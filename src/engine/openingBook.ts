import { Formation, GameState, Move, Position, Side, moveKey, samePosition } from './types';
import { applyMove, createGameState, generateLegalMoves, isLegalMove } from './rules';
import { createInitialBoard } from './setup';
import { computeZobristHash, hashToKey } from './hash';

export interface OpeningBookRecord {
  id: string;
  source?: string;
  group?: string;
  choChalim: string;
  hanChalim: string;
  result: 'cho' | 'han' | 'draw' | 'unknown';
  first16: string;
  moves16Ok?: boolean;
}

export interface ParsedOpeningMove {
  plyNumber: number;
  from: Position;
  to: Position;
  pieceLabel: string;
  suffix: string;
}

export interface OpeningBookMove {
  move: Move;
  playCount: number;
  winCount: number;
  lossCount: number;
  drawCount: number;
  scoreRate: number;
  bookScore: number;
  sources: string[];
}

export interface OpeningBookPosition {
  positionKey: string;
  ply: number;
  turn: Side;
  choFormation: Formation;
  hanFormation: Formation;
  moves: OpeningBookMove[];
}

export interface OpeningBook {
  positions: Record<string, OpeningBookPosition>;
  positionCount: number;
  moveCount: number;
  sourceGameCount: number;
  skippedGameCount: number;
  illegalMoveCount: number;
  parseFailureCount: number;
  createdAt: string;
}

export interface OpeningBookBuildOptions {
  maxPly?: number;
  minPlayCount?: number;
  maxMovesPerPosition?: number;
  includeUnknownResults?: boolean;
}

export interface OpeningBookLookupOptions {
  choFormation?: Formation;
  hanFormation?: Formation;
  minPlayCount?: number;
  maxMoves?: number;
}

type OutcomeForSide = 'win' | 'loss' | 'draw' | 'unknown';

const chalimByFormation: Record<Formation, string> = {
  'left-elephant': '마상마상',
  'right-elephant': '상마상마',
  'inner-elephant': '마상상마',
  'outer-elephant': '상마마상'
};

export function chalimStringToFormation(chalim: string): Formation | null {
  const normalized = normalizeChalim(chalim);
  for (const [formation, value] of Object.entries(chalimByFormation)) {
    if (value === normalized) return formation as Formation;
  }
  return null;
}

export function formationToChalimString(formation: Formation): string {
  return chalimByFormation[formation];
}

export function parseOpeningMoveToken(token: string): ParsedOpeningMove | null {
  const trimmed = token.trim();
  const match = /^(\d+)\.(.+)$/.exec(trimmed);
  if (!match) return null;

  const plyNumber = Number(match[1]);
  const body = match[2];
  const fromMatch = /^([0-8])([0-9])/.exec(body);
  if (!Number.isInteger(plyNumber) || !fromMatch) return null;

  const from = { x: Number(fromMatch[1]), y: Number(fromMatch[2]) };
  const rest = body.slice(fromMatch[0].length);
  const toMatches = Array.from(rest.matchAll(/([0-8])([0-9])/g));
  if (toMatches.length === 0) return null;

  const toMatch = toMatches[toMatches.length - 1];
  const to = { x: Number(toMatch[1]), y: Number(toMatch[2]) };
  if (!isOpeningPosition(from) || !isOpeningPosition(to)) return null;

  const pieceLabel = rest.slice(0, toMatch.index).trim();
  const suffix = rest.slice((toMatch.index ?? 0) + toMatch[0].length).trim();
  if (!pieceLabel) return null;

  return { plyNumber, from, to, pieceLabel, suffix };
}

export function parseFirst16Moves(first16: string): ParsedOpeningMove[] {
  return tokenizeFirst16(first16)
    .map(parseOpeningMoveToken)
    .filter((move): move is ParsedOpeningMove => Boolean(move));
}

export function parseOpeningRecordsCsv(csv: string): OpeningBookRecord[] {
  const rows = parseCsvRows(csv).filter((row) => row.some((cell) => cell.trim().length > 0));
  const [header, ...body] = rows;
  if (!header) return [];

  const indexes = new Map(header.map((name, index) => [name.trim(), index]));
  return body.map((row, index) => ({
    id: getCsvCell(row, indexes, 'id') || getCsvCell(row, indexes, 'game_index') || `record-${index + 1}`,
    source: getCsvCell(row, indexes, 'source') || undefined,
    group: getCsvCell(row, indexes, 'group') || undefined,
    choChalim: getCsvCell(row, indexes, 'cho_chalim'),
    hanChalim: getCsvCell(row, indexes, 'han_chalim'),
    result: parseRecordResult(getCsvCell(row, indexes, 'result')),
    first16: getCsvCell(row, indexes, 'first16'),
    moves16Ok: parseBooleanCell(getCsvCell(row, indexes, 'moves16_ok'))
  }));
}

export function buildOpeningBookFromRecords(
  records: OpeningBookRecord[],
  options: OpeningBookBuildOptions = {}
): OpeningBook {
  const maxPly = options.maxPly ?? 16;
  const minPlayCount = options.minPlayCount ?? 2;
  const maxMovesPerPosition = options.maxMovesPerPosition ?? 5;
  const includeUnknownResults = options.includeUnknownResults === true;
  const positions: Record<string, OpeningBookPosition> = {};
  let sourceGameCount = 0;
  let skippedGameCount = 0;
  let illegalMoveCount = 0;
  let parseFailureCount = 0;

  for (const record of records) {
    if (record.moves16Ok === false || (!includeUnknownResults && record.result === 'unknown')) {
      skippedGameCount += 1;
      continue;
    }

    const choFormation = chalimStringToFormation(record.choChalim);
    const hanFormation = chalimStringToFormation(record.hanChalim);
    if (!choFormation || !hanFormation) {
      skippedGameCount += 1;
      continue;
    }

    const tokens = tokenizeFirst16(record.first16);
    const parsedMoves = tokens.map(parseOpeningMoveToken);
    parseFailureCount += parsedMoves.filter((move) => !move).length;
    const moves = parsedMoves.filter((move): move is ParsedOpeningMove => Boolean(move)).slice(0, maxPly);
    if (moves.length === 0) {
      skippedGameCount += 1;
      continue;
    }

    sourceGameCount += 1;
    let state = createGameState(createInitialBoard(choFormation, hanFormation), 'CHO');
    for (let ply = 0; ply < moves.length; ply += 1) {
      const parsed = moves[ply];
      const legalMove = findMatchingLegalMove(state, { from: parsed.from, to: parsed.to });
      if (!legalMove) {
        illegalMoveCount += 1;
        break;
      }

      const positionKey = hashToKey(computeZobristHash(state));
      const position = (positions[positionKey] ??= {
        positionKey,
        ply,
        turn: state.turn,
        choFormation,
        hanFormation,
        moves: []
      });
      addBookMove(position, legalMove, record, state.turn);
      state = applyMove(state, legalMove, true);
    }
  }

  for (const position of Object.values(positions)) {
    position.moves = position.moves
      .map(finalizeBookMove)
      .sort((a, b) => compareBookMoves(a, b, minPlayCount))
      .slice(0, maxMovesPerPosition);
  }

  const moveCount = Object.values(positions).reduce((sum, position) => sum + position.moves.length, 0);
  return {
    positions,
    positionCount: Object.keys(positions).length,
    moveCount,
    sourceGameCount,
    skippedGameCount,
    illegalMoveCount,
    parseFailureCount,
    createdAt: new Date().toISOString()
  };
}

export function lookupOpeningMoves(
  book: OpeningBook,
  state: GameState,
  context: OpeningBookLookupOptions = {}
): OpeningBookMove[] {
  const positionKey = hashToKey(computeZobristHash(state));
  const position = book.positions[positionKey];
  if (!position) return [];
  if (context.choFormation && position.choFormation !== context.choFormation) return [];
  if (context.hanFormation && position.hanFormation !== context.hanFormation) return [];

  const minPlayCount = context.minPlayCount ?? 1;
  return position.moves
    .filter((bookMove) => bookMove.playCount >= minPlayCount && isLegalMove(state, bookMove.move))
    .slice(0, context.maxMoves ?? position.moves.length);
}

export function chooseOpeningBookMove(
  book: OpeningBook,
  state: GameState,
  context: OpeningBookLookupOptions = {}
): Move | null {
  return lookupOpeningMoves(book, state, context)[0]?.move ?? null;
}

export function isOpeningBookMoveAvailable(
  book: OpeningBook,
  state: GameState,
  context: OpeningBookLookupOptions = {}
): boolean {
  return lookupOpeningMoves(book, state, context).length > 0;
}

export function describeOpeningBookMove(bookMove: OpeningBookMove): string {
  return `book playCount=${bookMove.playCount} scoreRate=${(bookMove.scoreRate * 100).toFixed(1)}% bookScore=${bookMove.bookScore.toFixed(2)}`;
}

export function openingBookToJson(book: OpeningBook): string {
  return JSON.stringify(book, null, 2);
}

export function openingBookFromJson(json: string): OpeningBook {
  return JSON.parse(json) as OpeningBook;
}

export function resultForSide(result: OpeningBookRecord['result'], side: Side): OutcomeForSide {
  if (result === 'draw') return 'draw';
  if (result === 'unknown') return 'unknown';
  if (result === 'cho') return side === 'CHO' ? 'win' : 'loss';
  return side === 'HAN' ? 'win' : 'loss';
}

function addBookMove(position: OpeningBookPosition, move: Move, record: OpeningBookRecord, side: Side): void {
  const key = moveKey(move);
  let bookMove = position.moves.find((candidate) => moveKey(candidate.move) === key);
  if (!bookMove) {
    bookMove = {
      move,
      playCount: 0,
      winCount: 0,
      lossCount: 0,
      drawCount: 0,
      scoreRate: 0,
      bookScore: 0,
      sources: []
    };
    position.moves.push(bookMove);
  }

  bookMove.playCount += 1;
  const outcome = resultForSide(record.result, side);
  if (outcome === 'win') bookMove.winCount += 1;
  if (outcome === 'loss') bookMove.lossCount += 1;
  if (outcome === 'draw') bookMove.drawCount += 1;
  const source = record.source ?? record.id;
  if (!bookMove.sources.includes(source)) bookMove.sources.push(source);
}

function finalizeBookMove(move: OpeningBookMove): OpeningBookMove {
  const scoreRate = move.playCount > 0 ? (move.winCount + move.drawCount * 0.5) / move.playCount : 0;
  return {
    ...move,
    scoreRate,
    bookScore: Math.log(move.playCount + 1) * 0.35 + scoreRate * 0.65
  };
}

function compareBookMoves(a: OpeningBookMove, b: OpeningBookMove, minPlayCount: number): number {
  const aEnough = a.playCount >= minPlayCount ? 1 : 0;
  const bEnough = b.playCount >= minPlayCount ? 1 : 0;
  return bEnough - aEnough || b.bookScore - a.bookScore || b.scoreRate - a.scoreRate || b.playCount - a.playCount;
}

function findMatchingLegalMove(state: GameState, move: Move): Move | null {
  const legalMove =
    generateLegalMoves(state).find((candidate) => samePosition(candidate.from, move.from) && samePosition(candidate.to, move.to)) ?? null;
  if (!legalMove) return null;
  return { ...legalMove, piece: state.board[legalMove.from.y][legalMove.from.x] ?? undefined };
}

function tokenizeFirst16(first16: string): string[] {
  return first16.split(/\s+/).map((token) => token.trim()).filter(Boolean);
}

function isOpeningPosition(pos: Position): boolean {
  return pos.x >= 0 && pos.x <= 8 && pos.y >= 0 && pos.y <= 9;
}

function normalizeChalim(chalim: string): string {
  return chalim.replace(/[\s"'[\],]/g, '');
}

function parseCsvRows(csv: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = '';
  let inQuotes = false;

  for (let i = 0; i < csv.length; i += 1) {
    const char = csv[i];
    const next = csv[i + 1];
    if (char === '"' && inQuotes && next === '"') {
      cell += '"';
      i += 1;
      continue;
    }
    if (char === '"') {
      inQuotes = !inQuotes;
      continue;
    }
    if (char === ',' && !inQuotes) {
      row.push(cell);
      cell = '';
      continue;
    }
    if ((char === '\n' || char === '\r') && !inQuotes) {
      if (char === '\r' && next === '\n') i += 1;
      row.push(cell);
      rows.push(row);
      row = [];
      cell = '';
      continue;
    }
    cell += char;
  }

  row.push(cell);
  rows.push(row);
  return rows;
}

function getCsvCell(row: string[], indexes: Map<string, number>, name: string): string {
  const index = indexes.get(name);
  return index === undefined ? '' : (row[index] ?? '').trim();
}

function parseBooleanCell(value: string): boolean | undefined {
  if (!value) return undefined;
  return /^(true|1|yes|y)$/i.test(value);
}

function parseRecordResult(value: string): OpeningBookRecord['result'] {
  const normalized = value.trim().toLowerCase();
  if (['cho', '초', '초승', '1-0'].includes(normalized)) return 'cho';
  if (['han', '한', '한승', '0-1'].includes(normalized)) return 'han';
  if (['draw', '무', '무승부', '1/2-1/2'].includes(normalized)) return 'draw';
  return 'unknown';
}
