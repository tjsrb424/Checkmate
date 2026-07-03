import type { Board, Formation, GameState, Move, Piece, Position, Side } from './types';
import { BOARD_HEIGHT, BOARD_WIDTH, samePosition } from './types';
import { applyMove, createGameState, generateLegalMoves } from './rules';
import { createInitialBoard } from './setup';
import {
  OpeningBookRecord,
  chalimStringToFormation,
  parseOpeningMoveToken,
  resultForSide
} from './openingBook';

export interface TrainingExportOptions {
  maxPly?: number;
  includeUnknownResults?: boolean;
}

export interface PolicyTrainingExportSample {
  position: {
    board: Array<Array<Piece | null>>;
    turn: Side;
    history: Array<{ from: Position; to: Position }>;
    winner: Side | null;
    metadata: Record<string, string | number | boolean | null>;
  };
  move: { from: Position; to: Position };
  move_index: number;
  result: OpeningBookRecord['result'];
  source: string;
  ply: number;
  choFormation: Formation;
  hanFormation: Formation;
}

export interface PolicyTrainingExportStats {
  recordCount: number;
  sourceGameCount: number;
  sampleCount: number;
  skippedGameCount: number;
  illegalMoveCount: number;
  parseFailureCount: number;
  unknownResultSkipCount: number;
}

export interface PolicyTrainingExportResult {
  samples: PolicyTrainingExportSample[];
  stats: PolicyTrainingExportStats;
}

export function exportPolicySamplesFromOpeningRecords(
  records: OpeningBookRecord[],
  options: TrainingExportOptions = {}
): PolicyTrainingExportResult {
  const maxPly = options.maxPly ?? 16;
  const includeUnknownResults = options.includeUnknownResults === true;
  const samples: PolicyTrainingExportSample[] = [];
  const stats: PolicyTrainingExportStats = {
    recordCount: records.length,
    sourceGameCount: 0,
    sampleCount: 0,
    skippedGameCount: 0,
    illegalMoveCount: 0,
    parseFailureCount: 0,
    unknownResultSkipCount: 0
  };

  for (const record of records) {
    if (record.moves16Ok === false) {
      stats.skippedGameCount += 1;
      continue;
    }
    if (!includeUnknownResults && record.result === 'unknown') {
      stats.skippedGameCount += 1;
      stats.unknownResultSkipCount += 1;
      continue;
    }

    const choFormation = chalimStringToFormation(record.choChalim);
    const hanFormation = chalimStringToFormation(record.hanChalim);
    if (!choFormation || !hanFormation) {
      stats.skippedGameCount += 1;
      continue;
    }

    const tokens = tokenizeFirstMoves(record.first16);
    const parsedMoves = tokens.map(parseOpeningMoveToken);
    stats.parseFailureCount += parsedMoves.filter((move) => !move).length;
    const moves = parsedMoves.filter((move): move is NonNullable<typeof move> => Boolean(move)).slice(0, maxPly);
    if (moves.length === 0) {
      stats.skippedGameCount += 1;
      continue;
    }

    stats.sourceGameCount += 1;
    let state = createGameState(createInitialBoard(choFormation, hanFormation), 'CHO');
    for (let ply = 0; ply < moves.length; ply += 1) {
      const parsed = moves[ply];
      const legalMove = findMatchingLegalMove(state, { from: parsed.from, to: parsed.to });
      if (!legalMove) {
        stats.illegalMoveCount += 1;
        break;
      }

      samples.push(
        createPolicyTrainingSample({
          state,
          move: legalMove,
          record,
          ply,
          choFormation,
          hanFormation
        })
      );
      state = applyMove(state, legalMove, true);
    }
  }

  stats.sampleCount = samples.length;
  return { samples, stats };
}

export function policySampleToJsonLine(sample: PolicyTrainingExportSample): string {
  return JSON.stringify(sample);
}

export function policySamplesToJsonl(samples: PolicyTrainingExportSample[]): string {
  return samples.map(policySampleToJsonLine).join('\n') + (samples.length > 0 ? '\n' : '');
}

export function moveToPolicyIndex(move: Move): number {
  validatePosition(move.from);
  validatePosition(move.to);
  return (((move.from.x * BOARD_HEIGHT + move.from.y) * BOARD_WIDTH + move.to.x) * BOARD_HEIGHT + move.to.y);
}

function createPolicyTrainingSample(input: {
  state: GameState;
  move: Move;
  record: OpeningBookRecord;
  ply: number;
  choFormation: Formation;
  hanFormation: Formation;
}): PolicyTrainingExportSample {
  const source = input.record.source ?? input.record.id;
  return {
    position: {
      board: cloneBoardForJson(input.state.board),
      turn: input.state.turn,
      history: input.state.history.map((move) => ({ from: { ...move.from }, to: { ...move.to } })),
      winner: input.state.winner ?? null,
      metadata: {
        recordId: input.record.id,
        source,
        ply: input.ply,
        sideToMove: input.state.turn,
        outcomeForSide: resultForSide(input.record.result, input.state.turn),
        choFormation: input.choFormation,
        hanFormation: input.hanFormation
      }
    },
    move: { from: { ...input.move.from }, to: { ...input.move.to } },
    move_index: moveToPolicyIndex(input.move),
    result: input.record.result,
    source,
    ply: input.ply,
    choFormation: input.choFormation,
    hanFormation: input.hanFormation
  };
}

function findMatchingLegalMove(state: GameState, move: Move): Move | null {
  return (
    generateLegalMoves(state).find((candidate) => samePosition(candidate.from, move.from) && samePosition(candidate.to, move.to)) ??
    null
  );
}

function cloneBoardForJson(board: Board): Array<Array<Piece | null>> {
  return board.map((row) => row.map((piece) => (piece ? { ...piece } : null)));
}

function tokenizeFirstMoves(first16: string): string[] {
  return first16
    .split(/\s+/)
    .map((token) => token.trim())
    .filter(Boolean);
}

function validatePosition(position: Position): void {
  if (position.x < 0 || position.x >= BOARD_WIDTH || position.y < 0 || position.y >= BOARD_HEIGHT) {
    throw new Error(`position out of range: ${position.x},${position.y}`);
  }
}
