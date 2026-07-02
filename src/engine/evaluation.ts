import {
  BOARD_HEIGHT,
  BOARD_WIDTH,
  Board,
  GameState,
  Piece,
  Position,
  Side,
  getPiece,
  inBounds,
  otherSide,
  samePosition
} from './types';
import { generateLegalMoves, generatePseudoMoves, isInCheck, isInPalace, pieceMoves, findGeneral } from './rules';

const MOBILITY_WEIGHT = 7;

const orthogonalDirections: Position[] = [
  { x: 1, y: 0 },
  { x: -1, y: 0 },
  { x: 0, y: 1 },
  { x: 0, y: -1 }
];

// These values intentionally preserve the Sprint 4 score scale. Tune only after engine-vs-engine testing.
export const pieceValues: Record<Piece['kind'], number> = {
  GENERAL: 100000,
  CHARIOT: 1300,
  CANNON: 700,
  HORSE: 500,
  ELEPHANT: 300,
  GUARD: 300,
  SOLDIER: 220
};

export interface EvaluationBreakdown {
  material: number;
  positional: number;
  mobility: number;
  kingSafety: number;
  attackPressure: number;
  chariotActivity: number;
  cannonActivity: number;
  horseElephantActivity: number;
  soldierStructure: number;
  checkPressure: number;
  total: number;
}

export function evaluatePosition(state: GameState, side: Side): number {
  return evaluatePositionBreakdown(state, side).total;
}

export function evaluatePositionBreakdown(state: GameState, side: Side): EvaluationBreakdown {
  const enemy = otherSide(side);
  const material = materialScore(state.board, side);
  const positional = piecePairScore(state.board, side, (board, position, piece) => positionalBonus(piece, position.x, position.y));
  const mobility = mobilityScore(state, side);
  const kingSafety = kingSafetyFor(state.board, side) - kingSafetyFor(state.board, enemy);
  const attackPressure = attackPressureFor(state.board, side) - attackPressureFor(state.board, enemy);
  const chariotActivity = activityPairScore(state.board, side, chariotActivityForPiece);
  const cannonActivity = activityPairScore(state.board, side, cannonActivityForPiece);
  const horseElephantActivity = activityPairScore(state.board, side, horseElephantActivityForPiece);
  const soldierStructure = activityPairScore(state.board, side, soldierStructureForPiece);
  const checkPressure = checkPressureFor(state.board, side) - checkPressureFor(state.board, enemy);
  const total =
    material +
    positional +
    mobility +
    kingSafety +
    attackPressure +
    chariotActivity +
    cannonActivity +
    horseElephantActivity +
    soldierStructure +
    checkPressure;

  return {
    material,
    positional,
    mobility,
    kingSafety,
    attackPressure,
    chariotActivity,
    cannonActivity,
    horseElephantActivity,
    soldierStructure,
    checkPressure,
    total
  };
}

export function positionalBonus(piece: Piece, x: number, y: number): number {
  const progress = forwardProgress(piece.side, y);
  const centerFileBonus = x >= 3 && x <= 5 ? 16 : 0;

  switch (piece.kind) {
    case 'SOLDIER':
      return progress * 18 + centerFileBonus;
    case 'HORSE':
    case 'ELEPHANT':
      return progress > 1 ? 35 + centrality(x) * 3 : -20;
    case 'CHARIOT':
    case 'CANNON':
      return x >= 2 && x <= 6 ? 24 : 0;
    case 'GENERAL':
      return x === 4 ? 20 : -25;
    case 'GUARD':
      return 10 + (isInPalace({ x, y }, piece.side) ? 12 : 0);
  }
}

export function formatEvaluationBreakdown(breakdown: EvaluationBreakdown): string {
  return [
    `total=${breakdown.total}`,
    `material=${breakdown.material}`,
    `positional=${breakdown.positional}`,
    `mobility=${breakdown.mobility}`,
    `kingSafety=${breakdown.kingSafety}`,
    `attackPressure=${breakdown.attackPressure}`,
    `chariot=${breakdown.chariotActivity}`,
    `cannon=${breakdown.cannonActivity}`,
    `horseElephant=${breakdown.horseElephantActivity}`,
    `soldier=${breakdown.soldierStructure}`,
    `check=${breakdown.checkPressure}`
  ].join(' ');
}

function materialScore(board: Board, side: Side): number {
  return piecePairScore(board, side, (_board, _position, piece) => pieceValues[piece.kind]);
}

function mobilityScore(state: GameState, side: Side): number {
  const enemy = otherSide(side);
  const sideMoves = generateLegalMoves({ board: state.board, turn: side, history: state.history }, side).length;
  const enemyMoves = generateLegalMoves({ board: state.board, turn: enemy, history: state.history }, enemy).length;
  return (sideMoves - enemyMoves) * MOBILITY_WEIGHT;
}

function kingSafetyFor(board: Board, side: Side): number {
  const general = findGeneral(board, side);
  if (!general) return -1200;

  let score = 0;
  if (isInCheck(board, side)) score -= 280;

  const palacePieces = piecesInPalace(board, side);
  const guardsInPalace = palacePieces.filter(({ piece }) => piece.kind === 'GUARD').length;
  const defendersNearGeneral = ownPiecesNear(board, general, side, 1);
  const emptyAdjacent = adjacentSquares(general).filter((pos) => isInPalace(pos, side) && !getPiece(board, pos)).length;

  score += guardsInPalace * 45;
  score += defendersNearGeneral * 18;
  score -= emptyAdjacent * 12;
  score -= facingGeneralRisk(board, side, general);

  return score;
}

function attackPressureFor(board: Board, side: Side): number {
  const enemy = otherSide(side);
  const enemyGeneral = findGeneral(board, enemy);
  if (!enemyGeneral) return 400;

  let score = isInCheck(board, enemy) ? 160 : 0;
  const pressureSquares = palacePressureSquares(enemyGeneral, enemy);

  forEachPiece(board, side, (position, piece) => {
    const moves = pieceMoves(board, position, piece);
    for (const move of moves) {
      if (pressureSquares.some((target) => samePosition(target, move.to))) {
        score += attackMoveWeight(piece.kind);
      }
    }

    if (isInPalace(position, enemy)) {
      if (piece.kind === 'SOLDIER') score += 55;
      if (piece.kind === 'CHARIOT' || piece.kind === 'CANNON') score += 70;
    }
  });

  return score;
}

function chariotActivityForPiece(board: Board, position: Position, piece: Piece): number {
  if (piece.kind !== 'CHARIOT') return 0;

  let emptySquares = 0;
  let blockedByOwn = 0;
  for (const dir of orthogonalDirections) {
    const ray = scanRay(board, position, dir);
    emptySquares += ray.emptyCount;
    if (ray.firstPiece?.piece.side === piece.side) blockedByOwn += 1;
  }

  const enemyGeneral = findGeneral(board, otherSide(piece.side));
  let score = emptySquares * 8 - blockedByOwn * 14 + centrality(position.x) * 6 + forwardProgress(piece.side, position.y) * 4;
  if (enemyGeneral && (enemyGeneral.x === position.x || enemyGeneral.y === position.y)) {
    score += 32;
    if (isClearLine(board, position, enemyGeneral)) score += 50;
  }
  if (enemyGeneral && manhattan(position, enemyGeneral) <= 3) score += 24;
  return score;
}

function cannonActivityForPiece(board: Board, position: Position, piece: Piece): number {
  if (piece.kind !== 'CANNON') return 0;

  let score = centrality(position.x) * 4 + forwardProgress(piece.side, position.y) * 3;
  for (const dir of orthogonalDirections) {
    const screen = firstPieceInDirection(board, position, dir);
    if (!screen) continue;
    if (screen.piece.kind === 'CANNON') {
      score -= 12;
      continue;
    }

    score += 18;
    const target = firstPieceInDirection(board, screen.position, dir);
    if (!target) continue;
    if (target.piece.side !== piece.side && target.piece.kind !== 'CANNON') {
      score += Math.round(pieceValues[target.piece.kind] / 20);
    }
  }
  return score;
}

function horseElephantActivityForPiece(board: Board, position: Position, piece: Piece): number {
  if (piece.kind !== 'HORSE' && piece.kind !== 'ELEPHANT') return 0;

  const moveCount = pieceMoves(board, position, piece).length;
  let score = moveCount * (piece.kind === 'HORSE' ? 14 : 10);
  if (moveCount <= 1) score -= 28;
  if (forwardProgress(piece.side, position.y) <= 1) score -= 18;
  score += centrality(position.x) * 4;
  score += Math.min(4, forwardProgress(piece.side, position.y)) * 5;
  return score;
}

function soldierStructureForPiece(board: Board, position: Position, piece: Piece): number {
  if (piece.kind !== 'SOLDIER') return 0;

  const enemy = otherSide(piece.side);
  const enemyGeneral = findGeneral(board, enemy);
  let score = forwardProgress(piece.side, position.y) * 16 + centrality(position.x) * 5;
  if (isInPalace(position, enemy)) score += 90;
  if (enemyGeneral && manhattan(position, enemyGeneral) <= 2) score += 45;
  if (!hasFriendlySoldierNeighbor(board, position, piece.side)) score -= 10;

  const forward = piece.side === 'CHO' ? -1 : 1;
  const front = { x: position.x, y: position.y + forward };
  const blocker = getPiece(board, front);
  if (blocker?.side === piece.side) score -= 16;
  return score;
}

function checkPressureFor(board: Board, side: Side): number {
  const enemy = otherSide(side);
  let score = 0;
  if (isInCheck(board, enemy)) score += 220;
  if (isInCheck(board, side)) score -= 220;

  const pseudoChecks = generatePseudoMoves(board, side).filter((move) => {
    const moving = getPiece(board, move.from);
    const captured = getPiece(board, move.to);
    if (!moving || captured?.side === side) return false;
    board[move.from.y][move.from.x] = null;
    board[move.to.y][move.to.x] = moving;
    const givesCheck = isInCheck(board, enemy);
    board[move.from.y][move.from.x] = moving;
    board[move.to.y][move.to.x] = captured;
    return givesCheck;
  }).length;

  return score + Math.min(6, pseudoChecks) * 18;
}

function piecePairScore(
  board: Board,
  side: Side,
  scorer: (board: Board, position: Position, piece: Piece) => number
): number {
  return activityPairScore(board, side, scorer);
}

function activityPairScore(
  board: Board,
  side: Side,
  scorer: (board: Board, position: Position, piece: Piece) => number
): number {
  let score = 0;
  forEachPiece(board, undefined, (position, piece) => {
    const sign = piece.side === side ? 1 : -1;
    score += sign * scorer(board, position, piece);
  });
  return score;
}

function forEachPiece(board: Board, side: Side | undefined, visitor: (position: Position, piece: Piece) => void): void {
  for (let y = 0; y < BOARD_HEIGHT; y += 1) {
    for (let x = 0; x < BOARD_WIDTH; x += 1) {
      const piece = board[y][x];
      if (!piece || (side && piece.side !== side)) continue;
      visitor({ x, y }, piece);
    }
  }
}

function forwardProgress(side: Side, y: number): number {
  return side === 'CHO' ? 9 - y : y;
}

function centrality(x: number): number {
  return 4 - Math.abs(4 - x);
}

function piecesInPalace(board: Board, side: Side): Array<{ position: Position; piece: Piece }> {
  const pieces: Array<{ position: Position; piece: Piece }> = [];
  forEachPiece(board, side, (position, piece) => {
    if (isInPalace(position, side)) pieces.push({ position, piece });
  });
  return pieces;
}

function ownPiecesNear(board: Board, center: Position, side: Side, radius: number): number {
  let count = 0;
  for (let y = center.y - radius; y <= center.y + radius; y += 1) {
    for (let x = center.x - radius; x <= center.x + radius; x += 1) {
      const pos = { x, y };
      if (!inBounds(pos) || samePosition(pos, center)) continue;
      if (getPiece(board, pos)?.side === side) count += 1;
    }
  }
  return count;
}

function adjacentSquares(pos: Position): Position[] {
  return orthogonalDirections.map((dir) => ({ x: pos.x + dir.x, y: pos.y + dir.y })).filter(inBounds);
}

function facingGeneralRisk(board: Board, side: Side, general: Position): number {
  const enemyGeneral = findGeneral(board, otherSide(side));
  if (!enemyGeneral || enemyGeneral.x !== general.x) return 0;

  let blockers = 0;
  const step = enemyGeneral.y > general.y ? 1 : -1;
  for (let y = general.y + step; y !== enemyGeneral.y; y += step) {
    if (board[y][general.x]) blockers += 1;
  }
  if (blockers === 0) return 180;
  if (blockers === 1) return 45;
  return 0;
}

function palacePressureSquares(general: Position, side: Side): Position[] {
  const squares: Position[] = [];
  for (let y = general.y - 1; y <= general.y + 1; y += 1) {
    for (let x = general.x - 1; x <= general.x + 1; x += 1) {
      const pos = { x, y };
      if (inBounds(pos) && (isInPalace(pos, side) || manhattan(pos, general) <= 1)) {
        squares.push(pos);
      }
    }
  }
  return squares;
}

function attackMoveWeight(kind: Piece['kind']): number {
  switch (kind) {
    case 'CHARIOT':
      return 24;
    case 'CANNON':
      return 20;
    case 'HORSE':
    case 'ELEPHANT':
      return 14;
    case 'SOLDIER':
      return 18;
    case 'GENERAL':
    case 'GUARD':
      return 8;
  }
}

function scanRay(board: Board, from: Position, dir: Position): { emptyCount: number; firstPiece?: { position: Position; piece: Piece } } {
  let emptyCount = 0;
  let pos = { x: from.x + dir.x, y: from.y + dir.y };
  while (inBounds(pos)) {
    const piece = getPiece(board, pos);
    if (piece) return { emptyCount, firstPiece: { position: { ...pos }, piece } };
    emptyCount += 1;
    pos = { x: pos.x + dir.x, y: pos.y + dir.y };
  }
  return { emptyCount };
}

function firstPieceInDirection(board: Board, from: Position, dir: Position): { position: Position; piece: Piece } | null {
  let pos = { x: from.x + dir.x, y: from.y + dir.y };
  while (inBounds(pos)) {
    const piece = getPiece(board, pos);
    if (piece) return { position: { ...pos }, piece };
    pos = { x: pos.x + dir.x, y: pos.y + dir.y };
  }
  return null;
}

function isClearLine(board: Board, a: Position, b: Position): boolean {
  if (a.x !== b.x && a.y !== b.y) return false;
  const dx = Math.sign(b.x - a.x);
  const dy = Math.sign(b.y - a.y);
  let pos = { x: a.x + dx, y: a.y + dy };
  while (!samePosition(pos, b)) {
    if (getPiece(board, pos)) return false;
    pos = { x: pos.x + dx, y: pos.y + dy };
  }
  return true;
}

function hasFriendlySoldierNeighbor(board: Board, position: Position, side: Side): boolean {
  return [
    { x: position.x - 1, y: position.y },
    { x: position.x + 1, y: position.y }
  ].some((pos) => getPiece(board, pos)?.side === side && getPiece(board, pos)?.kind === 'SOLDIER');
}

function manhattan(a: Position, b: Position): number {
  return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
}
