import {
  Difficulty,
  GameState,
  Move,
  Piece,
  SearchLimits,
  Side,
  otherSide
} from './types';
import { applyMove, generateLegalMoves, isCheckmate, isInCheck } from './rules';

export const MATE_SCORE = 1_000_000;
export const DRAW_SCORE = 0;

const pieceValues: Record<Piece['kind'], number> = {
  GENERAL: 100000,
  CHARIOT: 1300,
  CANNON: 700,
  HORSE: 500,
  ELEPHANT: 300,
  GUARD: 300,
  SOLDIER: 220
};

export const difficultyLimits: Record<Difficulty, SearchLimits> = {
  easy: { maxDepth: 2, timeMs: 450 },
  normal: { maxDepth: 3, timeMs: 1100 },
  hard: { maxDepth: 4, timeMs: 2400 }
};

interface SearchContext {
  startedAt: number;
  timeMs: number;
  timedOut: boolean;
  nodes: number;
}

export interface SearchResult {
  move: Move | null;
  score: number;
  depth: number;
  nodes: number;
}

export function chooseBestMove(state: GameState, difficulty: Difficulty): SearchResult {
  const limits = difficultyLimits[difficulty];
  return searchBestMove(state, limits);
}

export function searchBestMove(state: GameState, limits: SearchLimits): SearchResult {
  const context: SearchContext = {
    startedAt: performance.now(),
    timeMs: limits.timeMs,
    timedOut: false,
    nodes: 0
  };
  let bestMove: Move | null = null;
  let bestScore = Number.NEGATIVE_INFINITY;
  let completedDepth = 0;

  for (let depth = 1; depth <= limits.maxDepth; depth += 1) {
    const result = searchRoot(state, depth, context);
    if (context.timedOut) break;
    if (result.move) {
      bestMove = result.move;
      bestScore = result.score;
      completedDepth = depth;
    }
  }

  if (!bestMove) {
    const legalMoves = orderMoves(state, generateLegalMoves(state));
    bestMove = legalMoves[0] ?? null;
    bestScore = bestMove ? evaluatePosition(applyMove(state, bestMove, false), state.turn) : 0;
  }

  return { move: bestMove, score: bestScore, depth: completedDepth, nodes: context.nodes };
}

function searchRoot(state: GameState, depth: number, context: SearchContext): SearchResult {
  const moves = orderMoves(state, generateLegalMoves(state));
  let bestMove: Move | null = null;
  let bestScore = Number.NEGATIVE_INFINITY;
  let alpha = Number.NEGATIVE_INFINITY;

  for (const move of moves) {
    if (isTimedOut(context)) break;
    const next = applyMove(state, move, false);
    const result = -negamax(next, depth - 1, Number.NEGATIVE_INFINITY, -alpha, context, 1);
    if (context.timedOut) break;
    if (result > bestScore) {
      bestScore = result;
      bestMove = move;
    }
    alpha = Math.max(alpha, bestScore);
  }

  return { move: bestMove, score: bestScore, depth, nodes: context.nodes };
}

function negamax(
  state: GameState,
  depth: number,
  alpha: number,
  beta: number,
  context: SearchContext,
  ply: number
): number {
  context.nodes += 1;
  if (isTimedOut(context)) return evaluatePosition(state, state.turn);

  const legalMoves = generateLegalMoves(state);
  if (legalMoves.length === 0) {
    return isInCheck(state.board, state.turn) ? -MATE_SCORE + ply : DRAW_SCORE;
  }
  if (depth === 0) {
    return evaluatePosition(state, state.turn);
  }

  let best = Number.NEGATIVE_INFINITY;
  let currentAlpha = alpha;
  for (const move of orderMoves(state, legalMoves)) {
    const next = applyMove(state, move, false);
    const score = -negamax(next, depth - 1, -beta, -currentAlpha, context, ply + 1);
    if (context.timedOut) break;
    best = Math.max(best, score);
    currentAlpha = Math.max(currentAlpha, score);
    if (currentAlpha >= beta) break;
  }
  return best;
}

export function evaluatePosition(state: GameState, side: Side): number {
  const board = state.board;
  let score = 0;
  for (let y = 0; y < board.length; y += 1) {
    for (let x = 0; x < board[y].length; x += 1) {
      const piece = board[y][x];
      if (!piece) continue;
      const sign = piece.side === side ? 1 : -1;
      score += sign * pieceValues[piece.kind];
      score += sign * positionalBonus(piece, x, y);
    }
  }

  const sideMobility = generateLegalMoves({ board, turn: side, history: state.history }, side).length;
  const enemy = otherSide(side);
  const enemyMobility = generateLegalMoves({ board, turn: enemy, history: state.history }, enemy).length;
  score += (sideMobility - enemyMobility) * 7;

  if (isInCheck(board, side)) score -= 350;
  if (isInCheck(board, enemy)) score += 350;
  return score;
}

function positionalBonus(piece: Piece, x: number, y: number): number {
  const forwardProgress = piece.side === 'CHO' ? 9 - y : y;
  switch (piece.kind) {
    case 'SOLDIER':
      return forwardProgress * 18 + (x >= 3 && x <= 5 ? 16 : 0);
    case 'HORSE':
    case 'ELEPHANT':
      return forwardProgress > 1 ? 35 : -20;
    case 'CHARIOT':
    case 'CANNON':
      return x >= 2 && x <= 6 ? 24 : 0;
    case 'GENERAL':
      return x === 4 ? 20 : -25;
    case 'GUARD':
      return 10;
  }
}

function orderMoves(state: GameState, moves: Move[]): Move[] {
  return [...moves].sort((a, b) => moveGuess(state, b) - moveGuess(state, a));
}

function moveGuess(state: GameState, move: Move): number {
  const target = state.board[move.to.y][move.to.x];
  const mover = state.board[move.from.y][move.from.x];
  const next = applyMove(state, move, false);
  if (isCheckmate(next.board, next.turn)) return MATE_SCORE;

  let score = 0;
  if (isInCheck(next.board, next.turn)) score += 50_000;
  if (mover && target) score += pieceValues[target.kind] * 10 - pieceValues[mover.kind];
  if (mover) score += positionalBonus(mover, move.to.x, move.to.y) - positionalBonus(mover, move.from.x, move.from.y);
  return score;
}

function isTimedOut(context: SearchContext): boolean {
  if (performance.now() - context.startedAt > context.timeMs) {
    context.timedOut = true;
    return true;
  }
  return false;
}
