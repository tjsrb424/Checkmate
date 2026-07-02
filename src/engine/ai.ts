import {
  Difficulty,
  GameState,
  Move,
  Piece,
  SearchLimits,
  Side,
  moveKey,
  otherSide
} from './types';
import { applyMove, generateLegalMoves, isCheckmate, isInCheck } from './rules';
import { computeZobristHash, hashToKey } from './hash';
import { TranspositionEntry, TranspositionTable } from './transposition';

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
  cutoffs: number;
  ttHits: number;
  ttMisses: number;
  ttStores: number;
  qNodes: number;
  qCutoffs: number;
  table?: TranspositionTable;
  enableTransposition: boolean;
  enableQuiescence: boolean;
  maxQuiescenceDepth: number;
  includeQuietChecks: boolean;
  rootMoveHint?: Move;
}

export interface SearchResult {
  move: Move | null;
  score: number;
  depth: number;
  nodes: number;
  pv: Move[];
  ttHits: number;
  ttMisses: number;
  ttStores: number;
  cutoffs: number;
  nps: number;
  elapsedMs: number;
  qNodes: number;
  qCutoffs: number;
  quiescenceEnabled: boolean;
}

export interface SearchOptions {
  reuseTable?: boolean;
  table?: TranspositionTable;
  enableTransposition?: boolean;
  enableQuiescence?: boolean;
  maxQuiescenceDepth?: number;
  includeQuietChecks?: boolean;
}

export function chooseBestMove(state: GameState, difficulty: Difficulty): SearchResult {
  const limits = difficultyLimits[difficulty];
  return searchBestMove(state, limits);
}

export function searchBestMove(state: GameState, limits: SearchLimits, options: SearchOptions = {}): SearchResult {
  const enableTransposition = options.enableTransposition !== false;
  const enableQuiescence = options.enableQuiescence !== false;
  const table = enableTransposition ? options.table ?? new TranspositionTable() : undefined;
  const context: SearchContext = {
    startedAt: performance.now(),
    timeMs: limits.timeMs,
    timedOut: false,
    nodes: 0,
    cutoffs: 0,
    ttHits: 0,
    ttMisses: 0,
    ttStores: 0,
    qNodes: 0,
    qCutoffs: 0,
    table,
    enableTransposition,
    enableQuiescence,
    maxQuiescenceDepth: options.maxQuiescenceDepth ?? 6,
    includeQuietChecks: options.includeQuietChecks ?? true
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
      context.rootMoveHint = result.move;
    }
  }

  if (!bestMove) {
    const legalMoves = orderMoves(state, generateLegalMoves(state), context);
    bestMove = legalMoves[0] ?? null;
    bestScore = bestMove ? evaluatePosition(applyMove(state, bestMove, false), state.turn) : 0;
  }

  const elapsedMs = Math.max(0, performance.now() - context.startedAt);
  const pv = bestMove && table ? extractPrincipalVariation(state, table, Math.max(1, completedDepth)) : bestMove ? [bestMove] : [];
  return {
    move: bestMove,
    score: bestScore,
    depth: completedDepth,
    nodes: context.nodes,
    pv,
    ttHits: context.ttHits,
    ttMisses: context.ttMisses,
    ttStores: context.ttStores,
    cutoffs: context.cutoffs,
    nps: Math.round(context.nodes / Math.max(0.001, elapsedMs / 1000)),
    elapsedMs,
    qNodes: context.qNodes,
    qCutoffs: context.qCutoffs,
    quiescenceEnabled: context.enableQuiescence
  };
}

function searchRoot(state: GameState, depth: number, context: SearchContext): SearchResult {
  const moves = orderMoves(state, generateLegalMoves(state), context);
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

  if (bestMove && context.table) {
    storeTransposition(state, depth, bestScore, Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY, bestMove, context, 0);
  }

  return {
    move: bestMove,
    score: bestScore,
    depth,
    nodes: context.nodes,
    pv: [],
    ttHits: context.ttHits,
    ttMisses: context.ttMisses,
    ttStores: context.ttStores,
    cutoffs: context.cutoffs,
    nps: 0,
    elapsedMs: performance.now() - context.startedAt,
    qNodes: context.qNodes,
    qCutoffs: context.qCutoffs,
    quiescenceEnabled: context.enableQuiescence
  };
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

  const originalAlpha = alpha;
  const ttEntry = getTransposition(state, depth, alpha, beta, context, ply);
  if (ttEntry?.usable) {
    return ttEntry.score;
  }
  let currentAlpha = ttEntry?.alpha ?? alpha;
  let currentBeta = ttEntry?.beta ?? beta;

  const legalMoves = generateLegalMoves(state);
  if (legalMoves.length === 0) {
    return isInCheck(state.board, state.turn) ? -MATE_SCORE + ply : DRAW_SCORE;
  }
  if (depth === 0) {
    return context.enableQuiescence
      ? quiescence(state, currentAlpha, currentBeta, context, ply, 0)
      : evaluatePosition(state, state.turn);
  }

  let best = Number.NEGATIVE_INFINITY;
  let bestMove: Move | undefined;
  for (const move of orderMoves(state, legalMoves, context)) {
    const next = applyMove(state, move, false);
    const score = -negamax(next, depth - 1, -currentBeta, -currentAlpha, context, ply + 1);
    if (context.timedOut) break;
    if (score > best) {
      best = score;
      bestMove = move;
    }
    currentAlpha = Math.max(currentAlpha, score);
    if (currentAlpha >= currentBeta) {
      context.cutoffs += 1;
      break;
    }
  }
  if (!context.timedOut) {
    storeTransposition(state, depth, best, originalAlpha, beta, bestMove, context, ply);
  }
  return best;
}

function quiescence(
  state: GameState,
  alpha: number,
  beta: number,
  context: SearchContext,
  ply: number,
  qDepth: number
): number {
  context.qNodes += 1;
  if (isTimedOut(context)) return evaluatePosition(state, state.turn);

  const legalMoves = generateLegalMoves(state);
  if (legalMoves.length === 0) {
    return isInCheck(state.board, state.turn) ? -MATE_SCORE + ply : DRAW_SCORE;
  }
  if (qDepth >= context.maxQuiescenceDepth) {
    return evaluatePosition(state, state.turn);
  }

  const standPat = evaluatePosition(state, state.turn);
  if (standPat >= beta) {
    context.qCutoffs += 1;
    return beta;
  }

  let currentAlpha = Math.max(alpha, standPat);
  const tacticalMoves = orderTacticalMoves(state, getTacticalMoves(state, legalMoves, context));
  for (const move of tacticalMoves) {
    const next = applyMove(state, move, false);
    const score = -quiescence(next, -beta, -currentAlpha, context, ply + 1, qDepth + 1);
    if (context.timedOut) break;
    if (score >= beta) {
      context.qCutoffs += 1;
      return beta;
    }
    if (score > currentAlpha) currentAlpha = score;
  }

  return currentAlpha;
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

function orderMoves(state: GameState, moves: Move[], context?: SearchContext): Move[] {
  const ttMove = context?.table ? peekBestMove(state, context.table) : undefined;
  const rootHint = context?.rootMoveHint;
  return [...moves].sort((a, b) => moveGuess(state, b, ttMove, rootHint) - moveGuess(state, a, ttMove, rootHint));
}

function moveGuess(state: GameState, move: Move, ttMove?: Move, rootHint?: Move): number {
  if (ttMove && sameMove(move, ttMove)) return MATE_SCORE * 2;
  if (rootHint && sameMove(move, rootHint)) return MATE_SCORE + 500_000;

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

function getTacticalMoves(state: GameState, moves: Move[], context: SearchContext): Move[] {
  return moves.filter((move) => {
    const capture = isCapture(state, move);
    const next = applyMove(state, move, false);
    const mate = isCheckmate(next.board, next.turn);
    const check = isInCheck(next.board, next.turn);

    if (mate) return true;
    if (capture) return true;
    if (check && context.includeQuietChecks) return true;
    return false;
  });
}

function orderTacticalMoves(state: GameState, moves: Move[]): Move[] {
  return [...moves].sort((a, b) => tacticalMoveGuess(state, b) - tacticalMoveGuess(state, a));
}

function tacticalMoveGuess(state: GameState, move: Move): number {
  const next = applyMove(state, move, false);
  if (isCheckmate(next.board, next.turn)) return MATE_SCORE;

  const captureScore = captureGuess(state, move);
  const checkScore = isInCheck(next.board, next.turn) ? 50_000 : 0;
  return checkScore + captureScore;
}

function captureGuess(state: GameState, move: Move): number {
  const victim = state.board[move.to.y][move.to.x] ?? move.captured;
  const attacker = state.board[move.from.y][move.from.x];
  if (!victim || !attacker) return 0;
  return pieceValues[victim.kind] * 10 - pieceValues[attacker.kind];
}

function isCapture(state: GameState, move: Move): boolean {
  return Boolean(move.captured ?? state.board[move.to.y][move.to.x]);
}

function getTransposition(
  state: GameState,
  depth: number,
  alpha: number,
  beta: number,
  context: SearchContext,
  ply: number
): { usable: true; score: number } | { usable: false; alpha: number; beta: number } | undefined {
  if (!context.enableTransposition || !context.table) return undefined;
  const entry = context.table.get(stateKey(state));
  if (!entry) {
    context.ttMisses += 1;
    return undefined;
  }

  context.ttHits += 1;
  if (entry.depth < depth) {
    return undefined;
  }

  const score = scoreFromTT(entry.score, ply);
  if (entry.flag === 'EXACT') return { usable: true, score };
  let nextAlpha = alpha;
  let nextBeta = beta;
  if (entry.flag === 'LOWERBOUND') nextAlpha = Math.max(nextAlpha, score);
  if (entry.flag === 'UPPERBOUND') nextBeta = Math.min(nextBeta, score);
  if (nextAlpha >= nextBeta) return { usable: true, score };
  return { usable: false, alpha: nextAlpha, beta: nextBeta };
}

function storeTransposition(
  state: GameState,
  depth: number,
  score: number,
  originalAlpha: number,
  beta: number,
  bestMove: Move | undefined,
  context: SearchContext,
  ply: number
): void {
  if (!context.enableTransposition || !context.table) return;
  let flag: TranspositionEntry['flag'] = 'EXACT';
  if (score <= originalAlpha) flag = 'UPPERBOUND';
  else if (score >= beta) flag = 'LOWERBOUND';
  context.table.set({
    key: stateKey(state),
    depth,
    score: scoreToTT(score, ply),
    flag,
    bestMove
  });
  context.ttStores += 1;
}

export function extractPrincipalVariation(state: GameState, table: TranspositionTable, maxDepth: number): Move[] {
  const pv: Move[] = [];
  let current = state;
  const seen = new Set<string>();

  for (let depth = 0; depth < maxDepth; depth += 1) {
    const key = stateKey(current);
    if (seen.has(key)) break;
    seen.add(key);

    const entry = table.peek(key);
    const move = entry?.bestMove;
    if (!move) break;
    if (!generateLegalMoves(current).some((legal) => sameMove(legal, move))) break;
    pv.push(move);
    current = applyMove(current, move, false);
  }

  return pv;
}

function peekBestMove(state: GameState, table: TranspositionTable): Move | undefined {
  const entry = table.peek(stateKey(state));
  return entry?.bestMove;
}

function stateKey(state: GameState): string {
  return hashToKey(computeZobristHash(state));
}

function sameMove(a: Move, b: Move): boolean {
  return moveKey(a) === moveKey(b);
}

function scoreToTT(score: number, ply: number): number {
  if (score > MATE_SCORE / 2) return score + ply;
  if (score < -MATE_SCORE / 2) return score - ply;
  return score;
}

function scoreFromTT(score: number, ply: number): number {
  if (score > MATE_SCORE / 2) return score - ply;
  if (score < -MATE_SCORE / 2) return score + ply;
  return score;
}

function isTimedOut(context: SearchContext): boolean {
  if (performance.now() - context.startedAt > context.timeMs) {
    context.timedOut = true;
    return true;
  }
  return false;
}
