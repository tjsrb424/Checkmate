import { GameState, Move, moveKey } from './types';
import { applyMove, generateLegalMoves } from './rules';

export interface PerftDivideResult {
  move: Move;
  nodes: number;
}

export function perft(state: GameState, depth: number): number {
  if (depth < 0) {
    throw new Error(`perft depth must be non-negative: ${depth}`);
  }
  if (depth === 0) return 1;

  const moves = generateLegalMoves(state);
  if (depth === 1) return moves.length;

  let nodes = 0;
  for (const move of moves) {
    nodes += perft(applyMove(state, move, false), depth - 1);
  }
  return nodes;
}

export function perftDivide(state: GameState, depth: number): PerftDivideResult[] {
  if (depth < 1) {
    throw new Error(`perft divide depth must be at least 1: ${depth}`);
  }

  return generateLegalMoves(state).map((move) => ({
    move,
    nodes: perft(applyMove(state, move, false), depth - 1)
  }));
}

export function formatMoveForDebug(move: Move): string {
  const capture = move.captured ? `x${move.captured.side}:${move.captured.kind}` : '';
  return `${moveKey(move)}${capture}`;
}
