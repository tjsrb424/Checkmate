import type { Board, PieceKind, Side } from './types';

export const janggiMaterialValues: Record<PieceKind, number> = {
  GENERAL: 0,
  GUARD: 3,
  ELEPHANT: 3,
  HORSE: 5,
  CHARIOT: 13,
  CANNON: 7,
  SOLDIER: 2
};

export interface MaterialScoreResult {
  cho: number;
  han: number;
  winner: Side | 'DRAW';
}

export function scoreSideMaterial(board: Board, side: Side): number {
  let score = side === 'HAN' ? 1.5 : 0;
  for (const row of board) {
    for (const piece of row) {
      if (piece?.side === side) score += janggiMaterialValues[piece.kind];
    }
  }
  return score;
}

export function scoreBoardMaterial(board: Board): MaterialScoreResult {
  const cho = scoreSideMaterial(board, 'CHO');
  const han = scoreSideMaterial(board, 'HAN');
  return {
    cho,
    han,
    winner: cho > han ? 'CHO' : han > cho ? 'HAN' : 'DRAW'
  };
}
