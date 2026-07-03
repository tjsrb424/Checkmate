import { describe, expect, it } from 'vitest';
import { createInitialBoard, emptyBoard, setPiece } from './index';
import { scoreBoardMaterial, scoreSideMaterial } from './scoring';

describe('janggi material scoring', () => {
  it('scores the initial board with Han deom', () => {
    const board = createInitialBoard('inner-elephant', 'inner-elephant');

    expect(scoreSideMaterial(board, 'CHO')).toBe(72);
    expect(scoreSideMaterial(board, 'HAN')).toBe(73.5);
    expect(scoreBoardMaterial(board)).toEqual({ cho: 72, han: 73.5, winner: 'HAN' });
  });

  it('reflects a removed chariot', () => {
    const board = createInitialBoard('inner-elephant', 'inner-elephant');
    board[0][0] = null;

    expect(scoreBoardMaterial(board)).toEqual({ cho: 72, han: 60.5, winner: 'CHO' });
  });

  it('applies Han deom even on bare generals', () => {
    const board = emptyBoard();
    setPiece(board, { x: 4, y: 8 }, { side: 'CHO', kind: 'GENERAL' });
    setPiece(board, { x: 4, y: 1 }, { side: 'HAN', kind: 'GENERAL' });

    expect(scoreBoardMaterial(board)).toEqual({ cho: 0, han: 1.5, winner: 'HAN' });
  });
});
