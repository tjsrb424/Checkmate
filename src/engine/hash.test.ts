import { describe, expect, it } from 'vitest';
import {
  applyMove,
  computeBoardHash,
  computeZobristHash,
  createGameState,
  createInitialBoard,
  emptyBoard,
  hashToKey,
  setPiece
} from './index';

describe('zobrist hash', () => {
  it('returns the same hash for the same position', () => {
    const first = createGameState(createInitialBoard('inner-elephant', 'inner-elephant'), 'CHO');
    const second = createGameState(createInitialBoard('inner-elephant', 'inner-elephant'), 'CHO');

    expect(computeZobristHash(first)).toBe(computeZobristHash(second));
  });

  it('changes when a piece moves', () => {
    const state = createGameState(createInitialBoard('inner-elephant', 'inner-elephant'), 'CHO');
    const next = applyMove(state, { from: { x: 0, y: 6 }, to: { x: 0, y: 5 } }, false);

    expect(computeZobristHash(next)).not.toBe(computeZobristHash(state));
  });

  it('changes when the turn changes on the same board', () => {
    const board = createInitialBoard('inner-elephant', 'inner-elephant');

    expect(computeBoardHash(board, 'CHO')).not.toBe(computeBoardHash(board, 'HAN'));
  });

  it('changes when one piece is different', () => {
    const first = emptyBoard();
    const second = emptyBoard();
    setPiece(first, { x: 4, y: 8 }, { side: 'CHO', kind: 'GENERAL' });
    setPiece(second, { x: 4, y: 8 }, { side: 'CHO', kind: 'GUARD' });

    expect(computeBoardHash(first, 'CHO')).not.toBe(computeBoardHash(second, 'CHO'));
  });

  it('keeps the initial position hash stable', () => {
    const state = createGameState(createInitialBoard('inner-elephant', 'inner-elephant'), 'CHO');

    expect(hashToKey(computeZobristHash(state))).toBe('5097c6170a131bd3');
  });
});
