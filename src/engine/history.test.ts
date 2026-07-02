import { describe, expect, it } from 'vitest';
import {
  Board,
  Move,
  PieceKind,
  Side,
  applyMove,
  canRedoMove,
  createGameState,
  emptyBoard,
  replayGame,
  setPiece,
  undoLastMove,
  undoToHumanTurn
} from './index';

function place(board: Board, x: number, y: number, side: Side, kind: PieceKind): void {
  setPiece(board, { x, y }, { side, kind });
}

function boardForHistory(): Board {
  const board = emptyBoard();
  place(board, 4, 8, 'CHO', 'GENERAL');
  place(board, 4, 1, 'HAN', 'GENERAL');
  place(board, 4, 5, 'CHO', 'SOLDIER');
  place(board, 0, 6, 'CHO', 'SOLDIER');
  place(board, 0, 3, 'HAN', 'SOLDIER');
  return board;
}

describe('game history helpers', () => {
  it('records moving and captured pieces when appending history', () => {
    const board = boardForHistory();
    place(board, 0, 5, 'HAN', 'HORSE');
    const state = createGameState(board, 'CHO');
    const next = applyMove(state, { from: { x: 0, y: 6 }, to: { x: 0, y: 5 } }, true);

    expect(next.history[0].piece).toEqual({ side: 'CHO', kind: 'SOLDIER' });
    expect(next.history[0].captured).toEqual({ side: 'HAN', kind: 'HORSE' });
  });

  it('replays history to the same board', () => {
    const initial = boardForHistory();
    const afterCho = applyMove(createGameState(initial, 'CHO'), { from: { x: 0, y: 6 }, to: { x: 0, y: 5 } }, true);
    const afterHan = applyMove(afterCho, { from: { x: 0, y: 3 }, to: { x: 0, y: 4 } }, true);
    const replayed = replayGame(initial, afterHan.history, 'CHO');

    expect(replayed.board).toEqual(afterHan.board);
    expect(replayed.turn).toBe(afterHan.turn);
  });

  it('undoes the last move by replaying from the initial board', () => {
    const initial = boardForHistory();
    const afterCho = applyMove(createGameState(initial, 'CHO'), { from: { x: 0, y: 6 }, to: { x: 0, y: 5 } }, true);
    const afterHan = applyMove(afterCho, { from: { x: 0, y: 3 }, to: { x: 0, y: 4 } }, true);
    const undone = undoLastMove(afterHan, initial, 'CHO');

    expect(undone.board).toEqual(afterCho.board);
    expect(undone.history).toHaveLength(1);
  });

  it('undoes back to the human turn in a human-vs-AI game', () => {
    const initial = boardForHistory();
    const afterHuman = applyMove(createGameState(initial, 'CHO'), { from: { x: 0, y: 6 }, to: { x: 0, y: 5 } }, true);
    const afterAi = applyMove(afterHuman, { from: { x: 0, y: 3 }, to: { x: 0, y: 4 } }, true);
    const undone = undoToHumanTurn(afterAi, initial, 'CHO', 'CHO');

    expect(undone.turn).toBe('CHO');
    expect(undone.history).toHaveLength(0);
  });

  it('allows legal redo moves and rejects illegal redo moves', () => {
    const initial = boardForHistory();
    const state = createGameState(initial, 'CHO');
    const legalRedo: Move = { from: { x: 0, y: 6 }, to: { x: 0, y: 5 } };
    const illegalRedo: Move = { from: { x: 8, y: 8 }, to: { x: 8, y: 7 } };

    expect(canRedoMove(state, legalRedo)).toBe(true);
    expect(canRedoMove(state, illegalRedo)).toBe(false);
  });
});
