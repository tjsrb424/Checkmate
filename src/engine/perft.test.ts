import { describe, expect, it } from 'vitest';
import {
  Board,
  GameState,
  Move,
  PieceKind,
  Side,
  applyMove,
  createGameState,
  createInitialBoard,
  emptyBoard,
  formatMoveForDebug,
  generateLegalMoves,
  isCheckmate,
  isInCheck,
  perft,
  perftDivide,
  setPiece
} from './index';

function place(board: Board, x: number, y: number, side: Side, kind: PieceKind): void {
  setPiece(board, { x, y }, { side, kind });
}

function state(board: Board, turn: Side = 'CHO'): GameState {
  return createGameState(board, turn);
}

function hasMove(moves: Move[], fromX: number, fromY: number, toX: number, toY: number): boolean {
  return moves.some((move) => move.from.x === fromX && move.from.y === fromY && move.to.x === toX && move.to.y === toY);
}

function addGenerals(board: Board): void {
  place(board, 3, 9, 'CHO', 'GENERAL');
  place(board, 5, 0, 'HAN', 'GENERAL');
}

describe('perft', () => {
  it('counts legal moves from the initial inner-elephant position', () => {
    const initial = state(createInitialBoard('inner-elephant', 'inner-elephant'), 'CHO');

    expect(perft(initial, 0)).toBe(1);
    expect(perft(initial, 1)).toBe(generateLegalMoves(initial).length);
  });

  it('keeps initial position perft snapshot stable', () => {
    const initial = state(createInitialBoard('inner-elephant', 'inner-elephant'), 'CHO');

    expect(perft(initial, 1)).toBe(31);
    expect(perft(initial, 2)).toBe(949);
    expect(perft(initial, 3)).toBe(29697);
  }, 30000);

  it('divides root moves and formats moves for debugging', () => {
    const initial = state(createInitialBoard('inner-elephant', 'inner-elephant'), 'CHO');
    const divide = perftDivide(initial, 2);

    expect(divide.reduce((total, item) => total + item.nodes, 0)).toBe(perft(initial, 2));
    expect(formatMoveForDebug(divide[0].move)).toMatch(/^\d,\d-\d,\d/);
  });

  it('reflects horse leg blocking in legal node counts', () => {
    const open = emptyBoard();
    addGenerals(open);
    place(open, 4, 4, 'CHO', 'HORSE');

    const blocked = emptyBoard();
    addGenerals(blocked);
    place(blocked, 4, 4, 'CHO', 'HORSE');
    place(blocked, 5, 4, 'HAN', 'SOLDIER');

    const moves = generateLegalMoves(state(blocked));
    expect(hasMove(moves, 4, 4, 6, 5)).toBe(false);
    expect(hasMove(moves, 4, 4, 6, 3)).toBe(false);
    expect(perft(state(open), 1) - perft(state(blocked), 1)).toBe(2);
  });

  it('reflects elephant path blocking in legal node counts', () => {
    const open = emptyBoard();
    addGenerals(open);
    place(open, 4, 4, 'CHO', 'ELEPHANT');

    const blocked = emptyBoard();
    addGenerals(blocked);
    place(blocked, 4, 4, 'CHO', 'ELEPHANT');
    place(blocked, 5, 4, 'HAN', 'SOLDIER');
    place(blocked, 2, 3, 'HAN', 'SOLDIER');

    const moves = generateLegalMoves(state(blocked));
    expect(hasMove(moves, 4, 4, 7, 6)).toBe(false);
    expect(hasMove(moves, 4, 4, 1, 2)).toBe(false);
    expect(perft(state(open), 1)).toBeGreaterThan(perft(state(blocked), 1));
  });

  it('validates cannon movement, capture, and cannon restrictions', () => {
    const board = emptyBoard();
    addGenerals(board);
    place(board, 0, 0, 'CHO', 'CANNON');
    place(board, 0, 2, 'CHO', 'SOLDIER');
    place(board, 0, 5, 'HAN', 'HORSE');
    place(board, 1, 2, 'HAN', 'CANNON');
    place(board, 2, 2, 'HAN', 'CANNON');

    const moves = generateLegalMoves(state(board));
    expect(hasMove(moves, 0, 0, 0, 1)).toBe(false);
    expect(hasMove(moves, 0, 0, 0, 3)).toBe(true);
    expect(hasMove(moves, 0, 0, 0, 5)).toBe(true);
    expect(hasMove(moves, 0, 0, 2, 2)).toBe(false);
    expect(perft(state(board), 1)).toBe(moves.length);
  });

  it('validates palace diagonal movement and blocking', () => {
    const board = emptyBoard();
    place(board, 4, 8, 'CHO', 'GENERAL');
    place(board, 5, 0, 'HAN', 'GENERAL');
    place(board, 3, 0, 'CHO', 'CHARIOT');
    place(board, 5, 2, 'HAN', 'HORSE');
    place(board, 4, 1, 'CHO', 'CANNON');
    place(board, 3, 7, 'CHO', 'SOLDIER');

    const moves = generateLegalMoves(state(board));
    expect(hasMove(moves, 3, 0, 5, 2)).toBe(false);
    expect(hasMove(moves, 4, 1, 5, 2)).toBe(false);
    expect(hasMove(moves, 3, 7, 4, 6)).toBe(false);
    expect(perft(state(board), 1)).toBe(moves.length);

    const openDiagonal = emptyBoard();
    addGenerals(openDiagonal);
    place(openDiagonal, 3, 0, 'CHO', 'CHARIOT');
    place(openDiagonal, 5, 2, 'HAN', 'HORSE');
    expect(hasMove(generateLegalMoves(state(openDiagonal)), 3, 0, 5, 2)).toBe(true);
  });

  it('only includes moves that resolve check', () => {
    const board = emptyBoard();
    place(board, 4, 8, 'CHO', 'GENERAL');
    place(board, 5, 0, 'HAN', 'GENERAL');
    place(board, 4, 4, 'HAN', 'CHARIOT');
    place(board, 0, 6, 'CHO', 'SOLDIER');

    const checked = state(board);
    expect(isInCheck(board, 'CHO')).toBe(true);
    expect(generateLegalMoves(checked).every((move) => !isInCheck(applyMove(checked, move, false).board, 'CHO'))).toBe(true);
    expect(perft(checked, 1)).toBe(generateLegalMoves(checked).length);
  });

  it('returns zero legal nodes from a checkmate position', () => {
    const board = emptyBoard();
    place(board, 3, 0, 'HAN', 'GENERAL');
    place(board, 3, 3, 'CHO', 'CHARIOT');
    place(board, 4, 3, 'CHO', 'CHARIOT');

    expect(isCheckmate(board, 'HAN')).toBe(true);
    expect(perft(state(board, 'HAN'), 1)).toBe(0);
  });
});
