import { describe, expect, it } from 'vitest';
import {
  Board,
  EvaluationBreakdown,
  PieceKind,
  Side,
  createGameState,
  emptyBoard,
  evaluatePosition,
  evaluatePositionBreakdown,
  setPiece
} from './index';

function place(board: Board, x: number, y: number, side: Side, kind: PieceKind): void {
  setPiece(board, { x, y }, { side, kind });
}

function baseBoard(): Board {
  const board = emptyBoard();
  place(board, 4, 8, 'CHO', 'GENERAL');
  place(board, 4, 1, 'HAN', 'GENERAL');
  place(board, 4, 5, 'CHO', 'SOLDIER');
  return board;
}

function withoutTotal(breakdown: EvaluationBreakdown): number {
  return (
    breakdown.material +
    breakdown.positional +
    breakdown.mobility +
    breakdown.kingSafety +
    breakdown.attackPressure +
    breakdown.chariotActivity +
    breakdown.cannonActivity +
    breakdown.horseElephantActivity +
    breakdown.soldierStructure +
    breakdown.checkPressure +
    breakdown.tacticalSafety
  );
}

describe('evaluation breakdown', () => {
  it('keeps total equal to the sum of all evaluation terms', () => {
    const state = createGameState(baseBoard(), 'CHO');
    const breakdown = evaluatePositionBreakdown(state, 'CHO');

    expect(breakdown.total).toBe(withoutTotal(breakdown));
    expect(evaluatePosition(state, 'CHO')).toBe(breakdown.total);
  });

  it('is symmetric by side on the same position', () => {
    const state = createGameState(baseBoard(), 'CHO');

    expect(evaluatePosition(state, 'CHO')).toBe(-evaluatePosition(state, 'HAN'));
  });

  it('reflects a chariot advantage in material', () => {
    const board = baseBoard();
    place(board, 0, 4, 'CHO', 'CHARIOT');
    const state = createGameState(board, 'CHO');

    expect(evaluatePositionBreakdown(state, 'CHO').material).toBeGreaterThan(0);
    expect(evaluatePositionBreakdown(state, 'HAN').material).toBeLessThan(0);
  });

  it('penalizes a side whose general is in check', () => {
    const board = emptyBoard();
    place(board, 4, 8, 'CHO', 'GENERAL');
    place(board, 3, 1, 'HAN', 'GENERAL');
    place(board, 4, 4, 'HAN', 'CHARIOT');
    const state = createGameState(board, 'CHO');
    const breakdown = evaluatePositionBreakdown(state, 'CHO');

    expect(breakdown.kingSafety + breakdown.checkPressure).toBeLessThan(0);
  });

  it('scores open positions with more mobility than blocked positions', () => {
    const open = baseBoard();
    place(open, 0, 5, 'CHO', 'CHARIOT');
    place(open, 8, 1, 'HAN', 'CHARIOT');

    const blocked = baseBoard();
    place(blocked, 0, 5, 'CHO', 'CHARIOT');
    place(blocked, 0, 4, 'CHO', 'SOLDIER');
    place(blocked, 1, 5, 'CHO', 'SOLDIER');
    place(blocked, 8, 1, 'HAN', 'CHARIOT');

    expect(evaluatePositionBreakdown(createGameState(open, 'CHO'), 'CHO').mobility).toBeGreaterThan(
      evaluatePositionBreakdown(createGameState(blocked, 'CHO'), 'CHO').mobility
    );
  });

  it('scores an open chariot higher than a blocked chariot', () => {
    const open = baseBoard();
    place(open, 0, 5, 'CHO', 'CHARIOT');

    const blocked = baseBoard();
    place(blocked, 0, 5, 'CHO', 'CHARIOT');
    place(blocked, 0, 4, 'CHO', 'SOLDIER');
    place(blocked, 1, 5, 'CHO', 'SOLDIER');

    expect(evaluatePositionBreakdown(createGameState(open, 'CHO'), 'CHO').chariotActivity).toBeGreaterThan(
      evaluatePositionBreakdown(createGameState(blocked, 'CHO'), 'CHO').chariotActivity
    );
  });

  it('rewards a cannon that can attack a major piece over a screen', () => {
    const active = baseBoard();
    place(active, 0, 7, 'CHO', 'CANNON');
    place(active, 0, 5, 'CHO', 'SOLDIER');
    place(active, 0, 2, 'HAN', 'CHARIOT');

    const idle = baseBoard();
    place(idle, 0, 7, 'CHO', 'CANNON');

    expect(evaluatePositionBreakdown(createGameState(active, 'CHO'), 'CHO').cannonActivity).toBeGreaterThan(
      evaluatePositionBreakdown(createGameState(idle, 'CHO'), 'CHO').cannonActivity
    );
  });

  it('rewards advanced soldiers and enemy palace entry', () => {
    const advanced = baseBoard();
    place(advanced, 4, 1, 'CHO', 'SOLDIER');

    const rear = baseBoard();
    place(rear, 0, 6, 'CHO', 'SOLDIER');

    expect(evaluatePositionBreakdown(createGameState(advanced, 'CHO'), 'CHO').soldierStructure).toBeGreaterThan(
      evaluatePositionBreakdown(createGameState(rear, 'CHO'), 'CHO').soldierStructure
    );
  });

  it('scores a developed horse or elephant above a blocked home piece', () => {
    const developed = baseBoard();
    place(developed, 4, 5, 'CHO', 'HORSE');

    const blocked = baseBoard();
    place(blocked, 1, 9, 'CHO', 'HORSE');
    place(blocked, 2, 9, 'CHO', 'SOLDIER');
    place(blocked, 1, 8, 'CHO', 'SOLDIER');

    expect(evaluatePositionBreakdown(createGameState(developed, 'CHO'), 'CHO').horseElephantActivity).toBeGreaterThan(
      evaluatePositionBreakdown(createGameState(blocked, 'CHO'), 'CHO').horseElephantActivity
    );
  });

  it('penalizes a hanging chariot in tactical safety', () => {
    const board = baseBoard();
    place(board, 0, 6, 'CHO', 'CHARIOT');
    place(board, 1, 6, 'HAN', 'SOLDIER');
    const state = createGameState(board, 'CHO');

    expect(evaluatePositionBreakdown(state, 'CHO').tacticalSafety).toBeLessThan(0);
  });

  it('rewards an enemy hanging chariot in tactical safety', () => {
    const board = baseBoard();
    place(board, 0, 6, 'HAN', 'CHARIOT');
    place(board, 1, 6, 'CHO', 'SOLDIER');
    const state = createGameState(board, 'CHO');

    expect(evaluatePositionBreakdown(state, 'CHO').tacticalSafety).toBeGreaterThan(0);
  });

  it('strongly penalizes immediate mate threats', () => {
    const board = emptyBoard();
    place(board, 3, 9, 'CHO', 'GENERAL');
    place(board, 3, 8, 'CHO', 'GUARD');
    place(board, 2, 6, 'CHO', 'CHARIOT');
    place(board, 8, 8, 'CHO', 'SOLDIER');
    place(board, 4, 1, 'HAN', 'GENERAL');
    place(board, 3, 6, 'HAN', 'CHARIOT');
    place(board, 5, 7, 'HAN', 'HORSE');
    const state = createGameState(board, 'CHO');

    expect(evaluatePositionBreakdown(state, 'CHO').tacticalSafety).toBeLessThan(-4000);
  });
});
