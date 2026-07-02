import { describe, expect, it } from 'vitest';
import {
  analyzeMoveSafety,
  createGameState,
  detectHangingPieces,
  detectImmediateMateThreat,
  emptyBoard,
  estimateMaterialSwingAfterMove,
  setPiece
} from './index';
import type { Board, Move, PieceKind, Side } from './index';

function place(board: Board, x: number, y: number, side: Side, kind: PieceKind): void {
  setPiece(board, { x, y }, { side, kind });
}

function safetyBaseBoard(): Board {
  const board = emptyBoard();
  place(board, 4, 8, 'CHO', 'GENERAL');
  place(board, 4, 1, 'HAN', 'GENERAL');
  place(board, 4, 5, 'CHO', 'SOLDIER');
  return board;
}

function hangingChariotBoard(): Board {
  const board = safetyBaseBoard();
  place(board, 0, 6, 'CHO', 'CHARIOT');
  place(board, 0, 3, 'HAN', 'CHARIOT');
  return board;
}

function hangingCannonBoard(): Board {
  const board = safetyBaseBoard();
  place(board, 0, 7, 'CHO', 'CANNON');
  place(board, 0, 3, 'HAN', 'CHARIOT');
  return board;
}

function mateThreatBoard(): Board {
  const board = emptyBoard();
  place(board, 3, 9, 'CHO', 'GENERAL');
  place(board, 3, 8, 'CHO', 'GUARD');
  place(board, 2, 6, 'CHO', 'CHARIOT');
  place(board, 8, 8, 'CHO', 'SOLDIER');
  place(board, 4, 1, 'HAN', 'GENERAL');
  place(board, 3, 6, 'HAN', 'CHARIOT');
  place(board, 5, 7, 'HAN', 'HORSE');
  return board;
}

describe('tactical safety', () => {
  it('detects immediate mate threats', () => {
    expect(detectImmediateMateThreat(createGameState(mateThreatBoard(), 'CHO'), 'CHO')).toBe(true);
    expect(detectImmediateMateThreat(createGameState(safetyBaseBoard(), 'CHO'), 'CHO')).toBe(false);
  });

  it('marks a move that allows immediate mate as losing', () => {
    const state = createGameState(mateThreatBoard(), 'CHO');
    const move: Move = { from: { x: 8, y: 8 }, to: { x: 7, y: 8 } };
    const safety = analyzeMoveSafety(state, move);

    expect(safety.allowsImmediateMate).toBe(true);
    expect(safety.riskLevel).toBe('losing');
    expect(safety.risks.map((risk) => risk.reason)).toContain('ALLOWS_IMMEDIATE_MATE');
  });

  it('detects a hanging chariot', () => {
    const state = createGameState(hangingChariotBoard(), 'CHO');
    const hanging = detectHangingPieces(state, 'CHO');

    expect(hanging[0]).toMatchObject({
      piece: { side: 'CHO', kind: 'CHARIOT' }
    });
  });

  it('detects a hanging cannon', () => {
    const state = createGameState(hangingCannonBoard(), 'CHO');
    const safety = analyzeMoveSafety(state, { from: { x: 4, y: 5 }, to: { x: 4, y: 4 } });

    expect(safety.risks.map((risk) => risk.reason)).toContain('HANGS_CANNON');
  });

  it('estimates opponent best capture as negative material swing', () => {
    const state = createGameState(hangingChariotBoard(), 'CHO');
    const swing = estimateMaterialSwingAfterMove(state, { from: { x: 4, y: 5 }, to: { x: 4, y: 4 } });

    expect(swing).toBeLessThan(-1000);
  });

  it('keeps a quiet safe move at a low risk score', () => {
    const state = createGameState(safetyBaseBoard(), 'CHO');
    const safety = analyzeMoveSafety(state, { from: { x: 4, y: 5 }, to: { x: 4, y: 4 } });

    expect(safety.riskLevel).toBe('safe');
    expect(safety.riskScore).toBeLessThan(300);
  });
});
