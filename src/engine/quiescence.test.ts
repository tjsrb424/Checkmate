import { describe, expect, it } from 'vitest';
import {
  Board,
  PieceKind,
  Side,
  createGameState,
  emptyBoard,
  searchBestMove,
  setPiece
} from './index';

function place(board: Board, x: number, y: number, side: Side, kind: PieceKind): void {
  setPiece(board, { x, y }, { side, kind });
}

function tacticalCaptureBoard(): Board {
  const board = emptyBoard();
  place(board, 4, 8, 'CHO', 'GENERAL');
  place(board, 5, 0, 'HAN', 'GENERAL');
  place(board, 0, 4, 'CHO', 'CHARIOT');
  place(board, 0, 3, 'HAN', 'HORSE');
  place(board, 1, 3, 'HAN', 'SOLDIER');
  return board;
}

function unstableCaptureBoard(): Board {
  const board = emptyBoard();
  place(board, 3, 9, 'CHO', 'GENERAL');
  place(board, 5, 0, 'HAN', 'GENERAL');
  place(board, 2, 5, 'CHO', 'HORSE');
  place(board, 4, 4, 'HAN', 'CHARIOT');
  place(board, 4, 2, 'HAN', 'CHARIOT');
  return board;
}

function quietCheckBoard(): Board {
  const board = emptyBoard();
  place(board, 4, 8, 'CHO', 'GENERAL');
  place(board, 5, 0, 'HAN', 'GENERAL');
  place(board, 0, 6, 'CHO', 'SOLDIER');
  place(board, 0, 3, 'HAN', 'CHARIOT');
  return board;
}

function captureChainBoard(): Board {
  const board = emptyBoard();
  place(board, 4, 8, 'CHO', 'GENERAL');
  place(board, 5, 0, 'HAN', 'GENERAL');
  place(board, 0, 6, 'CHO', 'CHARIOT');
  place(board, 0, 5, 'HAN', 'HORSE');
  place(board, 1, 5, 'CHO', 'SOLDIER');
  place(board, 2, 5, 'HAN', 'SOLDIER');
  place(board, 3, 5, 'CHO', 'SOLDIER');
  return board;
}

describe('quiescence search', () => {
  it('tracks qNodes only when enabled', () => {
    const state = createGameState(tacticalCaptureBoard(), 'CHO');
    const withQ = searchBestMove(state, { maxDepth: 1, timeMs: 1000 }, { enableQuiescence: true });
    const withoutQ = searchBestMove(state, { maxDepth: 1, timeMs: 1000 }, { enableQuiescence: false });

    expect(withQ.quiescenceEnabled).toBe(true);
    expect(withQ.qNodes).toBeGreaterThan(0);
    expect(withoutQ.quiescenceEnabled).toBe(false);
    expect(withoutQ.qNodes).toBe(0);
  });

  it('extends capture sequences at leaf nodes', () => {
    const state = createGameState(unstableCaptureBoard(), 'CHO');
    const withQ = searchBestMove(state, { maxDepth: 1, timeMs: 1000 }, { enableQuiescence: true });
    const withoutQ = searchBestMove(state, { maxDepth: 1, timeMs: 1000 }, { enableQuiescence: false });

    expect(withQ.qNodes).toBeGreaterThan(0);
    expect(withQ.score).not.toBe(withoutQ.score);
  });

  it('reduces overvaluing unstable captures', () => {
    const state = createGameState(unstableCaptureBoard(), 'CHO');
    const withQ = searchBestMove(state, { maxDepth: 1, timeMs: 1000 }, { enableQuiescence: true });
    const withoutQ = searchBestMove(state, { maxDepth: 1, timeMs: 1000 }, { enableQuiescence: false });

    expect(withQ.score).toBeLessThan(withoutQ.score);
  });

  it('includes quiet checks when configured', () => {
    const state = createGameState(quietCheckBoard(), 'CHO');
    const withChecks = searchBestMove(state, { maxDepth: 1, timeMs: 1000 }, { includeQuietChecks: true });
    const withoutChecks = searchBestMove(state, { maxDepth: 1, timeMs: 1000 }, { includeQuietChecks: false });

    expect(withChecks.qNodes).toBeGreaterThan(withoutChecks.qNodes);
  });

  it('respects max quiescence depth', () => {
    const state = createGameState(captureChainBoard(), 'CHO');
    const shallow = searchBestMove(state, { maxDepth: 1, timeMs: 1000 }, { maxQuiescenceDepth: 1 });
    const deeper = searchBestMove(state, { maxDepth: 1, timeMs: 1000 }, { maxQuiescenceDepth: 4 });

    expect(shallow.qNodes).toBeGreaterThan(0);
    expect(deeper.qNodes).toBeGreaterThanOrEqual(shallow.qNodes);
  });
});
