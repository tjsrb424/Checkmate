import { describe, expect, it } from 'vitest';
import {
  Board,
  PieceKind,
  Side,
  applyMove,
  createGameState,
  createInitialBoard,
  emptyBoard,
  generateLegalMoves,
  isCheckmate,
  MATE_SCORE,
  searchBestMove,
  setPiece,
  TranspositionTable
} from './index';
import { builtInOpeningBook, lookupOpeningMoves } from './index';
import { computeZobristHash, hashToKey } from './hash';
import type { OpeningBook } from './openingBook';

function place(board: Board, x: number, y: number, side: Side, kind: PieceKind): void {
  setPiece(board, { x, y }, { side, kind });
}

function hasImmediateMate(board: Board, side: Side): boolean {
  const state = createGameState(board, side);
  return generateLegalMoves(state).some((move) => {
    const next = applyMove(state, move, false);
    return isCheckmate(next.board, next.turn);
  });
}

function immediateMateBoard(): Board {
  const board = emptyBoard();
  place(board, 4, 8, 'CHO', 'GENERAL');
  place(board, 3, 0, 'HAN', 'GENERAL');
  place(board, 3, 3, 'CHO', 'CHARIOT');
  place(board, 4, 4, 'CHO', 'CHARIOT');
  place(board, 8, 4, 'HAN', 'CHARIOT');
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

function nodeCountBoard(): Board {
  const board = emptyBoard();
  place(board, 4, 8, 'CHO', 'GENERAL');
  place(board, 4, 6, 'CHO', 'SOLDIER');
  place(board, 4, 1, 'HAN', 'GENERAL');
  place(board, 4, 3, 'HAN', 'SOLDIER');
  return board;
}

describe('AI search stability', () => {
  it('chooses immediate checkmate when available', () => {
    const state = createGameState(immediateMateBoard(), 'CHO');
    const result = searchBestMove(state, { maxDepth: 1, timeMs: 1000 });

    expect(result.score).toBeGreaterThan(MATE_SCORE - 10);
    expect(result.nodes).toBeGreaterThan(0);

    const next = applyMove(state, result.move!, false);
    expect(isCheckmate(next.board, next.turn)).toBe(true);
  });

  it('avoids moves that allow immediate checkmate', () => {
    const state = createGameState(mateThreatBoard(), 'CHO');
    expect(hasImmediateMate(state.board, 'HAN')).toBe(true);

    const result = searchBestMove(state, { maxDepth: 2, timeMs: 1000 });
    const next = applyMove(state, result.move!, false);

    expect(result.move).not.toBeNull();
    expect(hasImmediateMate(next.board, next.turn)).toBe(false);
  });

  it('reports real visited nodes and completed depth', () => {
    const state = createGameState(nodeCountBoard(), 'CHO');
    const depthOne = searchBestMove(state, { maxDepth: 1, timeMs: 1000 });
    const depthTwo = searchBestMove(state, { maxDepth: 2, timeMs: 1000 });

    expect(depthOne.move).not.toBeNull();
    expect(depthOne.source).toBe('search');
    expect(depthOne.depth).toBe(1);
    expect(depthOne.nodes).toBeGreaterThan(0);
    expect(depthTwo.depth).toBe(2);
    expect(depthTwo.nodes).toBeGreaterThan(depthOne.nodes);
  });

  it('returns principal variation and search stats', () => {
    const state = createGameState(nodeCountBoard(), 'CHO');
    const result = searchBestMove(state, { maxDepth: 2, timeMs: 1000 });

    expect(result.pv.length).toBeGreaterThan(0);
    expect(result.pv[0]).toEqual(result.move);
    expect(result.ttHits + result.ttMisses).toBeGreaterThan(0);
    expect(result.ttStores).toBeGreaterThan(0);
    expect(result.nps).toBeGreaterThan(0);
    expect(result.elapsedMs).toBeGreaterThanOrEqual(0);
    expect(result.qNodes).toBeGreaterThan(0);
    expect(result.qCutoffs).toBeGreaterThanOrEqual(0);
    expect(result.quiescenceEnabled).toBe(true);
  });

  it('gets transposition hits when a table is reused', () => {
    const state = createGameState(createInitialBoard('inner-elephant', 'inner-elephant'), 'CHO');
    const table = new TranspositionTable();

    searchBestMove(state, { maxDepth: 2, timeMs: 1000 }, { table });
    const result = searchBestMove(state, { maxDepth: 2, timeMs: 1000 }, { table });

    expect(result.ttHits).toBeGreaterThan(0);
  });

  it('records alpha-beta cutoffs at deeper depth', () => {
    const state = createGameState(createInitialBoard('inner-elephant', 'inner-elephant'), 'CHO');
    const result = searchBestMove(state, { maxDepth: 3, timeMs: 2000 });

    expect(result.cutoffs).toBeGreaterThan(0);
  });

  it('keeps a fallback move when the search times out before completing a depth', () => {
    const state = createGameState(createInitialBoard('inner-elephant', 'inner-elephant'), 'CHO');
    const result = searchBestMove(state, { maxDepth: 4, timeMs: -1 });

    expect(result.move).not.toBeNull();
    expect(result.depth).toBe(0);
  });

  it('uses an opening book move before searching when available', () => {
    const state = createGameState(createInitialBoard('inner-elephant', 'inner-elephant'), 'CHO');
    const bookMove = lookupOpeningMoves(builtInOpeningBook, state, { minPlayCount: 2 })[0];
    const result = searchBestMove(state, { maxDepth: 2, timeMs: 1000 }, {
      useOpeningBook: true,
      openingBook: builtInOpeningBook,
      openingBookContext: { minPlayCount: 2 },
      maxBookPly: 16
    });

    expect(result.source).toBe('book');
    expect(result.move).toEqual(bookMove.move);
    expect(result.nodes).toBe(0);
    expect(result.pv[0]).toEqual(bookMove.move);
    expect(result.bookMove).toEqual(bookMove);
  });

  it('falls back to search when the opening book is unavailable or too late', () => {
    const state = createGameState(createInitialBoard('inner-elephant', 'inner-elephant'), 'CHO');
    const first = lookupOpeningMoves(builtInOpeningBook, state, { minPlayCount: 2 })[0].move;
    const later = applyMove(state, first, true);

    const noBook = searchBestMove(state, { maxDepth: 1, timeMs: 1000 }, { useOpeningBook: true });
    const tooLate = searchBestMove(later, { maxDepth: 1, timeMs: 1000 }, {
      useOpeningBook: true,
      openingBook: builtInOpeningBook,
      openingBookContext: { minPlayCount: 2 },
      maxBookPly: 0
    });

    expect(noBook.source).toBe('search');
    expect(tooLate.source).toBe('search');
  });

  it('ignores illegal opening book moves and searches normally', () => {
    const state = createGameState(createInitialBoard('inner-elephant', 'inner-elephant'), 'CHO');
    const key = hashToKey(computeZobristHash(state));
    const illegalBook: OpeningBook = {
      positions: {
        [key]: {
          positionKey: key,
          ply: 0,
          turn: 'CHO',
          choFormation: 'inner-elephant',
          hanFormation: 'inner-elephant',
          moves: [
            {
              move: { from: { x: 8, y: 8 }, to: { x: 8, y: 7 } },
              playCount: 10,
              winCount: 10,
              lossCount: 0,
              drawCount: 0,
              scoreRate: 1,
              bookScore: 2,
              sources: ['bad']
            }
          ]
        }
      },
      positionCount: 1,
      moveCount: 1,
      sourceGameCount: 1,
      skippedGameCount: 0,
      illegalMoveCount: 0,
      parseFailureCount: 0,
      createdAt: 'test'
    };

    const result = searchBestMove(state, { maxDepth: 1, timeMs: 1000 }, {
      useOpeningBook: true,
      openingBook: illegalBook,
      openingBookContext: { minPlayCount: 1 }
    });

    expect(result.source).toBe('search');
    expect(result.nodes).toBeGreaterThan(0);
  });
});
