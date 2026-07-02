import { describe, expect, it } from 'vitest';
import {
  analyzeGame,
  analyzePosition,
  createGameState,
  createInitialBoard,
  detectBlunders,
  emptyBoard,
  generateLegalMoves,
  searchBestMove,
  setPiece
} from './index';
import { lostGameFixtures } from './__fixtures__/lostGames';
import type { ScoreTimelineEntry } from './analysis';
import type { Board, PieceKind, Side } from './index';

function place(board: Board, x: number, y: number, side: Side, kind: PieceKind): void {
  setPiece(board, { x, y }, { side, kind });
}

function hangingChariotBoard(): Board {
  const board = emptyBoard();
  place(board, 4, 8, 'CHO', 'GENERAL');
  place(board, 4, 1, 'HAN', 'GENERAL');
  place(board, 4, 5, 'CHO', 'SOLDIER');
  place(board, 0, 6, 'CHO', 'CHARIOT');
  place(board, 0, 3, 'HAN', 'CHARIOT');
  return board;
}

describe('AI analysis tools', () => {
  it('analyzes a position with best move, score, and candidates', () => {
    const state = createGameState(createInitialBoard('inner-elephant', 'inner-elephant'), 'CHO');
    const analysis = analyzePosition(state, { limits: { maxDepth: 1, timeMs: 2500 }, maxCandidates: 4 });

    expect(analysis.bestMove).not.toBeNull();
    expect(analysis.depth).toBe(1);
    expect(analysis.nodes).toBeGreaterThan(0);
    expect(analysis.candidates.length).toBeGreaterThan(0);
    expect(analysis.candidates.length).toBeLessThanOrEqual(4);
    expect(analysis.candidates[0].move).toEqual(analysis.bestMove);
  });

  it('builds a score timeline for a legal game history', () => {
    const initialBoard = createInitialBoard('inner-elephant', 'inner-elephant');
    const state = createGameState(initialBoard, 'CHO');
    const firstMove = generateLegalMoves(state)[0];
    const analysis = analyzeGame(initialBoard, [firstMove], { limits: { maxDepth: 1, timeMs: 2500 }, maxCandidates: 2 });

    expect(analysis.error).toBeUndefined();
    expect(analysis.history).toHaveLength(1);
    expect(analysis.positions).toHaveLength(1);
    expect(analysis.scoreTimeline).toHaveLength(1);
    expect(analysis.scoreTimeline[0]).toMatchObject({
      ply: 1,
      side: 'CHO',
      move: firstMove
    });
  });

  it('detects mistake, blunder, and losing blunder severities', () => {
    const move = { from: { x: 0, y: 6 }, to: { x: 0, y: 5 } };
    const timeline: ScoreTimelineEntry[] = [
      { ply: 1, side: 'CHO', move, bestMove: move, scoreBefore: 500, scoreAfter: 199, loss: 301 },
      { ply: 2, side: 'HAN', move, bestMove: move, scoreBefore: 800, scoreAfter: 99, loss: 701 },
      { ply: 3, side: 'CHO', move, bestMove: move, scoreBefore: 1300, scoreAfter: 99, loss: 1201 }
    ];

    expect(detectBlunders(timeline).map((blunder) => blunder.severity)).toEqual([
      'mistake',
      'blunder',
      'losing-blunder'
    ]);
  });

  it('returns partial analysis for illegal history', () => {
    const initialBoard = createInitialBoard('inner-elephant', 'inner-elephant');
    const analysis = analyzeGame(
      initialBoard,
      [{ from: { x: 8, y: 8 }, to: { x: 8, y: 7 } }],
      { limits: { maxDepth: 1, timeMs: 1000 } }
    );

    expect(analysis.error).toContain('Illegal move at ply 1');
    expect(analysis.illegalPly).toBe(1);
    expect(analysis.history).toEqual([]);
  });

  it('keeps synthetic blunder fixtures available for regression coverage', () => {
    expect(lostGameFixtures.map((fixture) => fixture.riskKind)).toEqual([
      'free-chariot',
      'free-cannon',
      'allows-mate'
    ]);
  });

  it('does not choose the synthetic free-chariot blunder after tactical safety', () => {
    const state = createGameState(hangingChariotBoard(), 'CHO');
    const result = searchBestMove(state, { maxDepth: 1, timeMs: 1000 }, { maxCandidates: 50 });

    expect(result.selectedMoveSafety?.risks.some((risk) => risk.reason === 'HANGS_CHARIOT')).toBe(false);
  });
});
