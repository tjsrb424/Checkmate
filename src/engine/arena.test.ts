import { describe, expect, it } from 'vitest';
import {
  ArenaGameResult,
  ArenaPlayer,
  arenaResultToJson,
  arenaResultsToCsv,
  createSearchEnginePlayer,
  runArenaGame,
  runArenaSeries,
  summarizeArenaResults,
  builtInOpeningBook
} from './index';

function quickPlayer(id: string): ArenaPlayer {
  return createSearchEnginePlayer({
    id,
    label: id,
    limits: { maxDepth: 1, timeMs: 100 },
    options: { enableQuiescence: false, enableTransposition: false }
  });
}

function nullMovePlayer(id: string): ArenaPlayer {
  return {
    id,
    label: id,
    chooseMove() {
      return {
        move: null,
        score: 0,
        depth: 0,
        nodes: 0,
        pv: [],
        ttHits: 0,
        ttMisses: 0,
        ttStores: 0,
        cutoffs: 0,
        nps: 0,
        elapsedMs: 0,
        qNodes: 0,
        qCutoffs: 0,
        quiescenceEnabled: false,
        source: 'search'
      };
    }
  };
}

function illegalMovePlayer(id: string): ArenaPlayer {
  const player = nullMovePlayer(id);
  return {
    ...player,
    chooseMove() {
      return {
        ...player.chooseMove({ board: [], turn: 'CHO', history: [] }),
        move: { from: { x: 8, y: 8 }, to: { x: 8, y: 7 } }
      };
    }
  };
}

describe('AI arena', () => {
  it('runs a single arena game', () => {
    const result = runArenaGame({
      gameId: 'game-1',
      choPlayer: quickPlayer('cho'),
      hanPlayer: quickPlayer('han'),
      maxPlies: 2
    });

    expect(['CHO_WIN', 'HAN_WIN', 'DRAW', 'FORFEIT']).toContain(result.outcome);
    expect(result.plies).toBeGreaterThanOrEqual(0);
    expect(result.plies).toBeLessThanOrEqual(2);
  });

  it('draws when max plies is reached without a winner', () => {
    const result = runArenaGame({
      gameId: 'draw-1',
      choPlayer: quickPlayer('cho'),
      hanPlayer: quickPlayer('han'),
      maxPlies: 1
    });

    if (!result.winner) {
      expect(result.outcome).toBe('DRAW');
      expect(result.plies).toBe(1);
    }
  });

  it('swaps sides across a series', () => {
    const result = runArenaSeries({
      seriesId: 'swap',
      playerA: { id: 'a', label: 'A', limits: { maxDepth: 1, timeMs: 100 }, options: { enableQuiescence: false } },
      playerB: { id: 'b', label: 'B', limits: { maxDepth: 1, timeMs: 100 }, options: { enableQuiescence: false } },
      games: 2,
      maxPlies: 1,
      swapSides: true
    });

    expect(result.results[0].choPlayerId).toBe('a');
    expect(result.results[0].hanPlayerId).toBe('b');
    expect(result.results[1].choPlayerId).toBe('b');
    expect(result.results[1].hanPlayerId).toBe('a');
  });

  it('summarizes wins, draws, and score rates by player', () => {
    const games: ArenaGameResult[] = [
      resultStub('g1', 'a', 'b', 'CHO_WIN', 'CHO'),
      resultStub('g2', 'b', 'a', 'HAN_WIN', 'HAN'),
      resultStub('g3', 'a', 'b', 'DRAW')
    ];
    const summary = summarizeArenaResults(games, 'manual', { id: 'a', label: 'A' }, { id: 'b', label: 'B' });

    expect(summary.playerAWins).toBe(2);
    expect(summary.playerBWins).toBe(0);
    expect(summary.draws).toBe(1);
    expect(summary.playerAScoreRate).toBeCloseTo(2.5 / 3);
    expect(summary.playerBScoreRate).toBeCloseTo(0.5 / 3);
    expect(summary.choWins).toBe(1);
    expect(summary.hanWins).toBe(1);
  });

  it('forfeits when a player returns an illegal move', () => {
    const result = runArenaGame({
      gameId: 'illegal',
      choPlayer: illegalMovePlayer('bad'),
      hanPlayer: quickPlayer('han'),
      maxPlies: 2
    });

    expect(result.outcome).toBe('FORFEIT');
    expect(result.forfeitBy).toBe('CHO');
    expect(result.forfeitReason).toBe('ILLEGAL_MOVE');
    expect(result.winner).toBe('HAN');
  });

  it('forfeits when a player returns null move', () => {
    const result = runArenaGame({
      gameId: 'null',
      choPlayer: nullMovePlayer('bad'),
      hanPlayer: quickPlayer('han'),
      maxPlies: 2
    });

    expect(result.outcome).toBe('FORFEIT');
    expect(result.forfeitBy).toBe('CHO');
    expect(result.forfeitReason).toBe('NO_MOVE');
    expect(result.winner).toBe('HAN');
  });

  it('serializes arena results to CSV', () => {
    const summary = summarizeArenaResults([resultStub('g1', 'a', 'b', 'DRAW')], 'csv', { id: 'a', label: 'A' }, { id: 'b', label: 'B' });
    const csv = arenaResultsToCsv(summary);

    expect(csv).toContain('gameId,choPlayer,hanPlayer,choFormation,hanFormation,outcome,winner,plies,forfeitBy,forfeitReason');
    expect(csv).toContain('g1');
    expect(csv).toContain('DRAW');
    expect(csv).toContain(',10,');
  });

  it('serializes arena results to parseable JSON', () => {
    const summary = summarizeArenaResults([resultStub('g1', 'a', 'b', 'DRAW')], 'json', { id: 'a', label: 'A' }, { id: 'b', label: 'B' });
    const parsed = JSON.parse(arenaResultToJson(summary));

    expect(parsed.seriesId).toBe('json');
    expect(parsed.results[0].gameId).toBe('g1');
  });

  it('records opening book source in search summaries', () => {
    const result = runArenaGame({
      gameId: 'book',
      choPlayer: createSearchEnginePlayer({
        id: 'book-on',
        label: 'Book ON',
        limits: { maxDepth: 1, timeMs: 100 },
        options: {
          enableQuiescence: false,
          useOpeningBook: true,
          openingBook: builtInOpeningBook,
          openingBookContext: { minPlayCount: 2 },
          maxBookPly: 16
        }
      }),
      hanPlayer: quickPlayer('han'),
      maxPlies: 1,
      recordSearchStats: true
    });

    expect(result.searchSummaries?.[0].source).toBe('book');
    expect(result.searchSummaries?.[0].bookPlayCount).toBeGreaterThan(0);
  });
});

function resultStub(
  gameId: string,
  choPlayerId: string,
  hanPlayerId: string,
  outcome: ArenaGameResult['outcome'],
  winner?: ArenaGameResult['winner']
): ArenaGameResult {
  return {
    gameId,
    choPlayerId,
    hanPlayerId,
    choPlayerLabel: choPlayerId.toUpperCase(),
    hanPlayerLabel: hanPlayerId.toUpperCase(),
    choFormation: 'inner-elephant',
    hanFormation: 'inner-elephant',
    outcome,
    winner,
    plies: 10,
    history: []
  };
}
