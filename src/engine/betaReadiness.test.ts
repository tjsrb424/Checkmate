import { describe, expect, it } from 'vitest';
import {
  evaluateBetaReadiness,
  forbiddenBetaLabels,
  formatBetaReadinessMarkdown
} from './index';
import type { ArenaGameResult, ArenaLeagueResult, ArenaLeagueStanding } from './index';

function standing(playerId: string, scoreRate: number, overrides: Partial<ArenaLeagueStanding> = {}): ArenaLeagueStanding {
  return {
    playerId,
    label: playerId,
    games: 4,
    wins: 2,
    losses: 1,
    draws: 1,
    forfeits: 0,
    score: scoreRate * 4,
    scoreRate,
    averagePlies: 40,
    choWinRate: 0.5,
    hanWinRate: 0.5,
    ...overrides
  };
}

function game(id: string, winner: 'CHO' | 'HAN' | undefined, outcome: ArenaGameResult['outcome'] = winner ? `${winner}_WIN` as ArenaGameResult['outcome'] : 'DRAW'): ArenaGameResult {
  return {
    gameId: id,
    choPlayerId: id.endsWith('2') ? 'normal-search' : 'hard-search',
    hanPlayerId: id.endsWith('2') ? 'hard-search' : 'normal-search',
    choPlayerLabel: 'CHO',
    hanPlayerLabel: 'HAN',
    choFormation: 'inner-elephant',
    hanFormation: 'inner-elephant',
    winner,
    outcome,
    plies: 12,
    history: [],
    searchSummaries: [
      {
        ply: 0,
        side: id.endsWith('2') ? 'HAN' : 'CHO',
        move: null,
        score: 0,
        depth: 4,
        nodes: 1,
        qNodes: 0,
        nps: 1,
        ttHits: 0,
        cutoffs: 0,
        source: 'search'
      }
    ]
  };
}

function league(overrides: Partial<ArenaLeagueResult> = {}): ArenaLeagueResult {
  const base: ArenaLeagueResult = {
    leagueId: 'test',
    matches: [],
    games: [game('g1', 'CHO'), game('g2', 'HAN'), game('g3', 'CHO'), game('g4', 'HAN')],
    report: {
      createdAt: 'test',
      totalGames: 4,
      players: [
        { id: 'hard-search', label: 'Hard Search', tags: ['hard'] },
        { id: 'normal-search', label: 'Normal Search', tags: ['normal'] }
      ],
      settings: {
        gamesPerPair: 2,
        swapSides: true,
        maxPlies: 80,
        recordMoves: false,
        recordSearchStats: true,
        formationPairs: [{ choFormation: 'inner-elephant', hanFormation: 'inner-elephant' }]
      },
      standings: [standing('hard-search', 0.65), standing('normal-search', 0.45)],
      warnings: [],
      betaReadiness: { ready: true, passedChecks: [], failedChecks: [] }
    }
  };
  return { ...base, ...overrides, report: { ...base.report, ...overrides.report } };
}

const passingOptions = {
  blunderRegression: { passed: 4, total: 4 },
  manualSmoke: { candidatesVisible: true, openingBookFallbackOk: true, uiErrorRecoveryOk: true }
};

describe('beta readiness', () => {
  it('returns ready true when all gates pass', () => {
    const result = evaluateBetaReadiness(league(), passingOptions);

    expect(result.ready).toBe(true);
    expect(result.failedChecks).toEqual([]);
    expect(result.recommendedLabel).toBe('AI 대국 베타');
  });

  it('fails when a forfeit exists', () => {
    const badGame = { ...game('g5', 'HAN', 'FORFEIT'), forfeitBy: 'CHO' as const, forfeitReason: 'NO_MOVE' as const };
    const result = evaluateBetaReadiness(league({ games: [badGame] }), passingOptions);

    expect(result.ready).toBe(false);
    expect(result.failedChecks.map((check) => check.id)).toContain('hard-forfeit');
  });

  it('fails when hard is weaker than normal', () => {
    const result = evaluateBetaReadiness(
      league({ report: { ...league().report, standings: [standing('hard-search', 0.2), standing('normal-search', 0.8)] } }),
      passingOptions
    );

    expect(result.ready).toBe(false);
    expect(result.failedChecks.map((check) => check.id)).toContain('hard-vs-normal');
  });

  it('fails or warns when side bias is high', () => {
    const biased = league({ games: [game('g1', 'CHO'), game('g3', 'CHO'), game('g5', 'CHO'), game('g7', 'CHO')] });
    const result = evaluateBetaReadiness(biased, passingOptions);

    expect(result.ready).toBe(false);
    expect(result.failedChecks.map((check) => check.id)).toContain('side-bias');
  });

  it('does not recommend forbidden labels', () => {
    const result = evaluateBetaReadiness(league(), passingOptions);

    expect(forbiddenBetaLabels.some((word) => result.recommendedLabel.includes(word))).toBe(false);
  });

  it('formats markdown with failed checks', () => {
    const result = evaluateBetaReadiness(league({ games: [game('g1', undefined), game('g2', undefined)] }), passingOptions);
    const markdown = formatBetaReadinessMarkdown(result);

    expect(markdown).toContain('# AI Beta Readiness');
    expect(markdown).toContain('## Failed Checks');
    expect(markdown).toContain('drawRate');
  });
});
