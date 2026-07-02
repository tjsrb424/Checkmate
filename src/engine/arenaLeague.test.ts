import { describe, expect, it } from 'vitest';
import {
  ArenaLeaguePlayer,
  arenaLeagueResultToCsv,
  arenaLeagueResultToJson,
  arenaLeagueResultToMarkdown,
  createDefaultLeaguePlayers,
  runArenaLeague,
  summarizeLeagueStandings
} from './index';

function quickLeaguePlayers(): ArenaLeaguePlayer[] {
  return [
    {
      id: 'normal-search',
      label: 'Normal Search',
      tags: ['normal'],
      engineConfig: {
        id: 'normal-search',
        label: 'Normal Search',
        limits: { maxDepth: 1, timeMs: 200 },
        options: { enableQuiescence: false, enableTransposition: false, useTacticalSafety: false }
      }
    },
    {
      id: 'hard-search',
      label: 'Hard Search',
      tags: ['hard'],
      engineConfig: {
        id: 'hard-search',
        label: 'Hard Search',
        limits: { maxDepth: 1, timeMs: 200 },
        options: { enableQuiescence: false, enableTransposition: false }
      }
    }
  ];
}

describe('arena league', () => {
  it('creates default league players', () => {
    const players = createDefaultLeaguePlayers();

    expect(players.map((player) => player.id)).toContain('normal-search');
    expect(players.map((player) => player.id)).toContain('hard-safety');
    expect(players.map((player) => player.id)).toContain('hard-no-safety');
  });

  it('runs a two-player league and calculates standings', () => {
    const result = runArenaLeague({
      leagueId: 'test-league',
      players: quickLeaguePlayers(),
      gamesPerPair: 1,
      swapSides: true,
      maxPlies: 2,
      recordMoves: false,
      recordSearchStats: false,
      createdAt: 'test'
    });

    expect(result.matches).toHaveLength(1);
    expect(result.games).toHaveLength(2);
    expect(result.report.standings).toHaveLength(2);
    expect(result.report.standings.every((standing) => standing.games === 2)).toBe(true);
    expect(result.report.standings.every((standing) => standing.scoreRate >= 0 && standing.scoreRate <= 1)).toBe(true);
  });

  it('summarizes standings from an existing result', () => {
    const result = runArenaLeague({
      players: quickLeaguePlayers(),
      gamesPerPair: 1,
      swapSides: false,
      maxPlies: 1,
      createdAt: 'test'
    });
    const standings = summarizeLeagueStandings(result);

    expect(standings).toHaveLength(2);
    expect(standings[0].games).toBe(1);
  });

  it('renders markdown standings and warnings', () => {
    const result = runArenaLeague({
      players: quickLeaguePlayers(),
      gamesPerPair: 1,
      swapSides: false,
      maxPlies: 1,
      createdAt: 'test'
    });
    const markdown = arenaLeagueResultToMarkdown(result);

    expect(markdown).toContain('# Arena League Report');
    expect(markdown).toContain('## Standings');
    expect(markdown).toContain('## Warnings');
  });

  it('marks beta readiness false when warning conditions fail', () => {
    const result = runArenaLeague({
      players: quickLeaguePlayers(),
      gamesPerPair: 1,
      swapSides: false,
      maxPlies: 1,
      createdAt: 'test'
    });

    expect(result.report.warnings.length).toBeGreaterThan(0);
    expect(result.report.betaReadiness.ready).toBe(false);
    expect(result.report.betaReadiness.failedChecks.length).toBeGreaterThan(0);
  });

  it('serializes league CSV and JSON', () => {
    const result = runArenaLeague({
      players: quickLeaguePlayers(),
      gamesPerPair: 1,
      swapSides: false,
      maxPlies: 1,
      createdAt: 'test'
    });
    const csv = arenaLeagueResultToCsv(result);
    const parsed = JSON.parse(arenaLeagueResultToJson(result));

    expect(csv).toContain('rank,playerId,label,games,wins,draws,losses,forfeits,score,scoreRate,averagePlies');
    expect(parsed.leagueId).toBe('arena-league');
    expect(parsed.report.totalGames).toBe(1);
  });
});
