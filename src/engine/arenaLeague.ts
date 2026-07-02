import {
  allFormationPairs,
  defaultFormationPairs,
  runArenaSeries
} from './arena';
import type { ArenaGameResult, ArenaSeriesConfig, ArenaSeriesResult, EnginePlayerConfig, FormationPair } from './arena';
import { builtInOpeningBook } from './openingBookData';

export interface ArenaLeaguePlayer {
  id: string;
  label: string;
  engineConfig: EnginePlayerConfig;
  tags: string[];
}

export interface ArenaLeagueConfig {
  leagueId?: string;
  players?: ArenaLeaguePlayer[];
  gamesPerPair?: number;
  swapSides?: boolean;
  formationPairs?: FormationPair[];
  formationMode?: 'default' | 'all';
  maxPlies?: number;
  recordMoves?: boolean;
  recordSearchStats?: boolean;
  createdAt?: string;
}

export interface ArenaLeagueMatch {
  matchId: string;
  playerAId: string;
  playerBId: string;
  playerALabel: string;
  playerBLabel: string;
  games: number;
  playerAScoreRate: number;
  playerBScoreRate: number;
  result: ArenaSeriesResult;
}

export interface ArenaLeagueStanding {
  playerId: string;
  label: string;
  games: number;
  wins: number;
  losses: number;
  draws: number;
  forfeits: number;
  score: number;
  scoreRate: number;
  averagePlies: number;
  choWinRate: number;
  hanWinRate: number;
}

export interface ArenaLeagueBetaReadiness {
  ready: boolean;
  passedChecks: string[];
  failedChecks: string[];
}

export interface ArenaLeagueReport {
  createdAt: string;
  totalGames: number;
  players: Array<Pick<ArenaLeaguePlayer, 'id' | 'label' | 'tags'>>;
  settings: Required<Pick<ArenaLeagueConfig, 'gamesPerPair' | 'swapSides' | 'maxPlies' | 'recordMoves' | 'recordSearchStats'>> & {
    formationPairs: FormationPair[];
  };
  standings: ArenaLeagueStanding[];
  warnings: string[];
  betaReadiness: ArenaLeagueBetaReadiness;
}

export interface ArenaLeagueResult {
  leagueId: string;
  report: ArenaLeagueReport;
  matches: ArenaLeagueMatch[];
  games: ArenaGameResult[];
}

export function createDefaultLeaguePlayers(): ArenaLeaguePlayer[] {
  return [
    player('normal-search', 'Normal Search', { difficulty: 'normal' }, ['normal', 'search']),
    player('hard-search', 'Hard Search', { difficulty: 'hard' }, ['hard', 'search']),
    player(
      'hard-book',
      'Hard Book',
      {
        difficulty: 'hard',
        useOpeningBook: true,
        openingBook: builtInOpeningBook,
        openingBookContext: { minPlayCount: 2, maxMoves: 5 },
        maxBookPly: 16
      },
      ['hard', 'book']
    ),
    player('hard-no-q', 'Hard No Q', { difficulty: 'hard', options: { enableQuiescence: false } }, ['hard', 'no-q']),
    player('hard-no-safety', 'Hard No Safety', { difficulty: 'hard', options: { useTacticalSafety: false } }, ['hard', 'no-safety']),
    player('hard-safety', 'Hard Safety', { difficulty: 'hard', options: { useTacticalSafety: true } }, ['hard', 'safety'])
  ];
}

export function runArenaLeague(config: ArenaLeagueConfig = {}): ArenaLeagueResult {
  const players = config.players?.length ? config.players : createDefaultLeaguePlayers();
  const gamesPerPair = config.gamesPerPair ?? 2;
  const swapSides = config.swapSides ?? true;
  const maxPlies = config.maxPlies ?? 120;
  const recordMoves = config.recordMoves ?? false;
  const recordSearchStats = config.recordSearchStats ?? false;
  const formationPairs = config.formationPairs?.length
    ? config.formationPairs
    : config.formationMode === 'all'
      ? allFormationPairs()
      : defaultFormationPairs();
  const leagueId = config.leagueId ?? 'arena-league';
  const matches: ArenaLeagueMatch[] = [];

  for (let i = 0; i < players.length; i += 1) {
    for (let j = i + 1; j < players.length; j += 1) {
      const playerA = players[i];
      const playerB = players[j];
      const matchId = `${leagueId}-${playerA.id}-vs-${playerB.id}`;
      const seriesConfig: ArenaSeriesConfig = {
        seriesId: matchId,
        playerA: playerA.engineConfig,
        playerB: playerB.engineConfig,
        games: gamesPerPair * (swapSides ? 2 : 1),
        maxPlies,
        formationPairs,
        swapSides,
        recordMoves,
        recordSearchStats
      };
      const result = runArenaSeries(seriesConfig);
      matches.push({
        matchId,
        playerAId: playerA.id,
        playerBId: playerB.id,
        playerALabel: playerA.label,
        playerBLabel: playerB.label,
        games: result.games,
        playerAScoreRate: result.playerAScoreRate,
        playerBScoreRate: result.playerBScoreRate,
        result
      });
    }
  }

  const games = matches.flatMap((match) => match.result.results);
  const standings = summarizeLeagueStandings({ players, games });
  const warnings = buildWarnings(standings, games);
  const betaReadiness = buildBetaReadiness(standings, games, warnings);
  return {
    leagueId,
    matches,
    games,
    report: {
      createdAt: config.createdAt ?? new Date().toISOString(),
      totalGames: games.length,
      players: players.map(({ id, label, tags }) => ({ id, label, tags })),
      settings: { gamesPerPair, swapSides, maxPlies, recordMoves, recordSearchStats, formationPairs },
      standings,
      warnings,
      betaReadiness
    }
  };
}

export function summarizeLeagueStandings(input: ArenaLeagueResult | { players: ArenaLeaguePlayer[]; games: ArenaGameResult[] }): ArenaLeagueStanding[] {
  const players = 'report' in input
    ? input.report.players.map((entry) => ({ id: entry.id, label: entry.label }))
    : input.players.map((entry) => ({ id: entry.id, label: entry.label }));
  const rows = new Map<string, ArenaLeagueStanding & { choGames: number; hanGames: number }>();

  for (const item of players) {
    rows.set(item.id, {
      playerId: item.id,
      label: item.label,
      games: 0,
      wins: 0,
      losses: 0,
      draws: 0,
      forfeits: 0,
      score: 0,
      scoreRate: 0,
      averagePlies: 0,
      choWinRate: 0,
      hanWinRate: 0,
      choGames: 0,
      hanGames: 0
    });
  }

  for (const game of 'report' in input ? input.games : input.games) {
    applyGameToStanding(rows, game.choPlayerId, game, 'CHO');
    applyGameToStanding(rows, game.hanPlayerId, game, 'HAN');
  }

  return [...rows.values()]
    .map(({ choGames, hanGames, ...row }) => ({
      ...row,
      scoreRate: row.games > 0 ? row.score / row.games : 0,
      averagePlies: row.games > 0 ? row.averagePlies / row.games : 0,
      choWinRate: choGames > 0 ? row.choWinRate / choGames : 0,
      hanWinRate: hanGames > 0 ? row.hanWinRate / hanGames : 0
    }))
    .sort((a, b) => b.scoreRate - a.scoreRate || b.score - a.score || a.label.localeCompare(b.label));
}

export function arenaLeagueResultToMarkdown(result: ArenaLeagueResult): string {
  const lines = [
    '# Arena League Report',
    '',
    `- createdAt: ${result.report.createdAt}`,
    `- totalGames: ${result.report.totalGames}`,
    `- players: ${result.report.players.map((player) => player.id).join(', ')}`,
    `- maxPlies: ${result.report.settings.maxPlies}`,
    '',
    '## Standings',
    '',
    '| Rank | Player | Games | W | D | L | ScoreRate | AvgPlies |',
    '|---:|---|---:|---:|---:|---:|---:|---:|',
    ...result.report.standings.map(
      (standing, index) =>
        `| ${index + 1} | ${standing.label} | ${standing.games} | ${standing.wins} | ${standing.draws} | ${standing.losses} | ${formatRate(
          standing.scoreRate
        )} | ${standing.averagePlies.toFixed(1)} |`
    ),
    '',
    '## Match Results',
    '',
    '| Match | A | B | A ScoreRate | B ScoreRate |',
    '|---|---|---|---:|---:|',
    ...result.matches.map(
      (match) =>
        `| ${match.matchId} | ${match.playerALabel} | ${match.playerBLabel} | ${formatRate(match.playerAScoreRate)} | ${formatRate(
          match.playerBScoreRate
        )} |`
    ),
    '',
    '## Warnings',
    ''
  ];

  lines.push(...(result.report.warnings.length > 0 ? result.report.warnings.map((warning) => `- ${warning}`) : ['- none']));
  lines.push('', '## Beta Readiness', '', `- ready: ${result.report.betaReadiness.ready}`);
  lines.push(...result.report.betaReadiness.failedChecks.map((check) => `- failed: ${check}`));
  lines.push(...result.report.betaReadiness.passedChecks.map((check) => `- passed: ${check}`));
  return lines.join('\n');
}

export function arenaLeagueResultToCsv(result: ArenaLeagueResult): string {
  const header = ['rank', 'playerId', 'label', 'games', 'wins', 'draws', 'losses', 'forfeits', 'score', 'scoreRate', 'averagePlies'];
  const rows = result.report.standings.map((standing, index) =>
    [
      index + 1,
      standing.playerId,
      standing.label,
      standing.games,
      standing.wins,
      standing.draws,
      standing.losses,
      standing.forfeits,
      standing.score,
      standing.scoreRate,
      standing.averagePlies
    ]
      .map(csvEscape)
      .join(',')
  );
  return [header.join(','), ...rows].join('\n');
}

export function arenaLeagueResultToJson(result: ArenaLeagueResult): string {
  return JSON.stringify(result, null, 2);
}

function player(id: string, label: string, config: Omit<EnginePlayerConfig, 'id' | 'label'>, tags: string[]): ArenaLeaguePlayer {
  return { id, label, tags, engineConfig: { id, label, ...config } };
}

function applyGameToStanding(rows: Map<string, ArenaLeagueStanding & { choGames: number; hanGames: number }>, playerId: string, game: ArenaGameResult, side: 'CHO' | 'HAN'): void {
  const row = rows.get(playerId);
  if (!row) return;
  row.games += 1;
  row.averagePlies += game.plies;
  if (side === 'CHO') row.choGames += 1;
  if (side === 'HAN') row.hanGames += 1;
  if (game.outcome === 'DRAW') {
    row.draws += 1;
    row.score += 0.5;
    return;
  }
  if (game.forfeitBy === side) row.forfeits += 1;
  if (game.winner === side) {
    row.wins += 1;
    row.score += 1;
    if (side === 'CHO') row.choWinRate += 1;
    if (side === 'HAN') row.hanWinRate += 1;
  } else {
    row.losses += 1;
  }
}

function buildWarnings(standings: ArenaLeagueStanding[], games: ArenaGameResult[]): string[] {
  const warnings: string[] = [];
  const forfeitCount = games.filter((game) => game.outcome === 'FORFEIT').length;
  const drawRate = games.length > 0 ? games.filter((game) => game.outcome === 'DRAW').length / games.length : 0;
  const decisiveGames = games.filter((game) => game.winner === 'CHO' || game.winner === 'HAN');
  const choWinRate = decisiveGames.length > 0 ? decisiveGames.filter((game) => game.winner === 'CHO').length / decisiveGames.length : 0.5;
  const normal = standings.find((standing) => standing.playerId === 'normal-search');
  const hard = standings.find((standing) => standing.playerId === 'hard-search');
  const hardBook = standings.find((standing) => standing.playerId === 'hard-book');
  const hardSafety = standings.find((standing) => standing.playerId === 'hard-safety');
  const hardNoSafety = standings.find((standing) => standing.playerId === 'hard-no-safety');

  if (forfeitCount > 0) warnings.push(`forfeit count is ${forfeitCount}`);
  if (drawRate > 0.7) warnings.push(`draw rate is high: ${formatRate(drawRate)}`);
  if (decisiveGames.length > 0 && (choWinRate > 0.65 || choWinRate < 0.35)) warnings.push(`CHO/HAN bias detected: CHO win rate ${formatRate(choWinRate)}`);
  if (normal && hard && hard.scoreRate < normal.scoreRate) warnings.push('hard-search scoreRate is below normal-search');
  if (hard && hardBook && hardBook.scoreRate + 0.05 < hard.scoreRate) warnings.push('hard-book scoreRate is materially below hard-search');
  if (hardSafety && hardNoSafety && hardSafety.scoreRate + 0.05 < hardNoSafety.scoreRate) warnings.push('hard-safety scoreRate is materially below hard-no-safety');
  return warnings;
}

function buildBetaReadiness(standings: ArenaLeagueStanding[], games: ArenaGameResult[], warnings: string[]): ArenaLeagueBetaReadiness {
  const passedChecks: string[] = [];
  const failedChecks: string[] = [];
  const normal = standings.find((standing) => standing.playerId === 'normal-search');
  const hard = standings.find((standing) => standing.playerId === 'hard-search');
  const drawRate = games.length > 0 ? games.filter((game) => game.outcome === 'DRAW').length / games.length : 0;
  const decisiveGames = games.filter((game) => game.winner === 'CHO' || game.winner === 'HAN');
  const choWinRate = decisiveGames.length > 0 ? decisiveGames.filter((game) => game.winner === 'CHO').length / decisiveGames.length : 0.5;
  const forfeitCount = games.filter((game) => game.outcome === 'FORFEIT').length;

  addCheck(passedChecks, failedChecks, 'hard >= normal 55% scoreRate', !normal || !hard || hard.scoreRate >= normal.scoreRate * 0.55);
  addCheck(passedChecks, failedChecks, 'forfeit count is zero', forfeitCount === 0);
  addCheck(passedChecks, failedChecks, 'draw rate below 70%', drawRate < 0.7);
  addCheck(passedChecks, failedChecks, 'CHO/HAN bias within 35-65%', decisiveGames.length === 0 || (choWinRate >= 0.35 && choWinRate <= 0.65));
  addCheck(passedChecks, failedChecks, 'no critical warnings', warnings.length === 0);

  return { ready: failedChecks.length === 0, passedChecks, failedChecks };
}

function addCheck(passed: string[], failed: string[], label: string, ok: boolean): void {
  if (ok) passed.push(label);
  else failed.push(label);
}

function formatRate(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function csvEscape(value: unknown): string {
  const text = String(value);
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}
