import type { ArenaGameResult } from './arena';
import type { ArenaLeagueResult, ArenaLeagueStanding } from './arenaLeague';

export interface BetaReadinessCheck {
  id: string;
  label: string;
  passed: boolean;
  details: string;
  required: boolean;
}

export interface BetaReadinessResult {
  ready: boolean;
  score: number;
  passedChecks: BetaReadinessCheck[];
  failedChecks: BetaReadinessCheck[];
  warnings: string[];
  recommendedLabel: 'AI 대국 베타' | '초급 AI' | '실험적 AI' | '실전 투입 보류';
}

export interface BlunderRegressionSummary {
  passed: number;
  total: number;
}

export interface ManualSmokeSummary {
  candidatesVisible?: boolean;
  openingBookFallbackOk?: boolean;
  uiErrorRecoveryOk?: boolean;
}

export interface EvaluateBetaReadinessOptions {
  blunderRegression?: BlunderRegressionSummary;
  manualSmoke?: ManualSmokeSummary;
}

export const forbiddenBetaLabels = ['고수', '프로', '최강', '마스터', '알파고'];

export const defaultBetaReadinessRules = [
  'hard scoreRate >= normal scoreRate * 0.55',
  'hard forfeits = 0',
  'illegal move forfeits = 0',
  'drawRate < 0.70',
  'CHO decisive win rate between 0.35 and 0.65',
  'blunder regression passed',
  'hard average completed depth >= 4',
  'candidate Top N visible',
  'opening book fallback OK',
  'UI error recovery OK'
];

export function evaluateBetaReadiness(league: ArenaLeagueResult, options: EvaluateBetaReadinessOptions = {}): BetaReadinessResult {
  const hard = findStanding(league, 'hard-search');
  const normal = findStanding(league, 'normal-search');
  const hardGames = league.games.filter((game) => game.choPlayerId === 'hard-search' || game.hanPlayerId === 'hard-search');
  const illegalForfeits = league.games.filter((game) => game.outcome === 'FORFEIT' && game.forfeitReason === 'ILLEGAL_MOVE').length;
  const drawRate = league.games.length > 0 ? league.games.filter((game) => game.outcome === 'DRAW').length / league.games.length : 0;
  const decisiveGames = league.games.filter((game) => game.winner === 'CHO' || game.winner === 'HAN');
  const choWinRate =
    decisiveGames.length > 0 ? decisiveGames.filter((game) => game.winner === 'CHO').length / decisiveGames.length : 0.5;
  const hardAverageDepth = averageHardDepth(league.games);
  const blunder = options.blunderRegression;
  const smoke = options.manualSmoke;

  const checks: BetaReadinessCheck[] = [
    check(
      'hard-vs-normal',
      'hard가 normal 대비 Arena scoreRate 55% 이상',
      Boolean(!hard || !normal || hard.scoreRate >= normal.scoreRate * 0.55),
      `hard=${formatRate(hard?.scoreRate ?? 0)}, normal=${formatRate(normal?.scoreRate ?? 0)}`
    ),
    check('hard-forfeit', 'hard forfeit 0', hardGames.every((game) => !isForfeitByPlayer(game, 'hard-search')), `${hardGames.length} hard games`),
    check('illegal-forfeit', 'illegal move forfeit 0', illegalForfeits === 0, `illegal forfeits=${illegalForfeits}`),
    check('draw-rate', 'drawRate 70% 미만', drawRate < 0.7, `drawRate=${formatRate(drawRate)}`),
    check(
      'side-bias',
      'CHO/HAN 승률 편향 35%~65%',
      decisiveGames.length === 0 || (choWinRate >= 0.35 && choWinRate <= 0.65),
      decisiveGames.length === 0 ? 'no decisive games' : `CHO decisive winRate=${formatRate(choWinRate)}`
    ),
    check(
      'blunder-regression',
      'blunder regression fixture 전체 통과',
      Boolean(blunder && blunder.total > 0 && blunder.passed === blunder.total),
      blunder ? `${blunder.passed}/${blunder.total}` : 'not provided'
    ),
    check(
      'hard-depth',
      'hard completedDepth 평균 4 이상',
      hardAverageDepth >= 4,
      Number.isFinite(hardAverageDepth) ? `avgDepth=${hardAverageDepth.toFixed(2)}` : 'not measured'
    ),
    check('candidate-ui', 'AI 후보수 Top N 정상 표시', smoke?.candidatesVisible === true, smoke ? String(smoke.candidatesVisible) : 'not provided'),
    check(
      'opening-fallback',
      '오프닝북 ON/OFF fallback 정상',
      smoke?.openingBookFallbackOk === true,
      smoke ? String(smoke.openingBookFallbackOk) : 'not provided'
    ),
    check(
      'ui-error-recovery',
      'UI에서 AI 오류 발생 시 대국이 깨지지 않음',
      smoke?.uiErrorRecoveryOk === true,
      smoke ? String(smoke.uiErrorRecoveryOk) : 'not provided'
    )
  ];

  const warnings = [...league.report.warnings];
  if (!blunder) warnings.push('blunder regression summary was not provided');
  if (!smoke) warnings.push('manual UI smoke summary was not provided');
  if (!Number.isFinite(hardAverageDepth)) warnings.push('hard average completed depth was not measured');

  const passedChecks = checks.filter((item) => item.passed);
  const failedChecks = checks.filter((item) => !item.passed && item.required);
  const score = checks.length > 0 ? passedChecks.length / checks.length : 0;
  const ready = failedChecks.length === 0;
  return {
    ready,
    score,
    passedChecks,
    failedChecks,
    warnings,
    recommendedLabel: recommendLabel(ready, score)
  };
}

export function formatBetaReadinessMarkdown(result: BetaReadinessResult): string {
  return [
    '# AI Beta Readiness',
    '',
    `- ready: ${result.ready}`,
    `- score: ${formatRate(result.score)}`,
    `- recommendedLabel: ${result.recommendedLabel}`,
    '',
    '## Passed Checks',
    '',
    ...(result.passedChecks.length > 0 ? result.passedChecks.map((item) => `- ${item.label}: ${item.details}`) : ['- none']),
    '',
    '## Failed Checks',
    '',
    ...(result.failedChecks.length > 0 ? result.failedChecks.map((item) => `- ${item.label}: ${item.details}`) : ['- none']),
    '',
    '## Warnings',
    '',
    ...(result.warnings.length > 0 ? result.warnings.map((warning) => `- ${warning}`) : ['- none'])
  ].join('\n');
}

function check(id: string, label: string, passed: boolean, details: string, required = true): BetaReadinessCheck {
  return { id, label, passed, details, required };
}

function findStanding(league: ArenaLeagueResult, playerId: string): ArenaLeagueStanding | undefined {
  return league.report.standings.find((standing) => standing.playerId === playerId);
}

function isForfeitByPlayer(game: ArenaGameResult, playerId: string): boolean {
  if (game.outcome !== 'FORFEIT') return false;
  if (game.forfeitBy === 'CHO') return game.choPlayerId === playerId;
  if (game.forfeitBy === 'HAN') return game.hanPlayerId === playerId;
  return false;
}

function averageHardDepth(games: ArenaGameResult[]): number {
  const depths = games.flatMap((game) => {
    const hardSides = new Set<string>();
    if (game.choPlayerId === 'hard-search') hardSides.add('CHO');
    if (game.hanPlayerId === 'hard-search') hardSides.add('HAN');
    return (game.searchSummaries ?? [])
      .filter((summary) => hardSides.has(summary.side))
      .map((summary) => summary.depth);
  });
  return depths.length > 0 ? depths.reduce((sum, depth) => sum + depth, 0) / depths.length : Number.NaN;
}

function recommendLabel(ready: boolean, score: number): BetaReadinessResult['recommendedLabel'] {
  if (ready) return 'AI 대국 베타';
  if (score >= 0.8) return '초급 AI';
  if (score >= 0.5) return '실험적 AI';
  return '실전 투입 보류';
}

function formatRate(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}
