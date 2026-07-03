import {
  Difficulty,
  Formation,
  GameState,
  Move,
  SearchLimits,
  Side
} from './types';
import { applyMove, createGameState, isLegalMove } from './rules';
import { createInitialBoard } from './setup';
import { difficultyLimits, searchBestMove } from './ai';
import type { SearchOptions, SearchResult } from './ai';
import type { OpeningBook, OpeningBookLookupOptions } from './openingBook';
import type { JanggiRuleset, RulesetId } from './ruleset';
import { resolveRuleset } from './ruleset';
import { scoreBoardMaterial } from './scoring';

export type ArenaGameOutcome = 'CHO_WIN' | 'HAN_WIN' | 'DRAW' | 'FORFEIT';
export type ArenaForfeitReason = 'NO_MOVE' | 'ILLEGAL_MOVE' | 'SEARCH_ERROR';

export interface EnginePlayerConfig {
  id: string;
  label: string;
  difficulty?: Difficulty;
  limits?: SearchLimits;
  options?: SearchOptions;
  sidePreference?: Side;
  useOpeningBook?: boolean;
  openingBook?: OpeningBook;
  openingBookContext?: OpeningBookLookupOptions;
  maxBookPly?: number;
}

export interface ArenaPlayer {
  id: string;
  label: string;
  chooseMove(state: GameState): SearchResult;
}

export interface ArenaGameConfig {
  gameId: string;
  choPlayer: ArenaPlayer;
  hanPlayer: ArenaPlayer;
  choFormation?: Formation;
  hanFormation?: Formation;
  startingTurn?: Side;
  maxPlies?: number;
  recordMoves?: boolean;
  recordSearchStats?: boolean;
  ruleset?: RulesetId | JanggiRuleset;
}

export interface ArenaSeriesConfig {
  seriesId: string;
  playerA: EnginePlayerConfig | ArenaPlayer;
  playerB: EnginePlayerConfig | ArenaPlayer;
  games: number;
  maxPlies?: number;
  formationPairs?: FormationPair[];
  swapSides?: boolean;
  recordMoves?: boolean;
  recordSearchStats?: boolean;
  ruleset?: RulesetId | JanggiRuleset;
}

export interface FormationPair {
  choFormation: Formation;
  hanFormation: Formation;
}

export interface ArenaMoveSummary {
  ply: number;
  side: Side;
  move: Move | null;
  score: number;
  depth: number;
  nodes: number;
  qNodes: number;
  nps: number;
  ttHits: number;
  cutoffs: number;
  source: 'book' | 'search';
  bookPlayCount?: number;
  bookScoreRate?: number;
  bookScore?: number;
}

export interface ArenaGameResult {
  gameId: string;
  choPlayerId: string;
  hanPlayerId: string;
  choPlayerLabel: string;
  hanPlayerLabel: string;
  choFormation: Formation;
  hanFormation: Formation;
  winner?: Side;
  outcome: ArenaGameOutcome;
  forfeitBy?: Side;
  forfeitReason?: ArenaForfeitReason;
  plies: number;
  history: Move[];
  finalScore?: number;
  searchSummaries?: ArenaMoveSummary[];
}

export interface ArenaSeriesResult {
  seriesId: string;
  games: number;
  playerAId: string;
  playerBId: string;
  playerALabel: string;
  playerBLabel: string;
  playerAWins: number;
  playerBWins: number;
  draws: number;
  forfeits: number;
  playerAScoreRate: number;
  playerBScoreRate: number;
  choWins: number;
  hanWins: number;
  averagePlies: number;
  results: ArenaGameResult[];
}

const formations: Formation[] = ['inner-elephant', 'outer-elephant', 'left-elephant', 'right-elephant'];

export function createSearchEnginePlayer(config: EnginePlayerConfig): ArenaPlayer {
  return {
    id: config.id,
    label: config.label,
    chooseMove(state: GameState): SearchResult {
      const limits = config.limits ?? difficultyLimits[config.difficulty ?? 'normal'];
      return searchBestMove(state, limits, {
        ...config.options,
        useOpeningBook: config.useOpeningBook ?? config.options?.useOpeningBook,
        openingBook: config.openingBook ?? config.options?.openingBook,
        openingBookContext: config.openingBookContext ?? config.options?.openingBookContext,
        maxBookPly: config.maxBookPly ?? config.options?.maxBookPly
      });
    }
  };
}

export function runArenaGame(config: ArenaGameConfig): ArenaGameResult {
  const choFormation = config.choFormation ?? 'inner-elephant';
  const hanFormation = config.hanFormation ?? 'inner-elephant';
  const maxPlies = config.maxPlies ?? 200;
  const recordMoves = config.recordMoves !== false;
  const recordSearchStats = config.recordSearchStats === true;
  const ruleset = resolveRuleset(config.ruleset);
  let state = createGameState(createInitialBoard(choFormation, hanFormation), config.startingTurn ?? 'CHO');
  const searchSummaries: ArenaMoveSummary[] = [];
  let finalScore: number | undefined;

  for (let ply = 0; ply < maxPlies; ply += 1) {
    const side = state.turn;
    const player = side === 'CHO' ? config.choPlayer : config.hanPlayer;
    let result: SearchResult;

    try {
      result = player.chooseMove(state);
    } catch {
      return buildForfeitResult(config, state, choFormation, hanFormation, side, 'SEARCH_ERROR', recordMoves, searchSummaries, finalScore);
    }

    finalScore = result.score;
    if (recordSearchStats) {
      searchSummaries.push(searchResultToSummary(ply, side, result));
    }

    if (!result.move) {
      return buildForfeitResult(config, state, choFormation, hanFormation, side, 'NO_MOVE', recordMoves, searchSummaries, finalScore);
    }

    if (!isLegalMove(state, result.move, ruleset)) {
      return buildForfeitResult(config, state, choFormation, hanFormation, side, 'ILLEGAL_MOVE', recordMoves, searchSummaries, finalScore);
    }

    state = applyMove(state, result.move, true);
    if (state.winner) {
      return buildGameResult(config, state, choFormation, hanFormation, state.winner === 'CHO' ? 'CHO_WIN' : 'HAN_WIN', {
        winner: state.winner,
        history: recordMoves ? state.history : [],
        finalScore,
        searchSummaries: recordSearchStats ? searchSummaries : undefined
      });
    }
  }

  const material = ruleset.maxPlyPolicy === 'score-adjudication' ? scoreBoardMaterial(state.board) : undefined;
  const winner = material?.winner === 'CHO' || material?.winner === 'HAN' ? material.winner : undefined;
  return buildGameResult(config, state, choFormation, hanFormation, winner === 'CHO' ? 'CHO_WIN' : winner === 'HAN' ? 'HAN_WIN' : 'DRAW', {
    winner,
    history: recordMoves ? state.history : [],
    finalScore: material ? material.cho - material.han : finalScore,
    searchSummaries: recordSearchStats ? searchSummaries : undefined
  });
}

export function runArenaSeries(config: ArenaSeriesConfig): ArenaSeriesResult {
  const playerA = toArenaPlayer(config.playerA);
  const playerB = toArenaPlayer(config.playerB);
  const pairs = config.formationPairs?.length ? config.formationPairs : defaultFormationPairs();
  const results: ArenaGameResult[] = [];

  for (let gameIndex = 0; gameIndex < config.games; gameIndex += 1) {
    const swap = config.swapSides === true && gameIndex % 2 === 1;
    const pair = pairs[gameIndex % pairs.length];
    results.push(
      runArenaGame({
        gameId: `${config.seriesId}-${gameIndex + 1}`,
        choPlayer: swap ? playerB : playerA,
        hanPlayer: swap ? playerA : playerB,
        choFormation: pair.choFormation,
        hanFormation: pair.hanFormation,
        maxPlies: config.maxPlies,
        recordMoves: config.recordMoves,
        recordSearchStats: config.recordSearchStats,
        ruleset: config.ruleset
      })
    );
  }

  return summarizeArenaResults(results, config.seriesId, playerA, playerB);
}

export function summarizeArenaResults(
  results: ArenaGameResult[],
  seriesId = 'arena-series',
  playerA?: Pick<ArenaPlayer, 'id' | 'label'>,
  playerB?: Pick<ArenaPlayer, 'id' | 'label'>
): ArenaSeriesResult {
  const inferredA = playerA ?? {
    id: results[0]?.choPlayerId ?? '',
    label: results[0]?.choPlayerLabel ?? ''
  };
  const inferredB = playerB ?? {
    id: results.find((result) => result.hanPlayerId !== inferredA.id)?.hanPlayerId ?? results[0]?.hanPlayerId ?? '',
    label: results.find((result) => result.hanPlayerId !== inferredA.id)?.hanPlayerLabel ?? results[0]?.hanPlayerLabel ?? ''
  };
  let playerAWins = 0;
  let playerBWins = 0;
  let draws = 0;
  let forfeits = 0;
  let choWins = 0;
  let hanWins = 0;
  let totalPlies = 0;

  for (const result of results) {
    totalPlies += result.plies;
    if (result.outcome === 'DRAW') draws += 1;
    if (result.outcome === 'FORFEIT') forfeits += 1;
    if (result.winner === 'CHO') choWins += 1;
    if (result.winner === 'HAN') hanWins += 1;

    const winningPlayerId = result.winner === 'CHO' ? result.choPlayerId : result.winner === 'HAN' ? result.hanPlayerId : undefined;
    if (winningPlayerId === inferredA.id) playerAWins += 1;
    if (winningPlayerId === inferredB.id) playerBWins += 1;
  }

  const totalGames = results.length;
  return {
    seriesId,
    games: totalGames,
    playerAId: inferredA.id,
    playerBId: inferredB.id,
    playerALabel: inferredA.label,
    playerBLabel: inferredB.label,
    playerAWins,
    playerBWins,
    draws,
    forfeits,
    playerAScoreRate: totalGames > 0 ? (playerAWins + draws * 0.5) / totalGames : 0,
    playerBScoreRate: totalGames > 0 ? (playerBWins + draws * 0.5) / totalGames : 0,
    choWins,
    hanWins,
    averagePlies: totalGames > 0 ? totalPlies / totalGames : 0,
    results
  };
}

export function arenaResultsToCsv(result: ArenaSeriesResult): string {
  const header = [
    'gameId',
    'choPlayer',
    'hanPlayer',
    'choFormation',
    'hanFormation',
    'outcome',
    'winner',
    'plies',
    'forfeitBy',
    'forfeitReason'
  ];
  const rows = result.results.map((game) =>
    [
      game.gameId,
      game.choPlayerLabel,
      game.hanPlayerLabel,
      game.choFormation,
      game.hanFormation,
      game.outcome,
      game.winner ?? '',
      game.plies,
      game.forfeitBy ?? '',
      game.forfeitReason ?? ''
    ]
      .map(csvEscape)
      .join(',')
  );
  return [header.join(','), ...rows].join('\n');
}

export function arenaResultToJson(result: ArenaSeriesResult): string {
  return JSON.stringify(result, null, 2);
}

export function defaultFormationPairs(): FormationPair[] {
  return [{ choFormation: 'inner-elephant', hanFormation: 'inner-elephant' }];
}

export function allFormationPairs(): FormationPair[] {
  return formations.flatMap((choFormation) => formations.map((hanFormation) => ({ choFormation, hanFormation })));
}

function toArenaPlayer(player: EnginePlayerConfig | ArenaPlayer): ArenaPlayer {
  return 'chooseMove' in player ? player : createSearchEnginePlayer(player);
}

function buildForfeitResult(
  config: ArenaGameConfig,
  state: GameState,
  choFormation: Formation,
  hanFormation: Formation,
  forfeitBy: Side,
  forfeitReason: ArenaForfeitReason,
  recordMoves: boolean,
  searchSummaries: ArenaMoveSummary[],
  finalScore?: number
): ArenaGameResult {
  const winner = forfeitBy === 'CHO' ? 'HAN' : 'CHO';
  return buildGameResult(config, state, choFormation, hanFormation, 'FORFEIT', {
    winner,
    forfeitBy,
    forfeitReason,
    history: recordMoves ? state.history : [],
    finalScore,
    searchSummaries: searchSummaries.length > 0 ? searchSummaries : undefined
  });
}

function buildGameResult(
  config: ArenaGameConfig,
  state: GameState,
  choFormation: Formation,
  hanFormation: Formation,
  outcome: ArenaGameOutcome,
  extra: Partial<Pick<ArenaGameResult, 'winner' | 'forfeitBy' | 'forfeitReason' | 'history' | 'finalScore' | 'searchSummaries'>>
): ArenaGameResult {
  return {
    gameId: config.gameId,
    choPlayerId: config.choPlayer.id,
    hanPlayerId: config.hanPlayer.id,
    choPlayerLabel: config.choPlayer.label,
    hanPlayerLabel: config.hanPlayer.label,
    choFormation,
    hanFormation,
    outcome,
    plies: state.history.length,
    history: extra.history ?? state.history,
    winner: extra.winner,
    forfeitBy: extra.forfeitBy,
    forfeitReason: extra.forfeitReason,
    finalScore: extra.finalScore,
    searchSummaries: extra.searchSummaries
  };
}

function searchResultToSummary(ply: number, side: Side, result: SearchResult): ArenaMoveSummary {
  return {
    ply,
    side,
    move: result.move,
    score: result.score,
    depth: result.depth,
    nodes: result.nodes,
    qNodes: result.qNodes,
    nps: result.nps,
    ttHits: result.ttHits,
    cutoffs: result.cutoffs,
    source: result.source,
    bookPlayCount: result.bookMove?.playCount,
    bookScoreRate: result.bookMove?.scoreRate,
    bookScore: result.bookMove?.bookScore
  };
}

function csvEscape(value: unknown): string {
  const text = String(value);
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}
