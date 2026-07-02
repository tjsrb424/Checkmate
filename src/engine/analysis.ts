import { difficultyLimits, MATE_SCORE, searchBestMove } from './ai';
import type { SearchCandidate, SearchOptions } from './ai';
import { applyMove, createGameState, generateLegalMoves } from './rules';
import { cloneBoard, moveKey } from './types';
import type { Board, Difficulty, GameState, Move, SearchLimits, Side } from './types';

export type BlunderSeverity = 'mistake' | 'blunder' | 'losing-blunder';

export interface AnalyzePositionOptions {
  limits?: SearchLimits;
  difficulty?: Difficulty;
  searchOptions?: SearchOptions;
  maxCandidates?: number;
}

export interface AnalyzeGameOptions extends AnalyzePositionOptions {
  initialTurn?: Side;
}

export interface PositionAnalysis {
  state: GameState;
  turn: Side;
  bestMove: Move | null;
  score: number;
  depth: number;
  nodes: number;
  qNodes: number;
  nps: number;
  source: 'book' | 'search';
  candidates: SearchCandidate[];
}

export interface ScoreTimelineEntry {
  ply: number;
  side: Side;
  scoreBefore: number;
  scoreAfter: number;
  move: Move;
  bestMove: Move | null;
  loss: number;
}

export interface Blunder {
  ply: number;
  side: Side;
  move: Move;
  severity: BlunderSeverity;
  scoreBefore: number;
  scoreAfter: number;
  loss: number;
  bestMove: Move | null;
  bestScore: number;
}

export interface GameAnalysis {
  initialBoard: Board;
  history: Move[];
  positions: PositionAnalysis[];
  scoreTimeline: ScoreTimelineEntry[];
  blunders: Blunder[];
  error?: string;
  illegalPly?: number;
}

export function analyzePosition(state: GameState, options: AnalyzePositionOptions = {}): PositionAnalysis {
  const limits = options.limits ?? difficultyLimits[options.difficulty ?? 'normal'];
  const result = searchBestMove(state, limits, {
    ...options.searchOptions,
    maxCandidates: options.maxCandidates ?? options.searchOptions?.maxCandidates ?? 5
  });

  return {
    state,
    turn: state.turn,
    bestMove: result.move,
    score: result.score,
    depth: result.depth,
    nodes: result.nodes,
    qNodes: result.qNodes,
    nps: result.nps,
    source: result.source,
    candidates: result.candidates ?? []
  };
}

export function analyzeGame(initialBoard: Board, history: Move[], options: AnalyzeGameOptions = {}): GameAnalysis {
  const positions: PositionAnalysis[] = [];
  const scoreTimeline: ScoreTimelineEntry[] = [];
  const appliedHistory: Move[] = [];
  let state = createGameState(cloneBoard(initialBoard), options.initialTurn ?? 'CHO');

  for (let ply = 0; ply < history.length; ply += 1) {
    const before = analyzePosition(state, options);
    positions.push(before);

    const actualMove = findLegalMove(state, history[ply]);
    if (!actualMove) {
      const partial = {
        initialBoard: cloneBoard(initialBoard),
        history: appliedHistory,
        positions,
        scoreTimeline,
        blunders: detectBlunders(scoreTimeline),
        error: `Illegal move at ply ${ply + 1}: ${moveKey(history[ply])}`,
        illegalPly: ply + 1
      };
      return partial;
    }

    const next = applyMove(state, actualMove, true);
    const scoreAfter = next.winner === state.turn ? MATE_SCORE : -analyzePosition(next, options).score;
    scoreTimeline.push({
      ply: ply + 1,
      side: state.turn,
      scoreBefore: before.score,
      scoreAfter,
      move: actualMove,
      bestMove: before.bestMove,
      loss: before.score - scoreAfter
    });

    appliedHistory.push(actualMove);
    state = next;
    if (state.winner) break;
  }

  if (history.length === 0) {
    positions.push(analyzePosition(state, options));
  }

  return {
    initialBoard: cloneBoard(initialBoard),
    history: appliedHistory,
    positions,
    scoreTimeline,
    blunders: detectBlunders(scoreTimeline)
  };
}

export function detectBlunders(scoreTimeline: ScoreTimelineEntry[]): Blunder[] {
  return scoreTimeline.flatMap((entry) => {
    const severity = classifyLoss(entry.loss, entry.scoreBefore, entry.scoreAfter);
    if (!severity) return [];
    return [
      {
        ply: entry.ply,
        side: entry.side,
        move: entry.move,
        severity,
        scoreBefore: entry.scoreBefore,
        scoreAfter: entry.scoreAfter,
        loss: entry.loss,
        bestMove: entry.bestMove,
        bestScore: entry.scoreBefore
      }
    ];
  });
}

function classifyLoss(loss: number, scoreBefore: number, scoreAfter: number): BlunderSeverity | null {
  if (scoreBefore > MATE_SCORE / 2 && scoreAfter < MATE_SCORE / 2) return 'losing-blunder';
  if (loss >= 1200) return 'losing-blunder';
  if (loss >= 700) return 'blunder';
  if (loss >= 300) return 'mistake';
  return null;
}

function findLegalMove(state: GameState, move: Move): Move | null {
  return generateLegalMoves(state).find((legal) => moveKey(legal) === moveKey(move)) ?? null;
}
