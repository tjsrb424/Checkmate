import type { Difficulty, GameState, SearchOptions, SearchResult } from '../engine';
import { formatMoveWithPiece } from '../engine/notation';

export type SerializableSearchOptions = Pick<
  SearchOptions,
  | 'enableTransposition'
  | 'enableQuiescence'
  | 'maxQuiescenceDepth'
  | 'includeQuietChecks'
  | 'useOpeningBook'
  | 'openingBook'
  | 'openingBookContext'
  | 'maxBookPly'
  | 'maxCandidates'
>;

export interface AiSearchRequest {
  type: 'search';
  requestId: string;
  state: GameState;
  difficulty: Difficulty;
  options?: SerializableSearchOptions;
}

export interface AiSearchSuccessResponse {
  type: 'result';
  requestId: string;
  result: SearchResult;
}

export interface AiSearchErrorResponse {
  type: 'error';
  requestId: string;
  message: string;
  stack?: string;
}

export type AiWorkerRequest = AiSearchRequest;
export type AiWorkerResponse = AiSearchSuccessResponse | AiSearchErrorResponse;

export function createAiSearchRequest(
  state: GameState,
  difficulty: Difficulty,
  options?: SearchOptions,
  requestId = createRequestId()
): AiSearchRequest {
  return {
    type: 'search',
    requestId,
    state,
    difficulty,
    options: options ? toSerializableSearchOptions(options) : undefined
  };
}

export function isAiWorkerResponse(value: unknown): value is AiWorkerResponse {
  if (!isRecord(value) || typeof value.requestId !== 'string') return false;
  if (value.type === 'error') return typeof value.message === 'string';
  if (value.type === 'result') return isRecord(value.result);
  return false;
}

export function isLatestWorkerResponse(response: AiWorkerResponse, requestId: string | null): boolean {
  return Boolean(requestId) && response.requestId === requestId;
}

export function formatSearchSummary(result: SearchResult): string {
  if (result.source === 'book' && result.bookMove) {
    return `오프닝북: ${formatMoveWithPiece(result.bookMove.move, result.bookMove.move.piece)}, 표본 ${result.bookMove.playCount}, 승점률 ${(
      result.bookMove.scoreRate * 100
    ).toFixed(1)}%`;
  }
  if (result.source === 'book') return '오프닝북 착수';
  return `깊이 ${result.depth || 1}, 평가 ${Math.round(result.score)}, 노드 ${result.nodes}, Q ${result.qNodes}, NPS ${result.nps}, TT ${result.ttHits}, 컷 ${result.cutoffs}`;
}

function createRequestId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function toSerializableSearchOptions(options: SearchOptions): SerializableSearchOptions {
  return {
    enableTransposition: options.enableTransposition,
    enableQuiescence: options.enableQuiescence,
    maxQuiescenceDepth: options.maxQuiescenceDepth,
    includeQuietChecks: options.includeQuietChecks,
    useOpeningBook: options.useOpeningBook,
    openingBook: options.openingBook,
    openingBookContext: options.openingBookContext,
    maxBookPly: options.maxBookPly,
    maxCandidates: options.maxCandidates
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}
