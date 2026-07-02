import type { Difficulty, GameState, SearchOptions, SearchResult } from '../engine';

export type SerializableSearchOptions = Pick<
  SearchOptions,
  'enableTransposition' | 'enableQuiescence' | 'maxQuiescenceDepth' | 'includeQuietChecks'
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
    includeQuietChecks: options.includeQuietChecks
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}
