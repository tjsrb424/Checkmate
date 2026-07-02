import { describe, expect, it } from 'vitest';
import { createGameState, createInitialBoard } from '../engine';
import type { AiWorkerResponse } from './aiWorkerProtocol';
import {
  createAiSearchRequest,
  formatSearchSummary,
  isAiWorkerResponse,
  isLatestWorkerResponse
} from './aiWorkerProtocol';

describe('AI worker protocol', () => {
  it('creates serializable search requests with request metadata', () => {
    const state = createGameState(createInitialBoard('inner-elephant', 'inner-elephant'));
    const request = createAiSearchRequest(
      state,
      'hard',
      { enableQuiescence: false, maxQuiescenceDepth: 2 },
      'request-1'
    );

    expect(request.type).toBe('search');
    expect(request.requestId).toBe('request-1');
    expect(request.state).toBe(state);
    expect(request.difficulty).toBe('hard');
    expect(request.options).toEqual({
      enableTransposition: undefined,
      enableQuiescence: false,
      maxQuiescenceDepth: 2,
      includeQuietChecks: undefined
    });
  });

  it('recognizes result and error responses', () => {
    const resultResponse = {
      type: 'result',
      requestId: 'request-1',
      result: { move: null }
    };
    const errorResponse = {
      type: 'error',
      requestId: 'request-2',
      message: 'failed'
    };

    expect(isAiWorkerResponse(resultResponse)).toBe(true);
    expect(isAiWorkerResponse(errorResponse)).toBe(true);
    expect(isAiWorkerResponse({ type: 'result', requestId: 1, result: {} })).toBe(false);
    expect(isAiWorkerResponse({ type: 'error', requestId: 'request-3' })).toBe(false);
  });

  it('formats search summaries with the existing AI stats', () => {
    const summary = formatSearchSummary({
      move: null,
      score: 320.4,
      depth: 4,
      nodes: 48231,
      pv: [],
      ttHits: 132,
      ttMisses: 9,
      ttStores: 20,
      cutoffs: 840,
      nps: 20400,
      elapsedMs: 2364,
      qNodes: 9120,
      qCutoffs: 88,
      quiescenceEnabled: true
    });

    expect(summary).toContain('깊이 4');
    expect(summary).toContain('평가 320');
    expect(summary).toContain('노드 48231');
    expect(summary).toContain('Q 9120');
    expect(summary).toContain('NPS 20400');
    expect(summary).toContain('TT 132');
    expect(summary).toContain('컷 840');
  });

  it('checks whether a worker response belongs to the latest request', () => {
    const response: AiWorkerResponse = { type: 'error', requestId: 'current', message: 'failed' };

    expect(isLatestWorkerResponse(response, 'current')).toBe(true);
    expect(isLatestWorkerResponse(response, 'older')).toBe(false);
    expect(isLatestWorkerResponse(response, null)).toBe(false);
  });
});
