import { describe, expect, it } from 'vitest';
import { builtInOpeningBook, createGameState, createInitialBoard } from '../engine';
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
      {
        enableQuiescence: false,
        maxQuiescenceDepth: 2,
        useOpeningBook: true,
        openingBook: builtInOpeningBook,
        maxCandidates: 3,
        useTacticalSafety: false,
        table: undefined
      },
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
      includeQuietChecks: undefined,
      useOpeningBook: true,
      openingBook: builtInOpeningBook,
      openingBookContext: undefined,
      maxBookPly: undefined,
      maxCandidates: 3,
      useTacticalSafety: false
    });
    expect('table' in request.options!).toBe(false);
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
      quiescenceEnabled: true,
      source: 'search'
    });

    expect(summary).toContain('깊이 4');
    expect(summary).toContain('평가 320');
    expect(summary).toContain('노드 48231');
    expect(summary).toContain('Q 9120');
    expect(summary).toContain('NPS 20400');
    expect(summary).toContain('TT 132');
    expect(summary).toContain('컷 840');
  });

  it('formats opening book search summaries', () => {
    const state = createGameState(createInitialBoard('inner-elephant', 'inner-elephant'));
    const bookMove = Object.values(builtInOpeningBook.positions)[0].moves[0];
    const summary = formatSearchSummary({
      move: bookMove.move,
      score: 0,
      depth: 0,
      nodes: 0,
      pv: [bookMove.move],
      ttHits: 0,
      ttMisses: 0,
      ttStores: 0,
      cutoffs: 0,
      nps: 0,
      elapsedMs: 0,
      qNodes: 0,
      qCutoffs: 0,
      quiescenceEnabled: true,
      source: 'book',
      bookMove
    });

    expect(state.turn).toBe('CHO');
    expect(summary).toContain('오프닝북:');
    expect(summary).toContain('표본');
    expect(summary).toContain('승점률');
  });

  it('checks whether a worker response belongs to the latest request', () => {
    const response: AiWorkerResponse = { type: 'error', requestId: 'current', message: 'failed' };

    expect(isLatestWorkerResponse(response, 'current')).toBe(true);
    expect(isLatestWorkerResponse(response, 'older')).toBe(false);
    expect(isLatestWorkerResponse(response, null)).toBe(false);
  });
});
