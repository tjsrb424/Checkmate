/// <reference lib="webworker" />

import { difficultyLimits, searchBestMove } from '../engine';
import type { AiWorkerRequest, AiWorkerResponse } from './aiWorkerProtocol';

const workerScope = self as DedicatedWorkerGlobalScope;

workerScope.onmessage = (event: MessageEvent<AiWorkerRequest>) => {
  const request = event.data;
  if (request?.type !== 'search') {
    return;
  }

  try {
    const result = searchBestMove(request.state, difficultyLimits[request.difficulty], request.options);
    const response: AiWorkerResponse = {
      type: 'result',
      requestId: request.requestId,
      result
    };
    workerScope.postMessage(response);
  } catch (error) {
    const response: AiWorkerResponse = {
      type: 'error',
      requestId: request.requestId,
      message: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : undefined
    };
    workerScope.postMessage(response);
  }
};

export {};
