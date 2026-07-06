import type {
  ArenaResultsResponse,
  ModelRegistryResponse,
  StartAutoTrainRequest,
  StartAutoTrainResponse,
  StopAutoTrainResponse,
  TrainingHealth,
  TrainingLogsResponse,
  TrainingProgressResponse,
  TrainingStatus,
  TrainingSummaryResponse
} from './types';

const DEFAULT_TRAINING_SERVER_URL = 'http://127.0.0.1:8765';
const viteEnv = (import.meta as unknown as { env?: { VITE_TRAINING_SERVER_URL?: string } }).env;

export const trainingServerUrl =
  viteEnv?.VITE_TRAINING_SERVER_URL?.replace(/\/$/, '') ?? DEFAULT_TRAINING_SERVER_URL;

export function getTrainingHealth(): Promise<TrainingHealth> {
  return getJson<TrainingHealth>('/api/health');
}

export function getTrainingStatus(): Promise<TrainingStatus> {
  return getJson<TrainingStatus>('/api/training/status');
}

export function startAutoTrain(payload: StartAutoTrainRequest): Promise<StartAutoTrainResponse> {
  return postJson<StartAutoTrainResponse>('/api/training/autotrain/start', payload);
}

export function stopAutoTrain(): Promise<StopAutoTrainResponse> {
  return postJson<StopAutoTrainResponse>('/api/training/autotrain/stop', {});
}

export function getModelRegistry(): Promise<ModelRegistryResponse> {
  return getJson<ModelRegistryResponse>('/api/models/registry');
}

export function getTrainingLogs(limit = 50): Promise<TrainingLogsResponse> {
  return getJson<TrainingLogsResponse>(`/api/training/logs?limit=${encodeURIComponent(limit)}`);
}

export function getTrainingSummary(): Promise<TrainingSummaryResponse> {
  return getJson<TrainingSummaryResponse>('/api/training/summary');
}

export function getTrainingProgress(): Promise<TrainingProgressResponse> {
  return getJson<TrainingProgressResponse>('/api/training/progress');
}

export function getArenaResults(): Promise<ArenaResultsResponse> {
  return getJson<ArenaResultsResponse>('/api/arena/results');
}

async function getJson<T>(path: string): Promise<T> {
  return requestJson<T>(path, { method: 'GET' });
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  return requestJson<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
}

async function requestJson<T>(path: string, init: RequestInit): Promise<T> {
  try {
    const response = await fetch(`${trainingServerUrl}${path}`, init);
    const payload = await readJson(response);
    if (!response.ok) {
      return {
        ...(payload && typeof payload === 'object' ? payload : {}),
        error: extractError(payload) ?? `${response.status} ${response.statusText}`
      } as T;
    }
    return payload as T;
  } catch (error) {
    return {
      offline: true,
      error: error instanceof Error ? error.message : String(error)
    } as T;
  }
}

async function readJson(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return { raw: text };
  }
}

function extractError(payload: unknown): string | null {
  if (!payload || typeof payload !== 'object') return null;
  if ('detail' in payload && typeof payload.detail === 'string') return payload.detail;
  if ('error' in payload && typeof payload.error === 'string') return payload.error;
  return null;
}
