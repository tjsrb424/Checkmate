export interface TrainingApiEnvelope {
  offline?: boolean;
  error?: string;
}

export interface TrainingHealth extends TrainingApiEnvelope {
  ok?: boolean;
  server?: string;
  version?: string;
  cwd?: string;
  python?: string;
}

export type TrainingServerStatus = 'idle' | 'running' | 'failed';

export interface TrainingStatus extends TrainingApiEnvelope {
  serverStatus?: TrainingServerStatus;
  pid?: number | null;
  startedAt?: string | null;
  endedAt?: string | null;
  command?: string[] | null;
  lastError?: string | null;
  autotrainState?: Record<string, unknown> | null;
  autotrainSummary?: Record<string, unknown> | null;
  latestChampion?: ModelRegistryEntry | null;
  registryModelCount?: number;
  warnings?: string[];
}

export interface StartAutoTrainRequest {
  quick?: boolean;
  iterations?: number;
  gamesPerIteration?: number;
  simulations?: number;
  maxPlies?: number;
  trainEpochs?: number;
  batchSize?: number;
  promotionGames?: number;
  threshold?: number;
  ruleset?: 'kakao-like' | 'oetongsu-basic' | 'kja-like';
  selfplayWorkers?: number;
  parallelSelfPlay?: boolean;
}

export interface StartAutoTrainResponse extends TrainingApiEnvelope {
  started?: boolean;
  pid?: number;
  command?: string[];
  startedAt?: string;
}

export interface StopAutoTrainResponse extends TrainingApiEnvelope {
  requestedStop?: boolean;
  pid?: number;
  message?: string;
}

export interface ModelRegistryEntry {
  version?: string;
  status?: 'candidate' | 'promoted' | 'rejected' | string;
  path?: string;
  createdAt?: string;
  promotedAt?: string;
  rejectedAt?: string;
  metrics?: Record<string, unknown>;
  arena?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface ModelRegistryResponse extends TrainingApiEnvelope {
  registry?: {
    models?: ModelRegistryEntry[];
    [key: string]: unknown;
  };
  latestPromoted?: ModelRegistryEntry | null;
  promotedCount?: number;
  rejectedCount?: number;
  candidateCount?: number;
}

export interface TrainingLogEntry {
  event?: string;
  time?: string;
  timestamp?: string;
  iteration?: number;
  message?: string;
  raw?: string;
  [key: string]: unknown;
}

export interface TrainingLogsResponse extends TrainingApiEnvelope {
  entries?: TrainingLogEntry[];
}

export interface TrainingSummaryResponse extends TrainingApiEnvelope {
  summary?: Record<string, unknown> | null;
}

export interface ArenaResultSummary {
  file?: string;
  path?: string;
  candidateScoreRate?: number | null;
  championScoreRate?: number | null;
  promoted?: boolean | null;
  illegalMoves?: number | null;
  forfeits?: number | null;
  averagePlies?: number | null;
  modifiedAt?: string;
}

export interface ArenaResultsResponse extends TrainingApiEnvelope {
  results?: ArenaResultSummary[];
}
