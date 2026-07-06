import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  getArenaResults,
  getModelRegistry,
  getTrainingHealth,
  getTrainingLogs,
  getTrainingProgress,
  getTrainingStatus,
  getTrainingSummary,
  startAutoTrain,
  stopAutoTrain,
  trainingServerUrl
} from '../training/trainingApi';
import type {
  ArenaResultSummary,
  ModelRegistryEntry,
  ModelRegistryResponse,
  StartAutoTrainRequest,
  TrainingHealth,
  TrainingLogEntry,
  TrainingLogsResponse,
  TrainingProgressResponse,
  TrainingStatus,
  TrainingSummaryResponse
} from '../training/types';

const rulesetOptions: Array<NonNullable<StartAutoTrainRequest['ruleset']>> = [
  'kakao-like',
  'oetongsu-basic',
  'kja-like'
];

export function TrainingTab() {
  const [health, setHealth] = useState<TrainingHealth | null>(null);
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [registry, setRegistry] = useState<ModelRegistryResponse | null>(null);
  const [logs, setLogs] = useState<TrainingLogsResponse | null>(null);
  const [summary, setSummary] = useState<TrainingSummaryResponse | null>(null);
  const [progress, setProgress] = useState<TrainingProgressResponse | null>(null);
  const [arena, setArena] = useState<ArenaResultSummary[]>([]);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [form, setForm] = useState<Required<Omit<StartAutoTrainRequest, 'quick'>>>({
    iterations: 2,
    gamesPerIteration: 4,
    simulations: 8,
    maxPlies: 40,
    trainEpochs: 1,
    batchSize: 8,
    promotionGames: 4,
    threshold: 0.55,
    ruleset: 'kakao-like',
    selfplayWorkers: 2,
    parallelSelfPlay: true
  });

  const offline = Boolean(health?.offline || status?.offline);
  const running = status?.serverStatus === 'running';
  const latestModels = useMemo(() => registry?.registry?.models?.slice(-6).reverse() ?? [], [registry]);

  const refresh = useCallback(async () => {
    const [healthResult, statusResult, registryResult, logsResult, summaryResult, progressResult, arenaResult] = await Promise.all([
      getTrainingHealth(),
      getTrainingStatus(),
      getModelRegistry(),
      getTrainingLogs(50),
      getTrainingSummary(),
      getTrainingProgress(),
      getArenaResults()
    ]);
    setHealth(healthResult);
    setStatus(statusResult);
    setRegistry(registryResult);
    setLogs(logsResult);
    setSummary(summaryResult);
    setProgress(progressResult);
    setArena(arenaResult.results ?? []);
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), running ? 2500 : 6000);
    return () => window.clearInterval(timer);
  }, [refresh, running]);

  async function handleQuickStart() {
    await runAction('빠른 AutoTrain을 시작했습니다.', () =>
      startAutoTrain({
        quick: true,
        ruleset: form.ruleset,
        selfplayWorkers: form.selfplayWorkers,
        parallelSelfPlay: form.parallelSelfPlay
      })
    );
  }

  async function handleCustomStart() {
    await runAction('사용자 지정 AutoTrain을 시작했습니다.', () => startAutoTrain({ ...form }));
  }

  async function handleStop() {
    await runAction('AutoTrain 중지를 요청했습니다.', () => stopAutoTrain());
  }

  async function runAction(successMessage: string, action: () => Promise<{ error?: string; offline?: boolean }>) {
    setBusy(true);
    setMessage('');
    const result = await action();
    if (result.offline) {
      setMessage('훈련 서버가 실행되어 있지 않습니다. npm run training:server 를 먼저 실행하세요.');
    } else if (result.error) {
      setMessage(result.error);
    } else {
      setMessage(successMessage);
    }
    await refresh();
    setBusy(false);
  }

  return (
    <section className="trainingTab" aria-label="Local training control">
      <div className="trainingHero">
        <div>
          <p className="eyebrow">Local AlphaZero Pipeline</p>
          <h2>훈련 제어판</h2>
          <p>로컬 FastAPI 서버와 연결해 AutoTrain 실행, 모델 registry, arena 결과, 최근 로그를 확인합니다.</p>
        </div>
        <button className="secondaryAction" onClick={() => void refresh()} disabled={busy}>
          새로고침
        </button>
      </div>

      {offline && (
        <div className="trainingAlert">
          <strong>훈련 서버 연결 필요</strong>
          <span>터미널에서 npm run training:server 를 실행한 뒤 이 탭을 새로고침하세요.</span>
          <code>{trainingServerUrl}</code>
        </div>
      )}

      <ProgressPanel progress={progress} />

      <div className="trainingGrid">
        <section className="trainingPanel">
          <div className="panelHeader">
            <span className="groupLabel">서버 상태</span>
            <StatusPill status={status?.serverStatus ?? (offline ? 'failed' : 'idle')} />
          </div>
          <dl className="metricGrid">
            <Metric label="API" value={health?.server ?? '대기 중'} />
            <Metric label="버전" value={health?.version ?? '-'} />
            <Metric label="PID" value={status?.pid ? String(status.pid) : '-'} />
            <Metric label="모델 수" value={String(status?.registryModelCount ?? 0)} />
          </dl>
          <p className="panelText">{status?.lastError ?? status?.warnings?.[0] ?? '로컬 훈련 상태를 관찰할 준비가 되었습니다.'}</p>
          {message && <p className="actionMessage">{message}</p>}
          <div className="trainingActions">
            <button className="primaryAction" onClick={handleQuickStart} disabled={busy || running}>
              Quick Start
            </button>
            <button className="secondaryAction" onClick={handleCustomStart} disabled={busy || running}>
              Custom Start
            </button>
            <button className="dangerAction" onClick={handleStop} disabled={busy || !running}>
              Stop
            </button>
          </div>
        </section>

        <section className="trainingPanel">
          <div className="panelHeader">
            <span className="groupLabel">사용자 지정 실행</span>
          </div>
          <div className="trainingForm">
            <NumberField label="Iterations" value={form.iterations} min={1} onChange={(value) => updateForm('iterations', value)} />
            <NumberField
              label="Games"
              value={form.gamesPerIteration}
              min={1}
              onChange={(value) => updateForm('gamesPerIteration', value)}
            />
            <NumberField label="Simulations" value={form.simulations} min={1} onChange={(value) => updateForm('simulations', value)} />
            <NumberField label="Max plies" value={form.maxPlies} min={4} onChange={(value) => updateForm('maxPlies', value)} />
            <NumberField label="Epochs" value={form.trainEpochs} min={1} onChange={(value) => updateForm('trainEpochs', value)} />
            <NumberField label="Batch" value={form.batchSize} min={1} onChange={(value) => updateForm('batchSize', value)} />
            <NumberField
              label="Arena games"
              value={form.promotionGames}
              min={1}
              onChange={(value) => updateForm('promotionGames', value)}
            />
            <NumberField
              label="Workers"
              value={form.selfplayWorkers}
              min={1}
              onChange={(value) => updateForm('selfplayWorkers', value)}
            />
            <NumberField label="Threshold" value={form.threshold} min={0} step={0.01} onChange={(value) => updateForm('threshold', value)} />
            <label>
              <span>Ruleset</span>
              <select value={form.ruleset} onChange={(event) => updateForm('ruleset', event.target.value as typeof form.ruleset)}>
                {rulesetOptions.map((ruleset) => (
                  <option key={ruleset} value={ruleset}>
                    {ruleset}
                  </option>
                ))}
              </select>
            </label>
            <label className="trainingToggle">
              <input
                type="checkbox"
                checked={form.parallelSelfPlay}
                onChange={(event) => updateForm('parallelSelfPlay', event.target.checked)}
              />
              <span>Parallel self-play</span>
            </label>
          </div>
        </section>
      </div>

      <div className="trainingGrid three">
        <RegistryPanel registry={registry} models={latestModels} />
        <SummaryPanel summary={summary?.summary} />
        <ArenaPanel arena={arena} />
      </div>

      <section className="trainingPanel">
        <div className="panelHeader">
          <span className="groupLabel">최근 로그</span>
          <small>{logs?.entries?.length ?? 0} entries</small>
        </div>
        <ol className="logList">
          {(logs?.entries ?? []).slice(-10).reverse().map((entry, index) => (
            <li key={`${entry.time ?? entry.timestamp ?? 'log'}-${index}`}>
              <strong>{entry.event ?? entry.message ?? entry.raw ?? 'event'}</strong>
              <code>{compactJson(entry)}</code>
            </li>
          ))}
        </ol>
      </section>
    </section>
  );

  function updateForm<K extends keyof typeof form>(key: K, value: (typeof form)[K]) {
    setForm((current) => ({ ...current, [key]: value }));
  }
}

function ProgressPanel({ progress }: { progress: TrainingProgressResponse | null }) {
  const item = progress?.progress;
  const selfPlay = item?.selfPlay ?? {};
  const training = item?.training ?? {};
  const arena = item?.arena ?? {};
  const models = item?.models ?? {};
  const result = item?.result ?? {};
  const resultStatus = displayResult(result.status, result.promoted);

  if (!progress || progress.offline) {
    return (
      <section className="trainingPanel progressPanel">
        <div className="panelHeader">
          <span className="groupLabel">학습 진행 상황</span>
          <small>대기 중</small>
        </div>
        <p className="panelText">진행률 정보를 기다리고 있습니다.</p>
      </section>
    );
  }

  if (!progress.exists || !item) {
    return (
      <section className="trainingPanel progressPanel">
        <div className="panelHeader">
          <span className="groupLabel">학습 진행 상황</span>
          <small>progress.json 없음</small>
        </div>
        <p className="panelText">아직 실행 중인 AutoTrain 진행률 파일이 없습니다.</p>
      </section>
    );
  }

  return (
    <section className="trainingPanel progressPanel">
      <div className="panelHeader">
        <span className="groupLabel">학습 진행 상황</span>
        <StatusPill status={item.status ?? 'idle'} />
      </div>
      <div className="progressOverview">
        <Metric label="현재 상태" value={item.statusLabelKo ?? statusLabel(item.status)} />
        <Metric label="현재 단계" value={item.phaseLabelKo ?? phaseLabel(item.phaseKey)} />
        <Metric label="경과 시간" value={item.elapsedText ?? '-'} />
        <Metric label="예상 남은 시간" value={item.etaText ?? '-'} />
      </div>
      <div className="progressBars">
        <ProgressBar label="전체 진행률" value={item.overallPercent} />
        <ProgressBar label="단계 진행률" value={item.phasePercent} />
      </div>
      <p className="panelText">{item.messageKo ?? item.message ?? '진행률을 확인하는 중입니다.'}</p>
      <div className="progressCards">
        <MiniProgressCard
          title="자기대국 생성"
          rows={[
            ['대국', ratioText(selfPlay.currentGames, selfPlay.totalGames)],
            ['생성 샘플', valueText(selfPlay.currentSamples)],
            ['작업자', valueText(selfPlay.workers)]
          ]}
        />
        <MiniProgressCard
          title="후보 학습"
          rows={[
            ['반복 학습', ratioText(training.currentEpoch, training.totalEpochs)],
            ['묶음 학습', ratioText(training.currentBatch, training.totalBatches)],
            ['정책 손실', numberText(training.policyLoss)],
            ['가치 손실', numberText(training.valueLoss)]
          ]}
        />
        <MiniProgressCard
          title="승격 대국"
          rows={[
            ['대국', ratioText(arena.currentGames, arena.totalGames)],
            ['후보 점수율', percentText(arena.candidateScoreRate)],
            ['불법 수', valueText(arena.illegalMoves)],
            ['기권', valueText(arena.forfeits)]
          ]}
        />
        <MiniProgressCard
          title="모델 상태"
          rows={[
            ['현재 챔피언', valueText(models.championVersion ?? models.latestPromotedVersion)],
            ['후보 AI', valueText(models.candidateVersion)],
            ['승격 기준', '55%'],
            ['결과', resultStatus]
          ]}
        />
      </div>
      <p className="progressGlossary">
        자기대국은 AI가 스스로 대국 데이터를 만드는 단계입니다. 후보 학습은 그 데이터로 새 모델을 훈련하는 단계이고,
        승격 대국은 후보 AI가 현재 챔피언보다 강한지 확인하는 단계입니다.
      </p>
    </section>
  );
}

function ProgressBar({ label, value }: { label: string; value: number | undefined }) {
  const percent = clampPercent(value);
  return (
    <div className="progressBarRow">
      <div>
        <span>{label}</span>
        <strong>{percent.toFixed(percent % 1 === 0 ? 0 : 1)}%</strong>
      </div>
      <div className="progressTrack" aria-label={label}>
        <span style={{ width: `${percent}%` }} />
      </div>
    </div>
  );
}

function MiniProgressCard({ title, rows }: { title: string; rows: Array<[string, string]> }) {
  return (
    <div className="miniProgressCard">
      <strong>{title}</strong>
      <dl>
        {rows.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function NumberField({
  label,
  value,
  min,
  step = 1,
  onChange
}: {
  label: string;
  value: number;
  min: number;
  step?: number;
  onChange: (value: number) => void;
}) {
  return (
    <label>
      <span>{label}</span>
      <input type="number" min={min} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  );
}

function RegistryPanel({ registry, models }: { registry: ModelRegistryResponse | null; models: ModelRegistryEntry[] }) {
  return (
    <section className="trainingPanel">
      <div className="panelHeader">
        <span className="groupLabel">모델 Registry</span>
        <small>{registry?.promotedCount ?? 0} promoted</small>
      </div>
      <ul className="modelList">
        {models.length === 0 ? (
          <li className="emptyState">등록된 모델이 아직 없습니다.</li>
        ) : (
          models.map((model, index) => (
            <li key={`${model.version ?? 'model'}-${index}`}>
              <strong>{model.version ?? 'unknown'}</strong>
              <span>{model.status ?? 'unknown'}</span>
            </li>
          ))
        )}
      </ul>
    </section>
  );
}

function SummaryPanel({ summary }: { summary: Record<string, unknown> | null | undefined }) {
  const details = parallelSummaryDetails(summary);
  return (
    <section className="trainingPanel">
      <div className="panelHeader">
        <span className="groupLabel">AutoTrain 요약</span>
      </div>
      {details && (
        <dl className="metricGrid compact">
          <Metric label="Workers" value={details.workers} />
          <Metric label="Shards" value={details.shards} />
          <Metric label="Samples" value={details.samples} />
          <Metric label="Samples/sec" value={details.samplesPerSec} />
          <Metric label="Games/sec" value={details.gamesPerSec} />
          <Metric label="Inference ms" value={details.inferenceMs} />
          <Metric label="MCTS ms" value={details.mctsMs} />
        </dl>
      )}
      <pre className="summaryBox">{summary ? JSON.stringify(summary, null, 2) : '아직 summary 파일이 없습니다.'}</pre>
    </section>
  );
}

function ArenaPanel({ arena }: { arena: ArenaResultSummary[] }) {
  return (
    <section className="trainingPanel">
      <div className="panelHeader">
        <span className="groupLabel">Arena 결과</span>
        <small>{arena.length}</small>
      </div>
      <ul className="arenaList">
        {arena.slice(0, 5).map((result) => (
          <li key={result.file ?? result.path}>
            <strong>{result.promoted ? 'Promoted' : 'Checked'}</strong>
            <span>{formatPercent(result.candidateScoreRate)} candidate</span>
            <small>{result.file}</small>
          </li>
        ))}
        {arena.length === 0 && <li className="emptyState">arena 결과가 아직 없습니다.</li>}
      </ul>
    </section>
  );
}

function StatusPill({ status }: { status: string }) {
  return <span className={`statusPill ${status}`}>{status}</span>;
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function compactJson(entry: TrainingLogEntry): string {
  return JSON.stringify(entry).slice(0, 240);
}

function formatPercent(value: number | null | undefined): string {
  return typeof value === 'number' ? `${Math.round(value * 100)}%` : '-';
}

function clampPercent(value: number | undefined): number {
  if (typeof value !== 'number' || Number.isNaN(value)) return 0;
  return Math.max(0, Math.min(100, value));
}

function statusLabel(status: unknown): string {
  if (status === 'running') return '진행 중';
  if (status === 'completed') return '완료';
  if (status === 'failed') return '실패';
  return '대기 중';
}

function phaseLabel(phase: unknown): string {
  if (phase === 'selfplay') return '자기대국 생성';
  if (phase === 'train') return '후보 학습';
  if (phase === 'arena') return '승격 대국';
  if (phase === 'package') return '결과 정리';
  if (phase === 'completed') return '완료';
  if (phase === 'failed') return '실패';
  return '대기 중';
}

function valueText(value: unknown): string {
  if (value === null || value === undefined || value === '') return '-';
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(2);
  if (typeof value === 'string') return value;
  if (typeof value === 'boolean') return value ? '예' : '아니오';
  return '-';
}

function numberText(value: unknown): string {
  return typeof value === 'number' ? value.toFixed(4) : '-';
}

function ratioText(current: unknown, total: unknown): string {
  const left = valueText(current);
  const right = valueText(total);
  if (left === '-' && right === '-') return '-';
  return `${left} / ${right}`;
}

function percentText(value: unknown): string {
  return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '-';
}

function displayResult(status: unknown, promoted: unknown): string {
  if (promoted === true || status === 'promoted') return '승격';
  if (promoted === false || status === 'rejected') return '미승격';
  if (status === 'completed') return '완료';
  if (status === 'failed') return '실패';
  return '진행 중';
}

function parallelSummaryDetails(summary: Record<string, unknown> | null | undefined) {
  if (!summary) return null;
  const iterations = Array.isArray(summary.iterations) ? summary.iterations : [];
  const latest = iterations[iterations.length - 1];
  if (!latest || typeof latest !== 'object' || !('metrics' in latest)) return null;
  const metrics = latest.metrics;
  if (!metrics || typeof metrics !== 'object' || !('selfPlay' in metrics)) return null;
  const selfPlay = metrics.selfPlay as Record<string, unknown>;
  return {
    workers: String(selfPlay.workers ?? '-'),
    shards: String(selfPlay.shardCount ?? '-'),
    samples: String(selfPlay.sampleCount ?? '-'),
    samplesPerSec: formatNumber(selfPlay.samplesPerSecond),
    gamesPerSec: formatNumber(selfPlay.gamesPerSecond),
    inferenceMs: formatNumber(selfPlay.totalInferenceMs),
    mctsMs: formatNumber(selfPlay.totalMctsMs)
  };
}

function formatNumber(value: unknown): string {
  return typeof value === 'number' ? value.toFixed(value >= 100 ? 0 : 2) : '-';
}
