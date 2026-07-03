# Local Training Server And UI

Sprint 26 adds a local-only training control surface for the Oetongsu AlphaZero pipeline.

## Start The API Server

From the repository root:

```bash
npm run training:server
```

Development reload mode:

```bash
npm run training:server:dev
```

The server binds to `127.0.0.1:8765` only. This is intentional for the MVP because the endpoints can start and stop local training subprocesses.

## Open The UI

In a second terminal:

```bash
npm run dev
```

Open the app and select the `훈련` tab. The tab polls the local server, shows offline guidance when the server is not running, and exposes quick/custom AutoTrain controls.

If you need a different API URL for Vite, set:

```bash
VITE_TRAINING_SERVER_URL=http://127.0.0.1:8765
```

## API Endpoints

- `GET /api/health`
- `GET /api/training/status`
- `POST /api/training/autotrain/start`
- `POST /api/training/autotrain/stop`
- `GET /api/models/registry`
- `GET /api/training/logs?limit=50`
- `GET /api/training/summary`
- `GET /api/arena/results`

## Runtime Files

The server reads and writes under `data/`:

- `data/training/autotrain_state.json`
- `data/training/autotrain_summary.json`
- `data/training/autotrain_log.jsonl`
- `data/training/server_last_stdout.log`
- `data/training/server_last_stderr.log`
- `data/models/registry.json`
- `data/models/arena/*.json`

These files are local training artifacts and are not intended to be committed.

## Troubleshooting

If the Training tab says the server is offline, run `npm run training:server` from the repository root and press refresh.

If AutoTrain exits immediately, inspect `data/training/server_last_stderr.log` first. It captures the subprocess stderr from `python -m oetongsu_ml.autotrain`.

If dependencies are missing, activate the ML virtual environment and run:

```bash
cd ml
pip install -r requirements.txt
```
