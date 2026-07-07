#!/usr/bin/env bash
set -u
set -o pipefail

# Sprint 35 / RunPod A3 calibrated resume job.
# Requires /workspace/Checkmate/oetongsu_runpod_a2_artifacts.tgz plus RUNPOD_API_KEY/RUNPOD_POD_ID for autostop.

ROOT="${ROOT:-/workspace/Checkmate}"
TRAINING_DIR="$ROOT/data/training"
PROGRESS_JSON="$TRAINING_DIR/runpod_a3_progress.json"
MAIN_LOG="$TRAINING_DIR/autotrain_a3_runpod.log"
AUTOSTOP_LOG="$TRAINING_DIR/autostop.log"
INPUT_ARTIFACT="$ROOT/oetongsu_runpod_a2_artifacts.tgz"
OUTPUT_ARTIFACT="$ROOT/oetongsu_runpod_a3_artifacts.tgz"

MAX_SECONDS="${MAX_SECONDS:-28800}"
AUTOSTOP_SECONDS="${AUTOSTOP_SECONDS:-30600}"
AUTOSTOP_AFTER_SECONDS="${AUTOSTOP_AFTER_SECONDS:-$AUTOSTOP_SECONDS}"
WEBHOOK_URL="${WEBHOOK_URL:-}"

START_TS="$(date +%s)"
JOB_STATUS="starting"
AUTOTRAIN_PID=""
MONITOR_PID=""

mkdir -p "$TRAINING_DIR"

json_escape() {
  python -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$1"
}

write_progress() {
  local phase="$1"
  local percent="$2"
  local message="$3"
  local now elapsed
  now="$(date +%s)"
  elapsed="$((now - START_TS))"
  cat > "$PROGRESS_JSON" <<JSON
{
  "job": "runpod_autotrain_a3",
  "status": "$JOB_STATUS",
  "phase": "$phase",
  "estimatedPercent": $percent,
  "elapsedSeconds": $elapsed,
  "elapsedText": "$(printf '%02d:%02d:%02d' $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60)))",
  "startedAtUnix": $START_TS,
  "updatedAtUnix": $now,
  "message": $(json_escape "$message"),
  "progressAccuracy": "coarse_wall_clock_estimate"
}
JSON
}

log() {
  echo "[$(date -Is)] $*" | tee -a "$MAIN_LOG"
}

send_webhook() {
  local text="$1"
  if [ -n "$WEBHOOK_URL" ]; then
    python -c 'import json, sys; print(json.dumps({"text": sys.argv[1]}))' "$text" \
      | curl -sS -X POST "$WEBHOOK_URL" \
          -H "Content-Type: application/json" \
          --data-binary @- >/dev/null 2>&1 || true
  fi
}

try_stop_pod_rest() {
  if [ -z "${RUNPOD_API_KEY:-}" ] || [ -z "${RUNPOD_POD_ID:-}" ]; then
    echo "[autostop] RUNPOD_API_KEY or RUNPOD_POD_ID missing" | tee -a "$AUTOSTOP_LOG"
    return 1
  fi

  echo "[autostop] trying REST stop endpoint" | tee -a "$AUTOSTOP_LOG"
  curl -sS -X POST "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}/stop" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -o "$TRAINING_DIR/autostop_rest_response.json" \
    -w "%{http_code}" > "$TRAINING_DIR/autostop_rest_http_code.txt" || return 1

  local code
  code="$(cat "$TRAINING_DIR/autostop_rest_http_code.txt")"
  echo "[autostop] REST http_code=$code" | tee -a "$AUTOSTOP_LOG"
  if [ "$code" -ge 200 ] && [ "$code" -lt 300 ]; then
    return 0
  fi
  return 1
}

try_stop_pod_graphql() {
  if [ -z "${RUNPOD_API_KEY:-}" ] || [ -z "${RUNPOD_POD_ID:-}" ]; then
    echo "[autostop] RUNPOD_API_KEY or RUNPOD_POD_ID missing" | tee -a "$AUTOSTOP_LOG"
    return 1
  fi

  echo "[autostop] trying GraphQL podStop mutation" | tee -a "$AUTOSTOP_LOG"
  python - <<'PY' > "$TRAINING_DIR/autostop_graphql_body.json"
import json
import os

pod_id = os.environ["RUNPOD_POD_ID"]
query = f"""
mutation {{
  podStop(input: {{ podId: "{pod_id}" }}) {{
    id
    desiredStatus
  }}
}}
"""
print(json.dumps({"query": query}))
PY

  curl -sS -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    --data-binary @"$TRAINING_DIR/autostop_graphql_body.json" \
    -o "$TRAINING_DIR/autostop_graphql_response.json" \
    -w "%{http_code}" > "$TRAINING_DIR/autostop_graphql_http_code.txt" || return 1

  local code
  code="$(cat "$TRAINING_DIR/autostop_graphql_http_code.txt")"
  echo "[autostop] GraphQL http_code=$code" | tee -a "$AUTOSTOP_LOG"
  if [ "$code" -ge 200 ] && [ "$code" -lt 300 ]; then
    if grep -qi '"errors"' "$TRAINING_DIR/autostop_graphql_response.json"; then
      echo "[autostop] GraphQL response contains errors" | tee -a "$AUTOSTOP_LOG"
      cat "$TRAINING_DIR/autostop_graphql_response.json" | tee -a "$AUTOSTOP_LOG"
      return 1
    fi
    return 0
  fi
  return 1
}

stop_pod() {
  echo "[autostop] requested at $(date -Is)" | tee -a "$AUTOSTOP_LOG"
  if try_stop_pod_rest; then
    echo "[autostop] REST stop request accepted" | tee -a "$AUTOSTOP_LOG"
    return 0
  fi
  if try_stop_pod_graphql; then
    echo "[autostop] GraphQL stop request accepted" | tee -a "$AUTOSTOP_LOG"
    return 0
  fi
  echo "[autostop] FAILED. Manual Stop required in RunPod Console." | tee -a "$AUTOSTOP_LOG"
  return 1
}

start_watchdog() {
  if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    (
      sleep "$AUTOSTOP_AFTER_SECONDS"
      echo "[watchdog] max wall time reached: ${AUTOSTOP_AFTER_SECONDS}s" >> "$AUTOSTOP_LOG"
      package_artifacts >> "$AUTOSTOP_LOG" 2>&1 || true
      stop_pod >> "$AUTOSTOP_LOG" 2>&1 || true
    ) &
    echo $! > "$TRAINING_DIR/autostop_watchdog.pid"
    echo "[watchdog] started pid=$(cat "$TRAINING_DIR/autostop_watchdog.pid") seconds=$AUTOSTOP_AFTER_SECONDS" | tee -a "$AUTOSTOP_LOG"
  else
    echo "[watchdog] API key/pod id missing. Watchdog autostop disabled." | tee -a "$AUTOSTOP_LOG"
  fi
}

cancel_watchdog() {
  if [ -f "$TRAINING_DIR/autostop_watchdog.pid" ]; then
    local pid
    pid="$(cat "$TRAINING_DIR/autostop_watchdog.pid")"
    kill "$pid" >/dev/null 2>&1 || true
    echo "[watchdog] cancelled pid=$pid" | tee -a "$AUTOSTOP_LOG"
  fi
}

monitor_progress() {
  local pid="$1"
  while kill -0 "$pid" >/dev/null 2>&1; do
    local now elapsed estimated
    now="$(date +%s)"
    elapsed="$((now - START_TS))"
    estimated="$((15 + elapsed * 70 / MAX_SECONDS))"
    if [ "$estimated" -gt 85 ]; then
      estimated=85
    fi
    write_progress "autotrain_running" "$estimated" "AutoTrain A3 running. Percent is elapsed-time based, not exact internal game/batch progress."
    sleep 30
  done
}

preflight_cuda() {
  nvidia-smi | tee -a "$MAIN_LOG"
  python - <<'PY' | tee -a "$MAIN_LOG"
import sys
import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
if not torch.cuda.is_available():
    sys.exit(2)
PY
}

apply_a2_artifact() {
  cd "$ROOT"
  if [ ! -f "$INPUT_ARTIFACT" ]; then
    log "ERROR: missing artifact: $INPUT_ARTIFACT"
    return 1
  fi

  tar -tzf "$INPUT_ARTIFACT" | head -50 | tee -a "$MAIN_LOG"
  tar -xzf "$INPUT_ARTIFACT" -C "$ROOT"

  test -f "$ROOT/data/training/autotrain_state.json"
  test -f "$ROOT/data/models/registry.json"
  test -f "$ROOT/data/models/checkpoints/supervised_v0001.pt"
  test -f "$ROOT/data/models/checkpoints/az_iter_000002.pt"

  python - <<'PY' | tee -a "$MAIN_LOG"
import json
from pathlib import Path

state_path = Path("data/training/autotrain_state.json")
registry_path = Path("data/models/registry.json")
state = json.loads(state_path.read_text(encoding="utf-8"))
registry = json.loads(registry_path.read_text(encoding="utf-8"))
models = registry.get("models", [])

completed = int(state.get("completedIterations") or 0)
latest_champion = state.get("latestChampionVersion")
latest_candidate = state.get("latestCandidateVersion")
az1 = next((m for m in models if m.get("version") == "az_iter_000001"), None)
az2 = next((m for m in models if m.get("version") == "az_iter_000002"), None)

print("resume_completedIterations:", completed)
print("latestChampionVersion:", latest_champion)
print("latestCandidateVersion:", latest_candidate)
print("az_iter_000001_status:", az1.get("status") if az1 else None)
print("az_iter_000002_status:", az2.get("status") if az2 else None)

if completed < 2:
    raise SystemExit("completedIterations must be >= 2 for A3 resume")
if latest_champion != "supervised_v0001":
    raise SystemExit("latestChampionVersion must be supervised_v0001 before A3")
if latest_candidate != "az_iter_000002":
    raise SystemExit("latestCandidateVersion must be az_iter_000002 before A3")
if az1 is None:
    raise SystemExit("missing az_iter_000001 registry entry")
if az2 is None:
    raise SystemExit("missing az_iter_000002 registry entry")
if az2.get("status") != "rejected":
    raise SystemExit("az_iter_000002 status must be rejected before A3")
PY

  if ! (cd "$ROOT/ml" && python -m oetongsu_ml.autotrain --help | grep -q -- "--adjudicationDrawMargin"); then
    log "ERROR: python -m oetongsu_ml.autotrain --help does not expose --adjudicationDrawMargin"
    return 1
  fi
  log "AutoTrain help includes --adjudicationDrawMargin"
}

package_artifacts() {
  cd "$ROOT"
  tar -czf "$OUTPUT_ARTIFACT" \
    data/models/checkpoints \
    data/models/registry.json \
    data/models/arena \
    data/training \
    data/selfplay
  ls -lh "$OUTPUT_ARTIFACT" | tee -a "$MAIN_LOG"
}

fail_before_autotrain() {
  local phase="$1"
  local message="$2"
  JOB_STATUS="failed"
  write_progress "$phase" 100 "$message"
  log "ERROR: $message"
  send_webhook "Oetongsu RunPod A3 failed before AutoTrain: $message"
  package_artifacts || true
  stop_pod || true
}

run_autotrain_a3() {
  cd "$ROOT/ml"
  rm -f ../data/training/progress.json ../data/training/progress_events.jsonl
  timeout "$MAX_SECONDS" python -m oetongsu_ml.autotrain \
    --resume \
    --iterations 3 \
    --gamesPerIteration 100 \
    --simulations 48 \
    --arenaSimulations 48 \
    --maxPlies 150 \
    --trainEpochs 1 \
    --batchSize 64 \
    --channels 64 \
    --promotionGames 40 \
    --threshold 0.55 \
    --adjudicationDrawMargin 1.5 \
    --ruleset kakao-like \
    --strict \
    --selfplayWorkers 4 \
    --parallelSelfPlay \
    --progressPath ../data/training/progress.json \
    --progressEventsPath ../data/training/progress_events.jsonl 2>&1 | tee "$MAIN_LOG" &

  AUTOTRAIN_PID=$!
  monitor_progress "$AUTOTRAIN_PID" &
  MONITOR_PID=$!

  wait "$AUTOTRAIN_PID"
  local status=$?
  if [ -n "$MONITOR_PID" ]; then
    kill "$MONITOR_PID" >/dev/null 2>&1 || true
  fi
  return "$status"
}

main() {
  JOB_STATUS="running"
  write_progress "preflight" 2 "Preflight checks started"
  send_webhook "Oetongsu RunPod A3 started: preflight"

  cd "$ROOT"

  write_progress "cuda_check" 4 "Checking GPU and CUDA"
  if ! preflight_cuda; then
    fail_before_autotrain "failed" "CUDA preflight failed. A3 was not started."
    return 1
  fi

  write_progress "verify_artifact" 8 "Applying A2 artifact and verifying resume state for az_iter_000003"
  if ! apply_a2_artifact; then
    fail_before_autotrain "failed" "A2 artifact/resume preflight failed. A3 was not started."
    return 1
  fi

  write_progress "start_watchdog" 12 "Starting autostop watchdog"
  start_watchdog

  write_progress "autotrain_start" 15 "Starting AutoTrain A3"
  run_autotrain_a3
  local autotrain_status=$?

  if [ "$autotrain_status" -eq 0 ]; then
    JOB_STATUS="packaging"
    write_progress "packaging" 90 "AutoTrain complete. Packaging artifacts."
  else
    JOB_STATUS="failed"
    write_progress "failed" 90 "AutoTrain failed or timed out with status $autotrain_status. Packaging partial artifacts."
  fi

  package_artifacts

  if [ "$autotrain_status" -eq 0 ]; then
    JOB_STATUS="completed"
    write_progress "completed" 100 "A3 complete. Artifact packaged. Auto Stop requested."
    send_webhook "Oetongsu RunPod A3 completed: artifact packaged. Stop requested."
  else
    JOB_STATUS="failed"
    write_progress "failed" 100 "A3 failed or timed out. Partial artifact packaged. Stop requested."
    send_webhook "Oetongsu RunPod A3 failed or timed out: partial artifact packaged. Stop requested."
  fi

  cancel_watchdog
  stop_pod || true

  return "$autotrain_status"
}

main "$@"
