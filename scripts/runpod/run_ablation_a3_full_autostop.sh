#!/usr/bin/env bash
set -u
set -o pipefail

# Sprint 38 / RunPod A3 full ablation job.
# Requires /workspace/Checkmate/oetongsu_runpod_a3_artifacts.tgz plus RUNPOD_API_KEY/RUNPOD_POD_ID for autostop.

ROOT="${ROOT:-/workspace/Checkmate}"
TRAINING_DIR="$ROOT/data/training"
ABLATION_PROGRESS="$TRAINING_DIR/runpod_a3_ablation_progress.json"
ABLATION_LOG="$TRAINING_DIR/ablation_a3_runpod.log"
AUTOSTOP_LOG="$TRAINING_DIR/autostop.log"
INPUT_ARTIFACT="$ROOT/oetongsu_runpod_a3_artifacts.tgz"
OUTPUT_ARTIFACT="$ROOT/oetongsu_runpod_a3_ablation_artifacts.tgz"

MAX_SECONDS="${MAX_SECONDS:-28800}"
AUTOSTOP_SECONDS="${AUTOSTOP_SECONDS:-30600}"
AUTOSTOP_AFTER_SECONDS="${AUTOSTOP_AFTER_SECONDS:-$AUTOSTOP_SECONDS}"
WEBHOOK_URL="${WEBHOOK_URL:-}"

ABLATION_LRS="${ABLATION_LRS:-0.001 0.0003 0.0001}"
ABLATION_EPOCHS="${ABLATION_EPOCHS:-1}"
ABLATION_BATCH_SIZE="${ABLATION_BATCH_SIZE:-64}"
EVAL_GAMES="${EVAL_GAMES:-20}"
EVAL_SIMULATIONS="${EVAL_SIMULATIONS:-48}"
EVAL_MAX_PLIES="${EVAL_MAX_PLIES:-150}"
EVAL_DRAW_MARGIN="${EVAL_DRAW_MARGIN:-1.5}"
RUN_PAIRWISE_COMPARE="${RUN_PAIRWISE_COMPARE:-false}"

START_TS="$(date +%s)"
JOB_STATUS="starting"
WORK_PID=""
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
  cat > "$ABLATION_PROGRESS" <<JSON
{
  "job": "runpod_a3_full_ablation",
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
  echo "[$(date -Is)] $*" | tee -a "$ABLATION_LOG"
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

package_artifacts() {
  cd "$ROOT"
  local paths=()
  for path in \
    data/training/ablation_a3 \
    data/training/a3_regression_diagnostics_report.md \
    data/training/runpod_a3_ablation_progress.json \
    data/training/ablation_a3_runpod.log \
    data/training/autostop.log \
    data/models/arena \
    data/models/checkpoints/supervised_v0001.pt \
    data/models/checkpoints/az_iter_000002.pt \
    data/models/checkpoints/az_iter_000003.pt \
    data/selfplay/az_iter_000003.jsonl \
    data/selfplay/az_iter_000003_summary.json
  do
    if [ -e "$path" ]; then
      paths+=("$path")
    else
      log "package skip missing: $path"
    fi
  done

  if [ "${#paths[@]}" -eq 0 ]; then
    log "ERROR: no files available for packaging"
    return 1
  fi

  tar -czf "$OUTPUT_ARTIFACT" "${paths[@]}"
  ls -lh "$OUTPUT_ARTIFACT" | tee -a "$ABLATION_LOG"
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
    write_progress "ablation_running" "$estimated" "Full A3 ablation is running. Percent is elapsed-time based."
    sleep 30
  done
}

preflight_cuda() {
  nvidia-smi | tee -a "$ABLATION_LOG"
  python - <<'PY' | tee -a "$ABLATION_LOG"
import sys
import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
if not torch.cuda.is_available():
    sys.exit(2)
PY
}

apply_a3_artifact() {
  cd "$ROOT"
  if [ ! -f "$INPUT_ARTIFACT" ]; then
    log "ERROR: missing artifact: $INPUT_ARTIFACT"
    return 1
  fi

  tar -tzf "$INPUT_ARTIFACT" | head -50 | tee -a "$ABLATION_LOG"
  tar -xzf "$INPUT_ARTIFACT" -C "$ROOT"

  test -f "$ROOT/data/training/autotrain_state.json"
  test -f "$ROOT/data/models/checkpoints/supervised_v0001.pt"
  test -f "$ROOT/data/models/checkpoints/az_iter_000002.pt"
  test -f "$ROOT/data/models/checkpoints/az_iter_000003.pt"
  test -f "$ROOT/data/selfplay/az_iter_000003.jsonl"
  test -f "$ROOT/data/selfplay/az_iter_000003_summary.json"

  python - <<'PY' | tee -a "$ABLATION_LOG"
import json
from pathlib import Path

state = json.loads(Path("data/training/autotrain_state.json").read_text(encoding="utf-8"))
completed = int(state.get("completedIterations") or 0)
latest_champion = state.get("latestChampionVersion")
latest_candidate = state.get("latestCandidateVersion")

print("completedIterations:", completed)
print("latestChampionVersion:", latest_champion)
print("latestCandidateVersion:", latest_candidate)

if completed < 3:
    raise SystemExit("completedIterations must be >= 3 for A3 ablation")
if latest_champion != "supervised_v0001":
    raise SystemExit("latestChampionVersion must be supervised_v0001")
if latest_candidate != "az_iter_000003":
    raise SystemExit("latestCandidateVersion must be az_iter_000003")
PY

  if ! (cd "$ROOT/ml" && python -m oetongsu_ml.ablation_retrain --help >/dev/null); then
    log "ERROR: python -m oetongsu_ml.ablation_retrain --help failed"
    return 1
  fi
  if ! (cd "$ROOT/ml" && python -m oetongsu_ml.ablation_evaluate --help >/dev/null); then
    log "ERROR: python -m oetongsu_ml.ablation_evaluate --help failed"
    return 1
  fi
  log "A3 ablation preflight passed"
}

fail_before_ablation() {
  local message="$1"
  JOB_STATUS="failed"
  write_progress "failed" 100 "$message"
  log "ERROR: $message"
  send_webhook "Oetongsu RunPod A3 ablation failed before run: $message"
  package_artifacts || true
  stop_pod || true
}

run_ablation() {
  cd "$ROOT/ml"
  python -m oetongsu_ml.ablation_retrain \
    --data ../data/selfplay/az_iter_000003.jsonl \
    --resume ../data/models/checkpoints/supervised_v0001.pt \
    --outputDir ../data/training/ablation_a3 \
    --channels 64 \
    --batchSize "$ABLATION_BATCH_SIZE" \
    --epochs "$ABLATION_EPOCHS" \
    --lrs $ABLATION_LRS \
    --seed 7

  python -m oetongsu_ml.ablation_evaluate \
    --champion ../data/models/checkpoints/supervised_v0001.pt \
    --candidates ../data/training/ablation_a3/*.pt \
    --games "$EVAL_GAMES" \
    --simulations "$EVAL_SIMULATIONS" \
    --maxPlies "$EVAL_MAX_PLIES" \
    --adjudicationDrawMargin "$EVAL_DRAW_MARGIN" \
    --output ../data/training/ablation_a3/evaluation_summary.json

  if [ "$RUN_PAIRWISE_COMPARE" = "true" ]; then
    python -m oetongsu_ml.arena_compare \
      --left ../data/models/checkpoints/supervised_v0001.pt \
      --right ../data/models/checkpoints/az_iter_000003.pt \
      --leftName supervised_v0001 \
      --rightName az_iter_000003 \
      --games "$EVAL_GAMES" \
      --simulations "$EVAL_SIMULATIONS" \
      --maxPlies "$EVAL_MAX_PLIES" \
      --adjudicationDrawMargin "$EVAL_DRAW_MARGIN" \
      --ruleset kakao-like \
      --output ../data/training/arena_compare_supervised_vs_a3_full.json
  fi
}

run_ablation_with_timeout() {
  timeout "$MAX_SECONDS" bash -c 'run_ablation' 2>&1 | tee "$ABLATION_LOG" &
  WORK_PID=$!
  monitor_progress "$WORK_PID" &
  MONITOR_PID=$!

  wait "$WORK_PID"
  local status=$?
  if [ -n "$MONITOR_PID" ]; then
    kill "$MONITOR_PID" >/dev/null 2>&1 || true
  fi
  return "$status"
}

main() {
  JOB_STATUS="running"
  write_progress "preflight" 2 "Preflight checks started"
  send_webhook "Oetongsu RunPod A3 ablation started: preflight"

  cd "$ROOT"

  write_progress "cuda_check" 4 "Checking GPU and CUDA"
  if ! preflight_cuda; then
    fail_before_ablation "CUDA preflight failed. Ablation was not started."
    return 1
  fi

  write_progress "verify_artifact" 8 "Applying A3 artifact and verifying ablation inputs"
  if ! apply_a3_artifact; then
    fail_before_ablation "A3 artifact/preflight failed. Ablation was not started."
    return 1
  fi

  write_progress "start_watchdog" 12 "Starting autostop watchdog"
  start_watchdog

  write_progress "ablation_start" 15 "Starting full ablation retrain and evaluation"
  run_ablation_with_timeout
  local ablation_status=$?

  if [ "$ablation_status" -eq 0 ]; then
    JOB_STATUS="packaging"
    write_progress "packaging" 90 "Ablation complete. Packaging artifacts."
  else
    JOB_STATUS="failed"
    write_progress "failed" 90 "Ablation failed or timed out with status $ablation_status. Packaging partial artifacts."
  fi

  package_artifacts

  if [ "$ablation_status" -eq 0 ]; then
    JOB_STATUS="completed"
    write_progress "completed" 100 "A3 ablation complete. Artifact packaged. Auto Stop requested."
    send_webhook "Oetongsu RunPod A3 ablation completed: artifact packaged. Stop requested."
  else
    JOB_STATUS="failed"
    write_progress "failed" 100 "A3 ablation failed or timed out. Partial artifact packaged. Stop requested."
    send_webhook "Oetongsu RunPod A3 ablation failed or timed out: partial artifact packaged. Stop requested."
  fi

  cancel_watchdog
  stop_pod || true

  return "$ablation_status"
}

export ROOT ABLATION_BATCH_SIZE ABLATION_EPOCHS ABLATION_LRS
export EVAL_GAMES EVAL_SIMULATIONS EVAL_MAX_PLIES EVAL_DRAW_MARGIN RUN_PAIRWISE_COMPARE
export -f run_ablation
main "$@"
