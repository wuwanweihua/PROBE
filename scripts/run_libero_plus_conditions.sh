#!/usr/bin/env bash
set -euo pipefail

PROBE_ROOT="${PROBE_ROOT:-/home/nvidia/yutao/lyh/PROBE}"
OPENPI_ROOT="${OPENPI_ROOT:-$PROBE_ROOT/src/openpi}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-$PROBE_ROOT/src/LIBERO-plus}"
GPU_ID="${GPU_ID:-5}"
RUNTIME_GPU_ID="${RUNTIME_GPU_ID:-$GPU_ID}"
SERVER_GPU_ID="${SERVER_GPU_ID:-$GPU_ID}"
PORT="${PORT:-18000}"
TASK_SUITE="${TASK_SUITE:-libero_10}"
CONDITIONS_PATH="${CONDITIONS_PATH:-/data/libero_plus_conditions/libero10_gpt55_conditions.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/libero_plus_conditions/libero10_condition_rollouts}"
NUM_TRIALS_PER_CONDITION="${NUM_TRIALS_PER_CONDITION:-8}"
K_SAMPLES="${K_SAMPLES:-32}"
CONDITION_TYPES="${CONDITION_TYPES:-original,better,worse}"
MAX_BASE_STATES="${MAX_BASE_STATES:-}"
MAX_EPISODES="${MAX_EPISODES:-}"
EXEC_SEED_START="${EXEC_SEED_START:-700000}"
PROBE_SEED_START="${PROBE_SEED_START:-100000}"
CHECKPOINT_URI="${CHECKPOINT_URI:-gs://openpi-assets/checkpoints/pi05_libero}"
MIN_RECORDS="${MIN_RECORDS:-1}"
MIN_FAILURE_FRACTION="${MIN_FAILURE_FRACTION:-0.0}"
INSTRUCTION_REWRITER_MODEL_DIR="${INSTRUCTION_REWRITER_MODEL_DIR:-}"
INSTRUCTION_REWRITER_SOCKET="${INSTRUCTION_REWRITER_SOCKET:-}"
INSTRUCTION_REWRITER_SOCKET_HOST="$INSTRUCTION_REWRITER_SOCKET"
INSTRUCTION_REWRITER_SOURCE="${INSTRUCTION_REWRITER_SOURCE:-original}"
INSTRUCTION_REWRITER_MAX_NEW_TOKENS="${INSTRUCTION_REWRITER_MAX_NEW_TOKENS:-96}"
INSTRUCTION_REWRITER_TEMPERATURE="${INSTRUCTION_REWRITER_TEMPERATURE:-0.0}"
INSTRUCTION_REWRITER_TOP_P="${INSTRUCTION_REWRITER_TOP_P:-0.9}"

if [[ -n "$INSTRUCTION_REWRITER_MODEL_DIR" && "$INSTRUCTION_REWRITER_MODEL_DIR" == "$PROBE_ROOT/models"* ]]; then
  INSTRUCTION_REWRITER_MODEL_DIR="/app/models${INSTRUCTION_REWRITER_MODEL_DIR#$PROBE_ROOT/models}"
fi

if [[ -n "$INSTRUCTION_REWRITER_SOCKET" && "$INSTRUCTION_REWRITER_SOCKET" == "$PROBE_ROOT/tmp/qwen-ipc"* ]]; then
  INSTRUCTION_REWRITER_SOCKET="/app/qwen-ipc${INSTRUCTION_REWRITER_SOCKET#$PROBE_ROOT/tmp/qwen-ipc}"
fi

cd "$OPENPI_ROOT"

mkdir -p "$PROBE_ROOT/logs/libero_plus_conditions" "$PROBE_ROOT/tmp/qwen-ipc" "$OPENPI_ROOT/configs" "$OPENPI_ROOT/probe"
RSYNC_CODE_ARGS=(--archive --no-perms --no-owner --no-group --omit-dir-times --exclude='__pycache__/' --exclude='*.pyc')
rsync "${RSYNC_CODE_ARGS[@]}" "$PROBE_ROOT/probe/" "$OPENPI_ROOT/probe/"
rsync "${RSYNC_CODE_ARGS[@]}" "$PROBE_ROOT/configs/" "$OPENPI_ROOT/configs/"

if [[ -n "$INSTRUCTION_REWRITER_SOCKET_HOST" ]]; then
  echo "Waiting for Qwen rewriter socket at $INSTRUCTION_REWRITER_SOCKET_HOST ..."
  for _ in $(seq 1 60); do
    if [[ -S "$INSTRUCTION_REWRITER_SOCKET_HOST" ]]; then
      break
    fi
    sleep 2
  done
  if [[ ! -S "$INSTRUCTION_REWRITER_SOCKET_HOST" ]]; then
    echo "Qwen rewriter socket was not found: $INSTRUCTION_REWRITER_SOCKET_HOST" >&2
    exit 1
  fi
fi

export GPU_ID
export RUNTIME_GPU_ID
export SERVER_GPU_ID
export LIBERO_PLUS_ROOT
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$PROBE_ROOT/cache/openpi}"
export HF_HOME="${HF_HOME:-$PROBE_ROOT/cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROBE_ROOT/cache}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export SERVER_ARGS="--port $PORT --env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir $CHECKPOINT_URI"

COMPOSE_ARGS=(
  -f "$OPENPI_ROOT/examples/libero/compose.yml"
  -f "$PROBE_ROOT/scripts/compose.gpu-device.override.yml"
  -f "$PROBE_ROOT/scripts/compose.libero-plus.override.yml"
)

EXTRA_ARGS=()
if [[ -n "$MAX_BASE_STATES" ]]; then
  EXTRA_ARGS+=(--max-base-states "$MAX_BASE_STATES")
fi
if [[ -n "$MAX_EPISODES" ]]; then
  EXTRA_ARGS+=(--max-episodes "$MAX_EPISODES")
fi
if [[ -n "$INSTRUCTION_REWRITER_MODEL_DIR" ]]; then
  EXTRA_ARGS+=(--instruction-rewriter-model-dir "$INSTRUCTION_REWRITER_MODEL_DIR")
  EXTRA_ARGS+=(--instruction-rewriter-source "$INSTRUCTION_REWRITER_SOURCE")
  EXTRA_ARGS+=(--instruction-rewriter-max-new-tokens "$INSTRUCTION_REWRITER_MAX_NEW_TOKENS")
  EXTRA_ARGS+=(--instruction-rewriter-temperature "$INSTRUCTION_REWRITER_TEMPERATURE")
  EXTRA_ARGS+=(--instruction-rewriter-top-p "$INSTRUCTION_REWRITER_TOP_P")
fi
if [[ -n "$INSTRUCTION_REWRITER_SOCKET" ]]; then
  EXTRA_ARGS+=(--instruction-rewriter-socket "$INSTRUCTION_REWRITER_SOCKET")
fi

cleanup() {
  cd "$OPENPI_ROOT"
  docker compose "${COMPOSE_ARGS[@]}" logs openpi_server > "$PROBE_ROOT/logs/libero_plus_conditions/openpi_server.log" 2>/dev/null || true
  docker compose "${COMPOSE_ARGS[@]}" down --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker compose "${COMPOSE_ARGS[@]}" down --remove-orphans
docker compose "${COMPOSE_ARGS[@]}" up -d openpi_server

echo "Waiting for policy server on port $PORT ..."
for _ in $(seq 1 120); do
  if ss -ltn | grep -q ":$PORT "; then
    break
  fi
  sleep 5
done

docker compose "${COMPOSE_ARGS[@]}" logs --tail=80 openpi_server

docker compose "${COMPOSE_ARGS[@]}" run --rm --no-deps \
  -e INSTRUCTION_REWRITER_SOCKET="$INSTRUCTION_REWRITER_SOCKET" \
  -e INSTRUCTION_REWRITER_MODEL_DIR="$INSTRUCTION_REWRITER_MODEL_DIR" \
  -e INSTRUCTION_REWRITER_SOURCE="$INSTRUCTION_REWRITER_SOURCE" \
  -e INSTRUCTION_REWRITER_MAX_NEW_TOKENS="$INSTRUCTION_REWRITER_MAX_NEW_TOKENS" \
  -e INSTRUCTION_REWRITER_TEMPERATURE="$INSTRUCTION_REWRITER_TEMPERATURE" \
  -e INSTRUCTION_REWRITER_TOP_P="$INSTRUCTION_REWRITER_TOP_P" \
  --entrypoint /.venv/bin/python \
  runtime \
  -m probe.rollout.collect_condition_calls \
  --conditions "$CONDITIONS_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --task-suite-name "$TASK_SUITE" \
  --condition-types "$CONDITION_TYPES" \
  --num-trials-per-condition "$NUM_TRIALS_PER_CONDITION" \
  --k-samples "$K_SAMPLES" \
  --exec-seed-start "$EXEC_SEED_START" \
  --probe-seed-start "$PROBE_SEED_START" \
  "${EXTRA_ARGS[@]}"

docker compose "${COMPOSE_ARGS[@]}" run --rm --no-deps \
  --entrypoint /.venv/bin/python \
  runtime \
  -m probe.data.validate_week1 \
  --dataset "$OUTPUT_DIR" \
  --min-records "$MIN_RECORDS" \
  --min-failure-fraction "$MIN_FAILURE_FRACTION" \
  --expected-k "$K_SAMPLES"
