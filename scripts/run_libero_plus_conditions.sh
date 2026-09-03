#!/usr/bin/env bash
set -euo pipefail

PROBE_ROOT="${PROBE_ROOT:-/home/nvidia/yutao/lyh/PROBE}"
OPENPI_ROOT="${OPENPI_ROOT:-$PROBE_ROOT/src/openpi}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-$PROBE_ROOT/src/LIBERO-plus}"
GPU_ID="${GPU_ID:-5}"
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

cd "$OPENPI_ROOT"

mkdir -p "$PROBE_ROOT/logs/libero_plus_conditions" "$OPENPI_ROOT/configs" "$OPENPI_ROOT/probe"
rsync -a "$PROBE_ROOT/probe/" "$OPENPI_ROOT/probe/"
rsync -a "$PROBE_ROOT/configs/" "$OPENPI_ROOT/configs/"

export GPU_ID
export LIBERO_PLUS_ROOT
export OPENPI_DATA_HOME="$PROBE_ROOT/cache/openpi"
export HF_HOME="$PROBE_ROOT/cache/huggingface"
export XDG_CACHE_HOME="$PROBE_ROOT/cache"
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

docker compose "${COMPOSE_ARGS[@]}" run --rm --no-deps runtime \
  /.venv/bin/python -m probe.rollout.collect_condition_calls \
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

docker compose "${COMPOSE_ARGS[@]}" run --rm --no-deps runtime \
  /.venv/bin/python -m probe.data.validate_week1 \
  --dataset "$OUTPUT_DIR" \
  --min-records "$MIN_RECORDS" \
  --min-failure-fraction "$MIN_FAILURE_FRACTION" \
  --expected-k "$K_SAMPLES"
