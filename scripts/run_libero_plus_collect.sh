#!/usr/bin/env bash
set -euo pipefail

PROBE_ROOT="${PROBE_ROOT:-/home/nvidia/yutao/lyh/PROBE}"
OPENPI_ROOT="${OPENPI_ROOT:-$PROBE_ROOT/src/openpi}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-$PROBE_ROOT/src/LIBERO-plus}"
GPU_ID="${GPU_ID:-5}"
PORT="${PORT:-18000}"
TASK_SUITE="${TASK_SUITE:-libero_10}"
TASK_ORDER="${TASK_ORDER:-all}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
TARGET_RECORDS="${TARGET_RECORDS:-1000}"
K_SAMPLES="${K_SAMPLES:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/libero_plus_baseline/libero10_language_k1}"
CHECKPOINT_URI="${CHECKPOINT_URI:-gs://openpi-assets/checkpoints/pi05_libero}"
MIN_RECORDS="${MIN_RECORDS:-1}"
MIN_FAILURE_FRACTION="${MIN_FAILURE_FRACTION:-0.0}"

cd "$OPENPI_ROOT"

mkdir -p "$PROBE_ROOT/logs/libero_plus_collect" "$OPENPI_ROOT/configs" "$OPENPI_ROOT/probe"
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

cleanup() {
  cd "$OPENPI_ROOT"
  docker compose "${COMPOSE_ARGS[@]}" logs openpi_server > "$PROBE_ROOT/logs/libero_plus_collect/openpi_server.log" 2>/dev/null || true
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
  /.venv/bin/python -m probe.rollout.collect_calls \
  --config configs/week1_pi05_libero.yaml \
  --host 0.0.0.0 \
  --port "$PORT" \
  --task-suite-name "$TASK_SUITE" \
  --task-order "$TASK_ORDER" \
  --num-trials-per-task "$NUM_TRIALS_PER_TASK" \
  --target-records "$TARGET_RECORDS" \
  --k-samples "$K_SAMPLES" \
  --output-dir "$OUTPUT_DIR"

docker compose "${COMPOSE_ARGS[@]}" run --rm --no-deps runtime \
  /.venv/bin/python -m probe.data.validate_week1 \
  --dataset "$OUTPUT_DIR" \
  --min-records "$MIN_RECORDS" \
  --min-failure-fraction "$MIN_FAILURE_FRACTION" \
  --expected-k "$K_SAMPLES"
