#!/usr/bin/env bash
set -euo pipefail

PROBE_ROOT="${PROBE_ROOT:-/home/nvidia/yutao/lyh/PROBE}"
OPENPI_ROOT="${OPENPI_ROOT:-$PROBE_ROOT/src/openpi}"
GPU_ID="${GPU_ID:-5}"
PORT="${PORT:-18000}"
TARGET_RECORDS="${TARGET_RECORDS:-500}"
K_SAMPLES="${K_SAMPLES:-32}"
TASK_SUITE="${TASK_SUITE:-libero_10}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/week1/pi05_libero}"
CHECKPOINT_URI="${CHECKPOINT_URI:-gs://openpi-assets/checkpoints/pi05_libero}"
PERTURBATION_RATE="${PERTURBATION_RATE:-0.0}"
PERTURBATION_MODES="${PERTURBATION_MODES:-drop_object_words,shuffle_words,generic_instruction}"

cd "$OPENPI_ROOT"

mkdir -p "$PROBE_ROOT/logs/week1" "$OPENPI_ROOT/configs" "$OPENPI_ROOT/probe"
rsync -a "$PROBE_ROOT/probe/" "$OPENPI_ROOT/probe/"
rsync -a "$PROBE_ROOT/configs/" "$OPENPI_ROOT/configs/"

export GPU_ID
export OPENPI_DATA_HOME="$PROBE_ROOT/cache/openpi"
export HF_HOME="$PROBE_ROOT/cache/huggingface"
export XDG_CACHE_HOME="$PROBE_ROOT/cache"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export SERVER_ARGS="--port $PORT --env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir $CHECKPOINT_URI"

COMPOSE_ARGS=(
  -f "$OPENPI_ROOT/examples/libero/compose.yml"
  -f "$PROBE_ROOT/scripts/compose.gpu-device.override.yml"
)

cleanup() {
  cd "$OPENPI_ROOT"
  docker compose "${COMPOSE_ARGS[@]}" logs openpi_server > "$PROBE_ROOT/logs/week1/openpi_server.log" 2>/dev/null || true
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
  --target-records "$TARGET_RECORDS" \
  --k-samples "$K_SAMPLES" \
  --perturbation-rate "$PERTURBATION_RATE" \
  --perturbation-modes "$PERTURBATION_MODES" \
  --output-dir "$OUTPUT_DIR"

docker compose "${COMPOSE_ARGS[@]}" run --rm --no-deps runtime \
  /.venv/bin/python -m probe.data.validate_week1 \
  --dataset "$OUTPUT_DIR" \
  --min-records "$TARGET_RECORDS" \
  --min-failure-fraction 0.2 \
  --expected-k "$K_SAMPLES"
