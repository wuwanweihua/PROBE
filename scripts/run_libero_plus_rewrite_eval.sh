#!/usr/bin/env bash
set -euo pipefail

PROBE_ROOT="${PROBE_ROOT:-/home/nvidia/yutao/lyh/PROBE}"
OPENPI_ROOT="${OPENPI_ROOT:-$PROBE_ROOT/src/openpi}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-$PROBE_ROOT/src/LIBERO-plus}"
GPU_ID="${GPU_ID:-5}"
PORT="${PORT:-18000}"
TASK_SUITE="${TASK_SUITE:-libero_10}"
CHECKPOINT_URI="${CHECKPOINT_URI:-gs://openpi-assets/checkpoints/pi05_libero}"
REWRITES_PATH="${REWRITES_PATH:-/data/libero_plus_rewrites/libero10_hard50_gpt55_rewrites.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-/data/libero_plus_rewrite_eval/libero10_hard50_gpt55_results.jsonl}"
MAX_TASKS="${MAX_TASKS:-20}"
MAX_REWRITES_PER_TASK="${MAX_REWRITES_PER_TASK:-5}"

cd "$OPENPI_ROOT"

mkdir -p "$PROBE_ROOT/logs/libero_plus_rewrite" "$OPENPI_ROOT/configs" "$OPENPI_ROOT/probe"
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
  docker compose "${COMPOSE_ARGS[@]}" logs openpi_server > "$PROBE_ROOT/logs/libero_plus_rewrite/openpi_server.log" 2>/dev/null || true
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
  /.venv/bin/python -m probe.instruction_rewrite.eval_rewrites_libero_plus \
  --rewrites "$REWRITES_PATH" \
  --output "$OUTPUT_PATH" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --task-suite-name "$TASK_SUITE" \
  --max-tasks "$MAX_TASKS" \
  --max-rewrites-per-task "$MAX_REWRITES_PER_TASK"
