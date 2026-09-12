#!/usr/bin/env bash
set -euo pipefail

PROBE_ROOT="${PROBE_ROOT:-/home/nvidia/yutao/lyh/PROBE}"
OPENPI_ROOT="${OPENPI_ROOT:-$PROBE_ROOT/src/openpi}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-$PROBE_ROOT/src/LIBERO-plus}"
GPU_ID="${GPU_ID:-6}"
RUNTIME_GPU_ID="${RUNTIME_GPU_ID:-$GPU_ID}"
SERVER_GPU_ID="${SERVER_GPU_ID:-$GPU_ID}"
PORT="${PORT:-18320}"
COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-week2_env_probe_gpu${GPU_ID}}"
CONDITIONS_PATH="${CONDITIONS_PATH:-}"
TASK_IDS="${TASK_IDS:-}"
ORIGINAL_SUCCESS_JSON="${ORIGINAL_SUCCESS_JSON:-}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR to the Week 2 output directory}"
LABELS_PATH="${LABELS_PATH:-}"
MAX_BASE_STATES="${MAX_BASE_STATES:-2}"
K_SAMPLES="${K_SAMPLES:-4}"
TASK_SUITE="${TASK_SUITE:-libero_10}"
CHECKPOINT_URI="${CHECKPOINT_URI:-gs://openpi-assets/checkpoints/pi05_libero}"

cd "$OPENPI_ROOT"
mkdir -p "$PROBE_ROOT/logs" "$OPENPI_ROOT/probe"

rsync -a \
  --no-perms --no-owner --no-group --omit-dir-times \
  --exclude='__pycache__/' --exclude='*.pyc' \
  "$PROBE_ROOT/probe/" "$OPENPI_ROOT/probe/"

export GPU_ID RUNTIME_GPU_ID SERVER_GPU_ID LIBERO_PLUS_ROOT
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$PROBE_ROOT/cache/openpi}"
export HF_HOME="${HF_HOME:-$PROBE_ROOT/cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROBE_ROOT/cache}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export SERVER_ARGS="--port $PORT --env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir $CHECKPOINT_URI"

COMPOSE_ARGS=(
  -f "$OPENPI_ROOT/examples/libero/compose.yml"
  -f "$PROBE_ROOT/scripts/compose.gpu-device.override.yml"
  -f "$PROBE_ROOT/scripts/compose.libero-plus.override.yml"
)

test -f "$OPENPI_ROOT/examples/libero/compose.yml"
test -f "$PROBE_ROOT/scripts/compose.gpu-device.override.yml"
test -f "$PROBE_ROOT/scripts/compose.libero-plus.override.yml"
if [[ -n "$TASK_IDS" ]]; then
  CONDITION_TYPES="${CONDITION_TYPES:-original}"
  if [[ "$CONDITION_TYPES" != "original" ]]; then
    echo "TASK_IDS mode supports only CONDITION_TYPES=original" >&2
    exit 1
  fi
elif [[ -n "$CONDITIONS_PATH" ]]; then
  CONDITION_TYPES="${CONDITION_TYPES:-original,better,worse}"
  test -f "$CONDITIONS_PATH"
else
  echo "Set TASK_IDS or CONDITIONS_PATH" >&2
  exit 1
fi

echo "Expected policy server args: $SERVER_ARGS"
docker compose -p "$COMPOSE_PROJECT_NAME" "${COMPOSE_ARGS[@]}" config \
  | grep -E -- "pi05_libero|policy\.config[= ]+pi05_libero" >/dev/null \
  || { echo "compose config does not contain pi05_libero"; exit 1; }

docker compose -p "$COMPOSE_PROJECT_NAME" "${COMPOSE_ARGS[@]}" down --remove-orphans || true
docker compose -p "$COMPOSE_PROJECT_NAME" "${COMPOSE_ARGS[@]}" up -d openpi_server

for _ in $(seq 1 120); do
  if ss -ltn | grep -q ":$PORT "; then
    break
  fi
  sleep 5
done

docker compose -p "$COMPOSE_PROJECT_NAME" "${COMPOSE_ARGS[@]}" logs --tail=120 openpi_server
docker compose -p "$COMPOSE_PROJECT_NAME" "${COMPOSE_ARGS[@]}" logs openpi_server \
  | grep -F "pi05_libero" >/dev/null \
  || { echo "server log did not confirm pi05_libero"; exit 1; }

RUN_ARGS=(
  --output-dir /app/week2_output
  --host 0.0.0.0
  --port "$PORT"
  --task-suite-name "$TASK_SUITE"
  --condition-types "$CONDITION_TYPES"
  --k-samples "$K_SAMPLES"
  --max-base-states "$MAX_BASE_STATES"
  --checkpoint-uri "$CHECKPOINT_URI"
)
if [[ -n "$TASK_IDS" ]]; then
  RUN_ARGS+=(--task-ids "$TASK_IDS")
  if [[ -n "$ORIGINAL_SUCCESS_JSON" ]]; then
    RUN_ARGS+=(--original-success-json "$ORIGINAL_SUCCESS_JSON")
  fi
else
  RUN_ARGS+=(--conditions /app/week2_conditions.jsonl)
fi
if [[ -n "$LABELS_PATH" ]]; then
  RUN_ARGS+=(--labels /app/week2_labels.jsonl)
fi

mkdir -p "$OUTPUT_DIR"
MOUNTS=(-v "$OUTPUT_DIR:/app/week2_output")
if [[ -n "$CONDITIONS_PATH" ]]; then
  MOUNTS+=(-v "$CONDITIONS_PATH:/app/week2_conditions.jsonl:ro")
fi
if [[ -n "$LABELS_PATH" ]]; then
  MOUNTS+=(-v "$LABELS_PATH:/app/week2_labels.jsonl:ro")
fi

docker compose -p "$COMPOSE_PROJECT_NAME" "${COMPOSE_ARGS[@]}" run --rm --no-deps \
  "${MOUNTS[@]}" \
  --entrypoint /.venv/bin/python \
  runtime \
  -m probe.rollout.collect_week2_env_probe \
  "${RUN_ARGS[@]}" \
  2>&1 | tee "$PROBE_ROOT/logs/week2_env_probe_${COMPOSE_PROJECT_NAME}.log"
