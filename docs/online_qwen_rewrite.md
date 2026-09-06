# Local Qwen Rewrite

The observation-aware rewrite path runs entirely on the server. Qwen is loaded
once by the server's dedicated `.venv-qwen` process. The LIBERO runtime does
not import Qwen or share its Python dependencies.

## Communication

The Qwen process and runtime communicate through a Unix domain socket under
`PROBE/tmp/qwen-ipc`. No public port or API key is used. The runtime container
mounts that directory as `/app/qwen-ipc`.

Put the model weights under:

```text
PROBE/models/Qwen2.5-VL-7B-Instruct
```

## Runtime split

- `RUNTIME_GPU_ID` is used by the runtime container.
- `SERVER_GPU_ID` is used by `openpi_server`.
- If you do not set them, both fall back to `GPU_ID`.

## Start Qwen

Use the environment that has already been verified to load Qwen:

```bash
source "$PROBE_ROOT/.venv-qwen/bin/activate"
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH="$PROBE_ROOT:$OPENPI_ROOT"

python -m probe.instruction_rewrite.qwen_socket \
  --model-dir "$PROBE_ROOT/models/Qwen2.5-VL-7B-Instruct" \
  --socket-path "$PROBE_ROOT/tmp/qwen-ipc/qwen.sock"
```

## Enable runtime rewrite

Set:

```bash
export INSTRUCTION_REWRITER_SOCKET="$PROBE_ROOT/tmp/qwen-ipc/qwen.sock"
export INSTRUCTION_REWRITER_SOURCE=original
```

Then run the usual rollout script, for example:

```bash
./scripts/run_libero_plus_conditions.sh
```

When `INSTRUCTION_REWRITER_SOCKET` is set, the collector sends the current
observation and source instruction to Qwen at each replanning step, then sends
the returned instruction to the frozen VLA policy. If Qwen is unavailable, the
client falls back to the source instruction and records the error.
