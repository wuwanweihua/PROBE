# Local Qwen Rewrite

This repo now supports an observation-aware instruction rewrite path that runs entirely on the server.
The launch script rewrites any model path under `PROBE/models/...` to the container mount `/app/models/...`.

## Layout

- Put the model weights under `PROBE/models/Qwen2.5-VL-7B-Instruct` on the host.
- The runtime container sees that directory as `/app/models/Qwen2.5-VL-7B-Instruct`.

## Runtime split

- `RUNTIME_GPU_ID` is used by the runtime container.
- `SERVER_GPU_ID` is used by `openpi_server`.
- If you do not set them, both fall back to `GPU_ID`.

## Enable rewrite

Set:

```bash
export INSTRUCTION_REWRITER_MODEL_DIR=/app/models/Qwen2.5-VL-7B-Instruct
export INSTRUCTION_REWRITER_SOURCE=original
```

Then run the usual rollout script, for example:

```bash
./scripts/run_libero_plus_conditions.sh
```

When `INSTRUCTION_REWRITER_MODEL_DIR` is set, the collector rewrites the source instruction at each replanning step before sending it to the frozen VLA policy.
