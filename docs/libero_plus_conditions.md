# LIBERO-Plus Original/Better/Worse Conditions

新的 pilot 主线只用 LIBERO-Plus `libero_10`。每个 base state 生成三个 instruction condition：

- `original`: 清洗后的原始 instruction，去掉 `view ... initstate ...` 元信息。
- `better`: GPT 生成的更清楚、更适合 VLA 执行的语义等价 instruction。
- `worse`: GPT 生成的更不友好但仍语义等价的 instruction。

## Modules

- `probe.libero_plus.export_base_tasks`
  - 在服务器 Docker 中读取 LIBERO-Plus benchmark。
  - 导出 base task JSONL，包含 `base_id`、`task_id`、原始 instruction、difficulty/category。

- `probe.instruction_rewrite.generate_condition_pairs`
  - 在 Windows 本地读取 base task JSONL。
  - 从 `.env` 读取 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`。
  - 调 GPT 生成 `better` 和 `worse`，并和 `original` 一起写入 conditions JSONL。

- `probe.rollout.collect_condition_calls`
  - 在服务器 Docker 中读取 conditions JSONL。
  - 对每个 base state 的 `original/better/worse` 分别跑 `N` 次 rollout。
  - 每次 replanning 采 `K=32` action chunks。
  - 每条 record 记录 `base_id`、`condition_id`、`condition_type`、`exec_seed`、`probe_seed`、`replan_idx`。

## Seed Discipline

- `exec_seed`: episode/rollout 级别。用于环境执行、trial、action chunk 选择随机性。
- `probe_seed`: replanning/probe 级别。用于标记每一次 K-sampling 的随机流。

一个 episode 只有一个 `exec_seed`，但会有很多 `probe_seed`，因为 episode 中会多次 replanning。

## Typical Flow

1. 服务器导出 base tasks。
2. Windows 本地调用 GPT 生成 conditions。
3. 上传 conditions 到服务器。
4. 服务器按 three conditions 采集 rollout records。
