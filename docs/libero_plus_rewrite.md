# LIBERO-Plus Instruction Rewrite Workflow

这个中间任务分三段：

1. 从 LIBERO-Plus baseline 的 `records.jsonl` 里导出失败 episode。
2. 调用 GPT-5.5 对失败任务的 instruction 做语义等价改写。
3. 在服务器 Docker 里用改写后的 instruction 复跑失败任务，看是否从失败变成功。

## API Key

本地或服务器项目根目录放 `.env`：

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5.5
```

`.env` 已经在 `.gitignore` 中，不能提交到 GitHub。

## Data Flow

- baseline 输入：`/data/libero_plus_baseline/libero10_hard50_k1/records.jsonl`
- 失败 episode 清单：`/data/libero_plus_baseline/libero10_hard50_failed_episodes.jsonl`
- GPT 改写清单：`/data/libero_plus_rewrites/libero10_hard50_gpt55_rewrites.jsonl`
- 改写复测结果：`/data/libero_plus_rewrite_eval/libero10_hard50_gpt55_results.jsonl`

## What Each Module Does

- `probe.libero_plus.export_failed_episodes`
  - 读取 baseline 记录。
  - 按 `episode_id` 聚合每个 episode 的多次 replanning 记录。
  - 输出失败 episode，包括 `task_id`、原 instruction、失败步数、replanning 次数、difficulty/category 信息。

- `probe.instruction_rewrite.rewrite_failed_tasks`
  - 读取失败 episode。
  - 按 `task_id + instruction` 去重。
  - 调 GPT-5.5 生成多条语义等价 instruction。
  - 写成 JSONL，后续复测直接读取。

- `probe.instruction_rewrite.eval_rewrites_libero_plus`
  - 读取 GPT 改写清单。
  - 对每个改写 instruction 启动同一个 LIBERO-Plus task。
  - 每次 replanning 调 pi0.5 得到 action chunk 并执行。
  - 输出每条改写是否 rescue 成功。
