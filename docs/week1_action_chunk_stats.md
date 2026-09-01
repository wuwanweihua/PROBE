# Week 1 Action Chunk Stats

`probe.analysis.action_chunk_stats` 用来回答：同一个 `obs + instruction` 下，pi0.5 采样出的 32 个 action chunk 是否真的有差异。

它会读取每条记录的 `action_samples.npz`，计算：

- `flat_pairwise_rms_mean`: 32 个 chunk 两两展平后的平均 RMS 距离。
- `first_step_rms_std`: 32 个 chunk 的第一步动作标准差。
- `last_step_rms_std`: 32 个 chunk 的最后一步动作标准差。
- `endpoint_xyz_rms_std`: chunk 末端 xyz 位置维度的标准差。
- `gripper_std_mean`: gripper 维度的平均标准差。

服务器示例：

```bash
cd /home/nvidia/yutao/lyh/PROBE/src/openpi

docker compose \
  -f examples/libero/compose.yml \
  -f /home/nvidia/yutao/lyh/PROBE/scripts/compose.gpu-device.override.yml \
  run --rm --no-deps runtime \
  /.venv/bin/python -m probe.analysis.action_chunk_stats \
  --dataset /data/week1/pi05_libero_final_index \
  --manifest-name records_merged.jsonl \
  --csv-out /data/week1/pi05_libero_final_index/action_chunk_stats.csv
```

如果 `flat_pairwise_rms_mean_mean` 明显大于 0，说明 32 个 chunk 不是完全一样。再看 `success` 和 `failure` 两组的均值差异，可以初步判断“chunk 分布差异”是否和成败有关。
