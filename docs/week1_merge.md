# Week 1 Batch Merge

采集多批数据后，不需要复制 `.npz` 大文件。运行 `probe.data.merge_week1` 生成统一的 `records_merged.jsonl`，每条记录会保留 `source_dataset_dir` 和 `source_record_id`。

服务器示例：

```bash
cd /home/nvidia/yutao/lyh/PROBE/src/openpi

/.venv/bin/python -m probe.data.merge_week1 \
  --output-dir /data/week1/pi05_libero_final_index \
  /data/week1/pi05_libero_smoke10_v2 \
  /data/week1/pi05_libero_probe100_perturb25 \
  /data/week1/pi05_libero_probe150_perturb45_generic \
  /data/week1/pi05_libero_probe200_perturb35_generic \
  /data/week1/pi05_libero_probe100_perturb45_generic_v2
```

然后验收合并后的索引：

```bash
/.venv/bin/python -m probe.data.validate_week1 \
  --dataset /data/week1/pi05_libero_final_index \
  --manifest-name records_merged.jsonl \
  --min-records 500 \
  --min-failure-fraction 0.2 \
  --expected-k 32
```
