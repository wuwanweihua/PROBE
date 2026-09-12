# Week 3 后处理

`probe.data.build_week3_dataset` 将六个套件的 Week2 非执行采样数据合并成一个可训练数据集。它只读取已有的 `manifest.jsonl`、观测/动作/B 的 NPZ 和 8-retry 标签，不重新调用 VLA。

## 输入

每个 Week2 目录需要包含：

```text
manifest.jsonl
actions/*.npz
features/B/*.npz
observations/*.npz
```

使用 `--dataset SUITE=PATH` 重复指定套件，例如：

```bash
python -m probe.data.build_week3_dataset \
  --dataset camera=/data/week2_camera \
  --dataset language=/data/week2_language \
  --labels /data/boundary_tasks_8retry.jsonl \
  --output-dir /data/week3_dataset \
  --require-labels \
  --require-b
```

默认计算 K=`1,4,8,16,32` 五个前缀的 S。每个前缀仍是一条状态—条件记录，不能把 K 个动作样本展开成 K 条训练样本。

## 输出

```text
week3_dataset/
  manifest.jsonl
  labels.csv
  splits.json
  protocol.md
  summary.json
  summary.md
  features/
    B/*.npz
    B_matrix.npz
    S/K1/*.npz
    S/K4/*.npz
    S/K8/*.npz
    S/K16/*.npz
    S/K32/*.npz
    S_K1_matrix.npz
    ...
```

`manifest.jsonl` 保留源数据位置、真实 instruction、任务/条件、s/N 标签和输出特征引用。动作数组不复制，仍通过 `source_dataset_dir` 与 `source_action_samples_path` 回溯。

S 的 8 个特征为：

1. 整段平均分歧
2. 最大分歧
3. 前半段平均分歧
4. 后半段平均分歧
5. 候选动作片段成对 RMS 中位数
6. 成对 RMS 的 90% 分位数
7. 第一个时间位置的分歧
8. 最后一个时间位置的分歧

当前动作表示按原样计算，协议中写为 `identity`；如果以后需要尺度变换，必须只在训练 split 拟合。K=1 没有跨样本方差和成对距离，因此这些 S 特征写为 `NaN`，并在元数据中标记 `defined=false`。

## 划分

`splits.json` 按 `suite::base_id_or_group_id` 分组，保证同一任务的 original、better、worse 不跨 train/validation/test。固定随机种子默认是 `20260913`。

## 重要检查

进入训练前应确认：

- `summary.json` 的六个 suite 数量符合预期；
- `missing_labels=0`、`missing_b_features=0`；
- 每个任务的三个条件都存在；
- `splits.json` 中同一 group 只出现于一个 split；
- `action_shapes` 为预期的 `32x10x7`；
- B 与 S 的 record_id 一一对应。
