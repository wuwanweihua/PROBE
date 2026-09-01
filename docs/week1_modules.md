# PROBE Week 1 Modules

本周目标是跑通 `openpi + pi0.5 + LIBERO` 后，采集至少 500 条带标签调用数据。每条数据对应一次 `obs + instruction` 下的模型调用，包含 `K=32` 个 action chunk 和最终 `success/failure` 标签。

## 模块职责

`configs/week1_pi05_libero.yaml`

保存第一周实验配置：服务器路径、端口、GPU、checkpoint、LIBERO task suite、`K=32`、目标记录数、输出目录和验收阈值。

`probe/policies/pi05_client.py`

封装 openpi 的 websocket policy client。输入已经预处理好的 policy element，循环调用 pi0.5，得到 `K` 个 action chunk。

`probe/envs/libero_runner.py`

封装 LIBERO 环境相关逻辑：创建 task suite、创建仿真环境、预处理图像和 robot state、给出各 task suite 的默认最大步数。

`probe/rollout/collect_calls.py`

第一周主采集程序。它遍历 LIBERO 任务和 episode，在每次需要重新规划时采样 `K=32` 个 action chunk，按策略选一个执行，episode 结束后把该 episode 内所有调用写成带标签记录。

`probe/data/record_schema.py`

定义每条数据的字段，例如 `record_id`、`instruction`、`task_id`、`step_idx`、`action_samples_path`、`obs_path`、`final_success` 等。

`probe/data/writer.py`

负责落盘：元信息写入 `records.jsonl`，观测和 action chunk 写入压缩 `.npz` 文件。manifest 中保存相对路径，方便移动整个数据集目录。

`probe/data/validate_week1.py`

验收脚本。检查记录数、成功/失败类别、失败比例、action 文件是否存在、`K` 是否等于 32、selected action index 是否合法。

`probe/perturbations.py`

扰动工具。如果初始采集失败样本不足 20%，后续可以提高 `perturbation_rate`，通过改写 instruction 制造更难样本。

`scripts/run_week1_pi05_libero.sh`

服务器一键运行脚本：同步本地 `probe/` 和 `configs/` 到 openpi 目录，启动 policy server，运行采集程序，并执行数据验收。

`scripts/compose.gpu-device.override.yml`

Docker Compose 覆盖文件，用 `GPU_ID` 指定物理 GPU，避免误占其他卡。

## 推荐流程

1. Windows 本地写代码并同步到服务器 `/home/nvidia/yutao/lyh/PROBE`。
2. 服务器确认 `openpi_server:latest` 和 `libero:latest` 两个镜像都已构建。
3. 运行 `bash scripts/run_week1_pi05_libero.sh`。
4. 如果成功率太高导致失败样本不足，提高 `perturbation_rate` 后重跑一批。
