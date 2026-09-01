# PROBE

本仓库当前实现第一周的数据采集骨架：在远程 Linux 服务器上复用 `openpi + pi0.5 + LIBERO`，采集 `obs + instruction -> K=32 action chunk -> success/failure` 数据。

## Week 1 Quickstart

服务器路径约定：

```bash
export PROBE_ROOT=/home/nvidia/yutao/lyh/PROBE
```

确认你已经在服务器完成：

```text
$PROBE_ROOT/src/openpi
openpi_server:latest
libero:latest
$PROBE_ROOT/cache/openpi/openpi-assets/checkpoints/pi05_libero
```

把本地代码同步到服务器后运行：

```bash
cd /home/nvidia/yutao/lyh/PROBE
bash scripts/run_week1_pi05_libero.sh
```

数据默认写到容器内 `/data/week1/pi05_libero`，对应 openpi compose 挂载的宿主机数据目录。

更多模块说明见 `docs/week1_modules.md`。
