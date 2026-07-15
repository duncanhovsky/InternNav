# ArcDP v1.0.3：CPU 打包环境、NVMe 解压全量数据、8x4090 训练操作说明

本文从当前状态继续：`/root/data` 已经在西北二区同步完成，里面包含代码、checkpoint 和压缩态 InternData-N1 v0.5-full 数据集。后续目标是：

- CPU 开发机和 8 卡 4090 开发机均使用同一个预置镜像：`pytorch:2.7.0-cuda12.6-python3.10-ubuntu22.04`
- 先在 CPU 开发机配置好 conda 训练环境，并打包保存到 `/root/data/conda_env_packs`
- 后续租用 8 卡 4090 开发机后，用脚本一键解包训练环境
- 租用 NVMe 后挂载到 `/nvme`，把 `/root/data` 里的压缩数据解压到 `/nvme`
- 生成 preload index，训练时直接读 `/nvme` 中的解压数据
- 不使用 sharp，不使用 cache rotation

## 0. 路径约定

```bash
SOURCE_ROOT=/root/data
NVME_ROOT=/nvme
ENV_NAME=arcdp

SRC_PROJECT=/root/data/MyResearch/InternNav
DST_PROJECT=/nvme/MyResearch/InternNav

SRC_TRAJ=/root/data/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data
DST_TRAJ=/nvme/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data

ENV_PACK_DIR=/root/data/conda_env_packs
```

如果你的代码目录不是 `/root/data/MyResearch/InternNav_v1.0.3`，执行脚本时用 `SRC_PROJECT=/实际路径` 覆盖。

## 1. CPU 开发机上检查 /root/data

```bash
df -hT /root/data
df -ih /root/data

test -d /root/data/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data && echo dataset_archive_root_ok
test -d /root/data/MyResearch/InternNav && echo project_ok
test -f /root/data/MyResearch/InternNav/checkpoints/depth_anything_v2_vits.pth && echo depth_ckpt_ok
test -f /root/data/MyResearch/InternNav/scripts/setup_internnav_pytorch270_cu126.sh && echo setup_script_ok
```

建议先装基础工具：

```bash
apt-get update
apt-get install -y rsync tmux htop iotop lsof
```

## 2. 在 CPU 开发机配置并打包 conda 环境

进入项目：

```bash
cd /root/data/MyResearch/InternNav
```

一键配置 `arcdp` 环境并打包：

```bash
chmod +x scripts/arcdp_pack_training_env_cpu.sh

ENV_NAME=arcdp \
OUTPUT_DIR=/root/data/conda_env_packs \
INSTALL_APT=1 \
INTERNNAV_INSTALL_GIT_DEPS=skip \
INTERNNAV_INSTALL_FLASH_ATTN=skip \
bash scripts/arcdp_pack_training_env_cpu.sh
```

> 备注：`conda-pack` 不能打包 `pip install -e` 的 editable 包。脚本会在打包前自动从 `arcdp` 环境里卸载 editable 的 `internnav` 注册，但不会删除项目源码；后续 GPU 训练仍然通过 `/nvme/MyResearch/InternNav` 和 `PYTHONPATH` 导入源码。

说明：

- 脚本会调用 `scripts/setup_internnav_pytorch270_cu126.sh`
- 默认要求 PyTorch `2.7.0`、CUDA wheel `12.6`、Python `3.10`
- CPU 机没有 GPU 时，`torch.cuda.is_available()` 可以是 `False`，但 `torch.version.cuda` 应为 `12.6`
- CPU 机访问 GitHub 不稳定时，保持 `INTERNNAV_INSTALL_GIT_DEPS=skip`。这会跳过 `depth-camera-filtering` 和 `diffusion_policy` 两个 GitHub 源码依赖；当前 ArcDP/Bridge-DP full 训练不依赖它们，Habitat/RDP/InternVLA 路径需要时再补装。
- `flash_attn` 在 CPU 机上会触发很重的源码编译，可能导致会话被平台杀掉；ArcDP/Bridge-DP 训练通常不依赖它，所以 CPU 打包环境默认 `INTERNNAV_INSTALL_FLASH_ATTN=skip`
- 环境包会保存到 `/root/data/conda_env_packs`

检查产物：

```bash
ls -lh /root/data/conda_env_packs
ls -lh /root/data/conda_env_packs/arcdp_*.tar.gz*
test -f /root/data/conda_env_packs/arcdp_deploy_training_env_gpu.sh && echo deploy_script_ok
```

如果后续修改了依赖或代码环境，重新运行同一条打包命令即可生成新的包。

## 3. 租用并挂载 NVMe 到 /nvme

申请足够容量的 NVMe，并挂载到 CPU 开发机 `/nvme`。建议按 full 解压数据集准备，目标 inode：

```text
files_total >= 100M
最好 >= 120M
```

检查：

```bash
mkdir -p /nvme
findmnt -T /nvme || true
df -hT /nvme
df -ih /nvme
stat -f -c 'mount=%m fs_type=%T files_total=%c files_free=%d blocks_total=%b blocks_free=%f block_size=%S' /nvme
```

如果 `files_free` 明显低于 `100000000`，不建议直接开始 full 解压；可以等扩容完成后再执行下一步。

## 4. 解压 full 数据集到 /nvme 并生成 preload

执行：

```bash
cd /root/data/MyResearch/InternNav
chmod +x scripts/arcdp_prepare_full_nvme_from_root_data.sh

ENV_NAME=arcdp \
SOURCE_ROOT=/root/data \
NVME_ROOT=/nvme \
EXTRACT_JOBS=8 \
bash scripts/arcdp_prepare_full_nvme_from_root_data.sh
```

这个脚本会做四件事：

1. 检查 `/nvme` 挂载、容量和 inode
2. 将项目从 `/root/data/MyResearch/InternNav_v1.0.3` 同步到 `/nvme/MyResearch/InternNav_v1.0.3`
3. 将 `/root/data/datasets/.../traj_data` 下的 `.tar.gz` 解压到 `/nvme/datasets/.../traj_data`
4. 使用 `arcdp` conda 环境生成 `/nvme/MyResearch/InternNav_v1.0.3/checkpoints/preload_index.json`

如果中途失败或机器断开，直接重新运行同一条命令。已解压完成的压缩包会通过 `.extracted_*.done` marker 跳过，preload 生成也支持 `--resume`。

## 5. 检查 /nvme 数据和 preload

```bash
export PROJECT=/nvme/MyResearch/InternNav_v1.0.3
export DATA=/nvme/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data
export PRELOAD=${PROJECT}/checkpoints/preload_index.json
export DEPTH_CKPT=${PROJECT}/checkpoints/depth_anything_v2_vits.pth

test -d "${PROJECT}" && echo project_ok
test -d "${DATA}" && echo data_ok
test -f "${PRELOAD}" && echo preload_ok
test -f "${DEPTH_CKPT}" && echo depth_ckpt_ok

df -hT /nvme
df -ih /nvme
du -sh "${DATA}"
```

检查 preload episode 数量：

```bash
conda run -n arcdp python - <<'PY'
import json
from pathlib import Path

p = Path("/nvme/MyResearch/InternNav_v1.0.3/checkpoints/preload_index.json")
d = json.loads(p.read_text(encoding="utf-8"))
keys = ["trajectory_data_dir", "trajectory_rgb_path", "trajectory_depth_path", "trajectory_afford_path"]
for k in keys:
    print(k, len(d.get(k, [])))
lengths = [len(d.get(k, [])) for k in keys]
assert len(set(lengths)) == 1, lengths
assert lengths[0] == 196536, f"expected 196536, got {lengths[0]}"
print("preload_ok")
PY
```

## 6. 租用 8 卡 4090 开发机

等 `/nvme` 数据准备完成后，再租用 8 卡 4090 机器，仍选择预置镜像：

```text
pytorch:2.7.0-cuda12.6-python3.10-ubuntu22.04
```

挂载要求：

- 将已经准备好的 NVMe 挂载到 GPU 机器 `/nvme`
- 将保存环境包的 HDD 挂载到 GPU 机器 `/root/data`

检查：

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total --format=csv

df -hT /nvme /root/data
df -ih /nvme /root/data
test -d /nvme/MyResearch/InternNav_v1.0.3 && echo nvme_project_ok
test -f /root/data/conda_env_packs/arcdp_deploy_training_env_gpu.sh && echo env_deploy_script_ok
```

## 7. 在 8 卡 4090 机器上一键部署 conda 环境

```bash
cd /nvme/MyResearch/InternNav_v1.0.3
chmod +x scripts/arcdp_deploy_training_env_gpu.sh
chmod +x /root/data/conda_env_packs/arcdp_deploy_training_env_gpu.sh

bash /root/data/conda_env_packs/arcdp_deploy_training_env_gpu.sh \
  "$(ls -t /root/data/conda_env_packs/arcdp_*.tar.gz | head -n 1)"
```

如果目标环境已经存在，脚本会拒绝覆盖。确认要替换时：

```bash
OVERWRITE=1 bash /root/data/conda_env_packs/arcdp_deploy_training_env_gpu.sh \
  "$(ls -t /root/data/conda_env_packs/arcdp_*.tar.gz | head -n 1)"
```

打开新 shell 后：

```bash
conda activate arcdp
cd /nvme/MyResearch/InternNav_v1.0.3
export PYTHONPATH=/nvme/MyResearch/InternNav_v1.0.3:${PYTHONPATH:-}
```

验证：

```bash
python - <<'PY'
import torch
from internnav.model import get_policy
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
print("policy", get_policy("BridgeDP_Policy").__name__)
PY
```

## 8. 开始训练

训练直接读取 `/nvme` 中的解压全量数据：

```bash
cd /nvme/MyResearch/InternNav_v1.0.3
conda activate arcdp

chmod +x train_arcdp_full10_p0_1ep_8x4090_nvme.sh
chmod +x scripts/train/arcdp_4090/*.sh

mkdir -p logs
tmux new -s arcdp4090
```

在 tmux 中启动：

```bash
./train_arcdp_full10_p0_1ep_8x4090_nvme.sh \
  2>&1 | tee logs/arcdp_full10_p0_1ep_8x4090_$(date +%F_%H%M%S).log
```

脚本默认：

```text
BRIDGEDP_NUM_GPUS=8
BRIDGEDP_BATCH_SIZE=16
BRIDGEDP_GRAD_ACCUM=3
effective_global_batch=384
BRIDGEDP_DATASET_ROOT=/nvme/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data
BRIDGEDP_PRELOAD_INDEX=/nvme/MyResearch/InternNav_v1.0.3/checkpoints/preload_index.json
```

训练顺序：

```text
full 10ep
P0 rel 1ep
P0 no_bridge 1ep
P0 no_ordered_init 1ep
P0 no_scale_cond 1ep
P0 no_anchor_train 1ep
P0 no_gcs 1ep
```

## 9. 监控和恢复

监控 GPU：

```bash
watch -n 5 nvidia-smi
```

监控磁盘：

```bash
watch -n 60 'df -hT /nvme /root/data; echo; df -ih /nvme /root/data'
```

查看日志：

```bash
tail -f logs/arcdp_full10_p0_1ep_8x4090_*.log
```

训练脚本默认 `BRIDGEDP_AUTO_RESUME=1`。同一个 run 中断后，重新运行脚本会尝试从已有 checkpoint 续训。如果整套流程已经完成了前面的 stage，建议复制一份训练脚本，只保留未完成的 `run_stage` 行再继续。

## 10. 最小命令清单

CPU 机：

```bash
cd /root/data/MyResearch/InternNav_v1.0.3
bash scripts/arcdp_pack_training_env_cpu.sh

# 挂载 /nvme 后
ENV_NAME=arcdp SOURCE_ROOT=/root/data NVME_ROOT=/nvme EXTRACT_JOBS=8 \
bash scripts/arcdp_prepare_full_nvme_from_root_data.sh
```

8 卡 4090 机：

```bash
cd /nvme/MyResearch/InternNav_v1.0.3
bash /root/data/conda_env_packs/arcdp_deploy_training_env_gpu.sh \
  "$(ls -t /root/data/conda_env_packs/arcdp_*.tar.gz | head -n 1)"

conda activate arcdp
./train_arcdp_full10_p0_1ep_8x4090_nvme.sh
```
