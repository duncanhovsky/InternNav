# ArcDP v1.0.3 8x4090 full10 + P0 1ep 全量数据训练操作手册

本文面向“全量 InternData v0.5-full 直接放在 `/nvme` 里训练”的流程，不使用 sharp 数据，也不使用 cache rotation。目标训练方案：

- ArcDP full：10 epoch
- P0 消融：`rel`、`no_bridge`、`no_ordered_init`、`no_scale_cond`、`no_anchor_train`、`no_gcs` 各 1 epoch
- 推荐训练机器：8 卡 RTX 4090 24GB
- 备选训练机器：4 卡 RTX 4090 24GB

## 0. 固定路径和容量建议

统一使用以下路径，后续命令都按这个约定写：

```bash
HDD_ROOT=/hdd
NVME_ROOT=/nvme
PROJECT=/nvme/MyResearch/InternNav_v1.0.3
DATA=/nvme/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data
PRELOAD=/nvme/MyResearch/InternNav_v1.0.3/checkpoints/preload_index.json
DEPTH_CKPT=/nvme/MyResearch/InternNav_v1.0.3/checkpoints/depth_anything_v2_vits.pth
```

容量建议：

- `/hdd`：建议 8TB 起，最好 10TB；用于保存从原开发机复制来的 `/root/data`、全量压缩包和备份。
- `/nvme`：建议在西北二区 Ceph NVMe 申请 20TB 起；用于保存解压后的全量数据、代码、preload index、日志和 checkpoint。
- `/nvme` inode：建议 `files_total` 至少 100M，最好 120M 以上。低于 80M 风险较高。

已知数据规模：

```text
total_scenes   = 3725
total_episodes = 196536
total_bytes    = 5928075078118
num_shards     = 10
```

你之前实测西北二区 Ceph NVMe：

```text
993GB -> files_total=5486510，约 5.5M inode / 1TB
```

按这个配额线性估算，20TB 大约会给到 110M inode，比较适合承载解压后的 full 数据集。华北一区 Lustre 993GB 只有约 1.63M inode，20TB 也只有约 33M inode，不建议用来直接放解压后的 full 数据集。

## 1. 申请 CPU 开发机

先申请同区域的 CPU 开发机，用来做数据搬运、解压和 preload index 生成。建议放在最终训练所在区域，优先西北二区，避免跨区搬全量数据。

建议配置：

- CPU：普通多核即可，建议 16 核以上
- 内存：64GB 起
- 系统盘：够装基础环境即可
- 额外挂载：`/hdd` 和 `/nvme`
- 网络：能从原开发机 SSH/rsync 拉取 `/root/data`

登录 CPU 机器后先装常用工具：

```bash
apt-get update
apt-get install -y rsync tmux htop iotop lsof
```

如果平台镜像不是 Debian/Ubuntu，用对应包管理器安装 `rsync` 和 `tmux` 即可。

## 2. 挂载充足容量的 HDD 到 /hdd

在云平台控制台申请 HDD，并挂载到 CPU 开发机的 `/hdd`。不要随手格式化已有数据盘；如果平台已自动挂载，只需要检查。

```bash
mkdir -p /hdd

findmnt -T /hdd || true
df -hT /hdd
df -ih /hdd
stat -f -c 'mount=%m fs_type=%T files_total=%c files_free=%d blocks_total=%b blocks_free=%f block_size=%S' /hdd
```

判断标准：

- `df -hT /hdd` 能看到 `/hdd` 对应的容量和文件系统。
- `/hdd` 可用空间应大于原开发机 `/root/data` 的实际占用。
- 如果 `/hdd` 不是挂载点，先在平台控制台确认挂载路径；必要时用平台推荐方式挂载到 `/hdd`。

## 3. 拷贝原开发机 /root/data 到 /hdd

在新 CPU 开发机上执行。把 `<OLD_HOST>`、`<OLD_PORT>` 替换成原开发机 SSH 地址和端口。

```bash
export OLD_HOST=<OLD_HOST>
export OLD_PORT=22

rsync -aH --numeric-ids --info=progress2 \
  -e "ssh -p ${OLD_PORT}" \
  root@${OLD_HOST}:/root/data/ \
  /hdd/
```

如果就是在原开发机本机复制：

```bash
rsync -aH --numeric-ids --info=progress2 /root/data/ /hdd/
```

复制完成后检查：

```bash
du -sh /hdd

test -d /hdd/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data && echo hdd_dataset_ok
find /hdd/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data -type f -name '*.tar.gz' | wc -l

test -d /hdd/MyResearch/InternNav_v1.0.3 && echo hdd_project_ok
test -f /hdd/MyResearch/InternNav_v1.0.3/train_arcdp_full10_p0_1ep_8x4090_nvme.sh && echo arcdp_4090_script_ok
test -f /hdd/MyResearch/InternNav_v1.0.3/checkpoints/depth_anything_v2_vits.pth && echo depth_ckpt_ok
```

如果项目目录不是 `/hdd/MyResearch/InternNav_v1.0.3`，后续把 `SRC_PROJECT` 改成你的实际目录。如果没有看到 `arcdp_4090_script_ok`，说明原开发机里的代码还不是最新版本，需要先把新增的 4090 配置和启动脚本同步到 `/hdd` 里的项目，再继续后面的解压和训练准备。

## 4. 挂载充足容量 NVMe 到 /nvme，并检查 inode

在云平台申请 NVMe，并挂载到 CPU 开发机的 `/nvme`。建议先按 20TB 申请；如果平台只能先小容量申请再扩容，扩容后必须重新检查 inode。

```bash
mkdir -p /nvme

findmnt -T /nvme || true
df -hT /nvme
df -ih /nvme
stat -f -c 'mount=%m fs_type=%T files_total=%c files_free=%d blocks_total=%b blocks_free=%f block_size=%S' /nvme
```

快速判断 inode 是否够：

```bash
NEED_INODES=100000000
FREE_INODES=$(stat -f -c %d /nvme)
TOTAL_INODES=$(stat -f -c %c /nvme)

echo "total_inodes=${TOTAL_INODES}"
echo "free_inodes=${FREE_INODES}"

if [ "${FREE_INODES}" -ge "${NEED_INODES}" ]; then
  echo "inode_ok"
else
  echo "inode_low: recommend >= ${NEED_INODES}, better >= 120000000"
fi
```

建议：

- `files_total >= 100000000`：可以开始做 full 解压。
- `files_total >= 120000000`：更稳，给 checkpoint、日志和临时文件留余量。
- 如果 20TB 后 inode 仍明显低于 100M，说明该区域或文件系统 inode 配额不适合直接 full 解压训练，应换西北二区 Ceph 类 NVMe，或继续扩容。

## 5. 解压 /hdd 内压缩包数据集，并迁移代码到 /nvme

现有项目提供了准备脚本 `scripts/prepare_ssd_full.sh`，会完成：

1. 复制代码项目到 `/nvme`
2. 复制 `depth_anything_v2_vits.pth`
3. 从 `/hdd` 解压 full 数据到 `/nvme`
4. 生成 `/nvme/.../checkpoints/preload_index.json`

执行：

```bash
export SOURCE_ROOT=/hdd
export SSD_ROOT=/nvme
export SRC_PROJECT=/hdd/MyResearch/InternNav_v1.0.3
export DST_PROJECT=/nvme/MyResearch/InternNav_v1.0.3
export SRC_TRAJ=/hdd/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data
export DST_TRAJ=/nvme/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data
export EXTRACT_JOBS=8
export CONDA_ENV=base

cd "${SRC_PROJECT}"
bash scripts/prepare_ssd_full.sh
```

如果 CPU 机的 `base` 环境缺少生成 preload 所需依赖，可以先装最小依赖：

```bash
python -m pip install -U pip
python -m pip install jsonlines numpy tqdm
```

或者等第 7 步训练环境配好后，手动补生成 preload：

```bash
cd /nvme/MyResearch/InternNav_v1.0.3
PYTHONPATH=/nvme/MyResearch/InternNav_v1.0.3 \
python scripts/dataset/generate_preload_index.py \
  --root_dir /nvme/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data \
  --output /nvme/MyResearch/InternNav_v1.0.3/checkpoints/preload_index.json \
  --resume
```

准备完成后检查：

```bash
test -d /nvme/MyResearch/InternNav_v1.0.3 && echo project_ok
test -d /nvme/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data && echo data_ok
test -f /nvme/MyResearch/InternNav_v1.0.3/checkpoints/preload_index.json && echo preload_ok
test -f /nvme/MyResearch/InternNav_v1.0.3/checkpoints/depth_anything_v2_vits.pth && echo depth_ckpt_ok

du -sh /nvme/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data
df -hT /nvme
df -ih /nvme
```

如果解压中断，直接重新运行同一个 `bash scripts/prepare_ssd_full.sh`，脚本会通过 `.extracted_*.done` marker 跳过已完成的压缩包。

## 6. 租用 8 卡 RTX 4090 24GB 开发机

当 `/nvme` 数据准备完成后，再申请 8 卡 4090 机器，避免 GPU 空等数据解压。

要求：

- 与 `/nvme` 存储在同一区域，优先同可用区。
- 将已经准备好的 NVMe 挂载到 GPU 机器的 `/nvme`。
- 如果后续还要校验源压缩包，也把 HDD 挂载到 `/hdd`；训练本身不需要 `/hdd`。
- 确认 8 张 4090 都可见。

在 GPU 机器上检查：

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total --format=csv

df -hT /nvme
df -ih /nvme
test -d /nvme/MyResearch/InternNav_v1.0.3 && echo project_ok
test -d /nvme/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data && echo data_ok
```

如果只租到 4 卡 4090，也可以跑 4 卡脚本，训练会更慢，但 batch 等效保持 384。

## 7. 配置训练环境

进入 GPU 机器：

```bash
cd /nvme/MyResearch/InternNav_v1.0.3
```

创建环境：

```bash
conda create -n arcdp python=3.10 -y
conda activate arcdp

python -m pip install -U pip setuptools wheel
```

安装 PyTorch。下面以 CUDA 12.1 wheel 为例；如果平台镜像提供了匹配驱动和 CUDA 的 PyTorch，也可以直接使用平台预装版本。

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

安装项目依赖：

```bash
pip install -e ".[model]"
pip install jsonlines tensorboard
```

如果 `flash_attn` 构建失败，先确认 PyTorch 和 CUDA 可用：

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("gpu_count", torch.cuda.device_count())
print("gpu0", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

再按平台 CUDA/PyTorch 版本安装匹配的 `flash_attn`。常见做法：

```bash
MAX_JOBS=4 pip install flash_attn==2.7.4.post1 --no-build-isolation
```

安装后做 import 检查：

```bash
python - <<'PY'
import torch
import transformers
import diffusers
import internnav
print("env_ok")
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
PY
```

## 8. 检查数据、环境和训练前提条件

设置变量：

```bash
export PROJECT=/nvme/MyResearch/InternNav_v1.0.3
export DATA=/nvme/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data
export PRELOAD=${PROJECT}/checkpoints/preload_index.json
export DEPTH_CKPT=${PROJECT}/checkpoints/depth_anything_v2_vits.pth
```

基础检查：

```bash
cd "${PROJECT}"

test -d "${DATA}" && echo data_ok
test -f "${PRELOAD}" && echo preload_ok
test -f "${DEPTH_CKPT}" && echo depth_ckpt_ok

df -hT /nvme
df -ih /nvme
stat -f -c 'mount=%m fs_type=%T files_total=%c files_free=%d blocks_total=%b blocks_free=%f block_size=%S' /nvme
```

检查 preload 条目数量和路径可读性：

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("/nvme/MyResearch/InternNav_v1.0.3/checkpoints/preload_index.json")
d = json.loads(p.read_text(encoding="utf-8"))
keys = ["trajectory_data_dir", "trajectory_rgb_path", "trajectory_depth_path", "trajectory_afford_path"]
for k in keys:
    print(k, len(d.get(k, [])))

lengths = [len(d.get(k, [])) for k in keys]
assert len(set(lengths)) == 1, lengths
assert lengths[0] == 196536, f"expected 196536 episodes, got {lengths[0]}"

for k in ["trajectory_data_dir", "trajectory_afford_path"]:
    sample = d[k][0]
    assert Path(sample).exists(), f"missing sample path: {sample}"
print("preload_shape_ok")
PY
```

如果 `/hdd` 也挂载在 GPU 机上，可以做更完整的数据完整性检查：

```bash
cd "${PROJECT}"
python scripts/dataset/check_extracted_dataset_integrity.py \
  --src-traj /hdd/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data \
  --dst-traj "${DATA}" \
  --preload-index "${PRELOAD}"
```

严格模式会遍历 tar 内每个文件，耗时很长，只在怀疑解压不完整时使用：

```bash
python scripts/dataset/check_extracted_dataset_integrity.py \
  --src-traj /hdd/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data \
  --dst-traj "${DATA}" \
  --preload-index "${PRELOAD}" \
  --strict-tar
```

检查训练脚本权限：

```bash
chmod +x train_arcdp_full10_p0_1ep_8x4090_nvme.sh
chmod +x train_arcdp_full10_p0_1ep_4x4090_nvme.sh
chmod +x scripts/train/arcdp_4090/*.sh
```

## 9. 开始训练

建议用 `tmux` 保持会话：

```bash
tmux new -s arcdp4090
```

启动 8 卡 4090 训练：

```bash
cd /nvme/MyResearch/InternNav_v1.0.3
conda activate arcdp
mkdir -p logs

./train_arcdp_full10_p0_1ep_8x4090_nvme.sh 2>&1 | tee logs/arcdp_full10_p0_1ep_8x4090_$(date +%F_%H%M%S).log
```

如果是 4 卡 4090：

```bash
cd /nvme/MyResearch/InternNav_v1.0.3
conda activate arcdp
mkdir -p logs

./train_arcdp_full10_p0_1ep_4x4090_nvme.sh 2>&1 | tee logs/arcdp_full10_p0_1ep_4x4090_$(date +%F_%H%M%S).log
```

8 卡脚本默认参数：

```text
BRIDGEDP_NUM_GPUS=8
BRIDGEDP_BATCH_SIZE=16
BRIDGEDP_GRAD_ACCUM=3
effective_global_batch=384
```

4 卡脚本默认参数：

```text
BRIDGEDP_NUM_GPUS=4
BRIDGEDP_BATCH_SIZE=16
BRIDGEDP_GRAD_ACCUM=6
effective_global_batch=384
```

脚本会按顺序运行：

```text
arcdp_full10_8x4090_nvme / arcdp_full10_4x4090_nvme
arcdp_p0_rel_1ep_*
arcdp_p0_no_bridge_1ep_*
arcdp_p0_no_ordered_init_1ep_*
arcdp_p0_no_scale_cond_1ep_*
arcdp_p0_no_anchor_train_1ep_*
arcdp_p0_no_gcs_1ep_*
```

输出位置：

```bash
# full
/nvme/MyResearch/InternNav_v1.0.3/checkpoints/arcdp_full10_8x4090_nvme/ckpts

# P0
/nvme/MyResearch/InternNav_v1.0.3/checkpoints/arcdp_p0_4090_nvme/<run_name>/ckpts
```

## 10. 训练中监控和中断恢复

监控 GPU：

```bash
watch -n 5 nvidia-smi
```

监控磁盘和 inode：

```bash
watch -n 60 'df -hT /nvme; echo; df -ih /nvme'
```

查看日志：

```bash
tail -f logs/arcdp_full10_p0_1ep_8x4090_*.log
```

训练脚本默认：

```bash
BRIDGEDP_AUTO_RESUME=1
```

也就是同名 run 目录下已有 checkpoint 时会自动尝试续训。如果中断发生在 suite 中间，为避免重复跑已完成 stage，建议复制一份启动脚本并临时注释掉已完成的 `run_stage` 行，再重新运行。不要删除已有 checkpoint。

常见失败点：

- `Missing full dataset root`：`/nvme/datasets/.../traj_data` 路径不对，或 NVMe 没挂到 GPU 机。
- `Missing preload index`：未生成 `checkpoints/preload_index.json`，回到第 5 步补生成。
- `Missing DepthAnything encoder`：缺少 `checkpoints/depth_anything_v2_vits.pth`，从 `/hdd` 或原开发机复制到项目 checkpoint 目录。
- CUDA OOM：先保持单卡 batch 16；如果仍 OOM，可临时设置 `BRIDGEDP_BATCH_SIZE=12`，并相应调大 `BRIDGEDP_GRAD_ACCUM` 保持全局 batch 接近 384。
- inode 不足：不能靠清理少量日志解决，通常需要扩容 `/nvme` 或换 inode 配额更高的文件系统。

## 11. 最终检查清单

开始训练前至少确认这些项全部通过：

```bash
nvidia-smi
df -hT /nvme
df -ih /nvme
test -d /nvme/MyResearch/InternNav_v1.0.3
test -d /nvme/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data
test -f /nvme/MyResearch/InternNav_v1.0.3/checkpoints/preload_index.json
test -f /nvme/MyResearch/InternNav_v1.0.3/checkpoints/depth_anything_v2_vits.pth
```

preload 数量应为：

```text
trajectory_data_dir   196536
trajectory_rgb_path   196536
trajectory_depth_path 196536
trajectory_afford_path 196536
```

确认无误后再开始 8 卡训练。
