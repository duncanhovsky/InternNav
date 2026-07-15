# ArcDP 4 卡 A800 cache-rotation 训练流程

本文档记录当前推荐流程：先在 CPU/存储开发机上完成环境、数据、manifest 和 cache 准备，最后再租 4 卡 A800 训练。目标训练量为：

- ArcDP full：100 epoch 等量训练
- P0 消融：`rel`、`no_bridge`、`no_ordered_init`、`no_scale_cond`、`no_anchor_train`、`no_gcs` 各 10 epoch 等量训练
- full 100 epoch 训练中保留 20 个均匀 checkpoint，其中第 2 个约等价于 full 10 epoch

## 0. 当前结论

当前 cache 拷贝速度已明显恢复到可用范围。你在 2026-07-05 12:02:49 到 12:13:20 的采样为：

```text
cache_A.building: 602M -> 19G
/nvme used:       25G  -> 42G
inode used:       23K  -> 263K
elapsed:          10m31s
```

粗略速度：

- 数据量：约 95-105 GB/h
- inode：约 1.3-1.4M inode/h
- 单个 600GB cache slot 预计约 5.5-7 小时
- `cache_A + cache_B` 预计约 11-14 小时

如果速度长期掉到 30GB/h 以下，或者 5 分钟只增加 1GB 左右，就说明共享存储小文件/元数据性能又出现瓶颈，需要停下排查。

## 1. 关键路径和固定约定

当前推荐继续使用 4 卡、`1.5tb balanced` profile，即使实际 `/nvme` 已扩容到 8TB。这里的 `1.5tb` 不是实际盘大小，而是 cache profile 名称：

- cache slot 大小：600GB
- 双槽 `cache_A + cache_B` 目标占用：约 1.2TB
- manifest：10 个 shards
- 低 NVMe 模式：启用，切换 shard 时会先释放旧 slot
- cache build workers：4

路径约定：

```bash
PROJECT=/ssd/MyResearch/InternNav
NVME_PROJECT=/nvme/MyResearch/InternNav
HDD_ROOT=/ssd
NVME_ROOT=/nvme
DATA=/ssd/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data
MANIFEST=/ssd/bridgedp_rotation/manifests/shards_4gpu_1.5tb_balanced.json
CACHE_ROOT=/nvme/bridgedp_cache
```

注意：cache 准备是从 `/ssd` 已解压数据拷贝到 `/nvme`，不会从 `/root/data` 重新解压。

## 2. 训练量换算

cache-rotation 的等量 epoch 使用下式换算为训练步数：

```text
total_max_steps = ceil(50 * total_episodes * epochs / (gpus * per_gpu_batch * grad_accum))
```

当前数据集：

```text
total_episodes = 196536
gpus = 4
per_gpu_batch = 96
grad_accum = 1
global_batch = 384
```

因此 4 卡 A800 下：

| 等量 epoch | total_max_steps |
|---:|---:|
| 10 | 255,907 |
| 100 | 2,559,063 |
| 200 | 5,118,125 |
| 500 | 12,795,313 |
| 1000 | 25,590,625 |

full 100 epoch 训练会保存 20 个均匀 checkpoint，目标步数近似为 `total_max_steps * k / 20`。其中第 2 个 checkpoint 约为 `255,907` step，等价于 full 10 epoch。

## 3. CPU/存储开发机准备

进入项目：

```bash
cd /ssd/MyResearch/InternNav
```

确认数据和索引：

```bash
test -d /ssd/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data && echo data_ok
test -f /ssd/MyResearch/InternNav/checkpoints/preload_index.json && echo preload_ok
df -h /ssd /nvme
df -ih /ssd /nvme
```

正常应看到：

- `data_ok`
- `preload_ok`
- `/nvme` inode 总量约 22M，剩余 inode 远大于 17.3M
- `/ssd` 和 `/nvme` 都有足够空间

确认 manifest：

```bash
python - <<'PY'
import json
p="/ssd/bridgedp_rotation/manifests/shards_4gpu_1.5tb_balanced.json"
m=json.load(open(p, encoding="utf-8"))
print("manifest_ok")
print(m["summary"])
print("num_shards", len(m["shards"]))
PY
```

正常应看到：

```text
manifest_ok
{'total_scenes': 3725, 'total_episodes': 196536, 'total_bytes': 5928075078118, 'num_shards': 10}
num_shards 10
```

如果 manifest 不存在，再生成：

```bash
mkdir -p /ssd/bridgedp_rotation/manifests /ssd/bridgedp_rotation/logs

PYTHONPATH=/ssd/MyResearch/InternNav python scripts/cache_rotation/make_bridgedp_shards.py \
  --hdd-root /ssd \
  --manifest /ssd/bridgedp_rotation/manifests/shards_4gpu_1.5tb_balanced.json \
  --gpus 4 \
  --nvme-size 1.5tb \
  --preset balanced \
  --cache-slot-gb 600
```

已生成过 manifest 时不要重复做这一步。

## 4. 准备 cache_A/cache_B

先确认没有旧的构建进程：

```bash
pgrep -af "prepare_bridgedp|build_bridgedp_cache_shard|make_bridgedp_shards" || echo no_cache_process
```

如果还存在旧版单线程构建进程，先停止。确认没有进程后，如果旧的 `.building` 是旧版脚本产生的，可以删除后重跑：

```bash
rm -rf /nvme/bridgedp_cache/cache_A.building
rm -rf /nvme/bridgedp_cache/cache_B.building
```

启动准备：

```bash
mkdir -p /ssd/bridgedp_rotation/logs

bash scripts/cache_rotation/prepare_bridgedp_cache_rotation.sh \
  --gpus 4 \
  --nvme-size 1.5tb \
  --preset balanced \
  --hdd-root /ssd \
  --nvme-root /nvme \
  --no-swanlab \
  > /ssd/bridgedp_rotation/logs/prepare_cache_4gpu_1p5tb.log 2>&1 &

echo $! > /ssd/bridgedp_rotation/logs/prepare_cache_4gpu_1p5tb.pid
```

正常日志应出现：

```text
build workers:  4
[5/5] Prebuild cache_A and cache_B
[cache-build] shard=0 scenes=... workers=4 resume_scene_copy=1
[cache-build] copied scene: ...
```

如果日志里没有 `workers=4`，说明 `/ssd` 或 `/nvme` 的项目副本仍是旧脚本，需要先同步新版脚本。

监控命令：

```bash
tail -f /ssd/bridgedp_rotation/logs/prepare_cache_4gpu_1p5tb.log
```

另开窗口看速度：

```bash
for i in 1 2 3 4 5 6; do
  date
  df -h /nvme
  df -ih /nvme
  du -sh /nvme/bridgedp_cache/cache_A.building 2>/dev/null || true
  du -sh /nvme/bridgedp_cache/cache_A 2>/dev/null || true
  find /nvme/bridgedp_cache/cache_A.building/.scene_done -name '*.json' 2>/dev/null | wc -l
  sleep 300
done
```

完成判断：

```bash
test -f /nvme/bridgedp_cache/cache_A/.READY && echo cache_A_ready
test -f /nvme/bridgedp_cache/cache_B/.READY && echo cache_B_ready

python scripts/cache_rotation/check_bridgedp_cache.py --slot-root /nvme/bridgedp_cache/cache_A
python scripts/cache_rotation/check_bridgedp_cache.py --slot-root /nvme/bridgedp_cache/cache_B
```

正常应看到两个 slot 都 READY，且 check 脚本通过。

## 5. 打包 base conda 环境

CPU/存储开发机即使没有 GPU，也可以打包 CUDA 版 PyTorch 环境；关键是 `torch.version.cuda` 应为 `12.6`，不要求 `torch.cuda.is_available()` 为 True。

先安装/确认环境：

```bash
cd /ssd/MyResearch/InternNav
bash scripts/setup_internnav_pytorch270_cu126.sh
```

检查：

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
import swanlab
print("swanlab", getattr(swanlab, "__version__", "unknown"))
PY
```

正常：

- `torch_cuda` 为 `12.6`
- `swanlab` 能 import
- CPU 机上 `cuda_available` 可以是 False

打包：

```bash
OUTPUT_DIR=/ssd/bridgedp_rotation/conda_env_packs \
PACK_METHOD=auto \
REQUIRE_CUDA_TORCH=1 \
EXPECTED_TORCH_CUDA=12.6 \
bash scripts/pack_base_conda_env.sh
```

正常输出包含：

```text
[pack-base] done
archive:  /ssd/bridgedp_rotation/conda_env_packs/internnav_base_*.tar.gz
metadata: ...metadata.tar.gz
sha256:   ...sha256
```

预计耗时通常为 10-40 分钟，取决于环境大小和存储速度。

## 6. 租 4 卡 A800 后部署

进入 A800 机器后，先确认它能看到同一份 `/ssd` 和 `/nvme`。如果平台不是共享盘，需要把项目、cache 和 conda pack 复制过去。

基础检查：

```bash
df -h /ssd /nvme
df -ih /ssd /nvme
test -d /ssd/MyResearch/InternNav && echo project_ok
test -f /ssd/bridgedp_rotation/manifests/shards_4gpu_1.5tb_balanced.json && echo manifest_ok
test -f /nvme/bridgedp_cache/cache_A/.READY && echo cache_A_ready
test -f /nvme/bridgedp_cache/cache_B/.READY && echo cache_B_ready
nvidia-smi
```

部署 conda pack：

```bash
cd /ssd/MyResearch/InternNav

YES=1 \
EXPECTED_TORCH_CUDA=12.6 \
bash scripts/deploy_base_conda_env.sh /ssd/bridgedp_rotation/conda_env_packs/internnav_base_*.tar.gz
```

部署后开新 shell：

```bash
conda activate base
python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
PY
```

正常应看到：

- `torch_cuda` 为 `12.6`
- `cuda_available` 为 True
- `gpu_count` 为 4

## 7. SwanLab 配置

训练脚本默认启用 SwanLab。推荐用环境变量方式，适合非交互训练：

```bash
export SWANLAB_API_KEY=<your_api_key>
export SWANLAB_PROJECT=ArcDP-100ep
```

也可以交互式登录一次：

```bash
swanlab login <your_api_key>
```

训练启动时日志应显示：

```text
[SwanLab] SwanLab enabled for this training run.
[SwanLab] SwanLab project: ArcDP-100ep
[SwanLab] SwanLab experiment: ...
[SwanLab] report_to=swanlab
```

如果没有设置 key，脚本会告警。已经在机器上 `swanlab login` 过也可以继续训练；但集群/新机器更推荐显式设置 `SWANLAB_API_KEY`。

## 8. 训练前体检

先跑 full 100 epoch 的体检：

```bash
cd /ssd/MyResearch/InternNav

bash scripts/train/arcdp_cache_rotation/check_arcdp_training_ready.sh \
  --gpus 4 \
  --variant full \
  --epochs 100 \
  --nvme-size 1.5tb \
  --preset balanced \
  --hdd-root /ssd \
  --nvme-root /nvme
```

再抽查一个消融 10 epoch：

```bash
bash scripts/train/arcdp_cache_rotation/check_arcdp_training_ready.sh \
  --gpus 4 \
  --variant no_bridge \
  --epochs 10 \
  --nvme-size 1.5tb \
  --preset balanced \
  --hdd-root /ssd \
  --nvme-root /nvme
```

正常应看到：

- CUDA/GPU 检查通过
- manifest 检查通过
- `cache_A`、`cache_B` 检查通过
- `torchrun` 可用
- `swanlab` 可 import
- dry-run 命令能打印出来

体检脚本会尽量扫描所有条件，不会遇到第一个问题就中断。

## 9. Dry-run 训练命令

先打印完整 suite 命令，不真正启动：

```bash
bash scripts/train/arcdp_cache_rotation/train_arcdp_cache_rotation_4a800_100ep_suite.sh \
  --nvme-size 1.5tb \
  --preset balanced \
  --hdd-root /ssd \
  --nvme-root /nvme \
  --swanlab-project ArcDP-100ep \
  --dry-run
```

它应依次列出：

```text
full: 100 equivalent epochs
rel: 10 equivalent epochs
no_bridge: 10 equivalent epochs
no_ordered_init: 10 equivalent epochs
no_scale_cond: 10 equivalent epochs
no_anchor_train: 10 equivalent epochs
no_gcs: 10 equivalent epochs
```

## 10. 正式训练

建议用 `nohup` 或 tmux 启动：

```bash
cd /ssd/MyResearch/InternNav
mkdir -p /ssd/bridgedp_rotation/logs

export SWANLAB_API_KEY=<your_api_key>
export SWANLAB_PROJECT=ArcDP-100ep
export BRIDGEDP_LOGGING_STEPS=100
export BRIDGEDP_ETA_LOG_STEPS=500
export BRIDGEDP_SAVE_TOTAL_LIMIT=3
export BRIDGEDP_UNIFORM_CKPT_COUNT=20

nohup bash scripts/train/arcdp_cache_rotation/train_arcdp_cache_rotation_4a800_100ep_suite.sh \
  --nvme-size 1.5tb \
  --preset balanced \
  --hdd-root /ssd \
  --nvme-root /nvme \
  --swanlab-project ArcDP-100ep \
  > /ssd/bridgedp_rotation/logs/arcdp_4a800_full100_p0_10_suite.log 2>&1 &

echo $! > /ssd/bridgedp_rotation/logs/arcdp_4a800_full100_p0_10_suite.pid
```

查看日志：

```bash
tail -f /ssd/bridgedp_rotation/logs/arcdp_4a800_full100_p0_10_suite.log
```

训练会按顺序运行：

1. `full`，100 epoch 等量
2. `rel`，10 epoch 等量
3. `no_bridge`，10 epoch 等量
4. `no_ordered_init`，10 epoch 等量
5. `no_scale_cond`，10 epoch 等量
6. `no_anchor_train`，10 epoch 等量
7. `no_gcs`，10 epoch 等量

如果只跑某个版本：

```bash
bash scripts/train/arcdp_cache_rotation/train_arcdp_cache_rotation_4a800_100ep_suite.sh \
  --only full \
  --nvme-size 1.5tb \
  --preset balanced \
  --hdd-root /ssd \
  --nvme-root /nvme \
  --swanlab-project ArcDP-100ep
```

如果从某个消融继续：

```bash
bash scripts/train/arcdp_cache_rotation/train_arcdp_cache_rotation_4a800_100ep_suite.sh \
  --start-at no_scale_cond \
  --nvme-size 1.5tb \
  --preset balanced \
  --hdd-root /ssd \
  --nvme-root /nvme \
  --swanlab-project ArcDP-100ep
```

## 11. 训练监控闭环

### 本地日志

```bash
tail -f /ssd/bridgedp_rotation/logs/arcdp_4a800_full100_p0_10_suite.log
```

正常训练日志应包含：

```text
ArcDP cache-rotation training
Bridge-DP cache rotation training
total epochs: 100
shard epochs: 2
build workers:4
uniform ckpt: 20 -> /ssd/bridgedp_rotation/uniform_checkpoints/...
[ETA] ...
```

### GPU

```bash
watch -n 30 nvidia-smi
```

正常应看到 4 张 A800 有显存占用和利用率。

### Cache 轮转

训练过程中会一边训练当前 slot，一边用新版构建器准备下一个 inactive slot。确认新版生效：

```bash
grep -n "build-workers\|BRIDGEDP_BUILD_WORKERS" \
  /ssd/MyResearch/InternNav/train_bridgedp_cache_rotation.sh \
  /nvme/MyResearch/InternNav/train_bridgedp_cache_rotation.sh \
  /ssd/MyResearch/InternNav/scripts/cache_rotation/build_bridgedp_cache_shard.py \
  /nvme/MyResearch/InternNav/scripts/cache_rotation/build_bridgedp_cache_shard.py
```

训练时若看到后台构建命令，应包含：

```text
--build-workers 4
```

### SwanLab

打开 SwanLab 网页，在 `ArcDP-100ep` 项目下查看每个 run：

- `arcdp_full_100ep_4a800_1.5tb_balanced`
- `arcdp_rel_10ep_4a800_1.5tb_balanced`
- `arcdp_no_bridge_10ep_4a800_1.5tb_balanced`
- `arcdp_no_ordered_init_10ep_4a800_1.5tb_balanced`
- `arcdp_no_scale_cond_10ep_4a800_1.5tb_balanced`
- `arcdp_no_anchor_train_10ep_4a800_1.5tb_balanced`
- `arcdp_no_gcs_10ep_4a800_1.5tb_balanced`

重点看：

- loss 曲线
- learning rate
- `time/eta_hours`
- `progress/percent`
- global step

ETA 在最开始 30-60 分钟可能不稳定，进入稳定吞吐后再作为主要参考。

## 12. Checkpoint 位置

每个 run 的 live checkpoint 默认只保留最近 3 个：

```bash
/nvme/MyResearch/InternNav/checkpoints/<run_name>/ckpts
```

均匀 checkpoint 归档在：

```bash
/ssd/bridgedp_rotation/uniform_checkpoints/<run_name>
```

full 100 epoch 目标：

- 保留 20 个均匀 checkpoint
- 第 2 个约等价于 full 10 epoch
- 最后一个约等价于 full 100 epoch

检查：

```bash
find /ssd/bridgedp_rotation/uniform_checkpoints -maxdepth 2 -type d | sort
find /ssd/bridgedp_rotation/uniform_checkpoints/arcdp_full_100ep_4a800_1.5tb_balanced -maxdepth 1 -type d | sort
```

## 13. 常见异常

### cache 准备日志没有 workers=4

说明远端脚本不是新版。先同步 `/ssd/MyResearch/InternNav`，再重新同步到 `/nvme/MyResearch/InternNav`。

### kill 父脚本后还有 PID

`pgrep` 里出现的数字是 PID，不是进度。例如：

```text
255 python ... build_bridgedp_cache_shard.py ...
```

这表示 Python 子进程还在构建 cache，没有完成。如果要切换新版脚本，应停止该 PID。

### cache 速度太慢

如果 5 分钟只增加 1GB 左右，单 slot 会需要数十小时。优先确认：

```bash
tail -100 /ssd/bridgedp_rotation/logs/prepare_cache_4gpu_1p5tb.log
pgrep -af "build_bridgedp_cache_shard"
```

如果命令没有 `--build-workers 4`，说明还在跑旧脚本。

### SwanLab 没有曲线

检查：

```bash
python -c "import swanlab; print(swanlab.__version__)"
echo "$SWANLAB_API_KEY" | wc -c
grep -n "SwanLab" /ssd/bridgedp_rotation/logs/arcdp_4a800_full100_p0_10_suite.log | tail
```

如果没有 key：

```bash
export SWANLAB_API_KEY=<your_api_key>
```

然后重新启动训练。

## 14. 最小推荐执行顺序

1. 在 CPU/存储开发机确认 `/ssd` 数据、preload、manifest。
2. 用新版脚本准备 `/nvme/bridgedp_cache/cache_A` 和 `cache_B`。
3. 确认两个 cache slot 都 READY。
4. 在 CPU/存储开发机打包 CUDA 12.6 版 base conda 环境。
5. 租 4 卡 A800。
6. 在 A800 上确认能看到同一份 `/ssd` 和 `/nvme`。
7. 部署 conda pack，确认 4 张 GPU、PyTorch CUDA 12.6、SwanLab。
8. 跑 `check_arcdp_training_ready.sh`。
9. dry-run suite。
10. 正式启动 full 100 epoch + P0 消融 10 epoch suite。
11. 用日志、`nvidia-smi`、cache READY 状态和 SwanLab 形成监控闭环。
