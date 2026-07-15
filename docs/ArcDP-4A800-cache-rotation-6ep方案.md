# ArcDP 4 卡 A800 6ep/1ep 快速训练方案

本文档记录当前推荐的更短训练方案：

- ArcDP full：6 epoch 等量训练
- P0 消融：`rel`、`no_bridge`、`no_ordered_init`、`no_scale_cond`、`no_anchor_train`、`no_gcs` 各 1 epoch 等量训练
- full 6ep 训练保留 6 个均匀 checkpoint，第 1 个 checkpoint 等价于 full 1ep，用作 P0 消融对照
- 每个 shard 使用 `shard_epochs=10`，减少 cache 轮转次数和 GPU 等 cache 的时间

## 1. 重要取舍

这套方案优先降低训练时间，适合作为快速主实验/消融筛选。

`shard_epochs=10` 下，一个 600GB cache slot 平均训练约 1 个全数据 epoch 的等量步数。因此：

- full 6ep 约 6 个 stage
- 每个 P0 1ep 约 1 个 stage
- 全套约 12 个 stage

代价是：短训时更偏“等量训练量对比”，不是每个模型都完整扫遍 10 个 shards。full 1ep 对照和 P0 1ep 消融都默认从 shard 0 开始，因此对比是公平的，但覆盖的数据子集较少。如果后续需要更充分的数据覆盖，再把 `shard_epochs` 降低或增加 epoch。

## 2. 是否需要重新准备 cache

不需要。

`shard_epochs=10`、`full=6ep`、`P0=1ep` 都只改变训练步数和 cache 切换频率，不改变：

- manifest
- 10 个 shard 的划分
- `cache_A/cache_B` 文件内容
- `/ssd` 数据源

只要当前 cache 已 READY，就可以复用：

```bash
cd /ssd/MyResearch/InternNav

test -f /ssd/bridgedp_rotation/manifests/shards_4gpu_1.5tb_balanced.json && echo manifest_ok
test -f /nvme/bridgedp_cache/cache_A/.READY && echo cache_A_ready
test -f /nvme/bridgedp_cache/cache_B/.READY && echo cache_B_ready

python scripts/cache_rotation/check_bridgedp_cache.py --slot-root /nvme/bridgedp_cache/cache_A
python scripts/cache_rotation/check_bridgedp_cache.py --slot-root /nvme/bridgedp_cache/cache_B
```

## 3. 训练量和步数

当前数据集和 batch：

```text
total_episodes = 196536
gpus = 4
per_gpu_batch = 96
grad_accum = 1
global_batch = 384
```

步数公式：

```text
total_max_steps = ceil(50 * total_episodes * epochs / global_batch)
```

对应步数：

| 训练项 | 等量 epoch | 步数 |
|---|---:|---:|
| full | 6 | 153,544 |
| full 对照 checkpoint | 1 | 25,591 |
| 每个 P0 消融 | 1 | 25,591 |
| 6 个 P0 消融合计 | 6 | 153,546 |
| 全套合计 | 12 | 307,090 |

full 6ep 使用 6 个均匀 checkpoint：

```text
153544 / 6 = 25590.67
第 1 个均匀 checkpoint = 25591 steps ≈ full 1ep
```

## 4. 预计训练时间

以当前 cache 构建速度估计：

```text
600GB slot ≈ 5.5-7 小时
全套约 12 个 stage
cache 下限 ≈ 66-84 小时，即 2.8-3.5 天
```

按 4 卡 A800 不同 step time 估计：

| 假设 step time | 纯训练计算时间 | cache 下限 | 预计总时长 |
|---:|---:|---:|---:|
| 0.5 s/step | 1.8 天 | 2.8-3.5 天 | 3-4 天 |
| 1.0 s/step | 3.6 天 | 2.8-3.5 天 | 4 天左右 |
| 2.0 s/step | 7.1 天 | 2.8-3.5 天 | 7-8 天 |
| 3.0 s/step | 10.7 天 | 2.8-3.5 天 | 11 天左右 |

最可靠的估计仍然来自 4 卡 A800 上先跑 `--max-stages 1` 的 benchmark。

## 5. 同步新版脚本

正式跑前必须把新版脚本同步到服务器，至少确认：

```bash
cd /ssd/MyResearch/InternNav

grep -n "1|2|6|10|16|100" scripts/train/arcdp_cache_rotation/train_arcdp_cache_rotation.sh
grep -n "shard-epochs" train_bridgedp_cache_rotation.sh
test -f scripts/train/arcdp_cache_rotation/train_arcdp_cache_rotation_4a800_6ep_suite.sh && echo suite6_ok
```

如果 `/nvme/MyResearch/InternNav` 是旧副本，重新同步：

```bash
rsync -aH --info=progress2 \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  /ssd/MyResearch/InternNav/ /nvme/MyResearch/InternNav/
```

## 6. 训练前体检

full 6ep：

```bash
cd /ssd/MyResearch/InternNav

bash scripts/train/arcdp_cache_rotation/check_arcdp_training_ready.sh \
  --gpus 4 \
  --variant full \
  --epochs 6 \
  --shard-epochs 10 \
  --nvme-size 1.5tb \
  --preset balanced \
  --hdd-root /ssd \
  --nvme-root /nvme
```

抽查一个 P0 1ep：

```bash
bash scripts/train/arcdp_cache_rotation/check_arcdp_training_ready.sh \
  --gpus 4 \
  --variant no_bridge \
  --epochs 1 \
  --shard-epochs 10 \
  --nvme-size 1.5tb \
  --preset balanced \
  --hdd-root /ssd \
  --nvme-root /nvme
```

## 7. SwanLab

推荐非交互方式：

```bash
export SWANLAB_API_KEY=<your_api_key>
export SWANLAB_PROJECT=ArcDP-6ep
```

训练启动时应显示：

```text
[SwanLab] SwanLab enabled for this training run.
[SwanLab] SwanLab project: ArcDP-6ep
[SwanLab] report_to=swanlab
```

## 8. Benchmark

正式长跑前，先跑一个 stage：

```bash
cd /ssd/MyResearch/InternNav

export SWANLAB_API_KEY=<your_api_key>
export SWANLAB_PROJECT=ArcDP-benchmark

bash scripts/train/arcdp_cache_rotation/train_arcdp_cache_rotation_4a800.sh \
  --variant full \
  --epochs 6 \
  --shard-epochs 10 \
  --nvme-size 1.5tb \
  --preset balanced \
  --hdd-root /ssd \
  --nvme-root /nvme \
  --max-stages 1 \
  --swanlab-project ArcDP-benchmark
```

观察 30-60 分钟：

```bash
watch -n 30 nvidia-smi
tail -f /ssd/bridgedp_rotation/logs/arcdp_full_6ep_4a800_1.5tb_balanced/build_stage_0_next.log
```

SwanLab 里看：

- step time
- `time/eta_hours`
- `progress/percent`
- loss 曲线
- GPU 利用率

## 9. 一键启动全量 + 全部 P0 消融

```bash
cd /ssd/MyResearch/InternNav
mkdir -p /ssd/bridgedp_rotation/logs

export SWANLAB_API_KEY=<your_api_key>
export SWANLAB_PROJECT=ArcDP-6ep
export BRIDGEDP_LOGGING_STEPS=100
export BRIDGEDP_ETA_LOG_STEPS=500
export BRIDGEDP_SAVE_TOTAL_LIMIT=3

nohup bash scripts/train/arcdp_cache_rotation/train_arcdp_cache_rotation_4a800_6ep_suite.sh \
  --nvme-size 1.5tb \
  --preset balanced \
  --hdd-root /ssd \
  --nvme-root /nvme \
  --swanlab-project ArcDP-6ep \
  > /ssd/bridgedp_rotation/logs/arcdp_4a800_full6_p0_1_suite.log 2>&1 &

echo $! > /ssd/bridgedp_rotation/logs/arcdp_4a800_full6_p0_1_suite.pid
```

默认顺序：

1. `full`，6 epoch，6 个均匀 checkpoint
2. `rel`，1 epoch，1 个均匀 checkpoint
3. `no_bridge`，1 epoch，1 个均匀 checkpoint
4. `no_ordered_init`，1 epoch，1 个均匀 checkpoint
5. `no_scale_cond`，1 epoch，1 个均匀 checkpoint
6. `no_anchor_train`，1 epoch，1 个均匀 checkpoint
7. `no_gcs`，1 epoch，1 个均匀 checkpoint

## 10. 监控

主日志：

```bash
tail -f /ssd/bridgedp_rotation/logs/arcdp_4a800_full6_p0_1_suite.log
```

进程：

```bash
pgrep -af "train_arcdp_cache_rotation|train_bridgedp_cache_rotation|torchrun|build_bridgedp_cache_shard"
```

GPU：

```bash
watch -n 30 nvidia-smi
```

cache：

```bash
df -h /nvme
df -ih /nvme
find /nvme/bridgedp_cache -name .READY -print
```

checkpoint：

```bash
find /ssd/bridgedp_rotation/uniform_checkpoints -maxdepth 2 -type d | sort
find /nvme/MyResearch/InternNav/checkpoints -maxdepth 3 -type d -name 'checkpoint-*' | sort | tail
```

## 11. 输出位置

live checkpoint：

```bash
/nvme/MyResearch/InternNav/checkpoints/<run_name>/ckpts
```

均匀 checkpoint：

```bash
/ssd/bridgedp_rotation/uniform_checkpoints/<run_name>
```

full 6ep 的第 1 个均匀 checkpoint 约等价于 full 1ep，可用于和 P0 1ep 消融对比。
