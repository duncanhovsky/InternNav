# ArcDP 4 卡 A800 16ep/2ep 新训练方案

本文档记录 2026-07-05 调整后的轻量训练方案：

- ArcDP full：16 epoch 等量训练
- P0 消融：`rel`、`no_bridge`、`no_ordered_init`、`no_scale_cond`、`no_anchor_train`、`no_gcs` 各 2 epoch 等量训练
- full 16ep 训练保留 8 个均匀 checkpoint，第 1 个 checkpoint 等价于 full 2ep，用作消融对比基线
- 每个 shard 连续训练 `shard_epochs=10` 后再切换 cache，降低 GPU 等 cache 的概率

## 1. 是否需要重新准备 cache

不需要。

`shard_epochs=10` 只改变每个 READY cache slot 上训练多久，不改变以下内容：

- 数据集路径
- manifest 的 10 个 shards
- 每个 shard 包含哪些 scene
- `cache_A/cache_B` 里的文件布局

所以只要下面检查通过，就可以复用已准备好的 cache：

```bash
test -f /ssd/bridgedp_rotation/manifests/shards_4gpu_1.5tb_balanced.json && echo manifest_ok
test -f /nvme/bridgedp_cache/cache_A/.READY && echo cache_A_ready
test -f /nvme/bridgedp_cache/cache_B/.READY && echo cache_B_ready

cd /ssd/MyResearch/InternNav
python scripts/cache_rotation/check_bridgedp_cache.py --slot-root /nvme/bridgedp_cache/cache_A
python scripts/cache_rotation/check_bridgedp_cache.py --slot-root /nvme/bridgedp_cache/cache_B
```

只有在重新分 shard、改变 `cache_slot_gb`、换数据集或 cache 校验失败时，才需要重新准备 cache。

## 2. 训练量和步数

当前数据集：

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
| full | 16 | 409,450 |
| full 对照 checkpoint | 2 | 51,182 |
| 每个 P0 消融 | 2 | 51,182 |
| 6 个 P0 消融合计 | 12 | 307,092 |
| 全套合计 | 28 | 716,542 |

full 16ep 使用 8 个均匀 checkpoint：

```text
409450 / 8 = 51181.25
第 1 个均匀 checkpoint = 51182 steps ≈ full 2ep
```

## 3. shard_epochs=10 的影响

当前 manifest 是 10 个 shards，平均每个 shard 约：

```text
196536 / 10 = 19654 episodes
```

`shard_epochs=10` 时，每个 cache slot 平均训练步数：

```text
ceil(50 * 19654 * 10 / 384) ≈ 25,590 steps
```

因此 stage 数约为：

| 训练项 | 总步数 | stage 数 |
|---|---:|---:|
| full 16ep | 409,450 | 16 |
| 每个 P0 2ep | 51,182 | 2 |
| 6 个 P0 合计 | 307,092 | 12 |
| 全套合计 | 716,542 | 28 |

这比 `balanced` 默认 `shard_epochs=2` 的约 140 个 stage 少很多，cache 等待风险会明显下降。

## 4. 预计训练时间

当前 cache 构建速度估计：

```text
600GB slot ≈ 5.5-7 小时
```

`shard_epochs=10` 时，每个 stage 训练约 25,590 steps。按不同 step time 估计：

| 假设 step time | 训练计算时间 | cache 下限 | 预计总时长 |
|---:|---:|---:|---:|
| 0.5 s/step | 4.1 天 | 7-8 天 | 7-8 天 |
| 1.0 s/step | 8.3 天 | 7-8 天 | 8-9 天 |
| 2.0 s/step | 16.6 天 | 7-8 天 | 16-17 天 |
| 3.0 s/step | 24.9 天 | 7-8 天 | 25 天左右 |

实际 4 卡 A800 的总时间主要取决于真实 step time。建议先用 `--max-stages 1` 跑 benchmark，30-60 分钟后用 SwanLab 的 ETA 修正估计。

## 5. 同步新版脚本

正式训练前，确认服务器上有新版脚本：

```bash
grep -n "shard-epochs\|2|10|16|100" \
  /ssd/MyResearch/InternNav/scripts/train/arcdp_cache_rotation/train_arcdp_cache_rotation.sh

test -f /ssd/MyResearch/InternNav/scripts/train/arcdp_cache_rotation/train_arcdp_cache_rotation_4a800_16ep_suite.sh \
  && echo suite16_ok
```

如果 `/nvme/MyResearch/InternNav` 是旧副本，重新同步：

```bash
rsync -aH --info=progress2 \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  /ssd/MyResearch/InternNav/ /nvme/MyResearch/InternNav/
```

## 6. 训练前体检

full 16ep：

```bash
cd /ssd/MyResearch/InternNav

bash scripts/train/arcdp_cache_rotation/check_arcdp_training_ready.sh \
  --gpus 4 \
  --variant full \
  --epochs 16 \
  --shard-epochs 10 \
  --nvme-size 1.5tb \
  --preset balanced \
  --hdd-root /ssd \
  --nvme-root /nvme
```

抽查一个 P0 消融：

```bash
bash scripts/train/arcdp_cache_rotation/check_arcdp_training_ready.sh \
  --gpus 4 \
  --variant no_bridge \
  --epochs 2 \
  --shard-epochs 10 \
  --nvme-size 1.5tb \
  --preset balanced \
  --hdd-root /ssd \
  --nvme-root /nvme
```

## 7. SwanLab 配置

推荐用环境变量：

```bash
export SWANLAB_API_KEY=<your_api_key>
export SWANLAB_PROJECT=ArcDP-16ep
```

训练脚本会默认启用 SwanLab，并显示：

```text
[SwanLab] SwanLab enabled for this training run.
[SwanLab] SwanLab project: ArcDP-16ep
[SwanLab] report_to=swanlab
```

## 8. Benchmark

正式长跑前，建议先跑一个 stage：

```bash
cd /ssd/MyResearch/InternNav

export SWANLAB_API_KEY=<your_api_key>
export SWANLAB_PROJECT=ArcDP-benchmark

bash scripts/train/arcdp_cache_rotation/train_arcdp_cache_rotation_4a800.sh \
  --variant full \
  --epochs 16 \
  --shard-epochs 10 \
  --nvme-size 1.5tb \
  --preset balanced \
  --hdd-root /ssd \
  --nvme-root /nvme \
  --max-stages 1 \
  --swanlab-project ArcDP-benchmark
```

观察：

```bash
watch -n 30 nvidia-smi
tail -f /ssd/bridgedp_rotation/logs/arcdp_full_16ep_4a800_1.5tb_balanced/build_stage_0_next.log
```

SwanLab 中重点看：

- step time
- `time/eta_hours`
- `progress/percent`
- GPU 利用率
- 当前 stage 是否结束时下一个 cache 已 READY

## 9. 正式训练

```bash
cd /ssd/MyResearch/InternNav
mkdir -p /ssd/bridgedp_rotation/logs

export SWANLAB_API_KEY=<your_api_key>
export SWANLAB_PROJECT=ArcDP-16ep
export BRIDGEDP_LOGGING_STEPS=100
export BRIDGEDP_ETA_LOG_STEPS=500
export BRIDGEDP_SAVE_TOTAL_LIMIT=3

nohup bash scripts/train/arcdp_cache_rotation/train_arcdp_cache_rotation_4a800_16ep_suite.sh \
  --nvme-size 1.5tb \
  --preset balanced \
  --hdd-root /ssd \
  --nvme-root /nvme \
  --swanlab-project ArcDP-16ep \
  > /ssd/bridgedp_rotation/logs/arcdp_4a800_full16_p0_2_suite.log 2>&1 &

echo $! > /ssd/bridgedp_rotation/logs/arcdp_4a800_full16_p0_2_suite.pid
```

默认会依次运行：

1. `full`，16 epoch，8 个均匀 checkpoint
2. `rel`，2 epoch，2 个均匀 checkpoint
3. `no_bridge`，2 epoch，2 个均匀 checkpoint
4. `no_ordered_init`，2 epoch，2 个均匀 checkpoint
5. `no_scale_cond`，2 epoch，2 个均匀 checkpoint
6. `no_anchor_train`，2 epoch，2 个均匀 checkpoint
7. `no_gcs`，2 epoch，2 个均匀 checkpoint

## 10. 监控

主日志：

```bash
tail -f /ssd/bridgedp_rotation/logs/arcdp_4a800_full16_p0_2_suite.log
```

GPU：

```bash
watch -n 30 nvidia-smi
```

cache：

```bash
pgrep -af "build_bridgedp_cache_shard|train_cache_rotation|torchrun"
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

均匀 checkpoint 归档：

```bash
/ssd/bridgedp_rotation/uniform_checkpoints/<run_name>
```

full 16ep 的第 1 个均匀 checkpoint 约等价于 full 2ep，可用于和 P0 2ep 消融对比。
