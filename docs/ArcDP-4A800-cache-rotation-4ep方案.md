# ArcDP 4 卡 A800 4ep/1ep 默认训练方案

本文档记录当前默认训练方案：

- ArcDP full：4 epoch 等量训练
- P0 消融：`rel`、`no_bridge`、`no_ordered_init`、`no_scale_cond`、`no_anchor_train`、`no_gcs` 各 1 epoch 等量训练
- full 4ep 训练保留 4 个均匀 checkpoint，第 1 个 checkpoint 等价于 full 1ep，用作 P0 消融对照
- 每个 shard 使用 `shard_epochs=10`
- 最终 stage 不再预构建 next cache，避免短训结束前额外白等一个 cache slot

这份文档默认采用快速模式：`shard_epochs=10`。如果实验目标更强调每个模型均匀覆盖整个数据集，请使用本文第 4 节的覆盖优先模式。

## 1. 是否需要重新准备 cache

不需要。

这套方案只改变训练总步数和每个 shard 的训练时长，不改变：

- `/ssd` 已解压数据
- 10-shard manifest
- `cache_A/cache_B` 文件布局
- cache slot 大小

只要现有 cache 通过检查，就可以直接训练：

```bash
cd /ssd/MyResearch/InternNav

test -f /ssd/bridgedp_rotation/manifests/shards_4gpu_1.5tb_balanced.json && echo manifest_ok
test -f /nvme/bridgedp_cache/cache_A/.READY && echo cache_A_ready
test -f /nvme/bridgedp_cache/cache_B/.READY && echo cache_B_ready

python scripts/cache_rotation/check_bridgedp_cache.py --slot-root /nvme/bridgedp_cache/cache_A
python scripts/cache_rotation/check_bridgedp_cache.py --slot-root /nvme/bridgedp_cache/cache_B
```

## 2. 训练量和步数

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
| full | 4 | 102,363 |
| full 对照 checkpoint | 1 | 25,591 |
| 每个 P0 消融 | 1 | 25,591 |
| 6 个 P0 消融合计 | 6 | 153,546 |
| 全套合计 | 10 | 255,909 |

full 4ep 使用 4 个均匀 checkpoint：

```text
102363 / 4 = 25590.75
第 1 个均匀 checkpoint = 25591 steps ≈ full 1ep
```

## 3. GPU 空载率估计

这里估计的是 cache 等待导致的 GPU 空载率，不包含训练内部 dataloader 或模型计算导致的利用率波动。

当前 cache 构建速度估计：

```text
600GB slot ≈ 5.5-7 小时
```

`shard_epochs=10` 时，一个 stage 平均训练约：

```text
25,591 steps
```

已预构建 `cache_A=shard0` 和 `cache_B=shard1` 后，full 4ep 大约还需要构建 shard2、shard3；随后第一个 P0 需要把 `cache_A` 切回 shard0。由于最终 stage 会跳过 next-cache 预构建，全套额外 cache 等待约为：

```text
2 * max(0, cache_build_time - stage_train_time) + cache_build_time
```

估计表：

| 假设 step time | 纯训练时间 | cache 等待 | 总时长 | cache 空载率 |
|---:|---:|---:|---:|---:|
| 0.5 s/step | 35.5 h | 9.4-13.9 h | 1.9-2.1 天 | 21-28% |
| 1.0 s/step | 71.1 h | 5.5-7.0 h | 3.2-3.3 天 | 7-9% |
| 2.0 s/step | 142.2 h | 5.5-7.0 h | 6.2 天 | 4-5% |
| 3.0 s/step | 213.3 h | 5.5-7.0 h | 9.1 天 | 3% 左右 |

更准确的数值要以 4 卡 A800 上 `--max-stages 1` benchmark 为准。

## 4. 数据覆盖策略

### 当前默认是否均匀覆盖全数据集

不能。

当前默认 `shard_epochs=10` 时，一个 stage 大约就是 1 个全数据 epoch 的等量步数：

```text
每个 stage ≈ 25,591 steps
full 4ep ≈ 4 stages
每个 P0 1ep ≈ 1 stage
```

这意味着：

- full 4ep 默认主要覆盖约 4 个 shards。
- 每个 P0 1ep 默认主要覆盖 shard0。
- full 1ep checkpoint 和 P0 1ep 消融默认用同一个 shard0，对比公平，但不是全数据覆盖。

当前 manifest 主要按 scene group 和 shard 大小做均衡，不会按困难/简单样本分层；也就是说，目前没有 difficulty-aware 的均匀覆盖。

### 覆盖优先模式

如果希望 full 和每个 P0 都尽量均匀覆盖整个 10-shard 数据集，使用：

```bash
--shard-epochs 1
```

这样：

- full 4ep 约 40 stages，约完整覆盖 10 个 shards 4 轮。
- 每个 P0 1ep 约 10 stages，约完整覆盖 10 个 shards 1 轮。
- 全套约 100 stages。

启动命令：

```bash
nohup bash scripts/train/arcdp_cache_rotation/train_arcdp_cache_rotation_4a800_4ep_suite.sh \
  --nvme-size 1.5tb \
  --preset balanced \
  --hdd-root /ssd \
  --nvme-root /nvme \
  --shard-epochs 1 \
  --swanlab-project ArcDP-4ep-coverage \
  > /ssd/bridgedp_rotation/logs/arcdp_4a800_full4_p0_1_coverage.log 2>&1 &
```

代价是 cache 轮转显著增多。按当前 600GB slot 构建 `5.5-7 小时` 估计，覆盖优先模式大约需要 20-30 天。

### 折中模式

如果希望比默认覆盖更多数据，但又不想等完整覆盖模式，可以用：

```bash
--shard-epochs 4
```

这样：

- full 4ep 约 10 stages，可覆盖全部 10 个 shards 约 1 轮。
- 每个 P0 1ep 约 3 stages，只覆盖约 3 个 shards。
- 全套约 28 stages。
- 预计耗时约 6-8 天，取决于 step time 和 cache 构建速度。

启动命令：

```bash
nohup bash scripts/train/arcdp_cache_rotation/train_arcdp_cache_rotation_4a800_4ep_suite.sh \
  --nvme-size 1.5tb \
  --preset balanced \
  --hdd-root /ssd \
  --nvme-root /nvme \
  --shard-epochs 4 \
  --swanlab-project ArcDP-4ep-balanced-coverage \
  > /ssd/bridgedp_rotation/logs/arcdp_4a800_full4_p0_1_shard4.log 2>&1 &
```

### 困难/简单样本均匀覆盖

当前不能直接做到。要实现 difficulty-aware 覆盖，需要新增一套 manifest 生成逻辑：

1. 从 episode 或 scene metadata 中提取 difficulty 指标，例如轨迹长度、路径复杂度、失败/成功统计、语言长度或人工定义难度。
2. 将样本或 scene 分成 easy/medium/hard 等桶。
3. 生成 stratified shards，让每个 shard 都包含相近比例的困难/简单样本。
4. 重新准备 `cache_A/cache_B`。

这会改变 manifest 和 shard 内容，所以需要重建 cache。当前默认流程先不做这一步。

## 5. 同步新版脚本

正式跑前确认服务器上是最新版：

```bash
cd /ssd/MyResearch/InternNav

grep -n "1|2|4|6|10|16|100" scripts/train/arcdp_cache_rotation/train_arcdp_cache_rotation.sh
grep -n "Final stage reaches total max steps" train_bridgedp_cache_rotation.sh
test -f scripts/train/arcdp_cache_rotation/train_arcdp_cache_rotation_4a800_4ep_suite.sh && echo suite4_ok
```

如果 `/nvme/MyResearch/InternNav` 是旧副本，重新同步：

```bash
rsync -aH --info=progress2 \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  /ssd/MyResearch/InternNav/ /nvme/MyResearch/InternNav/
```

## 6. 训练前体检

full 4ep：

```bash
cd /ssd/MyResearch/InternNav

bash scripts/train/arcdp_cache_rotation/check_arcdp_training_ready.sh \
  --gpus 4 \
  --variant full \
  --epochs 4 \
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

```bash
export SWANLAB_API_KEY=<your_api_key>
export SWANLAB_PROJECT=ArcDP-4ep
```

训练启动时应显示：

```text
[SwanLab] SwanLab enabled for this training run.
[SwanLab] SwanLab project: ArcDP-4ep
[SwanLab] report_to=swanlab
```

## 8. Benchmark

正式长跑前建议先跑一个 stage：

```bash
cd /ssd/MyResearch/InternNav

export SWANLAB_API_KEY=<your_api_key>
export SWANLAB_PROJECT=ArcDP-benchmark

bash scripts/train/arcdp_cache_rotation/train_arcdp_cache_rotation_4a800.sh \
  --variant full \
  --epochs 4 \
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
tail -f /ssd/bridgedp_rotation/logs/arcdp_full_4ep_4a800_1.5tb_balanced/build_stage_0_next.log
```

SwanLab 里看：

- step time
- `time/eta_hours`
- `progress/percent`
- loss 曲线
- GPU 利用率

## 9. 一键启动 full + 全部 P0 消融

```bash
cd /ssd/MyResearch/InternNav
mkdir -p /ssd/bridgedp_rotation/logs

export SWANLAB_API_KEY=<your_api_key>
export SWANLAB_PROJECT=ArcDP-4ep
export BRIDGEDP_LOGGING_STEPS=100
export BRIDGEDP_ETA_LOG_STEPS=500
export BRIDGEDP_SAVE_TOTAL_LIMIT=3

nohup bash scripts/train/arcdp_cache_rotation/train_arcdp_cache_rotation_4a800_4ep_suite.sh \
  --nvme-size 1.5tb \
  --preset balanced \
  --hdd-root /ssd \
  --nvme-root /nvme \
  --swanlab-project ArcDP-4ep \
  > /ssd/bridgedp_rotation/logs/arcdp_4a800_full4_p0_1_suite.log 2>&1 &

echo $! > /ssd/bridgedp_rotation/logs/arcdp_4a800_full4_p0_1_suite.pid
```

默认顺序：

1. `full`，4 epoch，4 个均匀 checkpoint
2. `rel`，1 epoch，1 个均匀 checkpoint
3. `no_bridge`，1 epoch，1 个均匀 checkpoint
4. `no_ordered_init`，1 epoch，1 个均匀 checkpoint
5. `no_scale_cond`，1 epoch，1 个均匀 checkpoint
6. `no_anchor_train`，1 epoch，1 个均匀 checkpoint
7. `no_gcs`，1 epoch，1 个均匀 checkpoint

## 10. 监控

主日志：

```bash
tail -f /ssd/bridgedp_rotation/logs/arcdp_4a800_full4_p0_1_suite.log
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

full 4ep 的第 1 个均匀 checkpoint 约等价于 full 1ep，可用于和 P0 1ep 消融对比。
