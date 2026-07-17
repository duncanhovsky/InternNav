# ArcDP 8×4090 单 epoch 安全断点训练

本方案沿用 B+ 的计算与 DataLoader 参数：每卡 batch 24、梯度累积 2、每个 rank 4 个 worker、prefetch factor 1。独立 run 名为：

```text
arcdp_full1_b24_ga2_w4_pf1_safeckpt_8x4090_ceph
```

它不会读取原 B+ run 的 checkpoint。每 1% 训练进度保存一次完整、可恢复的 Trainer checkpoint；10%、20%……100% 永久保留，其余进度只保留最新 5 份。

## 更新并预检

当前正在运行的旧代码不会自动获得新保存逻辑。先在原 tmux 窗口按 `Ctrl+C` 停止旧训练，再更新代码：

```bash
cd /nvme/MyResearch/InternNav_v1.0.3
git pull origin bridgedp_v1.0.3
conda activate arcdp
chmod +x train_arcdp_full1_b24_ga2_w4_pf1_safeckpt_8x4090_ceph.sh
chmod +x scripts/train/arcdp_4090/train_arcdp_full1_8x4090_ceph_io.sh
```

只打印配置、不启动训练：

```bash
ARCDP_DRY_RUN=1 ./train_arcdp_full1_b24_ga2_w4_pf1_safeckpt_8x4090_ceph.sh
```

预期包含：

```text
per-device batch: 24
gradient accumulation: 2
effective global batch: 384
workers/process: 4
prefetch factor/worker: 1
checkpoint interval: 1%
rolling checkpoints kept: 5
permanent checkpoint interval: 10%
resume only complete checkpoints: 1
```

## 在 tmux 中直接启动

```bash
tmux new -s arcdp_safeckpt
cd /nvme/MyResearch/InternNav_v1.0.3
conda activate arcdp
set -o pipefail
LOG="logs/arcdp_safeckpt_$(date +%F_%H%M%S).log"
./train_arcdp_full1_b24_ga2_w4_pf1_safeckpt_8x4090_ceph.sh 2>&1 | tee "$LOG"
```

按 `Ctrl+B`、再按 `D` 退出 tmux；重新进入：

```bash
tmux attach -t arcdp_safeckpt
```

如果训练中断，重新执行同一条启动命令即可。脚本默认自动选择该 run 下 step 最大且带完整性标记的 checkpoint；没有标记的半成品目录会被跳过。

## 保存位置和检查方法

checkpoint 根目录：

```text
checkpoints/arcdp_full1_b24_ga2_w4_pf1_safeckpt_8x4090_ceph/ckpts
```

对于 25,590 个 optimizer steps，1%、10%、100% 分别对应 step 256、2,559、25,590。实际总 step 改变时会按 `ceil(total_steps × 百分比 / 100)` 自动重算。

查看已提交的 checkpoint 和保留清单：

```bash
CKPT_ROOT=checkpoints/arcdp_full1_b24_ga2_w4_pf1_safeckpt_8x4090_ceph/ckpts
find "$CKPT_ROOT" -maxdepth 2 -type f -name CHECKPOINT_COMPLETE.json -print | sort -V
python -m json.tool "$CKPT_ROOT/percent_checkpoint_manifest.json"
```

检查最新 checkpoint 中恢复训练所需的文件：

```bash
LATEST=$(find "$CKPT_ROOT" -maxdepth 1 -type d -name 'checkpoint-*' \
  -exec test -f '{}/CHECKPOINT_COMPLETE.json' ';' -print | sort -V | tail -n 1)
echo "$LATEST"
python -m json.tool "$LATEST/CHECKPOINT_COMPLETE.json"
ls -lh "$LATEST"/{pytorch_model.bin,bridgedp.ckpt,optimizer.pt,scheduler.pt,trainer_state.json}
ls -lh "$LATEST"/rng_state_*.pth
```

成功保存时日志会出现类似：

```text
[PercentCheckpoint] verified 1% at step 256: ...; permanent=0, pruned=0
[PercentCheckpoint] verified 10% at step 2559: ...; permanent=1, pruned=1
```

只有模型、优化器、调度器、Trainer 状态和 8 个 rank 的 RNG 状态全部存在且能加载后，程序才会原子写入 `CHECKPOINT_COMPLETE.json`。因此断电或强制结束时即使留下半写入目录，自动续训也不会使用它。

## 最终完成判据

训练正常退出前必须出现：

```text
[PercentCheckpoint] verified 100% ... permanent=1
[PercentCheckpoint] final checkpoint verified: .../checkpoint-25590
```

最终最多保留 15 份已完成 checkpoint：10 份永久 checkpoint 加 5 份滚动 checkpoint。写新 checkpoint 时还需要临时磁盘空间，请同时观察：

```bash
df -h /nvme
du -sh "$CKPT_ROOT"
```
