# ArcDP 8卡4090 Ceph I/O A/B测试

本方案只训练 full 模型1个epoch，不会启动 no-bridge 或其他消融实验。两套方案的有效全局batch均为384，学习率均为 `3e-4`。

## 方案参数

| 方案 | 每卡batch | 梯度累积 | 每rank workers | prefetch/worker | 有效全局batch |
|---|---:|---:|---:|---:|---:|
| A（优先） | 48 | 1 | 2 | 2 | 384 |
| B（A仍卡顿时） | 24 | 2 | 2 | 1 | 384 |

方案A只把DataLoader进程从32个降至16个，GPU计算batch不变。方案B进一步减少Ceph瞬时排队样本，但小micro-batch可能降低少量纯GPU效率。

## 1. 更新代码并进入环境

```bash
cd /nvme/MyResearch/InternNav_v1.0.3
git pull origin bridgedp_v1.0.3
conda activate arcdp

chmod +x train_arcdp_full1_b48_w2_8x4090_ceph.sh
chmod +x train_arcdp_full1_b24_ga2_w2_pf1_8x4090_ceph.sh
```

脚本会检查是否已经存在GPU计算进程。当前训练没有结束时，实际启动会被拒绝，避免误杀或与旧训练争抢显存。`ARCDP_DRY_RUN=1` 只打印配置，不占用GPU。

## 2. 检查方案A配置

```bash
ARCDP_DRY_RUN=1 ./train_arcdp_full1_b48_w2_8x4090_ceph.sh
```

确认输出包含：

```text
per-device batch: 48
gradient accumulation: 1
effective global batch: 384
workers/process: 2
prefetch factor/worker: 2
```

## 3. 启动方案A

前台启动并同时保存日志：

```bash
mkdir -p logs
./train_arcdp_full1_b48_w2_8x4090_ceph.sh \
  2>&1 | tee logs/arcdp_full1_b48_w2_$(date +%F_%H%M%S).log
```

需要断开SSH后继续运行时：

```bash
LOG=logs/arcdp_full1_b48_w2_$(date +%F_%H%M%S).log
nohup setsid ./train_arcdp_full1_b48_w2_8x4090_ceph.sh \
  >"$LOG" 2>&1 < /dev/null &
echo $! | tee logs/arcdp_full1_b48_w2.pid
echo "log: $LOG"
```

查看日志：

```bash
tail -f "$LOG"
```

方案A独立输出目录：

```text
checkpoints/arcdp_full1_b48_w2_8x4090_ceph/
```

## 4. 读取训练状态

至少完成100个optimizer steps后执行：

```bash
STATUS=checkpoints/arcdp_full1_b48_w2_8x4090_ceph/logs/training_status.json
python - "$STATUS" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    d = json.load(f)

for key in (
    "timestamp",
    "global_step",
    "max_steps",
    "progress_pct",
    "per_device_batch_size",
    "world_size",
    "gradient_accumulation_steps",
    "effective_global_batch",
    "avg_step_time",
    "last_step_time",
    "samples_per_second",
    "elapsed_human",
    "eta_human",
):
    print(f"{key}: {d.get(key)}")
print("gpu_memory:", d.get("gpu_memory"))
PY
```

这里的 `samples_per_second` 已修正为8卡全局吞吐，并正确计入梯度累积。旧版本显示的约 `1.02 samples/s` 实际漏乘了world size。

## 5. 采集120秒GPU统计

```bash
python - <<'PY'
import subprocess
import time

stats = {i: {"sm": [], "power": [], "mem": []} for i in range(8)}
query = "index,utilization.gpu,power.draw,memory.used"

for _ in range(120):
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        text=True,
    )
    for line in output.strip().splitlines():
        idx, sm, power, mem = [part.strip() for part in line.split(",")]
        record = stats[int(idx)]
        record["sm"].append(float(sm))
        record["power"].append(float(power))
        record["mem"].append(float(mem))
    time.sleep(1)

for idx, record in stats.items():
    zero = sum(value == 0 for value in record["sm"])
    print(
        f"GPU {idx}: avg_sm={sum(record['sm']) / len(record['sm']):6.2f}%, "
        f"zero={zero}/120, avg_power={sum(record['power']) / len(record['power']):6.1f}W, "
        f"max_mem={max(record['mem']):.0f}MiB"
    )
PY
```

## 6. 检查DataLoader是否仍卡在Ceph

```bash
RUN=arcdp_full1_b48_w2_8x4090_ceph

for RANK_PID in $(pgrep -f "scripts/train/base_train/train.py.*${RUN}"); do
    LOCAL_RANK=$(tr '\0' '\n' < "/proc/${RANK_PID}/environ" 2>/dev/null \
      | sed -n 's/^LOCAL_RANK=//p')
    [[ -n "$LOCAL_RANK" ]] || continue
    echo "===== rank=${LOCAL_RANK} pid=${RANK_PID} ====="
    ps -o pid,ppid,stat,psr,%cpu,%mem,wchan:32,etime --ppid "$RANK_PID"
done
```

重点检查是否反复出现：

```text
ceph_mdsc_wait_request
wait_on_page_bit_common
```

短暂出现一次不一定异常；同一个rank的全部worker长期处于 `D` 状态才说明Ceph问题仍在。

## 7. 是否需要测试方案B

方案A满足以下条件时，不需要测试方案B：

- 八张GPU均持续参与计算，没有单卡长期接近0%；
- 120秒内没有某张卡出现大量零利用率采样；
- DataLoader worker没有持续处于Ceph等待；
- 全局吞吐明显高于旧方案的约 `8.14 samples/s`；
- 平均optimizer-step时间明显低于旧方案的约 `47.17s`，且趋于稳定。

方案A仍存在rank长期掉队时，再停止A并测试B。

## 8. 启动方案B

```bash
ARCDP_DRY_RUN=1 ./train_arcdp_full1_b24_ga2_w2_pf1_8x4090_ceph.sh

LOG=logs/arcdp_full1_b24_ga2_w2_pf1_$(date +%F_%H%M%S).log
nohup setsid ./train_arcdp_full1_b24_ga2_w2_pf1_8x4090_ceph.sh \
  >"$LOG" 2>&1 < /dev/null &
echo $! | tee logs/arcdp_full1_b24_ga2_w2_pf1.pid
echo "log: $LOG"
```

方案B状态文件：

```text
checkpoints/arcdp_full1_b24_ga2_w2_pf1_8x4090_ceph/logs/training_status.json
```

方案B每个 `global_step` 包含两个micro-batches，但有效全局batch仍为384，因此可以直接比较两套方案的 `avg_step_time` 和 `samples_per_second`。

## 9. 回传给Codex的指标

请粘贴以下内容：

1. `training_status.json` 上述字段；
2. 120秒GPU统计的8行输出；
3. DataLoader worker状态；
4. 日志中最近的 `[ETA]` 和 `[Step ...]` 行；
5. 如有异常，粘贴完整异常栈。

有了这些指标即可判断方案A是否已经解决长尾阻塞，还是需要继续测试方案B。
