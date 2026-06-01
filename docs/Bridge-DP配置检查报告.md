# Bridge-DP 配置检查报告（4090 24GB 单卡）

> 检查时间: 2026-05-07  
> 硬件: NVIDIA RTX 4090 24GB  
> 配置文件: [`scripts/train/base_train/configs/bridgedp.py`](../scripts/train/base_train/configs/bridgedp.py)

---

## 1. 路径检查结果

| 配置项 | 当前值 | 状态 | 说明 |
|--------|--------|------|------|
| `root_dir` | `/media/monika/TSD302/.../vln_n1/traj_data` | ✅ 存在 | 数据集根目录正常 |
| `dataset_navdp` | `''` (空字符串) | ❌ **需修复** | 缺少预加载索引文件 |
| `dataset_r2r_root_dir` | `/media/monika/TSD302/.../vln_pe/raw_data/r2r` | ✅ 存在 | R2R 数据正常 |
| `lerobot_features_dir` | `/media/monika/TSD302/.../vln_pe/traj_data/r2r` | ✅ 存在 | LeRobot 特征正常 |
| DepthAnything 权重 | `checkpoints/depth_anything_v2_vits.pth` | ✅ 存在 (95MB) | 权重文件已下载 |

---

## 2. 关键问题与修复

### 🔴 问题 1：数据集未解压

**现状**：`traj_data/matterport3d_d435i/` 下有 68 个 `.tar.gz` 文件，未解压。

**影响**：数据集无法直接读取，训练会失败。

**修复方案**：

```bash
# 解压所有场景数据
cd /media/monika/TSD302/dataset/InternData/N1/InternData-N1-v0.1-mini/vln_n1/traj_data/matterport3d_d435i/

# 批量解压（耗时约 10-30 分钟，取决于磁盘速度）
for f in *.tar.gz; do
    echo "解压 $f ..."
    tar -xzf "$f"
done

# 验证解压结果
ls -d */ | head -5
# 预期输出：17DRP5sb8fy/ 1LXtFkjw3qL/ 1pXnuDYAj8r/ ...
```

---

### 🔴 问题 2：缺少预加载索引文件

**现状**：`dataset_navdp=''`（空字符串），数据集初始化会失败或极慢。

**修复方案**：

```bash
# 方法 1：生成索引文件（推荐）
python scripts/dataset/generate_preload_index.py \
    --root_dir /media/monika/TSD302/dataset/InternData/N1/InternData-N1-v0.1-mini/vln_n1/traj_data \
    --output /media/monika/TSD302/dataset/InternData/N1/InternData-N1-v0.1-mini/vln_n1/navdp_dataset_lerobot.json

# 方法 2：首次运行时自动扫描（慢，不推荐）
# 将 dataset_navdp 保持为空，preload=False
```

**修改配置**（`scripts/train/base_train/configs/bridgedp.py:59`）：

```python
# 修改前
dataset_navdp='',

# 修改后
dataset_navdp='/media/monika/TSD302/dataset/InternData/N1/InternData-N1-v0.1-mini/vln_n1/navdp_dataset_lerobot.json',
```

---

### 🟡 问题 3：显存可能不足

**现状**：`batch_size=32` 在 24GB 显存下可能 OOM。

**显存估算**：

| 组件 | 显存占用 |
|------|---------|
| DepthAnything ViT-S | ~100 MB |
| BridgeDPNet 模型 | ~200 MB |
| RGBD 编码 (8 frames × batch 32) | ~2 GB |
| Transformer 激活值 | ~4 GB |
| 梯度 + 优化器状态 | ~8 GB |
| 前向激活缓存 | ~6 GB |
| **总计** | **~20 GB** |

**风险**：24GB 卡理论可行，但实际运行时可能因碎片化或 PyTorch 内存管理导致 OOM。

**修复方案**：

```python
# scripts/train/base_train/configs/bridgedp.py:41
# 修改前
batch_size=32,

# 修改后（保守）
batch_size=16,  # 降低 50%，显存占用 ~12-14GB，安全

# 或（激进，需测试）
batch_size=24,  # 降低 25%，显存占用 ~16-18GB
```

**验证方法**：

```bash
# 训练前先跑一个 epoch 测试显存
torchrun --nproc_per_node=1 -m scripts.train.base_train.train \
    --name bridgedp_test --model_name bridgedp

# 观察 nvidia-smi 显存峰值
watch -n 1 nvidia-smi
```

---

### 🟢 问题 4：其他配置建议

#### 4.1 `num_workers` 调整

**现状**：`num_workers=8`

**建议**：4090 单卡训练时，CPU 瓶颈不明显，可降低 workers 减少内存占用：

```python
num_workers=4,  # 降低 CPU 内存占用
```

#### 4.2 `memory_size` 与 `predict_size`

**现状**：`memory_size=8, predict_size=24`（与 NavDP 一致）

**说明**：这是合理的默认值，无需修改。但如果显存紧张，可考虑：

```python
memory_size=5,   # 降至 5 帧历史（最小推荐值）
predict_size=16, # 降至 16 步预测（仍足够导航）
```

#### 4.3 `sigma_base` 需离线计算

**现状**：`sigma_base=1.0`（占位值）

**说明**：这是数据驱动的固定常数，需要在训练前离线计算：

```bash
# 数据解压后运行
python scripts/train/base_train/compute_sigma_base.py \
    --root_dir /media/monika/TSD302/dataset/InternData/N1/InternData-N1-v0.1-mini/vln_n1/traj_data \
    --dataset_index /media/monika/TSD302/dataset/InternData/N1/InternData-N1-v0.1-mini/vln_n1/navdp_dataset_lerobot.json \
    --num_samples 5000 \
    --output sigma_base_result.json

# 将结果填入配置
# sigma_base=0.847,  # 示例值，以实际计算结果为准
```

---

## 3. 修复后的完整配置

```python
# scripts/train/base_train/configs/bridgedp.py

bridgedp_exp_cfg = ExpCfg(
    name='bridgedp_train',
    model_name='bridgedp',
    torch_gpu_id=0,
    torch_gpu_ids=[0],
    # ... 其他配置保持不变 ...
    il=IlCfg(
        epochs=1000,
        batch_size=16,  # ← 修改：降低显存占用
        lr=1e-4,
        num_workers=4,  # ← 修改：降低 CPU 内存
        # ... 其他配置保持不变 ...
        
        # ← 修改：设置索引文件路径
        dataset_navdp='/media/monika/TSD302/dataset/InternData/N1/InternData-N1-v0.1-mini/vln_n1/navdp_dataset_lerobot.json',
        
        root_dir='/media/monika/TSD302/dataset/InternData/N1/InternData-N1-v0.1-mini/vln_n1/traj_data',
        
        # ... 其他配置保持不变 ...
        
        # ← 修改：填入离线计算的 sigma_base
        sigma_base=1.0,  # TODO: 运行 compute_sigma_base.py 后更新此值
        sigma_goal=0.1,
        n_prior_tokens=4,
    ),
    model=bridgedp_cfg,
)
```

---

## 4. 启动训练前的检查清单

- [ ] **解压数据集**（68 个 `.tar.gz` 文件）
- [ ] **生成预加载索引**（`generate_preload_index.py`）
- [ ] **修改 `dataset_navdp` 路径**（指向生成的索引文件）
- [ ] **降低 `batch_size` 至 16**（避免 OOM）
- [ ] **计算 `sigma_base`**（`compute_sigma_base.py`）
- [ ] **验证 DepthAnything 权重**（`checkpoints/depth_anything_v2_vits.pth` 已存在 ✅）
- [ ] **测试运行 1 个 epoch**（观察显存峰值）

---

## 5. 启动命令

```bash
# 单卡训练
torchrun --nproc_per_node=1 -m scripts.train.base_train.train \
    --name bridgedp_v1 --model_name bridgedp

# 监控显存
watch -n 1 nvidia-smi
```

---

## 6. 预期训练时间估算

| 项目 | 估算值 |
|------|--------|
| 数据集大小 | 68 场景 × 平均 50 轨迹 ≈ 3400 轨迹 |
| 单 epoch 时间 | 3400 / 16 (batch) ≈ 212 steps × 2s/step ≈ **7 分钟** |
| 1000 epochs | 7 分钟 × 1000 ≈ **5 天** |
| 保存间隔 | 每 5 epochs 保存一次 checkpoint |

**建议**：先训练 50 epochs（~6 小时）验证收敛趋势，再决定是否继续长训练。
