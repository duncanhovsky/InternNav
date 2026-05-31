# Bridge-DP: 弹性布朗桥扩散策略

> **Elastic Brownian Bridge Diffusion Policy for Navigation**

Bridge-DP 是基于 NavDP 架构的扩展版本，引入弹性布朗桥噪声调度器、先验轨迹编码器和方向自适应方差机制，在保持 NavDP 兼容性的同时提供更优的导航轨迹生成能力。

---

## 目录

- [1. 架构概览](#1-架构概览)
- [2. 与 NavDP 的关键差异](#2-与-navdp-的关键差异)
- [3. 文件结构](#3-文件结构)
- [4. 环境准备](#4-环境准备)
- [5. 数据集准备](#5-数据集准备)
- [6. 训练](#6-训练)
  - [6.1 单卡训练](#61-单卡训练)
  - [6.2 多卡训练](#62-多卡训练)
  - [6.3 训练配置说明](#63-训练配置说明)
- [7. 训练监控](#7-训练监控)
  - [7.1 Web 监控面板](#71-web-监控面板)
  - [7.2 TensorBoard](#72-tensorboard)
- [8. 离线工具](#8-离线工具)
  - [8.1 计算 σ_base](#81-计算-σ_base)
  - [8.2 生成预加载索引](#82-生成预加载索引)
- [9. 推理](#9-推理)
- [10. 核心算法](#10-核心算法)
- [11. 常见问题](#11-常见问题)
- [12. 参考文档](#12-参考文档)

---

## 1. 架构概览

```
输入                    编码器                    扩散策略                     输出
┌─────────┐       ┌──────────────┐        ┌────────────────┐         ┌──────────┐
│ RGB-D   │──────▶│ RGBD Backbone│──┐     │ Transformer    │         │ 绝对坐标 │
│ 图像序列 │       └──────────────┘  │     │ for Diffusion  │────────▶│ 轨迹     │
├─────────┤       ┌──────────────┐  ├────▶│ + Bridge Noise │         │ (x,y,z)  │
│ 目标点   │──────▶│ PixelGoal    │──┘     │   Scheduler    │         └──────────┘
│ (PointG) │      │ Backbone     │        └────────┬───────┘              │
├─────────┤       └──────────────┘                 │                      ▼
│ 先验轨迹 │──────▶┌──────────────┐                 │               ┌──────────┐
│ (直线)   │      │ Prior Encoder│─── Visual Gate ──┘               │ Critic   │
└─────────┘       │ + VisualGate │                                  │ 排序选择  │
                  └──────────────┘                                  └──────────┘
```

**核心创新：**
- **弹性布朗桥调度器** (`BridgeScheduler`)：用布朗桥而非标准高斯噪声，锚定起点+弹性终点
- **方向自适应指数** (`direction_adaptive_exponent`)：根据目标方向角自适应调节方差形状
- **先验轨迹编码** (`PriorEncoder + VisualGate`)：将直线先验注入扩散过程，加速收敛
- **绝对坐标动作空间**：不同于 NavDP 的增量动作，Bridge-DP 直接预测绝对坐标

---

## 2. 与 NavDP 的关键差异

| 特性 | NavDP | Bridge-DP |
|------|-------|-----------|
| 动作空间 | 增量 (Δx, Δy, Δz, Δyaw) × 4 步 | 绝对坐标 (x, y, z) × 24 步 |
| 噪声调度 | DDPM 高斯噪声 | 弹性布朗桥噪声 |
| 先验信息 | 无 | 直线先验轨迹 + 指数衰减偏移 |
| 方差特性 | 恒定方差 | 半椭圆 × 方向自适应指数 |
| 推理方式 | DDIM 10 步去噪 | DDIM 10 步 + 三次样条平滑 |
| Critic | 碰撞检测排序 | 碰撞检测排序（相同） |

---

## 3. 文件结构

```
InternNav/
├── internnav/
│   ├── model/basemodel/bridgedp/
│   │   ├── __init__.py                    # 包初始化
│   │   ├── bridgedp_policy.py             # 🔑 BridgeDPNet 模型策略
│   │   ├── bridge_scheduler.py            # 🔑 弹性布朗桥调度器
│   │   └── prior_encoder.py               # 🔑 先验编码器 + 视觉门控
│   ├── dataset/
│   │   └── bridgedp_lerobot_dataset.py    # 🔑 数据集读取
│   ├── trainer/
│   │   └── bridgedp_trainer.py            # 🔑 训练器
│   └── configs/model/
│       └── bridgedp.py                    # 模型配置
├── scripts/
│   ├── train/
│   │   ├── base_train/
│   │   │   ├── train.py                   # 训练入口
│   │   │   ├── compute_sigma_base.py      # σ_base 计算脚本
│   │   │   └── configs/
│   │   │       └── bridgedp.py            # 🔑 训练超参配置
│   │   ├── monitor_server.py              # 📊 训练监控服务器
│   │   └── templates/
│   │       └── monitor.html               # 📊 监控界面模板
│   └── dataset/
│       └── generate_preload_index.py      # 预加载索引生成
└── docs/
    ├── Bridge-DP推导.md                    # 📖 数学推导文档
    └── Bridge-DP实现.md                    # 📖 实现计划文档
```

---

## 4. 环境准备

### 4.1 Conda 环境

```bash
# 使用已有的 internnav 环境
conda activate internnav

# 确认 Python 版本
python --version  # 应为 3.10.x
```

### 4.2 依赖安装

```bash
# 核心依赖（如果尚未安装）
pip install numpy-quaternion
pip install "setuptools<81"          # pkg_resources 兼容性
pip install transformers==4.51.0     # ALL_LAYERNORM_LAYERS 兼容性
pip install flask psutil             # 训练监控界面
```

### 4.3 ROS2 路径冲突处理

如果系统安装了 ROS2 Humble，Python 路径中的 `/opt/ros/humble/...` 会与项目的 `scripts/` 包冲突。解决方法：

```bash
# 方法 1：训练命令前设置 PYTHONPATH（推荐）
PYTHONPATH="$(pwd)" python scripts/train/base_train/train.py ...

# 方法 2：使用 conda run 隔离
PYTHONPATH="$(pwd)" conda run -n internnav env PYTHONPATH="$(pwd)" python scripts/train/base_train/train.py ...
```

---

## 5. 数据集准备

### 5.1 数据目录结构

Bridge-DP 使用 LeRobot 格式的轨迹数据，支持两种目录结构：

**新格式（含 trajectory_XX 子目录）：**
```
traj_data/
└── r2r/                          # group
    └── 1pXnuDYAj8r/              # scene
        ├── trajectory_00/
        │   ├── data/
        │   │   └── chunk-000/
        │   │       ├── episode_000000.parquet
        │   │       └── path.ply
        │   ├── meta/
        │   │   └── episodes_stats.jsonl
        │   └── videos/
        │       └── chunk-000/
        │           ├── observation.images.rgb/
        │           │   ├── 000000.jpg
        │           │   ├── 000001.jpg
        │           │   └── ...
        │           └── observation.images.depth/
        │               ├── 000000.png
        │               └── ...
        ├── trajectory_01/
        │   └── ...
        └── trajectory_02/
            └── ...
```

**旧格式（无 trajectory 子目录）：**
```
traj_data/
└── r2r/
    └── scene_name/
        ├── data/chunk-000/
        ├── meta/episodes_stats.jsonl
        └── videos/chunk-000/
```

### 5.2 配置数据路径

编辑 [`scripts/train/base_train/configs/bridgedp.py`](scripts/train/base_train/configs/bridgedp.py) 中的路径：

```python
# 第 53 行 - R2R 原始数据目录
dataset_r2r_root_dir='/path/to/vln_pe/raw_data/r2r',

# 第 55 行 - LeRobot 特征数据目录
lerobot_features_dir='/path/to/vln_pe/traj_data/r2r',

# 第 60 行 - 轨迹数据根目录（训练数据扫描入口）
root_dir='/path/to/vln_n1/traj_data',
```

---

## 6. 训练

### 6.1 单卡训练

```bash
cd /path/to/InternNav

# 基础训练命令
PYTHONPATH="$(pwd)" conda run -n internnav env PYTHONPATH="$(pwd)" \
  python scripts/train/base_train/train.py \
  --name bridgedp_train \
  --model-name bridgedp

# 后台运行（推荐）
PYTHONPATH="$(pwd)" conda run -n internnav env PYTHONPATH="$(pwd)" \
  python scripts/train/base_train/train.py \
  --name bridgedp_train \
  --model-name bridgedp \
  2>&1 | tee logs/bridgedp_train_$(date +%Y%m%d_%H%M%S).log &
```

### 6.2 多卡训练

```bash
# 2 卡训练
PYTHONPATH="$(pwd)" conda run -n internnav env PYTHONPATH="$(pwd)" \
  torchrun --nproc_per_node=2 \
  scripts/train/base_train/train.py \
  --name bridgedp_train_2gpu \
  --model-name bridgedp
```

### 6.3 训练配置说明

所有训练超参定义在 [`scripts/train/base_train/configs/bridgedp.py`](scripts/train/base_train/configs/bridgedp.py)：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `epochs` | 1000 | 训练总轮数 |
| `batch_size` | 16 | 每设备 batch 大小 |
| `lr` | 1e-4 | 初始学习率 |
| `weight_decay` | 1e-4 | 权重衰减 |
| `warmup_ratio` | 0.05 | 学习率预热比例 |
| `num_workers` | 4 | DataLoader 工作进程数 |
| `image_size` | 224 | 输入图像尺寸 |
| `memory_size` | 8 | 历史帧数 |
| `predict_size` | 24 | 预测轨迹步数 |
| `sigma_base` | 1.0 | 布朗桥基础方差（建议用 `compute_sigma_base.py` 离线计算） |
| `sigma_goal` | 0.1 | 弹性尾端松弛方差 |
| `n_prior_tokens` | 4 | 先验编码器输出 token 数 |
| `save_interval_epochs` | 5 | 每 N 个 epoch 保存 checkpoint |
| `report_to` | tensorboard | 日志报告目标 |

### 6.4 输出目录

训练产物保存在 `checkpoints/bridgedp_train/`：

```
checkpoints/bridgedp_train/
├── ckpts/                    # 模型 checkpoint
│   ├── checkpoint-68/
│   ├── checkpoint-136/
│   └── ...
├── logs/
│   └── train.log             # 训练日志文件
└── tensorboard/
    └── events.out.tfevents.* # TensorBoard 事件
```

---

## 7. 训练监控

### 7.1 Web 监控面板

Bridge-DP 提供了一个实时 Web 监控界面，每 3 秒自动刷新：

**启动监控服务器：**

```bash
conda run -n internnav python scripts/train/monitor_server.py
```

**访问地址：** http://localhost:5000

**监控面板包含 7 个区域：**

| 区域 | 内容 |
|------|------|
| 🎮 GPU 状态 | GPU 利用率、显存占用、功耗、温度、风扇转速 |
| 💻 系统资源 | CPU 使用率、核心数、频率、内存占用 |
| 📈 训练进度 | 当前 Step / Epoch、最新 Loss、学习率、Checkpoint 数量 |
| 📉 Loss 曲线 | 基于 Chart.js 的实时 Loss 折线图（读取 TensorBoard 或 trainer_state） |
| ⚙️ 训练配置 | 所有超参数一览表 |
| 🔧 训练进程 | 训练进程 PID、CPU/RAM 占用、运行时间、GPU 显存 |
| 📋 训练日志 | 最后 50 行日志，错误/警告自动高亮 |

**同时启动训练和监控：**

```bash
# 终端 1：启动监控
conda run -n internnav python scripts/train/monitor_server.py &

# 终端 2：启动训练
PYTHONPATH="$(pwd)" conda run -n internnav env PYTHONPATH="$(pwd)" \
  python scripts/train/base_train/train.py \
  --name bridgedp_train --model-name bridgedp &

# 然后在浏览器中打开 http://localhost:5000
```

### 7.2 TensorBoard

除了 Web 面板，也可以使用 TensorBoard 查看训练曲线：

```bash
conda run -n internnav tensorboard --logdir checkpoints/bridgedp_train/tensorboard --port 6006
# 访问 http://localhost:6006
```

---

## 8. 离线工具

### 8.1 计算 σ_base

`σ_base` 是布朗桥调度器的关键超参数，需要根据实际数据分布离线计算：

```bash
PYTHONPATH="$(pwd)" conda run -n internnav env PYTHONPATH="$(pwd)" \
  python scripts/train/base_train/compute_sigma_base.py
```

该脚本会：
1. 扫描 `root_dir` 下所有轨迹数据
2. 计算所有轨迹的归一化动作标准差
3. 输出推荐的 `σ_base` 值

计算完成后，将结果填入 [`scripts/train/base_train/configs/bridgedp.py`](scripts/train/base_train/configs/bridgedp.py) 的 `sigma_base` 字段。

### 8.2 生成预加载索引

当数据量较大时，预加载索引可以加速数据集初始化：

```bash
PYTHONPATH="$(pwd)" conda run -n internnav env PYTHONPATH="$(pwd)" \
  python scripts/dataset/generate_preload_index.py \
  --root-dir /path/to/traj_data \
  --output /path/to/preload_index.pkl
```

生成索引后，修改配置启用预加载：

```python
# scripts/train/base_train/configs/bridgedp.py
dataset_navdp='/path/to/preload_index.pkl',  # 索引文件路径
preload=True,                                 # 启用预加载
```

---

## 9. 推理

Bridge-DP 提供两种推理模式：

### 9.1 PointGoal 推理（有目标）

```python
from internnav.model import get_policy, get_config

# 加载模型
model_class = get_policy('BridgeDP_Policy')
config_class = get_config('BridgeDP_Policy')
config = config_class(model_cfg={...})
model = model_class.from_pretrained('checkpoints/bridgedp_train/ckpts/checkpoint-XXX', config=config)
model.eval().cuda()

# 推理
trajectories, velocities = model.predict_pointgoal_batch_action_vel(
    goal_point=goal_point,      # [1, 3] 目标点坐标
    input_images=rgb_images,    # [1, memory_size, 3, 224, 224]
    input_depths=depth_images,  # [1, memory_size, 1, 224, 224]
    sample_num=32,              # 候选轨迹数量
)
```

### 9.2 NoGoal 推理（自由探索）

```python
trajectories, velocities = model.predict_nogoal_batch_action_vel(
    input_images=rgb_images,
    input_depths=depth_images,
    sample_num=32,
)
```

---

## 10. 核心算法

### 10.1 弹性布朗桥

标准布朗桥在 $t \in [0, 1]$ 上连接起点 $x_0$ 和终点 $x_1$：

$$q(x_t | x_0, x_1) = \mathcal{N}\left((1-t)x_0 + t x_1,\; t(1-t)\sigma^2 I\right)$$

Bridge-DP 将终点替换为弹性目标 $g$，方差使用半椭圆函数和方向自适应指数：

$$\sigma^2(t, \theta_g) = \sigma_{\text{base}}^2 \cdot \underbrace{\sqrt{t(1-t)}}_{\text{半椭圆}} \cdot \underbrace{t^{n(\theta_g)}}_{\text{方向自适应}}$$

方向自适应指数 $n(\theta_g)$：
$$n(\theta_g) = n_{\min} + (n_{\max} - n_{\min}) \cdot \frac{|\theta_g|}{\pi}$$

### 10.2 先验轨迹编码

将起点到目标的直线轨迹作为先验，通过 `PriorEncoder`（Transformer）编码为 token 序列，经 `VisualGate` 门控后注入扩散 Transformer 的 memory 中。

### 10.3 推理流程

1. 生成先验直线轨迹 + 指数衰减偏移补偿
2. 从弹性布朗桥采样初始噪声
3. DDIM 10 步去噪
4. Critic 网络排序候选轨迹
5. 三次样条后处理保证运动连续性
6. 提取速度指令

详细数学推导见 [`docs/Bridge-DP推导.md`](docs/Bridge-DP推导.md)。

---

## 11. 常见问题

### Q1: GPU 显存不足

RTX 4090 (24GB) 默认配置（batch_size=16）约需 ~1GB 显存。如果 OOM：
```python
# 减小 batch_size
batch_size=8,
# 或启用梯度检查点（需修改 TrainingArguments）
gradient_checkpointing=True,
```

### Q2: `ModuleNotFoundError: No module named 'quaternion'`

```bash
pip install numpy-quaternion
```

### Q3: `ModuleNotFoundError: No module named 'pkg_resources'`

```bash
pip install "setuptools<81"
```

### Q4: `ImportError: cannot import name 'ALL_LAYERNORM_LAYERS'`

```bash
pip install transformers==4.51.0
```

### Q5: `ModuleNotFoundError: No module named 'catkin_pkg'` (ROS2 冲突)

```bash
# 在训练命令前添加 PYTHONPATH
PYTHONPATH="$(pwd)" python scripts/train/base_train/train.py ...
```

### Q6: 训练时 GPU 利用率很低

这通常是正常的——数据集较小时，DataLoader 的 CPU 端数据预处理（图像加载、深度图处理）成为瓶颈。可以尝试：
- 增大 `num_workers`（如 8 或 16）
- 使用预加载索引（`preload=True`）
- 使用 `dataloader_pin_memory=True`

### Q7: 训练日志为空

确保使用最新版本的 `train.py`，单机训练现在也会写入日志文件到 `checkpoints/<name>/logs/train.log`。

---

## 12. 参考文档

| 文档 | 内容 |
|------|------|
| [`docs/Bridge-DP推导.md`](docs/Bridge-DP推导.md) | 完整数学推导（11 章节，含自检清单） |
| [`docs/Bridge-DP实现.md`](docs/Bridge-DP实现.md) | 代码实现计划（文件清单、类结构、注册点） |
| [`docs/Bridge-DP逻辑自检.md`](docs/Bridge-DP逻辑自检.md) | 导航场景逻辑自检报告 |
| [`docs/Bridge-DP配置检查报告.md`](docs/Bridge-DP配置检查报告.md) | 配置一致性检查 |

---

## License

本项目遵循 [LICENSE](../../LICENSE) 中定义的许可协议。
