# Bridge-DP 实现计划

> 基于 NavDP 代码库，分支：`bridgedp-dev`  
> 最后更新：2026-04-30  
> **核心原则：不修改任何 NavDP 原始代码，全部另起独立文件**

---

## 目录

1. [总体改动概览](#1-总体改动概览)
2. [文件清单与对应关系](#2-文件清单与对应关系)
3. [数据集：bridgedp_lerobot_dataset.py](#3-数据集bridgedp_lerobot_datasetpy)
4. [模型策略：bridgedp_policy.py](#4-模型策略bridgedp_policypy)
5. [桥调度器：bridge_scheduler.py](#5-桥调度器bridge_schedulerpy)
6. [训练器：bridgedp_trainer.py](#6-训练器bridgedp_trainerpy)
7. [框架注册点（不修改原文件的前提下）](#7-框架注册点)
8. [训练配置：configs/bridgedp.py](#8-训练配置)
9. [训练入口：train.py 修改](#9-训练入口)
10. [v1 实现检查清单](#10-v1-实现检查清单)

---

## 1. 总体改动概览

Bridge-DP 在 InternNav 框架中作为与 NavDP **同级的独立模型**存在。

**核心设计原则**：所有 Bridge-DP 类使用**与 NavDP 相同的基类**，而非继承 NavDP 的类：

| 组件 | NavDP 的类 | 共同基类 | Bridge-DP 的类 |
|------|-----------|---------|---------------|
| 数据集 | [`NavDP_Base_Datset`](internnav/dataset/navdp_lerobot_dataset.py:54) | `torch.utils.data.Dataset` | `BridgeDP_Base_Dataset` |
| 模型策略 | [`NavDPNet`](internnav/model/basemodel/navdp/navdp_policy.py:34) | `transformers.PreTrainedModel` | `BridgeDPNet` |
| 训练器 | [`NavDPTrainer`](internnav/trainer/navdp_trainer.py:11) | [`BaseTrainer`](internnav/trainer/base.py:32)（→ `transformers.Trainer`） | `BridgeDPTrainer` |
| 模型配置 | [`NavDPModelConfig`](internnav/model/basemodel/navdp/navdp_policy.py:19) | `transformers.PretrainedConfig` | `BridgeDPModelConfig` |

Bridge-DP 复用 NavDP 的视觉编码器（[`RGBDBackbone`](internnav/model/encoder/navdp_backbone.py:205)、[`ImageGoalBackbone`](internnav/model/encoder/navdp_backbone.py:316)、[`PixelGoalBackbone`](internnav/model/encoder/navdp_backbone.py:379)），这些编码器位于公共目录 `internnav/model/encoder/` 下，属于框架共享组件。

Bridge-DP 拥有独立的：

- 数据集类（独立继承 `Dataset`，绝对坐标 + 先验轨迹）
- 模型类（独立继承 `PreTrainedModel`，布朗桥 SDE + PriorEncoder + VisualGate）
- 训练器类（独立继承 `BaseTrainer`，桥损失计算）
- 配置文件
- 训练入口分支

---

## 2. 文件清单与对应关系

| NavDP 原文件（不修改） | Bridge-DP 新文件 | 说明 |
|----------------------|-----------------|------|
| [`internnav/dataset/navdp_lerobot_dataset.py`](internnav/dataset/navdp_lerobot_dataset.py) | `internnav/dataset/bridgedp_lerobot_dataset.py` | 独立继承 `Dataset`，仿照 NavDP 实现全部数据加载逻辑 |
| [`internnav/model/basemodel/navdp/navdp_policy.py`](internnav/model/basemodel/navdp/navdp_policy.py) | `internnav/model/basemodel/bridgedp/bridgedp_policy.py` | 独立继承 `PreTrainedModel`，仿照 NavDP 实现，复用共享 backbone |
| — | `internnav/model/basemodel/bridgedp/__init__.py` | 包初始化 |
| — | `internnav/model/basemodel/bridgedp/bridge_scheduler.py` | 方向自适应布朗桥调度器（替换 NavDP 的 DDPMScheduler） |
| — | `internnav/model/basemodel/bridgedp/prior_encoder.py` | PriorEncoder + VisualGate（Bridge-DP 独有模块） |
| [`internnav/trainer/navdp_trainer.py`](internnav/trainer/navdp_trainer.py) | `internnav/trainer/bridgedp_trainer.py` | 独立继承 `BaseTrainer`，仿照 NavDP 实现全部训练逻辑 |
| [`internnav/configs/model/navdp.py`](internnav/configs/model/navdp.py) | `internnav/configs/model/bridgedp.py` | 独立配置 |
| [`scripts/train/base_train/configs/navdp.py`](scripts/train/base_train/configs/navdp.py) | `scripts/train/base_train/configs/bridgedp.py` | 训练超参配置 |

**需要追加内容（而非修改原有逻辑）的框架文件**：

| 文件 | 追加内容 | 说明 |
|------|---------|------|
| [`internnav/model/__init__.py`](internnav/model/__init__.py) | 新增 `BridgeDP_Policy` 分支 | 模型注册 |
| [`internnav/configs/model/__init__.py`](internnav/configs/model/__init__.py) | 新增 `bridgedp_cfg` 导入 | 配置注册 |
| [`internnav/trainer/__init__.py`](internnav/trainer/__init__.py) | 新增 `BridgeDPTrainer` 导入 | 训练器注册 |
| [`scripts/train/base_train/configs/__init__.py`](scripts/train/base_train/configs/__init__.py) | 新增 `bridgedp_exp_cfg` 导入 | 训练配置注册 |
| [`scripts/train/base_train/train.py`](scripts/train/base_train/train.py) | 新增 `bridgedp` 分支 | 训练入口 |

---

## 3. 数据集：bridgedp_lerobot_dataset.py

**文件**：`internnav/dataset/bridgedp_lerobot_dataset.py`

**基类**：`torch.utils.data.Dataset`（与 [`NavDP_Base_Datset`](internnav/dataset/navdp_lerobot_dataset.py:54) 使用相同基类）

**设计原则**：不继承 `NavDP_Base_Datset`，而是独立继承 `Dataset`，仿照 NavDP 的实现思路从零实现全部数据加载逻辑。这样做的好处：
1. Bridge-DP 与 NavDP 完全解耦，任何一方的修改不会影响另一方
2. Bridge-DP 可以自由调整数据处理流程（例如动作空间、先验轨迹），不受父类约束
3. 代码结构清晰，方便独立测试和调试

### 3.1 需要独立实现的方法（仿照 NavDP）

以下方法参照 [`NavDP_Base_Datset`](internnav/dataset/navdp_lerobot_dataset.py:54) 的实现思路，在 `BridgeDP_Base_Dataset` 中独立实现：

| 方法 | NavDP 参照行 | Bridge-DP 实现说明 |
|------|------------|-------------------|
| `__init__()` | [L77–183](internnav/dataset/navdp_lerobot_dataset.py:77) | 相同的目录扫描、索引构建逻辑；新增 `sigma_base` 参数 |
| `__len__()` | [L185–186](internnav/dataset/navdp_lerobot_dataset.py:185) | 相同 |
| `load_image()` | [L189–204](internnav/dataset/navdp_lerobot_dataset.py:189) | 相同的 RGB 图像读取与预处理 |
| `load_depth()` | [L206–221](internnav/dataset/navdp_lerobot_dataset.py:206) | 相同的深度图读取与预处理 |
| `load_pointcloud()` | [L223–233](internnav/dataset/navdp_lerobot_dataset.py:223) | 相同的点云读取 |
| `process_image()` | [L235–261](internnav/dataset/navdp_lerobot_dataset.py:235) | 相同的图像增强（resize + normalize） |
| `process_depth()` | [L263–287](internnav/dataset/navdp_lerobot_dataset.py:263) | 相同的深度图增强 |
| `process_data_parquet()` | [L289–309](internnav/dataset/navdp_lerobot_dataset.py:289) | 相同的 Parquet 轨迹数据解析 |
| `process_obstacle_points()` | [L311–332](internnav/dataset/navdp_lerobot_dataset.py:311) | 相同的障碍点处理 |
| `process_memory()` | [L334–355](internnav/dataset/navdp_lerobot_dataset.py:334) | 相同的历史帧记忆构建 |
| `process_pixel_goal()` | [L357–412](internnav/dataset/navdp_lerobot_dataset.py:357) | 相同的像素目标投影 |
| `relative_pose()` | [L414–448](internnav/dataset/navdp_lerobot_dataset.py:414) | 相同的相对位姿计算 |
| `absolute_pose()` | [L450–484](internnav/dataset/navdp_lerobot_dataset.py:450) | 相同的绝对位姿计算 |
| `xyz_to_xyt()` | [L486–505](internnav/dataset/navdp_lerobot_dataset.py:486) | 相同的坐标变换 |
| `process_actions()` | [L507–590](internnav/dataset/navdp_lerobot_dataset.py:507) | 相同的轨迹增强（旋转+样条插值） |
| `rank_steps()` | [L592–633](internnav/dataset/navdp_lerobot_dataset.py:592) | 相同的障碍密度采样 |
| `__getitem__()` | [L635–809](internnav/dataset/navdp_lerobot_dataset.py:635) | **核心差异点**，见下表 |

### 3.2 `__getitem__` 与 NavDP 的差异

| 功能 | NavDP `__getitem__` | BridgeDP `__getitem__` |
|------|-------------------|----------------------|
| 动作空间 | `(pred_actions[1:] - pred_actions[:-1]) * 4.0`（[L760](internnav/dataset/navdp_lerobot_dataset.py:760)） | 直接使用 `pred_actions`（绝对坐标 xyt） |
| 先验轨迹 | 无 | 生成 `prior_traj`（含 30% 对抗训练） |
| 目标方位角 | 无 | 计算 `theta_g = atan2(g_y, g_x)` |
| 返回字段 | 10 个字段（[L798–809](internnav/dataset/navdp_lerobot_dataset.py:798)） | 12 个字段（+prior_traj, theta_g） |

### 3.3 关键代码结构

```python
from torch.utils.data import Dataset  # 使用 NavDP 的基类，而非 NavDP 的类

class BridgeDP_Base_Dataset(Dataset):
    """Bridge-DP 数据集。

    独立继承 Dataset，仿照 NavDP_Base_Datset 的实现思路，
    全部数据加载方法独立实现，不依赖 NavDP 任何代码。

    与 NavDP 的核心区别：
    1. 动作空间为绝对坐标 (x, y, θ)，不做差分×4
    2. 新增先验轨迹生成（含对抗训练）
    3. 新增目标方位角 θ_g 计算
    """

    def __init__(self, root_dirs, preload_path, memory_size, predict_size,
                 batch_size, image_size, scene_data_scale=1.0,
                 pixel_channel=7, action_dim=3, preload=False,
                 random_digit=False, prior_sample=False, sigma_base=1.0):
        super().__init__()
        # === 与 NavDP __init__ 相同的逻辑 ===
        # 目录扫描、索引构建、preload 机制
        # ...（仿照 NavDP_Base_Datset.__init__ 实现）

        # === Bridge-DP 新增 ===
        self.sigma_base = sigma_base  # 数据驱动固定常数

    # === 以下方法仿照 NavDP 独立实现（逻辑相同） ===
    def load_image(self, image_url): ...
    def load_depth(self, depth_url): ...
    def load_pointcloud(self, pcd_url): ...
    def process_image(self, image_path): ...
    def process_depth(self, depth_path): ...
    def process_data_parquet(self, index): ...
    def process_obstacle_points(self, index): ...
    def process_memory(self, rgb_paths, depth_paths, start_step, memory_digit=1): ...
    def process_pixel_goal(self, image_url, target_point, camera_intrinsic, camera_extrinsic): ...
    def relative_pose(self, R_base, T_base, R_world, T_world, base_extrinsic): ...
    def absolute_pose(self, R_base, T_base, R_frame, T_frame, base_extrinsic): ...
    def xyz_to_xyt(self, xyz_actions, init_vector): ...
    def process_actions(self, extrinsics, base_extrinsic, start_step, end_step, pred_digit=1): ...
    def rank_steps(self, extrinsics, obstacle_points, pred_digit=4): ...

    # === Bridge-DP 新增方法 ===
    def generate_prior_trajectory(self, pred_actions, point_goal):
        """生成先验轨迹（70%正确 + 30%对抗）。"""
        if np.random.random() < 0.3:
            # 对抗先验：随机旋转或缩放
            ...
        else:
            # 正确先验：直线插值 + 噪声
            ...

    def __getitem__(self, index):
        # === 与 NavDP 相同的数据加载流程 ===
        # process_data_parquet → process_obstacle_points → 采样 →
        # process_memory → process_actions → xyz_to_xyt →
        # critic 计算 → process_pixel_goal
        # ...

        # === Bridge-DP 差异点 ===
        # 1. 不做差分×4，直接使用绝对 xyt 坐标
        # pred_actions = pred_actions[1:]  # 去掉起点，保持 T 步
        # （不执行 NavDP 的 (pred_actions[1:] - pred_actions[:-1]) * 4.0）

        # 2. 生成先验轨迹
        prior_traj = self.generate_prior_trajectory(pred_actions, point_goal)

        # 3. 计算目标方位角
        theta_g = np.arctan2(point_goal[1], point_goal[0])

        return (
            point_goal, image_goal, pixel_goal,
            memory_images, depth_image,
            pred_actions, augment_actions,        # 绝对坐标
            pred_critic, augment_critic,
            float(pixel_flag),
            prior_traj, theta_g,                  # Bridge-DP 新增
        )
```

### 3.4 collate_fn

```python
def bridgedp_collate_fn(batch):
    collated = {
        "batch_pg": torch.stack([item[0] for item in batch]),
        "batch_ig": torch.stack([item[1] for item in batch]),
        "batch_tg": torch.stack([item[2] for item in batch]),
        "batch_rgb": torch.stack([item[3] for item in batch]),
        "batch_depth": torch.stack([item[4] for item in batch]),
        "batch_labels": torch.stack([item[5] for item in batch]),      # 绝对坐标
        "batch_augments": torch.stack([item[6] for item in batch]),    # 绝对坐标
        "batch_label_critic": torch.stack([item[7] for item in batch]),
        "batch_augment_critic": torch.stack([item[8] for item in batch]),
        # pixel_flag 在 NavDP 中未进 collate，此处同样跳过
        "batch_prior": torch.stack([item[10] for item in batch]),      # 新增
        "batch_theta_g": torch.stack([item[11] for item in batch]),    # 新增
    }
    return collated
```

---

## 4. 模型策略：bridgedp_policy.py

**文件**：`internnav/model/basemodel/bridgedp/bridgedp_policy.py`

**基类**：`transformers.PreTrainedModel` / `transformers.PretrainedConfig`（与 [`NavDPNet`](internnav/model/basemodel/navdp/navdp_policy.py:34) / [`NavDPModelConfig`](internnav/model/basemodel/navdp/navdp_policy.py:19) 使用相同基类）

### 4.1 类结构

```python
from transformers import PreTrainedModel, PretrainedConfig
from internnav.model.encoder.navdp_backbone import (  # 共享编码器，不修改
    RGBDBackbone, ImageGoalBackbone, PixelGoalBackbone,
    LearnablePositionalEncoding, SinusoidalPosEmb,
)
from .bridge_scheduler import BridgeScheduler
from .prior_encoder import PriorEncoder, VisualGate

class BridgeDPModelConfig(PretrainedConfig):
    """Bridge-DP 模型配置。独立继承 PretrainedConfig，仿照 NavDPModelConfig。"""
    model_type = 'bridgedp'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

class BridgeDPNet(PreTrainedModel):
    """Bridge-DP 模型策略。独立继承 PreTrainedModel，仿照 NavDPNet 实现思路。"""
    config_class = BridgeDPModelConfig

    def __init__(self, config: BridgeDPModelConfig):
        super().__init__(config)
        # === 复用 NavDP 的共享编码器（直接 import，不修改） ===
        self.rgbd_encoder = RGBDBackbone(...)
        self.pixel_encoder = PixelGoalBackbone(...)
        self.image_encoder = ImageGoalBackbone(...)
        self.point_encoder = nn.Linear(3, token_dim)

        # === 仿照 NavDP 实现 Transformer Decoder ===
        self.decoder = nn.TransformerDecoder(...)

        # === Bridge-DP 新增模块 ===
        self.prior_encoder = PriorEncoder(...)       # 新增
        self.visual_gate = VisualGate(...)           # 新增
        self.bridge_scheduler = BridgeScheduler(...) # 替换 NavDP 的 DDPMScheduler

        # 位置编码长度：原 (mem×16+4) → (mem×16+4+N_p)
        self.cond_pos_embed = LearnablePositionalEncoding(
            token_dim, memory_size * 16 + 4 + self.n_prior_tokens
        )

    # === 仿照 NavDPNet 实现以下方法 ===
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs): ...
    def to(self, device, *args, **kwargs): ...
    def _get_device(self): ...
```

### 4.2 与 NavDPNet 的方法对比

| 方法 | NavDPNet | BridgeDPNet |
|------|---------|-------------|
| `sample_noise()` | `DDPMScheduler.add_noise()` | `BridgeScheduler.add_noise(action, goal, theta_g)` |
| `predict_noise()` | memory = `[time, goal×3, rgbd]` | memory = `[time, goal×3, rgbd, G·prior]` |
| `predict_critic()` | 不变 | **完全不变**，不引入 prior |
| `forward()` | DDPM 前向 | 布朗桥前向 + 先验编码 |
| `predict_pointgoal_batch_action_vel()` | `torch.randn` → 10步去噪 → `cumsum/4` | `N(g, σ²_goal)` → 10步去噪 → 样条平滑 |

### 4.3 推理输出变化

```python
# NavDP 原始推理（增量→轨迹）
trajectory = torch.cumsum(naction / 4.0, dim=1)

# BridgeDP 推理（绝对坐标→样条平滑）
trajectory = smooth_trajectory_batch(x0_pred)  # (B, T, 3) 绝对坐标
```

---

## 5. 桥调度器：bridge_scheduler.py

**文件**：`internnav/model/basemodel/bridgedp/bridge_scheduler.py`

替代 NavDP 的 `DDPMScheduler`，实现方向自适应弹性布朗桥。

```python
class BridgeScheduler:
    def __init__(self, num_train_timesteps=10, sigma_base=1.0, sigma_goal=0.1):
        self.T = num_train_timesteps
        self.sigma_base = sigma_base  # 数据驱动固定常数
        self.sigma_goal = sigma_goal

    def variance(self, t_norm, theta_g):
        """方向自适应方差：σ²(t; θ_g) = σ²_base · [t(1-t)]^p + t² · σ²_goal"""
        p = 0.5 + 0.3 * torch.cos(theta_g)
        f = (t_norm * (1 - t_norm)) ** p
        return self.sigma_base ** 2 * f + t_norm ** 2 * self.sigma_goal ** 2

    def add_noise(self, x0, goal, theta_g, timesteps):
        """前向加噪：x_t = (1-t)x_0 + t·g + σ(t;θ_g)·ε"""
        ...

    def step(self, x_t, x0_pred, t):
        """DDIM 确定性反向步"""
        ...
```

---

## 6. 训练器：bridgedp_trainer.py

**文件**：`internnav/trainer/bridgedp_trainer.py`

**基类**：[`BaseTrainer`](internnav/trainer/base.py:32)（→ `transformers.Trainer`），与 [`NavDPTrainer`](internnav/trainer/navdp_trainer.py:11) 使用相同基类。

独立继承 `BaseTrainer`，仿照 `NavDPTrainer` 的实现思路，需独立实现以下方法：

| 方法 | NavDP 参照行 | Bridge-DP 实现说明 |
|------|------------|-------------------|
| `__init__()` | [L25–44](internnav/trainer/navdp_trainer.py:25) | 仿照 NavDP，加载 Bridge-DP 配置 |
| `compute_loss()` | [L46–183](internnav/trainer/navdp_trainer.py:46) | 核心差异：布朗桥损失 + 先验注入 |
| `create_optimizer()` | [L185–219](internnav/trainer/navdp_trainer.py:185) | 仿照 NavDP，分组设置学习率 |
| `create_scheduler()` | [L221–232](internnav/trainer/navdp_trainer.py:221) | 相同的 warmup scheduler |
| `create_optimizer_and_scheduler()` | [L234–244](internnav/trainer/navdp_trainer.py:234) | 相同 |
| `get_train_dataloader()` | [L246–270](internnav/trainer/navdp_trainer.py:246) | 仿照 NavDP，使用 Bridge-DP 的 collate_fn |
| `save_model()` | [L272–296](internnav/trainer/navdp_trainer.py:272) | 相同的 checkpoint 保存逻辑 |

```python
from internnav.trainer.base import BaseTrainer  # 与 NavDPTrainer 使用相同基类

class BridgeDPTrainer(BaseTrainer):
    """Bridge-DP 训练器。独立继承 BaseTrainer，仿照 NavDPTrainer 实现思路。"""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # ...仿照 NavDPTrainer.__init__ 实现

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 取出新增字段
        prior_traj = inputs["batch_prior"]
        theta_g = inputs["batch_theta_g"]

        # 前向（布朗桥加噪 + 先验注入 + 去噪预测）
        outputs = model(
            goal_point=inputs["batch_pg"],
            goal_image=inputs["batch_ig"],
            goal_pixel=inputs["batch_tg"],
            input_images=inputs["batch_rgb"],
            input_depths=inputs["batch_depth"],
            output_actions=inputs["batch_labels"],    # 绝对坐标
            augment_actions=inputs["batch_augments"],
            prior_traj=prior_traj,
            theta_g=theta_g,
        )

        # 损失结构与 NavDP 完全一致
        # loss = 0.8 * action_loss + 0.2 * critic_loss + 0.5 * aux_loss
        # action_loss = 0.5 * ng_loss + 0.5 * mg_loss

    def create_optimizer(self): ...      # 仿照 NavDPTrainer 实现
    def create_scheduler(self, optimizer, num_training_steps): ...
    def create_optimizer_and_scheduler(self, num_training_steps): ...
    def get_train_dataloader(self): ...  # 使用 bridgedp_collate_fn
    def save_model(self, output_dir, state_dict=None, **kwargs): ...
```

---

## 7. 框架注册点

以下文件需要**追加**内容（只加不改原有逻辑）：

### 7.1 模型注册 [`internnav/model/__init__.py`](internnav/model/__init__.py)

```python
# 在 get_policy() 中追加：
elif policy_name == 'BridgeDP_Policy':
    from .basemodel.bridgedp.bridgedp_policy import BridgeDPNet
    return BridgeDPNet

# 在 get_config() 中追加：
elif policy_name == 'BridgeDP_Policy':
    from .basemodel.bridgedp.bridgedp_policy import BridgeDPModelConfig
    return BridgeDPModelConfig
```

### 7.2 配置注册 [`internnav/configs/model/__init__.py`](internnav/configs/model/__init__.py)

```python
from .bridgedp import bridgedp_cfg
# 在 __all__ 中追加 'bridgedp_cfg'
```

### 7.3 训练器注册 [`internnav/trainer/__init__.py`](internnav/trainer/__init__.py)

```python
from .bridgedp_trainer import BridgeDPTrainer
```

### 7.4 训练配置注册 [`scripts/train/base_train/configs/__init__.py`](scripts/train/base_train/configs/__init__.py)

```python
from .bridgedp import bridgedp_exp_cfg
# 在 __all__ 中追加 'bridgedp_exp_cfg'
```

### 7.5 训练入口 [`scripts/train/base_train/train.py`](scripts/train/base_train/train.py)

需追加的位置（参照 NavDP 的模式）：

**导入区**（约第 23 行后）：
```python
from internnav.dataset.bridgedp_lerobot_dataset import BridgeDP_Base_Dataset, bridgedp_collate_fn
from internnav.trainer import BridgeDPTrainer
from scripts.train.base_train.configs import bridgedp_exp_cfg
```

**数据加载分支**（约第 319 行后，`elif config.model_name == "navdp":` 同级）：
```python
elif config.model_name == "bridgedp":
    train_dataset_data = BridgeDP_Base_Dataset(
        config.il.root_dir,
        config.il.dataset_navdp,      # 复用 NavDP 的数据索引
        config.il.memory_size,
        config.il.predict_size,
        config.il.batch_size,
        config.il.image_size,
        config.il.scene_scale,
        pixel_channel=config.il.pixel_channel,
        preload=config.il.preload,
        random_digit=config.il.random_digit,
        prior_sample=config.il.prior_sample,
    )
```

**Trainer 选择分支**（约第 479 行后）：
```python
elif config.model_name == 'bridgedp':
    policy_trainer = BridgeDPTrainer
    train_dataset = train_dataset_data
    collate_fn = bridgedp_collate_fn
```

**supported_cfg 字典**（约第 574 行后）：
```python
'bridgedp': [bridgedp_exp_cfg, "BridgeDP_Policy"],
```

**分布式初始化条件**（约第 242 行）：
```python
if config.model_name in ["navdp", "bridgedp", "flownav_static", ...]:
```

---

## 8. 训练配置

### 8.1 模型配置 `internnav/configs/model/bridgedp.py`

```python
from .base_encoders import ModelCfg

bridgedp_cfg = ModelCfg(
    policy_name='BridgeDP_Policy',
    state_encoder=None,
)
```

### 8.2 训练超参配置 `scripts/train/base_train/configs/bridgedp.py`

```python
from internnav.configs.model.bridgedp import bridgedp_cfg
from internnav.configs.trainer.eval import EvalCfg
from internnav.configs.trainer.exp import ExpCfg
from internnav.configs.trainer.il import FilterFailure, IlCfg, Loss

bridgedp_exp_cfg = ExpCfg(
    name='bridgedp_train',
    model_name='bridgedp',
    torch_gpu_id=0,
    torch_gpu_ids=[0],
    output_dir='checkpoints/%s/ckpts',
    tensorboard_dir='checkpoints/%s/tensorboard',
    checkpoint_folder='checkpoints/%s/ckpts',
    log_dir='checkpoints/%s/logs',
    local_rank=0,
    seed=0,
    eval=EvalCfg(
        use_ckpt_config=False, save_results=True, split=['val_seen'],
        ckpt_to_load='', max_steps=195, sample=False,
        success_distance=3.0, start_eval_epoch=-1, step_interval=50,
    ),
    il=IlCfg(
        epochs=1000,
        batch_size=32,
        lr=1e-4,
        num_workers=8,
        weight_decay=1e-4,
        warmup_ratio=0.05,
        use_iw=True,
        inflection_weight_coef=3.2,
        save_interval_epochs=5,
        save_filter_frozen_weights=False,
        load_from_ckpt=False,
        ckpt_to_load='',
        lmdb_map_size=1e12,
        dataset_r2r_root_dir='data/vln_pe/raw_data/r2r',
        lmdb_features_dir='r2r',
        lerobot_features_dir='data/vln_pe/traj_data/r2r',
        camera_name='pano_camera_0',
        report_to='tensorboard',
        dataset_navdp='data/datasets/navdp_dataset_lerobot.json',   # 复用 NavDP 索引
        root_dir='data/datasets/InternData-N1/vln_n1/traj_data',     # 复用 NavDP 数据
        image_size=224,
        scene_scale=1.0,
        preload=False,
        random_digit=False,
        prior_sample=False,
        memory_size=8,
        predict_size=24,
        pixel_channel=4,
        temporal_depth=16,
        heads=8,
        token_dim=384,
        channels=3,
        dropout=0.1,
        scratch=False,
        finetune=False,
        ddp_find_unused_parameters=True,
        filter_failure=FilterFailure(use=True, min_rgb_nums=15),
        loss=Loss(alpha=0.0001, dist_scale=1),
    ),
    model=bridgedp_cfg,
)
```

---

## 9. 训练入口

训练命令与 NavDP 完全一致，只需将 `--model-name` 改为 `bridgedp`：

```bash
# 单卡训练
python scripts/train/base_train/train.py --model-name bridgedp --name bridgedp_v1

# 多卡训练
torchrun --nproc_per_node=4 scripts/train/base_train/train.py \
    --model-name bridgedp --name bridgedp_v1_4gpu
```

---

## 10. v1 实现检查清单

### 新建文件（7 个）

| # | 文件 | 状态 |
|---|------|------|
| 1 | `internnav/dataset/bridgedp_lerobot_dataset.py` | ⬜ |
| 2 | `internnav/model/basemodel/bridgedp/__init__.py` | ⬜ |
| 3 | `internnav/model/basemodel/bridgedp/bridgedp_policy.py` | ⬜ |
| 4 | `internnav/model/basemodel/bridgedp/bridge_scheduler.py` | ⬜ |
| 5 | `internnav/model/basemodel/bridgedp/prior_encoder.py` | ⬜ |
| 6 | `internnav/trainer/bridgedp_trainer.py` | ⬜ |
| 7 | `internnav/configs/model/bridgedp.py` | ⬜ |

### 训练配置（1 个新建）

| # | 文件 | 状态 |
|---|------|------|
| 8 | `scripts/train/base_train/configs/bridgedp.py` | ⬜ |

### 追加注册（5 个文件追加内容，不修改原有逻辑）

| # | 文件 | 追加内容 | 状态 |
|---|------|---------|------|
| 9 | [`internnav/model/__init__.py`](internnav/model/__init__.py) | `BridgeDP_Policy` 分支 | ⬜ |
| 10 | [`internnav/configs/model/__init__.py`](internnav/configs/model/__init__.py) | `bridgedp_cfg` 导入 | ⬜ |
| 11 | [`internnav/trainer/__init__.py`](internnav/trainer/__init__.py) | `BridgeDPTrainer` 导入 | ⬜ |
| 12 | [`scripts/train/base_train/configs/__init__.py`](scripts/train/base_train/configs/__init__.py) | `bridgedp_exp_cfg` 导入 | ⬜ |
| 13 | [`scripts/train/base_train/train.py`](scripts/train/base_train/train.py) | `bridgedp` 数据/训练分支 | ⬜ |

### 离线脚本（1 个）

| # | 文件 | 状态 |
|---|------|------|
| 14 | `scripts/train/base_train/compute_sigma_base.py` | ⬜ |
