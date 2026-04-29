# Bridge-DP 实现计划

> 基于 NavDP 代码库，分支：`bridgedp-dev`  
> 最后更新：2026-04-29

---

## 目录

1. [总体改动概览](#1-总体改动概览)
2. [数据集改动](#2-数据集改动)
3. [模型架构改动](#3-模型架构改动)
4. [训练器改动](#4-训练器改动)
5. [推理流程改动](#5-推理流程改动)
6. [配置文件](#6-配置文件)
7. [v1 实现检查清单](#7-v1-实现检查清单)

---

## 1. 总体改动概览

Bridge-DP 在 NavDP 基础上做以下改动：

| 模块 | 文件 | 改动类型 | 说明 |
|------|------|---------|------|
| 数据集 | [`navdp_lerobot_dataset.py`](internnav/dataset/navdp_lerobot_dataset.py) | 修改 | 去掉差分×4，输出绝对坐标；计算 $\theta_g$ |
| 模型策略 | [`navdp_policy.py`](internnav/model/basemodel/navdp/navdp_policy.py) | 修改 | 替换 DDPMScheduler → BridgeScheduler；增加 PriorEncoder + 门控 |
| 训练器 | [`navdp_trainer.py`](internnav/trainer/navdp_trainer.py) | 轻微修改 | 传入 prior 轨迹；损失结构不变 |
| 新增文件 | `internnav/model/basemodel/bridgedp/bridge_scheduler.py` | 新建 | 方向自适应布朗桥采样/去噪逻辑 |
| 新增文件 | `internnav/model/basemodel/bridgedp/bridgedp_policy.py` | 新建 | BridgeDPNet 主模型 |
| 配置 | `internnav/configs/model/bridgedp.py` | 新建 | Bridge-DP 配置 |

**v1 目标**：PointGoal + NoGoal，先验注入，方向自适应方差，样条后处理。

---

## 2. 数据集改动

### 2.1 去掉差分×4，改为绝对坐标

**原始代码**（[`navdp_lerobot_dataset.py:760`](internnav/dataset/navdp_lerobot_dataset.py:760)）：

```python
pred_actions = (pred_actions[1:] - pred_actions[:-1]) * 4.0
augment_actions = (augment_actions[1:] - augment_actions[:-1]) * 4.0
```

**Bridge-DP 改动**：注释掉差分，直接使用绝对坐标（`xyt_actions` 已经是绝对坐标）：

```python
# Bridge-DP: 使用绝对坐标，不做差分
# pred_actions = (pred_actions[1:] - pred_actions[:-1]) * 4.0
# augment_actions = (augment_actions[1:] - augment_actions[:-1]) * 4.0
```

同时 `action_indexes` 的采样逻辑保持不变，但 `pred_actions` 现在是 `(predict_size, 3)` 的绝对坐标序列。

### 2.2 新增 $\theta_g$ 计算

在 `__getitem__` 中，`point_goal` 已经是 `target_xyt_actions[-1]`（[`navdp_lerobot_dataset.py:731`](internnav/dataset/navdp_lerobot_dataset.py:731)），其 `(x, y)` 分量即为目标在机体坐标系中的位置。

```python
# 计算目标方位角（用于方向自适应方差）
goal_theta_g = float(np.arctan2(point_goal[1].item(), point_goal[0].item()))
```

返回值中新增 `goal_theta_g`，并在 `navdp_collate_fn` 中对应添加 `"batch_theta_g"` 字段。

### 2.3 先验轨迹的处理

训练时先验轨迹来自**数据增强**（模拟上一帧预测）：

```python
# 以 70% 概率使用真实轨迹作为先验（加噪模拟跟踪误差）
# 以 30% 概率使用随机轨迹（对抗训练）
if np.random.rand() < 0.7:
    prior_traj = pred_actions + np.random.randn(*pred_actions.shape) * 0.05
else:
    prior_traj = np.random.randn(*pred_actions.shape) * 0.3

# 指数衰减偏移补偿
delta = prior_traj[0]  # 起点偏差
lambda_decay = 3.0
decay = np.exp(-lambda_decay * np.arange(len(prior_traj)) / len(prior_traj))
prior_traj = prior_traj - delta[None, :] * decay[:, None]
```

---

## 3. 模型架构改动

### 3.1 BridgeScheduler（新建）

**文件**：`internnav/model/basemodel/bridgedp/bridge_scheduler.py`

核心功能：

```python
class BridgeScheduler:
    def __init__(self, num_train_timesteps=10, sigma_goal=0.1, sigma_base=1.0):
        self.T = num_train_timesteps
        self.sigma_goal = sigma_goal
        self.sigma_base = sigma_base

    def variance(self, t_norm: float, theta_g: float) -> float:
        """t_norm ∈ [0,1]，theta_g 为目标方位角（弧度）"""
        p = 0.5 + 0.3 * np.cos(theta_g)
        f = (t_norm * (1 - t_norm)) ** p
        return self.sigma_base ** 2 * f + t_norm ** 2 * self.sigma_goal ** 2

    def add_noise(self, x0: Tensor, goal: Tensor, theta_g: Tensor,
                  timesteps: Tensor) -> tuple[Tensor, Tensor]:
        """前向加噪：返回 (x_t, epsilon)"""
        t_norm = timesteps.float() / self.T          # (B,)
        p = 0.5 + 0.3 * torch.cos(theta_g)          # (B,)
        f = (t_norm * (1 - t_norm)) ** p             # (B,)
        var = self.sigma_base ** 2 * f + t_norm ** 2 * self.sigma_goal ** 2
        sigma = var.sqrt()[:, None, None]            # (B,1,1)
        mean = (1 - t_norm)[:, None, None] * x0 + t_norm[:, None, None] * goal[:, None, :]
        epsilon = torch.randn_like(x0)
        x_t = mean + sigma * epsilon
        return x_t, epsilon

    def step(self, x_t: Tensor, x0_pred: Tensor, t: int) -> Tensor:
        """DDIM 确定性反向步：x_{t-1}"""
        t_norm = t / self.T
        t_prev_norm = (t - 1) / self.T
        if t_prev_norm <= 0:
            return x0_pred
        x_prev = (t_prev_norm / t_norm) * x_t + (1 - t_prev_norm / t_norm) * x0_pred
        return x_prev
```

### 3.2 PriorEncoder（新增模块）

在 [`navdp_backbone.py`](internnav/model/encoder/navdp_backbone.py) 中新增：

```python
class PriorEncoder(nn.Module):
    """将先验轨迹 (T, 3) 编码为 N_p 个 token。"""
    def __init__(self, traj_len=24, token_dim=512, n_tokens=4):
        super().__init__()
        self.input_proj = nn.Linear(3, token_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=8, dim_feedforward=2*token_dim,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.compress = nn.Linear(traj_len, n_tokens)  # 时间维度压缩

    def forward(self, prior_traj: Tensor) -> Tensor:
        """prior_traj: (B, T, 3) → (B, N_p, token_dim)"""
        x = self.input_proj(prior_traj)          # (B, T, D)
        x = self.encoder(x)                      # (B, T, D)
        x = x.transpose(1, 2)                    # (B, D, T)
        x = self.compress(x)                     # (B, D, N_p)
        return x.transpose(1, 2)                 # (B, N_p, D)
```

### 3.3 视觉门控模块

```python
class VisualGate(nn.Module):
    """根据视觉特征计算先验可信度门控系数 G ∈ (0,1)。"""
    def __init__(self, token_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim // 4),
            nn.GELU(),
            nn.Linear(token_dim // 4, 1),
        )

    def forward(self, rgbd_tokens: Tensor) -> Tensor:
        """rgbd_tokens: (B, N, D) → G: (B, 1, 1)"""
        h_vis = rgbd_tokens.mean(dim=1)          # (B, D) 全局平均池化
        return torch.sigmoid(self.mlp(h_vis)).unsqueeze(-1)  # (B, 1, 1)
```

### 3.4 BridgeDPNet 主模型（新建）

**文件**：`internnav/model/basemodel/bridgedp/bridgedp_policy.py`

与 [`NavDPNet`](internnav/model/basemodel/navdp/navdp_policy.py:34) 的差异：

| 组件 | NavDPNet | BridgeDPNet |
|------|---------|-------------|
| 噪声调度器 | `DDPMScheduler` | `BridgeScheduler` |
| `sample_noise()` | `torch.randn` + DDPM | 布朗桥前向加噪（含 $\theta_g$） |
| memory 序列 | `[time, goal×3, rgbd]` | `[time, goal×3, rgbd, G·prior_tokens]` |
| `cond_pos_embed` | 长度 `mem×16+4` | 长度 `mem×16+4+N_p` |
| 新增模块 | — | `PriorEncoder`, `VisualGate` |
| 输出 | 增量×4 | 绝对坐标轨迹 |

**`predict_noise` 改动**（memory 拼接）：

```python
def predict_noise(self, last_actions, timestep, goal_embed, rgbd_embed, prior_embed):
    action_embeds = self.input_embed(last_actions)
    time_embeds = self.time_emb(timestep).unsqueeze(1)
    G = self.visual_gate(rgbd_embed)              # (B, 1, 1)
    gated_prior = G * prior_embed                 # (B, N_p, D)
    cond = torch.cat([time_embeds, goal_embed, goal_embed, goal_embed,
                      rgbd_embed, gated_prior], dim=1)
    cond = cond + self.cond_pos_embed(cond)
    input_embedding = action_embeds + self.out_pos_embed(action_embeds)
    output = self.decoder(tgt=input_embedding, memory=cond,
                          tgt_mask=self.tgt_mask)
    return self.action_head(self.layernorm(output))
```

**`sample_noise` 改动**（布朗桥前向加噪）：

```python
def sample_noise(self, action, goal, theta_g):
    timesteps = torch.randint(0, self.bridge_scheduler.T,
                              (action.shape[0],), device=action.device).long()
    time_embeds = self.time_emb(timesteps).unsqueeze(1)
    noisy_action, noise = self.bridge_scheduler.add_noise(action, goal, theta_g, timesteps)
    noisy_action_embed = self.input_embed(noisy_action)
    return noise, time_embeds, noisy_action_embed
```

---

## 4. 训练器改动

[`navdp_trainer.py`](internnav/trainer/navdp_trainer.py) 中 `compute_loss` 的改动：

```python
# 新增：从 batch 中取出 prior 轨迹和 theta_g
prior_traj = inputs["batch_prior"]      # (B, T, 3)
theta_g = inputs["batch_theta_g"]       # (B,)

# 编码先验
prior_embed = model.prior_encoder(prior_traj)   # (B, N_p, D)

# 传入 sample_noise（布朗桥加噪）
goal_point = inputs["batch_pg"]         # (B, 3)
ng_noise, ng_time_embed, ng_noisy_embed = model.sample_noise(
    tensor_label_actions, goal_point, theta_g
)
```

损失结构**不变**：

```python
loss = 0.8 * action_loss + 0.2 * critic_loss + 0.5 * aux_loss
action_loss = 0.5 * ng_loss + 0.5 * mg_loss
```

---

## 5. 推理流程改动

[`bridgedp_policy.py`] 中的推理函数（对应 NavDP 的 [`predict_pointgoal_batch_action_vel`](internnav/model/basemodel/navdp/navdp_policy.py:302)）：

```python
def predict_pointgoal_batch_action_vel(self, goal_point, input_images, input_depths,
                                        prior_traj=None, sample_num=32):
    with torch.no_grad():
        goal = torch.as_tensor(goal_point, dtype=torch.float32, device=self._device)
        theta_g = torch.atan2(goal[:, 1], goal[:, 0])

        rgbd_embed = self.rgbd_encoder(input_images, input_depths)
        goal_embed = self.point_encoder(goal).unsqueeze(1)

        # 先验编码（无先验时用零向量）
        if prior_traj is not None:
            prior_embed = self.prior_encoder(prior_traj)
        else:
            prior_embed = torch.zeros(
                rgbd_embed.shape[0], self.n_prior_tokens, self.token_dim,
                device=self._device
            )

        # 推理起点：从目标附近采样
        x_t = goal.unsqueeze(1).expand(-1, self.predict_size, -1) + \
              torch.randn(sample_num, self.predict_size, 3, device=self._device) * \
              self.bridge_scheduler.sigma_goal

        # 10步反向去噪
        self.bridge_scheduler.set_timesteps(self.bridge_scheduler.T)
        for k in reversed(range(1, self.bridge_scheduler.T + 1)):
            x0_pred = self.predict_noise(x_t, torch.tensor([k], device=self._device),
                                         goal_embed, rgbd_embed, prior_embed)
            x_t = self.bridge_scheduler.step(x_t, x0_pred, k)

        # 三次样条后处理
        trajectories = smooth_trajectory_batch(x0_pred)  # (B, T, 3)

        # Critic 排序
        critic_values = self.predict_critic(trajectories, rgbd_embed)
        positive = trajectories[(-critic_values).argsort()[:8]]
        negative = trajectories[(critic_values).argsort()[:8]]
        return negative, positive
```

**样条后处理工具函数**：

```python
def smooth_trajectory_batch(traj: Tensor) -> Tensor:
    """traj: (B, T, 3) → 平滑后的 (B, T, 3)，使用三次样条"""
    from scipy.interpolate import CubicSpline
    B, T, _ = traj.shape
    t = np.linspace(0, 1, T)
    result = traj.clone().cpu().numpy()
    for b in range(B):
        for dim in range(3):
            cs = CubicSpline(t, result[b, :, dim])
            result[b, :, dim] = cs(t)
    return torch.from_numpy(result).to(traj.device)
```

---

## 6. 配置文件

**新建** `internnav/configs/model/bridgedp.py`：

```python
from internnav.configs.model.base_encoders import ModelCfg

def get_bridgedp_config():
    return ModelCfg(
        policy_name='bridgedp_Policy',
        # 扩散参数
        num_train_timesteps=10,
        sigma_base=1.0,       # 由数据统计确定后更新
        sigma_goal=0.1,       # PointGoal 时的尾端松弛
        # 先验编码器
        n_prior_tokens=4,
        prior_encoder_layers=2,
        # 门控
        use_visual_gate=True,
        adversarial_prior_prob=0.3,
        prior_noise_std=0.05,
        # 后处理
        use_spline_smooth=True,
    )
```

---

## 7. v1 实现检查清单

| # | 任务 | 文件 | 状态 |
|---|------|------|------|
| 1 | 数据集去掉差分×4 | [`navdp_lerobot_dataset.py:760`](internnav/dataset/navdp_lerobot_dataset.py:760) | ⬜ |
| 2 | 数据集新增 `theta_g` 计算和返回 | [`navdp_lerobot_dataset.py`](internnav/dataset/navdp_lerobot_dataset.py) | ⬜ |
| 3 | 数据集新增先验轨迹生成（含对抗训练） | [`navdp_lerobot_dataset.py`](internnav/dataset/navdp_lerobot_dataset.py) | ⬜ |
| 4 | 新建 `BridgeScheduler` | `internnav/model/basemodel/bridgedp/bridge_scheduler.py` | ⬜ |
| 5 | 新建 `PriorEncoder` | [`navdp_backbone.py`](internnav/model/encoder/navdp_backbone.py) | ⬜ |
| 6 | 新建 `VisualGate` | [`navdp_backbone.py`](internnav/model/encoder/navdp_backbone.py) | ⬜ |
| 7 | 新建 `BridgeDPNet` 主模型 | `internnav/model/basemodel/bridgedp/bridgedp_policy.py` | ⬜ |
| 8 | 更新 `cond_pos_embed` 长度（+N_p） | `bridgedp_policy.py` | ⬜ |
| 9 | 训练器传入 prior + theta_g | [`navdp_trainer.py`](internnav/trainer/navdp_trainer.py) | ⬜ |
| 10 | 推理函数加入样条后处理 | `bridgedp_policy.py` | ⬜ |
| 11 | 新建配置文件 | `internnav/configs/model/bridgedp.py` | ⬜ |
| 12 | 离线统计 `sigma_base`（脚本） | `scripts/train/base_train/compute_sigma_base.py` | ⬜ |
