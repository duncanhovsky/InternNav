# Bridge-DP 数学推导文档

> 基于 NavDP 代码库，分支：`bridgedp-dev`
> 最后更新：2026-04-29（含方向自适应方差、自由度分析、后处理平滑）

---

## 目录

1. [符号表](#1-符号表)
2. [问题设定与动作空间](#2-问题设定与动作空间)
3. [弹性布朗桥前向 SDE](#3-弹性布朗桥前向-sde)
4. [先验轨迹与偏移补偿](#4-先验轨迹与偏移补偿)
5. [视觉门控先验注入](#5-视觉门控先验注入)
6. [训练目标：条件得分匹配](#6-训练目标条件得分匹配)
7. [网络架构输入输出](#7-网络架构输入输出)
8. [反向去噪推理与后处理](#8-反向去噪推理与后处理)
9. [方向自适应方差形状](#9-方向自适应方差形状)
10. [目标形式映射](#10-目标形式映射)
11. [自检清单](#11-自检清单)

---

## 1. 符号表

| 符号 | 含义 | 维度 |
|------|------|------|
| $\mathbf{x}_0$ | 干净轨迹（绝对坐标，xyt） | $\mathbb{R}^{T \times 3}$，$T=24$ |
| $\mathbf{x}_t$ | 扩散时间步 $t$ 的含噪轨迹 | $\mathbb{R}^{T \times 3}$ |
| $\boldsymbol{\mu}$ | 先验轨迹（来自上一帧预测，完整 $T\times3$） | $\mathbb{R}^{T \times 3}$ |
| $\mathbf{g}$ | 目标端点（PointGoal 时为 $\mathbf{x}_0[T-1]$ 的近似） | $\mathbb{R}^3$ |
| $\theta_g$ | 目标在机体坐标系中的方位角 $\text{atan2}(g_y, g_x)$ | 标量 |
| $p(\theta_g)$ | 方向自适应方差指数 $= 0.5 + 0.3\cos\theta_g$ | 标量 |
| $\sigma^2_{\text{base}}$ | 基础方差幅度（数据驱动固定常数） | $\mathbb{R}^+$ |
| $\sigma^2_{\text{goal}}$ | 弹性尾端方差（控制终点松弛） | $\mathbb{R}^+$ |
| $\mathbf{h}_{\text{vis}}$ | 视觉编码器输出的全局特征 | $\mathbb{R}^{d}$ |
| $G$ | 视觉门控系数 $\in (0,1)$ | 标量 |
| $\hat{\mathbf{x}}_{0,\theta}$ | 网络预测的去噪轨迹 | $\mathbb{R}^{T \times 3}$ |
| $\boldsymbol{\epsilon}$ | 标准高斯噪声 | $\mathbb{R}^{T \times 3}$ |
| $\Delta t$ | 控制周期（秒） | 标量 |

---

## 2. 问题设定与动作空间

### 2.1 NavDP 原始动作空间（增量×4）

NavDP 在数据集中存储的是**差分增量乘以 4**：

```
# navdp_lerobot_dataset.py, line 760
pred_actions = (pred_actions[1:] - pred_actions[:-1]) * 4.0
```

推理时通过 cumsum 还原：

```
# navdp_policy.py, predict_pointgoal_batch_action_vel
trajectory = torch.cumsum(naction / 4.0, dim=1)
```

### 2.2 Bridge-DP 切换到绝对坐标（Plan A）

**动机**：布朗桥的端点约束 $\mathbf{x}_0[0] = \mathbf{p}_{\text{start}}$，$\mathbf{x}_0[T-1] \approx \mathbf{g}$ 在绝对坐标下是自然的；在增量空间中端点约束需要对所有增量求和，引入复杂的线性约束，破坏布朗桥的马尔可夫性。

**变换**：设当前帧机器人位姿为原点，轨迹点 $\mathbf{x}_0[i] \in \mathbb{R}^3$（xyt 格式）表示第 $i$ 步的**绝对位置**（相对于当前帧坐标系）。

**数据集修改**：`process_actions()` 中直接输出 `xyt_actions`（已有），不再做差分×4。

---

## 3. 弹性布朗桥前向 SDE

### 3.1 标准布朗桥回顾

标准布朗桥将扩散过程约束在两个端点之间。对于从 $\mathbf{a}$（$t=0$）到 $\mathbf{b}$（$t=1$）的桥，前向分布为：

$$
q(\mathbf{x}_t | \mathbf{x}_0, \mathbf{b}) = \mathcal{N}\!\left(\mathbf{x}_0 + t(\mathbf{b} - \mathbf{x}_0),\ t(1-t)\mathbf{I}\right)
$$

### 3.2 弹性尾端扩展

为了允许目标端点有不确定性（NoGoal 模式下 $\sigma^2_{\text{goal}} \to \infty$），引入弹性尾端：

$$
\boxed{
q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}\!\left(
  \underbrace{(1-t)\,\mathbf{x}_0 + t\,\mathbf{g}}_{\text{桥均值}},\;
  \underbrace{\sigma^2(t;\theta_g)\,\mathbf{I}}_{\text{方向自适应桥方差}}
\right)
}
$$

其中方差调度（见第 9 节详细推导）：

$$
\sigma^2(t;\theta_g) = \sigma^2_{\text{base}} \cdot \left[t(1-t)\right]^{p(\theta_g)} + t^2 \cdot \sigma^2_{\text{goal}}
$$

- $p(\theta_g) = 0.5 + 0.3\cos\theta_g$：方向自适应指数，由目标方位角决定
- $\sigma^2_{\text{base}}$：数据驱动的固定常数（训练集轨迹偏差 95 分位数），**不参与梯度反传**
- $\sigma^2_{\text{goal}}$：弹性尾端松弛方差

**退化性质**：
- $\sigma^2_{\text{goal}} = 0$，$\theta_g = 0°$：严格前向布朗桥，终点精确到达 $\mathbf{g}$
- $\sigma^2_{\text{goal}} \to \infty$：退化为标准扩散（NoGoal 模式），桥约束消失

### 3.3 采样公式

训练时，给定干净轨迹 $\mathbf{x}_0$、目标 $\mathbf{g}$ 和方位角 $\theta_g$，含噪样本为：

$$
\mathbf{x}_t = (1-t)\,\mathbf{x}_0 + t\,\mathbf{g} + \sigma(t;\theta_g)\,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

### 3.4 自由度分析

布朗桥的桥均值是起终点的线性插值，当目标在身后时（$\theta_g \approx 180°$）桥均值指向后方直线，与真实绕行轨迹偏差大。方向自适应方差通过增大 $p(\theta_g)$ 在后方场景下给予更大的方差预算：

| 目标方向 | $\theta_g$ | $p$ | 方差峰值 $\sigma(0.5)$ | 自由度 |
|---------|-----------|-----|----------------------|--------|
| 正前方 | 0° | 0.80 | $\sigma_{\text{base}} \cdot 0.25^{0.80}$ | 适中（信任桥均值） |
| 正侧方 | 90° | 0.50 | $\sigma_{\text{base}} \cdot 0.25^{0.50}$ | 中等 |
| 正后方 | 180° | 0.20 | $\sigma_{\text{base}} \cdot 0.25^{0.20}$ | 最大（允许大幅绕行） |

运动连续性由推理后的**三次样条平滑**保证（见第 8 节），不依赖增量空间。

---

## 4. 先验轨迹与偏移补偿

### 4.1 先验轨迹的来源

在连续导航中，上一帧（时间步 $k-1$）已预测了一条轨迹 $\hat{\mathbf{x}}_0^{(k-1)}$。将其作为当前帧扩散的先验均值，可以加速收敛并提供时序一致性。

### 4.2 坐标系对齐

上一帧轨迹在上一帧坐标系下定义，需变换到当前帧坐标系：

$$
\boldsymbol{\mu}^{(k)} = T_{k-1 \to k}\!\left(\hat{\mathbf{x}}_0^{(k-1)}\right)
$$

其中 $T_{k-1 \to k}$ 是相对位姿变换（由里程计/VIO 提供）。

### 4.3 指数衰减偏移补偿（新增）

**问题**：机器人实际执行轨迹时存在跟踪误差，导致先验轨迹的起点 $\boldsymbol{\mu}^{(k)}[0]$ 与当前实际位置（原点）之间存在偏差 $\boldsymbol{\delta} = \boldsymbol{\mu}^{(k)}[0] - \mathbf{0}$。

**补偿方案**：对先验轨迹施加指数衰减偏移修正，使得：
- 起点偏差被完全消除（$i=0$ 时修正量 = $-\boldsymbol{\delta}$）
- 远端点偏差逐渐消失（$i \to T$ 时修正量 $\to 0$）

$$
\boxed{
\tilde{\boldsymbol{\mu}}^{(k)}[i] = \boldsymbol{\mu}^{(k)}[i] - \boldsymbol{\delta} \cdot e^{-\lambda i / T}, \quad i = 0, 1, \ldots, T-1
}
$$

其中 $\lambda > 0$ 为衰减系数（推荐初始值 $\lambda = 3$，可调）。

**验证**：
- $i=0$：$\tilde{\boldsymbol{\mu}}^{(k)}[0] = \boldsymbol{\mu}^{(k)}[0] - \boldsymbol{\delta} = \mathbf{0}$ ✓（起点对齐当前位置）
- $i=T-1$：修正量 $= \boldsymbol{\delta} \cdot e^{-\lambda(T-1)/T} \approx \boldsymbol{\delta} \cdot e^{-\lambda}$（远端几乎不修正）✓

### 4.4 Prior Encoder 的输入

**澄清**：Prior Encoder 的输入是**完整的先验轨迹** $\tilde{\boldsymbol{\mu}} \in \mathbb{R}^{T \times 3}$，而非仅均值标量。

**原因**：
1. 完整轨迹包含形状信息（弯曲程度、速度变化），仅均值丢失这些信息
2. 网络需要感知先验轨迹的每个时间步，才能做精细的门控决策
3. 均值 $\boldsymbol{\mu}$ 在本文档中特指整条轨迹（$T \times 3$ 张量），不是标量均值

**编码方式**：

$$
\mathbf{h}_{\text{prior}} = \text{PriorEncoder}(\tilde{\boldsymbol{\mu}}) \in \mathbb{R}^{N_p \times d}
$$

其中 PriorEncoder 为轻量 Transformer（2层），将 $T$ 个轨迹点编码为 $N_p$ 个 token（$N_p \ll T$，推荐 $N_p = 4$）。

---

## 5. 视觉门控先验注入

### 5.1 动机

先验轨迹的可信度取决于当前视觉观测：
- 场景变化大（障碍物移动）→ 先验不可信，门控关闭
- 场景稳定 → 先验可信，门控开启

### 5.2 门控机制

$$
G = \sigma\!\left(\text{MLP}(\mathbf{h}_{\text{vis}})\right) \in (0, 1)
$$

其中 $\mathbf{h}_{\text{vis}}$ 为 RGBDBackbone 输出的全局视觉特征（对 memory tokens 做 mean pooling）。

### 5.3 注入方式

在 Transformer Decoder 的 memory 序列中，先验 token 通过门控加权后拼接：

$$
\text{memory} = \left[\mathbf{h}_{\text{time}},\ \mathbf{h}_{\text{goal}} \times 3,\ \mathbf{h}_{\text{rgbd}} \times N_{\text{mem}},\ G \cdot \mathbf{h}_{\text{prior}} \times N_p\right]
$$

### 5.4 对抗先验训练

训练时以 30% 概率注入**错误先验**（随机轨迹或上上帧轨迹），强制网络学会忽略不可信先验：

$$
\tilde{\boldsymbol{\mu}}_{\text{train}} = \begin{cases}
\tilde{\boldsymbol{\mu}}^{(k)} & \text{以概率 } 0.7 \\
\text{RandomTraj}() & \text{以概率 } 0.3
\end{cases}
$$

---

## 6. 训练目标：条件得分匹配

### 6.1 损失函数

采用预测 $\mathbf{x}_0$（而非预测噪声 $\boldsymbol{\epsilon}$）的参数化方式，损失为：

$$
\mathcal{L}_{\text{bridge}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\!\left[
  w(t) \cdot \left\| \hat{\mathbf{x}}_{0,\theta}(\mathbf{x}_t, t, \mathbf{g}, \tilde{\boldsymbol{\mu}}, \mathbf{o}) - \mathbf{x}_0 \right\|^2
\right]
$$

其中：
- $w(t) = 1/\sigma^2(t)$（SNR 加权，可选）
- $\mathbf{o}$ 为视觉观测（RGBD）
- $t \sim \mathcal{U}[0, 1]$（连续时间，或离散化为 $T_{\text{diff}}$ 步）

### 6.2 与 NavDP 损失的对比

NavDP 原始损失（`navdp_trainer.py`）：

```
loss = 0.8 * action_loss + 0.2 * critic_loss + 0.5 * aux_loss
action_loss = 0.5 * ng_loss + 0.5 * mg_loss  # no-goal + mixed-goal
```

Bridge-DP 损失：

```
loss = 0.8 * bridge_action_loss + 0.2 * critic_loss + 0.5 * aux_loss
bridge_action_loss = 0.5 * ng_loss + 0.5 * mg_loss  # 保持双分支结构
```

Critic 分支**不引入先验信息**（用户决策），保持与 NavDP 一致。

---

## 7. 网络架构输入输出

### 7.1 输入

| 输入 | 来源 | 处理 |
|------|------|------|
| $\mathbf{x}_t$ | 含噪轨迹（绝对坐标） | 直接输入 Transformer query |
| $t$ | 扩散时间步 | SinusoidalPosEmb → 1 token |
| $\mathbf{g}$ | 目标（PointGoal/ImageGoal/PixelGoal） | 对应 Backbone → 3 tokens |
| $\mathbf{o}$ | RGBD 观测（memory_size 帧） | RGBDBackbone → $N_{\text{mem}} \times 16$ tokens |
| $\tilde{\boldsymbol{\mu}}$ | 先验轨迹（完整 $T \times 3$） | PriorEncoder → $N_p$ tokens，门控加权 |

### 7.2 Transformer Decoder memory 序列

```
memory = [
    time_token,          # 1 token
    goal_tokens,         # 3 tokens (与 NavDP 一致)
    rgbd_tokens,         # memory_size × 16 tokens
    G * prior_tokens,    # N_p tokens (新增，门控加权)
]
```

### 7.3 输出

网络输出 $\hat{\mathbf{x}}_{0,\theta} \in \mathbb{R}^{T \times 3}$（绝对坐标轨迹）。

---

## 8. 反向去噪推理与后处理

### 8.1 DDIM 风格确定性采样

给定 $\mathbf{x}_t$，利用网络预测 $\hat{\mathbf{x}}_0$，按布朗桥后验均值更新：

$$
\mathbf{x}_{t-\Delta t} = \frac{(t - \Delta t)}{t}\,\mathbf{x}_t + \frac{\Delta t}{t}\,\hat{\mathbf{x}}_{0,\theta} + \sigma_{\text{reverse}}(t, \Delta t)\,\boldsymbol{\epsilon}'
$$

其中 $\sigma_{\text{reverse}}$ 在确定性采样时为 0。

### 8.2 推理步骤（10步，与 NavDP 一致）

```
x_T ~ N(g, σ²_goal · I)          # 从目标附近采样初始噪声
for t in [T_diff, ..., 1]:
    x0_pred = network(x_t, t, goal, prior, obs)
    x_{t-1} = bridge_reverse_step(x_t, x0_pred, t)
return x0_pred                    # 绝对坐标轨迹（T×3）
```

### 8.3 三次样条后处理（运动连续性保证）

扩散模型输出的是离散轨迹点序列，相邻点之间无曲率连续性保证。对输出轨迹做三次样条平滑（与 [`process_actions()`](internnav/dataset/navdp_lerobot_dataset.py:507) 中的 `CubicSpline` 逻辑一致）：

```python
from scipy.interpolate import CubicSpline
import numpy as np

def smooth_trajectory(traj: np.ndarray) -> np.ndarray:
    """traj: (T, 3) 绝对坐标轨迹，返回平滑后的 (T, 3)"""
    t = np.linspace(0, 1, traj.shape[0])
    cs_x = CubicSpline(t, traj[:, 0])
    cs_y = CubicSpline(t, traj[:, 1])
    cs_theta = CubicSpline(t, traj[:, 2])
    t_out = np.linspace(0, 1, traj.shape[0])
    return np.stack([cs_x(t_out), cs_y(t_out), cs_theta(t_out)], axis=-1)
```

**保证**：C² 连续（二阶导连续 → 曲率连续），旋转在移动中连续完成，不产生"走一步转一次"的割裂动作。

### 8.4 速度指令提取

平滑后的轨迹转换为速度指令：

$$
v_{\text{linear},i} = \frac{\|\Delta\mathbf{xy}_i\|}{\Delta t}, \quad
v_{\text{angular},i} = \frac{\Delta\theta_i}{\Delta t}, \quad i = 0, \ldots, T-2
$$

其中 $\Delta\mathbf{xy}_i = \hat{\mathbf{x}}_0[i+1, 0:2] - \hat{\mathbf{x}}_0[i, 0:2]$，$\Delta\theta_i = \hat{\mathbf{x}}_0[i+1, 2] - \hat{\mathbf{x}}_0[i, 2]$。

$v_{\text{linear}}$ 始终为非负值（机器人始终前进），旋转通过 $v_{\text{angular}}$ 完成，包括目标在身后时的 180° 转身。

---

## 9. 方向自适应方差形状

### 9.1 基函数：半椭圆

采用半椭圆 $[t(1-t)]^p$ 作为基函数，而非标准抛物线 $t(1-t)$。

**优势**：边界处斜率为无穷大（$\frac{d}{dt}[t(1-t)]^p\big|_{t=0^+} = +\infty$，$p < 1$ 时），方差从端点**陡然升起**，机器人在起点处立即获得探索自由度，无需等待方差缓慢爬升。

对比（$p=0.5$ 时）：

| $t$ | 抛物线 $t(1-t)$ | 半椭圆 $\sqrt{t(1-t)}$ |
|-----|----------------|----------------------|
| 0.1 | 0.09 | 0.30 |
| 0.3 | 0.21 | 0.46 |
| 0.5 | 0.25 | 0.50 |

### 9.2 方向自适应指数

$$
\boxed{
p(\theta_g) = 0.5 + 0.3\cos\theta_g, \quad \theta_g = \text{atan2}(g_y, g_x)
}
$$

| 方向 | $\theta_g$ | $p$ | 峰值位置 $t^*$ | 语义 |
|------|-----------|-----|--------------|------|
| 正前 | 0° | 0.80 | 0.75 | 前期信任桥均值，后期微调 |
| 右前 | 45° | 0.71 | 0.71 | 略大自由度 |
| 正右 | 90° | 0.50 | 0.50 | 标准半椭圆，对称 |
| 右后 | 135° | 0.29 | 0.29 | 前期大方差，允许大转弯 |
| 正后 | 180° | 0.20 | 0.25 | 近矩形，全程高方差，允许绕行 |

峰值位置公式：$t^* = \frac{p}{2p} = 0.5$（对称情况），非对称时 $t^* = \frac{\alpha}{\alpha+\beta}$（此处 $\alpha=\beta=p$，故 $t^*=0.5$；峰值高度随 $p$ 减小而升高）。

### 9.3 完整方差公式

$$
\boxed{
\sigma^2(t;\theta_g) = \sigma^2_{\text{base}} \cdot \left[t(1-t)\right]^{p(\theta_g)} + t^2 \cdot \sigma^2_{\text{goal}}
}
$$

**$\sigma^2_{\text{base}}$ 的确定**（数据驱动，离线计算）：

$$
\sigma_{\text{base}} = \text{quantile}_{95\%}\!\left(\left\{\max_{i}\left\|\mathbf{x}_0^{(j)}[i] - \left[(1-\tfrac{i}{T})\mathbf{x}_0^{(j)}[0] + \tfrac{i}{T}\mathbf{g}^{(j)}\right]\right\|\right\}_{j=1}^N\right)
$$

即训练集中所有轨迹与其对应桥均值最大偏差的 95 分位数。**不参与梯度反传**，无学习崩塌风险。

### 9.4 约束验证

1. $f(0) = 0^p \cdot 1^p = 0$ ✅
2. $f(1) = 1^p \cdot 0^p = 0$ ✅（终点由 $\sigma^2_{\text{goal}}$ 控制松弛）
3. $f(t) \geq 0$，$\forall t \in [0,1]$ ✅
4. $p \in [0.2, 0.8]$，有界，不退化 ✅

---

## 10. 目标形式映射

### 10.1 v1 实现计划（PointGoal + NoGoal）

**v1 阶段**仅实现两种目标形式：

| 目标类型 | 布朗桥端点 $\mathbf{g}$ | 弹性方差 $\sigma^2_{\text{goal}}$ |
|---------|----------------------|----------------------------------|
| **PointGoal** | 3D 坐标 $\mathbf{g} \in \mathbb{R}^3$（广播到所有时间步） | 小值（如 0.1），强约束 |
| **NoGoal** | 零向量（或最后一步位置的粗估计） | 大值（如 10.0），弱约束 |

**放弃 RelativeDistance**：相对距离目标在绝对坐标系下语义不清晰，且与布朗桥端点约束不兼容，v1 不实现。

### 10.2 后续扩展计划

| 目标类型 | 布朗桥端点 $\mathbf{g}$ | 备注 |
|---------|----------------------|------|
| **WaypointGoal** | 中间路径点序列 $\{\mathbf{g}_i\}$ | 多段布朗桥拼接 |
| **DirectionGoal** | 方向向量 → 投影到轨迹终点 | 需要距离假设 |
| **CorridorGoal** | 走廊中心线 → 采样终点 | 软约束 |
| **ImageGoal** | 目标图像 → 视觉编码器提取位置估计 | 需要额外位置解码器 |
| **PixelGoal** | 像素坐标 → 反投影到 3D | 需要深度信息 |

---

## 11. 自检清单

| # | 检查项 | 结论 |
|---|--------|------|
| 1 | 布朗桥 $t=0$ 时 $\mathbf{x}_0 = \mathbf{x}_0$？ | ✅ $(1-0)\mathbf{x}_0 + 0 \cdot \mathbf{g} + 0 = \mathbf{x}_0$ |
| 2 | 布朗桥 $t=1$ 时 $\mathbf{x}_1 \approx \mathbf{g}$？ | ✅ 均值 $= \mathbf{g}$，方差 $= \sigma^2_{\text{goal}}$ |
| 3 | NoGoal 退化正确？ | ✅ $\sigma^2_{\text{goal}} \to \infty$ 时方差主导，桥约束消失 |
| 4 | 先验偏移补偿起点对齐？ | ✅ $\tilde{\boldsymbol{\mu}}[0] = \mathbf{0}$（当前位置） |
| 5 | Prior Encoder 输入是完整轨迹？ | ✅ $\tilde{\boldsymbol{\mu}} \in \mathbb{R}^{T \times 3}$，非标量均值 |
| 6 | Critic 不含先验信息？ | ✅ `predict_critic` 使用 `nogoal_embed`，不接收 prior tokens |
| 7 | 方差形状满足边界条件？ | ✅ $[t(1-t)]^p\big|_{t=0,1} = 0$，$p \in [0.2, 0.8]$ 有界 |
| 8 | 动作空间切换到绝对坐标？ | ✅ 不再做差分×4，直接输出 xyt 绝对坐标 |
| 9 | 损失函数双分支结构保留？ | ✅ `0.5 * ng_loss + 0.5 * mg_loss` |
| 10 | 推理步数与 NavDP 一致？ | ✅ 10步 DDIM 采样 |
| 11 | 目标在身后时自由度足够？ | ✅ $\theta_g=180°$ 时 $p=0.2$，方差近矩形，全程高自由度 |
| 12 | 运动连续性保证？ | ✅ 推理后三次样条平滑，C² 连续，旋转在移动中完成 |
| 13 | $\sigma_{\text{base}}$ 无学习崩塌风险？ | ✅ 数据驱动固定常数，不参与梯度反传 |
