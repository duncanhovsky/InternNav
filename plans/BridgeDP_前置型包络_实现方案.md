# Bridge-DP 前置型包络实现方案

## 1. 目标与边界

当前 PointGoal 尺度相似桥在轨迹时间 `tau` 上使用对称噪声包络。它将最大法向自由度放在路径中段，适合普通转向，但对“机器人前方很近就需要开始侧移”的绕障场景可能偏晚。本文计划引入一个可连续控制的前置型包络，使候选轨迹的法向和航向探索在路径前段更早展开。

本次设计只改变 PointGoal 桥中的轨迹时间方差形状，不改变：

- 绝对轨迹 `x0` prediction 训练目标。
- 从原点到 PointGoal 的有序桥均值。
- 轨迹尺度归一化、scale token 与 RGB-D FiLM。
- NoGoal 分支的远端探索噪声。
- critic 监督或 MPC 执行逻辑。

前置包络并不能替代 clearance-aware critic；它解决的是“候选是否有机会早转”，不是“候选是否必然被安全地选中”。

## 2. 当前公式

实现位置：

- `internnav/model/basemodel/bridgedp/bridge_scheduler.py`
- `scripts/train/base_train/configs/bridgedp.py`

当前 PointGoal 的轨迹时间包络为：

$$
A_0(\tau;\,p)
=
\left(
\frac{\tau(1-\tau)}{1/4}
\right)^p,
\qquad
p=0.5+0.3\cos(\theta_g).
$$

其中 `tau=(i+1)/T`，当前 `T=24`。法向、切向和航向噪声共用这一包络：

$$
\sigma_n=d r_n A_0,\qquad
\sigma_t=d r_t A_0,\qquad
\sigma_\theta=d r_\theta A_0.
$$

`A_0` 的峰值位于 `tau=0.5`。这意味着候选路径在中段最容易产生左右绕行，而前几个控制点倾向于更靠近直线均值。

## 3. 参数设计

### 3.1 新配置项

训练配置中新增：

```python
bridge_envelope_frontload=1.0
```

语义约定：

| 值 | 行为 |
|---:|---|
| `0.0` | 完全关闭前置效应，严格复现当前对称包络 |
| `0.5` | 轻度前置，适合第一轮风险较低的消融 |
| `1.0` | 温和前置，峰值由路径中点移动到约前三分之一处 |
| `2.0` | 较强前置，峰值移动到约前四分之一处 |

参数取值应限制为 `>= 0.0`。不支持负值，因为后置探索不是当前问题需要解决的行为，并会增加配置解释成本。

当前实现已完成 scheduler 接线，训练配置启用 `1.0` 作为温和前置首轮设置；将该值设回 `0.0` 即可进行严格对称包络基线或复现旧行为。

### 3.2 为什么用 `frontload=0.0` 关闭

也可以定义 `tail_power=1.0` 表示关闭，但 `frontload=0.0` 更易读：

- 零值表示新增影响为零。
- TensorBoard、配置差异和论文消融表更容易解释。
- 可以连续扫参，无需把“原始指数 1”与“额外前置强度”混为一谈。

## 4. 新包络公式

定义前置强度：

$$
\lambda = \texttt{bridge\_envelope\_frontload},\qquad \lambda\ge0
$$

以及尾部指数：

$$
q=1+\lambda.
$$

法向/航向前置包络定义为：

$$
A_f(\tau;\,p,\lambda)
=
\left(
\frac{\tau(1-\tau)^q}
{q^q/(q+1)^{q+1}}
\right)^p.
$$

分母是 `tau * (1-tau)^q` 的解析峰值，因此 `A_f` 的最大值仍为 1。峰值所在的轨迹位置为：

$$
\tau^*=\frac{1}{q+1}=\frac{1}{2+\lambda}.
$$

因此：

| `frontload` | `q` | 峰值位置 |
|---:|---:|---:|
| `0.0` | `1.0` | `0.50` |
| `0.5` | `1.5` | `0.40` |
| `1.0` | `2.0` | `0.33` |
| `2.0` | `3.0` | `0.25` |

当 `frontload=0.0` 时：

$$
q=1,\qquad
\frac{q^q}{(q+1)^{q+1}}=\frac14,
$$

于是：

$$
A_f(\tau;\,p,0)=A_0(\tau;\,p).
$$

这给出严格的关闭路径，不是近似兼容。

## 5. 作用维度

前置包络不应无差别地作用于全部维度。推荐分离两个包络：

```python
shape_tau_base = symmetric_envelope(tau, p)
shape_tau_turn = frontloaded_envelope(tau, p, frontload)

sigma_tangent = dist_scale * bridge_tangent_sigma_ratio * shape_tau_base
sigma_normal = dist_scale * bridge_normal_sigma_ratio * shape_tau_turn
sigma_theta = dist_scale * bridge_theta_sigma_ratio * shape_tau_turn
```

理由如下：

- 法向 `normal` 是绕障轨迹早期横移的主要自由度，必须前置。
- 航向 `theta` 与早期转向有关，应与法向同步前置。
- 切向 `tangent` 控制 waypoint 沿目标方向的前后扰动。将其前置会增加前段航点顺序紊乱和速度形状变化，不能直接解决贴墙问题。

在 `frontload=0.0` 时，`shape_tau_turn == shape_tau_base`，三类噪声均与当前实现一致。

## 6. 实现步骤

### Step 1：训练配置参数（已实现）

文件：`scripts/train/base_train/configs/bridgedp.py`

当前配置：

```python
# 法向/航向自由度的前置程度；0.0 严格关闭前置影响，1.0 为温和前置实验。
bridge_envelope_frontload=1.0,
```

将其改为 `0.0` 会完全关闭新增影响，用于 E0 对照或旧模型行为复现。

### Step 2：InternNav 调度器接线（已实现）

文件：`internnav/model/basemodel/bridgedp/bridge_scheduler.py`

修改内容：

1. 在 `BridgeScheduler.__init__()` 新增 `bridge_envelope_frontload: float = 0.0`。
2. 检查 `frontload >= 0.0`，负值直接抛出 `ValueError`。
3. 新增私有 helper，例如 `_trajectory_envelopes(tau, p)`，集中生成 `shape_tau_base` 和 `shape_tau_turn`。
4. 在 `pointgoal_noise_params()` 中仅将 `sigma_normal` 与 `sigma_theta` 改用 `shape_tau_turn`。
5. 保持 `sample_pointgoal_bridge_noise()`、`add_noise_trajectory()`、`sample_initial_noise_ordered()` 复用同一个参数路径，从而保证训练前向加噪与推理初始化一致。
6. 保持 NoGoal 分支不变。

计划中的核心实现形态：

```python
def trajectory_time_envelopes(self, tau, p):
    symmetric = (tau * (1.0 - tau) / 0.25).clamp(min=0.0).pow(p)

    frontload = self.bridge_envelope_frontload
    q = 1.0 + frontload
    peak = (q ** q) / ((q + 1.0) ** (q + 1.0))
    turn = (tau * (1.0 - tau).pow(q) / peak).clamp(min=0.0).pow(p)
    return symmetric, turn
```

### Step 3：训练策略模型传参（已实现）

文件：`internnav/model/basemodel/bridgedp/bridgedp_policy.py`

修改内容：

1. 从 `il` 读取：

```python
self.bridge_envelope_frontload = il.get('bridge_envelope_frontload', 0.0)
```

2. 实例化 `BridgeScheduler` 时透传该参数。
3. checkpoint 权重不需要变化，因为该参数只影响调度器数学过程，不新增可训练张量。
4. 训练旧 checkpoint 时，必须用 `frontload=0.0` 做行为复现；改变此参数后应视为新的推理/训练配置，不与旧 checkpoint 的主结果混报。

### Step 4：NavDP 评估侧同步（已实现）

评估侧是独立副本，必须同步实现，否则训练和部署会使用不同包络。涉及文件：

- `NavDP/baselines/bridgedp/bridge_scheduler.py`
- `NavDP/baselines/bridgedp/policy_network.py`
- `NavDP/baselines/bridgedp/policy_agent.py`
- `NavDP/baselines/bridgedp/bridgedp_server.py`
- `NavDP/baselines/bridgedp/start_eval.sh`

需要完成：

1. 在独立 scheduler 中复制相同 helper 和参数校验。
2. 从 server CLI 增加：

```bash
--bridge_envelope_frontload 1.0
```

3. 由 `BridgeDP_Agent`、`BridgeDP_Policy` 向 scheduler 透传。
4. 所有正式评估日志记录该值，防止 checkpoint 与评估噪声形状不匹配。

### Step 5：监控指标（部分已实现）

在调参前增加至少以下诊断指标：

| 指标 | 用途 |
|---|---|
| `bridge/frontload` | 记录当前前置强度 |
| `bridge/normal_sigma_wp1` | 查看第一个 waypoint 的法向自由度 |
| `bridge/normal_sigma_wp6` | 查看前四分之一位置的法向自由度 |
| `bridge/normal_sigma_mid` | 检查中段自由度是否被过度压缩 |
| `early_y_std` | NavDP 评估日志中，多候选前 25% 执行轨迹的横向离散程度 |

其中 `bridge/*` 训练日志和评估侧 `early_y_std` 已接入。初始候选、critic 选中轨迹的分阶段横向离散度，以及 `trajectory/min_clearance_selected`，需要进一步接入评估环境的障碍或碰撞几何后再计算。

## 7. 单元验证

### 7.1 完全关闭等价测试

构造固定 `goal`、固定随机噪声和若干 `theta_g`，比较新增实现中：

```python
bridge_envelope_frontload=0.0
```

与原始公式得到的：

- `sigma_tangent`
- `sigma_normal`
- `sigma_theta`
- `sample_pointgoal_bridge_noise()` 输出

应满足 `torch.allclose(..., atol=1e-7, rtol=1e-6)`。

### 7.2 峰值移动测试

对于 `p=0.8`、`T=24`：

| 配置 | 预期最大值附近 waypoint |
|---|---:|
| `frontload=0.0` | `12` |
| `frontload=1.0` | `8` |
| `frontload=2.0` | `6` |

同时验证最后一个 waypoint 的法向与航向 sigma 为 0。

### 7.3 切向不受前置影响测试

在相同 `dist`、`theta_g` 下，对比 `frontload=0.0/1.0/2.0`：

```python
sigma_tangent
```

应完全相等；`sigma_normal` 与 `sigma_theta` 应发生预期前移。

### 7.4 训练与评估实现一致性测试

使用相同张量输入与固定随机噪声，比较 InternNav scheduler 和 NavDP baseline scheduler 的输出，确保两份独立实现完全对齐。

## 8. 实验矩阵

由于当前配置中的 `bridge_normal_sigma_ratio` 和 `bridge_theta_sigma_ratio` 已经是另一维度的调参项，前置包络实验应先固定幅度参数，只扫形状参数，避免无法解释收益来源。

建议第一轮固定：

```python
bridge_normal_sigma_ratio=1.0
bridge_tangent_sigma_ratio=0.05
bridge_theta_sigma_ratio=0.3
```

扫参：

| 实验 | `bridge_envelope_frontload` | 目的 |
|---|---:|---|
| E0 | `0.0` | 完全关闭，作为当前配置基线 |
| E1 | `0.5` | 轻度前置 |
| E2 | `1.0` | 温和前置，主候选 |
| E3 | `2.0` | 强前置，检验过早转向风险 |

评价至少包含：

- PointGoal success rate 与 SPL。
- 失败案例的接触次数或最小 clearance。
- 前 25% 轨迹的横向位移和候选横向方差。
- 轨迹总长度，识别是否出现不必要的大绕行。
- critic 选中轨迹相对于全部候选的 clearance 排名。

如果 `E1/E2` 的候选更早侧移但选中结果仍贴墙，下一步应改 critic clearance 监督或排序惩罚，而不是继续加大 `frontload`。

## 9. 风险与回退

| 风险 | 表现 | 回退措施 |
|---|---|---|
| 前置过强 | 开阔道路也过早摆动，SPL 下降 | 将 `frontload` 降至 `0.5` 或 `0.0` |
| 现有法向比例偏大 | 候选前段发散过宽，denoiser 不稳定 | 固定 `frontload` 后另行降低 `bridge_normal_sigma_ratio` |
| 仅推理侧启用 | 输出分布偏离训练条件 | 正式结果只使用训练与评估一致配置 |
| 安全候选未被选中 | 初始候选改善但碰撞不降 | 转向 clearance-aware critic/排序修改 |

回退开关就是：

```python
bridge_envelope_frontload=0.0
```

该值应在实现和测试中保证严格恢复当前对称包络行为。
