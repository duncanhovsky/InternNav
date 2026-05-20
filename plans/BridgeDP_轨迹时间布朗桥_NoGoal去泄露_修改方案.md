# Bridge-DP 轨迹时间布朗桥与 NoGoal 去泄露修改方案

## 1. 背景

当前 Bridge-DP 实现中，`sample_bridge_noise()` 为每条样本轨迹采样一个扩散时间步 `timestep`，然后在 `BridgeScheduler.add_noise()` 中用同一个归一化时间 `t` 作用到所有 24 个航点：

```text
x_t[i] = (1 - t) * x0[i] + t * goal + sigma(t, theta_g) * eps[i]
```

这个公式的问题是：当 `t` 接近 1 时，所有航点的均值都会接近同一个 `goal`，航点顺序结构被破坏。理论上，Bridge-DP 的桥均值应该随航点序号变化：第 1 个航点靠近机器人起点，第 24 个航点靠近导航终点，中间航点沿起点到终点有序分布。

此外，当前 no-goal 分支虽然将 goal embedding 置零，但前向加噪仍然使用真实 `goal` 构造 noisy action。这会让真实目标通过 `x_t` 泄露给 no-goal 分支，导致 no-goal 分支不是严格的无目标条件建模。

本方案只讨论实现修改计划，不直接修改代码。

## 2. 核心理论：拆分两个时间变量

需要将两个不同含义的时间变量拆开：

```text
tau_i: 轨迹时间，表示第 i 个航点在整条轨迹中的位置
s:     扩散时间，表示当前 noisy sample 离干净轨迹 x0 有多远
```

设预测航点数为 `T=24`，机器人起点为 `o=(0,0,0)`，导航终点为 `g`，则轨迹时间为：

```text
tau_i = (i + 1) / T, i = 0, 1, ..., T-1
```

不采样 `tau=0`，因为 `tau=0` 是机器人当前位置，即起点。第一个预测航点从 `1/T` 开始，最后一个航点为 `tau=1`。

PointGoal 分支的有序桥均值：

```text
mu_pg[i] = (1 - tau_i) * o + tau_i * g
```

轨迹时间方差采用中间大、两端小的布朗桥形状：

```text
sigma_tau[i]^2 = sigma_base^2 * [tau_i * (1 - tau_i)] ^ p(theta_g) + sigma_floor^2
p(theta_g) = 0.5 + 0.3 * cos(theta_g)
```

其中 `sigma_floor` 用来避免端点处方差为 0，尤其是后续若保留 eps 类损失时，避免除零或梯度爆炸。

扩散时间 `s` 控制从干净轨迹到桥先验的腐蚀程度：

```text
z_s[i] = (1 - s) * x0[i] + s * mu[i] + beta(s) * sigma_tau[i] * eps[i]
```

建议先使用简单稳定的：

```text
beta(s) = s
```

于是边界条件为：

```text
s = 0: z_s[i] = x0[i]
s = 1: z_s[i] = mu[i] + sigma_tau[i] * eps[i]
```

这与推理时 `sample_initial_noise_ordered()` 的语义一致：反向采样从一条有序布朗桥先验轨迹开始，而不是从所有航点都挤在 goal 附近开始。

## 3. 反向去噪推导

网络仍然预测干净轨迹：

```text
x0_hat = model(z_s, s, condition)
```

由前向公式可反推噪声：

```text
eps_hat[i] = (z_s[i] - (1 - s) * x0_hat[i] - s * mu[i])
             / (beta(s) * sigma_tau[i])
```

从当前扩散时间 `s` 更新到更小的 `s_prev`：

```text
z_s_prev[i] =
    (1 - s_prev) * x0_hat[i]
  + s_prev * mu[i]
  + beta(s_prev) * sigma_tau[i] * eps_hat[i]
```

当 `s_prev=0` 时：

```text
z_0 = x0_hat
```

这比当前 `step()` 中只用 `x_t` 与 `x0_pred` 做线性混合更一致，因为它显式保留了有序桥均值 `mu[i]` 和每个航点自己的 `sigma_tau[i]`。

## 4. NoGoal 去泄露设计

要求：保留 no-goal 分支，但 no-goal 的 forward noising 不依赖真实 `goal`。

### 4.1 NoGoal 桥均值

NoGoal 分支不能使用 `g`，因此推荐使用与目标无关的零均值桥先验：

```text
g_front = (d_front, 0, 0)
mu_ng[i] = tau_i * g_front
```

NoGoal 前向加噪：

```text
z_s_ng[i] = (1 - s) * x0[i] + s * mu_ng[i] + beta(s) * sigma_ng[i] * eps[i]
```

这里 `x0` 是监督目标本身，训练扩散模型时 noisy input 保留一部分 `x0` 信息是正常的；关键是当 `s=1` 时，noisy input 不再包含真实目标终点 `g`。

### 4.2 NoGoal 方差

NoGoal 没有目标方向 `theta_g`，建议先采用目标无关的固定轨迹方差：

```text
sigma_x[i]     = sigma_start + (sigma_x_end     - sigma_start) * tau_i^q
sigma_y[i]     = sigma_start + (sigma_y_end     - sigma_start) * tau_i^q
sigma_theta[i] = sigma_start + (sigma_theta_end - sigma_start) * tau_i^q
```

或使用弱布朗桥形状：

```text
sigma_y_end > sigma_x_end, sigma_theta_end > sigma_start
```

推荐第一阶段使用固定方差，减少变量：

```text
d_front = 0.8
sigma_start = 0.03
sigma_x_end = 0.35
sigma_y_end = 0.80
sigma_theta_end = 0.60
q = 2.0
```

### 4.3 NoGoal 推理初始化

`predict_nogoal_batch_action_vel()` 保持不使用真实目标：

```text
z_1_ng ~ N(mu_ng, diag(sigma_x^2, sigma_y^2, sigma_theta^2))
```

反向 `step()` 使用 `mu_ng=0`。这样 no-goal 分支从训练到推理都不依赖真实 goal。

## 5. 文件级修改方案

### 5.1 `internnav/model/basemodel/bridgedp/bridge_scheduler.py`

新增或重构以下方法。

#### 新增轨迹时间工具

```python
def trajectory_time(self, predict_size, device, dtype):
    # returns (1, T, 1), values [1/T, ..., 1]
```

#### 新增桥均值

```python
def bridge_mean_ordered(self, goal, origin, shape):
    # goal: (B, 3)
    # origin: (B, 3)
    # return mu: (B, T, 3)
```

PointGoal:

```text
mu = origin + tau * (goal - origin)
```

NoGoal:

```text
mu = tau * (d_front, 0, 0)
```

#### 新增轨迹时间方差

```python
def trajectory_std(self, goal=None, theta_g=None, shape=None, mode="pointgoal"):
```

PointGoal:

```text
sigma_tau = sqrt(sigma_base^2 * [tau(1-tau)]^p + sigma_floor^2)
```

NoGoal:

```text
sigma_tau 单调随 tau 增大，且 y/theta 方差大于 x 方差
```

#### 替换 `add_noise()`

旧接口：

```python
add_noise(x0, goal, theta_g, timesteps, noise=None)
```

建议改为：

```python
add_noise_trajectory(
    x0,
    timesteps,
    goal=None,
    theta_g=None,
    origin=None,
    mode="pointgoal",
    noise=None,
)
```

返回值建议包含调试所需的中间量：

```python
return noisy, noise, mu, sigma_tau, s_norm
```

#### 替换 `step()`

旧的 `step()` 只做：

```text
x_det = coeff_xt * x_t + coeff_x0 * x0_pred
```

建议新增：

```python
step_trajectory(
    x0_pred,
    x_s,
    timestep,
    goal=None,
    theta_g=None,
    origin=None,
    mode="pointgoal",
    eta=0.0,
)
```

内部按第 3 节公式计算 `eps_hat` 和 `z_s_prev`。

#### 保留 `sample_initial_noise_ordered()`

保留并改为调用统一的：

```python
mu = bridge_mean_ordered(...)
sigma_tau = trajectory_std(...)
return mu + sigma_tau * noise
```

新增 NoGoal 初始化：

```python
sample_initial_noise_nogoal(shape, device)
```

或在 `sample_initial_noise_ordered(..., mode="nogoal")` 内处理。

### 5.2 `internnav/model/basemodel/bridgedp/bridgedp_policy.py`

#### 修改 `sample_bridge_noise()`

当前 `sample_bridge_noise(x0, goal, theta_g, timesteps=None)` 同时服务 ng/mg，导致 ng 泄露 goal。

建议改为：

```python
def sample_bridge_noise(
    self,
    x0,
    timesteps=None,
    goal=None,
    theta_g=None,
    mode="pointgoal",
):
```

PointGoal/mixed-goal 调用：

```python
mg_x0_target, mg_time_embed, mg_noisy_embed, mg_timesteps, mg_noisy_action = \
    self.sample_bridge_noise(
        tensor_label_actions,
        goal=tensor_point_goal,
        theta_g=tensor_theta_g,
        mode="pointgoal",
    )
```

NoGoal 调用：

```python
ng_x0_target, ng_time_embed, ng_noisy_embed, ng_timesteps, ng_noisy_action = \
    self.sample_bridge_noise(
        tensor_label_actions,
        goal=None,
        theta_g=None,
        mode="nogoal",
    )
```

这样 no-goal 的 noisy action 不再依赖 `tensor_point_goal`。

#### 训练 forward

保留两个 action 分支：

```text
ng 分支: no goal embedding + no-goal noising
mg 分支: mixed goal embedding + pointgoal ordered-bridge noising
```

`use_prior_traj=False` 维持现状，prior token 继续使用零 token。

#### PointGoal 推理

`predict_pointgoal_batch_action_vel()` 使用：

```python
sample_initial_noise_ordered(..., mode="pointgoal")
step_trajectory(..., mode="pointgoal")
```

#### NoGoal 推理

`predict_nogoal_batch_action_vel()` 使用：

```python
sample_initial_noise_nogoal(...)
step_trajectory(..., mode="nogoal")
```

不要传真实 `goal` 或由真实 goal 推导的 `theta_g`。

### 5.3 `internnav/trainer/bridgedp_trainer.py`

#### Loss 主结构

第一阶段建议：

```text
action_loss = L_x0 + lambda_delta * L_delta
```

暂时设置：

```text
lambda_eps = 0.0
```

原因：新的前向过程已有显式有序桥先验，`L_eps` 容易成为按 `1 / sigma_tau^2` 加权的 x0 loss。端点 `sigma_tau` 小时尤其容易放大梯度。

#### 若保留 `L_eps`

必须使用 scheduler 返回的 `mu`、`sigma_tau`、`s_norm` 计算：

```text
eps_target = (z_s - (1-s) * x0_target - s * mu) / (beta(s) * sigma_tau)
eps_pred   = (z_s - (1-s) * x0_pred   - s * mu) / (beta(s) * sigma_tau)
```

并且：

```text
sigma_tau >= sigma_floor
beta(s) >= beta_floor
```

建议第二阶段再打开，权重从 `0.01` 开始。

#### 可选新增几何损失

可保留当前 `L_delta`。若轨迹仍抖动，再加入二阶平滑：

```text
L_smooth = MSE(
    x[:, 2:] - 2*x[:, 1:-1] + x[:, :-2],
    target[:, 2:] - 2*target[:, 1:-1] + target[:, :-2]
)
```

建议初始：

```text
lambda_smooth = 0.01
```

### 5.4 `scripts/train/base_train/configs/bridgedp.py`

建议新增或调整：

```python
bridge_use_trajectory_time=True
bridge_beta_schedule="linear"   # beta(s)=s
sigma_floor=0.01
sigma_nogoal=0.5
lambda_eps=0.0
lambda_delta=0.1
lambda_smooth=0.0
use_prior_traj=False
```

`sigma_base` 可以先保留 `0.2`。如果初始化轨迹过散，再降到 `0.1`；如果绕行能力不足，再升到 `0.3`。

### 5.5 `internnav/dataset/bridgedp_lerobot_dataset.py`

第一阶段无需改动。

当前数据集已经满足：

```text
point_goal = raw_pred_actions[-1]
theta_g = atan2(point_goal_y, point_goal_x)
Trainer 用 raw trajectory 生成 24 点弧长重采样监督
```

需要在验证中确认：

```text
batch_labels[:, -1, :2] ~= batch_pg[:, :2]
```

## 6. 验证清单

### 6.1 Scheduler 单元验证

PointGoal ordered mean：

```text
mu[:, 0]  ~= goal / T
mu[:, -1] == goal
mu 的 xy 范数整体递增
```

PointGoal `s=1`：

```text
z_1 的 24 个航点均值沿 origin -> goal 排列
不允许 24 个航点均值都等于 goal
```

NoGoal 去泄露：

```text
同一个 x0、noise、timesteps 下，传入不同 goal，ng_noisy_action 必须完全一致
```

端点方差：

```text
sigma_tau[:, -1] >= sigma_floor
不存在 NaN / Inf
```

### 6.2 训练小样本 overfit

用 16、32、64 条样本分别测试：

```text
L_x0 能快速下降
L_delta 同步下降
真实去噪推理轨迹贴近 GT，而不只是 teacher-forced loss 降低
```

建议先只看 PointGoal/mixed-goal 分支，不评价 prior。

### 6.3 可视化验证

检查训练可视化 JSONL：

```text
pred_traj 第一个点靠近起点前方，而不是靠近终点
pred_traj 最后一点靠近 nav_goal
24 个点有顺序，不出现全部堆在 goal 附近
NoGoal 可视化不因 batch_pg 改变而系统性改变初始化分布
```

### 6.4 分支对照实验

建议做 3 组 ablation：

```text
A: 旧 scalar-t bridge + ordered init
B: trajectory-time bridge + ordered init + lambda_eps=0
C: B + lambda_eps=0.01
```

优先比较：

```text
训练 L_x0
推理 ADE/FDE
终点误差
轨迹平滑度
可视化是否有 waypoint 堆叠
```

## 7. 推荐实施顺序

1. 先改 `BridgeScheduler`，实现 `tau_i` 有序桥均值、轨迹时间方差、新的 `add_noise_trajectory()` 和 `step_trajectory()`。
2. 改 `BridgeDPNet.sample_bridge_noise()`，加入 `mode="pointgoal" | "nogoal"`，让 no-goal 不再接收真实 goal。
3. 改 PointGoal 和 NoGoal 推理，统一使用新的 scheduler step。
4. 暂时关闭 `lambda_eps`，只训练 `L_x0 + lambda_delta * L_delta`。
5. 跑 scheduler 单元检查，确认 `s=1` 时 PointGoal 初始化是有序轨迹，而不是所有点在 goal 附近。
6. 跑 16/32/64 条样本 overfit。
7. 若 overfit 和可视化都稳定，再考虑打开小权重 `lambda_eps=0.01` 或 `lambda_smooth=0.01`。
8. prior 继续保持关闭，等主路径稳定后再单独启用。

## 8. 预期收益

修改后，训练和推理会满足同一个分布假设：

```text
训练 s=1: 从有序布朗桥先验轨迹恢复 x0
推理初始: 从有序布朗桥先验轨迹开始去噪
```

NoGoal 分支也会满足严格无目标条件：

```text
goal embedding 不出现真实 goal
forward noising 不使用真实 goal
reverse step 不使用真实 goal
```

这样可以同时解决两个核心问题：24 个航点不再在高扩散时间堆到同一个 goal 附近，no-goal 分支也不再通过 noisy action 偷看真实导航终点。

## 9. NoGoal 最终采用方案

根据后续讨论，NoGoal 分支不采用零均值自由扩散，而采用固定前方默认目标：

```text
g_front = (d_front, 0, 0)
mu_ng[i] = tau_i * g_front
```

该默认目标是固定超参，不来自样本真实 `point_goal`，因此不会造成目标泄露。NoGoal 的方差采用单调递增、各向异性的形式：

```text
sigma_x(tau)     = sigma_start + (sigma_x_end - sigma_start) * tau^q
sigma_y(tau)     = sigma_start + (sigma_y_end - sigma_start) * tau^q
sigma_theta(tau) = sigma_start + (sigma_theta_end - sigma_start) * tau^q
```

默认值：

```text
d_front = 0.8
sigma_start = 0.03
sigma_x_end = 0.35
sigma_y_end = 0.80
sigma_theta_end = 0.60
q = 2.0
```

语义是：NoGoal 默认向正前方探索，近端稳定，越靠近远端越开放，尤其允许横向和朝向产生更大变化。
