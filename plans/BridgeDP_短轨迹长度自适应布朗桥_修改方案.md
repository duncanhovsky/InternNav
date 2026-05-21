# Bridge-DP 尺度相似各向异性布朗桥修改方案

## 1. 背景与目标

当前训练现象：

1. 长轨迹样本（移动距离 > 0.8m）Bridge-DP 学习效果较好。
2. 短轨迹样本（移动距离 < 0.5m）和静止样本预测质量明显变差。
3. 现有 point-goal 有序布朗桥的均值会随 `origin -> goal` 缩放，但方差主要由 `sigma_base`、`sigma_floor`、`theta_g` 与航点时间 `tau` 决定，没有随起终点距离同步缩放。

本方案目标：

1. 让 point-goal 布朗桥的初始化分布随目标距离等比例缩放，使不同距离的轨迹在模型输入中呈现稳定、相似的归一化形状。
2. 将 xy 噪声从全局各向同性改成沿轨迹切向/法向的各向异性噪声：切向小方差，法向方差按起终点距离成比例缩放。
3. 在推理候选排序中加入可配置的“目标一致性项”，避免目标距离很小时安全但乱晃的候选被 critic 选中。
4. 增加距离区间监控，直接观察不同目标距离下的 loss 和终点误差。

## 2. 关键判断

### 2.1 关于“长轴 a 与短轴 b 成正比”

可以这样建模。

这里不需要真的引入椭圆函数，但建议显式引入“轨迹局部坐标系”里的各向异性噪声。把起点到终点的距离 `d = ||goal_xy - origin_xy||` 作为轨迹尺度，让桥的法向中段噪声幅度与 `d` 成正比，就能得到你说的效果：

- 长轴 `a`：起点到终点的距离 `d`。
- 短轴 `b`：布朗桥中段允许偏离直线均值的法向标准差。
- 比例关系：`b = ratio * a`。

这样每条轨迹在自身尺度下看到的桥形状一致。尤其数据已经做了坐标归一化，模型看到的是“相似形状、不同尺度”的稳定分布，而不是短轨迹被固定大噪声淹没、长轨迹相对更容易的分布。

更准确地说，这是一种“长度自适应 + 切法向各向异性”的尺度不变桥先验。航点均值已经通过 `mu_i = origin + tau_i * (goal - origin)` 在切向上均匀铺开，因此随机初始化不应该再显著打乱切向顺序；真正需要自由度的是法向偏离，也就是给模型绕开直线均值、形成曲线或绕障的空间。

### 2.2 当前代码里的问题点

主要文件：

- `internnav/model/basemodel/bridgedp/bridge_scheduler.py`
- `internnav/model/basemodel/bridgedp/bridgedp_policy.py`
- `internnav/trainer/bridgedp_trainer.py`
- `scripts/train/base_train/configs/bridgedp.py`

当前 `trajectory_std()` 的 point-goal 分支：

```python
var = (self.sigma_base ** 2) * (t_prod ** p) + (self.sigma_floor ** 2)
return var.sqrt().expand(B, T_pred, dim)
```

它没有使用 `goal - origin` 的距离。因此当 `goal == origin` 时：

- `mu` 全部接近 0。
- `sigma` 仍然非零。
- `sample_initial_noise_ordered()` 会采样出随机轨迹。

当前配置中：

- `sigma_base = 0.2`
- `sigma_floor = 0.01`
- `xy` 反归一化比例为 `5.0`

point-goal 中段 1σ 物理尺度大约在 0.58m 到 0.87m 之间。对 `<0.5m` 或静止样本，这个初始化噪声相对过大。

## 3. 方案 A：尺度相似 + 切法向各向异性布朗桥方差

### 3.1 新增配置

在 `scripts/train/base_train/configs/bridgedp.py` 的 Bridge-DP 专有超参中新增：

```python
# point-goal 尺度相似各向异性桥方差
bridge_scale_invariant_sigma=True,
bridge_anisotropic_xy=True,
bridge_normal_sigma_ratio=0.25,   # 法向中段 std = ratio * 起终点距离
bridge_tangent_sigma_ratio=0.03,  # 切向中段 std = ratio * 起终点距离
bridge_theta_sigma_ratio=0.05,    # theta 中段 std = ratio * 起终点距离
```

说明：

- 不再使用 `bridge_length_ref`。任何距离的轨迹都直接用自身起终点距离 `d` 作为尺度。
- 不再使用 `bridge_sigma_min_scale` / `bridge_sigma_max_scale`。`d=0` 时方差自然为 0；长轨迹也不会被截断，桥形状比例保持一致。
- 不再使用 `bridge_static_threshold` / `bridge_static_sigma_scale`。静止不是特殊规则，而是 `d=0` 时公式的自然退化。
- 不再使用 `bridge_scale_theta`。theta 噪声也由同一个轨迹尺度 `d` 和独立比例 `bridge_theta_sigma_ratio` 控制。
- `bridge_anisotropic_xy=True` 时，point-goal 的 xy 噪声不再是全局 `(x, y)` 独立同方差，而是在每条样本自己的切向/法向坐标系中采样。
- `bridge_tangent_sigma_ratio` 通常远小于 `bridge_normal_sigma_ratio`，因为切向航点位置已经由 `tau` 均匀铺开。

### 3.2 尺度相似与切法向公式

对 point-goal 分支先计算轨迹尺度。这里使用归一化空间中的距离，和模型输入输出保持一致：

```python
vec_xy = (goal_b - origin_b)[:, :2]
dist = torch.norm(vec_xy, dim=-1)  # normalized distance, shape=(B,)
```

布朗桥在航点时间上的形状使用归一化函数，保证每条轨迹中段峰值为 1，端点为 0：

```python
tau = self.trajectory_time(T_pred, device, dtype)  # (1, T, 1)
p = self.direction_adaptive_exponent(theta_g).view(B, 1, 1)
t_prod = (tau * (1.0 - tau)).clamp(min=0.0)
shape_tau = (t_prod / 0.25).clamp(min=0.0).pow(p)  # max(shape_tau)=1 at tau=0.5
```

然后构造轨迹局部坐标系。`eps` 只用于避免除零，不作为实际采样方差：

```python
dist_safe = dist.clamp(min=1e-6)
tangent = vec_xy / dist_safe.view(B, 1)         # e_t, shape=(B, 2)
normal = torch.stack([-tangent[:, 1], tangent[:, 0]], dim=-1)  # e_n
```

当 `dist=0` 时，`sigma_tangent/sigma_normal/sigma_theta` 全为 0，因此默认方向不会影响采样结果。可以只为了数值形状稳定使用默认方向：

```python
default_tangent = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
default_normal = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
zero_dist = dist <= 1e-6
tangent = torch.where(zero_dist.view(B, 1), default_tangent.view(1, 2), tangent)
normal = torch.where(zero_dist.view(B, 1), default_normal.view(1, 2), normal)
```

切向、法向和角度方差直接由 `dist` 成比例给出：

```python
dist_scale = dist.view(B, 1, 1)
sigma_normal = dist_scale * bridge_normal_sigma_ratio * shape_tau
sigma_tangent = dist_scale * bridge_tangent_sigma_ratio * shape_tau
sigma_theta = dist_scale * bridge_theta_sigma_ratio * shape_tau
```

xy 噪声为：

```python
eps_tangent = torch.randn(B, T_pred, 1, device=device, dtype=dtype)
eps_normal = torch.randn(B, T_pred, 1, device=device, dtype=dtype)

xy_noise = (
    tangent.view(B, 1, 2) * sigma_tangent * eps_tangent
    + normal.view(B, 1, 2) * sigma_normal * eps_normal
)
```

也就是说：

- 切向：航点已经由 `tau` 均匀铺开，只允许很小随机扰动。
- 法向：主要随机自由度，且幅度随 `origin-goal` 距离等比例缩放。
- `dist=0`：所有 sigma 都为 0，初始化自然为全零，不需要任何静止特殊判断。
- 长轨迹：sigma 随 `dist` 继续线性增长，不做 max clamp，因此桥形状比例不被破坏。
- `sigma_floor` 可继续保留给 NoGoal 或旧兼容逻辑；在 point-goal 尺度相似分支中不应加入空间方差 floor。

### 3.3 与现有逐维 sigma 的区别

当前实现是：

```python
noisy = mean + beta * sigma_xyz * randn_like(x0)
```

这只能表达全局 x/y 逐维独立噪声，不能表达“沿轨迹方向小、法向大”的旋转局部协方差。各向异性版本需要在 point-goal 分支新增 helper，而不是只返回 `(B, T, 3)` 的 `sigma`。

建议新增两个 helper：

```python
def pointgoal_noise_params(self, shape, device, dtype, goal, theta_g=None, origin=None):
    # 返回 mu, tangent, normal, sigma_tangent, sigma_normal, sigma_theta, sigma_diag

def sample_pointgoal_bridge_noise(self, shape, device, dtype, goal, theta_g=None, origin=None):
    # 使用上面的参数采样 xy_noise 和 theta_noise
```

NoGoal 分支仍然保留当前逐维 sigma 逻辑，不参与这次改动。

### 3.4 修改点

#### 3.4.1 `BridgeScheduler.__init__`

新增参数并保存为成员变量：

```python
bridge_scale_invariant_sigma: bool = False
bridge_anisotropic_xy: bool = True
bridge_normal_sigma_ratio: float = 0.25
bridge_tangent_sigma_ratio: float = 0.03
bridge_theta_sigma_ratio: float = 0.05
```

这些比例均作用在归一化空间的 `dist` 上，因此不需要 `action_scale_xy` 参与桥方差计算。若需要写物理量日志，再在 trainer 或 policy 层反归一化。

#### 3.4.2 `BridgeScheduler.trajectory_std()`

`trajectory_std()` 可保留为兼容接口，用于 NoGoal 分支和日志诊断。但 point-goal 的真实采样逻辑应切到新 helper。为了兼容现有返回值，可以让 point-goal `trajectory_std()` 在 `bridge_anisotropic_xy=False` 时走旧逻辑；在 `True` 时返回一个“诊断用 sigma”：

```python
sigma_diag[..., 0] = sigma_tangent
sigma_diag[..., 1] = sigma_normal
sigma_diag[..., 2] = sigma_theta
```

注意这个 `sigma_diag` 不再表示全局 x/y 逐维标准差，只用于监控，不应直接用于 eps-loss。

#### 3.4.3 `BridgeScheduler.add_noise_trajectory()`

point-goal 分支改为：

```python
mu, tangent, normal, sigma_t, sigma_n, sigma_theta, sigma_diag = \
    self.pointgoal_noise_params(...)

eps_t = torch.randn(B, T_pred, 1, device=device, dtype=dtype)
eps_n = torch.randn(B, T_pred, 1, device=device, dtype=dtype)
eps_th = torch.randn(B, T_pred, 1, device=device, dtype=dtype)

xy_noise = tangent[:, None, :] * sigma_t * eps_t + normal[:, None, :] * sigma_n * eps_n
theta_noise = sigma_theta * eps_th
noise_aniso = torch.cat([xy_noise, theta_noise], dim=-1)

noisy = (1.0 - s_norm) * x0 + s_norm * mu + s_norm * noise_aniso
return noisy, noise_aniso, mu, sigma_diag, s_norm
```

这里返回的 `noise_aniso` 是已经投影回全局 x/y/theta 空间的实际扰动，不是标准正态 `eps`。当前训练是 x0-prediction，`noise` 返回值不参与主损失，因此第一版这样最稳。

#### 3.4.4 `BridgeScheduler.sample_initial_noise_ordered()`

初始化也复用相同 helper：

```python
mu, tangent, normal, sigma_t, sigma_n, sigma_theta, _ = self.pointgoal_noise_params(...)
xy_noise = ...
theta_noise = ...
return mu + torch.cat([xy_noise, theta_noise], dim=-1)
```

这样 `dist=0` 时初始值接近全零；`dist>0` 时主要发生法向探索，且探索幅度与轨迹尺度成比例。

#### 3.4.5 `BridgeScheduler.step_trajectory()`

各向异性后，不建议继续使用逐维：

```python
eps_hat = residual / (beta * sigma)
x_prev = mean_prev + s_prev * sigma * eps_hat
```

因为 `sigma` 不是全局逐维标准差。更稳的写法是直接搬运当前 residual：

```python
mean_s = (1.0 - s) * x0_pred + s * mu
residual = x_s - mean_s
x_prev = (1.0 - s_prev) * x0_pred + s_prev * mu + (s_prev / s.clamp(min=1e-6)) * residual
```

这对旧的逐维 sigma 和新的切法向 sigma 都成立，本质是 DDIM 确定性路径中保持同一个噪声残差并按 diffusion time 缩放。

如果 `eta > 0.0`，额外随机扰动也应使用同一套切法向采样，而不是 `torch.randn_like(x_prev) * sigma_diag`。

#### 3.4.6 `BridgeDPNet.__init__`

从 `il` 读取新配置并传给 `BridgeScheduler`：

```python
self.bridge_scale_invariant_sigma = il.get('bridge_scale_invariant_sigma', False)
self.bridge_anisotropic_xy = il.get('bridge_anisotropic_xy', True)
...
self.bridge_scheduler = BridgeScheduler(
    ...,
    bridge_scale_invariant_sigma=self.bridge_scale_invariant_sigma,
    bridge_anisotropic_xy=self.bridge_anisotropic_xy,
    bridge_normal_sigma_ratio=self.bridge_normal_sigma_ratio,
    bridge_tangent_sigma_ratio=self.bridge_tangent_sigma_ratio,
    bridge_theta_sigma_ratio=self.bridge_theta_sigma_ratio,
)
```

#### 3.4.7 `lambda_eps` 约束

当前默认 `lambda_eps=0.0`，不会受影响。

如果未来重新开启 `lambda_eps > 0`，不能再用 `bridge_sigma` 逐维除法。需要改成切法向局部坐标中的 eps consistency，或者直接删除 eps consistency，保留 x0-prediction 主损失。

#### 3.4.8 推理初始化

不用额外改 `predict_pointgoal_batch_action_vel()` 的初始化流程，因为它已经调用：

```python
self.bridge_scheduler.sample_initial_noise_ordered(...)
```

只要 `sample_initial_noise_ordered()` 内部改成切法向采样，推理初始化自然会跟着变。

### 3.5 预期效果

假设归一化空间中 `d=0.16`（约 0.8m，因 xy scale 为 5m），且：

```python
bridge_normal_sigma_ratio=0.25
bridge_tangent_sigma_ratio=0.03
bridge_theta_sigma_ratio=0.05
```

则中段峰值附近：

- 法向 std ≈ `0.16 * 0.25 = 0.04`，反归一化约 0.20m。
- 切向 std ≈ `0.16 * 0.03 = 0.0048`，反归一化约 0.024m。
- 如果 `d` 翻倍，三类 std 同步翻倍；如果 `d=0`，三类 std 全为 0。

- 切向扰动明显小于法向扰动，航点顺序更稳定。
- 法向扰动保留主要探索自由度，适合表达绕行、曲线和避障。
- 所有距离的轨迹使用同一套比例公式，不存在短/长轨迹特例。

## 4. 方案 B：可开关目标一致性项

### 4.1 定位

目标一致性项建议先放在推理候选排序阶段，而不是训练 loss 中。

原因：

1. 当前问题最明显发生在推理候选生成与 critic 排序后。
2. Critic 当前主要评估轨迹安全性，不显式使用 point-goal，因此目标距离很小时可能选中“安全但多走”的轨迹。
3. 推理排序项可以通过 config 开关快速 ablation，不改变主训练目标。

### 4.2 新增配置

在 `scripts/train/base_train/configs/bridgedp.py` 增加：

```python
# 推理候选目标一致性排序项
enable_goal_consistency_score=True,
goal_consistency_terminal_weight=1.0,
goal_consistency_path_weight=0.2,
```

如果要做最小改动，也可以只保留：

```python
enable_goal_consistency_score=False
goal_consistency_weight=1.0
```

第一版建议默认 `True`，但如果希望完全复现实验旧行为，则默认 `False`。

### 4.3 排序公式

在 `predict_pointgoal_batch_action_vel()` 中，原逻辑：

```python
critic_values = self.predict_critic(naction, rgbd_embed)
positive_trajectory = trajectory[(-critic_values).argsort()[0:8]]
```

新增目标一致性后：

```python
score = critic_values

if self.enable_goal_consistency_score:
    terminal_err = torch.norm(naction[:, -1, :2] - goal_repeated[:, :2], dim=-1)
    path_len = torch.norm(naction[:, 1:, :2] - naction[:, :-1, :2], dim=-1).sum(dim=-1)
    goal_dist = torch.norm(goal_repeated[:, :2] - origin_repeated[:, :2], dim=-1)

    consistency_penalty = (
        terminal_w * terminal_err
        + path_w * torch.relu(path_len - goal_dist)
    )

    score = critic_values - consistency_penalty
```

这也是一个通用规则，不需要单独判断短轨迹或静止轨迹。当 `goal_dist=0` 时，`torch.relu(path_len - goal_dist)` 自然退化为 `path_len`，会惩罚无意义漂移；当轨迹变长时，允许的路径长度基准也随目标距离线性增长。

排序改为：

```python
negative_trajectory = trajectory[(score).argsort()[0:8]]
positive_trajectory = trajectory[(-score).argsort()[0:8]]
```

### 4.4 为什么需要 config 开关

目标一致性项可能影响绕障多样性：

- 当目标距离很小，它会自然抑制无意义漂移。
- 在复杂绕障场景下，如果权重过大，可能偏好短路径而压制必要绕行。

因此必须由配置控制，并记录到实验配置中，便于比较：

- `enable_goal_consistency_score=False`：纯 critic 排序。
- `enable_goal_consistency_score=True`：critic + 目标一致性排序。

### 4.5 可视化推理也要一致

`BridgeDPTrainer._infer_pred_traj_bridgedp()` 当前只生成单条可视化轨迹，不走多候选排序。第一版可以不加目标一致性项。

如果后续希望训练监控展示和真实推理完全一致，可以把监控推理也改成 `sample_num > 1` 并复用同一个 scoring helper。

## 5. 方案 C：距离分桶监控（只用于诊断）

### 5.1 实现形式

距离分桶只用于观测不同距离区间的训练表现，不参与布朗桥方差、初始化、反向去噪或候选排序。`static/short/mid/long` 只是日志标签，不是模型规则。它同时做两件事：

1. TensorBoard / trainer 日志中记录每个距离桶的 loss 和误差。
2. `traj_batches.jsonl` 中为每个可视化样本写入 `traj_length_m` 和 `length_bucket`，前端后续可以筛选或显示。

第一版先实现日志和 JSONL 元数据，不强制改前端。

### 5.2 新增配置

在 `scripts/train/base_train/configs/bridgedp.py` 增加：

```python
enable_distance_bucket_metrics=True,
distance_bucket_edges=(0.05, 0.5, 0.8),  # 物理米
distance_bucket_names=("static", "short", "mid", "long"),
```

桶定义：

- `static`: `d < 0.05m`
- `short`: `0.05m <= d < 0.5m`
- `mid`: `0.5m <= d < 0.8m`
- `long`: `d >= 0.8m`

### 5.3 Trainer 中记录的指标

在 `BridgeDPTrainer.compute_loss()` 中，使用当前 batch 的监督终点距离：

```python
target_dist_m = torch.norm(ng_x0_target[:, -1, :2], dim=-1) * 5.0
```

对每个 bucket 记录：

```text
bucket/static_count
bucket/static_L_x0
bucket/static_terminal_err_m
bucket/static_path_len_m

bucket/short_count
bucket/short_L_x0
bucket/short_terminal_err_m
bucket/short_path_len_m

bucket/mid_count
bucket/mid_L_x0
bucket/mid_terminal_err_m
bucket/mid_path_len_m

bucket/long_count
bucket/long_L_x0
bucket/long_terminal_err_m
bucket/long_path_len_m
```

其中：

```python
pred_avg = 0.5 * (x0_pred_ng.detach() + x0_pred_mg.detach())
target = ng_x0_target.detach()

point_mse = (pred_avg - target).square().mean(dim=(1, 2))
terminal_err_m = torch.norm(pred_avg[:, -1, :2] - target[:, -1, :2], dim=-1) * 5.0
path_len_m = torch.norm(pred_avg[:, 1:, :2] - pred_avg[:, :-1, :2], dim=-1).sum(dim=-1) * 5.0
```

每个 bucket 内取 mean。若某个 bucket 当前 batch 没有样本，只记录 count=0，其他指标跳过，避免写 NaN。

### 5.4 JSONL 可视化元数据

在 `_write_traj_snapshot()` 里为每个 sample 增加：

```json
{
  "traj_length_m": 0.37,
  "length_bucket": "short"
}
```

来源：

```python
traj_length_m = torch.norm(gt_phys[i, -1, :2]).item()
```

注意 `gt_phys` 已经是物理坐标，因此不需要再乘 5。

### 5.5 后续前端扩展

可选，不作为第一版必须项：

1. 在训练监控面板样本标题显示 `length_bucket` 和 `traj_length_m`。
2. 增加 bucket 下拉筛选，例如只看 `static/short` 样本。
3. 在 loss 面板增加 bucket 指标折线，便于比较不同距离区间是否同步改善。

## 6. 推荐实施顺序

### Step 1：配置参数落地

修改：

- `scripts/train/base_train/configs/bridgedp.py`
- `internnav/model/basemodel/bridgedp/bridgedp_policy.py`
- `internnav/model/basemodel/bridgedp/bridge_scheduler.py`

先只新增参数并保持默认关闭，保证旧实验可复现。

### Step 2：实现尺度相似 + 切法向各向异性 sigma

修改：

- `BridgeScheduler.__init__`
- `BridgeScheduler.trajectory_std()`
- `BridgeScheduler.pointgoal_noise_params()`（新增）
- `BridgeScheduler.sample_pointgoal_bridge_noise()`（新增）
- `BridgeScheduler.add_noise_trajectory()`
- `BridgeScheduler.sample_initial_noise_ordered()`
- `BridgeScheduler.step_trajectory()`

验证点：

1. `goal=(0,0,0)` 时，`sample_initial_noise_ordered()` 输出接近全零。
2. `goal=(0.4m,0,0)` 与 `goal=(0.8m,0,0)` 的法向中段 std 比值约为 1:2。
3. `goal=(1.5m,0,0)` 的法向中段 std 继续按距离线性增长，不被 clamp。
4. 同一 `goal` 下，切向/法向扰动标准差比约为 `bridge_tangent_sigma_ratio / bridge_normal_sigma_ratio`。
5. theta 噪声随 `dist * bridge_theta_sigma_ratio` 连续变化，`dist=0` 时自然为 0。
6. `step_trajectory()` 使用 residual carry 后，旧逐维 sigma 和新切法向 sigma 都能正常去噪。

### Step 3：实现目标一致性排序项

修改：

- `BridgeDPNet.__init__`
- `BridgeDPNet.predict_pointgoal_batch_action_vel()`

建议抽一个私有 helper：

```python
def _apply_goal_consistency_score(self, critic_values, trajectories, goals, origins):
    ...
```

这样后续训练监控或其他推理接口可以复用。

### Step 4：实现距离分桶监控

修改：

- `BridgeDPTrainer.compute_loss()`
- `BridgeDPTrainer._write_traj_snapshot()`

分桶指标只放入 `_monitor_logs`，不参与反向传播。

### Step 5：小 batch 自检

跑一个很小的 batch 或 debug 脚本，检查：

1. `bridge_sigma` 与 `goal_dist` 严格线性相关。
2. `bridge_sigma[..., 0]` 与 `bridge_sigma[..., 1]` 分别能反映切向/法向诊断尺度。
3. `goal_dist=0` 样本初始化不再明显漂移。
4. `bucket/static_count`、`bucket/short_count` 能正常出现。
5. 开关 `enable_goal_consistency_score=False` 时排序结果回到旧逻辑。

## 7. 风险与缓解

### 7.1 比例系数过小导致模型缺少修正空间

风险：

如果 `bridge_normal_sigma_ratio` 或 `bridge_tangent_sigma_ratio` 过小，所有轨迹的候选多样性都会下降，尤其法向绕行自由度会不足。

缓解：

- 优先调大 `bridge_normal_sigma_ratio`，因为法向是主要自由度。
- 切向比例保持较小，避免打乱航点顺序。
- 通过距离分桶指标观察不同距离区间是否同时改善，而不是只优化某个区间。

### 7.2 目标一致性项压制绕行

风险：

如果 `goal_consistency_path_weight` 太大，可能偏好短路径，影响避障。

缓解：

- 默认只轻微惩罚 `path_len - goal_dist`。
- 保留 critic 主导地位。
- 通过 config 开关做 ablation。

### 7.3 theta 维度尺度与 xy 不同

风险：

`theta` 是角度，直接使用距离比例可能过强或过弱。

缓解：

- 用独立的 `bridge_theta_sigma_ratio` 控制。
- 如果发现角度抖动过大，降低 `bridge_theta_sigma_ratio`；如果转向多样性不足，再调大。

### 7.4 各向异性噪声与 eps-consistency 不兼容

风险：

旧的 `L_eps` 写法默认 `bridge_sigma` 是全局逐维标准差，会做 `(residual / sigma)`。切法向噪声启用后，`bridge_sigma` 只是诊断值，不能再表示全局逐维协方差。

缓解：

- 第一版保持 `lambda_eps=0.0`。
- 如果未来需要 eps-consistency，必须在切向/法向局部坐标中计算 `eps_tangent`、`eps_normal`、`eps_theta`。
- `step_trajectory()` 使用 residual carry，避免在反向过程中依赖逐维 `sigma` 反解。

## 8. 建议默认配置

建议第一轮实验使用：

```python
bridge_scale_invariant_sigma=True,
bridge_anisotropic_xy=True,
bridge_normal_sigma_ratio=0.25,
bridge_tangent_sigma_ratio=0.03,
bridge_theta_sigma_ratio=0.05,

enable_goal_consistency_score=True,
goal_consistency_terminal_weight=1.0,
goal_consistency_path_weight=0.2,

enable_distance_bucket_metrics=True,
distance_bucket_edges=(0.05, 0.5, 0.8),
distance_bucket_names=("static", "short", "mid", "long"),
```

如果想保守一些，则：

```python
enable_goal_consistency_score=False
bridge_normal_sigma_ratio=0.15
bridge_tangent_sigma_ratio=0.02
```

## 9. 验收标准

训练 1 到 3 个 epoch 后，重点看：

1. `bucket/static_terminal_err_m` 明显下降。
2. `bucket/short_terminal_err_m` 明显下降。
3. `bucket/long_L_x0` 不明显恶化。
4. 可视化中小距离样本不再出现大范围随机漂移。
5. 初始化样本中切向抖动明显小于法向抖动，航点顺序更稳定。
6. `enable_goal_consistency_score=False/True` 的对比中，True 对小距离样本改善明显，但长距离绕行不明显变差。
