# Bridge-DP 轨迹终点感知修改方案

> **问题**：Bridge-DP 预测轨迹的终点远超真值轨迹终点，模型无法感知"需要走多远"
> **根因**：布朗桥构建方式混淆了"导航目标"与"轨迹终点"，初始化航点时所有 24 点堆在 goal 附近而非有序分布
> **方案**：引入绝对距离上限 `d_max`，将桥终点保持为导航目标，但 24 个航点的初始化与去噪在 `[x₀, x_n]` 区间内有序分布

---

## 一、核心思想

### 1.1 当前问题

```
当前实现：
  桥终点 = 轨迹片段终点（训练时）/ 导航目标（推理时）→ 训练-推理不一致
  初始化 24 航点 = 全部从 goal 附近采样 N(goal, σ²_goal·I) → 所有航点堆在远处
```

### 1.2 目标设计

```
修改后：
  桥终点 = 导航目标（训练和推理都一致）
  轨迹终点 x_n = min(d_max, ||goal - x₀||) 方向上的点
  初始化 24 航点 = 从 x₀ 到 x_n 之间有序插值分布，每个航点方差来自布朗桥公式
```

关键公式：
- `d_max`：从数据集统计标定的单次预测轨迹最大长度（归一化空间中的绝对距离）
- `x_n = x₀ + dir(goal) × min(d_max, ||goal - x₀||)`：合理轨迹终点
- 航点 i 的初始均值：`μ_i = x₀ + (i/23) × (x_n - x₀)` = 起点到 x_n 的线性插值
- 航点 i 在布朗桥上的位置参数：`s_i = ||μ_i - x₀|| / ||goal - x₀||`
- 航点 i 的初始方差：`σ²(s_i; θ_g)` = 布朗桥在 s_i 处的方差

当 `||goal - x₀|| ≤ d_max` 时，`x_n ≈ goal`，轨迹覆盖全程。

---

## 二、需要修改的文件清单

| # | 文件 | 改动类型 | 说明 |
|---|------|---------|------|
| 1 | `scripts/train/base_train/compute_sigma_base.py` | 新增功能 | 增加 `d_max` 计算功能 |
| 2 | `scripts/train/base_train/configs/bridgedp.py` | 配置新增 | 新增 `d_max` 超参 |
| 3 | `internnav/configs/trainer/il.py` | 配置新增 | IlCfg 增加 `d_max` 字段 |
| 4 | `internnav/dataset/bridgedp_lerobot_dataset.py` | 数据集改动 | `point_goal` 改为导航目标，新增 `nav_goal` 逻辑 |
| 5 | `internnav/model/basemodel/bridgedp/bridge_scheduler.py` | 核心改动 | `sample_initial_noise()` 支持有序区间采样 |
| 6 | `internnav/model/basemodel/bridgedp/bridgedp_policy.py` | 核心改动 | 训练/推理时的桥终点和初始化逻辑 |
| 7 | `internnav/trainer/bridgedp_trainer.py` | 可视化适配 | 推理可视化使用新逻辑 |

---

## 三、详细修改方案

### 3.1 Step 0: 数据集统计标定 `d_max`

**文件**: `scripts/train/base_train/compute_sigma_base.py`

在现有的 `compute_sigma_base()` 函数中增加轨迹长度统计功能，或新增一个独立函数 `compute_d_max()`。

**逻辑**：
```python
def compute_d_max(dataset_index, predict_size=24, num_samples=3000,
                  action_scale_xy=5.0, seed=42):
    """统计数据集中单次预测轨迹的长度分布。
    
    轨迹长度定义：起点到终点的直线距离（归一化空间）。
    d_max 取 90 分位数 × 1.2 冗余系数。
    
    返回:
        d_max: 归一化空间中的最大轨迹长度
        distribution_stats: 详细统计信息
    """
    # 遍历数据集，对每条轨迹：
    #   1. 读取 parquet 的 action 列
    #   2. 随机选起点，采样 predict_size 个帧（与 __getitem__ 一致）
    #   3. 转为局部坐标，计算起点到终点的直线距离
    #   4. 归一化（除以 action_scale_xy）
    # 收集所有距离，取 90 分位数 × 1.2
```

**输出示例**：
```json
{
    "d_max_recommended": 0.85,
    "d_max_raw_p90": 0.71,
    "redundancy_factor": 1.2,
    "distance_stats": {
        "mean": 0.42,
        "median": 0.38,
        "p75": 0.56,
        "p90": 0.71,
        "p95": 0.88,
        "max": 1.65
    },
    "num_trajectories": 2850,
    "action_scale_xy": 5.0
}
```

### 3.2 Step 1: 配置新增 `d_max`

**文件**: `internnav/configs/trainer/il.py`

在 `IlCfg` 中新增 `d_max` 字段：

```python
@dataclass
class IlCfg:
    ...
    # Bridge-DP 轨迹长度上限（归一化空间，由 compute_d_max 离线标定）
    d_max: float = 0.85  # 默认值，需根据数据集实际标定
```

**文件**: `scripts/train/base_train/configs/bridgedp.py`

```python
bridgedp_exp_cfg = ExpCfg(
    ...
    il=IlCfg(
        ...
        d_max=0.85,  # 归一化空间中单次预测的最大轨迹直线距离
    ),
)
```

### 3.3 Step 2: 数据集改动 — `point_goal` 改为导航目标

**文件**: `internnav/dataset/bridgedp_lerobot_dataset.py`

**核心改动**：`__getitem__()` 中 `point_goal` 不再使用轨迹片段终点，而是使用整条 episode 的最终目标位置。

当前代码（L653）：
```python
point_goal = target_xyt_actions[-1]  # 轨迹片段终点
```

修改为：
```python
# 导航目标 = 整条 episode 的终点（用 trajectory 最后一帧的位姿）
# 转为局部坐标（相对于 memory_start_choice 帧）
_, nav_goal_local = self.relative_pose(
    trajectory_extrinsics[memory_start_choice][0:3, 0:3],
    trajectory_extrinsics[memory_start_choice][0:3, 3],
    trajectory_extrinsics[-1][0:3, 0:3],
    trajectory_extrinsics[-1][0:3, 3],
    trajectory_base_extrinsic,
)
# 转为 xyt 格式（只取 x, y，theta 用终点方向角）
nav_goal_xyt = np.array([
    nav_goal_local[0], nav_goal_local[1],
    np.arctan2(nav_goal_local[1], nav_goal_local[0])
], dtype=np.float32)

# point_goal 始终是导航目标（与推理时一致）
point_goal = nav_goal_xyt
```

同时保留轨迹终点供加噪使用：
```python
traj_endpoint = target_xyt_actions[-1]  # 轨迹片段的实际终点（用于参考）
```

**归一化逻辑**保持不变：
```python
point_goal[0:2] = point_goal[0:2] / self.action_scale_xy
point_goal[2] = point_goal[2] / self.action_scale_theta
```

**返回值**：增加 `nav_goal`（与 `point_goal` 相同）和 `traj_endpoint_dist`（起点到轨迹终点的距离，用于调试/验证）。

**collate_fn** 对应调整。

### 3.4 Step 3: BridgeScheduler 核心改动 — 有序区间采样

**文件**: `internnav/model/basemodel/bridgedp/bridge_scheduler.py`

#### 3.4.1 新增 `sample_initial_noise_ordered()`

```python
def sample_initial_noise_ordered(
    self,
    goal: torch.Tensor,      # (B, 3) 导航目标，归一化坐标
    origin: torch.Tensor,     # (B, 3) 起点（通常为 0 或接近 0）
    d_max: float,             # 归一化空间中的最大轨迹距离
    shape: tuple,             # (B, T_pred, 3)
    device: torch.device,
) -> torch.Tensor:
    """有序区间初始化：24 个航点从 x₀ 到 x_n 线性分布。
    
    x_n = x₀ + dir(goal) × min(d_max, ||goal - x₀||)
    
    航点 i 的均值 = lerp(x₀, x_n, i/23)
    航点 i 的方差 = σ²(s_i; θ_g)，s_i = ||μ_i - x₀|| / ||goal - x₀||
    
    当 ||goal - x₀|| ≤ d_max 时，x_n ≈ goal，轨迹覆盖全程。
    """
    B, T_pred, dim = shape
    
    # 计算目标方向和距离
    goal_vec = goal - origin  # (B, 3)，如果 origin=0 则 = goal
    goal_dist = goal_vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # (B, 1)
    goal_dir = goal_vec / goal_dist  # (B, 3) 单位方向向量
    
    # 合理轨迹终点距离 = min(d_max, ||goal||)
    traj_dist = torch.clamp(goal_dist, max=d_max)  # (B, 1)
    x_n = origin + goal_dir * traj_dist  # (B, 3) 轨迹终点
    
    # 24 个航点均匀从 origin 到 x_n 插值
    t_traj = torch.linspace(0, 1, T_pred, device=device)  # (T,)
    t_traj = t_traj.view(1, T_pred, 1).expand(B, -1, -1)  # (B, T, 1)
    
    origin_exp = origin.unsqueeze(1).expand(-1, T_pred, -1)  # (B, T, 3)
    x_n_exp = x_n.unsqueeze(1).expand(-1, T_pred, -1)        # (B, T, 3)
    mu = origin_exp + t_traj * (x_n_exp - origin_exp)         # (B, T, 3)
    
    # 每个航点在布朗桥上的位置 s_i = 轨迹进度 × (traj_dist / goal_dist)
    # 即 s_i = (i/23) × (traj_dist / goal_dist)
    s_ratio = (traj_dist / goal_dist).unsqueeze(1)  # (B, 1, 1)
    s_values = t_traj * s_ratio  # (B, T, 1)，每个航点在 [0, s_n] 上的位置
    
    # 计算每个航点的方差（来自布朗桥公式）
    theta_g = torch.atan2(goal_vec[:, 1], goal_vec[:, 0])  # (B,)
    theta_g_exp = theta_g.view(-1, 1, 1)
    sigma_per_point = self.std(s_values, theta_g_exp)  # (B, T, 1)
    
    # 采样
    noise = torch.randn(shape, device=device)
    return mu + sigma_per_point * noise
```

#### 3.4.2 保持 `add_noise()` 兼容

训练时加噪公式保持不变，但 goal 参数现在传入的是导航目标（而非轨迹终点）。
由于真值轨迹 x₀ 的最后一个航点可能不在 goal 上（只覆盖了桥的前段），
桥均值 `(1-t)·x₀ + t·goal` 在 t→1 时会趋向 goal，但 x₀ 的最后几个航点
可能远离 goal——这没问题，因为 x₀-prediction 模式下网络直接预测 x₀ 本身，
加噪只是给 x₀ 加扰动，不需要 x₀ 的终点等于 goal。

**但需要验证一个数值问题**：当 goal 远在 10m 外（归一化后 ~2.0），而 x₀ 的终点在 2m 处（归一化后 ~0.4），
桥均值 `(1-t)·x₀[last] + t·2.0` 在 t=0.5 时为 `0.5·0.4 + 0.5·2.0 = 1.2`。
对于最后一个航点，含噪值的均值在 1.2 处，而真值在 0.4 处，偏差 0.8。
这个偏差被 `σ(t)` 覆盖吗？设 σ_base=0.5, t=0.5, p=0.5: σ ≈ 0.5×0.5 = 0.25。
0.8 / 0.25 = 3.2σ 偏差——偏大但可接受（3σ 覆盖 99.7%）。

**如果数据集中 goal 和 traj_endpoint 的距离差异更大，可能需要增大 sigma_base。**
这可以通过修改 `compute_sigma_base` 脚本来适配——用导航目标而非轨迹终点作为桥终点来计算偏差。

### 3.5 Step 4: BridgeDPNet 改动

**文件**: `internnav/model/basemodel/bridgedp/bridgedp_policy.py`

#### 3.5.1 `__init__()` 新增 `d_max`

```python
self.d_max = il.get('d_max', 0.85)  # 归一化空间中的最大轨迹距离
```

#### 3.5.2 `forward()` 训练时 goal 改为导航目标

当前代码中 `tensor_point_goal` 来自 `batch_pg`，改为导航目标后这里无需变化
（因为数据集已经把 `point_goal` 改成了导航目标）。

需要确认 `sample_bridge_noise()` 中的 goal 传入也是导航目标：
```python
# 当前（无需修改，因为 tensor_point_goal 已是导航目标）：
ng_x0_target, ng_time_embed, ng_noisy_embed, ng_timesteps = self.sample_bridge_noise(
    tensor_label_actions,   # x₀ = 真值轨迹
    tensor_point_goal,      # goal = 导航目标（修改后）
    tensor_theta_g
)
```

#### 3.5.3 `predict_pointgoal_batch_action_vel()` 推理初始化改动

将 `sample_initial_noise()` 替换为 `sample_initial_noise_ordered()`：

```python
# 当前代码（L650-656）替换为：
origin = torch.zeros_like(tensor_point_goal_n)  # 起点 = 原点（机器人当前位置）
naction = self.bridge_scheduler.sample_initial_noise_ordered(
    goal=tensor_point_goal_n.repeat(sample_num, 1),
    origin=origin.repeat(sample_num, 1),
    d_max=self.d_max,
    shape=(sample_num * B, self.predict_size, 3),
    device=self._device,
)
```

去噪循环中的 `endpoint_expanded` 也需要改为导航目标（保持桥的一致性）：
```python
# 去噪时 goal 仍然是导航目标（不是 x_n）
goal_expanded = tensor_point_goal_n.unsqueeze(1).expand(-1, self.predict_size, -1)
goal_expanded = goal_expanded.repeat(sample_num, 1, 1)
```

**去掉方向扰动**（L658-664 的 `dir_bias`），因为有序采样已经自然地提供了方向多样性。
如果仍需要多样性，可以保留但降低权重。

#### 3.5.4 `_infer_pred_traj_bridgedp()` 训练可视化适配

与推理函数同步改动。

### 3.6 Step 5: Trainer 适配

**文件**: `internnav/trainer/bridgedp_trainer.py`

`_infer_pred_traj_bridgedp()` 中的初始化逻辑同步改为 `sample_initial_noise_ordered()`。

---

## 四、关键设计决策记录

### 4.1 为什么用绝对距离 d_max 而非比例 s_n？

比例 s_n 依赖于 goal 距离：当 goal 极远时 s_n 极小，导致：
1. 方差 σ²(s_i) 对所有 s_i 都极小，压缩了探索空间
2. 24 个航点挤在很小的比例区间内，数值精度差

绝对距离 d_max 不依赖 goal 距离，直接反映"机器人物理上一次预测能走多远"，
与数据集样本的实际轨迹长度分布对齐。

### 4.2 为什么训练和推理的 goal 都用导航目标？

- **训练时**：网络通过 `point_encoder(nav_goal)` 获取远方目标信息，
  学会"即使目标在 10m 外，当前预测只走 2m"。布朗桥的方向自适应方差
  也依赖于正确的 θ_g（来自导航目标方向）。
- **推理时**：同上，保持一致。
- **如果训练用轨迹终点、推理用导航目标**：网络在训练中从未见过"goal 远在 10m 外"的场景，
  推理时遇到远目标会出现分布外泛化问题。

### 4.3 方差从布朗桥 s_i 处取值的合理性

当 s_n << 1 时（goal 很远，轨迹只走一小段），所有 s_i ∈ [0, s_n] 都接近 0，
方差 σ²(s_i) ≈ σ²_base · s_i^p 较小。这意味着轨迹初始值紧密围绕均值（直线插值），
是合理的——当目标远在 10m 外时，前 2m 的最优路径通常就是接近直线的。

当 s_n ≈ 1 时（goal 就在附近），s_i 分布在 [0, 1] 上，方差完整发挥，
允许更大的探索自由度——这也是合理的，因为短距离导航需要更灵活的避障。

### 4.4 sigma_base 可能需要重新标定

修改 goal 定义后，`compute_sigma_base.py` 中的"桥均值"也应该用导航目标而非轨迹终点
来计算线性插值基线。这会改变偏差分布，sigma_base 的推荐值可能变大。

---

## 五、实施步骤

```
[ ] Step 0: 扩展 compute_sigma_base.py，增加 d_max 计算功能
            运行标定，获取 d_max 值
[ ] Step 1: IlCfg 和 bridgedp config 新增 d_max 字段
[ ] Step 2: bridgedp_lerobot_dataset.py — point_goal 改为导航目标
[ ] Step 3: bridge_scheduler.py — 新增 sample_initial_noise_ordered()
[ ] Step 4: bridgedp_policy.py — 训练/推理逻辑适配
            - forward() 中确认 goal 为导航目标
            - predict_pointgoal_batch_action_vel() 使用有序采样
            - _normalize/_denormalize 保持不变
[ ] Step 5: bridgedp_trainer.py — 可视化推理适配
[ ] Step 6: 重新运行 compute_sigma_base.py（用导航目标作为桥终点）
[ ] Step 7: 训练验证 — 对比修改前后的可视化轨迹
```

---

## 六、风险评估

| 风险 | 等级 | 缓解措施 |
|------|------|---------|
| 数据集中获取 episode 终点 (`extrinsics[-1]`) 可能有异常值 | 中 | clip 距离到合理范围，增加 safety check |
| sigma_base 需要重新标定 | 低 | 新旧标定脚本可以并行运行 |
| 训练收敛速度可能变慢（goal 更远，学习目标更复杂） | 中 | 可以对 loss 增加距离感知的权重衰减 |
| d_max 取值敏感 | 低 | 数据驱动标定，不靠手动调参 |
| 向后兼容性 | 低 | 旧的 `sample_initial_noise()` 保留不删除 |
