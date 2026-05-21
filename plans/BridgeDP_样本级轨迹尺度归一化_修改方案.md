# Bridge-DP 样本级轨迹尺度归一化修改方案

## 1. 结论

这个方案是可行的，而且和 Bridge-DP 当前的目标很契合：让模型主要学习“轨迹形状”，而不是同时承担“轨迹形状 + 轨迹物理长度”的尺度混合建模。

核心思想是把每条有效 PointGoal 轨迹映射到统一尺度的形状空间：

```python
d_m = sqrt(x_n ** 2 + y_n ** 2)
R = trajectory_norm_target_distance  # 默认 2.0

x_i_shape = x_i_m * R / d_m
y_i_shape = y_i_m * R / d_m
theta_i_norm = theta_i_rad / pi
```

这样起点仍为 `(0, 0, 0)`，终点 xy 距离固定为 `R`。默认 `R=2.0`，不是写死为 `1.0`。整条 xy 轨迹只做等比例缩放，所以不会改变路径形状，也不会改变 `x/y` 比例。

反归一化必须使用同一个样本的距离因子：

```python
x_i_m = x_i_shape * d_m / R
y_i_m = y_i_shape * d_m / R
theta_i_rad = theta_i_norm * pi
```

只要训练、推理、可视化都保存并使用同一个 `d_m / R`，归一化前后不会产生位置漂移或尺度失真。

## 2. 边界规则

### 2.1 极短轨迹不训练

新增物理距离阈值：

```python
trajectory_norm_min_distance_m = 0.10
```

若 `d_m < trajectory_norm_min_distance_m`，该样本视为已经到达，不参与 Bridge-DP 轨迹生成训练。原因是这类样本没有稳定的可学习路径形状，强行归一化会把几厘米的数值误差放大成一条长轨迹，反而污染监督。

推理时同样使用该阈值。若输入 PointGoal 的 `sqrt(x^2 + y^2) < 0.10m`，应直接判定到达，而不是继续采样轨迹。

### 2.2 有效轨迹都要学习

只要 `d_m >= trajectory_norm_min_distance_m`，不论物理距离是 12cm、30cm、1m 还是 5m，都作为有效样本训练。Bridge-DP 学的是归一化后的 24 个航点如何拟合目标轨迹形状，而不是因为物理距离短就降低其训练地位。

### 2.3 xy 不再叠加固定 `/5`

开启样本级轨迹归一化后，xy 的模型空间定义改为：

```python
xy_shape = xy_m * R / d_m
```

因此不能再对 xy 额外执行固定 `/5.0`，否则默认 `R=2.0` 会被再次缩小到 `0.4`，破坏“终点距离固定为 R”的目标。

`theta` 仍沿用现有角度归一化：

```python
theta_norm = theta_rad / 3.14159
```

旧的 `xy / 5.0` 只保留为 legacy 模式或 NoGoal 专用路径。

## 3. 新增配置

在 `scripts/train/base_train/configs/bridgedp.py` 的 Bridge-DP 超参区增加：

```python
# PointGoal 样本级轨迹尺度归一化：让所有有效轨迹在模型空间具有统一终点距离。
enable_trajectory_normalization=True,
trajectory_norm_target_distance=2.0,
trajectory_norm_min_distance_m=0.10,
trajectory_norm_eps=1e-6,
drop_short_trajectory_samples=True,
max_short_trajectory_resample_attempts=20,
```

字段含义：

- `enable_trajectory_normalization`：是否启用样本级轨迹尺度归一化。
- `trajectory_norm_target_distance`：归一化后起点到终点的 xy 距离，默认 `2.0`。
- `trajectory_norm_min_distance_m`：小于该物理距离视为已到达，默认 `0.10m`。
- `trajectory_norm_eps`：防止除零的数值保护；正常有效样本不会用它替代阈值判断。
- `drop_short_trajectory_samples`：训练时短轨迹不参与 loss。
- `max_short_trajectory_resample_attempts`：Dataset 采样到短轨迹时可尝试重新抽样，避免 batch 内有效样本过少。

## 4. 训练数据流修改

### 4.1 当前链路

当前 `BridgeDP_Base_Dataset` 已经返回 raw 轨迹：

- `batch_raw_labels`
- `batch_raw_augments`
- `batch_raw_lengths`

Trainer 在 `_prepare_curve_supervision()` 中用 raw 轨迹做 GPU 三次样条重采样，再用 `_normalize_action_tensor()` 做固定 `/5` 和 `/pi`。

### 4.2 修改后链路

推荐把轨迹归一化放在三次样条拟合之前，流程如下：

1. Dataset 继续返回局部物理坐标 raw 轨迹，单位为米/弧度。
2. Trainer 读取每条 raw 轨迹的终点：

```python
end_xy = raw_labels[bid, raw_len - 1, :2]
d_m = torch.norm(end_xy)
```

3. 若 `d_m < trajectory_norm_min_distance_m`：

- 优先在 Dataset 中重新抽样一个有效片段。
- 若多次重试仍失败，则标记 `sample_valid=False`，Trainer 对该样本所有训练项置零权重。

4. 对有效样本，在三次样条拟合前缩放 raw 轨迹：

```python
scale_to_shape = R / d_m
raw_labels_shape[..., :2] = raw_labels[..., :2] * scale_to_shape
raw_augments_shape[..., :2] = raw_augments[..., :2] * scale_to_shape
raw_labels_shape[..., 2] = raw_labels[..., 2] / pi
raw_augments_shape[..., 2] = raw_augments[..., 2] / pi
```

5. 对 `raw_labels_shape` 和 `raw_augments_shape` 做 GPU 弧长三次样条重采样，得到固定 24 点监督：

```python
batch_labels = resample(raw_labels_shape)
batch_augments = resample(raw_augments_shape)
```

6. 覆盖 `batch_pg`，保证 point-goal 与重采样监督末端一致：

```python
batch_pg[:, :2] = batch_labels[:, -1, :2]  # xy 距离应约为 R
batch_pg[:, 2] = batch_labels[:, -1, 2]
```

7. 保存每个样本的反归一化尺度：

```python
batch_traj_distance_m = d_m
batch_traj_denorm_scale = d_m / R
```

8. 先验轨迹 `batch_prior` 基于归一化后的 `batch_labels` 生成，因此它也处于同一个形状空间。

## 5. Model 与 Scheduler 影响

### 5.1 BridgeScheduler

PointGoal 的 `goal` 已经在形状空间中，终点距离固定为 `R`。在当前 `bridge_scale_invariant_sigma=True` 下：

```python
dist = norm(goal_xy - origin_xy)  # 约等于 R
sigma_normal = dist * bridge_normal_sigma_ratio * shape_tau
sigma_tangent = dist * bridge_tangent_sigma_ratio * shape_tau
sigma_theta = dist * bridge_theta_sigma_ratio * shape_tau
```

因此所有有效轨迹的布朗桥均值形状一致，布朗桥噪声形状也一致。这正是该方案期望的行为：不再区分“短轨迹小噪声、长轨迹大噪声”，而是所有可学习轨迹都在统一形状空间中建模。

`bridge_scheduler.py` 可以少改或不改，重点是确保传入的 PointGoal、labels、prior 均已处于新形状空间。

### 5.2 BridgeDPNet

需要把 `_normalize_action()` / `_denormalize_action()` 拆成两个模式：

```python
legacy_normalize_action(action):
    xy = xy / 5.0
    theta = theta / pi

trajectory_normalize_action(action, distance_m, target_R):
    xy = xy * target_R / distance_m
    theta = theta / pi

trajectory_denormalize_action(action, distance_m, target_R):
    xy = xy * distance_m / target_R
    theta = theta * pi
```

推理时 PointGoal 使用 `trajectory_normalize_action()`；NoGoal 暂时保持原有 legacy 空间，因为 NoGoal 没有明确终点距离可用于样本级归一化。

## 6. 推理流程修改

PointGoal 推理入口收到物理坐标 `goal_point=(x, y, theta)`：

1. 计算物理距离：

```python
d_m = torch.norm(goal_point[:, :2], dim=-1)
```

2. 若 `d_m < trajectory_norm_min_distance_m`：

- 推荐在上层导航逻辑直接返回 arrived。
- 为兼容当前 `predict_pointgoal_batch_action_vel()` 返回格式，policy 内可返回全零轨迹，并可选增加 `return_arrival=True` 时返回 `arrived_mask`。

3. 对有效目标归一化：

```python
goal_shape[:, :2] = goal_point[:, :2] * R / d_m
goal_shape[:, 2] = goal_point[:, 2] / pi
```

4. 若有 `prior_traj`，使用同一个 `d_m` 和 `R` 做归一化：

```python
prior_shape[..., :2] = prior_traj[..., :2] * R / d_m
prior_shape[..., 2] = prior_traj[..., 2] / pi
```

5. BridgeScheduler 在形状空间采样和去噪。

6. 输出轨迹反归一化：

```python
trajectory_m[..., :2] = trajectory_shape[..., :2] * d_m / R
trajectory_m[..., 2] = trajectory_shape[..., 2] * pi
```

这样模型预测的形状会根据当前目标距离恢复到真实物理尺度。

## 7. Loss 与监控修改

### 7.1 Loss mask

新增 `batch_sample_valid`，形状 `(B,)`。短轨迹样本不参与：

- `L_x0`
- `L_delta`
- `L_eps`
- `critic_loss`
- `aux_loss`

若一个 batch 内全是无效样本，返回一个与模型参数相关联的零 loss，避免 DDP/AMP 报错，同时记录 `short_skipped_count`。

### 7.2 日志指标

新增监控项：

- `traj_norm/target_distance`：配置的 `R`。
- `traj_norm/valid_count`：当前 batch 有效样本数。
- `traj_norm/skipped_short_count`：短轨迹跳过数。
- `traj_norm/mean_distance_m`：有效样本原始物理终点距离均值。
- `traj_norm/mean_endpoint_dist_shape`：归一化后终点距离，应接近 `R`。
- `traj_norm/roundtrip_xy_error_m`：归一化再反归一化后的最大/均值误差。

原有距离分桶仍可保留，但分桶依据应使用 `batch_traj_distance_m`，而不是形状空间中的终点距离，因为后者恒为 `R`。

## 8. 可视化与反归一化

当前 `_denorm_batch()` 只能做固定：

```python
xy *= 5.0
theta *= pi
```

需要新增基于样本尺度的反归一化：

```python
def _denorm_shape_batch(batch_tensor, traj_distance_m, target_R):
    scale = (traj_distance_m / target_R).view(B, 1, 1)
    out[..., :2] = batch_tensor[..., :2] * scale
    out[..., 2] = batch_tensor[..., 2] * pi
    return out
```

对于 `(B, 3)` 的 goal，也要支持 `scale.view(B, 1)`。

训练可视化写 JSONL 时：

- `gt_phys = _denorm_shape_batch(batch_labels, batch_traj_distance_m, R)`
- `prior_phys = _denorm_shape_batch(batch_prior, batch_traj_distance_m, R)`
- `pred_phys = _denorm_shape_batch(pred_shape, batch_traj_distance_m, R)`
- `nav_goal_phys = _denorm_shape_batch(batch_pg, batch_traj_distance_m, R)`

这样前端看到的仍然是米/弧度坐标，且不会因为形状空间训练产生尺度误差。

## 9. 需要修改的文件

### 9.1 `scripts/train/base_train/configs/bridgedp.py`

增加第 3 节列出的配置项。

### 9.2 `internnav/trainer/bridgedp_trainer.py`

重点修改：

1. 新增配置读取。
2. 新增 `_compute_trajectory_norm_factors()`。
3. 新增 `_apply_trajectory_normalization_raw()`。
4. 修改 `_prepare_curve_supervision()`：
   - raw 轨迹先做样本级轨迹归一化；
   - 再做三次样条重采样；
   - 覆盖 `batch_pg`；
   - 写入 `batch_traj_distance_m`、`batch_traj_denorm_scale`、`batch_sample_valid`。
5. 修改 `compute_loss()`，用 `batch_sample_valid` 控制样本级 loss。
6. 修改 `_distance_bucket_logs()`，距离从 `batch_traj_distance_m` 读取。
7. 替换 `_denorm_batch()` 或新增 `_denorm_shape_batch()` 用于可视化。

### 9.3 `internnav/dataset/bridgedp_lerobot_dataset.py`

建议修改：

1. 保留 raw 物理轨迹输出。
2. Dataset 端在采样片段后先检查 `raw_pred_actions[-1, :2]` 距离。
3. 如果距离小于 `trajectory_norm_min_distance_m`，尝试重新采样，最多 `max_short_trajectory_resample_attempts` 次。
4. 若仍失败，返回 `sample_valid=False`，由 Trainer 跳过。

如果为了少改 Dataset，也可以第一阶段只在 Trainer 端跳过短轨迹；但长期建议 Dataset 端重采样，避免 batch 有效样本数过低。

### 9.4 `internnav/model/basemodel/bridgedp/bridgedp_policy.py`

重点修改：

1. 读取新配置。
2. 新增 PointGoal 轨迹归一化/反归一化 helper。
3. `predict_pointgoal_batch_action_vel()`：
   - 先判断 arrived；
   - 对 goal/prior 做样本级轨迹归一化；
   - 去噪后按 `d_m / R` 反归一化。
4. `_infer_pred_traj_bridgedp()` 或 Trainer 内推理可视化也要使用同一反归一化尺度。
5. NoGoal 暂不套用样本级轨迹归一化。

### 9.5 `internnav/model/basemodel/bridgedp/bridge_scheduler.py`

理论上不需要大改。只需检查：

1. PointGoal 的 `goal` 和 `origin` 都来自形状空间。
2. `bridge_scale_invariant_sigma=True` 时 `dist≈R`，符合统一桥形状目标。
3. 日志中若输出 sigma 物理量，需要乘 `d_m / R` 才是米制尺度。

## 10. 验证计划

### 10.1 单元验证

1. 轨迹 `(0,0)->(3,4)`，`R=2`：
   - `d_m=5`
   - 归一化终点应为 `(1.2, 1.6)`
   - 终点距离应为 `2`
   - 反归一化后应恢复 `(3,4)`。

2. 任意中间点 `(x_i, y_i)`：
   - `x_i / y_i` 比例保持。
   - 归一化再反归一化误差小于 `1e-5m`。

3. `d_m < 0.10m`：
   - 不执行除法。
   - 训练 loss 不包含该样本。
   - 推理直接返回 arrived 或全零轨迹。

### 10.2 训练监控

启动训练后检查：

1. `traj_norm/mean_endpoint_dist_shape` 接近 `2.0`。
2. `traj_norm/roundtrip_xy_error_m` 接近 0。
3. `traj_norm/skipped_short_count` 合理，不应大面积吞掉数据。
4. 轨迹面板中的 GT、pred、prior、nav_goal 坐标都回到米制空间，目标点和轨迹终点没有漂移。

### 10.3 Scheduler 验证

固定 `R=2`，不同物理距离样本进入 scheduler 后：

1. `bridge_mean_ordered()` 的终点距离均为 `2`。
2. `pointgoal_noise_params()` 中 `dist` 均约为 `2`。
3. 中段 `sigma_normal / R`、`sigma_tangent / R` 在不同物理距离样本上一致。

## 11. 风险与注意事项

1. 这是一次空间定义变更，开启后旧 checkpoint 与新训练空间不兼容，建议重新训练。
2. 必须避免 xy 同时执行样本级归一化和固定 `/5`，否则尺度会被双重缩小。
3. `batch_pg` 必须由归一化后的标签末端覆盖，避免 Dataset 旧逻辑和 Trainer 新逻辑不一致。
4. 距离分桶、可视化和物理误差指标必须使用原始 `d_m` 或反归一化后的米制轨迹。
5. NoGoal 没有明确目标距离，第一阶段建议保持原实现；后续若要统一，也需要单独定义 NoGoal 的虚拟尺度。

## 12. 推荐实施顺序

1. 先在 Trainer 中实现样本级轨迹归一化、反归一化可视化和 roundtrip 日志。
2. 再修改 PointGoal 推理入口，确保外部物理 goal 能进入形状空间并反归一化输出。
3. 然后补 Dataset 端短轨迹重采样，减少无效样本进入 batch。
4. 最后清理旧的 `/5` 调用路径，保留 legacy helper 但确保 PointGoal 新路径不会误用。

## 13. 当前实现状态

本次实现已经完成：

1. 配置默认开启样本级轨迹归一化，`trajectory_norm_target_distance=2.0`，`trajectory_norm_min_distance_m=0.10`。
2. Trainer 在三次样条重采样前对 raw xy 轨迹执行 `xy * R / d_m`，重采样后对 theta 执行 `/pi`。
3. `d_m < 0.10m` 的样本在 Trainer 端标记为无效，不参与 action、critic、aux loss。
4. `batch_pg` 被重采样后的归一化标签末端覆盖，确保目标 token 和监督终点一致。
5. 训练可视化、预测轨迹、GT、prior、nav_goal 均按 `xy * d_m / R` 和 `theta * pi` 反归一化回物理坐标。
6. PointGoal 推理入口对物理 goal/prior 做同样的样本级轨迹归一化；若目标距离小于 10cm，直接返回零轨迹。
7. NoGoal 分支保持 legacy `xy / 5.0` 训练空间，避免无目标推理时缺少 `d_m` 而无法反归一化；PointGoal/MixedGoal 分支使用新的形状空间。

尚未做 Dataset 端重新抽样。当前短轨迹样本会进入 batch，但被 Trainer loss mask 跳过；如果后续发现 `skipped_short_count` 较高，再把重采样逻辑下沉到 Dataset，减少无效 batch 占比。
