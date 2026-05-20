# Bridge-DP 轨迹拟合监督方案

## 1. 目标

Bridge-DP 的训练目标从“预测航点与真值航点按时间索引逐点对齐”改为“用固定 24 个预测航点拟合整段 GT 轨迹线”。

核心判断：

- Bridge-DP 不应该背诵样本中的第 i 个航点，而应该学会输出一组能表达整条轨迹几何形状的航点。
- 24 个输出航点不包含起点。起点是机器人当前位姿，归一化后固定为 `(0, 0, 0)`，无需预测。
- 监督标签由 GT 轨迹曲线按弧长重采样得到，不再因为短轨迹或末端静止段使用 `valid_mask` 截断。
- 推理时 24 个初始点应铺在“起点 -> 导航目标”的完整区间上，终点就是导航目标，不再使用 `d_max` 截断。

## 2. 当前代码现状

主要相关文件：

- `internnav/dataset/bridgedp_lerobot_dataset.py`
- `internnav/trainer/bridgedp_trainer.py`
- `internnav/model/basemodel/bridgedp/bridgedp_policy.py`
- `internnav/model/basemodel/bridgedp/bridge_scheduler.py`
- `scripts/train/base_train/configs/bridgedp.py`

当前数据链路中：

- Dataset 在 `process_actions()` 中用 `action_indexes = np.clip(...)` 从 GT 轨迹抽固定数量航点。
- 短轨迹或未来片段不足时，`np.clip` 会让后续索引重复落到末端附近。
- `__getitem__()` 中再用相邻位移生成 `valid_mask`，并在 trainer 中屏蔽这些“无效/重复”的末端点。
- 这使训练仍然是点对点 MSE，只是用 mask 回避短轨迹的部分点。

这个逻辑与新的目标不一致。新目标要求所有 24 个输出点都有效，因为它们是对连续轨迹线的重采样控制点。

## 3. 新监督定义

### 3.1 轨迹曲线输入

以 `target_xyt_actions` 作为原始 GT 轨迹：

```python
gt_traj = target_xyt_actions  # (N, 3), columns: x, y, theta
```

其中：

- `x, y` 用于拟合轨迹几何曲线。
- `theta` 来自样本自带朝向，是 Bridge-DP 应学习的主监督角度。

### 3.2 弧长参数化

使用 `x, y` 的累计弧长作为曲线参数：

```python
dxy = np.linalg.norm(gt_traj[1:, :2] - gt_traj[:-1, :2], axis=-1)
s = np.concatenate([[0.0], np.cumsum(dxy)])
s = s / max(s[-1], eps)
```

如果存在连续重复点，需要先去重或合并，保证传给样条的参数严格递增。

### 3.3 24 点重采样

因为输出的 24 个航点不包含起点，所以采样点应为：

```python
q = np.linspace(1.0 / T_pred, 1.0, T_pred)
```

也就是覆盖 `(起点, 终点]`，最后一个监督点严格对应导航目标。

不建议使用 `np.linspace(0.0, 1.0, T_pred)`，否则第 0 个预测航点会被监督到起点，和“起点已知、不预测”的接口语义冲突。

## 4. 样条与角度策略

### 4.1 x/y 曲线

对 `x, y` 使用弧长参数 `s` 做自然三次样条，然后在 `q` 上重采样。实现必须使用 torch 张量在模型所在 device 上执行，避免 SciPy/NumPy 曲线拟合结果再搬运到 GPU：

```python
x_q, y_q = torch_natural_cubic_resample(s, xy, q)
```

第一版采用自然三次样条。若后续发现急转弯过冲，再评估 GPU 版 PCHIP 或弧长线性插值。

### 4.2 theta 主监督

theta 的主监督应来自 GT 样本自带角度，而不是只由几何切线替代：

```python
theta_unwrapped = np.unwrap(gt_traj[:, 2])
theta_q = wrap_to_pi(torch_natural_cubic_resample(s, theta_unwrapped, q))
```

这样 Bridge-DP 学到的是数据中的真实姿态变化。曲线切线方向可以作为：

- GT theta 退化或缺失时的 fallback。
- 调试指标，用于检查 `theta_q` 与路径几何方向是否严重背离。
- 后续可选辅助正则，但不作为第一版主标签。

### 4.3 退化场景

如果有效运动弧长近似为 0：

- `x, y` 直接复制终点或保持全零到终点的线性退化。
- `theta` 复制最后一个有效 GT theta。
- `valid_mask` 仍为全 1，因为标签是“24 个轨迹控制点”，不是“有效运动步”。

如果去重后有效点数不足 4：

- 使用线性插值或低阶插值。
- 不强行 CubicSpline，避免数值过冲和报错。

## 5. Dataset 改动方案

### 5.1 Dataset 只返回 raw 轨迹

为避免 CPU/GPU 混用，Dataset 不执行“样条拟合 + 24 点重采样”。它只返回完整原始轨迹：

```python
raw_pred_actions = target_xyt_actions.astype(np.float32)
raw_augment_actions = augment_xyt_actions.astype(np.float32)
raw_traj_len = raw_pred_actions.shape[0]
```

职责：

- 保留完整 GT x/y/theta 轨迹。
- 保留完整增强轨迹。
- 在 collate 中 pad 为 `batch_raw_labels/batch_raw_augments`。
- 提供 `batch_raw_lengths` 和 `batch_is_task_start`。

### 5.2 标签字段占位

`batch_labels/batch_augments/batch_prior` 保持既有 shape 以兼容 trainer/model 接口，但 Dataset 端只放占位值：

```python
pred_actions = np.zeros((self.predict_size, self.action_dim), dtype=np.float32)
augment_actions = np.zeros((self.predict_size, self.action_dim), dtype=np.float32)
prior_traj = np.zeros((self.predict_size, self.action_dim), dtype=np.float32)
```

Trainer 会在模型 device 上用 raw 轨迹覆盖这些字段。

## 6. Trainer GPU 重采样方案

在 `BridgeDPTrainer` 中新增 torch 实现：

```python
labels_phys = _resample_trajectories_gpu(batch_raw_labels, batch_raw_lengths, T_pred)
augments_phys = _resample_trajectories_gpu(batch_raw_augments, batch_raw_lengths, T_pred)
batch_labels = normalize(labels_phys)
batch_augments = normalize(augments_phys)
batch_prior = _generate_prior_trajectory_gpu(batch_labels, batch_is_task_start)
batch_valid_mask = torch.ones(B, T_pred, device=device)
```

关键点：

- 弧长、自然三次样条求解、theta unwrap/wrap 都用 torch。
- 标签最后一点与 `point_goal` 一致。
- 先验轨迹也跟随 GPU 重采样标签在 GPU 上生成。
- `valid_mask` 固定全 1，仅作为兼容 loss mask。

### 6.1 point_goal 定义

`point_goal` 仍使用完整轨迹终点：

```python
point_goal = target_xyt_actions[-1].astype(np.float32)
```

这样布朗桥终点、训练标签最后一点、导航目标三者一致。

### 6.2 valid_mask 处理

第一版保留字段以兼容 collate 和可视化，但固定为全 1：

```python
valid_mask = np.ones((self.predict_size,), dtype=np.float32)
```

注释需要同步更新：它不再表示“真实运动步”，只表示“监督点参与训练”。

## 7. Trainer Loss 改动方案

第一阶段可以最小改动：

- 保留 `batch_valid_mask` 输入。
- 因 Dataset 已返回全 1，现有 `L_x0/L_delta/L_eps` 自动变为全序列监督。

第二阶段建议清理：

- 移除 action loss 中对 `valid_mask` 的依赖。
- 或将变量改名为 `loss_mask`，默认全 1，仅用于异常样本保护。
- 监控页面中“有效步”概念改成“监督步”，避免继续传达短轨迹截断的旧含义。

## 8. 推理初始化改动方案

### 8.1 删除 d_max 语义

当前 `sample_initial_noise_ordered()` 使用：

```python
x_n = origin + dir(goal) * min(d_max, ||goal-origin||)
```

新方案取消 `d_max`：

```python
x_n = goal
```

24 个初始点应铺在 `(origin, goal]`，不包含起点：

```python
t_traj = torch.linspace(1.0 / T_pred, 1.0, T_pred, device=device)
mu = origin + t_traj * (goal - origin)
```

### 8.2 需要修改的位置

- `bridge_scheduler.py`
  - `sample_initial_noise_ordered()` 删除 `d_max` 参数。
  - 文档和注释从“截断到合理终点”改为“完整起点到导航目标”。
  - 每个航点在完整桥上的位置参数就是 `t_traj`。

- `bridgedp_policy.py`
  - 初始化调用删除 `d_max=self.d_max`。
  - `BridgeDPNet.__init__()` 不再读取 `self.d_max`。
  - 推理注释同步改为“24 点铺在起点到导航目标完整区间上”。

- `bridgedp_trainer.py`
  - `_infer_pred_traj_bridgedp()` 中的可视化推理调用也删除 `d_max`。

- `scripts/train/base_train/configs/bridgedp.py`
  - 删除 `d_max=0.85` 配置。
  - 删除相关注释。

## 9. smooth_trajectory_batch 处理

当前 `smooth_trajectory_batch()` 在相同 24 个节点上求样条值，结果基本等价于恒等映射，不承担“轨迹拟合监督”的职责。

第一版建议：

- 推理输出不再调用 `smooth_trajectory_batch()`。
- 直接返回 denormalize 后的 24 点。
- 保留函数本身，作为后续“高频控制点上采样”的工具函数候选。

需要修改：

```python
naction = self._denormalize_action(naction)
trajectory = naction
```

替代：

```python
naction = self._denormalize_action(naction)
trajectory = smooth_trajectory_batch(naction)
```

## 10. 训练与推理一致性

新方案下应满足：

- 训练标签：24 个弧长重采样未来点，覆盖 `(origin, goal]`。
- 推理初始化：24 个布朗桥初始点，覆盖 `(origin, goal]`。
- 布朗桥终点：导航目标。
- 标签最后一点：导航目标。
- 输出接口：24 个预测未来点，不包含起点。

这比旧版更一致，也不需要用 `valid_mask` 解释短轨迹末端重复点。

## 11. 风险与应对

### 11.1 样条过冲

三次样条可能在急转弯或稀疏点上过冲。

应对：

- 先实现 CubicSpline 版本。
- 可视化检查过冲样本。
- 若过冲明显，将 x/y 插值替换为 PCHIP 或弧长线性插值。

### 11.2 theta 与路径切线不一致

GT theta 来自样本自带角度，可能和 x/y 曲线切线有差异。

应对：

- 第一版以 GT theta 为主监督。
- 同时在调试中记录 tangent theta 与 GT theta 的差异。
- 不把 tangent theta 直接覆盖为标签，避免丢失样本真实姿态信息。

### 11.3 critic 仍使用旧 action_indexes

第一版中 critic 可暂时沿用 `action_indexes` 对障碍距离评分。

应对：

- 主训练目标先完成曲线化。
- 后续再考虑 critic 是否也基于重采样后的轨迹点计算。

### 11.4 配置兼容

旧 checkpoint/config 可能仍包含 `d_max`。

应对：

- 模型读取端不再依赖 `d_max`。
- 保证配置里残留 `d_max` 不会导致报错。
- 新配置删除 `d_max`。

## 12. 验证清单

- Dataset 单样本检查：
  - `batch_labels.shape == (B, 24, 3)`。
  - `batch_labels[:, -1, :2]` 与 `batch_pg[:, :2]` 一致或非常接近。
  - `batch_valid_mask` 全 1。

- 可视化检查：
  - 短轨迹样本不再出现大量被截断的后半段。
  - GT 显示为 24 个沿整条路径均匀分布的点。
  - 预测轨迹应贴近路径形状，而不是只贴近局部时间索引。

- 训练指标：
  - `loss/L_x0` 稳定下降。
  - `loss/L_delta` 不应异常增大。
  - 去掉 `smooth_trajectory_batch()` 后可视化预测点不应出现明显锯齿失控。

- 推理检查：
  - PointGoal 推理不再访问 `self.d_max`。
  - `sample_initial_noise_ordered()` 不再需要 `d_max` 参数。
  - 24 个初始点从起点附近逐步铺到导航目标，且第 24 个点靠近目标。

## 13. 建议实施顺序

1. Dataset 返回 raw 轨迹、raw length 和 task-start 标记。
2. Trainer 新增 GPU 弧长自然三次样条重采样，并覆盖 `batch_labels/batch_augments`。
3. Trainer 在 GPU 上生成 `batch_prior`，并将 `valid_mask` 固定为全 1。
4. 修改 `sample_initial_noise_ordered()`，删除 `d_max` 和截断逻辑。
5. 修改所有调用方，删除 `d_max` 参数和配置读取。
6. 暂停使用 `smooth_trajectory_batch()`，推理直接返回 24 个预测点。
7. 跑一个小 batch 数据检查，再启动短训练观察可视化和 loss。
