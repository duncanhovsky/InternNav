# Bridge-DP 尺度条件 Token 与双尺度深度修改方案

## 1. 背景

样本级轨迹尺度归一化后，PointGoal 与监督轨迹进入“形状空间”：

```python
xy_shape = xy_m * R / d_m
```

其中 `d_m = sqrt(x_n^2 + y_n^2)`，`R = trajectory_norm_target_distance`，默认 `2.0`。

这会让所有有效样本的目标终点距离在模型空间中接近 `R`，有利于 Bridge-DP 学习统一尺度的轨迹形状。但当前深度图仍是米制：

```python
depth_m = raw_depth / 10000.0
```

因此模型同时看到：

- `goal / trajectory`：形状空间。
- `depth`：物理米制空间。

如果不把 `d_m` 或 `R / d_m` 显式交给模型，模型无法稳定知道“1 米深度”在当前形状空间中对应多少归一化距离。

## 2. 目标

本方案采用“方向 1 + 第三方向”的组合：

1. 显式增加尺度条件 token，让模型知道当前样本的物理距离尺度。
2. 保留原始米制深度，用于真实几何、安全距离和预训练 depth backbone 输入分布。
3. 额外提供 shape-space 深度，让深度障碍信息与归一化后的 PointGoal/轨迹空间直接对齐。
4. 训练、推理、可视化使用同一套 `d_m` 和 `R`，避免尺度漂移。

## 3. 核心设计

### 3.1 尺度条件特征

对每个有效 PointGoal 样本计算：

```python
d_m = norm(goal_xy_m)
R = trajectory_norm_target_distance
scale_to_shape = R / clamp(d_m, eps)
scale_to_metric = d_m / R
```

推荐输入给模型的尺度特征为：

```python
scale_feat = [
    d_m,
    log(d_m),
    scale_to_shape,   # R / d_m
    scale_to_metric,  # d_m / R
]
```

原因：

- `d_m` 给出真实物理目标距离。
- `log(d_m)` 对长短距离更平滑。
- `R / d_m` 是把米制深度映射到形状空间的直接比例。
- `d_m / R` 是把模型输出恢复到米制空间的比例。

这些特征比单独给 `d_m` 更容易被网络学习，因为模型不需要自己从零近似“除法关系”。

### 3.2 双尺度深度

保留现有米制深度：

```python
depth_metric = depth_m
```

新增形状空间深度：

```python
depth_shape = depth_m * scale_to_shape
```

这两个深度的含义不同：

- `depth_metric`：真实米制几何，保留绝对距离感。
- `depth_shape`：与 `goal_shape`、`trajectory_shape` 同尺度，便于模型判断障碍物位于归一化轨迹的哪个相对位置。

例子：

```text
R = 2

样本 A: d_m = 0.5m, depth_m = 1.0m
depth_shape = 1.0 * 2 / 0.5 = 4.0
障碍物在目标之后。

样本 B: d_m = 5.0m, depth_m = 1.0m
depth_shape = 1.0 * 2 / 5.0 = 0.4
障碍物在目标之前。
```

这样模型不需要单靠 attention 自己推理“1 米在当前样本里到底近还是远”。

## 4. 推荐架构

### 4.1 新增 ScaleEncoder

在 `BridgeDPNet.__init__` 中新增：

```python
self.scale_encoder = nn.Sequential(
    nn.Linear(4, self.token_dim),
    nn.GELU(),
    nn.Linear(self.token_dim, self.token_dim),
)
self.n_scale_tokens = 1
```

输入 `(B, 4)`，输出 `(B, 1, token_dim)`：

```python
scale_token = self.scale_encoder(scale_feat).unsqueeze(1)
```

memory 从当前：

```text
[time, goal x3, rgbd, prior]
```

变为：

```text
[time, scale, goal x3, rgbd, prior]
```

因此 `cond_len` 需要增加 `n_scale_tokens`：

```python
cond_len = memory_size * 16 + 4 + n_prior_tokens + n_scale_tokens
```

对应所有 `cond_pos_embed`、`cond_critic_mask`、memory 拼接位置都要同步更新。

### 4.2 双尺度 RGBD 编码的推荐实现

不要一开始改 Dataset 的深度输出格式。Dataset 继续返回米制 `batch_depth`。原因是：

1. 当前 `RGBDBackbone` 已经假设 depth 是米制。
2. Trainer/Policy 才知道 `d_m` 和 `R`，更适合在那里生成 `depth_shape`。
3. 避免把样本级 scale 逻辑散落到 Dataset。

在模型 forward / inference 中生成：

```python
depth_metric = input_depths
depth_shape = input_depths * scale_to_shape.view(B, 1, 1, 1)
```

如果 `input_depths` 是 `(B, H, W, 1)`，scale reshape 为 `(B, 1, 1, 1)`。

如果未来有 `(B, T, H, W, 1)`，scale reshape 为 `(B, 1, 1, 1, 1)`。

### 4.3 双尺度深度 token 的三种实现层级

#### 方案 A：只加 Scale Token

不新增 `depth_shape` 编码，只把 `scale_token` 放进 memory。

优点：

- 改动最小。
- 不改变 RGBDBackbone 输入分布。
- 训练稳定性最高。

缺点：

- 模型需要通过 attention 自己学习 `depth_m * R / d_m` 的隐式关系。

该方案适合作为第一阶段。

#### 方案 B：共享 RGBDBackbone，两次编码

分别编码：

```python
rgbd_metric = self.rgbd_encoder(input_images, depth_metric)
rgbd_shape = self.rgbd_encoder(input_images, depth_shape)
```

然后用一个融合层压回原 token 数：

```python
rgbd_embed = self.rgbd_scale_fuse(torch.cat([rgbd_metric, rgbd_shape], dim=-1))
```

其中：

```python
self.rgbd_scale_fuse = nn.Linear(2 * token_dim, token_dim)
```

优点：

- 保留 memory token 数量不变。
- `rgbd_shape` 直接与归一化轨迹空间对齐。

缺点：

- RGBDBackbone 计算量约翻倍。
- `depth_shape` 的数值分布可能超出现有 depth backbone 习惯范围，需要 clamp。

#### 方案 C：metric token + shape token 都进入 memory

编码：

```python
rgbd_metric = self.rgbd_encoder(input_images, depth_metric)
rgbd_shape = self.rgbd_encoder(input_images, depth_shape)
```

memory 拼接：

```text
[time, scale, goal x3, rgbd_metric, rgbd_shape, prior]
```

优点：

- 信息最完整。
- Transformer 可以自由决定使用 metric depth 还是 shape depth。

缺点：

- memory 长度增加一倍 RGBD token。
- 计算和显存压力最大。
- `cond_pos_embed`、critic mask、attention 成本都增加。

不建议第一阶段直接采用，除非训练资源充足。

## 5. 推荐落地路线

### 5.1 第一阶段：Scale Token 必做

目标是补上当前最缺的尺度桥梁。

新增配置：

```python
enable_scale_condition_token=True
scale_condition_features=("distance", "log_distance", "scale_to_shape", "scale_to_metric")
scale_condition_clamp_min_m=0.10
scale_condition_clamp_max_m=20.0
```

训练端：

- Trainer 已有 `batch_traj_distance_m`。
- 调用 model forward 时传入 `traj_distance_m`。

模型端：

- `BridgeDPNet.forward(..., traj_distance_m=None)`。
- PointGoal/MixedGoal memory 使用 `scale_token`。
- NoGoal 使用默认 scale token，可设为：

```python
d_m = action_scale_xy * nogoal_front_distance  # 例如 5m * 0.8 = 4m
```

或者使用全零 scale token，但推荐使用默认前向距离，避免 NoGoal 和 PointGoal memory 分布差异过大。

推理端：

- PointGoal 用输入 goal 的 `goal_distance_m` 构建 scale token。
- 小于 `trajectory_norm_min_distance_m` 时仍直接 arrived。

### 5.2 第二阶段：增加 shape depth，采用融合式方案 B

新增配置：

```python
enable_shape_depth=True
shape_depth_fusion="concat_linear"
shape_depth_clamp_max=5.0
```

生成：

```python
depth_shape = input_depths * scale_to_shape
depth_shape = depth_shape.clamp(min=0.0, max=shape_depth_clamp_max)
```

然后：

```python
rgbd_metric = self.rgbd_encoder(input_images, input_depths)
rgbd_shape = self.rgbd_encoder(input_images, depth_shape)
rgbd_embed = self.rgbd_scale_fuse(torch.cat([rgbd_metric, rgbd_shape], dim=-1))
```

第一版不建议把 `depth_shape` 直接替换原深度，也不建议只使用 `depth_shape`。保留 `depth_metric` 可以避免丢掉真实物理距离感。

## 6. 代码修改点

### 6.1 `scripts/train/base_train/configs/bridgedp.py`

新增：

```python
enable_scale_condition_token=True,
scale_condition_clamp_min_m=0.10,
scale_condition_clamp_max_m=20.0,

enable_shape_depth=False,  # 第一阶段先 False，第二阶段打开
shape_depth_fusion="concat_linear",
shape_depth_clamp_max=5.0,
```

### 6.2 `internnav/trainer/bridgedp_trainer.py`

当前已有：

```python
inputs_on_device["batch_traj_distance_m"]
```

需要把它传入模型：

```python
model(..., traj_distance_m=inputs_on_device["batch_traj_distance_m"])
```

训练可视化推理 `_infer_pred_traj_bridgedp()` 也要用同一个 `batch_traj_distance_m` 构建 scale token。

### 6.3 `internnav/model/basemodel/bridgedp/bridgedp_policy.py`

新增：

```python
self.enable_scale_condition_token
self.scale_encoder
self.n_scale_tokens
self.enable_shape_depth
self.rgbd_scale_fuse
```

新增 helper：

```python
def _build_scale_features(self, traj_distance_m):
    d = traj_distance_m.clamp(min=min_m, max=max_m)
    return torch.stack([
        d,
        torch.log(d),
        self.trajectory_norm_target_distance / d,
        d / self.trajectory_norm_target_distance,
    ], dim=-1)

def _build_scale_token(self, traj_distance_m):
    return self.scale_encoder(self._build_scale_features(traj_distance_m)).unsqueeze(1)
```

新增 depth helper：

```python
def _shape_depth(self, depths, traj_distance_m):
    scale = self.trajectory_norm_target_distance / traj_distance_m.clamp(min=eps)
    while scale.dim() < depths.dim():
        scale = scale.view(*scale.shape, 1)
    return (depths * scale).clamp(0.0, self.shape_depth_clamp_max)
```

注意 reshape 要分别支持 `(B,H,W,1)` 和 `(B,T,H,W,1)`。

### 6.4 `cond_len` 和 mask

当前：

```python
cond_len = memory_size * 16 + 4 + n_prior_tokens
```

第一阶段改为：

```python
cond_len = memory_size * 16 + 4 + n_prior_tokens + n_scale_tokens
```

memory 顺序建议固定为：

```text
0: time
1: scale
2-4: goal tokens
5 ...: rgbd tokens
last: prior tokens
```

critic mask 要同步调整：

- critic 仍应屏蔽 time、scale、goal tokens。
- critic 仍屏蔽 prior tokens。
- critic 保留 rgbd tokens。

也就是：

```python
self.cond_critic_mask[:, 0:5] = -inf
self.cond_critic_mask[:, 5 + memory_size * 16:] = -inf
```

若采用方案 C 双 RGBD token，则第二段起点要改为 `5 + memory_size * 16 * 2`。

## 7. 训练与推理数据流

### 7.1 训练 PointGoal/MixedGoal

```text
raw physical trajectory
  -> d_m
  -> xy_shape = xy_m * R / d_m
  -> pointgoal_shape
  -> scale_feat(d_m, R/d_m, ...)
  -> depth_metric + optional depth_shape
  -> Bridge-DP decoder
  -> predict x0_shape
```

### 7.2 推理 PointGoal

```text
physical goal
  -> if d_m < 0.10m: arrived
  -> goal_shape = goal_m * R / d_m
  -> scale_feat(d_m)
  -> depth_metric + optional depth_shape
  -> denoise in shape space
  -> trajectory_m = trajectory_shape * d_m / R
```

### 7.3 NoGoal

NoGoal 没有真实 `d_m`，第一阶段建议：

- 轨迹空间继续保持 legacy `xy / 5.0`。
- scale token 使用默认前向距离：

```python
default_nogoal_distance_m = nogoal_front_distance * action_scale_xy
```

- 不启用 `depth_shape`，或使用 `default_nogoal_distance_m` 生成 shape depth。

这样 NoGoal 不会因为缺少目标距离而引入不稳定尺度。

## 8. 监控指标

新增：

```text
scale/distance_m_mean
scale/distance_m_min
scale/distance_m_max
scale/scale_to_shape_mean
scale/scale_to_metric_mean
```

若启用 shape depth：

```text
shape_depth/max
shape_depth/mean
shape_depth/nonzero_ratio
shape_depth/clamped_ratio
```

还应保留：

```text
traj_norm/mean_endpoint_dist_shape
traj_norm/roundtrip_xy_error_m
```

判断标准：

- `traj_norm/mean_endpoint_dist_shape` 应接近 `R=2`。
- `scale/scale_to_shape_mean` 应随数据目标距离变化。
- `shape_depth/clamped_ratio` 不应过高，否则 shape depth 被大量截断。

## 9. 风险与对策

### 9.1 Scale token 被模型忽略

风险：Transformer 可能早期更依赖 rgbd/goal token，scale token 梯度弱。

对策：

- 把 `scale_token` 放在 memory 前部，紧跟 time token。
- 用多特征 `[d, log(d), R/d, d/R]`，降低学习难度。
- 在日志中监控不同距离 bucket 的 loss 是否收敛一致。

### 9.2 Shape depth 数值分布太偏

短目标时 `R/d_m` 较大，depth_shape 可能变大。

对策：

- 第一阶段不开启 shape depth。
- 第二阶段启用时设置 `shape_depth_clamp_max`。
- 监控 `shape_depth/clamped_ratio`。

### 9.3 双次 RGBD 编码显存增加

方案 B 会让 RGBDBackbone 前向计算近似翻倍。

对策：

- 先只做 Scale token。
- 如需 shape depth，优先使用共享 backbone + concat linear 融合，不增加 memory 长度。
- 若显存紧张，可冻结 shape depth 分支或降低 batch size。

### 9.4 NoGoal 空间与 PointGoal 空间不同

当前 NoGoal 保持 legacy 空间，PointGoal 使用 shape 空间。

对策：

- 保持 NoGoal 和 PointGoal 分支输入目标清晰分离。
- NoGoal 使用默认 scale token，不把其 loss 与 PointGoal 的物理尺度指标混在一起。
- 日志分别记录 NG 和 MG loss。

## 10. 推荐结论

推荐分两阶段做：

1. **立即做 Scale Token**：这是样本级轨迹归一化后必须补上的尺度条件，改动小，收益明确。
2. **再做 Shape Depth 融合**：保留 metric depth，同时额外编码 shape depth，并用 concat linear 融合回原 RGBD token 数。

不要直接把原始 depth 替换成 shape depth。最稳妥的路线是：

```text
metric depth 保留真实几何
scale token 显式给 d_m / R 与 R / d_m
shape depth 作为第二阶段增强，使视觉障碍与轨迹形状空间直接对齐
```

