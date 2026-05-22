# Bridge-DP Scale Token 引导 RGBD 特征调制修改方案

## 1. 背景

当前 Bridge-DP 已经引入了样本级轨迹尺度归一化：

```python
xy_shape = xy_m * R / d_m
theta_norm = theta_rad / pi
```

其中 `d_m = sqrt(x_n^2 + y_n^2)`，`R = trajectory_norm_target_distance`，默认 `R=2.0`。

这使 PointGoal、监督轨迹、先验轨迹进入统一的 shape-space。有效轨迹的终点距离基本固定为 `R`，Bridge-DP 更容易学习轨迹形状，而不是同时学习不同物理长度下的尺度变化。

但当前 RGBDBackbone 输入的深度仍然是 metric depth：

```python
depth_m = raw_depth / 10000.0
```

因此模型同时面对两个空间：

- `goal / trajectory / prior`：shape-space。
- `depth`：真实米制空间。

已经实现的 Scale Token 会把 `[d_m, log(d_m), R/d_m, d_m/R]` 作为一个 memory token 交给 Transformer decoder。这个方案是必要的第一步，但它仍然要求 decoder 自己通过 attention 学会：

```text
metric depth 中的 1m，在当前样本的 shape-space 中等价于 R / d_m 个归一化单位。
```

这个关系可以被学习，但不是强归纳偏置。为了更明确地引导模型使用尺度信息，建议在保留 Scale Token 的同时，用同一组尺度特征对 `rgbd_embed` 做特征调制。

## 2. 核心想法

不要只把 Scale Token 放在 memory 旁边，而是让尺度信息提前作用到视觉特征上：

```python
rgbd_embed = RGBDBackbone(input_images, input_depths)
scale_feat = [d_m, log(d_m), R / d_m, d_m / R]
gamma, beta = ScaleFiLM(scale_feat)
rgbd_embed = rgbd_embed * (1 + gamma) + beta
```

更准确地说，推荐使用残差 FiLM：

```python
rgbd_embed_mod = rgbd_embed + alpha * (norm(rgbd_embed) * gamma + beta)
```

这样做的含义是：

- Scale Token 继续作为全局尺度提示进入 Transformer memory。
- RGBD-FiLM 让视觉 token 本身变成 scale-aware。
- 模型在解释深度障碍物时，不必完全依赖 decoder 后期 attention 自己推理尺度比例。

## 3. 为什么比单独 Scale Token 更强

假设 `R=2`。

对于样本 A：

```text
d_m = 1m
depth_m = 1m
depth_shape = depth_m * R / d_m = 2
```

对于样本 B：

```text
d_m = 4m
depth_m = 1m
depth_shape = depth_m * R / d_m = 0.5
```

同样是深度图里前方 `1m` 的障碍物，在 shape-space 里的语义完全不同：

- `d_m=1m` 时，它接近目标终点尺度。
- `d_m=4m` 时，它只是近处局部障碍。

单独 Scale Token 的路径是：

```text
decoder query -> attend(scale_token) + attend(rgbd_token) -> 自己组合出尺度关系
```

RGBD-FiLM 的路径是：

```text
scale_feat -> 调制 rgbd_embed -> decoder 直接看到 scale-aware RGBD token
```

后者把“米制深度如何对应当前 shape-space”的先验提前注入视觉特征解释过程，正好对应当前 Bridge-DP 的尺度错位问题。

## 4. 推荐架构

### 4.1 保留现有 Scale Token

现有 memory 结构保持：

```text
[time, scale_token, goal x3, rgbd_embed, prior]
```

Scale Token 继续由当前函数生成：

```python
scale_feat = [
    d_m,
    log(d_m),
    R / d_m,
    d_m / R,
]
scale_token = scale_encoder(scale_feat).unsqueeze(1)
```

### 4.2 新增 Scale-FiLM 模块

在 `BridgeDPNet.__init__` 中新增：

```python
self.scale_rgbd_film = nn.Sequential(
    nn.Linear(4, self.token_dim),
    nn.GELU(),
    nn.Linear(self.token_dim, 2 * self.token_dim),
)
self.scale_rgbd_film_norm = nn.LayerNorm(self.token_dim)
```

输出拆成：

```python
gamma_beta = self.scale_rgbd_film(scale_feat)
gamma, beta = gamma_beta.chunk(2, dim=-1)
gamma = gamma.unsqueeze(1)  # (B, 1, token_dim)
beta = beta.unsqueeze(1)    # (B, 1, token_dim)
```

对 RGBD token 做广播调制：

```python
rgbd_norm = self.scale_rgbd_film_norm(rgbd_embed)
rgbd_embed = rgbd_embed + alpha * (rgbd_norm * gamma + beta)
```

其中 `rgbd_embed` 形状为：

```python
(B, memory_size * 16, token_dim)
```

`gamma` 和 `beta` 会沿 token 维度广播。

### 4.3 零初始化最后一层

为了避免一开始破坏现有视觉特征分布，建议将 `scale_rgbd_film` 最后一层零初始化：

```python
nn.init.zeros_(self.scale_rgbd_film[-1].weight)
nn.init.zeros_(self.scale_rgbd_film[-1].bias)
```

这样训练初始时：

```python
gamma = 0
beta = 0
rgbd_embed_mod = rgbd_embed
```

模型行为等价于当前版本，然后在训练中逐渐学会使用 scale 条件调制视觉特征。

## 5. 新增配置

建议在 `scripts/train/base_train/configs/bridgedp.py` 中新增：

```python
# Scale Token 已实现，继续保留。
enable_scale_condition_token=True,
scale_condition_clamp_min_m=0.10,
scale_condition_clamp_max_m=20.0,

# 用尺度条件调制 RGBD token。
enable_scale_rgbd_film=True,
scale_rgbd_film_alpha=1.0,
scale_rgbd_film_zero_init=True,
scale_rgbd_film_use_layernorm=True,
```

字段含义：

- `enable_scale_rgbd_film`：是否启用 RGBD-FiLM。
- `scale_rgbd_film_alpha`：残差调制强度，默认 `1.0`。
- `scale_rgbd_film_zero_init`：最后一层是否零初始化，推荐 `True`。
- `scale_rgbd_film_use_layernorm`：调制前是否对 `rgbd_embed` 做 LayerNorm，推荐 `True`。

## 6. 代码改动位置

### 6.1 `bridgedp_policy.py`

新增配置读取：

```python
self.enable_scale_rgbd_film = il.get('enable_scale_rgbd_film', False)
self.scale_rgbd_film_alpha = float(il.get('scale_rgbd_film_alpha', 1.0))
self.scale_rgbd_film_zero_init = il.get('scale_rgbd_film_zero_init', True)
self.scale_rgbd_film_use_layernorm = il.get('scale_rgbd_film_use_layernorm', True)
```

新增模块：

```python
if self.enable_scale_rgbd_film:
    self.scale_rgbd_film = nn.Sequential(
        nn.Linear(4, self.token_dim),
        nn.GELU(),
        nn.Linear(self.token_dim, 2 * self.token_dim),
    )
    self.scale_rgbd_film_norm = nn.LayerNorm(self.token_dim)
```

新增函数：

```python
def _apply_scale_rgbd_film(self, rgbd_embed, traj_distance_m):
    if not self.enable_scale_rgbd_film:
        return rgbd_embed

    feat = self._build_scale_features(traj_distance_m)
    gamma_beta = self.scale_rgbd_film(feat).to(
        device=rgbd_embed.device,
        dtype=rgbd_embed.dtype,
    )
    gamma, beta = gamma_beta.chunk(2, dim=-1)
    gamma = gamma.unsqueeze(1)
    beta = beta.unsqueeze(1)

    base = (
        self.scale_rgbd_film_norm(rgbd_embed)
        if self.scale_rgbd_film_use_layernorm
        else rgbd_embed
    )
    return rgbd_embed + self.scale_rgbd_film_alpha * (base * gamma + beta)
```

### 6.2 训练 forward

当前训练中有：

```python
rgbd_embed = self.rgbd_encoder(input_images, input_depths)
scale_embed_mg = self._build_scale_token(tensor_traj_distance_m, like_token=pointgoal_embed)
scale_embed_ng = self._build_scale_token(nogoal_distance_m, like_token=pointgoal_embed)
```

推荐改成两个 RGBD 版本：

```python
rgbd_embed_base = self.rgbd_encoder(input_images, input_depths)
rgbd_embed_mg = self._apply_scale_rgbd_film(rgbd_embed_base, tensor_traj_distance_m)
rgbd_embed_ng = self._apply_scale_rgbd_film(rgbd_embed_base, nogoal_distance_m)
```

然后：

- mixed-goal 分支使用 `rgbd_embed_mg`。
- no-goal 分支使用 `rgbd_embed_ng`。
- critic 如果仍保持 NoGoal 语义，则使用 `rgbd_embed_ng`。
- `VisualGate` 如果用于 PointGoal 先验，建议使用 `rgbd_embed_mg.mean(dim=1)`；如果服务 NoGoal，则使用 `rgbd_embed_ng.mean(dim=1)`。

如果为了最小改动，也可以第一阶段只构造：

```python
rgbd_embed = self._apply_scale_rgbd_film(rgbd_embed, tensor_traj_distance_m)
```

但这会让 NoGoal 分支也看到 PointGoal 的 `d_m`，语义不够干净。因此推荐分别构造 `mg/ng` 两套调制后的 RGBD token。

### 6.3 PointGoal 推理

当前 PointGoal 推理已经计算：

```python
goal_distance_m = torch.norm(tensor_point_goal[:, :2], dim=-1)
scale_embed = self._build_scale_token(goal_distance_m, like_token=pointgoal_embed)
```

新增：

```python
rgbd_embed = self.rgbd_encoder(input_images, input_depths)
rgbd_embed = self._apply_scale_rgbd_film(rgbd_embed, goal_distance_m)
```

然后 `predict_x0()` 和 `predict_critic()` 都使用调制后的 `rgbd_embed`。

### 6.4 NoGoal 推理

NoGoal 没有真实目标距离，继续使用默认前向距离：

```python
nogoal_distance_m = self.nogoal_front_distance * self.action_scale_xy
```

新增：

```python
rgbd_embed = self.rgbd_encoder(input_images, input_depths)
rgbd_embed = self._apply_scale_rgbd_film(rgbd_embed, nogoal_distance_m)
```

这样 NoGoal 的视觉特征也有稳定的默认尺度条件，不会和 PointGoal 的 memory 分布差异过大。

### 6.5 训练监控可视化推理

`bridgedp_trainer.py` 中 `_infer_pred_traj_bridgedp()` 已经使用 `batch_traj_distance_m` 构造 `scale_embed`。需要同步加入：

```python
rgbd_embed = model_ref.rgbd_encoder(...)
rgbd_embed = model_ref._apply_scale_rgbd_film(
    rgbd_embed,
    inputs_on_device["batch_traj_distance_m"],
)
```

保证可视化预测与正式 PointGoal 推理路径一致。

## 7. 与双尺度深度方案的关系

本方案不是双尺度深度。

它不新增 `depth_shape`，也不改变 `RGBDBackbone` 的输入深度分布。深度输入仍然是 metric depth：

```python
rgbd_embed = RGBDBackbone(image, depth_metric)
```

尺度只在 backbone 之后，以 FiLM 形式调制 token：

```python
rgbd_embed -> scale-aware rgbd_embed
```

因此它比“双尺度深度”更轻量，风险更低，适合作为当前 Scale Token 之后的第二步。

未来如果这个方案仍然不足，再考虑启用 `depth_shape = depth_metric * R / d_m` 的双尺度深度路线。

## 8. 训练引导与诊断

为了确认模型真的使用了尺度条件，建议增加两个诊断指标。

### 8.1 Zero-scale 诊断

训练监控时，额外计算一次将 scale feature 置为默认或全零后的预测误差：

```text
loss/with_scale
loss/zero_scale
```

如果两者接近，说明模型可能没有使用 Scale Token / RGBD-FiLM。

### 8.2 Shuffled-scale 诊断

在 batch 内打乱 `traj_distance_m`，观察预测误差：

```text
loss/shuffled_scale
```

期望现象：

```text
loss/with_scale < loss/shuffled_scale
```

如果 shuffled scale 不变差，说明尺度条件没有被有效利用。

这些诊断不一定第一版就实现，但适合在训练不明显受益时加入。

## 9. 实施优先级

推荐分三步落地。

### 第一步：最小 RGBD-FiLM

实现：

- 新增配置。
- 新增 `scale_rgbd_film`。
- 新增 `_apply_scale_rgbd_film()`。
- 训练和推理都对 `rgbd_embed` 做调制。
- 最后一层零初始化。

不实现：

- 不新增 shape depth。
- 不改 Dataset。
- 不改 RGBDBackbone 输入通道。
- 不改 Transformer memory 长度。

### 第二步：区分 PointGoal / NoGoal 的 RGBD 调制

训练中明确使用：

```python
rgbd_embed_mg = film(rgbd_embed_base, tensor_traj_distance_m)
rgbd_embed_ng = film(rgbd_embed_base, nogoal_distance_m)
```

避免 NoGoal 分支使用 PointGoal 的真实距离，保持语义干净。

### 第三步：加入尺度使用诊断

加入 zero-scale / shuffled-scale 诊断日志，用实验确认模型是否真的依赖尺度条件。

## 10. 预期效果

预期改进点：

- 模型更容易把 metric depth 和 shape-space 轨迹对齐。
- 同一深度障碍物在不同目标距离下的相对意义更容易被区分。
- Scale Token 不再只是一个旁路提示，而是直接参与视觉特征解释。
- 相比双尺度深度，改动更小，训练初期更稳定。

需要注意：

- 新增 FiLM 参数后，旧 checkpoint 不能完全结构兼容。
- 零初始化可以保证新模型初始行为接近当前 Scale Token 版本。
- 该方案仍然依赖训练数据中有足够的距离多样性；如果样本距离分布很偏，还需要配合距离分桶采样或 loss reweight。
