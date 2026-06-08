# Bridge-DP v1.0.3 更新实现文档

更新日期：2026-06-04

## 1. 更新目标

Bridge-DP v1.0.3 解决两个代码层面的能力边界：

1. PointGoal 布朗桥均值固定连接 `origin -> pointgoal`，在目标前方存在大障碍物时，初始化均值容易穿过障碍，候选绕障多样性主要依赖法向噪声和 critic 排序。
2. 训练监督默认按真实轨迹弧长均匀采样，而 PointGoal bridge 均值按起终点 chord 的 `tau=(i+1)/T` 均匀插值。复杂绕障轨迹中，监督 waypoint 的 chord 投影进度可能与 bridge 时间不一致。

本版本引入：

- **Bridge anchor sampling**：真实 `pointgoal` 不变，另采样一个等半径的 `bridge_anchor`，只用于布朗桥均值、方差坐标系、推理初始化和反向 step。
- **投影/弧长混合监督重采样**：在轨迹对起终点 chord 的投影基本单调且覆盖充分时，按 chord 投影均匀采样；否则回退旧版弧长采样。

## 2. 设计原则

- 真实 `pointgoal` 仍作为任务目标输入 point encoder，也用于 goal-consistency score。
- `bridge_anchor` 不是任务终点，只是 Brownian bridge 的几何 anchor。
- 训练和推理都支持 anchor sampling，避免只改推理导致 train-infer 分布错位。
- 投影采样默认使用 `hybrid_projection`，避免把起步侧移、回退、U 形绕行等非单调轨迹强行投影均匀化。
- 所有新增行为都由 config 超参数控制，可退回 v1.0.2 行为。

## 3. 代码改动

### 3.1 BridgeScheduler

文件：`internnav/model/basemodel/bridgedp/bridge_scheduler.py`

新增 `sample_bridge_anchor_goals()`：

- 输入真实 `goal`、`origin`、`sample_num`。
- 保持 `||anchor_xy - origin_xy|| = ||goal_xy - origin_xy||`。
- 在目标方向角附近采样扰动角 `delta`。
- `anchor_xy = origin_xy + radius * [cos(theta + delta), sin(theta + delta)]`。
- 第三维目标姿态默认保持真实 goal 第三维，不随 anchor 角度改写。
- 返回 `anchor_goal` 和 `anchor_theta`，后者用于 tangent/normal 方差坐标系。

扰动分布：

```text
中心候选：
delta = TruncatedGaussian(0, bridge_anchor_angle_std, [-max, max])

边缘候选：
|delta| ~ EdgeBiased(max, std),  p(|delta|=r) ∝ 1 - exp(-r^2 / (2 * std^2))
sign(delta) 在推理多候选时左右成对分配
```

默认配置关闭旧版 uniform 混合，改用 `bridge_anchor_edge_prob=0.5`：
推理多候选时，保留第一组原始 pointgoal anchor，其余候选约一半使用中心高斯，一半使用左右成对的 edge-biased 扰动；
训练 `sample_num=1` 时，edge 分支按 batch 内 Bernoulli 概率生效。

### 3.2 BridgeDPNet

文件：`internnav/model/basemodel/bridgedp/bridgedp_policy.py`

训练侧：

- `pointgoal_embed` 仍使用真实 `tensor_point_goal`。
- MixedGoal 分支前向加噪可按 `bridge_anchor_train_prob` 使用 sampled anchor。
- NoGoal 分支不使用 bridge anchor。

推理侧：

- 候选初始化使用 `bridge_goal_repeated`。
- 反向去噪 `step_trajectory()` 使用 `bridge_goal_repeated` 和 `bridge_theta_expanded`。
- `predict_x0()` 条件 token 仍使用真实 `pointgoal_embed`。
- goal-consistency score 仍使用真实 `goal_repeated`。
- `bridge_anchor_keep_original_sample=True` 时，第一组候选保留原始 pointgoal anchor，便于回退和对照。

### 3.3 BridgeDPTrainer

文件：`internnav/trainer/bridgedp_trainer.py`

新增监督重采样模式：

- `arc_length`：旧行为，按真实轨迹 xy 弧长均匀采样。
- `projection`：按起点到终点 chord 的投影进度均匀采样。
- `hybrid_projection`：投影单调且投影覆盖度达到阈值时使用 projection，否则回退 arc_length。

投影采样使用原点到轨迹末端的 chord：

```text
u_i = dot(xy_i, end_xy) / ||end_xy||^2
```

监督查询点仍为：

```text
q_i = (i+1) / T
```

因此在投影采样有效时，监督 waypoint 与 Brownian bridge 的 trajectory time 更一致。

## 4. Config 超参数

文件：`scripts/train/base_train/configs/bridgedp.py`

### 4.1 Bridge anchor sampling

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `enable_bridge_anchor_sampling` | `True` | 是否启用 PointGoal bridge anchor sampling。 |
| `bridge_anchor_train_prob` | `0.5` | 训练 MixedGoal 加噪时使用 sampled anchor 的样本比例。 |
| `bridge_anchor_keep_original_sample` | `True` | 推理候选第一组是否保留真实 pointgoal anchor。 |
| `bridge_anchor_angle_std` | `0.25` | anchor 角度扰动的高斯标准差，单位 rad。 |
| `bridge_anchor_angle_max` | `0.65` | anchor 最大角度偏移，单位 rad。 |
| `bridge_anchor_uniform_prob` | `0.0` | 旧版均匀角度扰动混合概率，默认关闭，保留用于兼容和 ablation。 |
| `bridge_anchor_edge_prob` | `0.5` | 使用 edge-biased 反高斯扰动的比例/概率；推理多候选时左右成对分配。 |

调参建议：

- 绕障候选不足：优先增大 `bridge_anchor_edge_prob`，其次增大 `bridge_anchor_angle_std` 或 `bridge_anchor_angle_max`。
- 终点偏移过大：减小 `bridge_anchor_angle_max`，或增大 goal-consistency terminal 权重。
- 训练不稳定：降低 `bridge_anchor_train_prob`，先只做推理侧 ablation。

### 4.1.1 Edge bridge noise

在 `bridge_scale_invariant_sigma=True` 且 PointGoal 分支下，支持结构化 edge bridge noise：

- `bridge_noise_edge_prob=0.5`：推理初始化候选中使用 edge noise 的比例。多候选时第一组原始候选保持普通初始化，其余 edge 候选按左右成对分配。
- `bridge_noise_edge_train_prob=0.2`：训练前向加噪中使用 edge noise 的样本概率。建议低于推理值，避免训练分布被边缘样本主导。
- `bridge_noise_edge_warmup_steps=4.0`：起点保护 gate，第一个 waypoint 的 edge 噪声为 0，随后平滑增大，避免 origin 到第一个航点横跳。
- `bridge_noise_edge_terminal_guard_steps=3.0`：终点保护 gate，最后一个 waypoint 的 edge 噪声强制为 0，终点附近平滑衰减。
- `bridge_noise_edge_normal_max=1.0`：法向反高斯 eps 最大幅值，最终偏移仍会乘以 `sigma_normal`。
- `bridge_noise_edge_tangent_scale=0.15`、`bridge_noise_edge_theta_scale=0.25`：切向与航向弱耦合比例，用于增加形状/姿态多样性，但不主导绕障。

### 4.2 监督重采样

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `trajectory_resample_mode` | `"hybrid_projection"` | 轨迹监督重采样模式。 |
| `trajectory_projection_monotonic_eps` | `1e-4` | 投影进度单调性容忍阈值。 |
| `trajectory_projection_min_span` | `0.80` | 投影覆盖度低于该值时回退弧长采样。 |
| `trajectory_projection_flat_lateral_eps` | `1e-3` | 投影几乎不变但横向移动超过该阈值时，hybrid 模式回退弧长采样。 |

调参建议：

- 保守复现实验：设为 `"arc_length"`。
- 验证 bridge-time 匹配收益：设为 `"projection"`。
- 推荐训练：保持 `"hybrid_projection"`。

### 4.3 目标一致性评分

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `enable_goal_consistency_score` | `True` | 在 critic 分数上叠加真实 pointgoal 一致性惩罚。 |
| `goal_consistency_terminal_weight` | `1.0` | 终端偏差惩罚权重。 |
| `goal_consistency_path_weight` | `0.05` | 路径长度过长惩罚权重。 |

启用原因：bridge anchor 会让 scheduler 几何终点与真实 pointgoal 分离，候选排序需要显式约束真实终点一致性。

## 5. 数据链路

训练链路：

```text
raw_labels
 -> arc_length / projection / hybrid_projection 重采样
 -> batch_labels, batch_pg
 -> pointgoal condition 使用 batch_pg
 -> MixedGoal bridge 加噪按概率使用 sampled bridge_anchor
 -> x0 prediction loss 仍监督真实 batch_labels
```

推理链路：

```text
真实 pointgoal
 -> point encoder / scale token / goal-consistency score
 -> sampled bridge_anchor
 -> sample_initial_noise_ordered()
 -> step_trajectory()
 -> learned critic + true-goal consistency 排序
```

## 6. 风险边界

- anchor sampling 增加的是候选多样性，不是硬避障约束。
- 如果 critic 对真实 inflated collision 相关性不足，anchor sampling 可能生成更多候选，但排序不一定更安全。
- projection 采样对非单调轨迹不稳，因此默认使用 hybrid fallback。
- 机器人 footprint、半径膨胀、速度/角速度约束仍未进入 Bridge-DP 主干。

## 7. 验证建议

最小验证：

- 单元测试 `tests/unit_test/test_bridgedp_v103_sampling.py`。
- 检查 anchor sampling 是否保持半径、是否保留原始候选、是否产生侧向偏移。
- 检查 projection resampling 的 chord 投影是否接近均匀。

实验验证：

- `arc_length` vs `hybrid_projection` vs `projection`。
- `enable_bridge_anchor_sampling=False/True`。
- `bridge_anchor_train_prob=0/0.25/0.5/1.0`。
- `bridge_anchor_angle_max=0.35/0.65/0.90`。
- 对不同机器人半径做离线 inflated collision 评估。
- 统计 critic 分数与 inflated collision 的相关性。
