# Bridge-DP 训练不收敛修复方案

> 诊断时间：2026-05-18  
> 症状：MSE 600+，最大点误差 60+，训练完全不收敛

---

## 根本原因总结

**一句话**：布朗桥框架下错误地使用了 ε-prediction，导致训练目标与推理逻辑根本不一致，且 σ 极小时反解公式数值爆炸。

---

## 问题清单（按严重程度）

### P0 — 致命，必须修复

#### 问题 1：训练用 ε-prediction，推理用 ε→x₀ 反解，两者数值不兼容

**位置**：
- [`bridgedp_policy.py:312`](internnav/model/basemodel/bridgedp/bridgedp_policy.py:312) — `sample_bridge_noise` 返回 `noise`（ε）
- [`bridgedp_policy.py:551`](internnav/model/basemodel/bridgedp/bridgedp_policy.py:551) — `action_head` 输出被当作 ε 预测
- [`bridgedp_policy.py:381`](internnav/model/basemodel/bridgedp/bridgedp_policy.py:381) — `_eps_to_x0` 反解：`x₀ = (x_t - t·g - σ·ε̂) / (1-t)`
- [`bridgedp_trainer.py:109`](internnav/trainer/bridgedp_trainer.py:109) — 损失：`(eps_abs_pred_ng - ng_noise).square().mean()`

**为什么爆炸**：布朗桥的 σ(t) 在 `sigma_base=0.1` 时约为 0.05，远小于 DDPM 的 √(1-ᾱ)≈O(1)。反解公式中 `σ·ε̂` 项被极小的 σ 压缩，任何预测误差经 `/(1-t)` 放大后（t=0.9 时放大 10 倍，t=0.99 时放大 100 倍）直接爆炸。

**修复**：改为 **x₀-prediction**，与推导文档 §6.1 一致。

**需要改动的文件**：
1. [`bridgedp_policy.py`](internnav/model/basemodel/bridgedp/bridgedp_policy.py) — `sample_bridge_noise` 改为返回 `x0`；删除 `_eps_to_x0`；`predict_noise` 改名为 `predict_x0`，直接输出 x̂₀
2. [`bridgedp_trainer.py`](internnav/trainer/bridgedp_trainer.py) — 损失改为 `(x0_pred - x0_target).square().mean()`；推理循环去掉 `_eps_to_x0` 调用

---

#### 问题 2：`sigma_base=0.1` 在归一化空间中太小，布朗桥退化为确定性映射

**位置**：[`bridgedp.py:91`](scripts/train/base_train/configs/bridgedp.py:91)

**为什么爆炸**：归一化后轨迹值范围约 [-2, 2]，但 σ(t=0.5) ≈ 0.05，噪声幅度不到轨迹幅度的 3%。x_t 几乎等于桥均值，网络看不到有意义的噪声，无法学习去噪。

**修复**：将 `sigma_base` 改为 **0.5**（归一化空间中合理值），`sigma_goal` 改为 **0.3**。

> 理想做法是在归一化空间中重新运行 `compute_sigma_base.py` 计算 95 分位数，但 0.5 是合理的初始估计值。

---

### P1 — 严重，强烈建议修复

#### 问题 3：双空间 L_rel 损失目标不合理，与 L_abs 产生梯度冲突

**位置**：
- [`bridgedp_policy.py:579`](internnav/model/basemodel/bridgedp/bridgedp_policy.py:579) — `ng_noise_rel = ng_noise[:, 1:] - ng_noise[:, :-1]`
- [`bridgedp_trainer.py:115`](internnav/trainer/bridgedp_trainer.py:115) — `ng_rel_loss = (eps_rel_pred_ng[:, :-1] - ng_noise_rel).square().mean()`

**问题**：噪声差分 `ε[i]-ε[i-1]` 方差为 2（不是 1），且 `action_head` 和 `delta_head` 共享同一 Transformer 输出，需要同时拟合两个不同分布，梯度冲突。

**修复**：切换到 x₀-prediction 后，L_rel 改为**增量一致性正则**：
```
x0_pred_delta = x0_pred[:, 1:] - x0_pred[:, :-1]   # 预测轨迹的增量
x0_target_delta = x0_target[:, 1:] - x0_target[:, :-1]  # 真实轨迹的增量
L_rel = (x0_pred_delta - x0_target_delta).square().mean()
```
这样 L_rel 有明确的物理意义（轨迹平滑性约束），且与 L_abs 梯度方向一致。

---

#### 问题 4：`valid_mask` 生成了但损失中完全没用

**位置**：
- [`bridgedp_lerobot_dataset.py:689`](internnav/dataset/bridgedp_lerobot_dataset.py:689) — 生成了 `valid_mask`
- [`bridgedp_trainer.py:109`](internnav/trainer/bridgedp_trainer.py:109) — 损失中没有使用

**问题**：静止重复帧（step_diff≈0）参与损失，这些点 x₀[i]≈x₀[i-1]，对网络学习无贡献但会稀释梯度。

**修复**：在 L_abs 中加入 mask 加权：
```python
mask = inputs_on_device["batch_valid_mask"].unsqueeze(-1)  # (B, T, 1)
ng_abs_loss = ((x0_pred_ng - x0_target) ** 2 * mask).sum() / mask.sum().clamp(min=1)
```

---

### P2 — 中等，建议修复

#### 问题 5：bridge endpoint 与实际轨迹终点不一致

**位置**：[`bridgedp_policy.py:473`](internnav/model/basemodel/bridgedp/bridgedp_policy.py:473)

训练时 `goal = tensor_point_goal`（来自数据集的 `point_goal = target_xyt_actions[-1]`），但 `pred_actions = target_xyt_actions[action_indexes][1:]` 经过下采样后，`pred_actions[-1]` 不一定等于 `point_goal`。

**修复**：训练时 bridge endpoint 应使用 `tensor_label_actions[:, -1, :]`（实际标签轨迹的最后一步），而非 `tensor_point_goal`。

---

#### 问题 6：推理时 `step()` 的 `eta=0.5` 在训练初期引入过多随机性

**位置**：[`bridge_scheduler.py:253`](internnav/model/basemodel/bridgedp/bridge_scheduler.py:253)

**修复**：推理时先用 `eta=0.0`（纯 DDIM 确定性）验证收敛，收敛后再开启随机扰动。

---

## 修复步骤（按执行顺序）

```
[ ] Step 1: 修改 bridgedp_policy.py — 切换到 x₀-prediction
[ ] Step 2: 修改 bridgedp_trainer.py — 损失改为 x₀-MSE + 增量一致性正则
[ ] Step 3: 修改 bridgedp.py 配置 — sigma_base=0.5, sigma_goal=0.3
[ ] Step 4: 修改 bridgedp_trainer.py — 加入 valid_mask 加权
[ ] Step 5: 修改 bridgedp_policy.py — bridge endpoint 用 label_actions[:,-1,:]
[ ] Step 6: 修改 bridge_scheduler.py — 推理 eta 默认改为 0.0
[ ] Step 7: 验证：跑 100 步，确认 action_loss < 0.5（归一化空间）
```

---

## 修改细节

### Step 1 & 2：x₀-prediction 核心改动

**[`bridgedp_policy.py`](internnav/model/basemodel/bridgedp/bridgedp_policy.py) 的 `sample_bridge_noise`**：

```python
# 改前：返回 noise（ε）
return noise, time_embeds, noisy_action_embed, timesteps

# 改后：返回 x0（干净轨迹，作为训练目标）
return x0, time_embeds, noisy_action_embed, timesteps
```

**[`bridgedp_policy.py`](internnav/model/basemodel/bridgedp/bridgedp_policy.py) 的 `forward` 返回值**：

```python
# 改前：返回 ng_noise（ε 目标）
# 改后：返回 tensor_label_actions（x₀ 目标）

return (
    x0_pred_ng,              # action_head 输出，语义为 x̂₀
    x0_pred_mg,
    eps_rel_pred_ng,         # delta_head 输出（用于增量一致性）
    eps_rel_pred_mg,
    cr_label_pred,
    cr_augment_pred,
    tensor_label_actions,    # x₀ 目标（ng/mg 共用同一标签）
    tensor_label_actions,    # x₀ 目标（mg 分支）
    imagegoal_aux_pred,
    pixelgoal_aux_pred,
)
```

**[`bridgedp_trainer.py`](internnav/trainer/bridgedp_trainer.py) 的 `compute_loss`**：

```python
# 改前（ε-MSE）：
ng_abs_loss = (eps_abs_pred_ng - ng_noise).square().mean()

# 改后（x₀-MSE + valid_mask）：
mask = inputs_on_device["batch_valid_mask"].unsqueeze(-1)  # (B, T, 1)
x0_target = inputs_on_device["batch_labels"]
ng_abs_loss = ((x0_pred_ng - x0_target) ** 2 * mask).sum() / mask.sum().clamp(min=1)
mg_abs_loss = ((x0_pred_mg - x0_target) ** 2 * mask).sum() / mask.sum().clamp(min=1)
L_abs = 0.5 * (ng_abs_loss + mg_abs_loss)

# L_rel 改为增量一致性正则：
x0_pred_ng_delta = x0_pred_ng[:, 1:] - x0_pred_ng[:, :-1]
x0_target_delta  = x0_target[:, 1:]  - x0_target[:, :-1]
ng_rel_loss = (x0_pred_ng_delta - x0_target_delta).square().mean()
# mg 同理
L_rel = 0.5 * (ng_rel_loss + mg_rel_loss)
```

**推理循环（`_infer_pred_traj_bridgedp` 和 `predict_pointgoal_batch_action_vel`）**：

```python
# 改前（ε→x₀ 反解）：
eps_pred = model_ref.predict_noise(naction, k, ...)
x0_pred = model_ref._eps_to_x0(eps_pred, naction, k, endpoint_exp, theta_exp)
naction = model_ref.bridge_scheduler.step(x0_pred, naction, k, ...)

# 改后（直接预测 x₀）：
x0_pred = model_ref.predict_x0(naction, k, ...)   # 函数改名，语义改变
naction = model_ref.bridge_scheduler.step(x0_pred, naction, k, ...)
```

### Step 3：配置修改

**[`bridgedp.py`](scripts/train/base_train/configs/bridgedp.py)**：

```python
# 改前：
sigma_base=0.1,
sigma_goal=0.1,

# 改后：
sigma_base=0.5,   # 归一化空间中合理值（轨迹范围 [-2,2]，噪声幅度约 25%）
sigma_goal=0.3,   # 终点松弛方差
```

---

## 预期效果

| 指标 | 修复前 | 修复后预期 |
|------|--------|-----------|
| action_loss（归一化空间） | 600+ | < 0.5 |
| 最大点误差（归一化空间） | 60+ | < 1.0 |
| 训练收敛步数 | 不收敛 | ~5000 步可见下降 |

---

## 不需要修改的部分

- [`bridge_scheduler.py`](internnav/model/basemodel/bridgedp/bridge_scheduler.py) 的 `add_noise()` 和 `step()` 逻辑本身是正确的，`step()` 接收 x₀ 预测并做 DDIM 更新，数学上没有问题
- [`prior_encoder.py`](internnav/model/basemodel/bridgedp/prior_encoder.py) 的 PriorEncoder 和 VisualGate 架构正确
- [`bridgedp_lerobot_dataset.py`](internnav/dataset/bridgedp_lerobot_dataset.py) 的数据处理逻辑基本正确（归一化、theta_g 计算、prior 生成）
- Transformer Decoder 架构、memory 序列结构、Critic 分支均正确
