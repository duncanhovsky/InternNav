# Bridge-DP 训练损失重设计方案（面向“逐步接近目标+速度匹配”）

## 1. 目标与约束

**目标**
- 让模型预测的 24 个航点“逐步接近导航目标”，而不是强制 24 步到达目标。
- 预测轨迹符合自然运动规律，速度分布与数据集样本一致。
- 预测点与真值点按时间步对齐（逐步匹配）。

**约束**
- **保持导航目标为端点**（episode 终点），不改为片段终点。
- **不改模型结构**（不新增 head）；仅调整训练损失设计与权重。
- 可增加日志指标用于验证效果。

## 2. 现有训练损失回顾（基于当前代码）

当前训练损失在 [internnav/trainer/bridgedp_trainer.py](internnav/trainer/bridgedp_trainer.py) 中，结构为：

- `L_x0`：x0 预测的逐点 MSE（全步同权重）
- `L_delta`：相邻步差分的 MSE
- `critic_loss`：障碍相关
- `aux_loss`：辅助目标

总体：

```
loss = 0.8 * action_loss + 0.2 * critic_loss + 0.5 * aux_loss
action_loss = L_x0 + lambda_delta * L_delta
```

**潜在问题**
- `L_x0` 全步等权 → 终端步与起步步同权，难以体现“逐步接近目标”的趋势。
- `L_delta` 只做平滑约束，对“速度分布匹配”没有直接约束。
- 目标是导航终点，但真值轨迹是片段 → 模型容易拉长轨迹以“更接近终点”。

## 3. 损失设计原则

1) **不强制 24 步到达导航目标**：仅鼓励“朝目标方向逐步推进”。
2) **按时间步匹配**：每步与真值同时间索引对齐。
3) **速度分布匹配**：让预测轨迹的速度统计与真值一致（均值/方差）。
4) **保持障碍与辅助损失不动**：避免影响已有稳定性。

## 4. 新损失设计（不改模型结构）

### 4.1 带权逐步重建损失（Weighted L_x0）

**核心思想**：
- 轻微提高后期步的权重（强调“逐步接近目标”）
- 用 GT 速度作为权重（大步位移给予更多学习强度）

定义：

- 预测与真值：$\hat{x}_0, x_0 \in \mathbb{R}^{B\times T\times 3}$
- 有效掩码：$M \in \{0,1\}^{B\times T}$
- 进度权重：

$$w_{t}^{\text{prog}} = 1 + \alpha \cdot \frac{t}{T-1},\; \alpha \in [0.3, 0.8]$$

- 速度权重（来自真值）：

$$v_t = \|x_0^{t} - x_0^{t-1}\|_2,\; w_{t}^{\text{speed}} = \text{clamp}(\frac{v_t}{\bar{v}}, w_{min}, w_{max})$$

- 综合权重：

$$w_t = w_{t}^{\text{prog}} \cdot w_{t}^{\text{speed}}$$

最终：

$$L_{x0}^\text{weighted} = \frac{\sum_{t} w_t \cdot M_t \cdot \|\hat{x}_0^t - x_0^t\|_2^2}{\sum_t M_t}$$

### 4.2 速度分布匹配正则（全局统计）

用预测与真值的“步长分布”在 batch 内对齐（均值+方差）：

- 真值步长：$v^{\text{gt}}_t = \|x_0^t - x_0^{t-1}\|_2$
- 预测步长：$v^{\text{pred}}_t = \|\hat{x}_0^t - \hat{x}_0^{t-1}\|_2$

统计损失：

$$L_{speed} = (\mu_{pred} - \mu_{gt})^2 + (\sigma_{pred} - \sigma_{gt})^2$$

> 这是“全局正则”，不强制逐步等速，但能抑制“整体变长”的趋势。

### 4.3 轨迹平滑损失（保留 L_delta）

保留现有 `L_delta`，但降低权重：

```
L_delta_weight = 0.05 ~ 0.1
```

避免过度影响轨迹长度。

### 4.4 终点轻约束（可选）

只做轻量正则，避免轨迹末端过度拉长：

$$L_{end} = \|\hat{x}_0^{T-1} - x_0^{T-1}\|_2^2$$

权重建议非常小（0.05 或更低），仅用于抑制漂移。

## 5. 新的 action_loss 组合建议

```
action_loss = L_x0_weighted
            + lambda_delta * L_delta
            + lambda_speed * L_speed
            + lambda_end * L_end
```

建议初值：
- `lambda_delta = 0.05`
- `lambda_speed = 0.1`
- `lambda_end = 0.05`
- `alpha (progress)` = 0.5

其余 `critic_loss`、`aux_loss` 保持不变。

## 6. 建议新增监控指标

用于验证是否“更像数据集轨迹”：

- `mean_speed_pred`, `mean_speed_gt`
- `std_speed_pred`, `std_speed_gt`
- `end_dist`（预测末端与 GT 末端距离）
- `traj_len_ratio`（预测路径长度 / GT 路径长度）

> 这些指标可用于可视化面板或日志输出。

## 7. 验证流程建议

1) **小规模训练对比**：保持其它超参不变，仅改损失，观察 1-2k steps。
2) **速度统计对比**：确认 `mean_speed_pred` 接近 `mean_speed_gt`。
3) **轨迹长度对比**：预测长度不再明显拉长。
4) **避障质量不下降**：`critic_loss` 不应明显恶化。

## 8. 备选方案（若仍出现拉长）

- 降低 `sigma_goal` 或 `d_max`，让桥初始化更接近 GT 范围。
- 进行课程式训练：前期降低导航目标强度，后期再恢复。

---

**备注**：此方案保持“导航目标为端点”的设计不变，但通过损失引导模型更像数据集轨迹，避免“过度拉长”。如需后续实施代码修改，可基于本方案直接落地。
