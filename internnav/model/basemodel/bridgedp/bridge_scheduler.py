"""方向自适应弹性布朗桥调度器。

本模块实现 Bridge-DP 的核心噪声调度逻辑，替代 NavDP 使用的
``diffusers.DDPMScheduler``。关键数学公式：

前向加噪（训练）::

    x_t = (1-t) * x_0 + t * g + σ(t; θ_g) * ε

方差调度::

    σ²(t; θ_g) = σ²_base · [t(1-t)]^{p(θ_g)} + t² · σ²_goal
    p(θ_g) = 0.5 + 0.3 · cos(θ_g)

反向去噪（推理，DDIM 确定性采样）::

    x_{t-Δt} = (t-Δt)/t · x_t + Δt/t · x̂_0

参考文献：
    - docs/Bridge-DP推导.md §3（弹性布朗桥 SDE）
    - docs/Bridge-DP推导.md §9（方向自适应方差）
    - docs/Bridge-DP推导.md §8（反向去噪推理）
"""

import math
from typing import Optional, Tuple

import torch


class BridgeScheduler:
    """方向自适应弹性布朗桥噪声调度器。

    与 ``diffusers.DDPMScheduler`` 接口对齐，但内部实现为布朗桥 SDE。

    核心接口对照：

    ================  =====================  ================================
    NavDP (DDPM)      BridgeScheduler        说明
    ================  =====================  ================================
    add_noise()       add_noise()            前向加噪
    step()            step()                 反向去噪一步
    set_timesteps()   set_timesteps()        设置推理时间步
    ================  =====================  ================================

    Attributes:
        num_train_timesteps: 训练扩散总步数（离散化步数），默认 10（与 NavDP 一致）。
        sigma_base: 数据驱动的基础方差常数（由 compute_sigma_base.py 离线计算），
                    代表训练集中轨迹偏离桥均值的 95 分位数，**不参与梯度反传**。
        sigma_goal: 弹性尾端松弛方差，控制目标端点的不确定性。
                    PointGoal 模式下较小（0.1），NoGoal 模式下可设为较大值。

    导航场景自检：
        1. **正前方目标** (θ_g≈0°): p≈0.80，方差峰值较小，桥均值（直线）可信度高，
           网络仅需做小幅修正即可通过正前方障碍物绕行。
        2. **正后方目标** (θ_g≈180°): p≈0.20，方差峰值接近 σ_base，给予网络充足自由度
           进行 U-turn 或大弧度绕行，不被桥均值（向后直线）锁死。
        3. **极端避障**: 即便桥均值穿过障碍物，σ(t) 提供的方差空间允许网络学会绕行路径。
           Critic 分支进一步在推理时排除穿越障碍的候选轨迹。
        4. **NoGoal 模式**: σ_goal → 大值，t²·σ²_goal 项在 t→1 时占主导，
           桥约束实质消失，退化为类似标准扩散的自由探索。
    """

    def __init__(
        self,
        num_train_timesteps: int = 100,
        sigma_base: float = 1.0,
        sigma_goal: float = 0.5,
        sigma_floor: Optional[float] = None,
        nogoal_front_distance: float = 0.8,
        nogoal_sigma_start: float = 0.03,
        nogoal_sigma_x_end: float = 0.35,
        nogoal_sigma_y_end: float = 0.80,
        nogoal_sigma_theta_end: float = 0.60,
        nogoal_sigma_power: float = 2.0,
        bridge_scale_invariant_sigma: bool = False,
        bridge_anisotropic_xy: bool = True,
        bridge_normal_sigma_ratio: float = 0.25,
        bridge_tangent_sigma_ratio: float = 0.03,
        bridge_theta_sigma_ratio: float = 0.05,
        bridge_virtual_prefix_steps: float = 8.0,
        bridge_anchor_angle_std: float = 0.0,
        bridge_anchor_angle_max: float = 0.0,
        bridge_anchor_uniform_prob: float = 0.0,
        bridge_anchor_edge_prob: float = 0.0,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.sigma_base = sigma_base
        self.sigma_goal = sigma_goal
        self.sigma_floor = sigma_goal if sigma_floor is None else sigma_floor
        self.nogoal_front_distance = nogoal_front_distance
        self.nogoal_sigma_start = nogoal_sigma_start
        self.nogoal_sigma_x_end = nogoal_sigma_x_end
        self.nogoal_sigma_y_end = nogoal_sigma_y_end
        self.nogoal_sigma_theta_end = nogoal_sigma_theta_end
        self.nogoal_sigma_power = nogoal_sigma_power
        self.bridge_scale_invariant_sigma = bridge_scale_invariant_sigma
        self.bridge_anisotropic_xy = bridge_anisotropic_xy
        self.bridge_normal_sigma_ratio = bridge_normal_sigma_ratio
        self.bridge_tangent_sigma_ratio = bridge_tangent_sigma_ratio
        self.bridge_theta_sigma_ratio = bridge_theta_sigma_ratio
        self.bridge_virtual_prefix_steps = float(bridge_virtual_prefix_steps)
        self.bridge_anchor_angle_std = float(max(bridge_anchor_angle_std, 0.0))
        self.bridge_anchor_angle_max = float(max(bridge_anchor_angle_max, 0.0))
        self.bridge_anchor_uniform_prob = float(
            min(max(bridge_anchor_uniform_prob, 0.0), 1.0)
        )
        self.bridge_anchor_edge_prob = float(
            min(max(bridge_anchor_edge_prob, 0.0), 1.0)
        )

        # 推理时使用的时间步序列（由 set_timesteps 设置）
        self._timesteps: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # 公共属性
    # ------------------------------------------------------------------

    @property
    def timesteps(self) -> torch.Tensor:
        """返回推理时间步序列（从 T-1 到 0），与 DDPMScheduler.timesteps 接口一致。"""
        if self._timesteps is None:
            self.set_timesteps(self.num_train_timesteps)
        return self._timesteps

    # ------------------------------------------------------------------
    # 时间步管理
    # ------------------------------------------------------------------

    def set_timesteps(self, num_inference_steps: int) -> None:
        """设置推理时的时间步序列。

        Args:
            num_inference_steps: 推理模型评估次数，范围为 [1, num_train_timesteps]。

        产出:
            self._timesteps: 覆盖完整训练时间轴、从 T-1 降至 0 的 LongTensor。
        """
        if not isinstance(num_inference_steps, int):
            raise TypeError("num_inference_steps must be an integer")
        if not 1 <= num_inference_steps <= self.num_train_timesteps:
            raise ValueError(
                "num_inference_steps must be between 1 and "
                f"{self.num_train_timesteps}, got {num_inference_steps}"
            )

        self._timesteps = torch.linspace(
            self.num_train_timesteps - 1,
            0,
            steps=num_inference_steps,
            dtype=torch.float64,
        ).round().long()

    # ------------------------------------------------------------------
    # 方差计算
    # ------------------------------------------------------------------

    def _normalized_time(self, timesteps: torch.Tensor) -> torch.Tensor:
        """将离散时间步转换为归一化连续时间 t ∈ (0, 1]。

        离散步 k ∈ {0, ..., T-1} 映射为 t = (k+1) / T，
        保证 t > 0（避免 t=0 时方差为零导致的数值问题）。

        Args:
            timesteps: 离散时间步，形状任意。

        Returns:
            归一化时间，形状与输入一致，dtype=float32。
        """
        return (timesteps.float() + 1.0) / self.num_train_timesteps

    def direction_adaptive_exponent(self, theta_g: torch.Tensor) -> torch.Tensor:
        """计算方向自适应指数 p(θ_g) = 0.5 + 0.3 · cos(θ_g)。

        物理含义：
        - 前方目标 (θ_g≈0): p≈0.80 → 方差更小 → 信任桥均值
        - 后方目标 (θ_g≈π): p≈0.20 → 方差更大 → 允许绕行

        Args:
            theta_g: 目标方位角（弧度），形状 (B,) 或 (B, 1)。

        Returns:
            指数 p，形状与 theta_g 一致。
        """
        return 0.5 + 0.3 * torch.cos(theta_g)

    def variance(
        self,
        t_norm: torch.Tensor,
        theta_g: torch.Tensor,
    ) -> torch.Tensor:
        """计算方向自适应桥方差 σ²(t; θ_g)。

        σ²(t; θ_g) = σ²_base · [t(1-t)]^{p(θ_g)} + t² · σ²_goal

        Args:
            t_norm: 归一化时间 ∈ (0, 1]，形状 (B, 1, 1) 或可广播形状。
            theta_g: 目标方位角（弧度），形状 (B,) 或 (B, 1, 1)。

        Returns:
            方差 σ²，形状与 t_norm 广播后一致。

        场景自检 — 方差数值验证：
            设 σ_base=1.0, σ_goal=0.1, t=0.5:
            - 正前方 (θ=0°):  σ² = 1.0·0.25^0.80 + 0.25·0.01 = 0.301 + 0.0025 ≈ 0.30
            - 正侧方 (θ=90°): σ² = 1.0·0.25^0.50 + 0.0025 ≈ 0.50
            - 正后方 (θ=180°): σ² = 1.0·0.25^0.20 + 0.0025 ≈ 0.76
            后方确实提供了更大方差，允许更大幅度偏离桥均值。
        """
        p = self.direction_adaptive_exponent(theta_g)
        # 确保 t(1-t) > 0，clamp 防止数值下溢
        t_prod = (t_norm * (1.0 - t_norm)).clamp(min=1e-8)
        bridge_term = (self.sigma_base ** 2) * (t_prod ** p)
        elastic_term = (t_norm ** 2) * (self.sigma_goal ** 2)
        return bridge_term + elastic_term

    def std(
        self,
        t_norm: torch.Tensor,
        theta_g: torch.Tensor,
    ) -> torch.Tensor:
        """计算标准差 σ(t; θ_g) = √(σ²(t; θ_g))。

        Args:
            t_norm: 归一化时间。
            theta_g: 目标方位角（弧度）。

        Returns:
            标准差 σ。
        """
        return self.variance(t_norm, theta_g).sqrt()

    # ------------------------------------------------------------------
    # Trajectory-time ordered bridge helpers
    # ------------------------------------------------------------------

    def trajectory_time(
        self,
        predict_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return tau_i=(i+1)/T for ordered waypoints, shaped (1, T, 1)."""
        return torch.linspace(
            1.0 / float(predict_size),
            1.0,
            predict_size,
            device=device,
            dtype=dtype,
        ).view(1, predict_size, 1)

    def trajectory_noise_time(
        self,
        predict_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return tau_i=(k+i)/(T+k) for point-goal bridge noise."""
        k = max(self.bridge_virtual_prefix_steps, 0.0)
        idx = torch.arange(1, predict_size + 1, device=device, dtype=dtype)
        tau = (idx + k) / (float(predict_size) + k)
        return tau.view(1, predict_size, 1)

    def _batch_endpoint(
        self,
        value: Optional[torch.Tensor],
        batch_size: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Normalize an endpoint-like tensor to (B, dim)."""
        if value is None:
            return torch.zeros(batch_size, dim, device=device, dtype=dtype)
        value = value.to(device=device, dtype=dtype)
        if value.dim() == 1:
            value = value.unsqueeze(0)
        if value.dim() == 3:
            value = value[:, -1, :]
        if value.shape[0] == 1 and batch_size > 1:
            value = value.expand(batch_size, -1)
        return value

    def _edge_anchor_count(self, available_samples: int) -> int:
        if available_samples <= 0 or self.bridge_anchor_edge_prob <= 0.0:
            return 0
        count = int(math.ceil(float(available_samples) * self.bridge_anchor_edge_prob))
        count = min(max(count, 0), available_samples)
        if available_samples > 1 and count == 1:
            count = 2
        if count % 2 == 1 and count < available_samples:
            count += 1
        elif count % 2 == 1 and count > 1:
            count -= 1
        return min(count, available_samples)

    def _sample_edge_anchor_delta(
        self,
        num: int,
        device: torch.device,
        dtype: torch.dtype,
        signs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if num <= 0 or self.bridge_anchor_angle_max <= 0.0:
            return torch.zeros(num, device=device, dtype=dtype)

        angle_max = self.bridge_anchor_angle_max
        angle_std = self.bridge_anchor_angle_std
        magnitude = torch.empty(num, device=device, dtype=dtype)

        if angle_std > 0.0:
            max_accept = 1.0 - math.exp(-(angle_max ** 2) / (2.0 * angle_std ** 2))
            filled = torch.zeros(num, device=device, dtype=torch.bool)
            for _ in range(16):
                remaining = (~filled).nonzero(as_tuple=False).view(-1)
                if remaining.numel() == 0:
                    break
                candidate = torch.rand(remaining.numel(), device=device, dtype=dtype) * angle_max
                accept_prob = (
                    1.0 - torch.exp(-(candidate ** 2) / (2.0 * angle_std ** 2))
                ) / max(max_accept, 1e-8)
                accepted = torch.rand(remaining.numel(), device=device) < accept_prob
                if accepted.any():
                    accepted_idx = remaining[accepted]
                    magnitude[accepted_idx] = candidate[accepted]
                    filled[accepted_idx] = True
            if (~filled).any():
                fallback_count = int((~filled).sum().item())
                magnitude[~filled] = angle_max * torch.sqrt(
                    torch.rand(fallback_count, device=device, dtype=dtype)
                )
        else:
            magnitude = angle_max * torch.sqrt(torch.rand(num, device=device, dtype=dtype))

        if signs is None:
            signs = torch.where(
                torch.rand(num, device=device) < 0.5,
                torch.full((num,), -1.0, device=device, dtype=dtype),
                torch.ones(num, device=device, dtype=dtype),
            )
        else:
            signs = signs.to(device=device, dtype=dtype).view(num)
        return magnitude * signs

    def sample_bridge_anchor_goals(
        self,
        goal: torch.Tensor,
        origin: Optional[torch.Tensor] = None,
        sample_num: int = 1,
        keep_first_sample: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample radius-preserving bridge anchors around the true point goal.

        The returned anchor is only intended for the Brownian-bridge mean/noise
        geometry. The task goal used by the policy condition and goal-consistency
        score should remain the original point goal.
        """
        if sample_num < 1:
            raise ValueError("sample_num must be >= 1")

        if goal.dim() == 1:
            base_batch = 1
        else:
            base_batch = goal.shape[0]
        dim = goal.shape[-1]
        device = goal.device
        dtype = goal.dtype
        goal_b = self._batch_endpoint(goal, base_batch, dim, device, dtype)
        origin_b = self._batch_endpoint(origin, base_batch, dim, device, dtype)

        goal_rep = goal_b.repeat(sample_num, 1)
        origin_rep = origin_b.repeat(sample_num, 1)
        vec_xy = goal_rep[:, :2] - origin_rep[:, :2]
        radius = torch.norm(vec_xy, dim=-1)
        base_theta = torch.atan2(vec_xy[:, 1], vec_xy[:, 0])

        if self.bridge_anchor_angle_max <= 0.0 and self.bridge_anchor_angle_std <= 0.0:
            return goal_rep, base_theta

        num = goal_rep.shape[0]
        delta = torch.zeros(num, device=device, dtype=dtype)
        if self.bridge_anchor_angle_std > 0.0:
            delta = torch.randn(num, device=device, dtype=dtype) * self.bridge_anchor_angle_std
        if self.bridge_anchor_angle_max > 0.0:
            delta = delta.clamp(
                min=-self.bridge_anchor_angle_max,
                max=self.bridge_anchor_angle_max,
            )
            if self.bridge_anchor_uniform_prob > 0.0:
                uniform_delta = (
                    torch.rand(num, device=device, dtype=dtype) * 2.0 - 1.0
                ) * self.bridge_anchor_angle_max
                use_uniform = (
                    torch.rand(num, device=device) < self.bridge_anchor_uniform_prob
                )
                delta = torch.where(use_uniform, uniform_delta, delta)

        if self.bridge_anchor_edge_prob > 0.0 and self.bridge_anchor_angle_max > 0.0:
            if sample_num > 1:
                start_sample = 1 if keep_first_sample else 0
                edge_count = self._edge_anchor_count(sample_num - start_sample)
                if edge_count > 0:
                    delta_view = delta.view(sample_num, base_batch)
                    edge_signs = torch.where(
                        torch.arange(edge_count, device=device) % 2 == 0,
                        torch.full((edge_count,), -1.0, device=device, dtype=dtype),
                        torch.ones(edge_count, device=device, dtype=dtype),
                    )
                    edge_signs = edge_signs.view(edge_count, 1).expand(edge_count, base_batch)
                    edge_delta = self._sample_edge_anchor_delta(
                        edge_count * base_batch,
                        device,
                        dtype,
                        signs=edge_signs.reshape(-1),
                    ).view(edge_count, base_batch)
                    delta_view[-edge_count:, :] = edge_delta
                    delta = delta_view.reshape(-1)
            else:
                use_edge = torch.rand(num, device=device) < self.bridge_anchor_edge_prob
                if use_edge.any():
                    edge_idx = use_edge.nonzero(as_tuple=False).view(-1)
                    edge_signs = torch.where(
                        torch.arange(edge_idx.numel(), device=device) % 2 == 0,
                        torch.full((edge_idx.numel(),), -1.0, device=device, dtype=dtype),
                        torch.ones(edge_idx.numel(), device=device, dtype=dtype),
                    )
                    delta[edge_idx] = self._sample_edge_anchor_delta(
                        edge_idx.numel(),
                        device,
                        dtype,
                        signs=edge_signs,
                    )

        if keep_first_sample and sample_num > 1:
            delta[:base_batch] = 0.0

        anchor_theta = base_theta + delta
        anchor = goal_rep.clone()
        if dim >= 2:
            anchor[:, 0] = origin_rep[:, 0] + radius * torch.cos(anchor_theta)
            anchor[:, 1] = origin_rep[:, 1] + radius * torch.sin(anchor_theta)
        return anchor, anchor_theta

    def bridge_mean_ordered(
        self,
        goal: torch.Tensor,
        origin: Optional[torch.Tensor],
        shape: tuple,
    ) -> torch.Tensor:
        """Ordered point-goal bridge mean from origin to goal, shaped (B, T, D)."""
        B, T_pred, dim = shape
        device = goal.device
        dtype = goal.dtype
        goal = self._batch_endpoint(goal, B, dim, device, dtype)
        origin = self._batch_endpoint(origin, B, dim, device, dtype)
        tau = self.trajectory_time(T_pred, device, dtype)
        return origin.unsqueeze(1) + tau * (goal.unsqueeze(1) - origin.unsqueeze(1))

    def bridge_mean_nogoal(
        self,
        shape: tuple,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """NoGoal default-front mean; independent of any real sample goal."""
        B, T_pred, dim = shape
        tau = self.trajectory_time(T_pred, device, dtype)
        front = torch.zeros(B, dim, device=device, dtype=dtype)
        front[:, 0] = self.nogoal_front_distance
        return tau * front.unsqueeze(1)

    def pointgoal_noise_params(
        self,
        shape: tuple,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        goal: Optional[torch.Tensor] = None,
        theta_g: Optional[torch.Tensor] = None,
        origin: Optional[torch.Tensor] = None,
    ):
        """Scale-similar point-goal bridge parameters in tangent/normal coordinates.

        The xy standard deviations are proportional to the normalized start-goal
        distance. No floor or max clamp is applied to the point-goal spatial
        variance, so zero-distance goals naturally produce zero spatial noise.
        """
        B, T_pred, dim = shape
        goal_b = self._batch_endpoint(goal, B, dim, device, dtype)
        origin_b = self._batch_endpoint(origin, B, dim, device, dtype)
        mu = self.bridge_mean_ordered(goal_b, origin_b, shape)

        if dim >= 2:
            vec_xy = goal_b[:, :2] - origin_b[:, :2]
        else:
            vec_xy = torch.zeros(B, 2, device=device, dtype=dtype)
        dist = torch.norm(vec_xy, dim=-1)

        if theta_g is None:
            theta_g = torch.atan2(vec_xy[:, 1], vec_xy[:, 0])
        else:
            theta_g = theta_g.to(device=device, dtype=dtype).view(-1)
            if theta_g.shape[0] == 1 and B > 1:
                theta_g = theta_g.expand(B)

        dist_safe = dist.clamp(min=1e-6)
        tangent = vec_xy / dist_safe.view(B, 1)
        normal = torch.stack([-tangent[:, 1], tangent[:, 0]], dim=-1)
        zero_dist = dist <= 1e-6
        if zero_dist.any():
            default_tangent = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
            default_normal = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
            tangent = torch.where(zero_dist.view(B, 1), default_tangent.view(1, 2), tangent)
            normal = torch.where(zero_dist.view(B, 1), default_normal.view(1, 2), normal)

        tau = self.trajectory_noise_time(T_pred, device, dtype)
        p = self.direction_adaptive_exponent(theta_g).view(B, 1, 1)
        t_prod = (tau * (1.0 - tau)).clamp(min=0.0)
        shape_tau = (t_prod / 0.25).clamp(min=0.0).pow(p)
        dist_scale = dist.view(B, 1, 1)

        sigma_normal = dist_scale * self.bridge_normal_sigma_ratio * shape_tau
        sigma_tangent = dist_scale * self.bridge_tangent_sigma_ratio * shape_tau
        sigma_theta = dist_scale * self.bridge_theta_sigma_ratio * shape_tau
        if not self.bridge_anisotropic_xy:
            sigma_tangent = sigma_normal

        sigma_diag = torch.zeros(shape, device=device, dtype=dtype)
        if dim >= 1:
            sigma_diag[..., 0:1] = sigma_tangent
        if dim >= 2:
            sigma_diag[..., 1:2] = sigma_normal
        if dim >= 3:
            sigma_diag[..., 2:3] = sigma_theta

        return mu, tangent, normal, sigma_tangent, sigma_normal, sigma_theta, sigma_diag

    def sample_pointgoal_bridge_noise(
        self,
        shape: tuple,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        goal: Optional[torch.Tensor] = None,
        theta_g: Optional[torch.Tensor] = None,
        origin: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ):
        """Sample scale-similar tangent/normal bridge noise in global x/y/theta."""
        B, T_pred, dim = shape
        (
            mu,
            tangent,
            normal,
            sigma_tangent,
            sigma_normal,
            sigma_theta,
            sigma_diag,
        ) = self.pointgoal_noise_params(shape, device, dtype, goal, theta_g, origin)

        if noise is None:
            eps = torch.randn(shape, device=device, dtype=dtype)
        else:
            eps = noise.to(device=device, dtype=dtype)

        bridge_noise = torch.zeros(shape, device=device, dtype=dtype)
        if dim >= 2:
            eps_tangent = eps[..., 0:1]
            eps_normal = eps[..., 1:2]
            xy_noise = (
                tangent.view(B, 1, 2) * sigma_tangent * eps_tangent
                + normal.view(B, 1, 2) * sigma_normal * eps_normal
            )
            bridge_noise[..., :2] = xy_noise
        elif dim == 1:
            bridge_noise[..., 0:1] = sigma_tangent * eps[..., 0:1]

        if dim >= 3:
            bridge_noise[..., 2:3] = sigma_theta * eps[..., 2:3]

        return bridge_noise, mu, sigma_diag

    def trajectory_std(
        self,
        shape: tuple,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        goal: Optional[torch.Tensor] = None,
        theta_g: Optional[torch.Tensor] = None,
        origin: Optional[torch.Tensor] = None,
        mode: str = "pointgoal",
    ) -> torch.Tensor:
        """Per-waypoint std for the trajectory-time bridge.

        PointGoal uses a Brownian-bridge shape with small endpoint variance.
        NoGoal uses a fixed front-biased prior whose uncertainty grows toward
        the far end, especially laterally and in heading.
        """
        B, T_pred, dim = shape
        if mode == "nogoal":
            tau = self.trajectory_time(T_pred, device, dtype)
            grow = tau.pow(self.nogoal_sigma_power)
            end = torch.tensor(
                [
                    self.nogoal_sigma_x_end,
                    self.nogoal_sigma_y_end,
                    self.nogoal_sigma_theta_end,
                ],
                device=device,
                dtype=dtype,
            ).view(1, 1, 3)
            if dim != 3:
                end = end[..., :dim]
            start = torch.full_like(end, self.nogoal_sigma_start)
            sigma = start + (end - start) * grow
            return sigma.expand(B, T_pred, dim).clamp(min=self.sigma_floor)

        if self.bridge_scale_invariant_sigma:
            _, _, _, _, _, _, sigma_diag = self.pointgoal_noise_params(
                shape, device, dtype, goal=goal, theta_g=theta_g, origin=origin,
            )
            return sigma_diag

        if theta_g is None:
            goal_b = self._batch_endpoint(goal, B, dim, device, dtype)
            origin_b = self._batch_endpoint(origin, B, dim, device, dtype)
            vec = goal_b - origin_b
            theta_g = torch.atan2(vec[:, 1], vec[:, 0])
        else:
            theta_g = theta_g.to(device=device, dtype=dtype).view(-1)
            if theta_g.shape[0] == 1 and B > 1:
                theta_g = theta_g.expand(B)

        tau = self.trajectory_noise_time(T_pred, device, dtype)
        p = self.direction_adaptive_exponent(theta_g).view(B, 1, 1)
        t_prod = (tau * (1.0 - tau)).clamp(min=0.0)
        var = (self.sigma_base ** 2) * (t_prod ** p) + (self.sigma_floor ** 2)
        return var.sqrt().expand(B, T_pred, dim)

    def add_noise_trajectory(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
        theta_g: Optional[torch.Tensor] = None,
        origin: Optional[torch.Tensor] = None,
        mode: str = "pointgoal",
        noise: Optional[torch.Tensor] = None,
    ):
        """Forward process with separate diffusion time s and trajectory time tau."""
        B, T_pred, dim = x0.shape
        device = x0.device
        dtype = x0.dtype
        s_norm = self._normalized_time(timesteps.to(device)).to(dtype=dtype).view(-1, 1, 1)
        if s_norm.shape[0] == 1 and B > 1:
            s_norm = s_norm.expand(B, 1, 1)

        if mode == "nogoal":
            if noise is None:
                noise = torch.randn_like(x0)
            mu = self.bridge_mean_nogoal(x0.shape, device, dtype)
            sigma = self.trajectory_std(x0.shape, device, dtype, mode="nogoal")
            beta = s_norm.clamp(min=1e-6)
            noisy = (1.0 - s_norm) * x0 + s_norm * mu + beta * sigma * noise
            return noisy, noise, mu, sigma, s_norm

        if origin is None:
            origin = torch.zeros(B, dim, device=device, dtype=dtype)

        if self.bridge_scale_invariant_sigma:
            bridge_noise, mu, sigma = self.sample_pointgoal_bridge_noise(
                x0.shape,
                device,
                dtype,
                goal=goal,
                theta_g=theta_g,
                origin=origin,
                noise=noise,
            )
            beta = s_norm.clamp(min=1e-6)
            noisy = (1.0 - s_norm) * x0 + s_norm * mu + beta * bridge_noise
            return noisy, bridge_noise, mu, sigma, s_norm

        if noise is None:
            noise = torch.randn_like(x0)
        mu = self.bridge_mean_ordered(goal, origin, x0.shape)
        sigma = self.trajectory_std(
            x0.shape, device, dtype,
            goal=goal, theta_g=theta_g, origin=origin, mode="pointgoal",
        )
        beta = s_norm.clamp(min=1e-6)
        noisy = (1.0 - s_norm) * x0 + s_norm * mu + beta * sigma * noise
        return noisy, noise, mu, sigma, s_norm

    # ------------------------------------------------------------------
    # 前向加噪（训练）
    # ------------------------------------------------------------------

    def add_noise(
        self,
        x0: torch.Tensor,
        goal: torch.Tensor,
        theta_g: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """布朗桥前向加噪：x_t = (1-t)·x_0 + t·g + σ(t; θ_g)·ε。

        对比 NavDP 的 ``DDPMScheduler.add_noise(action, noise, timesteps)``，
        本方法额外需要 goal 和 theta_g 参数。

        Args:
            x0: 干净轨迹，形状 (B, T_pred, 3)，绝对坐标 (x, y, θ)。
            goal: 目标位置，形状 (B, 1, 3) 或 (B, T_pred, 3)。
                  若为 (B, 1, 3)，会广播到每个预测步。
            theta_g: 目标方位角，形状 (B,)。
            timesteps: 离散时间步，形状 (B,)。
            noise: 可选的预生成高斯噪声 ε ~ N(0, I)，形状 (B, T_pred, 3)。
                   若为 None 则自动生成。

        Returns:
            含噪轨迹 x_t，形状 (B, T_pred, 3)。

        场景自检 — 极端情况：
            1. t=0 (不应出现，因为 _normalized_time 保证 t≥1/T):
               x_t ≈ x_0，几乎无噪声 → 正确。
            2. t→1: x_t ≈ g + σ(1)·ε，σ(1) = σ_goal →
               终点围绕目标波动，噪声幅度由 σ_goal 控制 → 正确。
            3. NoGoal 模式 (goal=0, σ_goal 大):
               x_t 在 t→1 时完全由噪声主导 → 退化为自由扩散 → 正确。
        """
        noisy, _, _, _, _ = self.add_noise_trajectory(
            x0,
            timesteps,
            goal=goal,
            theta_g=theta_g,
            mode="pointgoal",
            noise=noise,
        )
        return noisy

        if noise is None:
            noise = torch.randn_like(x0)

        # 归一化时间：(B,) → (B, 1, 1) 以广播到 (B, T_pred, 3)
        t_norm = self._normalized_time(timesteps).view(-1, 1, 1)

        # 目标方位角：(B,) → (B, 1, 1)
        theta_g_expanded = theta_g.view(-1, 1, 1)

        # 广播 goal 到 (B, T_pred, 3)
        if goal.dim() == 2:
            goal = goal.unsqueeze(1)  # (B, 3) → (B, 1, 3)
        goal = goal.expand_as(x0)

        # 桥均值：(1-t)·x_0 + t·g
        bridge_mean = (1.0 - t_norm) * x0 + t_norm * goal

        # 方向自适应标准差
        sigma = self.std(t_norm, theta_g_expanded)

        # x_t = bridge_mean + σ · ε
        return bridge_mean + sigma * noise

    def step_trajectory(
        self,
        x0_pred: torch.Tensor,
        x_s: torch.Tensor,
        timestep: torch.Tensor,
        prev_timestep: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
        theta_g: Optional[torch.Tensor] = None,
        origin: Optional[torch.Tensor] = None,
        mode: str = "pointgoal",
        eta: float = 0.0,
    ) -> torch.Tensor:
        """DDIM-style reverse step for the trajectory-time bridge process."""
        device = x_s.device
        dtype = x_s.dtype
        timestep = timestep.to(device)
        s_norm = self._normalized_time(timestep).to(dtype=dtype)
        if prev_timestep is None:
            prev_timestep = timestep - 1
        else:
            prev_timestep = torch.as_tensor(prev_timestep, device=device).to(
                dtype=timestep.dtype
            )
        invalid_order = (prev_timestep >= 0) & (prev_timestep >= timestep)
        if torch.any(invalid_order):
            raise ValueError("prev_timestep must be smaller than timestep")
        s_prev_norm = self._normalized_time(prev_timestep).clamp(min=0.0).to(
            dtype=dtype
        )

        B, _, dim = x_s.shape
        s = s_norm.view(-1, 1, 1)
        if s.shape[0] == 1 and B > 1:
            s = s.expand(B, 1, 1)
        s_prev = s_prev_norm.view(-1, 1, 1)
        if s_prev.shape[0] == 1 and B > 1:
            s_prev = s_prev.expand(B, 1, 1)

        if mode == "nogoal":
            mu = self.bridge_mean_nogoal(x_s.shape, device, dtype)
            sigma = self.trajectory_std(x_s.shape, device, dtype, mode="nogoal")
        else:
            if origin is None:
                origin = torch.zeros(B, dim, device=device, dtype=dtype)
            mu = self.bridge_mean_ordered(goal, origin, x_s.shape)
            sigma = None
            if not self.bridge_scale_invariant_sigma:
                sigma = self.trajectory_std(
                    x_s.shape, device, dtype,
                    goal=goal, theta_g=theta_g, origin=origin, mode="pointgoal",
                )

        mean_s = (1.0 - s) * x0_pred + s * mu
        residual = x_s - mean_s
        x_prev = (
            (1.0 - s_prev) * x0_pred
            + s_prev * mu
            + (s_prev / s.clamp(min=1e-6)) * residual
        )

        if eta > 0.0:
            if mode == "pointgoal" and self.bridge_scale_invariant_sigma:
                extra_noise, _, _ = self.sample_pointgoal_bridge_noise(
                    x_s.shape,
                    device,
                    dtype,
                    goal=goal,
                    theta_g=theta_g,
                    origin=origin,
                )
                x_prev = x_prev + eta * s_prev * extra_noise
            else:
                if sigma is None:
                    sigma = self.trajectory_std(
                        x_s.shape, device, dtype,
                        goal=goal, theta_g=theta_g, origin=origin, mode=mode,
                    )
                x_prev = x_prev + eta * s_prev * sigma * torch.randn_like(x_prev)

        return x_prev

    # ------------------------------------------------------------------
    # 反向去噪（推理）
    # ------------------------------------------------------------------

    def step(
        self,
        x0_pred: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        goal: torch.Tensor,
        theta_g: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """布朗桥 DDIM + 随机扰动反向去噪一步。

        x_{t-Δt} = (t-Δt)/t · x_t + Δt/t · x̂_0 + η · σ(t_prev; θ_g) · ε

        Args:
            eta: 随机扰动强度。0 为纯 DDIM 确定性；1 为完整随机扰动。
        """
        return self.step_trajectory(
            x0_pred,
            x_t,
            timestep,
            goal=goal,
            theta_g=theta_g,
            mode="pointgoal",
            eta=eta,
        )

        timestep = timestep.to(x_t.device)
        t_norm = self._normalized_time(timestep).float()
        dt = 1.0 / self.num_train_timesteps

        if t_norm.item() <= dt + 1e-6:
            return x0_pred

        t_prev = t_norm - dt
        coeff_xt = t_prev / t_norm
        coeff_x0 = dt / t_norm

        while coeff_xt.dim() < x_t.dim():
            coeff_xt = coeff_xt.unsqueeze(-1)
            coeff_x0 = coeff_x0.unsqueeze(-1)

        x_det = coeff_xt * x_t + coeff_x0 * x0_pred

        if eta > 0.0:
            # theta_g: (B*S,) → 取第一个元素用于方差估算（各样本共享同一目标方向）
            theta_scalar = theta_g.view(-1)[0:1].view(1, 1, 1)
            sigma = self.std(t_prev.view(1, 1, 1), theta_scalar) * eta
            x_det = x_det + sigma * torch.randn_like(x_det)

        return x_det

    # ------------------------------------------------------------------
    # 推理初始噪声
    # ------------------------------------------------------------------

    def sample_initial_noise(
        self,
        goal: torch.Tensor,
        shape: tuple,
        device: torch.device,
    ) -> torch.Tensor:
        """从目标附近采样推理初始噪声 x_T ~ N(g, σ²_goal · I)。

        **已废弃**：请使用 ``sample_initial_noise_ordered()``，
        该方法将 24 个航点有序分布在 x₀→x_n 区间上。

        本方法保留以兼容旧代码路径。

        Args:
            goal: 目标位置，形状 (B, 1, 3) 或 (B, 3)。
            shape: 输出形状 (B, T_pred, 3)。
            device: 计算设备。

        Returns:
            初始含噪轨迹 x_T，形状 (B, T_pred, 3)。
        """
        noise = torch.randn(shape, device=device)

        # 广播 goal 到 (B, T_pred, 3)
        if goal.dim() == 2:
            goal = goal.unsqueeze(1)
        goal = goal.expand(shape)

        return goal + self.sigma_goal * noise

    def sample_initial_noise_ordered(
        self,
        goal: torch.Tensor,
        origin: torch.Tensor,
        shape: tuple,
        device: torch.device,
    ) -> torch.Tensor:
        """有序区间初始化：24 个航点从起点到导航目标线性分布，方差来自布朗桥。

        航点 i 的初始值：
            均值 μ_i = lerp(origin, goal, (i+1)/T_pred)
            方差 σ²_i = σ²((i+1)/T_pred; θ_g)

        注意：输出的 24 个航点不包含起点，覆盖区间为 (origin, goal]。

        Args:
            goal: 导航目标，形状 (B, 3)，归一化坐标。
            origin: 起点，形状 (B, 3)，通常为零向量（当前机器人位置）。
            shape: 输出形状 (B, T_pred, 3)。
            device: 计算设备。

        Returns:
            初始含噪轨迹，形状 (B, T_pred, 3)，航点有序分布在 (origin, goal] 上。
        """
        B, T_pred, dim = shape

        # 保证 origin 和 goal 都是 (B, 3)
        if origin.dim() == 1:
            origin = origin.unsqueeze(0).expand(B, -1)
        if goal.dim() == 1:
            goal = goal.unsqueeze(0).expand(B, -1)

        if self.bridge_scale_invariant_sigma:
            bridge_noise, mu, _ = self.sample_pointgoal_bridge_noise(
                shape,
                device,
                goal.dtype,
                goal=goal,
                origin=origin,
            )
            return mu + bridge_noise

        # 目标方向和距离
        goal_vec = goal - origin  # (B, 3)

        # 24 个航点从 origin 到 goal 线性插值，不包含起点，包含导航目标。
        t_traj = torch.linspace(
            1.0 / float(T_pred), 1.0, T_pred, device=device
        )  # (T,)
        t_traj = t_traj.view(1, T_pred, 1)  # (1, T, 1)

        origin_exp = origin.unsqueeze(1)  # (B, 1, 3)
        goal_exp = goal.unsqueeze(1)      # (B, 1, 3)
        mu = origin_exp + t_traj * (goal_exp - origin_exp)  # (B, T, 3)

        # 计算方差：使用布朗桥在各航点弧长位置处的方差
        theta_g = torch.atan2(goal_vec[:, 1], goal_vec[:, 0])  # (B,)
        theta_g_exp = theta_g.view(-1, 1, 1)  # (B, 1, 1)
        sigma_per_point = self.trajectory_std(
            shape, device, goal.dtype,
            goal=goal, theta_g=theta_g, origin=origin, mode="pointgoal",
        )

        # 采样：μ_i + σ_i · ε
        noise = torch.randn(shape, device=device, dtype=goal.dtype)
        return mu + sigma_per_point * noise

    def sample_initial_noise_nogoal(
        self,
        shape: tuple,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """NoGoal initialization: default-front mean with growing uncertainty."""
        mu = self.bridge_mean_nogoal(shape, device, dtype)
        sigma = self.trajectory_std(shape, device, dtype, mode="nogoal")
        return mu + sigma * torch.randn(shape, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # 序列化（兼容 config 属性访问模式）
    # ------------------------------------------------------------------

    @property
    def config(self):
        """兼容 DDPMScheduler 的 config 访问模式。

        NavDP 代码中通过 ``self.noise_scheduler.config.num_train_timesteps``
        访问，此处提供相同的属性路径。
        """
        return _SchedulerConfig(self.num_train_timesteps)


class _SchedulerConfig:
    """轻量 config 容器，模仿 diffusers SchedulerConfig 的属性访问。"""

    def __init__(self, num_train_timesteps: int) -> None:
        self.num_train_timesteps = num_train_timesteps
