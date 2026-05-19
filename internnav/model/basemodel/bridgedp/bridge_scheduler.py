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
from typing import Optional

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
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.sigma_base = sigma_base
        self.sigma_goal = sigma_goal

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
            num_inference_steps: 推理步数（通常等于 num_train_timesteps=10）。

        产出:
            self._timesteps: 形如 [9, 8, ..., 1, 0] 的 LongTensor。
        """
        self._timesteps = torch.arange(num_inference_steps - 1, -1, -1).long()

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
        d_max: float,
        shape: tuple,
        device: torch.device,
    ) -> torch.Tensor:
        """有序区间初始化：24 个航点从 x₀ 到 x_n 线性分布，方差来自布朗桥。

        布朗桥构建在 x₀→goal（导航目标）上，但 24 个航点只采样
        前段 [x₀, x_n] 区间，其中 x_n 由 d_max 截断：

            x_n = origin + dir(goal) × min(d_max, ||goal - origin||)

        航点 i 的初始值：
            均值 μ_i = lerp(origin, x_n, i/23)
            方差 σ²_i = σ²(s_i; θ_g)，s_i 为该航点在完整桥上的位置参数

        当 ||goal - origin|| ≤ d_max 时，x_n ≈ goal，轨迹覆盖全程。

        Args:
            goal: 导航目标，形状 (B, 3)，归一化坐标。
            origin: 起点，形状 (B, 3)，通常为零向量（当前机器人位置）。
            d_max: 归一化空间中的最大轨迹直线距离（由离线标定）。
            shape: 输出形状 (B, T_pred, 3)。
            device: 计算设备。

        Returns:
            初始含噪轨迹，形状 (B, T_pred, 3)，航点有序分布在 [origin, x_n] 上。

        场景自检：
            1. goal 在 3m 外 (归一化 0.6), d_max=0.85:
               x_n = goal（0.6 < 0.85），航点覆盖全程 → 正确。
            2. goal 在 10m 外 (归一化 2.0), d_max=0.85:
               x_n = dir(goal) × 0.85，航点只覆盖前 4.25m → 正确。
            3. NoGoal (goal=0): x_n = origin = 0，
               所有航点 μ_i=0，方差小 → 退化为原点附近采样 → 需要 NoGoal 走独立路径。
        """
        B, T_pred, dim = shape

        # 保证 origin 和 goal 都是 (B, 3)
        if origin.dim() == 1:
            origin = origin.unsqueeze(0).expand(B, -1)
        if goal.dim() == 1:
            goal = goal.unsqueeze(0).expand(B, -1)

        # 目标方向和距离
        goal_vec = goal - origin  # (B, 3)
        goal_dist = goal_vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # (B, 1)
        goal_dir = goal_vec / goal_dist  # (B, 3) 单位方向

        # 合理轨迹终点距离 = min(d_max, ||goal - origin||)
        traj_dist = torch.clamp(goal_dist, max=d_max)  # (B, 1)
        x_n = origin + goal_dir * traj_dist  # (B, 3) 轨迹终点

        # 24 个航点从 origin 到 x_n 线性插值
        t_traj = torch.linspace(0, 1, T_pred, device=device)  # (T,)
        t_traj = t_traj.view(1, T_pred, 1)  # (1, T, 1)

        origin_exp = origin.unsqueeze(1)  # (B, 1, 3)
        x_n_exp = x_n.unsqueeze(1)        # (B, 1, 3)
        mu = origin_exp + t_traj * (x_n_exp - origin_exp)  # (B, T, 3)

        # 每个航点在完整布朗桥（x₀→goal）上的位置参数 s_i
        # s_i = (i/23) × (traj_dist / goal_dist)
        s_ratio = (traj_dist / goal_dist).unsqueeze(1)  # (B, 1, 1)
        s_values = t_traj * s_ratio  # (B, T, 1)

        # 计算方差：使用布朗桥在 s_i 处的方差
        theta_g = torch.atan2(goal_vec[:, 1], goal_vec[:, 0])  # (B,)
        theta_g_exp = theta_g.view(-1, 1, 1)  # (B, 1, 1)
        sigma_per_point = self.std(s_values, theta_g_exp)  # (B, T, 1)

        # 采样：μ_i + σ(s_i) · ε
        noise = torch.randn(shape, device=device)
        return mu + sigma_per_point * noise

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
