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
        num_train_timesteps: int = 10,
        sigma_base: float = 1.0,
        sigma_goal: float = 0.1,
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
    ) -> torch.Tensor:
        """布朗桥 DDIM 确定性反向去噪一步。

        更新公式（docs/Bridge-DP推导.md §8.1）：
            x_{t-Δt} = (t-Δt)/t · x_t + Δt/t · x̂_0

        当 t 到达最后一步（t_norm = 1/T），直接返回 x̂_0。

        对比 NavDP 的 ``DDPMScheduler.step(model_output, timestep, sample)``：
        - NavDP: model_output 是预测噪声 ε，内部做 DDPM 反向
        - Bridge-DP: x0_pred 是直接预测的干净轨迹 x̂_0，做布朗桥后验更新

        Args:
            x0_pred: 网络预测的干净轨迹 x̂_0，形状 (B, T_pred, 3)。
            x_t: 当前含噪轨迹，形状 (B, T_pred, 3)。
            timestep: 当前离散时间步（标量或形状 (1,)）。
            goal: 目标位置，形状 (B, 1, 3) 或 (B, T_pred, 3)。
            theta_g: 目标方位角，形状 (B,)。

        Returns:
            去噪后的轨迹 x_{t-Δt}，形状 (B, T_pred, 3)。

        场景自检 — 确定性采样的合理性：
            DDIM 确定性采样不引入额外随机噪声，好处：
            1. 推理结果可复现（给定相同 x_T 和 model weights）
            2. 导航策略更稳定（避免每帧输出轨迹抖动）
            3. 多样性由初始噪声 x_T ~ N(g, σ²_goal) 提供
        """
        # 归一化当前时间和步长
        t_norm = self._normalized_time(timestep).float()
        dt = 1.0 / self.num_train_timesteps  # Δt = 1/T

        # 如果已经是最后一步，直接返回预测结果
        if t_norm.item() <= dt + 1e-6:
            return x0_pred

        # 确定性 DDIM 桥更新
        t_prev = t_norm - dt
        # x_{t-Δt} = (t-Δt)/t · x_t + Δt/t · x̂_0
        coeff_xt = t_prev / t_norm
        coeff_x0 = dt / t_norm

        # 广播标量系数到张量维度
        while coeff_xt.dim() < x_t.dim():
            coeff_xt = coeff_xt.unsqueeze(-1)
            coeff_x0 = coeff_x0.unsqueeze(-1)

        return coeff_xt * x_t + coeff_x0 * x0_pred

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

        对比 NavDP 的 ``torch.randn(...)``（标准高斯），Bridge-DP 的初始噪声
        以目标为中心，幅度由 σ_goal 控制。

        Args:
            goal: 目标位置，形状 (B, 1, 3) 或 (B, 3)。
            shape: 输出形状 (B, T_pred, 3)。
            device: 计算设备。

        Returns:
            初始含噪轨迹 x_T，形状 (B, T_pred, 3)。

        场景自检 — 初始噪声的物理意义：
            - PointGoal (σ_goal=0.1): x_T 紧密围绕目标位置 → 去噪快速收敛
            - NoGoal (σ_goal=10.0): x_T 几乎是纯随机 → 等价于标准扩散起点
            - 目标在身后: x_T 仍围绕目标采样，去噪过程通过 U-turn 路径到达起点
        """
        noise = torch.randn(shape, device=device)

        # 广播 goal 到 (B, T_pred, 3)
        if goal.dim() == 2:
            goal = goal.unsqueeze(1)
        goal = goal.expand(shape)

        return goal + self.sigma_goal * noise

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
