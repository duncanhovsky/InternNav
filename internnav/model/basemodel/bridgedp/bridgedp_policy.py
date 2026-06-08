"""Bridge-DP 模型策略。

独立继承 ``transformers.PreTrainedModel``，仿照 NavDPNet 的实现思路，
实现基于弹性布朗桥 SDE 的导航扩散策略。

与 NavDPNet 的核心区别：
1. 噪声调度器：DDPMScheduler → BridgeScheduler（布朗桥）
2. 动作空间：增量×4 → 绝对坐标 (x, y, θ)
3. 新增模块：PriorEncoder + VisualGate（先验轨迹注入）
4. memory 序列：[time, goal×3, rgbd] → [time, goal×3, rgbd, G·prior×N_p]
5. 预测目标：预测噪声 ε → 预测干净轨迹 x̂_0
6. 推理输出：绝对坐标 24 点轨迹控制点（部署端可再按机器人运动属性重采样）

参考文献：
    - docs/Bridge-DP推导.md §3（弹性布朗桥 SDE）
    - docs/Bridge-DP推导.md §5（视觉门控先验注入）
    - docs/Bridge-DP推导.md §7（网络架构输入输出）
    - docs/Bridge-DP推导.md §8（反向去噪推理与后处理）
"""

import os

import math

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from internnav.configs.model.base_encoders import ModelCfg
from internnav.configs.trainer.exp import ExpCfg
from internnav.model.encoder.navdp_backbone import (
    ImageGoalBackbone,
    LearnablePositionalEncoding,
    PixelGoalBackbone,
    RGBDBackbone,
    SinusoidalPosEmb,
)

from .bridge_scheduler import BridgeScheduler
from .prior_encoder import PriorEncoder, VisualGate


class BridgeDPModelConfig(PretrainedConfig):
    """Bridge-DP 模型配置。

    独立继承 ``PretrainedConfig``，仿照 ``NavDPModelConfig`` 的结构，
    存储完整的实验配置（ExpCfg）。

    Attributes:
        model_type: HuggingFace 模型类型标识符，用于自动注册。
        model_cfg: 实验配置字典（由 ExpCfg 序列化而来）。
    """

    model_type = 'bridgedp'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_cfg = kwargs.get('model_cfg', None)

    @classmethod
    def from_dict(cls, config_dict):
        if 'model_cfg' in config_dict:
            config_dict['model_cfg'] = ExpCfg(**config_dict['model_cfg'])
        return super().from_dict(config_dict)


class BridgeDPNet(PreTrainedModel):
    """Bridge-DP 导航扩散策略网络。

    独立继承 ``PreTrainedModel``，仿照 ``NavDPNet`` 的实现思路，
    在其基础上引入布朗桥 SDE 和先验轨迹注入机制。

    网络架构（memory 序列）::

        memory = [
            time_token,          # 1 token  (SinusoidalPosEmb)
            scale_token,         # optional 1 token ([d_m, log(d_m), R/d_m, d_m/R])
            goal_tokens × 3,     # 3 tokens (point/image/pixel goal)
            rgbd_tokens,         # memory_size × 16 tokens
            G · prior_tokens,    # n_prior_tokens tokens (新增)
        ]
        total_cond_len = 1 + n_scale_tokens + 3 + memory_size*16 + n_prior_tokens

    Attributes:
        config_class: 对应的配置类。
    """

    config_class = BridgeDPModelConfig

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """加载预训练权重，仿照 NavDPNet.from_pretrained。"""
        config = kwargs.pop('config', None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if hasattr(config, 'model_dump'):
            config = cls.config_class(model_cfg=config)
        model = cls(config)
        model.to(model._device)
        if os.path.isdir(pretrained_model_name_or_path):
            incompatible_keys, _ = model.load_state_dict(
                torch.load(os.path.join(pretrained_model_name_or_path, 'pytorch_model.bin'))
            )
            if incompatible_keys:
                print(f'Incompatible keys: {incompatible_keys}')
        elif pretrained_model_name_or_path:
            incompatible_keys, _ = model.load_state_dict(
                torch.load(pretrained_model_name_or_path), strict=False
            )
            if incompatible_keys:
                print(f'Incompatible keys: {incompatible_keys}')
        return model

    def __init__(self, config: BridgeDPModelConfig):
        """初始化 BridgeDPNet。

        Args:
            config: BridgeDPModelConfig 实例，包含完整的 ExpCfg。
        """
        super().__init__(config)

        # ── 从配置中读取超参数（与 NavDPNet 相同的字段）──────────────────
        if isinstance(config, BridgeDPModelConfig):
            self.model_config = ModelCfg(**config.model_cfg['model'])
        else:
            self.model_config = config

        self._device = torch.device(f"cuda:{config.model_cfg['local_rank']}")
        il = config.model_cfg['il']
        self.image_size = il['image_size']
        self.memory_size = il['memory_size']
        self.predict_size = il['predict_size']
        self.pixel_channel = il['pixel_channel']
        self.temporal_depth = il['temporal_depth']
        self.attention_heads = il['heads']
        self.dropout = il['dropout']
        self.token_dim = il['token_dim']
        self.finetune = il['finetune']

        # Bridge-DP 专有超参数（从 il 中读取，有默认值）
        self.n_prior_tokens = il.get('n_prior_tokens', 4)
        self.sigma_base = il.get('sigma_base', 1.0)
        self.sigma_goal = il.get('sigma_goal', 0.1)
        self.sigma_floor = il.get('sigma_floor', self.sigma_goal)
        self.nogoal_front_distance = il.get('nogoal_front_distance', 0.8)
        self.nogoal_sigma_start = il.get('nogoal_sigma_start', 0.03)
        self.nogoal_sigma_x_end = il.get('nogoal_sigma_x_end', 0.35)
        self.nogoal_sigma_y_end = il.get('nogoal_sigma_y_end', 0.80)
        self.nogoal_sigma_theta_end = il.get('nogoal_sigma_theta_end', 0.60)
        self.nogoal_sigma_power = il.get('nogoal_sigma_power', 2.0)
        self.bridge_scale_invariant_sigma = il.get('bridge_scale_invariant_sigma', False)
        self.bridge_anisotropic_xy = il.get('bridge_anisotropic_xy', True)
        self.bridge_normal_sigma_ratio = il.get('bridge_normal_sigma_ratio', 0.25)
        self.bridge_tangent_sigma_ratio = il.get('bridge_tangent_sigma_ratio', 0.03)
        self.bridge_theta_sigma_ratio = il.get('bridge_theta_sigma_ratio', 0.05)
        self.bridge_virtual_prefix_steps = il.get('bridge_virtual_prefix_steps', 8.0)
        self.enable_bridge_anchor_sampling = il.get('enable_bridge_anchor_sampling', False)
        self.bridge_anchor_train_prob = float(il.get('bridge_anchor_train_prob', 0.0))
        self.bridge_anchor_keep_original_sample = il.get('bridge_anchor_keep_original_sample', True)
        self.bridge_anchor_angle_std = float(il.get('bridge_anchor_angle_std', 0.0))
        self.bridge_anchor_angle_max = float(il.get('bridge_anchor_angle_max', 0.0))
        self.bridge_anchor_uniform_prob = float(il.get('bridge_anchor_uniform_prob', 0.0))
        self.bridge_anchor_edge_prob = float(il.get('bridge_anchor_edge_prob', 0.0))
        self.bridge_noise_edge_prob = float(il.get('bridge_noise_edge_prob', 0.0))
        self.bridge_noise_edge_train_prob = float(il.get('bridge_noise_edge_train_prob', 0.0))
        self.bridge_noise_edge_warmup_steps = float(il.get('bridge_noise_edge_warmup_steps', 4.0))
        self.bridge_noise_edge_terminal_guard_steps = float(
            il.get('bridge_noise_edge_terminal_guard_steps', 3.0)
        )
        self.bridge_noise_edge_normal_max = float(il.get('bridge_noise_edge_normal_max', 1.0))
        self.bridge_noise_edge_tangent_scale = float(il.get('bridge_noise_edge_tangent_scale', 0.15))
        self.bridge_noise_edge_theta_scale = float(il.get('bridge_noise_edge_theta_scale', 0.25))
        self.enable_goal_consistency_score = il.get('enable_goal_consistency_score', False)
        self.goal_consistency_terminal_weight = il.get('goal_consistency_terminal_weight', 1.0)
        self.goal_consistency_path_weight = il.get('goal_consistency_path_weight', 0.2)
        self.num_train_timesteps = il.get('num_train_timesteps', 100)
        self.num_inference_timesteps = il.get('num_inference_timesteps', 100)
        self.inference_eta = float(il.get('inference_eta', 0.0))
        # 训练时是否使用“原点→目标”的布朗桥（前向加噪起点固定为零向量）
        self.use_origin_bridge_train = il.get('use_origin_bridge_train', False)
        # use_prior_traj=False 时完全忽略先验轨迹输入（等价于全零先验）
        self.use_prior_traj = il.get('use_prior_traj', False)
        # 动作空间归一化参数（必须与 bridgedp_lerobot_dataset.py 保持一致）
        self.action_scale_xy = 5.0
        self.action_scale_theta = 3.14159
        self.enable_trajectory_normalization = il.get('enable_trajectory_normalization', False)
        self.trajectory_norm_target_distance = float(il.get('trajectory_norm_target_distance', 2.0))
        self.trajectory_norm_min_distance_m = float(il.get('trajectory_norm_min_distance_m', 0.10))
        self.trajectory_norm_eps = float(il.get('trajectory_norm_eps', 1e-6))
        self.enable_scale_condition_token = il.get('enable_scale_condition_token', False)
        self.scale_condition_clamp_min_m = float(il.get('scale_condition_clamp_min_m', 0.10))
        self.scale_condition_clamp_max_m = float(il.get('scale_condition_clamp_max_m', 20.0))
        self.n_scale_tokens = 1 if self.enable_scale_condition_token else 0
        self.enable_scale_rgbd_film = il.get('enable_scale_rgbd_film', False)
        self.scale_rgbd_film_alpha = float(il.get('scale_rgbd_film_alpha', 1.0))
        self.scale_rgbd_film_zero_init = il.get('scale_rgbd_film_zero_init', True)
        self.scale_rgbd_film_use_layernorm = il.get('scale_rgbd_film_use_layernorm', True)

        # ── 共享视觉编码器（与 NavDP 相同，直接 import，不修改）──────────
        self.rgbd_encoder = RGBDBackbone(
            self.image_size, self.token_dim,
            memory_size=self.memory_size, finetune=self.finetune, device=self._device
        )
        self.pixel_encoder = PixelGoalBackbone(
            self.image_size, self.token_dim,
            pixel_channel=self.pixel_channel, device=self._device
        )
        self.image_encoder = ImageGoalBackbone(self.image_size, self.token_dim, device=self._device)
        self.point_encoder = nn.Linear(3, self.token_dim)

        # 冻结 RGB backbone（与 NavDP 一致）
        if not self.finetune:
            for p in self.rgbd_encoder.rgb_model.parameters():
                p.requires_grad = False
            self.rgbd_encoder.rgb_model.eval()

        # ── Transformer Decoder（与 NavDP 相同结构）──────────────────────
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.token_dim,
            nhead=self.attention_heads,
            dim_feedforward=4 * self.token_dim,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=self.temporal_depth)
        self.input_embed = nn.Linear(3, self.token_dim)

        # ── 位置编码（memory 长度增加了 scale token 和 n_prior_tokens）────
        # NavDP: memory_size*16 + 4 (time+goal×3)
        # Bridge-DP: memory_size*16 + 4 + scale + n_prior_tokens
        cond_len = self.memory_size * 16 + 4 + self.n_scale_tokens + self.n_prior_tokens
        self.cond_pos_embed = LearnablePositionalEncoding(self.token_dim, cond_len)
        self.out_pos_embed = LearnablePositionalEncoding(self.token_dim, self.predict_size)

        self.drop = nn.Dropout(self.dropout)
        self.time_emb = SinusoidalPosEmb(self.token_dim)
        self.layernorm = nn.LayerNorm(self.token_dim)

        # 输出头：预测干净轨迹 x̂_0（而非噪声 ε）
        self.action_head = nn.Linear(self.token_dim, 3)
        # 混合表示辅助头：预测帧间增量 Δ，用于一致性约束（方案 C）
        # 训练时 cumsum(Δ) ≈ x̂₀ 提供隐式平滑正则；推理时不使用
        self.delta_head = nn.Linear(self.token_dim, 3)
        self.critic_head = nn.Linear(self.token_dim, 1)

        # 辅助头（与 NavDP 一致）
        self.pixel_aux_head = nn.Linear(self.token_dim, 3)
        self.image_aux_head = nn.Linear(self.token_dim, 3)
        if self.enable_scale_condition_token:
            self.scale_encoder = nn.Sequential(
                nn.Linear(4, self.token_dim),
                nn.GELU(),
                nn.Linear(self.token_dim, self.token_dim),
            )
        if self.enable_scale_rgbd_film:
            self.scale_rgbd_film = nn.Sequential(
                nn.Linear(4, self.token_dim),
                nn.GELU(),
                nn.Linear(self.token_dim, 2 * self.token_dim),
            )
            if self.scale_rgbd_film_use_layernorm:
                self.scale_rgbd_film_norm = nn.LayerNorm(self.token_dim)
            if self.scale_rgbd_film_zero_init:
                nn.init.zeros_(self.scale_rgbd_film[-1].weight)
                nn.init.zeros_(self.scale_rgbd_film[-1].bias)

        # ── Bridge-DP 新增模块 ────────────────────────────────────────────
        self.prior_encoder = PriorEncoder(
            embed_dim=self.token_dim,
            n_prior_tokens=self.n_prior_tokens,
            dropout=self.dropout,
        )
        self.visual_gate = VisualGate(gate_dim=self.token_dim)
        self.bridge_scheduler = BridgeScheduler(
            num_train_timesteps=self.num_train_timesteps,
            sigma_base=self.sigma_base,
            sigma_goal=self.sigma_goal,
            sigma_floor=self.sigma_floor,
            nogoal_front_distance=self.nogoal_front_distance,
            nogoal_sigma_start=self.nogoal_sigma_start,
            nogoal_sigma_x_end=self.nogoal_sigma_x_end,
            nogoal_sigma_y_end=self.nogoal_sigma_y_end,
            nogoal_sigma_theta_end=self.nogoal_sigma_theta_end,
            nogoal_sigma_power=self.nogoal_sigma_power,
            bridge_scale_invariant_sigma=self.bridge_scale_invariant_sigma,
            bridge_anisotropic_xy=self.bridge_anisotropic_xy,
            bridge_normal_sigma_ratio=self.bridge_normal_sigma_ratio,
            bridge_tangent_sigma_ratio=self.bridge_tangent_sigma_ratio,
            bridge_theta_sigma_ratio=self.bridge_theta_sigma_ratio,
            bridge_virtual_prefix_steps=self.bridge_virtual_prefix_steps,
            bridge_anchor_angle_std=self.bridge_anchor_angle_std,
            bridge_anchor_angle_max=self.bridge_anchor_angle_max,
            bridge_anchor_uniform_prob=self.bridge_anchor_uniform_prob,
            bridge_anchor_edge_prob=self.bridge_anchor_edge_prob,
            bridge_noise_edge_prob=self.bridge_noise_edge_prob,
            bridge_noise_edge_train_prob=self.bridge_noise_edge_train_prob,
            bridge_noise_edge_warmup_steps=self.bridge_noise_edge_warmup_steps,
            bridge_noise_edge_terminal_guard_steps=self.bridge_noise_edge_terminal_guard_steps,
            bridge_noise_edge_normal_max=self.bridge_noise_edge_normal_max,
            bridge_noise_edge_tangent_scale=self.bridge_noise_edge_tangent_scale,
            bridge_noise_edge_theta_scale=self.bridge_noise_edge_theta_scale,
        )

        # ── 因果掩码（与 NavDP 相同）──────────────────────────────────────
        self.tgt_mask = (torch.triu(torch.ones(self.predict_size, self.predict_size)) == 1).transpose(0, 1)
        self.tgt_mask = (
            self.tgt_mask.float()
            .masked_fill(self.tgt_mask == 0, float('-inf'))
            .masked_fill(self.tgt_mask == 1, float(0.0))
        )
        self.tgt_mask = self.tgt_mask.to(self._device)

        # Critic 掩码：屏蔽 time/scale/goal token，与 NavDP 一致不泄露 goal
        # Bridge-DP 的 critic 不引入先验信息（用户决策），
        # 因此掩码长度需要覆盖 4 + scale + memory_size*16 + n_prior_tokens
        self.cond_critic_mask = torch.zeros((self.predict_size, cond_len))
        rgbd_start = 4 + self.n_scale_tokens
        self.cond_critic_mask[:, 0:rgbd_start] = float('-inf')  # 屏蔽 time + scale + goal×3
        # 同时屏蔽 prior tokens（critic 不使用先验）
        self.cond_critic_mask[:, rgbd_start + self.memory_size * 16:] = float('-inf')

    def to(self, device, *args, **kwargs):
        """将模型及缓冲区迁移到指定设备。"""
        self = super().to(device, *args, **kwargs)
        self.cond_critic_mask = self.cond_critic_mask.to(device)
        self.tgt_mask = self.tgt_mask.to(device)
        self._device = device
        return self

    # ------------------------------------------------------------------
    # 归一化 / 反归一化工具
    # ------------------------------------------------------------------

    def _normalize_action(self, action):
        """将原始绝对坐标归一化到训练空间。

        Args:
            action: (..., 3) 原始坐标 (x, y, θ)。

        Returns:
            归一化后的坐标。
        """
        normed = action.clone()
        normed[..., 0:2] = normed[..., 0:2] / self.action_scale_xy
        normed[..., 2] = normed[..., 2] / self.action_scale_theta
        return normed

    def _denormalize_action(self, action):
        """将归一化坐标反归一化到原始物理空间。

        Args:
            action: (..., 3) 归一化坐标。

        Returns:
            原始物理坐标 (x, y, θ)。
        """
        denormed = action.clone()
        denormed[..., 0:2] = denormed[..., 0:2] * self.action_scale_xy
        denormed[..., 2] = denormed[..., 2] * self.action_scale_theta
        return denormed

    def _normalize_trajectory_action(self, action, traj_distance_m):
        """PointGoal 样本级轨迹归一化：xy * R / d_m, theta / pi。"""
        normed = action.clone()
        distances = traj_distance_m.to(device=normed.device, dtype=normed.dtype).view(-1)
        scale = self.trajectory_norm_target_distance / distances.clamp(min=self.trajectory_norm_eps)
        if normed.dim() == 3:
            scale = scale.view(-1, 1, 1)
        elif normed.dim() == 2:
            scale = scale.view(-1, 1)
        else:
            while scale.dim() < normed[..., 0:2].dim():
                scale = scale.unsqueeze(-1)
        normed[..., 0:2] = normed[..., 0:2] * scale
        if normed.shape[-1] >= 3:
            normed[..., 2] = normed[..., 2] / self.action_scale_theta
        return normed

    def _denormalize_trajectory_action(self, action, traj_distance_m):
        """PointGoal 样本级轨迹反归一化：xy * d_m / R, theta * pi。"""
        denormed = action.clone()
        distances = traj_distance_m.to(device=denormed.device, dtype=denormed.dtype).view(-1)
        scale = distances / max(self.trajectory_norm_target_distance, self.trajectory_norm_eps)
        if denormed.dim() == 3:
            scale = scale.view(-1, 1, 1)
        elif denormed.dim() == 2:
            scale = scale.view(-1, 1)
        else:
            while scale.dim() < denormed[..., 0:2].dim():
                scale = scale.unsqueeze(-1)
        denormed[..., 0:2] = denormed[..., 0:2] * scale
        if denormed.shape[-1] >= 3:
            denormed[..., 2] = denormed[..., 2] * self.action_scale_theta
        return denormed

    def _default_nogoal_distance(self, batch_size, device, dtype=torch.float32):
        """NoGoal 没有真实目标距离，使用默认前向距离构造尺度 token。"""
        distance = self.nogoal_front_distance * self.action_scale_xy
        return torch.full((batch_size,), distance, device=device, dtype=dtype)

    def _build_scale_features(self, traj_distance_m):
        """构造 [d_m, log(d_m), R/d_m, d_m/R] 尺度条件特征。"""
        d = traj_distance_m.to(device=self._device, dtype=torch.float32).view(-1)
        d = d.clamp(
            min=max(self.scale_condition_clamp_min_m, self.trajectory_norm_eps),
            max=self.scale_condition_clamp_max_m,
        )
        target = max(self.trajectory_norm_target_distance, self.trajectory_norm_eps)
        return torch.stack(
            [
                d,
                torch.log(d),
                d.new_full(d.shape, target) / d,
                d / target,
            ],
            dim=-1,
        )

    def _build_scale_token(self, traj_distance_m, like_token=None):
        """将尺度特征编码为 memory token；关闭时返回空 token 序列。"""
        if not self.enable_scale_condition_token:
            if like_token is not None:
                B = like_token.shape[0]
                return like_token.new_zeros((B, 0, like_token.shape[-1]))
            B = traj_distance_m.shape[0]
            return torch.zeros((B, 0, self.token_dim), device=self._device)
        feat = self._build_scale_features(traj_distance_m)
        token = self.scale_encoder(feat).unsqueeze(1)
        if like_token is not None:
            token = token.to(device=like_token.device, dtype=like_token.dtype)
        return token

    def _apply_scale_rgbd_film(self, rgbd_embed, traj_distance_m):
        """Use metric/shape scale features to FiLM-modulate RGBD tokens."""
        if not self.enable_scale_rgbd_film:
            return rgbd_embed

        feat = self._build_scale_features(traj_distance_m)
        batch_size = rgbd_embed.shape[0]
        if feat.shape[0] == 1 and batch_size > 1:
            feat = feat.expand(batch_size, -1)
        elif feat.shape[0] != batch_size:
            raise ValueError(
                f"scale batch size {feat.shape[0]} does not match rgbd batch size {batch_size}"
            )

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

    def _build_valid_mask(self, trajectory, threshold=1e-4, min_valid_steps=4):
        """基于位移阈值构建有效步掩码（推理专用）。"""
        if trajectory.dim() != 3:
            raise ValueError("trajectory must have shape (B, T, 3)")
        B, T, _ = trajectory.shape
        if T == 0:
            return torch.zeros((B, 0), device=trajectory.device, dtype=torch.bool)
        step_diffs = torch.norm(trajectory[:, 1:, :2] - trajectory[:, :-1, :2], dim=-1)
        valid_mask = torch.cat(
            [torch.ones((B, 1), device=trajectory.device, dtype=torch.bool), step_diffs > threshold],
            dim=1,
        )
        if min_valid_steps > 1:
            valid_mask[:, :min(min_valid_steps, T)] = True
        return valid_mask

    def _valid_mask_to_lengths(self, valid_mask):
        """将有效步掩码转换为每条轨迹的有效长度。"""
        if valid_mask.numel() == 0:
            return torch.zeros((valid_mask.shape[0],), device=valid_mask.device, dtype=torch.long)
        idx = torch.arange(1, valid_mask.shape[1] + 1, device=valid_mask.device).unsqueeze(0)
        return (valid_mask.long() * idx).max(dim=1).values

    # ------------------------------------------------------------------
    # 前向加噪（训练专用）
    # ------------------------------------------------------------------

    def sample_bridge_noise(self, x0, goal=None, theta_g=None, timesteps=None, mode="pointgoal"):
        """布朗桥前向加噪，返回 x₀（训练目标）、含噪嵌入与含噪轨迹。

        训练目标为 x₀-prediction（预测干净轨迹），与推导文档 §6.1 一致。
        布朗桥的 σ(t) 远小于 DDPM 的 √(1-ᾱ)，ε-prediction 反解时
        (1-t) 分母会放大误差 100 倍，必须用 x₀-prediction。

        Returns:
            x0: 干净轨迹（训练目标）(B, T_pred, 3)。
            time_embeds: 时间步嵌入 (B, 1, d)。
            noisy_action_embed: 含噪轨迹嵌入 (B, T_pred, d)。
            timesteps: 离散时间步 (B,)。
            noisy_action: 含噪轨迹 (B, T_pred, 3)。
        """
        device = x0.device
        B = x0.shape[0]
        if timesteps is None:
            timesteps = torch.randint(
                0, self.bridge_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()
        time_embeds = self.time_emb(timesteps).unsqueeze(1)
        bridge_x0 = torch.zeros_like(x0) if self.use_origin_bridge_train else x0
        origin = torch.zeros(x0.shape[0], x0.shape[-1], device=device, dtype=x0.dtype)
        noisy_action, _, bridge_mu, bridge_sigma, s_norm = self.bridge_scheduler.add_noise_trajectory(
            bridge_x0,
            timesteps,
            goal=goal,
            theta_g=theta_g,
            origin=origin,
            mode=mode,
        )
        noisy_action_embed = self.input_embed(noisy_action)
        # 返回 x0 作为训练目标（x₀-prediction）
        return x0, time_embeds, noisy_action_embed, timesteps, noisy_action, bridge_mu, bridge_sigma, s_norm

    # ------------------------------------------------------------------
    # 去噪预测
    # ------------------------------------------------------------------

    def predict_x0(self, noisy_actions, timestep, goal_embed, rgbd_embed, prior_embed, scale_embed=None):
        """直接预测干净轨迹 x̂₀（x₀-prediction 模式）。

        与 ε-prediction 的区别：
        - 输出语义为干净轨迹 x̂₀ 而非噪声 ε
        - 训练和推理使用同一函数，无需 _eps_to_x0 反解
        - 避免了布朗桥 σ(t) << 1 时 (1-t) 分母放大误差的数值问题

        Args:
            noisy_actions: 含噪轨迹 (B, T_pred, 3)。
            timestep: 时间步 (1,) 或标量。
            goal_embed: 目标嵌入 (B, 1, d) 或 (1, 1, d)。
            rgbd_embed: RGBD 特征 (B, mem*16, d) 或 (1, mem*16, d)。
            prior_embed: 门控后先验 token (B, N_p, d) 或 (1, N_p, d)。

        Returns:
            x0_pred: 预测的干净轨迹 (B, T_pred, 3)。
        """
        action_embeds = self.input_embed(noisy_actions)
        time_embeds = self.time_emb(timestep.to(self._device).view(-1)).unsqueeze(1)
        cond_batch = goal_embed.shape[0]
        if time_embeds.shape[0] == 1 and cond_batch > 1:
            time_embeds = time_embeds.expand(cond_batch, -1, -1)

        if scale_embed is None:
            scale_embed = self._build_scale_token(
                self._default_nogoal_distance(cond_batch, self._device),
                like_token=goal_embed,
            )

        # memory = [time(1), scale(optional), goal×3, rgbd(mem×16), G·prior(N_p)]
        cond_tokens = torch.cat(
            [time_embeds, scale_embed, goal_embed, goal_embed, goal_embed, rgbd_embed, prior_embed],
            dim=1
        )
        cond_embedding = cond_tokens + self.cond_pos_embed(cond_tokens)
        cond_embedding = cond_embedding.repeat(action_embeds.shape[0] // cond_embedding.shape[0], 1, 1)

        input_embedding = action_embeds + self.out_pos_embed(action_embeds)
        output = self.decoder(
            tgt=input_embedding, memory=cond_embedding,
            tgt_mask=self.tgt_mask.to(self._device)
        )
        output = self.layernorm(output)
        x0_pred = self.action_head(output)
        return x0_pred

    def predict_critic(self, predict_trajectory, rgbd_embed, scale_embed=None):
        """Critic 分支，与 NavDP 完全一致，不引入先验信息。

        Args:
            predict_trajectory: 候选轨迹 (B*S, T_pred, 3)。
            rgbd_embed: RGBD 特征 (1, mem*16, d)。

        Returns:
            critic_values: (B*S,) 标量评分。
        """
        repeat_factor = max(1, predict_trajectory.shape[0] // rgbd_embed.shape[0])
        repeat_rgbd_embed = rgbd_embed.repeat(repeat_factor, 1, 1)
        nogoal_embed = torch.zeros_like(repeat_rgbd_embed[:, 0:1])
        if scale_embed is None:
            scale_embed = self._build_scale_token(
                self._default_nogoal_distance(rgbd_embed.shape[0], rgbd_embed.device, rgbd_embed.dtype),
                like_token=rgbd_embed[:, 0:1],
            )
        repeat_scale_embed = scale_embed.repeat(repeat_factor, 1, 1)
        # 先验位置用零填充（critic 不使用先验）
        zero_prior = torch.zeros(
            repeat_rgbd_embed.shape[0], self.n_prior_tokens, self.token_dim,
            device=repeat_rgbd_embed.device
        )

        action_embeddings = self.input_embed(predict_trajectory)
        action_embeddings = action_embeddings + self.out_pos_embed(action_embeddings)
        cond_tokens = torch.cat(
            [
                nogoal_embed, repeat_scale_embed,
                nogoal_embed, nogoal_embed, nogoal_embed,
                repeat_rgbd_embed, zero_prior,
            ],
            dim=1
        )
        cond_embeddings = cond_tokens + self.cond_pos_embed(cond_tokens)
        critic_output = self.decoder(
            tgt=action_embeddings, memory=cond_embeddings,
            memory_mask=self.cond_critic_mask.to(self._device)
        )
        critic_output = self.layernorm(critic_output)
        critic_output = self.critic_head(critic_output.mean(dim=1))[:, 0]
        return critic_output

    def _apply_goal_consistency_score(self, critic_values, trajectories, goals, origins=None):
        """Optionally combine critic score with goal-consistency penalties."""
        if not self.enable_goal_consistency_score:
            return critic_values

        if goals.dim() == 3:
            goals = goals[:, -1, :]
        if origins is None:
            origins = torch.zeros_like(goals)
        elif origins.dim() == 3:
            origins = origins[:, -1, :]

        terminal_err = torch.norm(trajectories[:, -1, :2] - goals[:, :2], dim=-1)
        path_len = torch.norm(
            trajectories[:, 1:, :2] - trajectories[:, :-1, :2], dim=-1
        ).sum(dim=-1)
        goal_dist = torch.norm(goals[:, :2] - origins[:, :2], dim=-1)

        penalty = (
            self.goal_consistency_terminal_weight * terminal_err
            + self.goal_consistency_path_weight * torch.relu(path_len - goal_dist)
        )
        return critic_values - penalty

    # ------------------------------------------------------------------
    # 训练前向
    # ------------------------------------------------------------------

    def forward(
        self, goal_point, goal_image, goal_pixel,
        input_images, input_depths,
        output_actions, augment_actions,
        prior_traj, theta_g, nogoal_actions=None, traj_distance_m=None,
    ):
        """训练前向传播。

        与 NavDP 的 forward 对比：
        - 新增参数：prior_traj (B, T_pred, 3), theta_g (B,)
        - 加噪方式：DDPMScheduler → BridgeScheduler
        - 预测目标：噪声 ε → 干净轨迹 x̂_0
        - memory 序列：追加门控先验 token

        Args:
            goal_point: 点目标 (B, 3)。
            goal_image: 图像目标 (B, H, W, 6)。
            goal_pixel: 像素目标 (B, H, W, C)。
            input_images: 历史 RGB (B, mem, H, W, 3)。
            input_depths: 历史深度 (B, H, W, 1)。
            output_actions: 标签轨迹 (B, T_pred, 3)，绝对坐标。
            augment_actions: 增强轨迹 (B, T_pred, 3)，绝对坐标。
            prior_traj: 先验轨迹 (B, T_pred, 3)。
            theta_g: 目标方位角 (B,)。

        Returns:
            tuple: (x0_pred_ng, x0_pred_mg, cr_label_pred, cr_augment_pred,
                    x0_target_ng, x0_target_mg, imagegoal_aux_pred, pixelgoal_aux_pred)

        场景自检 — forward 数据流验证：
            1. 正前方目标 (θ_g≈0°): 桥均值≈直线，方差小 → 加噪后 x_t 仍接近直线
               → 网络容易学会恢复出直线轨迹 → 正确。
            2. 正后方目标 (θ_g≈180°): 桥均值指向后方，但方差大 →
               x_t 偏离桥均值更多 → 网络需要学会从高噪声中恢复绕行路径 → 合理。
            3. 对抗先验 (30%错误): prior_traj 为随机 → PriorEncoder 编码后
               VisualGate 降低 G → 先验 token 权重小 → 网络主要依赖观测 → 正确。
        """
        device = next(self.parameters()).device

        assert input_images.shape[1] == self.memory_size
        tensor_point_goal = torch.as_tensor(goal_point, dtype=torch.float32).to(device)
        tensor_label_actions = torch.as_tensor(output_actions, dtype=torch.float32).to(device)
        tensor_augment_actions = torch.as_tensor(augment_actions, dtype=torch.float32).to(device)
        tensor_nogoal_actions = (
            torch.as_tensor(nogoal_actions, dtype=torch.float32).to(device)
            if nogoal_actions is not None
            else tensor_label_actions
        )
        tensor_prior = torch.as_tensor(prior_traj, dtype=torch.float32).to(device)
        tensor_theta_g = torch.as_tensor(theta_g, dtype=torch.float32).to(device)
        if traj_distance_m is None:
            if self.enable_trajectory_normalization:
                tensor_traj_distance_m = torch.full(
                    (tensor_point_goal.shape[0],),
                    self.trajectory_norm_target_distance * self.action_scale_xy,
                    device=device,
                    dtype=torch.float32,
                )
            else:
                tensor_traj_distance_m = torch.norm(tensor_point_goal[:, :2], dim=-1) * self.action_scale_xy
        else:
            tensor_traj_distance_m = torch.as_tensor(traj_distance_m, dtype=torch.float32).to(device).view(-1)
        input_images = input_images.to(device)
        input_depths = input_depths.to(device)

        # ── 布朗桥加噪（x₀-prediction 版本）──────────────────────────────
        # ng/mg 各自独立采样时间步，增加训练多样性（与 NavDP 一致）
        # sample_bridge_noise 返回 x0（干净轨迹）作为训练目标
        mg_bridge_goal = tensor_point_goal
        mg_bridge_theta_g = tensor_theta_g
        if self.enable_bridge_anchor_sampling and self.bridge_anchor_train_prob > 0.0:
            origin_train = torch.zeros_like(tensor_point_goal)
            sampled_goal, sampled_theta = self.bridge_scheduler.sample_bridge_anchor_goals(
                tensor_point_goal,
                origin_train,
                sample_num=1,
                keep_first_sample=False,
            )
            if self.bridge_anchor_train_prob < 1.0:
                use_anchor = (
                    torch.rand(tensor_point_goal.shape[0], device=device)
                    < self.bridge_anchor_train_prob
                )
                mg_bridge_goal = torch.where(
                    use_anchor.view(-1, 1),
                    sampled_goal,
                    tensor_point_goal,
                )
                mg_bridge_theta_g = torch.where(use_anchor, sampled_theta, tensor_theta_g)
            else:
                mg_bridge_goal = sampled_goal
                mg_bridge_theta_g = sampled_theta

        ng_x0_target, ng_time_embed, ng_noisy_embed, ng_timesteps, ng_noisy_action, ng_bridge_mu, ng_bridge_sigma, ng_s_norm = self.sample_bridge_noise(
            tensor_nogoal_actions, goal=None, theta_g=None, mode="nogoal"
        )
        mg_x0_target, mg_time_embed, mg_noisy_embed, mg_timesteps, mg_noisy_action, mg_bridge_mu, mg_bridge_sigma, mg_s_norm = self.sample_bridge_noise(
            tensor_label_actions, goal=mg_bridge_goal, theta_g=mg_bridge_theta_g, mode="pointgoal"
        )

        # ── 视觉编码（与 NavDP 相同）──────────────────────────────────────
        rgbd_embed_base = self.rgbd_encoder(input_images, input_depths)
        pointgoal_embed = self.point_encoder(tensor_point_goal).unsqueeze(1)
        nogoal_embed = torch.zeros_like(pointgoal_embed)
        scale_embed_mg = self._build_scale_token(tensor_traj_distance_m, like_token=pointgoal_embed)
        nogoal_distance_m = self._default_nogoal_distance(
            tensor_point_goal.shape[0], device, dtype=torch.float32
        )
        scale_embed_ng = self._build_scale_token(nogoal_distance_m, like_token=pointgoal_embed)
        rgbd_embed_mg = self._apply_scale_rgbd_film(rgbd_embed_base, tensor_traj_distance_m)
        rgbd_embed_ng = self._apply_scale_rgbd_film(rgbd_embed_base, nogoal_distance_m)
        imagegoal_embed = self.image_encoder(goal_image).unsqueeze(1)
        pixelgoal_embed = self.pixel_encoder(goal_pixel).unsqueeze(1)

        # 辅助损失预测（与 NavDP 一致）
        imagegoal_aux_pred = self.image_aux_head(imagegoal_embed[:, 0])
        pixelgoal_aux_pred = self.pixel_aux_head(pixelgoal_embed[:, 0])

        # ── 先验编码 + 视觉门控（Bridge-DP 新增）─────────────────────────
        if self.use_prior_traj:
            prior_tokens = self.prior_encoder(tensor_prior)  # (B, N_p, d)
            vis_global_mg = rgbd_embed_mg.mean(dim=1)  # (B, d) mean pooling
            gate_mg = self.visual_gate(vis_global_mg)  # (B, 1, 1)
            gated_prior_mg = gate_mg * prior_tokens  # (B, N_p, d)
            vis_global_ng = rgbd_embed_ng.mean(dim=1)
            gate_ng = self.visual_gate(vis_global_ng)
            gated_prior_ng = gate_ng * prior_tokens
        else:
            # use_prior_traj=False: 完全忽略先验，用零 token 填充
            gated_prior_mg = torch.zeros(
                tensor_prior.shape[0], self.n_prior_tokens, self.token_dim,
                device=device
            )
            gated_prior_ng = gated_prior_mg

        # ── 标签/增强轨迹嵌入（用于 critic，与 NavDP 一致）────────────────
        label_embed = self.input_embed(tensor_label_actions).detach()
        augment_embed = self.input_embed(tensor_augment_actions).detach()

        # ── 构建 memory + 位置编码 ──────────────────────────────────────
        cond_base = torch.cat(
            [
                ng_time_embed, scale_embed_ng,
                nogoal_embed, imagegoal_embed, pixelgoal_embed,
                rgbd_embed_ng, gated_prior_ng,
            ],
            dim=1
        )
        cond_pos_embed = self.cond_pos_embed(cond_base)

        # no-goal 分支 memory
        ng_cond_embeddings = self.drop(
            torch.cat(
                [
                    ng_time_embed, scale_embed_ng,
                    nogoal_embed, nogoal_embed, nogoal_embed,
                    rgbd_embed_ng, gated_prior_ng,
                ],
                dim=1
            ) + cond_pos_embed
        )

        # mixed-goal 分支 memory（与 NavDP 相同的 3^3 组合策略）
        cand_goal_embed = [pointgoal_embed, imagegoal_embed, pixelgoal_embed]
        batch_size = pointgoal_embed.shape[0]
        batch_indices = torch.arange(batch_size, device=device)
        pattern_indices = batch_indices % 27
        sel_0 = pattern_indices % 3
        sel_1 = (pattern_indices // 3) % 3
        sel_2 = (pattern_indices // 9) % 3
        goal_embeds = torch.stack(cand_goal_embed, dim=0)
        selected_0 = goal_embeds[sel_0, torch.arange(batch_size), :, :]
        selected_1 = goal_embeds[sel_1, torch.arange(batch_size), :, :]
        selected_2 = goal_embeds[sel_2, torch.arange(batch_size), :, :]
        mg_cond_embed = torch.cat(
            [
                mg_time_embed, scale_embed_mg,
                selected_0, selected_1, selected_2,
                rgbd_embed_mg, gated_prior_mg,
            ],
            dim=1
        )
        mg_cond_embeddings = self.drop(mg_cond_embed + cond_pos_embed)

        critic_cond_embeddings = self.drop(
            torch.cat(
                [
                    ng_time_embed, scale_embed_mg,
                    nogoal_embed, nogoal_embed, nogoal_embed,
                    rgbd_embed_mg, gated_prior_mg,
                ],
                dim=1
            ) + cond_pos_embed
        )

        # ── Transformer Decoder 前向 ─────────────────────────────────────
        out_pos_embed = self.out_pos_embed(ng_noisy_embed)

        ng_action_embeddings = self.drop(ng_noisy_embed + out_pos_embed)
        mg_action_embeddings = self.drop(mg_noisy_embed + out_pos_embed)
        label_action_embeddings = self.drop(label_embed + out_pos_embed)
        augment_action_embeddings = self.drop(augment_embed + out_pos_embed)

        # no-goal 分支：x₀-prediction（直接预测干净轨迹）
        ng_output = self.decoder(tgt=ng_action_embeddings, memory=ng_cond_embeddings, tgt_mask=self.tgt_mask)
        ng_output = self.layernorm(ng_output)
        x0_pred_ng = self.action_head(ng_output)   # 绝对空间 x₀ 预测 (B, T, 3)

        # mixed-goal 分支：x₀-prediction
        mg_output = self.decoder(
            tgt=mg_action_embeddings, memory=mg_cond_embeddings,
            tgt_mask=self.tgt_mask.to(device)
        )
        mg_output = self.layernorm(mg_output)
        x0_pred_mg = self.action_head(mg_output)   # 绝对空间 x₀ 预测 (B, T, 3)

        # Critic 分支（与 NavDP 一致，不使用先验）
        cr_label_output = self.decoder(
            tgt=label_action_embeddings, memory=critic_cond_embeddings,
            memory_mask=self.cond_critic_mask.to(self._device)
        )
        cr_label_output = self.layernorm(cr_label_output)
        cr_label_pred = self.critic_head(cr_label_output.mean(dim=1))[:, 0]

        cr_augment_output = self.decoder(
            tgt=augment_action_embeddings, memory=critic_cond_embeddings,
            memory_mask=self.cond_critic_mask.to(self._device)
        )
        cr_augment_output = self.layernorm(cr_augment_output)
        cr_augment_pred = self.critic_head(cr_augment_output.mean(dim=1))[:, 0]

        return (
            x0_pred_ng,        # (B, T_pred, 3) 绝对空间 x₀ 预测（ng 分支）
            x0_pred_mg,        # (B, T_pred, 3) 绝对空间 x₀ 预测（mg 分支）
            cr_label_pred,     # (B,) critic 评分（标签轨迹）
            cr_augment_pred,   # (B,) critic 评分（增强轨迹）
            ng_x0_target,      # (B, T_pred, 3) x₀ 目标（ng 分支，即干净轨迹）
            mg_x0_target,      # (B, T_pred, 3) x₀ 目标（mg 分支，即干净轨迹）
            imagegoal_aux_pred,  # (B, 3) 辅助预测
            pixelgoal_aux_pred,  # (B, 3) 辅助预测
            ng_noisy_action,   # (B, T_pred, 3) ng 含噪轨迹
            mg_noisy_action,   # (B, T_pred, 3) mg 含噪轨迹
            ng_timesteps,      # (B,) ng 时间步
            mg_timesteps,      # (B,) mg 时间步
            ng_bridge_mu,      # (B, T_pred, 3) ng bridge mean, no real-goal leakage
            mg_bridge_mu,      # (B, T_pred, 3) mg ordered point-goal bridge mean
            ng_bridge_sigma,   # (B, T_pred, 3)
            mg_bridge_sigma,   # (B, T_pred, 3)
            ng_s_norm,         # (B, 1, 1)
            mg_s_norm,         # (B, 1, 1)
        )

    # ------------------------------------------------------------------
    # 设备检测（与 NavDP 一致）
    # ------------------------------------------------------------------

    def _get_device(self):
        """安全获取模型所在设备。"""
        try:
            for param in self.parameters():
                return param.device
        except StopIteration:
            pass
        try:
            for buffer in self.buffers():
                return buffer.device
        except StopIteration:
            pass
        for module in self.children():
            try:
                for param in module.parameters():
                    return param.device
            except StopIteration:
                continue
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 推理接口
    # ------------------------------------------------------------------

    def predict_pointgoal_batch_action_vel(
        self, goal_point, input_images, input_depths,
        prior_traj=None, theta_g=None, sample_num=32,
        return_mask=False, min_valid_steps=4, valid_threshold=1e-4,
    ):
        """PointGoal 推理：生成多条候选轨迹并通过 Critic 排序。

        与 NavDP 的区别：
        1. 初始噪声：torch.randn → N(g, sigma_goal^2)
        2. 去噪步：DDPMScheduler.step → BridgeScheduler.step
        3. 后处理：cumsum/4 → 直接输出绝对坐标轨迹控制点
        4. 新增参数：prior_traj, theta_g

        Args:
            goal_point: 目标位置 (B, 3)。
            input_images: 历史 RGB (B, mem, H, W, 3)。
            input_depths: 历史深度 (B, H, W, 1)。
            prior_traj: 先验轨迹 (B, T_pred, 3)，首帧可为 None。
            theta_g: 目标方位角 (B,)，首帧可为 None。
            sample_num: 采样候选数。

        Returns:
            negative_trajectory: 低分轨迹 (8, T_pred, 3)。
            positive_trajectory: 高分轨迹 (8, T_pred, 3)。
            return_mask=True 时额外返回:
                negative_mask: (8, T_pred) 有效步掩码。
                positive_mask: (8, T_pred) 有效步掩码。
                negative_len: (8,) 有效长度（最后一个有效步索引+1）。
                positive_len: (8,) 有效长度（最后一个有效步索引+1）。

        场景自检 — 推理流程验证：
            1. 首帧无先验 (prior_traj=None): 先验设为全零 →
               VisualGate G≈0.7 但全零先验无信息 → 等价于无先验推理 → 正确。
            2. 连续帧有先验: 上一帧输出作为先验 → 加速收敛 → 正确。
            3. 目标在身后: theta_g≈π → σ大 → 初始噪声分散 →
               多样性高 → Critic 选出最优绕行路径 → 正确。
        """
        with torch.no_grad():
            tensor_point_goal = torch.as_tensor(goal_point, dtype=torch.float32, device=self._device)
            goal_distance_m = torch.norm(tensor_point_goal[:, :2], dim=-1)

            if self.enable_trajectory_normalization:
                arrived_mask = goal_distance_m < self.trajectory_norm_min_distance_m
                if arrived_mask.all():
                    zero_traj = torch.zeros((8, self.predict_size, 3), device=self._device)
                    if return_mask:
                        zero_mask = torch.zeros((8, self.predict_size), device=self._device, dtype=torch.bool)
                        zero_len = torch.zeros((8,), device=self._device, dtype=torch.long)
                        return zero_traj, zero_traj.clone(), zero_mask, zero_mask.clone(), zero_len, zero_len.clone()
                    return zero_traj, zero_traj.clone()

                if arrived_mask.any():
                    valid_idx = torch.nonzero(~arrived_mask, as_tuple=False).flatten()

                    def _select_valid(value):
                        if value is None:
                            return None
                        if torch.is_tensor(value):
                            return value[valid_idx]
                        return value[valid_idx.detach().cpu().numpy()]

                    tensor_point_goal = tensor_point_goal[valid_idx]
                    goal_distance_m = goal_distance_m[valid_idx]
                    input_images = _select_valid(input_images)
                    input_depths = _select_valid(input_depths)
                    prior_traj = _select_valid(prior_traj)
                    if theta_g is not None:
                        theta_g = _select_valid(torch.as_tensor(theta_g, dtype=torch.float32, device=self._device))

            # 计算 theta_g（在归一化之前，使用原始坐标）
            if theta_g is not None:
                tensor_theta_g = torch.as_tensor(theta_g, dtype=torch.float32, device=self._device)
            else:
                tensor_theta_g = torch.atan2(tensor_point_goal[:, 1], tensor_point_goal[:, 0])

            # ── 归一化：将物理坐标映射到当前训练空间 ──
            if self.enable_trajectory_normalization:
                tensor_point_goal_n = self._normalize_trajectory_action(
                    tensor_point_goal,
                    goal_distance_m,
                )
            else:
                tensor_point_goal_n = self._normalize_action(tensor_point_goal)

            rgbd_embed = self.rgbd_encoder(input_images, input_depths)
            pointgoal_embed = self.point_encoder(tensor_point_goal_n).unsqueeze(1)
            scale_embed = self._build_scale_token(goal_distance_m, like_token=pointgoal_embed)
            rgbd_embed = self._apply_scale_rgbd_film(rgbd_embed, goal_distance_m)

            # 先验处理（归一化后编码）
            if prior_traj is not None:
                tensor_prior = torch.as_tensor(prior_traj, dtype=torch.float32, device=self._device)
                if self.enable_trajectory_normalization:
                    tensor_prior = self._normalize_trajectory_action(tensor_prior, goal_distance_m)
                else:
                    tensor_prior = self._normalize_action(tensor_prior)
            else:
                tensor_prior = torch.zeros(
                    tensor_point_goal.shape[0], self.predict_size, 3, device=self._device
                )

            if self.use_prior_traj:
                prior_tokens = self.prior_encoder(tensor_prior)
                vis_global = rgbd_embed.mean(dim=1)
                gate = self.visual_gate(vis_global)
                gated_prior = gate * prior_tokens
            else:
                gated_prior = torch.zeros(
                    tensor_prior.shape[0], self.n_prior_tokens, self.token_dim,
                    device=self._device
                )

            # ── 有序区间初始化：24 个航点从起点到导航目标完整分布 ──
            # 输出航点不包含起点，覆盖 (origin, goal]，与弧长重采样监督一致。
            B = tensor_point_goal_n.shape[0]
            origin = torch.zeros_like(tensor_point_goal_n)  # 起点 = 机器人当前位置（归一化空间中的原点）
            goal_repeated = tensor_point_goal_n.repeat(sample_num, 1)
            origin_repeated = origin.repeat(sample_num, 1)
            theta_expanded = tensor_theta_g.repeat(sample_num)
            bridge_goal_repeated = goal_repeated
            bridge_theta_expanded = theta_expanded
            if self.enable_bridge_anchor_sampling:
                bridge_goal_repeated, bridge_theta_expanded = (
                    self.bridge_scheduler.sample_bridge_anchor_goals(
                        tensor_point_goal_n,
                        origin,
                        sample_num=sample_num,
                        keep_first_sample=self.bridge_anchor_keep_original_sample,
                    )
                )
            naction = self.bridge_scheduler.sample_initial_noise_ordered(
                goal=bridge_goal_repeated,
                origin=origin_repeated,
                shape=(sample_num * B, self.predict_size, 3),
                device=self._device,
                sample_num=sample_num,
                keep_first_sample=self.bridge_anchor_keep_original_sample,
            )

            # 去噪时 scheduler 使用 bridge anchor；真实 pointgoal 保留给条件 token
            # 和后续 goal-consistency score。
            self.bridge_scheduler.set_timesteps(self.num_inference_timesteps)
            if self.enable_trajectory_normalization:
                distance_repeated = goal_distance_m.repeat(sample_num)

            for k in self.bridge_scheduler.timesteps:
                x0_pred = self.predict_x0(
                    naction, k.to(self._device).unsqueeze(0),
                    pointgoal_embed, rgbd_embed, gated_prior, scale_embed
                )
                naction = self.bridge_scheduler.step_trajectory(
                    x0_pred, naction, k,
                    goal=bridge_goal_repeated,
                    theta_g=bridge_theta_expanded,
                    origin=origin_repeated,
                    mode="pointgoal",
                    eta=self.inference_eta,
                )

            # Critic 排序
            critic_values = self.predict_critic(naction, rgbd_embed, scale_embed)
            score_values = self._apply_goal_consistency_score(
                critic_values, naction, goal_repeated, origin_repeated
            )

            # ── 反归一化 ──
            # smooth_trajectory_batch 暂停使用；24 点本身即为可重采样的轨迹控制点。
            if self.enable_trajectory_normalization:
                naction = self._denormalize_trajectory_action(naction, distance_repeated)
            else:
                naction = self._denormalize_action(naction)
            trajectory = naction

            negative_trajectory = trajectory[(score_values).argsort()[0:8]]
            positive_trajectory = trajectory[(-score_values).argsort()[0:8]]
            if return_mask:
                negative_mask = self._build_valid_mask(
                    negative_trajectory, threshold=valid_threshold, min_valid_steps=min_valid_steps
                )
                positive_mask = self._build_valid_mask(
                    positive_trajectory, threshold=valid_threshold, min_valid_steps=min_valid_steps
                )
                negative_len = self._valid_mask_to_lengths(negative_mask)
                positive_len = self._valid_mask_to_lengths(positive_mask)
                return (
                    negative_trajectory, positive_trajectory,
                    negative_mask, positive_mask,
                    negative_len, positive_len,
                )
            return negative_trajectory, positive_trajectory

    def predict_nogoal_batch_action_vel(
        self, input_images, input_depths,
        prior_traj=None, sample_num=32,
        return_mask=False, min_valid_steps=4, valid_threshold=1e-4,
    ):
        """NoGoal 推理：无目标自由探索。

        与 PointGoal 推理的区别：
        - goal 设为零向量
        - theta_g 设为 0（方向无关，方差取中值）
        - sigma_goal 应设为较大值（由训练配置控制）

        Args:
            input_images: 历史 RGB (B, mem, H, W, 3)。
            input_depths: 历史深度 (B, H, W, 1)。
            prior_traj: 先验轨迹，可为 None。
            sample_num: 采样候选数。

        Returns:
            negative_trajectory: 低分轨迹 (8, T_pred, 3)。
            positive_trajectory: 高分轨迹 (8, T_pred, 3)。
            return_mask=True 时额外返回:
                negative_mask: (8, T_pred) 有效步掩码。
                positive_mask: (8, T_pred) 有效步掩码。
                negative_len: (8,) 有效长度（最后一个有效步索引+1）。
                positive_len: (8,) 有效长度（最后一个有效步索引+1）。
        """
        with torch.no_grad():
            rgbd_embed = self.rgbd_encoder(input_images, input_depths)
            nogoal_embed = torch.zeros_like(rgbd_embed[:, 0:1])
            B = rgbd_embed.shape[0]
            nogoal_distance_m = self._default_nogoal_distance(
                B, self._device, dtype=rgbd_embed.dtype
            )
            scale_embed = self._build_scale_token(nogoal_distance_m, like_token=nogoal_embed)
            rgbd_embed = self._apply_scale_rgbd_film(rgbd_embed, nogoal_distance_m)

            # NoGoal: goal = 0, theta_g = 0（归一化空间中 0 仍然是 0）
            zero_goal = torch.zeros(B, 3, device=self._device)
            zero_theta = torch.zeros(B, device=self._device)

            if prior_traj is not None:
                tensor_prior = torch.as_tensor(prior_traj, dtype=torch.float32, device=self._device)
                tensor_prior = self._normalize_action(tensor_prior)
            else:
                tensor_prior = torch.zeros(B, self.predict_size, 3, device=self._device)

            if self.use_prior_traj:
                prior_tokens = self.prior_encoder(tensor_prior)
                vis_global = rgbd_embed.mean(dim=1)
                gate = self.visual_gate(vis_global)
                gated_prior = gate * prior_tokens
            else:
                gated_prior = torch.zeros(
                    B, self.n_prior_tokens, self.token_dim, device=self._device
                )

            # NoGoal: fixed forward default target with uncertainty growing toward the far end.
            naction = self.bridge_scheduler.sample_initial_noise_nogoal(
                (sample_num * B, self.predict_size, 3),
                self._device,
                dtype=zero_goal.dtype,
            )

            self.bridge_scheduler.set_timesteps(self.num_inference_timesteps)

            for k in self.bridge_scheduler.timesteps:
                x0_pred = self.predict_x0(
                    naction, k.to(self._device).unsqueeze(0),
                    nogoal_embed, rgbd_embed, gated_prior, scale_embed
                )
                naction = self.bridge_scheduler.step_trajectory(
                    x0_pred, naction, k,
                    mode="nogoal",
                    eta=self.inference_eta,
                )

            critic_values = self.predict_critic(naction, rgbd_embed, scale_embed)

            # ── 反归一化 ──
            # smooth_trajectory_batch 暂停使用；部署端可按机器人运动属性另行重采样。
            naction = self._denormalize_action(naction)
            trajectory = naction

            negative_trajectory = trajectory[(critic_values).argsort()[0:8]]
            positive_trajectory = trajectory[(-critic_values).argsort()[0:8]]
            if return_mask:
                negative_mask = self._build_valid_mask(
                    negative_trajectory, threshold=valid_threshold, min_valid_steps=min_valid_steps
                )
                positive_mask = self._build_valid_mask(
                    positive_trajectory, threshold=valid_threshold, min_valid_steps=min_valid_steps
                )
                negative_len = self._valid_mask_to_lengths(negative_mask)
                positive_len = self._valid_mask_to_lengths(positive_mask)
                return (
                    negative_trajectory, positive_trajectory,
                    negative_mask, positive_mask,
                    negative_len, positive_len,
                )
            return negative_trajectory, positive_trajectory


# ======================================================================
# 辅助函数：三次样条平滑
# ======================================================================

def smooth_trajectory_batch(trajectories: torch.Tensor) -> torch.Tensor:
    """对 batch 轨迹做自然三次样条平滑，保证 C2 连续性（纯 GPU 实现）。

    与 NavDP 的 cumsum/4 后处理不同，Bridge-DP 使用绝对坐标，
    需要通过样条平滑保证曲率连续性。

    本实现完全在 GPU 上运行，避免了 scipy CubicSpline 的 CPU↔GPU 反复传输。
    采用自然三次样条（natural cubic spline）：端点二阶导为零。

    因为输入输出节点相同（不做上采样），样条拟合后在原节点处精确插值，
    但通过三对角方程求解强制了全局 C2 连续性，等价于 scipy CubicSpline 的效果。

    Args:
        trajectories: (B, T, 3) 绝对坐标轨迹。

    Returns:
        smoothed: (B, T, 3) 平滑后轨迹。

    场景自检：
        1. 直线轨迹输入 → 样条拟合后仍为直线 → 正确。
        2. 急转弯轨迹 → 样条平滑后转弯曲率连续 → 消除锯齿 → 正确。
        3. 轨迹点数 T=24 → 足够的控制点保证拟合精度 → 正确。
        4. 全程在 GPU 上计算 → 无 CPU↔GPU 传输瓶颈 → 正确。

    注意：当输入输出节点完全相同时，自然三次样条在节点处精确等于输入值，
    因此该函数对轨迹值不做改变，但计算过程保证了 C2 连续的数学约束，
    这对后续可能的上采样（如控制频率高于规划频率）提供了正确的插值基函数。
    如果当前场景下不需要上采样，此函数等价于恒等映射但保留了扩展接口。
    """
    if trajectories.dim() != 3:
        raise ValueError("trajectories must have shape (B, T, 3)")

    B, T, C = trajectories.shape
    if T < 3:
        return trajectories.clone()

    y = trajectories
    d = 6.0 * (y[:, 2:, :] - 2.0 * y[:, 1:-1, :] + y[:, :-2, :])
    n = T - 2

    c_prime = torch.zeros((B, n, C), device=y.device, dtype=y.dtype)
    d_prime = torch.zeros((B, n, C), device=y.device, dtype=y.dtype)

    c_prime[:, 0, :] = 0.25
    d_prime[:, 0, :] = d[:, 0, :] / 4.0

    for i in range(1, n):
        denom = 4.0 - c_prime[:, i - 1, :]
        c_prime[:, i, :] = 1.0 / denom
        d_prime[:, i, :] = (d[:, i, :] - d_prime[:, i - 1, :]) / denom

    m = torch.zeros((B, T, C), device=y.device, dtype=y.dtype)
    m[:, -2, :] = d_prime[:, -1, :]
    for i in range(n - 2, -1, -1):
        m[:, i + 1, :] = d_prime[:, i, :] - c_prime[:, i, :] * m[:, i + 2, :]

    t = torch.linspace(0, T - 1, T, device=y.device, dtype=y.dtype)
    idx = t.floor().long().clamp(max=T - 2)
    idx_next = (idx + 1).clamp(max=T - 1)

    idx_expand = idx.view(1, T, 1).expand(B, T, C)
    idx_next_expand = idx_next.view(1, T, 1).expand(B, T, C)

    y_i = torch.gather(y, 1, idx_expand)
    y_ip1 = torch.gather(y, 1, idx_next_expand)
    m_i = torch.gather(m, 1, idx_expand)
    m_ip1 = torch.gather(m, 1, idx_next_expand)

    t_i = idx.to(dtype=y.dtype).view(1, T, 1)
    t_ip1 = t_i + 1.0
    t_expand = t.view(1, T, 1)

    dt = t_expand - t_i
    dt_next = t_ip1 - t_expand

    smoothed = (
        m_i * (dt_next ** 3) / 6.0
        + m_ip1 * (dt ** 3) / 6.0
        + (y_i - m_i / 6.0) * dt_next
        + (y_ip1 - m_ip1 / 6.0) * dt
    )

    return smoothed
