"""先验轨迹编码器与视觉门控模块。

本模块实现 Bridge-DP 独有的两个组件：

1. **PriorEncoder**: 将上一帧预测的先验轨迹 μ̃ ∈ R^{T×3} 编码为 N_p 个
   token，供 Transformer Decoder 的 memory 序列使用。
2. **VisualGate**: 基于当前视觉观测生成门控值 G ∈ (0, 1)，控制先验 token
   的注入强度。场景变化大时 G→0（忽略过时先验），场景稳定时 G→1。

参考文献：
    - docs/Bridge-DP推导.md §4.4（Prior Encoder 的输入）
    - docs/Bridge-DP推导.md §5（视觉门控先验注入）
"""

import math

import torch
import torch.nn as nn


class PriorEncoder(nn.Module):
    """轻量 Transformer 编码器，将先验轨迹编码为固定数量的 token。

    架构：线性投影 → 可学习位置编码 → 2 层 Transformer Encoder → 可学习查询压缩

    将变长的 T 个轨迹点压缩为 N_p 个 token（N_p ≪ T），这些 token 后续会
    经过门控后拼接到 Transformer Decoder 的 memory 序列中。

    为什么使用 Transformer 而非简单 MLP：
        先验轨迹包含**顺序形状信息**（弯曲程度、速度变化），Transformer 的自注意力
        能够捕捉轨迹点之间的时序关系，而 MLP 只能逐点独立处理。

    Attributes:
        n_prior_tokens: 输出 token 数量 N_p，默认 4。
        embed_dim: token 嵌入维度 d，需与主模型 token_dim 一致。

    导航场景自检：
        1. **正确先验（场景稳定）**: 上一帧轨迹形状与当前最优路径接近，
           PriorEncoder 将其编码为有信息的 token → VisualGate 赋予高权重。
        2. **错误先验（对抗训练 30%）**: 随机轨迹无规律，编码后 token 与
           有意义的路径特征不同 → VisualGate 学会给低权重。
        3. **无先验（首帧）**: 输入全零 → 编码后接近零均值 token →
           VisualGate 给低权重，网络完全依赖当前观测。
    """

    def __init__(
        self,
        embed_dim: int = 384,
        n_prior_tokens: int = 4,
        action_dim: int = 3,
        num_layers: int = 2,
        nhead: int = 4,
        max_traj_len: int = 64,
        dropout: float = 0.1,
    ) -> None:
        """初始化 PriorEncoder。

        Args:
            embed_dim: token 嵌入维度，需与主模型 token_dim 一致。
            n_prior_tokens: 输出 token 数量 N_p。
            action_dim: 轨迹点维度（x, y, θ），默认 3。
            num_layers: Transformer Encoder 层数。
            nhead: 多头注意力头数。
            max_traj_len: 支持的最大轨迹长度（用于位置编码）。
            dropout: Dropout 概率。
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.n_prior_tokens = n_prior_tokens

        # 轨迹点线性投影：(x, y, θ) → d 维
        self.input_proj = nn.Linear(action_dim, embed_dim)

        # 可学习位置编码
        self.pos_embed = nn.Embedding(max_traj_len, embed_dim)

        # 2 层 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # 可学习查询 token，用于将 T 个输入压缩为 N_p 个输出
        self.query_tokens = nn.Parameter(
            torch.randn(1, n_prior_tokens, embed_dim) * 0.02
        )

        # 交叉注意力：query_tokens attend to encoded trajectory
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(embed_dim)
        self.cross_ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(self, prior_traj: torch.Tensor) -> torch.Tensor:
        """编码先验轨迹为 N_p 个 token。

        Args:
            prior_traj: 先验轨迹，形状 (B, T, 3)，绝对坐标 (x, y, θ)。
                        首帧无先验时可传入全零张量。

        Returns:
            先验 token，形状 (B, N_p, embed_dim)。
        """
        B, T, _ = prior_traj.shape

        # 线性投影 + 位置编码
        positions = torch.arange(T, device=prior_traj.device)
        h = self.input_proj(prior_traj) + self.pos_embed(positions).unsqueeze(0)

        # Transformer 自注意力编码
        h = self.transformer(h)  # (B, T, d)

        # 可学习查询压缩：T → N_p
        queries = self.query_tokens.expand(B, -1, -1)  # (B, N_p, d)

        # 交叉注意力
        attn_out, _ = self.cross_attn(
            query=queries,
            key=h,
            value=h,
        )
        queries = self.cross_norm(queries + attn_out)

        # FFN
        ffn_out = self.cross_ffn(queries)
        output = self.ffn_norm(queries + ffn_out)

        return output  # (B, N_p, d)


class VisualGate(nn.Module):
    """视觉门控模块，基于当前视觉特征生成先验可信度门控值。

    门控值 G = σ(MLP(h_vis)) ∈ (0, 1)，用于加权先验 token：
        gated_prior = G · prior_tokens

    当场景变化大时（障碍物移动、环境突变），G → 0，网络忽略过时先验；
    当场景稳定时，G → 1，网络充分利用先验信息加速收敛。

    初始化偏置为正值，使训练初期 G ≈ 0.7，倾向信任先验（因为大多数情况下
    先验是有意义的），后续通过对抗训练学会在必要时降低门控值。

    Attributes:
        gate_dim: 输入视觉特征维度。

    导航场景自检：
        1. **静态环境连续导航**: 连续帧视觉观测变化小 → h_vis 稳定 → G 高 →
           先验轨迹被充分利用，去噪收敛更快。
        2. **动态环境（障碍物突然出现）**: 视觉观测突变 → G 降低 →
           网络回退到仅依赖当前观测重新规划路径。
        3. **对抗先验训练（30%错误先验）**: 错误先验 + 正常视觉观测 →
           网络学会在先验与观测不一致时降低 G。
    """

    def __init__(
        self,
        gate_dim: int = 384,
        hidden_dim: int = 128,
    ) -> None:
        """初始化视觉门控模块。

        Args:
            gate_dim: 输入视觉特征维度（= token_dim）。
            hidden_dim: MLP 隐藏层维度。
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(gate_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # 初始化偏置为正值，使训练初期 G ≈ σ(1.0) ≈ 0.73
        # 这样模型默认信任先验，然后在对抗训练中学会何时不信任
        nn.init.constant_(self.mlp[-1].bias, 1.0)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """计算门控值。

        Args:
            visual_features: 全局视觉特征，形状 (B, gate_dim)。
                            通常由 RGBDBackbone 输出的 memory tokens 做 mean pooling 得到。

        Returns:
            门控值 G，形状 (B, 1, 1)，便于广播到 (B, N_p, d)。
        """
        # MLP → sigmoid
        gate = torch.sigmoid(self.mlp(visual_features))  # (B, 1)
        return gate.unsqueeze(-1)  # (B, 1, 1)，便于广播
