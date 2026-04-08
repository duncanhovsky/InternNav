import os
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from transformers import PretrainedConfig, PreTrainedModel

from internnav.configs.model.base_encoders import ModelCfg
from internnav.configs.trainer.exp import ExpCfg
from internnav.model.encoder.flownav_backbone import (
    ImageGoalBackbone,
    LearnablePositionalEncoding,
    PixelGoalBackbone,
    FlowNavFusionBackbone,
    SinusoidalPosEmb
)

class FlowNavModelConfig(PretrainedConfig):
    """FlowNav 模型配置类，继承自 Hugging Face 的 PretrainedConfig。"""
    model_type = "flownav_model"

    def __init__(self, **kwargs):
        """初始化配置对象。
           Args:
                **kwargs: 其他配置参数，允许通过字典传入。
        """
        super().__init__(**kwargs)
        # 存储FlowNav实验配置
        self.model_cfg = kwargs.get('model_cfg', None)
    
    @classmethod
    def from_dict(cls, config_dict):
        """从字典创建配置对象。
           Args:
                config_dict: 包含配置参数的字典。
           Returns:
                FlowNavModelConfig 实例。
        """
        if 'model_cfg' in config_dict:
            config_dict['model_cfg'] = ExpCfg(**config_dict['model_cfg'])
        return super().from_dict(config_dict)

class FlowNavNet(PreTrainedModel):
    """FlowNav主网络。
    
    该模型将RGB-D历史观测、4D时空条件与多种目标条件融合，
    通过TransformerDecoder预测动作噪声，并使用Critic进行轨迹评分。
    """
    config_class = FlowNavModelConfig

    @classmethod
    def from_pretrained(cls,pretrained_model_name_or_path,*model_args,**kwargs):
        """加载预训练模型
        
        Args:
            pretrained_model_name_or_path: 预训练模型的路径或名称。
            *model_args: 预留参数，保持与父类接口兼容。
            **kwargs: 额外参数，可包含‘config’。
        Returns：
            已加载权重的FlowNavNet实例。
        """
        config = kwargs.pop('config',None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path,**kwargs)
        
        # 若外部传入的是pydantic配置对象，转换为本模型配置类型
        if hasattr(config, 'model_dump'):
            config = cls.config_class(model_cfg=config)
        
        model = cls(config)
        model.to(model._device)

        # 加载预训练权重，兼容目录与文件两种形式
        if os.path.isdir(pretrained_model_name_or_path):
            incompatible_keys, _ = model.load_state_dict(
                torch.load(os.path.join(pretrained_model_name_or_path, 'pytorch_model_bin'))
            )
            if len(incompatible_keys) > 0:
                print(f"Incompatible keys: {incompatible_keys}")
        elif pretrained_model_name_or_path is None or len(pretrained_model_name_or_path) == 0:
            pass
        else:
            incompatible_keys, _ = model.load_state_dict(torch.load(pretrained_model_name_or_path), strict=False)
            if len(incompatible_keys) > 0:
                print(f'Incompatible keys: {incompatible_keys}')
        
        return model
    
    def __init__(self, config: FlowNavModelConfig):
        """初始化 FlowNav 网络结构
        
        Args:
            config: 模型配置对象
        """
        super().__init__(config)

        if isinstance(config, FlowNavModelConfig):
            self.model_config = ModelCfg(**config.model_cfg['model'])
        else:
            self.model_config = config
        
        self.config.model_cfg['il']
        self._device = torch.device(f"cuda:{config.model_cfg['local_rank']}")
        self.image_size = self.config.model_cfg['il']['image_size']
        self.memory_size = self.config.model_cfg['il']['memory_size']
        self.predict_size = self.config.model_cfg['il']['predict_size']
        self.pixel_channel = self.config.model_cfg['il']['pixel_channel']
        self.temporal_depth = self.config.model_cfg['il']['temporal_depth']
        self.attention_heads = self.config.model_cfg['il']['heads']
        self.input_channels = self.config.model_cfg['il']['channels']
        self.dropout = self.config.model_cfg['il']['dropout']
        self.token_dim = self.config.model_cfg['il']['token_dim']
        self.scratch = self.config.model_cfg['il']['scratch']
        self.finetune = self.config.model_cfg['il']['finetune']
        # 目标侧注意力掩码策略:
        # - none: 不使用tgt_mask，解码器可全局关注
        # - lookahead: 局部因果+短前瞻，第i步可看[0, i+lookahead]
        self.tgt_mask_mode = str(self.config.model_cfg['il'].get('tgt_mask_mode', 'none')).lower()
        self.tgt_mask_lookahead = int(self.config.model_cfg['il'].get('tgt_mask_lookahead', 0))
        if self.tgt_mask_lookahead < 0:
            raise ValueError(f"tgt_mask_lookahead must be >= 0, got {self.tgt_mask_lookahead}")

        # 编码器
        # fusion_encoder:
        #   input: 
        #     input_images: (B, T, C, H, W) RGB历史观测序列
        #     input_depths: (B, T, 1, H, W) 深度历史观测序列（单通道）
        #     dynamic_voxels: (B, T, X, Y, Z) 未来 T 帧的 4D 运动场
        #  output:
        #     memory_token: (B, memory_size*16, token_dim) 供 TransformerDecoder 使用的融合特征序列
        self.fusion_encoder = FlowNavFusionBackbone(
            image_size=self.image_size,
            embed_size=self.token_dim,
            memory_size=self.memory_size,
            finetune=self.finetune,
            rgb_checkpoint=self.config.model_cfg['il']['rgb_checkpoint'],
            dynamics_checkpoint=self.config.model_cfg['il']['dynamics_checkpoint'],
            input_dtype=self.config.model_cfg['il']['input_dtype'],
            hidden_dim=self.config.model_cfg['il']['hidden_dim'],
            device=self._device
        )
        self.pixel_encoder = PixelGoalBackbone(
            image_size=self.image_size,
            embed_size=self.token_dim,
            pixel_channel=self.pixel_channel,
            device=self._device
        )
        self.image_encoder = ImageGoalBackbone(
            image_size=self.image_size,
            embed_size=self.token_dim,
            device=self._device
        )
        self.point_encoder = nn.Linear(3,self.token_dim)

        # 冻结主干网络时，禁止其梯度并切换到eval
        if not self.finetune:
            for p in self.fusion_encoder.rgb_model.parameters():
                p.requires_grad = False
            self.fusion_encoder.rgb_model.eval()
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.token_dim,                 # Transformer维度与编码器输出维度保持一致
            nhead=self.attention_heads,             # 注意力头数
            dim_feedforward=self.token_dim * 4,     # 前馈网络维度通常是模型维度的4倍
            dropout=self.dropout,                   # Dropout率
            activation='gelu',                      # 激活函数
            batch_first=True,                       # 使用batch_first格式
            norm_first=True                         # 先归一化后计算，提升训练稳定性
        )
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=self.temporal_depth)
        self.input_embed = nn.Linear(3, self.token_dim) # 输入动作的线性映射

        # 条件序列位置编码长度：时间 token(1)+三个目标token(3)+memory token(memory_size*16)(8*16)
        # memory token的组成：每帧一个，包含 RGB-D 视觉特征 + 4D 时空动态特征，经过线性映射后得到 token_dim 维的特征向量
        self.cond_pos_embed = LearnablePositionalEncoding(
            self.token_dim, 
            self.memory_size*16+4   # 时间 token + 3 个目标 token = 4 + memory_size*16
        )
        self.out_pos_embed = LearnablePositionalEncoding(
            self.token_dim,
            self.predict_size   # 预测动作序列长度的可学习位置编码
        )
        self.drop = nn.Dropout(self.dropout)

        # shape:(B, token_dim)，为每个样本生成一个时间步嵌入，
        # 后续会通过 unsqueeze(1) 扩展为 (B, 1, token_dim)，与其他条件 token 拼接
        self.time_emb = SinusoidalPosEmb(self.token_dim)
        self.layernorm = nn.LayerNorm(self.token_dim)
        self.action_head = nn.Linear(self.token_dim, 3) # 输出动作的线性映射
        self.critic_head = nn.Linear(self.token_dim, 1) # Critic 的线性映射，输出单个标量分数
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=10,                 # 训练时的扩散步数，保持与 NavDP 一致
            beta_schedule='squaredcos_cap_v2',      # 使用与 NavDP 相同的 beta 调度策略
            clip_sample=True,                       # 采样时对预测的噪声进行裁剪，提升稳定性
            prediction_type='epsilon'               # 预测噪声的类型，保持与 NavDP 一致
        )

        # NavDP(原版)：自回归解码 mask: 仅允许当前位置关注历史与当前位置。
        # 输入是过去7帧RGB+当前帧RGB-D
        # self.tgt_mask = (torch.triu(torch.ones(self.predict_size, self.predict_size)) == 1).transpose(0, 1)
        # self.tgt_mask = (
        #     self.tgt_mask.float()
        #     .masked_fill(self.tgt_mask == 0, float('-inf'))
        #     .masked_fill(self.tgt_mask == 1, float(0.0))
        # )
        # self.tgt_mask = self.tgt_mask.to(self._device)
        
        # FlowNav(改进): 取消自回归限制，允许解码器在去噪过程中同时关注所有条件 token 和所有预测位置，
        # 输入是过去memory_size-1帧RGB + 当前RGB-D + 未来8帧4D时空动态体素，
        # 解码器可以同时关注所有条件token，去噪过程中不受自回归限制
        # 性能影响：解码器可以更灵活地利用条件信息，提升预测质量；同时增加计算量，需权衡使用。
        self.tgt_mask = None
        self._tgt_mask_seq_len = None
        self._refresh_tgt_mask(self.predict_size, self._device)

        # NavDP(原版): critic 条件 mask：屏蔽前 4 个目标相关 token，仅使用视觉 memory 进行估值。
        # self.cond_critic_mask = torch.zeros((self.predict_size, 4 + self.memory_size * 16))
        # self.cond_critic_mask[:, 0:4] = float('-inf')

        # FlowNav(改进): 同样屏蔽前4个目标相关token，
        # 但保留对 memory token 的访问，允许 critic 评估时利用视觉与动态信息进行更准确的价值估计。
        self.cond_critic_mask = torch.zeros((self.predict_size, 4 + self.memory_size * 16))
        self.cond_critic_mask[:, 0:4] = float('-inf')   # 屏蔽时间 token和三个目标 token，保留对 memory token 的访问权限
        # 注意这里cond_critic_mask尚未to(device)

        self.pixel_aux_head = nn.Linear(self.token_dim, 3)
        self.image_aux_head = nn.Linear(self.token_dim, 3)
    
    def to(self, device, *args, **kwargs):
        """重载'to'，确保自定义张量与模型参数同设备"""
        self = super().to(device, *args, **kwargs)

        # 将未注册为buffer的掩码手动迁移到目标设备
        self.cond_critic_mask = self.cond_critic_mask.to(device)
        if self.tgt_mask is not None:
            self.tgt_mask = self.tgt_mask.to(device)

        # 更新内部设备记录
        self._device = device

        return self

    def _build_tgt_mask(self, seq_len, device):
        """构建decoder目标序列掩码。

        模式:
            - none: 返回None，表示不限制目标侧自注意力。
            - lookahead: 局部因果掩码，第i步仅允许关注到i+lookahead。
        """
        if self.tgt_mask_mode == 'none':
            return None
        if self.tgt_mask_mode != 'lookahead':
            raise ValueError(
                f"Unsupported tgt_mask_mode: {self.tgt_mask_mode}. Expected 'none' or 'lookahead'."
            )

        step = torch.arange(seq_len, device=device)
        row = step.unsqueeze(1)
        col = step.unsqueeze(0)
        # bool mask语义: True表示该位置被屏蔽。
        return col > (row + self.tgt_mask_lookahead)

    def _refresh_tgt_mask(self, seq_len, device):
        """按序列长度与设备刷新tgt_mask缓存。"""
        if self.tgt_mask_mode == 'none':
            self.tgt_mask = None
            self._tgt_mask_seq_len = seq_len
            return

        needs_rebuild = (
            self.tgt_mask is None
            or self._tgt_mask_seq_len != seq_len
            or self.tgt_mask.device != device
        )
        if needs_rebuild:
            self.tgt_mask = self._build_tgt_mask(seq_len, device)
            self._tgt_mask_seq_len = seq_len

    def _align_batch(self, tensor, target_batch, name):
        """将张量在 batch 维对齐到 target_batch。

        规则:
            1) batch 已一致: 直接返回。
            2) batch=1: 使用 expand 进行广播。
            3) target_batch 可被当前 batch 整除: repeat_interleave 扩展。
            4) 其他情况: 抛错，避免静默形状错误。
        """
        batch = tensor.shape[0]
        if batch == target_batch:
            return tensor
        if batch == 1:
            return tensor.expand(target_batch, *tensor.shape[1:])
        if target_batch % batch == 0:
            repeat_factor = target_batch // batch
            return tensor.repeat_interleave(repeat_factor, dim=0)
        raise ValueError(
            f"Cannot align {name} batch from {batch} to target batch {target_batch}."
        )

    def sample_noise(self, action):
        """对动作序列采样扩散噪声并生成噪声动作嵌入。
        Args:
            action: (B, predict_size, 3) 原始动作序列
        Returns:
            tuple: (noise, time_embed, noisy_action_embed)
                noise: (B, predict_size, 3) 采样的扩散噪声
                time_embed: (B, 1, token_dim) 时间步嵌入
                noisy_action_embed: (B, predict_size, token_dim) 噪声动作的线性映射嵌入
        noise与action的关系: 根据 DDPM 的定义，噪声是通过向干净动作添加随机噪声得到的，即 noisy_action = action + noise。
        因此，noise 的分布通常是一个零均值的高斯分布，其标准差由噪声调度器（noise_scheduler）根据当前时间步动态调整。

        该函数的目的: 在训练过程中，为每个动作序列采样一个随机的扩散噪声，并生成对应的时间步嵌入和噪声动作嵌入。
        这些信息将被送入 TransformerDecoder 进行去噪训练，帮助模型学习如何从不同程度的噪声中恢复出干净的动作序列。
        """
        device = action.device
        noise = torch.randn(action.shape, device=device)
        # (action.shape[0],) -> (B,) 的随机时间步索引，范围在 [0, num_train_timesteps) 之间
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (action.shape[0],), device=device
        ).long()
        time_embeds = self.time_emb(timesteps).unsqueeze(1)  # (B, token_dim) -> (B, 1, token_dim)
        noisy_action = self.noise_scheduler.add_noise(action, noise, timesteps)
        noisy_action_embed = self.input_embed(noisy_action) # (B, predict_size, 3) -> (B, predict_size, token_dim)
        return noise, time_embeds, noisy_action_embed

    def predict_noise(self, last_actions, timestep, goal_embed, fusion_embed):
        """在给定条件下预测动作噪声。
        Args:
            last_actions: 当前扩散步的动作样本，形状为 (B, predict_size, 3)。
            timestep: 扩散时间步，通常为标量或形状为 (1,) 的张量，形状为 (B,) 时会在函数内部处理成 (B, 1, token_dim) 的时间嵌入。
            goal_embed: 目标嵌入，形状为 (B, 1, token_dim)。
            (NavDP原版)rgbd_embed: 视觉记忆嵌入，形状为 (B, memory_size*16, token_dim)。
            (FlowNav改进)fusion_embed: 融合了视觉与动态信息的时空条件嵌入，形状为 (B, memory_size*16, token_dim)。
        Returns:
            torch.Tensor: 预测噪声，形状为(B, predict_size, 3)。
        """
        action_embeds = self.input_embed(last_actions)  # (B, predict_size, 3) -> (B, predict_size, token_dim)

        timestep = timestep.to(self._device).long()
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        time_embeds = self.time_emb(timestep).unsqueeze(1)  # (B_t,) -> (B_t, 1, token_dim)

        # 条件序列结构: [time, goal, goal, goal, fusion_memory] shape: (B, 4(time+goal*3) + memory_size*16, token_dim)
        # 重复三遍goal_embed以匹配原来三个目标 token 的位置，保持与 NavDP 的条件结构一致
        cond_batch = max(time_embeds.shape[0], goal_embed.shape[0], fusion_embed.shape[0])
        time_embeds = self._align_batch(time_embeds, cond_batch, 'time_embeds')
        goal_embed = self._align_batch(goal_embed, cond_batch, 'goal_embed')
        fusion_embed = self._align_batch(fusion_embed, cond_batch, 'fusion_embed')
        cond_embedding = torch.cat(
            [time_embeds,goal_embed,goal_embed,goal_embed,fusion_embed],dim=1
        ) + self.cond_pos_embed(torch.cat([time_embeds,goal_embed,goal_embed,goal_embed,fusion_embed],dim=1))

        # 推理时可能进行多轨迹采样，按目标动作 batch 进行安全对齐。
        cond_embedding = self._align_batch(cond_embedding, action_embeds.shape[0], 'cond_embedding')
        # 经过 TransformerDecoder 进行去噪预测，输出形状为 (B, predict_size, token_dim)
        input_embedding = action_embeds + self.out_pos_embed(action_embeds) # 添加位置编码
        self._refresh_tgt_mask(input_embedding.shape[1], input_embedding.device)
        output = self.decoder(
            tgt=input_embedding, memory=cond_embedding, tgt_mask=self.tgt_mask
        )
        output = self.layernorm(output)
        output = self.action_head(output) # (B, predict_size, token_dim) -> (B, predict_size, 3)
        return output

    def predict_critic(self, predict_trajectory,fusion_embed):
        """对候选轨迹进行critic评分。
        Args:
            predict_trajectory: 待评估轨迹，形状为(B, predict_size, 3)。
            fusion_embed: 对应观测的 RGB-D 条件，形状为(1, memory_size*16, token_dim) 
                或可广播到N条轨迹的形状 (N, memory_size*16, token_dim)，其中 N 是候选轨迹的数量。
        Returns:
            torch.Tensor: 每条轨迹的标量分数，形状为(N,)
        """
        repeat_fusion_embed = self._align_batch(
            fusion_embed, predict_trajectory.shape[0], 'fusion_embed'
        )  # (N, memory_size*16, token_dim)
        nogoal_embed = torch.zeros_like(repeat_fusion_embed[:, 0:1])
        action_embeddings = self.input_embed(predict_trajectory)  # (N, predict_size, 3) -> (N, predict_size, token_dim)
        action_embeddings = action_embeddings + self.out_pos_embed(action_embeddings)

        # critic 阶段采用无目标条件（四个位置均为零向量）
        # 网络在 critic 阶段能感知到的输入信息: 
        #   时间 token 全为零向量，三个目标 token 全为零向量，融合条件 token 包含 RGB-D 视觉信息和 4D 时空动态信息。
        cond_embeddings = torch.cat(
            [nogoal_embed, nogoal_embed, nogoal_embed, nogoal_embed, repeat_fusion_embed], dim=1
        ) + self.cond_pos_embed(
            torch.cat([nogoal_embed, nogoal_embed, nogoal_embed, nogoal_embed, repeat_fusion_embed], dim=1)
        )
        critic_output = self.decoder(tgt=action_embeddings, memory=cond_embeddings, memory_mask=self.cond_critic_mask)
        critic_output = self.layernorm(critic_output)
        critic_output = self.critic_head(critic_output.mean(dim=1))[:,0]
        return critic_output    # 输出形状为 (N,)，每条轨迹对应一个标量分数

    def forward(self,goal_point,goal_image,goal_pixel,input_images,input_depths,dynamic_voxels,output_actions,augment_actions):
        """训练阶段前向过程。
        Args:
            goal_point: 目标点条件，形状为 (B, 3)。
            goal_image: 目标图像条件，形状为 (B, C, H, W)。
            goal_pixel: 目标像素条件，形状为 (B, pixel_channel, H, W)。
            input_images: 历史 RGB 图像序列，形状为 (B, T, C, H, W)。
            input_depths: 历史深度图像序列，形状为 (B, T, 1, H, W)。
            dynamic_voxels: 未来 T 帧的 4D 时空动态体素，形状为 (B, T, C=4, X, Y, Z)。
            output_actions: 干净的动作序列标签，形状为 (B, predict_size, 3)。
            augment_actions: 数据增强后的动作序列，形状为 (B, predict_size, 3)。
            PS: augment_actions = output_actions + 随机扰动，用于增强模型的鲁棒性。
        
        Returns:
            tuple: 依次返回
                noise_pred_ng: 预测的噪声，形状为 (B, predict_size, 3)。
                noise_pred_mg: 预测的噪声，形状为 (B, predict_size, 3)。
                cr_label_pred: 轨迹的 critic 评分，形状为 (B,)。
                cr_augment_pred: 轨迹的 critic 评分，形状为 (B,)。
                ng_noise: 采样的扩散噪声，形状为 (B, predict_size, 3)。
                mg_noise: 采样的扩散噪声，形状为 (B, predict_size, 3)。
                imagegoal_aux_pred: 目标图像的辅助预测，形状为 (B, 3)。
                pixelgoal_aux_pred: 目标像素的辅助预测，形状为 (B, 3)。
        PS:
            ng: no-goal 路径，输入条件不包含目标相关 token，仅包含时间 token 和融合条件 token。
            mg: mixed-goal 路径，输入条件包含时间 token、目标相关 token 和融合条件 token。
        """
        device = next(self.parameters()).device

        assert input_images.shape[1] == self.memory_size, f"输入的历史图像序列长度 {input_images.shape[1]} 必须与配置中的 memory_size {self.memory_size} 一致。"
        tensor_point_goal = torch.as_tensor(goal_point, dtype=torch.float32).to(device)
        tensor_label_actions = torch.as_tensor(output_actions, dtype=torch.float32).to(device)
        tensor_augment_actions = torch.as_tensor(augment_actions, dtype=torch.float32).to(device)
        input_images = input_images.to(device)
        input_depths = input_depths.to(device)
        dynamic_voxels = dynamic_voxels.to(device)

        # 生成 no-goal 与 mixed-goal 两路噪声样本
        ng_noise, ng_time_embed, ng_noisy_action_embed = self.sample_noise(tensor_label_actions)
        mg_noise, mg_time_embed, mg_noisy_action_embed = self.sample_noise(tensor_label_actions)

        #编码多模态感知条件与与目标条件
        fusion_embed = self.fusion_encoder(input_images, input_depths, dynamic_voxels) # (B, memory_size*16, token_dim)
        pointgoal_embed = self.point_encoder(tensor_point_goal).unsqueeze(1) # (B, 1, token_dim)
        nogoal_embed = torch.zeros_like(pointgoal_embed) # (B, 1, token_dim),全零向量表示 no-goal 条件
        imagegoal_embed = self.image_encoder(goal_image).unsqueeze(1) # (B, 1, token_dim)
        pixelgoal_embed = self.pixel_encoder(goal_pixel).unsqueeze(1) # (B, 1, token_dim)
        
        imagegoal_aux_pred = self.image_aux_head(imagegoal_embed[:, 0])
        pixelgoal_aux_pred = self.pixel_aux_head(pixelgoal_embed[:, 0])

        # 监督动作嵌入仅用于 decoder/critic 输入，不参与 input_embed 反向更新
        label_embed = self.input_embed(tensor_label_actions).detach() # (B, predict_size, token_dim)
        augment_embed = self.input_embed(tensor_augment_actions).detach() # (B, predict_size, token_dim)

        cond_pos_embed = self.cond_pos_embed(
            torch.cat([ng_time_embed, nogoal_embed, imagegoal_embed, pixelgoal_embed, fusion_embed], dim=1)
        )
        # no-goal 条件下的条件嵌入，时间 token 和融合条件 token 保持不变，目标相关 token 全为零向量
        # drop做了什么事？drop是为了在训练过程中随机丢弃部分条件信息，增强模型的鲁棒性和泛化能力。
        # 通过在条件嵌入上应用 dropout，可以模拟不同程度的条件缺失情况，迫使模型学会在不完整条件下仍然能够进行有效的去噪预测。
        # 这有助于提升模型在实际应用中的适应性，尤其是在某些条件信息可能不可靠或缺失的情况下。
        ng_cond_embeddings = self.drop(
            torch.cat([ng_time_embed, nogoal_embed, nogoal_embed, nogoal_embed, fusion_embed], dim=1) + cond_pos_embed
        )

        cand_goal_embed = [pointgoal_embed, imagegoal_embed, pixelgoal_embed]
        batch_size = pointgoal_embed.shape[0]

        # 使用确定性组合采样 mixed-goal：每个样本选择三个目标位的组合（共 3^3=27 种）
        batch_indices = torch.arange(batch_size, device=pointgoal_embed.device)
        pattern_indices = batch_indices % 27  # 3^3 = 27 种组合模式
        selections_0 = pattern_indices % 3
        selections_1 = (pattern_indices // 3) % 3
        selections_2 = (pattern_indices // 9) % 3
        # 根据选择的组合模式构建混合目标嵌入，0 表示 no-goal，1 表示 point-goal，2 表示 image-goal，3 表示 pixel-goal
        goal_embeds = torch.stack(cand_goal_embed, dim=0)   # [3, batch_size, 1, token_dim]
        selected_goals_0 = goal_embeds[selections_0, torch.arange(batch_size), :, :]    # [batch_size, 1, token_dim]
        selected_goals_1 = goal_embeds[selections_1, torch.arange(batch_size), :, :]    # [batch_size, 1, token_dim]
        selected_goals_2 = goal_embeds[selections_2, torch.arange(batch_size), :, :]    # [batch_size, 1, token_dim]
        mg_cond_embed_tensor = torch.cat(
            [mg_time_embed, selected_goals_0, selected_goals_1, selected_goals_2, fusion_embed], dim=1
        )
        # drop: 在 mixed-goal 条件下的条件嵌入上应用 dropout，增强模型的鲁棒性和泛化能力。
        mg_cond_embeddings = self.drop(mg_cond_embed_tensor + cond_pos_embed)

        # 构造各路解码输入
        out_pos_embed = self.out_pos_embed(ng_noisy_action_embed)
        ng_action_embeddings = self.drop(ng_noisy_action_embed + out_pos_embed)
        mg_action_embeddings = self.drop(mg_noisy_action_embed + out_pos_embed)
        label_action_embeddings = self.drop(label_embed + out_pos_embed)
        augment_action_embeddings = self.drop(augment_embed + out_pos_embed)
        self._refresh_tgt_mask(ng_action_embeddings.shape[1], ng_action_embeddings.device)

        # no-goal 噪声预测分支
        ng_output = self.decoder(
            tgt=ng_action_embeddings, memory=ng_cond_embeddings, tgt_mask=self.tgt_mask
        )
        ng_output = self.layernorm(ng_output)
        noise_pred_ng = self.action_head(ng_output)

        # mixel-goal 噪声预测分支
        mg_output = self.decoder(
            tgt=mg_action_embeddings, memory=mg_cond_embeddings, tgt_mask=self.tgt_mask
        )
        mg_output = self.layernorm(mg_output)
        noise_pred_mg = self.action_head(mg_output)

        # critic 对监督轨迹打分
        cr_label_output = self.decoder(
            tgt=label_action_embeddings, memory=ng_cond_embeddings, memory_mask=self.cond_critic_mask.to(self._device)
        )
        cr_label_output = self.layernorm(cr_label_output)
        cr_label_pred = self.critic_head(cr_label_output.mean(dim=1))[:, 0]

        # critic 对增广轨迹打分
        cr_augment_output = self.decoder(
            tgt=augment_action_embeddings, memory=ng_cond_embeddings, memory_mask=self.cond_critic_mask.to(self._device)
        )
        cr_augment_output = self.layernorm(cr_augment_output)
        cr_augment_pred = self.critic_head(cr_augment_output.mean(dim=1))[:, 0]

        return (
            noise_pred_ng,
            noise_pred_mg,
            cr_label_pred,
            cr_augment_pred,
            ng_noise,
            mg_noise,
            imagegoal_aux_pred,
            pixelgoal_aux_pred
        )

    def _get_device(self):
        """安全获取模型所在设备。

        Returns:
            torch.device: 当前模型推断到的设备。
        """
        # 1) 优先从模型参数获取设备。
        try:
            for param in self.parameters():
                return param.device
        except StopIteration:
            pass

        # 2) 若无参数，则尝试从 buffer 获取。
        try:
            for buffer in self.buffers():
                return buffer.device
        except StopIteration:
            pass

        # 3) 再尝试从子模块参数获取。
        for module in self.children():
            try:
                for param in module.parameters():
                    return param.device
            except StopIteration:
                continue

        # 4) 最后回退到系统默认设备。
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def predict_pointgoal_batch_action_vel(self, 
                                           goal_point, 
                                           input_images, 
                                           input_depths, 
                                           dynamic_voxels, 
                                           sample_num=32):
        """在点目标条件下批量采样动作速度轨迹
        Args:
            goal_point: 目标点条件，形状为 (B, 3)。
            input_images: 历史 RGB 图像序列，形状为 (B, T, C, H, W)。
            input_depths: 历史深度图像序列，形状为 (B, T, 1, H, W)。
            dynamic_voxels: 未来 T 帧的 4D 时空动态体素，形状为 (B, T, X, Y, Z)。
            sample_num: 每个样本采样的轨迹数量。
        Returns:
            tuple:
                negative_trajectory: 低分轨迹（8条）
                positive_trajectory: 高分轨迹（8条）
        """
        with torch.no_grad():
            tensor_point_goal = torch.as_tensor(goal_point, dtype=torch.float32, device=self._device)
            fusion_embed = self.fusion_encoder(input_images, input_depths, dynamic_voxels)
            pointgoal_embed = self.point_encoder(tensor_point_goal).unsqueeze(1)

            # 初始化高斯噪声轨迹，并执行DDPM反向去噪
            noisy_action = torch.randn(
                (sample_num * pointgoal_embed.shape[0], self.predict_size, 3), device=self._device
            )
            naction = noisy_action
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.predict_noise(naction, k.to(self._device).unsqueeze(0), pointgoal_embed, fusion_embed)
                naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample

            # 通过 critic 选择最优与最差轨迹，用于可视化或诊断。
            critic_values = self.predict_critic(naction, fusion_embed)

            negative_trajectory = torch.cumsum(naction / 4.0, dim=1)[(critic_values).argsort()[0:8]]
            positive_trajectory = torch.cumsum(naction / 4.0, dim=1)[(-critic_values).argsort()[0:8]]
            return negative_trajectory, positive_trajectory

    def predict_nogoal_batch_action_vel(self, input_images, input_depths, dynamic_voxels, sample_num=32):
        """在无目标条件下批量采样动作速度轨迹。

        Args:
            input_images: 历史 RGB 观测。
            input_depths: 历史深度观测。
            dynamic_voxels: 未来 T 帧的 4D 时空动态体素。
            sample_num: 每个样本采样轨迹数。

        Returns:
            tuple:
                negative_trajectory: 低分轨迹（8 条）。
                positive_trajectory: 高分轨迹（8 条）。
        """
        with torch.no_grad():
            fusion_embed = self.fusion_encoder(input_images, input_depths, dynamic_voxels)
            nogoal_embed = torch.zeros_like(fusion_embed[:, 0:1])

            # 以纯噪声初始化轨迹，在无目标条件下逐步去噪。
            noisy_action = torch.randn((sample_num * nogoal_embed.shape[0], self.predict_size, 3), device=self._device)
            naction = noisy_action
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.predict_noise(naction, k.unsqueeze(0), nogoal_embed, fusion_embed)
                naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample

            # 依据 critic 分数返回轨迹两端样本。
            critic_values = self.predict_critic(naction, fusion_embed)

            negative_trajectory = torch.cumsum(naction / 4.0, dim=1)[(critic_values).argsort()[0:8]]
            positive_trajectory = torch.cumsum(naction / 4.0, dim=1)[(-critic_values).argsort()[0:8]]
            return negative_trajectory, positive_trajectory
