import torch
import torch.nn as nn
import math
from internnav.model.encoder.depth_anything.depth_anything_v2.dpt import DepthAnythingV2
import spconv.pytorch as spconv


class SinusoidalPosEmb(nn.Module):
    """正弦位置编码。

    该模块将一维标量输入映射到固定维度的正余弦向量，常用于时间步
    或序列位置的连续表示。
    """
    def __init__(self, dim):
        """初始化正弦位置编码模块。

        Args:
            dim: 输出位置编码维度。
        """
        super().__init__()
        self.dim = dim
    def forward(self, x):
        """计算输入对应的正弦位置编码。
        Args:
            x: 形状为 (B,) 的一维张量。
        Returns:
            形状为 (B, dim) 的位置编码张量。
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class PositionalEncoding(nn.Module):
    """固定（不可学习）的正余弦位置编码模块"""
    def __init__(self, embed_dim, max_len=1000):
        """预计算并缓存位置编码
        Args:
            embed_dim: 位置编码维度。
            max_len: 最大序列长度。
        """
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0))/embed_dim))
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        self.register_buffer('pe',pe)
    def forward(self, x):
        """根据输入序列长度返回对应位置编码
        Args:
            x: 输入张量，约定第二维是序列长度
        Returns:
            形状为 (seq_len, embed_dim) 的位置编码张量。
        """
        return self.pe[: x.size(1)]

class LearnablePositionalEncoding(nn.Module):
    """基于 nn.Embedding 的可学习位置编码模块。"""
    def __init__(self, embed_dim, max_len=5000):
        """初始化可学习位置编码。
        Args:
            embed_dim: 嵌入维度。
            max_len: 可学习位置表的最大长度。
        """
        super(LearnablePositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        """生成与输入序列长度匹配的位置编码。
        Args:
            x: 输入 token 向量，形状为 (batch_size, seq_len, embed_dim)。
        Returns:
            形状为 (batch_size, seq_len, embed_dim) 的位置编码。
        """
        batch_size, seq_len, _ = x.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)  # (seq_len,)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)  # (batch_size, seq_len)
        position_encoding = self.position_embedding(position_ids)  # (batch_size, seq_len, embed_dim)
        return position_encoding

class TokenCompressor(nn.Module):
    """使用跨注意力将变长token序列压缩到固定长度。"""
    def __init__(self, embed_dim, num_heads, target_length):
        """初始化 Token 压缩模块。
        Args:
            embed_dim: 输入特征维度。
            num_heads: 注意力头数。
            target_length: 目标序列长度。
        """
        super(TokenCompressor, self).__init__()
        self.target_length = target_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 使用可学习查询序列，作为cross-attention的Query
        self.target_embedding = nn.Embedding(target_length, embed_dim)

        # 保留固定位置编码定义，兼容后续可能扩展
        self.positional_encoding = PositionalEncoding(embed_dim)

        self.token_positional_encoding = LearnablePositionalEncoding(embed_dim)
        self.query_positional_encoding = LearnablePositionalEncoding(embed_dim)

        # 跨注意力: query来自目标序列，key/value来自输入token
        self.cross_attention = nn.MultiheadAttention(embed_dim,num_heads,batch_first=True)

    def forward(self, x, padding_mask=None):
        """压缩输入token序列。
        Args:
            x: 输入序列，形状为(bs,N,embed_dim)，N可变
            padding_mask：可选的padding掩码，形状为(bs,N), True表示对应位置为padding
        Returns:
            形状为 (bs, target_length, embed_dim) 的压缩token。
        """
        bs, token_len, _ = x.shape

        # 为输入token添加可学习位置编码
        token_pe = self.token_positional_encoding(x)
        x = x + token_pe

        query = self.target_embedding.weight.unsqueeze(0).expand(bs,-1,-1)

        # 为查询序列补充位置编码，提升目标token的可区分性
        query_pe = self.query_positional_encoding(query)

        query = query + query_pe

        # 跨注意力：目标序列是query，输入序列同时作为key/value
        out, _ = self.cross_attention(query=query,key=x,value=x,key_padding_mask=padding_mask)
        return out
    
class RGBDBackbone(nn.Module):
    def __init__(self,
                image_size=224,
                embed_size=512,
                memory_size=8,
                device='cuda:0'):
        super().__init__()
        self.device = device
        self.memory_size = memory_size
        self.image_size = image_size
        self.embed_size = embed_size
        model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
        self.rgb_model = DepthAnythingV2(**model_configs['vits'])
        self.rgb_model = self.rgb_model.pretrained.float()
        self.rgb_model.eval()
        self.preprocess_mean = torch.tensor([0.485,0.456,0.406],dtype=torch.float32)
        self.preprocess_std = torch.tensor([0.229,0.224,0.225],dtype=torch.float32)
            
        self.depth_model = DepthAnythingV2(**model_configs['vits'])
        self.depth_model = self.depth_model.pretrained.float()
        self.depth_model.train()
        self.former_query = LearnablePositionalEncoding(384,self.memory_size*16)
        self.former_pe = LearnablePositionalEncoding(384,(self.memory_size+1)*256) 
        self.former_net = nn.TransformerDecoder(nn.TransformerDecoderLayer(384,8,batch_first=True),2)
        self.project_layer = nn.Linear(384,embed_size)
        
    def forward(self,images,depths):
        with torch.no_grad():
            if len(images.shape) == 4:  # (B, H, W, C) -> (B, C, H, W)
                tensor_images = torch.as_tensor(images,dtype=torch.float32,device=self.device).permute(0,3,1,2)
                tensor_images = tensor_images.reshape(-1,3,self.image_size,self.image_size)
                tensor_norm_images = (tensor_images - self.preprocess_mean.reshape(1,3,1,1).to(self.device))/self.preprocess_std.reshape(1,3,1,1).to(self.device)
                image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0]
            elif len(images.shape) == 5:
                tensor_images = torch.as_tensor(images,dtype=torch.float32,device=self.device).permute(0,1,4,2,3)
                B,T,C,H,W = tensor_images.shape
                tensor_images = tensor_images.reshape(-1,3,self.image_size,self.image_size)
                tensor_norm_images = (tensor_images - self.preprocess_mean.reshape(1,3,1,1).to(self.device))/self.preprocess_std.reshape(1,3,1,1).to(self.device)
                image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0].reshape(B,T*256,-1)
            if len(depths.shape) == 4:
                tensor_depths = torch.as_tensor(depths,dtype=torch.float32,device=self.device).permute(0,3,1,2)
                tensor_depths = tensor_depths.reshape(-1,1,self.image_size,self.image_size)
                tensor_depths = torch.concat([tensor_depths,tensor_depths,tensor_depths],dim=1)
                depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0]
            elif len(depths.shape) == 5:
                tensor_depths = torch.as_tensor(depths,dtype=torch.float32,device=self.device).permute(0,1,4,2,3)
                B,T,C,H,W = tensor_depths.shape
                tensor_depths = tensor_depths.reshape(-1,1,self.image_size,self.image_size)
                tensor_depths = torch.concat([tensor_depths,tensor_depths,tensor_depths],dim=1)
                depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0].reshape(B,T*256,-1)
            former_token = torch.concat((image_token,depth_token),dim=1) + self.former_pe(torch.concat((image_token,depth_token),dim=1))
            former_query = self.former_query(torch.zeros((image_token.shape[0], self.memory_size * 16, 384),device=self.device))
            memory_token = self.former_net(former_query,former_token)
            memory_token = self.project_layer(memory_token)
            return memory_token

class FlowNavFusionBackbone(nn.Module):
    """FlowNav 核心模态融合骨干网络。
    
    采用双流架构：
    1. 静态语义流：调用原版 RGBDBackbone 提取当前帧 (RGB + Depth) 的高维语义 Token。
    2. 动态物理流：利用 spconv 稀疏卷积极速编码未来 8 帧的 4D 运动场。
    3. 跨模态融合：通过 Cross-Attention 将未来的物理碰撞威胁注入当前的语义空间。
    输入：
        - input_images: (B, memory_size, H, W, C=3)
        - input_depths: (B, memory_size, H, W, C=1)
        - dynamic_voxels: (B, 8, C=4, X, Y, Z) 未来 8 帧的 4D 场，注意这里通道为4：[Occ, Vx, Vy, Vz]

    输出：严格保持 (B, memory_size*16, token_dim) 的 Token 序列，实现对后端的无缝欺骗。
    """

    def __init__(self, 
                 image_size=224,
                 embed_size=512,
                 finetune=True,
                 memory_size=8,
                 rgb_checkpoint="checkpoints/depth_anything_v2_vits.pth",
                 dynamics_checkpoint="checkpoints/flow_nav_dynamic.pth",
                 input_dtype="fp32",
                 hidden_dim=384,
                 device='cuda:0'):
        """初始化 FlowNavFusionBackbone。
        Args:
            config: 
                image_size: 输入图像尺寸，默认224。
                embed_size: 输出Token维度，默认512。
                finetune: 是否微调RGBD模型，默认True。
                memory_size: 记忆长度，默认8。
                rgb_checkpoint: RGBD模型权重路径，默认 "checkpoints/depth_anything_v2_vits.pth"。
                dynamics_checkpoint: 动态物理流权重路径，默认 "checkpoints/flow_nav_dynamic.pth"。
                input_dtype: 输入计算精度, 支持"bf16"或"fp32"，默认"fp32"。
            device: 设备字符串，如 'cuda:0'。
        """
        super().__init__()
        # 统一处理设备参数，避免外部传入格式不一致
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.finetune = finetune
        self.memory_size = memory_size
        self.image_size = image_size
        self.embed_size = embed_size
        model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
        self.rgb_checkpoint = rgb_checkpoint
        self.dynamics_checkpoint = dynamics_checkpoint  # 实时训练，不预先加载
        self.input_dtype = torch.bfloat16 if input_dtype == "bf16" else torch.float32
        self.version = self.version
        # self.hidden_dim = 384
        self.hidden_dim = hidden_dim
        
        # ---------------------------------------------------------
        # 1. 视锥语义流 (RGB & Depth)
        # ---------------------------------------------------------
        self.rgb_model = DepthAnythingV2(**model_configs['vits'])
        self.rgb_model.load_state_dict(torch.load(self.rgb_checkpoint), strict=False)
        self.rgb_model = self.rgb_model.pretrained.float()

        self.preprocess_mean = torch.tensor([0.485, 0.456, 0.406], dtype=self.input_dtype)
        self.preprocess_std = torch.tensor([0.229, 0.224, 0.225], dtype=self.input_dtype)

        self.rgb_model.eval() if not self.finetune else self.rgb_model.train()

        self.depth_model = DepthAnythingV2(**model_configs['vits'])
        self.depth_model = self.depth_model.pretrained.float()
        self.depth_model.train()

        # ---------------------------------------------------------
        # 2. 动态物理流 (带有显式速度特征的4D时空流,真·稀疏 4D 卷积，处理 LiteFlow+KF 预测的 8 帧)
        # ---------------------------------------------------------
        # 2.1. 稀疏3D卷积：处理单帧的[4(Occ,Vx,Vy,Vz), X, Y, Z]输入
        self.dynamic_encoder = spconv.SparseSequential(
            # 接收包含了占据概率Occ和绝对速度特征[Vx, Vy, Vz]的4通道输入
            spconv.SparseConv3d(
                in_channels=4, out_channels=64, 
                kernel_size=3, stride=2, padding=1, indice_key="spconv1"
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            spconv.SparseConv3d(
                in_channels=64, out_channels=128, 
                kernel_size=3, stride=2, padding=1, indice_key="spconv2"
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # 几何扩容：每帧池化产出 4x4x4 = 64 个 Token
        # 将稀疏特征转回致密后，用自适应池化统一空间尺寸为 4x4x4 = 64 个 Key/Value Token
        self.dynamic_pool = nn.AdaptiveAvgPool3d((4,4,4))
        self.dynamic_proj = nn.Linear(128, self.hidden_dim)

        # 时间因果自注意力层（处理 8*64=512 个带有动力学先验的 Token）
        self.temporal_pe = LearnablePositionalEncoding(self.hidden_dim, max_len=512)
        self.temporal_self_attn = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=8, 
            dim_feedforward=self.hidden_dim*4, 
            batch_first=True, norm_first=True
        )
        
        # ---------------------------------------------------------
        # 3. 多模态解码器 (Transformer Decoder)
        # ---------------------------------------------------------
        # RGB(256) + Depth(256) + 4D(512)
        # 总序列长度 = (memory_size*256)[RGB] + (memory_size*256)[Depth] + (8*64)[4D] = memory_size*512 + 512
        total_source_tokens = (self.memory_size * 2) * 256 + 512

        # 上帝视角的 Learnable Query
        # 目标Token数量严格保持与原版 RGBDBackbone 输出一致 (memory_size*16)个Query Token，实现对后端的无缝欺骗
        self.former_query = LearnablePositionalEncoding(self.hidden_dim, self.memory_size * 16)
        # 全局位置编码：让网络自动学习视锥与全向空间的对齐关系(对标NavDP的former_pe)
        self.global_pe = LearnablePositionalEncoding(self.hidden_dim, total_source_tokens)

        # 核心Decoder：用目标Query浓缩多模态源Token的信息
        self.former_net = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                self.hidden_dim, 
                8, 
                batch_first=True), 
                num_layers=2
            )
        # 输出投影：将384维拉升至Policy需要的维度（如512）
        self.project_layer = nn.Linear(self.hidden_dim, self.embed_size)
        self.to(device)

    def forward(self, input_images, input_depths, dynamic_voxels):
        """
        Args:
            input_images: 当前帧 RGB (B, memory_size, H, W, C) - 兼容 NavDP 格式
            input_depths: 当前帧 Depth (B, memory_size, H, W, C)
            dynamic_voxels: 未来 T 帧的 4D 运动场，稠密格式 (B, T=8, C=4, X, Y, Z)
        
        Returns:
            fused_tokens: (B, memory_size * 16, token_dim)
        """
        device = self._get_device()
        input_images = input_images.to(device)
        input_depths = input_depths.to(device)
        dynamic_voxels = dynamic_voxels.to(device)

        # --- 1. RGB 提取 ---
        # (Output: [B,memory_size*256,384])
        if len(input_images.shape) == 4:  # (B, H, W, C) -> (B, C, H, W)
            tensor_images = torch.as_tensor(input_images,dtype=torch.float32,device=device).permute(0,3,1,2)
            # ViT 需要固定尺寸输入，先 reshape 成 (B*memory_size, C, H, W)，提取特征后再 reshape 回 (B, memory_size*256, D)
            tensor_images = tensor_images.reshape(-1,3,self.image_size,self.image_size)
            # 均值方差归一化
            tensor_norm_images = (
                tensor_images - self.preprocess_mean.reshape(1,3,1,1).to(device)
            )/self.preprocess_std.reshape(1,3,1,1).to(device)
            # ViT 提取特征
            image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0]
        elif len(input_images.shape) == 5:  # 已经是 (B, memory_size, H, W, C)，直接处理
            # 多帧RGB：先平展时间维度，再恢复到（B，T*256，C）
            tensor_images = torch.as_tensor(input_images,dtype=torch.float32,device=device).permute(0,1,4,2,3) # [B,T,C,H,W]
            B,T_mem,C,H,W = tensor_images.shape
            # 先 reshape 成 (B*T_mem, C, H, W) 送入 ViT 提取特征，再 reshape 回 (B, T_mem*256, D)
            tensor_images = tensor_images.reshape(-1,3,self.image_size,self.image_size)

            # 均值方差归一化
            mean = self.preprocess_mean.reshape(1,3,1,1)
            std = self.preprocess_std.reshape(1,3,1,1)
            tensor_norm_images = (tensor_images - mean) / std
            image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0]
            image_token = image_token.reshape(B,T_mem*256,-1)

        if not self.finetune:
            # 推理或冻结模式下切断梯度
            image_token = image_token.detach()

        # --- 2. Depth 提取 ---
        # (Output: [B, memory_size*256,384])
        if len(input_depths.shape) == 4: # (B, H, W, C) -> (B, C, H, W)
            tensor_depths = torch.as_tensor(input_depths, dtype=torch.float32, device=device).permute(0,3,1,2)
            tensor_depths = tensor_depths.reshape(-1,1,self.image_size,self.image_size)
            # 单通道复制为三通道输入
            tensor_depths = torch.concat(
                [tensor_depths, tensor_depths, tensor_depths], dim=1
            )

            depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0]
        elif len(input_depths.shape) == 5:
            # 多帧深度：与RGB处理流程保持一致
            tensor_depths = torch.as_tensor(input_depths, dtype=torch.float32, device=device).permute(0,1,4,2,3) # [B,T,C,H,W]
            B,T_mem,C,H,W = tensor_depths.shape
            tensor_depths = tensor_depths.reshape(-1,1,self.image_size,self.image_size)
            tensor_depths = torch.concat(
                [tensor_depths, tensor_depths, tensor_depths], dim=1
            )
            depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0]
            depth_token = depth_token.reshape(B,T_mem*256,-1)
        
        # --- 3. 解析4D时空物理 ---
        # (Output: [B, 512, 384])
        B_dyn, T_dyn, C_dyn, X, Y, Z = dynamic_voxels.shape
        
        flat_voxels = dynamic_voxels.reshape(B_dyn*T_dyn,C_dyn,X,Y,Z)

        # 核心：根据Occ通道（通道0）判断占据状态
        mask = (flat_voxels[:,0,:,:,:].abs()>0)
        # 通过 mask 提取稀疏坐标 (Indices) 和特征 (Features)
        # indices 形状为 (N, 4)，第一列天然是 batch_index，第二列是时间维度 T_dyn
        indices = torch.nonzero(mask).int()  # (N, 4) -> [batch_index, C_dyn, x, y, z]

        # 提取4个通道的特征：[Occ,Vx,Vy,Vz],并且将时间维度 T_dyn 作为特征维度之一进行提取
        # 注意这里的 permute 是为了将时间维度 T_dyn 移动到特征维度，方便后续的稀疏卷积处理
        features = flat_voxels.permute(0,2,3,4,1)[mask]  # (N, C_dyn)
        
        sparse_tensor = spconv.SparseConvTensor(
            features=features, indices=indices, 
            spatial_shape=[X,Y,Z], batch_size=B_dyn*T_dyn
        )

        # 4D 卷积编码
        dense_out = self.dynamic_encoder(sparse_tensor).dense()
        # 膨胀回致密张量进行池化和维度对齐，最终得到 (B, 128, 4, 4, 4) 的动态物理特征
        pooled_out = self.dynamic_pool(dense_out)  # (B_dyn*T_dyn, 128, 4, 4, 4)
        
        # 整理成 Sequence 形式: (B_dyn*T_dyn, 128, 64) -> 投影 -> (B_dyn*T_dyn, 512, 384)
        # 这里的transpose是为了将空间维度的64个Token移到序列维度，方便后续的Transformer处理
        # (...,128,64) -> (...,64,128) -> 投影 -> (...,64,384)
        spatial_tokens = pooled_out.view(B_dyn*T_dyn,128,-1).transpose(1,2)
        spatial_tokens = self.dynamic_proj(spatial_tokens)

        physics_seq = spatial_tokens.reshape(B_dyn, T_dyn*64, self.hidden_dim)  # (B, 512, 384)
        # 加入时间位置编码，帮助模型区分未来8帧的时序关系
        # physics_seq[B, 512, 384] + temporal_pe[512, 384] -> (B, 512, 384)
        # 为什么physics_seq+temporal_pe后形状没变？因为temporal_pe是根据输入序列长度自动生成的，输出形状与输入相同，所以可以直接相加
        # temporal_pe是怎么加到physics_seq上的？是通过广播机制自动扩展到(B, 512, 384)的形状，然后逐元素相加的
        physics_seq = physics_seq + self.temporal_pe(physics_seq)   # (B, 512, 384)
        physics_tokens = self.temporal_self_attn(physics_seq)   # (B, 512, 384)

        # --- 4. 跨模态融合 ---
        unified_memory = torch.concat(
            (image_token,depth_token,physics_tokens),dim=1
            )
        # unified_memory[B, memory_size*512+512, 384] = concat(RGB[256], Depth[256], Physics[512]) + global_pe[memory_size*512+512, 384]
        unified_memory = unified_memory + self.global_pe(unified_memory)

        # Query (Q): 当前的语义 Token (探寻未来)，严格保持与原版 RGBDBackbone 输出一致 (B, memory_size*16, 384)
        query_zeros = torch.zeros(
            (B,self.memory_size*16,self.hidden_dim),device=device
            )
        # Key/Value (K, V): 未来的物理 Token (提供时空约束)，这里直接用 unified_memory 作为 K 和 V 输入 Transformer Decoder
        former_query = self.former_query(query_zeros)  # (B, memory_size*16, 384)

        # Cross_Attention跨模态检索
        memory_tokens = self.former_net(
            tgt=former_query,memory=unified_memory
            )
        
        fused_tokens = self.project_layer(memory_tokens)

        return fused_tokens  # (B, memory_size*16, embed_size)

    def _get_device(self):
        """安全获取当前模块所在设备。

        Returns:
            torch.device: 优先返回参数所在设备，若不可得则回退到可用默认设备。
        """
        # 1) 优先从模型参数中获取设备。
        try:
            for param in self.parameters():
                return param.device
        except StopIteration:
            pass

        # 2) 若无参数，则尝试从 buffer 获取设备。
        try:
            for buffer in self.buffers():
                return buffer.device
        except StopIteration:
            pass

        # 3) 进一步遍历子模块参数。
        for module in self.children():
            try:
                for param in module.parameters():
                    return param.device
            except StopIteration:
                continue

        # 4) 最后回退到系统可用默认设备。
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageGoalBackbone(nn.Module):
    """图像目标（Image Goal）编码骨干网络。"""

    def __init__(self, image_size=224, embed_size=512, device='cuda:0'):
        """初始化 ImageGoalBackbone。

        Args:
            image_size: 输入图像分辨率。
            embed_size: 输出特征维度。
            device: 目标设备，支持 None/int/str。
        """
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.image_size = image_size
        self.embed_size = embed_size
        model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
        self.imagegoal_encoder = DepthAnythingV2(**model_configs['vits'])
        self.imagegoal_encoder = self.imagegoal_encoder.pretrained.float()
        self.imagegoal_encoder.patch_embed.proj = nn.Conv2d(
            in_channels=6,
            out_channels=self.imagegoal_encoder.patch_embed.proj.out_channels,
            kernel_size=self.imagegoal_encoder.patch_embed.proj.kernel_size,
            stride=self.imagegoal_encoder.patch_embed.proj.stride,
            padding=self.imagegoal_encoder.patch_embed.proj.padding,
        )
        self.imagegoal_encoder.train()
        self.project_layer = nn.Linear(384, embed_size)
        self.to(device)

    def forward(self, images):
        """编码图像目标输入。

        Args:
            images: 输入图像，形状为 (B, H, W, C)。

        Returns:
            形状为 (B, embed_size) 的全局目标特征。
        """
        assert len(images.shape) == 4  # B,C,H,W
        device = self._get_device()
        images = images.to(device)
        tensor_images = torch.as_tensor(images, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        image_token = self.imagegoal_encoder.get_intermediate_layers(tensor_images)[0].mean(dim=1)
        image_token = self.project_layer(image_token)
        return image_token

    def _get_device(self):
        """安全获取当前模块所在设备。"""
        # 1) 优先从模型参数中获取设备。
        try:
            for param in self.parameters():
                return param.device
        except StopIteration:
            pass

        # 2) 若无参数，则尝试从 buffer 获取设备。
        try:
            for buffer in self.buffers():
                return buffer.device
        except StopIteration:
            pass

        # 3) 尝试从子模块参数中获取设备。
        for module in self.children():
            try:
                for param in module.parameters():
                    return param.device
            except StopIteration:
                continue

        # 4) 回退到系统默认设备。
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PixelGoalBackbone(nn.Module):
    """像素目标（Pixel Goal）编码骨干网络。
       像素目标的形式不是显式的(u,v)坐标，而是在数据集预处理阶段生成的带有目标掩码的图像输入。
       该骨干网络将这种特殊格式的图像输入编码成全局特征向量，供后续模块使用。
       真正喂给 PixelGoalBackbone 的是“像素位置热区/掩码 + 图像上下文”的多通道图，而不是裸坐标
    """

    def __init__(self, image_size=224, embed_size=512, pixel_channel=7, device='cuda:0'):
        """初始化 PixelGoalBackbone。

        Args:
            image_size: 输入图像分辨率。
            embed_size: 输出特征维度。
            pixel_channel: 输入通道数（例如 RGB + 额外掩码通道）。
            device: 目标设备，支持 None/int/str。
        """
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.image_size = image_size
        self.embed_size = embed_size
        model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
        self.pixelgoal_encoder = DepthAnythingV2(**model_configs['vits'])
        self.pixelgoal_encoder = self.pixelgoal_encoder.pretrained.float()
        self.pixelgoal_encoder.patch_embed.proj = nn.Conv2d(
            in_channels=pixel_channel,
            out_channels=self.pixelgoal_encoder.patch_embed.proj.out_channels,
            kernel_size=self.pixelgoal_encoder.patch_embed.proj.kernel_size,
            stride=self.pixelgoal_encoder.patch_embed.proj.stride,
            padding=self.pixelgoal_encoder.patch_embed.proj.padding,
        )
        self.pixelgoal_encoder.train()
        self.project_layer = nn.Linear(384, embed_size)
        self.to(device)

    def forward(self, images):
        """编码像素目标输入。

        Args:
            images: 输入图像，形状为 (B, H, W, C)。

        Returns:
            形状为 (B, embed_size) 的全局目标特征。
        """
        assert len(images.shape) == 4  # B,C,H,W
        device = self._get_device()
        images = images.to(device)
        tensor_images = torch.as_tensor(images, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        image_token = self.pixelgoal_encoder.get_intermediate_layers(tensor_images)[0].mean(dim=1)
        image_token = self.project_layer(image_token)
        return image_token

    def _get_device(self):
        """安全获取当前模块所在设备。"""
        # 1) 优先从模型参数中获取设备。
        try:
            for param in self.parameters():
                return param.device
        except StopIteration:
            pass

        # 2) 若无参数，则尝试从 buffer 获取设备。
        try:
            for buffer in self.buffers():
                return buffer.device
        except StopIteration:
            pass

        # 3) 尝试从子模块参数中获取设备。
        for module in self.children():
            try:
                for param in module.parameters():
                    return param.device
            except StopIteration:
                continue

        # 4) 回退到系统默认设备。
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
