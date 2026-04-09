import os
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from internnav.trainer.base import BaseTrainer


class FlowNavTrainer(BaseTrainer):
    """FlowNav 模型训练器。

    上下游链路说明：
    1. 上游数据链路：数据集 -> 数据加载器 -> 训练器
        'get_train_dataloader' 从 'self.train_dataset' 构建分布式 DataLoader, 
        每个 step 产出一个batch（由 'self.data_collator' 整理字段结构）。
    2. 中游训练链路：训练器 -> 模型 -> 损失函数 -> 优化器
        训练主循环（定义在父类 BaseTrainer）会调用 'compute_loss' 完成前向与损失计算，
        然后基于 `create_optimizer` / `create_scheduler` 产出的对象执行反向传播与参数更新。
    3. 下游产物链路：训练器 -> 模型权重 -> 模型评估 -> 模型部署
        `save_model` 负责将最终可部署权重写入磁盘，兼容 DDP 包装与非 DDP 模式。
    """

    def __init__(self, config, **kwargs):
        """初始化训练器运行时状态。

        参数:
            config: 实验配置对象，至少应包含 `config.il` 训练子配置。
            **kwargs: 透传给 `BaseTrainer` 的初始化参数。
        """
        super().__init__(**kwargs)
        self.config = config
        self.writer = None
        # self.iterations = config.checkpoint * 141
        self.start_time = time.time()
        # 为日志与调试保留一个统一设备标识。
        # DDP 模式下真实模型在 self.model.module 中。
        if hasattr(self.model, 'module'): # DDP 包装的模型
            self.model_device = self.model.module.device
        else:
            self.model_device = self.model.device
        
        # 仅在 rank 0 输出设备信息，避免 DDP 模式下重复打印。
        print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] Model device: {self.model_device}")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """执行一次前向并计算总损失。

        这是训练链路的核心函数：
        - 上游输入: DataLoader 产生的 batch 字段（pg/ig/tg/rgb/depth/labels/odom/odom_delta/...）
        - 中游处理: 将 batch 放到模型设备，调用 model 前向，组装多项损失
        - 下游输出: 返回标量 loss（用于 backward）以及可选的细分 outputs（用于日志/分析）

        参数:
            model: 当前参与训练的模型实例（可能是 DDP 包装后的模型）。
            inputs: 一个 batch 的字典输入。
            return_outputs: 是否额外返回细粒度中间输出。
            num_items_in_batch: 与父类接口保持一致，当前未使用。

        返回:
            return_outputs=False 时返回 Tensor 标量 loss；
            return_outputs=True 时返回 (loss, outputs) 元组。
        """
        # 通过参数设备推断当前模型所在设备，避免硬编码 cuda:0。
        model_device = next(model.parameters()).device

        # 第一轮通用搬运：将输入中的 Tensor 迁移到模型设备，
        # 非 Tensor 字段原样保留，便于兼容不同数据结构。
        inputs_on_device = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                # pin_memory + non_blocking=True 可减少 H2D 拷贝等待。
                inputs_on_device[key] = value.to(model_device, non_blocking=True)
            else:
                inputs_on_device[key] = value

        import os

        import psutil

        # 进程树调试信息：用于确认实际训练子进程数量，
        # 常见于多卡 DDP 排障（例如 world_size 与进程数不一致）。
        current_pid = os.getpid()
        process = psutil.Process(current_pid)
        parent = process.parent()

        if parent:
            children = parent.children()
            if len(children) == 8:
                print("There are 8 training processes running")
            else:
                print(f"There are {len(children)} training processes running")
        else:
            print("Cannot determine parent process")

        # 第二轮显式字段搬运：确保模型前向所需关键字段都在目标设备。
        # 这部分与上面的通用搬运存在冗余，但可明确约束关键输入字段。
        inputs_on_device = {
            "batch_pg": inputs["batch_pg"].to(model_device),
            "batch_ig": inputs["batch_ig"].to(model_device),
            "batch_tg": inputs["batch_tg"].to(model_device),
            "batch_rgb": inputs["batch_rgb"].to(model_device),
            "batch_depth": inputs["batch_depth"].to(model_device),
            "batch_labels": inputs["batch_labels"].to(model_device),
            "batch_augments": inputs["batch_augments"].to(model_device),
            "batch_label_critic": inputs["batch_label_critic"].to(model_device),
            "batch_augment_critic": inputs["batch_augment_critic"].to(model_device),
        }
        # 显式同步用于更稳定地观察性能/错误位置（会降低吞吐，偏调试用途）。
        torch.cuda.synchronize(model_device)

        # 仅取损失计算需要的监督信号；其余输入由model前向使用。
        # batch_pg = inputs["batch_pg"]
        # batch_ig = inputs["batch_ig"]
        # batch_rgb = inputs["batch_rgb"]
        # batch_depth = inputs["batch_depth"]
        # batch_labels = inputs["batch_labels"]
        # batch_augments = inputs["batch_augments"]
        batch_label_critic = inputs["batch_label_critic"]
        batch_augment_critic = inputs["batch_augment_critic"]
        # batch_odom = inputs["batch_odom"]
        # batch_odom_delta = inputs["batch_odom_delta"]

        # 前向输出语义：
        # - pred_ng / pred_mg: 两个动作分支预测
        # - critic_pred / augment_pred: 价值或辅助判别分支预测