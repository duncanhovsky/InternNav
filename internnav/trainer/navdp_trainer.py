import os
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from internnav.trainer.base import BaseTrainer


class NavDPTrainer(BaseTrainer):
    """NavDP 模型训练器。

    上下游链路说明：
    1. 上游数据链路：
       `get_train_dataloader` 从 `self.train_dataset` 构建分布式 DataLoader，
       每个 step 产出一个 batch（由 `self.data_collator` 整理字段结构）。
    2. 中游训练链路：
       训练主循环（定义在父类 BaseTrainer）会调用 `compute_loss` 完成前向与损失计算，
       然后基于 `create_optimizer` / `create_scheduler` 产出的对象执行反向传播与参数更新。
    3. 下游产物链路：
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
        if hasattr(self.model, 'module'):  # DDP wrapped model
            self.model_device = self.model.module.device
        else:
            self.model_device = self.model.device

        print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] Model device: {self.model_device}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """执行一次前向并计算总损失。

        这是训练链路的核心函数：
        - 上游输入: DataLoader 产生的 batch 字段（pg/ig/tg/rgb/depth/labels/...）
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

        # 仅取损失计算需要的监督信号；其余输入由 model 前向直接消费。
        # batch_pg = inputs["batch_pg"]
        # batch_ig = inputs["batch_ig"]
        # batch_rgb = inputs["batch_rgb"]
        # batch_depth = inputs["batch_depth"]
        # batch_labels = inputs["batch_labels"]
        # batch_augments = inputs["batch_augments"]
        batch_label_critic = inputs["batch_label_critic"]
        batch_augment_critic = inputs["batch_augment_critic"]

        # 前向输出语义：
        # - pred_ng / pred_mg: 两个动作分支预测
        # - critic_pred / augment_pred: 价值或辅助判别分支预测
        # - ng_noise / mg_noise: 动作分支监督目标（噪声回归目标）
        # - *_aux_pred: 对目标表征的辅助重建预测
        pred_ng, pred_mg, critic_pred, augment_pred, ng_noise, mg_noise, imagegoal_aux_pred, pixelgoal_aux_pred = model(
            inputs_on_device["batch_pg"],
            inputs_on_device["batch_ig"],
            inputs_on_device["batch_tg"],
            inputs_on_device["batch_rgb"],
            inputs_on_device["batch_depth"],
            inputs_on_device["batch_labels"],
            inputs_on_device["batch_augments"],
        )

        # 动作分支损失：噪声回归 MSE。
        ng_action_loss = (pred_ng - ng_noise).square().mean()
        mg_action_loss = (pred_mg - mg_noise).square().mean()

        # 辅助损失：将 imagegoal / pixelgoal 的辅助头都回归到 batch_pg 表征。
        # 该项用于稳定表征学习，提升主任务泛化。
        aux_loss = (
            0.5 * (inputs_on_device["batch_pg"] - imagegoal_aux_pred).square().mean()
            + 0.5 * (inputs_on_device["batch_pg"] - pixelgoal_aux_pred).square().mean()
        )

        # 主动作损失由 ng/mg 两个分支等权组成。
        action_loss = 0.5 * mg_action_loss + 0.5 * ng_action_loss

        # critic 分支损失：分别约束 critic_pred 与 augment_pred。
        critic_loss = (critic_pred - batch_label_critic).square().mean() + (
            augment_pred - batch_augment_critic
        ).square().mean()

        # 总损失配比：动作 0.8 + critic 0.2 + 辅助 0.5。
        # 注意这并非概率归一权重，而是经验加权系数。
        loss = 0.8 * action_loss + 0.2 * critic_loss + 0.5 * aux_loss

        # 将关键中间量暴露给上游 Trainer，便于日志、可视化和离线排障。
        outputs = {
            'pred_ng': pred_ng,
            'pred_mg': pred_mg,
            'critic_pred': critic_pred,
            'augment_pred': augment_pred,
            'noise': [ng_noise, mg_noise],
            'loss': loss,
            'ng_action_loss': ng_action_loss,
            'mg_action_loss': mg_action_loss,
            'aux_loss': aux_loss,
            'critic_loss': critic_loss,
        }
        # if self.logger:
        #     self.logger.info(
        #         f"[Step {self.state.global_step}] "
        #         f"Loss: {loss.item():.4f}, "
        #         f"Action Loss: {action_loss.item():.4f}, "
        #         f"Critic Loss: {critic_loss.item():.4f}"
        #     )

        # 与 HuggingFace Trainer 风格保持一致：
        # return_outputs 控制是否同时返回可分析字典。
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """创建并返回优化器。

        上游依赖:
            `self.config.il.lr` 学习率配置。
        下游使用:
            由父类训练循环在每个 step 中调用 `optimizer.step()`。
        """
        rank = dist.get_rank() if dist.is_initialized() else 0

        # 优先读取配置学习率，缺省时回退到常用默认值。
        try:
            lr = self.config.il.lr
            if rank == 0:
                print(f"[Rank 0] Using learning rate: {lr}")
        except AttributeError:
            lr = 1e-4
            if rank == 0:
                print(f"[Rank 0] Warning: Using default learning rate: {lr}")

        # DDP 模式下优化器应绑定原始模型参数（module）。
        if hasattr(self.model, 'module'):
            model_for_optim = self.model.module
        else:
            model_for_optim = self.model

        # 这里使用 Adam 作为参数更新器。
        optimizer = torch.optim.Adam(model_for_optim.parameters(), lr=lr)

        if rank == 0:
            print(f"[Rank 0] Optimizer created with {len(optimizer.param_groups)} param groups")
            total_params = sum(p.numel() for p in model_for_optim.parameters() if p.requires_grad)
            print(f"[Rank 0] Total trainable parameters: {total_params:,}")

        return optimizer

    def create_scheduler(self, optimizer, num_training_steps: int):
        """创建学习率调度器。

        参数:
            optimizer: 已构建完成的优化器。
            num_training_steps: 训练总步数（当前实现未直接使用）。

        返回:
            `LinearLR`，在前 10000 次迭代将学习率线性衰减到初始值的 0.5。
        """
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=10000)
        return scheduler

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """覆盖父类方法，统一控制优化器与调度器创建顺序。"""
        print("\n=== create optimizer and scheduler ===")

        # 先创建优化器，再基于该优化器创建调度器。
        self.optimizer = self.create_optimizer()

        # 参数顺序保持与本类 `create_scheduler` 签名一致。
        self.lr_scheduler = self.create_scheduler(self.optimizer, num_training_steps)

        return self.optimizer, self.lr_scheduler

    def get_train_dataloader(self):
        """构建训练 DataLoader（支持 DDP 切分）。

        上游:
            `self.train_dataset` 由外部初始化注入。
        下游:
            父类训练循环按迭代读取 batch，并传入 `compute_loss`。
        """
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        # 分布式采样器确保每个 rank 读取不同子集，并在每个 epoch 保持可复现打乱。
        sampler = DistributedSampler(self.train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=1234)

        loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.il.batch_size,
            sampler=sampler,
            num_workers=self.config.il.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.data_collator,
        )
        # print(loader)
        return loader

    def save_model(self, output_dir, state_dict=None, **kwargs):
        """
        保存模型到指定目录。

        关键点:
            1. 若处于 DDP 模式，需先解包到原始模型再导出权重。
            2. 仅保存 `state_dict`，便于后续灵活恢复到同构模型实例。

        参数:
            output_dir: 权重输出目录。
            state_dict: 预留参数，当前实现未使用。
            **kwargs: 预留扩展参数，兼容父类接口。
        """
        # 检查是否为 DDP 包装模型，决定保存对象。
        if hasattr(self.model, 'module'):
            # 取原始模型，避免保存 DDP wrapper 的外层结构。
            model_to_save = self.model.module
        else:
            model_to_save = self.model

        # 确保输出目录存在后再写入 checkpoint。
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model_to_save.state_dict(), output_dir + "navdp.ckpt")

        print(f"Saving model to {output_dir} (is DDP: {hasattr(self.model, 'module')})")
