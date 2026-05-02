"""Bridge-DP 模型训练器。

独立继承 ``BaseTrainer``（与 ``NavDPTrainer`` 使用相同基类），
仿照 ``NavDPTrainer`` 的实现思路，实现 Bridge-DP 的训练逻辑。

与 NavDPTrainer 的核心区别：
1. compute_loss: 预测目标为 x̂_0（干净轨迹）而非噪声 ε
2. 新增 batch_prior 和 batch_theta_g 输入字段
3. collate_fn 使用 bridgedp_collate_fn
4. 保存模型文件名为 bridgedp.ckpt

参考：
    - internnav/trainer/navdp_trainer.py（NavDP 训练器，不修改）
    - internnav/trainer/base.py（BaseTrainer 基类）
    - docs/Bridge-DP推导.md §6（训练目标）
"""

import os
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from internnav.trainer.base import BaseTrainer


class BridgeDPTrainer(BaseTrainer):
    """Bridge-DP 训练器。

    上下游链路说明（与 NavDPTrainer 一致）：
    1. 上游数据链路：get_train_dataloader 构建分布式 DataLoader。
    2. 中游训练链路：compute_loss 完成前向与损失计算。
    3. 下游产物链路：save_model 将权重写入磁盘。

    场景自检：
        1. 正前方目标：桥均值接近标签 → x̂_0 预测简单 → 损失小。
        2. 正后方目标：桥均值偏离标签 → 方差大 → x̂_0 预测困难 → 损失大但网络有充足方差空间。
        3. 对抗先验训练 (30%错误先验)：网络学会通过 VisualGate 降低 G → 不影响主损失。
    """

    def __init__(self, config, **kwargs):
        """初始化训练器。

        Args:
            config: 实验配置对象（ExpCfg），至少包含 config.il。
            **kwargs: 透传给 BaseTrainer。
        """
        super().__init__(**kwargs)
        self.config = config
        self.writer = None
        self.start_time = time.time()

        if hasattr(self.model, 'module'):
            self.model_device = self.model.module.device
        else:
            self.model_device = self.model.device

        print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] Model device: {self.model_device}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """执行一次前向并计算总损失。

        与 NavDPTrainer.compute_loss 的区别：
        - 新增 batch_prior, batch_theta_g 输入
        - 损失基于 x̂_0（干净轨迹预测）MSE 而非噪声 ε MSE
        - 损失配比保持一致：0.8 * action + 0.2 * critic + 0.5 * aux

        Args:
            model: 当前模型实例。
            inputs: batch 字典。
            return_outputs: 是否额外返回中间输出。
            num_items_in_batch: 未使用。

        Returns:
            loss 或 (loss, outputs)。
        """
        model_device = next(model.parameters()).device

        # 将输入迁移到模型设备
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
            "batch_prior": inputs["batch_prior"].to(model_device),
            "batch_theta_g": inputs["batch_theta_g"].to(model_device),
        }
        torch.cuda.synchronize(model_device)

        batch_label_critic = inputs["batch_label_critic"]
        batch_augment_critic = inputs["batch_augment_critic"]

        # 前向传播（Bridge-DP 新增 prior_traj 和 theta_g 参数）
        (x0_pred_ng, x0_pred_mg,
         critic_pred, augment_pred,
         x0_target_ng, x0_target_mg,
         imagegoal_aux_pred, pixelgoal_aux_pred) = model(
            inputs_on_device["batch_pg"],
            inputs_on_device["batch_ig"],
            inputs_on_device["batch_tg"],
            inputs_on_device["batch_rgb"],
            inputs_on_device["batch_depth"],
            inputs_on_device["batch_labels"],
            inputs_on_device["batch_augments"],
            inputs_on_device["batch_prior"],
            inputs_on_device["batch_theta_g"],
        )

        # 动作分支损失：预测 x̂_0 vs 真实 x_0 的 MSE
        # （NavDP 是 pred_noise vs noise 的 MSE）
        ng_action_loss = (x0_pred_ng - x0_target_ng).square().mean()
        mg_action_loss = (x0_pred_mg - x0_target_mg).square().mean()

        # 辅助损失（与 NavDP 一致）
        aux_loss = (
            0.5 * (inputs_on_device["batch_pg"] - imagegoal_aux_pred).square().mean()
            + 0.5 * (inputs_on_device["batch_pg"] - pixelgoal_aux_pred).square().mean()
        )

        # 主动作损失
        action_loss = 0.5 * mg_action_loss + 0.5 * ng_action_loss

        # Critic 损失（与 NavDP 一致）
        critic_loss = (
            (critic_pred - batch_label_critic).square().mean()
            + (augment_pred - batch_augment_critic).square().mean()
        )

        # 总损失（配比与 NavDP 一致）
        loss = 0.8 * action_loss + 0.2 * critic_loss + 0.5 * aux_loss

        outputs = {
            'x0_pred_ng': x0_pred_ng,
            'x0_pred_mg': x0_pred_mg,
            'critic_pred': critic_pred,
            'augment_pred': augment_pred,
            'loss': loss,
            'ng_action_loss': ng_action_loss,
            'mg_action_loss': mg_action_loss,
            'aux_loss': aux_loss,
            'critic_loss': critic_loss,
        }

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """创建优化器。仿照 NavDPTrainer.create_optimizer。"""
        rank = dist.get_rank() if dist.is_initialized() else 0
        try:
            lr = self.config.il.lr
            if rank == 0:
                print(f"[Rank 0] Using learning rate: {lr}")
        except AttributeError:
            lr = 1e-4
            if rank == 0:
                print(f"[Rank 0] Warning: Using default learning rate: {lr}")

        if hasattr(self.model, 'module'):
            model_for_optim = self.model.module
        else:
            model_for_optim = self.model

        optimizer = torch.optim.Adam(model_for_optim.parameters(), lr=lr)

        if rank == 0:
            total_params = sum(p.numel() for p in model_for_optim.parameters() if p.requires_grad)
            print(f"[Rank 0] Total trainable parameters: {total_params:,}")

        return optimizer

    def create_scheduler(self, optimizer, num_training_steps: int):
        """创建学习率调度器。仿照 NavDPTrainer.create_scheduler。"""
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.5, total_iters=10000
        )
        return scheduler

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """统一控制优化器与调度器创建。"""
        print("\n=== create optimizer and scheduler ===")
        self.optimizer = self.create_optimizer()
        self.lr_scheduler = self.create_scheduler(self.optimizer, num_training_steps)
        return self.optimizer, self.lr_scheduler

    def get_train_dataloader(self):
        """构建训练 DataLoader。仿照 NavDPTrainer.get_train_dataloader。"""
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        sampler = DistributedSampler(
            self.train_dataset, num_replicas=world_size,
            rank=rank, shuffle=True, seed=1234
        )

        loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.il.batch_size,
            sampler=sampler,
            num_workers=self.config.il.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.data_collator,
        )
        return loader

    def save_model(self, output_dir, state_dict=None, **kwargs):
        """保存模型到指定目录。仿照 NavDPTrainer.save_model。"""
        if hasattr(self.model, 'module'):
            model_to_save = self.model.module
        else:
            model_to_save = self.model

        os.makedirs(output_dir, exist_ok=True)
        torch.save(model_to_save.state_dict(), output_dir + "bridgedp.ckpt")
        print(f"Saving model to {output_dir} (is DDP: {hasattr(self.model, 'module')})")
