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

import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
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

        batch_label_critic = inputs_on_device["batch_label_critic"]
        batch_augment_critic = inputs_on_device["batch_augment_critic"]

        # 前向传播（Bridge-DP 新增 prior_traj 和 theta_g 参数）
        # 返回值新增 shared_timesteps 和 tensor_theta_g 用于 SNR 加权
        (x0_pred_ng, x0_pred_mg,
         critic_pred, augment_pred,
         x0_target_ng, x0_target_mg,
         imagegoal_aux_pred, pixelgoal_aux_pred,
         shared_timesteps, tensor_theta_g) = model(
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

        # ── SNR 加权：w(t) = 1 / σ²(t; θ_g) ──────────────────────────
        # 获取 bridge_scheduler 引用
        model_ref = model.module if hasattr(model, 'module') else model
        t_norm = model_ref.bridge_scheduler._normalized_time(shared_timesteps)  # (B,)
        theta_g_for_var = tensor_theta_g.view(-1)
        # 计算方差 σ²(t; θ_g)，形状 (B, 1, 1)
        variance = model_ref.bridge_scheduler.variance(
            t_norm.view(-1, 1, 1), theta_g_for_var.view(-1, 1, 1)
        )  # (B, 1, 1)
        # SNR 权重 w(t) = 1/σ²(t)，clamp 防止数值爆炸
        snr_weight = (1.0 / variance.clamp(min=0.01))  # (B, 1, 1)
        # 归一化使权重均值为 1（不改变总体损失量级）
        snr_weight = snr_weight / snr_weight.mean().clamp(min=1e-6)

        # ── 动作分支损失：SNR 加权 MSE ─────────────────────────────────
        ng_pointwise = (x0_pred_ng - x0_target_ng).square()  # (B, T, 3)
        mg_pointwise = (x0_pred_mg - x0_target_mg).square()  # (B, T, 3)
        ng_action_loss = (snr_weight * ng_pointwise).mean()
        mg_action_loss = (snr_weight * mg_pointwise).mean()

        # ── 轨迹结构正则化 ──────────────────────────────────────────────
        x0_pred_avg = 0.5 * x0_pred_ng + 0.5 * x0_pred_mg
        x0_target_avg = 0.5 * x0_target_ng + 0.5 * x0_target_mg

        # 1. 起点约束：预测轨迹第一个点与真值第一个点对齐
        start_loss = (x0_pred_avg[:, 0, :2] - x0_target_avg[:, 0, :2]).square().mean()

        # 2. 终点约束：与真值末端对齐（比 batch_pg 更稳定）
        terminal_loss = (x0_pred_avg[:, -1, :2] - x0_target_avg[:, -1, :2]).square().mean()

        # 3. 二阶方向平滑：惩罚方向突变而非绝对位移差
        pred_step = x0_pred_avg[:, 1:, :2] - x0_pred_avg[:, :-1, :2]  # (B, T-1, 2)
        smooth_loss = (pred_step[:, 1:, :] - pred_step[:, :-1, :]).square().mean() if pred_step.shape[1] > 1 else torch.tensor(0.0, device=model_device)

        # 4. 全局前进约束：只惩罚末端比起点更远离目标（允许绕行，不允许整体反转）
        goal_2d = inputs_on_device["batch_pg"][:, :2].unsqueeze(1)  # (B, 1, 2)
        start_dist = (x0_pred_avg[:, 0:1, :2] - goal_2d).norm(dim=-1)
        end_dist   = (x0_pred_avg[:, -1:, :2] - goal_2d).norm(dim=-1)
        global_forward_loss = torch.relu(end_dist - start_dist + 0.1).mean()

        # 5. 方向一致性约束
        pred_dir = pred_step
        gt_dir = x0_target_avg[:, 1:, :2] - x0_target_avg[:, :-1, :2]
        cos_sim = F.cosine_similarity(pred_dir + 1e-8, gt_dir + 1e-8, dim=-1)
        dir_loss = (1.0 - cos_sim).mean()

        # ── 辅助损失（与 NavDP 一致）──────────────────────────────────
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

        # ── 总损失 ──────────────────────────────────────────────────────
        loss = (0.8  * action_loss
                + 0.2  * critic_loss
                + 0.5  * aux_loss
                + 0.3  * start_loss           # 起点对齐
                + 0.2  * terminal_loss        # 终点对齐
                + 0.05 * smooth_loss          # 二阶方向平滑
                + 0.05 * global_forward_loss  # 全局前进（软约束）
                + 0.1  * dir_loss)            # 方向一致性

        # 先验 vs 真值偏差（量化先验质量）
        prior_gt_mse = (inputs_on_device["batch_prior"] - inputs_on_device["batch_labels"]).square().mean()

        # 收敛监控日志
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and not hasattr(self, '_log_step_count'):
            self._log_step_count = 0
        if rank == 0:
            self._log_step_count += 1
            if self._log_step_count % 50 == 1:
                print(f"[Step {self._log_step_count}] "
                      f"loss={loss.item():.4f}, action={action_loss.item():.4f}, "
                      f"start={start_loss.item():.4f}, term={terminal_loss.item():.4f}, "
                      f"smooth={smooth_loss.item():.4f}, gfwd={global_forward_loss.item():.4f}, "
                      f"dir={dir_loss.item():.4f}, "
                      f"snr_w=[{snr_weight.min().item():.2f},{snr_weight.max().item():.2f}]")

        # ── 监控增强：上报子 loss 到 HuggingFace log 系统 ──
        if rank == 0:
            # 计算梯度范数（在反向传播之前为上一步的梯度）
            grad_norm = self._compute_grad_norm(model)

            self._monitor_logs = {
                "loss/total":        loss.item(),
                "loss/action":       action_loss.item(),
                "loss/ng_action":    ng_action_loss.item(),
                "loss/mg_action":    mg_action_loss.item(),
                "loss/critic":       critic_loss.item(),
                "loss/aux":          aux_loss.item(),
                "loss/start":        start_loss.item(),
                "loss/terminal":     terminal_loss.item(),
                "loss/smooth":       smooth_loss.item(),
                "loss/global_fwd":   global_forward_loss.item(),
                "loss/direction":    dir_loss.item(),
                "loss/prior_gt_mse": prior_gt_mse.item(),
                "debug/snr_w_min":   snr_weight.min().item(),
                "debug/snr_w_max":   snr_weight.max().item(),
                "debug/x0_pred_min":   x0_pred_ng.min().item(),
                "debug/x0_pred_max":   x0_pred_ng.max().item(),
                "debug/x0_target_min": x0_target_ng.min().item(),
                "debug/x0_target_max": x0_target_ng.max().item(),
                "debug/grad_norm":   grad_norm,
            }

            # 每 N 步写入轨迹可视化数据（最后一个样本）
            self._write_traj_snapshot(
                x0_target_ng, x0_pred_ng,
                inputs_on_device["batch_prior"],
                inputs_on_device["batch_labels"],
                inputs_on_device["batch_theta_g"],
            )

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
            'start_loss': start_loss,
            'terminal_loss': terminal_loss,
            'smooth_loss': smooth_loss,
            'global_forward_loss': global_forward_loss,
            'dir_loss': dir_loss,
        }

        return (loss, outputs) if return_outputs else loss

    def _compute_grad_norm(self, model):
        """计算当前梯度的 L2 范数（用于监控梯度爆炸/消失）。"""
        try:
            model_ref = model.module if hasattr(model, 'module') else model
            total_norm = 0.0
            count = 0
            for p in model_ref.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
                    count += 1
            if count == 0:
                return 0.0
            return total_norm ** 0.5
        except Exception:
            return 0.0

    def _write_traj_snapshot(self, x0_target, x0_pred, prior_traj, gt_labels, batch_theta_g=None):
        """将整个 batch 所有样本的轨迹追加写入 JSONL，供前端翻页可视化。

        每 10 步写一次，避免 I/O 过于频繁。
        """
        if not hasattr(self, '_log_step_count') or self._log_step_count % 10 != 0:
            return
        try:
            log_dir = Path(self.args.output_dir).parent / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            batch_file = log_dir / 'traj_batches.jsonl'

            B = x0_target.shape[0]
            theta_g_list = batch_theta_g.detach().cpu().view(-1).tolist() if batch_theta_g is not None else [None] * B
            record = {
                "batch_idx": self._log_step_count,
                "step": self._log_step_count,
                "samples": [
                    {
                        "gt_traj":    x0_target[i].detach().cpu().tolist(),
                        "pred_traj":  x0_pred[i].detach().cpu().tolist(),
                        "prior_traj": prior_traj[i].detach().cpu().tolist(),
                        "gt_labels":  gt_labels[i].detach().cpu().tolist(),
                        "theta_g":    theta_g_list[i],
                    }
                    for i in range(B)
                ],
            }
            with open(batch_file, 'a') as f:
                f.write(json.dumps(record) + '\n')
        except Exception as e:
            print(f"[TrajectoryVis] Failed to write traj snapshot: {e}")

    def log(self, logs, *args, **kwargs):
        """重写 log 方法，注入子 loss 指标到 HuggingFace 日志系统。"""
        if hasattr(self, '_monitor_logs') and self._monitor_logs:
            logs.update(self._monitor_logs)
            self._monitor_logs = {}
        return super().log(logs, *args, **kwargs)

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
