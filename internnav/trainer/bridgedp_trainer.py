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
        """执行一次前向并计算 x₀-MSE 总损失。

        损失结构与 NavDP 对齐：
            loss = 0.8 * action_loss + 0.2 * critic_loss + 0.5 * aux_loss
        其中 action_loss = L_x0 + λ_delta * L_delta，均为均匀 MSE（无 SNR 加权）。
        valid_mask 用于屏蔽轨迹末端 padding 步。
        """
        model_device = next(model.parameters()).device

        inputs_on_device = {
            "batch_pg":             inputs["batch_pg"].to(model_device),
            "batch_ig":             inputs["batch_ig"].to(model_device),
            "batch_tg":             inputs["batch_tg"].to(model_device),
            "batch_rgb":            inputs["batch_rgb"].to(model_device),
            "batch_depth":          inputs["batch_depth"].to(model_device),
            "batch_labels":         inputs["batch_labels"].to(model_device),
            "batch_augments":       inputs["batch_augments"].to(model_device),
            "batch_label_critic":   inputs["batch_label_critic"].to(model_device),
            "batch_augment_critic": inputs["batch_augment_critic"].to(model_device),
            "batch_prior":          inputs["batch_prior"].to(model_device),
            "batch_theta_g":        inputs["batch_theta_g"].to(model_device),
            "batch_valid_mask":     inputs["batch_valid_mask"].to(model_device),
        }

        batch_label_critic   = inputs_on_device["batch_label_critic"]
        batch_augment_critic = inputs_on_device["batch_augment_critic"]
        # valid_mask: (B, T) bool，True 表示该时间步有效
        valid_mask = inputs_on_device["batch_valid_mask"].bool()  # (B, T)

        # 前向：返回 8 值元组（x₀-prediction）
        (x0_pred_ng, x0_pred_mg,
         critic_pred, augment_pred,
         ng_x0_target, mg_x0_target,
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

        # ── x₀-MSE（带 valid_mask，屏蔽 padding 步）─────────────────────
        mask = valid_mask.unsqueeze(-1).float()  # (B, T, 1)
        ng_x0_loss = ((x0_pred_ng - ng_x0_target).square() * mask).sum() / mask.sum().clamp(min=1)
        mg_x0_loss = ((x0_pred_mg - mg_x0_target).square() * mask).sum() / mask.sum().clamp(min=1)
        L_x0 = 0.5 * (ng_x0_loss + mg_x0_loss)

        # ── 增量一致性正则（鼓励预测轨迹平滑）────────────────────────────
        # L_delta = MSE(Δx̂₀, Δx₀_target)，Δ 为相邻步差分
        mask_delta = (mask[:, :-1] * mask[:, 1:])  # (B, T-1, 1)
        ng_delta_loss = (
            ((x0_pred_ng[:, 1:] - x0_pred_ng[:, :-1])
             - (ng_x0_target[:, 1:] - ng_x0_target[:, :-1])).square() * mask_delta
        ).sum() / mask_delta.sum().clamp(min=1)
        mg_delta_loss = (
            ((x0_pred_mg[:, 1:] - x0_pred_mg[:, :-1])
             - (mg_x0_target[:, 1:] - mg_x0_target[:, :-1])).square() * mask_delta
        ).sum() / mask_delta.sum().clamp(min=1)
        L_delta = 0.5 * (ng_delta_loss + mg_delta_loss)

        il_cfg = self.config.il if hasattr(self.config, 'il') else None
        lambda_delta = getattr(il_cfg, 'lambda_delta', 0.1) if il_cfg else 0.1
        action_loss = L_x0 + lambda_delta * L_delta

        # ── Critic 损失（与 NavDP 一致）──────────────────────────────
        critic_loss = (
            (critic_pred - batch_label_critic).square().mean()
            + (augment_pred - batch_augment_critic).square().mean()
        )

        # ── 辅助损失（与 NavDP 一致）──────────────────────────────────
        aux_loss = (
            0.5 * (inputs_on_device["batch_pg"] - imagegoal_aux_pred).square().mean()
            + 0.5 * (inputs_on_device["batch_pg"] - pixelgoal_aux_pred).square().mean()
        )

        # ── 总损失（3 项，与 NavDP 完全对齐）─────────────────────────
        loss = 0.8 * action_loss + 0.2 * critic_loss + 0.5 * aux_loss

        # ── 监控日志 ──────────────────────────────────────────────────
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            if not hasattr(self, '_log_step_count'):
                self._log_step_count = 0
            self._log_step_count += 1
            if self._log_step_count % 50 == 1:
                print(f"[Step {self._log_step_count}] "
                      f"loss={loss.item():.4f}, action={action_loss.item():.4f} "
                      f"(L_x0={L_x0.item():.4f}, L_delta={L_delta.item():.4f}), "
                      f"critic={critic_loss.item():.4f}, aux={aux_loss.item():.4f}")

            grad_norm = self._compute_grad_norm(model)
            self._monitor_logs = {
                "loss/total":      loss.item(),
                "loss/action":     action_loss.item(),
                "loss/L_x0":       L_x0.item(),
                "loss/L_delta":    L_delta.item(),
                "loss/ng_x0":      ng_x0_loss.item(),
                "loss/mg_x0":      mg_x0_loss.item(),
                "loss/critic":     critic_loss.item(),
                "loss/aux":        aux_loss.item(),
                "debug/grad_norm": grad_norm,
            }

            # 可视化：每 100 步执行一次真实去噪推理并写入 JSONL
            if self._log_step_count % 100 == 0:
                pred_traj = self._infer_pred_traj_bridgedp(model, inputs_on_device)
                self._write_traj_snapshot(
                    inputs_on_device["batch_labels"],
                    pred_traj,
                    inputs_on_device["batch_prior"],
                    inputs_on_device["batch_labels"],
                    inputs_on_device["batch_theta_g"],
                )

        outputs = {
            'x0_pred_ng':  x0_pred_ng,
            'x0_pred_mg':  x0_pred_mg,
            'critic_pred': critic_pred,
            'loss':        loss,
            'action_loss': action_loss,
            'L_x0':        L_x0.item(),
            'L_delta':     L_delta.item(),
            'critic_loss': critic_loss,
            'aux_loss':    aux_loss,
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

    def _infer_pred_traj_bridgedp(self, model, inputs_on_device):
        """对 batch[0] 执行布朗桥去噪推理，返回绝对坐标预测轨迹 (B, T, 3)。

        推理步数 10（与训练一致），仅取第 0 个样本，结果 expand 到 batch size。
        """
        model_ref = model.module if hasattr(model, 'module') else model
        B = inputs_on_device["batch_labels"].shape[0]
        device = inputs_on_device["batch_labels"].device

        def s(t):
            return t[0:1]

        was_training = model_ref.training
        model_ref.eval()
        try:
            with torch.no_grad():
                pg = s(inputs_on_device["batch_pg"])
                theta_g = s(inputs_on_device["batch_theta_g"])
                pg_n = model_ref._normalize_action(pg)

                pointgoal_embed = model_ref.point_encoder(pg_n).unsqueeze(1)
                rgbd_embed = model_ref.rgbd_encoder(
                    s(inputs_on_device["batch_rgb"]),
                    s(inputs_on_device["batch_depth"]),
                )

                # 先验（use_prior_traj=False 时 gated_prior 为零）
                gated_prior = torch.zeros(
                    1, model_ref.n_prior_tokens, model_ref.token_dim, device=device
                )

                # 有序区间初始化（与推理函数一致）
                origin = torch.zeros_like(pg_n)  # (1, 3)
                naction = model_ref.bridge_scheduler.sample_initial_noise_ordered(
                    goal=pg_n,
                    origin=origin,
                    d_max=model_ref.d_max,
                    shape=(1, model_ref.predict_size, 3),
                    device=device,
                )
                # 去噪时 goal = 导航目标
                goal_exp = pg_n.unsqueeze(1).expand(
                    -1, model_ref.predict_size, -1
                )
                theta_exp = theta_g

                model_ref.bridge_scheduler.set_timesteps(10)
                for k in model_ref.bridge_scheduler.timesteps:
                    x0_pred = model_ref.predict_x0(
                        naction, k.to(device).unsqueeze(0),
                        pointgoal_embed, rgbd_embed, gated_prior,
                    )
                    naction = model_ref.bridge_scheduler.step(
                        x0_pred, naction, k.to(device), goal_exp, theta_exp,
                    )

                pred_abs = model_ref._denormalize_action(naction)  # (1, T, 3)
        finally:
            if was_training:
                model_ref.train()

        return pred_abs.expand(B, -1, -1)

    def _write_traj_snapshot(self, x0_target, x0_pred, prior_traj, gt_labels, batch_theta_g=None):
        """将整个 batch 所有样本的轨迹追加写入 JSONL，供前端翻页可视化。由调用方控制写入频率。"""
        if not hasattr(self, '_log_step_count'):
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
