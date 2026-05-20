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
import math
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

    # ------------------------------------------------------------------
    # GPU 轨迹曲线监督
    # ------------------------------------------------------------------

    def _wrap_to_pi(self, angles: torch.Tensor) -> torch.Tensor:
        """将角度包裹到 [-pi, pi)。"""
        return torch.remainder(angles + math.pi, 2.0 * math.pi) - math.pi

    def _unwrap_angles(self, angles: torch.Tensor) -> torch.Tensor:
        """torch 版 unwrap，保持在当前 device 上执行。"""
        if angles.numel() <= 1:
            return angles
        diffs = angles[1:] - angles[:-1]
        wrapped = self._wrap_to_pi(diffs)
        wrapped = torch.where(
            (wrapped == -math.pi) & (diffs > 0),
            torch.full_like(wrapped, math.pi),
            wrapped,
        )
        correction = torch.cumsum(wrapped - diffs, dim=0)
        return torch.cat([angles[:1], angles[1:] + correction], dim=0)

    def _natural_cubic_eval(
        self,
        s: torch.Tensor,
        y: torch.Tensor,
        q: torch.Tensor,
    ) -> torch.Tensor:
        """在 GPU 上求自然三次样条并于 q 处重采样。

        Args:
            s: (N,) 严格递增、归一化到 [0, 1] 的弧长参数。
            y: (N, C) 曲线值。
            q: (T,) 查询点。
        """
        n = s.shape[0]
        if n == 1:
            return y.expand(q.shape[0], -1)
        if n == 2:
            denom = (s[1] - s[0]).clamp(min=1e-6)
            alpha = ((q - s[0]) / denom).clamp(0.0, 1.0).unsqueeze(-1)
            return (1.0 - alpha) * y[0:1] + alpha * y[1:2]

        h = (s[1:] - s[:-1]).clamp(min=1e-6)
        A = torch.zeros((n, n), device=s.device, dtype=s.dtype)
        rhs = torch.zeros((n, y.shape[-1]), device=s.device, dtype=y.dtype)
        A[0, 0] = 1.0
        A[-1, -1] = 1.0
        for i in range(1, n - 1):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2.0 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
            rhs[i] = 6.0 * (
                (y[i + 1] - y[i]) / h[i]
                - (y[i] - y[i - 1]) / h[i - 1]
            )
        second = torch.linalg.solve(A, rhs)

        idx = torch.searchsorted(s.contiguous(), q.contiguous(), right=True) - 1
        idx = idx.clamp(0, n - 2)
        s_i = s[idx]
        s_next = s[idx + 1]
        h_i = (s_next - s_i).clamp(min=1e-6)
        a = (s_next - q) / h_i
        b = (q - s_i) / h_i

        y_i = y[idx]
        y_next = y[idx + 1]
        m_i = second[idx]
        m_next = second[idx + 1]
        return (
            a.unsqueeze(-1) * y_i
            + b.unsqueeze(-1) * y_next
            + (((a ** 3 - a).unsqueeze(-1) * m_i
                + (b ** 3 - b).unsqueeze(-1) * m_next)
               * (h_i ** 2).unsqueeze(-1) / 6.0)
        )

    def _resample_one_trajectory_gpu(
        self,
        traj: torch.Tensor,
        num_points: int,
    ) -> torch.Tensor:
        """用 x/y 弧长参数在当前 GPU 上把一条原始轨迹重采样为 24 个未来点。"""
        traj = traj[:, :3]
        if traj.shape[0] == 0:
            return torch.zeros((num_points, 3), device=traj.device, dtype=traj.dtype)
        if traj.shape[0] == 1:
            return traj[-1:].expand(num_points, -1)

        theta_unwrapped = self._unwrap_angles(traj[:, 2])
        curve = torch.cat([traj[:, :2], theta_unwrapped.unsqueeze(-1)], dim=-1)

        dxy = torch.norm(traj[1:, :2] - traj[:-1, :2], dim=-1)
        s_full = torch.cat([dxy.new_zeros(1), torch.cumsum(dxy, dim=0)])
        total = s_full[-1]
        if total <= 1e-6:
            return curve[-1:].expand(num_points, -1).clone()

        keep = torch.cat([
            torch.ones(1, device=traj.device, dtype=torch.bool),
            dxy > 1e-6,
        ])
        keep_idx = torch.nonzero(keep, as_tuple=False).flatten()
        s = s_full[keep_idx]
        y = curve[keep_idx]
        # 若末端是重复静止点，用最后一帧的 theta 覆盖同弧长末端，保留 GT 姿态监督。
        y[-1] = curve[-1]
        s = s / total

        q = torch.linspace(
            1.0 / float(num_points),
            1.0,
            num_points,
            device=traj.device,
            dtype=traj.dtype,
        )
        sampled = self._natural_cubic_eval(s, y, q)
        sampled[:, 2] = self._wrap_to_pi(sampled[:, 2])
        return sampled

    def _resample_trajectories_gpu(
        self,
        raw_trajs: torch.Tensor,
        lengths: torch.Tensor,
        num_points: int,
    ) -> torch.Tensor:
        """对 batch 内变长原始轨迹做 GPU 弧长样条重采样。"""
        samples = []
        for bid in range(raw_trajs.shape[0]):
            n = int(lengths[bid].item())
            n = max(1, min(n, raw_trajs.shape[1]))
            samples.append(self._resample_one_trajectory_gpu(raw_trajs[bid, :n], num_points))
        return torch.stack(samples, dim=0)

    def _normalize_action_tensor(self, action: torch.Tensor) -> torch.Tensor:
        """与 BridgeDPNet/Dataset 一致的动作归一化。"""
        out = action.clone()
        out[..., 0:2] = out[..., 0:2] / 5.0
        out[..., 2] = out[..., 2] / 3.14159
        return out

    def _generate_prior_trajectory_gpu(
        self,
        labels: torch.Tensor,
        is_task_start: torch.Tensor,
    ) -> torch.Tensor:
        """在 GPU 上根据重采样标签生成 Bridge-DP 先验轨迹。"""
        B, T, _ = labels.shape
        prior = torch.zeros_like(labels)
        active = ~is_task_start.view(-1).bool()
        if not active.any():
            return prior

        q = torch.linspace(
            1.0 / float(T), 1.0, T, device=labels.device, dtype=labels.dtype
        ).view(1, T, 1)
        correct = torch.rand((B,), device=labels.device) < 0.7
        correct = correct & active
        if correct.any():
            end = labels[correct, -1:].clone()
            noise_std = 0.005 * torch.norm(end.squeeze(1), dim=-1, keepdim=True).view(-1, 1, 1)
            prior[correct] = q * end + torch.randn_like(labels[correct]) * noise_std

        wrong = active & ~correct
        if wrong.any():
            angles = torch.empty((int(wrong.sum().item()),), device=labels.device, dtype=labels.dtype)
            angles.uniform_(math.pi / 3.0, 5.0 * math.pi / 3.0)
            cos_a = torch.cos(angles)
            sin_a = torch.sin(angles)
            wrong_labels = labels[wrong]
            x = wrong_labels[..., 0]
            y = wrong_labels[..., 1]
            prior_wrong = wrong_labels.clone()
            prior_wrong[..., 0] = cos_a.view(-1, 1) * x - sin_a.view(-1, 1) * y
            prior_wrong[..., 1] = sin_a.view(-1, 1) * x + cos_a.view(-1, 1) * y
            prior[wrong] = prior_wrong
        return prior

    def _prepare_curve_supervision(self, inputs_on_device: dict, predict_size: int) -> None:
        """用 GPU 生成弧长样条监督标签，并原地覆盖 batch 字段。"""
        if "batch_raw_labels" not in inputs_on_device:
            return

        labels_phys = self._resample_trajectories_gpu(
            inputs_on_device["batch_raw_labels"],
            inputs_on_device["batch_raw_lengths"],
            predict_size,
        )
        augments_phys = self._resample_trajectories_gpu(
            inputs_on_device["batch_raw_augments"],
            inputs_on_device["batch_raw_lengths"],
            predict_size,
        )
        labels = self._normalize_action_tensor(labels_phys)
        augments = self._normalize_action_tensor(augments_phys)
        is_task_start = inputs_on_device.get(
            "batch_is_task_start",
            torch.zeros(labels.shape[0], device=labels.device, dtype=torch.bool),
        )

        inputs_on_device["batch_labels"] = labels
        inputs_on_device["batch_augments"] = augments
        inputs_on_device["batch_prior"] = self._generate_prior_trajectory_gpu(labels, is_task_start)
        inputs_on_device["batch_valid_mask"] = torch.ones(
            labels.shape[:2], device=labels.device, dtype=torch.float32
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """执行一次前向并计算 x₀-MSE 总损失。

        损失结构与 NavDP 对齐：
            loss = 0.8 * action_loss + 0.2 * critic_loss + 0.5 * aux_loss
        其中 action_loss = L_x0 + λ_delta * L_delta，均为均匀 MSE（无 SNR 加权）。
        轨迹监督由原始 GT 曲线在 GPU 上按弧长样条重采样得到，valid_mask 固定全 1。
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
        if "batch_raw_labels" in inputs:
            inputs_on_device.update({
                "batch_raw_labels":     inputs["batch_raw_labels"].to(model_device),
                "batch_raw_augments":   inputs["batch_raw_augments"].to(model_device),
                "batch_raw_lengths":    inputs["batch_raw_lengths"].to(model_device),
                "batch_is_task_start":  inputs["batch_is_task_start"].to(model_device),
            })

        model_ref = model.module if hasattr(model, 'module') else model
        self._prepare_curve_supervision(
            inputs_on_device,
            predict_size=getattr(model_ref, "predict_size", inputs_on_device["batch_labels"].shape[1]),
        )

        batch_label_critic   = inputs_on_device["batch_label_critic"]
        batch_augment_critic = inputs_on_device["batch_augment_critic"]
        # valid_mask: (B, T) bool，新轨迹拟合监督下全 1，保留为兼容 loss_mask。
        valid_mask = inputs_on_device["batch_valid_mask"].bool()  # (B, T)

        # 前向：返回 8 值元组（x₀-prediction）
        (x0_pred_ng, x0_pred_mg,
         critic_pred, augment_pred,
         ng_x0_target, mg_x0_target,
         imagegoal_aux_pred, pixelgoal_aux_pred,
         ng_noisy_action, mg_noisy_action,
         ng_timesteps, mg_timesteps,
         ng_bridge_mu, mg_bridge_mu,
         ng_bridge_sigma, mg_bridge_sigma,
         ng_s_norm, mg_s_norm) = model(
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

        # ── x₀-MSE（全 24 个重采样监督点参与训练）─────────────────────
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
        lambda_eps = getattr(il_cfg, 'lambda_eps', 0.0) if il_cfg else 0.0

        # ── optional eps-consistency under the trajectory-time bridge ──────
        # Disabled by default because it is effectively an SNR-weighted x0 loss.
        def _eps_loss(x0_pred, x0_target, noisy_action, bridge_mu, bridge_sigma, s_norm):
            beta = s_norm.clamp(min=1e-6)
            sigma = bridge_sigma.clamp(min=1e-6)
            eps_target = (
                noisy_action - (1.0 - s_norm) * x0_target - s_norm * bridge_mu
            ) / (beta * sigma)
            eps_pred = (
                noisy_action - (1.0 - s_norm) * x0_pred - s_norm * bridge_mu
            ) / (beta * sigma)
            return ((eps_pred - eps_target).square() * mask).sum() / mask.sum().clamp(min=1)

        if lambda_eps > 0:
            L_eps = 0.5 * (
                _eps_loss(x0_pred_ng, ng_x0_target, ng_noisy_action, ng_bridge_mu, ng_bridge_sigma, ng_s_norm)
                + _eps_loss(x0_pred_mg, mg_x0_target, mg_noisy_action, mg_bridge_mu, mg_bridge_sigma, mg_s_norm)
            )
        else:
            L_eps = L_x0.new_tensor(0.0)

        action_loss = L_x0 + lambda_delta * L_delta + lambda_eps * L_eps

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
                "loss/L_eps":      L_eps.item(),
                "loss/ng_x0":      ng_x0_loss.item(),
                "loss/mg_x0":      mg_x0_loss.item(),
                "loss/critic":     critic_loss.item(),
                "loss/aux":        aux_loss.item(),
                "debug/grad_norm": grad_norm,
            }

            # 可视化：每 100 步执行一次真实去噪推理并写入 JSONL
            if self._log_step_count % 100 == 0:
                pred_traj = self._infer_pred_traj_bridgedp(model, inputs_on_device)
                # 反归一化 gt 和 prior（统一为物理坐标，米/弧度）
                gt_phys = self._denorm_batch(inputs_on_device["batch_labels"])
                prior_phys = self._denorm_batch(inputs_on_device["batch_prior"])
                # 导航目标点（反归一化）
                nav_goal_phys = self._denorm_batch(inputs_on_device["batch_pg"])
                # 障碍物点（已在 Dataset 中做过局部化，物理坐标）
                obstacle_pts = inputs.get("batch_obstacle_pts", None)
                self._write_traj_snapshot(
                    gt_phys,
                    pred_traj,
                    prior_phys,
                    inputs_on_device["batch_labels"],
                    inputs_on_device["batch_theta_g"],
                    batch_nav_goal=nav_goal_phys,
                    batch_obstacle_pts=obstacle_pts,
                    batch_valid_mask=inputs_on_device["batch_valid_mask"],
                )

        outputs = {
            'x0_pred_ng':  x0_pred_ng,
            'x0_pred_mg':  x0_pred_mg,
            'critic_pred': critic_pred,
            'loss':        loss,
            'action_loss': action_loss,
            'L_x0':        L_x0.item(),
            'L_delta':     L_delta.item(),
            'L_eps':       L_eps.item(),
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
        """对整个 batch 执行布朗桥去噪推理，返回绝对坐标预测轨迹 (B, T, 3)。

        每个 batch 样本使用自己的 point-goal / RGBD / prior 单独推理，避免
        可视化时把 batch[0] 的预测轨迹复用到其它样本上。
        """
        model_ref = model.module if hasattr(model, 'module') else model
        B = inputs_on_device["batch_labels"].shape[0]
        device = inputs_on_device["batch_labels"].device

        was_training = model_ref.training
        model_ref.eval()
        try:
            with torch.no_grad():
                # batch_pg already lives in Bridge-DP normalized action space
                # because Dataset/Trainer apply xy / 5.0 and theta / pi before
                # model input. Normalizing it again would shrink the visualized
                # target by another factor of 5 in xy.
                pg_n = inputs_on_device["batch_pg"]
                theta_g = inputs_on_device["batch_theta_g"]

                pointgoal_embed = model_ref.point_encoder(pg_n).unsqueeze(1)
                rgbd_embed = model_ref.rgbd_encoder(
                    inputs_on_device["batch_rgb"],
                    inputs_on_device["batch_depth"],
                )

                if getattr(model_ref, "use_prior_traj", False):
                    prior_tokens = model_ref.prior_encoder(inputs_on_device["batch_prior"])
                    vis_global = rgbd_embed.mean(dim=1)
                    gate = model_ref.visual_gate(vis_global)
                    gated_prior = gate * prior_tokens
                else:
                    gated_prior = torch.zeros(
                        B, model_ref.n_prior_tokens, model_ref.token_dim, device=device
                    )

                # 有序区间初始化（与推理函数一致）
                origin = torch.zeros_like(pg_n)  # (B, 3)
                naction = model_ref.bridge_scheduler.sample_initial_noise_ordered(
                    goal=pg_n,
                    origin=origin,
                    shape=(B, model_ref.predict_size, 3),
                    device=device,
                )
                theta_exp = theta_g

                model_ref.bridge_scheduler.set_timesteps(model_ref.num_inference_timesteps)
                for k in model_ref.bridge_scheduler.timesteps:
                    x0_pred = model_ref.predict_x0(
                        naction, k.to(device).unsqueeze(0),
                        pointgoal_embed, rgbd_embed, gated_prior,
                    )
                    naction = model_ref.bridge_scheduler.step_trajectory(
                        x0_pred, naction, k.to(device),
                        goal=pg_n,
                        theta_g=theta_exp,
                        origin=origin,
                        mode="pointgoal",
                    )

                pred_abs = model_ref._denormalize_action(naction)  # (B, T, 3)
        finally:
            if was_training:
                model_ref.train()

        return pred_abs

    def _denorm_batch(self, batch_tensor):
        """将归一化的 batch 轨迹/目标点反归一化为物理坐标（米/弧度）。

        归一化规则与 BridgeDP_Base_Dataset.__getitem__ 一致：
            xy / 5.0, θ / π → 反归一化：xy * 5.0, θ * π
        """
        t = batch_tensor.detach().cpu().clone()
        t[..., 0:2] = t[..., 0:2] * 5.0
        if t.shape[-1] >= 3:
            t[..., 2] = t[..., 2] * 3.14159
        return t

    def _write_traj_snapshot(self, gt_phys, pred_phys, prior_phys, gt_labels,
                              batch_theta_g=None, batch_nav_goal=None,
                              batch_obstacle_pts=None, batch_valid_mask=None):
        """将整个 batch 所有样本的轨迹追加写入 JSONL，供前端翻页可视化。

        所有轨迹数据统一使用物理坐标（米/弧度），确保坐标系一致。
        新增字段：nav_goal（导航目标点）、obstacle_pts（局部化障碍物点）、valid_mask。
        """
        if not hasattr(self, '_log_step_count'):
            return
        try:
            log_dir = Path(self.args.output_dir).parent / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            batch_file = log_dir / 'traj_batches.jsonl'

            B = gt_phys.shape[0]
            theta_g_list = (batch_theta_g.detach().cpu().view(-1).tolist()
                            if batch_theta_g is not None else [None] * B)
            nav_goal_list = (batch_nav_goal.detach().cpu()[:, 0:2].tolist()
                             if batch_nav_goal is not None else [None] * B)
            if batch_valid_mask is not None:
                gt_valid_mask = batch_valid_mask.detach().cpu().bool()
            else:
                gt_valid_mask = None
            # 障碍物点：list of tensor/ndarray，每个样本点数不同
            if batch_obstacle_pts is not None:
                obs_list = []
                for pts in batch_obstacle_pts:
                    if hasattr(pts, 'detach'):
                        obs_list.append(pts.detach().cpu().tolist())
                    elif hasattr(pts, 'tolist'):
                        obs_list.append(pts.tolist())
                    else:
                        obs_list.append([])
            else:
                obs_list = [[] for _ in range(B)]

            pred_valid_mask = None
            if hasattr(pred_phys, 'detach'):
                pred_tensor = pred_phys.detach()
                if pred_tensor.dim() == 3 and pred_tensor.shape[1] > 0:
                    step_diffs = torch.norm(pred_tensor[:, 1:, :2] - pred_tensor[:, :-1, :2], dim=-1)
                    pred_valid_mask = torch.cat(
                        [torch.ones((pred_tensor.shape[0], 1), device=pred_tensor.device, dtype=torch.bool),
                         step_diffs > 1e-4],
                        dim=1,
                    )
                    pred_valid_mask[:, :min(4, pred_tensor.shape[1])] = True
                    pred_valid_mask = pred_valid_mask.cpu()

            record = {
                "batch_idx": self._log_step_count,
                "step": self._log_step_count,
                "samples": [
                    {
                        "gt_traj":      gt_phys[i].tolist() if hasattr(gt_phys[i], 'tolist') else gt_phys[i],
                        "pred_traj":    pred_phys[i].detach().cpu().tolist() if hasattr(pred_phys[i], 'detach') else pred_phys[i].tolist(),
                        "prior_traj":   prior_phys[i].tolist() if hasattr(prior_phys[i], 'tolist') else prior_phys[i],
                        "theta_g":      theta_g_list[i],
                        "nav_goal":     nav_goal_list[i],
                        "obstacle_pts": obs_list[i],
                        "gt_valid_mask": (gt_valid_mask[i].tolist() if gt_valid_mask is not None else None),
                        "pred_valid_mask": (pred_valid_mask[i].tolist() if pred_valid_mask is not None else None),
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
