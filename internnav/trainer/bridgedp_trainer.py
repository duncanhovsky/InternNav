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
        il_cfg = self.config.il if hasattr(self.config, 'il') else None
        self.action_scale_xy = 5.0
        self.action_scale_theta = 3.14159
        self.enable_trajectory_normalization = (
            getattr(il_cfg, 'enable_trajectory_normalization', False) if il_cfg else False
        )
        self.trajectory_norm_target_distance = float(
            getattr(il_cfg, 'trajectory_norm_target_distance', 2.0) if il_cfg else 2.0
        )
        self.trajectory_norm_min_distance_m = float(
            getattr(il_cfg, 'trajectory_norm_min_distance_m', 0.10) if il_cfg else 0.10
        )
        self.trajectory_norm_eps = float(
            getattr(il_cfg, 'trajectory_norm_eps', 1e-6) if il_cfg else 1e-6
        )
        self.drop_short_trajectory_samples = (
            getattr(il_cfg, 'drop_short_trajectory_samples', True) if il_cfg else True
        )
        self.trajectory_resample_mode = (
            getattr(il_cfg, 'trajectory_resample_mode', 'arc_length') if il_cfg else 'arc_length'
        )
        self.trajectory_projection_monotonic_eps = float(
            getattr(il_cfg, 'trajectory_projection_monotonic_eps', 1e-4) if il_cfg else 1e-4
        )
        self.trajectory_projection_min_span = float(
            getattr(il_cfg, 'trajectory_projection_min_span', 0.80) if il_cfg else 0.80
        )
        self.trajectory_projection_flat_lateral_eps = float(
            getattr(il_cfg, 'trajectory_projection_flat_lateral_eps', 1e-3) if il_cfg else 1e-3
        )

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
        include_start: bool = False,
    ) -> torch.Tensor:
        """用 x/y 弧长参数在当前 GPU 上把一条原始轨迹重采样为指定数量的点。"""
        mode = getattr(self, "trajectory_resample_mode", "arc_length")
        if mode in ("projection", "hybrid_projection"):
            sampled = self._resample_one_trajectory_projection_gpu(
                traj,
                num_points,
                include_start=include_start,
                strict=(mode == "hybrid_projection"),
            )
            if sampled is not None:
                return sampled

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

        q_start = 0.0 if include_start else 1.0 / float(num_points)
        q = torch.linspace(q_start, 1.0, num_points, device=traj.device, dtype=traj.dtype)
        sampled = self._natural_cubic_eval(s, y, q)
        sampled[:, 2] = self._wrap_to_pi(sampled[:, 2])
        return sampled

    def _resample_one_trajectory_projection_gpu(
        self,
        traj: torch.Tensor,
        num_points: int,
        include_start: bool = False,
        strict: bool = True,
    ) -> torch.Tensor:
        """Resample by uniform projection progress on the origin-to-end chord."""
        traj = traj[:, :3]
        if traj.shape[0] == 0:
            return torch.zeros((num_points, 3), device=traj.device, dtype=traj.dtype)
        if traj.shape[0] == 1:
            return traj[-1:].expand(num_points, -1)

        theta_unwrapped = self._unwrap_angles(traj[:, 2])
        curve = torch.cat([traj[:, :2], theta_unwrapped.unsqueeze(-1)], dim=-1)
        end_xy = curve[-1, :2]
        chord_sq = torch.dot(end_xy, end_xy)
        if chord_sq <= 1e-8:
            return None

        origin = torch.zeros((1, 3), device=traj.device, dtype=traj.dtype)
        curve = torch.cat([origin, curve], dim=0)
        progress = torch.matmul(curve[:, :2], end_xy) / chord_sq
        progress = progress.clamp(0.0, 1.0)
        progress[-1] = 1.0

        diffs = progress[1:] - progress[:-1]
        monotonic_eps = getattr(self, "trajectory_projection_monotonic_eps", 1e-4)
        if strict and (diffs < -monotonic_eps).any():
            return None
        flat_lateral_eps = getattr(self, "trajectory_projection_flat_lateral_eps", 1e-3)
        flat_progress = diffs.abs() <= monotonic_eps
        segment_xy = torch.norm(curve[1:, :2] - curve[:-1, :2], dim=-1)
        if strict and (flat_progress & (segment_xy > flat_lateral_eps)).any():
            return None
        min_span = getattr(self, "trajectory_projection_min_span", 0.80)
        if strict and (progress.max() - progress.min()) < min_span:
            return None

        keep = torch.cat([
            torch.ones(1, device=traj.device, dtype=torch.bool),
            diffs > monotonic_eps,
        ])
        keep[-1] = True
        s = progress[keep]
        y = curve[keep]
        if s.shape[0] < 2:
            return None
        if (s[1:] - s[:-1] <= 0.0).any():
            unique_keep = torch.cat([
                torch.ones(1, device=traj.device, dtype=torch.bool),
                (s[1:] - s[:-1]) > monotonic_eps,
            ])
            if not unique_keep[-1]:
                unique_keep[-2] = False
                unique_keep[-1] = True
            s = s[unique_keep]
            y = y[unique_keep]
        if s.shape[0] < 2:
            return None
        y[-1] = curve[-1]

        q_start = 0.0 if include_start else 1.0 / float(num_points)
        q = torch.linspace(q_start, 1.0, num_points, device=traj.device, dtype=traj.dtype)
        sampled = self._natural_cubic_eval(s, y, q)
        sampled[:, 2] = self._wrap_to_pi(sampled[:, 2])
        return sampled

    def _resample_trajectories_gpu(
        self,
        raw_trajs: torch.Tensor,
        lengths: torch.Tensor,
        num_points: int,
        include_start: bool = False,
    ) -> torch.Tensor:
        """对 batch 内变长原始轨迹做 GPU 弧长样条重采样。"""
        samples = []
        for bid in range(raw_trajs.shape[0]):
            n = int(lengths[bid].item())
            n = max(1, min(n, raw_trajs.shape[1]))
            samples.append(
                self._resample_one_trajectory_gpu(
                    raw_trajs[bid, :n],
                    num_points,
                    include_start=include_start,
                )
            )
        return torch.stack(samples, dim=0)

    def _normalize_action_tensor(self, action: torch.Tensor) -> torch.Tensor:
        """与 BridgeDPNet/Dataset 一致的动作归一化。"""
        out = action.clone()
        out[..., 0:2] = out[..., 0:2] / self.action_scale_xy
        out[..., 2] = out[..., 2] / self.action_scale_theta
        return out

    def _trajectory_denorm_tensor(
        self,
        action: torch.Tensor,
        traj_distance_m: torch.Tensor,
    ) -> torch.Tensor:
        """将形状空间轨迹恢复到物理 xy 坐标，theta 从 /pi 恢复为弧度。"""
        out = action.clone()
        distances = traj_distance_m.to(device=out.device, dtype=out.dtype).view(-1)
        scale = distances / max(self.trajectory_norm_target_distance, self.trajectory_norm_eps)
        if out.dim() == 3:
            scale = scale.view(-1, 1, 1)
        elif out.dim() == 2:
            scale = scale.view(-1, 1)
        else:
            while scale.dim() < out[..., 0:2].dim():
                scale = scale.unsqueeze(-1)
        out[..., 0:2] = out[..., 0:2] * scale
        if out.shape[-1] >= 3:
            out[..., 2] = out[..., 2] * self.action_scale_theta
        return out

    def _trajectory_norm_logs(
        self,
        labels: torch.Tensor,
        traj_distance_m: torch.Tensor,
        sample_valid: torch.Tensor,
    ) -> dict:
        """收集样本级轨迹归一化的轻量健康检查指标。"""
        valid = sample_valid.bool()
        logs = {
            "traj_norm/target_distance": self.trajectory_norm_target_distance,
            "traj_norm/valid_count": int(valid.sum().item()),
            "traj_norm/skipped_short_count": int((~valid).sum().item()),
        }
        if valid.any():
            endpoint_dist_shape = torch.norm(labels[valid, -1, :2], dim=-1)
            denorm = self._trajectory_denorm_tensor(labels[valid], traj_distance_m[valid])
            safe_dist = traj_distance_m[valid].clamp(min=self.trajectory_norm_eps)
            roundtrip = denorm.clone()
            roundtrip[..., 0:2] = (
                roundtrip[..., 0:2]
                * self.trajectory_norm_target_distance
                / safe_dist.view(-1, 1, 1)
            )
            roundtrip[..., 2] = roundtrip[..., 2] / self.action_scale_theta
            roundtrip_err = torch.abs(roundtrip - labels[valid]).amax()
            logs.update({
                "traj_norm/mean_distance_m": traj_distance_m[valid].mean().item(),
                "traj_norm/mean_endpoint_dist_shape": endpoint_dist_shape.mean().item(),
                "traj_norm/roundtrip_xy_error_m": roundtrip_err.item(),
            })
        return logs

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

    def _distance_bucket_config(self):
        il_cfg = self.config.il if hasattr(self.config, 'il') else None
        edges = getattr(il_cfg, 'distance_bucket_edges', (0.10, 0.5, 0.8)) if il_cfg else (0.10, 0.5, 0.8)
        names = getattr(il_cfg, 'distance_bucket_names', ("static", "short", "mid", "long")) if il_cfg else ("static", "short", "mid", "long")
        edges = tuple(float(v) for v in edges)
        names = tuple(str(v) for v in names)
        if len(names) != len(edges) + 1:
            names = tuple([f"bucket_{i}" for i in range(len(edges) + 1)])
        return edges, names

    def _distance_bucket_mask(self, distances: torch.Tensor, bucket_idx: int, edges: tuple) -> torch.Tensor:
        if bucket_idx == 0:
            return distances < edges[0]
        if bucket_idx == len(edges):
            return distances >= edges[-1]
        return (distances >= edges[bucket_idx - 1]) & (distances < edges[bucket_idx])

    def _distance_bucket_name(self, distance_m: float) -> str:
        edges, names = self._distance_bucket_config()
        value = torch.tensor([distance_m], dtype=torch.float32)
        for idx, name in enumerate(names):
            if bool(self._distance_bucket_mask(value, idx, edges)[0].item()):
                return name
        return names[-1]

    def _distance_bucket_logs(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        traj_distance_m: torch.Tensor = None,
        sample_valid: torch.Tensor = None,
    ) -> dict:
        il_cfg = self.config.il if hasattr(self.config, 'il') else None
        if not (getattr(il_cfg, 'enable_distance_bucket_metrics', False) if il_cfg else False):
            return {}

        edges, names = self._distance_bucket_config()
        pred = pred.detach()
        target = target.detach()
        if self.enable_trajectory_normalization and traj_distance_m is not None:
            target_dist_m = traj_distance_m.detach().to(device=target.device, dtype=target.dtype)
            scale_m = (
                target_dist_m
                / max(self.trajectory_norm_target_distance, self.trajectory_norm_eps)
            )
        else:
            target_dist_m = torch.norm(target[:, -1, :2], dim=-1) * self.action_scale_xy
            scale_m = target.new_full((target.shape[0],), self.action_scale_xy)
        valid = (
            sample_valid.detach().to(device=target.device).bool()
            if sample_valid is not None
            else torch.ones(target.shape[0], device=target.device, dtype=torch.bool)
        )
        point_mse = (pred - target).square().mean(dim=(1, 2))
        terminal_err_m = (
            torch.norm(pred[:, -1, :2] - target[:, -1, :2], dim=-1) * scale_m
        )
        path_len_m = (
            torch.norm(pred[:, 1:, :2] - pred[:, :-1, :2], dim=-1).sum(dim=-1) * scale_m
        )

        logs = {}
        for idx, name in enumerate(names):
            bucket_mask = self._distance_bucket_mask(target_dist_m, idx, edges) & valid
            count = int(bucket_mask.sum().item())
            prefix = f"bucket/{name}"
            logs[f"{prefix}_count"] = count
            if count > 0:
                logs[f"{prefix}_L_x0"] = point_mse[bucket_mask].mean().item()
                logs[f"{prefix}_terminal_err_m"] = terminal_err_m[bucket_mask].mean().item()
                logs[f"{prefix}_path_len_m"] = path_len_m[bucket_mask].mean().item()
        return logs

    def _prepare_curve_supervision(self, inputs_on_device: dict, predict_size: int) -> None:
        """用 GPU 生成弧长样条监督标签，并原地覆盖 batch 字段。"""
        if "batch_raw_labels" not in inputs_on_device:
            B = inputs_on_device["batch_labels"].shape[0]
            device = inputs_on_device["batch_labels"].device
            inputs_on_device.setdefault(
                "batch_sample_valid",
                torch.ones(B, device=device, dtype=torch.bool),
            )
            return

        is_task_start = inputs_on_device.get(
            "batch_is_task_start",
            torch.zeros(
                inputs_on_device["batch_raw_labels"].shape[0],
                device=inputs_on_device["batch_raw_labels"].device,
                dtype=torch.bool,
            ),
        )

        raw_labels = inputs_on_device["batch_raw_labels"]
        raw_augments = inputs_on_device["batch_raw_augments"]
        lengths = inputs_on_device["batch_raw_lengths"]
        B = raw_labels.shape[0]
        batch_idx = torch.arange(B, device=raw_labels.device)
        end_idx = (lengths.long().clamp(min=1, max=raw_labels.shape[1]) - 1)
        end_xy = raw_labels[batch_idx, end_idx, :2]
        traj_distance_m = torch.norm(end_xy, dim=-1)

        if self.enable_trajectory_normalization:
            sample_valid = traj_distance_m >= self.trajectory_norm_min_distance_m
            if not self.drop_short_trajectory_samples:
                sample_valid = torch.ones_like(sample_valid, dtype=torch.bool)
            safe_dist = traj_distance_m.clamp(min=self.trajectory_norm_eps)
            scale_to_shape = (
                self.trajectory_norm_target_distance
                / safe_dist
            ) * sample_valid.to(dtype=raw_labels.dtype)

            # 只缩放 xy；theta 仍以弧度参与 unwrap/spline，重采样后再除以 pi。
            labels_input = raw_labels.clone()
            augments_input = raw_augments.clone()
            labels_input[..., 0:2] = labels_input[..., 0:2] * scale_to_shape.view(B, 1, 1)
            augments_input[..., 0:2] = augments_input[..., 0:2] * scale_to_shape.view(B, 1, 1)
            labels_input[~sample_valid] = 0.0
            augments_input[~sample_valid] = 0.0

            labels = self._resample_trajectories_gpu(labels_input, lengths, predict_size)
            augments = self._resample_trajectories_gpu(augments_input, lengths, predict_size)
            labels[..., 2] = labels[..., 2] / self.action_scale_theta
            augments[..., 2] = augments[..., 2] / self.action_scale_theta
            labels[~sample_valid] = 0.0
            augments[~sample_valid] = 0.0
            # NoGoal 没有可用于反归一化的真实目标距离，保持 legacy xy/5 训练空间。
            labels_phys = self._resample_trajectories_gpu(raw_labels, lengths, predict_size)
            nogoal_labels = self._normalize_action_tensor(labels_phys)
            nogoal_labels[~sample_valid] = 0.0

            inputs_on_device["batch_pg"] = labels[:, -1, :].clone()
            inputs_on_device["batch_traj_distance_m"] = traj_distance_m
            inputs_on_device["batch_traj_denorm_scale"] = (
                traj_distance_m / max(self.trajectory_norm_target_distance, self.trajectory_norm_eps)
            )
            inputs_on_device["batch_sample_valid"] = sample_valid
            inputs_on_device["batch_traj_norm_target_distance"] = labels.new_full(
                (B,), self.trajectory_norm_target_distance
            )
        else:
            labels_phys = self._resample_trajectories_gpu(raw_labels, lengths, predict_size)
            augments_phys = self._resample_trajectories_gpu(raw_augments, lengths, predict_size)
            labels = self._normalize_action_tensor(labels_phys)
            augments = self._normalize_action_tensor(augments_phys)
            sample_valid = torch.ones(B, device=labels.device, dtype=torch.bool)
            nogoal_labels = labels
            inputs_on_device["batch_sample_valid"] = sample_valid
            inputs_on_device["batch_traj_distance_m"] = traj_distance_m
            inputs_on_device["batch_traj_denorm_scale"] = torch.full_like(
                traj_distance_m, self.action_scale_xy
            )

        inputs_on_device["batch_labels"] = labels
        inputs_on_device["batch_augments"] = augments
        inputs_on_device["batch_nogoal_labels"] = nogoal_labels
        inputs_on_device["batch_prior"] = self._generate_prior_trajectory_gpu(labels, is_task_start)
        inputs_on_device["batch_valid_mask"] = torch.ones(
            labels.shape[:2], device=labels.device, dtype=torch.float32
        ) * sample_valid.view(-1, 1).to(dtype=torch.float32)

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
        sample_valid = inputs_on_device.get(
            "batch_sample_valid",
            torch.ones(valid_mask.shape[0], device=valid_mask.device, dtype=torch.bool),
        ).bool()

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
            inputs_on_device.get("batch_nogoal_labels"),
            inputs_on_device.get("batch_traj_distance_m"),
        )

        # ── x₀-MSE（全 24 个重采样监督点参与训练）─────────────────────
        mask = (valid_mask & sample_valid.view(-1, 1)).unsqueeze(-1).float()  # (B, T, 1)
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
            if getattr(getattr(model_ref, "bridge_scheduler", None), "bridge_scale_invariant_sigma", False):
                raise ValueError(
                    "lambda_eps must stay 0 when bridge_scale_invariant_sigma=True; "
                    "eps consistency needs tangent/normal local-coordinate handling."
                )
            L_eps = 0.5 * (
                _eps_loss(x0_pred_ng, ng_x0_target, ng_noisy_action, ng_bridge_mu, ng_bridge_sigma, ng_s_norm)
                + _eps_loss(x0_pred_mg, mg_x0_target, mg_noisy_action, mg_bridge_mu, mg_bridge_sigma, mg_s_norm)
            )
        else:
            L_eps = L_x0.new_tensor(0.0)

        action_loss = L_x0 + lambda_delta * L_delta + lambda_eps * L_eps

        # ── Critic 损失（与 NavDP 一致）──────────────────────────────
        sample_weight = sample_valid.to(dtype=critic_pred.dtype)
        sample_denom = sample_weight.sum().clamp(min=1)
        critic_loss = (
            ((critic_pred - batch_label_critic).square() * sample_weight).sum() / sample_denom
            + ((augment_pred - batch_augment_critic).square() * sample_weight).sum() / sample_denom
        )

        # ── 辅助损失（与 NavDP 一致）──────────────────────────────────
        aux_point = (
            (inputs_on_device["batch_pg"] - imagegoal_aux_pred).square().mean(dim=-1)
            + (inputs_on_device["batch_pg"] - pixelgoal_aux_pred).square().mean(dim=-1)
        )
        aux_loss = (
            0.5 * (aux_point * sample_weight).sum() / sample_denom
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
            pred_avg = (
                x0_pred_mg.detach()
                if self.enable_trajectory_normalization
                else 0.5 * (x0_pred_ng.detach() + x0_pred_mg.detach())
            )
            bucket_target = mg_x0_target.detach() if self.enable_trajectory_normalization else ng_x0_target.detach()
            self._monitor_logs.update(
                self._distance_bucket_logs(
                    pred_avg,
                    bucket_target,
                    traj_distance_m=inputs_on_device.get("batch_traj_distance_m"),
                    sample_valid=sample_valid,
                )
            )
            if self.enable_trajectory_normalization:
                self._monitor_logs.update(
                    self._trajectory_norm_logs(
                        inputs_on_device["batch_labels"].detach(),
                        inputs_on_device["batch_traj_distance_m"].detach(),
                        sample_valid,
                    )
                )

            # 可视化：每 100 步执行一次真实去噪推理并写入 JSONL
            if self._log_step_count % 100 == 0:
                pred_traj = self._infer_pred_traj_bridgedp(model, inputs_on_device)
                # 反归一化 gt 和 prior（统一为物理坐标，米/弧度）
                gt_phys = self._denorm_batch(
                    inputs_on_device["batch_labels"],
                    inputs_on_device.get("batch_traj_distance_m"),
                )
                prior_phys = self._denorm_batch(
                    inputs_on_device["batch_prior"],
                    inputs_on_device.get("batch_traj_distance_m"),
                )
                # 导航目标点（反归一化）
                nav_goal_phys = self._denorm_batch(
                    inputs_on_device["batch_pg"],
                    inputs_on_device.get("batch_traj_distance_m"),
                )
                # 障碍物点（已在 Dataset 中做过局部化，物理坐标）
                obstacle_pts = inputs.get("batch_obstacle_pts", None)
                gt_spline_traj = None
                if "batch_raw_labels" in inputs_on_device:
                    spline_points = max(int(inputs_on_device["batch_labels"].shape[1]) * 4, 2)
                    gt_spline_traj = self._resample_trajectories_gpu(
                        inputs_on_device["batch_raw_labels"],
                        inputs_on_device["batch_raw_lengths"],
                        spline_points,
                        include_start=True,
                    )
                self._write_traj_snapshot(
                    gt_phys,
                    pred_traj,
                    prior_phys,
                    inputs_on_device["batch_labels"],
                    inputs_on_device["batch_theta_g"],
                    batch_nav_goal=nav_goal_phys,
                    batch_obstacle_pts=obstacle_pts,
                    batch_valid_mask=inputs_on_device["batch_valid_mask"],
                    batch_gt_spline_traj=gt_spline_traj,
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
                # batch_pg already lives in the active Bridge-DP model space.
                # In trajectory-normalized mode that is the fixed-distance shape
                # space; in legacy mode it is xy / 5.0 and theta / pi.
                pg_n = inputs_on_device["batch_pg"]
                theta_g = inputs_on_device["batch_theta_g"]

                pointgoal_embed = model_ref.point_encoder(pg_n).unsqueeze(1)
                scale_embed = model_ref._build_scale_token(
                    inputs_on_device["batch_traj_distance_m"],
                    like_token=pointgoal_embed,
                )
                rgbd_embed = model_ref.rgbd_encoder(
                    inputs_on_device["batch_rgb"],
                    inputs_on_device["batch_depth"],
                )
                rgbd_embed = model_ref._apply_scale_rgbd_film(
                    rgbd_embed,
                    inputs_on_device["batch_traj_distance_m"],
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
                        pointgoal_embed, rgbd_embed, gated_prior, scale_embed,
                    )
                    naction = model_ref.bridge_scheduler.step_trajectory(
                        x0_pred, naction, k.to(device),
                        goal=pg_n,
                        theta_g=theta_exp,
                        origin=origin,
                        mode="pointgoal",
                        eta=getattr(model_ref, "inference_eta", 0.0),
                    )

                if self.enable_trajectory_normalization:
                    pred_abs = self._trajectory_denorm_tensor(
                        naction,
                        inputs_on_device["batch_traj_distance_m"],
                    )
                else:
                    pred_abs = model_ref._denormalize_action(naction)  # (B, T, 3)
        finally:
            if was_training:
                model_ref.train()

        return pred_abs

    def _denorm_batch(self, batch_tensor, traj_distance_m=None):
        """将归一化的 batch 轨迹/目标点反归一化为物理坐标（米/弧度）。

        legacy 模式：
            xy / 5.0, θ / π → xy * 5.0, θ * π
        trajectory-normalized 模式：
            xy * R / d_m, θ / π → xy * d_m / R, θ * π
        """
        if self.enable_trajectory_normalization and traj_distance_m is not None:
            t = self._trajectory_denorm_tensor(
                batch_tensor.detach(),
                traj_distance_m.detach(),
            ).cpu()
        else:
            t = batch_tensor.detach().cpu().clone()
            t[..., 0:2] = t[..., 0:2] * self.action_scale_xy
            if t.shape[-1] >= 3:
                t[..., 2] = t[..., 2] * self.action_scale_theta
        return t

    def _write_traj_snapshot(self, gt_phys, pred_phys, prior_phys, gt_labels,
                              batch_theta_g=None, batch_nav_goal=None,
                              batch_obstacle_pts=None, batch_valid_mask=None,
                              batch_gt_spline_traj=None):
        """将整个 batch 所有样本的轨迹追加写入 JSONL，供前端翻页可视化。

        所有轨迹数据统一使用物理坐标（米/弧度），确保坐标系一致。
        新增字段：nav_goal（导航目标点）、obstacle_pts（局部化障碍物点）、valid_mask、
        gt_spline_traj（原始 GT 轨迹的三次样条重采样曲线，未做端点强制覆盖）。
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
            gt_spline_traj = (batch_gt_spline_traj.detach().cpu()
                              if batch_gt_spline_traj is not None else None)
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
                        "gt_resample_points": gt_phys[i].tolist() if hasattr(gt_phys[i], 'tolist') else gt_phys[i],
                        "pred_traj":    pred_phys[i].detach().cpu().tolist() if hasattr(pred_phys[i], 'detach') else pred_phys[i].tolist(),
                        "prior_traj":   prior_phys[i].tolist() if hasattr(prior_phys[i], 'tolist') else prior_phys[i],
                        "theta_g":      theta_g_list[i],
                        "nav_goal":     nav_goal_list[i],
                        "traj_length_m": float(torch.norm(gt_phys[i, -1, :2]).item()),
                        "length_bucket": self._distance_bucket_name(float(torch.norm(gt_phys[i, -1, :2]).item())),
                        "obstacle_pts": obs_list[i],
                        "gt_spline_traj": (gt_spline_traj[i].tolist() if gt_spline_traj is not None else None),
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
        model_state = state_dict if state_dict is not None else model_to_save.state_dict()
        torch.save(model_state, os.path.join(output_dir, "bridgedp.ckpt"))
        torch.save(model_state, os.path.join(output_dir, "pytorch_model.bin"))
        print(f"Saving model to {output_dir} (is DDP: {hasattr(self.model, 'module')})")
