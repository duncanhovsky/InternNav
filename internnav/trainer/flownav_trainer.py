import os
import time
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler

from internnav.model.encoder.dyn_module import (
    DynModuleConfig,
    FlowNavDynamicsRuntime,
    PointCloudFrame,
    PoseFrame,
)
from internnav.trainer.base import BaseTrainer


class FlowNavTrainer(BaseTrainer):
    """FlowNav 模型训练器。

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
        self.start_time = time.time()

        if hasattr(self.model, "module"):
            self.model_device = self.model.module.device
        else:
            self.model_device = self.model.device

        # dyn_module 配置：静态数据集缺失 dynamic_voxels 时，在线构造输入体素。
        self.dyn_cfg = self._build_dyn_module_config()

        print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] Model device: {self.model_device}")

    def _build_dyn_module_config(self) -> DynModuleConfig:
        """构建 dyn_module 配置。

        设计说明：
        1) 默认值可直接跑通，保证静态数据集训练不中断。
        2) 若配置中提供 dyn_module 子配置，则优先使用用户配置。
        """
        cfg = DynModuleConfig()
        cfg.device = str(self.model_device)

        dyn_cfg = getattr(getattr(self.config, "il", None), "dyn_module", None)
        if dyn_cfg is not None:
            for key, value in vars(dyn_cfg).items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
        return cfg

    @staticmethod
    def _depth_to_point_cloud(depth_m: np.ndarray, intrinsic: np.ndarray, stride: int = 2) -> np.ndarray:
        """把单帧深度图反投影为点云。

        说明：
        1) 该函数服务于 trainer 兜底分支，仅在缺失 dynamic_voxels 时调用。
        2) 采用 stride 下采样控制点数，避免训练时 CPU 反投影过慢。
        """
        d = np.asarray(depth_m, dtype=np.float32)
        if d.ndim == 3:
            d = d[..., 0]

        fx = float(intrinsic[0, 0])
        fy = float(intrinsic[1, 1])
        cx = float(intrinsic[0, 2])
        cy = float(intrinsic[1, 2])

        h, w = d.shape
        v = np.arange(0, h, stride, dtype=np.float32)
        u = np.arange(0, w, stride, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        zz = d[::stride, ::stride]

        valid = np.isfinite(zz) & (zz > 0.05)
        if valid.sum() == 0:
            return np.zeros((0, 3), dtype=np.float32)

        z = zz[valid]
        x = (uu[valid] - cx) * z / max(fx, 1e-6)
        y = (vv[valid] - cy) * z / max(fy, 1e-6)
        return np.stack([x, y, z], axis=-1).astype(np.float32)

    @staticmethod
    def _apply_depth_preprocess_to_intrinsic(intrinsic: np.ndarray, preprocess_meta: Optional[np.ndarray]) -> np.ndarray:
        """将深度预处理链路同步到相机内参，保证反投影几何一致。

        preprocess_meta 字段顺序约定：
        [scale_x, scale_y, pad_left, pad_top,
         crop_x0, crop_y0, crop_w, crop_h,
         final_scale_x, final_scale_y]
        """
        k = np.asarray(intrinsic, dtype=np.float32).copy()
        if preprocess_meta is None:
            return k

        m = np.asarray(preprocess_meta, dtype=np.float32).reshape(-1)
        if m.shape[0] < 10:
            return k

        scale_x, scale_y = float(m[0]), float(m[1])
        pad_left, pad_top = float(m[2]), float(m[3])
        crop_x0, crop_y0 = float(m[4]), float(m[5])
        final_scale_x, final_scale_y = float(m[8]), float(m[9])

        # 为什么这样改：深度经历 resize/pad/crop 后，像素坐标系发生变化；
        # 反投影若仍使用原始内参，会让点云坐标产生系统偏差。
        k[0, 0] = k[0, 0] * scale_x
        k[1, 1] = k[1, 1] * scale_y
        k[0, 2] = k[0, 2] * scale_x + pad_left
        k[1, 2] = k[1, 2] * scale_y + pad_top

        # 将主点映射到裁剪后的局部坐标系。
        k[0, 2] = k[0, 2] - crop_x0
        k[1, 2] = k[1, 2] - crop_y0

        # 将裁剪图再缩放到网络输入分辨率时，同步缩放焦距与主点。
        k[0, 0] = k[0, 0] * final_scale_x
        k[1, 1] = k[1, 1] * final_scale_y
        k[0, 2] = k[0, 2] * final_scale_x
        k[1, 2] = k[1, 2] * final_scale_y
        return k

    def _build_dynamic_voxels_from_dyn_module(self, inputs, model_device: torch.device) -> torch.Tensor:
        """使用 dyn_module 在线构建 dynamic_voxels。

        输入要求（静态数据集新增字段）：
        - batch_depth_raw_m_hist: (B, T, H, W, 1)
        - batch_pose_world_hist: (B, T, 4, 4)
        - batch_timestamp_hist_s: (B, T)
        - batch_camera_intrinsic: (B, 3, 3)

        返回：
            torch.Tensor: (B, T_future, 4, X, Y, Z)
        """
        required_keys = [
            "batch_depth_raw_m_hist",
            "batch_pose_world_hist",
            "batch_timestamp_hist_s",
            "batch_camera_intrinsic",
        ]
        for key in required_keys:
            if key not in inputs:
                raise KeyError(
                    f"Missing key '{key}' for dyn_module fallback. "
                    "Please ensure static FlowNav dataset exposes dyn-module auxiliary fields."
                )

        depth_hist = inputs["batch_depth_raw_m_hist"].detach().cpu().numpy()
        pose_hist = inputs["batch_pose_world_hist"].detach().cpu().numpy()
        ts_hist = inputs["batch_timestamp_hist_s"].detach().cpu().numpy()
        intr_hist = inputs["batch_camera_intrinsic"].detach().cpu().numpy()
        preprocess_meta_hist = None
        if "batch_depth_preprocess_meta" in inputs:
            preprocess_meta_hist = inputs["batch_depth_preprocess_meta"].detach().cpu().numpy()

        batch_vox = []
        for b in range(depth_hist.shape[0]):
            runtime = FlowNavDynamicsRuntime(self.dyn_cfg)
            for t in range(depth_hist.shape[1]):
                meta_bt = None if preprocess_meta_hist is None else preprocess_meta_hist[b, t]
                # 为什么这样改：确保“深度看到的感受野”和“点云重建时使用的相机模型”一致，
                # 尤其在启用裁剪/填充时避免几何偏移。
                intrinsic_bt = self._apply_depth_preprocess_to_intrinsic(intr_hist[b], meta_bt)
                pts = self._depth_to_point_cloud(depth_hist[b, t], intrinsic_bt)
                point_frame = PointCloudFrame(points_xyz=pts, stamp=float(ts_hist[b, t]))
                pose_frame = PoseFrame(t_world_ego=np.asarray(pose_hist[b, t], dtype=np.float32), stamp=float(ts_hist[b, t]))
                runtime.ingest(point_frame, pose_frame)

            vox_b = runtime.get_dynamic_voxels(batch_size=1, device=model_device, dtype=torch.float32)
            batch_vox.append(vox_b)

        return torch.cat(batch_vox, dim=0)

    def _resolve_dynamic_voxels(self, inputs, inputs_on_device, model_device: torch.device) -> torch.Tensor:
        """统一解析 dynamic_voxels。

        优先级：
        1) 若 batch 已包含 dynamic_voxels，直接使用（动态数据集路径）。
        1.5) 若同时提供 `batch_dynamic_voxel_valid_mask`，则仅对无效样本在线重建。
        2) 否则调用 dyn_module 在线构建（静态数据集路径）。
        3) 在线构建失败时回退零体素，保证训练主流程不中断。
        """
        if "batch_dynamic_voxels" in inputs_on_device:
            vox = inputs_on_device["batch_dynamic_voxels"]
            if "batch_dynamic_voxel_valid_mask" not in inputs_on_device:
                return vox

            valid_mask = inputs_on_device["batch_dynamic_voxel_valid_mask"].to(torch.bool)
            if bool(valid_mask.all()):
                return vox

            try:
                # 为什么这样改：混合训练时一个 batch 里会同时出现“有体素监督”的动态样本
                # 和“无体素监督”的静态样本。这里按样本掩码只重建缺失项，保留已有监督。
                fallback_vox = self._build_dynamic_voxels_from_dyn_module(inputs, model_device)
                merged = vox.clone()
                invalid_mask = ~valid_mask
                merged[invalid_mask] = fallback_vox[invalid_mask]
                return merged
            except Exception as exc:
                print(f"[FlowNavTrainer] mixed-batch dyn fallback failed: {exc}")
                grid = self.dyn_cfg.grid_size_xyz.tolist()
                merged = vox.clone()
                invalid_mask = ~valid_mask
                merged[invalid_mask] = torch.zeros(
                    (int(invalid_mask.sum().item()), self.dyn_cfg.horizon_frames, 4, grid[0], grid[1], grid[2]),
                    dtype=torch.float32,
                    device=model_device,
                )
                return merged

        try:
            return self._build_dynamic_voxels_from_dyn_module(inputs, model_device)
        except Exception as exc:
            # 兜底策略：生成零体素，避免训练因单批次数据异常直接中断。
            print(f"[FlowNavTrainer] dyn_module fallback failed: {exc}")
            grid = self.dyn_cfg.grid_size_xyz.tolist()
            bsz = int(inputs_on_device["batch_depth"].shape[0])
            return torch.zeros(
                (bsz, self.dyn_cfg.horizon_frames, 4, grid[0], grid[1], grid[2]),
                dtype=torch.float32,
                device=model_device,
            )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """执行一次前向并计算总损失。"""
        model_device = next(model.parameters()).device

        # 通用字段搬运：保持与 NavDP 相同框架，统一把 Tensor 移到模型设备。
        inputs_on_device = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                inputs_on_device[key] = value.to(model_device, non_blocking=True)
            else:
                inputs_on_device[key] = value

        # 统一解析 dynamic_voxels：兼容静态/动态两类数据集。
        dynamic_voxels = self._resolve_dynamic_voxels(inputs, inputs_on_device, model_device)

        batch_label_critic = inputs_on_device["batch_label_critic"]
        batch_augment_critic = inputs_on_device["batch_augment_critic"]

        (
            pred_ng,
            pred_mg,
            critic_pred,
            augment_pred,
            ng_noise,
            mg_noise,
            imagegoal_aux_pred,
            pixelgoal_aux_pred,
        ) = model(
            inputs_on_device["batch_pg"],
            inputs_on_device["batch_ig"],
            inputs_on_device["batch_tg"],
            inputs_on_device["batch_rgb"],
            inputs_on_device["batch_depth"],
            dynamic_voxels,
            inputs_on_device["batch_labels"],
            inputs_on_device["batch_augments"],
        )

        ng_action_loss = (pred_ng - ng_noise).square().mean()
        mg_action_loss = (pred_mg - mg_noise).square().mean()
        action_loss = 0.5 * mg_action_loss + 0.5 * ng_action_loss

        aux_loss = (
            0.5 * (inputs_on_device["batch_pg"] - imagegoal_aux_pred).square().mean()
            + 0.5 * (inputs_on_device["batch_pg"] - pixelgoal_aux_pred).square().mean()
        )

        critic_loss = (critic_pred - batch_label_critic).square().mean() + (
            augment_pred - batch_augment_critic
        ).square().mean()

        # 总损失配比：动作 0.8 + critic 0.2 + 辅助 0.5。
        # 与 NavDP 保持一致的损失组合，减少跨模型调参成本。
        loss = 0.8 * action_loss + 0.2 * critic_loss + 0.5 * aux_loss

        outputs = {
            "pred_ng": pred_ng,
            "pred_mg": pred_mg,
            "critic_pred": critic_pred,
            "augment_pred": augment_pred,
            "noise": [ng_noise, mg_noise],
            "loss": loss,
            "ng_action_loss": ng_action_loss,
            "mg_action_loss": mg_action_loss,
            "aux_loss": aux_loss,
            "critic_loss": critic_loss,
        }
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """创建并返回优化器。"""
        rank = dist.get_rank() if dist.is_initialized() else 0
        try:
            lr = self.config.il.lr
            if rank == 0:
                print(f"[Rank 0] Using learning rate: {lr}")
        except AttributeError:
            lr = 1e-4
            if rank == 0:
                print(f"[Rank 0] Warning: Using default learning rate: {lr}")

        model_for_optim = self.model.module if hasattr(self.model, "module") else self.model
        optimizer = torch.optim.Adam(model_for_optim.parameters(), lr=lr)

        if rank == 0:
            print(f"[Rank 0] Optimizer created with {len(optimizer.param_groups)} param groups")
            total_params = sum(p.numel() for p in model_for_optim.parameters() if p.requires_grad)
            print(f"[Rank 0] Total trainable parameters: {total_params:,}")
        return optimizer

    def create_scheduler(self, optimizer, num_training_steps: int):
        """创建学习率调度器。"""
        return torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=10000)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """覆盖父类方法，统一控制优化器与调度器创建顺序。"""
        print("\n=== create optimizer and scheduler ===")
        self.optimizer = self.create_optimizer()
        self.lr_scheduler = self.create_scheduler(self.optimizer, num_training_steps)
        return self.optimizer, self.lr_scheduler

    def get_train_dataloader(self):
        """构建训练 DataLoader（支持 DDP 切分）。"""
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        sampler = None
        # 为什么这样改：flownav_mix 需要按静态/动态权重采样。
        # 单机时可直接使用 WeightedRandomSampler；DDP 多机/多卡下先保持 DistributedSampler 兼容，
        # 避免各 rank 权重采样重复带来的统计偏差（后续可升级为分布式加权采样器）。
        if (
            world_size == 1
            and getattr(self.train_dataset, "use_weighted_sampler", False)
            and hasattr(self.train_dataset, "sample_weights")
        ):
            sampler = WeightedRandomSampler(
                weights=self.train_dataset.sample_weights,
                num_samples=len(self.train_dataset),
                replacement=True,
            )
        else:
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
        return loader

    def save_model(self, output_dir, state_dict=None, **kwargs):
        """保存模型到指定目录。"""
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        os.makedirs(output_dir, exist_ok=True)
        ckpt_path = os.path.join(output_dir, "flownav.ckpt")
        torch.save(model_to_save.state_dict(), ckpt_path)
        print(f"Saving model to {ckpt_path} (is DDP: {hasattr(self.model, 'module')})")