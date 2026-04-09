from __future__ import annotations
"""OpenSceneFlow 推理适配层。

该模块负责将运行时采集到的历史点云与位姿，转换为 OpenSceneFlow
可接受的 batch 字典，并将模型输出重新映射回当前帧点云索引。

实现原则：
1. 尽量复用 OpenSceneFlow 既有输入语义（pc0/pc1/pch*/pose*）。
2. 依赖异常时可回退为零流，保证主链路持续可用。
3. 对外暴露统一接口 ``predict_flow``，屏蔽模型细节差异。
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from .config import DynModuleConfig
from .types import FlowInferenceResult, PointCloudFrame, PoseFrame


def _to_tensor_points(points_xyz: np.ndarray, device: torch.device) -> torch.Tensor:
    """将点云数组转换为模型输入张量。

    Args:
        points_xyz: 点云坐标，形状 (N, 3)。
        device: 目标设备。

    Returns:
        torch.Tensor: 形状 (N, 3) 的 float32 张量。
    """
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points_xyz must be (N,3), got {pts.shape}")
    return torch.from_numpy(pts).to(device)


def _to_tensor_pose(t_world_ego: np.ndarray, device: torch.device) -> torch.Tensor:
    """将 4x4 位姿矩阵转换为模型输入张量。"""
    pose = np.asarray(t_world_ego, dtype=np.float32)
    if pose.shape != (4, 4):
        raise ValueError(f"Pose must be (4,4), got {pose.shape}")
    return torch.from_numpy(pose).to(device)


class OpenSceneFlowAdapter:
    """OpenSceneFlow 运行时适配器。

    支持 LiteFlow/DeltaFlow 两种后端，统一输出当前帧对齐的场景流结果。
    """

    def __init__(self, cfg: DynModuleConfig):
        """初始化适配器并尝试加载场景流模型。"""
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.repo_root = Path(__file__).resolve().parents[2]
        self.opensf_root = self.repo_root / "OpenSceneFlow"

        self.model: Optional[torch.nn.Module] = None
        self.model_name = cfg.model_name.lower()
        self.model_loaded = False
        self.using_fallback = False

        self._try_load_model()

    def _try_load_model(self) -> None:
        """尽力加载模型；失败时切换到安全回退模式。

        回退模式下 ``predict_flow`` 会返回零流，不抛异常中断主流程。
        """
        if not self.opensf_root.exists():
            self.using_fallback = True
            return

        # 将 OpenSceneFlow 根目录加入路径，便于导入其 src.models.*。
        if str(self.opensf_root) not in sys.path:
            sys.path.append(str(self.opensf_root))

        try:
            if self.model_name == "liteflow":
                from src.models.liteflow import LiteFlow

                self.model = LiteFlow(
                    voxel_size=self.cfg.voxel_size,
                    point_cloud_range=self.cfg.point_cloud_range,
                    grid_feature_size=self.cfg.grid_size_xyz.tolist(),
                    num_frames=self.cfg.history_frames,
                    decay_factor=0.9,
                )
            elif self.model_name == "deltaflow":
                from src.models.deltaflow import DeltaFlow

                self.model = DeltaFlow(
                    voxel_size=self.cfg.voxel_size,
                    point_cloud_range=self.cfg.point_cloud_range,
                    grid_feature_size=self.cfg.grid_size_xyz.tolist(),
                    num_frames=self.cfg.history_frames,
                )
            else:
                raise ValueError(f"Unsupported model_name: {self.model_name}")

            if self.cfg.checkpoint_path:
                ckpt = self.cfg.checkpoint_path
                if os.path.exists(ckpt):
                    try:
                        if hasattr(self.model, "load_from_checkpoint"):
                            # OpenSceneFlow 风格 checkpoint 加载接口。
                            self.model.load_from_checkpoint(ckpt)
                    except Exception:
                        # checkpoint 格式不匹配时保持运行，交由回退/默认权重兜底。
                        pass

            self.model = self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            self.using_fallback = False
        except Exception:
            # 任何导入或构建异常均进入回退模式。
            self.model = None
            self.model_loaded = False
            self.using_fallback = True

    def _build_batch(
        self,
        point_history: List[PointCloudFrame],
        pose_history: List[PoseFrame],
    ) -> Tuple[dict, np.ndarray]:
        """构建 OpenSceneFlow 所需 batch 字典。

        约定：
        - 最新帧作为 ``pc0``。
        - 前一帧作为 ``pc1``。
        - 更早历史帧映射到 ``pch{i}``。

        Returns:
            tuple:
                batch: OpenSceneFlow 推理输入。
                np.ndarray: 当前帧原始点云，用于输出回填。
        """
        if len(point_history) < 2:
            raise ValueError("Need at least 2 point cloud frames for scene-flow inference")
        if len(point_history) != len(pose_history):
            raise ValueError("point_history and pose_history must have same length")

        # 最新帧 -> pc0；上一帧 -> pc1；其余历史帧 -> pch*。
        pts = point_history
        poses = pose_history

        current = pts[-1]
        prev = pts[-2]
        current_pose = poses[-1]
        prev_pose = poses[-2]

        batch = {
            "pc0": _to_tensor_points(current.points_xyz, self.device).unsqueeze(0),
            "pc1": _to_tensor_points(prev.points_xyz, self.device).unsqueeze(0),
            "pose0": [_to_tensor_pose(current_pose.t_world_ego, self.device)],
            "pose1": [_to_tensor_pose(prev_pose.t_world_ego, self.device)],
        }

        # pch1 表示比 pc1 更早的一帧。
        max_hist = min(self.cfg.history_frames - 2, len(pts) - 2)
        for i in range(1, max_hist + 1):
            hist_idx = -2 - i
            batch[f"pch{i}"] = _to_tensor_points(pts[hist_idx].points_xyz, self.device).unsqueeze(0)
            batch[f"poseh{i}"] = [_to_tensor_pose(poses[hist_idx].t_world_ego, self.device)]

        return batch, np.asarray(current.points_xyz, dtype=np.float32)

    def _fallback_flow(
        self,
        point_history: List[PointCloudFrame],
    ) -> FlowInferenceResult:
        """生成零流回退结果。"""
        current = point_history[-1]
        points_xyz = np.asarray(current.points_xyz, dtype=np.float32)
        flow_xyz = np.zeros_like(points_xyz, dtype=np.float32)
        valid_mask = np.ones((points_xyz.shape[0],), dtype=bool)
        return FlowInferenceResult(
            points_xyz=points_xyz,
            flow_xyz=flow_xyz,
            valid_mask=valid_mask,
            stamp=current.stamp,
        )

    @torch.no_grad()
    def predict_flow(
        self,
        point_history: List[PointCloudFrame],
        pose_history: List[PoseFrame],
    ) -> FlowInferenceResult:
        """执行场景流推理，并回填到当前帧完整点集。

        OpenSceneFlow 输出通常仅包含有效点子集；本函数会根据
        ``pc0_valid_point_idxes`` 将结果散射回完整点云长度。
        """
        if self.model is None or self.using_fallback:
            return self._fallback_flow(point_history)

        batch, current_points = self._build_batch(point_history, pose_history)
        out = self.model(batch)

        flow_list = out.get("flow", None)
        valid_idx_list = out.get("pc0_valid_point_idxes", None)
        # 关键输出缺失时回退，避免异常传播到上层控制循环。
        if not flow_list or not valid_idx_list:
            return self._fallback_flow(point_history)

        flow = flow_list[0].detach().float().cpu().numpy()
        valid_idx_raw = valid_idx_list[0]
        if hasattr(valid_idx_raw, "detach"):
            valid_idx = valid_idx_raw.detach().cpu().numpy().astype(np.int64)
        else:
            valid_idx = np.asarray(valid_idx_raw, dtype=np.int64)

        full_flow = np.zeros_like(current_points, dtype=np.float32)
        valid_mask = np.zeros((current_points.shape[0],), dtype=bool)

        # 防御式裁剪：避免 flow 与索引长度不一致。
        m = min(flow.shape[0], valid_idx.shape[0])
        valid_idx = valid_idx[:m]
        flow = flow[:m]

        # 再次过滤越界索引，保证散射安全。
        in_range = (valid_idx >= 0) & (valid_idx < current_points.shape[0])
        valid_idx = valid_idx[in_range]
        flow = flow[in_range]

        full_flow[valid_idx] = flow
        valid_mask[valid_idx] = True

        return FlowInferenceResult(
            points_xyz=current_points,
            flow_xyz=full_flow,
            valid_mask=valid_mask,
            stamp=point_history[-1].stamp,
        )
