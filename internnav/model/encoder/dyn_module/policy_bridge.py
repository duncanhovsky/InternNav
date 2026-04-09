from __future__ import annotations
"""FlowNav 外置动态模块桥接层。

本模块将各子组件串联为统一运行时：
1. 接收点云与位姿输入。
2. 做历史帧 ego-motion 对齐。
3. 调 OpenSceneFlow 预测场景流。
4. 执行聚类、关联与 Kalman 跟踪。
5. 构建并缓存 policy 可消费的 dynamic_voxels。
"""

import time
from collections import deque
from typing import Deque, List, Optional

import numpy as np
import torch

from .config import DynModuleConfig
from .dynamic_voxel_builder import DynamicVoxelBuilder
from .motion_postprocess import MotionPostProcessor
from .opensceneflow_adapter import OpenSceneFlowAdapter
from .ros2_io import ApproxTimeSyncBuffer, odom_msg_to_pose, pointcloud_msg_to_xyz
from .types import PointCloudFrame, PoseFrame, RuntimeStats, TrackState


def _invert_t(t_world_ego: np.ndarray) -> np.ndarray:
    """求齐次变换的逆矩阵（SE(3)）。"""
    r = t_world_ego[:3, :3]
    p = t_world_ego[:3, 3]
    t_ego_world = np.eye(4, dtype=np.float32)
    t_ego_world[:3, :3] = r.T
    t_ego_world[:3, 3] = -r.T @ p
    return t_ego_world


def _transform_points(points_xyz: np.ndarray, t_dst_src: np.ndarray) -> np.ndarray:
    """将点云从源坐标系变换到目标坐标系。"""
    pts = np.asarray(points_xyz, dtype=np.float32)
    r = t_dst_src[:3, :3]
    t = t_dst_src[:3, 3]
    return (pts @ r.T) + t[None, :]


class FlowNavDynamicsRuntime:
    """对外动态模块运行时主类。

    目标是输出形状为 ``(B,T,C,X,Y,Z)`` 的 dynamic_voxels，且不修改 policy
    的既有接口。
    """

    def __init__(self, cfg: Optional[DynModuleConfig] = None):
        """初始化运行时组件与缓存。"""
        self.cfg = cfg if cfg is not None else DynModuleConfig()
        self.stats = RuntimeStats()

        self.flow_adapter = OpenSceneFlowAdapter(self.cfg)
        self.post = MotionPostProcessor(self.cfg)
        self.voxel_builder = DynamicVoxelBuilder(self.cfg, stats=self.stats)

        self.cloud_history: Deque[PointCloudFrame] = deque(maxlen=self.cfg.history_frames)
        self.pose_history: Deque[PoseFrame] = deque(maxlen=self.cfg.history_frames)

        self.sync = ApproxTimeSyncBuffer(max_dt_sec=max(0.02, 0.5 * self.cfg.frame_dt))

        self._last_heavy_stamp: Optional[float] = None

    def _extract_static_points(
        self,
        flow_points: np.ndarray,
        flow_vectors: np.ndarray,
        valid_mask: np.ndarray,
        tracks: List[TrackState],
    ) -> np.ndarray:
        """从场景流结果中提取“静态候选点”，用于 hybrid 占据构建。

        设计目标：
        1) 仅在 output_mode="hybrid" 时启用，避免对 dynamic_only 路径引入额外开销。
        2) 先按速度阈值筛出静态候选，再剔除动态轨迹邻域，减少拖影污染。
        3) 返回 numpy 数组，直接供 DynamicVoxelBuilder 更新静态图。

        Args:
            flow_points: 当前帧点云，形状 (N,3)。
            flow_vectors: 当前帧每点流向量，形状 (N,3)。
            valid_mask: 有效点掩码，形状 (N,)。
            tracks: 当前有效轨迹列表。

        Returns:
            np.ndarray: 静态候选点，形状 (M,3)。若无可用点则返回空数组。
        """
        # 仅 hybrid 模式需要静态占据；dynamic_only 直接返回空集合，避免额外计算。
        if str(getattr(self.cfg, "output_mode", "dynamic_only")).lower() != "hybrid":
            return np.zeros((0, 3), dtype=np.float32)

        points = np.asarray(flow_points, dtype=np.float32)
        vectors = np.asarray(flow_vectors, dtype=np.float32)
        valid = np.asarray(valid_mask, dtype=bool)
        if points.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # 步骤1：速度阈值筛选静态候选点。
        # 与动态筛选阈值 min_dynamic_speed 形成“分离带”，可降低边界误判。
        speed = np.linalg.norm(vectors, axis=-1)
        static_mask = valid & (speed <= float(self.cfg.static_speed_max))
        static_points = points[static_mask]
        if static_points.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # 步骤2：动态邻域剔除。
        # 目的：清理动态目标附近的拖影/残影点，避免写入静态图后形成“鬼影障碍”。
        if tracks:
            dynamic_centers = np.stack([tr.state[:3].astype(np.float32) for tr in tracks], axis=0)
            # 计算每个静态候选点到最近动态轨迹中心的距离。
            # 这里采用向量化实现，规模通常可控，且仅在 heavy 链路执行。
            diff = static_points[:, None, :] - dynamic_centers[None, :, :]
            min_dist = np.linalg.norm(diff, axis=-1).min(axis=1)
            keep = min_dist > float(self.cfg.dynamic_exclusion_radius)
            static_points = static_points[keep]

        if static_points.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return static_points.astype(np.float32)

    def _ego_compensate_history(
        self,
        clouds: List[PointCloudFrame],
        poses: List[PoseFrame],
    ) -> List[PointCloudFrame]:
        """将历史点云统一对齐到当前 ego 坐标系。

        该步骤用于去除自车运动影响，提高后续场景流与聚类稳定性。
        """
        if not clouds:
            return []

        current_pose = poses[-1].t_world_ego
        t_current_world = _invert_t(current_pose)

        aligned: List[PointCloudFrame] = []
        for c, p in zip(clouds, poses):
            # 点变换链路: src_local -> world -> current_local。
            t_world_src = p.t_world_ego
            t_current_src = t_current_world @ t_world_src
            pts = _transform_points(c.points_xyz, t_current_src)
            aligned.append(PointCloudFrame(points_xyz=pts, stamp=c.stamp, frame_id=c.frame_id))
        return aligned

    def _should_heavy_update(self, now_stamp: float) -> bool:
        """根据目标频率判断是否执行重链路计算。"""
        if self._last_heavy_stamp is None:
            return True
        period = 1.0 / max(self.cfg.heavy_rate_hz, 1e-3)
        return (now_stamp - self._last_heavy_stamp) >= period

    def ingest(self, point_frame: PointCloudFrame, pose_frame: PoseFrame) -> None:
        """输入一对同步后的点云与位姿，并按需触发重计算。"""
        self.cloud_history.append(point_frame)
        self.pose_history.append(pose_frame)

        if len(self.cloud_history) < 2:
            return
        if len(self.cloud_history) < self.cfg.history_frames:
            # 预热阶段允许短历史执行，避免冷启动等待过长。
            pass

        now_stamp = point_frame.stamp
        if not self._should_heavy_update(now_stamp):
            return

        # 记录重链路耗时用于运行诊断。
        t0 = time.perf_counter()

        clouds = list(self.cloud_history)
        poses = list(self.pose_history)
        clouds_aligned = self._ego_compensate_history(clouds, poses)

        # 场景流推理 -> 后处理 -> 动态体素构建。
        flow_res = self.flow_adapter.predict_flow(clouds_aligned, poses)
        dt = self.cfg.frame_dt
        tracks = self.post.process(flow_res, dt=dt)

        # 在 hybrid 模式下，从当前场景流中提取静态候选点并融合到占据图。
        # 在 dynamic_only 模式下，此函数会返回空数组，保持旧行为。
        static_points = self._extract_static_points(
            flow_points=flow_res.points_xyz,
            flow_vectors=flow_res.flow_xyz,
            valid_mask=flow_res.valid_mask,
            tracks=tracks,
        )

        self.voxel_builder.build_packet(
            tracks=tracks,
            now_stamp=now_stamp,
            static_points_xyz=static_points,
        )

        self._last_heavy_stamp = now_stamp
        self.stats.last_heavy_latency_ms = (time.perf_counter() - t0) * 1000.0

    def ingest_ros_msgs(self, cloud_msg, odom_msg) -> None:
        """输入 ROS1/ROS2 原始消息并自动完成同步与处理。"""
        cloud = pointcloud_msg_to_xyz(cloud_msg)
        odom = odom_msg_to_pose(odom_msg)

        self.sync.push_cloud(cloud)
        self.sync.push_odom(odom)

        # 取近似时间同步结果；若失败则统计丢帧。
        pair = self.sync.pop_synced()
        if pair is None:
            self.stats.dropped_frames += 1
            return
        self.ingest(pair.cloud, pair.odom)

    def get_dynamic_voxels(
        self,
        batch_size: int = 1,
        now_stamp: Optional[float] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """获取 policy 可消费的 dynamic_voxels 张量。

        Returns:
            torch.Tensor: 形状 ``(B, T, C, X, Y, Z)``。
        """
        if now_stamp is None:
            now_stamp = self.pose_history[-1].stamp if self.pose_history else time.time()

        packet = self.voxel_builder.get_or_empty(now_stamp)
        return self.voxel_builder.packet_to_policy_tensor(
            packet=packet,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    def get_stats(self) -> RuntimeStats:
        """返回运行时统计信息。"""
        return self.stats
