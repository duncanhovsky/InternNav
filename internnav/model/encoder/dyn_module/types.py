from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class PointCloudFrame:
    """单帧点云数据。

    Attributes:
        points_xyz: 点云坐标，形状为 (N, 3)，默认在车体或局部坐标系。
        stamp: 时间戳（秒）。
        frame_id: 坐标系名称（可选）。
    """

    points_xyz: np.ndarray
    stamp: float
    frame_id: str = ""


@dataclass
class PoseFrame:
    """单帧位姿数据。

    Attributes:
        t_world_ego: 4x4 齐次变换矩阵，表示 world <- ego。
        stamp: 时间戳（秒）。
        frame_id: 位姿所属坐标系名称（可选）。
    """

    t_world_ego: np.ndarray
    stamp: float
    frame_id: str = ""


@dataclass
class FlowInferenceResult:
    """场景流推理结果。

    该结果与当前帧点云按索引对齐，便于后处理链路直接消费。

    Attributes:
        points_xyz: 当前帧点云，形状 (N, 3)。
        flow_xyz: 与 points_xyz 对齐的速度/流向量，形状 (N, 3)。
        valid_mask: 有效点掩码，形状 (N,)。
        stamp: 当前帧时间戳（秒）。
    """

    points_xyz: np.ndarray
    flow_xyz: np.ndarray
    valid_mask: np.ndarray
    stamp: float


@dataclass
class TrackState:
    """目标跟踪状态。

    状态向量约定为 ``[px, py, pz, vx, vy, vz]``，用于未来轨迹 roll-out。

    Attributes:
        track_id: 轨迹唯一 ID。
        state: 状态向量，形状 (6,)。
        covariance: 状态协方差矩阵，形状 (6, 6)。
        age: 存活帧数。
        missed: 连续丢失帧数。
        stamp: 最近一次更新时间戳。
        cluster_size: 最近匹配簇的点数。
    """

    track_id: int
    state: np.ndarray
    covariance: np.ndarray
    age: int
    missed: int
    stamp: float
    cluster_size: int = 0


@dataclass
class DynamicVoxelPacket:
    """可直接送入 policy 的动态体素包。

    Attributes:
        voxels_t_cxyz: 动态体素，形状 (T, C, X, Y, Z)，通道 C=[Occ,Vx,Vy,Vz]。
        horizon_stamps: 每个未来帧对应时间戳列表，长度 T。
        voxel_size: 体素尺寸 [vx, vy, vz]。
        grid_origin: 网格原点（通常为 point_cloud_range 的最小角）。
        point_cloud_range: 点云空间范围 [xmin, ymin, zmin, xmax, ymax, zmax]。
        stamp: 本包生成时间戳。
        meta: 扩展元信息字典。
    """

    voxels_t_cxyz: np.ndarray
    horizon_stamps: List[float]
    voxel_size: np.ndarray
    grid_origin: np.ndarray
    point_cloud_range: np.ndarray
    stamp: float
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        """返回未来时间长度 T。"""
        return int(self.voxels_t_cxyz.shape[0])


@dataclass
class RuntimeStats:
    """运行时统计信息。

    Attributes:
        heavy_updates: 重链路（场景流+后处理）执行次数。
        cache_hits: 缓存命中次数。
        stale_fallbacks: 缓存过期或缺失后的回退次数。
        dropped_frames: 同步失败或输入丢弃次数。
        last_heavy_latency_ms: 最近一次重链路耗时（毫秒）。
    """

    heavy_updates: int = 0
    cache_hits: int = 0
    stale_fallbacks: int = 0
    dropped_frames: int = 0
    last_heavy_latency_ms: Optional[float] = None
