from __future__ import annotations
"""ROS 输入适配与近似时间同步工具。

该模块提供：
1. ROS1/ROS2 Odometry -> 4x4 位姿转换。
2. ROS1/ROS2 PointCloud2 -> Nx3 点云转换。
3. 点云与里程计的近似时间同步缓存。
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np

from .types import PointCloudFrame, PoseFrame


@dataclass
class SyncedPair:
    """同步后的传感器数据对。"""
    cloud: PointCloudFrame
    odom: PoseFrame


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """四元数转旋转矩阵。"""
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    r = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )
    return r


def odom_msg_to_pose(odom_msg) -> PoseFrame:
    """将 ROS1/ROS2 的 Odometry 消息转换为 PoseFrame。"""
    # 兼容 ROS1 与 ROS2 的常见字段结构。
    p = odom_msg.pose.pose.position
    q = odom_msg.pose.pose.orientation

    t = np.eye(4, dtype=np.float32)
    t[:3, :3] = _quat_to_rot(float(q.x), float(q.y), float(q.z), float(q.w))
    t[:3, 3] = np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float32)

    # ROS2: sec+nanosec；ROS1: to_sec()。
    stamp = float(odom_msg.header.stamp.sec) + float(odom_msg.header.stamp.nanosec) * 1e-9 if hasattr(odom_msg.header.stamp, "nanosec") else float(odom_msg.header.stamp.to_sec())
    frame_id = getattr(odom_msg.header, "frame_id", "")
    return PoseFrame(t_world_ego=t, stamp=stamp, frame_id=frame_id)


def pointcloud_msg_to_xyz(cloud_msg) -> PointCloudFrame:
    """将 ROS1/ROS2 的 PointCloud2 消息转换为 PointCloudFrame。

    运行时优先尝试 ROS2 的 ``sensor_msgs_py.point_cloud2``，
    失败后回退到 ROS1 的 ``sensor_msgs.point_cloud2``。
    """
    points = None

    # ROS2 解码路径。
    try:
        from sensor_msgs_py import point_cloud2 as pc2  # type: ignore

        arr = np.array(
            list(pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)),
            dtype=np.float32,
        )
        points = arr
    except Exception:
        pass

    # ROS1 回退解码路径。
    if points is None:
        try:
            from sensor_msgs import point_cloud2 as pc2  # type: ignore

            arr = np.array(
                list(pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)),
                dtype=np.float32,
            )
            points = arr
        except Exception as exc:
            raise RuntimeError(
                "Failed to decode PointCloud2. Install ROS point_cloud2 utilities."
            ) from exc

    # 防御式 reshape，确保输出始终是 (N,3)。
    if points.ndim != 2:
        points = points.reshape(-1, 3)

    stamp = float(cloud_msg.header.stamp.sec) + float(cloud_msg.header.stamp.nanosec) * 1e-9 if hasattr(cloud_msg.header.stamp, "nanosec") else float(cloud_msg.header.stamp.to_sec())
    frame_id = getattr(cloud_msg.header, "frame_id", "")
    return PointCloudFrame(points_xyz=points.astype(np.float32), stamp=stamp, frame_id=frame_id)


class ApproxTimeSyncBuffer:
    """点云与里程计的近似时间同步器。"""

    def __init__(self, max_dt_sec: float = 0.05, queue_size: int = 64):
        """初始化同步缓存。

        Args:
            max_dt_sec: 最大可接受时间差（秒）。
            queue_size: 缓存队列长度。
        """
        self.max_dt_sec = float(max_dt_sec)
        self.cloud_q: Deque[PointCloudFrame] = deque(maxlen=queue_size)
        self.odom_q: Deque[PoseFrame] = deque(maxlen=queue_size)

    def push_cloud(self, frame: PointCloudFrame) -> None:
        """压入点云帧。"""
        self.cloud_q.append(frame)

    def push_odom(self, frame: PoseFrame) -> None:
        """压入里程计帧。"""
        self.odom_q.append(frame)

    def pop_synced(self) -> Optional[SyncedPair]:
        """弹出与最新点云最匹配的里程计帧。"""
        if not self.cloud_q or not self.odom_q:
            return None

        cloud = self.cloud_q[-1]
        best_idx = -1
        best_dt = float("inf")

        # 在里程计队列中搜索时间差最小的候选。
        for idx, odom in enumerate(self.odom_q):
            dt = abs(cloud.stamp - odom.stamp)
            if dt < best_dt:
                best_dt = dt
                best_idx = idx

        if best_idx < 0 or best_dt > self.max_dt_sec:
            return None

        odom = list(self.odom_q)[best_idx]
        return SyncedPair(cloud=cloud, odom=odom)
