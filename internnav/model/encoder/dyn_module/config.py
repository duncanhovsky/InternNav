from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class DynModuleConfig:
    """外置动态模块配置。

    该配置覆盖从场景流推理到动态体素构建的关键超参数，分为：
    1. 模型与设备配置。
    2. 时间与频率配置。
    3. 体素网格配置。
    4. 聚类、关联与跟踪配置。
    """

    # 场景流模型名称，可选：liteflow / deltaflow。
    model_name: str = "liteflow"
    # 可选 checkpoint 路径；为空时使用模型默认初始化。
    checkpoint_path: str = ""
    # 运行设备字符串，如 "cuda:0" / "cpu"。
    device: str = "cuda:0"

    # 历史帧长度（输入场景流模型的点云帧数）。
    history_frames: int = 8
    # 未来滚动帧长度（dynamic_voxels 的时间维 T）。
    horizon_frames: int = 8
    # 相邻未来帧间隔（秒）。
    frame_dt: float = 0.1

    # 重计算链路目标频率（场景流+聚类+跟踪）。
    heavy_rate_hz: float = 7.5
    # policy 调用目标频率（通常高于 heavy_rate_hz）。
    policy_rate_hz: float = 15.0
    # 缓存有效期（秒），超时则触发空回退或重算。
    cache_ttl_sec: float = 0.5

    # 体素尺寸 [vx, vy, vz]，单位米。
    voxel_size: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    # 点云空间范围 [xmin, ymin, zmin, xmax, ymax, zmax]。
    point_cloud_range: List[float] = field(
        default_factory=lambda: [-20.0, -20.0, -3.0, 20.0, 20.0, 5.0]
    )

    # 动态点速度阈值（小于该阈值视为静态/噪声）。
    min_dynamic_speed: float = 0.15
    # 聚类最小点数。
    cluster_min_size: int = 12
    # 聚类邻域尺度（HDBSCAN 的 epsilon 或 DBSCAN 的 eps）。
    cluster_eps: float = 0.8

    # 轨迹最大丢失帧数，超过后删除轨迹。
    track_max_missed: int = 6
    # 关联最大距离阈值（米）。
    association_max_dist: float = 1.5
    # 是否在关联代价中使用速度项（当前实现预留开关）。
    use_velocity_in_association: bool = True

    # 占据膨胀半径（体素单位）。
    occ_inflate_radius: int = 1
    # 速度裁剪上限，避免异常值污染体素。
    vel_clip: float = 8.0

    # 输出模式控制：
    # - "dynamic_only": 仅输出动态占据与动态速度（兼容当前旧行为）。
    # - "hybrid": 输出“静态+动态联合占据图”，速度通道仍仅来自动态目标。
    # 说明：
    # 1) 该参数用于在不改 policy 输入签名的前提下切换两种语义。
    # 2) 推荐默认 "hybrid"，因为它在静态场景下更稳定，且不会污染速度通道。
    output_mode: str = "hybrid"

    # 静态点速度阈值（米/秒）：
    # 仅当点速低于该阈值时，才会被当作静态候选并写入静态占据图。
    # 与 min_dynamic_speed 搭配可形成“动态/静态”分离带。
    static_speed_max: float = 0.08

    # 静态占据图的指数滑动平均(EMA)系数：
    # new_static = static_ema_decay * old_static + (1 - static_ema_decay) * current_hit
    # 该机制可抑制单帧噪声，提升占据图时序稳定性。
    static_ema_decay: float = 0.95

    # 将 EMA 概率图二值化为占据图的阈值。
    # 阈值越高越保守，越低越容易把短时噪声点保留下来。
    static_occ_thresh: float = 0.5

    # 静态占据的高度裁剪区间（单位米，坐标系与体素网格一致）：
    # 仅保留 [static_z_min, static_z_max] 范围内的静态点，减少地面噪声和高空离群点影响。
    static_z_min: float = -1.5
    static_z_max: float = 2.5

    # 动态轨迹邻域剔除半径（单位米）：
    # 为避免“动态目标拖影”被误写入静态图，会剔除距离任意动态轨迹位置小于该半径的静态候选点。
    dynamic_exclusion_radius: float = 0.4   # 单位：米

    # 静态图是否在缓存过期且无重更新时进行时间衰减：
    # True 可降低陈旧静态结构的残留风险；False 可保持静态图更稳定。
    decay_static_on_fallback: bool = True

    # 回退阶段静态图衰减系数（仅在 decay_static_on_fallback=True 时生效）。
    # 每次回退调用：static_map *= fallback_static_decay
    fallback_static_decay: float = 0.995

    @property
    def grid_size_xyz(self) -> np.ndarray:
        """根据范围与体素尺寸计算网格维度。

        Returns:
            np.ndarray: 形状为 (3,) 的整数数组 [X, Y, Z]。
        """
        pcr = np.asarray(self.point_cloud_range, dtype=np.float32)
        vs = np.asarray(self.voxel_size, dtype=np.float32)
        dims = (pcr[3:] - pcr[:3]) / vs
        # 至少保证每个维度为 1，避免非法空网格。
        return np.maximum(np.floor(dims).astype(np.int32), 1)
