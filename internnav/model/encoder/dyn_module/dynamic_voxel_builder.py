from __future__ import annotations
"""动态体素构建与缓存模块。

输入为目标级轨迹状态（位置+速度），输出为 policy 可直接消费的
``dynamic_voxels``，通道约定为 ``[Occ, Vx, Vy, Vz]``。
"""

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from .config import DynModuleConfig
from .types import DynamicVoxelPacket, RuntimeStats, TrackState


@dataclass
class DynamicVoxelCache:
    """动态体素缓存容器。"""
    packet: Optional[DynamicVoxelPacket] = None

    def is_valid(self, now_stamp: float, ttl_sec: float) -> bool:
        """判断缓存是否仍在有效期内。"""
        if self.packet is None:
            return False
        return (now_stamp - self.packet.stamp) <= ttl_sec


class DynamicVoxelBuilder:
    """根据轨迹状态生成并缓存动态体素。"""

    def __init__(self, cfg: DynModuleConfig, stats: Optional[RuntimeStats] = None):
        """初始化网格参数与统计对象。"""
        self.cfg = cfg
        self.cache = DynamicVoxelCache()
        self.stats = stats if stats is not None else RuntimeStats()

        self.vs = np.asarray(cfg.voxel_size, dtype=np.float32)
        self.pcr = np.asarray(cfg.point_cloud_range, dtype=np.float32)
        self.origin = self.pcr[:3].copy()
        self.grid = cfg.grid_size_xyz.astype(np.int32)

        # 静态占据概率图（EMA）：
        # 该图仅在 output_mode="hybrid" 时被用于融合占据通道。
        # 设计原因：
        # 1) 在“几乎无动态目标”的场景中，纯动态占据会非常稀疏，导致动态分支信息不足。
        # 2) 将静态占据以低频、平滑的方式写入，可显著提升时空条件稳定性。
        self.static_occ_map = np.zeros((self.grid[0], self.grid[1], self.grid[2]), dtype=np.float32)

    def _mode(self) -> str:
        """标准化输出模式字符串，容忍大小写与异常值。"""
        mode = str(getattr(self.cfg, "output_mode", "dynamic_only")).strip().lower()
        if mode not in {"dynamic_only", "hybrid"}:
            # 对非法配置进行安全回退，确保运行时可持续。
            return "dynamic_only"
        return mode

    def _splat_occ_only(self, occ: np.ndarray, ijk: np.ndarray, radius: Optional[int] = None) -> None:
        """仅写入占据通道（不写速度），用于静态占据图构建。

        Args:
            occ: 形状为 (X,Y,Z) 的占据图。
            ijk: 体素索引 [i,j,k]。
            radius: 膨胀半径；若为 None 则复用配置中的 occ_inflate_radius。
        """
        r = int(self.cfg.occ_inflate_radius if radius is None else radius)
        xi, yi, zi = int(ijk[0]), int(ijk[1]), int(ijk[2])
        x0, x1 = max(0, xi - r), min(self.grid[0] - 1, xi + r)
        y0, y1 = max(0, yi - r), min(self.grid[1] - 1, yi + r)
        z0, z1 = max(0, zi - r), min(self.grid[2] - 1, zi + r)
        occ[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1] = 1.0

    def _update_static_map(self, static_points_xyz: Optional[np.ndarray]) -> np.ndarray:
        """根据当前静态点更新 EMA 静态占据图，并返回二值静态图。

        关键策略：
        1) 先将当前帧静态点体素化为 hit_map。
        2) 再通过 EMA 融合到长期静态图，抑制单帧噪声与闪烁。
        3) 最后按阈值二值化，供 hybrid 模式下的占据融合使用。

        Args:
            static_points_xyz: 当前重更新周期提取到的静态点，形状 (N,3)；可为 None。

        Returns:
            np.ndarray: 二值静态占据图，形状 (X,Y,Z)，取值 {0,1}。
        """
        # current_hit 记录“当前周期静态命中”，后续将与历史图做 EMA 融合。
        current_hit = np.zeros_like(self.static_occ_map, dtype=np.float32)

        if static_points_xyz is not None and static_points_xyz.size > 0:
            points = np.asarray(static_points_xyz, dtype=np.float32)

            # 高度裁剪：过滤地面噪声与高空离群点，减少静态图污染。
            z_ok = (points[:, 2] >= self.cfg.static_z_min) & (points[:, 2] <= self.cfg.static_z_max)
            points = points[z_ok]

            if points.size > 0:
                ijk = self._xyz_to_grid(points)
                valid = self._in_bounds(ijk)
                ijk = ijk[valid]

                if ijk.size > 0:
                    # 使用 unique 去重可显著减少重复写入，提高 CPU 侧效率。
                    uniq = np.unique(ijk, axis=0)
                    for voxel in uniq:
                        self._splat_occ_only(current_hit, voxel)

        # EMA 更新静态概率图：
        # - 在动态较弱、噪声较大的输入下，EMA 能提供更稳定的静态结构先验。
        # - static_ema_decay 越大，历史保留越强，响应越慢。
        decay = float(np.clip(self.cfg.static_ema_decay, 0.0, 0.9999))
        self.static_occ_map = decay * self.static_occ_map + (1.0 - decay) * current_hit

        # 阈值化后作为最终静态占据。
        threshold = float(np.clip(self.cfg.static_occ_thresh, 0.0, 1.0))
        static_binary = (self.static_occ_map >= threshold).astype(np.float32)
        return static_binary

    def _xyz_to_grid(self, xyz: np.ndarray) -> np.ndarray:
        """将世界/局部坐标转换为体素网格坐标。"""
        return np.floor((xyz - self.origin[None, :]) / self.vs[None, :]).astype(np.int32)

    def _in_bounds(self, ijk: np.ndarray) -> np.ndarray:
        """判断体素索引是否在网格范围内。"""
        x_ok = (ijk[:, 0] >= 0) & (ijk[:, 0] < self.grid[0])
        y_ok = (ijk[:, 1] >= 0) & (ijk[:, 1] < self.grid[1])
        z_ok = (ijk[:, 2] >= 0) & (ijk[:, 2] < self.grid[2])
        return x_ok & y_ok & z_ok

    def _splat_voxel(self, occ: np.ndarray, vel: np.ndarray, t: int, ijk: np.ndarray, vxyz: np.ndarray) -> None:
        """将单个轨迹点写入体素网格，并进行局部膨胀。

        Args:
            occ: 占据通道缓存，形状 (T, X, Y, Z)。
            vel: 速度通道缓存，形状 (T, 3, X, Y, Z)。
            t: 时间索引。
            ijk: 目标体素索引 [i, j, k]。
            vxyz: 对应速度向量 [vx, vy, vz]。
        """
        r = int(self.cfg.occ_inflate_radius)
        xi, yi, zi = int(ijk[0]), int(ijk[1]), int(ijk[2])
        x0, x1 = max(0, xi - r), min(self.grid[0] - 1, xi + r)
        y0, y1 = max(0, yi - r), min(self.grid[1] - 1, yi + r)
        z0, z1 = max(0, zi - r), min(self.grid[2] - 1, zi + r)

        occ[t, x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1] = 1.0
        vel[t, 0, x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1] = vxyz[0]
        vel[t, 1, x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1] = vxyz[1]
        vel[t, 2, x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1] = vxyz[2]

    def build_packet(
        self,
        tracks: List[TrackState],
        now_stamp: float,
        static_points_xyz: Optional[np.ndarray] = None,
    ) -> DynamicVoxelPacket:
        """从轨迹集合构建体素包，并按模式输出“仅动态”或“静态+动态联合占据”。

        Args:
            tracks: 动态轨迹状态列表（来自后处理模块）。
            now_stamp: 当前时间戳。
            static_points_xyz: 当前周期静态点集合，shape=(N,3)。
                仅在 hybrid 模式下用于更新静态占据图。
        """
        t_h = int(self.cfg.horizon_frames)
        occ_dynamic = np.zeros((t_h, self.grid[0], self.grid[1], self.grid[2]), dtype=np.float32)
        vel_dynamic = np.zeros((t_h, 3, self.grid[0], self.grid[1], self.grid[2]), dtype=np.float32)
        mode = self._mode()

        # 对每条轨迹做未来 T 帧常速度外推。
        for tr in tracks:
            pos = tr.state[:3].astype(np.float32)
            v = np.clip(tr.state[3:6].astype(np.float32), -self.cfg.vel_clip, self.cfg.vel_clip)
            for t in range(t_h):
                dt = (t + 1) * self.cfg.frame_dt
                p_t = pos + v * dt
                ijk = self._xyz_to_grid(p_t.reshape(1, 3))[0]
                if not self._in_bounds(ijk.reshape(1, 3))[0]:
                    continue
                self._splat_voxel(occ_dynamic, vel_dynamic, t, ijk, v)

        # 按输出模式决定占据通道的语义：
        # - dynamic_only: Occ 仅来自动态轨迹外推。
        # - hybrid: Occ = max(静态占据, 动态占据)。
        # 速度通道始终保持“仅动态”，避免静态结构产生伪速度。
        if mode == "hybrid":
            static_binary = self._update_static_map(static_points_xyz)
            # 将同一份静态图复制到未来 T 帧，再与每个时间步动态占据取并集。
            static_occ_t = np.repeat(static_binary[None, ...], t_h, axis=0)
            occ = np.maximum(static_occ_t, occ_dynamic)
        else:
            # dynamic_only 模式下不更新静态图，保证行为与旧版一致。
            occ = occ_dynamic

        # 拼接为 (T, C, X, Y, Z)，通道顺序固定为 [Occ,Vx,Vy,Vz]。
        vox = np.concatenate([occ[:, None, ...], vel_dynamic], axis=1)
        horizon_stamps = [now_stamp + (i + 1) * self.cfg.frame_dt for i in range(t_h)]

        packet = DynamicVoxelPacket(
            voxels_t_cxyz=vox,
            horizon_stamps=horizon_stamps,
            voxel_size=self.vs.copy(),
            grid_origin=self.origin.copy(),
            point_cloud_range=self.pcr.copy(),
            stamp=now_stamp,
            meta={
                "num_tracks": len(tracks),
                "output_mode": mode,
                "has_static_points": bool(static_points_xyz is not None and np.asarray(static_points_xyz).size > 0),
            },
        )
        self.cache.packet = packet
        self.stats.heavy_updates += 1
        return packet

    def get_or_empty(self, now_stamp: float) -> DynamicVoxelPacket:
        """获取缓存包；若缓存失效则返回空体素包。"""
        if self.cache.packet is not None and self.cache.is_valid(now_stamp, self.cfg.cache_ttl_sec):
            self.stats.cache_hits += 1
            return self.cache.packet

        self.stats.stale_fallbacks += 1
        mode = self._mode()

        # 回退阶段是否对静态图做轻微衰减：
        # 可缓解长时间无重更新时“陈旧静态结构”持续存在的问题。
        if mode == "hybrid" and self.cfg.decay_static_on_fallback:
            decay = float(np.clip(self.cfg.fallback_static_decay, 0.9, 1.0))
            self.static_occ_map *= decay

        # hybrid 模式下即便重链路暂时不可用，也尽量保留静态占据先验。
        if mode == "hybrid":
            static_binary = (self.static_occ_map >= self.cfg.static_occ_thresh).astype(np.float32)
            occ = np.repeat(static_binary[None, ...], self.cfg.horizon_frames, axis=0)
        else:
            occ = np.zeros(
                (self.cfg.horizon_frames, self.grid[0], self.grid[1], self.grid[2]),
                dtype=np.float32,
            )

        vel = np.zeros(
            (self.cfg.horizon_frames, 3, self.grid[0], self.grid[1], self.grid[2]),
            dtype=np.float32,
        )
        vox = np.concatenate([occ[:, None, ...], vel], axis=1)

        empty = DynamicVoxelPacket(
            voxels_t_cxyz=vox,
            horizon_stamps=[now_stamp + (i + 1) * self.cfg.frame_dt for i in range(self.cfg.horizon_frames)],
            voxel_size=self.vs.copy(),
            grid_origin=self.origin.copy(),
            point_cloud_range=self.pcr.copy(),
            stamp=now_stamp,
            meta={"num_tracks": 0, "fallback": True, "output_mode": mode},
        )
        return empty

    def packet_to_policy_tensor(
        self,
        packet: DynamicVoxelPacket,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """将体素包转换为 policy 输入张量。

        输出张量形状固定为 ``(B, T, C, X, Y, Z)``。
        """
        vox = packet.voxels_t_cxyz
        if vox.ndim != 5 or vox.shape[1] != 4:
            raise ValueError(f"Expected packet voxels (T,4,X,Y,Z), got {vox.shape}")

        tensor = torch.from_numpy(vox).to(dtype=dtype)
        tensor = tensor.unsqueeze(0).expand(batch_size, *tensor.shape)
        if device is not None:
            tensor = tensor.to(device)
        return tensor.contiguous()
