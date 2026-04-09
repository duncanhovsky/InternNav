from __future__ import annotations
"""运动后处理模块。

该模块接收逐点场景流结果，执行如下步骤：
1. 动态点筛选。
2. HDBSCAN/DBSCAN 聚类。
3. 簇级速度池化（中位数）。
4. 轨迹关联。
5. Kalman 预测与更新。

输出为可用于未来 roll-out 的稳定轨迹状态集合。
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .config import DynModuleConfig
from .types import FlowInferenceResult, TrackState


def _safe_norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """安全范数计算，防止数值下溢。"""
    return np.sqrt(np.maximum(np.sum(x * x, axis=axis), 1e-12))


@dataclass
class ClusterSummary:
    """单个聚类簇的摘要信息。"""
    centroid: np.ndarray
    velocity: np.ndarray
    size: int


class MotionPostProcessor:
    """基于聚类与 Kalman 的运动估计器。"""

    def __init__(self, cfg: DynModuleConfig):
        """初始化后处理器与轨迹容器。"""
        self.cfg = cfg
        self.tracks: Dict[int, TrackState] = {}
        self._next_track_id = 1

        self._cluster_backend = None
        self._init_cluster_backend()

    def _init_cluster_backend(self) -> None:
        """初始化聚类后端。

        优先级：
        1. hdbscan.HDBSCAN
        2. sklearn.cluster.DBSCAN
        3. 无可用后端（退化为全部噪声）
        """
        try:
            import hdbscan

            self._cluster_backend = ("hdbscan", hdbscan.HDBSCAN)
            return
        except Exception:
            pass

        try:
            from sklearn.cluster import DBSCAN

            self._cluster_backend = ("dbscan", DBSCAN)
        except Exception:
            self._cluster_backend = None

    def _cluster(self, points_xyz: np.ndarray) -> np.ndarray:
        """对动态点进行聚类。

        Args:
            points_xyz: 动态点坐标，形状 (N,3)。

        Returns:
            np.ndarray: 簇标签，形状 (N,)。噪声标签为 -1。
        """
        if points_xyz.shape[0] == 0:
            return np.zeros((0,), dtype=np.int64)

        if self._cluster_backend is None:
            # 无聚类库时全部视为噪声点。
            return np.full((points_xyz.shape[0],), -1, dtype=np.int64)

        kind, cls = self._cluster_backend
        if kind == "hdbscan":
            model = cls(
                min_cluster_size=max(self.cfg.cluster_min_size, 2),
                cluster_selection_epsilon=self.cfg.cluster_eps,
            )
            labels = model.fit_predict(points_xyz)
        else:
            model = cls(
                eps=self.cfg.cluster_eps,
                min_samples=max(self.cfg.cluster_min_size, 2),
            )
            labels = model.fit_predict(points_xyz)

        return np.asarray(labels, dtype=np.int64)

    def _summarize_clusters(
        self,
        points_xyz: np.ndarray,
        flow_xyz: np.ndarray,
        labels: np.ndarray,
    ) -> List[ClusterSummary]:
        """将逐点结果聚合为簇级摘要。

        采用中位数对质心与速度进行稳健估计，减弱离群点影响。
        """
        summaries: List[ClusterSummary] = []
        if points_xyz.shape[0] == 0:
            return summaries

        unique_labels = np.unique(labels)
        for lab in unique_labels:
            if lab < 0:
                continue
            mask = labels == lab
            size = int(np.sum(mask))
            if size < self.cfg.cluster_min_size:
                continue
            pts = points_xyz[mask]
            f = flow_xyz[mask]
            centroid = np.median(pts, axis=0).astype(np.float32)
            velocity = np.median(f, axis=0).astype(np.float32)
            summaries.append(ClusterSummary(centroid=centroid, velocity=velocity, size=size))
        return summaries

    def _predict_track(self, track: TrackState, dt: float) -> None:
        """Kalman 预测步（常速度模型）。"""
        # 状态向量: x = [px, py, pz, vx, vy, vz]
        f = np.eye(6, dtype=np.float32)
        f[0, 3] = dt
        f[1, 4] = dt
        f[2, 5] = dt

        q = np.eye(6, dtype=np.float32) * 0.05
        track.state = f @ track.state
        track.covariance = f @ track.covariance @ f.T + q

    def _update_track(self, track: TrackState, z: np.ndarray) -> None:
        """Kalman 更新步。

        Args:
            track: 待更新轨迹。
            z: 测量向量 [px, py, pz, vx, vy, vz]。
        """
        # 测量向量: z = [px, py, pz, vx, vy, vz]
        h = np.eye(6, dtype=np.float32)
        r = np.eye(6, dtype=np.float32) * 0.2

        y = z - (h @ track.state)
        s = h @ track.covariance @ h.T + r
        try:
            k = track.covariance @ h.T @ np.linalg.inv(s)
        except np.linalg.LinAlgError:
            k = track.covariance @ h.T @ np.linalg.pinv(s)

        track.state = track.state + k @ y
        i = np.eye(6, dtype=np.float32)
        track.covariance = (i - k @ h) @ track.covariance

    def _associate(
        self,
        track_ids: List[int],
        tracks_pred_pos: np.ndarray,
        clusters: List[ClusterSummary],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """贪心最近邻关联。

        Returns:
            matches: 匹配对列表 (track_id, cluster_idx)。
            unmatched_tracks: 未匹配轨迹 ID 列表。
            unmatched_clusters: 未匹配簇索引列表。
        """
        if len(track_ids) == 0:
            return [], [], list(range(len(clusters)))
        if len(clusters) == 0:
            return [], track_ids.copy(), []

        cluster_pos = np.stack([c.centroid for c in clusters], axis=0)
        dist = np.linalg.norm(
            tracks_pred_pos[:, None, :] - cluster_pos[None, :, :], axis=-1
        )

        matches: List[Tuple[int, int]] = []
        used_t = set()
        used_c = set()

        # 每次选取当前最小距离并将对应行列置为 inf。
        while True:
            idx = np.unravel_index(np.argmin(dist), dist.shape)
            i, j = int(idx[0]), int(idx[1])
            d = float(dist[i, j])
            if not np.isfinite(d) or d > self.cfg.association_max_dist:
                break
            if i in used_t or j in used_c:
                dist[i, j] = np.inf
                continue
            matches.append((track_ids[i], j))
            used_t.add(i)
            used_c.add(j)
            dist[i, :] = np.inf
            dist[:, j] = np.inf
            if len(used_t) == len(track_ids) or len(used_c) == len(clusters):
                break

        unmatched_tracks = [tid for k, tid in enumerate(track_ids) if k not in used_t]
        unmatched_clusters = [k for k in range(len(clusters)) if k not in used_c]
        return matches, unmatched_tracks, unmatched_clusters

    def process(self, flow_res: FlowInferenceResult, dt: float) -> List[TrackState]:
        """执行一帧完整后处理。

        Args:
            flow_res: 场景流推理结果。
            dt: 预测步长（秒）。

        Returns:
            List[TrackState]: 更新后的轨迹列表。
        """
        points = flow_res.points_xyz
        flow = flow_res.flow_xyz
        valid = flow_res.valid_mask

        if points.shape[0] == 0:
            return []

        # 仅保留速度超过阈值的动态点参与聚类。
        speed = _safe_norm(flow)
        dynamic_mask = valid & (speed >= self.cfg.min_dynamic_speed)

        dynamic_points = points[dynamic_mask]
        dynamic_flow = flow[dynamic_mask]

        labels = self._cluster(dynamic_points)
        clusters = self._summarize_clusters(dynamic_points, dynamic_flow, labels)

        # 先对所有现有轨迹执行预测步。
        track_ids = sorted(self.tracks.keys())
        for tid in track_ids:
            self._predict_track(self.tracks[tid], dt)
            self.tracks[tid].age += 1
            self.tracks[tid].missed += 1
            self.tracks[tid].stamp = flow_res.stamp

        track_pos = np.stack([self.tracks[tid].state[:3] for tid in track_ids], axis=0) if track_ids else np.zeros((0, 3), dtype=np.float32)
        matches, unmatched_tracks, unmatched_clusters = self._associate(track_ids, track_pos, clusters)

        # 对匹配成功轨迹做 Kalman 更新。
        for tid, cidx in matches:
            c = clusters[cidx]
            z = np.concatenate([c.centroid, c.velocity], axis=0).astype(np.float32)
            t = self.tracks[tid]
            self._update_track(t, z)
            t.missed = 0
            t.cluster_size = c.size
            t.stamp = flow_res.stamp

        # 对未匹配轨迹计数并按阈值清理。
        for tid in unmatched_tracks:
            if self.tracks[tid].missed > self.cfg.track_max_missed:
                del self.tracks[tid]

        # 对未匹配簇创建新轨迹。
        for cidx in unmatched_clusters:
            c = clusters[cidx]
            x0 = np.concatenate([c.centroid, c.velocity], axis=0).astype(np.float32)
            p0 = np.eye(6, dtype=np.float32) * 0.5
            tid = self._next_track_id
            self._next_track_id += 1
            self.tracks[tid] = TrackState(
                track_id=tid,
                state=x0,
                covariance=p0,
                age=1,
                missed=0,
                stamp=flow_res.stamp,
                cluster_size=c.size,
            )

        return [self.tracks[tid] for tid in sorted(self.tracks.keys())]
