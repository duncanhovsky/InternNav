from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .config import SceneLayerConfig
from .errors import ErrorCode, SceneCompileError
from .models import SceneMetrics


@dataclass
class MetricsEvaluator:
    """场景指标校验与复杂度分桶。"""

    cfg: SceneLayerConfig

    @staticmethod
    def _bucket_from_bins(value: float, bins: List[float]) -> int:
        for idx in range(len(bins) - 1):
            if bins[idx] <= value < bins[idx + 1]:
                return idx
        return len(bins) - 2

    def validate_and_build(self, scene_id: str, raw_metrics: dict) -> SceneMetrics:
        path_count = int(raw_metrics["path_count"])
        shortest = float(raw_metrics["shortest_path_m"])
        second_shortest = float(raw_metrics["second_shortest_path_m"])
        detour = float(raw_metrics["detour_margin_m"])
        free_ratio = float(raw_metrics["free_space_ratio"])
        static_density = float(raw_metrics["static_density"])

        # 复杂度分：同时考虑静态密度、路径冗余不足、自由空间稀缺。
        static_complexity = max(0.0, min(1.0, 0.55 * static_density + 0.25 * (1.0 - free_ratio) + 0.20 * (1.0 / max(path_count, 1))))

        if path_count < self.cfg.min_path_count:
            raise SceneCompileError(
                code=ErrorCode.REACHABILITY_FAIL,
                message=f"path_count={path_count} 小于阈值 {self.cfg.min_path_count}",
                scene_id=scene_id,
            )

        detour_floor = max(self.cfg.min_detour_margin_m, self.cfg.detour_margin_ratio_floor * shortest)
        if detour < detour_floor:
            raise SceneCompileError(
                code=ErrorCode.METRIC_REJECT,
                message=f"detour_margin_m={detour:.3f} 小于阈值 {detour_floor:.3f}",
                scene_id=scene_id,
            )

        if free_ratio < self.cfg.min_free_space_ratio:
            raise SceneCompileError(
                code=ErrorCode.METRIC_REJECT,
                message=f"free_space_ratio={free_ratio:.3f} 小于阈值 {self.cfg.min_free_space_ratio:.3f}",
                scene_id=scene_id,
            )

        return SceneMetrics(
            path_count=path_count,
            shortest_path_m=shortest,
            second_shortest_path_m=second_shortest,
            detour_margin_m=detour,
            free_space_ratio=free_ratio,
            static_density=static_density,
            static_complexity_score=round(static_complexity, 4),
        )

    def complexity_bucket(self, scene_metrics: SceneMetrics) -> str:
        # 这里只看静态复杂度，动态复杂度在动态层补齐。
        bins = self.cfg.static_complexity_bins
        idx = self._bucket_from_bins(scene_metrics.static_complexity_score, bins)
        if idx <= 0:
            return "easy"
        if idx == 1:
            return "medium"
        return "hard"
