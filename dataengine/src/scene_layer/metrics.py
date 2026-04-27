from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .config import SceneLayerConfig
from .errors import ErrorCode, SceneCompileError
from .models import SceneMetrics


@dataclass
class RawSceneMetrics:
    """NavMesh worker 返回的原始指标。

    仅负责表达和轻量归一化，不承载业务阈值策略。
    """

    path_count: int
    shortest_path_m: float
    second_shortest_path_m: float
    detour_margin_m: float
    free_space_ratio: float
    static_density: float


class RawMetricsAdapter:
    """原始指标适配层：将 dict 转为结构化 RawSceneMetrics。"""

    @staticmethod
    def from_dict(raw_metrics: dict) -> RawSceneMetrics:
        return RawSceneMetrics(
            path_count=max(0, int(raw_metrics["path_count"])),
            shortest_path_m=max(0.0, float(raw_metrics["shortest_path_m"])),
            second_shortest_path_m=max(0.0, float(raw_metrics["second_shortest_path_m"])),
            detour_margin_m=max(0.0, float(raw_metrics["detour_margin_m"])),
            free_space_ratio=max(0.0, min(1.0, float(raw_metrics["free_space_ratio"]))),
            static_density=max(0.0, min(1.0, float(raw_metrics["static_density"]))),
        )


@dataclass
class SceneMetricPolicy:
    """策略阈值层：负责质量判定与复杂度分桶。"""

    cfg: SceneLayerConfig

    @staticmethod
    def _bucket_from_bins(value: float, bins: List[float]) -> int:
        for idx in range(len(bins) - 1):
            if bins[idx] <= value < bins[idx + 1]:
                return idx
        return len(bins) - 2

    def _build_scene_metrics(self, raw: RawSceneMetrics) -> SceneMetrics:
        path_count = raw.path_count
        shortest = raw.shortest_path_m
        second_shortest = raw.second_shortest_path_m
        detour = raw.detour_margin_m
        free_ratio = raw.free_space_ratio
        static_density = raw.static_density
        # 复杂度分：同时考虑静态密度、路径冗余不足、自由空间稀缺。
        static_complexity = max(
            0.0,
            min(
                1.0,
                0.55 * static_density + 0.25 * (1.0 - free_ratio) + 0.20 * (1.0 / max(path_count, 1)),
            ),
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

    def validate_and_build(self, scene_id: str, raw: RawSceneMetrics) -> SceneMetrics:
        scene_metrics = self._build_scene_metrics(raw)

        if scene_metrics.path_count < self.cfg.min_path_count:
            raise SceneCompileError(
                code=ErrorCode.REACHABILITY_FAIL,
                message=f"path_count={scene_metrics.path_count} 小于阈值 {self.cfg.min_path_count}",
                scene_id=scene_id,
            )

        detour_floor = max(
            self.cfg.min_detour_margin_m,
            self.cfg.detour_margin_ratio_floor * scene_metrics.shortest_path_m,
        )
        if scene_metrics.detour_margin_m < detour_floor:
            raise SceneCompileError(
                code=ErrorCode.METRIC_REJECT,
                message=f"detour_margin_m={scene_metrics.detour_margin_m:.3f} 小于阈值 {detour_floor:.3f}",
                scene_id=scene_id,
            )

        if scene_metrics.free_space_ratio < self.cfg.min_free_space_ratio:
            raise SceneCompileError(
                code=ErrorCode.METRIC_REJECT,
                message=(
                    f"free_space_ratio={scene_metrics.free_space_ratio:.3f} "
                    f"小于阈值 {self.cfg.min_free_space_ratio:.3f}"
                ),
                scene_id=scene_id,
            )

        return scene_metrics

    def complexity_bucket(self, scene_metrics: SceneMetrics) -> str:
        # 这里只看静态复杂度，动态复杂度在动态层补齐。
        bins = self.cfg.static_complexity_bins
        idx = self._bucket_from_bins(scene_metrics.static_complexity_score, bins)
        if idx <= 0:
            return "easy"
        if idx == 1:
            return "medium"
        return "hard"


@dataclass
class MetricsEvaluator:
    """兼容入口：组合原始指标层与策略阈值层。"""

    cfg: SceneLayerConfig

    def __post_init__(self) -> None:
        self.policy = SceneMetricPolicy(cfg=self.cfg)

    def validate_and_build(self, scene_id: str, raw_metrics: dict) -> SceneMetrics:
        raw = RawMetricsAdapter.from_dict(raw_metrics)
        return self.policy.validate_and_build(scene_id=scene_id, raw=raw)

    def complexity_bucket(self, scene_metrics: SceneMetrics) -> str:
        return self.policy.complexity_bucket(scene_metrics)
