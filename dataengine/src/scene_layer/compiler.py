from __future__ import annotations

import time
from dataclasses import dataclass

from .composer import SceneComposer
from .config import SceneLayerConfig
from .errors import SceneCompileError
from .metrics import MetricsEvaluator
from .models import SceneCandidate, SceneCompileResult, SceneMetrics
from .navmesh import NavMeshBuilder


@dataclass
class SceneCompiler:
    """单场景编译器。

    职责：
    1. 调用 composer 生成场景 stage。
    2. 调用 navmesh 构建与评估。
    3. 调用 metrics 进行质量阈值检查并输出复杂度桶。
    4. 对可重试错误执行有限重试，返回结构化 SceneCompileResult。
    """

    cfg: SceneLayerConfig

    def __post_init__(self) -> None:
        self.composer = SceneComposer(cfg=self.cfg)
        self.navmesh_builder = NavMeshBuilder(
            bake_timeout_s=self.cfg.navmesh_bake_timeout_s,
            cell_size_m=self.cfg.navmesh_cell_size_m,
            cell_height_m=self.cfg.navmesh_cell_height_m,
            max_slope_deg=self.cfg.navmesh_max_slope_deg,
            step_height_m=self.cfg.navmesh_step_height_m,
            navmesh_agent_profiles=self.cfg.navmesh_agent_profiles,
            agent_radius_m_by_profile=self.cfg.agent_radius_m_by_profile,
            isaac_headless=self.cfg.isaac_headless,
            isaac_renderer=self.cfg.isaac_renderer,
        )
        self.metrics_eval = MetricsEvaluator(cfg=self.cfg)

    @staticmethod
    def _zero_metrics() -> SceneMetrics:
        """返回失败场景的零占位指标，保证结果结构稳定。"""
        return SceneMetrics(
            path_count=0,
            shortest_path_m=0.0,
            second_shortest_path_m=0.0,
            detour_margin_m=0.0,
            free_space_ratio=0.0,
            static_density=0.0,
            static_complexity_score=0.0,
        )

    def compile(self, candidate: SceneCandidate) -> SceneCompileResult:
        """执行一次完整编译流程。

        参数:
            candidate: 已采样的场景候选（含模板、seed、scene_id）。

        返回:
            SceneCompileResult: 成功结果（status=DONE）。
        """
        t0 = time.time()

        composed = self.composer.compose(candidate)
        navmesh = self.navmesh_builder.build_and_eval(
            scene_id=candidate.scene_id,
            scene_dir=composed["scene_dir"],
            seed=candidate.seed,
            stage_usd=composed["stage_usd"],
        )

        scene_metrics = self.metrics_eval.validate_and_build(
            scene_id=candidate.scene_id,
            raw_metrics=navmesh["metrics"],
        )
        complexity_bucket = self.metrics_eval.complexity_bucket(scene_metrics)

        _ = time.time() - t0

        return SceneCompileResult(
            scene_id=candidate.scene_id,
            scene_type=candidate.scene_type,
            mode=candidate.mode,
            seed=candidate.seed,
            status="DONE",
            stage_usd=composed["stage_usd"],
            navmesh_file=navmesh["navmesh_file"],
            metrics=scene_metrics,
            complexity_bucket=complexity_bucket,
            reason=None,
            layout_hash=composed["layout_hash"],
            template_id=candidate.template.template_id,
        )

    def compile_with_retry(self, candidate: SceneCandidate) -> SceneCompileResult:
        """带重试的编译入口。

        仅对 cfg.retryable_error_codes 中定义的错误码执行重试。
        任何失败都收敛为 status=FAILED 的结构化结果，避免上游流水线中断。
        """
        last_exc = None
        for attempt in range(self.cfg.max_compile_retries + 1):
            try:
                return self.compile(candidate)
            except SceneCompileError as exc:
                last_exc = exc
                if exc.code.value not in self.cfg.retryable_error_codes:
                    break
                if attempt >= self.cfg.max_compile_retries:
                    break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= self.cfg.max_compile_retries:
                    break

        if isinstance(last_exc, SceneCompileError):
            return SceneCompileResult(
                scene_id=candidate.scene_id,
                scene_type=candidate.scene_type,
                mode=candidate.mode,
                seed=candidate.seed,
                status="FAILED",
                stage_usd=candidate.template.usd_path,
                navmesh_file="",
                metrics=self._zero_metrics(),
                complexity_bucket="easy",
                reason=str(last_exc),
                layout_hash="",
                template_id=candidate.template.template_id,
            )

        reason = str(last_exc) if last_exc is not None else "unknown"
        return SceneCompileResult(
            scene_id=candidate.scene_id,
            scene_type=candidate.scene_type,
            mode=candidate.mode,
            seed=candidate.seed,
            status="FAILED",
            stage_usd=candidate.template.usd_path,
            navmesh_file="",
            metrics=self._zero_metrics(),
            complexity_bucket="easy",
            reason=reason,
            layout_hash="",
            template_id=candidate.template.template_id,
        )
