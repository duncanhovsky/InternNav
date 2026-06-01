from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict

from .composer import SceneComposer
from .config import SceneLayerConfig
from .dynamics_layer import DynamicsLayer
from .errors import SceneCompileError
from .metrics import MetricsEvaluator
from .models import SceneCandidate, SceneCompileResult, SceneMetrics
from .navmesh import NavMeshBuilder

logger = logging.getLogger(__name__)


def _build_ira_dynamic_layer(cfg: SceneLayerConfig):
    """尝试构建基于 IRA 的动态层。

    仅当 dynamics_backend == 'ira' 时启用。
    """
    try:
        from dataengine.src.dynamic_layer.config import DynamicLayerConfig
        from dataengine.src.dynamic_layer.dynamic_layer import IRADynamicLayer

        ira_cfg = DynamicLayerConfig(
            enable_dynamics=cfg.enable_dynamics,
            backend="ira",
            seed=cfg.global_seed,
            simulation_length=getattr(cfg, "ira_simulation_length", 300),
            character_asset_path=getattr(cfg, "ira_character_asset_path", ""),
            character_filters=list(getattr(cfg, "ira_character_filters", ["male", "medical"])),
            character_num_min=cfg.dynamic_num_people_min,
            character_num_max=cfg.dynamic_num_people_max,
            character_spawn_area=list(getattr(cfg, "ira_character_spawn_area", ["Walkable"])),
            character_navigation_area=list(getattr(cfg, "ira_character_navigation_area", ["Walkable"])),
            idle_ratio=cfg.dynamic_people_idle_ratio,
            max_commands_per_character=getattr(cfg, "ira_max_commands_per_character", 5),
            camera_num=getattr(cfg, "ira_camera_num", 5),
            headless=cfg.isaac_headless,
            timeout_s=cfg.dynamics_timeout_s,
            isaac_sim_python=getattr(cfg, "ira_isaac_sim_python", ""),
            fallback_to_synthetic=getattr(cfg, "ira_fallback_to_synthetic", True),
            output_dir=getattr(cfg, "ira_output_dir", ""),
        )
        ira_cfg.validate()
        return IRADynamicLayer(cfg=ira_cfg)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to build IRA dynamic layer, using legacy: %s", exc)
        return None


@dataclass
class SceneCompiler:
    """单场景编译器。

    职责：
    1. 调用 composer 生成场景 stage。
    2. 调用 navmesh 构建与评估。
    3. 调用动态层生成 NPC 行为（支持 IRA 或旧合成后端）。
    4. 调用 metrics 进行质量阈值检查并输出复杂度桶。
    5. 对可重试错误执行有限重试，返回结构化 SceneCompileResult。
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

        # 根据 dynamics_backend 选择动态层实现
        self._ira_layer = None
        if self.cfg.dynamics_backend == "ira":
            self._ira_layer = _build_ira_dynamic_layer(self.cfg)

        # 旧合成后端（作为 fallback 或默认）
        self.dynamics_layer = DynamicsLayer(
            enable_dynamics=self.cfg.enable_dynamics,
            backend=self.cfg.dynamics_backend,
            timeout_s=self.cfg.dynamics_timeout_s,
            steps=self.cfg.dynamic_steps,
            dt_s=self.cfg.dynamic_dt_s,
            num_people_min=self.cfg.dynamic_num_people_min,
            num_people_max=self.cfg.dynamic_num_people_max,
            num_objects_min=self.cfg.dynamic_num_objects_min,
            num_objects_max=self.cfg.dynamic_num_objects_max,
            asset_root=self.cfg.asset_root,
            asset_strict=self.cfg.dynamic_asset_strict,
            people_character_relpaths=list(self.cfg.dynamic_people_character_relpaths),
            people_idle_animation_relpaths=list(self.cfg.dynamic_people_idle_animation_relpaths),
            people_walk_animation_relpaths=list(self.cfg.dynamic_people_walk_animation_relpaths),
            vehicle_relpaths=list(self.cfg.dynamic_vehicle_relpaths),
            require_animation_binding=self.cfg.dynamic_require_animation_binding,
            people_idle_ratio=self.cfg.dynamic_people_idle_ratio,
            vehicle_parked_ratio=self.cfg.dynamic_vehicle_parked_ratio,
        )

    def _run_dynamics(
        self,
        scene_id: str,
        scene_dir: str,
        seed: int,
        stage_usd: str,
    ) -> Dict[str, Any]:
        """执行动态层，优先使用 IRA 后端，失败时退化到旧后端。

        返回兼容旧 DynamicsLayer.generate() 格式的字典。
        """
        if self._ira_layer is not None:
            logger.info("Using IRA dynamic layer for scene %s", scene_id)
            return self._ira_layer.generate_legacy_dict(
                scene_id=scene_id,
                scene_dir=scene_dir,
                seed=seed,
                stage_usd=stage_usd,
            )

        # 旧合成后端
        return self.dynamics_layer.generate(
            scene_id=scene_id,
            scene_dir=scene_dir,
            seed=seed,
            stage_usd=stage_usd,
        )

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
        dynamics = self._run_dynamics(
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
            schema_version=self.cfg.schema_version,
            navmesh_metrics_by_profile=dict(navmesh.get("metrics_by_profile", {})),
            navmesh_files_by_profile=dict(navmesh.get("navmesh_files_by_profile", {})),
            navmesh_debug_by_profile=dict(navmesh.get("debug_files_by_profile", {})),
            dynamic_enabled=bool(dynamics.get("enabled", False)),
            dynamic_backend=str(dynamics.get("backend", "")),
            dynamic_track_file=str(dynamics.get("track_file", "")),
            dynamic_behavior_event_file=str(dynamics.get("behavior_event_file", "")),
            dynamic_overlay_usd=str(dynamics.get("overlay_usd", "")),
            dynamic_object_count=int(dynamics.get("object_count", 0)),
            dynamic_sample_count=int(dynamics.get("sample_count", 0)),
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
                schema_version=self.cfg.schema_version,
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
            schema_version=self.cfg.schema_version,
        )
