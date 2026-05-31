from __future__ import annotations

from pathlib import Path

from dataengine.src.scene_layer.composer import SceneComposer
from dataengine.src.scene_layer.config import SceneLayerConfig
from dataengine.src.scene_layer.models import SceneCandidate, SceneCompileResult, SceneMetrics, SceneTemplate
from dataengine.src.scene_layer.navmesh import NavMeshBuilder
from dataengine.src.scene_layer.pipeline import SceneLayerPipeline


def _make_cfg(tmp_path: Path) -> SceneLayerConfig:
    out_root = tmp_path / "out"
    cfg = SceneLayerConfig(
        asset_root=str(tmp_path / "assets"),
        scene_backend="stub",
        target_scene_count=1,
        trajectories_per_scene=2,
        usd_output_root=str(out_root / "usd"),
        scene_manifest_path=str(out_root / "manifests" / "scene_manifest.jsonl"),
        compile_log_path=str(out_root / "logs" / "scene_compile_log.jsonl"),
        task_manifest_path=str(out_root / "manifests" / "trajectory_tasks.jsonl"),
    )
    cfg.validate()
    return cfg


def test_stub_layout_hash_is_deterministic(tmp_path: Path):
    assets_root = tmp_path / "assets"
    assets_root.mkdir(parents=True, exist_ok=True)
    template_path = assets_root / "template.usda"
    template_path.write_text("#usda 1.0\n", encoding="utf-8")

    cfg = _make_cfg(tmp_path)
    composer = SceneComposer(cfg=cfg)

    template = SceneTemplate(
        template_id="tmp_template",
        scene_type="warehouse",
        usd_path=str(template_path),
        supports_complete_mode=True,
    )
    candidate = SceneCandidate(
        scene_id="scene_000001_warehouse_complete",
        scene_type="warehouse",
        mode="complete",
        seed=20260419,
        template=template,
    )

    first = composer.compose(candidate)
    second = composer.compose(candidate)

    assert first["layout_hash"] == second["layout_hash"]


def test_manifest_rows_include_v1alpha_contract_fields(tmp_path: Path):
    (tmp_path / "assets").mkdir(parents=True, exist_ok=True)
    pipeline = SceneLayerPipeline(cfg=_make_cfg(tmp_path))

    result = SceneCompileResult(
        scene_id="scene_000001_warehouse_complete",
        scene_type="warehouse",
        mode="complete",
        seed=11,
        status="DONE",
        stage_usd="/tmp/stage.usda",
        navmesh_file="/tmp/navmesh_go2.bin",
        metrics=SceneMetrics(
            path_count=8,
            shortest_path_m=10.0,
            second_shortest_path_m=12.0,
            detour_margin_m=2.0,
            free_space_ratio=0.5,
            static_density=0.3,
            static_complexity_score=0.42,
        ),
        complexity_bucket="medium",
        template_id="tmp_template",
        schema_version="v1alpha",
        navmesh_metrics_by_profile={"go2": {"path_count": 8}},
        navmesh_files_by_profile={"go2": "/tmp/navmesh_go2.bin"},
        navmesh_debug_by_profile={"go2": "/tmp/navmesh_debug_go2.json"},
    )

    row = pipeline._result_to_row(result)
    assert row["schema_version"] == "v1alpha"
    assert row["contract"] == "scene_manifest.v1alpha"
    assert "navmesh" in row
    assert "metrics_by_profile" in row["navmesh"]

    task = pipeline._build_tasks_for_scene(row)[0]
    task_row = pipeline._task_to_row(task, row)
    assert task_row["schema_version"] == "v1alpha"
    assert task_row["contract"] == "task_manifest.v1alpha"
    assert task_row["scene_layout_hash"] == row["layout_hash"]


def test_navmesh_profile_metrics_aggregation_is_conservative():
    metrics_by_profile = {
        "go2": {
            "path_count": 10,
            "shortest_path_m": 9.0,
            "second_shortest_path_m": 12.0,
            "detour_margin_m": 3.0,
            "free_space_ratio": 0.45,
            "static_density": 0.33,
        },
        "g1": {
            "path_count": 6,
            "shortest_path_m": 8.0,
            "second_shortest_path_m": 11.0,
            "detour_margin_m": 2.2,
            "free_space_ratio": 0.39,
            "static_density": 0.41,
        },
    }

    agg = NavMeshBuilder._aggregate_profile_metrics(metrics_by_profile)
    assert agg["path_count"] == 6
    assert agg["shortest_path_m"] == 8.0
    assert agg["detour_margin_m"] == 2.2
    assert agg["free_space_ratio"] == 0.39
    assert agg["static_density"] == 0.41
