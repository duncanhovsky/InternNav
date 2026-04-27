from __future__ import annotations

from pathlib import Path

import pytest

from dataengine.src.scene_layer.config import SceneLayerConfig
from dataengine.src.scene_layer.errors import SceneCompileError
from dataengine.src.scene_layer.metrics import RawMetricsAdapter, RawSceneMetrics, SceneMetricPolicy


def _make_cfg(tmp_path: Path) -> SceneLayerConfig:
    assets = tmp_path / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    cfg = SceneLayerConfig(asset_root=str(assets), scene_backend="stub")
    cfg.validate()
    return cfg


def test_raw_metrics_adapter_normalizes_values():
    raw = RawMetricsAdapter.from_dict(
        {
            "path_count": -3,
            "shortest_path_m": -1.0,
            "second_shortest_path_m": 2.0,
            "detour_margin_m": -0.2,
            "free_space_ratio": 1.7,
            "static_density": -0.4,
        }
    )
    assert raw.path_count == 0
    assert raw.shortest_path_m == 0.0
    assert raw.detour_margin_m == 0.0
    assert raw.free_space_ratio == 1.0
    assert raw.static_density == 0.0


def test_policy_validate_and_bucket(tmp_path: Path):
    cfg = _make_cfg(tmp_path)
    policy = SceneMetricPolicy(cfg=cfg)
    metrics = policy.validate_and_build(
        scene_id="scene_1",
        raw=RawSceneMetrics(
            path_count=10,
            shortest_path_m=6.0,
            second_shortest_path_m=8.0,
            detour_margin_m=2.0,
            free_space_ratio=0.5,
            static_density=0.4,
        ),
    )

    assert metrics.path_count == 10
    assert 0.0 <= metrics.static_complexity_score <= 1.0
    assert policy.complexity_bucket(metrics) in {"easy", "medium", "hard"}


def test_policy_rejects_under_threshold(tmp_path: Path):
    cfg = _make_cfg(tmp_path)
    cfg.min_path_count = 3
    policy = SceneMetricPolicy(cfg=cfg)

    with pytest.raises(SceneCompileError):
        policy.validate_and_build(
            scene_id="scene_reject",
            raw=RawSceneMetrics(
                path_count=1,
                shortest_path_m=10.0,
                second_shortest_path_m=11.0,
                detour_margin_m=1.0,
                free_space_ratio=0.6,
                static_density=0.2,
            ),
        )
