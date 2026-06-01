from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataengine.src.scene_layer.dynamics_layer import DynamicsLayer
from dataengine.src.scene_layer.errors import SceneCompileError


def test_dynamics_layer_generates_tracks(tmp_path: Path):
    scene_dir = tmp_path / "scene_000"
    scene_dir.mkdir(parents=True, exist_ok=True)

    layer = DynamicsLayer(
        enable_dynamics=True,
        backend="synthetic",
        timeout_s=30,
        steps=6,
        dt_s=0.1,
        num_people_min=1,
        num_people_max=1,
        num_objects_min=1,
        num_objects_max=1,
    )

    out = layer.generate(scene_id="scene_000", scene_dir=str(scene_dir), seed=7)
    assert out["enabled"] is True
    assert out["behavior_event_file"] == ""
    assert "overlay_usd" in out
    assert out["object_count"] == 2
    assert out["sample_count"] == 12
    assert Path(out["track_file"]).is_file()


def test_dynamics_layer_disabled(tmp_path: Path):
    layer = DynamicsLayer(
        enable_dynamics=False,
        backend="synthetic",
        timeout_s=30,
        steps=5,
        dt_s=0.1,
        num_people_min=1,
        num_people_max=1,
        num_objects_min=1,
        num_objects_max=1,
    )
    out = layer.generate(scene_id="scene_001", scene_dir=str(tmp_path), seed=1)
    assert out["enabled"] is False
    assert out["behavior_event_file"] == ""
    assert out["overlay_usd"] == ""
    assert out["sample_count"] == 0


def test_dynamics_layer_asset_driven_generates_asset_fields(tmp_path: Path):
    asset_root = tmp_path / "assets"
    (asset_root / "People" / "Characters" / "F_Business_02").mkdir(parents=True, exist_ok=True)
    (asset_root / "People" / "Animations").mkdir(parents=True, exist_ok=True)
    (asset_root / "Props" / "Forklift").mkdir(parents=True, exist_ok=True)

    (asset_root / "People" / "Characters" / "F_Business_02" / "F_Business_02.usd").write_text(
        "#usda", encoding="utf-8"
    )
    (asset_root / "People" / "Animations" / "stand_idle_loop.skelanim.usd").write_text(
        "#usda", encoding="utf-8"
    )
    (asset_root / "People" / "Animations" / "stand_walk_loop.skelanim.usd").write_text(
        "#usda", encoding="utf-8"
    )
    (asset_root / "Props" / "Forklift" / "forklift.usd").write_text("#usda", encoding="utf-8")

    scene_dir = tmp_path / "scene_asset"
    scene_dir.mkdir(parents=True, exist_ok=True)
    layer = DynamicsLayer(
        enable_dynamics=True,
        backend="asset_driven",
        timeout_s=30,
        steps=4,
        dt_s=0.1,
        num_people_min=1,
        num_people_max=1,
        num_objects_min=1,
        num_objects_max=1,
        asset_root=str(asset_root),
        asset_strict=True,
        people_character_relpaths=["People/Characters/F_Business_02/F_Business_02.usd"],
        people_idle_animation_relpaths=["People/Animations/stand_idle_loop.skelanim.usd"],
        people_walk_animation_relpaths=["People/Animations/stand_walk_loop.skelanim.usd"],
        vehicle_relpaths=["Props/Forklift/forklift.usd"],
        people_idle_ratio=0.0,
        vehicle_parked_ratio=0.0,
    )

    out = layer.generate(scene_id="scene_asset", scene_dir=str(scene_dir), seed=9)
    assert out["enabled"] is True
    assert out["backend"] == "asset_driven"
    assert out["behavior_event_file"] == ""
    assert "overlay_usd" in out

    rows = [x for x in Path(out["track_file"]).read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) == 8
    first = json.loads(rows[0])
    assert first["source_backend"] == "asset_driven"
    assert first["asset_relpath"] != ""
    if first["category"] == "people":
        if first["motion_mode"] == "stand_idle":
            assert first["animation_behavior"] == "idle"
        if first["motion_mode"] == "walk":
            assert first["animation_behavior"] == "walk"


def test_dynamics_layer_asset_driven_writes_overlay_without_pxr(tmp_path: Path):
    asset_root = tmp_path / "assets"
    (asset_root / "People" / "Characters" / "F_Business_02").mkdir(parents=True, exist_ok=True)
    (asset_root / "People" / "Animations").mkdir(parents=True, exist_ok=True)
    (asset_root / "Props" / "Forklift").mkdir(parents=True, exist_ok=True)

    (asset_root / "People" / "Characters" / "F_Business_02" / "F_Business_02.usd").write_text(
        "#usda 1.0\n",
        encoding="utf-8",
    )
    (asset_root / "People" / "Animations" / "stand_idle_loop.skelanim.usd").write_text(
        "#usda 1.0\n",
        encoding="utf-8",
    )
    (asset_root / "People" / "Animations" / "stand_walk_loop.skelanim.usd").write_text(
        "#usda 1.0\n",
        encoding="utf-8",
    )
    (asset_root / "Props" / "Forklift" / "forklift.usd").write_text("#usda 1.0\n", encoding="utf-8")

    stage_usd = tmp_path / "base_stage.usda"
    stage_usd.write_text("#usda 1.0\n", encoding="utf-8")

    scene_dir = tmp_path / "scene_overlay"
    scene_dir.mkdir(parents=True, exist_ok=True)

    layer = DynamicsLayer(
        enable_dynamics=True,
        backend="asset_driven",
        timeout_s=30,
        steps=3,
        dt_s=0.1,
        num_people_min=1,
        num_people_max=1,
        num_objects_min=1,
        num_objects_max=1,
        asset_root=str(asset_root),
        asset_strict=True,
        people_character_relpaths=["People/Characters/F_Business_02/F_Business_02.usd"],
        people_idle_animation_relpaths=["People/Animations/stand_idle_loop.skelanim.usd"],
        people_walk_animation_relpaths=["People/Animations/stand_walk_loop.skelanim.usd"],
        vehicle_relpaths=["Props/Forklift/forklift.usd"],
    )

    out = layer.generate(scene_id="scene_overlay", scene_dir=str(scene_dir), seed=11, stage_usd=str(stage_usd))
    overlay_path = Path(out["overlay_usd"])
    assert overlay_path.is_file()

    overlay_text = overlay_path.read_text(encoding="utf-8")
    assert "subLayers" in overlay_text
    assert "dynamic:source_backend" in overlay_text
    assert "references" in overlay_text


def test_dynamics_layer_asset_driven_strict_missing_asset_fails(tmp_path: Path):
    scene_dir = tmp_path / "scene_asset_fail"
    scene_dir.mkdir(parents=True, exist_ok=True)
    layer = DynamicsLayer(
        enable_dynamics=True,
        backend="asset_driven",
        timeout_s=30,
        steps=3,
        dt_s=0.1,
        num_people_min=1,
        num_people_max=1,
        num_objects_min=0,
        num_objects_max=0,
        asset_root=str(tmp_path / "missing_assets"),
        asset_strict=True,
        people_character_relpaths=["People/Characters/F_Business_02/F_Business_02.usd"],
        people_idle_animation_relpaths=["People/Animations/stand_idle_loop.skelanim.usd"],
        people_walk_animation_relpaths=["People/Animations/stand_walk_loop.skelanim.usd"],
        vehicle_relpaths=["Props/Forklift/forklift.usd"],
    )

    with pytest.raises(SceneCompileError):
        layer.generate(scene_id="scene_asset_fail", scene_dir=str(scene_dir), seed=1)


def test_dynamics_layer_asset_driven_binding_requires_real_animation_file(tmp_path: Path):
    asset_root = tmp_path / "assets"
    (asset_root / "People" / "Characters" / "F_Business_02").mkdir(parents=True, exist_ok=True)
    (asset_root / "People" / "Animations").mkdir(parents=True, exist_ok=True)

    (asset_root / "People" / "Characters" / "F_Business_02" / "F_Business_02.usd").write_text(
        "#usda", encoding="utf-8"
    )
    # Deliberately create only idle animation; walk animation is missing.
    (asset_root / "People" / "Animations" / "stand_idle_loop.skelanim.usd").write_text(
        "#usda", encoding="utf-8"
    )

    scene_dir = tmp_path / "scene_asset_binding_fail"
    scene_dir.mkdir(parents=True, exist_ok=True)
    layer = DynamicsLayer(
        enable_dynamics=True,
        backend="asset_driven",
        timeout_s=30,
        steps=4,
        dt_s=0.1,
        num_people_min=1,
        num_people_max=1,
        num_objects_min=0,
        num_objects_max=0,
        asset_root=str(asset_root),
        asset_strict=False,
        people_character_relpaths=["People/Characters/F_Business_02/F_Business_02.usd"],
        people_idle_animation_relpaths=["People/Animations/stand_idle_loop.skelanim.usd"],
        people_walk_animation_relpaths=["People/Animations/stand_walk_loop.skelanim.usd"],
        vehicle_relpaths=["Props/Forklift/forklift.usd"],
        people_idle_ratio=0.0,
    )

    with pytest.raises(SceneCompileError):
        layer.generate(scene_id="scene_asset_binding_fail", scene_dir=str(scene_dir), seed=13)


def test_dynamics_layer_ira_character_graph_writes_behavior_events(tmp_path: Path):
    asset_root = tmp_path / "assets"
    (asset_root / "People" / "Characters" / "F_Business_02").mkdir(parents=True, exist_ok=True)
    (asset_root / "People" / "Animations").mkdir(parents=True, exist_ok=True)
    (asset_root / "Props" / "Forklift").mkdir(parents=True, exist_ok=True)

    (asset_root / "People" / "Characters" / "F_Business_02" / "F_Business_02.usd").write_text(
        "#usda", encoding="utf-8"
    )
    (asset_root / "People" / "Animations" / "stand_idle_loop.skelanim.usd").write_text(
        "#usda", encoding="utf-8"
    )
    (asset_root / "People" / "Animations" / "stand_walk_loop.skelanim.usd").write_text(
        "#usda", encoding="utf-8"
    )
    (asset_root / "Props" / "Forklift" / "forklift.usd").write_text("#usda", encoding="utf-8")

    scene_dir = tmp_path / "scene_ira"
    scene_dir.mkdir(parents=True, exist_ok=True)
    layer = DynamicsLayer(
        enable_dynamics=True,
        backend="ira_character_graph",
        timeout_s=30,
        steps=10,
        dt_s=0.1,
        num_people_min=1,
        num_people_max=1,
        num_objects_min=1,
        num_objects_max=1,
        asset_root=str(asset_root),
        asset_strict=True,
        people_character_relpaths=["People/Characters/F_Business_02/F_Business_02.usd"],
        people_idle_animation_relpaths=["People/Animations/stand_idle_loop.skelanim.usd"],
        people_walk_animation_relpaths=["People/Animations/stand_walk_loop.skelanim.usd"],
        vehicle_relpaths=["Props/Forklift/forklift.usd"],
        people_idle_ratio=0.0,
    )

    out = layer.generate(scene_id="scene_ira", scene_dir=str(scene_dir), seed=42)
    assert out["enabled"] is True
    assert out["backend"] == "ira_character_graph"
    assert Path(out["track_file"]).is_file()
    assert Path(out["behavior_event_file"]).is_file()

    event_rows = [x for x in Path(out["behavior_event_file"]).read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(event_rows) > 0
    event0 = json.loads(event_rows[0])
    assert event0["contract"] == "dynamic_events.v1alpha"
    assert event0["event_type"] == "state_enter"
