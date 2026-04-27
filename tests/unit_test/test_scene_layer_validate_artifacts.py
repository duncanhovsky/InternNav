from __future__ import annotations

import json
from pathlib import Path

from dataengine.src.scene_layer.validate_artifacts import validate_many


def _mk_scene_dir(root: Path, scene_id: str, with_debug: bool = True) -> Path:
    scene_dir = root / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "stage_spec.json").write_text(
        json.dumps({"backend": "isaac_replicator", "status": "DONE"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (scene_dir / "navmesh_go2.bin").write_bytes(b"ok")
    if with_debug:
        (scene_dir / "navmesh_debug_go2.json").write_text("{}", encoding="utf-8")
    return scene_dir


def test_validate_many_passes_with_expected_files(tmp_path: Path):
    scene_dir = _mk_scene_dir(tmp_path, "scene_ok")
    summary = validate_many([{"scene_id": "scene_ok", "scene_dir": str(scene_dir)}], strict_warn=False)

    assert summary.ok is True
    assert summary.total == 1
    assert summary.failed == 0


def test_validate_many_warns_without_debug(tmp_path: Path):
    scene_dir = _mk_scene_dir(tmp_path, "scene_warn", with_debug=False)
    summary = validate_many([{"scene_id": "scene_warn", "scene_dir": str(scene_dir)}], strict_warn=False)

    assert summary.ok is True
    assert summary.warnings >= 1



def test_validate_many_fails_when_strict_warn(tmp_path: Path):
    scene_dir = _mk_scene_dir(tmp_path, "scene_strict", with_debug=False)
    summary = validate_many([{"scene_id": "scene_strict", "scene_dir": str(scene_dir)}], strict_warn=True)

    assert summary.ok is False
    assert summary.warnings >= 1


def test_validate_many_fails_when_navmesh_missing(tmp_path: Path):
    scene_dir = tmp_path / "scene_bad"
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "stage_spec.json").write_text("{}", encoding="utf-8")

    summary = validate_many([{"scene_id": "scene_bad", "scene_dir": str(scene_dir)}], strict_warn=False)

    assert summary.ok is False
    assert summary.failed == 1


def test_validate_many_checks_dynamic_tracks_when_expected(tmp_path: Path):
    scene_dir = _mk_scene_dir(tmp_path, "scene_dyn")
    track_path = scene_dir / "dynamic_tracks.jsonl"
    track_path.write_text(
        json.dumps(
            {
                "schema_version": "v1alpha",
                "contract": "dynamic_tracks.v1alpha",
                "scene_id": "scene_dyn",
                "object_id": "people_000",
                "category": "people",
                "timestamp_ns": 1,
                "position_xyz": [0.0, 0.0, 0.0],
                "velocity_xyz": [0.0, 0.0, 0.0],
                "bbox_xyz": [0.6, 0.6, 1.7],
                "yaw_deg": 0.0,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = validate_many(
        [
            {
                "scene_id": "scene_dyn",
                "scene_dir": str(scene_dir),
                "expect_dynamics": True,
                "expected_track_file": str(track_path),
            }
        ],
        strict_warn=False,
    )
    assert summary.ok is True


def test_validate_many_fails_when_dynamic_expected_but_missing(tmp_path: Path):
    scene_dir = _mk_scene_dir(tmp_path, "scene_dyn_missing")
    summary = validate_many(
        [{"scene_id": "scene_dyn_missing", "scene_dir": str(scene_dir), "expect_dynamics": True}],
        strict_warn=False,
    )
    assert summary.ok is False


def test_validate_many_fails_when_asset_driven_people_missing_animation(tmp_path: Path):
    scene_dir = _mk_scene_dir(tmp_path, "scene_dyn_asset_bad")
    track_path = scene_dir / "dynamic_tracks.jsonl"
    track_path.write_text(
        json.dumps(
            {
                "schema_version": "v1alpha",
                "contract": "dynamic_tracks.v1alpha",
                "scene_id": "scene_dyn_asset_bad",
                "object_id": "people_000",
                "category": "people",
                "timestamp_ns": 1,
                "position_xyz": [0.0, 0.0, 0.0],
                "velocity_xyz": [0.0, 0.0, 0.0],
                "bbox_xyz": [0.6, 0.6, 1.7],
                "yaw_deg": 0.0,
                "source_backend": "asset_driven",
                "motion_mode": "walk",
                "asset_relpath": "People/Characters/F_Business_02/F_Business_02.usd",
                "animation_relpath": "",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = validate_many(
        [
            {
                "scene_id": "scene_dyn_asset_bad",
                "scene_dir": str(scene_dir),
                "expect_dynamics": True,
                "expected_track_file": str(track_path),
            }
        ],
        strict_warn=False,
    )
    assert summary.ok is False


def test_validate_many_fails_when_expected_dynamic_overlay_missing(tmp_path: Path):
    scene_dir = _mk_scene_dir(tmp_path, "scene_dyn_overlay_missing")
    track_path = scene_dir / "dynamic_tracks.jsonl"
    track_path.write_text(
        json.dumps(
            {
                "schema_version": "v1alpha",
                "contract": "dynamic_tracks.v1alpha",
                "scene_id": "scene_dyn_overlay_missing",
                "object_id": "people_000",
                "category": "people",
                "timestamp_ns": 1,
                "position_xyz": [0.0, 0.0, 0.0],
                "velocity_xyz": [0.0, 0.0, 0.0],
                "bbox_xyz": [0.6, 0.6, 1.7],
                "yaw_deg": 0.0,
                "source_backend": "asset_driven",
                "motion_mode": "stand_idle",
                "asset_relpath": "People/Characters/F_Business_02/F_Business_02.usd",
                "animation_relpath": "People/Animations/stand_idle_loop.skelanim.usd",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = validate_many(
        [
            {
                "scene_id": "scene_dyn_overlay_missing",
                "scene_dir": str(scene_dir),
                "expect_dynamics": True,
                "expected_track_file": str(track_path),
                "expected_overlay_usd": str(scene_dir / "dynamic_overlay.usda"),
            }
        ],
        strict_warn=False,
    )
    assert summary.ok is False


def test_validate_many_fails_when_animation_behavior_mismatch(tmp_path: Path):
    scene_dir = _mk_scene_dir(tmp_path, "scene_dyn_anim_mismatch")
    track_path = scene_dir / "dynamic_tracks.jsonl"
    track_path.write_text(
        json.dumps(
            {
                "schema_version": "v1alpha",
                "contract": "dynamic_tracks.v1alpha",
                "scene_id": "scene_dyn_anim_mismatch",
                "object_id": "people_000",
                "category": "people",
                "timestamp_ns": 1,
                "position_xyz": [0.0, 0.0, 0.0],
                "velocity_xyz": [0.0, 0.0, 0.0],
                "bbox_xyz": [0.6, 0.6, 1.7],
                "yaw_deg": 0.0,
                "source_backend": "asset_driven",
                "motion_mode": "walk",
                "asset_relpath": "People/Characters/F_Business_02/F_Business_02.usd",
                "animation_relpath": "People/Animations/stand_walk_loop.skelanim.usd",
                "animation_behavior": "idle",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = validate_many(
        [
            {
                "scene_id": "scene_dyn_anim_mismatch",
                "scene_dir": str(scene_dir),
                "expect_dynamics": True,
                "expected_track_file": str(track_path),
            }
        ],
        strict_warn=False,
    )
    assert summary.ok is False


def test_validate_many_fails_when_expected_behavior_event_missing(tmp_path: Path):
    scene_dir = _mk_scene_dir(tmp_path, "scene_dyn_events_missing")
    track_path = scene_dir / "dynamic_tracks.jsonl"
    track_path.write_text(
        json.dumps(
            {
                "schema_version": "v1alpha",
                "contract": "dynamic_tracks.v1alpha",
                "scene_id": "scene_dyn_events_missing",
                "object_id": "people_000",
                "category": "people",
                "timestamp_ns": 1,
                "position_xyz": [0.0, 0.0, 0.0],
                "velocity_xyz": [0.0, 0.0, 0.0],
                "bbox_xyz": [0.6, 0.6, 1.7],
                "yaw_deg": 0.0,
                "source_backend": "ira_character_graph",
                "motion_mode": "walk",
                "asset_relpath": "People/Characters/F_Business_02/F_Business_02.usd",
                "animation_relpath": "People/Animations/stand_walk_loop.skelanim.usd",
                "animation_behavior": "walk",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = validate_many(
        [
            {
                "scene_id": "scene_dyn_events_missing",
                "scene_dir": str(scene_dir),
                "expect_dynamics": True,
                "expected_track_file": str(track_path),
                "expected_behavior_event_file": str(scene_dir / "behavior_events.jsonl"),
            }
        ],
        strict_warn=False,
    )
    assert summary.ok is False


def test_validate_many_passes_with_valid_behavior_event_file(tmp_path: Path):
    scene_dir = _mk_scene_dir(tmp_path, "scene_dyn_events_ok")
    track_path = scene_dir / "dynamic_tracks.jsonl"
    track_path.write_text(
        json.dumps(
            {
                "schema_version": "v1alpha",
                "contract": "dynamic_tracks.v1alpha",
                "scene_id": "scene_dyn_events_ok",
                "object_id": "people_000",
                "category": "people",
                "timestamp_ns": 1,
                "position_xyz": [0.0, 0.0, 0.0],
                "velocity_xyz": [0.0, 0.0, 0.0],
                "bbox_xyz": [0.6, 0.6, 1.7],
                "yaw_deg": 0.0,
                "source_backend": "ira_character_graph",
                "motion_mode": "stand_idle",
                "asset_relpath": "People/Characters/F_Business_02/F_Business_02.usd",
                "animation_relpath": "People/Animations/stand_idle_loop.skelanim.usd",
                "animation_behavior": "idle",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    event_path = scene_dir / "behavior_events.jsonl"
    event_path.write_text(
        json.dumps(
            {
                "schema_version": "v1alpha",
                "contract": "dynamic_events.v1alpha",
                "scene_id": "scene_dyn_events_ok",
                "object_id": "people_000",
                "category": "people",
                "timestamp_ns": 1,
                "event_type": "state_enter",
                "from_motion_mode": "",
                "to_motion_mode": "stand_idle",
                "command_name": "Idle",
                "animation_behavior": "idle",
                "source_backend": "ira_character_graph",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = validate_many(
        [
            {
                "scene_id": "scene_dyn_events_ok",
                "scene_dir": str(scene_dir),
                "expect_dynamics": True,
                "expected_track_file": str(track_path),
                "expected_behavior_event_file": str(event_path),
            }
        ],
        strict_warn=False,
    )
    assert summary.ok is True
