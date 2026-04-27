from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

try:
    import pytest
except Exception:  # noqa: BLE001
    pytest = None


def test_scene_layer_smoke_stub_manifest_contract(tmp_path: Path):
    if importlib.util.find_spec("isaacsim") is None:
        if pytest is not None:
            pytest.skip("isaacsim is not available in current environment")
        return

    root = Path(__file__).resolve().parents[2]
    out_dir = tmp_path / "out"
    cfg = {
        "schema_version": "v1alpha",
        "mode": "train",
        "asset_root": "/home/monika/dyishere/dataset/assets/isaac/isaac-sim-assets-complete-5.1.0/Assets/Isaac/5.1",
        "scene_backend": "stub",
        "target_scene_count": 1,
        "trajectories_per_scene": 1,
        "enabled_scene_types": ["warehouse"],
        "scene_type_weights": {"warehouse": 1.0},
        "scene_mode_weights": {"complete": 1.0, "modular": 0.01},
        "global_seed": 123,
        "scene_seed_offset": 0,
        "enable_dynamics": True,
        "dynamics_backend": "synthetic",
        "dynamic_steps": 5,
        "dynamic_dt_s": 0.1,
        "dynamic_num_people_min": 1,
        "dynamic_num_people_max": 1,
        "dynamic_num_objects_min": 1,
        "dynamic_num_objects_max": 1,
        "min_path_count": 2,
        "min_detour_margin_m": 0.0,
        "detour_margin_ratio_floor": 0.0,
        "min_free_space_ratio": 0.01,
        "usd_output_root": str(out_dir / "scenes" / "usd"),
        "scene_manifest_path": str(out_dir / "manifests" / "scene_manifest.jsonl"),
        "compile_log_path": str(out_dir / "logs" / "scene_compile_log.jsonl"),
        "task_manifest_path": str(out_dir / "manifests" / "trajectory_tasks.jsonl"),
    }
    cfg_path = tmp_path / "scene_smoke_config.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "-m", "dataengine.src.scene_layer", "--config-json", str(cfg_path)],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    scene_manifest = out_dir / "manifests" / "scene_manifest.jsonl"
    task_manifest = out_dir / "manifests" / "trajectory_tasks.jsonl"
    assert scene_manifest.is_file()
    assert task_manifest.is_file()

    scene_rows = [json.loads(x) for x in scene_manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
    task_rows = [json.loads(x) for x in task_manifest.read_text(encoding="utf-8").splitlines() if x.strip()]

    assert any(row.get("contract") == "scene_manifest.v1alpha" for row in scene_rows)
    assert any(row.get("contract") == "task_manifest.v1alpha" for row in task_rows)
    done_row = next((x for x in scene_rows if x.get("status") == "DONE"), None)
    assert done_row is not None
    dynamic = done_row.get("dynamic", {})
    assert dynamic.get("enabled") is True
    assert dynamic.get("sample_count", 0) > 0
    assert Path(dynamic.get("track_file", "")).is_file()
