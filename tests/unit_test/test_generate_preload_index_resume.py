import importlib.util
import json
from pathlib import Path


def _load_generator_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "dataset"
        / "generate_preload_index.py"
    )
    spec = importlib.util.spec_from_file_location(
        "generate_preload_index_resume", script_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_scene(root: Path, group: str, scene: str) -> None:
    unit = root / group / scene
    (unit / "data" / "chunk-000").mkdir(parents=True)
    (unit / "meta").mkdir()
    (unit / "videos" / "chunk-000" / "observation.images.rgb").mkdir(parents=True)
    (unit / "videos" / "chunk-000" / "observation.images.depth").mkdir(parents=True)

    (unit / "data" / "chunk-000" / "episode_000000.parquet").touch()
    (unit / "videos" / "chunk-000" / "observation.images.rgb" / "000000.png").touch()
    (unit / "videos" / "chunk-000" / "observation.images.depth" / "000000.png").touch()
    (unit / "meta" / "pointcloud.ply").touch()
    with open(unit / "meta" / "episodes_stats.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"episode_index": 0}) + "\n")


def test_resumable_preload_index_continues_from_first_incomplete_unit(tmp_path):
    module = _load_generator_module()
    root = tmp_path / "traj_data"
    _write_scene(root, "group_a", "scene_001")
    _write_scene(root, "group_a", "scene_002")

    output = tmp_path / "preload_index.json"
    work_dir = tmp_path / "preload_index.work"

    partial = module.generate_preload_index_resumable(
        root_dir=str(root),
        output_path=output,
        work_dir=work_dir,
        max_units=1,
    )

    assert partial["complete"] is False
    assert partial["completed_units"] == 1
    assert partial["next_unit"] == "group_a/scene_002"
    assert not output.exists()

    progress = json.loads((work_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress["last_completed"] == "group_a/scene_001"
    assert progress["next_unit"] == "group_a/scene_002"

    final = module.generate_preload_index_resumable(
        root_dir=str(root),
        output_path=output,
        work_dir=work_dir,
    )

    assert final["complete"] is True
    assert final["completed_units"] == 2
    assert final["total_episodes"] == 2

    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["trajectory_data_dir"] == [
        str(root / "group_a" / "scene_001" / "data" / "chunk-000" / "episode_000000.parquet"),
        str(root / "group_a" / "scene_002" / "data" / "chunk-000" / "episode_000000.parquet"),
    ]
    assert len(data["trajectory_rgb_path"]) == 2
    assert len(data["trajectory_depth_path"]) == 2
    assert len(data["trajectory_afford_path"]) == 2


def test_resumable_preload_index_does_not_resummarize_all_units_after_each_scan(tmp_path, monkeypatch):
    module = _load_generator_module()
    root = tmp_path / "traj_data"
    for idx in range(6):
        _write_scene(root, "group_a", f"scene_{idx:03d}")

    output = tmp_path / "preload_index.json"
    work_dir = tmp_path / "preload_index.work"

    load_calls = 0
    original_load_completed_shard = module._load_completed_shard

    def counted_load_completed_shard(*args, **kwargs):
        nonlocal load_calls
        load_calls += 1
        return original_load_completed_shard(*args, **kwargs)

    monkeypatch.setattr(module, "_load_completed_shard", counted_load_completed_shard)

    module.generate_preload_index_resumable(
        root_dir=str(root),
        output_path=output,
        work_dir=work_dir,
        max_units=3,
    )

    assert load_calls <= 12
