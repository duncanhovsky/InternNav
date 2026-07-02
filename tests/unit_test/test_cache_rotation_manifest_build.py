import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cache_rotation.cache_rotation_lib import (
    build_cache_slot,
    discover_scenes,
    load_manifest,
    validate_cache_slot,
    write_shard_manifest,
)


def _write_fake_episode(unit: Path, episode_id: int = 0):
    data_dir = unit / "data" / "chunk-000"
    meta_dir = unit / "meta"
    rgb_dir = unit / "videos" / "chunk-000" / "observation.images.rgb"
    depth_dir = unit / "videos" / "chunk-000" / "observation.images.depth"
    for path in (data_dir, meta_dir, rgb_dir, depth_dir):
        path.mkdir(parents=True, exist_ok=True)

    (data_dir / f"episode_{episode_id:06d}.parquet").write_bytes(b"parquet")
    (meta_dir / "episodes_stats.jsonl").write_text(
        '{"image_index":{"min":0,"max":1}}\n',
        encoding="utf-8",
    )
    (meta_dir / "pointcloud.ply").write_text("ply\n", encoding="utf-8")
    for idx in range(2):
        (rgb_dir / f"{idx:06d}.png").write_bytes(b"rgb")
        (depth_dir / f"{idx:06d}.png").write_bytes(b"depth")


def test_manifest_discovers_groups_and_builds_ready_cache_slot(tmp_path):
    hdd_traj = tmp_path / "hdd" / "datasets" / "InternData-N1" / "v0.5-full-vln-n1" / "vln_n1" / "traj_data"
    _write_fake_episode(hdd_traj / "hssd_zed" / "scene_000" / "trajectory_00")
    _write_fake_episode(hdd_traj / "hm3d_d435i" / "scene_001" / "trajectory_00")

    scenes = discover_scenes(hdd_traj)
    assert {scene.group for scene in scenes} == {"hssd_zed", "hm3d_d435i"}

    manifest_path = tmp_path / "manifests" / "shards.json"
    manifest = write_shard_manifest(
        scenes=scenes,
        output_path=manifest_path,
        cache_slot_gb=1,
        max_scenes_per_shard=1,
        seed=7,
    )

    loaded = load_manifest(manifest_path)
    assert loaded["summary"]["total_scenes"] == 2
    assert loaded["summary"]["total_episodes"] == 2
    assert len(loaded["shards"]) == len(manifest["shards"]) == 2

    slot_root = tmp_path / "nvme" / "bridgedp_cache" / "cache_A"
    metadata = build_cache_slot(
        manifest_path=manifest_path,
        shard_index=0,
        hdd_traj=hdd_traj,
        slot_root=slot_root,
        force=True,
    )

    assert (slot_root / ".READY").exists()
    assert (slot_root / "preload_index.json").exists()
    assert metadata["status"] == "READY"
    assert metadata["shard_index"] == 0

    report = validate_cache_slot(slot_root)
    assert report.ok, report.examples
    preload = json.loads((slot_root / "preload_index.json").read_text(encoding="utf-8"))
    assert len(preload["trajectory_data_dir"]) == 1
    assert preload["trajectory_data_dir"][0].startswith(str(slot_root))
