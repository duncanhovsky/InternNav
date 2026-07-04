import json
import shutil
from pathlib import Path


def _make_minimal_scene(root: Path) -> Path:
    scene = root / "hm3d_d435i" / "scene_000"
    chunk = "chunk-000"
    (scene / "data" / chunk).mkdir(parents=True)
    (scene / "videos" / chunk / "observation.images.rgb").mkdir(parents=True)
    (scene / "videos" / chunk / "observation.images.depth").mkdir(parents=True)
    (scene / "meta").mkdir(parents=True)
    (scene / "data" / chunk / "episode_000.parquet").write_bytes(b"parquet")
    (scene / "videos" / chunk / "observation.images.rgb" / "000.png").write_bytes(b"rgb")
    (scene / "videos" / chunk / "observation.images.depth" / "000.png").write_bytes(b"depth")
    (scene / "meta" / "episodes_stats.jsonl").write_text('{"episode_index": 0}\n', encoding="utf-8")
    (scene / "meta" / "pointcloud.ply").write_bytes(b"ply")
    return scene


def test_1p5tb_profile_uses_600gb_slots_and_low_nvme_mode():
    from scripts.cache_rotation.cache_rotation_lib import resolve_profile

    profile = resolve_profile(gpus=4, nvme_size="1.5tb", preset="balanced")

    assert profile.nvme_size == "1.5tb"
    assert profile.cache_slot_gb == 600
    assert profile.low_nvme_mode is True

    alias_profile = resolve_profile(gpus=4, nvme_size="1500gb", preset="balanced")
    assert alias_profile.nvme_size == "1.5tb"


def test_low_nvme_cache_build_removes_old_slot_before_copy(tmp_path, monkeypatch):
    from scripts.cache_rotation import cache_rotation_lib

    hdd_traj = tmp_path / "hdd" / "traj_data"
    scene = _make_minimal_scene(hdd_traj)
    manifest = {
        "version": 1,
        "cache_slot_gb": 600,
        "summary": {"total_scenes": 1, "total_episodes": 1, "total_bytes": 1, "num_shards": 1},
        "shards": [
            {
                "shard_index": 0,
                "scene_count": 1,
                "episode_count": 1,
                "bytes": 1,
                "scenes": [
                    {
                        "group": "hm3d_d435i",
                        "scene": "scene_000",
                        "rel_path": scene.relative_to(hdd_traj).as_posix(),
                        "bytes": 1,
                        "episodes": 1,
                    }
                ],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    slot = tmp_path / "cache_B"
    slot.mkdir()
    (slot / ".READY").write_text("old", encoding="utf-8")
    (slot / "old_payload").write_text("must be removed before copy", encoding="utf-8")

    original_copytree = shutil.copytree
    observed_slot_absent = []

    def copytree_with_peak_check(src, dst, *args, **kwargs):
        observed_slot_absent.append(not slot.exists())
        return original_copytree(src, dst, *args, **kwargs)

    monkeypatch.setattr(cache_rotation_lib.shutil, "copytree", copytree_with_peak_check)

    cache_rotation_lib.build_cache_slot(
        manifest_path=manifest_path,
        shard_index=0,
        hdd_traj=hdd_traj,
        slot_root=slot,
        force=True,
        drop_existing_before_build=True,
    )

    assert observed_slot_absent
    assert all(observed_slot_absent)
    assert not (slot / "old_payload").exists()
    assert (slot / ".READY").exists()
