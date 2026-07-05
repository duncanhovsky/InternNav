#!/usr/bin/env python3
"""Core helpers for Bridge-DP cache rotation.

The functions here intentionally use only the Python standard library so the
dataset/cache preparation path can run before the full training environment is
installed.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


PRELOAD_REPEAT_FACTOR = 50
DEFAULT_TRAJ_SUFFIX = Path("datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data")
SLOT_TRAJ_SUFFIX = Path("vln_n1/traj_data")


@dataclass(frozen=True)
class RotationProfile:
    gpus: int
    nvme_size: str
    preset: str
    cache_slot_gb: int
    shard_epochs: int
    per_gpu_batch: int
    num_workers: int
    build_workers: int
    low_nvme_mode: bool = False
    grad_accum: int = 1
    save_steps: int = 500
    validation_level: str = "fast"


@dataclass(frozen=True)
class TrainingPlan:
    total_max_steps: int
    stage_steps: int
    stage_end_step: int
    global_batch: int


@dataclass(frozen=True)
class SceneRecord:
    group: str
    scene: str
    rel_path: str
    bytes: int
    episodes: int

    def as_dict(self) -> dict:
        return {
            "group": self.group,
            "scene": self.scene,
            "rel_path": self.rel_path,
            "bytes": self.bytes,
            "episodes": self.episodes,
        }


@dataclass
class ValidationReport:
    ok: bool
    missing_paths: int = 0
    bad_shape: int = 0
    marker_ready: bool = False
    examples: list[str] = field(default_factory=list)


_PROFILE_TABLE = {
    (4, "1tb", "balanced"): dict(cache_slot_gb=320, shard_epochs=3, num_workers=10, build_workers=3),
    (4, "1.5tb", "balanced"): dict(cache_slot_gb=600, shard_epochs=2, num_workers=10, build_workers=4, low_nvme_mode=True),
    (4, "2tb", "balanced"): dict(cache_slot_gb=750, shard_epochs=2, num_workers=10, build_workers=4),
    (4, "4tb", "balanced"): dict(cache_slot_gb=1600, shard_epochs=1, num_workers=10, build_workers=4),
    (8, "1tb", "balanced"): dict(cache_slot_gb=320, shard_epochs=5, num_workers=8, build_workers=4),
    (8, "1.5tb", "balanced"): dict(cache_slot_gb=600, shard_epochs=4, num_workers=8, build_workers=5, low_nvme_mode=True),
    (8, "2tb", "balanced"): dict(cache_slot_gb=750, shard_epochs=4, num_workers=8, build_workers=5),
    (8, "4tb", "balanced"): dict(cache_slot_gb=1600, shard_epochs=2, num_workers=8, build_workers=6),
    (4, "1tb", "quality"): dict(cache_slot_gb=320, shard_epochs=2, num_workers=10, build_workers=3),
    (4, "1.5tb", "quality"): dict(cache_slot_gb=600, shard_epochs=1, num_workers=10, build_workers=4, low_nvme_mode=True),
    (4, "2tb", "quality"): dict(cache_slot_gb=750, shard_epochs=1, num_workers=10, build_workers=4),
    (4, "4tb", "quality"): dict(cache_slot_gb=1600, shard_epochs=1, num_workers=10, build_workers=4),
    (8, "1tb", "quality"): dict(cache_slot_gb=320, shard_epochs=4, num_workers=8, build_workers=4),
    (8, "1.5tb", "quality"): dict(cache_slot_gb=600, shard_epochs=3, num_workers=8, build_workers=5, low_nvme_mode=True),
    (8, "2tb", "quality"): dict(cache_slot_gb=750, shard_epochs=2, num_workers=8, build_workers=5),
    (8, "4tb", "quality"): dict(cache_slot_gb=1600, shard_epochs=2, num_workers=8, build_workers=6),
    (4, "1tb", "throughput"): dict(cache_slot_gb=320, shard_epochs=4, num_workers=10, build_workers=4),
    (4, "1.5tb", "throughput"): dict(cache_slot_gb=600, shard_epochs=3, num_workers=10, build_workers=4, low_nvme_mode=True),
    (4, "2tb", "throughput"): dict(cache_slot_gb=750, shard_epochs=3, num_workers=10, build_workers=4),
    (4, "4tb", "throughput"): dict(cache_slot_gb=1600, shard_epochs=2, num_workers=10, build_workers=5),
    (8, "1tb", "throughput"): dict(cache_slot_gb=320, shard_epochs=6, num_workers=8, build_workers=5),
    (8, "1.5tb", "throughput"): dict(cache_slot_gb=600, shard_epochs=5, num_workers=8, build_workers=5, low_nvme_mode=True),
    (8, "2tb", "throughput"): dict(cache_slot_gb=750, shard_epochs=5, num_workers=8, build_workers=6),
    (8, "4tb", "throughput"): dict(cache_slot_gb=1600, shard_epochs=3, num_workers=8, build_workers=6),
}


def normalize_nvme_size(value: str) -> str:
    text = str(value).strip().lower().replace(" ", "")
    aliases = {
        "1.5t": "1.5tb",
        "1.5tb": "1.5tb",
        "1500g": "1.5tb",
        "1500gb": "1.5tb",
    }
    if text in aliases:
        return aliases[text]
    if text.endswith("t"):
        text += "b"
    if text not in {"1tb", "1.5tb", "2tb", "4tb"}:
        raise ValueError(f"unsupported nvme size: {value}; expected 1tb, 1.5tb, 2tb, or 4tb")
    return text


def resolve_profile(gpus: int, nvme_size: str, preset: str = "balanced") -> RotationProfile:
    preset = str(preset).strip().lower()
    key = (int(gpus), normalize_nvme_size(nvme_size), preset)
    if key not in _PROFILE_TABLE:
        raise ValueError(f"unsupported cache rotation profile: gpus={gpus}, nvme_size={nvme_size}, preset={preset}")
    values = dict(_PROFILE_TABLE[key])
    return RotationProfile(
        gpus=key[0],
        nvme_size=key[1],
        preset=key[2],
        per_gpu_batch=96,
        grad_accum=1,
        save_steps=500,
        validation_level="fast",
        **values,
    )


def compute_training_plan(
    *,
    total_episodes: int,
    shard_episodes: int,
    total_epochs: int,
    shard_epochs: int,
    gpus: int,
    per_gpu_batch: int,
    grad_accum: int,
    current_global_step: int,
) -> TrainingPlan:
    global_batch = int(gpus) * int(per_gpu_batch) * int(grad_accum)
    if global_batch <= 0:
        raise ValueError("global batch must be positive")
    total_max_steps = math.ceil(PRELOAD_REPEAT_FACTOR * int(total_episodes) * int(total_epochs) / global_batch)
    stage_steps = math.ceil(PRELOAD_REPEAT_FACTOR * int(shard_episodes) * int(shard_epochs) / global_batch)
    stage_end_step = min(int(current_global_step) + stage_steps, total_max_steps)
    return TrainingPlan(
        total_max_steps=max(total_max_steps, 1),
        stage_steps=max(stage_steps, 1),
        stage_end_step=max(stage_end_step, 1),
        global_batch=global_batch,
    )


def directory_size(path: Path) -> int:
    total = 0
    stack = [os.fspath(path)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        pass
        except OSError:
            pass
    return total


def _scene_record_from_dict(payload: dict) -> SceneRecord:
    return SceneRecord(
        group=str(payload["group"]),
        scene=str(payload["scene"]),
        rel_path=str(payload["rel_path"]),
        bytes=int(payload.get("bytes", 0)),
        episodes=int(payload.get("episodes", 0)),
    )


def _scene_inventory_file(work_dir: Path, group: str, scene: str) -> Path:
    return Path(work_dir) / "scenes" / group / f"{scene}.json"


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def discover_scenes_resumable(
    hdd_traj: Path,
    *,
    work_dir: Path,
    log_progress: bool = True,
) -> list[SceneRecord]:
    """Discover scenes while checkpointing each scene inventory record.

    Large shared filesystems can interrupt long metadata scans. This keeps one
    JSON file per completed scene so reruns resume from the last completed scene
    instead of restarting the whole dataset walk.
    """
    root = Path(hdd_traj)
    if not root.is_dir():
        raise FileNotFoundError(f"missing HDD trajectory root: {root}")

    scene_dirs: list[Path] = []
    for group_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        scene_dirs.extend(sorted(path for path in group_dir.iterdir() if path.is_dir()))

    scenes: list[SceneRecord] = []
    total = len(scene_dirs)
    for index, scene_dir in enumerate(scene_dirs, start=1):
        group = scene_dir.parent.name
        scene_name = scene_dir.name
        rel_path = scene_dir.relative_to(root).as_posix()
        inventory = _scene_inventory_file(work_dir, group, scene_name)

        if inventory.exists():
            try:
                payload = json.loads(inventory.read_text(encoding="utf-8"))
                if payload.get("skipped"):
                    if log_progress:
                        print(f"[manifest-resume] skip cached empty scene {index}/{total}: {rel_path}", flush=True)
                    continue
                record = _scene_record_from_dict(payload)
                scenes.append(record)
                if log_progress:
                    print(f"[manifest-resume] cached scene {index}/{total}: {rel_path} episodes={record.episodes}", flush=True)
                continue
            except (OSError, json.JSONDecodeError, KeyError, ValueError):
                inventory.unlink(missing_ok=True)

        if log_progress:
            print(f"[manifest-resume] scan scene {index}/{total}: {rel_path}", flush=True)
        episodes = count_scene_episodes(scene_dir)
        if episodes <= 0:
            _write_json_atomic(
                inventory,
                {
                    "group": group,
                    "scene": scene_name,
                    "rel_path": rel_path,
                    "bytes": 0,
                    "episodes": 0,
                    "skipped": True,
                },
            )
            continue

        record = SceneRecord(
            group=group,
            scene=scene_name,
            rel_path=rel_path,
            bytes=directory_size(scene_dir),
            episodes=episodes,
        )
        _write_json_atomic(inventory, record.as_dict())
        scenes.append(record)
        if log_progress:
            print(
                f"[manifest-resume] done scene {index}/{total}: {rel_path} "
                f"episodes={record.episodes} bytes={record.bytes}",
                flush=True,
            )

    summary = {
        "hdd_traj": str(root),
        "total_scene_dirs": total,
        "trainable_scenes": len(scenes),
        "total_episodes": sum(scene.episodes for scene in scenes),
        "total_bytes": sum(scene.bytes for scene in scenes),
    }
    _write_json_atomic(Path(work_dir) / "summary.json", summary)
    return scenes


def count_scene_episodes(scene_path: Path) -> int:
    trajectory_dirs = sorted(p for p in scene_path.iterdir() if p.is_dir() and p.name.startswith("trajectory_"))
    units = trajectory_dirs if trajectory_dirs else [scene_path]
    count = 0
    for unit in units:
        data_dir = unit / "data"
        if not data_dir.is_dir():
            continue
        count += sum(1 for path in data_dir.glob("chunk-*/*.parquet") if path.is_file())
    return count


def discover_scenes(hdd_traj: Path) -> list[SceneRecord]:
    root = Path(hdd_traj)
    if not root.is_dir():
        raise FileNotFoundError(f"missing HDD trajectory root: {root}")

    scenes = []
    for group_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for scene_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            episodes = count_scene_episodes(scene_dir)
            if episodes <= 0:
                continue
            rel_path = scene_dir.relative_to(root).as_posix()
            scenes.append(
                SceneRecord(
                    group=group_dir.name,
                    scene=scene_dir.name,
                    rel_path=rel_path,
                    bytes=directory_size(scene_dir),
                    episodes=episodes,
                )
            )
    return scenes


def _group_balanced_order(scenes: list[SceneRecord], seed: int) -> list[SceneRecord]:
    by_group: dict[str, list[SceneRecord]] = {}
    rng = random.Random(seed)
    for scene in scenes:
        by_group.setdefault(scene.group, []).append(scene)
    for group_scenes in by_group.values():
        group_scenes.sort(key=lambda item: item.scene)
        rng.shuffle(group_scenes)

    ordered = []
    groups = sorted(by_group)
    while any(by_group.values()):
        for group in groups:
            if by_group[group]:
                ordered.append(by_group[group].pop(0))
    return ordered


def write_shard_manifest(
    *,
    scenes: list[SceneRecord],
    output_path: Path,
    cache_slot_gb: int,
    max_scenes_per_shard: int | None = None,
    seed: int = 1234,
) -> dict:
    target_bytes = int(cache_slot_gb) * 1024**3
    ordered = _group_balanced_order(list(scenes), seed=seed)
    shards: list[dict] = []
    current: list[SceneRecord] = []
    current_bytes = 0

    def flush() -> None:
        nonlocal current, current_bytes
        if not current:
            return
        shard_index = len(shards)
        shards.append(
            {
                "shard_index": shard_index,
                "scene_count": len(current),
                "episode_count": sum(item.episodes for item in current),
                "bytes": sum(item.bytes for item in current),
                "scenes": [item.as_dict() for item in current],
            }
        )
        current = []
        current_bytes = 0

    for scene in ordered:
        would_exceed_bytes = current and current_bytes + scene.bytes > target_bytes
        would_exceed_count = max_scenes_per_shard is not None and current and len(current) >= max_scenes_per_shard
        if would_exceed_bytes or would_exceed_count:
            flush()
        current.append(scene)
        current_bytes += scene.bytes
    flush()

    manifest = {
        "version": 1,
        "cache_slot_gb": int(cache_slot_gb),
        "summary": {
            "total_scenes": len(scenes),
            "total_episodes": sum(item.episodes for item in scenes),
            "total_bytes": sum(item.bytes for item in scenes),
            "num_shards": len(shards),
        },
        "shards": shards,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def load_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def shard_signature(shard: dict) -> str:
    payload = [
        {
            "rel_path": scene["rel_path"],
            "bytes": scene.get("bytes", 0),
            "episodes": scene.get("episodes", 0),
        }
        for scene in shard.get("scenes", [])
    ]
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _process_data_unit(unit_path: Path) -> dict:
    result = {
        "trajectory_data_dir": [],
        "trajectory_rgb_path": [],
        "trajectory_depth_path": [],
        "trajectory_afford_path": [],
    }
    data_path = unit_path / "data"
    meta_path = unit_path / "meta" / "episodes_stats.jsonl"
    afford_path = unit_path / "meta" / "pointcloud.ply"
    if not data_path.exists() or not meta_path.exists():
        return result

    chunks = sorted(path.name for path in data_path.iterdir() if path.is_dir() and path.name.startswith("chunk-"))
    if not chunks:
        return result
    chunk_name = chunks[0]
    data_dir = unit_path / "data" / chunk_name
    rgb_dir = unit_path / "videos" / chunk_name / "observation.images.rgb"
    depth_dir = unit_path / "videos" / chunk_name / "observation.images.depth"
    if not rgb_dir.is_dir() or not depth_dir.is_dir():
        return result

    episode_info = _read_jsonl(meta_path)
    rgb_paths = [str(path) for path in sorted(rgb_dir.iterdir()) if path.is_file()]
    depth_paths = [str(path) for path in sorted(depth_dir.iterdir()) if path.is_file()]
    data_paths = [str(path) for path in sorted(data_dir.iterdir()) if path.suffix == ".parquet"]
    if not data_paths:
        return result

    if len(episode_info) == 1 and "image_index" not in episode_info[0]:
        result["trajectory_data_dir"].append(data_paths[0])
        result["trajectory_rgb_path"].append(rgb_paths)
        result["trajectory_depth_path"].append(depth_paths)
        result["trajectory_afford_path"].append(str(afford_path))
        return result

    for episode_idx, episode in enumerate(episode_info):
        if episode_idx >= len(data_paths):
            break
        if "image_index" in episode:
            start = int(episode["image_index"]["min"])
            end = int(episode["image_index"]["max"])
            episode_rgb_paths = rgb_paths[start : end + 1]
            episode_depth_paths = depth_paths[start : end + 1]
        else:
            episode_rgb_paths = rgb_paths
            episode_depth_paths = depth_paths
        result["trajectory_data_dir"].append(data_paths[episode_idx])
        result["trajectory_rgb_path"].append(episode_rgb_paths)
        result["trajectory_depth_path"].append(episode_depth_paths)
        result["trajectory_afford_path"].append(str(afford_path))
    return result


def generate_preload_index(root_dir: Path, output_path: Path) -> dict:
    root = Path(root_dir)
    index = {
        "trajectory_data_dir": [],
        "trajectory_rgb_path": [],
        "trajectory_depth_path": [],
        "trajectory_afford_path": [],
    }
    for group_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for scene_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            trajectory_dirs = sorted(path for path in scene_dir.iterdir() if path.is_dir() and path.name.startswith("trajectory_"))
            units = trajectory_dirs if trajectory_dirs else [scene_dir]
            for unit in units:
                unit_result = _process_data_unit(unit)
                for key in index:
                    index[key].extend(unit_result[key])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    return index


def _read_active_pid(slot_root: Path) -> int | None:
    active = Path(slot_root) / ".ACTIVE"
    if not active.exists():
        return None
    try:
        text = active.read_text(encoding="utf-8").strip()
        return int(text) if text else None
    except (OSError, ValueError):
        return None


def is_pid_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def ensure_slot_can_be_rebuilt(slot_root: Path, force: bool) -> None:
    slot = Path(slot_root)
    active_pid = _read_active_pid(slot)
    if active_pid and is_pid_running(active_pid):
        raise RuntimeError(f"cache slot is ACTIVE with live pid {active_pid}: {slot}")
    if slot.exists() and not force and (slot / ".READY").exists():
        return
    if slot.exists() and not force:
        raise RuntimeError(f"cache slot exists but is not READY; pass --force to rebuild: {slot}")


def _cache_scene_marker(building: Path, scene: dict) -> Path:
    rel_path = Path(f"{scene['rel_path']}.json")
    return Path(building) / ".scene_done" / rel_path


def _cache_scene_marker_payload(scene: dict) -> dict:
    return {
        "rel_path": scene["rel_path"],
        "bytes": int(scene.get("bytes", 0)),
        "episodes": int(scene.get("episodes", 0)),
    }


def _cache_scene_is_done(building: Path, scene: dict, dst: Path) -> bool:
    marker = _cache_scene_marker(building, scene)
    if not marker.exists() or not dst.is_dir():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        marker.unlink(missing_ok=True)
        return False
    return payload == _cache_scene_marker_payload(scene)


def _copy_cache_scene(scene: dict, *, hdd_traj: Path, slot_traj: Path, building: Path) -> str:
    rel_path = Path(scene["rel_path"])
    src = Path(hdd_traj) / rel_path
    dst = Path(slot_traj) / rel_path
    if _cache_scene_is_done(building, scene, dst):
        print(f"[cache-build] reuse scene: {scene['rel_path']}", flush=True)
        return scene["rel_path"]

    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    _write_json_atomic(_cache_scene_marker(building, scene), _cache_scene_marker_payload(scene))
    print(f"[cache-build] copied scene: {scene['rel_path']}", flush=True)
    return scene["rel_path"]


def _remove_extra_scene_dirs(slot_traj: Path, expected_rel_paths: set[str]) -> None:
    root = Path(slot_traj)
    if not root.exists():
        return
    for group_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for scene_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            rel_path = scene_dir.relative_to(root).as_posix()
            if rel_path not in expected_rel_paths:
                shutil.rmtree(scene_dir)


def build_cache_slot(
    *,
    manifest_path: Path,
    shard_index: int,
    hdd_traj: Path,
    slot_root: Path,
    force: bool = False,
    drop_existing_before_build: bool = False,
    build_workers: int = 1,
    resume_scene_copy: bool = True,
) -> dict:
    manifest = load_manifest(manifest_path)
    shards = manifest.get("shards", [])
    if shard_index < 0 or shard_index >= len(shards):
        raise IndexError(f"shard index out of range: {shard_index}; num_shards={len(shards)}")
    shard = shards[shard_index]
    slot = Path(slot_root)
    ensure_slot_can_be_rebuilt(slot, force=force)
    signature = shard_signature(shard)
    if slot.exists() and (slot / ".READY").exists() and not force:
        metadata = json.loads((slot / "metadata.json").read_text(encoding="utf-8"))
        if int(metadata.get("shard_index", -1)) == int(shard_index) and metadata.get("shard_signature") == signature:
            return metadata
        raise RuntimeError(
            f"cache slot {slot} is READY for shard {metadata.get('shard_index')}, "
            f"not requested shard/signature {shard_index}; pass --force to rebuild"
        )

    if drop_existing_before_build and slot.exists():
        shutil.rmtree(slot)
    building = slot.with_name(slot.name + ".building")
    build_state_path = building / ".BUILDING.json"
    if building.exists() and not resume_scene_copy:
        shutil.rmtree(building)
    elif building.exists() and build_state_path.exists():
        try:
            build_state = json.loads(build_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            build_state = {}
        if int(build_state.get("shard_index", -1)) != int(shard_index) or build_state.get("shard_signature") != signature:
            shutil.rmtree(building)

    building.mkdir(parents=True, exist_ok=True)
    (building / ".BUILDING").write_text(str(os.getpid()), encoding="utf-8")
    _write_json_atomic(
        build_state_path,
        {
            "shard_index": int(shard_index),
            "shard_signature": signature,
            "scene_count": int(shard["scene_count"]),
        },
    )

    slot_traj = building / SLOT_TRAJ_SUFFIX
    scenes = list(shard["scenes"])
    workers = max(int(build_workers or 1), 1)
    print(
        f"[cache-build] shard={shard_index} scenes={len(scenes)} "
        f"workers={workers} resume_scene_copy={int(bool(resume_scene_copy))}",
        flush=True,
    )
    if workers == 1 or len(scenes) <= 1:
        for scene in scenes:
            _copy_cache_scene(scene, hdd_traj=hdd_traj, slot_traj=slot_traj, building=building)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_copy_cache_scene, scene, hdd_traj=hdd_traj, slot_traj=slot_traj, building=building)
                for scene in scenes
            ]
            for future in as_completed(futures):
                future.result()
    _remove_extra_scene_dirs(slot_traj, {scene["rel_path"] for scene in scenes})

    (building / ".BUILDING").unlink(missing_ok=True)
    build_state_path.unlink(missing_ok=True)
    (building / ".READY").write_text(json.dumps({"shard_index": shard_index}), encoding="utf-8")

    if slot.exists():
        shutil.rmtree(slot)
    building.rename(slot)
    preload_index = generate_preload_index(slot / SLOT_TRAJ_SUFFIX, slot / "preload_index.json")
    metadata = {
        "status": "READY",
        "shard_index": shard_index,
        "scene_count": shard["scene_count"],
        "episode_count": shard["episode_count"],
        "bytes": shard["bytes"],
        "trajectory_root": str(slot / SLOT_TRAJ_SUFFIX),
        "preload_index": str(slot / "preload_index.json"),
        "preload_episode_count": len(preload_index["trajectory_data_dir"]),
        "shard_signature": signature,
        "build_workers": workers,
        "resume_scene_copy": bool(resume_scene_copy),
    }
    (slot / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def _flatten(value) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _flatten(item)


def validate_cache_slot(slot_root: Path, example_limit: int = 20) -> ValidationReport:
    slot = Path(slot_root)
    examples: list[str] = []
    missing = 0
    bad_shape = 0
    ready = (slot / ".READY").exists()
    preload = slot / "preload_index.json"
    if not ready:
        examples.append(f"missing READY marker: {slot / '.READY'}")
    if not preload.exists():
        return ValidationReport(ok=False, missing_paths=1, bad_shape=bad_shape, marker_ready=ready, examples=examples + [f"missing preload index: {preload}"])
    try:
        data = json.loads(preload.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ValidationReport(ok=False, bad_shape=1, marker_ready=ready, examples=examples + [f"bad preload index: {exc}"])

    keys = ["trajectory_data_dir", "trajectory_rgb_path", "trajectory_depth_path", "trajectory_afford_path"]
    lengths = []
    for key in keys:
        value = data.get(key)
        if not isinstance(value, list):
            bad_shape += 1
            if len(examples) < example_limit:
                examples.append(f"preload key is missing or not list: {key}")
            continue
        lengths.append(len(value))
        for path_text in _flatten(value):
            if not Path(path_text).exists():
                missing += 1
                if len(examples) < example_limit:
                    examples.append(f"missing path: {path_text}")
    if lengths and len(set(lengths)) != 1:
        bad_shape += 1
        if len(examples) < example_limit:
            examples.append(f"preload list lengths differ: {dict(zip(keys, lengths))}")
    return ValidationReport(
        ok=ready and missing == 0 and bad_shape == 0,
        missing_paths=missing,
        bad_shape=bad_shape,
        marker_ready=ready,
        examples=examples,
    )


def find_latest_global_step(output_dir: Path) -> int:
    root = Path(output_dir)
    if not root.is_dir():
        return 0
    latest_step = 0
    for checkpoint_dir in root.glob("checkpoint-*"):
        state_path = checkpoint_dir / "trainer_state.json"
        if not state_path.exists():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            latest_step = max(latest_step, int(state.get("global_step", 0)))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return latest_step


def mark_active(slot_root: Path) -> None:
    Path(slot_root, ".ACTIVE").write_text(str(os.getpid()), encoding="utf-8")


def clear_active(slot_root: Path) -> None:
    Path(slot_root, ".ACTIVE").unlink(missing_ok=True)
