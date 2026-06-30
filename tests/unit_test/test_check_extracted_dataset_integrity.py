import json
import subprocess
import sys
import tarfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "dataset" / "check_extracted_dataset_integrity.py"


def _write_unit(unit: Path):
    data_dir = unit / "data" / "chunk-000"
    meta_dir = unit / "meta"
    rgb_dir = unit / "videos" / "chunk-000" / "observation.images.rgb"
    depth_dir = unit / "videos" / "chunk-000" / "observation.images.depth"
    for path in (data_dir, meta_dir, rgb_dir, depth_dir):
        path.mkdir(parents=True, exist_ok=True)

    (data_dir / "episode_000.parquet").write_bytes(b"parquet")
    (meta_dir / "episodes_stats.jsonl").write_text('{"image_index":{"min":0,"max":0}}\n', encoding="utf-8")
    (meta_dir / "pointcloud.ply").write_text("ply\n", encoding="utf-8")
    (rgb_dir / "000000.png").write_bytes(b"rgb")
    (depth_dir / "000000.png").write_bytes(b"depth")


def _make_archive(src_root: Path, work_dir: Path):
    unit = work_dir / "scene-a" / "trajectory_00"
    _write_unit(unit)
    archive = src_root / "group-a" / "scene-a.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(work_dir / "scene-a", arcname="scene-a")
    return archive


def _extract_archive(archive: Path, dst_root: Path):
    dst_group = dst_root / "group-a"
    dst_group.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(dst_group)
    (dst_group / f".extracted_{archive.name}.done").write_text("", encoding="utf-8")


def _write_preload_index(path: Path, dst_root: Path):
    unit = dst_root / "group-a" / "scene-a" / "trajectory_00"
    index = {
        "trajectory_data_dir": [str(unit / "data" / "chunk-000" / "episode_000.parquet")],
        "trajectory_rgb_path": [[str(unit / "videos" / "chunk-000" / "observation.images.rgb" / "000000.png")]],
        "trajectory_depth_path": [[str(unit / "videos" / "chunk-000" / "observation.images.depth" / "000000.png")]],
        "trajectory_afford_path": [str(unit / "meta" / "pointcloud.ply")],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index), encoding="utf-8")


def test_integrity_checker_passes_for_complete_extraction(tmp_path):
    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    archive = _make_archive(src_root, tmp_path / "archive-work")
    _extract_archive(archive, dst_root)
    preload = tmp_path / "preload_index.json"
    _write_preload_index(preload, dst_root)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--src-traj",
            str(src_root),
            "--dst-traj",
            str(dst_root),
            "--preload-index",
            str(preload),
            "--strict-tar",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert result.returncode == 0, result.stdout
    assert "archives_checked=1" in result.stdout
    assert "missing_files=0" in result.stdout
    assert "preload_missing_paths=0" in result.stdout


def test_integrity_checker_fails_when_extracted_file_is_missing(tmp_path):
    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    archive = _make_archive(src_root, tmp_path / "archive-work")
    _extract_archive(archive, dst_root)
    missing = dst_root / "group-a" / "scene-a" / "trajectory_00" / "videos" / "chunk-000" / "observation.images.depth" / "000000.png"
    missing.unlink()

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--src-traj",
            str(src_root),
            "--dst-traj",
            str(dst_root),
            "--strict-tar",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert result.returncode == 1
    assert "missing_files=1" in result.stdout
    assert "observation.images.depth" in result.stdout
