#!/usr/bin/env python3
"""Check InternData-N1 extracted trajectory dataset integrity.

This checker is designed for the layout used by scripts/prepare_ssd_full.sh:

  SRC_TRAJ/group_name/*.tar.gz
  DST_TRAJ/group_name/.extracted_<archive>.done
  DST_TRAJ/group_name/<archive contents...>

Fast mode checks archive markers and preload paths. Strict mode additionally
walks every tar member and verifies that each extracted file exists with the
expected size.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable


DEFAULT_DATASET_SUFFIX = Path("datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data")
DEFAULT_PRELOAD_SUFFIX = Path("MyResearch/InternNav/checkpoints/preload_index.json")


@dataclass
class CheckReport:
    archives_checked: int = 0
    markers_missing: int = 0
    archives_unreadable: int = 0
    tar_files_checked: int = 0
    missing_files: int = 0
    size_mismatches: int = 0
    unsafe_members: int = 0
    preload_entries: int = 0
    preload_missing_paths: int = 0
    preload_bad_shape: int = 0
    examples: list[str] = field(default_factory=list)

    def add_example(self, message: str, limit: int) -> None:
        if len(self.examples) < limit:
            self.examples.append(message)

    @property
    def ok(self) -> bool:
        return (
            self.markers_missing == 0
            and self.archives_unreadable == 0
            and self.missing_files == 0
            and self.size_mismatches == 0
            and self.unsafe_members == 0
            and self.preload_missing_paths == 0
            and self.preload_bad_shape == 0
        )


def _default_source_root() -> Path:
    source_root = Path(os.environ.get("SOURCE_ROOT", "/data"))
    home_data = Path.home() / "data"
    if not source_root.is_dir() and home_data.is_dir():
        source_root = home_data
    return source_root


def default_src_traj() -> Path:
    return _default_source_root() / DEFAULT_DATASET_SUFFIX


def default_dst_traj() -> Path:
    return Path(os.environ.get("SSD_ROOT", "/ssd")) / DEFAULT_DATASET_SUFFIX


def default_preload_index() -> Path:
    if "DST_PROJECT" in os.environ:
        return Path(os.environ["DST_PROJECT"]) / "checkpoints" / "preload_index.json"
    return Path(os.environ.get("SSD_ROOT", "/ssd")) / DEFAULT_PRELOAD_SUFFIX


def iter_archives(src_traj: Path) -> Iterable[Path]:
    if not src_traj.exists():
        raise FileNotFoundError(f"missing source archive root: {src_traj}")
    for group_dir in sorted(src_traj.iterdir()):
        if not group_dir.is_dir():
            continue
        yield from sorted(group_dir.glob("*.tar.gz"))


def marker_path(src_traj: Path, dst_traj: Path, archive: Path) -> Path:
    rel_parent = archive.parent.relative_to(src_traj)
    return dst_traj / rel_parent / f".extracted_{archive.name}.done"


def _safe_member_path(dst_dir: Path, member_name: str) -> Path | None:
    posix = PurePosixPath(member_name)
    if posix.is_absolute() or any(part == ".." for part in posix.parts):
        return None
    parts = [part for part in posix.parts if part not in ("", ".")]
    if not parts:
        return None
    return dst_dir.joinpath(*parts)


def check_archives(
    src_traj: Path,
    dst_traj: Path,
    strict_tar: bool,
    report: CheckReport,
    example_limit: int,
) -> None:
    for archive in iter_archives(src_traj):
        report.archives_checked += 1
        rel_parent = archive.parent.relative_to(src_traj)
        dst_dir = dst_traj / rel_parent
        marker = marker_path(src_traj, dst_traj, archive)

        if not marker.exists():
            report.markers_missing += 1
            report.add_example(f"missing marker: {marker}", example_limit)

        if not strict_tar:
            continue

        try:
            with tarfile.open(archive, "r:*") as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    report.tar_files_checked += 1
                    extracted = _safe_member_path(dst_dir, member.name)
                    if extracted is None:
                        report.unsafe_members += 1
                        report.add_example(f"unsafe tar member: {archive}::{member.name}", example_limit)
                        continue
                    if not extracted.exists():
                        report.missing_files += 1
                        report.add_example(f"missing extracted file: {extracted} (from {archive})", example_limit)
                        continue
                    try:
                        actual_size = extracted.stat().st_size
                    except OSError as exc:
                        report.missing_files += 1
                        report.add_example(f"cannot stat extracted file: {extracted} ({exc})", example_limit)
                        continue
                    if actual_size != member.size:
                        report.size_mismatches += 1
                        report.add_example(
                            f"size mismatch: {extracted} expected={member.size} actual={actual_size} (from {archive})",
                            example_limit,
                        )
        except (tarfile.TarError, OSError) as exc:
            report.archives_unreadable += 1
            report.add_example(f"unreadable archive: {archive} ({exc})", example_limit)


def _flatten_preload_paths(value) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _flatten_preload_paths(item)


def check_preload_index(preload_index: Path, report: CheckReport, example_limit: int) -> None:
    if not preload_index.exists():
        report.preload_missing_paths += 1
        report.add_example(f"missing preload index: {preload_index}", example_limit)
        return

    try:
        data = json.loads(preload_index.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.preload_bad_shape += 1
        report.add_example(f"cannot read preload index: {preload_index} ({exc})", example_limit)
        return

    keys = [
        "trajectory_data_dir",
        "trajectory_rgb_path",
        "trajectory_depth_path",
        "trajectory_afford_path",
    ]
    lengths = []
    for key in keys:
        value = data.get(key)
        if not isinstance(value, list):
            report.preload_bad_shape += 1
            report.add_example(f"preload key is missing or not a list: {key}", example_limit)
            continue
        lengths.append(len(value))
        for path_text in _flatten_preload_paths(value):
            report.preload_entries += 1
            if not Path(path_text).exists():
                report.preload_missing_paths += 1
                report.add_example(f"missing preload path: {path_text}", example_limit)

    if lengths and len(set(lengths)) != 1:
        report.preload_bad_shape += 1
        report.add_example(f"preload list lengths differ: {dict(zip(keys, lengths))}", example_limit)


def print_report(report: CheckReport) -> None:
    print("Integrity check summary")
    print(f"  archives_checked={report.archives_checked}")
    print(f"  markers_missing={report.markers_missing}")
    print(f"  archives_unreadable={report.archives_unreadable}")
    print(f"  tar_files_checked={report.tar_files_checked}")
    print(f"  missing_files={report.missing_files}")
    print(f"  size_mismatches={report.size_mismatches}")
    print(f"  unsafe_members={report.unsafe_members}")
    print(f"  preload_entries={report.preload_entries}")
    print(f"  preload_missing_paths={report.preload_missing_paths}")
    print(f"  preload_bad_shape={report.preload_bad_shape}")
    if report.examples:
        print()
        print("Examples")
        for example in report.examples:
            print(f"  - {example}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check extracted InternData-N1 trajectory data against source archives and preload index.",
    )
    parser.add_argument("--src-traj", type=Path, default=default_src_traj(), help="Source traj_data archive root.")
    parser.add_argument("--dst-traj", type=Path, default=default_dst_traj(), help="Extracted traj_data root.")
    parser.add_argument(
        "--preload-index",
        type=Path,
        default=default_preload_index(),
        help="Generated preload_index.json to verify.",
    )
    parser.add_argument(
        "--strict-tar",
        action="store_true",
        help="Verify every regular file listed in every tar archive exists and has the expected size.",
    )
    parser.add_argument(
        "--skip-preload",
        action="store_true",
        help="Skip preload_index.json path validation.",
    )
    parser.add_argument(
        "--example-limit",
        type=int,
        default=50,
        help="Maximum number of failing examples to print.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = CheckReport()

    print("Checking extracted dataset")
    print(f"  src_traj:      {args.src_traj}")
    print(f"  dst_traj:      {args.dst_traj}")
    print(f"  preload_index: {args.preload_index}")
    print(f"  strict_tar:    {args.strict_tar}")

    try:
        check_archives(args.src_traj, args.dst_traj, args.strict_tar, report, args.example_limit)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not args.skip_preload:
        check_preload_index(args.preload_index, report, args.example_limit)

    print()
    print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
