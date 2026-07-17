"""Pure helpers for percentage-based resumable checkpoint retention."""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


COMPLETE_MARKER_FILE = "CHECKPOINT_COMPLETE.json"
MANIFEST_FILE = "percent_checkpoint_manifest.json"


@dataclass(frozen=True)
class PercentCheckpointTarget:
    percent: int
    step: int
    permanent: bool


def compute_percent_checkpoint_targets(
    total_steps: int,
    interval_percent: int = 1,
    permanent_interval_percent: int = 10,
) -> list[PercentCheckpointTarget]:
    """Return exact integer-percentage targets, always including the final step."""
    total = max(int(total_steps), 1)
    interval = max(int(interval_percent), 1)
    permanent_interval = max(int(permanent_interval_percent), 1)
    percentages = list(range(interval, 101, interval))
    if not percentages or percentages[-1] != 100:
        percentages.append(100)

    by_step: dict[int, PercentCheckpointTarget] = {}
    for percent in percentages:
        step = math.ceil(total * percent / 100)
        by_step[step] = PercentCheckpointTarget(
            percent=percent,
            step=step,
            permanent=(percent % permanent_interval == 0 or percent == 100),
        )
    return [by_step[step] for step in sorted(by_step)]


def atomic_write_json(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_complete_checkpoint_marker(checkpoint_dir: str | Path) -> dict | None:
    marker_path = Path(checkpoint_dir) / COMPLETE_MARKER_FILE
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _completed_checkpoint_records(output_dir: Path) -> list[dict]:
    records: list[dict] = []
    if not output_dir.is_dir():
        return records
    for checkpoint_dir in output_dir.glob("checkpoint-*"):
        if not checkpoint_dir.is_dir():
            continue
        marker = load_complete_checkpoint_marker(checkpoint_dir)
        if marker is None:
            continue
        try:
            global_step = int(marker["global_step"])
            percent = int(marker["percent"])
        except (KeyError, TypeError, ValueError):
            continue
        records.append(
            {
                **marker,
                "global_step": global_step,
                "percent": percent,
                "permanent": bool(marker.get("permanent", False)),
                "path": checkpoint_dir.name,
                "_path": checkpoint_dir,
            }
        )
    records.sort(key=lambda record: int(record["global_step"]))
    return records


def apply_percent_checkpoint_retention(output_dir: str | Path, rolling_keep: int = 5) -> list[Path]:
    """Prune old completed rolling checkpoints while preserving permanent and incomplete ones."""
    root = Path(output_dir)
    records = _completed_checkpoint_records(root)
    rolling = [record for record in records if not record["permanent"]]
    keep_count = max(int(rolling_keep), 0)
    remove_records = rolling[:-keep_count] if keep_count else rolling
    removed: list[Path] = []
    for record in remove_records:
        path = record["_path"]
        shutil.rmtree(path)
        removed.append(path)

    remaining = _completed_checkpoint_records(root)
    manifest_records = []
    for record in remaining:
        manifest_records.append({key: value for key, value in record.items() if key != "_path"})
    atomic_write_json(
        root / MANIFEST_FILE,
        {
            "completed_count": len(manifest_records),
            "permanent_count": sum(bool(record["permanent"]) for record in manifest_records),
            "rolling_count": sum(not bool(record["permanent"]) for record in manifest_records),
            "rolling_keep": keep_count,
            "checkpoints": manifest_records,
        },
    )
    return removed
