"""Checkpoint resume helpers for base training entrypoints."""

import re
import shutil
from pathlib import Path

from scripts.train.base_train.percent_checkpoint_utils import load_complete_checkpoint_marker


TRAINER_STATE_FILE = "trainer_state.json"
HF_WEIGHT_FILE = "pytorch_model.bin"
BRIDGEDP_WEIGHT_FILE = "bridgedp.ckpt"

_AUTO_VALUES = {"1", "auto", "last", "latest", "true", "yes", "y"}
_DISABLED_VALUES = {"", "0", "none", "false", "no", "n", "off"}
_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def _checkpoint_step(path):
    match = _CHECKPOINT_RE.match(path.name)
    return int(match.group(1)) if match else -1


def is_valid_trainer_checkpoint(path, require_complete=False):
    """Return True when a directory has the Trainer state needed for exact resume."""
    checkpoint_dir = Path(path)
    if not checkpoint_dir.is_dir() or not (checkpoint_dir / TRAINER_STATE_FILE).is_file():
        return False
    if not require_complete:
        return True

    marker = load_complete_checkpoint_marker(checkpoint_dir)
    if marker is None:
        return False
    try:
        marker_step = int(marker["global_step"])
        required_files = marker["required_files"]
        file_sizes = marker["file_sizes"]
    except (KeyError, TypeError, ValueError):
        return False
    if (
        marker_step != _checkpoint_step(checkpoint_dir)
        or not isinstance(required_files, list)
        or not isinstance(file_sizes, dict)
    ):
        return False
    for relative_name in required_files:
        if not isinstance(relative_name, str) or not relative_name:
            return False
        required_path = checkpoint_dir / relative_name
        try:
            expected_size = int(file_sizes[relative_name])
        except (KeyError, TypeError, ValueError):
            return False
        if (
            expected_size <= 0
            or not required_path.is_file()
            or required_path.stat().st_size != expected_size
        ):
            return False
    return True


def find_latest_checkpoint(output_dir, require_complete=False):
    """Find the latest valid checkpoint-* directory under an output directory."""
    root = Path(output_dir)
    if not root.is_dir():
        return None

    checkpoints = []
    for path in root.iterdir():
        if not path.is_dir() or not path.name.startswith("checkpoint-"):
            continue
        if not is_valid_trainer_checkpoint(path, require_complete=require_complete):
            continue
        checkpoints.append((_checkpoint_step(path), path.stat().st_mtime, path))

    if not checkpoints:
        return None
    checkpoints.sort(key=lambda item: (item[0], item[1]))
    return str(checkpoints[-1][2])


def resolve_resume_checkpoint(output_dir, requested="", auto_resume=False, require_complete=False):
    """Resolve a checkpoint directory, or return None to start from scratch."""
    requested_text = str(requested or "").strip()
    requested_key = requested_text.lower()

    if requested_key == "":
        return find_latest_checkpoint(output_dir, require_complete=require_complete) if auto_resume else None

    if requested_key in _DISABLED_VALUES:
        return None

    if requested_key in _AUTO_VALUES:
        return find_latest_checkpoint(output_dir, require_complete=require_complete)

    checkpoint_dir = Path(requested_text).expanduser()
    if not is_valid_trainer_checkpoint(checkpoint_dir, require_complete=require_complete):
        return None
    return str(checkpoint_dir)


def ensure_checkpoint_model_weight(checkpoint_dir):
    """Mirror legacy Bridge-DP weights to the Transformers default weight filename."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.is_dir():
        return None

    hf_weight = checkpoint_path / HF_WEIGHT_FILE
    bridge_weight = checkpoint_path / BRIDGEDP_WEIGHT_FILE
    if not hf_weight.exists() and bridge_weight.is_file():
        shutil.copy2(bridge_weight, hf_weight)
    return str(hf_weight) if hf_weight.exists() else None
