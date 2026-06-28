"""Checkpoint resume helpers for base training entrypoints."""

import re
import shutil
from pathlib import Path


TRAINER_STATE_FILE = "trainer_state.json"
HF_WEIGHT_FILE = "pytorch_model.bin"
BRIDGEDP_WEIGHT_FILE = "bridgedp.ckpt"

_AUTO_VALUES = {"1", "auto", "last", "latest", "true", "yes", "y"}
_DISABLED_VALUES = {"", "0", "none", "false", "no", "n", "off"}
_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def _checkpoint_step(path):
    match = _CHECKPOINT_RE.match(path.name)
    return int(match.group(1)) if match else -1


def is_valid_trainer_checkpoint(path):
    """Return True when a directory has the Trainer state needed for exact resume."""
    checkpoint_dir = Path(path)
    return checkpoint_dir.is_dir() and (checkpoint_dir / TRAINER_STATE_FILE).is_file()


def find_latest_checkpoint(output_dir):
    """Find the latest valid checkpoint-* directory under an output directory."""
    root = Path(output_dir)
    if not root.is_dir():
        return None

    checkpoints = []
    for path in root.iterdir():
        if not path.is_dir() or not path.name.startswith("checkpoint-"):
            continue
        if not is_valid_trainer_checkpoint(path):
            continue
        checkpoints.append((_checkpoint_step(path), path.stat().st_mtime, path))

    if not checkpoints:
        return None
    checkpoints.sort(key=lambda item: (item[0], item[1]))
    return str(checkpoints[-1][2])


def resolve_resume_checkpoint(output_dir, requested="", auto_resume=False):
    """Resolve a checkpoint directory, or return None to start from scratch."""
    requested_text = str(requested or "").strip()
    requested_key = requested_text.lower()

    if requested_key == "":
        return find_latest_checkpoint(output_dir) if auto_resume else None

    if requested_key in _DISABLED_VALUES:
        return None

    if requested_key in _AUTO_VALUES:
        return find_latest_checkpoint(output_dir)

    checkpoint_dir = Path(requested_text).expanduser()
    if not is_valid_trainer_checkpoint(checkpoint_dir):
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
