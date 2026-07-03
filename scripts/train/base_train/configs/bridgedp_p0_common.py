"""Shared helpers for ArcDP P0 ablation training configs."""

import copy
import os

from .bridgedp_full import PROJECT_ROOT, bridgedp_full_exp_cfg


def _env_bool(name, default):
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.lower() not in ("0", "false", "no", "n")


def make_arcdp_p0_exp_cfg(default_run_name, il_overrides=None):
    """Return an isolated Bridge-DP/ArcDP P0 ablation config.

    Each ablation keeps ``model_name='bridgedp'`` so the existing dataset,
    policy, and trainer paths are reused, but output roots and run names are
    separated from the full ArcDP checkpoint tree.
    """
    cfg = copy.deepcopy(bridgedp_full_exp_cfg)
    checkpoint_root = os.environ.get(
        "ARCDP_P0_CHECKPOINT_ROOT",
        f"{PROJECT_ROOT}/checkpoints/arcdp_p0",
    )
    cfg.name = os.environ.get("ARCDP_P0_RUN_NAME", default_run_name)
    cfg.resume_from_checkpoint = os.environ.get("BRIDGEDP_RESUME_FROM", "")
    cfg.auto_resume = _env_bool("BRIDGEDP_AUTO_RESUME", True)
    cfg.output_dir = f"{checkpoint_root}/%s/ckpts"
    cfg.tensorboard_dir = f"{checkpoint_root}/%s/tensorboard"
    cfg.checkpoint_folder = f"{checkpoint_root}/%s/ckpts"
    cfg.log_dir = f"{checkpoint_root}/%s/logs"

    for key, value in (il_overrides or {}).items():
        setattr(cfg.il, key, value)
    return cfg
