"""Shared helpers for ArcDP P0 ablation training configs."""

import copy
import os

from .bridgedp_full import PROJECT_ROOT, _env_float, _env_int, bridgedp_full_exp_cfg


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
    cfg.il.epochs = _env_int("BRIDGEDP_EPOCHS", cfg.il.epochs)
    cfg.il.batch_size = _env_int("BRIDGEDP_BATCH_SIZE", cfg.il.batch_size)
    cfg.il.gradient_accumulation_steps = _env_int(
        "BRIDGEDP_GRAD_ACCUM",
        cfg.il.gradient_accumulation_steps,
    )
    cfg.il.num_workers = _env_int("BRIDGEDP_NUM_WORKERS", cfg.il.num_workers)
    cfg.il.lr = _env_float("BRIDGEDP_LR", cfg.il.lr)
    cfg.il.save_interval_epochs = _env_int(
        "BRIDGEDP_SAVE_INTERVAL_EPOCHS",
        cfg.il.save_interval_epochs,
    )
    cfg.il.save_total_limit = _env_int("BRIDGEDP_SAVE_TOTAL_LIMIT", cfg.il.save_total_limit)

    for key, value in (il_overrides or {}).items():
        setattr(cfg.il, key, value)
    return cfg
