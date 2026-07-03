"""ArcDP P0 ablation: train and infer without bridge anchor sampling."""

import os

from .bridgedp_p0_common import make_arcdp_p0_exp_cfg


ARCDP_P0_CHECKPOINT_ROOT = os.environ.get("ARCDP_P0_CHECKPOINT_ROOT", "checkpoints/arcdp_p0")

bridgedp_p0_no_anchor_train_exp_cfg = make_arcdp_p0_exp_cfg(
    default_run_name="arcdp_p0_no_anchor_train",
    il_overrides={
        "enable_bridge_anchor_sampling": False,
        "bridge_anchor_train_prob": 0.0,
    },
)
