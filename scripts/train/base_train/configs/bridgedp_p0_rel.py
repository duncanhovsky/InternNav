"""ArcDP P0 ablation: relative-motion DDPM policy."""

import os

from .bridgedp_p0_common import make_arcdp_p0_exp_cfg


ARCDP_P0_CHECKPOINT_ROOT = os.environ.get("ARCDP_P0_CHECKPOINT_ROOT", "checkpoints/arcdp_p0")

bridgedp_p0_rel_exp_cfg = make_arcdp_p0_exp_cfg(
    default_run_name="arcdp_p0_rel",
    il_overrides={
        "ablation_prediction_space": "relative_delta",
        "ablation_diffusion_mode": "ddpm",
        "ablation_initialization_mode": "gaussian",
        "enable_trajectory_normalization": False,
        "enable_bridge_anchor_sampling": False,
        "bridge_anchor_train_prob": 0.0,
    },
)
