"""ArcDP P0 ablation: bridge denoising without ordered bridge initialization."""

import os

from .bridgedp_p0_common import make_arcdp_p0_exp_cfg


ARCDP_P0_CHECKPOINT_ROOT = os.environ.get("ARCDP_P0_CHECKPOINT_ROOT", "checkpoints/arcdp_p0")

bridgedp_p0_no_ordered_init_exp_cfg = make_arcdp_p0_exp_cfg(
    default_run_name="arcdp_p0_no_ordered_init",
    il_overrides={
        "ablation_prediction_space": "absolute",
        "ablation_diffusion_mode": "bridge",
        "ablation_initialization_mode": "gaussian",
    },
)
