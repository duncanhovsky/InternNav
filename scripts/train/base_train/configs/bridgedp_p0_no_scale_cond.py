"""ArcDP P0 ablation: remove all physical-scale conditioning."""

import os

from .bridgedp_p0_common import make_arcdp_p0_exp_cfg


ARCDP_P0_CHECKPOINT_ROOT = os.environ.get("ARCDP_P0_CHECKPOINT_ROOT", "checkpoints/arcdp_p0")

bridgedp_p0_no_scale_cond_exp_cfg = make_arcdp_p0_exp_cfg(
    default_run_name="arcdp_p0_no_scale_cond",
    il_overrides={
        "enable_scale_condition_token": False,
        "enable_scale_rgbd_film": False,
    },
)
