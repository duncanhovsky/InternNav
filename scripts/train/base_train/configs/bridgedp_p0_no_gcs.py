"""ArcDP P0 ablation: disable goal-consistency candidate scoring."""

import os

from .bridgedp_p0_common import make_arcdp_p0_exp_cfg


ARCDP_P0_CHECKPOINT_ROOT = os.environ.get("ARCDP_P0_CHECKPOINT_ROOT", "checkpoints/arcdp_p0")

bridgedp_p0_no_gcs_exp_cfg = make_arcdp_p0_exp_cfg(
    default_run_name="arcdp_p0_no_gcs",
    il_overrides={
        "enable_goal_consistency_score": False,
    },
)
