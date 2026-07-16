"""Compatibility helpers for SwanLab-backed Transformers training."""

import os


def initialize_swanlab_run(report_to, *, is_main_process: bool, run_name: str):
    """Create the single SwanLab Run needed by the Transformers callback.

    Transformers 4.51 expects ``swanlab.get_run()`` to return ``None`` before
    initialization, while SwanLab 0.8.5 raises ``RuntimeError`` instead.  Only
    world-process-zero owns the external logging Run during distributed training.
    """
    if not is_main_process or "swanlab" not in str(report_to).lower():
        return None

    import swanlab

    try:
        active_run = swanlab.get_run()
    except RuntimeError as exc:
        if "no active run" not in str(exc).lower():
            raise
        active_run = None

    if active_run is not None:
        return active_run

    init_kwargs = {
        "experiment_name": os.environ.get("SWANLAB_EXP_NAME") or run_name,
    }
    project_name = os.environ.get("SWANLAB_PROJ_NAME")
    if project_name:
        init_kwargs["project"] = project_name

    return swanlab.init(**init_kwargs)
