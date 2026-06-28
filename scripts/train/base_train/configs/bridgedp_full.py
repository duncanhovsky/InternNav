"""Full Bridge-DP v1.0.3 training config for a 4x A800 SSD workspace.

This config keeps ``model_name='bridgedp'`` so the existing Bridge-DP
dataset, model, and trainer branches are reused unchanged. The external CLI
key is ``bridgedp_full`` and is registered in ``train.py``.
"""

import copy
import os

from .bridgedp import bridgedp_exp_cfg


def _env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


def _env_float(name, default):
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else default


SSD_ROOT = os.environ.get("BRIDGEDP_SSD_ROOT", "/ssd")
PROJECT_ROOT = os.environ.get("BRIDGEDP_PROJECT_ROOT", f"{SSD_ROOT}/MyResearch/InternNav")
DATASET_ROOT = os.environ.get(
    "BRIDGEDP_DATASET_ROOT",
    f"{SSD_ROOT}/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data",
)
PRELOAD_INDEX = os.environ.get(
    "BRIDGEDP_PRELOAD_INDEX",
    f"{PROJECT_ROOT}/checkpoints/preload_index.json",
)

NUM_GPUS = _env_int("BRIDGEDP_NUM_GPUS", 4)

bridgedp_full_exp_cfg = copy.deepcopy(bridgedp_exp_cfg)
bridgedp_full_exp_cfg.name = os.environ.get("BRIDGEDP_RUN_NAME", "bridgedp_full")
bridgedp_full_exp_cfg.model_name = "bridgedp"
bridgedp_full_exp_cfg.torch_gpu_id = 0
bridgedp_full_exp_cfg.torch_gpu_ids = list(range(NUM_GPUS))
bridgedp_full_exp_cfg.output_dir = f"{PROJECT_ROOT}/checkpoints/%s/ckpts"
bridgedp_full_exp_cfg.tensorboard_dir = f"{PROJECT_ROOT}/checkpoints/%s/tensorboard"
bridgedp_full_exp_cfg.checkpoint_folder = f"{PROJECT_ROOT}/checkpoints/%s/ckpts"
bridgedp_full_exp_cfg.log_dir = f"{PROJECT_ROOT}/checkpoints/%s/logs"

il = bridgedp_full_exp_cfg.il
il.batch_size = _env_int("BRIDGEDP_BATCH_SIZE", 96)
il.lr = _env_float("BRIDGEDP_LR", 3e-4)
il.num_workers = _env_int("BRIDGEDP_NUM_WORKERS", 10)
il.dataset_navdp = PRELOAD_INDEX
il.root_dir = DATASET_ROOT
il.preload = True
il.report_to = os.environ.get("BRIDGEDP_REPORT_TO", "tensorboard")

# Extra fields are allowed by IlCfg and consumed by the training entry.
il.gradient_accumulation_steps = _env_int("BRIDGEDP_GRAD_ACCUM", 1)
il.bf16 = os.environ.get("BRIDGEDP_BF16", "1") != "0"
il.tf32 = os.environ.get("BRIDGEDP_TF32", "1") != "0"
il.dataloader_pin_memory = os.environ.get("BRIDGEDP_PIN_MEMORY", "1") != "0"
il.save_total_limit = _env_int("BRIDGEDP_SAVE_TOTAL_LIMIT", 20)
il.logging_steps = _env_int("BRIDGEDP_LOGGING_STEPS", 10)

