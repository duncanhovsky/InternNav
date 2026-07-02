"""Bridge-DP cache rotation training config.

This config is intentionally separate from bridgedp_full.py so the original
epoch-based training path remains unchanged.
"""

import copy
import os

from .bridgedp_full import bridgedp_full_exp_cfg, _env_float, _env_int


NVME_ROOT = os.environ.get("BRIDGEDP_NVME_ROOT", "/nvme")
PROJECT_ROOT = os.environ.get("BRIDGEDP_PROJECT_ROOT", "/nvme/MyResearch/InternNav")
CACHE_SLOT_ROOT = os.environ.get("BRIDGEDP_CACHE_SLOT_ROOT", f"{NVME_ROOT}/bridgedp_cache/cache_A")
DATASET_ROOT = os.environ.get("BRIDGEDP_DATASET_ROOT", f"{CACHE_SLOT_ROOT}/vln_n1/traj_data")
PRELOAD_INDEX = os.environ.get("BRIDGEDP_PRELOAD_INDEX", f"{CACHE_SLOT_ROOT}/preload_index.json")
NUM_GPUS = _env_int("BRIDGEDP_NUM_GPUS", 4)


bridgedp_cache_rotation_exp_cfg = copy.deepcopy(bridgedp_full_exp_cfg)
bridgedp_cache_rotation_exp_cfg.name = os.environ.get("BRIDGEDP_RUN_NAME", "bridgedp_cache_rotation")
bridgedp_cache_rotation_exp_cfg.model_name = "bridgedp"
bridgedp_cache_rotation_exp_cfg.resume_from_checkpoint = os.environ.get("BRIDGEDP_RESUME_FROM", "")
bridgedp_cache_rotation_exp_cfg.auto_resume = os.environ.get("BRIDGEDP_AUTO_RESUME", "1") != "0"
bridgedp_cache_rotation_exp_cfg.torch_gpu_id = 0
bridgedp_cache_rotation_exp_cfg.torch_gpu_ids = list(range(NUM_GPUS))
bridgedp_cache_rotation_exp_cfg.output_dir = f"{PROJECT_ROOT}/checkpoints/%s/ckpts"
bridgedp_cache_rotation_exp_cfg.tensorboard_dir = f"{PROJECT_ROOT}/checkpoints/%s/tensorboard"
bridgedp_cache_rotation_exp_cfg.checkpoint_folder = f"{PROJECT_ROOT}/checkpoints/%s/ckpts"
bridgedp_cache_rotation_exp_cfg.log_dir = f"{PROJECT_ROOT}/checkpoints/%s/logs"

il = bridgedp_cache_rotation_exp_cfg.il
il.root_dir = DATASET_ROOT
il.dataset_navdp = PRELOAD_INDEX
il.batch_size = _env_int("BRIDGEDP_BATCH_SIZE", 96)
il.num_workers = _env_int("BRIDGEDP_NUM_WORKERS", 10)
il.lr = _env_float("BRIDGEDP_LR", 3e-4)
il.gradient_accumulation_steps = _env_int("BRIDGEDP_GRAD_ACCUM", 1)
il.total_max_steps = _env_int("BRIDGEDP_TOTAL_MAX_STEPS", 100000)
il.stage_end_step = _env_int("BRIDGEDP_STAGE_END_STEP", 1000)
il.save_steps = _env_int("BRIDGEDP_SAVE_STEPS", 500)
il.ignore_data_skip = os.environ.get("BRIDGEDP_IGNORE_DATA_SKIP", "1") != "0"
il.save_strategy = os.environ.get("BRIDGEDP_SAVE_STRATEGY", "steps")
il.cache_stage_id = os.environ.get("BRIDGEDP_CACHE_STAGE_ID", "")
il.cache_slot_root = CACHE_SLOT_ROOT
