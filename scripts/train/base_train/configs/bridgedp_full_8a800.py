"""Bridge-DP full training config for 8x A800 on an SSD workspace."""

import copy
import os

from .bridgedp_full import bridgedp_full_exp_cfg, _env_float, _env_int


NUM_GPUS = _env_int("BRIDGEDP_NUM_GPUS", 8)


bridgedp_full_8a800_exp_cfg = copy.deepcopy(bridgedp_full_exp_cfg)
bridgedp_full_8a800_exp_cfg.name = os.environ.get("BRIDGEDP_RUN_NAME", "bridgedp_full_8a800")
bridgedp_full_8a800_exp_cfg.model_name = "bridgedp"
bridgedp_full_8a800_exp_cfg.torch_gpu_id = 0
bridgedp_full_8a800_exp_cfg.torch_gpu_ids = list(range(NUM_GPUS))
bridgedp_full_8a800_exp_cfg.resume_from_checkpoint = os.environ.get("BRIDGEDP_RESUME_FROM", "")
bridgedp_full_8a800_exp_cfg.auto_resume = os.environ.get("BRIDGEDP_AUTO_RESUME", "1") != "0"

il = bridgedp_full_8a800_exp_cfg.il
il.batch_size = _env_int("BRIDGEDP_BATCH_SIZE", 96)
il.num_workers = _env_int("BRIDGEDP_NUM_WORKERS", 5)
il.lr = _env_float("BRIDGEDP_LR", 3e-4)
