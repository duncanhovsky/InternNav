"""Bridge-DP full10 training config for 8x RTX 4090 24GB on NVMe."""

import copy
import os

from .bridgedp_full import _env_float, _env_int, bridgedp_full_exp_cfg


NUM_GPUS = _env_int("BRIDGEDP_NUM_GPUS", 8)


bridgedp_full_8x4090_exp_cfg = copy.deepcopy(bridgedp_full_exp_cfg)
bridgedp_full_8x4090_exp_cfg.name = os.environ.get("BRIDGEDP_RUN_NAME", "arcdp_full10_8x4090_nvme")
bridgedp_full_8x4090_exp_cfg.model_name = "bridgedp"
bridgedp_full_8x4090_exp_cfg.torch_gpu_id = 0
bridgedp_full_8x4090_exp_cfg.torch_gpu_ids = list(range(NUM_GPUS))
bridgedp_full_8x4090_exp_cfg.resume_from_checkpoint = os.environ.get("BRIDGEDP_RESUME_FROM", "")
bridgedp_full_8x4090_exp_cfg.auto_resume = os.environ.get("BRIDGEDP_AUTO_RESUME", "1") != "0"

il = bridgedp_full_8x4090_exp_cfg.il
il.epochs = _env_int("BRIDGEDP_EPOCHS", 10)
il.batch_size = _env_int("BRIDGEDP_BATCH_SIZE", 16)
il.gradient_accumulation_steps = _env_int("BRIDGEDP_GRAD_ACCUM", 3)
il.num_workers = _env_int("BRIDGEDP_NUM_WORKERS", 4)
il.lr = _env_float("BRIDGEDP_LR", 3e-4)
il.save_interval_epochs = _env_int("BRIDGEDP_SAVE_INTERVAL_EPOCHS", 1)
il.save_total_limit = _env_int("BRIDGEDP_SAVE_TOTAL_LIMIT", 20)
