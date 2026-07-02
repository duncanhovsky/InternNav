import logging
import os
import sys
import traceback
from datetime import datetime

if os.path.isdir('./third_party/diffusion-policy'):
    sys.path.append('./third_party/diffusion-policy')
elif os.path.isdir('./src/diffusion-policy'):
    sys.path.append('./src/diffusion-policy')

import torch
import torch.distributed as dist
import tyro
from pydantic import BaseModel
from transformers import TrainerCallback, TrainingArguments

from internnav.dataset.bridgedp_lerobot_dataset import BridgeDP_Base_Dataset, bridgedp_collate_fn
from internnav.model import get_config, get_policy
from internnav.model.utils.logger import MyLogger
from internnav.trainer import BridgeDPTrainer
from scripts.train.base_train.configs.bridgedp_cache_rotation import bridgedp_cache_rotation_exp_cfg
from scripts.train.base_train.resume_utils import ensure_checkpoint_model_weight, resolve_resume_checkpoint
from scripts.train.base_train.train import CheckpointFormatCallback, DetailedProgressCallback, _make_dir


class CacheTrainCfg(BaseModel):
    """CLI switches for the independent Bridge-DP cache rotation entrypoint."""

    name: str = "bridgedp_cache_rotation"
    resume_from_checkpoint: str = ""
    auto_resume: bool = True


class StageStopCallback(TrainerCallback):
    """Stop a short cache stage at an absolute global step and request a checkpoint."""

    def __init__(self, stage_end_step: int):
        self.stage_end_step = int(stage_end_step)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stage_end_step:
            control.should_save = True
            control.should_training_stop = True
        return control


def _is_main_process() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def _init_distributed():
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    return local_rank, world_size, rank, device


def _build_dataset(config):
    return BridgeDP_Base_Dataset(
        config.il.root_dir,
        config.il.dataset_navdp,
        config.il.memory_size,
        config.il.predict_size,
        config.il.batch_size,
        config.il.image_size,
        config.il.scene_scale,
        pixel_channel=config.il.pixel_channel,
        preload=config.il.preload,
        random_digit=config.il.random_digit,
        prior_sample=config.il.prior_sample,
        critic_near_threshold=getattr(config.il, "critic_near_threshold", 0.2),
        critic_hard_threshold=getattr(config.il, "critic_hard_threshold", 0.1),
        critic_soft_beta=getattr(config.il, "critic_soft_beta", 4.0),
        critic_max_weight=getattr(config.il, "critic_max_weight", 5.0),
        critic_mean_weight=getattr(config.il, "critic_mean_weight", 2.0),
        critic_trend_weight=getattr(config.il, "critic_trend_weight", 0.5),
        critic_safe_score=getattr(config.il, "critic_safe_score", 2.0),
    )


def _build_model(config, model_class, model_config_class, device, world_size, local_rank):
    model_cfg = model_config_class(model_cfg=config.model_dump())
    model = model_class.from_pretrained(pretrained_model_name_or_path=config.il.ckpt_to_load, config=model_cfg)
    model.to(device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    return model


def main(config, model_class, model_config_class):
    try:
        _make_dir(config)
        torch.backends.cuda.matmul.allow_tf32 = bool(getattr(config.il, "tf32", False))
        torch.backends.cudnn.allow_tf32 = bool(getattr(config.il, "tf32", False))

        print("=== Start Bridge-DP cache rotation stage ===")
        print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"cache_stage_id: {getattr(config.il, 'cache_stage_id', '')}")
        print(f"cache_slot_root: {getattr(config.il, 'cache_slot_root', '')}")
        print(f"dataset_root: {config.il.root_dir}")
        print(f"preload_index: {config.il.dataset_navdp}")
        print(f"total_max_steps: {getattr(config.il, 'total_max_steps')}")
        print(f"stage_end_step: {getattr(config.il, 'stage_end_step')}")

        local_rank, world_size, _, device = _init_distributed()
        model = _build_model(config, model_class, model_config_class, device, world_size, local_rank)

        train_logger_filename = os.path.join(config.log_dir, "train_cache_rotation.log")
        if _is_main_process():
            train_logger = MyLogger(
                name="train_cache_rotation",
                level=logging.INFO,
                format_str="%(asctime)-15s %(message)s",
                filename=train_logger_filename,
            )
        else:
            train_logger = MyLogger(
                name="train_cache_rotation",
                level=logging.INFO,
                format_str="%(asctime)-15s %(message)s",
            )

        transformers_logger = logging.getLogger("transformers")
        if transformers_logger.hasHandlers():
            transformers_logger.handlers = []
        if local_rank in [0, -1]:
            transformers_logger.addHandler(train_logger.handlers[0])
        transformers_logger.setLevel(logging.INFO)

        train_dataset = _build_dataset(config)
        total_max_steps = int(getattr(config.il, "total_max_steps"))
        stage_end_step = int(getattr(config.il, "stage_end_step"))
        ignore_data_skip = bool(getattr(config.il, "ignore_data_skip", True))

        training_args = TrainingArguments(
            output_dir=config.output_dir,
            logging_dir=config.tensorboard_dir,
            run_name=config.name,
            remove_unused_columns=False,
            deepspeed="",
            gradient_checkpointing=False,
            bf16=bool(getattr(config.il, "bf16", False)),
            tf32=bool(getattr(config.il, "tf32", False)),
            per_device_train_batch_size=config.il.batch_size,
            gradient_accumulation_steps=int(getattr(config.il, "gradient_accumulation_steps", 1)),
            dataloader_num_workers=config.il.num_workers,
            dataloader_pin_memory=bool(getattr(config.il, "dataloader_pin_memory", False)),
            optim="adamw_torch",
            learning_rate=config.il.lr,
            lr_scheduler_type="cosine",
            logging_steps=float(getattr(config.il, "logging_steps", 10.0)),
            max_steps=total_max_steps,
            save_strategy=getattr(config.il, "save_strategy", "steps"),
            save_steps=int(getattr(config.il, "save_steps", 500)),
            save_total_limit=int(getattr(config.il, "save_total_limit", 20)),
            report_to=config.il.report_to,
            seed=0,
            do_eval=False,
            ddp_find_unused_parameters=config.il.ddp_find_unused_parameters,
            ddp_bucket_cap_mb=100,
            torch_compile_mode=None,
            dataloader_drop_last=True,
            disable_tqdm=True,
            log_level="info",
            ignore_data_skip=ignore_data_skip,
        )

        trainer = BridgeDPTrainer(
            config=config,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=bridgedp_collate_fn,
        )
        trainer.add_callback(CheckpointFormatCallback(run_name=config.name, exp_cfg_dir=config.log_dir))
        trainer.add_callback(
            DetailedProgressCallback(
                log_dir=config.log_dir,
                dataset_size=len(train_dataset),
                batch_size=config.il.batch_size,
            )
        )
        trainer.add_callback(StageStopCallback(stage_end_step=stage_end_step))

        resume_checkpoint = resolve_resume_checkpoint(
            config.output_dir,
            requested=getattr(config, "resume_from_checkpoint", ""),
            auto_resume=bool(getattr(config, "auto_resume", True)),
        )
        if _is_main_process():
            if resume_checkpoint:
                ensure_checkpoint_model_weight(resume_checkpoint)
                print(f"[Resume] Resume from checkpoint: {resume_checkpoint}")
            else:
                print(f"[Resume] No valid checkpoint found in {config.output_dir}; start from scratch.")
        if dist.is_initialized():
            dist.barrier()
        if resume_checkpoint and not _is_main_process():
            ensure_checkpoint_model_weight(resume_checkpoint)

        trainer.train(resume_from_checkpoint=resume_checkpoint)
        if _is_main_process():
            final_ckpt = os.path.join(config.output_dir, f"checkpoint-{trainer.state.global_step}")
            if os.path.isdir(final_ckpt):
                trainer.save_model(final_ckpt)
                trainer.state.save_to_json(os.path.join(final_ckpt, "trainer_state.json"))
                print(f"[StageStop] Final stage checkpoint ready: {final_ckpt}")
            else:
                print(f"[StageStop] Warning: expected checkpoint was not created: {final_ckpt}")
        if dist.is_initialized():
            dist.barrier()
        for handler in train_logger.handlers:
            handler.flush()
    except Exception as exc:
        print(f"Unhandled exception: {exc}")
        print("Stack trace:")
        traceback.print_exc()
        if dist.is_initialized():
            dist.destroy_process_group()
        raise


if __name__ == "__main__":
    cli_cfg = tyro.cli(CacheTrainCfg)
    exp_cfg = bridgedp_cache_rotation_exp_cfg
    exp_cfg.name = cli_cfg.name
    exp_cfg.resume_from_checkpoint = cli_cfg.resume_from_checkpoint
    exp_cfg.auto_resume = cli_cfg.auto_resume or bool(getattr(exp_cfg, "auto_resume", False))
    exp_cfg.num_gpus = len(exp_cfg.torch_gpu_ids)
    exp_cfg.world_size = exp_cfg.num_gpus

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    assert available_gpus >= exp_cfg.num_gpus, (
        f"Number of GPUs requested ({exp_cfg.num_gpus}) is greater than available GPUs ({available_gpus})"
    )
    model_class = get_policy("BridgeDP_Policy")
    model_config_class = get_config("BridgeDP_Policy")
    main(exp_cfg, model_class, model_config_class)
