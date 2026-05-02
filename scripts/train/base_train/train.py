import os
import sys

if os.path.isdir('./third_party/diffusion-policy'):
    sys.path.append('./third_party/diffusion-policy')
elif os.path.isdir('./src/diffusion-policy'):
    sys.path.append('./src/diffusion-policy')
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import tyro
from pydantic import BaseModel
from torch.utils.data import Dataset
from transformers import TrainerCallback, TrainingArguments

from internnav.dataset.cma_lerobot_dataset import CMALerobotDataset, cma_collate_fn
from internnav.dataset.flownav_dyn_lerobot_dataset import FlowNav_Dyn_Lerobot_Dataset, flownav_dyn_collate_fn
from internnav.dataset.flownav_lerobot_dataset import FlowNav_Base_Datset, flownav_collate_fn
from internnav.dataset.navdp_lerobot_dataset import NavDP_Base_Datset, navdp_collate_fn
from internnav.dataset.bridgedp_lerobot_dataset import BridgeDP_Base_Dataset, bridgedp_collate_fn
from internnav.dataset.rdp_lerobot_dataset import RDP_LerobotDataset, rdp_collate_fn
from internnav.model import get_config, get_policy
from internnav.model.utils.logger import MyLogger
from internnav.model.utils.utils import load_dataset
from internnav.trainer import BridgeDPTrainer, CMATrainer, FlowNavTrainer, NavDPTrainer, RDPTrainer
from scripts.train.base_train.configs import (
    bridgedp_exp_cfg,
    cma_exp_cfg,
    cma_plus_exp_cfg,
    flownav_dyn_exp_cfg,
    flownav_mix_exp_cfg,
    flownav_static_exp_cfg,
    navdp_exp_cfg,
    rdp_exp_cfg,
    seq2seq_exp_cfg,
    seq2seq_plus_exp_cfg,
)


class TrainCfg(BaseModel):
    """Training configuration class"""

    name: str = 'cma_train'  # Experiment name
    model_name: str = 'cma'  # Model name, options: 'cma', 'cma_plus', 'seq2seq', 'seq2seq_plus', 'rdp', 'navdp', 'bridgedp', 'flownav_static', 'flownav_dyn', 'flownav_mix'


class FlowNavMixDataset(Dataset):
    """将静态/动态两个 FlowNav 数据集拼接，并提供 WeightedRandomSampler 所需权重。

    设计目标：
    1) 使用同一个 FlowNav 模型联合训练静态+动态数据。
    2) 采样比例由 `static_ratio` / `dyn_ratio` 控制，而不是固定轮询。
    3) 对小数据集子域（如动态样本较少）按权重采样，缓解机械循环造成的模式过拟合。
    """

    def __init__(self, static_dataset: Dataset, dyn_dataset: Dataset, static_ratio: float = 1.0, dyn_ratio: float = 1.0):
        self.static_dataset = static_dataset
        self.dyn_dataset = dyn_dataset

        self.static_len = len(self.static_dataset)
        self.dyn_len = len(self.dyn_dataset)
        self.total_len = self.static_len + self.dyn_len

        if self.total_len <= 0:
            raise ValueError("FlowNavMixDataset requires at least one sample.")

        # 为什么这样改：WeightedRandomSampler 期望“每个样本一个权重”，
        # 这里把“子集比例”均匀分配到各自子集内部，保证总体期望比例可控。
        s_ratio = max(float(static_ratio), 1e-6)
        d_ratio = max(float(dyn_ratio), 1e-6)
        s_per_sample = s_ratio / max(self.static_len, 1)
        d_per_sample = d_ratio / max(self.dyn_len, 1)
        # sample_weights作为权重由什么决定权重大小：sample_weights的大小由s_per_sample和d_per_sample决定
        self.sample_weights = torch.cat(
            [
                torch.full((self.static_len,), s_per_sample, dtype=torch.double),
                torch.full((self.dyn_len,), d_per_sample, dtype=torch.double),
            ]
        )

        # trainer 检测此标记后会启用 WeightedRandomSampler。
        self.use_weighted_sampler = True

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        if index < self.static_len:
            return {"source": "static", "sample": self.static_dataset[index]}
        dyn_index = index - self.static_len
        return {"source": "dyn", "sample": self.dyn_dataset[dyn_index]}


def make_flownav_mix_collate_fn(predict_frames: int, dynamic_grid_shape, image_size: int, fallback_fps: float):
    """构建 FlowNav 混合训练用 collate 函数。

    关键策略：
    1) 动态样本保留 `batch_dynamic_voxels`（高质量监督优先）。
    2) 静态样本写入零体素并设置 `batch_dynamic_voxel_valid_mask=0`。
    3) 同时为全 batch 补齐 dyn_module fallback 所需字段，
       便于 trainer 对 mask=0 的样本在线重建 dynamic_voxels。
    """

    x, y, z = dynamic_grid_shape
    dt = 1.0 / max(float(fallback_fps), 1e-6)

    def _default_depth_meta(history_frames: int):
        base = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, float(image_size), float(image_size), 1.0, 1.0], dtype=torch.float32)
        return base.repeat(history_frames, 1)

    def _collate(batch):
        pg, ig, tg = [], [], []
        rgb, depth = [], []
        labels, augments = [], []
        label_critic, augment_critic = [], []
        odom_pose, odom_delta = [], []
        dyn_voxels, dyn_valid = [], []
        depth_raw_hist, pose_world_hist = [], []
        ts_hist_s, camera_intr = [], []
        depth_meta_hist, pixel_flag = [], []

        for item in batch:
            source = item["source"]
            sample = item["sample"]

            pg.append(sample[0])
            ig.append(sample[1])
            tg.append(sample[2])
            rgb.append(sample[3])
            depth.append(sample[4])
            labels.append(sample[5])
            augments.append(sample[6])
            label_critic.append(sample[7])
            augment_critic.append(sample[8])
            odom_pose.append(sample[9])
            odom_delta.append(sample[10])

            if source == "dyn":
                # 动态样本直接使用数据集提供的 dynamic_voxels。
                dyn_voxels.append(sample[11])
                dyn_valid.append(torch.tensor(True))
                pixel_flag.append(sample[12])

                # 为字段对齐补齐 fallback 输入（该类样本通常不会走 fallback）。
                h = sample[4].shape[0]
                depth_raw_hist.append(sample[4].clone().to(torch.float32))
                pose_world_hist.append(torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(h, 1, 1))
                ts_hist_s.append(torch.arange(h, dtype=torch.float32) * dt)
                camera_intr.append(torch.eye(3, dtype=torch.float32))
                depth_meta_hist.append(_default_depth_meta(h))
            else:
                # 静态样本先放置零体素，后续由 trainer 按 mask=0 进行在线重建。
                dyn_voxels.append(torch.zeros((predict_frames, 4, x, y, z), dtype=torch.float32))
                dyn_valid.append(torch.tensor(False))
                pixel_flag.append(sample[16])

                depth_raw_hist.append(sample[11].to(torch.float32))
                pose_world_hist.append(sample[12].to(torch.float32))
                ts_hist_s.append(sample[13].to(torch.float32))
                camera_intr.append(sample[14].to(torch.float32))
                depth_meta_hist.append(sample[15].to(torch.float32))

        return {
            "batch_pg": torch.stack(pg),
            "batch_ig": torch.stack(ig),
            "batch_tg": torch.stack(tg),
            "batch_rgb": torch.stack(rgb),
            "batch_depth": torch.stack(depth),
            "batch_labels": torch.stack(labels),
            "batch_augments": torch.stack(augments),
            "batch_label_critic": torch.stack(label_critic),
            "batch_augment_critic": torch.stack(augment_critic),
            "batch_odom_pose": torch.stack(odom_pose),
            "batch_odom_delta": torch.stack(odom_delta),
            "batch_dynamic_voxels": torch.stack(dyn_voxels),
            "batch_dynamic_voxel_valid_mask": torch.stack(dyn_valid).to(torch.bool),
            "batch_depth_raw_m_hist": torch.stack(depth_raw_hist),
            "batch_pose_world_hist": torch.stack(pose_world_hist),
            "batch_timestamp_hist_s": torch.stack(ts_hist_s),
            "batch_camera_intrinsic": torch.stack(camera_intr),
            "batch_depth_preprocess_meta": torch.stack(depth_meta_hist),
            "batch_pixel_flag": torch.tensor(pixel_flag, dtype=torch.float32),
        }

    return _collate


class CheckpointFormatCallback(TrainerCallback):
    """This callback format checkpoint to make them standalone. For now, it copies all config
    files to /checkpoint-{step}/experiment_cfg/:
    - conf.yaml
    - initial_actions.npz
    - metadata.json
    """

    def __init__(self, run_name: str, exp_cfg_dir: Path | None = None):
        """
        Args:
            run_name: Name of the experiment run
            exp_cfg_dir: Path to the directory containing all experiment metadata
        """
        self.exp_cfg_dir = exp_cfg_dir

    def on_save(self, args, state, control, **kwargs):
        """Called after the trainer saves a checkpoint."""
        if state.is_world_process_zero:
            checkpoint_dir = Path(args.output_dir) / f'checkpoint-{state.global_step}'  # noqa: F841


def _make_dir(config):
    config.tensorboard_dir = config.tensorboard_dir % config.name
    config.checkpoint_folder = config.checkpoint_folder % config.name
    config.log_dir = config.log_dir % config.name
    config.output_dir = config.output_dir % config.name
    if not os.path.exists(config.tensorboard_dir):
        os.makedirs(config.tensorboard_dir, exist_ok=True)
    if not os.path.exists(config.checkpoint_folder):
        os.makedirs(config.checkpoint_folder, exist_ok=True)
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir, exist_ok=True)


def main(config, model_class, model_config_class):
    try:
        """Main training function."""
        _make_dir(config)

        print("=== Start training ===")
        print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print("Environment variables:")
        print(f"  RANK: {os.getenv('RANK', 'Not set')}")
        print(f"  LOCAL_RANK: {os.getenv('LOCAL_RANK', 'Not set')}")
        print(f"  WORLD_SIZE: {os.getenv('WORLD_SIZE', 'Not set')}")
        print(f"  MASTER_ADDR: {os.getenv('MASTER_ADDR', 'Not set')}")
        print(f"  MASTER_PORT: {os.getenv('MASTER_PORT', 'Not set')}")

        if config.model_name in ["navdp", "bridgedp", "flownav_static", "flownav_dyn", "flownav_mix"]:
            local_rank = int(os.getenv('LOCAL_RANK', '0'))
            world_size = int(os.getenv('WORLD_SIZE', '1'))
            rank = int(os.getenv('RANK', '0'))

            # Set CUDA device for each process
            device_id = local_rank
            torch.cuda.set_device(device_id)
            device = torch.device(f'cuda:{device_id}')
            print(f"World size: {world_size}, Local rank: {local_rank}, Global rank: {rank}")

            # Initialize distributed training environment
            if world_size > 1:
                try:
                    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
                    print("Distributed initialization SUCCESS")
                except Exception as e:
                    print(f"Distributed initialization FAILED: {str(e)}")
                    world_size = 1

            print("=" * 50)
            print("After distributed init:")
            print(f"LOCAL_RANK: {local_rank}")
            print(f"WORLD_SIZE: {world_size}")

        if dist.is_initialized():
            print(f"Dist WORLD_SIZE: {dist.get_world_size()}")
            print(f"Dist RANK: {dist.get_rank()}")
        else:
            print("Distributed NOT initialized")

        # ------------ load model ------------
        model_cfg = model_config_class(model_cfg=config.model_dump())
        if config.il.ckpt_to_load:
            print(f"load model from:{config.il.ckpt_to_load}")
        model = model_class.from_pretrained(pretrained_model_name_or_path=config.il.ckpt_to_load, config=model_cfg)
        if config.model_name in ["navdp", "bridgedp", "flownav_static", "flownav_dyn", "flownav_mix"]:
            model.to(device)
            for name, param in model.named_parameters():
                if config.model_name == "navdp" and 'mask_token' in name:
                    param.requires_grad = False
            # Check that all parameters and buffers are on the correct device
            for name, param in model.named_parameters():
                if param.device != device:
                    print(f"Parameter {name} is on wrong device {param.device}, should be moved to {device}")
                    param.data = param.data.to(device)

            for name, buffer in model.named_buffers():
                if buffer.device != device:
                    print(f"Buffer {name} is on wrong device {buffer.device}, should be moved to {device}")
                    buffer.data = buffer.data.to(device)

            # If distributed training, wrap the model with DDP
            if world_size > 1:
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
                )
        # ------------ load logger ------------
        train_logger_filename = os.path.join(config.log_dir, 'train.log')
        if dist.is_initialized() and dist.get_rank() == 0:
            train_logger = MyLogger(
                name='train',
                level=logging.INFO,
                format_str='%(asctime)-15s %(message)s',
                filename=train_logger_filename,
            )
        else:
            # Other processes use console logging
            train_logger = MyLogger(name='train', level=logging.INFO, format_str='%(asctime)-15s %(message)s')
        transformers_logger = logging.getLogger("transformers")
        if transformers_logger.hasHandlers():
            transformers_logger.handlers = []
        if config.model_name in ["navdp", "bridgedp", "flownav_static", "flownav_dyn", "flownav_mix"] and local_rank in [0, -1]:  # Only main process or non-distributed
            transformers_logger.addHandler(train_logger.handlers[0])
        transformers_logger.setLevel(logging.INFO)

        # ------------ load dataset ------------
        if config.model_name == "navdp":
            train_dataset_data = NavDP_Base_Datset(
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
            )
        elif config.model_name == "bridgedp":
            train_dataset_data = BridgeDP_Base_Dataset(
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
            )
        elif config.model_name == "flownav_static":
            train_dataset_data = FlowNav_Base_Datset(
                root_dirs=config.il.root_dir,
                preload_path=config.il.dataset_flownav,
                memory_size=config.il.memory_size,
                history_frames=config.il.history_frames,
                predict_frames=config.il.predict_frames,
                predict_size=config.il.predict_size,
                batch_size=config.il.batch_size,
                image_size=config.il.image_size,
                scene_data_scale=config.il.scene_scale,
                pixel_channel=config.il.pixel_channel,
                action_dim=config.il.action_dim,
                fallback_fps=config.il.fallback_fps,
                preload=config.il.preload,
                random_digit=config.il.random_digit,
                prior_sample=config.il.prior_sample,
            )
        elif config.model_name == "flownav_dyn":
            train_dataset_data = FlowNav_Dyn_Lerobot_Dataset(
                root_dirs=config.il.root_dir,
                preload_path=config.il.dataset_flownav,
                memory_size=config.il.memory_size,
                history_frames=config.il.history_frames,
                predict_frames=config.il.predict_frames,
                predict_size=config.il.predict_size,
                batch_size=config.il.batch_size,
                image_size=config.il.image_size,
                scene_data_scale=config.il.scene_scale,
                pixel_channel=config.il.pixel_channel,
                action_dim=config.il.action_dim,
                fallback_fps=config.il.fallback_fps,
                preload=config.il.preload,
                random_digit=config.il.random_digit,
                prior_sample=config.il.prior_sample,
                dynamic_time_tolerance_ns=config.il.dynamic_time_tolerance_ns,
                dynamic_grid_shape=config.il.dynamic_grid_shape,
                dynamic_grid_resolution=config.il.dynamic_grid_resolution,
                dynamic_grid_z_min=config.il.dynamic_grid_z_min,
                use_cached_dynamic_voxels=config.il.use_cached_dynamic_voxels,
                use_cached_est_dynamic_voxels=config.il.use_cached_est_dynamic_voxels,
                est_dynamic_voxel_subdir=config.il.est_dynamic_voxel_subdir,
                est_voxel_ratio=config.il.est_voxel_ratio,
                est_voxel_seed=config.il.est_voxel_seed,
                use_cached_pred_critic=config.il.use_cached_pred_critic,
                dynamic_weight=config.il.dynamic_weight,
                static_weight=config.il.static_weight,
                near_threshold=config.il.near_threshold,
            )
        elif config.model_name == "flownav_mix":
            static_root = getattr(config.il, "root_dir_static", config.il.root_dir)
            dyn_root = getattr(config.il, "root_dir_dyn", config.il.root_dir)
            static_index = getattr(config.il, "dataset_flownav_static", config.il.dataset_flownav)
            dyn_index = getattr(config.il, "dataset_flownav_dyn", config.il.dataset_flownav)

            static_dataset = FlowNav_Base_Datset(
                root_dirs=static_root,
                preload_path=static_index,
                memory_size=config.il.memory_size,
                history_frames=config.il.history_frames,
                predict_frames=config.il.predict_frames,
                predict_size=config.il.predict_size,
                batch_size=config.il.batch_size,
                image_size=config.il.image_size,
                scene_data_scale=config.il.scene_scale,
                pixel_channel=config.il.pixel_channel,
                action_dim=config.il.action_dim,
                fallback_fps=config.il.fallback_fps,
                preload=config.il.preload,
                random_digit=config.il.random_digit,
                prior_sample=config.il.prior_sample,
            )

            dyn_dataset = FlowNav_Dyn_Lerobot_Dataset(
                root_dirs=dyn_root,
                preload_path=dyn_index,
                memory_size=config.il.memory_size,
                history_frames=config.il.history_frames,
                predict_frames=config.il.predict_frames,
                predict_size=config.il.predict_size,
                batch_size=config.il.batch_size,
                image_size=config.il.image_size,
                scene_data_scale=config.il.scene_scale,
                pixel_channel=config.il.pixel_channel,
                action_dim=config.il.action_dim,
                fallback_fps=config.il.fallback_fps,
                preload=config.il.preload,
                random_digit=config.il.random_digit,
                prior_sample=config.il.prior_sample,
                dynamic_time_tolerance_ns=config.il.dynamic_time_tolerance_ns,
                dynamic_grid_shape=config.il.dynamic_grid_shape,
                dynamic_grid_resolution=config.il.dynamic_grid_resolution,
                dynamic_grid_z_min=config.il.dynamic_grid_z_min,
                use_cached_dynamic_voxels=config.il.use_cached_dynamic_voxels,
                use_cached_est_dynamic_voxels=config.il.use_cached_est_dynamic_voxels,
                est_dynamic_voxel_subdir=config.il.est_dynamic_voxel_subdir,
                est_voxel_ratio=config.il.est_voxel_ratio,
                est_voxel_seed=config.il.est_voxel_seed,
                use_cached_pred_critic=config.il.use_cached_pred_critic,
                dynamic_weight=config.il.dynamic_weight,
                static_weight=config.il.static_weight,
                near_threshold=config.il.near_threshold,
            )

            train_dataset_data = FlowNavMixDataset(
                static_dataset=static_dataset,
                dyn_dataset=dyn_dataset,
                static_ratio=getattr(config.il, "mix_static_ratio", 1.0),
                dyn_ratio=getattr(config.il, "mix_dyn_ratio", 1.0),
            )
        else:
            if '3dgs' in config.il.lmdb_features_dir or '3dgs' in config.il.lmdb_features_dir:
                dataset_root_dir = config.il.dataset_six_floor_root_dir
                dataset_type = '3dgs'
            elif 'grutopia' in config.il.lmdb_features_dir:
                dataset_root_dir = config.il.dataset_grutopia10_root_dir
                dataset_type = 'grutopia'
            else:
                dataset_root_dir = config.il.dataset_r2r_root_dir
                dataset_type = 'r2r'
            train_dataset_data = load_dataset(dataset_root_dir, 'train', logger=train_logger, dataset_type=dataset_type)
            global_batch_size = config.il.batch_size * len(config.torch_gpu_ids)

        # ------------ data_loader ------------
        if config.model_name in ['cma', 'seq2seq']:
            policy_trainer = CMATrainer
            train_dataset = CMALerobotDataset(
                config,
                config.il.lerobot_features_dir,
                config.il.use_iw,
                dataset_data=train_dataset_data,
                inflection_weight_coef=config.il.inflection_weight_coef,
                lmdb_map_size=config.il.lmdb_map_size,
                batch_size=config.il.batch_size,
            )
            collate_fn = cma_collate_fn

        elif config.model_name == 'rdp':
            policy_trainer = RDPTrainer
            train_dataset = RDP_LerobotDataset(
                config,
                config.il.lerobot_features_dir,
                dataset_data=train_dataset_data,
                batch_size=config.il.batch_size,
            )
            collate_fn = rdp_collate_fn(global_batch_size=global_batch_size)
        elif config.model_name == 'navdp':
            policy_trainer = NavDPTrainer
            train_dataset = train_dataset_data
            collate_fn = navdp_collate_fn
        elif config.model_name == 'bridgedp':
            policy_trainer = BridgeDPTrainer
            train_dataset = train_dataset_data
            collate_fn = bridgedp_collate_fn
        elif config.model_name == 'flownav_static':
            policy_trainer = FlowNavTrainer
            train_dataset = train_dataset_data
            collate_fn = flownav_collate_fn
        elif config.model_name == 'flownav_dyn':
            policy_trainer = FlowNavTrainer
            train_dataset = train_dataset_data
            collate_fn = flownav_dyn_collate_fn
        elif config.model_name == 'flownav_mix':
            policy_trainer = FlowNavTrainer
            train_dataset = train_dataset_data
            collate_fn = make_flownav_mix_collate_fn(
                predict_frames=config.il.predict_frames,
                dynamic_grid_shape=config.il.dynamic_grid_shape,
                image_size=config.il.image_size,
                fallback_fps=config.il.fallback_fps,
            )

        # ------------ training args ------------
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            run_name=config.name,
            remove_unused_columns=False,
            deepspeed='',
            gradient_checkpointing=False,
            bf16=False,  # fp16=False,
            tf32=False,
            per_device_train_batch_size=config.il.batch_size,
            gradient_accumulation_steps=1,
            dataloader_num_workers=config.il.num_workers,
            dataloader_pin_memory=False,
            optim='adamw_torch',
            learning_rate=config.il.lr,
            lr_scheduler_type='cosine',
            logging_steps=10.0,
            num_train_epochs=config.il.epochs,
            save_strategy='epoch',  # no
            save_steps=config.il.save_interval_epochs,
            save_total_limit=8,
            report_to=config.il.report_to,
            seed=0,
            do_eval=False,
            ddp_find_unused_parameters=config.il.ddp_find_unused_parameters,
            ddp_bucket_cap_mb=100,
            torch_compile_mode=None,
            dataloader_drop_last=True,
            disable_tqdm=True,
            log_level="info",
        )

        # Create the trainer
        trainer = policy_trainer(
            config=config, model=model, args=training_args, train_dataset=train_dataset, data_collator=collate_fn
        )

        # Add checkpoint format callback to ensure experiment_cfg is copied to each checkpoint
        run_name = config.name
        ckpt_format_callback = CheckpointFormatCallback(run_name=run_name, exp_cfg_dir=config.log_dir)
        trainer.add_callback(ckpt_format_callback)

        trainer.train()
        if train_logger:
            for handler in train_logger.handlers:
                handler.flush()
    except Exception as e:
        import traceback

        print(f"Unhandled exception: {str(e)}")
        print("Stack trace:")
        traceback.print_exc()

        # If distributed environment, ensure all processes exit
        if dist.is_initialized():
            dist.destroy_process_group()

        raise


if __name__ == '__main__':
    # Parse command line arguments using tyro
    config = tyro.cli(TrainCfg)

    # Print configuration information
    print('\n' + '=' * 50)
    print('FINE-TUNING CONFIGURATION:')
    print('=' * 50)
    for key, value in vars(config).items():
        print(f'{key}: {value}')
    print('=' * 50 + '\n')

    # Select configuration based on model_name
    supported_cfg = {
        'seq2seq': [seq2seq_exp_cfg, "Seq2Seq_Policy"],
        'seq2seq_plus': [seq2seq_plus_exp_cfg, 'Seq2Seq_Policy'],
        'cma': [cma_exp_cfg, "CMA_Policy"],
        'cma_plus': [cma_plus_exp_cfg, "CMA_Policy"],
        'rdp': [rdp_exp_cfg, "RDP_Policy"],
        'navdp': [navdp_exp_cfg, "NavDP_Policy"],
        'bridgedp': [bridgedp_exp_cfg, "BridgeDP_Policy"],
        'flownav_static': [flownav_static_exp_cfg, "FlowNav_Policy"],
        'flownav_dyn': [flownav_dyn_exp_cfg, "FlowNav_Policy"],
        'flownav_mix': [flownav_mix_exp_cfg, "FlowNav_Policy"],
    }

    if config.model_name not in supported_cfg:
        raise ValueError(f'Invalid model name: {config.model_name}. Supported models are: {list(supported_cfg.keys())}')

    exp_cfg, policy_name = supported_cfg[config.model_name]
    model_class, model_config_class = get_policy(policy_name), get_config(policy_name)

    exp_cfg.name = config.name
    exp_cfg.num_gpus = len(exp_cfg.torch_gpu_ids)
    exp_cfg.world_size = exp_cfg.num_gpus

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Validate GPU configuration
    assert (
        exp_cfg.num_gpus <= available_gpus
    ), f'Number of GPUs requested ({exp_cfg.num_gpus}) is greater than the available GPUs ({available_gpus})'
    assert exp_cfg.num_gpus > 0, 'Number of GPUs must be greater than 0'
    print(f'Using {exp_cfg.num_gpus} GPUs')

    main(exp_cfg, model_class, model_config_class)
