"""Bridge-DP full training config for 4x A800 on an SSD workspace."""

import os

from internnav.configs.model.bridgedp import bridgedp_cfg
from internnav.configs.trainer.eval import EvalCfg
from internnav.configs.trainer.exp import ExpCfg
from internnav.configs.trainer.il import FilterFailure, IlCfg, Loss


def _env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


def _env_float(name, default):
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else default


SSD_ROOT = os.environ.get("BRIDGEDP_SSD_ROOT", "/ssd")
PROJECT_ROOT = os.environ.get("BRIDGEDP_PROJECT_ROOT", f"{SSD_ROOT}/MyResearch/InternNav")
DATASET_BASE = os.environ.get(
    "BRIDGEDP_DATASET_BASE",
    f"{SSD_ROOT}/datasets/InternData-N1/v0.5-full-vln-n1",
)
DATASET_ROOT = os.environ.get(
    "BRIDGEDP_DATASET_ROOT",
    f"{DATASET_BASE}/vln_n1/traj_data",
)
PRELOAD_INDEX = os.environ.get(
    "BRIDGEDP_PRELOAD_INDEX",
    f"{PROJECT_ROOT}/checkpoints/preload_index.json",
)
NUM_GPUS = _env_int("BRIDGEDP_NUM_GPUS", 4)


bridgedp_full_exp_cfg = ExpCfg(
    name=os.environ.get("BRIDGEDP_RUN_NAME", "bridgedp_full"),
    model_name='bridgedp',
    resume_from_checkpoint=os.environ.get("BRIDGEDP_RESUME_FROM", ""),
    auto_resume=os.environ.get("BRIDGEDP_AUTO_RESUME", "1") != "0",
    torch_gpu_id=0,
    torch_gpu_ids=list(range(NUM_GPUS)),
    output_dir=f'{PROJECT_ROOT}/checkpoints/%s/ckpts',
    tensorboard_dir=f'{PROJECT_ROOT}/checkpoints/%s/tensorboard',
    checkpoint_folder=f'{PROJECT_ROOT}/checkpoints/%s/ckpts',
    log_dir=f'{PROJECT_ROOT}/checkpoints/%s/logs',
    local_rank=0,
    seed=0,
    eval=EvalCfg(
        use_ckpt_config=False,
        save_results=True,
        split=['val_seen'],
        ckpt_to_load='',
        max_steps=195,
        sample=False,
        success_distance=3.0,
        start_eval_epoch=-1,
        step_interval=50,
    ),
    il=IlCfg(
        epochs=1000,
        batch_size=_env_int("BRIDGEDP_BATCH_SIZE", 96),
        lr=_env_float("BRIDGEDP_LR", 3e-4),
        num_workers=_env_int("BRIDGEDP_NUM_WORKERS", 10),
        weight_decay=1e-4,
        warmup_ratio=0.05,
        use_iw=True,
        inflection_weight_coef=3.2,
        save_interval_epochs=5,
        save_filter_frozen_weights=False,
        load_from_ckpt=False,
        ckpt_to_load='',
        lmdb_map_size=1e12,
        dataset_r2r_root_dir=os.environ.get(
            "BRIDGEDP_R2R_ROOT",
            f"{DATASET_BASE}/vln_pe/raw_data/r2r",
        ),
        lmdb_features_dir='r2r',
        lerobot_features_dir=os.environ.get(
            "BRIDGEDP_LEROBOT_FEATURES_DIR",
            f"{DATASET_BASE}/vln_pe/traj_data/r2r",
        ),
        camera_name='pano_camera_0',
        report_to=os.environ.get("BRIDGEDP_REPORT_TO", "tensorboard"),
        dataset_navdp=PRELOAD_INDEX,
        root_dir=DATASET_ROOT,
        image_size=224,
        scene_scale=1.0,
        preload=True,
        random_digit=True,
        prior_sample=True,
        memory_size=8,
        predict_size=24,
        pixel_channel=4,
        temporal_depth=16,
        heads=8,
        token_dim=384,
        channels=3,
        dropout=0.1,
        scratch=False,
        finetune=False,
        ddp_find_unused_parameters=True,
        gradient_accumulation_steps=_env_int("BRIDGEDP_GRAD_ACCUM", 1),
        bf16=os.environ.get("BRIDGEDP_BF16", "1") != "0",
        tf32=os.environ.get("BRIDGEDP_TF32", "1") != "0",
        dataloader_pin_memory=os.environ.get("BRIDGEDP_PIN_MEMORY", "1") != "0",
        save_total_limit=_env_int("BRIDGEDP_SAVE_TOTAL_LIMIT", 20),
        logging_steps=_env_int("BRIDGEDP_LOGGING_STEPS", 10),
        filter_failure=FilterFailure(
            use=True,
            min_rgb_nums=15,
        ),
        loss=Loss(
            alpha=0.0001,
            dist_scale=1,
        ),
        sigma_base=0.2,
        sigma_goal=0.01,
        sigma_floor=0.01,
        nogoal_front_distance=0.8,
        nogoal_sigma_start=0.03,
        nogoal_sigma_x_end=0.35,
        nogoal_sigma_y_end=0.80,
        nogoal_sigma_theta_end=0.60,
        nogoal_sigma_power=2.0,
        bridge_scale_invariant_sigma=True,
        bridge_anisotropic_xy=True,
        bridge_normal_sigma_ratio=6.0,
        bridge_tangent_sigma_ratio=0.3,
        bridge_theta_sigma_ratio=1.2,
        bridge_virtual_prefix_steps=8.0,
        enable_bridge_anchor_sampling=True,
        bridge_anchor_train_prob=0.5,
        bridge_anchor_keep_original_sample=True,
        bridge_anchor_angle_std=0.25,
        bridge_anchor_angle_max=0.65,
        bridge_anchor_uniform_prob=0.0,
        bridge_anchor_edge_prob=0.5,
        trajectory_resample_mode="hybrid_projection",
        trajectory_projection_monotonic_eps=1e-4,
        trajectory_projection_min_span=0.80,
        trajectory_projection_flat_lateral_eps=1e-3,
        enable_trajectory_normalization=True,
        trajectory_norm_target_distance=2.0,
        trajectory_norm_min_distance_m=0.10,
        trajectory_norm_eps=1e-6,
        drop_short_trajectory_samples=True,
        enable_scale_condition_token=True,
        scale_condition_clamp_min_m=0.10,
        scale_condition_clamp_max_m=20.0,
        enable_scale_rgbd_film=True,
        scale_rgbd_film_alpha=1.0,
        scale_rgbd_film_zero_init=True,
        scale_rgbd_film_use_layernorm=True,
        enable_goal_consistency_score=True,
        goal_consistency_terminal_weight=1.0,
        goal_consistency_path_weight=0.05,
        critic_near_threshold=0.3,
        critic_hard_threshold=0.1,
        critic_soft_beta=4.0,
        critic_max_weight=5.0,
        critic_mean_weight=2.0,
        critic_trend_weight=0.5,
        critic_safe_score=2.0,
        enable_distance_bucket_metrics=True,
        distance_bucket_edges=(0.10, 0.5, 0.8),
        distance_bucket_names=("static", "short", "mid", "long"),
        n_prior_tokens=4,
        num_train_timesteps=10,
        num_inference_timesteps=10,
        inference_eta=0.10,
        use_origin_bridge_train=False,
        use_prior_traj=False,
        lambda_delta=0.0,
        lambda_eps=0.0,
    ),
    model=bridgedp_cfg,
)
