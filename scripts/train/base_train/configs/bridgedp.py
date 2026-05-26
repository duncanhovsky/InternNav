"""Bridge-DP 训练超参配置。

仿照 ``scripts/train/base_train/configs/navdp.py`` 的结构，
为 Bridge-DP 提供完整的训练参数配置。

与 NavDP 配置的差异：
1. model_name = 'bridgedp'
2. model = bridgedp_cfg（指向 BridgeDP_Policy）
3. 新增 Bridge-DP 专有超参：sigma_base, sigma_goal, n_prior_tokens
"""

from internnav.configs.model.bridgedp import bridgedp_cfg
from internnav.configs.trainer.eval import EvalCfg
from internnav.configs.trainer.exp import ExpCfg
from internnav.configs.trainer.il import FilterFailure, IlCfg, Loss

bridgedp_exp_cfg = ExpCfg(
    name='bridgedp_train',
    model_name='bridgedp',
    torch_gpu_id=0,
    torch_gpu_ids=[0],
    output_dir='checkpoints/%s/ckpts',
    tensorboard_dir='checkpoints/%s/tensorboard',
    checkpoint_folder='checkpoints/%s/ckpts',
    log_dir='checkpoints/%s/logs',
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
        batch_size=32,
        lr=1e-4,
        num_workers=8,
        weight_decay=1e-4,
        warmup_ratio=0.05,
        use_iw=True,
        inflection_weight_coef=3.2,
        save_interval_epochs=5,
        save_filter_frozen_weights=False,
        load_from_ckpt=False,
        ckpt_to_load='',
        lmdb_map_size=1e12,
        dataset_r2r_root_dir='/home/monika/dyishere/dataset/InternData/N1/InternData-N1-v0.5-mini/vln_pe/raw_data/r2r',
        lmdb_features_dir='r2r',
        lerobot_features_dir='/home/monika/dyishere/dataset/InternData/N1/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r',
        camera_name='pano_camera_0',
        report_to='tensorboard',
        # 复用 NavDP 的数据索引和数据目录
        dataset_navdp='checkpoints/preload_index.json',
        root_dir='/home/monika/dyishere/dataset/InternData/N1/InternData-N1-v0.5-mini/vln_n1/traj_data',
        image_size=224,
        scene_scale=1.0,
        preload=True,
        random_digit=False,
        prior_sample=False,
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
        filter_failure=FilterFailure(
            use=True, 
            min_rgb_nums=15
        ),
        loss=Loss(
            alpha=0.0001, 
            dist_scale=1, 
        ),
        # ── Bridge-DP 专有超参数 ──
        # sigma_base: 数据驱动固定常数（由 compute_sigma_base.py 离线计算）
        # sigma_goal: 弹性尾端松弛方差（PointGoal 模式下适中约束）
        # n_prior_tokens: PriorEncoder 输出 token 数量
        # num_train_timesteps: 训练时扩散步数（与 NavDP 保持一致，布朗桥端点约束加速收敛）
        # num_inference_timesteps: 推理时 DDIM 去噪步数（与 NavDP 保持一致）
        sigma_base=0.2,
        sigma_goal=0.01,
        sigma_floor=0.01,
        # NoGoal 分支不使用真实目标；默认向正前方展开，远端方差更大。
        nogoal_front_distance=0.8,  # normalized x distance, 0.8 ~= 4m before denorm
        nogoal_sigma_start=0.03,
        nogoal_sigma_x_end=0.35,
        nogoal_sigma_y_end=0.80,
        nogoal_sigma_theta_end=0.60,
        nogoal_sigma_power=2.0,
        # PointGoal 分支使用尺度相似的切向/法向各向异性桥方差。
        bridge_scale_invariant_sigma=True,
        bridge_anisotropic_xy=True,
        bridge_normal_sigma_ratio=1.0,
        bridge_tangent_sigma_ratio=0.05,
        bridge_theta_sigma_ratio=0.3,
        # 法向/航向自由度的前置程度；0.0 严格关闭前置影响，1.0 为温和前置实验。
        bridge_envelope_frontload=1.0,
        # PointGoal 样本级轨迹尺度归一化：有效轨迹统一到固定终点距离的形状空间。
        enable_trajectory_normalization=True,
        trajectory_norm_target_distance=2.0,
        trajectory_norm_min_distance_m=0.10,
        trajectory_norm_eps=1e-6,
        drop_short_trajectory_samples=True,
        # 尺度条件 token：显式告诉模型米制深度与形状空间之间的比例。
        enable_scale_condition_token=True,
        scale_condition_clamp_min_m=0.10,
        scale_condition_clamp_max_m=20.0,
        enable_scale_rgbd_film=True,
        scale_rgbd_film_alpha=1.0,
        scale_rgbd_film_zero_init=True,
        scale_rgbd_film_use_layernorm=True,
        # 推理候选排序：critic 分数减去目标一致性惩罚。
        enable_goal_consistency_score=True,
        goal_consistency_terminal_weight=1.0,
        goal_consistency_path_weight=0.05,
        # 距离分桶仅用于训练监控诊断，不参与模型规则。
        enable_distance_bucket_metrics=True,
        distance_bucket_edges=(0.10, 0.5, 0.8),
        distance_bucket_names=("static", "short", "mid", "long"),
        n_prior_tokens=4,
        num_train_timesteps=10,
        num_inference_timesteps=10,
        # use_origin_bridge_train: 训练时布朗桥起点固定为零向量（原点→目标）
        use_origin_bridge_train=False,
        use_prior_traj=False,
        # ── 增量一致性正则超参数 ──
        # lambda_delta: 增量一致性正则权重
        #   action_loss = L_x0 + lambda_delta * L_delta
        lambda_delta=0.1,
        # lambda_eps: x0 -> eps 反推噪声回归权重
        #   action_loss = L_x0 + lambda_delta * L_delta + lambda_eps * L_eps
        lambda_eps=0.0,
    ),
    model=bridgedp_cfg,
)
