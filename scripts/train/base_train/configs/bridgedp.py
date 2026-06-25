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
        # 训练数据多样性：随机 memory/pred 时间步长，扩大 24 点轨迹的尺度与曲率分布。
        random_digit=True,
        # 困难样本采样：按障碍密度偏置起终点选择，增加绕障片段占比。
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
        # 这些桥方差会进入训练前向 sample_bridge_noise()，不是仅用于推理采样。
        bridge_scale_invariant_sigma=True,
        bridge_anisotropic_xy=True,
        # 横向扰动，最直接增加左右绕障、多路径绕行的可能性。这个增大最有利于绕障多样性。
        bridge_normal_sigma_ratio=6.0,
        # 沿目标方向的前后扰动，增大会让轨迹更容易提前/滞后、拉长/回退。它会增加差异，但对绕障帮助不如 normal，太大会带来绕路、抖动或目标一致性下降。
        bridge_tangent_sigma_ratio=0.08,
        # 航向扰动，增大会让轨迹姿态更多样，有利于转向探索，但太大会让 MPC 跟踪变难，出现大角速度或姿态摆动。
        bridge_theta_sigma_ratio=0.8,
        bridge_virtual_prefix_steps=8.0,
        # v1.0.3: PointGoal bridge anchor sampling。真实 pointgoal 仍作为任务目标；
        # sampled anchor 只用于布朗桥均值、方差坐标系、推理初始化和反向 step。
        enable_bridge_anchor_sampling=True,
        # 训练时使用 sampled anchor 的样本比例；0 表示训练完全保持原行为。
        bridge_anchor_train_prob=0.40,
        # 推理候选中保留第一组原始 pointgoal anchor，保证有无偏移候选可回退。
        bridge_anchor_keep_original_sample=True,

        # anchor 角度扰动：中心候选使用截断高斯，边缘候选使用左右成对的反高斯/edge-biased 扰动。
        # 表示主要采样分布的高斯标准差，单位是 rad。
        bridge_anchor_angle_std=0.60,
        # 这是硬限制。只要该值大于 0，扰动角会被限制在 [-max, max] 范围内。过大可能导致训练不稳定，过小可能限制多样性。
        bridge_anchor_angle_max=1.30,
        # 旧版 uniform 混合，默认关闭；需要完全随机覆盖边界时再打开。
        bridge_anchor_uniform_prob=0.0,
        # 一半扰动候选改用 edge-biased 分布；推理多候选时按左右成对分配，训练 sample_num=1 时按 batch 概率生效。
        bridge_anchor_edge_prob=0.40,

        # v1.0.3: 监督重采样模式。
        # arc_length 保持旧行为；projection 强制按起终点 chord 投影均匀；
        # hybrid_projection 在投影单调且覆盖充分时使用 projection，否则回退 arc_length。
        trajectory_resample_mode="hybrid_projection",
        trajectory_projection_monotonic_eps=1e-4,
        trajectory_projection_min_span=0.80,
        trajectory_projection_flat_lateral_eps=1e-3,
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
        # Treat the hard core as an approximate robot footprint and the near
        # shell as a clearance margin, not just a sparse centerline distance.
        critic_near_threshold=0.30,
        critic_hard_threshold=0.15,
        critic_soft_beta=4.0,
        critic_max_weight=10.0,
        critic_mean_weight=4.0,
        critic_trend_weight=0.15,
        critic_safe_score=2.0,
        critic_densify_step=0.05,
        # 距离分桶仅用于训练监控诊断，不参与模型规则。
        enable_distance_bucket_metrics=True,
        distance_bucket_edges=(0.10, 0.5, 0.8),
        distance_bucket_names=("static", "short", "mid", "long"),
        n_prior_tokens=4,
        num_train_timesteps=10,
        num_inference_timesteps=10,
        inference_eta=0.03,
        # use_origin_bridge_train: 训练时布朗桥起点固定为零向量（原点→目标）
        use_origin_bridge_train=False,
        use_prior_traj=False,
        # ── 增量一致性正则超参数 ──
        # lambda_delta: 增量一致性正则权重
        # Keep velocity, acceleration and turning trend close to the supervised path.
        lambda_delta=0.10,
        lambda_acc=0.05,
        lambda_turn_consistency=0.05,
        # lambda_eps: x0 -> eps 反推噪声回归权重
        lambda_eps=0.0,
        # Pairwise critic ranking: when label/augment have different geometric
        # safety scores, train the critic to order the safer trajectory higher.
        lambda_critic_rank=0.4,
        critic_rank_margin=0.8,
        critic_rank_target_min_gap=0.05,
        # Generator-side safety loss: directly penalize predicted x0 trajectories
        # whose densified local path collides with or brushes obstacle points.
        lambda_generator_safety=0.08,
        generator_safety_hard_threshold=0.15,
        generator_safety_near_threshold=0.30,
        generator_safety_hard_weight=6.0,
        generator_safety_near_weight=1.0,
        generator_safety_segment_substeps=4,
        generator_safety_hard_topk=8,
        generator_safety_warmup_steps=5000,
    ),
    model=bridgedp_cfg,
)
