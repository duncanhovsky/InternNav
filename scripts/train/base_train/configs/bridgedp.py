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
        num_workers=4,
        weight_decay=1e-4,
        warmup_ratio=0.05,
        use_iw=True,
        inflection_weight_coef=3.2,
        save_interval_epochs=5,
        save_filter_frozen_weights=False,
        load_from_ckpt=False,
        ckpt_to_load='',
        lmdb_map_size=1e12,
        dataset_r2r_root_dir='/media/monika/TSD302/dataset/InternData/N1/InternData-N1-v0.1-mini/vln_pe/raw_data/r2r',
        lmdb_features_dir='r2r',
        lerobot_features_dir='/media/monika/TSD302/dataset/InternData/N1/InternData-N1-v0.1-mini/vln_pe/traj_data/r2r',
        camera_name='pano_camera_0',
        report_to='tensorboard',
        # 复用 NavDP 的数据索引和数据目录
        dataset_navdp='checkpoints/bridgedp_train/preload_index.json',
        root_dir='/media/monika/TSD302/dataset/InternData/N1/InternData-N1-v0.1-mini/vln_n1/traj_data',
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
        # sigma_goal: 弹性尾端松弛方差
        # n_prior_tokens: PriorEncoder 输出 token 数量
        # num_train_timesteps: 训练时扩散步数（增大以提供足够的噪声水平覆盖）
        # num_inference_timesteps: 推理时 DDIM 去噪步数
        sigma_base=0.0813,
        sigma_goal=0.001,
        n_prior_tokens=4,
        num_train_timesteps=100,
        num_inference_timesteps=100,
        use_prior_traj=False,
    ),
    model=bridgedp_cfg,
)
