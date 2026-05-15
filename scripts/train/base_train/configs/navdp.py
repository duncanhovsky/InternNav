from internnav.configs.model.navdp import navdp_cfg
from internnav.configs.trainer.eval import EvalCfg
from internnav.configs.trainer.exp import ExpCfg
from internnav.configs.trainer.il import FilterFailure, IlCfg, Loss

navdp_exp_cfg = ExpCfg(
    name='navdp_train',  # 实验名；会被拼接到输出目录中。
    model_name='navdp',  # 训练分支选择标识，train.py 中据此选择数据集与 Trainer。
    # num_gpus = 4,
    torch_gpu_id=0,  # 单卡场景主卡 ID（历史字段，当前脚本主要使用 torch_gpu_ids）。
    torch_gpu_ids=[0],  # 参与训练的 GPU 列表；其长度会影响全局 batch 计算。
    output_dir='checkpoints/%s/ckpts',  # HF Trainer 输出目录模板（%s 会替换为 name）。
    tensorboard_dir='checkpoints/%s/tensorboard',  # TensorBoard 日志目录模板。
    checkpoint_folder='checkpoints/%s/ckpts',  # 额外 checkpoint 目录模板（与 output_dir 保持一致）。
    log_dir='checkpoints/%s/logs',  # 文本日志目录模板。
    local_rank=0,  # DDP 本地进程序号（通常由启动器注入覆盖）。
    # device = None,
    seed=0,  # 全局随机种子，控制采样与初始化可复现性。
    eval=EvalCfg(
        use_ckpt_config=False,  # 评估时是否优先读取 checkpoint 内配置。
        save_results=True,  # 是否保存评估结果文件。
        split=['val_seen'],  # 评估划分集合。
        ckpt_to_load='',  # 评估加载的权重路径；空串表示按训练流程使用。
        max_steps=195,  # 单条轨迹评估最大步数。
        sample=False,  # 是否开启采样式评估。
        success_distance=3.0,  # 成功判定阈值（米）。
        start_eval_epoch=-1,  # 从哪个 epoch 开始评估；-1 常表示按默认策略。
        step_interval=50,  # 评估间隔（步数/周期，按上游解释使用）。
    ),
    il=IlCfg(
        epochs=1000,  # 训练总 epoch 数；配合较小学习率走长训练。
        batch_size=32,  # 单卡 batch；多卡时全局 batch=该值*卡数。
        lr=1e-4,  # 基础学习率。
        num_workers=4,  # DataLoader 工作进程数；受 CPU 核数与磁盘吞吐影响。
        weight_decay=1e-4,  # L2 正则强度。
        warmup_ratio=0.05,  # 学习率预热比例（前 5% 训练步）。
        use_iw=True,  # 是否启用 inflection weighting（转折点加权）。
        inflection_weight_coef=3.2,  # 转折点样本损失系数。
        save_interval_epochs=5,  # 每隔多少 epoch 保存一次 checkpoint。
        save_filter_frozen_weights=False,  # 保存时是否过滤冻结参数。
        load_from_ckpt=False,  # 是否从已有权重继续训练。
        ckpt_to_load='',  # 继续训练加载路径；空串表示从头训练。
        lmdb_map_size=1e12,  # LMDB 映射上限（大数据集防止 map 满）。
        dataset_r2r_root_dir='/media/monika/TSD302/dataset/InternData/N1/InternData-N1-v0.1-mini/vln_pe/raw_data/r2r',  # R2R 原始数据根目录。
        dataset_3dgs_root_dir='',  # 3DGS 数据根目录（当前该配置未使用）。
        dataset_grutopia10_root_dir='',  # Grutopia 数据根目录（当前该配置未使用）。
        lmdb_features_dir='r2r',  # 特征目录标识；train.py 据此推断数据类型。
        lerobot_features_dir='/media/monika/TSD302/dataset/InternData/N1/InternData-N1-v0.1-mini/vln_pe/traj_data/r2r',  # LeRobot 特征路径（非 navdp 分支常用）。
        camera_name='pano_camera_0',  # 相机名称约定（用于特征选择/兼容字段）。
        report_to='tensorboard',  # 日志后端：wandb / tensorboard / none。
        dataset_navdp='checkpoints/bridgedp_train/preload_index.json',  # NavDP 预索引文件路径。
        root_dir='/media/monika/TSD302/dataset/InternData/N1/InternData-N1-v0.1-mini/vln_n1/traj_data',  # NavDP 轨迹数据根目录。
        image_size=224,  # 输入图像分辨率（RGB/Depth 统一到该尺寸）。
        scene_scale=1.0,  # 场景采样比例，1.0 表示不过滤场景。
        preload=True,  # 是否直接从 dataset_navdp 读取预构建索引。
        random_digit=False,  # 是否随机采样时序步长。
        prior_sample=False,  # 是否启用障碍密度优先采样。
        memory_size=8,  # 历史记忆帧长度（时序上下文窗口）。
        predict_size=24,  # 未来动作预测长度（时间步数）。
        pixel_channel=4,  # 像素目标通道数（常见 4 或 7）。
        temporal_depth=16,  # 时序编码深度（模型结构参数）。
        heads=8,  # Transformer 多头注意力头数。
        token_dim=384,  # token 隐向量维度。
        channels=3,  # 图像输入通道数。
        dropout=0.1,  # Dropout 概率。
        scratch=False,  # 是否从随机初始化训练。
        finetune=False,  # 是否进入微调模式。
        ddp_find_unused_parameters=True,  # DDP 未使用参数检查开关；复杂分支模型建议开启。
        filter_failure=FilterFailure(
            use=True,  # 是否过滤失败轨迹样本。
            min_rgb_nums=15,  # 最少 RGB 帧数阈值
最大点误差: 0.0689，小于该值的轨迹会被过滤。
        ),
        loss=Loss(
            alpha=0.0001,  # 损失项权重超参（由模型内部读取）。
            dist_scale=1,  # 距离损失缩放系数。
        ),
    ),
    model=navdp_cfg,  # 模型结构配置入口（policy_name/state_encoder 等）。
)
