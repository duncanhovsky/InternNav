from internnav.configs.model.flownav import flownav_cfg
from internnav.configs.trainer.eval import EvalCfg
from internnav.configs.trainer.exp import ExpCfg
from internnav.configs.trainer.il import FilterFailure, IlCfg, Loss


# ==============================
# FlowNav 静态数据集训练配置
# ==============================
flownav_static_exp_cfg = ExpCfg(
	name='flownav_static_train',  # 实验名；会写入输出目录与日志目录。
	model_name='flownav_static',  # 训练入口分支名：FlowNav + 静态数据集。
	torch_gpu_id=0,  # 主 GPU id（历史字段，实际多卡以 torch_gpu_ids 为准）。
	torch_gpu_ids=[0],  # 使用的 GPU 列表，长度影响全局 batch。
	output_dir='checkpoints/%s/ckpts',  # checkpoint 根目录模板。
	tensorboard_dir='checkpoints/%s/tensorboard',  # TensorBoard 日志目录模板。
	checkpoint_folder='checkpoints/%s/ckpts',  # 额外 checkpoint 目录模板。
	log_dir='checkpoints/%s/logs',  # 文本日志目录模板。
	local_rank=0,  # DDP 本地 rank，分布式启动时会被环境变量覆盖。
	seed=0,  # 随机种子，影响采样与初始化的可复现性。
	eval=EvalCfg(
		use_ckpt_config=False,  # 评估阶段是否优先读取 ckpt 自带配置。
		save_results=True,  # 是否保存评估结果文件。
		split=['val_seen'],  # 评估 split。
		ckpt_to_load='',  # 指定评估权重路径；空串为默认行为。
		max_steps=195,  # 单 episode 最大评估步数。
		sample=False,  # 是否启用采样式评估。
		success_distance=3.0,  # 成功阈值（米）。
		start_eval_epoch=-1,  # 从哪个 epoch 起开始评估。
		step_interval=50,  # 评估间隔。
	),
	il=IlCfg(
		epochs=1000,  # 总训练 epoch；调大可提升收敛上限但耗时更长。
		batch_size=16,  # 单卡 batch；增大提升吞吐但会增加显存占用。
		lr=1e-4,  # 学习率；增大收敛更快但更易震荡。
		num_workers=8,  # 数据加载进程数；过小会导致 GPU 等数据。
		weight_decay=1e-4,  # L2 正则强度；增大可抑制过拟合。
		warmup_ratio=0.05,  # 预热比例；增大可降低训练初期不稳定。
		save_interval_epochs=5,  # 保存间隔 epoch；减小可更频繁留档。
		save_filter_frozen_weights=False,  # 保存时是否过滤冻结参数。
		load_from_ckpt=False,  # 是否从 checkpoint 续训。
		ckpt_to_load='',  # 续训权重路径。
		report_to='tensorboard',  # 日志后端。

		# ====== FlowNav 静态数据集相关 ======
		root_dir='data/datasets/InternData-N1/vln_n1/traj_data',  # 数据根目录。
		dataset_flownav='data/datasets/flownav_dataset_lerobot.json',  # 预构建索引路径。
		preload=False,  # True: 直接读索引；False: 扫描目录重建索引。
		scene_scale=1.0,  # 场景采样比例；<1 可减小训练集规模。
		random_digit=False,  # 是否随机采样时间步长；True 可增强时序鲁棒性。
		prior_sample=False,  # 是否按障碍密度优先采样；True 样本更“难”。
		image_size=224,  # 图像统一分辨率；增大可保留细节但更耗显存。
		memory_size=8,  # 历史 RGB 帧长度；增大可提升时序上下文。
		history_frames=2,  # 历史深度帧长度；增大可增强动态估计上下文。
		predict_size=24,  # 动作预测长度；增大可覆盖更远规划。
		predict_frames=8,  # dynamic_voxels 时间维长度；需与模型假设一致。
		pixel_channel=4,  # 像素目标通道数（4/7）。
		action_dim=3,  # 动作维度；若与数据不一致会触发 pad/截断。
		fallback_fps=30.0,  # 缺失时间戳时的回退帧率；影响时间轴物理量。

		# ====== FlowNav 模型关键超参（flownav_policy 必需） ======
		temporal_depth=16,  # TransformerDecoder 层数。
		heads=8,  # 多头注意力头数。
		token_dim=384,  # token 维度；增大提升容量但更耗显存。
		channels=3,  # 图像输入通道数。
		dropout=0.1,  # Dropout 概率。
		scratch=False,  # 是否从头训练。
		finetune=False,  # 是否解冻 backbone 微调。
		rgb_checkpoint='',  # RGB backbone 预训练权重路径。
		dynamics_checkpoint='',  # Dynamics backbone 权重路径。
		input_dtype='float32',  # 输入数据精度；可改为 bfloat16 降显存。
		hidden_dim=256,  # 融合模块隐藏维度。
		tgt_mask_mode='none',  # 解码 mask 模式：none/lookahead。
		tgt_mask_lookahead=0,  # lookahead 模式可见未来步数。

		ddp_find_unused_parameters=True,  # 分支模型建议开启，避免 DDP 参数遗漏报错。
		filter_failure=FilterFailure(
			use=True,  # 是否过滤异常/失败轨迹。
			min_rgb_nums=15,  # 最少 RGB 帧阈值。
		),
		loss=Loss(
			alpha=0.0001,  # 损失缩放超参（按模型内部读取）。
			dist_scale=1,  # 距离损失缩放系数。
		),
	),
	model=flownav_cfg,
)


# ==============================
# FlowNav 动态数据集训练配置
# ==============================
flownav_dyn_exp_cfg = ExpCfg(
	name='flownav_dyn_train',  # 动态场景实验名。
	model_name='flownav_dyn',  # 训练入口分支名：FlowNav + 动态数据集。
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
		batch_size=16,
		lr=1e-4,
		num_workers=8,
		weight_decay=1e-4,
		warmup_ratio=0.05,
		save_interval_epochs=5,
		save_filter_frozen_weights=False,
		load_from_ckpt=False,
		ckpt_to_load='',
		report_to='tensorboard',

		# ====== FlowNav 动态数据集通用项 ======
		root_dir='data/datasets/InternData-N1/vln_n1/traj_data',
		dataset_flownav='data/datasets/flownav_dyn_dataset_lerobot.json',
		preload=False,
		scene_scale=1.0,
		random_digit=False,
		prior_sample=False,
		image_size=224,
		memory_size=8,
		history_frames=2,
		predict_size=24,
		predict_frames=8,
		pixel_channel=4,
		action_dim=3,
		fallback_fps=30.0,

		# ====== 动态版新增项（影响 dynamic_voxels 与 critic 监督） ======
		dynamic_time_tolerance_ns=20_000_000,  # 动态轨迹时间对齐容差；过小会丢目标，过大易错配。
		dynamic_grid_shape=(32, 32, 16),  # 动态体素网格大小；增大更细致但显存和算力开销更大。
		dynamic_grid_resolution=0.25,  # 体素分辨率（米）；减小可提升几何精度。
		dynamic_grid_z_min=-1.0,  # 体素 z 轴起点（局部坐标）。
		use_cached_dynamic_voxels=True,  # 是否优先使用 GT 动态体素缓存。
		use_cached_est_dynamic_voxels=True,  # 是否读取 dyn_module 估计体素缓存。
		est_dynamic_voxel_subdir='dyn_module_voxel',  # 估计体素缓存目录名。
		est_voxel_ratio=0.0,  # 估计体素混入比例；从 0->1 可做课程学习。
		est_voxel_seed=0,  # 混入采样随机种子。
		use_cached_pred_critic=True,  # pred_critic 是否优先读离线缓存。
		dynamic_weight=1.0,  # critic 动态距离权重。
		static_weight=1.0,  # critic 静态距离权重。
		near_threshold=0.1,  # critic 近碰阈值（米）。

		# ====== FlowNav 模型关键超参 ======
		temporal_depth=16,
		heads=8,
		token_dim=384,
		channels=3,
		dropout=0.1,
		scratch=False,
		finetune=False,
		rgb_checkpoint='',
		dynamics_checkpoint='',
		input_dtype='float32',
		hidden_dim=256,
		tgt_mask_mode='none',
		tgt_mask_lookahead=0,

		ddp_find_unused_parameters=True,
		filter_failure=FilterFailure(
			use=True,
			min_rgb_nums=15,
		),
		loss=Loss(
			alpha=0.0001,
			dist_scale=1,
		),
	),
	model=flownav_cfg,
)


# ==============================
# FlowNav 静态+动态混合训练配置
# ==============================
flownav_mix_exp_cfg = ExpCfg(
	name='flownav_mix_train',  # 混合训练实验名。
	model_name='flownav_mix',  # 单模型混合数据分支名：同一个 FlowNav 同时学习静态+动态数据。
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
		batch_size=16,
		lr=1e-4,
		num_workers=8,
		weight_decay=1e-4,
		warmup_ratio=0.05,
		save_interval_epochs=5,
		save_filter_frozen_weights=False,
		load_from_ckpt=False,
		ckpt_to_load='',
		report_to='tensorboard',

		# ====== 混合数据集路径配置 ======
		root_dir='data/datasets/InternData-N1/vln_n1/traj_data',  # 默认根目录；未单独指定静态/动态根目录时使用。
		root_dir_static='data/datasets/InternData-N1/vln_n1/traj_data',  # 静态数据根目录。
		root_dir_dyn='data/datasets/InternData-N1/vln_n1/traj_data',  # 动态数据根目录。
		dataset_flownav='data/datasets/flownav_dataset_lerobot.json',  # 默认索引路径（兜底）。
		dataset_flownav_static='data/datasets/flownav_dataset_lerobot.json',  # 静态索引路径。
		dataset_flownav_dyn='data/datasets/flownav_dyn_dataset_lerobot.json',  # 动态索引路径。
		preload=False,
		scene_scale=1.0,
		random_digit=False,
		prior_sample=False,

		# ====== 混合采样策略 ======
		mix_static_ratio=1.0,  # 静态样本采样权重；增大可提升静态数据占比。
		mix_dyn_ratio=1.0,  # 动态样本采样权重；增大可提升动态数据占比。

		# ====== 通用数据参数 ======
		image_size=224,
		memory_size=8,
		history_frames=2,
		predict_size=24,
		predict_frames=8,
		pixel_channel=4,
		action_dim=3,
		fallback_fps=30.0,

		# ====== 动态子集参数 ======
		dynamic_time_tolerance_ns=20_000_000,
		dynamic_grid_shape=(32, 32, 16),
		dynamic_grid_resolution=0.25,
		dynamic_grid_z_min=-1.0,
		use_cached_dynamic_voxels=True,
		use_cached_est_dynamic_voxels=True,
		est_dynamic_voxel_subdir='dyn_module_voxel',
		est_voxel_ratio=0.0,
		est_voxel_seed=0,
		use_cached_pred_critic=True,
		dynamic_weight=1.0,
		static_weight=1.0,
		near_threshold=0.1,

		# ====== FlowNav 模型关键超参 ======
		temporal_depth=16,
		heads=8,
		token_dim=384,
		channels=3,
		dropout=0.1,
		scratch=False,
		finetune=False,
		rgb_checkpoint='',
		dynamics_checkpoint='',
		input_dtype='float32',
		hidden_dim=256,
		tgt_mask_mode='none',
		tgt_mask_lookahead=0,

		ddp_find_unused_parameters=True,
		filter_failure=FilterFailure(
			use=True,
			min_rgb_nums=15,
		),
		loss=Loss(
			alpha=0.0001,
			dist_scale=1,
		),
	),
	model=flownav_cfg,
)
