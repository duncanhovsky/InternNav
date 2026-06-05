# Bridge-DP 纯代码分析报告

> 分析日期：2026-06-04  
> 分析范围：静态代码与配置，不包含训练结果、仿真结果或真实机器人实验结果。  
> 核心原则：只把代码中已经接通、默认配置已启用、训练/推理链路实际使用的机制认定为当前现状；已实现但默认关闭、仅文档设想、或缺少部署闭环的内容单独标注为潜在能力或未验证能力。

## 1. 总体结论

Bridge-DP 当前代码已经形成了一条完整的训练链路：`bridgedp` 配置注册到训练入口，使用独立的 `BridgeDP_Base_Dataset`、`BridgeDPTrainer`、`BridgeDPNet`、`BridgeScheduler` 和 `bridgedp_critic`。默认实验形态可以概括为：

**绝对轨迹监督 + GPU 弧长重采样 + 轨迹尺度归一化 + 布朗桥 x0 预测 + RGB-D 条件 + learned critic 候选排序。**

但需要特别强调：默认配置中 `use_prior_traj=False`（`scripts/train/base_train/configs/bridgedp.py:149`）。因此，虽然模型代码实现了 `PriorEncoder` 和 `VisualGate`（`internnav/model/basemodel/bridgedp/bridgedp_policy.py:254-260`），默认训练/推理并不会真正使用上一帧轨迹先验。当前默认版本更适合被称为“布朗桥轨迹生成版 Bridge-DP”，还不能被严格认定为完整的“连续任务先验注入版 Bridge-DP”。

从应用场景看，当前代码对常规室内导航、较大静态障碍物、目标明确的点目标导航有结构基础；对室外泛化、起步贴近障碍、必须先原地转向、大碰撞体积机器人、不同机器人尺寸迁移、细长障碍物和低矮小障碍物，代码层面都存在明确短板或需要数据/实验验证的风险。

## 2. 系统级代码分析

### 2.1 注册与训练入口

Bridge-DP 与 NavDP 平级注册，而不是继承 NavDP 类后做局部覆盖。训练入口中直接导入 Bridge-DP 数据集和 collate 函数（`scripts/train/base_train/train.py:26`），并在 `bridgedp` 分支创建 `BridgeDP_Base_Dataset`（`scripts/train/base_train/train.py:613`）。Trainer 选择也有独立分支：`BridgeDPTrainer` 在训练入口中被导入（`scripts/train/base_train/train.py:31`），并在 `config.model_name == 'bridgedp'` 时使用（`scripts/train/base_train/train.py:784`）。模型注册通过 `BridgeDP_Policy` 映射到 `BridgeDPNet` 和 `BridgeDPModelConfig`（`internnav/model/__init__.py:29`, `internnav/model/__init__.py:69`）。

这说明 Bridge-DP 的系统架构已经接入主训练框架，不是一个孤立草稿。但它目前主要覆盖训练和模型推理接口，缺少面向机器人闭环部署的系统层模块，例如 footprint-aware local planner、速度/角速度约束器、急停策略、不同机器人尺寸配置接口。

### 2.2 默认配置真实启用的能力

默认训练配置启用了困难样本采样和时序步长随机化：`prior_sample=True`、`random_digit=True`、`memory_size=8`、`predict_size=24`（`scripts/train/base_train/configs/bridgedp.py:65-69`）。布朗桥相关配置中，`sigma_base=0.2`、`sigma_goal=0.01`，并启用 `bridge_scale_invariant_sigma=True` 和较大的横向扰动比例 `bridge_normal_sigma_ratio=2.0`（`scripts/train/base_train/configs/bridgedp.py:93-108`）。这说明当前思路明显鼓励横向候选扩展和绕障多样性。

默认也启用了轨迹尺度归一化与尺度条件 token：`enable_trajectory_normalization=True`、`enable_scale_condition_token=True`（`scripts/train/base_train/configs/bridgedp.py:115-121`）。这对不同目标距离的轨迹形状统一有帮助。

但默认关闭了两个关键能力：

- `enable_goal_consistency_score=False`（`scripts/train/base_train/configs/bridgedp.py:129`），推理排序默认主要依赖 learned critic，而不是显式终点一致性惩罚。
- `use_prior_traj=False`（`scripts/train/base_train/configs/bridgedp.py:149`），先验轨迹注入模块默认不参与主干决策。

此外，`lambda_delta=0.0`、`lambda_eps=0.0`（`scripts/train/base_train/configs/bridgedp.py:153-156`），说明增量平滑正则和 eps 一致性正则虽然在 Trainer 中有代码路径，但默认并不进入损失。

### 2.3 数据链路

Dataset 从 Parquet 的 `action` 字段读取逐帧世界位姿，而不是固定相机外参；这个逻辑在 `process_data_parquet()` 中实现（`internnav/dataset/bridgedp_lerobot_dataset.py:312`）。障碍物来自 `pointcloud.ply` 的颜色标注，`process_obstacle_points()` 通过颜色接近 `[0, 0, 0.5]` 提取障碍物点（`internnav/dataset/bridgedp_lerobot_dataset.py:339`）。

训练样本构造时，`rank_steps()` 按未来轨迹包围盒内的障碍点密度偏置采样起止片段（`internnav/dataset/bridgedp_lerobot_dataset.py:527`）。这对增加绕障、狭窄、障碍密集片段的训练频率有帮助，但它只统计障碍点数量，不理解机器人尺寸、障碍形状、通道可通过宽度，也不区分低矮障碍和真正阻挡身体的障碍。

Dataset 的 `pred_actions`、`augment_actions` 和 `prior_traj` 在当前实现中是占位零数组（`internnav/dataset/bridgedp_lerobot_dataset.py:665-667`）；真正用于训练的轨迹来自 `raw_pred_actions` 和 `raw_augment_actions`（`internnav/dataset/bridgedp_lerobot_dataset.py:660`），在 collate 后进入 Trainer。因此分析训练监督时不能只看 Dataset 返回的 `pred_actions`，必须看 Trainer 后续覆盖逻辑。

### 2.4 训练链路

`BridgeDPTrainer._prepare_curve_supervision()` 是当前训练链路的核心之一（`internnav/trainer/bridgedp_trainer.py:390`）。它读取 batch 中的 raw 轨迹，计算终点距离 `traj_distance_m`（`internnav/trainer/bridgedp_trainer.py:417`），在启用轨迹归一化时按固定目标距离缩放轨迹形状，并用 GPU 样条/弧长方式重采样为固定 `predict_size` 的监督序列（`internnav/trainer/bridgedp_trainer.py:419-449`）。

这意味着当前 Bridge-DP 的监督目标不是 Dataset 里简单裁剪出的固定索引点，而是 Trainer 中重新生成的“弧长均匀控制点”。优点是轨迹几何更平滑、监督尺度更统一；缺点是训练目标更依赖重采样实现，且短距离/静止/起步障碍场景会被 `trajectory_norm_min_distance_m` 和 `sample_valid` 逻辑影响。

Trainer 会在 GPU 上生成训练先验 `batch_prior`（`internnav/trainer/bridgedp_trainer.py:473`），生成函数为 `_generate_prior_trajectory_gpu()`（`internnav/trainer/bridgedp_trainer.py:280`）。但由于模型默认 `use_prior_traj=False`，这些 prior 在默认模型前向中会被零 token 替代。也就是说，Trainer 确实准备了先验，但默认模型不消费它。

损失仍是 NavDP 风格的三项组合：`loss = 0.8 * action_loss + 0.2 * critic_loss + 0.5 * aux_loss`（`internnav/trainer/bridgedp_trainer.py:617`）。`action_loss` 的主体是 NoGoal 与 PointGoal 的 x0 MSE，critic loss 监督 learned critic 预测几何风险分数（`internnav/trainer/bridgedp_trainer.py:602`）。

### 2.5 模型链路

`BridgeDPNet` 的模型主体在 `bridgedp_policy.py` 中定义（`internnav/model/basemodel/bridgedp/bridgedp_policy.py:67`）。它复用 RGB-D、PointGoal、ImageGoal、PixelGoal 编码器，并新增 `PriorEncoder`、`VisualGate` 和 `BridgeScheduler`（`internnav/model/basemodel/bridgedp/bridgedp_policy.py:254-260`）。

训练前向使用 `sample_bridge_noise()` 进行布朗桥前向加噪（`internnav/model/basemodel/bridgedp/bridgedp_policy.py:464`），预测目标是干净轨迹 `x0` 而不是噪声。模型同时构建 NoGoal 和 PointGoal 两条生成分支（`internnav/model/basemodel/bridgedp/bridgedp_policy.py:623`）。

关键风险在 prior 分支：当 `use_prior_traj=False` 时，模型明确用全零 prior token 填充，完全忽略输入先验（`internnav/model/basemodel/bridgedp/bridgedp_policy.py:717-726`）。推理接口同样遵循这个逻辑（`internnav/model/basemodel/bridgedp/bridgedp_policy.py:979-986`, `internnav/model/basemodel/bridgedp/bridgedp_policy.py:1102-1109`）。因此任何关于“上一帧轨迹先验提升连续性”的结论，都不能基于当前默认配置直接成立。

### 2.6 布朗桥生成逻辑

`BridgeScheduler` 提供 ordered point-goal bridge。PointGoal 均值由 `bridge_mean_ordered()` 从当前原点插值到目标（`internnav/model/basemodel/bridgedp/bridge_scheduler.py:259`），NoGoal 均值由 `bridge_mean_nogoal()` 构造默认前向目标（`internnav/model/basemodel/bridgedp/bridge_scheduler.py:274`）。

各向异性噪声在 `pointgoal_noise_params()` 中实现（`internnav/model/basemodel/bridgedp/bridge_scheduler.py:287`），其中横向 `sigma_normal` 和切向 `sigma_tangent` 分开建模（`internnav/model/basemodel/bridgedp/bridge_scheduler.py:336-337`）。这使得代码能够沿目标方向和法向方向生成不同扰动，对绕障候选多样性有利。

但布朗桥均值本身不感知障碍物。对于“目标在前方但中间有障碍”的场景，桥均值仍可能穿障碍；安全性依赖后续模型去噪学到偏离均值，以及推理阶段 learned critic 对候选轨迹排序。

推理时，PointGoal 使用 `sample_initial_noise_ordered()` 从有序桥分布初始化候选轨迹（`internnav/model/basemodel/bridgedp/bridgedp_policy.py:994`，调度器实现见 `internnav/model/basemodel/bridgedp/bridge_scheduler.py:737`），然后用 `step_trajectory()` 逐步去噪（`internnav/model/basemodel/bridgedp/bridge_scheduler.py:578`）。候选排序由 `predict_critic()` 和可选 goal consistency 完成；默认 goal consistency 关闭，最终通过 `score_values` 取前 8 条正/负轨迹（`internnav/model/basemodel/bridgedp/bridgedp_policy.py:1025-1038`）。

### 2.7 Critic 安全标签

`bridgedp_critic.py` 的核心几何标签是二维点到障碍点的最小 L2 距离（`internnav/dataset/bridgedp_critic.py:8`）。风险函数使用 hard/soft threshold 将 clearance 映射为风险（`internnav/dataset/bridgedp_critic.py:21`），最终分数由最大风险、平均风险和距离趋势组成（`internnav/dataset/bridgedp_critic.py:54`）。

这比只看平均 clearance 更合理，因为单点硬碰撞会被 `max_risk` 惩罚。但它仍然不是完整碰撞检测：没有机器人 footprint、多边形/圆形膨胀、朝向相关碰撞体、速度相关安全距离，也没有区分障碍高度。对狭小走道和大碰撞体积机器人，轨迹中心点安全不等于机器人实体安全。

## 3. 方法级能力与缺陷

### 3.1 数据侧

优势：
- 使用真实轨迹位姿 `action` 作为监督来源，语义清晰。
- RGB、Depth、pointcloud 同时进入数据链路，具备视觉避障和几何 critic 的基础。
- 障碍密度采样可以增加困难片段出现概率。

缺陷：
- 障碍监督依赖点云颜色标注，标注缺失或类别错误会直接影响 critic 标签。
- Critic 标签只使用二维 xy 最小距离，不处理高度、坡度、台阶、悬空障碍和低矮小物体。
- 没有从数据层显式注入机器人半径、形状、载荷宽度或传感器安装差异。

### 3.2 训练侧

优势：
- GPU 重采样将变长 raw 轨迹变成固定长度控制点，适合 Transformer 输出。
- 轨迹尺度归一化缓解不同目标距离的监督尺度差异。
- NoGoal/PointGoal 双分支保留了目标条件和无目标探索两种模式。

缺陷：
- 默认 `lambda_delta=0`，轨迹增量平滑正则没有真正启用。
- prior 在 Trainer 中生成，但默认模型不使用；这会让“连续帧先验”停留在潜在能力。
- 起步障碍、原地转向、急停等行为没有专门 loss 或状态标签。

### 3.3 生成侧

优势：
- 布朗桥有序初始化比纯高斯随机轨迹更符合导航任务结构。
- 各向异性噪声允许横向绕行探索，默认 `bridge_normal_sigma_ratio=2.0` 明确增强法向扰动。
- x0 prediction 避免小方差布朗桥下 eps 反推误差放大。

缺陷：
- 桥均值不看障碍，因此大障碍物、死胡同、狭小通道仍依赖模型和 critic 学出来。
- 候选数和噪声方差决定能否覆盖安全绕行模态，没有硬保证。
- NoGoal 目前是默认前向均值加远端方差，更像探索启发式，而不是完整的局部安全策略。

### 3.4 评估侧

优势：
- Learned critic 在推理时对多候选轨迹排序，保留了反事实评估思路。
- Critic 默认屏蔽先验，有助于避免错误 prior 污染安全评分。

缺陷：
- Critic 是学习式软判断，不是可证明安全约束。
- 安全标签本身没有 footprint-aware 膨胀，导致 learned critic 继承标签盲区。
- 对动态人、细长腿、小台阶等场景，没有显式时空占据或局部几何细化。

## 4. 思路级分析

Bridge-DP 的核心思路是把 NavDP 的“从随机噪声生成轨迹”改造成“在起点-目标桥分布附近生成轨迹”，同时保留 RGB-D 条件和 critic 排序。这一思路的优点是清晰的：生成空间更接近导航任务结构，目标边界更明确，轨迹输出为绝对 xyt 后也减少了增量累积误差。

但是，当前默认实现还没有把“上一轮轨迹先验”变成主干能力。`PriorEncoder` 与 `VisualGate` 已经在代码里，但默认关闭，且没有看到完整的部署端 shift-and-align 逻辑：例如上一帧轨迹如何扣除已执行段、如何根据真实里程计对齐、如何补齐末端、跟踪误差过大时如何降权或丢弃先验。因此从思路级看，Bridge-DP 的论文级核心卖点需要继续补全：不是只实现 prior token，而是要形成从部署端轨迹缓存到训练分布一致性的闭环链路。

另一个关键思路缺口是安全性没有几何内核。当前系统把安全主要交给 learned critic，但机器人导航中的“碰不碰”高度依赖机器人几何、速度、执行误差和障碍物高度。没有 footprint-aware critic 或尺寸条件输入时，模型很难自然泛化到大碰撞体积机器人或不同机器人平台。

## 5. 典型场景能力判断

### 室内常规场景

当前代码较适配。RGB-D 编码、点云障碍标签、critic 排序和轨迹数据分布都围绕室内导航设计。若训练数据覆盖充分，常规墙体、家具、大障碍物的避障能力有合理基础。风险主要来自局部几何细节和 learned critic 误判。

### 室外场景

代码层面不能直接证明室外泛化。室外问题涉及光照、尺度、地面材质、远距离深度、坡度、植被、动态行人等分布变化。当前配置和数据路径更像 InternData/N1 室内/仿真导航管线，是否适合室外必须靠数据审计和实验验证。

### 起步存在障碍物

历史 RGB-D 和当前深度可以让模型感知起点附近障碍，critic 也可以惩罚穿障候选。但代码没有显式“起步安全检查”“起步先后退/先转向”“距离过近急停”等策略。若障碍物贴近机器人，轨迹中心点和深度特征可能不足以保证身体不碰撞。

### 起步需要先转动避障

Bridge-DP 输出 xyt，theta 进入监督和预测，所以理论上可以学到转向轨迹。但没有看到单独的原地旋转动作空间、角速度约束或“线速度为零时先旋转”的控制器逻辑。因此它可能生成带大 yaw 变化的轨迹控制点，但实际机器人是否能先转身避障，取决于下游控制器。

### 面前存在大障碍物

大障碍物通常更容易被深度和点云捕捉，critic 标签也更稳定。缺点是 PointGoal 桥均值可能直接穿过障碍；如果候选采样未覆盖绕行模态，或 critic 排序失败，仍可能选出不安全轨迹。

### 狭小走道

这是当前代码风险较高的场景。二维中心线 clearance 不等于机器人 footprint clearance。缺少机器人半径膨胀后，模型可能认为走道中心线安全，但大体积机器人实际会擦碰两侧。

### 机器人碰撞体积大

当前代码没有机器人尺寸条件输入，也没有按半径扩张障碍的 critic 标签。大体积机器人只靠同一套 learned critic 和同一套阈值评估，属于明确短板。

### 应用机器人碰撞体积不同

当前默认系统不支持真正的尺寸条件泛化。不同机器人应至少在训练和推理中提供 `robot_radius` 或 footprint token，并在 critic 标签中按尺寸重算 clearance。否则只能通过为每种机器人重新训练/微调来弥补。

### 细长障碍物：桌椅腿、人

若点云中细长障碍点足够密集且被标注为障碍，二维 L2 critic 可以惩罚靠近。但细长物体点稀疏、深度噪声大、遮挡严重，人又可能动态移动，所以当前静态二维点距离方案并不稳。对人尤其缺少时序预测和动态占据建模。

### 小障碍物：小台阶、小玩具

当前风险很高。小台阶和小玩具是否进入障碍点云、是否被颜色标注、是否在深度预处理中保留，都会显著影响结果。二维平面距离还无法表达“可跨越台阶”和“不可跨越障碍”的区别。

## 6. 改进优先级

### P0：补齐安全标签的几何真实性

- 在 critic 标签中加入机器人半径/footprint 膨胀，至少支持圆形半径。
- 将当前 `min_l2_distances_xy` 扩展为 inflated clearance，并记录不同半径下的 collision label。
- 对狭小走道、大体积机器人、贴近障碍起步优先验证。

### P1：让机器人尺寸成为条件

- 在配置和模型条件中加入 `robot_radius` 或 footprint token。
- 训练时随机化机器人半径，让同一视觉场景下的可通行性随尺寸变化。
- 推理时按目标机器人尺寸选择或条件化生成轨迹。

### P2：把先验注入从潜在能力变成主干能力

- 启用 `use_prior_traj=True` 的受控实验。
- 增加部署端 prior shift-and-align：丢弃已执行段、用当前里程计对齐剩余轨迹、补齐末端。
- 对错误先验、突发障碍、跟踪误差大等情况加入 gate 诊断。

### P3：补细长/低矮/动态障碍建模

- 对障碍点加入高度统计或局部 voxel 表征。
- 对桌椅腿、人腿、小玩具、小台阶建立专项数据审计。
- 对动态人引入时序占据或与 FlowNav 动态层结合，而不是只靠静态点云。

### P4：补部署控制约束

- 将轨迹控制点转成速度/角速度时加入动力学限制。
- 起步近障时加入停止、原地旋转、慢速探索策略。
- 对输出轨迹做执行前几何碰撞复核，避免 learned critic 单点失败。

## 7. 后续实验建议

这些实验用于验证上述代码分析，不作为当前报告的主体结论。

1. 数据审计：统计障碍点密度、最近障碍距离、通道宽度、低矮/细长障碍比例、室内/室外来源比例。
2. 离线几何评估：对 GT、augment、模型预测候选按不同机器人半径计算 inflated collision rate。
3. Critic 相关性检查：比较 learned critic 分数与真实 inflated collision、最小 clearance、路径贴边率的相关性。
4. 场景专项集：构建起步障碍、先转向避障、大障碍、狭窄走道、大体积机器人、桌椅腿/小玩具/小台阶测试集。
5. 消融：比较 `use_prior_traj` 开/关、候选数、critic 阈值、NoGoal 方差、`bridge_normal_sigma_ratio` 和 footprint-aware critic。

## 8. 结语

从纯代码角度看，Bridge-DP 当前已经具备一个清晰、可训练、可推理的生成式导航策略骨架。它相对 NavDP 的主要实际变化是：轨迹表示改为绝对 xyt，训练监督改为 GPU 弧长重采样后的 x0 预测，生成过程改为有目标结构的布朗桥，并继续使用 learned critic 做候选排序。

但当前默认实现的能力边界也很明确：先验注入没有默认启用，安全评估没有进入机器人几何层，部署控制闭环尚未补齐。因此，对复杂场景的客观判断应保持克制：它有较好的方法基础，但在起步贴近障碍、狭窄走道、大体积机器人、跨机器人尺寸、细长/低矮障碍物等场景下，还不能仅凭当前代码宣称可靠。
