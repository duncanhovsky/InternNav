# FlowNav 代码全览与入口使用说明

## 1. 文档范围
本清单覆盖当前仓库中与 FlowNav 直接相关的代码与配置，包括：
- FlowNav 模型本体（policy/backbone）
- FlowNav 数据集与 collate 逻辑（静态/动态/混合训练相关）
- FlowNav Trainer 与训练入口
- FlowNav 动态模块（dyn_module）
- FlowNav 配置注册与入口映射文件

说明：以下按“核心程度”分组，尽量做到可直接定位与上手。

---

## 2. FlowNav 相关代码文件总表

### 2.1 训练入口与实验脚本
1. `scripts/train/base_train/train.py`
- 统一训练主入口。
- 通过 `--model-name` 选择 `flownav_static` / `flownav_dyn` / `flownav_mix`。
- 负责：配置映射、模型构建、数据集选择、Trainer 选择、HF Trainer 启动。

2. `scripts/train/base_train/start_train.sh`
- Bash 启动脚本。
- 对 FlowNav 系列使用 `torchrun` 启动。
- 可通过 `--model` 传入 `flownav_static` / `flownav_dyn` / `flownav_mix`。

3. `scripts/train/base_train/flownav_intermediate_visualizer.py`
- FlowNav 中间结果可视化预检入口。
- 支持 `flownav_static` 与 `flownav_dyn` 两种模式。
- 会输出位姿轨迹、点云 BEV、动态体素可视化 GIF 等。

---

### 2.2 FlowNav 训练配置
4. `scripts/train/base_train/configs/flownav.py`
- 三套实验配置：
  - `flownav_static_exp_cfg`
  - `flownav_dyn_exp_cfg`
  - `flownav_mix_exp_cfg`
- 包含数据路径、训练超参、动态体素参数、混合采样比例等。

5. `scripts/train/base_train/configs/__init__.py`
- 导出 FlowNav 三套配置，供 `train.py` 统一注册。

6. `internnav/configs/model/flownav.py`
- 模型级配置入口：`policy_name='FlowNav_Policy'`。

7. `internnav/configs/model/__init__.py`
- 将 `flownav_cfg` 纳入模型配置总导出。

8. `internnav/configs/trainer/il.py`
- IL 通用配置定义。
- 与 FlowNav 直接关联字段如 `fallback_fps` 在此声明。

---

### 2.3 FlowNav 数据集与数据拼接
9. `internnav/dataset/flownav_lerobot_dataset.py`
- 静态场景 FlowNav 数据集实现：`FlowNav_Base_Datset`。
- 包含 `flownav_collate_fn`。
- 为静态样本提供 dyn_module 在线回退所需辅助字段。

10. `internnav/dataset/flownav_dyn_lerobot_dataset.py`
- 动态场景数据集实现：`FlowNav_Dyn_Lerobot_Dataset`。
- 包含 `flownav_dyn_collate_fn`。
- 支持动态轨迹对齐、动态体素输入、critic 相关动态信息。

11. `internnav/dataset/dyn_lerobot_dataset.md`
- 动态数据集设计说明文档。
- 描述字段规范、目录组织、时间对齐与接入建议。

---

### 2.4 FlowNav 训练器
12. `internnav/trainer/flownav_trainer.py`
- FlowNav 专用 Trainer：`FlowNavTrainer`。
- 负责：
  - 训练损失计算（动作/critic/辅助）
  - dynamic_voxels 解析优先级（已有体素 or 在线构建）
  - 混合训练的 mask 回填
  - 保存 `flownav.ckpt`

13. `internnav/trainer/__init__.py`
- 将 `FlowNavTrainer` 注册到 trainer 导出入口。

---

### 2.5 FlowNav 模型与编码器
14. `internnav/model/basemodel/flownav/flownav_policy.py`
- FlowNav 主模型：`FlowNavNet`。
- 配置类：`FlowNavModelConfig`。
- 支持从目录加载 `flownav.ckpt` 或 `pytorch_model.bin`。

15. `internnav/model/encoder/flownav_backbone.py`
- 视觉与动态融合骨干。
- 关键类：`FlowNavFusionBackbone`。
- 包含 RGB/Depth 特征提取、动态体素编码、跨模态融合。

16. `internnav/model/__init__.py`
- 将 `FlowNav_Policy` 映射到 `FlowNavNet` / `FlowNavModelConfig`。

---

### 2.6 FlowNav dyn_module（外置动态模块）
17. `internnav/model/encoder/dyn_module/__init__.py`
- dyn_module 包导出入口。

18. `internnav/model/encoder/dyn_module/config.py`
- `DynModuleConfig`：动态模块总配置。

19. `internnav/model/encoder/dyn_module/types.py`
- 运行时数据结构：点云帧、位姿帧、轨迹状态、体素包等。

20. `internnav/model/encoder/dyn_module/opensceneflow_adapter.py`
- OpenSceneFlow 适配层，负责场景流推理及回退策略。

21. `internnav/model/encoder/dyn_module/motion_postprocess.py`
- 动态点筛选、聚类、关联、Kalman 跟踪。

22. `internnav/model/encoder/dyn_module/dynamic_voxel_builder.py`
- 轨迹到体素的构建与缓存，输出 `(B,T,C,X,Y,Z)`。

23. `internnav/model/encoder/dyn_module/policy_bridge.py`
- 对外统一运行时：`FlowNavDynamicsRuntime`。
- 连接输入、重计算链路、缓存和 policy 张量输出。

24. `internnav/model/encoder/dyn_module/ros2_io.py`
- ROS 消息转换与近似时间同步。

25. `internnav/model/encoder/dyn_module/ros2_node.py`
- ROS2 节点包装：`FlowNavDynROS2Node`。

26. `internnav/model/encoder/dyn_module/README.md`
- dyn_module 快速使用说明与示例。

---

## 3. FlowNav 各入口使用说明

## 3.1 入口 A：统一训练入口（推荐）
文件：`scripts/train/base_train/train.py`

### 命令模板
```bash
python scripts/train/base_train/train.py --name <实验名> --model-name <模型分支>
```

### 可用分支
- `flownav_static`：FlowNav + 静态数据集
- `flownav_dyn`：FlowNav + 动态数据集
- `flownav_mix`：FlowNav + 静态+动态混合采样

### 示例
```bash
python scripts/train/base_train/train.py --name flownav_static_run1 --model-name flownav_static
python scripts/train/base_train/train.py --name flownav_dyn_run1 --model-name flownav_dyn
python scripts/train/base_train/train.py --name flownav_mix_run1 --model-name flownav_mix
```

### 运行要点
- 实际数据路径、batch、动态参数由 `scripts/train/base_train/configs/flownav.py` 控制。
- 若使用分布式，需保证环境变量与 GPU 可见性配置正确。

---

## 3.2 入口 B：脚本化启动入口（Linux/Bash）
文件：`scripts/train/base_train/start_train.sh`

### 命令模板
```bash
bash scripts/train/base_train/start_train.sh --name <实验名> --model <模型分支>
```

### 示例
```bash
bash scripts/train/base_train/start_train.sh --name flownav_static_run1 --model flownav_static
bash scripts/train/base_train/start_train.sh --name flownav_dyn_run1 --model flownav_dyn
bash scripts/train/base_train/start_train.sh --name flownav_mix_run1 --model flownav_mix
```

### 说明
- 脚本内部会根据 `--model` 自动设置 `CUDA_VISIBLE_DEVICES` 和 `NUM_GPUS`。
- 对 FlowNav 分支默认走 `torchrun`。
- 在 Windows PowerShell 下通常不直接执行 `.sh`，建议直接使用入口 A。

---

## 3.3 入口 C：FlowNav 中间结果可视化预检
文件：`scripts/train/base_train/flownav_intermediate_visualizer.py`

### 命令模板
```bash
python scripts/train/base_train/flownav_intermediate_visualizer.py \
  --model_name <flownav_static|flownav_dyn> \
  --num_samples 4 \
  --batch_size 2 \
  --output_dir checkpoints/flownav_intermediate_vis \
  --device cuda:0
```

### 示例
```bash
python scripts/train/base_train/flownav_intermediate_visualizer.py --model_name flownav_static --num_samples 4 --batch_size 2
python scripts/train/base_train/flownav_intermediate_visualizer.py --model_name flownav_dyn --num_samples 4 --batch_size 2
```

### 输出内容
- `ego_pose_history.png`
- `history_pointcloud_bev.png`
- `future_4d_occ_t0.png`
- `future_4d_motion.gif`
- `intermediate_tensors.npz`
- `summary.txt`

---

## 3.4 入口 D：数据集离线自测入口
文件：`internnav/dataset/flownav_lerobot_dataset.py`

### 触发方式
该文件含 `if __name__ == "__main__":` 测试逻辑，可直接运行进行样本可视化导出。

```bash
python internnav/dataset/flownav_lerobot_dataset.py
```

### 说明
- 该入口更多用于 dataset 读数与可视化自检，不是训练入口。
- 默认路径示例可能需按本地数据实际位置修改。

---

## 3.5 入口 E：代码级模型加载入口
关键文件：
- `internnav/model/__init__.py`
- `internnav/model/basemodel/flownav/flownav_policy.py`

### 方式 1（框架统一注册方式）
```python
from internnav.model import get_policy, get_config

ModelClass = get_policy("FlowNav_Policy")
ConfigClass = get_config("FlowNav_Policy")
```

### 方式 2（直接加载 checkpoint）
```python
from internnav.model.basemodel.flownav.flownav_policy import FlowNavNet

model = FlowNavNet.from_pretrained("checkpoints/your_run/ckpts")
# 目录下优先找 flownav.ckpt，其次 pytorch_model.bin
```

---

## 3.6 入口 F：dyn_module 运行时入口（代码调用）
关键文件：
- `internnav/model/encoder/dyn_module/policy_bridge.py`
- `internnav/model/encoder/dyn_module/config.py`

### 最小示例
```python
from internnav.model.encoder.dyn_module import DynModuleConfig, FlowNavDynamicsRuntime

cfg = DynModuleConfig()
runtime = FlowNavDynamicsRuntime(cfg)

# 每次传入一帧点云与位姿
runtime.ingest(point_frame, pose_frame)

# 在调用 policy 前取动态体素
dynamic_voxels = runtime.get_dynamic_voxels(batch_size=1)
```

---

## 3.7 入口 G：dyn_module ROS2 节点入口
关键文件：`internnav/model/encoder/dyn_module/ros2_node.py`

### 调用方式
```python
from internnav.model.encoder.dyn_module import DynModuleConfig, FlowNavDynROS2Node

node = FlowNavDynROS2Node(DynModuleConfig())
node.spin(cloud_topic="/points", odom_topic="/odom")
```

### 说明
- 依赖 ROS2：`rclpy`, `sensor_msgs`, `nav_msgs`。
- 适合把动态体素构建独立为 ROS 侧在线模块。

---

## 4. 建议阅读顺序（快速上手）
1. `scripts/train/base_train/train.py`（先理解入口分支）
2. `scripts/train/base_train/configs/flownav.py`（再看参数）
3. `internnav/dataset/flownav_lerobot_dataset.py` 与 `internnav/dataset/flownav_dyn_lerobot_dataset.py`（理解数据契约）
4. `internnav/trainer/flownav_trainer.py`（理解训练时 dynamic_voxels 处理）
5. `internnav/model/basemodel/flownav/flownav_policy.py` + `internnav/model/encoder/flownav_backbone.py`（理解模型结构）
6. `internnav/model/encoder/dyn_module/*`（理解动态模块在线链路）

---

## 5. 常见易错点
1. `model-name` 参数必须是 `flownav_static` / `flownav_dyn` / `flownav_mix` 之一。
2. 动态训练需要动态数据字段完整（尤其时间戳与 dynamic voxel 相关缓存/配置）。
3. 混合训练依赖静态与动态索引路径同时可用（见 `dataset_flownav_static`、`dataset_flownav_dyn`）。
4. `start_train.sh` 更适合 Linux Bash；Windows 下优先直接运行 `train.py`。

---

## 6. 一句话总结
FlowNav 在本仓库是“统一训练入口 + 三套配置分支 + 静态/动态双数据管线 + Trainer 动态体素回退 + 外置 dyn_module”的完整闭环，建议从 `train.py` 与 `configs/flownav.py` 两个文件作为主入口开始。