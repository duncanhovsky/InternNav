# FlowNav 外置动态模块

该模块提供如下外置运行时链路：

历史点云 + 当前点云 + odom
-> OpenSceneFlow 推理（LiteFlow/DeltaFlow）
-> HDBSCAN 聚类
-> 关联 + Kalman 跟踪
-> dynamic_voxels（T=8, C=4）

输出张量与 policy 对齐：`(B, T, C, X, Y, Z)`，通道定义如下：

1. `Occ`：占据概率/占据标记
2. `Vx`：x 方向速度
3. `Vy`：y 方向速度
4. `Vz`：z 方向速度

## 快速使用（同进程）

```python
from flownav.dyn_module import DynModuleConfig, FlowNavDynamicsRuntime

cfg = DynModuleConfig(
  model_name="liteflow",          # 或 "deltaflow"
  checkpoint_path="",             # 可选
    history_frames=8,
    horizon_frames=8,
    heavy_rate_hz=7.5,
    policy_rate_hz=15.0,
)
runtime = FlowNavDynamicsRuntime(cfg)

# 每次拿到同步后的点云+位姿后调用：
runtime.ingest(point_frame, pose_frame)

# 在调用 FlowNav policy 前获取 dynamic_voxels：
dynamic_voxels = runtime.get_dynamic_voxels(
    batch_size=1,
    device=None,
)
```

## ROS 消息输入

如果你已经拿到 ROS1/ROS2 的消息对象：

```python
runtime.ingest_ros_msgs(cloud_msg, odom_msg)
```

## 说明

- 该模块不修改现有 FlowNav policy 方法签名。
- 若 OpenSceneFlow 依赖缺失，适配器会自动进入安全回退模式并输出零流，保证主流程不中断。
- 聚类优先使用 `hdbscan`，不可用时回退到 `sklearn.DBSCAN`。
