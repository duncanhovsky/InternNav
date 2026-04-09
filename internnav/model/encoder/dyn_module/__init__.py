"""FlowNav 外置动态模块。

该包用于在 policy 外部构建时空动态先验，核心职责如下：
1. 接收 LiDAR 点云与里程计位姿输入。
2. 调用场景流估计与后处理链路，生成动态体素。
3. 输出与 FlowNav policy 无缝对接的 ``dynamic_voxels`` 张量。

设计目标：
- 不修改现有 policy 方法签名。
- 支持 ROS1/ROS2 消息输入。
- 支持重计算低频、策略消费高频的缓存复用模式。
"""

from .config import DynModuleConfig
from .types import DynamicVoxelPacket, PointCloudFrame, PoseFrame, TrackState
from .policy_bridge import FlowNavDynamicsRuntime
from .ros2_node import FlowNavDynROS2Node

__all__ = [
    "DynModuleConfig",
    "DynamicVoxelPacket",
    "PointCloudFrame",
    "PoseFrame",
    "TrackState",
    "FlowNavDynamicsRuntime",
    "FlowNavDynROS2Node",
]
