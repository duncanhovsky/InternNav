from __future__ import annotations
"""ROS2 节点封装。

该文件提供可选的独立运行入口，用于在 ROS2 环境中订阅点云与里程计，
并驱动 ``FlowNavDynamicsRuntime`` 持续更新动态体素缓存。
"""

from typing import Optional

from .config import DynModuleConfig
from .policy_bridge import FlowNavDynamicsRuntime


class FlowNavDynROS2Node:
    """FlowNav 动态模块 ROS2 节点包装器。"""

    def __init__(self, cfg: Optional[DynModuleConfig] = None):
        """初始化运行时对象。"""
        self.runtime = FlowNavDynamicsRuntime(cfg)
        self._node = None

    def spin(self, cloud_topic: str = "/points", odom_topic: str = "/odom") -> None:
        """启动 ROS2 订阅循环。

        Args:
            cloud_topic: 点云话题名。
            odom_topic: 里程计话题名。
        """
        try:
            import rclpy
            from rclpy.node import Node
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import PointCloud2
        except Exception as exc:
            raise RuntimeError("ROS2 (rclpy/nav_msgs/sensor_msgs) is required for spin().") from exc

        rclpy.init()

        class _Node(Node):
            """内部 ROS2 节点实现。

            策略：
            - 里程计高频缓存最新值。
            - 点云到达时与最新里程计配对并送入 runtime。
            """

            def __init__(self, outer: "FlowNavDynROS2Node"):
                super().__init__("flownav_dyn_module")
                self.outer = outer
                self._last_cloud = None
                self._last_odom = None
                self.create_subscription(PointCloud2, cloud_topic, self.on_cloud, 10)
                self.create_subscription(Odometry, odom_topic, self.on_odom, 50)

            def on_cloud(self, msg: PointCloud2) -> None:
                """点云回调：到达后尝试与最新里程计配对。"""
                self._last_cloud = msg
                if self._last_odom is not None:
                    self.outer.runtime.ingest_ros_msgs(self._last_cloud, self._last_odom)

            def on_odom(self, msg: Odometry) -> None:
                """里程计回调：仅更新最新缓存。"""
                self._last_odom = msg

        self._node = _Node(self)
        rclpy.spin(self._node)
        self._node.destroy_node()
        rclpy.shutdown()
