"""动态层配置定义。

该配置覆盖 IRA 插件驱动所需的全部参数，是动态层的唯一运行时输入源。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# IRA 支持的 character action 类型
SUPPORTED_ACTIONS = {"Idle", "GoTo", "LookAround"}

# 支持的运动模式
SUPPORTED_MOTION_MODES = {"stand_idle", "walk", "mixed"}

# 支持的 IRA writer 类型
SUPPORTED_WRITERS = {"IRABasicWriter"}

# 支持的 character filter 关键字（Isaac Sim 5.1 People 资产分类）
SUPPORTED_CHARACTER_FILTERS = {
    "male", "female",
    "business", "casual", "medical", "construction", "military",
}

# 支持的 robot 类型
SUPPORTED_ROBOT_TYPES = {"nova_carter", "iw_hub"}


@dataclass
class DynamicLayerConfig:
    """动态层配置契约。

    该对象控制 IRA 插件的全部行为参数，包括：
    1. 人物角色配置（数量、筛选条件、资产路径）
    2. 行为命令生成策略
    3. 机器人配置（预留）
    4. 传感器配置
    5. Replicator 输出配置
    6. 仿真控制参数

    所有路径字段使用绝对路径或相对于工作目录的路径。
    """

    # ─── 全局控制 ───
    enable_dynamics: bool = True
    backend: str = "ira"
    seed: int = 123456
    simulation_length: int = 300

    # ─── 场景引用 ───
    # scene_usd 在运行时由场景层传入，不需要在配置中预设
    scene_usd: str = ""

    # ─── 角色资产 ───
    character_asset_path: str = ""
    character_filters: List[str] = field(default_factory=lambda: ["male", "medical"])
    character_num_min: int = 5
    character_num_max: int = 15
    character_spawn_area: List[str] = field(default_factory=lambda: ["Walkable"])
    character_navigation_area: List[str] = field(default_factory=lambda: ["Walkable"])

    # ─── 行为命令生成策略 ───
    # idle_ratio: 静止 NPC 占比
    idle_ratio: float = 0.3
    # walk_ratio: 行走 NPC 占比 (1 - idle_ratio - look_around_ratio)
    look_around_ratio: float = 0.1
    # 每个 NPC 的最大命令条数
    max_commands_per_character: int = 5
    # GoTo 目标坐标范围
    goto_x_range: List[float] = field(default_factory=lambda: [-30.0, 10.0])
    goto_y_range: List[float] = field(default_factory=lambda: [-25.0, 30.0])
    goto_z: float = 0.0
    # Idle 持续时间范围（秒）
    idle_duration_range: List[float] = field(default_factory=lambda: [1.5, 5.0])
    # LookAround 持续时间范围（秒）
    look_around_duration_range: List[float] = field(default_factory=lambda: [1.5, 4.0])

    # ─── Robot 配置（预留） ───
    robot_nova_carter_num: int = 0
    robot_iw_hub_num: int = 0
    robot_spawn_area: List[str] = field(default_factory=list)
    robot_navigation_area: List[str] = field(default_factory=list)
    robot_write_data: bool = False

    # ─── 传感器 ───
    camera_num: int = 5

    # ─── Replicator 输出 ───
    writer: str = "IRABasicWriter"
    output_dir: str = ""
    image_output_format: str = "png"
    rgb: bool = True
    camera_params: bool = True
    # Bounding box 标注
    bbox_2d_tight: bool = True
    bbox_2d_loose: bool = True
    bbox_3d: bool = True
    skeleton_data: bool = False
    semantic_filter_predicate: str = "class:character|robot;id:*"
    # 分割标注
    semantic_segmentation: bool = False
    instance_id_segmentation: bool = False
    instance_segmentation: bool = False
    # 深度
    distance_to_camera: bool = False
    distance_to_image_plane: bool = False
    colorize_depth: bool = False
    # 其他标注
    occlusion: bool = False
    normals: bool = False
    motion_vectors: bool = False
    pointcloud: bool = False

    # ─── IRA 驱动配置 ───
    # Isaac Sim 可执行文件路径（为空时自动探测）
    isaac_sim_python: str = ""
    # IRA 扩展模块名
    ira_module: str = "isaacsim.replicator.agent"
    # headless 模式
    headless: bool = True
    # 超时秒数
    timeout_s: int = 600
    # IRA 配置文件输出目录（为空时使用 scene_dir 下的 ira_config/）
    ira_config_dir: str = ""

    # ─── 预定义场景行为模板 ───
    # 指定预定义行为方案的路径，为空时使用默认生成策略
    scenario_template: str = ""

    # ─── 兼容旧流程 ───
    # 当 IRA 不可用时是否 fallback 到旧的 synthetic 后端
    fallback_to_synthetic: bool = True

    def validate(self) -> None:
        """执行配置合法性校验。"""
        if not self.enable_dynamics:
            return  # 禁用时不校验

        if self.backend not in {"ira", "synthetic"}:
            raise ValueError(f"backend 必须是 'ira' 或 'synthetic', got={self.backend}")

        if self.simulation_length <= 0:
            raise ValueError(f"simulation_length 必须 > 0, got={self.simulation_length}")

        if self.seed < 0:
            raise ValueError(f"seed 不能为负数, got={self.seed}")

        # 角色数量
        if self.character_num_min < 0:
            raise ValueError(f"character_num_min 不能为负, got={self.character_num_min}")
        if self.character_num_max < self.character_num_min:
            raise ValueError(
                f"character_num_max({self.character_num_max}) 必须 >= character_num_min({self.character_num_min})"
            )

        # 行为比例
        total_ratio = self.idle_ratio + self.look_around_ratio
        if total_ratio > 1.0:
            raise ValueError(
                f"idle_ratio({self.idle_ratio}) + look_around_ratio({self.look_around_ratio}) = {total_ratio} > 1.0"
            )
        if self.idle_ratio < 0 or self.look_around_ratio < 0:
            raise ValueError("idle_ratio 和 look_around_ratio 不能为负")

        # 命令参数
        if self.max_commands_per_character <= 0:
            raise ValueError(f"max_commands_per_character 必须 > 0, got={self.max_commands_per_character}")

        if len(self.goto_x_range) != 2 or self.goto_x_range[0] >= self.goto_x_range[1]:
            raise ValueError(f"goto_x_range 必须是 [min, max] 且 min < max, got={self.goto_x_range}")
        if len(self.goto_y_range) != 2 or self.goto_y_range[0] >= self.goto_y_range[1]:
            raise ValueError(f"goto_y_range 必须是 [min, max] 且 min < max, got={self.goto_y_range}")

        if len(self.idle_duration_range) != 2 or self.idle_duration_range[0] >= self.idle_duration_range[1]:
            raise ValueError(f"idle_duration_range 格式错误: {self.idle_duration_range}")
        if len(self.look_around_duration_range) != 2 or self.look_around_duration_range[0] >= self.look_around_duration_range[1]:
            raise ValueError(f"look_around_duration_range 格式错误: {self.look_around_duration_range}")

        # 传感器
        if self.camera_num < 0:
            raise ValueError(f"camera_num 不能为负, got={self.camera_num}")

        # Writer
        if self.writer not in SUPPORTED_WRITERS:
            raise ValueError(f"writer 必须是 {SUPPORTED_WRITERS} 之一, got={self.writer}")

        # Robot
        if self.robot_nova_carter_num < 0 or self.robot_iw_hub_num < 0:
            raise ValueError("robot 数量不能为负")

        # 超时
        if self.timeout_s <= 0:
            raise ValueError(f"timeout_s 必须 > 0, got={self.timeout_s}")

    def pick_character_num(self, seed: int) -> int:
        """根据 seed 确定性选择角色数量。"""
        span = max(0, self.character_num_max - self.character_num_min)
        if span == 0:
            return self.character_num_min
        return self.character_num_min + (seed % (span + 1))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DynamicLayerConfig":
        """从字典构建并校验配置。"""
        cfg = cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
        cfg.validate()
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        """导出配置字典。"""
        return self.__dict__.copy()
