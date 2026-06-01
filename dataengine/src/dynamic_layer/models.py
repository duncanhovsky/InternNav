"""动态层数据模型定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CharacterSpec:
    """单个 NPC 角色规格。

    Attributes:
        character_id: NPC 标识符，如 ``Character``, ``Character_01``。
        asset_relpath: 角色资产相对于 asset_root 的路径。
        motion_mode: 运动模式 (``stand_idle`` / ``walk`` / ``mixed``)。
        spawn_position: 初始出生点 ``[x, y, z]``；为空时由 IRA 自动采样。
    """

    character_id: str
    asset_relpath: str = ""
    motion_mode: str = "walk"
    spawn_position: Optional[List[float]] = None


@dataclass
class RobotSpec:
    """主视角 Robot 规格（预留）。

    Attributes:
        robot_id: 机器人标识符。
        robot_type: 机器人类型 (``nova_carter`` / ``iw_hub``)。
        spawn_position: 初始出生点。
    """

    robot_id: str = ""
    robot_type: str = "nova_carter"
    spawn_position: Optional[List[float]] = None


@dataclass
class CommandEntry:
    """单条 NPC 行为命令。

    格式对应 IRA command.txt 中的一行：
    ``<character_id> <action> <param1> [param2] [param3] [param4]``

    Attributes:
        character_id: 目标角色 ID。
        action: 行为类型 (``Idle`` / ``GoTo`` / ``LookAround``)。
        params: 行为参数列表。
    """

    character_id: str
    action: str
    params: List[str] = field(default_factory=list)

    def to_line(self) -> str:
        """序列化为 command.txt 的单行格式。"""
        parts = [self.character_id, self.action] + self.params
        return " ".join(parts)

    @classmethod
    def idle(cls, character_id: str, duration: float) -> "CommandEntry":
        """创建 Idle 命令。"""
        return cls(character_id=character_id, action="Idle", params=[f"{duration:.1f}"])

    @classmethod
    def goto(cls, character_id: str, x: float, y: float, z: float = 0.0) -> "CommandEntry":
        """创建 GoTo 命令。"""
        return cls(
            character_id=character_id,
            action="GoTo",
            params=[f"{x:.2f}", f"{y:.2f}", f"{z:.1f}", "_"],
        )

    @classmethod
    def look_around(cls, character_id: str, duration: float) -> "CommandEntry":
        """创建 LookAround 命令。"""
        return cls(character_id=character_id, action="LookAround", params=[f"{duration:.2f}"])


@dataclass
class RobotCommandEntry:
    """单条 Robot 行为命令（预留）。

    格式对应 IRA robot_command.txt 中的一行。
    """

    robot_id: str
    action: str
    params: List[str] = field(default_factory=list)

    def to_line(self) -> str:
        """序列化为 robot_command.txt 的单行格式。"""
        parts = [self.robot_id, self.action] + self.params
        return " ".join(parts)


@dataclass
class IRAConfig:
    """完整的 IRA 配置结构，对应 default_config.yaml。

    该结构与 ``isaacsim.replicator.agent`` 插件的 YAML schema 一一映射。
    """

    # --- global ---
    seed: int = 123456
    simulation_length: int = 300

    # --- scene ---
    scene_asset_path: str = ""

    # --- character ---
    character_asset_path: str = ""
    character_filters: List[str] = field(default_factory=list)
    character_num: int = 10
    character_spawn_area: List[str] = field(default_factory=lambda: ["Walkable"])
    character_navigation_area: List[str] = field(default_factory=lambda: ["Walkable"])
    character_command_file: str = ""

    # --- robot ---
    robot_command_file: str = ""
    robot_nova_carter_num: int = 0
    robot_iw_hub_num: int = 0
    robot_spawn_area: List[str] = field(default_factory=list)
    robot_navigation_area: List[str] = field(default_factory=list)
    robot_write_data: bool = False

    # --- sensor ---
    camera_num: int = 5

    # --- replicator ---
    writer: str = "IRABasicWriter"
    output_dir: str = ""
    rgb: bool = True
    camera_params: bool = True
    image_output_format: str = "png"
    object_info_bounding_box_2d_tight: bool = True
    object_info_bounding_box_2d_loose: bool = True
    object_info_bounding_box_3d: bool = True
    agent_info_skeleton_data: bool = False
    semantic_filter_predicate: str = "class:character|robot;id:*"
    semantic_segmentation: bool = False
    instance_id_segmentation: bool = False
    instance_segmentation: bool = False
    distance_to_camera: bool = False
    distance_to_image_plane: bool = False
    colorize_depth: bool = False
    colorize_semantic_segmentation: bool = True
    colorize_instance_id_segmentation: bool = True
    colorize_instance_segmentation: bool = True
    occlusion: bool = False
    normals: bool = False
    motion_vectors: bool = False
    pointcloud: bool = False
    pointcloud_include_unlabelled: bool = False
    use_common_output_dir: bool = False

    # --- response ---
    response_list: List[Any] = field(default_factory=list)

    # --- event ---
    event_list: List[Any] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为 IRA YAML 兼容的嵌套字典。"""
        return {
            "isaacsim.replicator.agent": {
                "version": "0.7.1",
                "global": {
                    "seed": self.seed,
                    "simulation_length": self.simulation_length,
                },
                "scene": {
                    "asset_path": self.scene_asset_path,
                },
                "character": {
                    "asset_path": self.character_asset_path,
                    "filters": self.character_filters,
                    "num": self.character_num,
                    "spawn_area": self.character_spawn_area,
                    "navigation_area": self.character_navigation_area,
                    "command_file": self.character_command_file,
                },
                "robot": {
                    "command_file": self.robot_command_file,
                    "nova_carter_num": self.robot_nova_carter_num,
                    "iw_hub_num": self.robot_iw_hub_num,
                    "spawn_area": self.robot_spawn_area,
                    "navigation_area": self.robot_navigation_area,
                    "write_data": self.robot_write_data,
                },
                "sensor": {
                    "camera_num": self.camera_num,
                },
                "replicator": {
                    "writer": self.writer,
                    "parameters": {
                        "output_dir": self.output_dir,
                        "rgb": self.rgb,
                        "camera_params": self.camera_params,
                        "image_output_format": self.image_output_format,
                        "object_info_bounding_box_2d_tight": self.object_info_bounding_box_2d_tight,
                        "object_info_bounding_box_2d_loose": self.object_info_bounding_box_2d_loose,
                        "object_info_bounding_box_3d": self.object_info_bounding_box_3d,
                        "agent_info_skeleton_data": self.agent_info_skeleton_data,
                        "semantic_filter_predicate": self.semantic_filter_predicate,
                        "semantic_segmentation": self.semantic_segmentation,
                        "instance_id_segmentation": self.instance_id_segmentation,
                        "instance_segmentation": self.instance_segmentation,
                        "distance_to_camera": self.distance_to_camera,
                        "distance_to_image_plane": self.distance_to_image_plane,
                        "colorize_depth": self.colorize_depth,
                        "colorize_semantic_segmentation": self.colorize_semantic_segmentation,
                        "colorize_instance_id_segmentation": self.colorize_instance_id_segmentation,
                        "colorize_instance_segmentation": self.colorize_instance_segmentation,
                        "occlusion": self.occlusion,
                        "normals": self.normals,
                        "motion_vectors": self.motion_vectors,
                        "pointcloud": self.pointcloud,
                        "pointcloud_include_unlabelled": self.pointcloud_include_unlabelled,
                        "use_common_output_dir": self.use_common_output_dir,
                    },
                },
                "response": {
                    "response_list": self.response_list,
                },
                "event": {
                    "event_list": self.event_list,
                },
            }
        }


@dataclass
class DynamicLayerResult:
    """动态层执行结果。

    Attributes:
        enabled: 动态层是否启用。
        backend: 使用的后端 (``ira`` / ``synthetic``)。
        ira_config_path: 生成的 IRA 配置文件路径。
        command_file: 生成的 NPC 命令文件路径。
        robot_command_file: 生成的 Robot 命令文件路径。
        output_dir: IRA 输出的数据目录。
        track_file: 标准化后的动态轨迹文件路径。
        behavior_event_file: 行为事件文件路径。
        overlay_usd: 动态叠加 USD 路径（兼容旧流程）。
        object_count: 动态对象总数。
        sample_count: 生成的样本帧数。
        character_count: NPC 人物数量。
        robot_count: Robot 数量。
        camera_count: 摄像头数量。
        simulation_length: 仿真时长（秒）。
        ira_return_code: IRA 进程返回码。
        error_message: 错误信息（成功时为空）。
    """

    enabled: bool = False
    backend: str = "ira"
    ira_config_path: str = ""
    command_file: str = ""
    robot_command_file: str = ""
    output_dir: str = ""
    track_file: str = ""
    behavior_event_file: str = ""
    overlay_usd: str = ""
    object_count: int = 0
    sample_count: int = 0
    character_count: int = 0
    robot_count: int = 0
    camera_count: int = 0
    simulation_length: int = 0
    ira_return_code: int = -1
    error_message: str = ""

    @property
    def ok(self) -> bool:
        """是否执行成功。"""
        return self.ira_return_code == 0 and self.error_message == ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化字典，兼容 scene_manifest.dynamic 格式。"""
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "track_file": self.track_file,
            "behavior_event_file": self.behavior_event_file,
            "overlay_usd": self.overlay_usd,
            "object_count": self.object_count,
            "sample_count": self.sample_count,
            "ira_config_path": self.ira_config_path,
            "command_file": self.command_file,
            "robot_command_file": self.robot_command_file,
            "output_dir": self.output_dir,
            "character_count": self.character_count,
            "robot_count": self.robot_count,
            "camera_count": self.camera_count,
            "simulation_length": self.simulation_length,
        }

    def to_legacy_dict(self) -> Dict[str, Any]:
        """转换为兼容旧 DynamicsLayer.generate() 返回格式的字典。

        保证 ``SceneCompiler`` 不需要修改即可使用新动态层。
        """
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "track_file": self.track_file,
            "behavior_event_file": self.behavior_event_file,
            "overlay_usd": self.overlay_usd,
            "object_count": self.object_count,
            "sample_count": self.sample_count,
        }
