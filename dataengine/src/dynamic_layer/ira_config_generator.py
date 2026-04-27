"""IRA 配置文件生成器。

根据 DynamicLayerConfig + 场景信息生成 isaacsim.replicator.agent 所需的 YAML 配置文件。
"""

from __future__ import annotations

import os
from typing import Any, Dict

from .config import DynamicLayerConfig
from .models import IRAConfig

# 延迟导入 yaml，支持无 PyYAML 环境的 fallback
_yaml = None


def _get_yaml():
    global _yaml
    if _yaml is None:
        try:
            import yaml
            _yaml = yaml
        except ImportError:
            _yaml = None
    return _yaml


def _dump_yaml_fallback(data: Dict[str, Any]) -> str:
    """无 PyYAML 时的简易 YAML 序列化。

    仅支持 IRA 配置中用到的类型：dict / list / str / int / float / bool。
    """
    lines: list[str] = []

    def _write(obj: Any, indent: int = 0) -> None:
        prefix = "  " * indent
        if isinstance(obj, dict):
            for key, val in obj.items():
                if isinstance(val, (dict,)):
                    lines.append(f"{prefix}{key}:")
                    _write(val, indent + 1)
                elif isinstance(val, list):
                    if len(val) == 0:
                        lines.append(f"{prefix}{key}: []")
                    else:
                        lines.append(f"{prefix}{key}:")
                        for item in val:
                            if isinstance(item, dict):
                                lines.append(f"{prefix}- ")
                                _write(item, indent + 1)
                            else:
                                lines.append(f"{prefix}- {_scalar(item)}")
                elif isinstance(val, bool):
                    lines.append(f"{prefix}{key}: {str(val).lower()}")
                elif isinstance(val, (int, float)):
                    lines.append(f"{prefix}{key}: {val}")
                elif isinstance(val, str):
                    # 含特殊字符时加引号
                    if any(c in val for c in ":{}[]#&*!|>'\"%@`"):
                        lines.append(f'{prefix}{key}: "{val}"')
                    else:
                        lines.append(f"{prefix}{key}: {val}")
                else:
                    lines.append(f"{prefix}{key}: {val}")
        elif isinstance(obj, list):
            for item in obj:
                lines.append(f"{prefix}- {_scalar(item)}")

    def _scalar(val: Any) -> str:
        if isinstance(val, bool):
            return str(val).lower()
        if isinstance(val, str):
            if any(c in val for c in ":{}[]#&*!|>'\"%@`"):
                return f'"{val}"'
            return val
        return str(val)

    _write(data)
    return "\n".join(lines) + "\n"


class IRAConfigGenerator:
    """IRA 配置文件生成器。

    核心职责：
    1. 将 DynamicLayerConfig 映射到 IRAConfig 模型
    2. 序列化为 YAML 文件写入指定目录
    3. 返回生成的配置文件路径
    """

    def __init__(self, cfg: DynamicLayerConfig) -> None:
        self.cfg = cfg

    def build_ira_config(
        self,
        scene_usd: str,
        seed: int,
        character_num: int,
        command_file: str,
        robot_command_file: str,
        output_dir: str,
    ) -> IRAConfig:
        """构建 IRAConfig 实例。

        Args:
            scene_usd: 场景 USD 文件的绝对路径。
            seed: 随机种子。
            character_num: 确定的角色数量。
            command_file: NPC 命令文件的绝对路径。
            robot_command_file: Robot 命令文件的绝对路径。
            output_dir: Replicator 输出目录。

        Returns:
            填充完毕的 IRAConfig 实例。
        """
        return IRAConfig(
            # global
            seed=seed,
            simulation_length=self.cfg.simulation_length,
            # scene
            scene_asset_path=scene_usd,
            # character
            character_asset_path=self.cfg.character_asset_path,
            character_filters=list(self.cfg.character_filters),
            character_num=character_num,
            character_spawn_area=list(self.cfg.character_spawn_area),
            character_navigation_area=list(self.cfg.character_navigation_area),
            character_command_file=command_file,
            # robot
            robot_command_file=robot_command_file,
            robot_nova_carter_num=self.cfg.robot_nova_carter_num,
            robot_iw_hub_num=self.cfg.robot_iw_hub_num,
            robot_spawn_area=list(self.cfg.robot_spawn_area),
            robot_navigation_area=list(self.cfg.robot_navigation_area),
            robot_write_data=self.cfg.robot_write_data,
            # sensor
            camera_num=self.cfg.camera_num,
            # replicator
            writer=self.cfg.writer,
            output_dir=output_dir,
            rgb=self.cfg.rgb,
            camera_params=self.cfg.camera_params,
            image_output_format=self.cfg.image_output_format,
            object_info_bounding_box_2d_tight=self.cfg.bbox_2d_tight,
            object_info_bounding_box_2d_loose=self.cfg.bbox_2d_loose,
            object_info_bounding_box_3d=self.cfg.bbox_3d,
            agent_info_skeleton_data=self.cfg.skeleton_data,
            semantic_filter_predicate=self.cfg.semantic_filter_predicate,
            semantic_segmentation=self.cfg.semantic_segmentation,
            instance_id_segmentation=self.cfg.instance_id_segmentation,
            instance_segmentation=self.cfg.instance_segmentation,
            distance_to_camera=self.cfg.distance_to_camera,
            distance_to_image_plane=self.cfg.distance_to_image_plane,
            colorize_depth=self.cfg.colorize_depth,
            occlusion=self.cfg.occlusion,
            normals=self.cfg.normals,
            motion_vectors=self.cfg.motion_vectors,
            pointcloud=self.cfg.pointcloud,
        )

    def generate(
        self,
        scene_usd: str,
        seed: int,
        character_num: int,
        command_file: str,
        robot_command_file: str,
        output_dir: str,
        config_dir: str,
    ) -> str:
        """生成 IRA YAML 配置文件并写入磁盘。

        Args:
            scene_usd: 场景 USD 文件路径。
            seed: 随机种子。
            character_num: 角色数量。
            command_file: NPC 命令文件路径。
            robot_command_file: Robot 命令文件路径。
            output_dir: Replicator 输出目录。
            config_dir: 配置文件输出目录。

        Returns:
            生成的 YAML 配置文件绝对路径。
        """
        ira_config = self.build_ira_config(
            scene_usd=scene_usd,
            seed=seed,
            character_num=character_num,
            command_file=command_file,
            robot_command_file=robot_command_file,
            output_dir=output_dir,
        )

        config_dict = ira_config.to_dict()
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, "default_config.yaml")

        yaml_mod = _get_yaml()
        if yaml_mod is not None:
            yaml_text = yaml_mod.dump(
                config_dict,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        else:
            yaml_text = _dump_yaml_fallback(config_dict)

        with open(config_path, "w", encoding="utf-8") as f:
            f.write(yaml_text)

        return os.path.abspath(config_path)
