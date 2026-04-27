"""动态层主编排器。

IRADynamicLayer 是动态层的唯一对外接口，负责编排：
1. 配置生成（IRAConfigGenerator）
2. 命令生成（CommandGenerator）
3. IRA 执行（IRADriver）
4. 后处理（PostProcessor）

对外暴露 ``generate()`` 方法，接口签名兼容旧 ``DynamicsLayer.generate()``，
使 ``SceneCompiler`` 可以无感切换。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .command_generator import CommandGenerator
from .config import DynamicLayerConfig
from .ira_config_generator import IRAConfigGenerator
from .ira_driver import IRADriver
from .models import DynamicLayerResult
from .post_processor import PostProcessor

logger = logging.getLogger(__name__)


@dataclass
class IRADynamicLayer:
    """基于 IRA 的动态层主编排器。

    该类编排整个动态层流程：

    1. 根据配置计算角色数量
    2. 生成 NPC 行为命令文件 (command.txt)
    3. 生成 IRA 配置文件 (default_config.yaml)
    4. 驱动 IRA 插件执行仿真
    5. 后处理 IRA 输出为标准格式
    6. 返回兼容旧流程的结果字典

    当 IRA 不可用且 ``fallback_to_synthetic`` 为 True 时，
    退化到旧的 ``DynamicsLayer`` 合成后端。
    """

    cfg: DynamicLayerConfig

    def __post_init__(self) -> None:
        self.config_generator = IRAConfigGenerator(self.cfg)
        self.command_generator = CommandGenerator(self.cfg)
        self.driver = IRADriver(self.cfg)

    def generate(
        self,
        scene_id: str,
        scene_dir: str,
        seed: int,
        stage_usd: str = "",
        dry_run: bool = False,
    ) -> DynamicLayerResult:
        """执行动态层完整流程。

        Args:
            scene_id: 场景标识符。
            scene_dir: 场景工作目录。
            seed: 随机种子。
            stage_usd: 场景 USD 文件路径。
            dry_run: 仅生成配置不执行 IRA。

        Returns:
            DynamicLayerResult 包含执行结果和所有产物路径。
        """
        if not self.cfg.enable_dynamics:
            return DynamicLayerResult(
                enabled=False,
                backend="ira",
                ira_return_code=0,
            )

        # ─── 1. 准备目录 ───
        config_dir = self.cfg.ira_config_dir or os.path.join(scene_dir, "ira_config")
        output_dir = self.cfg.output_dir or os.path.join(scene_dir, "ira_output")
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # ─── 2. 确定角色数量 ───
        character_num = self.cfg.pick_character_num(seed)
        logger.info(
            "DynamicLayer: scene_id=%s, seed=%d, character_num=%d",
            scene_id, seed, character_num,
        )

        # ─── 3. 生成命令文件 ───
        cmd_path, robot_cmd_path, commands = self.command_generator.generate_and_write(
            seed=seed,
            character_num=character_num,
            config_dir=config_dir,
        )
        logger.info(
            "Generated %d commands -> %s",
            len(commands), cmd_path,
        )

        # ─── 4. 生成 IRA 配置文件 ───
        ira_config_path = self.config_generator.generate(
            scene_usd=stage_usd or self.cfg.scene_usd,
            seed=seed,
            character_num=character_num,
            command_file=cmd_path,
            robot_command_file=robot_cmd_path,
            output_dir=output_dir,
            config_dir=config_dir,
        )
        logger.info("Generated IRA config -> %s", ira_config_path)

        # ─── 5. 检查 IRA 可用性 ───
        if not dry_run and not self.driver.is_available():
            if self.cfg.fallback_to_synthetic:
                logger.warning(
                    "IRA not available, falling back to synthetic backend. "
                    "Config files were generated at: %s", config_dir,
                )
                return self._fallback_synthetic(
                    scene_id=scene_id,
                    scene_dir=scene_dir,
                    seed=seed,
                    stage_usd=stage_usd,
                    config_dir=config_dir,
                    ira_config_path=ira_config_path,
                    cmd_path=cmd_path,
                    robot_cmd_path=robot_cmd_path,
                    character_num=character_num,
                )
            else:
                return DynamicLayerResult(
                    enabled=True,
                    backend="ira",
                    ira_config_path=ira_config_path,
                    command_file=cmd_path,
                    robot_command_file=robot_cmd_path,
                    output_dir=output_dir,
                    character_count=character_num,
                    ira_return_code=-1,
                    error_message="IRA not available and fallback disabled",
                )

        # ─── 6. 执行 IRA ───
        driver_result = self.driver.run(
            config_path=ira_config_path,
            output_dir=output_dir,
            dry_run=dry_run,
        )

        if dry_run:
            return DynamicLayerResult(
                enabled=True,
                backend="ira",
                ira_config_path=ira_config_path,
                command_file=cmd_path,
                robot_command_file=robot_cmd_path,
                output_dir=output_dir,
                character_count=character_num,
                camera_count=self.cfg.camera_num,
                simulation_length=self.cfg.simulation_length,
                ira_return_code=0,
            )

        if not driver_result.success:
            logger.error(
                "IRA execution failed: return_code=%d, error=%s",
                driver_result.return_code,
                driver_result.error_summary,
            )
            # 尝试 fallback
            if self.cfg.fallback_to_synthetic:
                logger.warning("Falling back to synthetic backend after IRA failure")
                return self._fallback_synthetic(
                    scene_id=scene_id,
                    scene_dir=scene_dir,
                    seed=seed,
                    stage_usd=stage_usd,
                    config_dir=config_dir,
                    ira_config_path=ira_config_path,
                    cmd_path=cmd_path,
                    robot_cmd_path=robot_cmd_path,
                    character_num=character_num,
                )

            return DynamicLayerResult(
                enabled=True,
                backend="ira",
                ira_config_path=ira_config_path,
                command_file=cmd_path,
                robot_command_file=robot_cmd_path,
                output_dir=output_dir,
                character_count=character_num,
                ira_return_code=driver_result.return_code,
                error_message=driver_result.error_summary,
            )

        # ─── 7. 后处理 ───
        processor = PostProcessor(
            scene_id=scene_id,
            output_dir=output_dir,
            dt_s=self.cfg.simulation_length / max(1, len(self._count_frames(output_dir))),
        )
        post_result = processor.process(base_timestamp_ns=seed * 1_000_000)

        return DynamicLayerResult(
            enabled=True,
            backend="ira",
            ira_config_path=ira_config_path,
            command_file=cmd_path,
            robot_command_file=robot_cmd_path,
            output_dir=output_dir,
            track_file=post_result.get("track_file", ""),
            object_count=post_result.get("object_count", 0),
            sample_count=post_result.get("sample_count", 0),
            character_count=character_num,
            camera_count=self.cfg.camera_num,
            simulation_length=self.cfg.simulation_length,
            ira_return_code=0,
        )

    def _count_frames(self, output_dir: str) -> list:
        """统计 IRA 输出帧数。"""
        if not os.path.isdir(output_dir):
            return []
        return [
            d for d in sorted(os.listdir(output_dir))
            if os.path.isdir(os.path.join(output_dir, d))
        ]

    def _fallback_synthetic(
        self,
        scene_id: str,
        scene_dir: str,
        seed: int,
        stage_usd: str,
        config_dir: str,
        ira_config_path: str,
        cmd_path: str,
        robot_cmd_path: str,
        character_num: int,
    ) -> DynamicLayerResult:
        """Fallback 到旧的合成后端。

        调用旧的 dynamics_worker 生成合成轨迹，
        同时保留已生成的 IRA 配置文件供后续手动使用。
        """
        try:
            from dataengine.src.scene_layer.dynamics_layer import DynamicsLayer

            legacy = DynamicsLayer(
                enable_dynamics=True,
                backend="ira_character_graph",
                timeout_s=90,
                steps=max(10, self.cfg.simulation_length // 10),
                dt_s=1.0,
                num_people_min=self.cfg.character_num_min,
                num_people_max=self.cfg.character_num_max,
                num_objects_min=0,
                num_objects_max=0,
                asset_root="",
                asset_strict=False,
                require_animation_binding=False,
            )
            legacy_result = legacy.generate(
                scene_id=scene_id,
                scene_dir=scene_dir,
                seed=seed,
                stage_usd=stage_usd,
            )
            return DynamicLayerResult(
                enabled=True,
                backend="synthetic_fallback",
                ira_config_path=ira_config_path,
                command_file=cmd_path,
                robot_command_file=robot_cmd_path,
                output_dir=os.path.join(scene_dir, "ira_output"),
                track_file=legacy_result.get("track_file", ""),
                behavior_event_file=legacy_result.get("behavior_event_file", ""),
                overlay_usd=legacy_result.get("overlay_usd", ""),
                object_count=legacy_result.get("object_count", 0),
                sample_count=legacy_result.get("sample_count", 0),
                character_count=character_num,
                simulation_length=self.cfg.simulation_length,
                ira_return_code=-1,
                error_message="Fallback to synthetic: IRA unavailable",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Synthetic fallback also failed: %s", exc)
            return DynamicLayerResult(
                enabled=True,
                backend="ira",
                ira_config_path=ira_config_path,
                command_file=cmd_path,
                robot_command_file=robot_cmd_path,
                character_count=character_num,
                ira_return_code=-1,
                error_message=f"IRA unavailable and fallback failed: {exc}",
            )

    def generate_legacy_dict(
        self,
        scene_id: str,
        scene_dir: str,
        seed: int,
        stage_usd: str = "",
    ) -> Dict[str, Any]:
        """兼容旧 DynamicsLayer.generate() 的接口。

        返回与旧 ``dynamics_layer.py`` 的 ``generate()`` 相同格式的字典，
        使 ``SceneCompiler`` 可以直接替换使用。
        """
        result = self.generate(
            scene_id=scene_id,
            scene_dir=scene_dir,
            seed=seed,
            stage_usd=stage_usd,
        )
        return result.to_legacy_dict()
