"""动态层单元测试。

覆盖：
1. DynamicLayerConfig 校验
2. CommandGenerator 命令生成与确定性
3. IRAConfigGenerator 配置生成
4. CommandEntry 模型序列化
5. DynamicLayerResult 格式兼容
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dataengine.src.dynamic_layer.config import DynamicLayerConfig
from dataengine.src.dynamic_layer.models import (
    CommandEntry,
    DynamicLayerResult,
    IRAConfig,
    RobotCommandEntry,
)
from dataengine.src.dynamic_layer.command_generator import CommandGenerator
from dataengine.src.dynamic_layer.ira_config_generator import IRAConfigGenerator, _dump_yaml_fallback


# ─── DynamicLayerConfig 测试 ───


class TestDynamicLayerConfig:
    """配置校验测试。"""

    def test_default_config_valid(self):
        """默认配置应通过校验。"""
        cfg = DynamicLayerConfig()
        cfg.validate()

    def test_disabled_skips_validation(self):
        """禁用时跳过校验，即使参数非法。"""
        cfg = DynamicLayerConfig(enable_dynamics=False, simulation_length=-1)
        cfg.validate()  # 应不抛异常

    def test_invalid_backend_rejected(self):
        """非法 backend 被拒绝。"""
        cfg = DynamicLayerConfig(backend="invalid")
        with pytest.raises(ValueError, match="backend"):
            cfg.validate()

    def test_invalid_simulation_length(self):
        cfg = DynamicLayerConfig(simulation_length=0)
        with pytest.raises(ValueError, match="simulation_length"):
            cfg.validate()

    def test_invalid_character_num_range(self):
        cfg = DynamicLayerConfig(character_num_min=10, character_num_max=5)
        with pytest.raises(ValueError, match="character_num_max"):
            cfg.validate()

    def test_invalid_idle_ratio(self):
        cfg = DynamicLayerConfig(idle_ratio=0.8, look_around_ratio=0.5)
        with pytest.raises(ValueError, match="idle_ratio"):
            cfg.validate()

    def test_invalid_goto_range(self):
        cfg = DynamicLayerConfig(goto_x_range=[10.0, -10.0])
        with pytest.raises(ValueError, match="goto_x_range"):
            cfg.validate()

    def test_pick_character_num_deterministic(self):
        """相同 seed 产生相同角色数量。"""
        cfg = DynamicLayerConfig(character_num_min=5, character_num_max=15)
        n1 = cfg.pick_character_num(42)
        n2 = cfg.pick_character_num(42)
        assert n1 == n2
        assert 5 <= n1 <= 15

    def test_pick_character_num_fixed_range(self):
        """min == max 时返回固定值。"""
        cfg = DynamicLayerConfig(character_num_min=10, character_num_max=10)
        assert cfg.pick_character_num(99) == 10

    def test_from_dict(self):
        data = {"simulation_length": 500, "camera_num": 8}
        cfg = DynamicLayerConfig.from_dict(data)
        assert cfg.simulation_length == 500
        assert cfg.camera_num == 8

    def test_to_dict_roundtrip(self):
        cfg = DynamicLayerConfig(simulation_length=200)
        d = cfg.to_dict()
        assert d["simulation_length"] == 200
        assert "character_num_min" in d


# ─── CommandEntry 模型测试 ───


class TestCommandEntry:
    """命令条目模型测试。"""

    def test_idle_command(self):
        cmd = CommandEntry.idle("Character", 4.1)
        assert cmd.to_line() == "Character Idle 4.1"

    def test_goto_command(self):
        cmd = CommandEntry.goto("Character_01", -14.16, 7.29, 0.0)
        line = cmd.to_line()
        assert line.startswith("Character_01 GoTo")
        assert "-14.16" in line
        assert "7.29" in line
        assert "_" in line  # trailing underscore

    def test_look_around_command(self):
        cmd = CommandEntry.look_around("Character_02", 2.5)
        assert cmd.to_line() == "Character_02 LookAround 2.50"

    def test_robot_command_entry(self):
        cmd = RobotCommandEntry(robot_id="Robot_01", action="GoTo", params=["1.0", "2.0", "0.0"])
        assert "Robot_01" in cmd.to_line()


# ─── CommandGenerator 测试 ───


class TestCommandGenerator:
    """命令生成器测试。"""

    def test_generate_commands_nonempty(self):
        """生成命令应非空。"""
        cfg = DynamicLayerConfig(character_num_min=3, character_num_max=3)
        gen = CommandGenerator(cfg)
        commands = gen.generate_commands(seed=42, character_num=3)
        assert len(commands) > 0

    def test_generate_commands_deterministic(self):
        """相同 seed 生成相同命令。"""
        cfg = DynamicLayerConfig(character_num_min=5, character_num_max=5)
        gen = CommandGenerator(cfg)
        cmds1 = gen.generate_commands(seed=100, character_num=5)
        cmds2 = gen.generate_commands(seed=100, character_num=5)
        lines1 = [c.to_line() for c in cmds1]
        lines2 = [c.to_line() for c in cmds2]
        assert lines1 == lines2

    def test_generate_commands_different_seeds(self):
        """不同 seed 生成不同命令。"""
        cfg = DynamicLayerConfig(character_num_min=5, character_num_max=5)
        gen = CommandGenerator(cfg)
        cmds1 = gen.generate_commands(seed=100, character_num=5)
        cmds2 = gen.generate_commands(seed=200, character_num=5)
        lines1 = [c.to_line() for c in cmds1]
        lines2 = [c.to_line() for c in cmds2]
        assert lines1 != lines2

    def test_character_id_naming(self):
        """角色 ID 命名约定。"""
        cfg = DynamicLayerConfig(character_num_min=3, character_num_max=3)
        gen = CommandGenerator(cfg)
        commands = gen.generate_commands(seed=42, character_num=3)
        char_ids = {c.character_id for c in commands}
        assert "Character" in char_ids  # 第一个角色
        assert "Character_01" in char_ids
        assert "Character_02" in char_ids

    def test_write_command_file(self):
        """命令写入文件并可读。"""
        cfg = DynamicLayerConfig()
        gen = CommandGenerator(cfg)
        commands = gen.generate_commands(seed=42, character_num=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = gen.write_command_file(commands, os.path.join(tmpdir, "command.txt"))
            assert os.path.isfile(path)

            with open(path, "r") as f:
                content = f.read()
            lines = [l for l in content.strip().split("\n") if l.strip()]
            assert len(lines) == len(commands)

    def test_write_robot_command_file_empty(self):
        """空 robot 命令文件。"""
        cfg = DynamicLayerConfig()
        gen = CommandGenerator(cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = gen.write_robot_command_file(None, os.path.join(tmpdir, "robot_command.txt"))
            assert os.path.isfile(path)

    def test_generate_and_write(self):
        """一站式生成并写入。"""
        cfg = DynamicLayerConfig()
        gen = CommandGenerator(cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            cmd_path, robot_path, commands = gen.generate_and_write(
                seed=42, character_num=5, config_dir=tmpdir,
            )
            assert os.path.isfile(cmd_path)
            assert os.path.isfile(robot_path)
            assert len(commands) > 0


# ─── IRAConfigGenerator 测试 ───


class TestIRAConfigGenerator:
    """IRA 配置生成器测试。"""

    def test_build_ira_config(self):
        """构建 IRAConfig 实例。"""
        cfg = DynamicLayerConfig()
        gen = IRAConfigGenerator(cfg)
        ira_cfg = gen.build_ira_config(
            scene_usd="/tmp/test_scene.usd",
            seed=42,
            character_num=10,
            command_file="/tmp/command.txt",
            robot_command_file="/tmp/robot_command.txt",
            output_dir="/tmp/output",
        )
        assert ira_cfg.scene_asset_path == "/tmp/test_scene.usd"
        assert ira_cfg.seed == 42
        assert ira_cfg.character_num == 10
        assert ira_cfg.character_command_file == "/tmp/command.txt"

    def test_ira_config_to_dict_structure(self):
        """IRAConfig.to_dict() 返回正确的嵌套结构。"""
        ira_cfg = IRAConfig(seed=999, simulation_length=100)
        d = ira_cfg.to_dict()
        assert "isaacsim.replicator.agent" in d
        root = d["isaacsim.replicator.agent"]
        assert root["version"] == "0.7.1"
        assert root["global"]["seed"] == 999
        assert root["global"]["simulation_length"] == 100
        assert "character" in root
        assert "robot" in root
        assert "sensor" in root
        assert "replicator" in root

    def test_generate_yaml_file(self):
        """生成 YAML 配置文件。"""
        cfg = DynamicLayerConfig()
        gen = IRAConfigGenerator(cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = gen.generate(
                scene_usd="/tmp/scene.usd",
                seed=42,
                character_num=5,
                command_file="/tmp/cmd.txt",
                robot_command_file="/tmp/robot_cmd.txt",
                output_dir="/tmp/out",
                config_dir=tmpdir,
            )
            assert os.path.isfile(path)
            assert path.endswith("default_config.yaml")

            with open(path, "r") as f:
                content = f.read()
            assert "isaacsim.replicator.agent" in content
            assert "seed: 42" in content

    def test_yaml_fallback_serializer(self):
        """无 PyYAML 时的 fallback 序列化。"""
        data = {
            "key1": "value1",
            "key2": 42,
            "key3": True,
            "nested": {
                "a": [1, 2, 3],
                "b": "hello",
            },
            "empty_list": [],
        }
        text = _dump_yaml_fallback(data)
        assert "key1: value1" in text
        assert "key2: 42" in text
        assert "key3: true" in text
        assert "- 1" in text
        assert "empty_list: []" in text


# ─── DynamicLayerResult 测试 ───


class TestDynamicLayerResult:
    """结果模型测试。"""

    def test_ok_property(self):
        result = DynamicLayerResult(ira_return_code=0, error_message="")
        assert result.ok is True

    def test_not_ok_with_error(self):
        result = DynamicLayerResult(ira_return_code=0, error_message="some error")
        assert result.ok is False

    def test_not_ok_with_code(self):
        result = DynamicLayerResult(ira_return_code=1, error_message="")
        assert result.ok is False

    def test_to_dict(self):
        result = DynamicLayerResult(enabled=True, backend="ira", character_count=10)
        d = result.to_dict()
        assert d["enabled"] is True
        assert d["backend"] == "ira"
        assert d["character_count"] == 10

    def test_to_legacy_dict(self):
        """兼容旧格式的字典。"""
        result = DynamicLayerResult(
            enabled=True,
            backend="ira",
            track_file="/tmp/tracks.jsonl",
            object_count=5,
            sample_count=100,
        )
        d = result.to_legacy_dict()
        assert set(d.keys()) == {
            "enabled", "backend", "track_file",
            "behavior_event_file", "overlay_usd",
            "object_count", "sample_count",
        }
        assert d["track_file"] == "/tmp/tracks.jsonl"


# ─── IRAConfig 测试 ───


class TestIRAConfig:
    """IRA 配置模型详细测试。"""

    def test_default_values(self):
        cfg = IRAConfig()
        assert cfg.writer == "IRABasicWriter"
        assert cfg.character_num == 10

    def test_replicator_parameters(self):
        cfg = IRAConfig(rgb=True, semantic_segmentation=True)
        d = cfg.to_dict()
        params = d["isaacsim.replicator.agent"]["replicator"]["parameters"]
        assert params["rgb"] is True
        assert params["semantic_segmentation"] is True
        assert params["object_info_bounding_box_2d_tight"] is True
