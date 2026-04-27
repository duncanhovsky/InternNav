"""动态层集成测试。

覆盖：
1. IRADynamicLayer 端到端 dry-run 流程
2. IRADynamicLayer 禁用路径
3. IRADynamicLayer fallback 路径（无 Isaac Sim 环境）
4. 完整配置文件生成 + 命令文件验证
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dataengine.src.dynamic_layer.config import DynamicLayerConfig
from dataengine.src.dynamic_layer.dynamic_layer import IRADynamicLayer
from dataengine.src.dynamic_layer.models import DynamicLayerResult


class TestIRADynamicLayerDryRun:
    """IRA 动态层 dry-run 集成测试。"""

    def test_dry_run_generates_config_files(self):
        """dry-run 应生成配置文件但不执行 IRA。"""
        cfg = DynamicLayerConfig(
            character_num_min=3,
            character_num_max=3,
            simulation_length=100,
        )
        layer = IRADynamicLayer(cfg=cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = layer.generate(
                scene_id="test_scene_001",
                scene_dir=tmpdir,
                seed=42,
                stage_usd="/tmp/fake_scene.usd",
                dry_run=True,
            )

            # 基本属性
            assert result.enabled is True
            assert result.backend == "ira"
            assert result.ira_return_code == 0
            assert result.character_count == 3

            # 配置文件应已生成
            assert result.ira_config_path != ""
            assert os.path.isfile(result.ira_config_path)

            # 命令文件应已生成
            assert result.command_file != ""
            assert os.path.isfile(result.command_file)

            # robot 命令文件应已生成
            assert result.robot_command_file != ""
            assert os.path.isfile(result.robot_command_file)

    def test_dry_run_config_content(self):
        """dry-run 生成的 IRA 配置内容应正确。"""
        cfg = DynamicLayerConfig(
            character_num_min=5,
            character_num_max=5,
            simulation_length=200,
            camera_num=8,
        )
        layer = IRADynamicLayer(cfg=cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = layer.generate(
                scene_id="test_scene_002",
                scene_dir=tmpdir,
                seed=100,
                stage_usd="/tmp/scene.usd",
                dry_run=True,
            )

            with open(result.ira_config_path, "r") as f:
                content = f.read()

            assert "isaacsim.replicator.agent" in content
            assert "seed: 100" in content
            assert "simulation_length: 200" in content

    def test_dry_run_command_file_content(self):
        """dry-run 生成的命令文件应有正确格式。"""
        cfg = DynamicLayerConfig(
            character_num_min=3,
            character_num_max=3,
            max_commands_per_character=4,
        )
        layer = IRADynamicLayer(cfg=cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = layer.generate(
                scene_id="test_scene_003",
                scene_dir=tmpdir,
                seed=42,
                stage_usd="/tmp/scene.usd",
                dry_run=True,
            )

            with open(result.command_file, "r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]

            # 每行应以角色 ID 开头
            for line in lines:
                parts = line.split()
                assert len(parts) >= 2
                assert parts[0].startswith("Character")
                assert parts[1] in {"Idle", "GoTo", "LookAround"}

    def test_dry_run_deterministic(self):
        """相同参数的 dry-run 应生成相同命令序列。

        注意：IRA YAML 中包含绝对路径（tmpdir），因此只比较命令文件内容。
        """
        cfg = DynamicLayerConfig(character_num_min=5, character_num_max=5)

        with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
            layer1 = IRADynamicLayer(cfg=cfg)
            layer2 = IRADynamicLayer(cfg=cfg)

            r1 = layer1.generate(
                scene_id="scene", scene_dir=tmpdir1, seed=42,
                stage_usd="/tmp/s.usd", dry_run=True,
            )
            r2 = layer2.generate(
                scene_id="scene", scene_dir=tmpdir2, seed=42,
                stage_usd="/tmp/s.usd", dry_run=True,
            )

            # 命令文件内容应完全一致（不含路径依赖）
            with open(r1.command_file) as f1, open(r2.command_file) as f2:
                assert f1.read() == f2.read()

            # 角色数量应一致
            assert r1.character_count == r2.character_count


class TestIRADynamicLayerDisabled:
    """动态层禁用路径测试。"""

    def test_disabled_returns_empty_result(self):
        cfg = DynamicLayerConfig(enable_dynamics=False)
        layer = IRADynamicLayer(cfg=cfg)
        result = layer.generate(
            scene_id="test_disabled",
            scene_dir="/tmp/test",
            seed=42,
        )
        assert result.enabled is False
        assert result.ira_return_code == 0

    def test_disabled_legacy_dict(self):
        cfg = DynamicLayerConfig(enable_dynamics=False)
        layer = IRADynamicLayer(cfg=cfg)
        d = layer.generate_legacy_dict(
            scene_id="test_disabled",
            scene_dir="/tmp/test",
            seed=42,
        )
        assert d["enabled"] is False
        assert "track_file" in d


class TestIRADynamicLayerFallback:
    """无 Isaac Sim 环境时的 fallback 测试。"""

    def test_fallback_to_synthetic_when_ira_unavailable(self):
        """IRA 不可用时应 fallback 到合成后端。"""
        cfg = DynamicLayerConfig(
            character_num_min=2,
            character_num_max=2,
            isaac_sim_python="/nonexistent/python.sh",
            fallback_to_synthetic=True,
        )
        layer = IRADynamicLayer(cfg=cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = layer.generate(
                scene_id="test_fallback",
                scene_dir=tmpdir,
                seed=42,
                stage_usd="",
            )

            # 应该有配置文件（即使 IRA 不可用也会先生成）
            assert result.ira_config_path != ""
            assert result.command_file != ""
            # backend 应该标记为 fallback
            assert "fallback" in result.backend or result.ira_return_code != 0

    def test_no_fallback_returns_error(self):
        """禁止 fallback 且 IRA 不可用时返回错误结果。"""
        cfg = DynamicLayerConfig(
            character_num_min=2,
            character_num_max=2,
            isaac_sim_python="/nonexistent/python.sh",
            fallback_to_synthetic=False,
        )
        layer = IRADynamicLayer(cfg=cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = layer.generate(
                scene_id="test_no_fallback",
                scene_dir=tmpdir,
                seed=42,
            )
            assert result.error_message != ""


class TestIRADynamicLayerLegacyCompat:
    """旧接口兼容性测试。"""

    def test_generate_legacy_dict_keys(self):
        """generate_legacy_dict 返回的键集合应与旧 DynamicsLayer 一致。"""
        cfg = DynamicLayerConfig(enable_dynamics=False)
        layer = IRADynamicLayer(cfg=cfg)
        d = layer.generate_legacy_dict(
            scene_id="compat_test",
            scene_dir="/tmp/test",
            seed=42,
        )
        expected_keys = {"enabled", "backend", "track_file", "behavior_event_file",
                         "overlay_usd", "object_count", "sample_count"}
        assert set(d.keys()) == expected_keys

    def test_dry_run_legacy_dict(self):
        """dry-run 模式下 generate_legacy_dict 应返回合理值。"""
        cfg = DynamicLayerConfig(character_num_min=3, character_num_max=3)
        layer = IRADynamicLayer(cfg=cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            # 通过 generate 再转换
            result = layer.generate(
                scene_id="compat_dry",
                scene_dir=tmpdir,
                seed=42,
                stage_usd="/tmp/scene.usd",
                dry_run=True,
            )
            d = result.to_legacy_dict()
            assert d["enabled"] is True
            assert d["backend"] == "ira"
