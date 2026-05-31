"""NPC 行为命令生成器。

为每个 NPC 角色生成 IRA 的行为命令序列 (command.txt)，
以及空的 robot_command.txt（后续 Robot 接入时扩展）。

命令格式（每行一条）::

    Character Idle 4.1
    Character GoTo -14.16 7.29 0.0 _
    Character_01 LookAround 2.5
    Character_01 GoTo -19.28 28.59 0.0 _

支持的行为模式：
- stand_idle: NPC 原地站立，仅生成 Idle 命令
- walk: NPC 在可行走区域间漫游，生成 GoTo + Idle 交替序列
- mixed: 按概率分配 Idle / GoTo / LookAround
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Tuple

from .config import DynamicLayerConfig
from .models import CommandEntry, RobotCommandEntry


def _character_id(index: int) -> str:
    """生成 IRA 角色 ID。

    按照 IRA 约定：第一个角色为 ``Character``，后续为 ``Character_01``, ``Character_02``, ...
    """
    if index == 0:
        return "Character"
    return f"Character_{index:02d}"


class CommandGenerator:
    """NPC 行为命令生成器。

    根据配置中的行为比例和参数范围，为每个 NPC 生成确定性的命令序列。
    """

    def __init__(self, cfg: DynamicLayerConfig) -> None:
        self.cfg = cfg

    def _assign_motion_modes(
        self,
        rng: random.Random,
        character_num: int,
    ) -> List[str]:
        """为每个角色分配运动模式。

        根据 idle_ratio 和 look_around_ratio 确定每种模式的角色数量：
        - stand_idle: 仅 Idle
        - walk: GoTo + Idle 交替
        - mixed: Idle / GoTo / LookAround 混合

        Returns:
            长度为 character_num 的模式列表。
        """
        n_idle = max(0, int(character_num * self.cfg.idle_ratio))
        n_look = max(0, int(character_num * self.cfg.look_around_ratio))
        n_walk = max(0, character_num - n_idle - n_look)

        modes: List[str] = (
            ["stand_idle"] * n_idle
            + ["mixed"] * n_look
            + ["walk"] * n_walk
        )
        # 不足时补 walk
        while len(modes) < character_num:
            modes.append("walk")
        # 截断
        modes = modes[:character_num]
        rng.shuffle(modes)
        return modes

    def _gen_idle_command(self, rng: random.Random, char_id: str) -> CommandEntry:
        """生成单条 Idle 命令。"""
        lo, hi = self.cfg.idle_duration_range
        duration = round(rng.uniform(lo, hi), 2)
        return CommandEntry.idle(char_id, duration)

    def _gen_goto_command(self, rng: random.Random, char_id: str) -> CommandEntry:
        """生成单条 GoTo 命令。"""
        x = round(rng.uniform(*self.cfg.goto_x_range), 2)
        y = round(rng.uniform(*self.cfg.goto_y_range), 2)
        z = self.cfg.goto_z
        return CommandEntry.goto(char_id, x, y, z)

    def _gen_look_around_command(self, rng: random.Random, char_id: str) -> CommandEntry:
        """生成单条 LookAround 命令。"""
        lo, hi = self.cfg.look_around_duration_range
        duration = round(rng.uniform(lo, hi), 2)
        return CommandEntry.look_around(char_id, duration)

    def _gen_commands_stand_idle(
        self,
        rng: random.Random,
        char_id: str,
        max_cmds: int,
    ) -> List[CommandEntry]:
        """生成站立模式的命令序列（仅 Idle）。"""
        cmds: List[CommandEntry] = []
        for _ in range(min(max_cmds, 2)):
            cmds.append(self._gen_idle_command(rng, char_id))
        return cmds

    def _gen_commands_walk(
        self,
        rng: random.Random,
        char_id: str,
        max_cmds: int,
    ) -> List[CommandEntry]:
        """生成行走模式的命令序列（GoTo + Idle 交替）。"""
        cmds: List[CommandEntry] = []
        for i in range(max_cmds):
            if i % 2 == 0:
                cmds.append(self._gen_goto_command(rng, char_id))
            else:
                cmds.append(self._gen_idle_command(rng, char_id))
        return cmds

    def _gen_commands_mixed(
        self,
        rng: random.Random,
        char_id: str,
        max_cmds: int,
    ) -> List[CommandEntry]:
        """生成混合模式的命令序列。"""
        cmds: List[CommandEntry] = []
        actions = ["Idle", "GoTo", "LookAround"]
        weights = [0.25, 0.50, 0.25]
        for _ in range(max_cmds):
            action = rng.choices(actions, weights=weights, k=1)[0]
            if action == "Idle":
                cmds.append(self._gen_idle_command(rng, char_id))
            elif action == "GoTo":
                cmds.append(self._gen_goto_command(rng, char_id))
            else:
                cmds.append(self._gen_look_around_command(rng, char_id))
        return cmds

    def generate_commands(
        self,
        seed: int,
        character_num: int,
    ) -> List[CommandEntry]:
        """为所有角色生成完整的命令序列。

        Args:
            seed: 随机种子，保证确定性。
            character_num: 角色总数。

        Returns:
            所有角色的命令列表（按角色顺序排列）。
        """
        rng = random.Random(seed + 8001)
        modes = self._assign_motion_modes(rng, character_num)
        all_commands: List[CommandEntry] = []

        for idx in range(character_num):
            char_id = _character_id(idx)
            mode = modes[idx]
            max_cmds = self.cfg.max_commands_per_character

            if mode == "stand_idle":
                cmds = self._gen_commands_stand_idle(rng, char_id, max_cmds)
            elif mode == "walk":
                cmds = self._gen_commands_walk(rng, char_id, max_cmds)
            elif mode == "mixed":
                cmds = self._gen_commands_mixed(rng, char_id, max_cmds)
            else:
                cmds = self._gen_commands_walk(rng, char_id, max_cmds)

            all_commands.extend(cmds)

        return all_commands

    def write_command_file(
        self,
        commands: List[CommandEntry],
        output_path: str,
    ) -> str:
        """将命令序列写入 command.txt 文件。

        Args:
            commands: 命令列表。
            output_path: 输出文件路径。

        Returns:
            写入的文件绝对路径。
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        lines = [cmd.to_line() for cmd in commands]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return os.path.abspath(output_path)

    def write_robot_command_file(
        self,
        commands: Optional[List[RobotCommandEntry]],
        output_path: str,
    ) -> str:
        """将 Robot 命令写入 robot_command.txt。

        当前版本生成空文件（无 Robot 命令），后续扩展。

        Args:
            commands: Robot 命令列表（当前为 None 或空）。
            output_path: 输出文件路径。

        Returns:
            写入的文件绝对路径。
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if commands:
            lines = [cmd.to_line() for cmd in commands]
            content = "\n".join(lines) + "\n"
        else:
            content = "\n"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        return os.path.abspath(output_path)

    def generate_and_write(
        self,
        seed: int,
        character_num: int,
        config_dir: str,
    ) -> Tuple[str, str, List[CommandEntry]]:
        """一站式生成并写入命令文件。

        Args:
            seed: 随机种子。
            character_num: 角色数量。
            config_dir: 配置文件输出目录。

        Returns:
            (command_file_path, robot_command_file_path, commands)
        """
        commands = self.generate_commands(seed=seed, character_num=character_num)
        cmd_path = self.write_command_file(
            commands=commands,
            output_path=os.path.join(config_dir, "command.txt"),
        )
        robot_cmd_path = self.write_robot_command_file(
            commands=None,
            output_path=os.path.join(config_dir, "robot_command.txt"),
        )
        return cmd_path, robot_cmd_path, commands
