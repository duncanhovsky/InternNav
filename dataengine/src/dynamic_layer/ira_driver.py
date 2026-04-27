"""IRA 插件驱动器。

负责启动 Isaac Sim 并加载 isaacsim.replicator.agent 扩展执行仿真。

支持两种执行模式：
1. **子进程模式**（默认）：通过 Isaac Sim Python 解释器启动 IRA
2. **进程内模式**（Isaac Sim 环境内）：直接调用 IRA API

当 Isaac Sim 不可用时，根据配置决定是否 fallback 到 dry-run。
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .config import DynamicLayerConfig

logger = logging.getLogger(__name__)


def _find_isaac_sim_python() -> Optional[str]:
    """尝试自动探测 Isaac Sim Python 解释器路径。

    搜索策略（按优先级）：
    1. 环境变量 ``ISAAC_SIM_PYTHON``
    2. 环境变量 ``ISAAC_SIM_PATH`` 下的 ``python.sh``
    3. 常见安装路径
    """
    # 策略 1：环境变量
    env_python = os.environ.get("ISAAC_SIM_PYTHON", "").strip()
    if env_python and os.path.isfile(env_python):
        return env_python

    # 策略 2：ISAAC_SIM_PATH
    sim_path = os.environ.get("ISAAC_SIM_PATH", "").strip()
    if sim_path:
        candidates = [
            os.path.join(sim_path, "python.sh"),
            os.path.join(sim_path, "isaac-sim.sh"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c

    # 策略 3：常见路径
    common_paths = [
        os.path.expanduser("~/.local/share/ov/pkg/isaac-sim-5.1.0/python.sh"),
        "/isaac-sim/python.sh",
        os.path.expanduser("~/isaacsim/python.sh"),
    ]
    for p in common_paths:
        if os.path.isfile(p):
            return p

    return None


@dataclass
class IRADriverResult:
    """IRA 驱动执行结果。"""

    return_code: int
    stdout: str
    stderr: str
    config_path: str
    output_dir: str
    success: bool

    @property
    def error_summary(self) -> str:
        """提取错误摘要（stderr 最后几行）。"""
        if not self.stderr:
            return ""
        lines = self.stderr.strip().splitlines()
        tail = lines[-5:] if len(lines) > 5 else lines
        return " | ".join(tail)


class IRADriver:
    """IRA 插件驱动器。

    职责：
    1. 构建 Isaac Sim + IRA 的启动命令
    2. 以子进程方式执行仿真
    3. 监控执行状态并返回结果
    """

    def __init__(self, cfg: DynamicLayerConfig) -> None:
        self.cfg = cfg
        self._isaac_python: Optional[str] = None

    def _resolve_isaac_python(self) -> str:
        """解析 Isaac Sim Python 路径。"""
        if self._isaac_python:
            return self._isaac_python

        if self.cfg.isaac_sim_python:
            self._isaac_python = self.cfg.isaac_sim_python
        else:
            found = _find_isaac_sim_python()
            if found:
                self._isaac_python = found
            else:
                self._isaac_python = ""

        return self._isaac_python

    def is_available(self) -> bool:
        """检查 IRA 运行环境是否可用。"""
        python_path = self._resolve_isaac_python()
        return bool(python_path) and os.path.isfile(python_path)

    def _build_command(self, config_path: str) -> List[str]:
        """构建 IRA 启动命令行。

        根据 Isaac Sim 5.1 的启动方式，有两种方案：

        方案 A（推荐）：通过 Isaac Sim Python 直接运行 IRA 模块::

            /path/to/isaac-sim/python.sh -m isaacsim.replicator.agent \\
                --config /path/to/default_config.yaml

        方案 B：通过 Isaac Sim 启动器加载扩展::

            /path/to/isaac-sim/isaac-sim.sh \\
                --ext-folder /path/to/extensions \\
                --enable isaacsim.replicator.agent

        当前实现采用方案 A。
        """
        python_path = self._resolve_isaac_python()
        cmd = [python_path]

        # IRA 模块入口
        cmd.extend(["-m", self.cfg.ira_module])

        # 配置文件
        cmd.extend(["--config", config_path])

        # headless 模式
        if self.cfg.headless:
            cmd.append("--headless")

        return cmd

    def run(
        self,
        config_path: str,
        output_dir: str,
        dry_run: bool = False,
    ) -> IRADriverResult:
        """执行 IRA 仿真。

        Args:
            config_path: IRA YAML 配置文件路径。
            output_dir: 预期的输出目录。
            dry_run: 为 True 时仅记录命令不实际执行。

        Returns:
            IRADriverResult 执行结果。
        """
        if dry_run:
            cmd = self._build_command(config_path) if self.is_available() else ["[dry-run]"]
            logger.info("[dry-run] IRA command: %s", " ".join(cmd))
            return IRADriverResult(
                return_code=0,
                stdout="[dry-run] IRA not executed",
                stderr="",
                config_path=config_path,
                output_dir=output_dir,
                success=True,
            )

        if not self.is_available():
            msg = (
                "Isaac Sim Python 解释器不可用。"
                f" isaac_sim_python={self.cfg.isaac_sim_python!r},"
                " 尝试自动探测也未找到。"
                " 请设置环境变量 ISAAC_SIM_PYTHON 或在配置中指定 isaac_sim_python。"
            )
            logger.warning(msg)
            return IRADriverResult(
                return_code=-1,
                stdout="",
                stderr=msg,
                config_path=config_path,
                output_dir=output_dir,
                success=False,
            )

        cmd = self._build_command(config_path)
        logger.info("Starting IRA: %s", " ".join(cmd))

        os.makedirs(output_dir, exist_ok=True)

        try:
            completed = subprocess.run(  # noqa: S603
                cmd,
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
                timeout=max(1, self.cfg.timeout_s),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            logger.error("IRA execution timeout after %ds", self.cfg.timeout_s)
            return IRADriverResult(
                return_code=-2,
                stdout=str(getattr(exc, "stdout", "") or ""),
                stderr=f"IRA timeout after {self.cfg.timeout_s}s: {exc}",
                config_path=config_path,
                output_dir=output_dir,
                success=False,
            )
        except FileNotFoundError as exc:
            logger.error("Isaac Sim executable not found: %s", exc)
            return IRADriverResult(
                return_code=-3,
                stdout="",
                stderr=f"Isaac Sim executable not found: {exc}",
                config_path=config_path,
                output_dir=output_dir,
                success=False,
            )

        success = completed.returncode == 0
        if not success:
            logger.warning(
                "IRA exited with code %d. stderr tail: %s",
                completed.returncode,
                (completed.stderr or "")[-500:],
            )

        return IRADriverResult(
            return_code=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            config_path=config_path,
            output_dir=output_dir,
            success=success,
        )

    def run_in_process(self, config_path: str) -> IRADriverResult:
        """进程内执行 IRA（需要在 Isaac Sim 环境中调用）。

        该方法供在 Isaac Sim 的 Python 环境内直接使用，
        避免子进程启动的开销。

        Args:
            config_path: IRA 配置文件路径。

        Returns:
            IRADriverResult。
        """
        try:
            # 尝试导入 IRA 模块（仅在 Isaac Sim 环境内可用）
            import importlib
            ira_mod = importlib.import_module(self.cfg.ira_module)

            # IRA 通常提供 load_config + run 入口
            if hasattr(ira_mod, "load_config"):
                ira_mod.load_config(config_path)
            if hasattr(ira_mod, "run"):
                ira_mod.run()

            return IRADriverResult(
                return_code=0,
                stdout="in-process execution completed",
                stderr="",
                config_path=config_path,
                output_dir=self.cfg.output_dir,
                success=True,
            )
        except ImportError as exc:
            return IRADriverResult(
                return_code=-4,
                stdout="",
                stderr=f"Cannot import IRA module in-process: {exc}",
                config_path=config_path,
                output_dir=self.cfg.output_dir,
                success=False,
            )
        except Exception as exc:  # noqa: BLE001
            return IRADriverResult(
                return_code=-5,
                stdout="",
                stderr=f"IRA in-process execution failed: {exc}",
                config_path=config_path,
                output_dir=self.cfg.output_dir,
                success=False,
            )
