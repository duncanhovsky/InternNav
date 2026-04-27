from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List

from .errors import ErrorCode, SceneCompileError


@dataclass
class NavMeshBuilder:
    """NavMesh 构建与可达性估计（Isaac Sim 官方插件实现）。"""

    bake_timeout_s: int
    cell_size_m: float = 0.05
    cell_height_m: float = 0.05
    max_slope_deg: float = 35.0
    step_height_m: float = 0.25
    navmesh_agent_profiles: List[str] = None
    agent_radius_m_by_profile: Dict[str, float] = None
    isaac_headless: bool = True
    isaac_renderer: str = "RayTracedLighting"

    def __post_init__(self) -> None:
        if self.navmesh_agent_profiles is None:
            self.navmesh_agent_profiles = ["go2"]
        if self.agent_radius_m_by_profile is None:
            self.agent_radius_m_by_profile = {"go2": 0.28}

    @staticmethod
    def _error_code_from_str(code: str) -> ErrorCode:
        try:
            return ErrorCode(code)
        except ValueError:
            return ErrorCode.RUNTIME_FAIL

    def _resolve_stage_usd(self, scene_dir: str, stage_usd: str | None) -> str:
        if stage_usd and os.path.isfile(stage_usd):
            return stage_usd

        generated = os.path.join(scene_dir, "stage_composed.usda")
        if os.path.isfile(generated):
            return generated

        raise SceneCompileError(
            code=ErrorCode.NAVMESH_FAIL,
            message=(
                "未找到可用于 NavMesh 烘焙的 stage USD。"
                f" stage_usd={stage_usd}, fallback={generated}"
            ),
        )

    def _pick_agent_radius_m(self) -> float:
        radii: List[float] = []
        for profile in self.navmesh_agent_profiles:
            r = self.agent_radius_m_by_profile.get(profile)
            if r is not None:
                radii.append(float(r))
        if not radii:
            return 0.28
        return max(radii)

    def build_and_eval(self, scene_id: str, scene_dir: str, seed: int, stage_usd: str | None = None) -> dict:
        if not os.path.isdir(scene_dir):
            raise SceneCompileError(
                code=ErrorCode.NAVMESH_FAIL,
                message=f"scene_dir 不存在，无法构建 navmesh: {scene_dir}",
                scene_id=scene_id,
            )

        resolved_stage = self._resolve_stage_usd(scene_dir=scene_dir, stage_usd=stage_usd)

        request_path = os.path.join(scene_dir, "navmesh_request.json")
        response_path = os.path.join(scene_dir, "navmesh_response.json")

        request_payload = {
            "scene_id": scene_id,
            "scene_dir": scene_dir,
            "stage_usd": resolved_stage,
            "seed": int(seed),
            "bake_timeout_s": int(self.bake_timeout_s),
            "cell_size_m": float(self.cell_size_m),
            "cell_height_m": float(self.cell_height_m),
            "max_slope_deg": float(self.max_slope_deg),
            "step_height_m": float(self.step_height_m),
            "agent_radius_m": float(self._pick_agent_radius_m()),
            # Conservative default for quadruped + humanoid mixed scenes.
            "agent_height_m": 1.0,
            "headless": bool(self.isaac_headless),
            "renderer": str(self.isaac_renderer),
        }

        with open(request_path, "w", encoding="utf-8") as f:
            json.dump(request_payload, f, ensure_ascii=False, indent=2)

        cmd = [
            sys.executable,
            "-m",
            "dataengine.src.scene_layer.navmesh_worker",
            "--request",
            request_path,
            "--response",
            response_path,
        ]

        try:
            completed = subprocess.run(  # noqa: S603
                cmd,
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
                timeout=max(1, int(self.bake_timeout_s) + 60),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SceneCompileError(
                code=ErrorCode.GPU_TIMEOUT,
                message=f"NavMesh 子进程超时({self.bake_timeout_s}s): {exc}",
                scene_id=scene_id,
            ) from exc

        if not os.path.isfile(response_path):
            stderr_tail = (completed.stderr or "").strip().splitlines()[-3:]
            stderr_summary = " | ".join(stderr_tail) if stderr_tail else "no stderr"
            raise SceneCompileError(
                code=ErrorCode.RUNTIME_FAIL,
                message=(
                    "NavMesh 子进程未生成 response 文件; "
                    f"returncode={completed.returncode}; stderr_tail={stderr_summary}"
                ),
                scene_id=scene_id,
            )

        with open(response_path, "r", encoding="utf-8") as f:
            response = json.load(f)

        if not response.get("ok", False):
            err_code = self._error_code_from_str(str(response.get("error_code", ErrorCode.RUNTIME_FAIL.value)))
            msg = str(response.get("message", "navmesh worker failed"))
            raise SceneCompileError(code=err_code, message=msg, scene_id=scene_id)

        return {
            "navmesh_file": str(response["navmesh_file"]),
            "metrics": dict(response["metrics"]),
        }
