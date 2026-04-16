from __future__ import annotations

import os
import random
from dataclasses import dataclass

from .errors import ErrorCode, SceneCompileError


@dataclass
class NavMeshBuilder:
    """NavMesh 构建与可达性估计（可替换为 Isaac 实现）。

    当前实现提供确定性“伪构建”用于流程联调。
    后续接入 Isaac API 时，仅需替换 build_and_eval 方法内部逻辑。
    """

    bake_timeout_s: int

    def build_and_eval(self, scene_id: str, scene_dir: str, seed: int) -> dict:
        if not os.path.isdir(scene_dir):
            raise SceneCompileError(
                code=ErrorCode.NAVMESH_FAIL,
                message=f"scene_dir 不存在，无法构建 navmesh: {scene_dir}",
                scene_id=scene_id,
            )

        rng = random.Random(seed + 97)

        # 用固定概率模拟运行时构建失败，便于验证重试逻辑。
        if rng.random() < 0.02:
            raise SceneCompileError(
                code=ErrorCode.RUNTIME_FAIL,
                message="NavMesh 构建过程超时/运行时异常（模拟）",
                scene_id=scene_id,
            )

        navmesh_file = os.path.join(scene_dir, "navmesh.bin")
        with open(navmesh_file, "wb") as f:
            f.write(b"NAVMESH_PLACEHOLDER")

        shortest_path = rng.uniform(8.0, 40.0)
        second_shortest = shortest_path + rng.uniform(0.5, 8.0)
        detour_margin = second_shortest - shortest_path

        metrics = {
            "path_count": int(rng.randint(2, 7)),
            "shortest_path_m": float(round(shortest_path, 3)),
            "second_shortest_path_m": float(round(second_shortest, 3)),
            "detour_margin_m": float(round(detour_margin, 3)),
            "free_space_ratio": float(round(rng.uniform(0.15, 0.45), 3)),
            "static_density": float(round(rng.uniform(0.15, 0.6), 3)),
        }
        return {"navmesh_file": navmesh_file, "metrics": metrics}
