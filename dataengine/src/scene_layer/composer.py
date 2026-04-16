from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict

from .errors import ErrorCode, SceneCompileError
from .models import SceneCandidate
from .utils import atomic_write_text, stable_hash


@dataclass
class SceneComposer:
    """场景组合器（轻量实现）。

    说明：
    1. 这里不直接依赖 Isaac Python API，保持离线可测。
    2. 输出 stage 描述文件（JSON）+ 逻辑 stage_usd 路径。
    3. 真正加载 USD 并启动仿真可在上层 runner 中对接 Isaac Sim。
    """

    usd_output_root: str

    def compose(self, candidate: SceneCandidate) -> Dict[str, str]:
        if not os.path.isfile(candidate.template.usd_path):
            raise SceneCompileError(
                code=ErrorCode.ASSET_MISSING,
                message=f"模板 USD 不存在: {candidate.template.usd_path}",
                scene_id=candidate.scene_id,
            )

        rng = random.Random(candidate.seed)
        layout_variant = {
            "obstacle_density": round(rng.uniform(0.2, 0.6), 3),
            "aisle_width_m": round(rng.uniform(1.0, 2.8), 3),
            "door_open_ratio": round(rng.uniform(0.4, 1.0), 3),
            "light_intensity_scale": round(rng.uniform(0.7, 1.3), 3),
        }

        # 模块化模式额外写入结构参数，后续可驱动真实拼装器。
        if candidate.mode == "modular":
            layout_variant.update(
                {
                    "module_rows": int(rng.randint(2, 6)),
                    "module_cols": int(rng.randint(2, 8)),
                    "turn_probability": round(rng.uniform(0.1, 0.5), 3),
                }
            )

        scene_dir = os.path.join(self.usd_output_root, candidate.scene_id)
        os.makedirs(scene_dir, exist_ok=True)

        # 使用“引用模板 + 参数配置”的形式表达 stage，便于审计和复现。
        stage_spec = {
            "scene_id": candidate.scene_id,
            "scene_type": candidate.scene_type,
            "mode": candidate.mode,
            "seed": candidate.seed,
            "template_id": candidate.template.template_id,
            "template_usd": candidate.template.usd_path,
            "layout_variant": layout_variant,
        }

        spec_path = os.path.join(scene_dir, "stage_spec.json")
        import json

        atomic_write_text(spec_path, json.dumps(stage_spec, ensure_ascii=False, indent=2))

        stage_usd = candidate.template.usd_path
        layout_hash = stable_hash(json.dumps(stage_spec, ensure_ascii=False, sort_keys=True))

        return {
            "scene_dir": scene_dir,
            "stage_usd": stage_usd,
            "stage_spec_path": spec_path,
            "layout_hash": layout_hash,
        }
