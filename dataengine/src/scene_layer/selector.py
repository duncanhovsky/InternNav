from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

from .models import SceneTemplate
from .utils import normalize_weights


@dataclass
class WeightedSceneSelector:
    """按场景类型权重与模式权重采样模板。"""

    scene_type_weights: Dict[str, float]
    scene_mode_weights: Dict[str, float]
    seed: int

    def __post_init__(self) -> None:
        """初始化确定性随机采样器与归一化权重。"""
        self._rng = random.Random(self.seed)
        self._scene_type_weights = normalize_weights(self.scene_type_weights)
        self._scene_mode_weights = normalize_weights(self.scene_mode_weights)

    def _sample_key(self, weighted: Dict[str, float]) -> str:
        """从离散权重分布采样一个键。"""
        keys = list(weighted.keys())
        probs = list(weighted.values())
        return self._rng.choices(keys, weights=probs, k=1)[0]

    def sample(self, templates: Dict[str, List[SceneTemplate]]) -> tuple[str, SceneTemplate]:
        """返回 (scene_mode, template)。"""
        for _ in range(100):
            scene_type = self._sample_key(self._scene_type_weights)
            mode = self._sample_key(self._scene_mode_weights)
            candidates = templates.get(scene_type, [])
            if mode == "complete":
                candidates = [x for x in candidates if x.supports_complete_mode]
            else:
                candidates = [x for x in candidates if x.supports_modular_mode]
            if len(candidates) == 0:
                continue
            return mode, self._rng.choice(candidates)
        raise RuntimeError("无法采样到合法模板，请检查资产与权重配置")
