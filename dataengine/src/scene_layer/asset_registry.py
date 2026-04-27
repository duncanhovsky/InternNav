from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .models import SceneTemplate


@dataclass
class AssetRegistry:
    """资产注册器。

    说明：
    1. 优先使用已知高质量完整场景模板。
    2. 模块化场景通过模板标记 supports_modular_mode=True。
    """

    asset_root: str

    def _all_exist(self, relpaths: List[str]) -> bool:
        for rel in relpaths:
            if not os.path.isfile(os.path.join(self.asset_root, rel)):
                return False
        return True

    def build_registry(self) -> Dict[str, List[SceneTemplate]]:
        templates: Dict[str, List[SceneTemplate]] = {
            "warehouse": [],
            "hospital": [],
            "outdoor": [],
        }

        # 已知模板：若路径不存在，不报错，但不会纳入候选池。
        # required_relpaths 用于过滤“模板文件存在，但关键依赖缺失”的情况。
        known: List[Tuple[str, str, str, bool, bool, List[str], List[str]]] = [
            (
                "warehouse_digital_twin_small",
                "warehouse",
                "Isaac/Environments/Digital_Twin_Warehouse/small_warehouse_digital_twin.usd",
                True,
                False,
                ["complete", "warehouse", "indoor"],
                [],
            ),
            (
                "warehouse_modular",
                "warehouse",
                "Isaac/Environments/Modular_Warehouse/Props/warehouse_h10m_center.usd",
                False,
                True,
                ["modular", "warehouse", "indoor"],
                [],
            ),
            (
                "hospital_complete",
                "hospital",
                "Isaac/Environments/Hospital/hospital.usd",
                True,
                False,
                ["complete", "hospital", "indoor"],
                [],
            ),
            (
                "office_complete",
                "hospital",
                "Isaac/Environments/Office/office.usd",
                True,
                False,
                ["complete", "office", "indoor"],
                [],
            ),
            (
                "rivermark_outdoor",
                "outdoor",
                "Isaac/Environments/Outdoor/Rivermark/rivermark.usd",
                True,
                True,
                ["complete", "modular", "outdoor"],
                [
                    "Isaac/Environments/Outdoor/Rivermark/dsready_content/nv_content/common_assets/props_general/patio_chair_01/patio_chair_01.usd",
                    "Isaac/Environments/Outdoor/Rivermark/dsready_content/nv_content/common_assets/props_vegetation/hedge_round_01/hedge_round_01.usd",
                ],
            ),
        ]

        for tid, scene_type, rel_path, complete_mode, modular_mode, tags, required_relpaths in known:
            abs_path = os.path.join(self.asset_root, rel_path)
            if os.path.isfile(abs_path):
                if required_relpaths and not self._all_exist(required_relpaths):
                    continue
                templates[scene_type].append(
                    SceneTemplate(
                        template_id=tid,
                        scene_type=scene_type,
                        usd_path=abs_path,
                        supports_complete_mode=complete_mode,
                        supports_modular_mode=modular_mode,
                        tags=tags,
                    )
                )

        return templates
