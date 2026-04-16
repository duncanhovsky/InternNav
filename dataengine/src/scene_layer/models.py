from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SceneTemplate:
    """可用于生成场景的模板资产。"""

    template_id: str
    scene_type: str
    usd_path: str
    supports_complete_mode: bool = True
    supports_modular_mode: bool = False
    tags: List[str] = field(default_factory=list)


@dataclass
class SceneCandidate:
    """待编译场景候选。"""

    scene_id: str
    scene_type: str
    mode: str
    seed: int
    template: SceneTemplate
    params: Dict[str, float] = field(default_factory=dict)


@dataclass
class SceneMetrics:
    """场景层关键指标。"""

    path_count: int
    shortest_path_m: float
    second_shortest_path_m: float
    detour_margin_m: float
    free_space_ratio: float
    static_density: float
    static_complexity_score: float


@dataclass
class SceneCompileResult:
    """场景编译结果。"""

    scene_id: str
    scene_type: str
    mode: str
    seed: int
    status: str
    stage_usd: str
    navmesh_file: str
    metrics: SceneMetrics
    complexity_bucket: str
    reason: Optional[str] = None
    layout_hash: str = ""
    template_id: str = ""


@dataclass
class TrajectoryTask:
    """后续轨迹生成任务定义。"""

    task_id: str
    scene_id: str
    agent_type: str
    episode_idx: int
    global_seed: int
    scene_seed: int
    complexity_bucket: str
    scene_type: str
    mode: str
