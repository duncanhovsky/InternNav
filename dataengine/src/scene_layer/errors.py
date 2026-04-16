from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ErrorCode(str, Enum):
    """场景编译错误码。"""

    ASSET_MISSING = "ASSET_MISSING"
    USD_COMPOSE_FAIL = "USD_COMPOSE_FAIL"
    NAVMESH_FAIL = "NAVMESH_FAIL"
    REACHABILITY_FAIL = "REACHABILITY_FAIL"
    METRIC_REJECT = "METRIC_REJECT"
    RUNTIME_FAIL = "RUNTIME_FAIL"
    GPU_TIMEOUT = "GPU_TIMEOUT"
    IO_TRANSIENT = "IO_TRANSIENT"


@dataclass
class SceneCompileError(Exception):
    """场景编译异常，携带标准化错误码。"""

    code: ErrorCode
    message: str
    scene_id: Optional[str] = None

    def __str__(self) -> str:
        sid = f" scene_id={self.scene_id}" if self.scene_id else ""
        return f"[{self.code}] {self.message}{sid}"
