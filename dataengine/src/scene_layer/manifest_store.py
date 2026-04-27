from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set

from .utils import append_jsonl, load_jsonl


@dataclass
class ManifestStore:
    """场景层清单读写封装。

    统一管理三类 JSONL:
    1. scene_manifest: 场景编译结果（DONE/FAILED）。
    2. compile_log: 每次编译尝试日志。
    3. task_manifest: 下游轨迹生成任务。
    """

    scene_manifest_path: str
    compile_log_path: str
    task_manifest_path: str

    def load_done_scene_ids(self) -> Set[str]:
        """读取历史 DONE 场景，用于断点恢复与去重。"""
        rows = load_jsonl(self.scene_manifest_path)
        return {str(x.get("scene_id")) for x in rows if x.get("status") == "DONE"}

    def append_scene_rows(self, rows: Iterable[Dict]) -> None:
        """追加场景结果行。"""
        append_jsonl(self.scene_manifest_path, rows)

    def append_compile_logs(self, rows: Iterable[Dict]) -> None:
        """追加编译日志行。"""
        append_jsonl(self.compile_log_path, rows)

    def append_task_rows(self, rows: Iterable[Dict]) -> None:
        """追加轨迹任务行。"""
        append_jsonl(self.task_manifest_path, rows)

    def read_scene_rows(self) -> List[Dict]:
        """读取全部场景结果行。"""
        return load_jsonl(self.scene_manifest_path)
