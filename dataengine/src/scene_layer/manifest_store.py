from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set

from .utils import append_jsonl, load_jsonl


@dataclass
class ManifestStore:
    scene_manifest_path: str
    compile_log_path: str
    task_manifest_path: str

    def load_done_scene_ids(self) -> Set[str]:
        rows = load_jsonl(self.scene_manifest_path)
        return {str(x.get("scene_id")) for x in rows if x.get("status") == "DONE"}

    def append_scene_rows(self, rows: Iterable[Dict]) -> None:
        append_jsonl(self.scene_manifest_path, rows)

    def append_compile_logs(self, rows: Iterable[Dict]) -> None:
        append_jsonl(self.compile_log_path, rows)

    def append_task_rows(self, rows: Iterable[Dict]) -> None:
        append_jsonl(self.task_manifest_path, rows)

    def read_scene_rows(self) -> List[Dict]:
        return load_jsonl(self.scene_manifest_path)
