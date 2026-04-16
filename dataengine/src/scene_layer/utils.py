from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Any, Dict, Iterable, List


def ensure_parent_dir(file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


def atomic_write_text(file_path: str, content: str, encoding: str = "utf-8") -> None:
    ensure_parent_dir(file_path)
    fd, tmp_path = tempfile.mkstemp(prefix="tmp_", suffix=".tmp", dir=os.path.dirname(file_path))
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
        os.replace(tmp_path, file_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def append_jsonl(file_path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_parent_dir(file_path)
    with open(file_path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(file_path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def stable_hash(data: str) -> str:
    h = hashlib.sha1()
    h.update(data.encode("utf-8"))
    return h.hexdigest()


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    s = sum(weights.values())
    if s <= 0:
        raise ValueError(f"非法权重和: {s}")
    return {k: v / s for k, v in weights.items()}
