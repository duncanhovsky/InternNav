from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List

from .errors import ErrorCode, SceneCompileError


@dataclass
class DynamicsLayer:
    enable_dynamics: bool
    backend: str
    timeout_s: int
    steps: int
    dt_s: float
    num_people_min: int
    num_people_max: int
    num_objects_min: int
    num_objects_max: int
    asset_root: str = ""
    asset_strict: bool = False
    people_character_relpaths: List[str] | None = None
    people_idle_animation_relpaths: List[str] | None = None
    people_walk_animation_relpaths: List[str] | None = None
    vehicle_relpaths: List[str] | None = None
    require_animation_binding: bool = True
    people_idle_ratio: float = 0.35
    vehicle_parked_ratio: float = 0.40

    def _pick_counts(self, seed: int) -> Dict[str, int]:
        # Keep deterministic counts for same seed.
        span_people = max(0, self.num_people_max - self.num_people_min)
        span_objects = max(0, self.num_objects_max - self.num_objects_min)
        n_people = self.num_people_min + (seed % (span_people + 1 if span_people >= 0 else 1))
        n_objects = self.num_objects_min + ((seed + 13) % (span_objects + 1 if span_objects >= 0 else 1))
        return {"num_people": int(n_people), "num_objects": int(n_objects)}

    @staticmethod
    def _error_code_from_str(code: str) -> ErrorCode:
        try:
            return ErrorCode(code)
        except ValueError:
            return ErrorCode.RUNTIME_FAIL

    def generate(self, scene_id: str, scene_dir: str, seed: int, stage_usd: str = "") -> Dict:
        if not self.enable_dynamics:
            return {
                "enabled": False,
                "backend": self.backend,
                "track_file": "",
                "behavior_event_file": "",
                "overlay_usd": "",
                "object_count": 0,
                "sample_count": 0,
            }

        os.makedirs(scene_dir, exist_ok=True)
        request_path = os.path.join(scene_dir, "dynamics_request.json")
        response_path = os.path.join(scene_dir, "dynamics_response.json")
        track_file = os.path.join(scene_dir, "dynamic_tracks.jsonl")
        behavior_event_file = os.path.join(scene_dir, "behavior_events.jsonl")
        overlay_usd = os.path.join(scene_dir, "dynamic_overlay.usda")

        counts = self._pick_counts(seed=seed)
        payload = {
            "scene_id": scene_id,
            "seed": int(seed),
            "backend": self.backend,
            "steps": int(self.steps),
            "dt_s": float(self.dt_s),
            "track_path": track_file,
            "behavior_event_path": behavior_event_file,
            "overlay_path": overlay_usd,
            "stage_usd": str(stage_usd or ""),
            "asset_root": self.asset_root,
            "asset_strict": bool(self.asset_strict),
            "people_character_relpaths": list(self.people_character_relpaths or []),
            "people_idle_animation_relpaths": list(self.people_idle_animation_relpaths or []),
            "people_walk_animation_relpaths": list(self.people_walk_animation_relpaths or []),
            "vehicle_relpaths": list(self.vehicle_relpaths or []),
            "require_animation_binding": bool(self.require_animation_binding),
            "people_idle_ratio": float(self.people_idle_ratio),
            "vehicle_parked_ratio": float(self.vehicle_parked_ratio),
            **counts,
        }
        with open(request_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        cmd = [
            sys.executable,
            "-m",
            "dataengine.src.scene_layer.dynamics_worker",
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
                timeout=max(1, int(self.timeout_s)),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SceneCompileError(
                code=ErrorCode.GPU_TIMEOUT,
                message=f"Dynamics worker timeout({self.timeout_s}s): {exc}",
                scene_id=scene_id,
            ) from exc

        if not os.path.isfile(response_path):
            stderr_tail = (completed.stderr or "").strip().splitlines()[-3:]
            stderr_summary = " | ".join(stderr_tail) if stderr_tail else "no stderr"
            raise SceneCompileError(
                code=ErrorCode.RUNTIME_FAIL,
                message=(
                    "Dynamics worker missing response file; "
                    f"returncode={completed.returncode}; stderr_tail={stderr_summary}"
                ),
                scene_id=scene_id,
            )

        with open(response_path, "r", encoding="utf-8") as f:
            response = json.load(f)

        if not response.get("ok", False):
            err_code = self._error_code_from_str(str(response.get("error_code", ErrorCode.RUNTIME_FAIL.value)))
            msg = str(response.get("message", "dynamics worker failed"))
            raise SceneCompileError(code=err_code, message=msg, scene_id=scene_id)

        return {
            "enabled": True,
            "backend": str(response.get("backend", self.backend)),
            "track_file": str(response.get("track_file", track_file)),
            "behavior_event_file": str(response.get("behavior_event_file", "")),
            "overlay_usd": str(response.get("overlay_usd", "")),
            "object_count": int(response.get("object_count", 0)),
            "sample_count": int(response.get("sample_count", 0)),
        }
