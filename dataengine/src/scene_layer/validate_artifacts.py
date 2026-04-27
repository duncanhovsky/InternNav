from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence

from .utils import load_jsonl


@dataclass
class ArtifactIssue:
    level: str
    code: str
    message: str
    scene_id: str = ""


@dataclass
class SceneValidationResult:
    scene_id: str
    scene_dir: str
    ok: bool
    issues: List[ArtifactIssue] = field(default_factory=list)


@dataclass
class ValidationSummary:
    ok: bool
    total: int
    passed: int
    failed: int
    warnings: int
    strict_warn: bool
    results: List[SceneValidationResult] = field(default_factory=list)


def _is_navmesh_debug_name(name: str) -> bool:
    return name == "navmesh_debug.json" or name.startswith("navmesh_debug_")


def _scene_id_from_dir(scene_dir: str) -> str:
    return os.path.basename(os.path.normpath(scene_dir))


def _load_jsonl_rows(path: str) -> List[Dict]:
    if not os.path.isfile(path):
        return []
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _validate_dynamic_tracks(track_path: str, scene_id: str, issues: List[ArtifactIssue]) -> None:
    rows = _load_jsonl_rows(track_path)
    if len(rows) == 0:
        issues.append(
            ArtifactIssue(
                level="ERROR",
                code="DYNAMIC_TRACKS_EMPTY",
                message="dynamic tracks file is empty",
                scene_id=scene_id,
            )
        )
        return

    required_keys = {
        "schema_version",
        "contract",
        "scene_id",
        "object_id",
        "category",
        "timestamp_ns",
        "position_xyz",
        "velocity_xyz",
        "bbox_xyz",
        "yaw_deg",
    }
    valid_people_modes = {"synthetic", "stand_idle", "walk"}
    valid_object_modes = {"synthetic", "parked", "patrol"}
    valid_animation_behaviors = {"none", "idle", "walk"}
    last_ts_by_obj: Dict[str, int] = {}
    for idx, row in enumerate(rows):
        missing = [k for k in required_keys if k not in row]
        if missing:
            issues.append(
                ArtifactIssue(
                    level="ERROR",
                    code="DYNAMIC_TRACKS_SCHEMA_MISSING",
                    message=f"row={idx} missing keys: {missing}",
                    scene_id=scene_id,
                )
            )
            continue

        if row.get("scene_id") != scene_id:
            issues.append(
                ArtifactIssue(
                    level="ERROR",
                    code="DYNAMIC_TRACKS_SCENE_ID_MISMATCH",
                    message=f"row={idx} scene_id mismatch: {row.get('scene_id')}",
                    scene_id=scene_id,
                )
            )

        oid = str(row.get("object_id"))
        category = str(row.get("category"))
        motion_mode = str(row.get("motion_mode", "synthetic"))
        source_backend = str(row.get("source_backend", "synthetic"))
        animation_relpath = str(row.get("animation_relpath", "")).strip()
        animation_behavior = str(row.get("animation_behavior", "none")).strip()

        if category == "people" and motion_mode not in valid_people_modes:
            issues.append(
                ArtifactIssue(
                    level="ERROR",
                    code="DYNAMIC_TRACKS_PEOPLE_MODE_INVALID",
                    message=f"row={idx} invalid people motion_mode={motion_mode}",
                    scene_id=scene_id,
                )
            )
        if category == "object" and motion_mode not in valid_object_modes:
            issues.append(
                ArtifactIssue(
                    level="ERROR",
                    code="DYNAMIC_TRACKS_OBJECT_MODE_INVALID",
                    message=f"row={idx} invalid object motion_mode={motion_mode}",
                    scene_id=scene_id,
                )
            )

        if source_backend == "asset_driven":
            if not str(row.get("asset_relpath", "")).strip():
                issues.append(
                    ArtifactIssue(
                        level="ERROR",
                        code="DYNAMIC_TRACKS_ASSET_MISSING",
                        message=f"row={idx} asset_driven row missing asset_relpath",
                        scene_id=scene_id,
                    )
                )
            if category == "people" and motion_mode in {"stand_idle", "walk"} and not animation_relpath:
                issues.append(
                    ArtifactIssue(
                        level="ERROR",
                        code="DYNAMIC_TRACKS_ANIM_MISSING",
                        message=(
                            f"row={idx} people asset_driven row missing animation_relpath "
                            f"for motion_mode={motion_mode}"
                        ),
                        scene_id=scene_id,
                    )
                )

            if animation_behavior not in valid_animation_behaviors:
                issues.append(
                    ArtifactIssue(
                        level="ERROR",
                        code="DYNAMIC_TRACKS_ANIM_BEHAVIOR_INVALID",
                        message=f"row={idx} invalid animation_behavior={animation_behavior}",
                        scene_id=scene_id,
                    )
                )

            if category == "people":
                if motion_mode == "stand_idle" and animation_behavior != "idle":
                    issues.append(
                        ArtifactIssue(
                            level="ERROR",
                            code="DYNAMIC_TRACKS_ANIM_BEHAVIOR_MISMATCH",
                            message=(
                                f"row={idx} motion_mode=stand_idle requires animation_behavior=idle, "
                                f"got={animation_behavior}"
                            ),
                            scene_id=scene_id,
                        )
                    )
                if motion_mode == "walk" and animation_behavior != "walk":
                    issues.append(
                        ArtifactIssue(
                            level="ERROR",
                            code="DYNAMIC_TRACKS_ANIM_BEHAVIOR_MISMATCH",
                            message=(
                                f"row={idx} motion_mode=walk requires animation_behavior=walk, "
                                f"got={animation_behavior}"
                            ),
                            scene_id=scene_id,
                        )
                    )

        ts = int(row.get("timestamp_ns"))
        prev = last_ts_by_obj.get(oid)
        if prev is not None and ts < prev:
            issues.append(
                ArtifactIssue(
                    level="ERROR",
                    code="DYNAMIC_TRACKS_TS_NOT_MONOTONIC",
                    message=f"object_id={oid} timestamp not monotonic: prev={prev}, now={ts}",
                    scene_id=scene_id,
                )
            )
        last_ts_by_obj[oid] = ts


def _validate_behavior_events(event_path: str, scene_id: str, issues: List[ArtifactIssue]) -> None:
    rows = _load_jsonl_rows(event_path)
    if len(rows) == 0:
        issues.append(
            ArtifactIssue(
                level="ERROR",
                code="DYNAMIC_EVENTS_EMPTY",
                message="behavior events file is empty",
                scene_id=scene_id,
            )
        )
        return

    required_keys = {
        "schema_version",
        "contract",
        "scene_id",
        "object_id",
        "category",
        "timestamp_ns",
        "event_type",
        "to_motion_mode",
        "source_backend",
    }
    last_ts_by_obj: Dict[str, int] = {}
    for idx, row in enumerate(rows):
        missing = [k for k in required_keys if k not in row]
        if missing:
            issues.append(
                ArtifactIssue(
                    level="ERROR",
                    code="DYNAMIC_EVENTS_SCHEMA_MISSING",
                    message=f"row={idx} missing keys: {missing}",
                    scene_id=scene_id,
                )
            )
            continue

        if row.get("contract") != "dynamic_events.v1alpha":
            issues.append(
                ArtifactIssue(
                    level="ERROR",
                    code="DYNAMIC_EVENTS_CONTRACT_INVALID",
                    message=f"row={idx} contract must be dynamic_events.v1alpha",
                    scene_id=scene_id,
                )
            )

        if row.get("scene_id") != scene_id:
            issues.append(
                ArtifactIssue(
                    level="ERROR",
                    code="DYNAMIC_EVENTS_SCENE_ID_MISMATCH",
                    message=f"row={idx} scene_id mismatch: {row.get('scene_id')}",
                    scene_id=scene_id,
                )
            )

        oid = str(row.get("object_id"))
        ts = int(row.get("timestamp_ns"))
        prev = last_ts_by_obj.get(oid)
        if prev is not None and ts < prev:
            issues.append(
                ArtifactIssue(
                    level="ERROR",
                    code="DYNAMIC_EVENTS_TS_NOT_MONOTONIC",
                    message=f"object_id={oid} timestamp not monotonic: prev={prev}, now={ts}",
                    scene_id=scene_id,
                )
            )
        last_ts_by_obj[oid] = ts


def _validate_scene_dir(
    scene_dir: str,
    scene_id: str = "",
    expect_dynamics: bool = False,
    expected_track_file: str = "",
    expected_behavior_event_file: str = "",
    expected_overlay_usd: str = "",
) -> SceneValidationResult:
    sid = scene_id or _scene_id_from_dir(scene_dir)
    issues: List[ArtifactIssue] = []

    if not os.path.isdir(scene_dir):
        issues.append(ArtifactIssue(level="ERROR", code="SCENE_DIR_MISSING", message=f"scene_dir not found: {scene_dir}", scene_id=sid))
        return SceneValidationResult(scene_id=sid, scene_dir=scene_dir, ok=False, issues=issues)

    names = sorted(os.listdir(scene_dir))
    stage_spec = os.path.join(scene_dir, "stage_spec.json")
    if not os.path.isfile(stage_spec):
        issues.append(ArtifactIssue(level="ERROR", code="STAGE_SPEC_MISSING", message="stage_spec.json not found", scene_id=sid))
    else:
        with open(stage_spec, "r", encoding="utf-8") as f:
            try:
                payload = json.load(f)
            except json.JSONDecodeError as exc:
                issues.append(ArtifactIssue(level="ERROR", code="STAGE_SPEC_INVALID_JSON", message=str(exc), scene_id=sid))
                payload = {}
        status = str(payload.get("status", "")).strip()
        backend = str(payload.get("backend", "")).strip()
        if backend == "isaac_replicator" and status not in {"COMPOSING", "DONE", "FAILED"}:
            issues.append(
                ArtifactIssue(
                    level="WARN",
                    code="STAGE_SPEC_STATUS_UNEXPECTED",
                    message=f"backend=isaac_replicator but status={status!r}",
                    scene_id=sid,
                )
            )

    navmesh_files = [x for x in names if x == "navmesh.bin" or x.startswith("navmesh_") and x.endswith(".bin")]
    if len(navmesh_files) == 0:
        issues.append(ArtifactIssue(level="ERROR", code="NAVMESH_BIN_MISSING", message="no navmesh bin file found", scene_id=sid))

    navmesh_debug_files = [x for x in names if _is_navmesh_debug_name(x) and x.endswith(".json")]
    if len(navmesh_debug_files) == 0:
        issues.append(
            ArtifactIssue(
                level="WARN",
                code="NAVMESH_DEBUG_MISSING",
                message="no navmesh debug json found (legacy/stub runs may omit this)",
                scene_id=sid,
            )
        )

    request_files = [x for x in names if x.startswith("navmesh_request") and x.endswith(".json")]
    response_files = [x for x in names if x.startswith("navmesh_response") and x.endswith(".json")]
    if len(request_files) > 0 and len(response_files) == 0:
        issues.append(
            ArtifactIssue(
                level="WARN",
                code="NAVMESH_RESPONSE_MISSING",
                message="navmesh requests found but no navmesh response json",
                scene_id=sid,
            )
        )

    dynamic_track_path = ""
    if expected_track_file:
        dynamic_track_path = expected_track_file
        if not os.path.isabs(dynamic_track_path):
            dynamic_track_path = os.path.join(os.getcwd(), dynamic_track_path)
    else:
        default_path = os.path.join(scene_dir, "dynamic_tracks.jsonl")
        if os.path.isfile(default_path):
            dynamic_track_path = default_path

    if expect_dynamics:
        if not dynamic_track_path or not os.path.isfile(dynamic_track_path):
            issues.append(
                ArtifactIssue(
                    level="ERROR",
                    code="DYNAMIC_TRACKS_MISSING",
                    message="dynamics expected but dynamic_tracks file is missing",
                    scene_id=sid,
                )
            )
        else:
            _validate_dynamic_tracks(track_path=dynamic_track_path, scene_id=sid, issues=issues)
    elif dynamic_track_path and os.path.isfile(dynamic_track_path):
        # If file exists, validate it even when not strictly expected.
        _validate_dynamic_tracks(track_path=dynamic_track_path, scene_id=sid, issues=issues)

    behavior_event_path = expected_behavior_event_file.strip()
    if behavior_event_path:
        if not os.path.isabs(behavior_event_path):
            behavior_event_path = os.path.join(os.getcwd(), behavior_event_path)
        if not os.path.isfile(behavior_event_path):
            issues.append(
                ArtifactIssue(
                    level="ERROR",
                    code="DYNAMIC_EVENTS_MISSING",
                    message=f"behavior events file not found: {expected_behavior_event_file}",
                    scene_id=sid,
                )
            )
        else:
            _validate_behavior_events(event_path=behavior_event_path, scene_id=sid, issues=issues)

    overlay_path = expected_overlay_usd.strip()
    if overlay_path:
        if not os.path.isabs(overlay_path):
            overlay_path = os.path.join(os.getcwd(), overlay_path)
        if not os.path.isfile(overlay_path):
            issues.append(
                ArtifactIssue(
                    level="ERROR",
                    code="DYNAMIC_OVERLAY_MISSING",
                    message=f"dynamic overlay usd not found: {expected_overlay_usd}",
                    scene_id=sid,
                )
            )

    ok = all(x.level != "ERROR" for x in issues)
    return SceneValidationResult(scene_id=sid, scene_dir=scene_dir, ok=ok, issues=issues)


def _collect_scene_dirs_from_manifest(scene_manifest: str, usd_output_root: str, only_done: bool) -> List[Dict[str, str]]:
    rows = load_jsonl(scene_manifest)
    out: List[Dict[str, str]] = []
    for row in rows:
        if only_done and row.get("status") != "DONE":
            continue
        sid = str(row.get("scene_id", "")).strip()
        if sid == "":
            continue
        dynamic = row.get("dynamic") if isinstance(row.get("dynamic"), dict) else {}
        out.append(
            {
                "scene_id": sid,
                "scene_dir": os.path.join(usd_output_root, sid),
                "expect_dynamics": bool(dynamic.get("enabled", False)),
                "expected_track_file": str(dynamic.get("track_file", "")),
                "expected_behavior_event_file": str(dynamic.get("behavior_event_file", "")),
                "expected_overlay_usd": str(dynamic.get("overlay_usd", "")),
            }
        )
    return out


def validate_many(scene_entries: Sequence[Dict[str, str]], strict_warn: bool) -> ValidationSummary:
    results: List[SceneValidationResult] = []
    for item in scene_entries:
        results.append(
            _validate_scene_dir(
                scene_dir=item["scene_dir"],
                scene_id=item.get("scene_id", ""),
                expect_dynamics=bool(item.get("expect_dynamics", False)),
                expected_track_file=str(item.get("expected_track_file", "")),
                expected_behavior_event_file=str(item.get("expected_behavior_event_file", "")),
                expected_overlay_usd=str(item.get("expected_overlay_usd", "")),
            )
        )

    passed = sum(1 for x in results if x.ok)
    failed = sum(1 for x in results if not x.ok)
    warnings = sum(1 for r in results for issue in r.issues if issue.level == "WARN")

    ok = failed == 0 and (warnings == 0 if strict_warn else True)
    return ValidationSummary(
        ok=ok,
        total=len(results),
        passed=passed,
        failed=failed,
        warnings=warnings,
        strict_warn=strict_warn,
        results=results,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate scene_layer artifacts consistency")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--scene-dir", type=str, default="", help="single scene directory to validate")
    src.add_argument("--scene-manifest", type=str, default="", help="scene manifest jsonl for batch validation")

    parser.add_argument("--usd-output-root", type=str, default="dataengine/out/scenes/usd", help="scene root for manifest mode")
    parser.add_argument("--only-done", action="store_true", help="manifest mode: validate only DONE scenes")
    parser.add_argument("--strict-warn", action="store_true", help="treat warnings as failure")
    parser.add_argument("--json", action="store_true", help="print json output")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.scene_dir:
        entries = [{"scene_id": _scene_id_from_dir(args.scene_dir), "scene_dir": args.scene_dir}]
    else:
        entries = _collect_scene_dirs_from_manifest(
            scene_manifest=args.scene_manifest,
            usd_output_root=args.usd_output_root,
            only_done=bool(args.only_done),
        )

    summary = validate_many(entries, strict_warn=bool(args.strict_warn))

    if args.json:
        payload = asdict(summary)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(
            f"ok={summary.ok} total={summary.total} passed={summary.passed} "
            f"failed={summary.failed} warnings={summary.warnings} strict_warn={summary.strict_warn}"
        )
        for result in summary.results:
            print(f"- scene_id={result.scene_id} ok={result.ok} dir={result.scene_dir}")
            for issue in result.issues:
                print(f"  [{issue.level}] {issue.code}: {issue.message}")

    if not summary.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
