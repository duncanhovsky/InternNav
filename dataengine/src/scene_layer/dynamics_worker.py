from __future__ import annotations

import argparse
import json
import os
import random
import traceback
from typing import Dict, List


def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _pick_relpath(
    rng: random.Random,
    asset_root: str,
    candidates: List[str],
    fallback: str,
    strict: bool,
) -> str:
    options = [x.strip() for x in candidates if x.strip()]
    if not options:
        if strict:
            raise ValueError("asset candidate list is empty under strict mode")
        return fallback

    idx = rng.randrange(0, len(options))
    relpath = options[idx]
    if not strict:
        return relpath

    full = _asset_abspath(asset_root=asset_root, asset_relpath=relpath, strict=True)
    if not full:
        raise ValueError(f"asset not found under strict mode: {relpath}")
    return relpath


def _to_prim_token(value: str) -> str:
    out = []
    for ch in value:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    token = "".join(out).strip("_")
    return token or "actor"


def _asset_abspath(asset_root: str, asset_relpath: str, strict: bool) -> str:
    if not asset_relpath:
        return ""

    candidates = [asset_relpath]
    if not asset_relpath.startswith("Isaac/"):
        # Backward compatibility: historical configs used "People/..." and
        # "Props/..." while current asset_root points to Assets/Isaac/5.1.
        candidates.append(f"Isaac/{asset_relpath}")

    for rel in candidates:
        full = os.path.join(asset_root, rel)
        if os.path.isfile(full):
            return full

    if strict:
        raise ValueError(f"asset not found under strict mode: {asset_relpath}")
    return os.path.join(asset_root, asset_relpath)


def _usd_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _require_asset_exists(asset_root: str, relpath: str, label: str) -> None:
    if not relpath.strip():
        raise ValueError(f"{label} relpath is empty")
    _ = _asset_abspath(asset_root=asset_root, asset_relpath=relpath, strict=True)


def _validate_people_animation_binding(
    people_mode: str,
    animation_relpath: str,
    idle_relpaths: List[str],
    walk_relpaths: List[str],
) -> str:
    idle_set = {x.strip() for x in idle_relpaths if x.strip()}
    walk_set = {x.strip() for x in walk_relpaths if x.strip()}
    anim = animation_relpath.strip()

    if people_mode == "stand_idle":
        if anim not in idle_set:
            raise ValueError(
                "animation does not match behavior: "
                f"motion_mode=stand_idle requires one of idle animations, got={animation_relpath}"
            )
        return "idle"

    if people_mode == "walk":
        if anim not in walk_set:
            raise ValueError(
                "animation does not match behavior: "
                f"motion_mode=walk requires one of walk animations, got={animation_relpath}"
            )
        return "walk"

    return "none"


def _build_mode_timeline(
    rng: random.Random,
    steps: int,
    initial_mode: str,
    transition_map: Dict[str, Dict[str, float]],
    min_state_steps: int,
    max_state_steps: int,
) -> List[str]:
    if steps <= 0:
        return []

    timeline: List[str] = []
    mode = initial_mode
    while len(timeline) < steps:
        block = rng.randint(max(1, min_state_steps), max(1, max_state_steps))
        for _ in range(block):
            if len(timeline) >= steps:
                break
            timeline.append(mode)

        transitions = transition_map.get(mode, {})
        next_modes = [k for k, v in transitions.items() if v > 0.0]
        if not next_modes:
            continue
        weights = [float(transitions[k]) for k in next_modes]
        mode = str(rng.choices(next_modes, weights=weights, k=1)[0])

    return timeline


def _write_dynamic_overlay_fallback(
    stage_usd: str,
    overlay_path: str,
    actors: List[Dict],
    asset_root: str,
    strict: bool,
) -> str:
    if not stage_usd or not os.path.isfile(stage_usd):
        raise ValueError(f"base stage_usd not found: {stage_usd}")

    lines: List[str] = [
        "#usda 1.0",
        "(",
        f'    subLayers = [@{stage_usd}@]',
        ")",
        "",
        'def Xform "World"',
        "{",
        '    def Xform "Dynamics"',
        "    {",
    ]

    for actor in actors:
        asset_relpath = str(actor.get("asset_relpath", ""))
        if not asset_relpath:
            continue

        asset_full = _asset_abspath(asset_root=asset_root, asset_relpath=asset_relpath, strict=strict)
        if not asset_full:
            continue

        token = _to_prim_token(str(actor.get("object_id", "actor")))
        pos = actor.get("position_xyz", [0.0, 0.0, 0.0])
        yaw = float(actor.get("yaw_deg", 0.0))
        object_id = _usd_escape(str(actor.get("object_id", "")))
        category = _usd_escape(str(actor.get("category", "")))
        motion_mode = _usd_escape(str(actor.get("motion_mode", "")))
        asset_rel = _usd_escape(asset_relpath)
        anim_rel = _usd_escape(str(actor.get("animation_relpath", "")))
        anim_behavior = _usd_escape(str(actor.get("animation_behavior", "none")))

        lines.extend(
            [
                f'        def Xform "{token}"',
                "        {",
                f'            custom string dynamic:object_id = "{object_id}"',
                f'            custom string dynamic:category = "{category}"',
                f'            custom string dynamic:motion_mode = "{motion_mode}"',
                f'            custom string dynamic:asset_relpath = "{asset_rel}"',
                '            custom string dynamic:source_backend = "asset_driven"',
                f'            custom string dynamic:animation_relpath = "{anim_rel}"',
                f'            custom string dynamic:animation_behavior = "{anim_behavior}"',
                (
                    "            double3 xformOp:translate:dynamic = "
                    f"({float(pos[0])}, {float(pos[1])}, {float(pos[2])})"
                ),
                f"            double3 xformOp:rotateXYZ:dynamic = (0, 0, {yaw})",
                '            uniform token[] xformOpOrder = ["xformOp:translate:dynamic", "xformOp:rotateXYZ:dynamic"]',
                f'            def Xform "Asset" (references = @{asset_full}@)',
                "            {",
                "            }",
                "        }",
            ]
        )

    lines.extend(
        [
            "    }",
            "}",
            "",
        ]
    )

    with open(overlay_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return overlay_path


def _write_dynamic_overlay(
    stage_usd: str,
    overlay_path: str,
    actors: List[Dict],
    asset_root: str,
    strict: bool,
) -> str:
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return _write_dynamic_overlay_fallback(
            stage_usd=stage_usd,
            overlay_path=overlay_path,
            actors=actors,
            asset_root=asset_root,
            strict=strict,
        )

    if not stage_usd or not os.path.isfile(stage_usd):
        raise ValueError(f"base stage_usd not found: {stage_usd}")

    stage = Usd.Stage.CreateNew(overlay_path)
    if stage is None:
        raise RuntimeError(f"failed to create overlay stage: {overlay_path}")

    root_layer = stage.GetRootLayer()
    if root_layer is None:
        raise RuntimeError("overlay root layer missing")

    root_layer.subLayerPaths = [stage_usd]

    world = stage.DefinePrim("/World", "Xform")
    dynamics_root = stage.DefinePrim("/World/Dynamics", "Xform")
    if world is None or dynamics_root is None:
        raise RuntimeError("failed to define dynamics root prims")

    for actor in actors:
        asset_relpath = str(actor.get("asset_relpath", ""))
        if not asset_relpath:
            continue
        asset_full = _asset_abspath(asset_root=asset_root, asset_relpath=asset_relpath, strict=strict)
        if not asset_full and strict:
            raise ValueError(f"asset path resolution failed: {asset_relpath}")
        if not asset_full:
            continue

        token = _to_prim_token(str(actor.get("object_id", "actor")))
        anchor_path = Sdf.Path(f"/World/Dynamics/{token}")
        anchor_prim = stage.DefinePrim(anchor_path, "Xform")
        ref_prim = stage.DefinePrim(anchor_path.AppendChild("Asset"), "Xform")
        ref_prim.GetReferences().AddReference(asset_full)

        xformable = UsdGeom.Xformable(anchor_prim)
        pos = actor.get("position_xyz", [0.0, 0.0, 0.0])
        yaw = float(actor.get("yaw_deg", 0.0))
        xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "dynamic").Set(
            Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2]))
        )
        xformable.AddRotateXYZOp(UsdGeom.XformOp.PrecisionDouble, "dynamic").Set(
            Gf.Vec3d(0.0, 0.0, yaw)
        )

        anchor_prim.CreateAttribute("dynamic:object_id", Sdf.ValueTypeNames.String).Set(str(actor.get("object_id", "")))
        anchor_prim.CreateAttribute("dynamic:category", Sdf.ValueTypeNames.String).Set(str(actor.get("category", "")))
        anchor_prim.CreateAttribute("dynamic:motion_mode", Sdf.ValueTypeNames.String).Set(str(actor.get("motion_mode", "")))
        anchor_prim.CreateAttribute("dynamic:asset_relpath", Sdf.ValueTypeNames.String).Set(asset_relpath)
        anchor_prim.CreateAttribute("dynamic:source_backend", Sdf.ValueTypeNames.String).Set("asset_driven")
        anchor_prim.CreateAttribute("dynamic:animation_relpath", Sdf.ValueTypeNames.String).Set(
            str(actor.get("animation_relpath", ""))
        )
        anchor_prim.CreateAttribute("dynamic:animation_behavior", Sdf.ValueTypeNames.String).Set(
            str(actor.get("animation_behavior", "none"))
        )

    root_layer.Save()
    return overlay_path


def _generate_tracks(request: Dict) -> Dict:
    scene_id = str(request["scene_id"])
    seed = int(request["seed"])
    track_path = str(request["track_path"])
    behavior_event_path = str(request.get("behavior_event_path", "")).strip()
    overlay_path = str(request.get("overlay_path", "")).strip()
    stage_usd = str(request.get("stage_usd", "")).strip()
    steps = int(request.get("steps", 20))
    dt_s = float(request.get("dt_s", 0.1))
    n_people = int(request.get("num_people", 1))
    n_objects = int(request.get("num_objects", 1))

    backend = str(request.get("backend", "synthetic"))
    asset_root = str(request.get("asset_root", "")).strip()
    asset_strict = bool(request.get("asset_strict", False))
    people_character_relpaths = list(request.get("people_character_relpaths", []))
    people_idle_animation_relpaths = list(request.get("people_idle_animation_relpaths", []))
    people_walk_animation_relpaths = list(request.get("people_walk_animation_relpaths", []))
    vehicle_relpaths = list(request.get("vehicle_relpaths", []))
    require_animation_binding = bool(request.get("require_animation_binding", True))
    people_idle_ratio = float(request.get("people_idle_ratio", 0.35))
    vehicle_parked_ratio = float(request.get("vehicle_parked_ratio", 0.40))
    ira_transition_map = request.get(
        "ira_transition_map",
        {
            "stand_idle": {"stand_idle": 0.5, "walk": 0.5},
            "walk": {"walk": 0.6, "stand_idle": 0.4},
        },
    )
    ira_min_state_steps = int(request.get("ira_min_state_steps", 2))
    ira_max_state_steps = int(request.get("ira_max_state_steps", 5))

    rng = random.Random(seed + 701)
    rows: List[Dict] = []
    events: List[Dict] = []
    actor_specs: List[Dict] = []
    ts0 = int(seed) * 1_000_000

    def _emit_actor(
        object_id: str,
        category: str,
        speed_mode: str,
        speed_modes_timeline: List[str],
        asset_relpath: str,
        animation_relpath: str,
        animation_behavior: str,
    ) -> int:
        x = rng.uniform(-4.0, 4.0)
        y = rng.uniform(-4.0, 4.0)
        if speed_mode in {"stand_idle", "parked"}:
            vx = 0.0
            vy = 0.0
        elif category == "people":
            vx = rng.uniform(-0.35, 0.35)
            vy = rng.uniform(-0.35, 0.35)
        else:
            vx = rng.uniform(-1.0, 1.0)
            vy = rng.uniform(-1.0, 1.0)
        samples = 0
        first_yaw = 0.0
        prev_mode = ""
        for step in range(steps):
            current_mode = speed_modes_timeline[step] if speed_modes_timeline else speed_mode
            if current_mode in {"stand_idle", "parked"}:
                vx_now = 0.0
                vy_now = 0.0
            elif category == "people":
                vx_now = vx
                vy_now = vy
            else:
                vx_now = vx
                vy_now = vy

            t_ns = ts0 + int(step * dt_s * 1e9)
            px = x + vx_now * step * dt_s
            py = y + vy_now * step * dt_s
            yaw = (rng.uniform(-180.0, 180.0) + step * 2.0) % 360.0
            if yaw >= 180.0:
                yaw -= 360.0
            if step == 0:
                first_yaw = yaw

            if prev_mode != current_mode:
                events.append(
                    {
                        "schema_version": "v1alpha",
                        "contract": "dynamic_events.v1alpha",
                        "scene_id": scene_id,
                        "object_id": object_id,
                        "category": category,
                        "timestamp_ns": int(t_ns),
                        "event_type": "state_enter",
                        "from_motion_mode": prev_mode or "",
                        "to_motion_mode": current_mode,
                        "command_name": "Idle" if current_mode == "stand_idle" else ("GoTo" if current_mode == "walk" else current_mode),
                        "animation_behavior": animation_behavior,
                        "source_backend": backend,
                    }
                )
                prev_mode = current_mode

            rows.append(
                {
                    "schema_version": "v1alpha",
                    "contract": "dynamic_tracks.v1alpha",
                    "scene_id": scene_id,
                    "object_id": object_id,
                    "category": category,
                    "timestamp_ns": int(t_ns),
                    "position_xyz": [round(px, 4), round(py, 4), 0.0],
                    "velocity_xyz": [round(vx_now, 4), round(vy_now, 4), 0.0],
                    "bbox_xyz": [0.6, 0.6, 1.7] if category == "people" else [0.8, 0.8, 1.2],
                    "yaw_deg": round(yaw, 4),
                    "source_backend": backend,
                    "motion_mode": current_mode,
                    "asset_relpath": asset_relpath,
                    "animation_relpath": animation_relpath,
                    "animation_behavior": animation_behavior,
                }
            )
            samples += 1

        actor_specs.append(
            {
                "object_id": object_id,
                "category": category,
                "motion_mode": speed_mode,
                "asset_relpath": asset_relpath,
                "animation_relpath": animation_relpath,
                "animation_behavior": animation_behavior,
                "position_xyz": [round(x, 4), round(y, 4), 0.0],
                "yaw_deg": round(first_yaw, 4),
            }
        )
        return samples

    object_count = 0
    sample_count = 0

    def _pick_people_mode() -> str:
        return "stand_idle" if rng.random() < people_idle_ratio else "walk"

    def _pick_vehicle_mode() -> str:
        return "parked" if rng.random() < vehicle_parked_ratio else "patrol"

    for idx in range(max(0, n_people)):
        if backend in {"asset_driven", "ira_character_graph"}:
            people_mode = _pick_people_mode()
            char_relpath = _pick_relpath(
                rng,
                asset_root,
                people_character_relpaths,
                fallback="Isaac/People/Characters/F_Business_02/F_Business_02.usd",
                strict=asset_strict,
            )
            if people_mode == "stand_idle":
                anim_relpath = _pick_relpath(
                    rng,
                    asset_root,
                    people_idle_animation_relpaths,
                    fallback="Isaac/People/Animations/stand_idle_loop.skelanim.usd",
                    strict=asset_strict,
                )
            else:
                anim_relpath = _pick_relpath(
                    rng,
                    asset_root,
                    people_walk_animation_relpaths,
                    fallback="Isaac/People/Animations/stand_walk_loop.skelanim.usd",
                    strict=asset_strict,
                )

            if require_animation_binding:
                _require_asset_exists(asset_root, char_relpath, label="people character")
                _require_asset_exists(asset_root, anim_relpath, label="people animation")
                anim_behavior = _validate_people_animation_binding(
                    people_mode=people_mode,
                    animation_relpath=anim_relpath,
                    idle_relpaths=people_idle_animation_relpaths,
                    walk_relpaths=people_walk_animation_relpaths,
                )
            else:
                anim_behavior = "none"

            people_timeline: List[str] = []
            if backend == "ira_character_graph":
                people_timeline = _build_mode_timeline(
                    rng=rng,
                    steps=steps,
                    initial_mode=people_mode,
                    transition_map=ira_transition_map,
                    min_state_steps=ira_min_state_steps,
                    max_state_steps=ira_max_state_steps,
                )
        else:
            people_mode = "synthetic"
            char_relpath = ""
            anim_relpath = ""
            anim_behavior = "none"
            people_timeline = []

        object_count += 1
        sample_count += _emit_actor(
            object_id=f"people_{idx:03d}",
            category="people",
            speed_mode=people_mode,
            speed_modes_timeline=people_timeline,
            asset_relpath=char_relpath,
            animation_relpath=anim_relpath,
            animation_behavior=anim_behavior,
        )

    for idx in range(max(0, n_objects)):
        if backend in {"asset_driven", "ira_character_graph"}:
            vehicle_mode = _pick_vehicle_mode()
            vehicle_relpath = _pick_relpath(
                rng,
                asset_root,
                vehicle_relpaths,
                fallback="Isaac/Props/Forklift/forklift.usd",
                strict=asset_strict,
            )
            if require_animation_binding:
                _require_asset_exists(asset_root, vehicle_relpath, label="vehicle asset")
        else:
            vehicle_mode = "synthetic"
            vehicle_relpath = ""

        object_timeline: List[str] = []
        if backend == "ira_character_graph":
            object_transition_map = {
                "parked": {"parked": 0.7, "patrol": 0.3},
                "patrol": {"patrol": 0.65, "parked": 0.35},
            }
            object_timeline = _build_mode_timeline(
                rng=rng,
                steps=steps,
                initial_mode=vehicle_mode,
                transition_map=object_transition_map,
                min_state_steps=ira_min_state_steps,
                max_state_steps=ira_max_state_steps,
            )

        object_count += 1
        sample_count += _emit_actor(
            object_id=f"object_{idx:03d}",
            category="object",
            speed_mode=vehicle_mode,
            speed_modes_timeline=object_timeline,
            asset_relpath=vehicle_relpath,
            animation_relpath="",
            animation_behavior="none",
        )

    _write_jsonl(track_path, rows)
    if backend == "ira_character_graph" and behavior_event_path:
        _write_jsonl(behavior_event_path, events)

    overlay_usd = ""
    if backend in {"asset_driven", "ira_character_graph"} and overlay_path and stage_usd:
        try:
            overlay_usd = _write_dynamic_overlay(
                stage_usd=stage_usd,
                overlay_path=overlay_path,
                actors=actor_specs,
                asset_root=asset_root,
                strict=asset_strict,
            )
        except Exception:
            if asset_strict:
                raise
            overlay_usd = ""

    return {
        "ok": True,
        "backend": backend,
        "track_file": track_path,
        "behavior_event_file": behavior_event_path if backend == "ira_character_graph" else "",
        "overlay_usd": overlay_usd,
        "object_count": int(object_count),
        "sample_count": int(sample_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scene dynamics worker")
    parser.add_argument("--request", type=str, required=True)
    parser.add_argument("--response", type=str, required=True)
    args = parser.parse_args()

    try:
        request = _read_json(args.request)
        payload = _generate_tracks(request)
    except Exception as exc:  # noqa: BLE001
        payload = {
            "ok": False,
            "error_code": "DYNAMICS_FAIL",
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    _write_json(args.response, payload)


if __name__ == "__main__":
    main()
