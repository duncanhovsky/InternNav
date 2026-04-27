from __future__ import annotations

import argparse
import json
import math
import os
import random
import traceback
from typing import Dict, List, Sequence, Tuple


def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _set_if_has_attr(settings_obj, settings_cls, attr_name: str, value) -> None:
    path = getattr(settings_cls, attr_name, None)
    if not path:
        return
    settings_obj.set(path, value)


def _apply_navmesh_settings(request: Dict) -> None:
    import carb.settings  # type: ignore[import-not-found]
    from omni.anim.navigation.core import NavMeshSettings  # type: ignore[import-not-found]

    settings = carb.settings.get_settings()

    _set_if_has_attr(settings, NavMeshSettings, "AUTO_REBAKE_SETTING_PATH", False)
    _set_if_has_attr(settings, NavMeshSettings, "CACHE_ENABLED_SETTING_PATH", False)
    _set_if_has_attr(settings, NavMeshSettings, "VIEW_NAVMESH_SETTING_PATH", False)

    # Most navmesh size settings in Omniverse are expressed in centimeters.
    radius_cm = float(request.get("agent_radius_m", 0.28)) * 100.0
    _set_if_has_attr(settings, NavMeshSettings, "AGENT_MIN_RADIUS_SETTING_PATH", radius_cm)
    _set_if_has_attr(settings, NavMeshSettings, "AGENT_MAX_RADIUS_SETTING_PATH", radius_cm)
    _set_if_has_attr(
        settings,
        NavMeshSettings,
        "AGENT_MIN_HEIGHT_SETTING_PATH",
        float(request.get("agent_height_m", 1.0)) * 100.0,
    )
    _set_if_has_attr(
        settings,
        NavMeshSettings,
        "AGENT_MAX_STEP_HEIGHT_SETTING_PATH",
        float(request.get("step_height_m", 0.25)) * 100.0,
    )
    _set_if_has_attr(
        settings,
        NavMeshSettings,
        "AGENT_MAX_FLOOR_SLOPE_SETTING_PATH",
        float(request.get("max_slope_deg", 35.0)),
    )


def _ensure_navmesh_volume(stage) -> str:
    import NavSchema  # type: ignore[import-not-found]
    from pxr import Sdf, UsdGeom  # type: ignore[import-not-found]

    for prim in stage.Traverse():
        if prim.GetTypeName() == "NavMeshVolume":
            return prim.GetPath().pathString

    volume_path = Sdf.Path("/NavMeshVolume")
    volume = NavSchema.NavMeshVolume.Define(stage, volume_path)
    volume_type_attr = volume.GetNavVolumeTypeAttr()
    if volume_type_attr:
        volume_type_attr.Set("Include")

    prim = stage.GetPrimAtPath(volume_path)
    boundable = UsdGeom.Boundable(prim)
    extent_attr = boundable.GetExtentAttr()
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
    half_extent_stage = 30.0 / max(meters_per_unit, 1e-6)
    if extent_attr:
        extent_attr.Set(
            [
                (-half_extent_stage, -half_extent_stage, -half_extent_stage),
                (half_extent_stage, half_extent_stage, half_extent_stage),
            ]
        )

    for prim in stage.Traverse():
        if prim.GetTypeName() == "NavMeshVolume":
            return prim.GetPath().pathString

    return ""


def _wait_for_navmesh(app, nav_iface, timeout_s: int) -> List[int]:
    import omni.anim.navigation.core as nav  # type: ignore[import-not-found]
    import omni.kit.app  # type: ignore[import-not-found]

    events: List[int] = []
    state = {"ready": False, "failed": False}
    updated_event = getattr(nav, "EVENT_TYPE_NAVMESH_UPDATED", None)
    ready_event = getattr(nav, "EVENT_TYPE_NAVMESH_READY", None)
    ready_events = {e for e in (updated_event, ready_event) if e is not None}

    def _on_event(e) -> None:
        events.append(int(e.type))
        if e.type == nav.EVENT_TYPE_NAVMESH_BAKE_FAILED:
            state["failed"] = True
        if e.type in ready_events:
            state["ready"] = True

    sub = (
        nav_iface.get_navmesh_event_stream().create_subscription_to_pop(
            _on_event,
            name="scene_layer_navmesh_worker",
            order=omni.kit.app.EVENT_ORDER_DEFAULT,
        )
    )

    started = False
    for _ in range(240):
        if bool(nav_iface.start_navmesh_baking()):
            started = True
            break
        app.update()

    if not started:
        sub = None
        raise RuntimeError("start_navmesh_baking returned False (possibly no bake volume or invalid stage)")

    max_steps = max(1, int(timeout_s * 120))
    for _ in range(max_steps):
        app.update()
        if state["failed"]:
            sub = None
            raise RuntimeError("navmesh bake failed")

        navmesh_obj = nav_iface.get_navmesh()
        if navmesh_obj is not None and (state["ready"] or not ready_events or len(events) > 0):
            sub = None
            return events

    sub = None
    raise TimeoutError(f"navmesh bake timed out after {timeout_s}s")


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    dz = float(a[2]) - float(b[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _path_length(path_points: Sequence[Sequence[float]]) -> float:
    if len(path_points) < 2:
        return 0.0
    s = 0.0
    for i in range(len(path_points) - 1):
        s += _dist(path_points[i], path_points[i + 1])
    return s


def _to_xyz(v) -> Tuple[float, float, float]:
    if hasattr(v, "x") and hasattr(v, "y") and hasattr(v, "z"):
        return float(v.x), float(v.y), float(v.z)
    return float(v[0]), float(v[1]), float(v[2])


def _sample_navmesh_metrics(nav_iface, navmesh_obj, seed: int) -> Dict[str, float]:
    rng = random.Random(seed + 131)

    sample_tag = "scene_layer"
    nav_iface.set_random_seed(sample_tag, int(seed))

    max_random_queries = 320
    target_points = 96

    points: List[Tuple[float, float, float]] = []
    success_random = 0

    for _ in range(max_random_queries):
        try:
            p = navmesh_obj.query_random_point(sample_tag)
        except TypeError:
            p = navmesh_obj.query_random_point()
        except Exception:
            p = None
        if p is not None:
            success_random += 1
            points.append(_to_xyz(p))
            if len(points) >= target_points:
                break

    if len(points) < 4:
        raise RuntimeError("navmesh random-point sampling failed (too few valid points)")

    pair_candidates: List[Tuple[int, int]] = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            pair_candidates.append((i, j))
    rng.shuffle(pair_candidates)

    path_lengths: List[float] = []
    max_path_queries = 220
    for i, j in pair_candidates[:max_path_queries]:
        path = navmesh_obj.query_shortest_path(start_pos=points[i], end_pos=points[j])
        if path is None:
            continue
        pts = path.get_points()
        if len(pts) < 2:
            continue
        length = _path_length([_to_xyz(x) for x in pts])
        if length > 0.5:
            path_lengths.append(length)

    if len(path_lengths) == 0:
        raise RuntimeError("navmesh shortest-path queries returned no valid paths")

    path_lengths.sort()
    shortest = path_lengths[0]
    second_shortest = path_lengths[1] if len(path_lengths) > 1 else path_lengths[0]
    detour_margin = max(0.0, second_shortest - shortest)

    free_space_ratio = max(0.01, min(0.99, success_random / float(max_random_queries)))
    # Static density is a heuristic; higher free-space usually implies lower density.
    static_density = max(0.1, min(0.9, 0.8 - free_space_ratio + 0.08 * (1.0 / max(len(path_lengths), 1))))

    return {
        "path_count": int(len(path_lengths)),
        "shortest_path_m": float(round(shortest, 3)),
        "second_shortest_path_m": float(round(second_shortest, 3)),
        "detour_margin_m": float(round(detour_margin, 3)),
        "free_space_ratio": float(round(free_space_ratio, 3)),
        "static_density": float(round(static_density, 3)),
    }


def _bake_and_eval(request: Dict) -> Dict:
    from isaacsim import SimulationApp  # type: ignore[import-not-found]

    app = SimulationApp(
        {
            "headless": bool(request.get("headless", True)),
            "renderer": str(request.get("renderer", "RayTracedLighting")),
        }
    )

    try:
        import omni.kit.app  # type: ignore[import-not-found]
        import omni.usd  # type: ignore[import-not-found]

        ext_mgr = omni.kit.app.get_app().get_extension_manager()
        for ext_name in ("omni.anim.navigation.bundle", "omni.anim.navigation.core"):
            try:
                if not ext_mgr.is_extension_enabled(ext_name):
                    ext_mgr.set_extension_enabled_immediate(ext_name, True)
            except Exception:
                # Some versions may not expose one of the bundle names; continue with best effort.
                pass

        for _ in range(20):
            app.update()

        import omni.anim.navigation.core as nav  # type: ignore[import-not-found]
        stage_path = str(request["stage_usd"])
        if not os.path.isfile(stage_path):
            raise FileNotFoundError(f"stage usd not found: {stage_path}")

        usd_ctx = omni.usd.get_context()
        ok = usd_ctx.open_stage(stage_path)
        if not ok:
            raise RuntimeError(f"failed to open stage: {stage_path}")

        for _ in range(120):
            app.update()

        _apply_navmesh_settings(request)

        stage = usd_ctx.get_stage()
        if stage is None:
            raise RuntimeError("failed to get opened stage")

        navmesh_volume_path = _ensure_navmesh_volume(stage)

        nav_iface = nav.acquire_interface()
        events = _wait_for_navmesh(app, nav_iface, int(request.get("bake_timeout_s", 300)))
        navmesh_obj = nav_iface.get_navmesh()
        if navmesh_obj is None:
            raise RuntimeError("navmesh object unavailable after bake")
        metrics = _sample_navmesh_metrics(nav_iface, navmesh_obj, int(request.get("seed", 0)))

        scene_dir = str(request["scene_dir"])
        profile = str(request.get("profile", "default"))
        navmesh_file = os.path.join(scene_dir, f"navmesh_{profile}.bin")
        with open(navmesh_file, "wb") as f:
            f.write(b"ISAAC_NAVMESH_READY")

        debug_path = os.path.join(scene_dir, f"navmesh_debug_{profile}.json")
        _write_json(
            debug_path,
            {
                "stage_usd": stage_path,
                "profile": profile,
                "navmesh_volume_path": navmesh_volume_path,
                "events": events,
                "metrics": metrics,
            },
        )

        return {
            "ok": True,
            "profile": profile,
            "navmesh_file": navmesh_file,
            "debug_file": debug_path,
            "metrics": metrics,
        }
    finally:
        # Intentionally skip app.close() to reduce close-time crashes on some Isaac builds.
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Isaac NavMesh bake worker")
    parser.add_argument("--request", type=str, required=True, help="Path to request JSON")
    parser.add_argument("--response", type=str, required=True, help="Path to response JSON")
    args = parser.parse_args()

    try:
        request = _read_json(args.request)
        payload = _bake_and_eval(request)
    except TimeoutError as exc:
        payload = {
            "ok": False,
            "error_code": "GPU_TIMEOUT",
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    except FileNotFoundError as exc:
        payload = {
            "ok": False,
            "error_code": "NAVMESH_FAIL",
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    except Exception as exc:  # noqa: BLE001
        payload = {
            "ok": False,
            "error_code": "RUNTIME_FAIL",
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    _write_json(args.response, payload)


if __name__ == "__main__":
    main()
