from __future__ import annotations

import argparse
import json
import traceback
from typing import Dict, List

from .utils import stable_hash


def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _strip_anonymous_sublayers(stage_path: str) -> None:
    if not stage_path:
        return

    try:
        with open(stage_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return

    out: List[str] = []
    in_sublayers = False
    for line in lines:
        if not in_sublayers and "subLayers = [" in line:
            in_sublayers = True
            out.append(line)
            continue

        if in_sublayers:
            if "@anon:" in line:
                continue
            out.append(line)
            if "]" in line:
                in_sublayers = False
            continue

        out.append(line)

    with open(stage_path, "w", encoding="utf-8") as f:
        f.writelines(out)


def _compose(request: Dict) -> Dict:
    from isaacsim import SimulationApp  # type: ignore[import-not-found]

    app = SimulationApp(
        {
            "headless": bool(request.get("headless", True)),
            "renderer": str(request.get("renderer", "RayTracedLighting")),
        }
    )

    try:
        import omni.usd  # type: ignore[import-not-found]
        import omni.replicator.core as rep  # type: ignore[import-not-found]
        from pxr import Gf, UsdGeom  # type: ignore[import-not-found]

        if bool(request.get("enable_replicator", True)):
            try:
                rep.set_global_seed(int(request.get("replicator_seed", 0)))
            except Exception:
                # Compat: older versions may not expose this API.
                pass

        usd_ctx = omni.usd.get_context()
        usd_ctx.new_stage()
        stage = usd_ctx.get_stage()
        if stage is None:
            raise RuntimeError("failed to create stage")

        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        stage.DefinePrim("/World", "Xform")

        template_usd = str(request["template_usd"])
        scene_root = stage.DefinePrim("/World/SceneRoot", "Xform")
        scene_root.GetReferences().AddReference(template_usd)

        stage.DefinePrim("/World/ProceduralProps", "Xform")
        props = request.get("props", [])
        for item in props:
            idx = int(item["idx"])
            prim_path = f"/World/ProceduralProps/prop_{idx:04d}"
            # 变换施加在锚点 prim，资产引用挂在子 prim，避免 xformOp 命名冲突。
            anchor_prim = stage.DefinePrim(prim_path, "Xform")
            ref_prim = stage.DefinePrim(f"{prim_path}/Asset", "Xform")
            ref_prim.GetReferences().AddReference(str(item["usd_path"]))

            xformable = UsdGeom.Xformable(anchor_prim)
            t = item["position"]
            r = item["rotation_euler_deg"]
            s = item["scale_xyz"]
            xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "placement").Set(
                Gf.Vec3d(float(t[0]), float(t[1]), float(t[2]))
            )
            xformable.AddRotateXYZOp(UsdGeom.XformOp.PrecisionDouble, "placement").Set(
                Gf.Vec3d(float(r[0]), float(r[1]), float(r[2]))
            )
            xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble, "placement").Set(
                Gf.Vec3d(float(s[0]), float(s[1]), float(s[2]))
            )

        warmup_frames = int(request.get("warmup_frames", 0))
        for _ in range(max(0, warmup_frames)):
            app.update()

        stage_path = str(request["stage_path"])
        flattened = stage.Flatten()
        if not flattened.Export(stage_path):
            raise RuntimeError(f"failed to export composed stage: {stage_path}")
        _strip_anonymous_sublayers(stage_path)

        layout_hash = stable_hash(json.dumps(props, ensure_ascii=False, sort_keys=True))
        return {"ok": True, "stage_usd": stage_path, "layout_hash": layout_hash}
    finally:
        # Avoid explicit close by default to reduce close-time crashes in some Isaac builds.
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Isaac Replicator compose worker")
    parser.add_argument("--request", type=str, required=True, help="Path to request JSON")
    parser.add_argument("--response", type=str, required=True, help="Path to response JSON")
    args = parser.parse_args()

    try:
        request = _read_json(args.request)
        payload = _compose(request)
    except Exception as exc:  # noqa: BLE001
        payload = {
            "ok": False,
            "error_code": "USD_COMPOSE_FAIL",
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    _write_json(args.response, payload)


if __name__ == "__main__":
    main()
