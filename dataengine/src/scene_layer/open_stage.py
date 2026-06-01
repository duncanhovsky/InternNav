from __future__ import annotations

import argparse
import time


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open a generated USD stage in Isaac Sim GUI")
    parser.add_argument("--stage", type=str, required=True, help="Path to composed stage USD/USDA")
    parser.add_argument(
        "--overlay",
        type=str,
        default="",
        help="Optional dynamic overlay USD. If provided, preview opens overlay (which sublayers base stage).",
    )
    parser.add_argument("--renderer", type=str, default="RayTracedLighting", help="Isaac renderer")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": False, "renderer": args.renderer})
    try:
        import omni.usd

        usd_ctx = omni.usd.get_context()
        target_stage = args.overlay.strip() or args.stage
        ok = usd_ctx.open_stage(target_stage)
        if not ok:
            raise RuntimeError(f"failed to open stage: {target_stage}")

        while app.is_running():
            app.update()
            time.sleep(0.01)
    finally:
        # Keep default behavior minimal; if your local build crashes on close,
        # the generated files are already on disk.
        app.close()


if __name__ == "__main__":
    main()
