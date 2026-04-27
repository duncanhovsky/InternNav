from __future__ import annotations

import argparse
import time


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open a generated USD stage in Isaac Sim GUI")
    parser.add_argument("--stage", type=str, required=True, help="Path to composed stage USD/USDA")
    parser.add_argument("--renderer", type=str, default="RayTracedLighting", help="Isaac renderer")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": False, "renderer": args.renderer})
    try:
        import omni.usd

        usd_ctx = omni.usd.get_context()
        ok = usd_ctx.open_stage(args.stage)
        if not ok:
            raise RuntimeError(f"failed to open stage: {args.stage}")

        while app.is_running():
            app.update()
            time.sleep(0.01)
    finally:
        # Keep default behavior minimal; if your local build crashes on close,
        # the generated files are already on disk.
        app.close()


if __name__ == "__main__":
    main()
