from __future__ import annotations

import argparse
import json

from .config import SceneLayerConfig
from .pipeline import SceneLayerPipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scene Layer Pipeline")
    parser.add_argument(
        "--config-json",
        type=str,
        default="",
        help="场景层 JSON 配置路径。为空时使用默认配置。",
    )
    parser.add_argument(
        "--target-scene-count",
        type=int,
        default=-1,
        help="可选覆盖：目标场景数量。",
    )
    parser.add_argument(
        "--trajectories-per-scene",
        type=int,
        default=-1,
        help="可选覆盖：每个场景轨迹任务数量。",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--isaac-headless",
        action="store_true",
        help="强制 Isaac 以后端无头模式运行。",
    )
    mode_group.add_argument(
        "--isaac-gui",
        action="store_true",
        help="强制 Isaac 以后端有头模式运行（显示窗口）。",
    )
    parser.add_argument(
        "--inspect-gui",
        action="store_true",
        help="GUI 模式下保持窗口运行，直到手动关闭（仅调试用）。",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.config_json:
        cfg = SceneLayerConfig.from_json(args.config_json)
    else:
        cfg = SceneLayerConfig()
        cfg.validate()

    if args.target_scene_count > 0:
        cfg.target_scene_count = args.target_scene_count
    if args.trajectories_per_scene > 0:
        cfg.trajectories_per_scene = args.trajectories_per_scene

    if args.isaac_headless:
        cfg.isaac_headless = True
    if args.isaac_gui:
        cfg.isaac_headless = False
    if args.inspect_gui:
        cfg.isaac_gui_inspect_mode = True
        cfg.isaac_headless = False

    cfg.validate()

    pipeline = SceneLayerPipeline(cfg)
    stats = pipeline.run()
    print(json.dumps({"status": "ok", "stats": stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
