"""动态层 CLI 入口。

用法:
    python -m dataengine.src.dynamic_layer --scene-usd <path> --output-dir <dir> [--seed 123456]
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import DynamicLayerConfig
from .dynamic_layer import IRADynamicLayer


def main() -> None:
    parser = argparse.ArgumentParser(description="DataEngine Dynamic Layer (IRA)")
    parser.add_argument("--scene-usd", type=str, required=True, help="场景 USD 文件路径")
    parser.add_argument("--scene-dir", type=str, required=True, help="场景工作目录")
    parser.add_argument("--scene-id", type=str, default="scene_000000", help="场景 ID")
    parser.add_argument("--seed", type=int, default=123456, help="随机种子")
    parser.add_argument("--config", type=str, default="", help="动态层配置 JSON 路径")
    parser.add_argument("--dry-run", action="store_true", help="仅生成配置不执行 IRA")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)
        cfg = DynamicLayerConfig.from_dict(cfg_data)
    else:
        cfg = DynamicLayerConfig()

    layer = IRADynamicLayer(cfg=cfg)
    result = layer.generate(
        scene_id=args.scene_id,
        scene_dir=args.scene_dir,
        seed=args.seed,
        stage_usd=args.scene_usd,
        dry_run=args.dry_run,
    )

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    sys.exit(0 if result.ok or args.dry_run else 1)


if __name__ == "__main__":
    main()
