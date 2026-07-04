#!/usr/bin/env python3
"""Build one READY Bridge-DP cache slot from a shard manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.cache_rotation.cache_rotation_lib import DEFAULT_TRAJ_SUFFIX, build_cache_slot, validate_cache_slot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Bridge-DP cache_A/cache_B from HDD extracted data.")
    parser.add_argument("--manifest", type=Path, default=Path("/hdd/bridgedp_rotation/manifests/shards.json"))
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--hdd-root", type=Path, default=Path("/hdd"))
    parser.add_argument("--hdd-traj", type=Path, default=None)
    parser.add_argument("--slot-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--drop-existing-before-build",
        action="store_true",
        help="Remove the previous slot before copying the new shard to reduce NVMe peak usage.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hdd_traj = args.hdd_traj or (args.hdd_root / DEFAULT_TRAJ_SUFFIX)
    metadata = build_cache_slot(
        manifest_path=args.manifest,
        shard_index=args.shard_index,
        hdd_traj=hdd_traj,
        slot_root=args.slot_root,
        force=args.force,
        drop_existing_before_build=args.drop_existing_before_build,
    )
    report = validate_cache_slot(args.slot_root)
    if not report.ok:
        print("cache validation failed")
        for example in report.examples:
            print(f"  - {example}")
        return 1
    print(f"slot={args.slot_root}")
    print(f"status={metadata['status']}")
    print(f"shard_index={metadata['shard_index']}")
    print(f"episode_count={metadata['episode_count']}")
    print(f"preload_index={metadata['preload_index']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
