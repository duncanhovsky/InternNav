#!/usr/bin/env python3
"""Create Bridge-DP cache rotation shard manifests from an extracted HDD dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.cache_rotation.cache_rotation_lib import (
    DEFAULT_TRAJ_SUFFIX,
    discover_scenes,
    resolve_profile,
    write_shard_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Bridge-DP cache rotation shard manifest.")
    parser.add_argument("--hdd-root", type=Path, default=Path("/hdd"))
    parser.add_argument("--hdd-traj", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=Path("/hdd/bridgedp_rotation/manifests/shards.json"))
    parser.add_argument("--gpus", type=int, choices=[4, 8], required=True)
    parser.add_argument(
        "--nvme-size",
        choices=["1tb", "1.5tb", "2tb", "4tb", "1t", "1.5t", "1500g", "1500gb", "2t", "4t"],
        required=True,
    )
    parser.add_argument("--preset", choices=["balanced", "quality", "throughput"], default="balanced")
    parser.add_argument("--cache-slot-gb", type=int, default=0)
    parser.add_argument("--max-scenes-per-shard", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = resolve_profile(args.gpus, args.nvme_size, args.preset)
    hdd_traj = args.hdd_traj or (args.hdd_root / DEFAULT_TRAJ_SUFFIX)
    cache_slot_gb = args.cache_slot_gb or profile.cache_slot_gb
    max_scenes = args.max_scenes_per_shard or None

    scenes = discover_scenes(hdd_traj)
    if not scenes:
        raise RuntimeError(f"no trainable scenes found under {hdd_traj}")
    manifest = write_shard_manifest(
        scenes=scenes,
        output_path=args.manifest,
        cache_slot_gb=cache_slot_gb,
        max_scenes_per_shard=max_scenes,
        seed=args.seed,
    )
    summary = manifest["summary"]
    print(f"manifest={args.manifest}")
    print(f"hdd_traj={hdd_traj}")
    print(f"total_scenes={summary['total_scenes']}")
    print(f"total_episodes={summary['total_episodes']}")
    print(f"num_shards={summary['num_shards']}")
    print(f"cache_slot_gb={cache_slot_gb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
