#!/usr/bin/env python3
"""Emit shell variables for one Bridge-DP cache rotation stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.cache_rotation.cache_rotation_lib import (
    compute_training_plan,
    find_latest_global_step,
    load_manifest,
    resolve_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan Bridge-DP cache rotation stage steps.")
    parser.add_argument("--manifest", type=Path, default=Path("/hdd/bridgedp_rotation/manifests/shards.json"))
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", type=int, choices=[4, 8], required=True)
    parser.add_argument(
        "--nvme-size",
        choices=["1tb", "1.5tb", "2tb", "4tb", "1t", "1.5t", "1500g", "1500gb", "2t", "4t"],
        required=True,
    )
    parser.add_argument("--preset", choices=["balanced", "quality", "throughput"], default="balanced")
    parser.add_argument("--total-epochs", type=int, required=True)
    parser.add_argument("--shard-epochs", type=int, default=0)
    parser.add_argument("--per-gpu-batch", type=int, default=0)
    parser.add_argument("--grad-accum", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = resolve_profile(args.gpus, args.nvme_size, args.preset)
    manifest = load_manifest(args.manifest)
    shard = manifest["shards"][args.shard_index]
    current_step = find_latest_global_step(args.output_dir)
    shard_epochs = args.shard_epochs or profile.shard_epochs
    per_gpu_batch = args.per_gpu_batch or profile.per_gpu_batch
    plan = compute_training_plan(
        total_episodes=manifest["summary"]["total_episodes"],
        shard_episodes=shard["episode_count"],
        total_epochs=args.total_epochs,
        shard_epochs=shard_epochs,
        gpus=args.gpus,
        per_gpu_batch=per_gpu_batch,
        grad_accum=args.grad_accum,
        current_global_step=current_step,
    )
    print(f"BRIDGEDP_CURRENT_GLOBAL_STEP={current_step}")
    print(f"BRIDGEDP_TOTAL_MAX_STEPS={plan.total_max_steps}")
    print(f"BRIDGEDP_STAGE_STEPS={plan.stage_steps}")
    print(f"BRIDGEDP_STAGE_END_STEP={plan.stage_end_step}")
    print(f"BRIDGEDP_GLOBAL_BATCH={plan.global_batch}")
    print(f"BRIDGEDP_SHARD_EPISODES={shard['episode_count']}")
    print(f"BRIDGEDP_TOTAL_EPISODES={manifest['summary']['total_episodes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
