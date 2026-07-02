#!/usr/bin/env python3
"""Emit Bridge-DP cache rotation profile defaults as shell variables."""

from __future__ import annotations

import argparse

from scripts.cache_rotation.cache_rotation_lib import resolve_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print Bridge-DP cache rotation profile defaults.")
    parser.add_argument("--gpus", type=int, choices=[4, 8], required=True)
    parser.add_argument("--nvme-size", choices=["1tb", "2tb", "4tb", "1t", "2t", "4t"], required=True)
    parser.add_argument("--preset", choices=["balanced", "quality", "throughput"], default="balanced")
    return parser.parse_args()


def main() -> int:
    profile = resolve_profile(**vars(parse_args()))
    print(f"BRIDGEDP_NUM_GPUS={profile.gpus}")
    print(f"BRIDGEDP_NVME_SIZE={profile.nvme_size}")
    print(f"BRIDGEDP_CACHE_PRESET={profile.preset}")
    print(f"BRIDGEDP_CACHE_SLOT_GB={profile.cache_slot_gb}")
    print(f"BRIDGEDP_SHARD_EPOCHS={profile.shard_epochs}")
    print(f"BRIDGEDP_BATCH_SIZE={profile.per_gpu_batch}")
    print(f"BRIDGEDP_NUM_WORKERS={profile.num_workers}")
    print(f"BRIDGEDP_BUILD_WORKERS={profile.build_workers}")
    print(f"BRIDGEDP_GRAD_ACCUM={profile.grad_accum}")
    print(f"BRIDGEDP_SAVE_STEPS={profile.save_steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
