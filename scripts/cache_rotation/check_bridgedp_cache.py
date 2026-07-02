#!/usr/bin/env python3
"""Validate a READY Bridge-DP cache slot."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.cache_rotation.cache_rotation_lib import validate_cache_slot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Bridge-DP cache slot.")
    parser.add_argument("--slot-root", type=Path, required=True)
    parser.add_argument("--example-limit", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_cache_slot(args.slot_root, example_limit=args.example_limit)
    print(f"slot={args.slot_root}")
    print(f"ready={int(report.marker_ready)}")
    print(f"missing_paths={report.missing_paths}")
    print(f"bad_shape={report.bad_shape}")
    if report.examples:
        print("examples:")
        for example in report.examples:
            print(f"  - {example}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
