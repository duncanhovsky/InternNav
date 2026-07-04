"""Helpers for evenly spaced Bridge-DP checkpoint archives."""

from __future__ import annotations

import math


def compute_uniform_checkpoint_targets(total_steps: int, checkpoint_count: int) -> list[int]:
    """Return monotonically increasing target steps spread over the full run."""
    total = max(int(total_steps), 1)
    count = int(checkpoint_count)
    if count <= 0:
        return []

    targets: list[int] = []
    for index in range(1, count + 1):
        step = math.ceil(total * index / count)
        if not targets or step != targets[-1]:
            targets.append(step)
    return targets
