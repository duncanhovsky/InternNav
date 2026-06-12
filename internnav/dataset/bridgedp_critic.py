"""Bridge-DP critic label geometry utilities."""

from __future__ import annotations

import numpy as np


def densify_trajectory_xy(points_xy: np.ndarray, max_step: float = 0.05) -> np.ndarray:
    """Sample a sparse XY trajectory so segment interiors contribute to clearance."""
    points = np.asarray(points_xy, dtype=np.float32)[..., :2]
    if points.size == 0 or points.shape[0] <= 1:
        return points.reshape((-1, 2)).astype(np.float32)

    step = float(max_step)
    if step <= 0.0:
        return points.astype(np.float32)

    dense_parts = [points[0:1]]
    for start, end in zip(points[:-1], points[1:]):
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length < 1e-6:
            continue
        count = max(1, int(np.ceil(length / step)))
        alpha = np.linspace(1.0 / count, 1.0, count, dtype=np.float32)[:, None]
        dense_parts.append(start[None, :] + alpha * delta[None, :])

    if len(dense_parts) == 1:
        return dense_parts[0].astype(np.float32)
    return np.concatenate(dense_parts, axis=0).astype(np.float32)


def min_l2_distances_xy(points_xy: np.ndarray, obstacle_xy: np.ndarray) -> np.ndarray:
    """Return each point's minimum planar L2 distance to obstacle points."""
    points = np.asarray(points_xy, dtype=np.float32)[..., :2]
    obstacles = np.asarray(obstacle_xy, dtype=np.float32)[..., :2]
    if points.size == 0:
        return np.empty((0,), dtype=np.float32)
    if obstacles.size == 0:
        return np.full((points.shape[0],), np.inf, dtype=np.float32)

    delta = points[:, np.newaxis, :] - obstacles[np.newaxis, :, :]
    return np.linalg.norm(delta, axis=-1).min(axis=-1).astype(np.float32)


def bridge_dp_soft_risk(
    distances: np.ndarray,
    hard_threshold: float = 0.1,
    soft_threshold: float = 0.2,
    beta: float = 4.0,
) -> np.ndarray:
    """Map clearance distances to [0, 1] risk with a hard core and soft shell."""
    d = np.asarray(distances, dtype=np.float32)
    risk = np.zeros_like(d, dtype=np.float32)
    if d.size == 0:
        return risk

    hard = float(hard_threshold)
    soft = float(soft_threshold)
    risk[d <= hard] = 1.0
    if soft <= hard:
        return risk

    shell = (d > hard) & (d < soft)
    if not np.any(shell):
        return risk

    u = (d[shell] - hard) / (soft - hard)
    beta = float(beta)
    if abs(beta) < 1e-6:
        shell_risk = 1.0 - u
    else:
        exp_tail = np.exp(-beta)
        shell_risk = (np.exp(-beta * u) - exp_tail) / (1.0 - exp_tail)
    risk[shell] = np.clip(shell_risk, 0.0, 1.0)
    return risk.astype(np.float32)


def compute_bridge_dp_critic_score_from_distances(
    distances: np.ndarray,
    action_indexes: np.ndarray,
    hard_threshold: float = 0.1,
    soft_threshold: float = 0.2,
    beta: float = 4.0,
    max_weight: float = 5.0,
    mean_weight: float = 2.0,
    trend_weight: float = 0.5,
    safe_score: float = 2.0,
) -> float:
    """Score a candidate trajectory from precomputed obstacle clearances."""
    d = np.asarray(distances, dtype=np.float32)
    if d.size == 0 or np.all(np.isinf(d)):
        return float(safe_score)

    idx = np.asarray(action_indexes, dtype=np.int64)
    if idx.size == 0:
        idx = np.arange(d.shape[0], dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < d.shape[0])]
    if idx.size == 0:
        return float(safe_score)

    risk_idx = idx[:-1] if idx.size > 1 else idx
    risk = bridge_dp_soft_risk(
        d[risk_idx],
        hard_threshold=hard_threshold,
        soft_threshold=soft_threshold,
        beta=beta,
    )
    max_risk = float(risk.max()) if risk.size else 0.0
    mean_risk = float(risk.mean()) if risk.size else 0.0

    selected_d = d[idx]
    if selected_d.size > 1:
        trend = float((selected_d[1:] - selected_d[:-1]).sum())
    else:
        trend = 0.0

    return float(
        -float(max_weight) * max_risk
        - float(mean_weight) * mean_risk
        + float(trend_weight) * trend
    )


def compute_bridge_dp_critic_score(
    trajectory_world_points: np.ndarray,
    obstacle_points: np.ndarray,
    action_indexes: np.ndarray,
    hard_threshold: float = 0.1,
    soft_threshold: float = 0.2,
    beta: float = 4.0,
    max_weight: float = 5.0,
    mean_weight: float = 2.0,
    trend_weight: float = 0.5,
    safe_score: float = 2.0,
    densify_step: float = 0.05,
) -> float:
    """Compute Bridge-DP's geometric critic label for one trajectory."""
    if obstacle_points is None or np.asarray(obstacle_points).shape[0] == 0:
        return float(safe_score)
    dense_points = densify_trajectory_xy(trajectory_world_points, max_step=densify_step)
    distances = min_l2_distances_xy(dense_points, obstacle_points)
    if float(densify_step) > 0.0:
        action_indexes = np.arange(distances.shape[0], dtype=np.int64)
    return compute_bridge_dp_critic_score_from_distances(
        distances,
        action_indexes,
        hard_threshold=hard_threshold,
        soft_threshold=soft_threshold,
        beta=beta,
        max_weight=max_weight,
        mean_weight=mean_weight,
        trend_weight=trend_weight,
        safe_score=safe_score,
    )
