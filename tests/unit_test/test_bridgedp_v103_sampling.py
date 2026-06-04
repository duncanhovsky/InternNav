import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from internnav.model.basemodel.bridgedp.bridge_scheduler import BridgeScheduler
from internnav.trainer.bridgedp_trainer import BridgeDPTrainer


def test_bridge_anchor_sampling_preserves_radius_and_keeps_first_block():
    torch.manual_seed(7)
    scheduler = BridgeScheduler(
        bridge_anchor_angle_std=0.0,
        bridge_anchor_angle_max=0.5,
        bridge_anchor_uniform_prob=1.0,
    )
    goal = torch.tensor([[2.0, 0.0, 0.3], [0.0, 3.0, -0.2]])
    origin = torch.zeros_like(goal)

    anchor, theta = scheduler.sample_bridge_anchor_goals(
        goal,
        origin,
        sample_num=4,
        keep_first_sample=True,
    )

    expected = goal.repeat(4, 1)
    assert torch.allclose(anchor[:2], expected[:2])
    assert torch.allclose(theta[:2], torch.atan2(goal[:, 1], goal[:, 0]))

    radius = torch.norm(anchor[:, :2] - origin.repeat(4, 1)[:, :2], dim=-1)
    expected_radius = torch.norm(expected[:, :2], dim=-1)
    assert torch.allclose(radius, expected_radius, atol=1e-5)
    assert torch.any(torch.abs(anchor[2:, 1]) > 1e-4)
    assert torch.allclose(anchor[:, 2], expected[:, 2])


def test_bridge_anchor_sampling_adds_paired_edge_candidates_after_original():
    torch.manual_seed(11)
    angle_max = 0.65
    scheduler = BridgeScheduler(
        bridge_anchor_angle_std=0.25,
        bridge_anchor_angle_max=angle_max,
        bridge_anchor_uniform_prob=0.0,
        bridge_anchor_edge_prob=0.5,
    )
    goal = torch.tensor([[2.0, 0.0, 0.3]])
    origin = torch.zeros_like(goal)

    _, theta = scheduler.sample_bridge_anchor_goals(
        goal,
        origin,
        sample_num=17,
        keep_first_sample=True,
    )
    delta = theta - torch.atan2(goal[:, 1], goal[:, 0]).repeat(17)

    assert math.isclose(float(delta[0]), 0.0, abs_tol=1e-6)
    edge_delta = delta[-8:]
    assert torch.all(edge_delta.abs() <= angle_max + 1e-6)
    assert torch.count_nonzero(edge_delta < 0).item() == 4
    assert torch.count_nonzero(edge_delta > 0).item() == 4


def test_bridge_anchor_edge_probability_applies_to_single_sample_batches():
    torch.manual_seed(13)
    angle_max = 0.65
    scheduler = BridgeScheduler(
        bridge_anchor_angle_std=0.25,
        bridge_anchor_angle_max=angle_max,
        bridge_anchor_uniform_prob=0.0,
        bridge_anchor_edge_prob=1.0,
    )
    goal = torch.tensor([[2.0, 0.0, 0.3], [2.0, 0.0, 0.3]])
    origin = torch.zeros_like(goal)

    _, theta = scheduler.sample_bridge_anchor_goals(
        goal,
        origin,
        sample_num=1,
        keep_first_sample=False,
    )
    delta = theta - torch.atan2(goal[:, 1], goal[:, 0])

    assert torch.all(delta.abs() <= angle_max + 1e-6)
    assert torch.count_nonzero(delta < 0).item() == 1
    assert torch.count_nonzero(delta > 0).item() == 1


def test_projection_resampling_uses_uniform_chord_progress():
    trainer = BridgeDPTrainer.__new__(BridgeDPTrainer)
    trainer.trajectory_resample_mode = "projection"
    trainer.trajectory_projection_monotonic_eps = 1e-5
    trainer.trajectory_projection_min_span = 0.80

    traj = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    sampled = trainer._resample_one_trajectory_gpu(
        traj,
        num_points=4,
        include_start=False,
    )
    projection = sampled[:, 0] / 2.0

    assert torch.allclose(
        projection,
        torch.tensor([0.25, 0.50, 0.75, 1.00]),
        atol=1e-4,
    )
    assert math.isclose(float(sampled[-1, 0]), 2.0, abs_tol=1e-4)


def test_hybrid_projection_falls_back_on_flat_lateral_motion():
    trainer = BridgeDPTrainer.__new__(BridgeDPTrainer)
    trainer.trajectory_resample_mode = "hybrid_projection"
    trainer.trajectory_projection_monotonic_eps = 1e-5
    trainer.trajectory_projection_min_span = 0.80
    trainer.trajectory_projection_flat_lateral_eps = 1e-3

    traj = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    sampled = trainer._resample_one_trajectory_gpu(
        traj,
        num_points=4,
        include_start=False,
    )
    projection = sampled[:, 0] / 2.0

    assert not torch.allclose(
        projection,
        torch.tensor([0.25, 0.50, 0.75, 1.00]),
        atol=1e-4,
    )
