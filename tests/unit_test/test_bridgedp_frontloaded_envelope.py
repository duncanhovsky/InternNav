import importlib.util
from pathlib import Path

import pytest
import torch

from internnav.model.basemodel.bridgedp.bridge_scheduler import BridgeScheduler


SHAPE = (1, 24, 3)
DEVICE = torch.device("cpu")
GOAL = torch.tensor([[2.0, 0.0, 0.0]])


def _scheduler(frontload, scheduler_cls=BridgeScheduler):
    return scheduler_cls(
        bridge_scale_invariant_sigma=True,
        bridge_anisotropic_xy=True,
        bridge_normal_sigma_ratio=0.5,
        bridge_tangent_sigma_ratio=0.05,
        bridge_theta_sigma_ratio=0.1,
        bridge_envelope_frontload=frontload,
    )


def _params(frontload, scheduler_cls=BridgeScheduler):
    scheduler = _scheduler(frontload, scheduler_cls)
    return scheduler, scheduler.pointgoal_noise_params(SHAPE, DEVICE, goal=GOAL)


def _load_eval_scheduler():
    path = Path(__file__).resolve().parents[3] / "NavDP" / "baselines" / "bridgedp" / "bridge_scheduler.py"
    spec = importlib.util.spec_from_file_location("navdp_bridgedp_eval_scheduler", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.BridgeScheduler


def test_zero_frontload_matches_original_symmetric_envelope():
    scheduler, params = _params(0.0)
    _, _, _, sigma_tangent, sigma_normal, sigma_theta, _ = params
    tau = scheduler.trajectory_time(SHAPE[1], DEVICE)
    p = scheduler.direction_adaptive_exponent(torch.zeros(1)).view(1, 1, 1)
    original = ((tau * (1.0 - tau)).clamp(min=0.0) / 0.25).pow(p)

    torch.testing.assert_close(sigma_tangent, 2.0 * 0.05 * original)
    torch.testing.assert_close(sigma_normal, 2.0 * 0.5 * original)
    torch.testing.assert_close(sigma_theta, 2.0 * 0.1 * original)


def test_frontload_moves_turn_peak_forward_without_moving_tangent_envelope():
    _, params_0 = _params(0.0)
    _, params_1 = _params(1.0)
    _, params_2 = _params(2.0)

    tangent_0, normal_0, theta_0 = params_0[3], params_0[4], params_0[5]
    tangent_1, normal_1, theta_1 = params_1[3], params_1[4], params_1[5]
    tangent_2, normal_2, theta_2 = params_2[3], params_2[4], params_2[5]

    assert normal_0[0, :, 0].argmax().item() + 1 == 12
    assert normal_1[0, :, 0].argmax().item() + 1 == 8
    assert normal_2[0, :, 0].argmax().item() + 1 == 6
    torch.testing.assert_close(tangent_0, tangent_1)
    torch.testing.assert_close(tangent_0, tangent_2)
    assert normal_1[0, 0, 0] > normal_0[0, 0, 0]
    assert theta_1[0, 0, 0] > theta_0[0, 0, 0]
    assert normal_1[0, -1, 0] == 0.0
    assert theta_2[0, -1, 0] == 0.0


@pytest.mark.parametrize("frontload", [-0.01, -1.0])
def test_negative_frontload_is_rejected(frontload):
    with pytest.raises(ValueError, match="non-negative"):
        _scheduler(frontload)


def test_eval_scheduler_matches_training_scheduler():
    eval_scheduler_cls = _load_eval_scheduler()
    _, train_params = _params(1.0)
    _, eval_params = _params(1.0, eval_scheduler_cls)

    for train_tensor, eval_tensor in zip(train_params, eval_params):
        torch.testing.assert_close(train_tensor, eval_tensor)

