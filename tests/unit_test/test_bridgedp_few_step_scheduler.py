import unittest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from internnav.model.basemodel.bridgedp.bridge_scheduler import BridgeScheduler


class BridgeDPFewStepSchedulerTest(unittest.TestCase):
    def setUp(self):
        self.scheduler = BridgeScheduler(num_train_timesteps=10)

    def test_sparse_schedules_span_the_training_time_axis(self):
        expected = {
            2: [9, 0],
            4: [9, 6, 3, 0],
            8: [9, 8, 6, 5, 4, 3, 1, 0],
            10: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        }

        for steps, timesteps in expected.items():
            with self.subTest(steps=steps):
                self.scheduler.set_timesteps(steps)
                self.assertEqual(self.scheduler.timesteps.tolist(), timesteps)

    def test_inference_budget_must_fit_the_training_axis(self):
        for steps in (0, 11):
            with self.subTest(steps=steps):
                with self.assertRaises(ValueError):
                    self.scheduler.set_timesteps(steps)

    def test_sparse_reverse_step_jumps_to_next_selected_time(self):
        x_s = torch.ones(1, 2, 3)
        x0_pred = torch.zeros_like(x_s)
        endpoint = torch.zeros(1, 3)

        x_prev = self.scheduler.step_trajectory(
            x0_pred,
            x_s,
            torch.tensor(9),
            prev_timestep=torch.tensor(6),
            goal=endpoint,
            theta_g=torch.zeros(1),
            origin=endpoint,
            eta=0.0,
        )

        self.assertTrue(torch.allclose(x_prev, torch.full_like(x_s, 0.7)))

    def test_adjacent_reverse_step_keeps_ten_step_behavior(self):
        x_s = torch.ones(1, 2, 3)
        x0_pred = torch.zeros_like(x_s)
        endpoint = torch.zeros(1, 3)

        implicit = self.scheduler.step_trajectory(
            x0_pred,
            x_s,
            torch.tensor(9),
            goal=endpoint,
            theta_g=torch.zeros(1),
            origin=endpoint,
            eta=0.0,
        )
        explicit = self.scheduler.step_trajectory(
            x0_pred,
            x_s,
            torch.tensor(9),
            prev_timestep=torch.tensor(8),
            goal=endpoint,
            theta_g=torch.zeros(1),
            origin=endpoint,
            eta=0.0,
        )

        self.assertTrue(torch.allclose(implicit, explicit))
        self.assertTrue(torch.allclose(explicit, torch.full_like(x_s, 0.9)))


if __name__ == "__main__":
    unittest.main()
