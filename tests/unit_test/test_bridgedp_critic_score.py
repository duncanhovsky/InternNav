import numpy as np
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from internnav.dataset.bridgedp_critic import (
    bridge_dp_soft_risk,
    compute_bridge_dp_critic_score_from_distances,
    min_l2_distances_xy,
)


class BridgeDPCriticScoreTest(unittest.TestCase):
    def test_min_l2_distance_marks_diagonal_near_obstacle(self):
        trajectory_xy = np.array([[0.12, 0.12]], dtype=np.float32)
        obstacle_xy = np.array([[0.0, 0.0]], dtype=np.float32)

        distances = min_l2_distances_xy(trajectory_xy, obstacle_xy)

        self.assertTrue(np.isclose(distances[0], np.sqrt(0.12**2 + 0.12**2)))
        self.assertLess(distances[0], 0.2)


    def test_soft_risk_is_hard_inside_0_1_and_exponential_to_soft_threshold(self):
        distances = np.array([0.05, 0.1, 0.15, 0.2, 0.25], dtype=np.float32)

        risk = bridge_dp_soft_risk(
            distances,
            hard_threshold=0.1,
            soft_threshold=0.2,
            beta=4.0,
        )

        self.assertTrue(np.isclose(risk[0], 1.0))
        self.assertTrue(np.isclose(risk[1], 1.0))
        self.assertGreater(risk[2], 0.0)
        self.assertLess(risk[2], 1.0)
        self.assertTrue(np.isclose(risk[3], 0.0))
        self.assertTrue(np.isclose(risk[4], 0.0))


    def test_critic_score_uses_max_plus_mean_so_single_hard_collision_dominates(self):
        distances = np.array([1.0, 1.0, 0.05, 1.0], dtype=np.float32)
        action_indexes = np.arange(distances.shape[0])

        score = compute_bridge_dp_critic_score_from_distances(
            distances,
            action_indexes,
            hard_threshold=0.1,
            soft_threshold=0.2,
            beta=4.0,
            max_weight=5.0,
            mean_weight=2.0,
            trend_weight=0.0,
        )

        self.assertTrue(np.isclose(score, -5.0 - 2.0 / 3.0))


    def test_critic_score_keeps_distance_trend_reward(self):
        distances = np.array([0.4, 0.6, 0.9], dtype=np.float32)
        action_indexes = np.arange(distances.shape[0])

        score = compute_bridge_dp_critic_score_from_distances(
            distances,
            action_indexes,
            hard_threshold=0.1,
            soft_threshold=0.2,
            trend_weight=0.5,
        )

        self.assertTrue(np.isclose(score, 0.5 * (0.9 - 0.4)))


if __name__ == "__main__":
    unittest.main()
