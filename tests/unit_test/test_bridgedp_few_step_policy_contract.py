import unittest
from pathlib import Path


class BridgeDPFewStepPolicyContractTest(unittest.TestCase):
    def test_pointgoal_and_nogoal_pass_the_next_selected_timestep(self):
        policy_path = (
            Path(__file__).resolve().parents[2]
            / "internnav"
            / "model"
            / "basemodel"
            / "bridgedp"
            / "bridgedp_policy.py"
        )
        source = policy_path.read_text(encoding="utf-8")

        self.assertEqual(
            source.count("for step_index, k in enumerate(denoise_timesteps):"),
            2,
        )
        self.assertEqual(source.count("prev_timestep=next_k"), 2)


if __name__ == "__main__":
    unittest.main()

