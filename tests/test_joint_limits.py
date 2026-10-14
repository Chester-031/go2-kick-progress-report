"""Joint hard limits end nothing: the simulator enforces them, and slamming a
joint into its stop only matters for a real robot.  Each episode records
its largest overshoot past a limit, as a diagnostic."""

from __future__ import annotations

import unittest

import torch

from tests.fake_isaac import FakeGymEnv, FakeSoccerBaseEnv, install_isaaclab_stub

install_isaaclab_stub()

from robot_soccer import environment  # noqa: E402
from robot_soccer.adr import AutomaticDomainRandomization  # noqa: E402
from robot_soccer.controls import DirectJointControl  # noqa: E402
from robot_soccer.shaping import PotentialShaping, ShapingWeights  # noqa: E402

NUM_ENVS = 32


class JointLimitTest(unittest.TestCase):
    def test_overshooting_a_joint_limit_ends_nothing_and_is_recorded(self) -> None:
        # The fake drives a joint 0.01 rad past its limit now and then.
        base = FakeSoccerBaseEnv(NUM_ENVS, direct_joint_control=True)
        env = environment.CurrentTaskProgressHighGoalEnv(
            low_env=environment.FrozenLowInferenceEnv(FakeGymEnv(base)),
            control=DirectJointControl(num_envs=NUM_ENVS, device=torch.device("cpu")),
            reset_sampler=AutomaticDomainRandomization(seed=0),
            shaping=PotentialShaping(ShapingWeights(), gamma=0.995),
            native_reward_scale=1.0,
        )
        self.assertNotIn("joint_hard_limit", environment.OUTCOMES)
        overshoots = []
        for _ in range(400):
            _, _, _, extras = env.step(torch.zeros(NUM_ENVS, 12))
            if extras["episodes"]:
                overshoots += extras["episodes"]["max_joint_overshoot_rad"].tolist()
        self.assertTrue(overshoots, "no episode finished")
        self.assertTrue(any(abs(value - 0.01) < 1e-5 for value in overshoots), "an overshoot was not recorded")
        self.assertTrue(all(value >= 0.0 for value in overshoots))


if __name__ == "__main__":
    unittest.main()
