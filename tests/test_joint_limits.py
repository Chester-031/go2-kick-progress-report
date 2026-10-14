"""Joint hard limits: a small overshoot no longer ends the episode, a large
one still does, and the shaping sees how deep the joints are in the
margin next to their limits."""

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


def rollout(overshoot_rad: float, steps: int = 400) -> dict[str, int]:
    base = FakeSoccerBaseEnv(NUM_ENVS, direct_joint_control=True, joint_fault_overshoot_rad=overshoot_rad)
    env = environment.CurrentTaskProgressHighGoalEnv(
        low_env=environment.FrozenLowInferenceEnv(FakeGymEnv(base)),
        control=DirectJointControl(num_envs=NUM_ENVS, device=torch.device("cpu")),
        reset_sampler=AutomaticDomainRandomization(seed=0),
        shaping=PotentialShaping(ShapingWeights(), gamma=0.995),
        native_reward_scale=1.0,
    )
    counts: dict[str, int] = {}
    for _ in range(steps):
        _, _, _, extras = env.step(torch.zeros(NUM_ENVS, 12))
        episodes = extras["episodes"]
        if episodes:
            for code in episodes["outcome"].tolist():
                name = environment.OUTCOMES[code]
                counts[name] = counts.get(name, 0) + 1
    return counts


class JointLimitTest(unittest.TestCase):
    def test_a_small_overshoot_does_not_end_the_episode(self) -> None:
        self.assertEqual(rollout(0.02).get("joint_hard_limit", 0), 0)

    def test_a_large_overshoot_still_does(self) -> None:
        self.assertGreater(rollout(0.2).get("joint_hard_limit", 0), 0)

    def test_proximity_counts_how_deep_joints_are_in_the_margin(self) -> None:
        base = FakeSoccerBaseEnv(2, direct_joint_control=True)
        env = environment.CurrentTaskProgressHighGoalEnv(
            low_env=environment.FrozenLowInferenceEnv(FakeGymEnv(base)),
            control=DirectJointControl(num_envs=2, device=torch.device("cpu")),
            reset_sampler=AutomaticDomainRandomization(seed=0),
            shaping=PotentialShaping(ShapingWeights(), gamma=0.995),
        )
        joints = base.scene["robot"].data.joint_pos
        joints[:] = 0.0
        # The fake's limits are +-1 rad and the margin is 0.1 rad: a joint at
        # 0.95 is half way in, one at -1.05 is 1.5 margins deep.
        joints[1, 0] = 0.95
        joints[1, 1] = -1.05
        proximity = env._joint_limit_proximity().tolist()
        self.assertEqual(proximity[0], 0.0)
        self.assertAlmostEqual(proximity[1], 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
