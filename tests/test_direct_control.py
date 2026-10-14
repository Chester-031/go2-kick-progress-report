"""Direct joint control: the policy's 12 outputs drive the joints every
native 50 Hz tick, and the scene's own regularizer reward joins the goal
reward and the potential-based shaping."""

from __future__ import annotations

import unittest

import torch

from tests.fake_isaac import NATIVE_PENALTY, FakeGymEnv, FakeSoccerBaseEnv, install_isaaclab_stub

install_isaaclab_stub()

from robot_soccer import environment  # noqa: E402
from robot_soccer.adr import AutomaticDomainRandomization  # noqa: E402
from robot_soccer.controls import DirectJointControl  # noqa: E402
from robot_soccer.shaping import PotentialShaping, ShapingState, ShapingWeights  # noqa: E402

GAMMA = 0.995
NUM_ENVS = 32
STEPS = 600


def make_direct_env(num_envs: int = NUM_ENVS) -> environment.CurrentTaskProgressHighGoalEnv:
    low_env = environment.FrozenLowInferenceEnv(
        FakeGymEnv(FakeSoccerBaseEnv(num_envs, direct_joint_control=True))
    )
    return environment.CurrentTaskProgressHighGoalEnv(
        low_env=low_env,
        control=DirectJointControl(num_envs=num_envs, device=torch.device("cpu")),
        reset_sampler=AutomaticDomainRandomization(seed=0, probe_fraction=0.0, evaluation_fraction=0.0),
        shaping=PotentialShaping(ShapingWeights(), gamma=GAMMA),
        native_reward_scale=1.0,
    )


class DirectControlTest(unittest.TestCase):
    def test_one_tick_per_action_and_a_50_hz_horizon(self) -> None:
        env = make_direct_env(4)
        self.assertEqual(env.num_actions, 12)
        self.assertEqual(env.low_steps_per_high_action, 1)
        self.assertEqual(env.max_episode_length, 1000)
        self.assertEqual(tuple(env.get_observations().shape), (4, 70))

    def test_reward_is_goal_plus_shaping_plus_the_scenes_regularizers(self) -> None:
        env = make_direct_env()
        generator = torch.Generator().manual_seed(0)
        observation = env.get_observations()
        discounted = torch.zeros(NUM_ENVS, dtype=torch.float64)
        penalty = torch.zeros(NUM_ENVS, dtype=torch.float64)
        age = torch.zeros(NUM_ENVS, dtype=torch.long)
        checked = 0
        for _ in range(STEPS):
            # Push forward with some noise: the robot starts behind the ball.
            action = 0.6 + 0.5 * torch.randn(NUM_ENVS, 12, generator=generator)
            action[:, 8:12] = 0.2 * torch.randn(NUM_ENVS, 4, generator=generator)
            native = -NATIVE_PENALTY * action.double().square().mean(dim=1)
            observation, reward, done, extras = env.step(action)
            episodes = extras["episodes"]
            sparse = torch.zeros(NUM_ENVS, dtype=torch.float64)
            done_ids = torch.nonzero(done).flatten()
            if done_ids.numel():
                sparse[done_ids] = episodes["goal"].double() - episodes["own_goal"].double()
            discounted += GAMMA ** age.double() * (reward.double() - sparse - native)
            penalty += native
            age += 1
            if done_ids.numel():
                start = ShapingState(
                    robot_xy=episodes["reset_robot_xy_m"],
                    ball_xy=episodes["reset_ball_xy_m"],
                    goal_xy=torch.stack(
                        (episodes["goal_side"].float() * 4.11, torch.zeros(done_ids.numel())), dim=1
                    ),
                    base_height_m=torch.full((done_ids.numel(),), 0.30),
                    up_z=torch.ones(done_ids.numel()),
                    joint_limit_proximity=torch.zeros(done_ids.numel()),
                )
                # The shaping still discounts to -Phi(start) ...
                expected = -env.shaping.potential(start).double()
                self.assertLess(float((discounted[done_ids] - expected).abs().max()), 1e-3)
                # ... and the scene's regularizer reward is added and logged.
                self.assertTrue(
                    torch.allclose(episodes["native_reward_return"].double(), penalty[done_ids], atol=1e-4)
                )
                checked += int(done_ids.numel())
                discounted[done_ids] = 0.0
                penalty[done_ids] = 0.0
                age[done_ids] = 0
        self.assertGreater(checked, 20)
        self.assertIsNotNone(observation)


if __name__ == "__main__":
    unittest.main()
