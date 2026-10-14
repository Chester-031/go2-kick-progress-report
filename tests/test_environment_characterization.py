"""Characterization of the soccer environment wrapper on the fake scene.

The wrapper's episode bookkeeping (terminal outcome precedence, the frozen
terminal state, contact timing, progress reward) is the part a speed
refactor can silently break.  This test drives the wrapper on the
deterministic fake Isaac scene with scripted resets and a scripted high
policy and compares every finished episode, every reward sum and the final
observation with a fixture recorded from the Stage B wrapper.

Regenerate the fixture only when a behavior change is intended:
``REGENERATE_FIXTURE=1 python -m unittest tests.test_environment_characterization``.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import unittest

import torch

from tests.fake_isaac import (
    FakeGymEnv,
    FakeSoccerBaseEnv,
    frozen_low_actor,
    install_isaaclab_stub,
    scripted_latent_action,
)

install_isaaclab_stub()

from robot_soccer import environment as environment_module  # noqa: E402
from robot_soccer.config import (  # noqa: E402
    COMMAND_AXIS_RANGES,
    COMMAND_RATE_LIMITS_PER_SECOND,
    COMMAND_VERTICES,
)
from robot_soccer.resets import ResetSample  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "environment_characterization.json"
NUM_ENVS = 24
STEPS = 230
# Episode fields every version of the wrapper must reproduce exactly.
COMPARED_FIELDS = (
    "goal_side",
    "reset_robot_xy_m",
    "reset_robot_yaw_rad",
    "reset_ball_xy_m",
    "outcome",
    "goal",
    "own_goal",
    "boundary",
    "fall",
    "joint_hard_limit",
    "timeout",
    "unsafe_pre_goal",
    "requested_goal_line_reached",
    "contact_seen",
    "first_true_contact_native_tick",
    "maximum_contact_force_n",
    "terminal_contact_force_n",
    "terminal_base_contact_force_n",
    "terminal_joint_hard_limit_excess_rad",
    "native_safe_guard_changed_elements",
    "native_safe_guard_max_correction_rad",
    "episode_high_ticks",
    "episode_native_ticks",
    "terminal_robot_xy_m",
    "terminal_ball_xy_m",
    "initial_l_m",
    "final_l_m",
    "cumulative_progress_reward",
    "requested_goal_axis_ball_displacement_m",
    "abs_lateral_ball_drift_m",
)
# (ball distance from the goal along -x, robot offset from the ball, yaw):
# a quick goal, a slow approach, a run out of bounds, an own goal, a
# robot facing away that times out, and a near-goal shot from an angle.
SCRIPTED_PLACEMENTS = (
    ((3.45, 0.05), (2.95, 0.05), 0.0),
    ((2.10, -0.30), (1.30, -0.70), 0.4),
    ((0.0, 2.10), (0.0, 1.80), 1.57),
    ((-3.55, 0.0), (-3.05, 0.0), math.pi),
    ((-1.0, 1.0), (-2.5, -1.2), -2.4),
    ((3.30, 0.35), (2.70, 0.70), -0.5),
)


def scripted_reset(seed: int, serial: int) -> tuple[ResetSample, str]:
    """Cycle through SCRIPTED_PLACEMENTS with a small serial-keyed shift."""

    del seed
    ball, robot, yaw = SCRIPTED_PLACEMENTS[(serial - 1) % len(SCRIPTED_PLACEMENTS)]
    shift = 0.01 * ((serial * 37) % 11 - 5)
    return (
        ResetSample(
            robot_xy=(robot[0] + shift, robot[1] - shift),
            robot_yaw=yaw,
            ball_xy=(ball[0], ball[1] + shift),
            goal_side=1,
        ),
        "SCRIPTED",
    )


def run_scripted_rollout(make_env) -> dict:
    """Run the scripted rollout on a wrapper built by ``make_env(low_env,
    actor)`` and return everything the fixture records."""

    base = FakeSoccerBaseEnv(NUM_ENVS)
    env = make_env(environment_module.FrozenLowInferenceEnv(FakeGymEnv(base)), frozen_low_actor())
    observation = env.get_observations()
    events: list[dict] = []
    reward_sum = 0.0
    done_count = 0
    for _ in range(STEPS):
        observation, reward, done, extras = env.step(scripted_latent_action(observation))
        reward_sum += float(reward.sum())
        done_count += int(done.sum())
        events.extend(extras["terminal_events"])
    return {
        "events": [{name: event[name] for name in COMPARED_FIELDS} for event in events],
        "reward_sum": reward_sum,
        "done_count": done_count,
        "final_observation_sum": float(observation.double().sum()),
    }


def make_wrapper(low_env, actor):
    environment_module.sample_training_reset = scripted_reset
    return environment_module.CurrentTaskProgressHighGoalEnv(
        low_env=low_env,
        frozen_low_actor=actor,
        seed=7,
        command_vertices=COMMAND_VERTICES,
        command_axis_ranges=COMMAND_AXIS_RANGES,
        command_rate_limits=COMMAND_RATE_LIMITS_PER_SECOND,
    )


def assert_close(test: unittest.TestCase, expected, actual, path: str) -> None:
    if isinstance(expected, float) or isinstance(actual, float):
        test.assertTrue(
            math.isclose(float(expected), float(actual), rel_tol=1.0e-5, abs_tol=1.0e-5),
            f"{path}: expected {expected}, got {actual}",
        )
    elif isinstance(expected, list):
        test.assertEqual(len(expected), len(actual), path)
        for index, (left, right) in enumerate(zip(expected, actual)):
            assert_close(test, left, right, f"{path}[{index}]")
    elif isinstance(expected, dict):
        test.assertEqual(sorted(expected), sorted(actual), path)
        for key in expected:
            assert_close(test, expected[key], actual[key], f"{path}.{key}")
    else:
        test.assertEqual(expected, actual, path)


class EnvironmentCharacterizationTest(unittest.TestCase):
    def test_scripted_rollout_matches_fixture(self) -> None:
        torch.manual_seed(0)
        result = run_scripted_rollout(make_wrapper)
        outcomes = {event["outcome"] for event in result["events"]}
        # The script must keep exercising every terminal branch.
        self.assertTrue({"goal", "own_goal", "boundary", "fall", "timeout"} <= outcomes, outcomes)
        if os.environ.get("REGENERATE_FIXTURE") == "1":
            FIXTURE.parent.mkdir(parents=True, exist_ok=True)
            FIXTURE.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
        assert_close(self, expected, result, "rollout")


if __name__ == "__main__":
    unittest.main()
