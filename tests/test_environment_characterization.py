"""Characterization of the soccer environment wrapper on the fake scene.

The wrapper's episode bookkeeping (terminal outcome precedence, the frozen
terminal state, contact timing, progress reward) is the part a speed
refactor can silently break.  This test drives the wrapper on the
deterministic fake Isaac scene with scripted resets and a scripted high
policy and compares every finished episode, every reward sum and the final
observation with a fixture recorded from the wrapper (first from Stage B,
re-recorded once for the exact command projection).

Regenerate the fixture only when a behavior change is intended:
``REGENERATE_FIXTURE=1 python -m unittest tests.test_environment_characterization``.
"""

from __future__ import annotations

from dataclasses import dataclass
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
from robot_soccer.commands import CommandSurface  # noqa: E402
from robot_soccer.controls import HeldCommandControl  # noqa: E402
from robot_soccer.resets import ResetBatch  # noqa: E402
from robot_soccer.shaping import PotentialShaping, ShapingState, ShapingWeights  # noqa: E402

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


@dataclass(frozen=True)
class ScriptedStart:
    robot_xy: tuple[float, float]
    robot_yaw: float
    ball_xy: tuple[float, float]
    goal_side: int


def scripted_reset(serial: int) -> ScriptedStart:
    """Cycle through SCRIPTED_PLACEMENTS with a small serial-keyed shift."""

    ball, robot, yaw = SCRIPTED_PLACEMENTS[(serial - 1) % len(SCRIPTED_PLACEMENTS)]
    shift = 0.01 * ((serial * 37) % 11 - 5)
    return ScriptedStart(
        robot_xy=(robot[0] + shift, robot[1] - shift),
        robot_yaw=yaw,
        ball_xy=(ball[0], ball[1] + shift),
        goal_side=1,
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
        events.extend(episode_rows(extras["episodes"]))
    return {
        "events": [{name: event[name] for name in COMPARED_FIELDS} for event in events],
        "reward_sum": reward_sum,
        "done_count": done_count,
        "final_observation_sum": float(observation.double().sum()),
    }


def episode_rows(columns: dict) -> list[dict]:
    """Turn one step's columnar episode batch into per-episode dicts in the
    fixture's format (outcome names, None for 'no contact')."""

    if not columns:
        return []
    rows = []
    for index in range(len(columns["outcome"])):
        row = {name: value[index].tolist() for name, value in columns.items()}
        row["outcome"] = environment_module.OUTCOMES[row["outcome"]]
        if row["first_true_contact_native_tick"] < 0:
            row["first_true_contact_native_tick"] = None
        rows.append(row)
    return rows


class ScriptedResetSampler:
    """``scripted_reset`` for a running serial, as a reset sampler."""

    def __init__(self) -> None:
        self.serial = 0

    def __call__(self, count: int) -> ResetBatch:
        samples = []
        for _ in range(count):
            self.serial += 1
            samples.append(scripted_reset(self.serial))
        return ResetBatch(
            robot_xy=torch.tensor([sample.robot_xy for sample in samples]),
            robot_yaw=torch.tensor([sample.robot_yaw for sample in samples]),
            ball_xy=torch.tensor([sample.ball_xy for sample in samples]),
            goal_side=torch.tensor([sample.goal_side for sample in samples]),
        )


def held_command_control(low_env, actor) -> HeldCommandControl:
    """The Stage B hierarchy on the fake scene."""

    device = torch.device(low_env.device)
    return HeldCommandControl(
        low_actor=actor,
        command_surface=CommandSurface(
            vertices=COMMAND_VERTICES,
            axis_ranges=COMMAND_AXIS_RANGES,
            rate_limits_per_second=COMMAND_RATE_LIMITS_PER_SECOND,
            high_control_dt=0.1,
            device=device,
        ),
        command=environment_module.writable_command(low_env.unwrapped),
        num_envs=low_env.num_envs,
        device=device,
    )


def make_wrapper(low_env, actor):
    return environment_module.CurrentTaskProgressHighGoalEnv(
        low_env=low_env,
        control=held_command_control(low_env, actor),
        reset_sampler=ScriptedResetSampler(),
        shaping=PotentialShaping(ShapingWeights(), gamma=0.997),
    )


def assert_close(test: unittest.TestCase, expected, actual, path: str) -> None:
    if path.endswith("_yaw_rad"):
        # A heading of +pi and -pi is the same pose.
        difference = math.remainder(float(expected) - float(actual), 2.0 * math.pi)
        test.assertLess(abs(difference), 1.0e-5, f"{path}: expected {expected}, got {actual}")
    elif isinstance(expected, float) or isinstance(actual, float):
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


class ShapingInEnvironmentTest(unittest.TestCase):
    def test_every_episode_discounts_to_minus_its_start_potential(self) -> None:
        """Ng et al.'s invariance, end to end: for each finished episode the
        discounted shaping reward (reward minus the +-1 goal terms) adds up
        to -Phi(start state), so shaping cannot favour any behaviour."""

        gamma = 0.997
        base = FakeSoccerBaseEnv(NUM_ENVS)
        env = make_wrapper(environment_module.FrozenLowInferenceEnv(FakeGymEnv(base)), frozen_low_actor())
        observation = env.get_observations()
        discounted = torch.zeros(NUM_ENVS, dtype=torch.float64)
        age = torch.zeros(NUM_ENVS, dtype=torch.long)
        checked = 0
        for _ in range(STEPS):
            observation, reward, done, extras = env.step(scripted_latent_action(observation))
            episodes = extras["episodes"]
            sparse = torch.zeros(NUM_ENVS, dtype=torch.float64)
            done_ids = torch.nonzero(done).flatten()
            if done_ids.numel():
                sparse[done_ids] = episodes["goal"].double() - episodes["own_goal"].double()
            discounted += gamma ** age.double() * (reward.double() - sparse)
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
                expected = -env.shaping.potential(start).double()
                self.assertLess(float((discounted[done_ids] - expected).abs().max()), 1e-3)
                checked += int(done_ids.numel())
                discounted[done_ids] = 0.0
                age[done_ids] = 0
        self.assertGreater(checked, 20)


if __name__ == "__main__":
    unittest.main()
