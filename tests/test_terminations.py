"""What may end an episode: a goal, an own goal, the ball leaving the
arena, the robot leaving the arena, or the 20 s horizon.  Falling ends
nothing (the robot can get up and carry on) and never cancels a goal;
stepping into a goal box ends nothing either."""

from __future__ import annotations

import unittest

import torch

from tests.fake_isaac import FakeGymEnv, FakeSoccerBaseEnv, install_isaaclab_stub

install_isaaclab_stub()

from robot_soccer import environment  # noqa: E402
from robot_soccer.controls import DirectJointControl  # noqa: E402
from robot_soccer.resets import ResetBatch  # noqa: E402
from robot_soccer.shaping import PotentialShaping, ShapingWeights  # noqa: E402


class FixedStart:
    """Every episode starts the robot at ``robot_xy`` facing +x with the
    ball at ``ball_xy``, attacking the +x goal."""

    def __init__(self, robot_xy: tuple[float, float], ball_xy: tuple[float, float]) -> None:
        self.robot_xy, self.ball_xy = robot_xy, ball_xy

    def __call__(self, count: int) -> ResetBatch:
        return ResetBatch(
            robot_xy=torch.tensor([self.robot_xy]).repeat(count, 1),
            robot_yaw=torch.zeros(count),
            ball_xy=torch.tensor([self.ball_xy]).repeat(count, 1),
            goal_side=torch.ones(count, dtype=torch.long),
        )


def make_env(start: FixedStart, num_envs: int = 4):
    base = FakeSoccerBaseEnv(num_envs, direct_joint_control=True)
    env = environment.CurrentTaskProgressHighGoalEnv(
        low_env=environment.FrozenLowInferenceEnv(FakeGymEnv(base)),
        control=DirectJointControl(num_envs=num_envs, device=torch.device("cpu")),
        reset_sampler=start,
        shaping=PotentialShaping(ShapingWeights(), gamma=0.995),
    )
    return base, env


def run(env, steps: int, action: float = 0.0) -> list[dict]:
    rows = []
    for _ in range(steps):
        _, _, _, extras = env.step(torch.full((env.num_envs, 12), action))
        episodes = extras["episodes"]
        if episodes:
            for index in range(len(episodes["outcome"])):
                rows.append({name: value[index].tolist() for name, value in episodes.items()})
    return rows


class TerminationTest(unittest.TestCase):
    def test_lying_on_the_ground_ends_nothing_and_is_recorded(self) -> None:
        self.assertNotIn("fall", environment.OUTCOMES)
        base, env = make_env(FixedStart((-2.0, 0.0), (0.0, 1.5)))
        base.trunk_on_ground[:] = True
        # Two seconds on the ground: no episode ends.
        self.assertEqual(run(env, 100), [])
        base.trunk_on_ground[:] = False
        rows = run(env, 1000)
        self.assertTrue(rows, "episodes should end at the 20 s horizon")
        first = rows[0]
        self.assertEqual(environment.OUTCOMES[first["outcome"]], "timeout")
        self.assertGreaterEqual(first["falls"], 1)
        self.assertGreaterEqual(first["seconds_on_ground"], 2.0)

    def test_a_goal_scored_while_falling_still_counts(self) -> None:
        # Robot right behind the ball on the goal line; push forward.
        base, env = make_env(FixedStart((3.6, 0.0), (3.95, 0.0)), num_envs=2)
        base.trunk_on_ground[:] = True
        rows = run(env, 60, action=0.6)
        self.assertTrue(rows)
        self.assertEqual(environment.OUTCOMES[rows[0]["outcome"]], "goal")

    def test_stepping_into_the_goal_box_ends_nothing_but_leaving_the_arena_does(self) -> None:
        base, env = make_env(FixedStart((-2.0, 0.0), (0.0, 1.5)), num_envs=2)
        robot = base.scene["robot"].data
        origins = base.scene.env_origins
        # Slot 0 stands inside the +x goal box, slot 1 behind the back wall.
        robot.root_pos_w[0, 0] = origins[0, 0] + 4.3
        robot.root_pos_w[1, 0] = origins[1, 0] + 4.8
        rows = run(env, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(environment.OUTCOMES[rows[0]["outcome"]], "boundary")
        self.assertEqual(rows[0]["terminal_robot_xy_m"][0] > 4.7, True)


if __name__ == "__main__":
    unittest.main()
