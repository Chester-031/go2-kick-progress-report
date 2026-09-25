from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_soccer.commands import (  # noqa: E402
    limit_high_command_tensor,
    project_to_convex_hull,
)
from robot_soccer.config import (  # noqa: E402
    COMMAND_AXIS_RANGES,
    COMMAND_RATE_LIMITS_PER_SECOND,
    COMMAND_VERTICES,
    POLICY_OBSERVATION_DIM,
)
from robot_soccer.observations import build_entity_observation  # noqa: E402
from robot_soccer.resets import (  # noqa: E402
    FULL,
    REFERENCE,
    fixed_evaluation_samples,
    is_hard_goal_facing_ball_behind,
    sample_full_reset,
    sample_training_reset,
    stationary_training_regime,
)
from robot_soccer.rewards import (  # noqa: E402
    PATH_NORMALIZER_M,
    scalar_progress,
    terminal_sparse_reward,
    torch_progress,
    torch_total_reward,
)


class ResetTests(unittest.TestCase):
    def test_stationary_pattern_is_exactly_75_25_and_side_balanced(self) -> None:
        assignments = [stationary_training_regime(index) for index in range(1, 9)]
        self.assertEqual(
            assignments,
            [
                (REFERENCE, -1),
                (REFERENCE, 1),
                (REFERENCE, -1),
                (FULL, None),
                (REFERENCE, 1),
                (REFERENCE, -1),
                (REFERENCE, 1),
                (FULL, None),
            ],
        )
        self.assertEqual(Counter(item[0] for item in assignments), {REFERENCE: 6, FULL: 2})
        self.assertEqual(
            Counter(item[1] for item in assignments if item[0] == REFERENCE),
            {-1: 3, 1: 3},
        )

    def test_training_sampling_is_deterministic(self) -> None:
        for serial in range(1, 17):
            left = sample_training_reset(83_001 * 1_000_003 + serial, serial)
            right = sample_training_reset(83_001 * 1_000_003 + serial, serial)
            self.assertEqual(left, right)

    def test_full_sample_respects_support(self) -> None:
        sample = sample_full_reset(12345)
        distance = math.dist(sample.robot_xy, sample.ball_xy)
        self.assertGreaterEqual(distance, 0.65)
        self.assertLessEqual(distance, 2.40)
        self.assertLessEqual(abs(sample.robot_xy[0]), 3.65)
        self.assertLessEqual(abs(sample.robot_xy[1]), 2.15)
        self.assertIn(sample.goal_side, (-1, 1))

    def test_fixed_evaluation_contains_declared_hard_subset(self) -> None:
        samples = fixed_evaluation_samples(32, 75_001, hard_case_count=4)
        self.assertEqual(len(samples), 32)
        self.assertEqual(
            sum(is_hard_goal_facing_ball_behind(sample) for sample in samples),
            4,
        )


class ObservationTests(unittest.TestCase):
    def test_coordinate_repaired_tail_is_robot_from_ball_in_goal_frame(self) -> None:
        low = torch.zeros(1, 48, dtype=torch.float32)
        robot = torch.tensor([[0.0, 1.0, 0.30]], dtype=torch.float32)
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        ball = torch.tensor([[1.0, 0.0, 0.11]], dtype=torch.float32)
        velocity = torch.zeros(1, 3, dtype=torch.float32)
        goal = torch.tensor([[4.0, 0.0]], dtype=torch.float32)
        origin = torch.zeros(1, 2, dtype=torch.float32)
        observation, state = build_entity_observation(
            low_observation=low,
            robot_position_w=robot,
            robot_quaternion_wxyz=quaternion,
            ball_position_w=ball,
            ball_velocity_w=velocity,
            goal_xy_w=goal,
            env_origin_xy_w=origin,
            time_remaining=torch.ones(1),
            contact_seen=torch.zeros(1, dtype=torch.bool),
        )
        self.assertEqual(tuple(observation.shape), (1, POLICY_OBSERVATION_DIM))
        self.assertAlmostEqual(float(observation[0, -2]), -1.0 / 8.0, places=7)
        self.assertAlmostEqual(float(observation[0, -1]), 1.0 / 2.5, places=7)
        torch.testing.assert_close(
            state["robot_from_ball_goal_axis"],
            torch.tensor([-1.0]),
        )
        torch.testing.assert_close(
            state["robot_from_ball_goal_left"],
            torch.tensor([1.0]),
        )

    def test_coordinate_tail_is_independent_of_robot_yaw(self) -> None:
        common = {
            "low_observation": torch.zeros(1, 48),
            "robot_position_w": torch.tensor([[0.0, 1.0, 0.30]]),
            "ball_position_w": torch.tensor([[1.0, 0.0, 0.11]]),
            "ball_velocity_w": torch.zeros(1, 3),
            "goal_xy_w": torch.tensor([[4.0, 0.0]]),
            "env_origin_xy_w": torch.zeros(1, 2),
            "time_remaining": torch.ones(1),
            "contact_seen": torch.zeros(1, dtype=torch.bool),
        }
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        yaw_90 = torch.tensor(
            [[math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)]]
        )
        first, _ = build_entity_observation(
            robot_quaternion_wxyz=identity,
            **common,
        )
        second, _ = build_entity_observation(
            robot_quaternion_wxyz=yaw_90,
            **common,
        )
        torch.testing.assert_close(first[:, -2:], second[:, -2:])


class RewardTests(unittest.TestCase):
    def test_scalar_and_tensor_progress_match(self) -> None:
        pre_robot = (0.0, 0.0)
        pre_ball = (1.0, 0.0)
        post_robot = (0.25, 0.0)
        post_ball = (1.10, 0.0)
        _, _, scalar = scalar_progress(
            pre_robot,
            pre_ball,
            post_robot,
            post_ball,
            1,
        )
        tensor, _ = torch_progress(
            torch.tensor([pre_robot]),
            torch.tensor([pre_ball]),
            torch.tensor([post_robot]),
            torch.tensor([post_ball]),
            torch.tensor([1]),
        )
        self.assertAlmostEqual(float(tensor.item()), scalar, places=6)
        self.assertGreater(scalar, 0.0)
        self.assertAlmostEqual(
            scalar,
            0.25 / PATH_NORMALIZER_M,
            places=6,
        )

    def test_strict_terminal_reward_and_total(self) -> None:
        self.assertEqual(terminal_sparse_reward(goal=True, own_goal=False), 1.0)
        self.assertEqual(terminal_sparse_reward(goal=False, own_goal=True), -1.0)
        self.assertEqual(terminal_sparse_reward(goal=False, own_goal=False), 0.0)
        total, parts = torch_total_reward(
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([1]),
            goal=torch.tensor([True]),
            own_goal=torch.tensor([False]),
        )
        torch.testing.assert_close(total, torch.tensor([1.0]))
        torch.testing.assert_close(parts["strict_goal_reward"], torch.tensor([1.0]))


class CommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vertices = torch.tensor(COMMAND_VERTICES, dtype=torch.float32)
        self.ranges = torch.tensor(COMMAND_AXIS_RANGES, dtype=torch.float32)
        self.rates = torch.tensor(
            COMMAND_RATE_LIMITS_PER_SECOND,
            dtype=torch.float32,
        )

    def test_projection_leaves_vertices_unchanged(self) -> None:
        projected = project_to_convex_hull(self.vertices, self.vertices)
        torch.testing.assert_close(projected, self.vertices)

    def test_rate_limit_applies_after_command_surface_projection(self) -> None:
        applied = limit_high_command_tensor(
            normalized_action=torch.tensor([[1.0, 0.0, 0.0]]),
            previous_command=torch.zeros(1, 3),
            vertices=self.vertices,
            axis_ranges=self.ranges,
            rate_limits_per_second=self.rates,
            high_control_dt=0.1,
        )
        torch.testing.assert_close(
            applied,
            torch.tensor([[0.1, 0.0, 0.0]]),
            rtol=0.0,
            atol=1.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
