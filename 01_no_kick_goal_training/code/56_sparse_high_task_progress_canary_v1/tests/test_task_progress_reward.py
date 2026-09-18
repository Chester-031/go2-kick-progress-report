from __future__ import annotations

import math
import random
import struct
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from task_progress_reward import (
    GOAL_CENTER_X_M, PATH_ABS_TOLERANCE, PATH_NORMALIZER_M,
    REWARD_ABS_TOLERANCE, STAGING_OFFSET_M, STAGING_SAFE_MAX_X_M,
    STAGING_SAFE_MAX_Y_M, STAGING_SAFE_MIN_X_M, STAGING_SAFE_MIN_Y_M,
    RewardGeometryError, close_scalar, scalar_progress, scalar_task_geometry,
    strict_goal_xy, torch_progress,
)


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


class TaskProgressRewardTests(unittest.TestCase):
    def test_reward_constants_are_canonical_float32(self) -> None:
        self.assertEqual(GOAL_CENTER_X_M, f32(4.11))
        self.assertEqual(STAGING_OFFSET_M, f32(0.55))
        self.assertEqual(STAGING_SAFE_MAX_X_M, f32(3.65))
        self.assertEqual(STAGING_SAFE_MIN_X_M, f32(-3.65))
        self.assertEqual(STAGING_SAFE_MAX_Y_M, f32(2.15))
        self.assertEqual(STAGING_SAFE_MIN_Y_M, f32(-2.15))
        self.assertEqual(PATH_NORMALIZER_M, f32(math.hypot(8.0, 5.0)))
        self.assertEqual(PATH_ABS_TOLERANCE, 1.0e-5)
        self.assertEqual(REWARD_ABS_TOLERANCE, 2.0e-6)

    def test_exact_goal_center_uses_exact_zero_fallback(self) -> None:
        for side in (-1, 1):
            with self.subTest(side=side):
                ball = (side * GOAL_CENTER_X_M, 0.0)
                geometry = scalar_task_geometry((0.0, 0.0), ball, side)
                self.assertEqual(geometry.raw_staging_target_xy, (ball[0] - side * STAGING_OFFSET_M, 0.0))
                self.assertTrue(math.isfinite(geometry.path_length_m))

    def test_coordinate_projection_is_total_outside_box_and_mirrored(self) -> None:
        probes = [((5.2, 3.4), 1), ((-5.2, -3.4), -1), ((-4.7, 0.0), 1), ((4.7, 0.0), -1), ((0.0, 3.0), 1), ((0.0, -3.0), -1)]
        for ball, side in probes:
            value = scalar_task_geometry((0.0, 0.0), ball, side)
            x, y = value.staging_target_xy
            self.assertLessEqual(STAGING_SAFE_MIN_X_M, x); self.assertLessEqual(x, STAGING_SAFE_MAX_X_M)
            self.assertLessEqual(STAGING_SAFE_MIN_Y_M, y); self.assertLessEqual(y, STAGING_SAFE_MAX_Y_M)
        left = scalar_task_geometry((-0.4, 0.7), (-3.8, 2.3), 1)
        right = scalar_task_geometry((0.4, -0.7), (3.8, -2.3), -1)
        self.assertAlmostEqual(right.staging_target_xy[0], -left.staging_target_xy[0])
        self.assertAlmostEqual(right.staging_target_xy[1], -left.staging_target_xy[1])
        self.assertAlmostEqual(right.path_length_m, left.path_length_m)

    def test_nextafter_projection_boundaries(self) -> None:
        for bound in (STAGING_SAFE_MIN_X_M, STAGING_SAFE_MAX_X_M, STAGING_SAFE_MIN_Y_M, STAGING_SAFE_MAX_Y_M):
            for direction in (-math.inf, math.inf):
                probe = math.nextafter(bound, direction)
                ball = (probe, 0.2) if abs(bound) > 3.0 else (0.2, probe)
                geometry = scalar_task_geometry((0.0, 0.0), ball, 1)
                self.assertTrue(STAGING_SAFE_MIN_X_M <= geometry.staging_target_xy[0] <= STAGING_SAFE_MAX_X_M)
                self.assertTrue(STAGING_SAFE_MIN_Y_M <= geometry.staging_target_xy[1] <= STAGING_SAFE_MAX_Y_M)

    def test_torch_float32_and_scalar_reconstruction_margin(self) -> None:
        rng = random.Random(56001); robot = []; ball = []; sides = []
        for _ in range(20_000):
            robot.append((rng.uniform(-4.2, 4.2), rng.uniform(-2.7, 2.7)))
            ball.append((rng.uniform(-4.3, 4.3), rng.uniform(-2.8, 2.8)))
            sides.append(-1 if rng.random() < 0.5 else 1)
        robot_t = torch.tensor(robot, dtype=torch.float32); ball_t = torch.tensor(ball, dtype=torch.float32)
        post_robot_t = robot_t + torch.tensor([0.013, -0.009], dtype=torch.float32)
        post_ball_t = ball_t + torch.tensor([0.017, 0.006], dtype=torch.float32)
        progress, receipt = torch_progress(robot_t, ball_t, post_robot_t, post_ball_t, torch.tensor(sides, dtype=torch.long))
        max_l_error = 0.0; max_progress_error = 0.0
        for index in range(len(robot)):
            pre, post, scalar_value = scalar_progress(robot_t[index].tolist(), ball_t[index].tolist(), post_robot_t[index].tolist(), post_ball_t[index].tolist(), sides[index])
            max_l_error = max(max_l_error, abs(pre.path_length_m - float(receipt["l_pre_m"][index])), abs(post.path_length_m - float(receipt["l_post_m"][index])))
            max_progress_error = max(max_progress_error, abs(scalar_value - float(progress[index])))
        self.assertLess(max_l_error, PATH_ABS_TOLERANCE)
        self.assertLess(max_progress_error, REWARD_ABS_TOLERANCE)

    def test_goal_and_own_goal_total_reward_signs(self) -> None:
        for side in (-1, 1):
            requested_goal = (side * GOAL_CENTER_X_M, 0.0); own_goal = (-side * GOAL_CENTER_X_M, 0.0)
            for robot_x in (-4.0, -2.0, 0.0, 2.0, 4.0):
                for robot_y in (-2.5, 0.0, 2.5):
                    for ball_x in (-4.0, 0.0, 4.0):
                        _, _, good = scalar_progress((robot_x, robot_y), (ball_x, 0.3), (robot_x, robot_y), requested_goal, side)
                        _, _, own = scalar_progress((robot_x, robot_y), (ball_x, 0.3), (robot_x, robot_y), own_goal, side)
                        self.assertGreater(good + 1.0, 0.0); self.assertLess(own - 1.0, 0.0)

    def test_requested_mouth_overshoot_and_own_goal_runtime_sign_guards(self) -> None:
        for side in (-1, 1):
            for overshoot_m in (0.5, 1.0):
                with self.subTest(side=side, overshoot_m=overshoot_m):
                    pre_ball = (side * (GOAL_CENTER_X_M - 0.05), 0.48)
                    post_ball = (side * (GOAL_CENTER_X_M + overshoot_m), 0.48)
                    _, _, progress = scalar_progress((0.0, 0.0), pre_ball, (0.0, 0.0), post_ball, side)
                    self.assertGreater(progress + 1.0, 0.0)
            with self.subTest(side=side, terminal="own_goal"):
                pre_ball = (-side * (GOAL_CENTER_X_M - 0.05), -0.48)
                post_ball = (-side * (GOAL_CENTER_X_M + 1.0), -0.48)
                _, _, progress = scalar_progress((0.0, 0.0), pre_ball, (0.0, 0.0), post_ball, side)
                self.assertLess(progress - 1.0, 0.0)

    def test_frozen_tolerance_edges(self) -> None:
        self.assertTrue(close_scalar(1.0 + 9.0e-6, 1.0, PATH_ABS_TOLERANCE))
        self.assertFalse(close_scalar(1.0 + 1.1e-5, 1.0, PATH_ABS_TOLERANCE))
        self.assertTrue(close_scalar(1.0 + 1.0e-6, 1.0, REWARD_ABS_TOLERANCE))
        self.assertFalse(close_scalar(1.0 + 3.0e-6, 1.0, REWARD_ABS_TOLERANCE))
        self.assertFalse(close_scalar(True, 1.0, REWARD_ABS_TOLERANCE))

    def test_boolean_coordinates_and_coercible_goal_sides_are_rejected(self) -> None:
        with self.assertRaises(RewardGeometryError):
            scalar_task_geometry((True, 0.0), (0.0, 0.0), 1)
        for side in (True, 1.0, "1"):
            with self.subTest(side=side):
                with self.assertRaises(RewardGeometryError):
                    strict_goal_xy(side)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
