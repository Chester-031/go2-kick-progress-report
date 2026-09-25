"""Task-progress shaping plus strict goal/own-goal terminal reward."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Sequence

import torch


def _canonical_float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


GOAL_CENTER_X_M = _canonical_float32(4.11)
GOAL_CENTER_Y_M = _canonical_float32(0.0)
STAGING_OFFSET_M = _canonical_float32(0.55)
STAGING_SAFE_MIN_X_M = _canonical_float32(-3.65)
STAGING_SAFE_MAX_X_M = _canonical_float32(3.65)
STAGING_SAFE_MIN_Y_M = _canonical_float32(-2.15)
STAGING_SAFE_MAX_Y_M = _canonical_float32(2.15)
PATH_NORMALIZER_M = _canonical_float32(math.hypot(8.0, 5.0))
TERMINAL_GOAL_HALF_WIDTH_M = _canonical_float32(0.49)
ROBOT_FIELD_HALF_LENGTH_M = _canonical_float32(4.0)
ROBOT_FIELD_HALF_WIDTH_M = _canonical_float32(2.5)
BALL_SIDE_BOUNDARY_M = _canonical_float32(2.61)
CONTACT_FORCE_THRESHOLD_N = _canonical_float32(1.0)
BASE_FALL_FORCE_THRESHOLD_N = _canonical_float32(1.0)
JOINT_HARD_LIMIT_TOLERANCE_RAD = _canonical_float32(1.0e-5)


class RewardGeometryError(ValueError):
    """A state cannot define the task-progress potential."""


@dataclass(frozen=True)
class ScalarGeometry:
    goal_xy: tuple[float, float]
    raw_staging_target_xy: tuple[float, float]
    staging_target_xy: tuple[float, float]
    staging_projection_applied: bool
    path_length_m: float


def _xy(value: Sequence[float], label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise RewardGeometryError(f"{label} must be a two-vector.")
    if any(
        not isinstance(item, (int, float)) or isinstance(item, bool)
        for item in value
    ):
        raise RewardGeometryError(f"{label} must contain only real numbers.")
    result = float(value[0]), float(value[1])
    if not all(math.isfinite(item) for item in result):
        raise RewardGeometryError(f"{label} is non-finite.")
    return result


def strict_goal_xy(goal_side: int) -> tuple[float, float]:
    if type(goal_side) is not int or goal_side not in (-1, 1):
        raise RewardGeometryError("goal_side must be exactly -1 or 1.")
    return float(goal_side) * GOAL_CENTER_X_M, GOAL_CENTER_Y_M


def scalar_task_geometry(
    robot_xy: Sequence[float],
    ball_xy: Sequence[float],
    goal_side: int,
) -> ScalarGeometry:
    """Compute L(s) = robot-to-staging + ball-to-goal."""

    robot = _xy(robot_xy, "robot_xy")
    ball = _xy(ball_xy, "ball_xy")
    goal = strict_goal_xy(goal_side)
    goal_dx = goal[0] - ball[0]
    goal_dy = goal[1] - ball[1]
    ball_goal = math.hypot(goal_dx, goal_dy)
    if not math.isfinite(ball_goal):
        raise RewardGeometryError("Goal distance is non-finite.")
    if ball_goal == 0.0:
        unit_x, unit_y = float(goal_side), 0.0
    else:
        unit_x, unit_y = goal_dx / ball_goal, goal_dy / ball_goal
    raw_staging = (
        ball[0] - STAGING_OFFSET_M * unit_x,
        ball[1] - STAGING_OFFSET_M * unit_y,
    )
    staging = (
        min(STAGING_SAFE_MAX_X_M, max(STAGING_SAFE_MIN_X_M, raw_staging[0])),
        min(STAGING_SAFE_MAX_Y_M, max(STAGING_SAFE_MIN_Y_M, raw_staging[1])),
    )
    path = math.hypot(robot[0] - staging[0], robot[1] - staging[1]) + ball_goal
    if not all(math.isfinite(item) for item in (*raw_staging, *staging, path)):
        raise RewardGeometryError("Task geometry is non-finite.")
    return ScalarGeometry(
        goal_xy=goal,
        raw_staging_target_xy=raw_staging,
        staging_target_xy=staging,
        staging_projection_applied=raw_staging != staging,
        path_length_m=path,
    )


def scalar_progress(
    pre_robot_xy: Sequence[float],
    pre_ball_xy: Sequence[float],
    post_robot_xy: Sequence[float],
    post_ball_xy: Sequence[float],
    goal_side: int,
) -> tuple[ScalarGeometry, ScalarGeometry, float]:
    pre = scalar_task_geometry(pre_robot_xy, pre_ball_xy, goal_side)
    post = scalar_task_geometry(post_robot_xy, post_ball_xy, goal_side)
    progress = (pre.path_length_m - post.path_length_m) / PATH_NORMALIZER_M
    if not math.isfinite(progress):
        raise RewardGeometryError("Progress is non-finite.")
    return pre, post, progress


def torch_task_geometry(
    robot_xy: torch.Tensor,
    ball_xy: torch.Tensor,
    goal_side: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized runtime formula in the simulator tensor dtype."""

    if (
        robot_xy.ndim != 2
        or robot_xy.shape[1] != 2
        or ball_xy.shape != robot_xy.shape
    ):
        raise RewardGeometryError("Robot/ball coordinates must have shape [N,2].")
    if goal_side.ndim != 1 or goal_side.shape[0] != robot_xy.shape[0]:
        raise RewardGeometryError("goal_side must have shape [N].")
    if not bool(torch.isfinite(robot_xy).all()) or not bool(
        torch.isfinite(ball_xy).all()
    ):
        raise RewardGeometryError("Runtime physical state is non-finite.")
    if not bool(((goal_side == -1) | (goal_side == 1)).all()):
        raise RewardGeometryError("goal_side must contain only -1 or 1.")

    goal = torch.zeros_like(robot_xy)
    goal[:, 0] = goal_side.to(robot_xy.dtype) * GOAL_CENTER_X_M
    delta = goal - ball_xy
    distance = torch.linalg.vector_norm(delta, dim=1)
    regular = distance != 0.0
    safe_distance = torch.where(regular, distance, torch.ones_like(distance))
    unit = delta / safe_distance.unsqueeze(1)
    fallback = torch.zeros_like(unit)
    fallback[:, 0] = goal_side.to(robot_xy.dtype)
    unit = torch.where(regular.unsqueeze(1), unit, fallback)

    raw_staging = ball_xy - STAGING_OFFSET_M * unit
    staging = torch.stack(
        (
            raw_staging[:, 0].clamp(
                STAGING_SAFE_MIN_X_M,
                STAGING_SAFE_MAX_X_M,
            ),
            raw_staging[:, 1].clamp(
                STAGING_SAFE_MIN_Y_M,
                STAGING_SAFE_MAX_Y_M,
            ),
        ),
        dim=1,
    )
    path = torch.linalg.vector_norm(robot_xy - staging, dim=1) + distance
    if not bool(torch.isfinite(staging).all()) or not bool(
        torch.isfinite(path).all()
    ):
        raise RewardGeometryError("Runtime task geometry is non-finite.")
    return path, raw_staging, staging, goal


def torch_progress(
    pre_robot_xy: torch.Tensor,
    pre_ball_xy: torch.Tensor,
    post_robot_xy: torch.Tensor,
    post_ball_xy: torch.Tensor,
    goal_side: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pre_l, pre_raw_target, pre_target, goal = torch_task_geometry(
        pre_robot_xy,
        pre_ball_xy,
        goal_side,
    )
    post_l, post_raw_target, post_target, post_goal = torch_task_geometry(
        post_robot_xy,
        post_ball_xy,
        goal_side,
    )
    if not bool(torch.equal(goal, post_goal)):
        raise RewardGeometryError("Goal changed within a transition.")
    progress = (pre_l - post_l) / PATH_NORMALIZER_M
    if not bool(torch.isfinite(progress).all()):
        raise RewardGeometryError("Runtime progress is non-finite.")
    return progress, {
        "goal_xy": goal,
        "raw_staging_target_pre_xy": pre_raw_target,
        "raw_staging_target_post_xy": post_raw_target,
        "staging_target_pre_xy": pre_target,
        "staging_target_post_xy": post_target,
        "staging_projection_pre_applied": (pre_raw_target != pre_target).any(dim=1),
        "staging_projection_post_applied": (post_raw_target != post_target).any(dim=1),
        "l_pre_m": pre_l,
        "l_post_m": post_l,
    }


def terminal_sparse_reward(*, goal: bool, own_goal: bool) -> float:
    if type(goal) is not bool or type(own_goal) is not bool:
        raise RewardGeometryError("goal and own_goal must be Booleans.")
    if goal and own_goal:
        raise RewardGeometryError("goal and own_goal are mutually exclusive.")
    return 1.0 if goal else -1.0 if own_goal else 0.0


def torch_total_reward(
    pre_robot_xy: torch.Tensor,
    pre_ball_xy: torch.Tensor,
    post_robot_xy: torch.Tensor,
    post_ball_xy: torch.Tensor,
    goal_side: torch.Tensor,
    *,
    goal: torch.Tensor,
    own_goal: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine progress shaping with strict +1/-1 terminal outcomes."""

    if goal.shape != goal_side.shape or own_goal.shape != goal_side.shape:
        raise RewardGeometryError("Terminal masks must have shape [N].")
    if goal.dtype != torch.bool or own_goal.dtype != torch.bool:
        raise RewardGeometryError("Terminal masks must be Boolean.")
    if bool((goal & own_goal).any()):
        raise RewardGeometryError("Goal and own-goal masks overlap.")
    progress, geometry = torch_progress(
        pre_robot_xy,
        pre_ball_xy,
        post_robot_xy,
        post_ball_xy,
        goal_side,
    )
    strict = goal.to(progress.dtype) - own_goal.to(progress.dtype)
    total = progress + strict
    geometry = dict(geometry)
    geometry["progress_reward"] = progress
    geometry["strict_goal_reward"] = strict
    return total, geometry
