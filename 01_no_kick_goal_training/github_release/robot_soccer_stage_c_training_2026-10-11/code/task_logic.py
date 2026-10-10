"""EXPLANATORY PROJECTION — NOT THE PRODUCTION TRAINER.

Pure reward geometry and same-tick terminal precedence for Stage C.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from parameters import (
    BALL_SIDE_BOUNDARY_M,
    BASE_FALL_FORCE_THRESHOLD_N,
    FIELD_HALF_LENGTH_M,
    FIELD_HALF_WIDTH_M,
    GOAL_CENTER_ABS_X_M,
    GOAL_HALF_WIDTH_FOR_BALL_CENTER_M,
    JOINT_HARD_LIMIT_TOLERANCE_RAD,
    STAGING_OFFSET_M,
    STAGING_SAFE_X_M,
    STAGING_SAFE_Y_M,
    TRUE_CONTACT_FORCE_THRESHOLD_N,
)


PATH_NORMALIZER_M = math.hypot(8.0, 5.0)


@dataclass(frozen=True)
class TaskGeometry:
    selected_goal_xy: tuple[float, float]
    raw_staging_target_xy: tuple[float, float]
    staging_target_xy: tuple[float, float]
    staging_projection_applied: bool
    path_length_m: float


@dataclass(frozen=True)
class TerminalInputs:
    goal_side: int
    robot_xy: tuple[float, float]
    ball_xy: tuple[float, float]
    contact_seen: bool
    base_contact_force_n: float
    joint_limit_excess_rad: float
    timeout: bool = False


@dataclass(frozen=True)
class TerminalDecision:
    goal: bool
    own_goal: bool
    boundary: bool
    fall: bool
    joint_hard_limit: bool
    timeout: bool
    done: bool
    outcome: str | None


def _validate_xy(value: tuple[float, float], label: str) -> None:
    if len(value) != 2 or not all(math.isfinite(item) for item in value):
        raise ValueError(f"{label} must be a finite two-vector")


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return min(bounds[1], max(bounds[0], value))


def task_geometry(
    robot_xy: tuple[float, float],
    ball_xy: tuple[float, float],
    goal_side: int,
) -> TaskGeometry:
    """Return L(s) = robot-to-staging + ball-to-selected-goal."""

    _validate_xy(robot_xy, "robot_xy")
    _validate_xy(ball_xy, "ball_xy")
    if goal_side not in (-1, 1):
        raise ValueError("goal_side must be -1 or 1")
    goal = (goal_side * GOAL_CENTER_ABS_X_M, 0.0)
    goal_delta = (goal[0] - ball_xy[0], goal[1] - ball_xy[1])
    ball_goal_distance = math.hypot(*goal_delta)
    if ball_goal_distance == 0.0:
        direction = (float(goal_side), 0.0)
    else:
        direction = (
            goal_delta[0] / ball_goal_distance,
            goal_delta[1] / ball_goal_distance,
        )
    raw_staging = (
        ball_xy[0] - STAGING_OFFSET_M * direction[0],
        ball_xy[1] - STAGING_OFFSET_M * direction[1],
    )
    staging = (
        _clamp(raw_staging[0], STAGING_SAFE_X_M),
        _clamp(raw_staging[1], STAGING_SAFE_Y_M),
    )
    path_length = math.hypot(
        robot_xy[0] - staging[0],
        robot_xy[1] - staging[1],
    ) + ball_goal_distance
    return TaskGeometry(
        selected_goal_xy=goal,
        raw_staging_target_xy=raw_staging,
        staging_target_xy=staging,
        staging_projection_applied=raw_staging != staging,
        path_length_m=path_length,
    )


def progress_reward(
    pre_robot_xy: tuple[float, float],
    pre_ball_xy: tuple[float, float],
    post_robot_xy: tuple[float, float],
    post_ball_xy: tuple[float, float],
    goal_side: int,
) -> float:
    """Normalize one transition's reduction in task path length."""

    before = task_geometry(pre_robot_xy, pre_ball_xy, goal_side)
    after = task_geometry(post_robot_xy, post_ball_xy, goal_side)
    return (before.path_length_m - after.path_length_m) / PATH_NORMALIZER_M


def true_contact(contact_force_n: float) -> bool:
    if not math.isfinite(contact_force_n):
        raise ValueError("contact force must be finite")
    return contact_force_n >= TRUE_CONTACT_FORCE_THRESHOLD_N


def terminal_decision(inputs: TerminalInputs) -> TerminalDecision:
    """Apply safety-first termination after one native 50 Hz tick."""

    _validate_xy(inputs.robot_xy, "robot_xy")
    _validate_xy(inputs.ball_xy, "ball_xy")
    if inputs.goal_side not in (-1, 1):
        raise ValueError("goal_side must be -1 or 1")
    if not all(
        math.isfinite(value)
        for value in (inputs.base_contact_force_n, inputs.joint_limit_excess_rad)
    ):
        raise ValueError("terminal physical values must be finite")

    selected_axis = inputs.goal_side * inputs.ball_xy[0]
    inside_goal = abs(inputs.ball_xy[1]) <= GOAL_HALF_WIDTH_FOR_BALL_CENTER_M
    requested_goal = (
        inputs.contact_seen
        and selected_axis >= GOAL_CENTER_ABS_X_M
        and inside_goal
    )
    own_goal = selected_axis <= -GOAL_CENTER_ABS_X_M and inside_goal
    boundary = (
        abs(inputs.robot_xy[0]) > FIELD_HALF_LENGTH_M
        or abs(inputs.robot_xy[1]) > FIELD_HALF_WIDTH_M
        or abs(inputs.ball_xy[1]) > BALL_SIDE_BOUNDARY_M
        or (
            abs(inputs.ball_xy[0]) > GOAL_CENTER_ABS_X_M
            and not inside_goal
        )
    )
    fall = inputs.base_contact_force_n > BASE_FALL_FORCE_THRESHOLD_N
    joint_hard_limit = (
        inputs.joint_limit_excess_rad > JOINT_HARD_LIMIT_TOLERANCE_RAD
    )
    unsafe = own_goal or boundary or fall or joint_hard_limit
    goal = requested_goal and not unsafe
    done = goal or unsafe or inputs.timeout
    outcome = (
        "goal" if goal
        else "own_goal" if own_goal
        else "fall" if fall
        else "boundary" if boundary
        else "joint_hard_limit" if joint_hard_limit
        else "timeout" if inputs.timeout
        else None
    )
    return TerminalDecision(
        goal=goal,
        own_goal=own_goal,
        boundary=boundary,
        fall=fall,
        joint_hard_limit=joint_hard_limit,
        timeout=inputs.timeout,
        done=done,
        outcome=outcome,
    )


def total_reward(progress: float, decision: TerminalDecision) -> float:
    """Add only the strict +1/−1 terminal outcome to task progress."""

    if not math.isfinite(progress):
        raise ValueError("progress must be finite")
    sparse = 1.0 if decision.goal else -1.0 if decision.own_goal else 0.0
    return progress + sparse
