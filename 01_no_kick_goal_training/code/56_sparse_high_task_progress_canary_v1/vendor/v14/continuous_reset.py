"""Full-distribution no-curriculum reset sampling."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

from interface_contract import FieldGeometry


@dataclass(frozen=True)
class ResetSample:
    robot_xy: tuple[float, float]
    robot_yaw: float
    ball_xy: tuple[float, float]
    goal_side: int


@dataclass(frozen=True)
class ResetConfig:
    minimum_robot_ball_distance: float = 0.65
    maximum_robot_ball_distance: float = 2.40
    geometry: FieldGeometry = FieldGeometry()

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_robot_ball_distance < self.maximum_robot_ball_distance:
            raise ValueError("Robot-ball reset distance interval is invalid.")


def sample_reset(seed: int, *, config: ResetConfig = ResetConfig()) -> ResetSample:
    """Sample the complete bearing/range distribution without a difficulty level."""

    rng = random.Random(int(seed))
    geometry = config.geometry
    for _ in range(256):
        ball_x = rng.uniform(
            -geometry.field_half_length + geometry.reset_ball_margin,
            geometry.field_half_length - geometry.reset_ball_margin,
        )
        ball_y = rng.uniform(
            -geometry.field_half_width + geometry.reset_ball_margin,
            geometry.field_half_width - geometry.reset_ball_margin,
        )
        bearing = rng.uniform(-math.pi, math.pi)
        distance = rng.uniform(config.minimum_robot_ball_distance, config.maximum_robot_ball_distance)
        robot_x = ball_x - distance * math.cos(bearing)
        robot_y = ball_y - distance * math.sin(bearing)
        if (
            abs(robot_x) <= geometry.field_half_length - geometry.robot_margin
            and abs(robot_y) <= geometry.field_half_width - geometry.robot_margin
        ):
            return ResetSample(
                robot_xy=(robot_x, robot_y),
                robot_yaw=rng.uniform(-math.pi, math.pi),
                ball_xy=(ball_x, ball_y),
                goal_side=-1 if rng.random() < 0.5 else 1,
            )
    raise RuntimeError("Unable to sample a valid full-distribution reset.")


def hard_goal_facing_ball_behind_sample(
    seed: int,
    *,
    goal_side: int,
    config: ResetConfig = ResetConfig(),
) -> ResetSample:
    """Construct the named hard case for read-only evaluation only."""

    if goal_side not in (-1, 1):
        raise ValueError("goal_side must be -1 or 1.")
    rng = random.Random(int(seed))
    geometry = config.geometry
    for _ in range(256):
        robot_x = rng.uniform(-2.20, 2.20)
        robot_y = rng.uniform(-1.50, 1.50)
        goal_x = float(goal_side) * geometry.field_half_length
        yaw = math.atan2(-robot_y, goal_x - robot_x)
        distance = rng.uniform(0.80, 1.80)
        ball_x = robot_x - distance * math.cos(yaw)
        ball_y = robot_y - distance * math.sin(yaw)
        if (
            abs(ball_x) <= geometry.field_half_length - geometry.reset_ball_margin
            and abs(ball_y) <= geometry.field_half_width - geometry.reset_ball_margin
        ):
            return ResetSample(
                robot_xy=(robot_x, robot_y),
                robot_yaw=yaw,
                ball_xy=(ball_x, ball_y),
                goal_side=goal_side,
            )
    raise RuntimeError("Unable to construct the hard goal-facing ball-behind case.")


def is_hard_goal_facing_ball_behind(
    sample: ResetSample,
    *,
    geometry: FieldGeometry = FieldGeometry(),
    yaw_tolerance_radians: float = 1.0e-6,
    lateral_tolerance_m: float = 1.0e-6,
) -> bool:
    goal_x = float(sample.goal_side) * geometry.field_half_length
    goal_yaw = math.atan2(-sample.robot_xy[1], goal_x - sample.robot_xy[0])
    yaw_error = math.atan2(
        math.sin(sample.robot_yaw - goal_yaw),
        math.cos(sample.robot_yaw - goal_yaw),
    )
    dx = sample.ball_xy[0] - sample.robot_xy[0]
    dy = sample.ball_xy[1] - sample.robot_xy[1]
    cosine, sine = math.cos(sample.robot_yaw), math.sin(sample.robot_yaw)
    body_x = cosine * dx + sine * dy
    body_y = -sine * dx + cosine * dy
    return (
        abs(yaw_error) <= yaw_tolerance_radians
        and body_x <= -0.65
        and abs(body_y) <= lateral_tolerance_m
    )


def fixed_evaluation_samples(
    count: int,
    seed: int,
    *,
    hard_case_count: int | None = None,
) -> tuple[ResetSample, ...]:
    if count <= 0:
        raise ValueError("Fixed evaluation count must be positive.")
    if hard_case_count is None:
        hard_case_count = count // 8
    if not 0 <= hard_case_count <= count:
        raise ValueError("hard_case_count must be within the evaluation count.")
    hard_indices = {
        (ordinal * count) // hard_case_count for ordinal in range(hard_case_count)
    } if hard_case_count else set()
    values: list[ResetSample] = []
    hard_ordinal = 0
    for index in range(count):
        item_seed = seed * 1_000_003 + index
        if index in hard_indices:
            goal_side = 1 if hard_ordinal % 2 == 0 else -1
            values.append(hard_goal_facing_ball_behind_sample(item_seed, goal_side=goal_side))
            hard_ordinal += 1
        else:
            values.append(sample_reset(item_seed))
    result = tuple(values)
    actual_hard_count = sum(is_hard_goal_facing_ball_behind(value) for value in result)
    if actual_hard_count != hard_case_count:
        raise RuntimeError(
            f"Fixed hard-case construction changed: {actual_hard_count}/{hard_case_count}."
        )
    return result


def relative_bucket(sample: ResetSample) -> str:
    dx = sample.ball_xy[0] - sample.robot_xy[0]
    dy = sample.ball_xy[1] - sample.robot_xy[1]
    cosine, sine = math.cos(sample.robot_yaw), math.sin(sample.robot_yaw)
    body_x = cosine * dx + sine * dy
    body_y = -sine * dx + cosine * dy
    far = math.hypot(dx, dy) >= 1.45
    if abs(body_x) >= abs(body_y):
        side = "front" if body_x >= 0.0 else "back"
    else:
        side = "left" if body_y >= 0.0 else "right"
    return f"{side}_{'far' if far else 'near'}"
