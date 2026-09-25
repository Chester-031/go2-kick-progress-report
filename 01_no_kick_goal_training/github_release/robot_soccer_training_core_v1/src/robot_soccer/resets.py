"""Stationary Reference/Full reset sampling for no-Kick training."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

from .config import FieldGeometry


REFERENCE = "REFERENCE"
FULL = "FULL"
EVALUATION = "EVALUATION"


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
        if not (
            0.0
            < self.minimum_robot_ball_distance
            < self.maximum_robot_ball_distance
        ):
            raise ValueError("Robot-ball reset distance interval is invalid.")


# Every eight reset serials contain six Reference and two Full samples.
# This pattern is stationary from the first reset; it is not a curriculum.
TRAINING_PATTERN_PERIOD = 8
REFERENCE_PATTERN_SIDES = {0: -1, 1: 1, 2: -1, 4: 1, 5: -1, 6: 1}

REFERENCE_BALL_AXIS_MAGNITUDE_M = (3.84, 3.88)
REFERENCE_BALL_LATERAL_M = (-0.06, 0.06)
REFERENCE_ROBOT_BEHIND_BALL_M = (0.56, 0.60)
REFERENCE_ROBOT_LATERAL_OFFSET_M = (-0.04, 0.04)
REFERENCE_ROBOT_YAW_ERROR_RAD = (-0.08, 0.08)

BALANCED_REFERENCE_CASE_COUNT = 512
BALANCED_REFERENCE_SEED = 91_001


def sample_full_reset(
    seed: int,
    *,
    config: ResetConfig = ResetConfig(),
) -> ResetSample:
    """Sample the complete bearing/range distribution."""

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
        distance = rng.uniform(
            config.minimum_robot_ball_distance,
            config.maximum_robot_ball_distance,
        )
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
    raise RuntimeError("Unable to sample a valid Full-distribution reset.")


def sample_reference_reset(
    seed: int,
    *,
    goal_side: int,
    config: ResetConfig = ResetConfig(),
) -> ResetSample:
    """Sample the fixed Reference-condition support distribution."""

    if goal_side not in (-1, 1):
        raise ValueError("goal_side must be -1 or 1.")
    rng = random.Random(int(seed))
    axis_magnitude = rng.uniform(*REFERENCE_BALL_AXIS_MAGNITUDE_M)
    ball_y = rng.uniform(*REFERENCE_BALL_LATERAL_M)
    behind = rng.uniform(*REFERENCE_ROBOT_BEHIND_BALL_M)
    lateral = rng.uniform(*REFERENCE_ROBOT_LATERAL_OFFSET_M)
    yaw_error = rng.uniform(*REFERENCE_ROBOT_YAW_ERROR_RAD)

    ball_x = float(goal_side) * axis_magnitude
    goal_x = float(goal_side) * config.geometry.strict_goal_center_x
    goal_dx = goal_x - ball_x
    goal_dy = -ball_y
    goal_distance = math.hypot(goal_dx, goal_dy)
    if goal_distance <= 0.0:
        raise RuntimeError("Reference reset has no goal direction.")
    unit_x, unit_y = goal_dx / goal_distance, goal_dy / goal_distance
    left_x, left_y = -unit_y, unit_x
    signed_lateral = float(goal_side) * lateral
    robot_x = ball_x - behind * unit_x + signed_lateral * left_x
    robot_y = ball_y - behind * unit_y + signed_lateral * left_y
    target_yaw = math.atan2(unit_y, unit_x)
    robot_yaw = math.atan2(
        math.sin(target_yaw + float(goal_side) * yaw_error),
        math.cos(target_yaw + float(goal_side) * yaw_error),
    )
    return ResetSample(
        robot_xy=(robot_x, robot_y),
        robot_yaw=robot_yaw,
        ball_xy=(ball_x, ball_y),
        goal_side=goal_side,
    )


def stationary_training_regime(sample_serial: int) -> tuple[str, int | None]:
    """Return the fixed six-Reference/two-Full assignment for one serial."""

    if type(sample_serial) is not int or sample_serial <= 0:
        raise ValueError("sample_serial must be a positive integer.")
    side = REFERENCE_PATTERN_SIDES.get(
        (sample_serial - 1) % TRAINING_PATTERN_PERIOD
    )
    return (FULL, None) if side is None else (REFERENCE, side)


def sample_training_reset(
    seed: int,
    sample_serial: int,
    *,
    config: ResetConfig = ResetConfig(),
) -> tuple[ResetSample, str]:
    """Sample the exact stationary 75% Reference / 25% Full mixture."""

    regime, goal_side = stationary_training_regime(sample_serial)
    if regime == REFERENCE:
        if goal_side is None:
            raise AssertionError("Reference goal side is absent.")
        return (
            sample_reference_reset(seed, goal_side=goal_side, config=config),
            regime,
        )
    return sample_full_reset(seed, config=config), regime


def balanced_reference_side(assignment: int) -> int:
    if type(assignment) is not int or not (
        0 <= assignment < BALANCED_REFERENCE_CASE_COUNT
    ):
        raise ValueError("Reference assignment is out of range.")
    return -1 if assignment % 2 == 0 else 1


def balanced_reference_samples(
    *,
    config: ResetConfig = ResetConfig(),
) -> tuple[ResetSample, ...]:
    """Build the fixed 512-case balanced Reference evaluation set."""

    return tuple(
        sample_reference_reset(
            BALANCED_REFERENCE_SEED * 1_000_003 + assignment,
            goal_side=balanced_reference_side(assignment),
            config=config,
        )
        for assignment in range(BALANCED_REFERENCE_CASE_COUNT)
    )


def hard_goal_facing_ball_behind_sample(
    seed: int,
    *,
    goal_side: int,
    config: ResetConfig = ResetConfig(),
) -> ResetSample:
    """Construct the named hard subset used only for fixed evaluation."""

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
    raise RuntimeError("Unable to construct the named hard reset.")


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
    config: ResetConfig = ResetConfig(),
) -> tuple[ResetSample, ...]:
    """Build a deterministic Full-distribution set with an explicit hard subset."""

    if count <= 0:
        raise ValueError("Fixed evaluation count must be positive.")
    if hard_case_count is None:
        hard_case_count = count // 8
    if not 0 <= hard_case_count <= count:
        raise ValueError("hard_case_count must be within the evaluation count.")
    hard_indices = (
        {
            (ordinal * count) // hard_case_count
            for ordinal in range(hard_case_count)
        }
        if hard_case_count
        else set()
    )
    values: list[ResetSample] = []
    hard_ordinal = 0
    for index in range(count):
        item_seed = seed * 1_000_003 + index
        if index in hard_indices:
            goal_side = 1 if hard_ordinal % 2 == 0 else -1
            values.append(
                hard_goal_facing_ball_behind_sample(
                    item_seed,
                    goal_side=goal_side,
                    config=config,
                )
            )
            hard_ordinal += 1
        else:
            values.append(sample_full_reset(item_seed, config=config))
    result = tuple(values)
    if sum(is_hard_goal_facing_ball_behind(value) for value in result) != hard_case_count:
        raise RuntimeError("Fixed hard-case construction changed.")
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
