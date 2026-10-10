"""EXPLANATORY PROJECTION — NOT THE PRODUCTION TRAINER.

Pure Stage C reset allocation and analytic geometry construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

from parameters import (
    BALL_GOAL_DISTANCE_SUPPORT_M,
    BALL_GOAL_DISTANCE_WEIGHT_UNITS,
    BALANCED_SIGN_CELLS,
    COMPLETE_RESET_ALLOCATION_PERIOD,
    ORIENTATION_SUPPORT_DEG,
    ROBOT_BALL_DISTANCE_SUPPORT_M,
    TRAINING_SEED,
)


BALL_GOAL_DISTANCE_SLOTS_M = (
    6.0, 5.0, 4.5, 4.0, 2.5, 3.5, 3.0, 6.0,
    2.0, 5.0, 4.5, 4.0, 2.5, 6.0, 3.5, 5.0,
    4.5, 3.0, 6.0, 4.0, 2.5, 5.0, 4.5, 6.0,
    2.0, 3.5, 3.0, 4.0, 2.5, 5.0, 4.5, 4.0,
)
STRICT_GOAL_CENTER_ABS_X_M = 4.11
FIELD_HALF_LENGTH_M = 4.0
FIELD_HALF_WIDTH_M = 2.5
ROBOT_MARGIN_M = 0.35
BALL_MARGIN_M = 0.35
MIN_BALL_GOAL_LATERAL_ANGLE_DEG = 0.10
MAX_BALL_GOAL_LATERAL_ANGLE_DEG = 1.00
MIN_ROBOT_ORBIT_OFFSET_DEG = 122.0
MAX_ROBOT_ORBIT_OFFSET_DEG = 136.0
MAX_COMMON_SUPPORT_REJECTION_ATTEMPTS = 20_000


@dataclass(frozen=True)
class ResetTargets:
    absolute_facing_degrees: float
    facing_sign: int
    robot_ball_distance_m: float
    ball_goal_distance_m: float
    goal_side: int
    ball_lateral_sign: int
    robot_orbit_sign: int


@dataclass(frozen=True)
class ResetSample:
    robot_xy: tuple[float, float]
    robot_yaw: float
    ball_xy: tuple[float, float]
    goal_side: int


@dataclass(frozen=True)
class AllocationCount:
    ball_goal_distance_m: float
    weight_units: int
    resets_per_complete_cycle: int


def item_seed(training_seed: int, sample_serial: int) -> int:
    """Derive each reset's local RNG seed without shared RNG state."""

    if type(training_seed) is not int or isinstance(training_seed, bool):
        raise TypeError("training_seed must be an integer")
    if type(sample_serial) is not int or sample_serial <= 0:
        raise ValueError("sample_serial must be a positive integer")
    return training_seed * 1_000_003 + sample_serial


def targets_for_serial(sample_serial: int) -> ResetTargets:
    """Select one balanced three-axis cell and one four-sign cell."""

    if type(sample_serial) is not int or sample_serial <= 0:
        raise ValueError("sample_serial must be a positive integer")
    offset = sample_serial - 1
    sign_cell = offset % BALANCED_SIGN_CELLS
    combinations = tuple(
        (angle, robot_ball, ball_goal)
        for angle in ORIENTATION_SUPPORT_DEG
        for robot_ball in ROBOT_BALL_DISTANCE_SUPPORT_M
        for ball_goal in BALL_GOAL_DISTANCE_SLOTS_M
    )
    angle, robot_ball, ball_goal = combinations[
        (offset // BALANCED_SIGN_CELLS) % len(combinations)
    ]
    return ResetTargets(
        absolute_facing_degrees=angle,
        facing_sign=-1 if (sign_cell // 2) % 2 == 0 else 1,
        robot_ball_distance_m=robot_ball,
        ball_goal_distance_m=ball_goal,
        goal_side=-1 if sign_cell % 2 == 0 else 1,
        ball_lateral_sign=-1 if (sign_cell // 4) % 2 == 0 else 1,
        robot_orbit_sign=-1 if (sign_cell // 8) % 2 == 0 else 1,
    )


def _wrap_radians(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _geometry(
    targets: ResetTargets,
    lateral_angle_magnitude_rad: float,
    robot_orbit_magnitude_rad: float,
) -> ResetSample:
    lateral = targets.ball_lateral_sign * lateral_angle_magnitude_rad
    orbit = targets.robot_orbit_sign * robot_orbit_magnitude_rad
    goal_x = targets.goal_side * STRICT_GOAL_CENTER_ABS_X_M
    ball_x = (
        goal_x
        - targets.goal_side
        * targets.ball_goal_distance_m
        * math.cos(lateral)
    )
    ball_y = targets.ball_goal_distance_m * math.sin(lateral)
    ball_to_goal_yaw = math.atan2(-ball_y, goal_x - ball_x)
    robot_bearing = ball_to_goal_yaw + orbit
    robot_x = ball_x + targets.robot_ball_distance_m * math.cos(robot_bearing)
    robot_y = ball_y + targets.robot_ball_distance_m * math.sin(robot_bearing)
    robot_to_ball_yaw = math.atan2(ball_y - robot_y, ball_x - robot_x)
    robot_yaw = _wrap_radians(
        robot_to_ball_yaw
        + targets.facing_sign * math.radians(targets.absolute_facing_degrees)
    )
    return ResetSample(
        robot_xy=(robot_x, robot_y),
        robot_yaw=robot_yaw,
        ball_xy=(ball_x, ball_y),
        goal_side=targets.goal_side,
    )


def _legal(sample: ResetSample) -> bool:
    return (
        abs(sample.robot_xy[0]) <= FIELD_HALF_LENGTH_M - ROBOT_MARGIN_M
        and abs(sample.robot_xy[1]) <= FIELD_HALF_WIDTH_M - ROBOT_MARGIN_M
        and abs(sample.ball_xy[0]) <= FIELD_HALF_LENGTH_M - BALL_MARGIN_M
        and abs(sample.ball_xy[1]) <= FIELD_HALF_WIDTH_M - BALL_MARGIN_M
    )


def _common_support_is_legal(
    targets: ResetTargets,
    lateral_angle_magnitude_rad: float,
    robot_orbit_magnitude_rad: float,
) -> bool:
    """Require one nuisance draw to be legal for the complete support."""

    for angle in ORIENTATION_SUPPORT_DEG:
        for robot_ball in ROBOT_BALL_DISTANCE_SUPPORT_M:
            for ball_goal in BALL_GOAL_DISTANCE_SUPPORT_M:
                candidate = ResetTargets(
                    absolute_facing_degrees=angle,
                    facing_sign=targets.facing_sign,
                    robot_ball_distance_m=robot_ball,
                    ball_goal_distance_m=ball_goal,
                    goal_side=targets.goal_side,
                    ball_lateral_sign=targets.ball_lateral_sign,
                    robot_orbit_sign=targets.robot_orbit_sign,
                )
                if not _legal(
                    _geometry(
                        candidate,
                        lateral_angle_magnitude_rad,
                        robot_orbit_magnitude_rad,
                    )
                ):
                    return False
    return True


def build_reset_sample(
    sample_serial: int,
    *,
    training_seed: int = TRAINING_SEED,
) -> ResetSample:
    """Construct one exact Stage C sample without clipping or clamping."""

    targets = targets_for_serial(sample_serial)
    rng = random.Random(item_seed(training_seed, sample_serial))
    for _ in range(MAX_COMMON_SUPPORT_REJECTION_ATTEMPTS):
        lateral = rng.uniform(
            math.radians(MIN_BALL_GOAL_LATERAL_ANGLE_DEG),
            math.radians(MAX_BALL_GOAL_LATERAL_ANGLE_DEG),
        )
        orbit = rng.uniform(
            math.radians(MIN_ROBOT_ORBIT_OFFSET_DEG),
            math.radians(MAX_ROBOT_ORBIT_OFFSET_DEG),
        )
        if _common_support_is_legal(targets, lateral, orbit):
            sample = _geometry(targets, lateral, orbit)
            if not _legal(sample):
                raise RuntimeError("common-support acceptance produced an illegal sample")
            return sample
    raise RuntimeError("unable to find a common-support legal nuisance draw")


def allocation_counts() -> tuple[AllocationCount, ...]:
    """Return exact ball–goal counts in one 4,096-reset allocation cycle."""

    per_weight_unit = (
        len(ORIENTATION_SUPPORT_DEG)
        * len(ROBOT_BALL_DISTANCE_SUPPORT_M)
        * BALANCED_SIGN_CELLS
    )
    return tuple(
        AllocationCount(distance, weight, weight * per_weight_unit)
        for distance, weight in zip(
            BALL_GOAL_DISTANCE_SUPPORT_M,
            BALL_GOAL_DISTANCE_WEIGHT_UNITS,
            strict=True,
        )
    )


if len(BALL_GOAL_DISTANCE_SLOTS_M) != 32:
    raise AssertionError("Stage C must contain exactly 32 distance slots")
if sum(row.resets_per_complete_cycle for row in allocation_counts()) != COMPLETE_RESET_ALLOCATION_PERIOD:
    raise AssertionError("Stage C reset allocation period changed")
