"""EXPLANATORY PROJECTION — NOT THE PRODUCTION TRAINER.

Peer definitions for Stage C orientation, robot–ball, and ball–goal analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

from parameters import (
    BALL_GOAL_DISTANCE_SUPPORT_M,
    BALL_GOAL_DISTANCE_WEIGHT_UNITS,
    COMPLETE_RESET_ALLOCATION_PERIOD,
    ORIENTATION_SUPPORT_DEG,
    ROBOT_BALL_DISTANCE_SUPPORT_M,
)


@dataclass(frozen=True)
class AnalysisAxis:
    name: str
    unit: str
    support: tuple[float, ...]
    allocation_weights: tuple[int, ...]
    interpretation: str


@dataclass(frozen=True)
class AxisSupportCount:
    axis: str
    value: float
    expected_resets_per_complete_cycle: int


PRIMARY_ANALYSIS_AXES = (
    AnalysisAxis(
        name="absolute_orientation_error",
        unit="degrees",
        support=ORIENTATION_SUPPORT_DEG,
        allocation_weights=(1, 1, 1, 1),
        interpretation="initial robot heading relative to the robot-to-ball bearing",
    ),
    AnalysisAxis(
        name="robot_ball_distance",
        unit="meters",
        support=ROBOT_BALL_DISTANCE_SUPPORT_M,
        allocation_weights=(1, 1),
        interpretation="initial distance the robot must close before transport",
    ),
    AnalysisAxis(
        name="ball_goal_distance",
        unit="meters",
        support=BALL_GOAL_DISTANCE_SUPPORT_M,
        allocation_weights=BALL_GOAL_DISTANCE_WEIGHT_UNITS,
        interpretation="initial transport distance to the selected goal",
    ),
)


def axis_support_counts() -> tuple[AxisSupportCount, ...]:
    """Return marginal counts for one complete 4,096-reset allocation cycle."""

    orientation_count = COMPLETE_RESET_ALLOCATION_PERIOD // len(ORIENTATION_SUPPORT_DEG)
    robot_ball_count = COMPLETE_RESET_ALLOCATION_PERIOD // len(ROBOT_BALL_DISTANCE_SUPPORT_M)
    ball_goal_unit = COMPLETE_RESET_ALLOCATION_PERIOD // sum(BALL_GOAL_DISTANCE_WEIGHT_UNITS)
    return (
        *(
            AxisSupportCount("absolute_orientation_error", value, orientation_count)
            for value in ORIENTATION_SUPPORT_DEG
        ),
        *(
            AxisSupportCount("robot_ball_distance", value, robot_ball_count)
            for value in ROBOT_BALL_DISTANCE_SUPPORT_M
        ),
        *(
            AxisSupportCount("ball_goal_distance", value, weight * ball_goal_unit)
            for value, weight in zip(
                BALL_GOAL_DISTANCE_SUPPORT_M,
                BALL_GOAL_DISTANCE_WEIGHT_UNITS,
                strict=True,
            )
        ),
    )


def primary_axis_names() -> tuple[str, str, str]:
    """The order is presentational; all three axes are analysis peers."""

    return tuple(axis.name for axis in PRIMARY_ANALYSIS_AXES)


def comparison_rule() -> tuple[str, ...]:
    """Return the interpretation rules for distribution-aware comparisons."""

    return (
        "report every primary axis directly",
        "do not let an aggregate rate hide a weak support value",
        "condition cross-stage distance comparisons on exact distance",
        "remember that only ball_goal_distance is allocation-weighted",
        "do not extrapolate beyond robot_ball_distance values 0.8 and 1.6 meters",
        "do not interpret zero-filled opponent slots as learned 1v1 behavior",
    )


if primary_axis_names() != (
    "absolute_orientation_error",
    "robot_ball_distance",
    "ball_goal_distance",
):
    raise AssertionError("the three primary analysis axes changed")
