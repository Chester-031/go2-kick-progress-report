"""Fixed Stage B distance-then-orientation reset curriculum."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

from .config import FieldGeometry


BG2_ONLY = "BG2_ONLY"
BG2_BG4 = "BG2_BG4"
BG2_BG4_BG6 = "BG2_BG4_BG6"
ORIENTATION_GRID = "ORIENTATION_GRID"
ROBOT_BALL_SUPPORT_M = (0.8, 1.6)
BALL_GOAL_SUPPORT_M = (2.0, 4.0, 6.0)
ORIENTATION_SUPPORT_DEG = (0.0, 45.0, 90.0, 180.0)


@dataclass(frozen=True)
class ResetSample:
    robot_xy: tuple[float, float]
    robot_yaw: float
    ball_xy: tuple[float, float]
    goal_side: int


@dataclass(frozen=True)
class Phase:
    name: str
    first_iteration: int
    last_iteration: int
    angles_deg: tuple[float, ...]
    robot_ball_m: tuple[float, ...]
    ball_goal_m: tuple[float, ...]


PHASES = (
    Phase(BG2_ONLY, 1, 5, (45.0,), ROBOT_BALL_SUPPORT_M, (2.0,)),
    Phase(BG2_BG4, 6, 10, (45.0,), ROBOT_BALL_SUPPORT_M, (2.0, 4.0)),
    Phase(BG2_BG4_BG6, 11, 15, (45.0,), ROBOT_BALL_SUPPORT_M, BALL_GOAL_SUPPORT_M),
    Phase(
        ORIENTATION_GRID,
        16,
        30,
        ORIENTATION_SUPPORT_DEG,
        ROBOT_BALL_SUPPORT_M,
        BALL_GOAL_SUPPORT_M,
    ),
)

_iteration = 0
_phase: Phase | None = None


def phase_for_iteration(iteration: int) -> Phase:
    if type(iteration) is not int:
        raise TypeError("iteration must be an integer")
    for phase in PHASES:
        if phase.first_iteration <= iteration <= phase.last_iteration:
            return phase
    raise ValueError("Stage B iteration must be in 1..30")


def activate_for_iteration(iteration: int) -> Phase:
    """Advance the non-adaptive curriculum exactly once per PPO iteration."""

    global _iteration, _phase
    if iteration != _iteration + 1:
        raise ValueError("curriculum iterations must be sequential")
    _iteration = iteration
    _phase = phase_for_iteration(iteration)
    return _phase


def active_phase() -> Phase:
    if _phase is None:
        raise RuntimeError("activate iteration 1 before creating the environment")
    return _phase


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _targets(phase: Phase, serial: int) -> tuple[float, int, float, float, int, int, int]:
    if type(serial) is not int or serial <= 0:
        raise ValueError("sample serial must be positive")
    offset = serial - 1
    sign_cell = offset % 16
    combinations = tuple(
        (angle, robot_ball, ball_goal)
        for angle in phase.angles_deg
        for robot_ball in phase.robot_ball_m
        for ball_goal in phase.ball_goal_m
    )
    angle, robot_ball, ball_goal = combinations[(offset // 16) % len(combinations)]
    return (
        angle,
        -1 if (sign_cell // 2) % 2 == 0 else 1,
        robot_ball,
        ball_goal,
        -1 if sign_cell % 2 == 0 else 1,
        -1 if (sign_cell // 4) % 2 == 0 else 1,
        -1 if (sign_cell // 8) % 2 == 0 else 1,
    )


def _sample_geometry(target, lateral_angle: float, orbit_angle: float) -> ResetSample:
    angle_deg, facing_sign, robot_ball, ball_goal, goal_side, lateral_sign, orbit_sign = target
    lateral = lateral_sign * lateral_angle
    goal_x = goal_side * 4.11
    ball_x = goal_x - goal_side * ball_goal * math.cos(lateral)
    ball_y = ball_goal * math.sin(lateral)
    ball_to_goal = math.atan2(-ball_y, goal_x - ball_x)
    bearing = ball_to_goal + orbit_sign * orbit_angle
    robot_x = ball_x + robot_ball * math.cos(bearing)
    robot_y = ball_y + robot_ball * math.sin(bearing)
    robot_to_ball = math.atan2(ball_y - robot_y, ball_x - robot_x)
    yaw = _wrap(robot_to_ball + facing_sign * math.radians(angle_deg))
    return ResetSample((robot_x, robot_y), yaw, (ball_x, ball_y), goal_side)


def _legal(sample: ResetSample) -> bool:
    return (
        abs(sample.robot_xy[0]) <= 3.65
        and abs(sample.robot_xy[1]) <= 2.15
        and abs(sample.ball_xy[0]) <= 3.65
        and abs(sample.ball_xy[1]) <= 2.15
    )


def _legal_for_final_support(target, lateral: float, orbit: float) -> bool:
    _, facing_sign, _, _, goal_side, lateral_sign, orbit_sign = target
    for angle in ORIENTATION_SUPPORT_DEG:
        for robot_ball in ROBOT_BALL_SUPPORT_M:
            for ball_goal in BALL_GOAL_SUPPORT_M:
                candidate = (
                    angle,
                    facing_sign,
                    robot_ball,
                    ball_goal,
                    goal_side,
                    lateral_sign,
                    orbit_sign,
                )
                if not _legal(_sample_geometry(candidate, lateral, orbit)):
                    return False
    return True


def sample_training_reset(seed: int, sample_serial: int) -> tuple[ResetSample, str]:
    """Sample one balanced cell from the active Stage B phase."""

    phase = active_phase()
    target = _targets(phase, sample_serial)
    rng = random.Random(int(seed))
    for _ in range(20_000):
        lateral = rng.uniform(math.radians(0.10), math.radians(1.00))
        orbit = rng.uniform(math.radians(122.0), math.radians(136.0))
        if _legal_for_final_support(target, lateral, orbit):
            return _sample_geometry(target, lateral, orbit), phase.name
    raise RuntimeError("unable to sample common legal Stage B support")


def is_hard_goal_facing_ball_behind(
    sample: ResetSample,
    *,
    geometry: FieldGeometry = FieldGeometry(),
) -> bool:
    goal_x = sample.goal_side * geometry.field_half_length
    goal_yaw = math.atan2(-sample.robot_xy[1], goal_x - sample.robot_xy[0])
    yaw_error = _wrap(sample.robot_yaw - goal_yaw)
    dx = sample.ball_xy[0] - sample.robot_xy[0]
    dy = sample.ball_xy[1] - sample.robot_xy[1]
    cosine, sine = math.cos(sample.robot_yaw), math.sin(sample.robot_yaw)
    body_x = cosine * dx + sine * dy
    body_y = -sine * dx + cosine * dy
    return abs(yaw_error) <= 1.0e-6 and body_x <= -0.65 and abs(body_y) <= 1.0e-6


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
