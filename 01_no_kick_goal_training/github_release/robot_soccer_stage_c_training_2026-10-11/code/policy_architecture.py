"""EXPLANATORY PROJECTION — NOT THE PRODUCTION TRAINER.

Pure descriptions of the 70-D recurrent high policy and command surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from parameters import (
    ACTIVATION,
    ACTOR_HIDDEN_DIMS,
    COMMAND_AXIS_RANGES,
    COMMAND_RATE_LIMITS_PER_SECOND,
    COMMAND_VERTICES,
    CRITIC_HIDDEN_DIMS,
    ENTITY_FEATURE_DIM,
    GRU_HIDDEN_DIM,
    GRU_LAYERS,
    HIGH_POLICY_HZ,
    LOW_OBSERVATION_DIM,
    LOW_POLICY_HZ,
    LOW_TICKS_PER_HIGH_ACTION,
    POLICY_ACTION_DIM,
    POLICY_OBSERVATION_DIM,
)


ENTITY_FEATURE_FIELDS = (
    "ball_body_x",
    "ball_body_y",
    "goal_body_x",
    "goal_body_y",
    "ball_velocity_body_x",
    "ball_velocity_body_y",
    "goal_direction_body_x",
    "goal_direction_body_y",
    "boundary_ray_front",
    "boundary_ray_left",
    "boundary_ray_back",
    "boundary_ray_right",
    "opponent_body_x",
    "opponent_body_y",
    "opponent_velocity_body_x",
    "opponent_velocity_body_y",
    "opponent_present",
    "selected_goal_side",
    "time_remaining",
    "contact_seen",
    "robot_from_ball_goal_axis",
    "robot_from_ball_goal_left",
)
OPPONENT_FIELDS = ENTITY_FEATURE_FIELDS[12:17]


@dataclass(frozen=True)
class RecurrentPolicy:
    observation_dim: int
    action_dim: int
    recurrent_type: str
    recurrent_hidden_dim: int
    recurrent_layers: int
    actor_hidden_dims: tuple[int, ...]
    critic_hidden_dims: tuple[int, ...]
    activation: str


@dataclass(frozen=True)
class ControlHierarchy:
    high_policy_hz: int
    low_policy_hz: int
    low_ticks_per_high_action: int
    low_policy_trainable: bool
    learned_kick_action: bool


HIGH_POLICY = RecurrentPolicy(
    observation_dim=POLICY_OBSERVATION_DIM,
    action_dim=POLICY_ACTION_DIM,
    recurrent_type="gru",
    recurrent_hidden_dim=GRU_HIDDEN_DIM,
    recurrent_layers=GRU_LAYERS,
    actor_hidden_dims=ACTOR_HIDDEN_DIMS,
    critic_hidden_dims=CRITIC_HIDDEN_DIMS,
    activation=ACTIVATION,
)
CONTROL_HIERARCHY = ControlHierarchy(
    high_policy_hz=HIGH_POLICY_HZ,
    low_policy_hz=LOW_POLICY_HZ,
    low_ticks_per_high_action=LOW_TICKS_PER_HIGH_ACTION,
    low_policy_trainable=False,
    learned_kick_action=False,
)


def observation_layout() -> tuple[tuple[str, int, int], ...]:
    """Return half-open feature slices for the published 70-D observation."""

    return (
        ("frozen_low_observation", 0, LOW_OBSERVATION_DIM),
        ("entity_and_task_features", LOW_OBSERVATION_DIM, POLICY_OBSERVATION_DIM),
    )


def stage_c_opponent_features() -> tuple[tuple[str, float], ...]:
    """Stage C reserved opponent slots but supplied no live opponent."""

    return tuple((name, 0.0) for name in OPPONENT_FIELDS)


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _subtract(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(a - b for a, b in zip(left, right, strict=True))


def _add_scaled(
    base: tuple[float, float, float],
    direction: tuple[float, float, float],
    scale: float,
) -> tuple[float, float, float]:
    return tuple(a + scale * b for a, b in zip(base, direction, strict=True))


def project_to_command_hull(
    desired: tuple[float, float, float],
    *,
    maximum_iterations: int = 64,
    tolerance: float = 1.0e-9,
) -> tuple[float, float, float]:
    """Project one command with the production Frank–Wolfe construction."""

    if len(desired) != 3 or not all(math.isfinite(value) for value in desired):
        raise ValueError("desired command must be a finite three-vector")
    if maximum_iterations <= 0 or tolerance <= 0.0:
        raise ValueError("projection bounds must be positive")
    current = min(
        COMMAND_VERTICES,
        key=lambda vertex: _dot(_subtract(desired, vertex), _subtract(desired, vertex)),
    )
    for _ in range(maximum_iterations):
        gradient = _subtract(current, desired)
        vertex = min(COMMAND_VERTICES, key=lambda item: _dot(gradient, item))
        direction = _subtract(vertex, current)
        gap = -_dot(gradient, direction)
        denominator = _dot(direction, direction)
        if gap <= tolerance or denominator <= tolerance:
            break
        step = min(1.0, max(0.0, gap / denominator))
        current = _add_scaled(current, direction, step)
    return current


def map_high_action(
    latent_action: tuple[float, float, float],
    previous_command: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Apply tanh, physical scaling, hull projection, and coupled rate limit."""

    if len(latent_action) != 3 or len(previous_command) != 3:
        raise ValueError("actions and commands must be three-vectors")
    if not all(math.isfinite(value) for value in (*latent_action, *previous_command)):
        raise ValueError("actions and commands must be finite")
    bounded = tuple(math.tanh(value) for value in latent_action)
    physical = tuple(
        (lower + upper) / 2.0 + value * (upper - lower) / 2.0
        for value, (lower, upper) in zip(
            bounded,
            COMMAND_AXIS_RANGES,
            strict=True,
        )
    )
    target = project_to_command_hull(physical)
    previous = project_to_command_hull(previous_command)
    delta = _subtract(target, previous)
    permitted = tuple(
        rate / HIGH_POLICY_HZ for rate in COMMAND_RATE_LIMITS_PER_SECOND
    )
    ratios = tuple(
        limit / abs(change) if change != 0.0 else 1.0
        for limit, change in zip(permitted, delta, strict=True)
    )
    scale = min(1.0, max(0.0, min(ratios)))
    return _add_scaled(previous, delta, scale)


if len(ENTITY_FEATURE_FIELDS) != ENTITY_FEATURE_DIM:
    raise AssertionError("entity feature schema changed")
if LOW_OBSERVATION_DIM + ENTITY_FEATURE_DIM != POLICY_OBSERVATION_DIM:
    raise AssertionError("observation dimensions do not sum to 70")
