"""Stable tensor and field interfaces for sparse high-level soccer."""

from __future__ import annotations

from dataclasses import dataclass
import math


LOW_OBSERVATION_DIM = 48
ENTITY_FEATURE_DIM = 22
POLICY_OBSERVATION_DIM = LOW_OBSERVATION_DIM + ENTITY_FEATURE_DIM
POLICY_ACTION_DIM = 3

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
    "ball_goal_longitudinal",
    "ball_goal_lateral",
)


@dataclass(frozen=True)
class FieldGeometry:
    field_half_length: float = 4.0
    field_half_width: float = 2.5
    goal_half_width: float = 0.60
    ball_radius: float = 0.11
    robot_margin: float = 0.35
    reset_ball_margin: float = 0.35

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value) and value > 0.0 for value in vars(self).values()
        ):
            raise ValueError("Field geometry values must be positive and finite.")
        if self.goal_center_lateral_limit <= 0.0:
            raise ValueError("The declared ball does not fit through the goal.")

    @property
    def strict_goal_center_x(self) -> float:
        return self.field_half_length + self.ball_radius

    @property
    def goal_center_lateral_limit(self) -> float:
        return self.goal_half_width - self.ball_radius


def validate_static_contract() -> None:
    if len(ENTITY_FEATURE_FIELDS) != ENTITY_FEATURE_DIM:
        raise AssertionError("Entity feature names and dimension differ.")
    if POLICY_OBSERVATION_DIM != 70 or POLICY_ACTION_DIM != 3:
        raise AssertionError("Published sparse-high tensor schema changed.")


validate_static_contract()

