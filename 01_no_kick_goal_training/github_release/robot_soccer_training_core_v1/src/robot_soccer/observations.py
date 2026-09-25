"""Build the 70-D high-policy observation used by the no-Kick policy."""

from __future__ import annotations

import math
from typing import Any

import torch

from .config import (
    ENTITY_FEATURE_DIM,
    LOW_OBSERVATION_DIM,
    POLICY_OBSERVATION_DIM,
    FieldGeometry,
)


GOAL_DIRECTION_EPSILON_M = 1.0e-6
LONGITUDINAL_NORMALIZER_M = 8.0


def yaw_from_quaternion_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ValueError("quaternion must have shape [N,4].")
    w, x, y, z = quaternion.unbind(dim=1)
    return torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y.square() + z.square()),
    )


def world_to_body(vector_xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    if vector_xy.ndim != 2 or vector_xy.shape[1] != 2:
        raise ValueError("vector_xy must have shape [N,2].")
    if yaw.shape != (vector_xy.shape[0],):
        raise ValueError("yaw must have shape [N].")
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    return torch.stack(
        (
            cosine * vector_xy[:, 0] + sine * vector_xy[:, 1],
            -sine * vector_xy[:, 0] + cosine * vector_xy[:, 1],
        ),
        dim=1,
    )


def boundary_rays(
    robot_xy_local: torch.Tensor,
    yaw: torch.Tensor,
    *,
    geometry: FieldGeometry = FieldGeometry(),
) -> torch.Tensor:
    if robot_xy_local.ndim != 2 or robot_xy_local.shape[1] != 2:
        raise ValueError("robot_xy_local must have shape [N,2].")
    if yaw.shape != (robot_xy_local.shape[0],):
        raise ValueError("yaw must have shape [N].")
    x = robot_xy_local[:, 0].clamp(
        -geometry.field_half_length,
        geometry.field_half_length,
    )
    y = robot_xy_local[:, 1].clamp(
        -geometry.field_half_width,
        geometry.field_half_width,
    )
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    directions = torch.stack(
        (
            torch.stack((cosine, sine), dim=1),
            torch.stack((-sine, cosine), dim=1),
            torch.stack((-cosine, -sine), dim=1),
            torch.stack((sine, -cosine), dim=1),
        ),
        dim=1,
    )
    dx, dy = directions[:, :, 0], directions[:, :, 1]
    epsilon = torch.finfo(robot_xy_local.dtype).eps
    infinity = torch.full_like(dx, float("inf"))
    tx = torch.where(
        dx > epsilon,
        (geometry.field_half_length - x[:, None]) / dx,
        torch.where(
            dx < -epsilon,
            (-geometry.field_half_length - x[:, None]) / dx,
            infinity,
        ),
    )
    ty = torch.where(
        dy > epsilon,
        (geometry.field_half_width - y[:, None]) / dy,
        torch.where(
            dy < -epsilon,
            (-geometry.field_half_width - y[:, None]) / dy,
            infinity,
        ),
    )
    diagonal = math.hypot(
        2.0 * geometry.field_half_length,
        2.0 * geometry.field_half_width,
    )
    return (torch.minimum(tx, ty).clamp(min=0.0) / diagonal).clamp(0.0, 1.0)


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if not torch.is_tensor(value) or tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite.")
    return value


def build_entity_observation(
    *,
    low_observation: torch.Tensor,
    robot_position_w: torch.Tensor,
    robot_quaternion_wxyz: torch.Tensor,
    ball_position_w: torch.Tensor,
    ball_velocity_w: torch.Tensor,
    goal_xy_w: torch.Tensor,
    env_origin_xy_w: torch.Tensor,
    time_remaining: torch.Tensor,
    contact_seen: torch.Tensor,
    opponent_position_w: torch.Tensor | None = None,
    opponent_velocity_w: torch.Tensor | None = None,
    opponent_present: torch.Tensor | None = None,
    geometry: FieldGeometry = FieldGeometry(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the policy observation and useful geometric state.

    The final two policy features are the robot position relative to the ball,
    expressed in the ball-to-selected-goal frame.  This is the coordinate
    repair used by the recovered training run; robot yaw does not enter those
    two coordinates.
    """

    if not torch.is_tensor(low_observation) or low_observation.ndim != 2:
        raise ValueError("low_observation must have shape [N,48].")
    batch = low_observation.shape[0]
    _require_tensor(
        "low_observation",
        low_observation,
        (batch, LOW_OBSERVATION_DIM),
    )
    _require_tensor("robot_position_w", robot_position_w, (batch, 3))
    _require_tensor(
        "robot_quaternion_wxyz",
        robot_quaternion_wxyz,
        (batch, 4),
    )
    _require_tensor("ball_position_w", ball_position_w, (batch, 3))
    _require_tensor("ball_velocity_w", ball_velocity_w, (batch, 3))
    _require_tensor("goal_xy_w", goal_xy_w, (batch, 2))
    _require_tensor("env_origin_xy_w", env_origin_xy_w, (batch, 2))
    _require_tensor("time_remaining", time_remaining, (batch,))
    if not torch.is_tensor(contact_seen) or contact_seen.shape != (batch,):
        raise ValueError("contact_seen must have shape [N].")
    if contact_seen.dtype != torch.bool:
        raise ValueError("contact_seen must be Boolean.")

    device, dtype = low_observation.device, low_observation.dtype
    if opponent_present is None:
        opponent_present = torch.zeros(batch, dtype=torch.bool, device=device)
    if opponent_position_w is None:
        opponent_position_w = torch.zeros(batch, 3, dtype=dtype, device=device)
    if opponent_velocity_w is None:
        opponent_velocity_w = torch.zeros(batch, 3, dtype=dtype, device=device)
    if opponent_present.shape != (batch,) or opponent_present.dtype != torch.bool:
        raise ValueError("opponent_present must be Boolean [N].")
    _require_tensor("opponent_position_w", opponent_position_w, (batch, 3))
    _require_tensor("opponent_velocity_w", opponent_velocity_w, (batch, 3))

    yaw = yaw_from_quaternion_wxyz(robot_quaternion_wxyz)
    robot_xy = robot_position_w[:, :2]
    ball_xy = ball_position_w[:, :2]
    ball_body = world_to_body(ball_xy - robot_xy, yaw)
    goal_body = world_to_body(goal_xy_w - robot_xy, yaw)
    ball_velocity_body = world_to_body(ball_velocity_w[:, :2], yaw)

    goal_delta = goal_xy_w - ball_xy
    goal_distance = torch.linalg.vector_norm(goal_delta, dim=1)
    safe_goal_distance = goal_distance.clamp_min(GOAL_DIRECTION_EPSILON_M)
    goal_direction_w = goal_delta / safe_goal_distance[:, None]
    goal_direction_body = world_to_body(goal_direction_w, yaw)
    goal_left_w = torch.stack(
        (-goal_direction_w[:, 1], goal_direction_w[:, 0]),
        dim=1,
    )
    rays = boundary_rays(
        robot_xy - env_origin_xy_w,
        yaw,
        geometry=geometry,
    )

    opponent_relative = world_to_body(
        opponent_position_w[:, :2] - robot_xy,
        yaw,
    )
    opponent_velocity = world_to_body(opponent_velocity_w[:, :2], yaw)
    present = opponent_present.to(dtype=dtype)[:, None]
    opponent_relative = opponent_relative * present
    opponent_velocity = opponent_velocity * present

    selected_goal_side = torch.sign(
        goal_xy_w[:, 0] - env_origin_xy_w[:, 0]
    )

    # Coordinate-repaired tail: robot relative to ball, in the ball-goal frame.
    robot_from_ball = robot_xy - ball_xy
    longitudinal = torch.sum(robot_from_ball * goal_direction_w, dim=1)
    lateral = torch.sum(robot_from_ball * goal_left_w, dim=1)

    entity = torch.cat(
        (
            ball_body / 5.0,
            goal_body / 8.0,
            ball_velocity_body / 3.0,
            goal_direction_body,
            rays,
            opponent_relative / 5.0,
            opponent_velocity / 3.0,
            present,
            selected_goal_side[:, None],
            time_remaining.clamp(0.0, 1.0)[:, None],
            contact_seen.to(dtype=dtype)[:, None],
            longitudinal[:, None] / LONGITUDINAL_NORMALIZER_M,
            lateral[:, None] / geometry.field_half_width,
        ),
        dim=1,
    )
    if entity.shape != (batch, ENTITY_FEATURE_DIM):
        raise AssertionError(f"Entity observation drifted: {tuple(entity.shape)}")
    observation = torch.cat((low_observation, entity), dim=1)
    if observation.shape != (batch, POLICY_OBSERVATION_DIM):
        raise AssertionError(f"Policy observation drifted: {tuple(observation.shape)}")
    if not bool(torch.isfinite(observation).all()):
        raise RuntimeError("Policy observation contains a non-finite value.")

    return observation, {
        "robot_yaw": yaw,
        "robot_ball_distance": torch.linalg.vector_norm(ball_xy - robot_xy, dim=1),
        "ball_goal_distance": goal_distance,
        "goal_direction_w": goal_direction_w,
        "selected_goal_side": selected_goal_side,
        "robot_from_ball_goal_axis": longitudinal,
        "robot_from_ball_goal_left": lateral,
    }
