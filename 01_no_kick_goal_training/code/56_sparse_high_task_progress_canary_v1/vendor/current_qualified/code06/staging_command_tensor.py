"""Batched high-command projection into an accepted low command envelope."""

from __future__ import annotations

import torch


def normalized_action_to_physical(
    action: torch.Tensor,
    axis_ranges: torch.Tensor,
) -> torch.Tensor:
    if action.ndim != 2 or action.shape[1] != 3:
        raise ValueError("action must have shape [N,3].")
    if axis_ranges.shape != (3, 2):
        raise ValueError("axis_ranges must have shape [3,2].")
    if not bool(torch.isfinite(action).all()) or not bool(torch.isfinite(axis_ranges).all()):
        raise ValueError("Action mapping inputs must be finite.")
    if bool((axis_ranges[:, 0] > axis_ranges[:, 1]).any()):
        raise ValueError("An axis range has minimum greater than maximum.")
    clipped = torch.clamp(action, -1.0, 1.0)
    center = axis_ranges.mean(dim=1)
    half = (axis_ranges[:, 1] - axis_ranges[:, 0]) / 2.0
    return center.unsqueeze(0) + clipped * half.unsqueeze(0)


def project_to_convex_hull(
    desired: torch.Tensor,
    vertices: torch.Tensor,
    *,
    maximum_iterations: int = 64,
    tolerance: float = 1.0e-9,
) -> torch.Tensor:
    """Batched Frank-Wolfe projection; every iterate remains admissible."""

    if desired.ndim != 2 or desired.shape[1] != 3:
        raise ValueError("desired must have shape [N,3].")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] <= 0:
        raise ValueError("vertices must have shape [V,3] with V > 0.")
    if maximum_iterations <= 0 or tolerance <= 0.0:
        raise ValueError("Projection iterations and tolerance must be positive.")
    distances = torch.sum((desired[:, None, :] - vertices[None, :, :]).square(), dim=2)
    current = vertices[torch.argmin(distances, dim=1)].clone()
    for _ in range(maximum_iterations):
        gradient = current - desired
        scores = torch.sum(gradient[:, None, :] * vertices[None, :, :], dim=2)
        vertex = vertices[torch.argmin(scores, dim=1)]
        direction = vertex - current
        gap = -torch.sum(gradient * direction, dim=1)
        denominator = torch.sum(direction.square(), dim=1)
        active = (gap > tolerance) & (denominator > tolerance)
        if not bool(active.any()):
            break
        step = torch.zeros_like(gap)
        step[active] = torch.clamp(gap[active] / denominator[active], 0.0, 1.0)
        current = current + step.unsqueeze(1) * direction
    return current


def limit_high_command_tensor(
    *,
    normalized_action: torch.Tensor,
    previous_command: torch.Tensor,
    vertices: torch.Tensor,
    axis_ranges: torch.Tensor,
    rate_limits_per_second: torch.Tensor,
    high_control_dt: float,
) -> torch.Tensor:
    """Map, project and rate-limit a batch without any behavioral router."""

    if previous_command.shape != normalized_action.shape:
        raise ValueError("previous_command and normalized_action shapes differ.")
    if rate_limits_per_second.shape != (3,) or bool((rate_limits_per_second <= 0.0).any()):
        raise ValueError("rate_limits_per_second must be a positive [3] tensor.")
    if high_control_dt <= 0.0:
        raise ValueError("high_control_dt must be positive.")
    physical = normalized_action_to_physical(normalized_action, axis_ranges)
    target = project_to_convex_hull(physical, vertices)
    previous = project_to_convex_hull(previous_command, vertices)
    delta = target - previous
    permitted = rate_limits_per_second * high_control_dt
    ratios = torch.where(delta.abs() > 0.0, permitted.unsqueeze(0) / delta.abs(), 1.0)
    scale = torch.clamp(ratios.amin(dim=1), 0.0, 1.0)
    return previous + scale.unsqueeze(1) * delta
