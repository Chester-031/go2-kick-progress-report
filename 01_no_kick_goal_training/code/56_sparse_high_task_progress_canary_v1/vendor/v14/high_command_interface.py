"""Smooth latent high action mapped only into the qualified low command hull."""

from __future__ import annotations

import torch

from staging_command_tensor import limit_high_command_tensor


def latent_to_bounded_action(latent_action: torch.Tensor) -> torch.Tensor:
    if latent_action.ndim != 2 or latent_action.shape[1] != 3:
        raise ValueError("latent_action must have shape [N,3].")
    if not bool(torch.isfinite(latent_action).all()):
        raise ValueError("latent_action contains a non-finite value.")
    bounded = torch.tanh(latent_action)
    if not bool(torch.isfinite(bounded).all()) or bool((bounded.abs() > 1.0).any()):
        raise RuntimeError("Bounded high action is invalid.")
    return bounded


def limit_latent_high_command(
    *,
    latent_action: torch.Tensor,
    previous_command: torch.Tensor,
    vertices: torch.Tensor,
    axis_ranges: torch.Tensor,
    rate_limits_per_second: torch.Tensor,
    high_control_dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    bounded = latent_to_bounded_action(latent_action)
    applied = limit_high_command_tensor(
        normalized_action=bounded,
        previous_command=previous_command,
        vertices=vertices,
        axis_ranges=axis_ranges,
        rate_limits_per_second=rate_limits_per_second,
        high_control_dt=high_control_dt,
    )
    return applied, bounded

