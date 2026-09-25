"""Safety reward and termination terms for the native low executor."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def _action_term(env, action_name: str):
    manager = env.action_manager
    if hasattr(manager, "get_term"):
        return manager.get_term(action_name)
    return manager._terms[action_name]


def guard_correction_l2(env, action_name: str = "joint_pos") -> torch.Tensor:
    """Penalize actor targets that depend on the safety guard."""

    term = _action_term(env, action_name)
    correction = getattr(term, "last_guard_correction", None)
    if correction is None:
        raise RuntimeError("Native safe joint action is not installed.")
    return torch.sum(torch.square(correction), dim=1)


def joint_soft_excess_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize measured positions outside the live soft interval."""

    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    position = asset.data.joint_pos[:, joint_ids]
    limits = asset.data.soft_joint_pos_limits[:, joint_ids, :]
    below = torch.clamp(limits[..., 0] - position, min=0.0)
    above = torch.clamp(position - limits[..., 1], min=0.0)
    return torch.sum(torch.square(below + above), dim=1)


def actual_joint_hard_limit(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tolerance_rad: float = 1.0e-5,
) -> torch.Tensor:
    """Terminate an environment if a measured joint crosses a hard limit."""

    if tolerance_rad < 0.0:
        raise ValueError("Hard-limit tolerance must be non-negative.")
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    position = asset.data.joint_pos[:, joint_ids]
    limits = asset.data.joint_pos_limits[:, joint_ids, :]
    outside = (position < limits[..., 0] - tolerance_rad) | (
        position > limits[..., 1] + tolerance_rad
    )
    return torch.any(outside, dim=1)
