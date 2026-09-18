"""The complete high-level reward: goal +1, own goal -1, otherwise zero."""

from __future__ import annotations

import torch


def sparse_goal_reward(
    *, goal: torch.Tensor, own_goal: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if goal.dtype != torch.bool or own_goal.dtype != torch.bool:
        raise ValueError("Goal masks must be Boolean.")
    if goal.shape != own_goal.shape:
        raise ValueError("Goal masks differ in shape.")
    if bool((goal & own_goal).any()):
        raise ValueError("A transition cannot be both a goal and an own goal.")
    goal_term = goal.to(dtype=torch.float32)
    own_goal_term = -own_goal.to(dtype=torch.float32)
    reward = goal_term + own_goal_term
    if not bool(torch.isfinite(reward).all()):
        raise RuntimeError("Sparse goal reward is non-finite.")
    return reward, {"goal": goal_term, "own_goal": own_goal_term}

