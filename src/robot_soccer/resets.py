"""Where episodes start: the boundary between the environment and whatever
chooses the starts.

The environment asks a ``ResetSampler`` for the starts of the slots it is
resetting and writes them into the simulator; it knows nothing about how
they were chosen.  Training uses automatic domain randomization
(``adr.AutomaticDomainRandomization``); tests use scripted samplers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch


@dataclass(frozen=True)
class ResetBatch:
    """Where a batch of new episodes starts, in env-local field coordinates.

    ``robot_xy`` and ``ball_xy`` are ``[n,2]`` metres, ``robot_yaw`` is
    ``[n]`` radians and ``goal_side`` ``[n]`` is +1 or -1 (the goal the
    robot attacks).  ``tags`` holds any per-episode values the sampler
    wants back in the finished-episode records (each ``[n]`` or ``[n,k]``),
    keyed by the record field name.
    """

    robot_xy: torch.Tensor
    robot_yaw: torch.Tensor
    ball_xy: torch.Tensor
    goal_side: torch.Tensor
    tags: dict[str, torch.Tensor] = field(default_factory=dict)


class ResetSampler(Protocol):
    """Gives the environment the starts of ``count`` new episodes, in the
    order of the slots being reset."""

    def __call__(self, count: int) -> ResetBatch: ...
