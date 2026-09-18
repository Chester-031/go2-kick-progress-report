"""Native guarded joint-position action used during learning and inference.

Unlike the rejected post-hoc compound parent, the new actor is optimized with
this exact action mapping active from its first rollout.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction


class NativeSafeJointPositionAction(JointPositionAction):
    """Clamp post-affine targets to a fixed live-limit guard interval."""

    GUARD_GAP_MULTIPLIER = 1.10

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        soft = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, :]
        hard = self._asset.data.joint_pos_limits[:, self._joint_ids, :]
        if not bool(torch.isfinite(soft).all().item()):
            raise RuntimeError("Live soft joint limits are unavailable.")
        if not bool(torch.isfinite(hard).all().item()):
            raise RuntimeError("Live hard joint limits are unavailable.")
        lower_gap = soft[..., 0] - hard[..., 0]
        upper_gap = hard[..., 1] - soft[..., 1]
        if not bool(torch.all(lower_gap > 0.0).item()):
            raise RuntimeError("Lower hard-to-soft gaps are not strict.")
        if not bool(torch.all(upper_gap > 0.0).item()):
            raise RuntimeError("Upper hard-to-soft gaps are not strict.")
        self._guard_lower = (
            soft[..., 0] + self.GUARD_GAP_MULTIPLIER * lower_gap
        ).clone()
        self._guard_upper = (
            soft[..., 1] - self.GUARD_GAP_MULTIPLIER * upper_gap
        ).clone()
        if not bool(torch.all(self._guard_lower < self._guard_upper).item()):
            raise RuntimeError("Native guarded target interval is empty.")

        self._unlimited_processed_actions = torch.zeros_like(self.processed_actions)
        self._last_guard_correction = torch.zeros_like(self.processed_actions)
        self._processed_element_count = 0
        self._guard_changed_element_count = 0
        self._maximum_guard_correction_rad = 0.0

    @property
    def unlimited_processed_actions(self) -> torch.Tensor:
        return self._unlimited_processed_actions

    @property
    def last_guard_correction(self) -> torch.Tensor:
        return self._last_guard_correction

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        self._unlimited_processed_actions.copy_(self._processed_actions)
        limited = torch.clamp(
            self._processed_actions, min=self._guard_lower, max=self._guard_upper
        )
        correction = limited - self._processed_actions
        self._last_guard_correction.copy_(correction)
        self._processed_actions.copy_(limited)
        self._processed_element_count += int(correction.numel())
        self._guard_changed_element_count += int(
            torch.count_nonzero(torch.abs(correction) > 1.0e-9).item()
        )
        if correction.numel():
            self._maximum_guard_correction_rad = max(
                self._maximum_guard_correction_rad,
                float(torch.max(torch.abs(correction)).item()),
            )

    def telemetry(self) -> dict[str, int | float]:
        return {
            "processed_elements": self._processed_element_count,
            "guard_changed_elements": self._guard_changed_element_count,
            "maximum_guard_correction_rad": self._maximum_guard_correction_rad,
        }

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        selected = slice(None) if env_ids is None else env_ids
        self._unlimited_processed_actions[selected] = 0.0
        self._last_guard_correction[selected] = 0.0
