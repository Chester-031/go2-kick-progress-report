#!/usr/bin/env python3
"""R8.60 isolated skill router.

This module deliberately has no Isaac/torch dependency. It validates the
exclusive action-owner contract before the router is inserted into the exact
V4D/R8.58 environment source.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


class Skill(str, Enum):
    STAND = "stand"
    LOCOMOTION = "locomotion"
    PUSH = "push"
    KICK = "kick"


class Phase(str, Enum):
    NORMAL = "normal"
    KICK_STRIKE = "kick_strike"
    KICK_RECOVERY = "kick_recovery"
    ABORT_RECOVERY = "abort_recovery"


@dataclass(frozen=True)
class KickController:
    strength: float
    thigh: float
    calf: float
    ramp: float


KICK_CONTROLLERS = {
    0.75: KickController(0.75, 0.32, 0.12, 0.35),
    1.00: KickController(1.00, 0.32, 0.16, 0.70),
    1.25: KickController(1.25, -0.04, 0.04, 0.00),
}


@dataclass
class RouterState:
    phase: Phase = Phase.NORMAL
    skill: Skill = Skill.LOCOMOTION
    strength: float = 1.00
    elapsed: int = 0
    total_steps: int = 0
    owner_steps: int = 0
    episode_id: int = 0


class R860ActionRouter:
    """Exclusive 12D owner switch executed immediately before low_env.step."""

    def __init__(self, *, strength: float = 1.00, strike_steps: int = 60,
                 recovery_steps: int = 40) -> None:
        self.state = RouterState(strength=self._nearest_strength(strength))
        self.strike_steps = int(strike_steps)
        self.recovery_steps = int(recovery_steps)
        self.low_env_step_count = 0
        self.events: list[dict[str, Any]] = []

    @staticmethod
    def _nearest_strength(value: float) -> float:
        return min(KICK_CONTROLLERS, key=lambda x: abs(float(x) - float(value)))

    @staticmethod
    def _vec(x: Sequence[float]) -> tuple[float, ...]:
        result = tuple(float(v) for v in x)
        if len(result) != 12:
            raise ValueError(f"expected 12D action, got {len(result)}D")
        return result

    def _record(self, event: str, **extra: Any) -> None:
        self.events.append({
            "event": event, "phase": self.state.phase.value,
            "skill": self.state.skill.value, "strength": self.state.strength,
            "step": self.state.total_steps, **extra,
        })

    def request(self, skill: Skill | str, *, strength: float | None = None,
                ready: bool = True) -> bool:
        """Request a skill only while normal; kick is safety-gated."""
        skill = Skill(skill)
        if self.state.phase is not Phase.NORMAL:
            self._record("request_ignored_busy", requested=skill.value)
            return False
        if skill is Skill.KICK and not ready:
            self._record("kick_rejected_not_ready")
            return False
        self.state.skill = skill
        self.state.elapsed = 0
        if skill is Skill.KICK:
            self.state.phase = Phase.KICK_STRIKE
            if strength is not None:
                self.state.strength = self._nearest_strength(strength)
        self._record("skill_acquired")
        return True

    def _owner_action(self, v4d: Sequence[float], kick: Sequence[float],
                      recovery: Sequence[float]) -> tuple[tuple[float, ...], str]:
        # Exactly one branch owns the output; this is intentionally not a sum.
        if self.state.phase is Phase.KICK_STRIKE:
            return self._vec(kick), "r858_kick"
        if self.state.phase in (Phase.KICK_RECOVERY, Phase.ABORT_RECOVERY):
            return self._vec(recovery), "r855_recovery"
        return self._vec(v4d), f"v4d_{self.state.skill.value}"

    def step(self, v4d: Sequence[float], kick: Sequence[float],
             recovery: Sequence[float], *, contact: bool = False,
             recovered: bool = False, abort: bool = False) -> tuple[tuple[float, ...], str]:
        action, owner = self._owner_action(v4d, kick, recovery)
        self.low_env_step_count += 1
        self.state.total_steps += 1
        self.state.owner_steps += 1
        self._record("low_env_step", owner=owner)

        if self.state.phase is Phase.KICK_STRIKE:
            self.state.elapsed += 1
            if abort:
                self.state.phase = Phase.ABORT_RECOVERY
                self.state.elapsed = 0
                self._record("kick_abort")
            elif contact or self.state.elapsed >= self.strike_steps:
                self.state.phase = Phase.KICK_RECOVERY
                self.state.elapsed = 0
                self._record("strike_to_recovery", contact=bool(contact))
        elif self.state.phase in (Phase.KICK_RECOVERY, Phase.ABORT_RECOVERY):
            self.state.elapsed += 1
            if recovered or self.state.elapsed >= self.recovery_steps:
                self.state.phase = Phase.NORMAL
                self.state.skill = Skill.LOCOMOTION
                self.state.elapsed = 0
                self._record("recovery_to_normal")
        return action, owner

    def summary(self) -> dict[str, Any]:
        return {
            "phase": self.state.phase.value,
            "skill": self.state.skill.value,
            "strength": self.state.strength,
            "low_env_step_count": self.low_env_step_count,
            "events": len(self.events),
            "controllers": {str(k): vars(v) for k, v in KICK_CONTROLLERS.items()},
        }
