"""How a policy action becomes the native 50 Hz joint actions.

The soccer environment steps the Isaac Lab scene one native 50 Hz tick at a
time and checks goals and safety after every tick.  A control owns
everything between the policy's action and those ticks: how many ticks one
policy action is held for, what joint action each tick sends, and the
velocity command the native observation carries.

- ``DirectJointControl``: the policy itself outputs the 12 joint actions
  every tick (one tick per action).  What the 12 numbers mean (position or
  velocity targets) is the scene's action term, chosen in the scene
  config.  The velocity-command slot of the observation stays zero.
- ``HeldCommandControl``: the policy outputs a ``(vx, vy, wz)`` body
  velocity command, projected onto the qualified command surface and held
  for five ticks, which a frozen low-level locomotion policy turns into
  joint actions (the Stage B hierarchy).
"""

from __future__ import annotations

from typing import Callable, Protocol

import torch

from .commands import CommandSurface


class Control(Protocol):
    """What the environment needs from a control."""

    num_actions: int
    ticks_per_action: int

    def start(self, action: torch.Tensor) -> torch.Tensor:
        """Take a policy action for all slots; return the bounded action
        for logging."""

    def tick_action(
        self, tick_active: torch.Tensor, native_observation: Callable[[], torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The ``[N,12]`` joint action for this tick (zero for slots that
        already terminated) and a scalar failure flag."""

    def held_command(self) -> torch.Tensor:
        """The ``[N,3]`` velocity command the native observation shows."""

    def reset(self, env_ids: torch.Tensor) -> None:
        """Forget the held action of slots starting a new episode."""


class DirectJointControl:
    """The policy's 12 outputs go straight to the joint action term, one
    native tick per policy step."""

    ticks_per_action = 1
    num_actions = 12

    def __init__(self, *, num_envs: int, device: torch.device) -> None:
        self._action = torch.zeros(num_envs, 12, device=device)
        self._command = torch.zeros(num_envs, 3, device=device)

    def start(self, action: torch.Tensor) -> torch.Tensor:
        self._action = action
        return action

    def tick_action(
        self, tick_active: torch.Tensor, native_observation: Callable[[], torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del native_observation
        joint_action = torch.where(tick_active.unsqueeze(1), self._action, torch.zeros_like(self._action))
        return joint_action, torch.zeros((), dtype=torch.bool, device=joint_action.device)

    def held_command(self) -> torch.Tensor:
        return self._command

    def reset(self, env_ids: torch.Tensor) -> None:
        del env_ids


class HeldCommandControl:
    """A body-velocity command held for five native ticks and executed by a
    frozen low-level policy."""

    ticks_per_action = 5
    num_actions = 3

    def __init__(
        self,
        *,
        low_actor: torch.nn.Module,
        command_surface: CommandSurface,
        command: torch.Tensor,
        num_envs: int,
        device: torch.device,
    ) -> None:
        low_actor.eval()
        if any(parameter.requires_grad for parameter in low_actor.parameters()):
            raise RuntimeError("qualified low actor is unexpectedly trainable")
        devices = {value.device for value in tuple(low_actor.parameters()) + tuple(low_actor.buffers())}
        if devices != {torch.device(device)}:
            raise RuntimeError(f"low actor and environment devices differ: {devices}")
        self.low_actor = low_actor
        self.command_surface = command_surface
        # The scene's writable velocity-command tensor the low policy reads.
        self.command = command
        self.previous_command = torch.zeros(num_envs, 3, device=device)
        self._applied = self.previous_command

    def start(self, action: torch.Tensor) -> torch.Tensor:
        applied, bounded = self.command_surface.limit(action, self.previous_command)
        self.previous_command = applied.clone()
        self._applied = applied
        return bounded

    def tick_action(
        self, tick_active: torch.Tensor, native_observation: Callable[[], torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.command[:, :3] = torch.where(
            tick_active.unsqueeze(1), self._applied, torch.zeros_like(self._applied)
        )
        observation = native_observation()
        failed = (observation[:, 9:12] - self.command[:, :3]).abs().amax() > 1.0e-6
        with torch.no_grad():
            joint_action = torch.where(
                tick_active.unsqueeze(1),
                self.low_actor(observation),
                torch.zeros(observation.shape[0], 12, device=observation.device),
            )
        return joint_action, failed

    def held_command(self) -> torch.Tensor:
        return self.previous_command

    def reset(self, env_ids: torch.Tensor) -> None:
        self.previous_command[env_ids] = 0.0
