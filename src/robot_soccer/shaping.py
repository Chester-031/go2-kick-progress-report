"""Potential-based reward shaping that cannot change the optimal policy.

Ng, Harada & Russell, "Policy invariance under reward transformations"
(ICML 1999), Theorem 1: adding F(s, s') = gamma * Phi(s') - Phi(s) to the
reward, for any function Phi of the state, keeps every optimal policy
optimal, and no other kind of additive shaping guarantees that.  Two
conditions make it hold here:

- ``gamma`` must be the discount PPO uses for the same transitions;
- Phi is 0 at absorbing states, so on the transition into a terminal state
  (goal, own goal, fall, boundary, joint limit, or the end of the finite
  horizon, which the observation's time-remaining feature makes part of
  the state) the shaping is just -Phi(s).

Then the discounted shaping along any episode adds up to -Phi(s0)
(Corollary 2: V' = V - Phi), whatever the policy does, so shaping only
changes how quickly credit reaches the actions, never which behaviour is
best.  Falling early, for example, earns nothing net: the -Phi(s) bonus on
the fall transition exactly cancels the shaping collected before it.

Phi is a weighted sum of four potentials, each higher in better states:

- ``ball_goal``: minus the ball's distance to the centre of the goal it is
  to be scored in (pulls the ball to the middle of the goal);
- ``robot_ball``: minus the robot's distance to the ball;
- ``base_height``: minus how far the base is from its nominal standing
  height;
- ``upright``: minus (1 - cos of the base's tilt from vertical).

The last two only make staying up and getting back up easier to learn.
Because they are potentials, holding a good posture earns nothing per
step; only changes in posture are scored.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ShapingWeights:
    """Weight of each potential (reward per unit of its distance) and the
    nominal standing height of the Go2 base."""

    ball_goal: float = 1.0
    robot_ball: float = 0.5
    base_height: float = 2.0
    upright: float = 0.5
    # The stock Go2 settles at 0.28 m standing on its PD drive (Isaac Lab).
    nominal_base_height_m: float = 0.28


@dataclass(frozen=True)
class ShapingState:
    """What the potentials read from a batch of states: robot and ball
    positions and the goal centre (``[N,2]``, one frame), the base height
    (``[N]`` m) and ``up_z`` (``[N]``), the world-z component of the base's
    up axis (1 upright, 0 on its side, -1 upside down)."""

    robot_xy: torch.Tensor
    ball_xy: torch.Tensor
    goal_xy: torch.Tensor
    base_height_m: torch.Tensor
    up_z: torch.Tensor


class PotentialShaping:
    """F(s, s') = gamma Phi(s') - Phi(s), with Phi(s') = 0 when s' is terminal."""

    def __init__(self, weights: ShapingWeights, *, gamma: float) -> None:
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1] and equal to the PPO discount")
        self.weights = weights
        self.gamma = float(gamma)

    def potentials(self, state: ShapingState) -> dict[str, torch.Tensor]:
        """Each weighted potential of each state, by name."""

        weights = self.weights
        return {
            "ball_goal": -weights.ball_goal * torch.linalg.vector_norm(state.ball_xy - state.goal_xy, dim=1),
            "robot_ball": -weights.robot_ball * torch.linalg.vector_norm(state.ball_xy - state.robot_xy, dim=1),
            "base_height": -weights.base_height * (state.base_height_m - weights.nominal_base_height_m).abs(),
            "upright": -weights.upright * (1.0 - state.up_z),
        }

    def potential(self, state: ShapingState) -> torch.Tensor:
        """Phi(s), the sum of the weighted potentials."""

        return sum(self.potentials(state).values())

    def __call__(
        self, before: ShapingState, after: ShapingState, terminal: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """The shaping reward of each transition and its part from each
        potential (the parts add up to the reward)."""

        keep = (~terminal).to(before.robot_xy.dtype)
        start = self.potentials(before)
        end = self.potentials(after)
        components = {name: self.gamma * keep * end[name] - start[name] for name in start}
        return sum(components.values()), components
