"""Potential-based shaping (Ng, Harada & Russell 1999).

The property that makes shaping safe is Theorem 1 / Corollary 2: with
F(s, s') = gamma * Phi(s') - Phi(s) and Phi = 0 at absorbing states, the
discounted shaping collected along ANY episode adds up to -Phi(s0), so the
value of every policy shifts by the same -Phi(s0) and the optimal policy is
unchanged.  These tests check that sum on random trajectories that end in
each way an episode can end, and check that each potential points the way
it should.
"""

from __future__ import annotations

import math
import unittest

import torch

from robot_soccer.shaping import PotentialShaping, ShapingState, ShapingWeights

GAMMA = 0.99


def random_states(count: int, generator: torch.Generator) -> ShapingState:
    def uniform(low: float, high: float, *shape: int) -> torch.Tensor:
        return low + (high - low) * torch.rand(*shape, generator=generator)

    return ShapingState(
        robot_xy=uniform(-4.0, 4.0, count, 2),
        ball_xy=uniform(-4.0, 4.0, count, 2),
        goal_xy=torch.tensor([[4.11, 0.0]]).expand(count, 2),
        base_height_m=uniform(0.05, 0.45, count),
        up_z=uniform(-1.0, 1.0, count),
    )


class TelescopingTest(unittest.TestCase):
    def test_discounted_shaping_sums_to_minus_initial_potential(self) -> None:
        shaping = PotentialShaping(ShapingWeights(), gamma=GAMMA)
        generator = torch.Generator().manual_seed(0)
        envs, steps = 64, 40
        states = [random_states(envs, generator) for _ in range(steps + 1)]
        # Each env's episode ends at its own step (absorbing afterwards).
        end = torch.randint(1, steps + 1, (envs,), generator=generator)
        total = torch.zeros(envs, dtype=torch.float64)
        for t in range(steps):
            alive = t < end
            terminal = (t + 1) == end
            reward, _ = shaping(states[t], states[t + 1], terminal)
            total += torch.where(alive, (GAMMA ** t) * reward.double(), torch.zeros_like(total))
        expected = -shaping.potential(states[0]).double()
        self.assertLess(float((total - expected).abs().max()), 1e-4)

    def test_terminal_transition_drops_the_next_potential(self) -> None:
        shaping = PotentialShaping(ShapingWeights(), gamma=GAMMA)
        generator = torch.Generator().manual_seed(1)
        before, after = random_states(8, generator), random_states(8, generator)
        reward, _ = shaping(before, after, torch.ones(8, dtype=torch.bool))
        self.assertTrue(torch.allclose(reward, -shaping.potential(before), atol=1e-6))

    def test_components_add_up_to_the_reward(self) -> None:
        shaping = PotentialShaping(ShapingWeights(), gamma=GAMMA)
        generator = torch.Generator().manual_seed(2)
        before, after = random_states(8, generator), random_states(8, generator)
        terminal = torch.tensor([True, False] * 4)
        reward, components = shaping(before, after, terminal)
        self.assertEqual(set(components), {"ball_goal", "robot_ball", "base_height", "upright"})
        self.assertTrue(torch.allclose(sum(components.values()), reward, atol=1e-6))


class DirectionTest(unittest.TestCase):
    def state(self, *, robot=(0.0, 0.0), ball=(1.0, 0.0), height=0.32, up=1.0) -> ShapingState:
        return ShapingState(
            robot_xy=torch.tensor([robot]),
            ball_xy=torch.tensor([ball]),
            goal_xy=torch.tensor([[4.11, 0.0]]),
            base_height_m=torch.tensor([height]),
            up_z=torch.tensor([up]),
        )

    def shaped(self, before: ShapingState, after: ShapingState) -> dict[str, float]:
        shaping = PotentialShaping(ShapingWeights(), gamma=1.0)
        _, components = shaping(before, after, torch.tensor([False]))
        return {name: float(value) for name, value in components.items()}

    def test_ball_toward_the_goal_centre_is_rewarded(self) -> None:
        components = self.shaped(self.state(ball=(2.0, 1.0)), self.state(ball=(3.0, 0.5)))
        self.assertGreater(components["ball_goal"], 0.0)

    def test_robot_toward_the_ball_is_rewarded(self) -> None:
        components = self.shaped(self.state(robot=(-1.0, 0.0)), self.state(robot=(0.0, 0.0), ball=(1.0, 0.0)))
        self.assertGreater(components["robot_ball"], 0.0)

    def test_sagging_and_tipping_are_penalized_and_recovering_is_rewarded(self) -> None:
        nominal = ShapingWeights().nominal_base_height_m
        sag = self.shaped(self.state(height=nominal), self.state(height=nominal - 0.1))
        self.assertLess(sag["base_height"], 0.0)
        recover = self.shaped(self.state(height=nominal - 0.1), self.state(height=nominal))
        self.assertGreater(recover["base_height"], 0.0)
        tip = self.shaped(self.state(up=1.0), self.state(up=math.cos(0.5)))
        self.assertLess(tip["upright"], 0.0)

    def test_gamma_must_match_ppo(self) -> None:
        with self.assertRaises(ValueError):
            PotentialShaping(ShapingWeights(), gamma=1.5)


if __name__ == "__main__":
    unittest.main()
