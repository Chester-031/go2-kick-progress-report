"""The command surface projection must be the exact Euclidean projection.

The reference solves the projection as a small quadratic program over the
convex weights of the hull vertices in float64 with SciPy, independently of
the implementation under test.
"""

from __future__ import annotations

import unittest

import numpy as np
from scipy.optimize import minimize
import torch

from robot_soccer.commands import project_to_convex_hull
from robot_soccer.config import COMMAND_VERTICES

VERTICES = np.asarray(COMMAND_VERTICES, dtype=np.float64)


def reference_projection(point: np.ndarray) -> np.ndarray:
    count = VERTICES.shape[0]
    result = minimize(
        lambda weights: float(np.sum((weights @ VERTICES - point) ** 2)),
        np.full(count, 1.0 / count),
        jac=lambda weights: 2.0 * VERTICES @ (weights @ VERTICES - point),
        bounds=[(0.0, 1.0)] * count,
        constraints=({"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},),
        method="SLSQP",
        options={"ftol": 1.0e-15, "maxiter": 1000},
    )
    return result.x @ VERTICES


class ProjectionTest(unittest.TestCase):
    def test_matches_exact_projection(self) -> None:
        generator = np.random.default_rng(0)
        # The whole physical command box plus points well outside it.
        points = generator.uniform(-0.8, 0.8, size=(300, 3))
        expected = np.stack([reference_projection(point) for point in points])
        actual = project_to_convex_hull(
            torch.as_tensor(points, dtype=torch.float32),
            torch.as_tensor(VERTICES, dtype=torch.float32),
        ).double().numpy()
        error = np.abs(actual - expected).max()
        self.assertLess(error, 1.0e-5, f"max projection error {error:.2e}")

    def test_points_inside_are_unchanged(self) -> None:
        weights = torch.distributions.Dirichlet(torch.ones(len(VERTICES))).sample((200,))
        inside = weights @ torch.as_tensor(VERTICES, dtype=torch.float32)
        projected = project_to_convex_hull(inside, torch.as_tensor(VERTICES, dtype=torch.float32))
        self.assertLess(float((projected - inside).abs().max()), 1.0e-6)


if __name__ == "__main__":
    unittest.main()
