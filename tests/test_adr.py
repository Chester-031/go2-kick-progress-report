"""Automatic domain randomization over the three reset knobs."""

from __future__ import annotations

import itertools
import math
import unittest

import torch

from robot_soccer.adr import (
    EPISODE_KINDS,
    LOWER,
    PARAMETERS,
    UPPER,
    AutomaticDomainRandomization,
    reset_geometry,
)
from robot_soccer.metrics import initial_geometry

INDEX = {parameter.name: index for index, parameter in enumerate(PARAMETERS)}
KIND = {name: index for index, name in enumerate(EPISODE_KINDS)}


def probe_results(parameter: str, edge: int, successes: list[bool]) -> dict[str, torch.Tensor]:
    count = len(successes)
    return {
        "adr_kind": torch.full((count,), KIND["probe"]),
        "adr_parameter": torch.full((count,), INDEX[parameter]),
        "adr_edge": torch.full((count,), edge),
        "goal": torch.tensor(successes),
    }


class RangeTest(unittest.TestCase):
    def test_ranges_start_as_points(self) -> None:
        adr = AutomaticDomainRandomization()
        for parameter, (low, high) in zip(PARAMETERS, adr.ranges().tolist()):
            self.assertEqual(low, parameter.start)
            self.assertEqual(high, parameter.start)

    def test_full_buffer_of_successes_moves_the_edge_out_one_step(self) -> None:
        adr = AutomaticDomainRandomization(buffer_size=8)
        moves = adr.record(probe_results("ball_goal_distance_m", UPPER, [True] * 7))
        self.assertEqual(moves, 0)
        moves = adr.record(probe_results("ball_goal_distance_m", UPPER, [True]))
        self.assertEqual(moves, 1)
        parameter = PARAMETERS[INDEX["ball_goal_distance_m"]]
        low, high = adr.ranges()[INDEX["ball_goal_distance_m"]].tolist()
        self.assertAlmostEqual(high, parameter.start + parameter.step)
        self.assertAlmostEqual(low, parameter.start)

    def test_failures_move_the_edge_in_but_never_past_the_other_edge(self) -> None:
        adr = AutomaticDomainRandomization(buffer_size=4)
        adr.set_range("robot_ball_distance_m", 0.8, 0.85)
        adr.record(probe_results("robot_ball_distance_m", UPPER, [False] * 4))
        self.assertEqual(adr.ranges()[INDEX["robot_ball_distance_m"]].tolist(), [0.8, 0.8])

    def test_middling_success_holds_and_clears_the_buffer(self) -> None:
        adr = AutomaticDomainRandomization(buffer_size=4)
        adr.set_range("facing_error_deg", 0.0, 30.0)
        self.assertEqual(adr.record(probe_results("facing_error_deg", UPPER, [True, False, True, False])), 0)
        self.assertEqual(adr.ranges()[INDEX["facing_error_deg"]].tolist(), [0.0, 30.0])
        # The next decision uses only new results.
        self.assertEqual(adr.record(probe_results("facing_error_deg", UPPER, [True] * 4)), 1)

    def test_edges_stop_at_the_limits(self) -> None:
        adr = AutomaticDomainRandomization(buffer_size=1)
        for _ in range(100):
            adr.record(probe_results("facing_error_deg", UPPER, [True]))
            adr.record(probe_results("facing_error_deg", LOWER, [True]))
        self.assertEqual(adr.ranges()[INDEX["facing_error_deg"]].tolist(), [0.0, 180.0])

    def test_only_probes_move_edges(self) -> None:
        adr = AutomaticDomainRandomization(buffer_size=1)
        results = probe_results("ball_goal_distance_m", UPPER, [True] * 3)
        results["adr_kind"] = torch.tensor([KIND["train"], KIND["evaluation"], KIND["train"]])
        self.assertEqual(adr.record(results), 0)

    def test_state_round_trip(self) -> None:
        adr = AutomaticDomainRandomization(buffer_size=8)
        adr.set_range("ball_goal_distance_m", 0.75, 3.0)
        adr.record(probe_results("ball_goal_distance_m", UPPER, [True] * 5))
        restored = AutomaticDomainRandomization(buffer_size=8)
        restored.load_state_dict(adr.state_dict())
        self.assertTrue(torch.equal(restored.ranges(), adr.ranges()))
        # The partly filled buffer survives: three more results decide.
        self.assertEqual(restored.record(probe_results("ball_goal_distance_m", UPPER, [True] * 3)), 1)


class SamplingTest(unittest.TestCase):
    def test_kinds_and_values(self) -> None:
        adr = AutomaticDomainRandomization(probe_fraction=0.3, evaluation_fraction=0.1, seed=3)
        adr.set_range("ball_goal_distance_m", 1.0, 4.0)
        adr.set_range("robot_ball_distance_m", 0.6, 1.5)
        adr.set_range("facing_error_deg", 0.0, 90.0)
        batch = adr(20_000)
        kind = batch.tags["adr_kind"]
        values = batch.tags["adr_values"]
        self.assertAlmostEqual(float((kind == KIND["probe"]).float().mean()), 0.3, delta=0.02)
        self.assertAlmostEqual(float((kind == KIND["evaluation"]).float().mean()), 0.1, delta=0.02)
        ranges = adr.ranges()
        training = kind == KIND["train"]
        self.assertTrue(bool(((values[training] >= ranges[:, 0]) & (values[training] <= ranges[:, 1])).all()))
        probes = kind == KIND["probe"]
        pinned = ranges[batch.tags["adr_parameter"][probes], batch.tags["adr_edge"][probes]]
        chosen = values[probes].gather(1, batch.tags["adr_parameter"][probes].unsqueeze(1)).squeeze(1)
        self.assertTrue(torch.allclose(chosen.double(), pinned, atol=1e-6))
        evaluation = kind == KIND["evaluation"]
        for difficulty in adr.evaluation_difficulties:
            rows = evaluation & (batch.tags["adr_difficulty"] == difficulty)
            self.assertGreater(int(rows.sum()), 100)
            fixed = adr.difficulty_ranges(difficulty)
            self.assertTrue(bool(((values[rows] >= fixed[:, 0] - 1e-6) & (values[rows] <= fixed[:, 1] + 1e-6)).all()))

    def test_difficulty_spans_from_the_start_to_the_limits(self) -> None:
        adr = AutomaticDomainRandomization()
        for parameter, (low, high) in zip(PARAMETERS, adr.difficulty_ranges(0.0).tolist()):
            self.assertAlmostEqual(low, parameter.start)
            self.assertAlmostEqual(high, parameter.start)
        for parameter, (low, high) in zip(PARAMETERS, adr.difficulty_ranges(1.0).tolist()):
            self.assertAlmostEqual(low, parameter.minimum)
            self.assertAlmostEqual(high, parameter.maximum, places=5)


class GeometryTest(unittest.TestCase):
    def test_start_realizes_the_requested_knobs_and_stays_on_the_field(self) -> None:
        # Every corner of the knob limits, many draws of the free choices.
        corners = torch.tensor(
            list(itertools.product(*[(parameter.minimum, parameter.maximum) for parameter in PARAMETERS]))
        ).repeat(500, 1)
        batch = reset_geometry(corners, torch.Generator().manual_seed(0))
        for name in ("robot_xy", "ball_xy"):
            xy = getattr(batch, name)
            self.assertLessEqual(float(xy[:, 0].abs().max()), 3.65, name)
            self.assertLessEqual(float(xy[:, 1].abs().max()), 2.15, name)
        measured = initial_geometry(
            {
                "reset_robot_xy_m": batch.robot_xy,
                "reset_ball_xy_m": batch.ball_xy,
                "reset_robot_yaw_rad": batch.robot_yaw,
                "goal_side": batch.goal_side,
            }
        )
        self.assertLess(float((measured["ball_goal_m"] - corners[:, INDEX["ball_goal_distance_m"]]).abs().max()), 1e-4)
        self.assertLess(float((measured["robot_ball_m"] - corners[:, INDEX["robot_ball_distance_m"]]).abs().max()), 1e-4)
        self.assertLess(float((measured["facing_deg"] - corners[:, INDEX["facing_error_deg"]]).abs().max()), 1e-2)
        self.assertEqual(set(batch.goal_side.tolist()), {-1, 1})

    def test_both_goal_sides_and_facing_signs_occur(self) -> None:
        values = torch.tensor([[3.0, 1.0, 90.0]]).repeat(2000, 1)
        batch = reset_geometry(values, torch.Generator().manual_seed(1))
        to_ball = batch.ball_xy - batch.robot_xy
        bearing = torch.atan2(to_ball[:, 1], to_ball[:, 0])
        signed = torch.atan2(torch.sin(batch.robot_yaw - bearing), torch.cos(batch.robot_yaw - bearing))
        self.assertAlmostEqual(float((signed > 0).float().mean()), 0.5, delta=0.05)
        self.assertAlmostEqual(float((batch.goal_side > 0).float().mean()), 0.5, delta=0.05)
        self.assertTrue(math.isclose(float(signed.abs().mean()), math.pi / 2, rel_tol=1e-3))


if __name__ == "__main__":
    unittest.main()
