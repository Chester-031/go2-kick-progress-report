"""Automatic domain randomization over the four reset knobs."""

from __future__ import annotations

import itertools
import math
import unittest

import torch

from robot_soccer.adr import (
    BALL_FIELD_LIMIT_XY,
    EPISODE_KINDS,
    LOWER,
    PARAMETERS,
    ROBOT_FIELD_LIMIT_XY,
    UPPER,
    AutomaticDomainRandomization,
    reset_geometry,
)
from robot_soccer.metrics import initial_geometry

INDEX = {parameter.name: index for index, parameter in enumerate(PARAMETERS)}
KIND = {name: index for index, name in enumerate(EPISODE_KINDS)}
MEASURED = {
    "ball_goal_distance_m": "ball_goal_m",
    "robot_ball_distance_m": "robot_ball_m",
    "facing_error_deg": "facing_deg",
    "robot_bearing_deg": "bearing_deg",
}


def probe_results(parameter: str, edge: int, successes: list[bool]) -> dict[str, torch.Tensor]:
    count = len(successes)
    return {
        "adr_kind": torch.full((count,), KIND["probe"]),
        "adr_parameter": torch.full((count,), INDEX[parameter]),
        "adr_edge": torch.full((count,), edge),
        "goal": torch.tensor(successes),
    }


def measure(batch) -> dict[str, torch.Tensor]:
    return initial_geometry(
        {
            "reset_robot_xy_m": batch.robot_xy,
            "reset_ball_xy_m": batch.ball_xy,
            "reset_robot_yaw_rad": batch.robot_yaw,
            "goal_side": batch.goal_side,
        }
    )


def on_field(batch, margin: float = 1e-5) -> torch.Tensor:
    robot, ball = ROBOT_FIELD_LIMIT_XY, BALL_FIELD_LIMIT_XY
    return (
        (batch.robot_xy[:, 0].abs() <= robot[0] + margin)
        & (batch.robot_xy[:, 1].abs() <= robot[1] + margin)
        & (batch.ball_xy[:, 0].abs() <= ball[0] + margin)
        & (batch.ball_xy[:, 1].abs() <= ball[1] + margin)
    )


class RangeTest(unittest.TestCase):
    def test_ranges_start_as_points(self) -> None:
        adr = AutomaticDomainRandomization()
        for parameter, (low, high) in zip(PARAMETERS, adr.ranges().tolist()):
            self.assertEqual(low, parameter.start)
            self.assertEqual(high, parameter.start)

    def test_first_move_is_the_smallest_step(self) -> None:
        adr = AutomaticDomainRandomization(buffer_size=8)
        adr.record(probe_results("ball_goal_distance_m", UPPER, [True] * 7))
        self.assertEqual(adr.record(probe_results("ball_goal_distance_m", UPPER, [True])), 1)
        parameter = PARAMETERS[INDEX["ball_goal_distance_m"]]
        low, high = adr.ranges()[INDEX["ball_goal_distance_m"]].tolist()
        self.assertAlmostEqual(high, parameter.start + parameter.min_step)
        self.assertAlmostEqual(low, parameter.start)

    def test_steps_grow_with_distance_from_the_start(self) -> None:
        parameter = PARAMETERS[INDEX["ball_goal_distance_m"]]
        self.assertAlmostEqual(parameter.step_at(parameter.start), parameter.min_step)
        middle = (parameter.start + parameter.maximum) / 2.0
        self.assertAlmostEqual(parameter.step_at(middle), (parameter.min_step + parameter.max_step) / 2.0)
        self.assertAlmostEqual(parameter.step_at(parameter.maximum), parameter.max_step)
        # An edge that moves out every time reaches the limit in far fewer
        # moves than it would at the smallest step.
        adr = AutomaticDomainRandomization(buffer_size=1)
        moves = 0
        while adr.ranges()[INDEX["ball_goal_distance_m"], UPPER] < parameter.maximum:
            adr.record(probe_results("ball_goal_distance_m", UPPER, [True]))
            moves += 1
        self.assertLess(moves, (parameter.maximum - parameter.start) / parameter.min_step / 4)

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
        for _ in range(200):
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
    def test_kinds_values_and_every_start_on_the_field(self) -> None:
        adr = AutomaticDomainRandomization(probe_fraction=0.3, evaluation_fraction=0.1, seed=3)
        adr.set_range("ball_goal_distance_m", 0.15, 4.0)
        adr.set_range("robot_ball_distance_m", 0.4, 2.3)
        adr.set_range("facing_error_deg", 0.0, 90.0)
        adr.set_range("robot_bearing_deg", 0.0, 180.0)
        batch = adr(20_000)
        self.assertTrue(bool(on_field(batch).all()))
        kind = batch.tags["adr_kind"]
        values = batch.tags["adr_values"].double()
        self.assertAlmostEqual(float((kind == KIND["probe"]).float().mean()), 0.3, delta=0.02)
        self.assertAlmostEqual(float((kind == KIND["evaluation"]).float().mean()), 0.1, delta=0.02)
        ranges = adr.ranges()
        training = kind == KIND["train"]
        self.assertTrue(bool(((values[training] >= ranges[:, 0] - 1e-6) & (values[training] <= ranges[:, 1] + 1e-6)).all()))
        probes = kind == KIND["probe"]
        parameter = batch.tags["adr_parameter"][probes]
        pinned = ranges[parameter, batch.tags["adr_edge"][probes]]
        chosen = values[probes].gather(1, parameter.unsqueeze(1)).squeeze(1)
        self.assertTrue(torch.allclose(chosen, pinned, atol=1e-6))
        # The starts realize the recorded knob values.
        measured = measure(batch)
        for name, key in MEASURED.items():
            tolerance = 1e-2 if name.endswith("deg") else 1e-4
            self.assertLess(float((measured[key] - values[:, INDEX[name]]).abs().max()), tolerance, name)
        evaluation = kind == KIND["evaluation"]
        for difficulty in adr.evaluation_difficulties:
            rows = evaluation & (batch.tags["adr_difficulty"] == difficulty)
            self.assertGreater(int(rows.sum()), 100)

    def test_difficulty_spans_from_the_start_to_the_limits(self) -> None:
        adr = AutomaticDomainRandomization()
        for parameter, (low, high) in zip(PARAMETERS, adr.difficulty_ranges(0.0).tolist()):
            self.assertAlmostEqual(low, parameter.start)
            self.assertAlmostEqual(high, parameter.start)
        for parameter, (low, high) in zip(PARAMETERS, adr.difficulty_ranges(1.0).tolist()):
            self.assertAlmostEqual(low, parameter.minimum)
            self.assertAlmostEqual(high, parameter.maximum, places=5)


class GeometryTest(unittest.TestCase):
    def test_every_corner_realizes_its_knobs(self) -> None:
        corners = torch.tensor(
            list(itertools.product(*[(parameter.minimum, parameter.maximum) for parameter in PARAMETERS]))
        ).repeat(200, 1)
        batch, legal = reset_geometry(corners, torch.Generator().manual_seed(0))
        # The flag agrees with the field limits (up to float32 rounding).
        self.assertTrue(bool((~legal | on_field(batch)).all()))
        self.assertTrue(bool((legal | ~on_field(batch, margin=-1e-4)).all()))
        measured = measure(batch)
        for name, key in MEASURED.items():
            tolerance = 1e-2 if name.endswith("deg") else 1e-4
            self.assertLess(float((measured[key] - corners[:, INDEX[name]]).abs().max()), tolerance, name)

    def test_start_is_the_robot_right_behind_the_ball_facing_the_goal(self) -> None:
        start = torch.tensor([[parameter.start for parameter in PARAMETERS]]).repeat(500, 1)
        batch, legal = reset_geometry(start, torch.Generator().manual_seed(1))
        self.assertTrue(bool(legal.all()))
        goal_x = batch.goal_side.double() * 4.11
        to_goal = torch.atan2(-batch.robot_xy[:, 1].double(), goal_x - batch.robot_xy[:, 0].double())
        heading_error = torch.atan2(torch.sin(batch.robot_yaw - to_goal), torch.cos(batch.robot_yaw - to_goal))
        self.assertLess(float(heading_error.abs().max()), math.radians(1.5))
        ball_ahead = batch.goal_side.double() * (batch.ball_xy[:, 0] - batch.robot_xy[:, 0]).double()
        self.assertTrue(bool((ball_ahead > 0.4).all()))

    def test_both_goal_sides_and_facing_signs_occur(self) -> None:
        values = torch.tensor([[3.0, 1.0, 90.0, 0.0]]).repeat(2000, 1)
        batch, _ = reset_geometry(values, torch.Generator().manual_seed(1))
        to_ball = batch.ball_xy - batch.robot_xy
        bearing = torch.atan2(to_ball[:, 1], to_ball[:, 0])
        signed = torch.atan2(torch.sin(batch.robot_yaw - bearing), torch.cos(batch.robot_yaw - bearing))
        self.assertAlmostEqual(float((signed > 0).float().mean()), 0.5, delta=0.05)
        self.assertAlmostEqual(float((batch.goal_side > 0).float().mean()), 0.5, delta=0.05)


if __name__ == "__main__":
    unittest.main()
