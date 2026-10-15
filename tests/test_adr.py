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
    "ball_angle_deg": "ball_angle_deg",
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
        adr.set_range("robot_ball_distance_m", 0.8, 0.81)
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

    def test_a_checkpoint_from_before_a_knob_existed_still_loads(self) -> None:
        old = AutomaticDomainRandomization(buffer_size=8)
        old.set_range("ball_goal_distance_m", 0.5, 6.0)
        state = old.state_dict()
        # A four-knob state, as checkpoints before ball_angle_deg wrote it.
        state["ranges"] = state["ranges"][:4]
        state["buffer_count"] = state["buffer_count"][:4]
        state["buffer_successes"] = state["buffer_successes"][:4]
        state.pop("parameter_names", None)
        restored = AutomaticDomainRandomization(buffer_size=8)
        restored.load_state_dict(state)
        self.assertEqual(restored.ranges()[INDEX["ball_goal_distance_m"]].tolist(), [0.5, 6.0])
        start = PARAMETERS[INDEX["ball_angle_deg"]].start
        self.assertEqual(restored.ranges()[INDEX["ball_angle_deg"]].tolist(), [start, start])

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
        adr.set_range("ball_angle_deg", 0.0, 60.0)
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


class LeadKnobTest(unittest.TestCase):
    """A lead knob widens alone: until its upper edge reaches a value, it is
    the only knob probed and every other knob stays at its start."""

    def wide_ranges(self, adr: AutomaticDomainRandomization) -> None:
        adr.set_range("ball_goal_distance_m", 0.15, 5.0)
        adr.set_range("facing_error_deg", 0.0, 120.0)
        adr.set_range("robot_bearing_deg", 0.0, 90.0)
        adr.set_range("ball_angle_deg", 0.0, 40.0)

    def test_before_the_gate_only_the_lead_knob_moves(self) -> None:
        adr = AutomaticDomainRandomization(
            seed=1, probe_fraction=0.5, evaluation_fraction=0.0,
            lead_parameter="robot_ball_distance_m", lead_until=4.0,
        )
        self.wide_ranges(adr)
        adr.set_range("robot_ball_distance_m", 0.45, 2.0)
        batch = adr(4000)
        kind, values = batch.tags["adr_kind"], batch.tags["adr_values"].double()
        probes = kind == KIND["probe"]
        self.assertTrue(bool((batch.tags["adr_parameter"][probes] == INDEX["robot_ball_distance_m"]).all()))
        for parameter in PARAMETERS:
            if parameter.name != "robot_ball_distance_m":
                column = values[:, INDEX[parameter.name]]
                self.assertTrue(torch.allclose(column, torch.full_like(column, parameter.start)), parameter.name)
        self.assertGreater(float(values[:, INDEX["robot_ball_distance_m"]].max()), 1.5)
        # Probe results for other knobs (from before the gate) move nothing.
        self.assertEqual(adr.record(probe_results("ball_goal_distance_m", UPPER, [True] * 400)), 0)

    def test_once_the_lead_knob_reaches_its_target_every_knob_is_in_play(self) -> None:
        adr = AutomaticDomainRandomization(
            seed=2, probe_fraction=0.5, evaluation_fraction=0.0,
            lead_parameter="robot_ball_distance_m", lead_until=4.0, buffer_size=1,
        )
        self.wide_ranges(adr)
        adr.set_range("robot_ball_distance_m", 0.45, 3.9)
        adr.record(probe_results("robot_ball_distance_m", UPPER, [True]))
        self.assertGreaterEqual(float(adr.ranges()[INDEX["robot_ball_distance_m"], UPPER]), 4.0)
        # The gate stays open even if the lead knob later narrows again.
        adr.set_range("robot_ball_distance_m", 0.45, 3.0)
        batch = adr(4000)
        probes = batch.tags["adr_kind"] == KIND["probe"]
        self.assertEqual(set(batch.tags["adr_parameter"][probes].tolist()), set(range(len(PARAMETERS))))
        self.assertGreater(float(batch.tags["adr_values"][:, INDEX["ball_goal_distance_m"]].max()), 1.0)
        restored = AutomaticDomainRandomization(lead_parameter="robot_ball_distance_m", lead_until=4.0)
        restored.load_state_dict(adr.state_dict())
        self.assertTrue(restored.lead_done)


class WholeFieldTest(unittest.TestCase):
    def test_the_limits_reach_every_corner_of_the_field(self) -> None:
        adr = AutomaticDomainRandomization(seed=5, probe_fraction=0.0, evaluation_fraction=0.0)
        for parameter in PARAMETERS:
            adr.set_range(parameter.name, parameter.minimum, parameter.maximum)
        batch = adr(40_000)
        for name, limit in (("ball_xy", BALL_FIELD_LIMIT_XY), ("robot_xy", ROBOT_FIELD_LIMIT_XY)):
            xy = getattr(batch, name)
            # Starts reach within 0.5 m of every edge of the field.
            self.assertGreater(float(xy[:, 0].max()), limit[0] - 0.5, name)
            self.assertLess(float(xy[:, 0].min()), -limit[0] + 0.5, name)
            self.assertGreater(float(xy[:, 1].max()), limit[1] - 0.5, name)
            self.assertLess(float(xy[:, 1].min()), -limit[1] + 0.5, name)
        # The robot can start across the field from the ball.
        separation = torch.linalg.vector_norm(batch.robot_xy - batch.ball_xy, dim=1)
        self.assertGreater(float(separation.max()), 6.0)


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
        values = torch.tensor([[3.0, 1.0, 90.0, 0.0, 0.5]]).repeat(2000, 1)
        batch, _ = reset_geometry(values, torch.Generator().manual_seed(1))
        to_ball = batch.ball_xy - batch.robot_xy
        bearing = torch.atan2(to_ball[:, 1], to_ball[:, 0])
        signed = torch.atan2(torch.sin(batch.robot_yaw - bearing), torch.cos(batch.robot_yaw - bearing))
        self.assertAlmostEqual(float((signed > 0).float().mean()), 0.5, delta=0.05)
        self.assertAlmostEqual(float((batch.goal_side > 0).float().mean()), 0.5, delta=0.05)


if __name__ == "__main__":
    unittest.main()
