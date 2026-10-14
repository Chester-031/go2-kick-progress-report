"""The whole training loop on the fake scene: real RSL-RL PPO, ADR resets,
metrics, checkpoints and resume.  Only the Isaac Lab environment
construction is replaced (``runtime.make_environment``)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from tests.fake_isaac import FakeGymEnv, FakeSoccerBaseEnv, frozen_low_actor, install_isaaclab_stub

install_isaaclab_stub()

from robot_soccer import environment, runtime, train  # noqa: E402
from robot_soccer.shaping import PotentialShaping, ShapingWeights  # noqa: E402
from robot_soccer.config import (  # noqa: E402
    GAMMA,
    COMMAND_AXIS_RANGES,
    COMMAND_RATE_LIMITS_PER_SECOND,
    COMMAND_VERTICES,
)


def fake_make_environment(*, num_envs, reset_sampler, device, **_):
    actor = frozen_low_actor(device)
    env = environment.CurrentTaskProgressHighGoalEnv(
        low_env=environment.FrozenLowInferenceEnv(FakeGymEnv(FakeSoccerBaseEnv(num_envs, device=device))),
        frozen_low_actor=actor,
        reset_sampler=reset_sampler,
        shaping=PotentialShaping(ShapingWeights(), gamma=GAMMA),
        command_vertices=COMMAND_VERTICES,
        command_axis_ranges=COMMAND_AXIS_RANGES,
        command_rate_limits=COMMAND_RATE_LIMITS_PER_SECOND,
    )
    return env, environment.CurrentTaskProgressHighRslAdapter(env), actor


def arguments(output: Path, **overrides) -> argparse.Namespace:
    args = train._parser().parse_args(["--low-checkpoint", "unused", "--low-checkpoint-sha256", "unused"])
    args.output_dir = output
    args.device = "cpu"
    args.num_envs = 64
    args.steps_per_env = 40
    args.iterations = 3
    args.checkpoint_interval = 2
    args.no_tensorboard = True
    args.adr_buffer_size = 2
    args.adr_probe_fraction = 0.5
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


class TrainLoopTest(unittest.TestCase):
    def test_train_then_resume(self) -> None:
        with tempfile.TemporaryDirectory() as root, mock.patch.object(
            runtime, "make_environment", fake_make_environment
        ):
            first = Path(root) / "first"
            self.assertEqual(train.run(arguments(first)), 0)
            records = [json.loads(line) for line in (first / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([record["iteration"] for record in records], [1, 2, 3])
            for record in records:
                self.assertGreater(record["completed_episodes"], 0)
                self.assertEqual(
                    sum(values["episodes"] for values in record["by_kind"].values()),
                    record["completed_episodes"],
                )
                self.assertEqual(
                    sum(row[4] for row in record["knob_grid"]), record["completed_episodes"]
                )
                self.assertEqual(
                    set(record["adr"]["knobs"]),
                    {"ball_goal_distance_m", "robot_ball_distance_m", "facing_error_deg", "robot_bearing_deg"},
                )
                self.assertGreater(record["timing"]["transitions_per_second"], 0.0)
            probes = sum(
                edge["probe_results"]
                for record in records
                for knob in record["adr"]["knobs"].values()
                for edge in (knob["low_edge"], knob["high_edge"])
            )
            self.assertEqual(probes, sum(record["by_kind"]["probe"]["episodes"] for record in records))
            info = json.loads((first / "run_info.json").read_text())
            self.assertEqual(len(info["adr_parameters"]), 4)
            self.assertEqual(
                sorted(path.name for path in first.glob("*.pt")),
                ["model_iteration_0002.pt", "model_iteration_0003.pt"],
            )

            # Plant distinctive ranges in the checkpoint and resume from it
            # with no edge moves and no fixed-difficulty episodes: every
            # start must then come from the planted ranges.
            checkpoint = torch.load(first / "model_iteration_0002.pt", weights_only=True)
            planted_ranges = [[3.0, 4.0], [1.0, 1.2], [90.0, 120.0], [0.0, 10.0]]
            checkpoint["adr_state"]["ranges"] = planted_ranges
            planted = Path(root) / "planted.pt"
            torch.save(checkpoint, planted)
            second = Path(root) / "second"
            self.assertEqual(
                train.run(
                    arguments(second, resume=planted, adr_buffer_size=10_000, adr_evaluation_fraction=0.0)
                ),
                0,
            )
            resumed = [json.loads(line) for line in (second / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([record["iteration"] for record in resumed], [3])
            cells = {tuple(row[:4]) for row in resumed[0]["knob_grid"]}
            # Grid cells (metrics.KNOB_GRID_EDGES): ball-goal [3,4) or, for
            # an upper-edge probe at 4.0, [4,5); robot-ball [1.0,1.4);
            # facing [90,135); bearing [0,10) or, at 10, [10,20).
            self.assertTrue(cells, "no episodes finished")
            for ball_goal, robot_ball, facing, bearing in cells:
                self.assertIn(ball_goal, (7, 8))
                self.assertEqual(robot_ball, 4)
                self.assertEqual(facing, 4)
                self.assertIn(bearing, (0, 1))
            summary = json.loads((second / "training_summary.json").read_text())
            self.assertEqual(summary["first_iteration"], 3)
            self.assertEqual(summary["adr_final_ranges"], planted_ranges)

if __name__ == "__main__":
    unittest.main()
