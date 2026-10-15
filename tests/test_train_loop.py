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
from robot_soccer.controls import DirectJointControl  # noqa: E402
from tests.test_environment_characterization import held_command_control  # noqa: E402


MADE_WITH_GAMMA: list[float] = []


def fake_make_environment(*, control, num_envs, reset_sampler, device, gamma, **_):
    """runtime.make_environment on the fake scene."""

    MADE_WITH_GAMMA.append(gamma)
    if control == "hierarchical":
        actor = frozen_low_actor(device)
        low_env = environment.FrozenLowInferenceEnv(FakeGymEnv(FakeSoccerBaseEnv(num_envs, device=device)))
        env = environment.CurrentTaskProgressHighGoalEnv(
            low_env=low_env,
            control=held_command_control(low_env, actor),
            reset_sampler=reset_sampler,
            shaping=PotentialShaping(ShapingWeights(), gamma=gamma),
        )
        return env, environment.CurrentTaskProgressHighRslAdapter(env), actor
    low_env = environment.FrozenLowInferenceEnv(
        FakeGymEnv(FakeSoccerBaseEnv(num_envs, device=device, direct_joint_control=True))
    )
    env = environment.CurrentTaskProgressHighGoalEnv(
        low_env=low_env,
        control=DirectJointControl(num_envs=num_envs, device=torch.device(device)),
        reset_sampler=reset_sampler,
        shaping=PotentialShaping(ShapingWeights(), gamma=gamma),
        native_reward_scale=1.0,
    )
    return env, environment.CurrentTaskProgressHighRslAdapter(env), None


def arguments(output: Path, **overrides) -> argparse.Namespace:
    args = train._parser().parse_args(
        ["--control", "hierarchical", "--low-checkpoint", "unused", "--low-checkpoint-sha256", "unused"]
    )
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
                    sum(row[5] for row in record["knob_grid"]), record["completed_episodes"]
                )
                self.assertEqual(
                    set(record["adr"]["knobs"]),
                    {"ball_goal_distance_m", "robot_ball_distance_m", "facing_error_deg", "robot_bearing_deg",
                     "ball_angle_deg"},
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
            self.assertEqual(len(info["adr_parameters"]), 5)
            self.assertEqual(
                sorted(path.name for path in first.glob("*.pt")),
                ["model_iteration_0002.pt", "model_iteration_0003.pt"],
            )

            # Plant distinctive ranges in the checkpoint and resume from it
            # with no edge moves and no fixed-difficulty episodes: every
            # start must then come from the planted ranges.
            checkpoint = torch.load(first / "model_iteration_0002.pt", weights_only=True)
            planted_ranges = [[3.0, 4.0], [1.0, 1.2], [90.0, 120.0], [0.0, 10.0], [5.0, 15.0]]
            checkpoint["adr_state"]["ranges"] = planted_ranges
            planted = Path(root) / "planted.pt"
            torch.save(checkpoint, planted)
            second = Path(root) / "second"
            self.assertEqual(
                train.run(
                    arguments(second, resume=planted, adr_buffer_size=10_000, adr_evaluation_fraction=0.0,
                              adr_lead_knob="none")
                ),
                0,
            )
            resumed = [json.loads(line) for line in (second / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([record["iteration"] for record in resumed], [3])
            cells = {tuple(row[:5]) for row in resumed[0]["knob_grid"]}
            # Grid cells (metrics.KNOB_GRID_EDGES): ball-goal [3,4) or, for
            # an upper-edge probe at 4.0, [4,5); robot-ball [1.0,1.4);
            # facing [90,135); bearing [0,10) or, at 10, [10,20); ball angle
            # [2,10) or [10,20).
            self.assertTrue(cells, "no episodes finished")
            for ball_goal, robot_ball, facing, bearing, ball_angle in cells:
                self.assertIn(ball_angle, (1, 2))
                self.assertIn(ball_goal, (7, 8))
                self.assertEqual(robot_ball, 4)
                self.assertEqual(facing, 4)
                self.assertIn(bearing, (0, 1))
            summary = json.loads((second / "training_summary.json").read_text())
            self.assertEqual(summary["first_iteration"], 3)
            self.assertEqual(summary["adr_final_ranges"], planted_ranges)

class DirectJointTrainLoopTest(unittest.TestCase):
    def test_position_and_velocity_controls_train_a_feed_forward_policy(self) -> None:
        for control in ("position", "velocity"):
            with self.subTest(control=control), tempfile.TemporaryDirectory() as root, mock.patch.object(
                runtime, "make_environment", fake_make_environment
            ):
                output = Path(root) / control
                args = arguments(output, control=control, steps_per_env=60, iterations=2, checkpoint_interval=2)
                self.assertEqual(train.run(args), 0)
                records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
                self.assertEqual(len(records), 2)
                self.assertGreater(sum(record["completed_episodes"] for record in records), 0)
                checkpoint = torch.load(output / "model_iteration_0002.pt", weights_only=True)
                self.assertFalse(any(name.startswith("memory_") for name in checkpoint["model_state_dict"]))
                summary = json.loads((output / "training_summary.json").read_text())
                self.assertEqual(summary["control"], control)

    def test_gamma_override_reaches_ppo_and_the_shaping(self) -> None:
        with tempfile.TemporaryDirectory() as root, mock.patch.object(
            runtime, "make_environment", fake_make_environment
        ):
            output = Path(root) / "gamma"
            args = arguments(output, control="position", steps_per_env=30, iterations=1, checkpoint_interval=1,
                             gamma=0.998)
            MADE_WITH_GAMMA.clear()
            self.assertEqual(train.run(args), 0)
            self.assertEqual(MADE_WITH_GAMMA, [0.998])
            info = json.loads((output / "run_info.json").read_text())
            self.assertEqual(info["ppo_gamma"], 0.998)


if __name__ == "__main__":
    unittest.main()
