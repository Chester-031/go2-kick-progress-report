from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "record_current_policy.py"
SHELL = HERE / "record_current_policy.sh"
PACKAGER = HERE / "prepare_video_inputs.sh"


def load_module():
    spec = importlib.util.spec_from_file_location("record_current_policy", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("recorder module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecorderStaticTests(unittest.TestCase):
    def test_help_does_not_require_isaac(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--assignment", result.stdout)
        self.assertIn("--high-checkpoint", result.stdout)
        self.assertIn("--low-checkpoint", result.stdout)
        self.assertIn("--evidence-rows", result.stdout)

    def test_missing_inputs_fail_before_isaac_import(self):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--assignment",
                "0",
                "--high-checkpoint",
                "/does/not/exist/high.pt",
                "--low-checkpoint",
                "/does/not/exist/low.pt",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("high.pt", result.stderr)
        self.assertNotIn("No module named 'isaaclab'", result.stderr)

    def test_shell_uses_local_isaac_launcher(self):
        result = subprocess.run(
            ["bash", "-n", str(SHELL)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        source = SHELL.read_text(encoding="utf-8")
        self.assertIn("/home/ubuntu/IsaacLab_232/isaaclab.sh -p", source)
        self.assertIn("--enable_cameras", source)

    def test_packaged_input_directory_matches_recorder_default(self):
        module = load_module()
        source = PACKAGER.read_text(encoding="utf-8")
        self.assertEqual(module.DEFAULT_INPUT_DIR.name, "current_policy_inputs")
        self.assertIn('tar --sort=name --owner=0 --group=0 --numeric-owner --mtime=\'UTC 1970-01-01\' -czf "$OUTPUT_TAR" current_policy_inputs', source)

    def test_defaults_and_checkpoint_identities(self):
        module = load_module()
        parser = module.build_parser()
        args = parser.parse_args(["--assignment", "7"])
        self.assertEqual(args.assignment, 7)
        self.assertEqual(
            args.output_dir,
            Path("/home/ubuntu/下载/go2_soccer_phaseA_results/r898_video_bundle"),
        )
        self.assertEqual(
            module.HIGH_CHECKPOINT_SHA256,
            "2cec710ac921b220fe69a7819483dc0fc841da9a9ee35fda6e85927ebd33a919",
        )
        self.assertEqual(
            module.LOW_CHECKPOINT_SHA256,
            "f46d3b760c43664350aac8099ae5bbd32bc1207f97c8814f5d21c9b531c3d96c",
        )

    def test_file_identity_guard(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            payload = b"checkpoint"
            path.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()
            self.assertEqual(module.verify_regular_file(path, expected), path.resolve())
            with self.assertRaises(RuntimeError):
                module.verify_regular_file(path, "0" * 64)

    def test_native_frame_is_captured_after_each_step(self):
        module = load_module()
        events = []

        def step(action):
            events.append(("step", action))
            return "result"

        capture = module.NativeStepFrameCapture(step, lambda: events.append(("frame", None)))
        self.assertEqual(capture("action"), "result")
        self.assertEqual(events, [("step", "action"), ("frame", None)])
        self.assertEqual(capture.count, 1)

    def test_recorder_resets_recurrent_state_before_inference(self):
        source = SCRIPT.read_text(encoding="utf-8")
        reset_position = source.index("policy.reset()")
        inference_position = source.index("policy.act_inference(observation)")
        self.assertLess(reset_position, inference_position)
        self.assertIn("recurrent hidden state did not reset on terminal step", source)

    def test_replay_must_match_archived_case(self):
        module = load_module()
        expected = {
            "evaluation_assignment": 7,
            "outcome": "timeout",
            "contact_seen": True,
            "hard_goal_facing_ball_behind": False,
            "goal_side": 1,
        }
        module.verify_replay(expected, dict(expected), 7)
        for field, replacement in {
            "outcome": "own_goal",
            "contact_seen": False,
            "hard_goal_facing_ball_behind": True,
            "goal_side": -1,
        }.items():
            observed = dict(expected)
            observed[field] = replacement
            with self.assertRaisesRegex(RuntimeError, field):
                module.verify_replay(expected, observed, 7)

    def test_simulation_app_closes_on_every_exit(self):
        source = SCRIPT.read_text(encoding="utf-8")
        finally_position = source.rindex("finally:")
        close_position = source.rindex("simulation_app.close()")
        self.assertLess(finally_position, close_position)


if __name__ == "__main__":
    unittest.main()
