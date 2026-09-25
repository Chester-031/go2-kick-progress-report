from __future__ import annotations

import ast
import pathlib
import sys
import tempfile
import unittest

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_soccer.evaluate import _parser as evaluation_parser  # noqa: E402
from robot_soccer.low_policy import file_sha256  # noqa: E402
from robot_soccer.runtime import load_high_policy, prepare_output_directory  # noqa: E402
from robot_soccer.train import (  # noqa: E402
    _parser as training_parser,
    _validate_training_dimensions,
)


class TrainingContractTests(unittest.TestCase):
    def test_training_dimensions_match_recurrent_minibatches(self) -> None:
        _validate_training_dimensions(4, 2, 1)
        _validate_training_dimensions(4096, 200, 10)
        for num_envs in (1, 2, 3, 5, 6, 7):
            with self.subTest(num_envs=num_envs):
                with self.assertRaises(ValueError):
                    _validate_training_dimensions(num_envs, 200, 1)
        with self.assertRaises(ValueError):
            _validate_training_dimensions(4, 1, 1)

    def test_pre_action_checkpoint_identity_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "initial.pt"
            policy = torch.nn.Linear(2, 1)
            payload = {
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": None,
                "iter": 0,
                "infos": {
                    "architecture": "fresh_recurrent_high_v56_task_progress_70d_3d",
                    "source": "RUN_LOCAL_INITIAL_EVALUATION",
                    "kind": "initial_before_any_act",
                    "v4d_weights_loaded": False,
                    "teacher_loaded": False,
                },
            }
            torch.save(payload, path)
            self.assertEqual(
                load_high_policy(
                    policy,
                    path,
                    expected_checkpoint_sha256=file_sha256(path),
                    require_pre_action_initial=True,
                ),
                0,
            )
            with self.assertRaises(RuntimeError):
                load_high_policy(
                    policy,
                    path,
                    expected_checkpoint_sha256="0" * 64,
                    require_pre_action_initial=True,
                )

    def test_output_directory_must_be_empty(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            output = pathlib.Path(parent) / "run"
            self.assertEqual(prepare_output_directory(output), output.resolve())
            (output / "existing.txt").write_text("occupied\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_output_directory(output)

    def test_device_argument_is_owned_by_isaac_app_launcher(self) -> None:
        for factory in (training_parser, evaluation_parser):
            parser = factory()
            self.assertNotIn("--device", parser._option_string_actions)
            parser.add_argument("--device", default="cuda:0")
            self.assertIn("--device", parser._option_string_actions)

    def test_no_kick_router_or_state_machine(self) -> None:
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "src").rglob("*.py")
        ).lower()
        for token in (
            "kick_strength",
            "kick_action",
            "route_first",
            "runtime_router",
            "phase_latch",
        ):
            self.assertNotIn(token, source)

    def test_compute_returns_uses_no_grad(self) -> None:
        path = ROOT / "src" / "robot_soccer" / "train.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "compute_returns"
        ]
        self.assertEqual(len(calls), 1)
        parent = parents[calls[0]]
        while not isinstance(parent, ast.With):
            parent = parents[parent]
        self.assertEqual(ast.unparse(parent.items[0].context_expr), "torch.no_grad()")

    def test_environment_step_clock_is_initialized_and_advanced(self) -> None:
        path = ROOT / "src" / "robot_soccer" / "environment.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        environment = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "CurrentTaskProgressHighGoalEnv"
        )
        methods = {
            node.name: node
            for node in environment.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        initialized = any(
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == "vector_steps"
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and node.value.value == 0
            for node in ast.walk(methods["__init__"])
        )
        advanced = any(
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Attribute)
            and isinstance(node.target.value, ast.Name)
            and node.target.value.id == "self"
            and node.target.attr == "vector_steps"
            and isinstance(node.op, ast.Add)
            and isinstance(node.value, ast.Constant)
            and node.value.value == 1
            for node in ast.walk(methods["step"])
        )
        self.assertTrue(initialized)
        self.assertTrue(advanced)

    def test_runtime_seed_is_set_after_environment_and_before_runner(self) -> None:
        for name in ("train.py", "evaluate.py"):
            path = ROOT / "src" / "robot_soccer" / name
            tree = ast.parse(path.read_text(encoding="utf-8"))
            run = next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "run"
            )
            calls = {
                "make_environment": [],
                "manual_seed": [],
                "OnPolicyRunner": [],
            }
            for node in ast.walk(run):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Name) and node.func.id in calls:
                    calls[node.func.id].append(node.lineno)
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "torch"
                    and node.func.attr == "manual_seed"
                ):
                    calls["manual_seed"].append(node.lineno)
            self.assertTrue(all(len(values) == 1 for values in calls.values()), name)
            self.assertLess(
                calls["make_environment"][0],
                calls["manual_seed"][0],
            )
            self.assertLess(calls["manual_seed"][0], calls["OnPolicyRunner"][0])


if __name__ == "__main__":
    unittest.main()
