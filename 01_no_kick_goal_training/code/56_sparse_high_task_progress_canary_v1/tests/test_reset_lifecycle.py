from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / "current_task_progress_env.py"
SOURCE = ENV_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE, filename=str(ENV_PATH))


def _class_method(class_name: str, method_name: str) -> ast.FunctionDef:
    for node in TREE.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"missing {class_name}.{method_name}")


def _helper_namespace() -> dict[str, Any]:
    wanted = {
        "BALL_CONTACT_RESET_CACHE_FIELDS",
        "BASE_CONTACT_RESET_CACHE_FIELDS",
        "CONTACT_SENSOR_RESET_TIMER_FIELDS",
        "cached_contact_sensor_state",
        "cached_contact_reset_receipt",
    }
    nodes = [
        node
        for node in TREE.body
        if (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name) and target.id in wanted
                for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            )
        )
        or (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted)
    ]
    namespace: dict[str, Any] = {
        "Any": Any,
        "Sequence": Sequence,
        "torch": torch,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ENV_PATH), "exec"), namespace)
    return namespace


@dataclass
class _FakeContactData:
    net_forces_w: torch.Tensor
    net_forces_w_history: torch.Tensor
    force_matrix_w: torch.Tensor | None = None
    force_matrix_w_history: torch.Tensor | None = None
    current_air_time: torch.Tensor | None = None
    last_air_time: torch.Tensor | None = None
    current_contact_time: torch.Tensor | None = None
    last_contact_time: torch.Tensor | None = None


class _FakeContactSensor:
    def __init__(self, *, num_envs: int, filtered: bool) -> None:
        self._num_envs = num_envs
        self._timestamp = torch.arange(1, num_envs + 1, dtype=torch.float32)
        self._timestamp_last_update = self._timestamp + 10.0
        self._is_outdated = torch.zeros(num_envs, dtype=torch.bool)
        self._data = _FakeContactData(
            net_forces_w=torch.arange(1, num_envs * 6 + 1, dtype=torch.float32).reshape(num_envs, 2, 3),
            net_forces_w_history=torch.arange(1, num_envs * 18 + 1, dtype=torch.float32).reshape(num_envs, 3, 2, 3),
            force_matrix_w=(
                torch.arange(1, num_envs * 6 + 1, dtype=torch.float32).reshape(num_envs, 1, 2, 3)
                if filtered
                else None
            ),
            force_matrix_w_history=(
                torch.arange(1, num_envs * 24 + 1, dtype=torch.float32).reshape(num_envs, 4, 1, 2, 3)
                if filtered
                else None
            ),
            current_air_time=(torch.arange(1, num_envs * 2 + 1, dtype=torch.float32).reshape(num_envs, 2) if not filtered else None),
            last_air_time=(torch.arange(11, 11 + num_envs * 2, dtype=torch.float32).reshape(num_envs, 2) if not filtered else None),
            current_contact_time=(
                torch.arange(21, 21 + num_envs * 2, dtype=torch.float32).reshape(num_envs, 2)
                if not filtered
                else None
            ),
            last_contact_time=(
                torch.arange(31, 31 + num_envs * 2, dtype=torch.float32).reshape(num_envs, 2)
                if not filtered
                else None
            ),
        )
        self.public_data_reads = 0

    @property
    def data(self) -> Any:
        self.public_data_reads += 1
        raise AssertionError("the reset receipt must not trigger lazy sensor.data")

    def reset(self, env_ids: torch.Tensor) -> None:
        ids = env_ids.to(dtype=torch.long).flatten()
        self._timestamp[ids] = 0.0
        self._timestamp_last_update[ids] = 0.0
        self._is_outdated[ids] = True
        self._data.net_forces_w[ids] = 0.0
        self._data.net_forces_w_history[ids] = 0.0
        if self._data.force_matrix_w is not None:
            self._data.force_matrix_w[ids] = 0.0
        if self._data.force_matrix_w_history is not None:
            self._data.force_matrix_w_history[ids] = 0.0
        for field in ("current_air_time", "last_air_time", "current_contact_time", "last_contact_time"):
            value = getattr(self._data, field)
            if value is not None:
                value[ids] = 0.0


class ResetLifecycleStaticTests(unittest.TestCase):
    def test_reset_uses_canonical_ball_state_and_final_sensor_resets(self) -> None:
        reset = _class_method("CurrentTaskProgressHighGoalEnv", "_reset_idx")
        text = ast.unparse(reset)
        self.assertIn("self.ball.data.default_root_state[ids]", text)
        self.assertIn("ball_pose[:, :3] += origins", text)
        self.assertIn("self.ball.write_root_velocity_to_sim(ball_velocity, env_ids=ids)", text)
        self.assertNotIn("ball_z", text)
        self.assertNotIn("self._ball_contact_force()", text)

        reset_calls: dict[str, int] = {}
        build_observation_line = -1
        public_data_lines: list[int] = []
        for node in ast.walk(reset):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (
                    isinstance(node.func.value, ast.Attribute)
                    and isinstance(node.func.value.value, ast.Name)
                    and node.func.value.value.id == "self"
                    and node.func.value.attr in {"ball_contact", "base_contact"}
                    and node.func.attr == "reset"
                ):
                    self.assertEqual(len(node.args), 1)
                    self.assertIsInstance(node.args[0], ast.Name)
                    self.assertEqual(node.args[0].id, "ids")
                    reset_calls[node.func.value.attr] = node.lineno
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"
                    and node.func.attr == "_build_observation"
                ):
                    build_observation_line = node.lineno
            if isinstance(node, ast.Attribute) and node.attr == "data":
                public_data_lines.append(node.lineno)

        self.assertEqual(set(reset_calls), {"ball_contact", "base_contact"})
        self.assertGreater(min(reset_calls.values()), build_observation_line)
        self.assertTrue(public_data_lines)
        self.assertLess(max(public_data_lines), min(reset_calls.values()))


class ResetCacheReceiptTests(unittest.TestCase):
    def test_selected_slots_clear_and_unselected_slots_remain_bit_exact_without_lazy_read(self) -> None:
        helpers = _helper_namespace()
        snapshot = helpers["cached_contact_sensor_state"]
        receipt = helpers["cached_contact_reset_receipt"]
        ball_fields = helpers["BALL_CONTACT_RESET_CACHE_FIELDS"]
        base_fields = helpers["BASE_CONTACT_RESET_CACHE_FIELDS"]

        ball = _FakeContactSensor(num_envs=4, filtered=True)
        base = _FakeContactSensor(num_envs=4, filtered=False)
        selected = torch.tensor([0, 2], dtype=torch.long)
        unselected = torch.tensor([1, 3], dtype=torch.long)
        before_unselected = {
            "ball": snapshot(ball, unselected, ball_fields, sensor_name="robot_ball_contact"),
            "base": snapshot(base, unselected, base_fields, sensor_name="base_contact"),
        }

        ball.reset(selected)
        base.reset(selected)
        result = receipt(ball, base, selected)
        after_unselected = {
            "ball": snapshot(ball, unselected, ball_fields, sensor_name="robot_ball_contact"),
            "base": snapshot(base, unselected, base_fields, sensor_name="base_contact"),
        }

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["selected_env_ids"], [0, 2])
        self.assertEqual(result["read_path"], "PRIVATE_CACHE_NONLAZY_NO_PUBLIC_DATA_ACCESS")
        self.assertEqual(ball.public_data_reads, 0)
        self.assertEqual(base.public_data_reads, 0)
        for sensor_name in before_unselected:
            self.assertEqual(set(before_unselected[sensor_name]), set(after_unselected[sensor_name]))
            for field in before_unselected[sensor_name]:
                self.assertTrue(
                    torch.equal(before_unselected[sensor_name][field], after_unselected[sensor_name][field]),
                    f"unselected {sensor_name}.{field} changed",
                )

    def test_receipt_rejects_one_stale_cached_history_value(self) -> None:
        helpers = _helper_namespace()
        receipt = helpers["cached_contact_reset_receipt"]
        ball = _FakeContactSensor(num_envs=3, filtered=True)
        base = _FakeContactSensor(num_envs=3, filtered=False)
        selected = torch.tensor([1], dtype=torch.long)
        ball.reset(selected)
        base.reset(selected)
        ball._data.force_matrix_w_history[1, 0, 0, 0, 0] = 1.0

        result = receipt(ball, base, selected)

        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["sensor_checks"]["robot_ball_contact"]["cached_reset_buffers_zero"])
        self.assertTrue(result["sensor_checks"]["base_contact"]["pass"])
        self.assertEqual(ball.public_data_reads, 0)
        self.assertEqual(base.public_data_reads, 0)


if __name__ == "__main__":
    unittest.main()
