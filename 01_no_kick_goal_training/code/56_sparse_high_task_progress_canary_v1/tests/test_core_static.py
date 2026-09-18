from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def source(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


class CoreStaticTests(unittest.TestCase):
    def test_all_core_python_parses_and_has_no_unsafe_torch_load(self) -> None:
        for path in sorted(ROOT.glob("*.py")):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            self.assertNotIn("weights_only=False", path.read_text(encoding="utf-8"))

    def test_trainer_durable_reward_gate_precedes_returns_and_update(self) -> None:
        text = source("train_task_progress_canary.py")
        gate_write = text.index('atomic_json(output / "first_rollout_reward_support_summary.json"')
        gate_guard = text.index('atomic_bytes(output / "FIRST_ROLLOUT_REWARD_SUPPORT_FINAL_GUARD"')
        compute = text.index("runner.alg.compute_returns(observation)")
        update = text.index("losses = runner.alg.update()")
        self.assertLess(gate_write, gate_guard)
        self.assertLess(gate_guard, compute)
        self.assertLess(compute, update)
        self.assertIn('output / "first_rollout_transitions.jsonl"', text)
        self.assertIn('output / "model_first_update.pt"', text)
        self.assertIn('output / "model_final.pt"', text)

    def test_rejected_branch_has_durable_no_update_evidence(self) -> None:
        text = source("train_task_progress_canary.py")
        for token in (
            "storage_cleared_without_update",
            "policy_state_unchanged_without_update",
            "low_state_unchanged_without_update",
            "optimizer_state_empty_after_rejection",
            "optimizer_steps_after_rejection",
            "learned_checkpoint_count",
            "REWARD_SUPPORT_REJECTED_FINAL_GUARD",
        ):
            self.assertIn(token, text)

    def test_environment_freezes_post_state_only_for_tick_active_slots(self) -> None:
        text = source("current_task_progress_env.py")
        self.assertIn("tick_active = active.clone()", text)
        self.assertIn("post_robot_xy[tick_active] = robot_xy[tick_active]", text)
        self.assertIn("post_ball_xy[tick_active] = ball_xy[tick_active]", text)
        self.assertIn('"post_state_source": "SAME_TICK_PHYSICAL_PRE_RESET"', text)
        self.assertIn("self._reset_idx(done_ids)", text)
        self.assertLess(text.index('"post_state_source": "SAME_TICK_PHYSICAL_PRE_RESET"'), text.index("self._reset_idx(done_ids)"))

    def test_no_forbidden_behavior_modules_are_imported(self) -> None:
        imports: set[str] = set()
        for path in sorted(ROOT.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module)
        forbidden_fragments = ("geometric_teacher", "offline_bc", "v4d", "kick", "router", "residual")
        self.assertFalse(any(fragment in name.lower() for name in imports for fragment in forbidden_fragments))

    def test_source_bootstrap_is_package_local_only(self) -> None:
        text = source("source_bootstrap.py")
        self.assertIn("SOURCE_DIR / \"vendor/v14\"", text)
        self.assertIn("SOURCE_DIR / \"vendor/current_qualified\"", text)
        self.assertNotIn("parents[1]", text)

    def test_checkpoint_provenance_and_safe_load_are_v56_specific(self) -> None:
        text = source("runtime_common.py")
        self.assertIn("fresh_recurrent_high_v56_task_progress_70d_3d", text)
        self.assertEqual(text.count("weights_only=True"), 2)
        self.assertIn("V56_PARENT_V55_TEACHER_FIXED_ROWS_SHA256", text)
        self.assertIn("parent_v55_identity", text)


if __name__ == "__main__":
    unittest.main()
