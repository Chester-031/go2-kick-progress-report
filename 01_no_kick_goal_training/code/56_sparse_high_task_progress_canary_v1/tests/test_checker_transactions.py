from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import check_sparse_high_task_progress_canary as checker


def _load_core_fixture():
    name = "v56_transaction_core_fixture"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests/test_canary_protocol.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE_FIXTURE = _load_core_fixture()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(checker.canonical_json_bytes(value))


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(checker.canonical_json_bytes(row) for row in rows))


def _component_update(name: str) -> dict[str, object]:
    return {
        "changed": True,
        "parameter_names": [name],
        "parameter_count": 1,
        "maximum_absolute_delta": 0.1,
        "l2_delta": 0.1,
    }


def _heads() -> dict[str, object]:
    return {
        "actor": {
            "weight_shape": [3, 128],
            "weight_l2": 1.0,
            "weight_max_abs": 0.1,
            "weight_nonzero": True,
        },
        "critic": {
            "weight_shape": [1, 128],
            "weight_l2": 1.0,
            "weight_max_abs": 0.1,
            "weight_nonzero": True,
        },
    }


class SyntheticTransaction:
    """A complete filesystem transaction for either legal V56 branch."""

    INITIAL_STATE = "a" * 64
    FIRST_STATE = "b" * 64
    FINAL_STATE = "c" * 64
    SOURCE_MANIFEST = "d" * 64

    def __init__(self, *, support_pass: bool) -> None:
        self.support_pass = support_pass
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        suffix = "pass" if support_pass else "reject"
        self.run = self.base / f"sparse_high_task_progress_canary_v1_123_synthetic_{suffix}"
        self.evidence = self.run / "evidence"
        self.contact_reset_dir = self.evidence / "contact_reset_probe"
        self.initial_dir = self.evidence / "initial"
        self.training_dir = self.evidence / "training"
        self.final_dir = self.evidence / "final"
        self.result_dir = self.evidence / "result"
        for path in (self.contact_reset_dir, self.initial_dir, self.training_dir, self.final_dir, self.result_dir):
            path.mkdir(parents=True, exist_ok=True)

        self._build_task_snapshot()
        self._build_runtime_roots()
        self.environment = self._runtime_environment()
        self._write_environment()
        _write_json(self.contact_reset_dir / "contact_reset_lifecycle_probe.json", {})
        (self.contact_reset_dir / "CONTACT_RESET_LIFECYCLE_PROBE_FINAL_GUARD").write_text(
            "PASS\n", encoding="ascii"
        )
        self.source_receipts = self._source_receipts()
        self.initial = self._build_evaluation("initial")
        self.support = self._build_support()
        self.first: dict[str, object] | None = None
        self.training: dict[str, object] | None = None
        self.final: dict[str, object] | None = None
        if self.support_pass:
            self.first, self.training = self._build_training()
            self.final = self._build_evaluation("final")
        self._write_result()
        self._write_transaction_files(in_job=True)
        self._make_snapshot_read_only()
        self._backup_mutable_files()

    def cleanup(self) -> None:
        snapshot = self.run / "source/runtime_project/source/isaac_go2_low_level_rl"
        if snapshot.exists():
            for path in sorted(snapshot.rglob("*"), key=lambda item: len(item.parts), reverse=True):
                try:
                    path.chmod(0o755 if path.is_dir() else 0o644)
                except OSError:
                    pass
            snapshot.chmod(0o755)
        self.temporary.cleanup()

    def _build_task_snapshot(self) -> None:
        source_parent = self.run / "source/runtime_project/source"
        snapshot = source_parent / "isaac_go2_low_level_rl"
        package = snapshot / "isaac_go2_low_level_rl"
        package.mkdir(parents=True)
        (snapshot / "setup.py").write_text("# synthetic task snapshot\n", encoding="utf-8")
        (package / "__init__.py").write_text("# synthetic task package\n", encoding="utf-8")
        lines = [
            f"{_sha(path)}  {path.relative_to(source_parent).as_posix()}\n"
            for path in sorted(item for item in snapshot.rglob("*") if item.is_file())
        ]
        manifest = self.run / "source/TASK_PACKAGE_RUNTIME.sha256"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("".join(lines), encoding="ascii")
        self.task_manifest_sha = _sha(manifest)
        _write_json(
            self.evidence / "task_package_snapshot_receipt.json",
            {
                "schema": "sparse_high_task_progress_canary_v1/task_snapshot/v1",
                "status": "PASS",
                "failures": [],
                "manifest_sha256": self.task_manifest_sha,
                "regular_file_count": len(lines),
                "symlink_count": 0,
                "cache_or_bytecode_count": 0,
            },
        )

    def _build_runtime_roots(self) -> None:
        self.isaac_roots: dict[str, Path] = {}
        for package in ("isaaclab", "isaaclab_tasks", "isaaclab_assets", "isaaclab_rl"):
            root = self.base / f"{package}_distribution"
            root.mkdir()
            self.isaac_roots[package] = root
        self.rsl_root = self.base / "rsl_rl_distribution"
        self.rsl_root.mkdir()

    def _runtime_environment(self) -> dict[str, str]:
        snapshot = (self.run / "source/runtime_project/source/isaac_go2_low_level_rl").resolve()
        values = {
            "SOURCE": str(ROOT.resolve()),
            "BOUND_SOURCE_MANIFEST_SHA256": self.SOURCE_MANIFEST,
            "TASK_PACKAGE_SNAPSHOT": str(snapshot),
            "TASK_PACKAGE_MANIFEST_SHA256": self.task_manifest_sha,
            "V56_RSL_RL_DISTRIBUTION_ROOT": str(self.rsl_root.resolve()),
            "V56_SUBMITTER_FULL_STATIC_SUITE": "PASS",
            "V56_SUBMITTER_RSL_RL_SMOKE": "PASS",
            "RSL_RL_NO_SIM_RECURRENT_LIFECYCLE_SMOKE": "PASS",
        }
        for key, value in checker.EXPECTED_PARENT_V55.items():
            values[f"PARENT_V55_{key.upper()}"] = str(value)
        for package, root in self.isaac_roots.items():
            values[f"V56_{package.upper()}_DISTRIBUTION_ROOT"] = str(root.resolve())
        return values

    def _write_environment(self) -> None:
        payload = "".join(f"{key}={value}\n" for key, value in self.environment.items())
        (self.evidence / "runtime_environment.txt").write_text(payload, encoding="utf-8")

    @staticmethod
    def _source_receipts() -> dict[str, object]:
        roots = {
            "v14": ROOT / "vendor/v14",
            "current": ROOT / "vendor/current_qualified",
        }
        value: dict[str, object] = {
            "roots": {group: str(path.resolve()) for group, path in roots.items()}
        }
        for group, expected in checker.SOURCE_RECEIPT_HASHES.items():
            receipts: dict[str, object] = {}
            for relative, digest in expected.items():
                path = roots[group] / relative
                receipts[relative] = {"sha256": digest, "bytes": path.stat().st_size}
            value[group] = receipts
        return value

    def _runtime_identity(self, seed: int) -> dict[str, object]:
        snapshot = str((self.run / "source/runtime_project/source/isaac_go2_low_level_rl").resolve())
        return {
            "num_envs": 128,
            "reset_seed": seed,
            "sim_dt_seconds": 0.005,
            "decimation": 4,
            "high_policy_hz": 10,
            "low_policy_hz": 50,
            "low_steps_per_high_action": 5,
            "finite_horizon": True,
            "time_outs_bootstrap_key_emitted": False,
            "native_auto_terminations_disabled": ["actual_joint_hard_limit", "base_contact", "time_out"],
            "current_low_checkpoint": {
                "basename": "model_final.pt",
                "bytes": checker.LOW_CHECKPOINT_BYTES,
                "sha256": checker.EXPECTED_LOW_CHECKPOINT_SHA256,
                "runtime_module_state_sha256": checker.EXPECTED_LOW_MODULE_SHA256,
            },
            "parent_v55_identity": copy.deepcopy(checker.EXPECTED_PARENT_V55),
            "task_package_import_origin": {
                "status": "PASS",
                "distribution_root": snapshot,
                "loaded_module_count": 3,
                "module_origins_sha256": "1" * 64,
            },
            "isaac_import_origins": {
                package: {
                    "status": "PASS",
                    "distribution_root": str(root.resolve()),
                    "loaded_module_count": 1,
                    "module_origins_sha256": "2" * 64,
                }
                for package, root in self.isaac_roots.items()
            },
            "rsl_rl_import_origin": {
                "status": "PASS",
                "version": "3.1.2",
                "distribution_root": str(self.rsl_root.resolve()),
                "loaded_module_count": 1,
                "module_origins_sha256": "3" * 64,
            },
        }

    @staticmethod
    def _environment_summary(*, evaluation: bool, metrics: dict[str, object] | None = None) -> dict[str, object]:
        high_steps = 200 if evaluation else checker.TOTAL_ROLLOUT_STEPS
        return {
            "status": "CURRENT_TASK_PROGRESS_HIGH_ENV_COMPLETE",
            "observation_dim": 70,
            "action_dim": 3,
            "high_policy_rate_hz": 10,
            "low_policy_rate_hz": 50,
            "low_steps_per_high_action": 5,
            "finite_horizon": True,
            "time_outs_bootstrap_key_emitted": False,
            "terminal_semantics": "V53_STRICT_UNSAFE_SAME_TICK_REJECTS_GOAL_PRE_RESET_SNAPSHOT",
            "actual_joint_hard_limit_is_terminal": True,
            "native_safe_guard_correction_is_terminal": False,
            "curriculum_registered": False,
            "teacher_registered": False,
            "runtime_router_registered": False,
            "scripted_phase_transition": False,
            "kick_registered": False,
            "native_base_reward_may_be_registered": True,
            "native_base_reward_discarded": True,
            "native_base_reward_consumed_by_high_or_ppo": False,
            "high_ppo_reward_kind": "TASK_PROGRESS_PLUS_STRICT_TERMINAL",
            "strict_terminal_reward_values": [-1.0, 0.0, 1.0],
            "dense_return_authorizes_behavior": False,
            "contact_is_terminal": False,
            "high_vector_steps": high_steps,
            "low_vector_steps": 5 * high_steps,
            "contact_positive_low_ticks": 0,
            "contact_negative_low_ticks": 5 * high_steps * 128,
            "outcome_counts": ({"timeout": 288} if metrics is None else metrics["outcome_counts"])
            if evaluation else {"timeout": 128},
            "bucket_outcome_counts": {},
            "reward_term_means": {"progress": 0.0, "strict_goal": 0.0, "total": 0.0},
            "mean_abs_applied_command": [0.0, 0.0, 0.0],
            "native_safe_guard_changed_elements": 0,
            "native_safe_guard_max_correction_rad": 0.0,
            "evaluation_cases_completed": 288 if evaluation else 0,
        }

    @staticmethod
    def _fixed_timeout_rows() -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for case in checker.fixed_reset_cases():
            geometry = checker.scalar_task_geometry(case.robot_xy, case.ball_xy, case.goal_side)
            rows.append({
                "schema": checker.EPISODE_ROW_SCHEMA,
                "evaluation_assignment": case.assignment,
                "counts_for_evaluation": True,
                "hard_goal_facing_ball_behind": case.hard,
                "bucket": case.bucket,
                "goal_side": case.goal_side,
                "sample_serial": case.assignment + 1,
                "item_seed": checker.FIXED_EVALUATION_SEED * 1_000_003 + case.assignment,
                "reset_robot_xy_m": list(case.robot_xy),
                "reset_robot_yaw_rad": case.robot_yaw,
                "reset_ball_xy_m": list(case.ball_xy),
                "outcome": "timeout",
                "goal": False,
                "own_goal": False,
                "boundary": False,
                "fall": False,
                "joint_hard_limit": False,
                "timeout": True,
                "unsafe_pre_goal": False,
                "requested_goal_line_reached": False,
                "contact_seen": False,
                "first_true_contact_native_tick": None,
                "maximum_contact_force_n": 0.0,
                "terminal_contact_force_n": 0.0,
                "terminal_base_contact_force_n": 0.0,
                "terminal_joint_hard_limit_excess_rad": 0.0,
                "native_safe_guard_changed_elements": 0,
                "native_safe_guard_max_correction_rad": 0.0,
                "episode_high_ticks": 200,
                "episode_native_ticks": 1000,
                "terminal_robot_xy_m": list(case.robot_xy),
                "terminal_ball_xy_m": list(case.ball_xy),
                "initial_l_m": geometry.path_length_m,
                "final_l_m": geometry.path_length_m,
                "cumulative_progress_reward": 0.0,
                "requested_goal_axis_ball_displacement_m": 0.0,
                "abs_lateral_ball_drift_m": 0.0,
            })
        return rows

    def _build_evaluation(self, phase: str) -> dict[str, object]:
        directory = self.initial_dir if phase == "initial" else self.final_dir
        rows = self._fixed_timeout_rows()
        rows_path = directory / f"{phase}_episode_results.jsonl"
        _write_jsonl(rows_path, rows)
        metrics = checker.validate_episode_rows(rows)
        del rows

        value = CORE_FIXTURE._evaluation(phase)
        for key, metric in metrics.items():
            value[key] = metric
        value.update({
            "episode_rows_sha256": _sha(rows_path),
            "runner_identity": copy.deepcopy(checker.RUNNER_IDENTITY),
            "runtime_identity": self._runtime_identity(checker.FIXED_EVALUATION_SEED),
            "source_receipts": copy.deepcopy(self.source_receipts),
            "action_std": {
                "shape": [3],
                "values": [1.0, 1.0, 1.0] if phase == "initial" else [0.9, 0.8, 0.7],
                "all_finite_positive": True,
                "unit_initial_required": phase == "initial",
                "unit_initial_pass": phase == "initial",
            },
            "output_heads": _heads(),
            "environment": self._environment_summary(evaluation=True, metrics=metrics),
            "total_reward_min": 0.0,
            "total_reward_max": 0.0,
        })
        if phase == "initial":
            checkpoint = directory / "model_initial.pt"
            checkpoint.write_bytes(b"synthetic-initial-checkpoint\n")
            value["policy_state_sha256"] = self.INITIAL_STATE
            value["initial_checkpoint"] = {
                "relative_path": checkpoint.name,
                "bytes": checkpoint.stat().st_size,
                "sha256": _sha(checkpoint),
                "policy_state_sha256": self.INITIAL_STATE,
            }
            (directory / "INITIAL_EVALUATION_FINAL_GUARD").write_text("PASS\n", encoding="ascii")
        else:
            assert self.training is not None
            final_receipt = self.training["final_checkpoint"]
            value.update({
                "policy_state_sha256": self.FINAL_STATE,
                "loaded_policy_state_sha256": self.FINAL_STATE,
                "checkpoint_sha256": final_receipt["sha256"],
            })
            (directory / "FINAL_EVALUATION_FINAL_GUARD").write_text("PASS\n", encoding="ascii")
        _write_json(directory / f"{phase}_evaluation_summary.json", value)
        return value

    @staticmethod
    def _install_nonzero_action_trace(rows: list[dict[str, object]]) -> None:
        previous = {env_id: [0.0, 0.0, 0.0] for env_id in range(128)}
        for row in rows:
            env_id = int(row["env_id"])
            raw = [10.0, 0.0, 0.0] if env_id == 0 and int(row["high_step_index"]) < 4 else [0.0, 0.0, 0.0]
            applied = checker.reconstruct_applied_command(raw, previous[env_id])
            row["raw_action"] = raw
            row["applied_command"] = applied
            previous[env_id] = [0.0, 0.0, 0.0] if row["done"] is True else applied

    def _build_support(self) -> dict[str, object]:
        rows = CORE_FIXTURE._raw_receipt(straddle_side=None if self.support_pass else -1)
        self._install_nonzero_action_trace(rows)
        rows_path = self.training_dir / "first_rollout_transitions.jsonl"
        _write_jsonl(rows_path, rows)
        support = checker.validate_first_rollout_transitions(rows)
        del rows
        support.update({
            "transition_rows_relative_path": rows_path.name,
            "transition_rows_bytes": rows_path.stat().st_size,
            "transition_rows_sha256": _sha(rows_path),
            "optimizer_constructed_before_gate": True,
            "optimizer_state_empty_before_gate": True,
            "optimizer_steps_before_gate": 0,
            "initial_policy_state_sha256": self.INITIAL_STATE,
            "initial_checkpoint_sha256": self.initial["initial_checkpoint"]["sha256"],
            "frozen_low_state_sha256_before_gate": checker.EXPECTED_LOW_MODULE_SHA256,
            "dense_return_authorizes_behavior": False,
            "long_training_authorized": False,
        })
        signal_names = (
            "global_nonzero_support",
            "global_positive_support",
            "global_negative_support",
            "positive_support_goal_side_negative",
            "positive_support_goal_side_positive",
        )
        support["gate_pass"] = all(support["checks"][key] for key in signal_names)
        if not self.support_pass:
            support["rejection_integrity"] = {
                "storage_cleared_without_update": True,
                "policy_state_unchanged_without_update": True,
                "policy_state_sha256_after_rejection": self.INITIAL_STATE,
                "low_state_unchanged_without_update": True,
                "low_state_sha256_after_rejection": checker.EXPECTED_LOW_MODULE_SHA256,
                "optimizer_state_empty_after_rejection": True,
                "optimizer_steps_after_rejection": 0,
                "learned_checkpoint_count": 0,
            }
        checks = checker.reward_support_checks(support)
        support["checks"] = checks
        support["blockers"] = sorted(key for key in signal_names if checks.get(key) is not True)
        if not self.support_pass:
            support["independent_integrity_checks"] = copy.deepcopy(checks)
        _write_json(self.training_dir / "first_rollout_reward_support_summary.json", support)
        (self.training_dir / "FIRST_ROLLOUT_REWARD_SUPPORT_FINAL_GUARD").write_text("PASS\n", encoding="ascii")
        if not self.support_pass:
            (self.training_dir / "REWARD_SUPPORT_REJECTED_FINAL_GUARD").write_text("PASS\n", encoding="ascii")
        return support

    def _build_training(self) -> tuple[dict[str, object], dict[str, object]]:
        first_checkpoint = self.training_dir / "model_first_update.pt"
        final_checkpoint = self.training_dir / "model_final.pt"
        first_checkpoint.write_bytes(b"synthetic-first-update-checkpoint\n")
        final_checkpoint.write_bytes(b"synthetic-final-checkpoint\n")

        first = CORE_FIXTURE._first_update()
        first.update({
            "gate_pass": True,
            "normalization_update_calls": checker.FIRST_ROLLOUT_STEPS,
            "actor_update": _component_update("actor.weight"),
            "critic_update": _component_update("critic.weight"),
            "initial_policy_state_sha256": self.INITIAL_STATE,
            "initial_checkpoint_sha256": self.initial["initial_checkpoint"]["sha256"],
            "reward_support_rows_sha256": self.support["transition_rows_sha256"],
            "first_update_checkpoint": {
                "relative_path": first_checkpoint.name,
                "bytes": first_checkpoint.stat().st_size,
                "sha256": _sha(first_checkpoint),
                "policy_state_sha256": self.FIRST_STATE,
            },
            "runner_identity": copy.deepcopy(checker.RUNNER_IDENTITY),
            "runtime_identity": self._runtime_identity(checker.TRAINING_SEED),
            "source_receipts": copy.deepcopy(self.source_receipts),
        })
        first["checks"] = checker.first_update_checks(first)
        _write_json(self.training_dir / "first_update_summary.json", first)
        (self.training_dir / "FIRST_UPDATE_FINAL_GUARD").write_text("PASS\n", encoding="ascii")

        training = CORE_FIXTURE._training()
        training.update({
            "normalization_update_calls": checker.TOTAL_ROLLOUT_STEPS,
            "actor_update": _component_update("actor.weight"),
            "critic_update": _component_update("critic.weight"),
            "initial_policy_state_sha256": self.INITIAL_STATE,
            "final_policy_state_sha256": self.FINAL_STATE,
            "initial_checkpoint_sha256": self.initial["initial_checkpoint"]["sha256"],
            "reward_support_rows_sha256": self.support["transition_rows_sha256"],
            "first_update_checkpoint": copy.deepcopy(first["first_update_checkpoint"]),
            "final_checkpoint": {
                "relative_path": final_checkpoint.name,
                "bytes": final_checkpoint.stat().st_size,
                "sha256": _sha(final_checkpoint),
                "policy_state_sha256": self.FINAL_STATE,
            },
            "initial_action_std": {
                "shape": [3], "values": [1.0, 1.0, 1.0],
                "all_finite_positive": True, "unit_initial_required": True,
                "unit_initial_pass": True,
            },
            "final_action_std": {
                "shape": [3], "values": [0.9, 0.8, 0.7],
                "all_finite_positive": True, "unit_initial_required": False,
                "unit_initial_pass": False,
            },
            "initial_output_heads": _heads(),
            "final_output_heads": _heads(),
            "full_distribution_from_iteration_zero": True,
            "online_curriculum_registered": False,
            "training_initial_reset_count": 128,
            "training_initial_resets": checker.expected_training_initial_resets(),
            "normalization_lifecycle": {
                "installed": True,
                "original_method_present": True,
                "normalization_enabled": False,
                "semantics": "COUNTED_DISABLED_NORMALIZATION_LIFECYCLE",
            },
            "runner_identity": copy.deepcopy(checker.RUNNER_IDENTITY),
            "runtime_identity": self._runtime_identity(checker.TRAINING_SEED),
            "source_receipts": copy.deepcopy(self.source_receipts),
            "environment": self._environment_summary(evaluation=False),
        })
        _write_json(self.training_dir / "training_summary.json", training)
        _write_json(
            self.training_dir / "iteration_progress.json",
            {
                "status": "COMPLETE",
                "completed_iterations": checker.TRAINING_ITERATIONS,
                "rollout_steps": checker.TOTAL_ROLLOUT_STEPS,
                "optimizer_steps": checker.TOTAL_OPTIMIZER_STEPS,
            },
        )
        (self.training_dir / "TRAINING_FINAL_GUARD").write_text("PASS\n", encoding="ascii")
        return first, training

    def _write_result(self) -> None:
        contract = checker.load_contract(ROOT / "SPARSE_HIGH_TASK_PROGRESS_CANARY_V1_CONTRACT.json")
        result = checker.build_result(
            contract,
            checker.ResultInputs(self.initial, self.support, self.first, self.training, self.final),
        )
        if result["status"] != "COMPLETE":
            raise AssertionError(f"synthetic result is not complete: {result['failures']}")
        summary_paths = {
            "initial_evaluation": self.initial_dir / "initial_evaluation_summary.json",
            "reward_support": self.training_dir / "first_rollout_reward_support_summary.json",
        }
        if self.support_pass:
            summary_paths.update({
                "first_update": self.training_dir / "first_update_summary.json",
                "training": self.training_dir / "training_summary.json",
                "final_evaluation": self.final_dir / "final_evaluation_summary.json",
            })
        result["input_summary_sha256"] = {key: _sha(path) for key, path in summary_paths.items()}
        self.result = result
        _write_json(self.result_dir / "task_progress_canary_result.json", result)
        (self.result_dir / "V56_RESULT_FINAL_GUARD").write_text("PASS\n", encoding="ascii")

    def _phase_text(self, *, in_job: bool) -> str:
        events = [
            ("wrapper_static_preflight", "START", ""),
            ("wrapper_static_preflight", "COMPLETE", "0"),
            ("task_package_snapshot", "START", ""),
            ("task_package_snapshot", "COMPLETE", "0"),
            ("contact_reset_lifecycle_probe", "START", ""),
            ("contact_reset_lifecycle_probe", "COMPLETE", "0"),
            ("initial_fixed_evaluation", "START", ""),
            ("initial_fixed_evaluation", "COMPLETE", "0"),
            ("bounded_task_progress_ppo", "START", ""),
            ("bounded_task_progress_ppo", "COMPLETE" if self.support_pass else "REJECT", "0"),
        ]
        if self.support_pass:
            events.extend([
                ("final_fixed_evaluation", "START", ""),
                ("final_fixed_evaluation", "COMPLETE", "0"),
            ])
        events.extend([
            ("terminal_aggregate", "START", ""),
            ("terminal_aggregate", "COMPLETE", "0"),
            ("independent_result_check", "START", ""),
        ])
        if not in_job:
            events.append(("independent_result_check", "COMPLETE", "0"))
        return "timestamp\tphase\tstate\trc\n" + "".join(
            f"2026-09-18T00:00:00+00:00\t{phase}\t{state}\t{rc}\n"
            for phase, state, rc in events
        )

    def _write_transaction_files(self, *, in_job: bool) -> None:
        (self.evidence / "contact_reset_probe_child_rc.txt").write_text("0\n", encoding="ascii")
        (self.evidence / "initial_child_rc.txt").write_text("0\n", encoding="ascii")
        (self.evidence / "training_child_rc.txt").write_text("0\n", encoding="ascii")
        if self.support_pass:
            (self.evidence / "final_child_rc.txt").write_text("0\n", encoding="ascii")
        (self.evidence / "process_rc.txt").write_text("0\n", encoding="ascii")
        (self.evidence / "SPARSE_HIGH_TASK_PROGRESS_CANARY_TRANSACTION_FINAL_GUARD").write_text("PASS\n", encoding="ascii")
        _write_json(
            self.evidence / "transaction_status.json",
            {
                "schema": "sparse_high_task_progress_canary_v1/transaction_status/v1",
                "status": "COMPLETE",
                "process_rc": 0,
                "failed_phase": None,
                "failure": None,
                "parent_v55_run_label": checker.EXPECTED_PARENT_V55["run_label"],
                "decision": self.result["decision"],
                "initial_evaluation_executed": True,
                "reward_support_executed": True,
                "reward_support_gate_pass": self.support_pass,
                "first_update_executed": self.support_pass,
                "training_executed": self.support_pass,
                "final_evaluation_executed": self.support_pass,
                "ppo_optimizer_steps": checker.TOTAL_OPTIMIZER_STEPS if self.support_pass else 0,
                "long_training_authorized": False,
            },
        )
        (self.evidence / "phase_events.tsv").write_text(self._phase_text(in_job=in_job), encoding="utf-8")

    def set_phase_mode(self, *, in_job: bool) -> None:
        (self.evidence / "phase_events.tsv").write_text(self._phase_text(in_job=in_job), encoding="utf-8")

    def _make_snapshot_read_only(self) -> None:
        root = self.run / "source/runtime_project/source/isaac_go2_low_level_rl"
        for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
        root.chmod(0o555)

    def _backup_mutable_files(self) -> None:
        self.backups = self.base / "backups"
        self.backups.mkdir()
        for path in (
            self.training_dir / "first_rollout_transitions.jsonl",
            self.training_dir / "first_update_summary.json",
            self.training_dir / "training_summary.json",
            self.result_dir / "task_progress_canary_result.json",
            self.evidence / "phase_events.tsv",
            self.evidence / "process_rc.txt",
        ):
            if path.exists():
                shutil.copyfile(path, self.backups / path.name)

    def restore(self) -> None:
        for backup in self.backups.iterdir():
            destinations = {
                "first_rollout_transitions.jsonl": self.training_dir / backup.name,
                "first_update_summary.json": self.training_dir / backup.name,
                "training_summary.json": self.training_dir / backup.name,
                "task_progress_canary_result.json": self.result_dir / backup.name,
                "phase_events.tsv": self.evidence / backup.name,
                "process_rc.txt": self.evidence / backup.name,
            }
            shutil.copyfile(backup, destinations[backup.name])
        forbidden = self.training_dir / "model_first_update.pt"
        if not self.support_pass and forbidden.exists():
            forbidden.unlink()

    def run_main(self, *, in_job: bool) -> tuple[int, str]:
        arguments = [str(ROOT / "check_sparse_high_task_progress_canary.py"), str(self.run)]
        if in_job:
            arguments.append("--in-job")
        output = io.StringIO()
        errors = io.StringIO()
        with (
            mock.patch.object(checker, "PARENT_TASK_MANIFEST_SHA256", self.task_manifest_sha),
            mock.patch.object(checker, "validate_source_binding", return_value=self.SOURCE_MANIFEST),
            mock.patch.object(checker, "validate_contact_reset_probe"),
            mock.patch.object(sys, "argv", arguments),
            redirect_stdout(output),
            redirect_stderr(errors),
        ):
            return_code = checker.main()
        return return_code, output.getvalue() + errors.getvalue()

    def mutate_transition(self, index: int, mutate) -> None:
        path = self.training_dir / "first_rollout_transitions.jsonl"
        replacement = self.base / "mutated-transitions.jsonl"
        with path.open("rb") as source, replacement.open("wb") as target:
            for row_index, line in enumerate(source):
                if row_index == index:
                    row = json.loads(line)
                    mutate(row)
                    line = checker.canonical_json_bytes(row)
                target.write(line)
        os.replace(replacement, path)


class CheckerTransactionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reject = SyntheticTransaction(support_pass=False)
        cls.passing = SyntheticTransaction(support_pass=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.reject.cleanup()
        cls.passing.cleanup()

    def tearDown(self) -> None:
        self.reject.restore()
        self.passing.restore()

    def test_main_accepts_complete_reject_transaction_in_job(self) -> None:
        self.reject.set_phase_mode(in_job=True)
        return_code, output = self.reject.run_main(in_job=True)
        self.assertEqual(return_code, 0, output)
        self.assertIn("V56_CHECK=PASS", output)
        self.assertIn("REWARD_SUPPORT_GATE_PASS=NO", output)
        self.assertIn("TOTAL_PPO_OPTIMIZER_STEPS=0", output)

    def test_main_accepts_complete_pass_transaction_external(self) -> None:
        self.passing.set_phase_mode(in_job=False)
        return_code, output = self.passing.run_main(in_job=False)
        self.assertEqual(return_code, 0, output)
        self.assertIn("V56_CHECK=PASS", output)
        self.assertIn("REWARD_SUPPORT_GATE_PASS=YES", output)
        self.assertIn("TOTAL_PPO_OPTIMIZER_STEPS=240", output)

    def test_raw_reward_serial_and_terminal_mutations_fail_closed(self) -> None:
        cases = (
            ("reward", 0, lambda row: row.__setitem__("progress_reward", float(row["progress_reward"]) + 0.01)),
            ("serial", 0, lambda row: row.__setitem__("sample_serial", int(row["sample_serial"]) + 1)),
            ("terminal", 25_472, lambda row: row.__setitem__("timeout", False)),
        )
        for label, index, mutate in cases:
            with self.subTest(label=label):
                self.passing.mutate_transition(index, mutate)
                with self.assertRaises(checker.ProtocolError):
                    checker.validate_transition_file(
                        self.passing.training_dir / "first_rollout_transitions.jsonl",
                        self.passing.support,
                    )
                self.passing.restore()

    def test_action_hull_and_rate_mutations_fail_closed(self) -> None:
        # Slot zero reaches the +vx hull vertex after four rate-limited steps.
        # A tiny +vy displacement remains inside the gross replay tolerance and
        # the rate box, but leaves the qualified convex hull.
        def outside_hull(row: dict[str, object]) -> None:
            command = list(row["applied_command"])
            command[1] += 1.0e-4
            row["applied_command"] = command

        # The first command is exactly at the +vx 0.1-second rate limit.  This
        # perturbation is hull-feasible and replay-close but rate-infeasible.
        def outside_rate(row: dict[str, object]) -> None:
            command = list(row["applied_command"])
            command[0] += 1.0e-4
            row["applied_command"] = command

        for label, index, mutate, expected in (
            ("hull", 384, outside_hull, "convex hull"),
            ("rate", 0, outside_rate, "rate limit"),
        ):
            with self.subTest(label=label):
                self.passing.mutate_transition(index, mutate)
                with self.assertRaisesRegex(checker.ProtocolError, expected):
                    checker.validate_transition_file(
                        self.passing.training_dir / "first_rollout_transitions.jsonl",
                        self.passing.support,
                    )
                self.passing.restore()

    def test_result_input_hash_and_decision_mutations_fail_closed(self) -> None:
        path = self.passing.result_dir / "task_progress_canary_result.json"
        for label, mutate in (
            (
                "input hash",
                lambda value: value["input_summary_sha256"].__setitem__("initial_evaluation", "0" * 64),
            ),
            (
                "decision",
                lambda value: value.__setitem__("decision", checker.DECISION_HALF_GATE),
            ),
        ):
            with self.subTest(label=label):
                value = json.loads(path.read_bytes())
                mutate(value)
                contract = checker.load_contract(ROOT / "SPARSE_HIGH_TASK_PROGRESS_CANARY_V1_CONTRACT.json")
                hashes = {
                    "initial_evaluation": _sha(self.passing.initial_dir / "initial_evaluation_summary.json"),
                    "reward_support": _sha(self.passing.training_dir / "first_rollout_reward_support_summary.json"),
                    "first_update": _sha(self.passing.training_dir / "first_update_summary.json"),
                    "training": _sha(self.passing.training_dir / "training_summary.json"),
                    "final_evaluation": _sha(self.passing.final_dir / "final_evaluation_summary.json"),
                }
                with self.assertRaisesRegex(checker.ProtocolError, "terminal result differs"):
                    checker.validate_result_exact(
                        value, contract, self.passing.initial, self.passing.support,
                        self.passing.first, self.passing.training, self.passing.final, hashes,
                    )
                self.passing.restore()

    def test_phase_and_process_rc_mutations_fail_closed(self) -> None:
        self.reject.set_phase_mode(in_job=True)
        checker.validate_phase_events(
            self.reject.evidence / "phase_events.tsv", support_pass=False, in_job=True
        )
        self.reject.set_phase_mode(in_job=False)
        checker.validate_phase_events(
            self.reject.evidence / "phase_events.tsv", support_pass=False, in_job=False
        )
        with self.assertRaisesRegex(checker.ProtocolError, "phase event sequence"):
            checker.validate_phase_events(
                self.reject.evidence / "phase_events.tsv", support_pass=False, in_job=True
            )
        self.reject.restore()

        (self.reject.evidence / "process_rc.txt").write_text("1\n", encoding="ascii")
        with self.assertRaisesRegex(checker.ProtocolError, "process RC"):
            checker.validate_transaction(self.reject.run, self.reject.result, support_pass=False)

    def test_reject_branch_forbids_downstream_artifacts(self) -> None:
        forbidden = self.reject.training_dir / "model_first_update.pt"
        forbidden.write_bytes(b"forbidden\n")
        with self.assertRaisesRegex(checker.ProtocolError, "forbidden artifact"):
            checker.require_absent([forbidden], "reward support reject branch")

    def test_actual_checkpoint_policy_state_must_change(self) -> None:
        first_path = self.passing.training_dir / "first_update_summary.json"
        training_path = self.passing.training_dir / "training_summary.json"
        first = json.loads(first_path.read_bytes())
        training = json.loads(training_path.read_bytes())
        first["first_update_checkpoint"]["policy_state_sha256"] = self.passing.INITIAL_STATE
        training["first_update_checkpoint"]["policy_state_sha256"] = self.passing.INITIAL_STATE
        _write_json(first_path, first)
        _write_json(training_path, training)
        return_code, output = self.passing.run_main(in_job=True)
        self.assertEqual(return_code, 2, output)
        self.assertIn("first PPO update did not change actual checkpoint state", output)


if __name__ == "__main__":
    unittest.main()
