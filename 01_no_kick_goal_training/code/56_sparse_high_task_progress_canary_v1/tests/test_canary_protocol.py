from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canary_protocol import (
    CONTRACT_SCHEMA,
    DECISION_INVALID,
    DECISION_NO_BREAKTHROUGH,
    DECISION_REWARD_SUPPORT_REJECTED,
    EPISODE_ROW_SCHEMA,
    EVALUATION_SCHEMA,
    EXPECTED_PARENT_V55,
    EXPECTED_ASSIGNMENT_SHA256,
    EXPECTED_FIXED_CASE_TABLE_SHA256,
    EXPECTED_LOW_MODULE_SHA256,
    FIRST_UPDATE_SCHEMA,
    PENDING_DECISION,
    PENDING_STATUS,
    ProtocolError,
    RELEASE_DECISION,
    RELEASE_STATUS,
    RESULT_SCHEMA,
    ResultInputs,
    SUPPORT_EPSILON,
    TRAINING_SCHEMA,
    TRANSITION_ROW_SCHEMA,
    build_result,
    expected_training_initial_resets,
    load_contract,
    validate_evaluation_summary,
    validate_first_rollout_transitions,
)
from task_progress_reward import GOAL_CENTER_X_M, PATH_NORMALIZER_M, scalar_progress


CONTRACT = ROOT / "SPARSE_HIGH_TASK_PROGRESS_CANARY_V1_CONTRACT.json"
CLOSURE = ROOT / "EVIDENCE_CLOSURE.json"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _leaf_paths(value: object, prefix: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    if isinstance(value, dict):
        return [path for key, item in value.items() for path in _leaf_paths(item, (*prefix, key))]
    if isinstance(value, list):
        return [path for index, item in enumerate(value) for path in _leaf_paths(item, (*prefix, index))]
    return [prefix]


def _mutate_leaf(value: object) -> object:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.123456789
    if isinstance(value, str):
        return value + "_DRIFT"
    if value is None:
        return "DRIFT"
    raise AssertionError(f"unsupported contract leaf: {type(value).__name__}")


def _replace_at_path(value: object, path: tuple[object, ...]) -> None:
    cursor = value
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    key = path[-1]
    cursor[key] = _mutate_leaf(cursor[key])  # type: ignore[index]


def _released_contract() -> dict:
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    value["status"] = RELEASE_STATUS
    value["decision"] = RELEASE_DECISION
    value["parent_v55"].update(EXPECTED_PARENT_V55)
    return value


def _raw_receipt(*, straddle_side: int | None = None) -> list[dict]:
    resets = expected_training_initial_resets()
    state_robot = [list(row["robot_xy"]) for row in resets]
    state_ball = [list(row["ball_xy"]) for row in resets]
    positive_env_by_side = {
        side: next(index for index, row in enumerate(resets) if row["goal_side"] == side)
        for side in (-1, 1)
    }
    negative_env = next(index for index in range(128) if index not in positive_env_by_side.values())
    rows: list[dict] = []
    for step in range(200):
        for env_id, reset in enumerate(resets):
            pre_robot = list(state_robot[env_id])
            pre_ball = list(state_ball[env_id])
            post_robot = list(pre_robot)
            post_ball = list(pre_ball)
            if step == 0 and (env_id in positive_env_by_side.values() or env_id == negative_env):
                pre_geometry, _, _ = scalar_progress(
                    pre_robot, pre_ball, pre_robot, pre_ball, reset["goal_side"]
                )
                dx = pre_geometry.staging_target_xy[0] - pre_robot[0]
                dy = pre_geometry.staging_target_xy[1] - pre_robot[1]
                norm = math.hypot(dx, dy)
                if norm == 0.0:
                    dx, dy, norm = 1.0, 0.0, 1.0
                sign = -1.0 if env_id == negative_env else 1.0
                distance = 0.02
                if (
                    straddle_side == reset["goal_side"]
                    and env_id == positive_env_by_side[reset["goal_side"]]
                ):
                    distance = 0.95 * SUPPORT_EPSILON * PATH_NORMALIZER_M
                post_robot = [
                    pre_robot[0] + sign * distance * dx / norm,
                    pre_robot[1] + sign * distance * dy / norm,
                ]
            state_robot[env_id] = list(post_robot)
            state_ball[env_id] = list(post_ball)
            pre_geometry, post_geometry, progress = scalar_progress(
                pre_robot, pre_ball, post_robot, post_ball, reset["goal_side"]
            )
            recorded_progress = progress
            if (
                straddle_side == reset["goal_side"]
                and step == 0
                and env_id == positive_env_by_side[reset["goal_side"]]
            ):
                if not progress < SUPPORT_EPSILON:
                    raise AssertionError("straddle fixture reconstruction did not stay below epsilon")
                recorded_progress = SUPPORT_EPSILON + 0.5e-6
            timeout = step == 199
            rows.append(
                {
                    "schema": TRANSITION_ROW_SCHEMA,
                    "global_transition_index": step * 128 + env_id,
                    "rollout_iteration": 0,
                    "high_step_index": step,
                    "env_id": env_id,
                    "sample_serial": env_id + 1,
                    "episode_serial_before_step": env_id + 1,
                    "item_seed": reset["item_seed"],
                    "episode_high_step_index_before_step": step,
                    "goal_side": reset["goal_side"],
                    "reset_robot_xy_m": list(reset["robot_xy"]),
                    "reset_robot_yaw_rad": reset["robot_yaw"],
                    "reset_ball_xy_m": list(reset["ball_xy"]),
                    "pre_robot_xy_m": pre_robot,
                    "pre_ball_xy_m": pre_ball,
                    "post_robot_xy_pre_reset_m": post_robot,
                    "post_ball_xy_pre_reset_m": post_ball,
                    "goal_xy_m": list(pre_geometry.goal_xy),
                    "raw_staging_target_pre_xy_m": list(pre_geometry.raw_staging_target_xy),
                    "raw_staging_target_post_xy_m": list(post_geometry.raw_staging_target_xy),
                    "staging_target_pre_xy_m": list(pre_geometry.staging_target_xy),
                    "staging_target_post_xy_m": list(post_geometry.staging_target_xy),
                    "staging_projection_pre_applied": pre_geometry.staging_projection_applied,
                    "staging_projection_post_applied": post_geometry.staging_projection_applied,
                    "l_pre_m": pre_geometry.path_length_m,
                    "l_post_m": post_geometry.path_length_m,
                    "normalizer_m": PATH_NORMALIZER_M,
                    "progress_reward": recorded_progress,
                    "strict_goal_reward": 0.0,
                    "total_reward": recorded_progress,
                    "raw_action": [0.0, 0.0, 0.0],
                    "applied_command": [0.0, 0.0, 0.0],
                    "native_ticks_executed": 5,
                    "terminal_reason": "TIMEOUT" if timeout else "NONE",
                    "goal": False,
                    "own_goal": False,
                    "boundary": False,
                    "fall": False,
                    "joint_hard_limit": False,
                    "timeout": timeout,
                    "unsafe_same_tick": False,
                    "done": timeout,
                    "reset_after_step": timeout,
                    "contact_seen": False,
                    "maximum_contact_force_n": 0.0,
                    "terminal_contact_force_n": 0.0,
                    "terminal_base_contact_force_n": 0.0,
                    "terminal_joint_hard_limit_excess_rad": 0.0,
                    "post_state_source": "SAME_TICK_PHYSICAL_PRE_RESET",
                }
            )
    return rows


def _early_terminal_receipt(*, straddle_side: int | None = None) -> list[dict]:
    """Build a complete grid with tick-1/tick-4 terminals and exact resets."""

    reset_table = expected_training_initial_resets(count=512)
    current = [copy.deepcopy(reset_table[index]) for index in range(128)]
    state_robot = [list(row["robot_xy"]) for row in current]
    state_ball = [list(row["ball_xy"]) for row in current]
    episode_step = [0] * 128
    next_serial = 129
    positive_env_by_side = {
        side: next(
            index
            for index, row in enumerate(current)
            if index not in (0, 1) and row["goal_side"] == side
        )
        for side in (-1, 1)
    }
    negative_env = next(
        index
        for index in range(128)
        if index not in (0, 1) and index not in positive_env_by_side.values()
    )
    rows: list[dict] = []
    for high_step in range(200):
        for env_id in range(128):
            reset = current[env_id]
            side = int(reset["goal_side"])
            pre_robot = list(state_robot[env_id])
            pre_ball = list(state_ball[env_id])
            post_robot = list(pre_robot)
            post_ball = list(pre_ball)
            native_ticks = 5
            contact_seen = False
            maximum_contact = 0.0
            terminal_contact = 0.0
            boundary = False

            if high_step == 0 and env_id == 0:
                # Contact reaches the requested line on the same native tick
                # as a robot boundary violation. Unsafe has priority: no goal.
                post_robot = [4.1, pre_robot[1]]
                post_ball = [side * GOAL_CENTER_X_M, 0.0]
                native_ticks = 1
                contact_seen = True
                maximum_contact = terminal_contact = 1.25
                boundary = True
            elif high_step == 0 and env_id == 1:
                post_robot = [pre_robot[0], 2.6]
                native_ticks = 4
                boundary = True
            elif high_step == 0 and (
                env_id in positive_env_by_side.values() or env_id == negative_env
            ):
                pre_geometry, _, _ = scalar_progress(
                    pre_robot, pre_ball, pre_robot, pre_ball, side
                )
                dx = pre_geometry.staging_target_xy[0] - pre_robot[0]
                dy = pre_geometry.staging_target_xy[1] - pre_robot[1]
                norm = math.hypot(dx, dy)
                if norm == 0.0:
                    dx, dy, norm = 1.0, 0.0, 1.0
                sign = -1.0 if env_id == negative_env else 1.0
                distance = 0.02
                if straddle_side == side and env_id == positive_env_by_side[side]:
                    distance = 0.95 * SUPPORT_EPSILON * PATH_NORMALIZER_M
                post_robot = [
                    pre_robot[0] + sign * distance * dx / norm,
                    pre_robot[1] + sign * distance * dy / norm,
                ]

            requested_line = (
                contact_seen
                and side * post_ball[0] >= GOAL_CENTER_X_M
                and abs(post_ball[1]) <= 0.49000000953674316
            )
            goal = bool(requested_line and not boundary)
            timeout = bool(not requested_line and not boundary and episode_step[env_id] == 199 and native_ticks == 5)
            done = bool(requested_line or boundary or timeout)
            pre_geometry, post_geometry, progress = scalar_progress(
                pre_robot, pre_ball, post_robot, post_ball, side
            )
            recorded_progress = progress
            if (
                straddle_side == side
                and high_step == 0
                and env_id == positive_env_by_side[side]
            ):
                self_check = progress < SUPPORT_EPSILON
                if not self_check:
                    raise AssertionError("straddle fixture reconstruction did not stay below epsilon")
                recorded_progress = SUPPORT_EPSILON + 0.5e-6
            row = {
                "schema": TRANSITION_ROW_SCHEMA,
                "global_transition_index": high_step * 128 + env_id,
                "rollout_iteration": 0,
                "high_step_index": high_step,
                "env_id": env_id,
                "sample_serial": reset["sample_serial"],
                "episode_serial_before_step": reset["sample_serial"],
                "item_seed": reset["item_seed"],
                "episode_high_step_index_before_step": episode_step[env_id],
                "goal_side": side,
                "reset_robot_xy_m": list(reset["robot_xy"]),
                "reset_robot_yaw_rad": reset["robot_yaw"],
                "reset_ball_xy_m": list(reset["ball_xy"]),
                "pre_robot_xy_m": pre_robot,
                "pre_ball_xy_m": pre_ball,
                "post_robot_xy_pre_reset_m": post_robot,
                "post_ball_xy_pre_reset_m": post_ball,
                "goal_xy_m": list(pre_geometry.goal_xy),
                "raw_staging_target_pre_xy_m": list(pre_geometry.raw_staging_target_xy),
                "raw_staging_target_post_xy_m": list(post_geometry.raw_staging_target_xy),
                "staging_target_pre_xy_m": list(pre_geometry.staging_target_xy),
                "staging_target_post_xy_m": list(post_geometry.staging_target_xy),
                "staging_projection_pre_applied": pre_geometry.staging_projection_applied,
                "staging_projection_post_applied": post_geometry.staging_projection_applied,
                "l_pre_m": pre_geometry.path_length_m,
                "l_post_m": post_geometry.path_length_m,
                "normalizer_m": PATH_NORMALIZER_M,
                "progress_reward": recorded_progress,
                "strict_goal_reward": 0.0,
                "total_reward": recorded_progress,
                "raw_action": [0.0, 0.0, 0.0],
                "applied_command": [0.0, 0.0, 0.0],
                "native_ticks_executed": native_ticks,
                "terminal_reason": "BOUNDARY" if boundary else "TIMEOUT" if timeout else "NONE",
                "goal": goal,
                "own_goal": False,
                "boundary": boundary,
                "fall": False,
                "joint_hard_limit": False,
                "timeout": timeout,
                "unsafe_same_tick": boundary,
                "done": done,
                "reset_after_step": done,
                "contact_seen": contact_seen,
                "maximum_contact_force_n": maximum_contact,
                "terminal_contact_force_n": terminal_contact,
                "terminal_base_contact_force_n": 0.0,
                "terminal_joint_hard_limit_excess_rad": 0.0,
                "post_state_source": "SAME_TICK_PHYSICAL_PRE_RESET",
            }
            rows.append(row)
            if done:
                replacement = copy.deepcopy(reset_table[next_serial - 1])
                self_serial = int(replacement["sample_serial"])
                if self_serial != next_serial:
                    raise AssertionError("reset table serial mismatch")
                next_serial += 1
                current[env_id] = replacement
                state_robot[env_id] = list(replacement["robot_xy"])
                state_ball[env_id] = list(replacement["ball_xy"])
                episode_step[env_id] = 0
            else:
                state_robot[env_id] = list(post_robot)
                state_ball[env_id] = list(post_ball)
                episode_step[env_id] += 1
    return rows


POLICY_SHA = "a" * 64
INITIAL_CHECKPOINT_SHA = "b" * 64
ROWS_SHA = "c" * 64
FIRST_CHECKPOINT_SHA = "d" * 64
FIRST_POLICY_SHA = "e" * 64
FINAL_CHECKPOINT_SHA = "f" * 64
FINAL_POLICY_SHA = "1" * 64


def _evaluation(phase: str) -> dict:
    value = {
        "schema": EVALUATION_SCHEMA,
        "status": "COMPLETE",
        "phase": phase,
        "fixed_evaluation_case_count": 288,
        "hard_case_count": 36,
        "assignments_exactly_once": True,
        "assignment_sha256": EXPECTED_ASSIGNMENT_SHA256,
        "fixed_case_table_sha256": EXPECTED_FIXED_CASE_TABLE_SHA256,
        "episode_rows_sha256": "2" * 64,
        "goal_count": 0,
        "hard_goal_count": 0,
        "unsafe_count": 0,
        "contact_count": 0,
        "case_counts_by_goal_side": {"-1": 132, "1": 156},
        "hard_case_counts_by_goal_side": {"-1": 18, "1": 18},
        "goal_counts_by_goal_side": {"-1": 0, "1": 0},
        "hard_goal_counts_by_goal_side": {"-1": 0, "1": 0},
        "outcome_counts": {
            "goal": 0,
            "own_goal": 0,
            "boundary": 0,
            "fall": 0,
            "joint_hard_limit": 0,
            "timeout": 288,
        },
        "maximum_live_reset_error": 0.0,
        "mean_initial_l_m": 1.0,
        "mean_final_l_m": 1.0,
        "mean_cumulative_progress_reward": 0.0,
        "mean_requested_goal_axis_ball_displacement_m": 0.0,
        "mean_abs_lateral_ball_drift_m": 0.0,
        "ppo_optimizer_steps": 0,
        "policy_state_unchanged": True,
        "frozen_low_state_unchanged": True,
        "low_state_before_sha256": EXPECTED_LOW_MODULE_SHA256,
        "low_state_after_sha256": EXPECTED_LOW_MODULE_SHA256,
        "policy_state_sha256": POLICY_SHA if phase == "initial" else FINAL_POLICY_SHA,
        "finite_horizon": True,
        "time_outs_bootstrap_key_emitted": False,
        "total_reward_min": -0.1,
        "total_reward_max": 0.1,
        "strict_terminal_reward_values": [-1.0, 0.0, 1.0],
        "dense_return_authorizes_behavior": False,
        "long_training_authorized": False,
        "policy_initialization_seed": 83_001,
        "policy_seed_reapplied_after_environment_construction": True,
        "failures": [],
    }
    if phase == "initial":
        value.update({
            "checkpoint_loaded": False,
            "fresh_policy_initialized": True,
            "checkpoint_sha256": None,
            "loaded_policy_state_sha256": None,
            "initial_checkpoint": {
                "relative_path": "model_initial.pt",
                "bytes": 100,
                "sha256": INITIAL_CHECKPOINT_SHA,
                "policy_state_sha256": POLICY_SHA,
            },
        })
    else:
        value.update({
            "checkpoint_loaded": True,
            "fresh_policy_initialized": False,
            "checkpoint_sha256": FINAL_CHECKPOINT_SHA,
            "loaded_policy_state_sha256": FINAL_POLICY_SHA,
            "initial_checkpoint": None,
        })
    return value


def _reward_support(*, passes: bool) -> dict:
    checks = {
        "row_count_exact": True,
        "complete_unique_grid": True,
        "all_finite": True,
        "all_formula_reconstructed": True,
        "global_nonzero_support": True,
        "global_positive_support": True,
        "global_negative_support": passes,
        "positive_support_goal_side_negative": True,
        "positive_support_goal_side_positive": True,
    }
    value = {
        "schema": "sparse_high_task_progress_canary_v1/reward_support/v1",
        "status": "COMPLETE",
        "failures": [],
        "checks": checks,
        "gate_pass": passes,
        "optimizer_constructed_before_gate": True,
        "optimizer_state_empty_before_gate": True,
        "optimizer_steps_before_gate": 0,
        "transition_rows_relative_path": "first_rollout_transitions.jsonl",
        "transition_rows_bytes": 1234,
        "transition_rows_sha256": ROWS_SHA,
        "initial_policy_state_sha256": POLICY_SHA,
        "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA,
        "frozen_low_state_sha256_before_gate": EXPECTED_LOW_MODULE_SHA256,
    }
    if not passes:
        value["rejection_integrity"] = {
            "storage_cleared_without_update": True,
            "policy_state_unchanged_without_update": True,
            "policy_state_sha256_after_rejection": POLICY_SHA,
            "low_state_unchanged_without_update": True,
            "low_state_sha256_after_rejection": EXPECTED_LOW_MODULE_SHA256,
            "optimizer_state_empty_after_rejection": True,
            "optimizer_steps_after_rejection": 0,
            "learned_checkpoint_count": 0,
        }
    return value


def _hidden_receipt() -> dict:
    return {
        "actor_equal_after_restore": True,
        "critic_equal_after_restore": True,
        "actor_unchanged_by_update": True,
        "critic_unchanged_by_update": True,
    }


def _first_update() -> dict:
    return {
        "schema": FIRST_UPDATE_SCHEMA,
        "status": "COMPLETE",
        "failures": [],
        "rollout_steps": 200,
        "rollout_samples": 25_600,
        "ppo_optimizer_steps": 20,
        "actor_update": {"changed": True},
        "critic_update": {"changed": True},
        "returns_complete": True,
        "returns_all_finite": True,
        "losses_all_finite": True,
        "storage_reset_after_update": True,
        "checkpoint_loaded": True,
        "checkpoint_origin": "RUN_LOCAL_INITIAL_EVALUATION",
        "optimizer_constructed": True,
        "optimizer_loaded": False,
        "fresh_optimizer_state_empty_before_first_update": True,
        "policy_all_finite": True,
        "frozen_low_state_unchanged": True,
        "low_state_before_sha256": EXPECTED_LOW_MODULE_SHA256,
        "low_state_after_sha256": EXPECTED_LOW_MODULE_SHA256,
        "recurrent_hidden_restore": _hidden_receipt(),
        "initial_policy_state_sha256": POLICY_SHA,
        "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA,
        "reward_support_gate_pass": True,
        "reward_support_rows_sha256": ROWS_SHA,
        "first_update_checkpoint": {
            "relative_path": "model_first_update.pt",
            "bytes": 200,
            "sha256": FIRST_CHECKPOINT_SHA,
            "policy_state_sha256": FIRST_POLICY_SHA,
        },
        "iteration_receipt": {
            "iteration": 0,
            "optimizer_steps": 20,
            "cumulative_optimizer_steps": 20,
            "returns_all_finite": True,
            "losses_all_finite": True,
            "storage_reset_after_update": True,
        },
        "long_training_authorized": False,
    }


def _training() -> dict:
    receipts = []
    for iteration in range(12):
        receipts.append({
            "iteration": iteration,
            "rollout_steps": 200,
            "optimizer_steps": 20,
            "returns_all_finite": True,
            "losses_all_finite": True,
            "storage_reset_after_update": True,
            "recurrent_hidden_restore": _hidden_receipt(),
        })
    return {
        "schema": TRAINING_SCHEMA,
        "status": "COMPLETE",
        "failures": [],
        "initial_policy_state_sha256": POLICY_SHA,
        "final_policy_state_sha256": FINAL_POLICY_SHA,
        "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA,
        "reward_support_rows_sha256": ROWS_SHA,
        "reward_support_gate_pass": True,
        "training_iterations": 12,
        "rollout_steps": 2_400,
        "rollout_samples": 307_200,
        "ppo_optimizer_steps": 240,
        "optimizer_constructed": True,
        "actor_update": {"changed": True},
        "critic_update": {"changed": True},
        "policy_all_finite": True,
        "frozen_low_state_unchanged": True,
        "low_state_before_sha256": EXPECTED_LOW_MODULE_SHA256,
        "low_state_after_sha256": EXPECTED_LOW_MODULE_SHA256,
        "dense_return_authorizes_behavior": False,
        "long_training_authorized": False,
        "first_update_checkpoint": {
            "relative_path": "model_first_update.pt",
            "bytes": 200,
            "sha256": FIRST_CHECKPOINT_SHA,
            "policy_state_sha256": FIRST_POLICY_SHA,
        },
        "final_checkpoint": {
            "relative_path": "model_final.pt",
            "bytes": 300,
            "sha256": FINAL_CHECKPOINT_SHA,
            "policy_state_sha256": FINAL_POLICY_SHA,
        },
        "iteration_receipts": receipts,
    }


class CanaryProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.valid_rows = _raw_receipt()

    def test_checked_in_contract_is_exact_parent_release_ready(self) -> None:
        value = load_contract(CONTRACT, allow_unreleased_design=True)
        self.assertEqual(value["contract_schema"], CONTRACT_SCHEMA)
        self.assertEqual(value["status"], RELEASE_STATUS)
        self.assertEqual(value["decision"], RELEASE_DECISION)
        parent = load_contract(CONTRACT)["parent_v55"]
        self.assertEqual({key: parent[key] for key in EXPECTED_PARENT_V55}, EXPECTED_PARENT_V55)

    def test_evidence_closure_carries_the_same_exact_parent_release(self) -> None:
        closure = json.loads(CLOSURE.read_text(encoding="utf-8"))
        self.assertEqual(closure["status"], RELEASE_STATUS)
        parent = closure["parent_v55_own_artifacts"]
        self.assertEqual({key: parent[key] for key in EXPECTED_PARENT_V55}, EXPECTED_PARENT_V55)
        self.assertTrue(parent["required_before_gpu"])
        self.assertFalse(closure["long_training_authorized"])

    def test_exact_parent_release_and_wrong_null_rejections(self) -> None:
        released = _released_contract()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "released.json"
            _write_json(path, released)
            self.assertEqual(load_contract(path)["parent_v55"], released["parent_v55"])

            wrong = copy.deepcopy(released)
            wrong["parent_v55"]["teacher_fixed_rows_sha256"] = "0" * 64
            _write_json(path, wrong)
            with self.assertRaisesRegex(ProtocolError, "identity"):
                load_contract(path)

            null = copy.deepcopy(released)
            null["parent_v55"]["transaction_status_sha256"] = None
            _write_json(path, null)
            with self.assertRaises(ProtocolError):
                load_contract(path)

    def test_contract_rejects_duplicate_nonfinite_nonobject_partial_and_nested_json(self) -> None:
        released = _released_contract()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            bad_inputs = (
                ('{"contract_schema":"x","contract_schema":"y"}\n', "duplicate"),
                ('{"contract_schema":NaN}\n', "non-finite"),
                ('{"contract_schema":Infinity}\n', "non-finite"),
                ('[]\n', "root"),
            )
            for payload, message in bad_inputs:
                with self.subTest(payload=payload):
                    path.write_text(payload, encoding="utf-8")
                    with self.assertRaisesRegex(ProtocolError, message):
                        load_contract(path, allow_unreleased_design=True)

            partial = copy.deepcopy(released)
            partial["parent_v55"]["teacher_fixed_summary_sha256"] = None
            partial["status"] = PENDING_STATUS
            partial["decision"] = PENDING_DECISION
            _write_json(path, partial)
            with self.assertRaisesRegex(ProtocolError, "partial"):
                load_contract(path, allow_unreleased_design=True)

            nested = copy.deepcopy(released)
            nested["parent_v55"]["result_sha256"] = None
            nested["parent_v55"]["nested_v54"] = {
                "result_sha256": EXPECTED_PARENT_V55["result_sha256"]
            }
            _write_json(path, nested)
            with self.assertRaises(ProtocolError):
                load_contract(path)

    def test_contract_exact_locks_causal_command_forbidden_and_authorization_blocks(self) -> None:
        released = _released_contract()
        mutations = []
        changed_command = copy.deepcopy(released)
        changed_command["command_interface"]["rate_limits_per_second"][0] = 1.1
        mutations.append(changed_command)
        changed_cause = copy.deepcopy(released)
        changed_cause["causal_change"]["only_changed_mechanism"] = "TWO_CHANGES"
        mutations.append(changed_cause)
        changed_authorization = copy.deepcopy(released)
        changed_authorization["authorization_provenance"]["long_training_authorized"] = True
        mutations.append(changed_authorization)
        empty_forbidden = copy.deepcopy(released)
        empty_forbidden["forbidden"] = {}
        mutations.append(empty_forbidden)
        missing_forbidden = copy.deepcopy(released)
        missing_forbidden["forbidden"].pop("kick_action")
        mutations.append(missing_forbidden)
        enabled_forbidden = copy.deepcopy(released)
        enabled_forbidden["forbidden"]["kick_action"] = False
        mutations.append(enabled_forbidden)
        changed_low_activation = copy.deepcopy(released)
        changed_low_activation["qualified_low"]["actor_activation"] = "relu"
        mutations.append(changed_low_activation)
        changed_low_path = copy.deepcopy(released)
        changed_low_path["qualified_low"]["checkpoint_cluster_path"] += ".other"
        mutations.append(changed_low_path)
        changed_low_bytes = copy.deepcopy(released)
        changed_low_bytes["qualified_low"]["checkpoint_bytes"] += 1
        mutations.append(changed_low_bytes)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mutated.json"
            for index, value in enumerate(mutations):
                with self.subTest(index=index):
                    _write_json(path, value)
                    with self.assertRaises(ProtocolError):
                        load_contract(path)

    def test_every_released_contract_leaf_is_exact_locked(self) -> None:
        released = _released_contract()
        paths = _leaf_paths(released)
        self.assertGreater(len(paths), 150)
        accepted: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "leaf_mutation.json"
            for leaf_path in paths:
                mutated = copy.deepcopy(released)
                _replace_at_path(mutated, leaf_path)
                _write_json(path, mutated)
                try:
                    load_contract(path)
                except ProtocolError:
                    continue
                accepted.append(".".join(str(item) for item in leaf_path))
        self.assertEqual(accepted, [])
        extra = copy.deepcopy(released)
        extra["EXTRA_SENTINEL"] = True
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "extra.json"
            _write_json(path, extra)
            with self.assertRaisesRegex(ProtocolError, "top-level"):
                load_contract(path)

    def test_full_25600_transition_receipt_passes_and_has_both_side_support(self) -> None:
        summary = validate_first_rollout_transitions(self.valid_rows)
        self.assertEqual(len(self.valid_rows), 25_600)
        self.assertTrue(summary["gate_pass"])
        self.assertGreaterEqual(summary["positive_progress_count_by_goal_side"]["-1"], 1)
        self.assertGreaterEqual(summary["positive_progress_count_by_goal_side"]["1"], 1)
        self.assertGreaterEqual(summary["negative_progress_count"], 1)
        self.assertGreater(abs(summary["progress_reward_min"]), SUPPORT_EPSILON)

    def test_transition_mutations_are_rejected(self) -> None:
        cases = (
            ("terminal_reason", "GOAL", "terminal reason"),
            ("sample_serial", 999, "global_sample_serial"),
            ("goal_xy_m", [4.11, 0.0], "canonical goal"),
            ("goal_xy_m", [True, 0.0], "goal_xy_m"),
        )
        for key, replacement, message in cases:
            with self.subTest(key=key):
                rows = list(self.valid_rows)
                rows[1] = dict(rows[1])
                rows[1][key] = replacement
                with self.assertRaisesRegex(ProtocolError, message):
                    validate_first_rollout_transitions(rows)

    def test_nonterminal_continuity_mutation_is_rejected(self) -> None:
        rows = list(self.valid_rows)
        index = 128 + 7
        rows[index] = dict(rows[index])
        rows[index]["pre_robot_xy_m"] = [
            rows[index]["pre_robot_xy_m"][0] + 1.0e-3,
            rows[index]["pre_robot_xy_m"][1],
        ]
        with self.assertRaisesRegex(ProtocolError, "l_pre_m changed|next-pre discontinuity"):
            validate_first_rollout_transitions(rows)

    def test_early_terminal_priority_and_contiguous_global_reset_serials(self) -> None:
        rows = _early_terminal_receipt()
        summary = validate_first_rollout_transitions(rows)
        self.assertTrue(summary["gate_pass"])
        tick_one = rows[0]
        tick_four = rows[1]
        self.assertEqual(tick_one["native_ticks_executed"], 1)
        self.assertTrue(tick_one["contact_seen"])
        self.assertTrue(tick_one["boundary"])
        self.assertFalse(tick_one["goal"])
        self.assertEqual(tick_one["terminal_reason"], "BOUNDARY")
        self.assertEqual(tick_four["native_ticks_executed"], 4)
        self.assertEqual(rows[128]["sample_serial"], 129)
        self.assertEqual(rows[129]["sample_serial"], 130)
        self.assertEqual(rows[128]["episode_high_step_index_before_step"], 0)
        self.assertEqual(rows[129]["episode_high_step_index_before_step"], 0)

    def test_support_threshold_requires_recorded_and_reconstructed_intersection(self) -> None:
        rows = _raw_receipt(straddle_side=1)
        summary = validate_first_rollout_transitions(rows)
        self.assertFalse(summary["gate_pass"])
        self.assertEqual(summary["positive_progress_count_by_goal_side"]["1"], 0)
        self.assertFalse(summary["checks"]["positive_support_goal_side_positive"])

    def test_valid_behavioral_support_rejection_is_not_engineering_failure(self) -> None:
        result = build_result(
            _released_contract(),
            ResultInputs(_evaluation("initial"), _reward_support(passes=False), None, None, None),
        )
        self.assertEqual(result["result_schema"], RESULT_SCHEMA)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["decision"], DECISION_REWARD_SUPPORT_REJECTED)
        self.assertTrue(result["engineering_integrity_pass"])

    def test_support_cross_run_and_receipt_mutations_are_invalid(self) -> None:
        mutations = (
            ("initial_policy_state_sha256", "9" * 64),
            ("initial_checkpoint_sha256", "8" * 64),
            ("frozen_low_state_sha256_before_gate", "7" * 64),
            ("transition_rows_relative_path", "other.jsonl"),
            ("transition_rows_bytes", 0),
            ("transition_rows_sha256", "NOT_A_SHA"),
        )
        for key, replacement in mutations:
            with self.subTest(key=key):
                support = _reward_support(passes=False)
                support[key] = replacement
                result = build_result(
                    _released_contract(),
                    ResultInputs(_evaluation("initial"), support, None, None, None),
                )
                self.assertEqual(result["status"], "ERROR")
                self.assertEqual(result["decision"], DECISION_INVALID)

    def test_full_branch_identity_and_checkpoint_bridges(self) -> None:
        inputs = ResultInputs(
            _evaluation("initial"),
            _reward_support(passes=True),
            _first_update(),
            _training(),
            _evaluation("final"),
        )
        result = build_result(_released_contract(), inputs)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["decision"], DECISION_NO_BREAKTHROUGH)
        self.assertEqual(result["failures"], [])

        mutations = (
            ("first.initial_policy_state_sha256", "9" * 64),
            ("first.initial_checkpoint_sha256", "8" * 64),
            ("first.reward_support_rows_sha256", "7" * 64),
            ("first.first_update_checkpoint.relative_path", "wrong.pt"),
            ("training.first_update_checkpoint.sha256", "6" * 64),
            ("training.final_checkpoint.policy_state_sha256", "5" * 64),
        )
        for label, replacement in mutations:
            with self.subTest(label=label):
                first = _first_update()
                training = _training()
                target: dict = first if label.startswith("first.") else training
                pieces = label.split(".")[1:]
                for piece in pieces[:-1]:
                    target = target[piece]
                target[pieces[-1]] = replacement
                mutated = build_result(
                    _released_contract(),
                    ResultInputs(
                        _evaluation("initial"),
                        _reward_support(passes=True),
                        first,
                        training,
                        _evaluation("final"),
                    ),
                )
                self.assertEqual(mutated["status"], "ERROR")
                self.assertEqual(mutated["decision"], DECISION_INVALID)

    def test_evaluation_count_conservation_and_phase_provenance(self) -> None:
        for phase in ("initial", "final"):
            validate_evaluation_summary(_evaluation(phase), phase=phase)
        mutations = []
        bad_sum = _evaluation("final")
        bad_sum["goal_count"] = 1
        mutations.append(bad_sum)
        bad_hard = _evaluation("final")
        bad_hard["hard_goal_counts_by_goal_side"]["1"] = 1
        bad_hard["hard_goal_count"] = 1
        mutations.append(bad_hard)
        bad_outcomes = _evaluation("final")
        bad_outcomes["outcome_counts"]["timeout"] = 287
        mutations.append(bad_outcomes)
        bad_phase = _evaluation("final")
        bad_phase["fresh_policy_initialized"] = True
        mutations.append(bad_phase)
        for index, value in enumerate(mutations):
            with self.subTest(index=index):
                with self.assertRaises(ProtocolError):
                    validate_evaluation_summary(value, phase="final")

    def test_finalizer_records_only_exact_present_input_summary_hashes(self) -> None:
        import finalize_task_progress_canary

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract_path = root / "contract.json"
            initial_path = root / "initial.json"
            support_path = root / "support.json"
            _write_json(contract_path, _released_contract())
            _write_json(initial_path, _evaluation("initial"))
            _write_json(support_path, _reward_support(passes=False))
            initial_sha = hashlib.sha256(initial_path.read_bytes()).hexdigest()
            support_sha = hashlib.sha256(support_path.read_bytes()).hexdigest()
            argv = [
                "finalize_task_progress_canary.py",
                "--contract", str(contract_path),
                "--evidence-root", str(root),
                "--output-dir", str(root / "result"),
                "--initial-evaluation", str(initial_path),
                "--initial-evaluation-sha256", initial_sha,
                "--reward-support", str(support_path),
                "--reward-support-sha256", support_sha,
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(finalize_task_progress_canary.main(), 0)
            result = json.loads((root / "result/task_progress_canary_result.json").read_text(encoding="utf-8"))
            self.assertEqual(
                result["input_summary_sha256"],
                {"initial_evaluation": initial_sha, "reward_support": support_sha},
            )
            self.assertEqual(result["decision"], DECISION_REWARD_SUPPORT_REJECTED)


if __name__ == "__main__":
    unittest.main()
