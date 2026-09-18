from __future__ import annotations

import ast
import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import check_sparse_high_task_progress_canary as checker


ACTUAL_17944428_BASE_BODY_NAMES = [
    "base",
    "FL_hip", "FL_thigh", "FL_calf", "FL_foot",
    "FR_hip", "FR_thigh", "FR_calf", "FR_foot",
    "Head_upper", "Head_lower",
    "RL_hip", "RL_thigh", "RL_calf", "RL_foot",
    "RR_hip", "RR_thigh", "RR_calf", "RR_foot",
]


def valid_probe() -> dict[str, object]:
    def reset_state(ids: list[int]) -> dict[str, object]:
        expected = [
            [float(env_id), -float(env_id), 0.115, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            for env_id in ids
        ]
        return {
            "selected_ball_cache_zero_after": True,
            "selected_base_cache_zero_after": True,
            "selected_ball_cache_max_abs_after": 0.0,
            "selected_base_cache_max_abs_after": 0.0,
            "selected_ball_metadata_reset": True,
            "selected_base_metadata_reset": True,
            "selected_ball_root_state_shape": [len(ids), 13],
            "expected_ball_root_state_13d": expected,
            "actual_ball_root_state_13d": copy.deepcopy(expected),
            "selected_ball_root_state_13d_max_error": 0.0,
            "selected_ball_root_state_13d_canonical": True,
        }

    tick_rows = [
        {
            "env_id": env_id,
            "ball_current_n": 2.0 + local,
            "robot_current_n": 2.5 + local,
            "dual_threshold_hit": True,
        }
        for local, env_id in enumerate(checker.CONTACT_RESET_PARTIAL_IDS)
    ]
    trial_rows = [
        {
            "env_id": row["env_id"],
            "ball_current_peak_n": row["ball_current_n"],
            "robot_current_peak_n": row["robot_current_n"],
            "first_dual_hit_tick": 1,
            "ever_same_tick_dual_hit": True,
        }
        for row in tick_rows
    ]
    qualification = {
        "qualification_job": checker.CONTACT_RESET_QUALIFICATION_JOB,
        "contact_body_contract_sha256": checker.CONTACT_RESET_BODY_CONTRACT_SHA256,
        "accepted_body_names": list(checker.CONTACT_RESET_BODY_NAMES),
        "live_body_names": list(checker.CONTACT_RESET_BODY_NAMES),
        "base_contact_body_names": list(ACTUAL_17944428_BASE_BODY_NAMES),
        "base_body_index_by_accepted_body": [
            ACTUAL_17944428_BASE_BODY_NAMES.index(name)
            for name in checker.CONTACT_RESET_BODY_NAMES
        ],
        "body_count": 19,
        "ball_sensor_body_count": 1,
        "base_sensor_body_count": 19,
        "filter_count": 19,
        "ball_force_matrix_shape": [8, 1, 19, 3],
        "base_net_force_shape": [8, 19, 3],
        "candidate_order": list(checker.CONTACT_RESET_CALF_NAMES),
        "max_raw_physics_ticks_per_candidate": 4,
        "contact_threshold_n": 1.0,
        "current_same_tick_same_body_required": True,
        "history_used_as_contact_truth": False,
        "ball_displacement_used_as_contact_truth": False,
        "low_actor_used": False,
        "low_env_step_used": False,
        "action_manager_processing_used": False,
        "trials": [{
            "body_name": "FL_calf",
            "body_index": 11,
            "filter_index": 11,
            "base_body_index": 3,
            "ticks_run": 1,
            "ticks": [{"tick": 1, "per_env": tick_rows}],
            "per_env": trial_rows,
            "all_selected_same_tick_hit": True,
            "witness_tick": 1,
        }],
        "winning_body_name": "FL_calf",
        "winning_body_index": 11,
        "winning_filter_index": 11,
        "winning_base_body_index": 3,
        "winning_tick": 1,
        "pass": True,
    }
    physical = {
        "selected_ids": list(checker.CONTACT_RESET_PARTIAL_IDS),
        "reset_called_immediately_after_witness": True,
        "selected_physical_caches_overwritten_before_reset": False,
        **reset_state(list(checker.CONTACT_RESET_PARTIAL_IDS)),
        "unselected_ball_sensor_unchanged": True,
        "unselected_base_sensor_unchanged": True,
        "unselected_ball_state_unchanged": True,
        "pass": True,
    }
    raw = {
        "boundary": "AFTER_RESET_BEFORE_PUBLIC_SENSOR_DATA_OR_PHYSICS",
        "physics_dt_s": 0.005,
        "ball_raw_native_shape": [8, 19, 3],
        "ball_raw_reshaped_shape": [8, 1, 19, 3],
        "base_raw_native_shape": [152, 3],
        "base_raw_reshaped_shape": [8, 19, 3],
        "ball_sensor_body_count": 1,
        "ball_filter_count": 19,
        "base_sensor_body_count": 19,
        "base_body_name": "FL_calf",
        "base_body_index": 3,
        "ball_raw_selected_max_n": 2.0,
        "base_raw_selected_body_max_n": 2.0,
        "raw_values_finite": True,
        "selected_private_digest_before_sha256": "a" * 64,
        "selected_private_digest_after_sha256": "a" * 64,
        "unselected_private_digest_before_sha256": "b" * 64,
        "unselected_private_digest_after_sha256": "b" * 64,
        "selected_private_max_delta": 0.0,
        "unselected_private_max_delta": 0.0,
        "private_cache_unchanged_selected": True,
        "private_cache_unchanged_unselected": True,
        "raw_force_magnitude_used_as_pass_criterion": False,
        "public_contact_sensor_data_reads": 0,
        "pass": True,
    }
    one_tick = {
        "boundary": "EXACTLY_ONE_RAW_PHYSICS_TICK_AFTER_PHYSICAL_PARTIAL_RESET",
        "raw_physics_ticks": 1,
        "physics_dt_s": 0.005,
        "public_contact_sensor_data_reads_before_tick": 0,
        "public_contact_sensor_data_reads_after_tick": 2,
        "contact_body_name": "FL_calf",
        "contact_body_index": 11,
        "base_body_name": "FL_calf",
        "base_body_index": 3,
        "ball_force_matrix_shape": [8, 1, 19, 3],
        "base_net_force_shape": [8, 19, 3],
        "ball_history_shape": [8, 4, 1, 19, 3],
        "base_history_shape": [8, 3, 19, 3],
        "ball_history_tail_max_abs": 0.0,
        "base_history_tail_max_abs": 0.0,
        "ball_index_zero_selected_max_n": 0.25,
        "base_index_zero_selected_body_max_n": 0.5,
        "index_zero_allowed_any_contact": True,
        "ball_timestamp_s": [0.005] * 3,
        "ball_timestamp_last_update_s": [0.005] * 3,
        "base_timestamp_s": [0.005] * 3,
        "base_timestamp_last_update_s": [0.005] * 3,
        "ball_outdated_after_tick": False,
        "base_outdated_after_tick": False,
        "finite": True,
        "pass": True,
    }
    production = {
        "boundary": "INDEPENDENT_FRESH_RESET_THEN_ONE_NORMAL_LOW_ENV_STEP",
        "fresh_reset": {**reset_state(list(checker.CONTACT_RESET_PARTIAL_IDS)), "pass": True},
        "configured_decimation": 4,
        "normal_low_steps": 1,
        "raw_physics_ticks": 4,
        "physics_dt_s": 0.005,
        "low_action_max_abs": 0.4,
        "low_action_min": -0.4,
        "low_action_max": 0.3,
        "ball_history_shape": [8, 4, 1, 19, 3],
        "base_history_shape": [8, 3, 19, 3],
        "ball_timestamp_s": [0.02] * 3,
        "ball_timestamp_last_update_s": [0.02] * 3,
        "base_timestamp_s": [0.02] * 3,
        "base_timestamp_last_update_s": [0.02] * 3,
        "ball_outdated_after_step": False,
        "base_outdated_after_step": False,
        "finite": True,
        "pass": True,
    }
    sentinel_partial = {
        "selected_ids": list(checker.CONTACT_RESET_PARTIAL_IDS),
        "sentinel_scheme": "DISTINCT_BASE_PLUS_FIELD_PLUS_ENV_SLOT",
        "selected_sensor_cache_dirty_before": True,
        **reset_state(list(checker.CONTACT_RESET_PARTIAL_IDS)),
        "unselected_ball_sensor_unchanged": True,
        "unselected_base_sensor_unchanged": True,
        "unselected_ball_state_unchanged": True,
        "pass": True,
    }
    sentinel_full = {
        "selected_ids": list(range(8)),
        "sentinel_scheme": "DISTINCT_BASE_PLUS_FIELD_PLUS_ENV_SLOT",
        "selected_sensor_cache_dirty_before": True,
        **reset_state(list(range(8))),
        "pass": True,
    }
    checks = {
        "contact_qualification": True,
        "physical_partial_reset": True,
        "post_reset_raw_query": True,
        "first_post_reset_physics_tick": True,
        "production_low_step": True,
        "sentinel_partial": True,
        "sentinel_full": True,
    }
    return {
        "schema": checker.CONTACT_RESET_PROBE_SCHEMA,
        "status": "PASS",
        "num_envs": 8,
        "partial_reset_ids": list(checker.CONTACT_RESET_PARTIAL_IDS),
        "unselected_ids": list(checker.CONTACT_RESET_UNSELECTED_IDS),
        "zero_tolerance": 0.0,
        "state_tolerance": 1.0e-6,
        "contact_threshold_n": 1.0,
        "contact_qualification": qualification,
        "physical_partial_reset": physical,
        "post_reset_raw_query": raw,
        "first_post_reset_physics_tick": one_tick,
        "production_low_step": production,
        "sentinel_partial": sentinel_partial,
        "sentinel_full": sentinel_full,
        "checks": checks,
        "ppo_constructed": False,
        "optimizer_constructed": False,
        "runtime_identity": {},
        "source_receipts": {},
        "failures": [],
        "long_training_authorized": False,
    }


class ContactResetProbeDeliveryTests(unittest.TestCase):
    def test_probe_is_real_isaac_fresh_process_without_ppo(self) -> None:
        path = ROOT / "probe_contact_reset_lifecycle.py"
        text = path.read_text(encoding="utf-8")
        ast.parse(text, filename=str(path))
        self.assertLess(text.index("from isaaclab.app import AppLauncher"), text.index("import torch"))
        self.assertNotIn("OnPolicyRunner", text)
        self.assertNotIn("optimizer.step", text)
        for token in (
            "CONTACT_QUALIFICATION_JOB",
            "CONTACT_BODY_CONTRACT_SHA256",
            'CALF_CANDIDATES = ("FL_calf", "FR_calf", "RL_calf", "RR_calf")',
            '"base_body_index_by_accepted_body": base_body_index_by_accepted_body',
            'base_body_index = base_names.index(body_name)',
            'qualification["winning_base_body_index"] = base_body_index',
            "_establish_contact_and_reset",
            'trial["all_selected_same_tick_hit"] = bool(dual.all().item())',
            'qualification["winning_tick"] = trial["witness_tick"]',
            '"selected_physical_caches_overwritten_before_reset": False',
            "_post_reset_raw_query",
            "get_contact_force_matrix",
            "get_net_contact_forces",
            '"raw_force_magnitude_used_as_pass_criterion": False',
            "_first_post_reset_tick",
            '"raw_physics_ticks": 1',
            "ball_history[ids, 1:]",
            "base_history[ids, 1:]",
            "_production_low_step",
            '"configured_decimation": decimation',
            "_sentinel_partial_lane",
            "_sentinel_full_lane",
            "_dirty_ball_state",
            "unselected_ball_sensor_unchanged",
            "unselected_base_sensor_unchanged",
            "selected_ball_root_state_13d_canonical",
            '"failure_classification": classification',
            '"diagnostics": {',
            "def _json_safe",
            "return value if math.isfinite(value) else None",
            '"lane_receipts": _json_safe(LANE_RECEIPTS)',
            '"error_details": _json_safe(error_details)',
            "CONTACT_RESET_LIFECYCLE_PROBE_FINAL_GUARD",
            "os._exit(92)",
        ):
            self.assertIn(token, text)
        self.assertLess(
            text.index('atomic_json(output / "contact_reset_lifecycle_probe.json"'),
            text.index('atomic_bytes(output / "CONTACT_RESET_LIFECYCLE_PROBE_FINAL_GUARD"'),
        )

    def test_job_runs_probe_before_any_policy_process_and_not_full_suite(self) -> None:
        job = (ROOT / "run_sparse_high_task_progress_canary.sbatch").read_text(encoding="utf-8")
        probe = job.index("phase contact_reset_lifecycle_probe START")
        self.assertLess(probe, job.index("phase initial_fixed_evaluation START"))
        self.assertLess(probe, job.index("phase bounded_task_progress_ppo START"))
        self.assertIn('run_isaac_child 180s "$LOGS/contact_reset_lifecycle_probe.log"', job)
        self.assertNotIn("unittest discover", job)
        self.assertIn("tests.test_contact_reset_probe_delivery", job)
        self.assertIn("RSL_RL_3_1_2_NO_SIM_RECURRENT_LIFECYCLE_SMOKE_FAILED", job)
        self.assertIn("V56_SUBMITTER_FULL_STATIC_SUITE_NOT_PASS", job)
        self.assertIn("V56_SUBMITTER_RSL_RL_SMOKE_NOT_PASS", job)
        self.assertIn("FAILURE_REASON=$1", job)
        self.assertIn('phase "$ACTIVE_PHASE" FAIL "$rc"', job)

    def test_independent_checker_accepts_only_complete_probe_receipt(self) -> None:
        value = valid_probe()
        with mock.patch.object(checker, "validate_runtime"), mock.patch.object(
            checker, "validate_source_receipts"
        ):
            checker.validate_contact_reset_probe(value, environment={}, source=ROOT)

            mutations = {
                "body contract pin": lambda item: item["contact_qualification"].__setitem__(
                    "contact_body_contract_sha256", "0" * 64
                ),
                "body/filter pairing": lambda item: item["contact_qualification"]["trials"][0].__setitem__(
                    "filter_index", 12
                ),
                "accepted-to-base mapping": lambda item: item["contact_qualification"][
                    "base_body_index_by_accepted_body"
                ].__setitem__(11, 11),
                "forced articulation index on base tensor": lambda item: (
                    item["contact_qualification"]["trials"][0].__setitem__("base_body_index", 11),
                    item["contact_qualification"].__setitem__("winning_base_body_index", 11),
                    item["post_reset_raw_query"].__setitem__("base_body_index", 11),
                    item["first_post_reset_physics_tick"].__setitem__("base_body_index", 11),
                ),
                "physical private cache": lambda item: item["physical_partial_reset"].__setitem__(
                    "selected_ball_cache_max_abs_after", 1.0
                ),
                "physical canonical 13D": lambda item: item["physical_partial_reset"][
                    "actual_ball_root_state_13d"
                ][0].__setitem__(2, 0.25),
                "physical unselected slot": lambda item: item["physical_partial_reset"].__setitem__(
                    "unselected_base_sensor_unchanged", False
                ),
                "raw cache digest": lambda item: item["post_reset_raw_query"].__setitem__(
                    "selected_private_digest_after_sha256", "c" * 64
                ),
                "raw non-finite declaration": lambda item: item["post_reset_raw_query"].__setitem__(
                    "raw_values_finite", False
                ),
                "Boolean selected ID": lambda item: item.__setitem__(
                    "partial_reset_ids", [True, 4, 6]
                ),
                "first refresh tick count": lambda item: item["first_post_reset_physics_tick"].__setitem__(
                    "raw_physics_ticks", 4
                ),
                "first refresh timestamp": lambda item: item["first_post_reset_physics_tick"].__setitem__(
                    "ball_timestamp_s", [0.02] * 3
                ),
                "first refresh history tail": lambda item: item["first_post_reset_physics_tick"].__setitem__(
                    "base_history_tail_max_abs", 1.0
                ),
                "production decimation": lambda item: item["production_low_step"].__setitem__(
                    "raw_physics_ticks", 3
                ),
                "partial sentinel dirtiness": lambda item: item["sentinel_partial"].__setitem__(
                    "selected_sensor_cache_dirty_before", False
                ),
                "full sentinel zero": lambda item: item["sentinel_full"].__setitem__(
                    "selected_base_cache_max_abs_after", 1.0
                ),
                "undeclared field": lambda item: item.__setitem__("extra", True),
            }
            for label, mutate in mutations.items():
                with self.subTest(label=label):
                    invalid = copy.deepcopy(value)
                    mutate(invalid)
                    with self.assertRaises(checker.ProtocolError):
                        checker.validate_contact_reset_probe(invalid, environment={}, source=ROOT)

    def test_checker_rejects_cross_tick_selected_contact_splice(self) -> None:
        value = valid_probe()
        trial = value["contact_qualification"]["trials"][0]
        trial["ticks_run"] = 2
        trial["ticks"] = [
            {
                "tick": 1,
                "per_env": [
                    {"env_id": 1, "ball_current_n": 2.0, "robot_current_n": 2.0, "dual_threshold_hit": True},
                    {"env_id": 4, "ball_current_n": 0.0, "robot_current_n": 0.0, "dual_threshold_hit": False},
                    {"env_id": 6, "ball_current_n": 0.0, "robot_current_n": 0.0, "dual_threshold_hit": False},
                ],
            },
            {
                "tick": 2,
                "per_env": [
                    {"env_id": 1, "ball_current_n": 0.0, "robot_current_n": 0.0, "dual_threshold_hit": False},
                    {"env_id": 4, "ball_current_n": 2.0, "robot_current_n": 2.0, "dual_threshold_hit": True},
                    {"env_id": 6, "ball_current_n": 2.0, "robot_current_n": 2.0, "dual_threshold_hit": True},
                ],
            },
        ]
        trial["per_env"] = [
            {"env_id": 1, "ball_current_peak_n": 2.0, "robot_current_peak_n": 2.0, "first_dual_hit_tick": 1, "ever_same_tick_dual_hit": True},
            {"env_id": 4, "ball_current_peak_n": 2.0, "robot_current_peak_n": 2.0, "first_dual_hit_tick": 2, "ever_same_tick_dual_hit": True},
            {"env_id": 6, "ball_current_peak_n": 2.0, "robot_current_peak_n": 2.0, "first_dual_hit_tick": 2, "ever_same_tick_dual_hit": True},
        ]
        trial["all_selected_same_tick_hit"] = True
        trial["witness_tick"] = 2
        value["contact_qualification"]["winning_tick"] = 2
        with mock.patch.object(checker, "validate_runtime"), mock.patch.object(
            checker, "validate_source_receipts"
        ), self.assertRaises(checker.ProtocolError):
            checker.validate_contact_reset_probe(value, environment={}, source=ROOT)

    def test_checker_closes_probe_artifact_phase_and_output(self) -> None:
        source = (ROOT / "check_sparse_high_task_progress_canary.py").read_text(encoding="utf-8")
        for token in (
            "validate_contact_reset_probe",
            'child_rc(evidence / "contact_reset_probe_child_rc.txt"',
            '"CONTACT_RESET_LIFECYCLE_PROBE_FINAL_GUARD"',
            '("contact_reset_lifecycle_probe", "START", "")',
            '("contact_reset_lifecycle_probe", "COMPLETE", "0")',
            "CONTACT_RESET_LIFECYCLE_PROBE_SHA256=",
        ):
            self.assertIn(token, source)

    def test_failure_receipt_diagnostics_are_strict_and_independently_classified(self) -> None:
        trace = "Traceback (most recent call last):\nRuntimeError: raw query failed\n"
        value = {
            "schema": checker.CONTACT_RESET_PROBE_SCHEMA,
            "status": "ERROR",
            "failure_classification": "POST_RESET_RAW_QUERY_ERROR",
            "failed_stage": "post_reset_raw_query",
            "exception_type": "RuntimeError",
            "message": "raw query failed",
            "diagnostics": {
                "completed_lanes": ["contact_qualification", "physical_partial_reset"],
                "current_lane": "post_reset_raw_query",
                "lane_receipts": {
                    "contact_qualification": {"pass": True},
                    "physical_partial_reset": {"pass": True},
                    "post_reset_raw_query": {"pass": False},
                },
                "error_details": {"raw_exception_type": "RuntimeError"},
            },
            "traceback": trace,
            "long_training_authorized": False,
        }
        classification, trace_sha = checker.validate_contact_reset_probe_failure(value)
        self.assertEqual(classification, "POST_RESET_RAW_QUERY_ERROR")
        self.assertEqual(trace_sha, hashlib.sha256(trace.encode("utf-8")).hexdigest())

        mutations = {
            "producer classification": lambda item: item.__setitem__(
                "failure_classification", "UNCLASSIFIED_ERROR"
            ),
            "stage/current mismatch": lambda item: item["diagnostics"].__setitem__(
                "current_lane", "production_low_step"
            ),
            "completed lane gap": lambda item: item["diagnostics"].__setitem__(
                "completed_lanes", ["physical_partial_reset"]
            ),
            "unreached receipt": lambda item: item["diagnostics"]["lane_receipts"].__setitem__(
                "sentinel_full", {"pass": False}
            ),
            "non-finite diagnostic": lambda item: item["diagnostics"]["error_details"].__setitem__(
                "force", float("inf")
            ),
            "extra diagnostic key": lambda item: item["diagnostics"].__setitem__("extra", True),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                invalid = copy.deepcopy(value)
                mutate(invalid)
                with self.assertRaises(checker.ProtocolError):
                    checker.validate_contact_reset_probe_failure(invalid)

    def test_failed_probe_transaction_prints_durable_attribution_without_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "sparse_high_task_progress_canary_v1_123_failed_probe"
            evidence = run / "evidence"
            probe = evidence / "contact_reset_probe"
            logs = run / "logs"
            probe.mkdir(parents=True)
            logs.mkdir()
            (evidence / "process_rc.txt").write_text("2\n", encoding="ascii")
            (evidence / "contact_reset_probe_child_rc.txt").write_text("92\n", encoding="ascii")
            (evidence / "SPARSE_HIGH_TASK_PROGRESS_CANARY_TRANSACTION_FINAL_GUARD").write_text(
                "FAIL phase=contact_reset_lifecycle_probe process_rc=2\n", encoding="utf-8"
            )
            (evidence / "phase_events.tsv").write_text(
                "timestamp\tphase\tstate\trc\n"
                "2026-09-18T00:00:00+00:00\twrapper_static_preflight\tSTART\t\n"
                "2026-09-18T00:00:01+00:00\twrapper_static_preflight\tCOMPLETE\t0\n"
                "2026-09-18T00:00:02+00:00\ttask_package_snapshot\tSTART\t\n"
                "2026-09-18T00:00:03+00:00\ttask_package_snapshot\tCOMPLETE\t0\n"
                "2026-09-18T00:00:04+00:00\tcontact_reset_lifecycle_probe\tSTART\t\n"
                "2026-09-18T00:00:05+00:00\tcontact_reset_lifecycle_probe\tFAIL\t92\n",
                encoding="utf-8",
            )
            (evidence / "transaction_status.json").write_text(
                json.dumps({
                    "schema": "sparse_high_task_progress_canary_v1/transaction_status/v1",
                    "status": "ERROR",
                    "process_rc": 2,
                    "failed_phase": "contact_reset_lifecycle_probe",
                    "failure": "CONTACT_RESET_PROBE_CHILD_RC_92",
                    "ppo_optimizer_steps": 0,
                }, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            failure_trace = "Traceback (most recent call last):\nProbeFailure: no qualified calf\n"
            (probe / "contact_reset_lifecycle_probe_failure.json").write_text(
                json.dumps({
                    "schema": checker.CONTACT_RESET_PROBE_SCHEMA,
                    "status": "ERROR",
                    "failure_classification": "PHYSICAL_CONTACT_ESTABLISHMENT_ERROR",
                    "failed_stage": "contact_qualification",
                    "exception_type": "ProbeFailure",
                    "message": "no qualified calf established same-tick contact",
                    "diagnostics": {
                        "completed_lanes": [],
                        "current_lane": "contact_qualification",
                        "lane_receipts": {"contact_qualification": {"pass": False}},
                        "error_details": {
                            "failure_kind": "PHYSICAL_CONTACT_ESTABLISHMENT",
                            "actual_contact_established": False,
                        },
                    },
                    "traceback": failure_trace,
                    "long_training_authorized": False,
                }, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (logs / "contact_reset_lifecycle_probe.log").write_text("probe failed\n", encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                rc = checker.diagnose_failed_transaction(run, evidence)
            rendered = output.getvalue()
            self.assertEqual(rc, 2)
            for line in (
                "V56_CHECK=FAIL",
                "ENGINEERING_INTEGRITY=FAIL",
                "PROCESS_RC=2",
                "FAILED_PHASE=contact_reset_lifecycle_probe",
                "ERROR=CONTACT_RESET_PROBE_CHILD_RC_92",
                "FAILURE_KIND=CHILD_PROCESS",
                "FAILED_PHASE_RC=92",
                "FAILED_CHILD_RC=92",
                "PPO_OPTIMIZER_STEPS=0",
                "PROBE_ERROR=ProbeFailure: no qualified calf established same-tick contact",
                "PROBE_FAILURE_CLASSIFICATION=PHYSICAL_CONTACT_ESTABLISHMENT_ERROR",
                f"PROBE_TRACEBACK_SHA256={hashlib.sha256(failure_trace.encode('utf-8')).hexdigest()}",
                "PROBE_FAILURE_SHA256=",
                "PROBE_LOG_SHA256=",
                "LONG_TRAINING_AUTHORIZED=NO",
            ):
                self.assertIn(line, rendered)
            self.assertNotIn("REQUIRED_FILE_MISSING", rendered)

            (evidence / "SPARSE_HIGH_TASK_PROGRESS_CANARY_TRANSACTION_FINAL_GUARD").write_text(
                "FAIL phase=wrong process_rc=2\n", encoding="utf-8"
            )
            with self.assertRaises(checker.ProtocolError):
                checker.diagnose_failed_transaction(run, evidence)

    def test_failed_probe_rc_zero_missing_guard_reports_wrapper_gate_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "sparse_high_task_progress_canary_v1_124_missing_guard"
            evidence = run / "evidence"
            probe = evidence / "contact_reset_probe"
            logs = run / "logs"
            probe.mkdir(parents=True)
            logs.mkdir()
            (evidence / "process_rc.txt").write_text("2\n", encoding="ascii")
            (evidence / "contact_reset_probe_child_rc.txt").write_text("0\n", encoding="ascii")
            (evidence / "SPARSE_HIGH_TASK_PROGRESS_CANARY_TRANSACTION_FINAL_GUARD").write_text(
                "FAIL phase=contact_reset_lifecycle_probe process_rc=2\n", encoding="utf-8"
            )
            (evidence / "phase_events.tsv").write_text(
                "timestamp\tphase\tstate\trc\n"
                "2026-09-18T00:00:00+00:00\tcontact_reset_lifecycle_probe\tSTART\t\n"
                "2026-09-18T00:00:01+00:00\tcontact_reset_lifecycle_probe\tFAIL\t2\n",
                encoding="utf-8",
            )
            (evidence / "transaction_status.json").write_text(
                json.dumps({
                    "schema": "sparse_high_task_progress_canary_v1/transaction_status/v1",
                    "status": "ERROR",
                    "process_rc": 2,
                    "failed_phase": "contact_reset_lifecycle_probe",
                    "failure": "CONTACT_RESET_LIFECYCLE_PROBE_GUARD_MISSING",
                    "ppo_optimizer_steps": 0,
                }, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (logs / "contact_reset_lifecycle_probe.log").write_text(
                "probe exited zero without guard\n", encoding="utf-8"
            )
            output = io.StringIO()
            with redirect_stdout(output):
                rc = checker.diagnose_failed_transaction(run, evidence)
            rendered = output.getvalue()
            self.assertEqual(rc, 2)
            for line in (
                "V56_CHECK=FAIL",
                "ENGINEERING_INTEGRITY=FAIL",
                "ERROR=CONTACT_RESET_LIFECYCLE_PROBE_GUARD_MISSING",
                "FAILURE_KIND=WRAPPER_OR_GATE",
                "FAILED_PHASE_RC=2",
                "FAILED_CHILD_RC=0",
                "PROBE_FAILURE_CLASSIFICATION=ABSENT",
                "PROBE_TRACEBACK_SHA256=ABSENT",
                "PROBE_FAILURE_RECEIPT=ABSENT",
                "PROBE_LOG_SHA256=",
            ):
                self.assertIn(line, rendered)

    def test_late_training_failure_preserves_completed_optimizer_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "sparse_high_task_progress_canary_v1_125_late_training"
            evidence = run / "evidence"
            training = evidence / "training"
            logs = run / "logs"
            training.mkdir(parents=True)
            logs.mkdir()
            (evidence / "process_rc.txt").write_text("2\n", encoding="ascii")
            (evidence / "training_child_rc.txt").write_text("139\n", encoding="ascii")
            (evidence / "SPARSE_HIGH_TASK_PROGRESS_CANARY_TRANSACTION_FINAL_GUARD").write_text(
                "FAIL phase=bounded_task_progress_ppo process_rc=2\n", encoding="utf-8"
            )
            (evidence / "phase_events.tsv").write_text(
                "timestamp\tphase\tstate\trc\n"
                "2026-09-18T00:00:00+00:00\tbounded_task_progress_ppo\tSTART\t\n"
                "2026-09-18T00:00:01+00:00\tbounded_task_progress_ppo\tFAIL\t139\n",
                encoding="utf-8",
            )
            (evidence / "transaction_status.json").write_text(
                json.dumps({
                    "schema": "sparse_high_task_progress_canary_v1/transaction_status/v1",
                    "status": "ERROR",
                    "process_rc": 2,
                    "failed_phase": "bounded_task_progress_ppo",
                    "failure": "TRAINING_CHILD_RC_139",
                    "ppo_optimizer_steps": 160,
                }, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (training / "training_failure.json").write_text(
                '{"status":"ERROR","message":"reset failed"}\n', encoding="utf-8"
            )
            (training / "iteration_progress.json").write_text(
                json.dumps({
                    "status": "ACTIVE",
                    "completed_iterations": 8,
                    "rollout_steps": 1600,
                    "optimizer_steps": 160,
                    "last_iteration": {
                        "iteration": 7,
                        "rollout_steps": 200,
                        "optimizer_steps": 20,
                        "cumulative_optimizer_steps": 160,
                    },
                }, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (logs / "training.log").write_text("training reset failed\n", encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                rc = checker.diagnose_failed_transaction(run, evidence)
            rendered = output.getvalue()
            self.assertEqual(rc, 2)
            for line in (
                "FAILED_PHASE=bounded_task_progress_ppo",
                "FAILURE_KIND=CHILD_PROCESS",
                "FAILED_CHILD_RC=139",
                "PPO_OPTIMIZER_STEPS=160",
                "ITERATION_PROGRESS_SHA256=",
                "TRAINING_FAILURE_RECEIPT=",
                "TRAINING_FAILURE_SHA256=",
            ):
                self.assertIn(line, rendered)
            status = json.loads((evidence / "transaction_status.json").read_text(encoding="utf-8"))
            status["ppo_optimizer_steps"] = 0
            (evidence / "transaction_status.json").write_text(
                json.dumps(status, sort_keys=True) + "\n", encoding="utf-8"
            )
            with self.assertRaises(checker.ProtocolError):
                checker.diagnose_failed_transaction(run, evidence)

    def test_result_present_late_failures_dispatch_to_actionable_diagnosis(self) -> None:
        cases = (
            ("terminal_aggregate", "FINALIZATION_GUARD_NOT_PASS", "finalization.log"),
            ("independent_result_check", "IN_JOB_CHECKER_FAILED", "in_job_checker.log"),
        )
        for phase, failure, log_name in cases:
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary:
                run = Path(temporary) / f"sparse_high_task_progress_canary_v1_126_{phase}"
                evidence = run / "evidence"
                training = evidence / "training"
                result = evidence / "result"
                logs = run / "logs"
                training.mkdir(parents=True)
                result.mkdir()
                logs.mkdir()
                (result / "task_progress_canary_result.json").write_text("{}\n", encoding="utf-8")
                (evidence / "process_rc.txt").write_text("2\n", encoding="ascii")
                (evidence / "SPARSE_HIGH_TASK_PROGRESS_CANARY_TRANSACTION_FINAL_GUARD").write_text(
                    f"FAIL phase={phase} process_rc=2\n", encoding="utf-8"
                )
                (evidence / "phase_events.tsv").write_text(
                    "timestamp\tphase\tstate\trc\n"
                    f"2026-09-18T00:00:00+00:00\t{phase}\tSTART\t\n"
                    f"2026-09-18T00:00:01+00:00\t{phase}\tFAIL\t2\n",
                    encoding="utf-8",
                )
                (evidence / "transaction_status.json").write_text(
                    json.dumps({
                        "schema": "sparse_high_task_progress_canary_v1/transaction_status/v1",
                        "status": "ERROR",
                        "process_rc": 2,
                        "failed_phase": phase,
                        "failure": failure,
                        "ppo_optimizer_steps": 240,
                    }, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                (training / "iteration_progress.json").write_text(
                    json.dumps({
                        "status": "COMPLETE",
                        "completed_iterations": 12,
                        "rollout_steps": 2400,
                        "optimizer_steps": 240,
                        "last_iteration": {
                            "iteration": 11,
                            "rollout_steps": 200,
                            "optimizer_steps": 20,
                            "cumulative_optimizer_steps": 240,
                        },
                    }, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                (logs / log_name).write_text(f"{failure}\n", encoding="utf-8")
                output = io.StringIO()
                with mock.patch.object(sys, "argv", ["checker", str(run)]), redirect_stdout(output):
                    rc = checker.main()
                rendered = output.getvalue()
                self.assertEqual(rc, 2)
                for line in (
                    "V56_CHECK=FAIL",
                    "ENGINEERING_INTEGRITY=FAIL",
                    f"FAILED_PHASE={phase}",
                    f"ERROR={failure}",
                    "FAILURE_KIND=WRAPPER_OR_GATE",
                    "PPO_OPTIMIZER_STEPS=240",
                    f"FAILED_PHASE_LOG={logs / log_name}",
                    "FAILED_PHASE_LOG_SHA256=",
                ):
                    self.assertIn(line, rendered)
                self.assertNotIn("REQUIRED_FILE_MISSING", rendered)


if __name__ == "__main__":
    unittest.main()
