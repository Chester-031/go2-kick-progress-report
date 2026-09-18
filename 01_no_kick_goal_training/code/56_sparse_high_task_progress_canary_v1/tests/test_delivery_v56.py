from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import check_sparse_high_task_progress_canary as checker
from validate_parent_v55 import ParentError, release_pins
from validate_source_manifest import SourceManifestError, validate as validate_source_manifest


class DeliveryStaticTests(unittest.TestCase):
    def test_checker_is_stdlib_only_and_isolated_help_runs(self) -> None:
        path = ROOT / "check_sparse_high_task_progress_canary.py"
        tree = ast.parse(path.read_bytes(), filename=str(path))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
        self.assertTrue({"torch", "rsl_rl", "canary_protocol", "task_progress_reward"}.isdisjoint(imported))
        completed = subprocess.run(
            [sys.executable, "-B", "-I", "-S", str(path), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_shells_parse_and_encode_single_transaction_branches(self) -> None:
        paths = [
            ROOT / "check_sparse_high_task_progress_canary.sh",
            ROOT / "submit_and_check_sparse_high_task_progress_canary.sh",
            ROOT / "run_sparse_high_task_progress_canary.sbatch",
        ]
        completed = subprocess.run(["bash", "-n", *(str(path) for path in paths)], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        job = paths[-1].read_text(encoding="utf-8")
        for token in (
            "V56_DELIVERY_SCHEMA_STATE",
            "PARENT_V55_OWN_FACT_AND_HASH_GATE_FAILED",
            "INDEPENDENT_REWARD_SUPPORT_RECONSTRUCTION_FAILED",
            "REWARD_SUPPORT_REJECTED_FINAL_GUARD",
            "ACTUAL_CHECKPOINT_POLICY_STATES_DID_NOT_CHANGE",
            "durable_line",
            "os.fsync",
            "V56_CHECK_IN_JOB=1",
        ):
            self.assertIn(token, job)
        self.assertIn("#SBATCH --time=06:00:00", job)

    def test_queued_drift_failures_are_inside_durable_transaction_trap(self) -> None:
        job = (ROOT / "run_sparse_high_task_progress_canary.sbatch").read_text(encoding="utf-8")
        trap_offset = job.index("trap on_exit EXIT")
        phase_offset = job.index("phase wrapper_static_preflight START")
        for failure_token in (
            "V56_DELIVERY_SCHEMA_NOT_RELEASED",
            "V56_SOURCE_MANIFEST_CHANGED_AFTER_SUBMIT",
            "CURRENT_QUALIFIED_LOW_CHECKPOINT",
            "PARENT_V55_RELEASE_PIN_GATE_FAILED",
            "EXPORTED_PARENT_V55_BINDING_INVALID",
        ):
            self.assertLess(trap_offset, job.index(failure_token))
            self.assertLess(phase_offset, job.index(failure_token))
        # Missing scheduler exports must reach the trapped preflight instead of
        # aborting during shell parameter expansion before evidence exists.
        for name in (
            "V56_PARENT_V55_RUN_LABEL",
            "V56_PARENT_V55_RESULT_SHA256",
            "V56_PARENT_V55_TEACHER_FIXED_SUMMARY_SHA256",
            "V56_PARENT_V55_TEACHER_FIXED_ROWS_SHA256",
            "V56_PARENT_V55_TRANSACTION_STATUS_SHA256",
            "V56_PARENT_V55_TASK_MANIFEST_SHA256",
            "V56_BOUND_SOURCE_MANIFEST_SHA256",
        ):
            self.assertIn(f"${{{name}:-}}", job)
            self.assertNotIn(f"${{{name}:?}}", job)
        self.assertIn('os.environ.get("V56_PARENT_V55_RUN_LABEL", "")', job)

    def test_r1_removes_only_periodic_child_traceback_watchdog(self) -> None:
        for name in ("evaluate_task_progress_policy.py", "train_task_progress_canary.py"):
            child = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn("faulthandler.enable(all_threads=True)", child)
            self.assertNotIn("dump_traceback_later", child)
        job = (ROOT / "run_sparse_high_task_progress_canary.sbatch").read_text(encoding="utf-8")
        self.assertIn('run_isaac_child 35m "$LOGS/initial_evaluation.log"', job)
        self.assertIn('run_isaac_child 215m "$LOGS/training.log"', job)
        self.assertIn('run_isaac_child 35m "$LOGS/final_evaluation.log"', job)
        closure = json.loads((ROOT / "EVIDENCE_CLOSURE.json").read_text(encoding="utf-8"))
        repair = closure["engineering_repair_r1"]
        self.assertEqual(
            repair["failed_run"],
            "sparse_high_task_progress_canary_v1_17939578_20260918T071005373934975Z",
        )
        self.assertIn("child RC 139", repair["observed_fact"])

    def test_r2_closure_records_exact_failure_and_atomic_reset_boundary(self) -> None:
        closure = json.loads((ROOT / "EVIDENCE_CLOSURE.json").read_text(encoding="utf-8"))
        repair = closure["engineering_repair_r2"]
        self.assertEqual(repair["failed_job_id"], 17940941)
        self.assertEqual(
            repair["failed_run"],
            "sparse_high_task_progress_canary_v1_17940941_20260918T111911340609851Z",
        )
        self.assertIn("eight of twelve", repair["observed_fact"])
        self.assertIn("160 of 240", repair["observed_fact"])
        self.assertIn("ninth rollout", repair["observed_fact"])
        self.assertIn("public ContactSensor.data", repair["root_cause"])
        self.assertIn("previous-episode", repair["root_cause"])
        self.assertIn("live pre-reset world z", repair["ball_z_sibling_risk"])
        self.assertIn("same atomic reset repair", repair["ball_z_sibling_risk"])
        gate = repair["pre_training_probe_gate"]
        self.assertIn("before initial fixed evaluation", gate["position"])
        self.assertIn("actual nonzero robot-ball PhysX contact", gate["pass"])
        self.assertIn("before evaluation, fresh high PPO or checkpoint creation", gate["fail"])
        self.assertFalse(closure["long_training_authorized"])

    def test_r3_closure_records_fixture_failure_and_one_tick_causal_gate(self) -> None:
        closure = json.loads((ROOT / "EVIDENCE_CLOSURE.json").read_text(encoding="utf-8"))
        repair = closure["engineering_repair_r3"]
        self.assertEqual(repair["failed_job_id"], 17943801)
        self.assertEqual(
            repair["failed_run"],
            "sparse_high_task_progress_canary_v1_17943801_20260918T121821021175688Z",
        )
        self.assertIn("before any call to _reset_idx", repair["observed_fact"])
        self.assertIn("zero PPO optimizer steps", repair["observed_fact"])
        self.assertIn("calf-center positive-contact construction", repair["smallest_isolated_change"])
        gate = repair["pre_training_probe_gate"]
        self.assertIn("one and the same tick", gate["physical_contact_lane"])
        self.assertIn("exactly one raw physics tick", gate["one_tick_lane"])
        self.assertIn("cannot substitute", gate["sentinel_lane"])
        self.assertIn("PPO_OPTIMIZER_STEPS=0", gate["fail"])
        self.assertFalse(closure["long_training_authorized"])

    def test_r4_closure_records_live_order_failure_and_three_index_join(self) -> None:
        closure = json.loads((ROOT / "EVIDENCE_CLOSURE.json").read_text(encoding="utf-8"))
        repair = closure["engineering_repair_r4"]
        expected = [
            "base", "FL_hip", "FR_hip", "Head_upper", "RL_hip", "RR_hip",
            "FL_thigh", "FR_thigh", "Head_lower", "RL_thigh", "RR_thigh",
            "FL_calf", "FR_calf", "RL_calf", "RR_calf",
            "FL_foot", "FR_foot", "RL_foot", "RR_foot",
        ]
        expected_base = [
            "base", "FL_hip", "FL_thigh", "FL_calf", "FL_foot",
            "FR_hip", "FR_thigh", "FR_calf", "FR_foot",
            "Head_upper", "Head_lower",
            "RL_hip", "RL_thigh", "RL_calf", "RL_foot",
            "RR_hip", "RR_thigh", "RR_calf", "RR_foot",
        ]
        self.assertEqual(repair["failed_job_id"], 17944428)
        self.assertEqual(repair["elapsed_seconds_approx"], 41)
        self.assertEqual(repair["actual_accepted_body_names"], expected)
        self.assertEqual(repair["actual_live_body_names"], expected)
        self.assertEqual(repair["actual_base_contact_body_names"], expected_base)
        self.assertEqual(set(expected_base), set(expected))
        self.assertNotEqual(expected_base, expected)
        self.assertIn("contact-contract receipt", repair["observed_fact"])
        self.assertIn("No call to _reset_idx", repair["observed_fact"])
        self.assertIn("PPO optimizer steps remained zero", repair["observed_fact"])
        self.assertIn("75 static tests", repair["root_cause"])
        self.assertIn("three distinct live namespaces", repair["root_cause"])
        self.assertIn("robot_body_index", repair["falsifiable_hypothesis"])
        self.assertIn("ball_filter_index", repair["falsifiable_hypothesis"])
        self.assertIn("base_sensor_body_index", repair["falsifiable_hypothesis"])
        mapping = repair["pre_training_probe_gate"]["contract_mapping"]
        self.assertIn("duplicate-free permutation", repair["smallest_isolated_change"])
        self.assertIn("order is not required", mapping)
        self.assertIn("PPO_OPTIMIZER_STEPS=0", repair["pre_training_probe_gate"]["fail"])
        self.assertFalse(closure["long_training_authorized"])

    def test_standard_flow_runs_login_suite_once_and_optional_preflight_stops(self) -> None:
        submit = (ROOT / "submit_and_check_sparse_high_task_progress_canary.sh").read_text(encoding="utf-8")
        job = (ROOT / "run_sparse_high_task_progress_canary.sbatch").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertEqual(submit.count('-m unittest discover -v -s "$SOURCE/tests"'), 1)
        self.assertEqual(submit.count('"$SOURCE/tests/test_recurrent_hidden_guard.py"'), 1)
        self.assertIn("V56_SUBMITTER_FULL_STATIC_SUITE=PASS", submit)
        self.assertIn("V56_SUBMITTER_RSL_RL_SMOKE=PASS", submit)
        self.assertIn('V56_SUBMITTER_FULL_STATIC_SUITE="$V56_SUBMITTER_FULL_STATIC_SUITE"', submit)
        self.assertIn('V56_SUBMITTER_RSL_RL_SMOKE="$V56_SUBMITTER_RSL_RL_SMOKE"', submit)
        self.assertIn("V56_STANDARD_SUBMISSION_FLOW=ONE_FORMAL_COMMAND", submit)
        self.assertIn("V56_IN_JOB_FULL_STATIC_SUITE=NOT_REPEATED", submit)
        self.assertIn("V56_PREFLIGHT_MODE=OPTIONAL_STANDALONE_NOT_A_SUBMISSION_PREREQUISITE", submit)
        self.assertIn("V56_PREFLIGHT_ONLY_NEXT_ACTION=STOP_NO_IMMEDIATE_FORMAL_SUBMISSION", submit)
        self.assertNotIn('-m unittest discover -v -s "$SOURCE/tests"', job)
        self.assertIn("V56_SUBMITTER_FULL_STATIC_SUITE", job)
        self.assertIn("V56_SUBMITTER_RSL_RL_SMOKE", job)
        cluster_section = readme.split("## Cluster use", 1)[1]
        formal = "bash submit_and_check_sparse_high_task_progress_canary.sh"
        optional = "V56_PREFLIGHT_ONLY=1 bash submit_and_check_sparse_high_task_progress_canary.sh"
        self.assertLess(cluster_section.index(formal), cluster_section.index(optional))
        self.assertIn("It is not a prerequisite", cluster_section)
        self.assertIn("Do not run this optional command and then run the", cluster_section)


class ActionChainTests(unittest.TestCase):
    def test_cpu_replay_is_gross_guard_but_hull_and_rate_are_decisive(self) -> None:
        previous = [0.18, -0.04, 0.12]
        raw = [1.1, -0.7, 0.35]
        applied = checker.reconstruct_applied_command(raw, previous)
        self.assertTrue(checker.command_in_qualified_hull(applied))
        self.assertTrue(checker.command_respects_rate_limit(applied, previous))
        # A known-size CPU/CUDA near-tie discrepancy must remain admissible.
        drifted = [applied[0] + 3.0e-5, applied[1], applied[2]]
        self.assertTrue(checker.close_vector(drifted, applied, checker.ACTION_CHAIN_GROSS_TOLERANCE))
        self.assertTrue(checker.command_in_qualified_hull([0.1, 0.0, 0.0]))
        self.assertFalse(checker.command_respects_rate_limit([0.2, 0.0, 0.0], [0.0, 0.0, 0.0]))
        # Axis-wise legal does not imply membership in the qualified convex hull.
        self.assertFalse(checker.command_in_qualified_hull([0.4, 0.3, 0.0]))

    def test_rate_reference_is_reprojected_previous_not_raw_previous(self) -> None:
        previous = [0.19, 0.07, -0.09]
        projected = checker.project_command(previous)
        command = list(projected)
        self.assertTrue(checker.command_respects_rate_limit(command, previous))

    def test_full_25600_core_fixture_is_independently_reconstructed(self) -> None:
        fixture_path = ROOT / "tests/test_canary_protocol.py"
        spec = importlib.util.spec_from_file_location("v56_protocol_fixture", fixture_path)
        self.assertIsNotNone(spec)
        assert spec is not None and spec.loader is not None
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        import canary_protocol

        rows = fixture._raw_receipt()
        summary = canary_protocol.validate_first_rollout_transitions(rows)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "first_rollout_transitions.jsonl"
            payload = b"".join(checker.canonical_json_bytes(row) for row in rows)
            path.write_bytes(payload)
            summary.update({
                "transition_rows_relative_path": path.name,
                "transition_rows_bytes": len(payload),
                "transition_rows_sha256": hashlib.sha256(payload).hexdigest(),
                "optimizer_constructed_before_gate": True,
                "optimizer_state_empty_before_gate": True,
                "optimizer_steps_before_gate": 0,
                "initial_policy_state_sha256": "1" * 64,
                "initial_checkpoint_sha256": "2" * 64,
                "frozen_low_state_sha256_before_gate": checker.EXPECTED_LOW_MODULE_SHA256,
                "dense_return_authorizes_behavior": False,
                "long_training_authorized": False,
            })
            checks = canary_protocol.reward_support_checks(summary)
            signal_names = (
                "global_nonzero_support", "global_positive_support", "global_negative_support",
                "positive_support_goal_side_negative", "positive_support_goal_side_positive",
            )
            summary["gate_pass"] = all(checks[name] for name in signal_names)
            summary["checks"] = checks
            summary["blockers"] = []
            derived = checker.validate_transition_file(path, summary)
            self.assertEqual(derived["transition_row_count"], 25_600)
            self.assertEqual(derived["positive_progress_count"], 2)
            self.assertEqual(derived["negative_progress_count"], 1)

            original = rows[0]
            rows[0] = dict(original, applied_command=[0.4, 0.3, 0.0])
            path.write_bytes(b"".join(checker.canonical_json_bytes(row) for row in rows))
            with self.assertRaises(checker.ProtocolError):
                checker.validate_transition_file(path, summary)
            rows[0] = dict(original, applied_command=[0.2, 0.0, 0.0])
            path.write_bytes(b"".join(checker.canonical_json_bytes(row) for row in rows))
            with self.assertRaises(checker.ProtocolError):
                checker.validate_transition_file(path, summary)

    def test_reset_serial_and_support_threshold_mutations_are_rejected(self) -> None:
        fixture_path = ROOT / "tests/test_canary_protocol.py"
        spec = importlib.util.spec_from_file_location("v56_protocol_mutation_fixture", fixture_path)
        assert spec is not None and spec.loader is not None
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)

        rows = fixture._early_terminal_receipt()
        checker.validate_first_rollout_transitions(rows)
        following = 128  # high-step 1, env 0 follows the early done at row 0.
        mutated = list(rows)
        mutated[following] = dict(mutated[following])
        mutated[following]["sample_serial"] += 1
        mutated[following]["episode_serial_before_step"] += 1
        with self.assertRaises(checker.ProtocolError):
            checker.validate_first_rollout_transitions(mutated)

        straddled = fixture._raw_receipt(straddle_side=-1)
        derived = checker.validate_first_rollout_transitions(straddled)
        self.assertEqual(derived["positive_progress_count_by_goal_side"]["-1"], 0)
        self.assertFalse(derived["checks"]["positive_support_goal_side_negative"])


class ParentAndManifestTests(unittest.TestCase):
    def test_exact_parent_release_contract_passes_and_byte_drift_fails(self) -> None:
        contract = ROOT / "SPARSE_HIGH_TASK_PROGRESS_CANARY_V1_CONTRACT.json"
        self.assertEqual(release_pins(contract)["run_label"], checker.EXPECTED_PARENT_V55["run_label"])
        with tempfile.TemporaryDirectory() as temporary:
            drifted = Path(temporary) / contract.name
            value = json.loads(contract.read_text(encoding="utf-8"))
            value["parent_v55"]["result_sha256"] = "0" * 64
            drifted.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaises(ParentError):
                release_pins(drifted)

    def test_source_manifest_reconstructs_exact_membership_and_rejects_extra(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "56_sparse_high_task_progress_canary_v1"
            root.mkdir()
            payload_path = root / "payload.txt"
            payload_path.write_bytes(b"payload\n")
            manifest = root / "SPARSE_HIGH_TASK_PROGRESS_CANARY_V1_SOURCE.sha256"
            manifest.write_text(f"{hashlib.sha256(payload_path.read_bytes()).hexdigest()}  payload.txt\n", encoding="ascii")
            receipt = parent / f"{root.name}.source-manifest.sha256"
            receipt.write_text(f"{hashlib.sha256(manifest.read_bytes()).hexdigest()}  {manifest.name}\n", encoding="ascii")
            report = validate_source_manifest(root, manifest, receipt)
            self.assertEqual(report["status"], "PASS")
            (root / "extra.txt").write_text("extra", encoding="utf-8")
            with self.assertRaises(SourceManifestError):
                validate_source_manifest(root, manifest, receipt)


class CheckerConstantsTests(unittest.TestCase):
    def test_frozen_core_hashes_and_parent_pins_are_exact(self) -> None:
        expected = {
            "SPARSE_HIGH_TASK_PROGRESS_CANARY_V1_CONTRACT.json": "3b96dbfffcfee02c24f50eba1e8ae1884fdbd3f020dc1b85acb950c0c9379289",
            "EVIDENCE_CLOSURE.json": "40d81a0084ea564a54ade38f7a6ace08c368fec94011f9c073361757b11bcbb9",
            "canary_protocol.py": "4b9e3f1ea30981c5e751aef57635b7ca12eca004593e0d93927fed66583e0388",
            "task_progress_reward.py": "6f8a36adb8db70e9f0ee6260e919ba216fcadfb1d213a9ac5f584ec16e9e33cf",
            "current_task_progress_env.py": "66f47b233dade10719ab54b82ec4871c5d26674adfbf1a97fec536b3d9ef398d",
            "runtime_common.py": "a7223c16db1825a38bdd7b3c9e232d5a1b3670048401e7bbb5ae609d3073f960",
            "train_task_progress_canary.py": "d0f41b598d8248601f8f6b5ba899f9feb7f4889ec0faf185db271ea49d65535b",
            "evaluate_task_progress_policy.py": "bdbb2a8dae4ef83094de45d6fdfeb86f3eafd1a7e72c9e554421c234b170f571",
            "finalize_task_progress_canary.py": "1a3095e3afd7124ca584e7fe6265c117e9a5202d30135e7afddae475f0ce9490",
        }
        for relative, digest in expected.items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), digest)

    def test_checkpoint_state_inequality_guard_is_present(self) -> None:
        source = (ROOT / "check_sparse_high_task_progress_canary.py").read_text(encoding="utf-8")
        self.assertIn("initial_policy_sha != first_policy_sha", source)
        self.assertIn("first_policy_sha != final_policy_sha", source)


if __name__ == "__main__":
    unittest.main()
