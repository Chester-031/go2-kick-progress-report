#!/usr/bin/env python3
"""Fail-closed validation of the exact V55 OWN parent evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any


CONTRACT_SHA256 = "3b96dbfffcfee02c24f50eba1e8ae1884fdbd3f020dc1b85acb950c0c9379289"
RELEASE_STATUS = "RELEASE_READY_EXACT_PARENT_V55_OWN_ARTIFACTS_PINNED"
RUN_LABEL = "sparse_high_offline_bc_warm_start_v1_17938567_20260918T045821635312091Z"
DECISION = "V55_GEOMETRIC_TEACHER_FIXED_GATE_REJECTED_STOP_PRE_BC"
PINS = {
    "result_sha256": "30b1369ff05a084733d0278e7fb31dbae0261a243f9a678b3d595e8fe3f7b4e3",
    "teacher_fixed_summary_sha256": "00b54c399abb05cd34fc7f86a0f5caf62dc4dc3b6eb238aa80d8e0ca07444356",
    "teacher_fixed_rows_sha256": "248d7e158becb03ec8480cf687d5b9b368380b5c20bca86abfb4ae25fc15a496",
    "transaction_status_sha256": "09a842bc23895489113f104ec63b640087115619583a7cc08f1eea24680ebfcc",
    "task_manifest_sha256": "b3af08bf243fa793257ebf625012ef05a43eb38df8e533293be68dfda9b36e3f",
}
ARTIFACTS = {
    "result_sha256": "evidence/result/result.json",
    "teacher_fixed_summary_sha256": "evidence/teacher_fixed/teacher_evaluation_summary.json",
    "teacher_fixed_rows_sha256": "evidence/teacher_fixed/teacher_fixed_episode_results.jsonl",
    "transaction_status_sha256": "evidence/transaction_status.json",
    "task_manifest_sha256": "source/TASK_PACKAGE_RUNTIME.sha256",
}


class ParentError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ParentError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def duplicate_guard(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def nonfinite(value: str) -> None:
    raise ParentError(f"non-finite JSON constant: {value}")


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=duplicate_guard, parse_constant=nonfinite)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ParentError(f"{label} unreadable: {error}") from error
    require(isinstance(value, dict), f"{label} is not an object")
    return value


def regular_under(root: Path, relative: str) -> Path:
    candidate = root / relative
    metadata = candidate.lstat()
    require(stat.S_ISREG(metadata.st_mode) and not candidate.is_symlink(), f"unsafe parent artifact: {relative}")
    require(metadata.st_nlink == 1, f"parent artifact has multiple hard links: {relative}")
    resolved = candidate.resolve(strict=True)
    require(resolved.is_relative_to(root), f"parent artifact escaped run: {relative}")
    return resolved


def release_pins(contract_path: Path) -> dict[str, Any]:
    contract = load_json(contract_path.resolve(strict=True), "V56 contract")
    require(sha256_file(contract_path.resolve(strict=True)) == CONTRACT_SHA256, "V56 release contract bytes changed")
    require(contract.get("status") == RELEASE_STATUS, "V56 contract is not release-ready")
    require(contract.get("decision") == "AUTHORIZE_EXACTLY_ONE_FRESH_HIGH_TASK_PROGRESS_BOUNDED_CANARY", "V56 release decision changed")
    parent = contract.get("parent_v55")
    require(isinstance(parent, dict), "V56 parent_v55 block missing")
    expected = {
        "required_decision": DECISION,
        "required_checker_rc": 0,
        "required_engineering_integrity": "PASS",
        "exact_own_artifact_pin_required_before_gpu": True,
        "run_label": RUN_LABEL,
        **PINS,
    }
    require(parent == expected, "V55 exact OWN parent block differs")
    return parent


def validate(contract_path: Path, run_path: Path | None) -> dict[str, Any]:
    parent = release_pins(contract_path)
    if run_path is None:
        return parent
    metadata = run_path.lstat()
    require(stat.S_ISDIR(metadata.st_mode) and not run_path.is_symlink(), "V55 run is not a real directory")
    run = run_path.resolve(strict=True)
    require(run.name == RUN_LABEL, "V55 run label differs")
    paths: dict[str, Path] = {}
    for field, relative in ARTIFACTS.items():
        path = regular_under(run, relative)
        require(sha256_file(path) == PINS[field], f"V55 OWN {field} differs")
        paths[field] = path

    result = load_json(paths["result_sha256"], "V55 result")
    require(result.get("result_schema") == "sparse_high_offline_bc_warm_start_v1/result/v1", "V55 result schema changed")
    require(result.get("status") == "COMPLETE" and result.get("failures") == [], "V55 result incomplete")
    require(result.get("engineering_integrity_pass") is True and result.get("decision") == DECISION, "V55 result decision/integrity changed")
    expected_stage = {
        "teacher_fixed_gate_executed": True,
        "teacher_fixed_gate_pass": False,
        "teacher_dataset_executed": False,
        "offline_bc_executed": False,
        "bc_fixed_evaluation_executed": False,
        "first_sparse_rollout_executed": False,
        "ppo_training_executed": False,
        "bounded_continuation_executed": False,
        "fixed_final_evaluation_executed": False,
        "ppo_optimizer_steps": 0,
        "long_training_authorized": False,
    }
    require(all(result.get(key) == value for key, value in expected_stage.items()), "V55 stopped-branch facts changed")
    summary = load_json(paths["teacher_fixed_summary_sha256"], "V55 teacher-fixed summary")
    require(summary.get("schema") == "sparse_high_offline_bc_warm_start_v1/teacher_evaluation/v1", "V55 teacher schema changed")
    require(summary.get("status") == "COMPLETE" and summary.get("phase") == "fixed_gate" and summary.get("failures") == [], "V55 teacher summary incomplete")
    metrics = summary.get("metrics", {})
    require(summary.get("behavioral_gate_pass") is False and summary.get("ppo_optimizer_steps") == 0, "V55 teacher decision changed")
    require(metrics.get("case_count") == 288 and metrics.get("hard_case_count") == 36, "V55 teacher denominators changed")
    require(metrics.get("contact_count") == 91 and metrics.get("hard_contact_count") == 9, "V55 teacher contact facts changed")
    require(metrics.get("goal_count") == 0 and metrics.get("unsafe_count") == 1, "V55 teacher goal/safety facts changed")
    require(summary.get("episode_rows_sha256") == PINS["teacher_fixed_rows_sha256"], "V55 summary does not bind OWN rows")
    require(result.get("teacher_fixed") == summary, "V55 result embedded teacher summary differs")
    require(result.get("input_summary_sha256") == {"teacher_fixed": PINS["teacher_fixed_summary_sha256"]}, "V55 result input hash mapping changed")
    require(all(result.get(name) is None for name in ("teacher_dataset", "offline_bc", "bc_initial_evaluation", "first_sparse_rollout", "training", "final_evaluation")), "V55 rejected branch contains downstream evidence")

    transaction = load_json(paths["transaction_status_sha256"], "V55 transaction")
    require(transaction.get("schema") == "sparse_high_offline_bc_warm_start_v1/transaction_status/v1", "V55 transaction schema changed")
    require(transaction.get("status") == "COMPLETE" and transaction.get("process_rc") == 0, "V55 transaction incomplete")
    require(transaction.get("decision") == DECISION and transaction.get("ppo_optimizer_steps") == 0, "V55 transaction decision changed")
    require(transaction.get("long_training_authorized") is False, "V55 transaction authorized long training")
    return parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pins-only", action="store_true")
    group.add_argument("--run", type=Path)
    args = parser.parse_args()
    try:
        parent = validate(args.contract, None if args.pins_only else args.run)
    except (OSError, ParentError, ValueError, TypeError) as error:
        print(f"V56_PARENT_V55_GATE=FAIL {error}", file=os.sys.stderr)
        return 2
    print("V56_PARENT_V55_GATE=PASS")
    print(f"V56_PARENT_V55_RUN_LABEL={parent['run_label']}")
    for field in PINS:
        print(f"V56_PARENT_V55_{field.upper()}={parent[field]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
