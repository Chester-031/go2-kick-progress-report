#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys


def fail(code: str, detail: str, errors: list[dict[str, str]]) -> None:
    errors.append({"code": code, "detail": detail})


def main() -> int:
    package = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent)
    train = package / "train_go2_soccer_v4k1_r5_stable_kick.py"
    play = package / "play_go2_soccer_v4k1_r5_stable_kick.py"
    installer = package / "install_submit_v4k1_r5_stable_kick.sh"
    worker = package / "run_go2_soccer_v4k1_r5_stable_kick.sbatch"
    base_validator = package / "validate_v4k1_r5_package.py"

    errors: list[dict[str, str]] = []
    ok: list[str] = []

    result = subprocess.run(
        [sys.executable, str(base_validator), str(package)],
        capture_output=True,
        text=True,
        check=False,
    )
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0 or "V4K1_R5_PACKAGE_AUDIT_CLEAN" not in result.stdout:
        fail("BASE_VALIDATOR_FAILED", f"rc={result.returncode}", errors)
    else:
        ok.append("base_validator_clean")

    for path in (train, play, installer, worker):
        if not path.is_file():
            fail("MISSING_FILE", str(path), errors)

    if errors:
        print(json.dumps({"errors": errors, "ok": ok}, indent=2))
        print("V4K1_R5_3_COLLECT_ALL_AUDIT_FAILED")
        return 1

    train_text = train.read_text(encoding="utf-8")
    play_text = play.read_text(encoding="utf-8")
    installer_text = installer.read_text(encoding="utf-8")
    worker_text = worker.read_text(encoding="utf-8")

    if hashlib.sha256(train.read_bytes()).hexdigest() != hashlib.sha256(play.read_bytes()).hexdigest():
        fail("TRAIN_PLAY_MISMATCH", "SHA256 differs", errors)
    else:
        ok.append("train_play_identical")

    try:
        tree = ast.parse(train_text, filename=str(train))
    except SyntaxError as exc:
        fail("TRAIN_AST", f"{exc.msg}:{exc.lineno}", errors)
        tree = ast.Module(body=[], type_ignores=[])

    kick_method = None
    training_fn = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "IsolatedDeltaPoseKickEnv":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "kick_step":
                    kick_method = item
        if isinstance(node, ast.FunctionDef) and node.name == "run_v4k1_training":
            training_fn = node

    if kick_method is None:
        fail("KICK_STEP_MISSING", "IsolatedDeltaPoseKickEnv.kick_step", errors)
    else:
        if len(kick_method.decorator_list) != 1:
            fail(
                "KICK_STEP_DECORATOR_COUNT",
                f"expected=1 observed={len(kick_method.decorator_list)}",
                errors,
            )
        else:
            ok.append("kick_step_single_decorator")

    required_tokens = (
        "V4K1_R5_3_CAUSAL_DISCOVERY_REPAIR",
        "before_ball_speed=before_ball_speed",
        "wrong_foot_contact = wrong_foot_proximity & ball_response_now",
        "body_ball_contact = body_ball_proximity & ball_response_now",
        "harmless near-contact",
        "standing_positive_baseline_removed_in_stage1=true",
        "episode_foot_lift_rate",
        "episode_foot_swing_rate",
        "episode_motion_ready_rate",
        "episode_geometric_contact_rate",
        "episode_passive_response_rate",
        "episode_precontact_motion_rate",
        "episode_invalid_contact_rate",
        "episode_body_contact_rate",
        "episode_wrong_foot_contact_rate",
        "V4K1_STAGE1_MIN_LIFT_RATE",
        "V4K1_STAGE1_MIN_SWING_RATE",
    )
    for token in required_tokens:
        if token not in train_text:
            fail("TRAIN_TOKEN_MISSING", token, errors)
    if not errors:
        ok.append("causal_reason_metrics_present")

    forbidden_tokens = (
        "del before_state",
        "self.contact_duration_steps += geometric_contact.long()",
        'final_stable_activation = mean_available("stable_activation_fraction")',
        'final_violation = mean_available("violation_fraction")',
        "@torch.inference_mode(False)\n    @torch.inference_mode(False)",
    )
    for token in forbidden_tokens:
        if token in train_text:
            fail("FORBIDDEN_TOKEN", token, errors)

    # Causal source ordering.
    order_tokens = (
        "before_state = self._state()",
        "self.low_env.step",
        "after_state = self._state()",
        "before_ball_speed =",
        "ball_response_now =",
        "precontact_ball_motion = precontact_ball_motion_mask(",
        "valid_contact = valid_kick_contact_mask(",
        "invalid_contact, passive_selected_response = invalid_kick_contact_mask(",
    )
    positions = [train_text.find(token) for token in order_tokens]
    if any(pos < 0 for pos in positions) or positions != sorted(positions):
        fail("CAUSAL_SOURCE_ORDER", repr(list(zip(order_tokens, positions))), errors)
    else:
        ok.append("causal_source_order")

    # Ensure stage-1 reward does not contain the old positive standing baseline.
    start = train_text.find('if self.curriculum_stage == "contact_discovery":', train_text.find("def _kick_reward"))
    end = train_text.find("else:", start)
    stage1_block = train_text[start:end]
    for old_positive in ("pose_track", "yaw_track", '+ 0.06 * stability["stable"].float()'):
        if old_positive in stage1_block:
            fail("STAGE1_STANDING_BASELINE_PRESENT", old_positive, errors)
    if "new_lift.float()" not in stage1_block or "new_swing.float()" not in stage1_block:
        fail("STAGE1_MOTION_REWARD_MISSING", "new_lift/new_swing", errors)
    else:
        ok.append("stage1_motion_only_positive_signal")

    # Shell syntax.
    for path in package.glob("*.sh"):
        proc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        if proc.returncode:
            fail("SHELL_SYNTAX", f"{path.name}:{proc.stderr}", errors)
    for path in package.glob("*.sbatch"):
        proc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        if proc.returncode:
            fail("SBATCH_SYNTAX", f"{path.name}:{proc.stderr}", errors)
    if not any(e["code"] in {"SHELL_SYNTAX", "SBATCH_SYNTAX"} for e in errors):
        ok.append("shell_syntax_clean")

    # Parse submit_job calls and verify 22 arguments and schedules.
    normalized = installer_text.replace("\\\n", " ")
    expected = {
        "PRE_JOB": {"envs": "512", "iterations": "3", "s1": "1", "s2": "1", "s3": "1"},
        "PILOT_JOB": {"envs": "128", "iterations": "300", "s1": "200", "s2": "70", "s3": "30"},
        "FULL_JOB": {"envs": "512", "iterations": "900", "s1": "300", "s2": "400", "s3": "200"},
    }
    for variable, contract in expected.items():
        match = re.search(
            rf'{variable}="\$\(\s*submit_job\s+(.*?)\s*\)"',
            normalized,
            flags=re.S,
        )
        if not match:
            fail("SUBMIT_CALL_MISSING", variable, errors)
            continue
        try:
            args = shlex.split(match.group(1))
        except ValueError as exc:
            fail("SUBMIT_CALL_PARSE", f"{variable}:{exc}", errors)
            continue
        if len(args) != 22:
            fail("SUBMIT_ARITY", f"{variable}:expected=22 observed={len(args)} args={args}", errors)
            continue
        values = {
            "envs": args[4],
            "iterations": args[5],
            "s1": args[6],
            "s2": args[7],
            "s3": args[8],
        }
        for key, expected_value in contract.items():
            if values[key] != expected_value:
                fail("SUBMIT_CONTRACT", f"{variable}.{key}:expected={expected_value} observed={values[key]}", errors)
        try:
            if int(values["s1"]) + int(values["s2"]) + int(values["s3"]) != int(values["iterations"]):
                fail("SCHEDULE_SUM", f"{variable}:{values}", errors)
        except ValueError:
            fail("SCHEDULE_NUMERIC", f"{variable}:{values}", errors)
    if not any(e["code"].startswith(("SUBMIT_", "SCHEDULE_")) for e in errors):
        ok.append("submit_arity_and_schedule_clean")

    required_installer = (
        "PRE_ROLLOUT_STEPS=240",
        "PRE_TEMPORAL_MIN_STEPS=$((150 + 6 + 60 + 12))",
        "V4K1_R5_3_TEMPORAL_PREFLIGHT_OK",
        "pilot=128_envs_300_iters_stages_200_70_30",
        "full=512_envs_900_iters_stages_300_400_200",
        "V4K1_STAGE1_MIN_LIFT_RATE",
        "V4K1_STAGE1_MIN_SWING_RATE",
        "V4K1_PRECONTACT_SPEED=0.030",
        "V4K1_TOWARD_BALL_SPEED=0.060",
    )
    for token in required_installer:
        if token not in installer_text:
            fail("INSTALLER_TOKEN_MISSING", token, errors)

    required_worker = (
        "precontact_motion_uses_before_action_state=true",
        "first_valid_strike_ball_response_allowed=true",
        "wrong_foot_and_body_proximity_require_ball_response=true",
        "reason_decomposed_episode_metrics=true",
        'touch "${RUN_DIR}/V4K1_ISOLATED_KICK_FAILED"',
    )
    for token in required_worker:
        if token not in worker_text:
            fail("WORKER_TOKEN_MISSING", token, errors)

    payload = {
        "package": str(package),
        "errors": errors,
        "ok": ok,
        "counts": {"errors": len(errors), "warnings": 0, "ok": len(ok)},
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(
        "V4K1_R5_3_COLLECT_ALL_AUDIT_COUNTS "
        f"errors={len(errors)} warnings=0 ok={len(ok)}"
    )
    if errors:
        print("V4K1_R5_3_COLLECT_ALL_AUDIT_FAILED")
        return 1
    print("V4K1_R5_3_COLLECT_ALL_AUDIT_CLEAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
