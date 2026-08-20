#!/usr/bin/env python3
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any
import ast
import builtins
import hashlib
import json
import subprocess
import symtable
import sys
import traceback

import torch


class Audit:
    def __init__(self) -> None:
        self.errors: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []
        self.ok: list[str] = []

    def error(self, code: str, message: str, path: str = "", line: int | None = None) -> None:
        self.errors.append({"code": code, "path": path, "line": line, "message": message})

    def warning(self, code: str, message: str, path: str = "", line: int | None = None) -> None:
        self.warnings.append({"code": code, "path": path, "line": line, "message": message})

    def passed(self, message: str) -> None:
        self.ok.append(message)


def function_node(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    return next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ),
        None,
    )


def class_method_node(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef | None:
    cls = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ),
        None,
    )
    if cls is None:
        return None
    return next(
        (
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == method_name
        ),
        None,
    )


def source_segment(text: str, node: ast.AST | None) -> str:
    if node is None:
        return ""
    return ast.get_source_segment(text, node) or ""


def check_unresolved_globals(text: str, filename: str, audit: Audit) -> None:
    try:
        table = symtable.symtable(text, filename, "exec")
    except Exception as exc:
        audit.error("SYMTABLE", repr(exc), filename)
        return
    module_names = set(table.get_identifiers())
    builtin_names = set(dir(builtins))
    issues: list[tuple[str, str]] = []

    def walk(scope: symtable.SymbolTable, stack: list[str]) -> None:
        for child in scope.get_children():
            child_stack = [*stack, child.get_name()]
            for name in child.get_identifiers():
                symbol = child.lookup(name)
                if (
                    symbol.is_referenced()
                    and symbol.is_global()
                    and name not in module_names
                    and name not in builtin_names
                ):
                    issues.append((".".join(child_stack), name))
            walk(child, child_stack)

    walk(table, [])
    if issues:
        for scope, name in issues:
            audit.error("UNRESOLVED_GLOBAL", f"{scope}:{name}", filename)
    else:
        audit.passed(f"unresolved_globals_clean:{filename}")


def extract_dict_keys(node: ast.AST, variable_name: str) -> set[str]:
    keys: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == variable_name for target in child.targets):
            continue
        if not isinstance(child.value, ast.Dict):
            continue
        for key in child.value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.add(key.value)
    return keys


def exec_functions(text: str, tree: ast.Module, names: tuple[str, ...], namespace: dict[str, Any]) -> dict[str, Any]:
    chunks: list[str] = []
    for name in names:
        node = function_node(tree, name)
        if node is None:
            raise RuntimeError(f"missing function {name}")
        chunks.append(source_segment(text, node))
    exec("\n\n".join(chunks), namespace)
    return namespace


def pure_predicate_tests(text: str, tree: ast.Module, audit: Audit) -> None:
    namespace: dict[str, Any] = {"torch": torch, "os": __import__("os")}
    try:
        exec_functions(
            text,
            tree,
            (
                "stable_stance_mask",
                "precontact_ball_motion_mask",
                "valid_kick_contact_mask",
                "invalid_kick_contact_mask",
                "next_valid_contact_duration",
                "stable_kick_success_mask",
            ),
            namespace,
        )
        stable = namespace["stable_stance_mask"]
        precontact = namespace["precontact_ball_motion_mask"]
        valid = namespace["valid_kick_contact_mask"]
        invalid = namespace["invalid_kick_contact_mask"]
        duration = namespace["next_valid_contact_duration"]
        success = namespace["stable_kick_success_mask"]
        true = torch.tensor([True])
        false = torch.tensor([False])

        stable_ok = stable(
            vertical_speed=torch.tensor([0.01]),
            horizontal_speed=torch.tensor([0.03]),
            angular_xy_speed=torch.tensor([0.04]),
            upright_error=torch.tensor([0.002]),
            base_height=torch.tensor([0.32]),
            support_count=torch.tensor([4]),
        )
        falling = stable(
            vertical_speed=torch.tensor([0.40]),
            horizontal_speed=torch.tensor([0.03]),
            angular_xy_speed=torch.tensor([0.04]),
            upright_error=torch.tensor([0.002]),
            base_height=torch.tensor([0.32]),
            support_count=torch.tensor([4]),
        )
        if not torch.equal(stable_ok, true):
            audit.error("PREDICATE_STABLE_TRUE", str(stable_ok.tolist()))
        if not torch.equal(falling, false):
            audit.error("PREDICATE_FALLING_REJECT", str(falling.tolist()))

        static_before = precontact(
            active=true,
            contact_seen=false,
            before_ball_speed=torch.tensor([0.0]),
            before_ball_displacement=torch.tensor([0.0]),
        )
        moving_before = precontact(
            active=true,
            contact_seen=false,
            before_ball_speed=torch.tensor([0.08]),
            before_ball_displacement=torch.tensor([0.0]),
        )
        if not torch.equal(static_before, false):
            audit.error("CAUSAL_STATIC_BEFORE_ALLOWED", str(static_before.tolist()))
        if not torch.equal(moving_before, true):
            audit.error("CAUSAL_PREEXISTING_MOTION_REJECT", str(moving_before.tolist()))

        valid_first_strike = valid(
            active=true,
            geometric_contact=true,
            stable_at_contact=true,
            foot_lift_seen=true,
            foot_swing_seen=true,
            deliberate_approach=true,
            wrong_foot_contact=false,
            body_ball_contact=false,
            precontact_ball_motion=static_before,
        )
        no_approach = valid(
            active=true,
            geometric_contact=true,
            stable_at_contact=true,
            foot_lift_seen=true,
            foot_swing_seen=true,
            deliberate_approach=false,
            wrong_foot_contact=false,
            body_ball_contact=false,
            precontact_ball_motion=false,
        )
        if not torch.equal(valid_first_strike, true):
            audit.error("CAUSAL_FIRST_STRIKE_ACCEPT", str(valid_first_strike.tolist()))
        if not torch.equal(no_approach, false):
            audit.error("CAUSAL_PASSIVE_PROXIMITY_REJECT", str(no_approach.tolist()))

        harmless_near, passive_near = invalid(
            geometric_contact=true,
            valid_contact=false,
            ball_response_now=false,
            wrong_foot_contact=false,
            body_ball_contact=false,
            precontact_ball_motion=false,
        )
        passive_invalid, passive_response = invalid(
            geometric_contact=true,
            valid_contact=false,
            ball_response_now=true,
            wrong_foot_contact=false,
            body_ball_contact=false,
            precontact_ball_motion=false,
        )
        if not torch.equal(harmless_near, false) or not torch.equal(passive_near, false):
            audit.error("DISCOVERY_NEAR_CONTACT_NOT_PERMANENT", f"{harmless_near.tolist()} {passive_near.tolist()}")
        if not torch.equal(passive_invalid, true) or not torch.equal(passive_response, true):
            audit.error("PASSIVE_BALL_RESPONSE_REJECT", f"{passive_invalid.tolist()} {passive_response.tolist()}")

        duration_before_valid = duration(
            previous_duration=torch.tensor([0]),
            contact_seen_after_update=false,
            geometric_contact=true,
        )
        duration_first_valid = duration(
            previous_duration=torch.tensor([0]),
            contact_seen_after_update=true,
            geometric_contact=true,
        )
        if int(duration_before_valid.item()) != 0:
            audit.error("CONTACT_DURATION_PREVALID_ZERO", str(duration_before_valid.tolist()))
        if int(duration_first_valid.item()) != 1:
            audit.error("CONTACT_DURATION_FIRST_VALID_ONE", str(duration_first_valid.tolist()))

        success_ok = success(
            valid_contact_seen=true,
            impact_seen=true,
            peak_direction_speed=torch.tensor([0.45]),
            directional_displacement=torch.tensor([0.30]),
            contact_duration_steps=torch.tensor([5]),
            violation_seen=false,
            base_displacement=torch.tensor([0.04]),
            recovery_seen=true,
        )
        push_fail = success(
            valid_contact_seen=true,
            impact_seen=true,
            peak_direction_speed=torch.tensor([0.45]),
            directional_displacement=torch.tensor([0.30]),
            contact_duration_steps=torch.tensor([20]),
            violation_seen=false,
            base_displacement=torch.tensor([0.04]),
            recovery_seen=true,
        )
        if not torch.equal(success_ok, true):
            audit.error("PREDICATE_SUCCESS_TRUE", str(success_ok.tolist()))
        if not torch.equal(push_fail, false):
            audit.error("PREDICATE_PUSH_REJECT", str(push_fail.tolist()))
        if not any(e["code"].startswith(("PREDICATE", "CAUSAL", "DISCOVERY", "PASSIVE", "CONTACT_DURATION")) for e in audit.errors):
            audit.passed("pure_causal_predicate_tests")
    except Exception as exc:
        audit.error("PURE_PREDICATE_EXCEPTION", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")



def warm_start_tests(text: str, tree: ast.Module, audit: Audit) -> None:
    node = function_node(tree, "warm_start_v4k1_policy")
    if node is None:
        audit.error("WARM_START_MISSING", "warm_start_v4k1_policy")
        return
    source = source_segment(text, node)

    class KickEnv:
        KICK_OBS_DIM = 70
        NATIVE_COMMAND_SLICE = slice(9, 12)

    class Policy(torch.nn.Module):
        def __init__(self, obs: int) -> None:
            super().__init__()
            self.actor = torch.nn.Sequential(
                torch.nn.Linear(obs, 512), torch.nn.ELU(),
                torch.nn.Linear(512, 256), torch.nn.ELU(),
                torch.nn.Linear(256, 128), torch.nn.ELU(),
                torch.nn.Linear(128, 12),
            )
            self.critic = torch.nn.Sequential(
                torch.nn.Linear(obs, 512), torch.nn.ELU(),
                torch.nn.Linear(512, 256), torch.nn.ELU(),
                torch.nn.Linear(256, 128), torch.nn.ELU(),
                torch.nn.Linear(128, 1),
            )

    class Runner:
        def __init__(self, module: Policy) -> None:
            self.module = module
            self.current_learning_iteration = -1

    std_calls: list[float] = []
    namespace: dict[str, Any] = {
        "Any": Any,
        "Path": Path,
        "torch": torch,
        "IsolatedDeltaPoseKickEnv": KickEnv,
        "policy_module": lambda runner: runner.module,
        "load_state_dictionary": lambda checkpoint: checkpoint,
        "set_action_std": lambda module, value: std_calls.append(float(value)) or True,
    }
    try:
        exec(source, namespace)
        warm = namespace["warm_start_v4k1_policy"]
        source62 = Policy(62).state_dict()
        target_boot = Policy(70)
        runner_boot = Runner(target_boot)
        with redirect_stdout(StringIO()):
            warm(runner=runner_boot, checkpoint=source62, requested_std=0.35)
        loaded = target_boot.state_dict()
        for prefix in ("actor", "critic"):
            key = f"{prefix}.0.weight"
            if not torch.equal(loaded[key][:, :9], source62[key][:, :9]):
                audit.error("BOOTSTRAP_MAP_0_9", key)
            if torch.count_nonzero(loaded[key][:, 9:12]):
                audit.error("BOOTSTRAP_SCRUB_9_12", key)
            if not torch.equal(loaded[key][:, 12:56], source62[key][:, 12:56]):
                audit.error("BOOTSTRAP_MAP_12_56", key)
            if not torch.equal(loaded[key][:, 56:59], source62[key][:, 56:59]):
                audit.error("BOOTSTRAP_MAP_56_59", key)
            if torch.count_nonzero(loaded[key][:, 59]):
                audit.error("BOOTSTRAP_ZERO_KICK_ONEHOT", key)
            if not torch.equal(loaded[key][:, 60:62], source62[key][:, 59:61]):
                audit.error("BOOTSTRAP_MAP_BALL_CONTEXT", key)
            if torch.count_nonzero(loaded[key][:, 62:70]):
                audit.error("BOOTSTRAP_ZERO_NEW_CONTEXT", key)

        source70 = Policy(70).state_dict()
        target_resume = Policy(70)
        runner_resume = Runner(target_resume)
        with redirect_stdout(StringIO()):
            warm(runner=runner_resume, checkpoint=source70, requested_std=0.20)
        resumed = target_resume.state_dict()
        for key, expected in source70.items():
            if not torch.equal(resumed[key], expected):
                audit.error("RESUME_EXACT_COPY", key)
        if std_calls != [0.35, 0.20]:
            audit.error("STD_RESET_CALLS", f"observed={std_calls}")
        if runner_boot.current_learning_iteration != 0 or runner_resume.current_learning_iteration != 0:
            audit.error("WARM_START_ITERATION", "runner iteration not reset")
        if not any(error["code"].startswith(("BOOTSTRAP", "RESUME", "STD", "WARM")) for error in audit.errors):
            audit.passed("warm_start_62_to_70_and_70_to_70")
    except Exception as exc:
        audit.error("WARM_START_EXCEPTION", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")


def main() -> int:
    package = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
    audit = Audit()
    train = package / "train_go2_soccer_v4k1_r5_stable_kick.py"
    play = package / "play_go2_soccer_v4k1_r5_stable_kick.py"
    worker = package / "run_go2_soccer_v4k1_r5_stable_kick.sbatch"
    installer = package / "install_submit_v4k1_r5_stable_kick.sh"

    required_paths = [train, play, worker, installer, package / "README.md", package / "V4K1_DESIGN.md"]
    for path in required_paths:
        if not path.is_file() or path.stat().st_size == 0:
            audit.error("MISSING_FILE", "required file missing or empty", str(path))
        else:
            audit.passed(f"file:{path.name}")

    python_files = sorted(package.glob("*.py"))
    shell_files = sorted([*package.glob("*.sh"), *package.glob("*.sbatch")])
    for path in python_files:
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
            audit.passed(f"python_compile:{path.name}")
        except SyntaxError as exc:
            audit.error("PYTHON_SYNTAX", exc.msg, str(path), exc.lineno)
        except Exception as exc:
            audit.error("PYTHON_COMPILE", repr(exc), str(path))
    for path in shell_files:
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        if result.returncode:
            audit.error("SHELL_SYNTAX", result.stderr.strip(), str(path))
        else:
            audit.passed(f"shell_syntax:{path.name}")

    if train.is_file() and play.is_file():
        train_hash = hashlib.sha256(train.read_bytes()).hexdigest()
        play_hash = hashlib.sha256(play.read_bytes()).hexdigest()
        if train_hash != play_hash:
            audit.error("TRAIN_PLAY_MISMATCH", f"train={train_hash} play={play_hash}")
        else:
            audit.passed("train_play_identical")

    if train.is_file():
        text = train.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(train))
            audit.passed("train_ast_parse")
        except SyntaxError as exc:
            audit.error("TRAIN_AST", exc.msg, str(train), exc.lineno)
            tree = ast.Module(body=[], type_ignores=[])
        check_unresolved_globals(text, str(train), audit)

        required_tokens = (
            "V4K1_R5_STABLE_STANCE_DELTA_POSE_KICK",
            "V4K1_R5_3_CAUSAL_DISCOVERY_REPAIR",
            'V4K1_SETTLE_MAX_STEPS", "150"',
            "required_activation_rate=",
            "settle_timeout_rate_max=",
            "arming_violation_rate_max=",
            "never_activated_envs=",
            "os._exit(1)",
            "os._exit(0)",
            "run_v4k1_stable_reset_guard",
            "[V4K1_R5_STABLE_RESET_GUARD]",
            "stable_stance_mask(",
            "valid_kick_contact_mask(",
            "stable_kick_success_mask(",
            "self.ball_placed",
            "self.kick_active",
            "self.settle_step_buf",
            "self.ball_arm_step_buf",
            "self.active_kick_step_buf",
            "applied_actions = torch.where",
            "self.ball.write_root_pose_to_sim",
            "selected[env_ids, :2]",
            "precontact_ball_motion_mask(",
            "before_ball_speed=before_ball_speed",
            "deliberate_approach=deliberate_approach",
            "invalid_kick_contact_mask(",
            "next_valid_contact_duration(",
            "standing_positive_baseline_removed_in_stage1=true",
            "body_ball_contact",
            "wrong_foot_contact",
            "self.foot_lift_seen",
            "self.foot_swing_seen",
            "self.recovery_seen",
            "contact_duration_steps <= 12",
            "reward = torch.where(active",
            "V4K1_R5_STABLE_STANCE_CONTACT_CAUSALITY=true",
            "V4K1-R5 invalid-contact/exploit guard failed",
            "V4K1-R5 stable-activation guard failed",
            'final_stable_activation = mean_available("episode_stable_activation_rate")',
            'final_violation = mean_available("episode_violation_rate")',
            '"completed_env_fraction"',
            "V4K1_MIN_COMPLETED_ENV_FRACTION",
            "V4K1-R5.3 rollout-completion guard failed",
            "V4K1_STAGE1_MIN_LIFT_RATE",
            "V4K1_STAGE1_MIN_SWING_RATE",
            "episode_foot_lift_rate",
            "episode_foot_swing_rate",
            "bootstrap_62_to_70",
            "resume_70_to_70",
        )
        for token in required_tokens:
            if token not in text:
                audit.error("MISSING_TOKEN", token, str(train))

        forbidden_tokens = (
            'V4K1_SETTLE_MAX_STEPS", "75"',
            "simulation_app.close()",
            "stage3_two_leg_generalization=true",
            "load_result.",
            'f"warm_start_mode={"',
            "actor_input != 62",
            "reward = torch.where(~active",
            "del before_state",
            "self.contact_duration_steps += geometric_contact.long()",
        )
        for token in forbidden_tokens:
            if token in text:
                audit.error("FORBIDDEN_TOKEN", token, str(train))

        for method in (
            "__init__",
            "_resolve_all_foot_ids",
            "_stability_state",
            "_place_ball_for_kick",
            "_advance_pre_kick_state",
            "_kick_reward",
            "kick_step",
            "_reset_idx",
        ):
            if class_method_node(tree, "IsolatedDeltaPoseKickEnv", method) is None:
                audit.error("MISSING_METHOD", method, str(train))

        kick_step = class_method_node(tree, "IsolatedDeltaPoseKickEnv", "kick_step")
        training = function_node(tree, "run_v4k1_training")
        if kick_step is not None and training is not None:
            diagnostic_keys = extract_dict_keys(kick_step, "diagnostics")
            totals_keys = extract_dict_keys(training, "totals")
            missing_diag = sorted(totals_keys - diagnostic_keys)
            if missing_diag:
                audit.error("TOTALS_DIAGNOSTIC_KEY_MISMATCH", f"missing={missing_diag}", str(train))
            else:
                audit.passed("training_totals_match_diagnostics")

        reset_source = source_segment(text, class_method_node(tree, "IsolatedDeltaPoseKickEnv", "_reset_idx"))
        placement_source = source_segment(text, class_method_node(tree, "IsolatedDeltaPoseKickEnv", "_place_ball_for_kick"))
        if "origins[:, 0] + 2.20" not in reset_source or "origins[:, 1] + 1.80" not in reset_source:
            audit.error("BALL_NOT_PARKED_DURING_SETTLE", "park location missing", str(train))
        if "selected[env_ids, :2]" not in placement_source:
            audit.error("BALL_NOT_PLACED_FROM_ACTUAL_FOOT", "actual selected foot placement missing", str(train))
        if "self.kick_leg[env_ids] = 0" not in reset_source:
            audit.error("LEFT_LEG_NOT_ISOLATED", "R5 must train left leg only", str(train))

        training_source = source_segment(
            text,
            function_node(tree, "run_v4k1_training"),
        )
        semantic_pairs = (
            (
                'final_stable_activation = mean_available("episode_stable_activation_rate")',
                "FINAL_STABLE_EPISODE_METRIC",
            ),
            (
                'final_violation = mean_available("episode_violation_rate")',
                "FINAL_VIOLATION_EPISODE_METRIC",
            ),
            (
                "completed_env_seen |= dones.to(env.device).bool()",
                "COMPLETION_COVERAGE_TRACKING",
            ),
        )
        for token, code in semantic_pairs:
            if token not in training_source:
                audit.error(code, token, str(train))
            else:
                audit.passed(code.lower())

        pure_predicate_tests(text, tree, audit)
        warm_start_tests(text, tree, audit)

    # Worker and installer must enforce the new guard before accepting a job.
    if worker.is_file():
        worker_text = worker.read_text(encoding="utf-8")
        for token in (
            "V4K1_R5_STABLE_STANCE_CONTACT_CAUSALITY=true",
            "[V4K1_R5_STABLE_RESET_GUARD]",
            "V4K1_MAX_VIOLATION_RATE",
            "V4K1_MIN_STABLE_ACTIVATION_RATE",
            "stage3_fixed_leg=left",
            "V4K1_SETTLE_MAX_STEPS:-150",
            "experiments/v4k1_r5_3_stable_kick",
            "stage1_min_lift_rate=",
            "stage1_min_swing_rate=",
            "V4K1_MIN_COMPLETED_ENV_FRACTION",
            "V4K1_ROLLOUT_STEPS",
            'touch "${RUN_DIR}/V4K1_ISOLATED_KICK_FAILED"',
        ):
            if token not in worker_text:
                audit.error("WORKER_MISSING_TOKEN", token, str(worker))
    if installer.is_file():
        installer_text = installer.read_text(encoding="utf-8")
        for token in (
            "V4K1_R5_PACKAGE_AUDIT_CLEAN",
            "validate_v4k1_r5_package.py",
            "afterok",
            "V4K1_MAX_VIOLATION_RATE",
            "V4K1_MIN_STABLE_ACTIVATION_RATE",
            "V4K1_SETTLE_MAX_STEPS=150",
            "runs/go2_soccer_v4k1_r5_3_stable_kick",
            "pilot=128_envs_300_iters_stages_200_70_30",
            "full=512_envs_900_iters_stages_300_400_200",
            "V4K1_STAGE1_MIN_LIFT_RATE",
            "V4K1_STAGE1_MIN_SWING_RATE",
            "512 3 1 1 1 00:45:00",
            "PRE_ROLLOUT_STEPS=240",
            "PRE_TEMPORAL_MIN_STEPS",
            "V4K1_R5_3_TEMPORAL_PREFLIGHT_OK",
            "V4K1_MIN_COMPLETED_ENV_FRACTION",
        ):
            if token not in installer_text:
                audit.error("INSTALLER_MISSING_TOKEN", token, str(installer))

    payload = {
        "package": str(package),
        "errors": audit.errors,
        "warnings": audit.warnings,
        "ok": audit.ok,
        "counts": {
            "errors": len(audit.errors),
            "warnings": len(audit.warnings),
            "ok": len(audit.ok),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(
        "V4K1_R5_PACKAGE_AUDIT_COUNTS "
        f"errors={len(audit.errors)} warnings={len(audit.warnings)} ok={len(audit.ok)}"
    )
    if audit.errors:
        print("V4K1_R5_PACKAGE_AUDIT_FAILED")
        return 1
    print("V4K1_R5_PACKAGE_AUDIT_CLEAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
