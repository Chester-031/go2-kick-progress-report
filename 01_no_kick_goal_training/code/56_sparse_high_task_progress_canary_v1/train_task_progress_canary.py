#!/usr/bin/env python3
"""One bounded fresh-high PPO transaction with a pre-update reward gate."""

from __future__ import annotations

import argparse
import faulthandler
import math
import os
from pathlib import Path
import traceback
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--contract", type=Path, required=True)
parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path, required=True)
parser.add_argument("--initial-checkpoint", type=Path, required=True)
parser.add_argument("--initial-checkpoint-sha256", required=True)
parser.add_argument("--expected-initial-policy-state-sha256", required=True)
parser.add_argument("--task", default="Isaac-Go2-Soccer-Milestone2-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

from source_bootstrap import configure_source_roots  # noqa: E402

SOURCE_RECEIPTS = configure_source_roots()

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
import isaac_go2_low_level_rl.tasks  # noqa: E402,F401
import isaac_go2_low_level_rl.tasks.manager_based.soccer  # noqa: E402,F401

from canary_protocol import (  # noqa: E402
    FIRST_ROLLOUT_SAMPLES,
    FIRST_ROLLOUT_STEPS,
    FIRST_UPDATE_OPTIMIZER_STEPS,
    FIRST_UPDATE_SCHEMA,
    TOTAL_OPTIMIZER_STEPS,
    TOTAL_ROLLOUT_SAMPLES,
    TOTAL_ROLLOUT_STEPS,
    TRAINING_ITERATIONS,
    TRAINING_SCHEMA,
    first_update_checks,
    load_contract,
    reward_support_checks,
    validate_first_rollout_transitions,
)
from check_frozen_low_state import module_state_snapshot  # noqa: E402
from recurrent_hidden_guard import clone_hidden_state, hidden_state_equal, restore_hidden_state  # noqa: E402
from runtime_common import (  # noqa: E402
    action_std_report,
    atomic_bytes,
    atomic_json,
    atomic_jsonl,
    clone_state,
    component_update,
    file_sha256,
    install_counted_normalization,
    load_same_run_initial_policy,
    make_runtime,
    module_bit_state_sha256,
    output_head_report,
    policy_module,
    runner_cfg,
    runner_identity,
    runtime_provenance,
    save_checkpoint,
    state_finite,
)


def _json_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item()) if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _loss_values_all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if torch.is_tensor(value):
        return value.numel() > 0 and bool(torch.isfinite(value).all())
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_loss_values_all_finite(item) for item in value)
    if isinstance(value, dict):
        return bool(value) and all(isinstance(key, str) and _loss_values_all_finite(item) for key, item in value.items())
    return False


def _storage_all_finite(algorithm: Any) -> bool:
    storage = algorithm.storage
    tensors = [
        getattr(storage, name)
        for name in ("returns", "advantages", "values", "actions")
        if torch.is_tensor(getattr(storage, name, None))
    ]
    return bool(tensors) and all(bool(torch.isfinite(value).all()) for value in tensors)


def _state_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[name], right[name]) for name in left)


def run() -> int:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract = load_contract(args.contract)
    bounded = contract["bounded_canary"]
    env, adapter, low, low_before, runtime_identity = make_runtime(
        contract=contract,
        num_envs=int(bounded["num_envs"]),
        reset_seed=int(bounded["training_reset_seed"]),
        device=args.device,
        task=args.task,
        capture_transition_rows=True,
    )
    runtime_identity.update(runtime_provenance(contract))

    torch.manual_seed(int(bounded["policy_initialization_seed"]))
    runner = OnPolicyRunner(
        adapter,
        runner_cfg(contract=contract, device=args.device),
        log_dir=None,
        device=args.device,
    )
    policy = policy_module(runner)
    identity = runner_identity(runner, policy, contract)
    optimizer = runner.alg.optimizer
    if optimizer.state:
        raise RuntimeError("fresh PPO optimizer contains state before first rollout")
    loaded = load_same_run_initial_policy(
        policy=policy,
        checkpoint=args.initial_checkpoint,
        expected_file_sha256=args.initial_checkpoint_sha256,
        expected_state_sha256=args.expected_initial_policy_state_sha256,
    )
    if optimizer.state:
        raise RuntimeError("same-run initial policy load populated optimizer state")
    normalization = install_counted_normalization(policy)
    initial_std = action_std_report(policy, require_initial_unit=True)
    initial_heads = output_head_report(policy)
    initial_state = clone_state(policy)
    initial_state_sha = module_bit_state_sha256(policy)
    if initial_state_sha != args.expected_initial_policy_state_sha256:
        raise RuntimeError("trainer initial state differs from initial evaluator")

    original_optimizer_step = optimizer.step
    optimizer_steps = 0

    def counted_optimizer_step(*values, **kwargs):
        nonlocal optimizer_steps
        optimizer_steps += 1
        return original_optimizer_step(*values, **kwargs)

    optimizer.step = counted_optimizer_step
    observation = adapter.get_observations().to(args.device)
    policy.train()
    rollout_steps = 0
    reward_min = math.inf
    reward_max = -math.inf
    iteration_receipts: list[dict[str, Any]] = []
    first_rows: list[dict[str, Any]] = []
    first_update_summary: dict[str, Any] | None = None
    reward_support_summary: dict[str, Any] | None = None

    for iteration in range(TRAINING_ITERATIONS):
        optimizer_before = optimizer_steps
        normalization_before = int(policy.normalization_update_calls)
        with torch.inference_mode():
            for _ in range(FIRST_ROLLOUT_STEPS):
                action = runner.alg.act(observation)
                if not bool(torch.isfinite(action).all()):
                    raise RuntimeError("PPO rollout action is non-finite")
                observation, reward, done, extras = adapter.step(action.to(adapter.device))
                if "time_outs" in extras:
                    raise RuntimeError("finite-horizon training exposed RSL timeout bootstrap key")
                if iteration == 0:
                    rows = extras.get("v56_transition_rows")
                    if not isinstance(rows, list) or len(rows) != adapter.num_envs:
                        raise RuntimeError("first rollout raw transition receipt is incomplete")
                    first_rows.extend(dict(row) for row in rows)
                observation = observation.to(args.device)
                reward = reward.to(args.device)
                done = done.to(args.device)
                runner.alg.process_env_step(observation, reward, done, extras)
                reward_min = min(reward_min, float(reward.min().item()))
                reward_max = max(reward_max, float(reward.max().item()))
                rollout_steps += 1

        # The complete 25,600-row receipt is durable and independently
        # reconstructed before compute_returns or any optimizer update.
        if iteration == 0:
            if optimizer_steps != 0 or optimizer.state:
                raise RuntimeError("optimizer changed before reward-support gate")
            rows_path = output / "first_rollout_transitions.jsonl"
            atomic_jsonl(rows_path, first_rows)
            reward_support_summary = validate_first_rollout_transitions(first_rows)
            reward_support_summary.update({
                "transition_rows_relative_path": rows_path.name,
                "transition_rows_bytes": rows_path.stat().st_size,
                "transition_rows_sha256": file_sha256(rows_path),
                "optimizer_constructed_before_gate": True,
                "optimizer_state_empty_before_gate": not bool(optimizer.state),
                "optimizer_steps_before_gate": optimizer_steps,
                "initial_policy_state_sha256": initial_state_sha,
                "initial_checkpoint_sha256": loaded["sha256"],
                "frozen_low_state_sha256_before_gate": low_before["sha256"],
                "dense_return_authorizes_behavior": False,
                "long_training_authorized": False,
            })
            checks = reward_support_checks(reward_support_summary)
            structural_names = (
                "terminal_complete",
                "row_count_exact",
                "complete_unique_grid",
                "all_finite",
                "all_formula_reconstructed",
                "pre_optimizer_update_gate",
                "transition_rows_receipt_valid",
                "initial_identity_fields_well_formed",
            )
            structural_failures = [name for name in structural_names if checks.get(name) is not True]
            if structural_failures:
                reward_support_summary["status"] = "ERROR"
                reward_support_summary["failures"] = [
                    "reward-support structural/integrity failure: " + ",".join(structural_failures)
                ]
                reward_support_summary["gate_pass"] = False
                reward_support_summary["checks"] = checks
                reward_support_summary["blockers"] = structural_failures
                atomic_json(output / "first_rollout_reward_support_summary.json", reward_support_summary)
                atomic_bytes(output / "FIRST_ROLLOUT_REWARD_SUPPORT_FINAL_GUARD", b"FAIL\n")
                raise RuntimeError(reward_support_summary["failures"][0])
            signal_names = (
                "global_nonzero_support",
                "global_positive_support",
                "global_negative_support",
                "positive_support_goal_side_negative",
                "positive_support_goal_side_positive",
            )
            reward_support_summary["gate_pass"] = all(checks.get(name) is True for name in signal_names)
            reward_support_summary["checks"] = checks
            reward_support_summary["blockers"] = sorted(name for name in signal_names if checks.get(name) is not True)
            atomic_json(output / "first_rollout_reward_support_summary.json", reward_support_summary)
            atomic_bytes(output / "FIRST_ROLLOUT_REWARD_SUPPORT_FINAL_GUARD", b"PASS\n")
            env.disable_transition_capture()

            if not reward_support_summary["gate_pass"]:
                state_after_rollout = clone_state(policy)
                if not _state_equal(initial_state, state_after_rollout):
                    raise RuntimeError("policy weights changed during rejected rollout")
                runner.alg.storage.clear()
                if int(runner.alg.storage.step) != 0 or optimizer.state or optimizer_steps != 0:
                    raise RuntimeError("rejected rollout did not preserve pristine PPO state")
                low_after = module_state_snapshot(low)
                if low_after["sha256"] != low_before["sha256"]:
                    raise RuntimeError("frozen low changed during rejected rollout")
                if (output / "model_first_update.pt").exists() or (output / "model_final.pt").exists():
                    raise RuntimeError("rejected reward-support branch left a learned checkpoint")
                reward_support_summary["rejection_integrity"] = {
                    "storage_cleared_without_update": True,
                    "policy_state_unchanged_without_update": True,
                    "policy_state_sha256_after_rejection": module_bit_state_sha256(policy),
                    "low_state_unchanged_without_update": low_after["sha256"] == low_before["sha256"],
                    "low_state_sha256_after_rejection": low_after["sha256"],
                    "optimizer_state_empty_after_rejection": not bool(optimizer.state),
                    "optimizer_steps_after_rejection": optimizer_steps,
                    "learned_checkpoint_count": 0,
                }
                final_rejection_checks = reward_support_checks(reward_support_summary)
                if not (
                    final_rejection_checks.get("terminal_complete") is True
                    and final_rejection_checks.get("pre_optimizer_update_gate") is True
                    and final_rejection_checks.get("rejected_branch_no_update_integrity") is True
                ):
                    raise RuntimeError(f"reward-support rejection integrity failed: {final_rejection_checks}")
                reward_support_summary["checks"] = final_rejection_checks
                reward_support_summary["blockers"] = sorted(
                    name
                    for name in (
                        "global_nonzero_support",
                        "global_positive_support",
                        "global_negative_support",
                        "positive_support_goal_side_negative",
                        "positive_support_goal_side_positive",
                    )
                    if final_rejection_checks.get(name) is not True
                )
                reward_support_summary["independent_integrity_checks"] = final_rejection_checks
                atomic_json(output / "first_rollout_reward_support_summary.json", reward_support_summary)
                atomic_bytes(output / "REWARD_SUPPORT_REJECTED_FINAL_GUARD", b"PASS\n")
                print("V56_REWARD_SUPPORT_GATE=REJECTED_STOP_BEFORE_PPO_UPDATE", flush=True)
                return 0

        actor_hidden_snapshot = clone_hidden_state(policy.memory_a.hidden_state)
        critic_hidden_snapshot = clone_hidden_state(policy.memory_c.hidden_state)
        if actor_hidden_snapshot is None or critic_hidden_snapshot is None:
            raise RuntimeError("recurrent hidden state is absent after rollout")
        with torch.inference_mode():
            runner.alg.compute_returns(observation)
        restore_hidden_state(policy.memory_a, actor_hidden_snapshot)
        restore_hidden_state(policy.memory_c, critic_hidden_snapshot)
        hidden_receipt = {
            "actor_equal_after_restore": hidden_state_equal(policy.memory_a.hidden_state, actor_hidden_snapshot),
            "critic_equal_after_restore": hidden_state_equal(policy.memory_c.hidden_state, critic_hidden_snapshot),
        }
        if not all(hidden_receipt.values()):
            raise RuntimeError("recurrent hidden state restore failed")
        returns_finite = _storage_all_finite(runner.alg)
        losses = runner.alg.update()
        hidden_receipt["actor_unchanged_by_update"] = hidden_state_equal(policy.memory_a.hidden_state, actor_hidden_snapshot)
        hidden_receipt["critic_unchanged_by_update"] = hidden_state_equal(policy.memory_c.hidden_state, critic_hidden_snapshot)
        losses_all_finite = (
            isinstance(losses, dict)
            and set(losses) == {"value_function", "surrogate", "entropy"}
            and _loss_values_all_finite(losses)
        )
        receipt = {
            "iteration": iteration,
            "rollout_steps": FIRST_ROLLOUT_STEPS,
            "rollout_samples": FIRST_ROLLOUT_SAMPLES,
            "normalization_update_calls": int(policy.normalization_update_calls) - normalization_before,
            "optimizer_steps": optimizer_steps - optimizer_before,
            "cumulative_optimizer_steps": optimizer_steps,
            "returns_complete": True,
            "returns_all_finite": returns_finite,
            "recurrent_hidden_restore": hidden_receipt,
            "storage_reset_after_update": int(runner.alg.storage.step) == 0,
            "losses_all_finite": losses_all_finite,
            "finite_horizon": True,
            "time_outs_bootstrap_key_emitted": False,
            "losses": _json_value(losses),
        }
        if (
            not returns_finite
            or not losses_all_finite
            or receipt["optimizer_steps"] != FIRST_UPDATE_OPTIMIZER_STEPS
            or receipt["normalization_update_calls"] != FIRST_ROLLOUT_STEPS
            or receipt["storage_reset_after_update"] is not True
            or not all(hidden_receipt.values())
        ):
            raise RuntimeError(f"bounded PPO iteration integrity failed: {receipt}")
        iteration_receipts.append(receipt)

        if iteration == 0:
            if reward_support_summary is None or reward_support_summary["gate_pass"] is not True or optimizer_before != 0:
                raise RuntimeError("first PPO update bypassed reward-support gate")
            after_first = clone_state(policy)
            low_after_first = module_state_snapshot(low)
            first_checkpoint = save_checkpoint(
                output / "model_first_update.pt",
                policy=policy,
                optimizer=optimizer,
                iteration=1,
                kind="first_update",
            )
            first_update_summary = {
                "schema": FIRST_UPDATE_SCHEMA,
                "status": "COMPLETE",
                "gate_pass": True,
                "rollout_steps": FIRST_ROLLOUT_STEPS,
                "rollout_samples": FIRST_ROLLOUT_SAMPLES,
                "normalization_update_calls": int(policy.normalization_update_calls),
                "returns_complete": True,
                "returns_all_finite": returns_finite,
                "storage_reset_after_update": int(runner.alg.storage.step) == 0,
                "losses_all_finite": losses_all_finite,
                "ppo_optimizer_steps": optimizer_steps,
                "expected_ppo_optimizer_steps": FIRST_UPDATE_OPTIMIZER_STEPS,
                "actor_update": component_update(initial_state, after_first, ("memory_a.", "actor.")),
                "critic_update": component_update(initial_state, after_first, ("memory_c.", "critic.")),
                "policy_all_finite": state_finite(after_first),
                "initial_policy_state_sha256": initial_state_sha,
                "initial_checkpoint_sha256": loaded["sha256"],
                "checkpoint_loaded": True,
                "checkpoint_origin": "RUN_LOCAL_INITIAL_EVALUATION",
                "optimizer_constructed": True,
                "optimizer_loaded": False,
                "fresh_optimizer_state_empty_before_first_update": True,
                "reward_support_gate_pass": True,
                "reward_support_rows_sha256": reward_support_summary["transition_rows_sha256"],
                "recurrent_hidden_restore": hidden_receipt,
                "first_update_checkpoint": first_checkpoint,
                "frozen_low_state_unchanged": low_before["sha256"] == low_after_first["sha256"],
                "low_state_before_sha256": low_before["sha256"],
                "low_state_after_sha256": low_after_first["sha256"],
                "iteration_receipt": receipt,
                "runner_identity": identity,
                "runtime_identity": runtime_identity,
                "source_receipts": SOURCE_RECEIPTS,
                "failures": [],
                "long_training_authorized": False,
            }
            checks = first_update_checks(first_update_summary)
            first_update_summary["checks"] = checks
            if not all(checks.values()):
                raise RuntimeError(f"first-update integrity checks failed: {checks}")
            atomic_json(output / "first_update_summary.json", first_update_summary)
            atomic_bytes(output / "FIRST_UPDATE_FINAL_GUARD", b"PASS\n")

        atomic_json(output / "iteration_progress.json", {
            "status": "ACTIVE" if iteration + 1 < TRAINING_ITERATIONS else "COMPLETE",
            "completed_iterations": iteration + 1,
            "rollout_steps": rollout_steps,
            "optimizer_steps": optimizer_steps,
            "last_iteration": receipt,
        })

    if first_update_summary is None or reward_support_summary is None:
        raise RuntimeError("passing branch lacks first-rollout/first-update evidence")
    final_state = clone_state(policy)
    final_state_sha = module_bit_state_sha256(policy)
    low_after = module_state_snapshot(low)
    final_checkpoint = save_checkpoint(
        output / "model_final.pt",
        policy=policy,
        optimizer=optimizer,
        iteration=TRAINING_ITERATIONS,
        kind="bounded_training_final",
    )
    training_summary = {
        "schema": TRAINING_SCHEMA,
        "status": "COMPLETE",
        "training_iterations": len(iteration_receipts),
        "rollout_steps": rollout_steps,
        "rollout_samples": rollout_steps * int(bounded["num_envs"]),
        "normalization_update_calls": int(policy.normalization_update_calls),
        "ppo_optimizer_steps": optimizer_steps,
        "expected_ppo_optimizer_steps": TOTAL_OPTIMIZER_STEPS,
        "optimizer_constructed": True,
        "iteration_receipts": iteration_receipts,
        "actor_update": component_update(initial_state, final_state, ("memory_a.", "actor.")),
        "critic_update": component_update(initial_state, final_state, ("memory_c.", "critic.")),
        "policy_all_finite": state_finite(final_state),
        "initial_policy_state_sha256": initial_state_sha,
        "final_policy_state_sha256": final_state_sha,
        "initial_checkpoint_sha256": loaded["sha256"],
        "reward_support_gate_pass": True,
        "reward_support_rows_sha256": reward_support_summary["transition_rows_sha256"],
        "first_update_checkpoint": first_update_summary["first_update_checkpoint"],
        "initial_action_std": initial_std,
        "final_action_std": action_std_report(policy, require_initial_unit=False),
        "initial_output_heads": initial_heads,
        "final_output_heads": output_head_report(policy),
        "reward_min": reward_min,
        "reward_max": reward_max,
        "frozen_low_state_unchanged": low_before["sha256"] == low_after["sha256"],
        "low_state_before_sha256": low_before["sha256"],
        "low_state_after_sha256": low_after["sha256"],
        "full_distribution_from_iteration_zero": True,
        "online_curriculum_registered": False,
        "training_initial_reset_count": env.summary()["training_initial_reset_count"],
        "training_initial_resets": env.summary()["training_initial_resets"],
        "finite_horizon": True,
        "time_outs_bootstrap_key_emitted": False,
        "final_checkpoint": final_checkpoint,
        "runner_identity": identity,
        "runtime_identity": runtime_identity,
        "source_receipts": SOURCE_RECEIPTS,
        "normalization_lifecycle": normalization,
        "environment": env.summary(),
        "dense_return_authorizes_behavior": False,
        "failures": [],
        "long_training_authorized": False,
    }
    if (
        rollout_steps != TOTAL_ROLLOUT_STEPS
        or rollout_steps * int(bounded["num_envs"]) != TOTAL_ROLLOUT_SAMPLES
        or optimizer_steps != TOTAL_OPTIMIZER_STEPS
        or int(policy.normalization_update_calls) != TOTAL_ROLLOUT_STEPS
        or not training_summary["frozen_low_state_unchanged"]
        or final_checkpoint["policy_state_sha256"] != final_state_sha
    ):
        raise RuntimeError("bounded PPO totals or frozen-low identity changed")
    atomic_json(output / "training_summary.json", training_summary)
    atomic_bytes(output / "TRAINING_FINAL_GUARD", b"PASS\n")
    print("V56_BOUNDED_TRAINING=COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    faulthandler.enable(all_threads=True)
    try:
        return_code = run()
    except BaseException as error:
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "training_failure.json", {
            "status": "ERROR",
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "long_training_authorized": False,
        })
        print(traceback.format_exc(), flush=True)
        os._exit(91)
    print("V56_TRAINING_PROCESS=COMPLETE", flush=True)
    os._exit(return_code)
