#!/usr/bin/env python3
"""Read-only initial/final fixed evaluation for the V56 progress canary."""

from __future__ import annotations

import argparse
import faulthandler
import os
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--phase", choices=("initial", "final"), required=True)
parser.add_argument("--contract", type=Path, required=True)
parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--checkpoint-sha256")
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
    EVALUATION_SCHEMA,
    FIXED_CASE_COUNT,
    EXPECTED_FIXED_CASE_TABLE_SHA256,
    load_contract,
    validate_episode_rows,
)
from check_frozen_low_state import module_state_snapshot  # noqa: E402
from runtime_common import (  # noqa: E402
    action_std_report,
    atomic_bytes,
    atomic_json,
    atomic_jsonl,
    clone_state,
    file_sha256,
    load_same_run_final_policy,
    make_runtime,
    module_bit_state_sha256,
    output_head_report,
    policy_module,
    runner_cfg,
    runner_identity,
    runtime_provenance,
    save_checkpoint,
)


def _state_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[name], right[name]) for name in left)


def run() -> int:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract = load_contract(args.contract)
    run_contract = contract["bounded_canary"]
    if args.phase == "initial" and (args.checkpoint is not None or args.checkpoint_sha256 is not None):
        raise RuntimeError("initial evaluation must not load a checkpoint")
    if args.phase == "final" and (args.checkpoint is None or args.checkpoint_sha256 is None):
        raise RuntimeError("final evaluation requires the same-run final checkpoint and SHA256")

    env = adapter = low = None
    env, adapter, low, low_before, runtime_identity = make_runtime(
        contract=contract,
        num_envs=int(run_contract["num_envs"]),
        reset_seed=int(run_contract["fixed_evaluation_seed"]),
        device=args.device,
        task=args.task,
        evaluation_case_count=FIXED_CASE_COUNT,
    )
    runtime_identity.update(runtime_provenance(contract))
    try:
        # Isaac/reset construction may consume torch RNG.  Policy initialization
        # is intentionally reseeded immediately before the fresh constructor.
        policy_seed = int(run_contract["policy_initialization_seed"])
        torch.manual_seed(policy_seed)
        runner = OnPolicyRunner(adapter, runner_cfg(contract=contract, device=args.device), log_dir=None, device=args.device)
        policy = policy_module(runner)
        identity = runner_identity(runner, policy, contract)
        checkpoint_loaded = False
        loaded_identity: dict[str, object] | None = None
        if args.phase == "final":
            loaded_identity = load_same_run_final_policy(
                policy=policy,
                checkpoint=args.checkpoint,
                expected_file_sha256=str(args.checkpoint_sha256),
            )
            if loaded_identity["training_iteration"] != int(run_contract["learning_iterations"]):
                raise RuntimeError("final checkpoint iteration changed")
            checkpoint_loaded = True
        initial_std = action_std_report(policy, require_initial_unit=args.phase == "initial")
        head_receipt = output_head_report(policy)
        policy_state_before = clone_state(policy)
        policy_sha_before = module_bit_state_sha256(policy)
        initial_checkpoint = None
        if args.phase == "initial":
            initial_checkpoint = save_checkpoint(
                output / "model_initial.pt",
                policy=policy,
                optimizer=None,
                iteration=0,
                kind="initial_before_any_act",
            )
            if initial_checkpoint["policy_state_sha256"] != policy_sha_before:
                raise RuntimeError("durable initial checkpoint differs from live fresh policy")

        optimizer = runner.alg.optimizer
        original_step = optimizer.step
        optimizer_steps = 0

        def forbidden_optimizer_step(*values, **kwargs):
            nonlocal optimizer_steps
            optimizer_steps += 1
            raise RuntimeError("fixed evaluation attempted an optimizer step")

        optimizer.step = forbidden_optimizer_step
        rows: list[dict[str, object]] = []
        reward_min = float("inf")
        reward_max = float("-inf")
        observation = adapter.get_observations().to(args.device)
        policy.eval()
        vector_steps = 0
        with torch.inference_mode():
            while len(rows) < FIXED_CASE_COUNT:
                action = policy.act_inference(observation)
                if not bool(torch.isfinite(action).all()):
                    raise RuntimeError("fixed-evaluation action is non-finite")
                observation, reward, done, extras = adapter.step(action.to(adapter.device))
                if "time_outs" in extras:
                    raise RuntimeError("finite-horizon evaluation exposed RSL timeout bootstrap key")
                reward_min = min(reward_min, float(reward.min().item()))
                reward_max = max(reward_max, float(reward.max().item()))
                for event in extras.get("terminal_events", []):
                    if event.get("counts_for_evaluation"):
                        rows.append(dict(event))
                policy.reset(done.to(args.device))
                observation = observation.to(args.device)
                vector_steps += 1
                if vector_steps > 800:
                    raise RuntimeError("fixed evaluation exceeded its finite completion bound")
        optimizer.step = original_step
        if len(rows) != FIXED_CASE_COUNT:
            raise RuntimeError("fixed evaluation did not finish exactly 288 cases")
        metrics = validate_episode_rows(rows)
        rows_path = output / f"{args.phase}_episode_results.jsonl"
        atomic_jsonl(rows_path, rows)
        policy_after = clone_state(policy)
        policy_sha_after = module_bit_state_sha256(policy)
        low_after = module_state_snapshot(low)
        policy_unchanged = _state_equal(policy_state_before, policy_after) and policy_sha_after == policy_sha_before
        low_unchanged = low_before["sha256"] == low_after["sha256"]
        summary = {
            "schema": EVALUATION_SCHEMA,
            "status": "COMPLETE",
            "phase": args.phase,
            "fixed_evaluation_case_count": FIXED_CASE_COUNT,
            "hard_case_count": metrics["hard_case_count"],
            "assignments_exactly_once": True,
            "assignment_sha256": metrics["assignment_sha256"],
            "fixed_case_table_sha256": EXPECTED_FIXED_CASE_TABLE_SHA256,
            "episode_rows_sha256": file_sha256(rows_path),
            "goal_count": metrics["goal_count"],
            "hard_goal_count": metrics["hard_goal_count"],
            "unsafe_count": metrics["unsafe_count"],
            "contact_count": metrics["contact_count"],
            "case_counts_by_goal_side": metrics["case_counts_by_goal_side"],
            "hard_case_counts_by_goal_side": metrics["hard_case_counts_by_goal_side"],
            "goal_counts_by_goal_side": metrics["goal_counts_by_goal_side"],
            "hard_goal_counts_by_goal_side": metrics["hard_goal_counts_by_goal_side"],
            "outcome_counts": metrics["outcome_counts"],
            "maximum_live_reset_error": metrics["maximum_live_reset_error"],
            "mean_initial_l_m": metrics["mean_initial_l_m"],
            "mean_final_l_m": metrics["mean_final_l_m"],
            "mean_cumulative_progress_reward": metrics["mean_cumulative_progress_reward"],
            "mean_requested_goal_axis_ball_displacement_m": metrics["mean_requested_goal_axis_ball_displacement_m"],
            "mean_abs_lateral_ball_drift_m": metrics["mean_abs_lateral_ball_drift_m"],
            "dense_return_authorizes_behavior": False,
            "ppo_optimizer_steps": optimizer_steps,
            "policy_state_sha256": policy_sha_before,
            "loaded_policy_state_sha256": None if loaded_identity is None else loaded_identity["policy_state_sha256"],
            "policy_state_unchanged": policy_unchanged,
            "fresh_policy_initialized": args.phase == "initial",
            "checkpoint_loaded": checkpoint_loaded,
            "checkpoint_sha256": None if loaded_identity is None else loaded_identity["sha256"],
            "initial_checkpoint": initial_checkpoint,
            "action_std": initial_std,
            "output_heads": head_receipt,
            "policy_initialization_seed": policy_seed,
            "policy_seed_reapplied_after_environment_construction": True,
            "runner_identity": identity,
            "runtime_identity": runtime_identity,
            "source_receipts": SOURCE_RECEIPTS,
            "total_reward_min": reward_min,
            "total_reward_max": reward_max,
            "strict_terminal_reward_values": [-1.0, 0.0, 1.0],
            "finite_horizon": True,
            "time_outs_bootstrap_key_emitted": False,
            "frozen_low_state_unchanged": low_unchanged,
            "low_state_before_sha256": low_before["sha256"],
            "low_state_after_sha256": low_after["sha256"],
            "environment": env.summary(),
            "failures": [],
            "long_training_authorized": False,
        }
        if not policy_unchanged or not low_unchanged or optimizer_steps != 0:
            raise RuntimeError("read-only fixed evaluation mutated trainable or frozen state")
        summary_path = output / f"{args.phase}_evaluation_summary.json"
        atomic_json(summary_path, summary)
        guard = "INITIAL_EVALUATION_FINAL_GUARD" if args.phase == "initial" else "FINAL_EVALUATION_FINAL_GUARD"
        atomic_bytes(output / guard, b"PASS\n")
        print(f"V56_{args.phase.upper()}_EVALUATION=COMPLETE")
        return 0
    finally:
        # Omniverse teardown is intentionally bypassed by the process-level
        # hard exit after durable evidence is written.  Synchronous close has
        # previously masked valid evidence with unrelated cache/quota errors.
        pass


if __name__ == "__main__":
    faulthandler.enable(all_threads=True)
    try:
        return_code = run()
    except BaseException as error:
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(
            output / f"{args.phase}_evaluation_failure.json",
            {"status": "ERROR", "phase": args.phase, "exception_type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(), "long_training_authorized": False},
        )
        print(traceback.format_exc(), flush=True)
        os._exit(90)
    print(f"V56_{args.phase.upper()}_PROCESS=COMPLETE", flush=True)
    os._exit(return_code)
