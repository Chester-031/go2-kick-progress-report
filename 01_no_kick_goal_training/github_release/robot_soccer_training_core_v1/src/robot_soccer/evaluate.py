"""Run deterministic fixed evaluation for Reference, Full, or named-hard cases."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping

from .config import SEED


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-checkpoint", type=Path)
    parser.add_argument("--low-checkpoint-sha256")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--split", choices=("reference", "full", "hard"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--task", default="Isaac-Go2-Soccer-Milestone2-v0")
    parser.add_argument("--num-envs", type=int, default=128)
    return parser


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _side_counts(rows: list[Mapping[str, Any]], key: str) -> dict[str, int]:
    return {
        str(side): sum(int(row[key]) for row in rows if int(row["goal_side"]) == side)
        for side in (-1, 1)
    }


def run(args: argparse.Namespace) -> int:
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from .metrics import summarize_episodes
    from .runtime import (
        clone_module_state,
        fixed_samples,
        load_high_policy,
        make_environment,
        module_state_equal,
        policy_module,
        prepare_output_directory,
        runner_config,
    )

    if args.num_envs <= 0:
        raise ValueError("num-envs must be positive")
    samples, seed = fixed_samples(args.split)
    expected_count = len(samples)
    output = prepare_output_directory(args.output_dir)

    env, adapter, low_policy = make_environment(
        low_checkpoint=args.low_checkpoint,
        low_checkpoint_sha256=args.low_checkpoint_sha256,
        num_envs=args.num_envs,
        seed=seed,
        device=args.device,
        task=args.task,
        split=args.split,
    )
    torch.manual_seed(SEED)
    low_before = clone_module_state(low_policy)
    runner = OnPolicyRunner(
        adapter,
        runner_config(device=args.device, seed=SEED),
        log_dir=None,
        device=args.device,
    )
    policy = policy_module(runner)
    checkpoint_iteration = load_high_policy(
        policy,
        args.checkpoint,
        expected_checkpoint_sha256=args.checkpoint_sha256,
    )
    policy.eval()
    policy_before = clone_module_state(policy)

    def forbidden_step(*_values: Any, **_kwargs: Any) -> None:
        raise RuntimeError("evaluation attempted an optimizer step")

    runner.alg.optimizer.step = forbidden_step
    observation = adapter.get_observations().to(args.device)
    rows: list[dict[str, Any]] = []
    vector_steps = 0
    maximum_vector_steps = (math.ceil(expected_count / args.num_envs) + 1) * 200
    with torch.inference_mode():
        while len(rows) < expected_count:
            action = policy.act_inference(observation)
            if not torch.isfinite(action).all():
                raise RuntimeError("policy produced a non-finite evaluation action")
            observation, _reward, done, extras = adapter.step(action.to(adapter.device))
            for event in extras["terminal_events"]:
                if event["counts_for_evaluation"]:
                    rows.append(dict(event))
            policy.reset(done.to(args.device))
            observation = observation.to(args.device)
            vector_steps += 1
            if vector_steps > maximum_vector_steps:
                raise RuntimeError("fixed evaluation exceeded its completion bound")

    if len(rows) != expected_count:
        raise RuntimeError("fixed evaluation returned the wrong number of cases")
    assignments = [int(row["evaluation_assignment"]) for row in rows]
    if sorted(assignments) != list(range(expected_count)):
        raise RuntimeError("fixed evaluation assignments are not exact")
    rows.sort(key=lambda row: int(row["evaluation_assignment"]))

    policy_after = clone_module_state(policy)
    low_after = clone_module_state(low_policy)
    if not module_state_equal(policy_before, policy_after):
        raise RuntimeError("evaluation changed the high policy")
    if not module_state_equal(low_before, low_after):
        raise RuntimeError("evaluation changed the frozen low policy")

    aggregate = summarize_episodes(rows)
    hard_rows = [row for row in rows if row["hard_goal_facing_ball_behind"]]
    hard_metrics = summarize_episodes(hard_rows)
    outcome_names = (
        "goal",
        "own_goal",
        "boundary",
        "fall",
        "joint_hard_limit",
        "timeout",
    )
    outcome_counter = Counter(str(row["outcome"]) for row in rows)
    summary = {
        "status": "COMPLETE",
        "split": args.split,
        "checkpoint_iteration": checkpoint_iteration,
        "case_count": expected_count,
        "fixed_evaluation_case_count": expected_count,
        "hard_case_count": len(hard_rows),
        "goal_count": aggregate["goal_count"],
        "hard_goal_count": hard_metrics["goal_count"],
        "contact_count": aggregate["contact_count"],
        "unsafe_count": aggregate["unsafe_count"],
        "mean_episode_return": aggregate["mean_episode_return"],
        "mean_cumulative_progress_reward": aggregate[
            "mean_cumulative_progress_reward"
        ],
        "goal_counts_by_side": _side_counts(rows, "goal"),
        "contact_counts_by_side": _side_counts(rows, "contact_seen"),
        "outcome_counts": {
            name: int(outcome_counter.get(name, 0)) for name in outcome_names
        },
        "metrics": aggregate,
        "hard_metrics": hard_metrics,
        "policy_unchanged": True,
        "frozen_low_unchanged": True,
        "optimizer_steps": 0,
        "vector_steps": vector_steps,
    }
    _write_jsonl(output / f"{args.split}_episodes.jsonl", rows)
    _write_json(output / f"{args.split}_summary.json", summary)
    print(
        f"split={args.split} cases={expected_count} "
        f"goals={aggregate['goal_count']} contacts={aggregate['contact_count']} "
        f"return={aggregate['mean_episode_return']:.4f}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    missing = [
        name
        for name in (
            "low_checkpoint",
            "low_checkpoint_sha256",
            "checkpoint",
            "checkpoint_sha256",
            "split",
            "output_dir",
        )
        if getattr(args, name) is None
    ]
    if missing:
        names = ", ".join("--" + name.replace("_", "-") for name in missing)
        parser.error("missing required arguments: " + names)
    launcher = AppLauncher(args)
    _ = launcher.app
    return run(args)


if __name__ == "__main__":
    try:
        result = main()
    except Exception:
        traceback.print_exc()
        result = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(result)
