"""Train the recurrent no-Kick high policy on the stationary Reference/Full mix."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping

from .config import (
    CHECKPOINT_ITERATIONS,
    NUM_ENVS,
    NUM_MINI_BATCHES,
    ROLLOUT_STEPS,
    SEED,
    TRAINING_ITERATIONS,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-checkpoint", type=Path)
    parser.add_argument("--low-checkpoint-sha256")
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--initial-checkpoint-sha256")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--task", default="Isaac-Go2-Soccer-Milestone2-v0")
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--steps-per-env", type=int, default=ROLLOUT_STEPS)
    parser.add_argument("--iterations", type=int, default=TRAINING_ITERATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-tensorboard", action="store_true")
    return parser


def _losses_finite(losses: Mapping[str, Any]) -> bool:
    if not losses:
        return False
    for value in losses.values():
        scalar = float(value.detach().cpu().item()) if hasattr(value, "detach") else float(value)
        if not math.isfinite(scalar):
            return False
    return True


def _component_changed(before, after, prefixes: tuple[str, ...]) -> bool:
    names = [name for name in before if name.startswith(prefixes)]
    return bool(names) and any(not after[name].equal(before[name]) for name in names)


def _validate_training_dimensions(
    num_envs: int,
    steps_per_env: int,
    iterations: int,
) -> None:
    if num_envs < NUM_MINI_BATCHES or num_envs % NUM_MINI_BATCHES != 0:
        raise ValueError(
            f"num-envs must be a positive multiple of {NUM_MINI_BATCHES}"
        )
    if steps_per_env < 2 or iterations <= 0:
        raise ValueError("steps-per-env must be at least 2 and iterations must be positive")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> int:
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from .metrics import TrainingMetrics
    from .recurrent import clone_hidden_state, restore_hidden_state
    from .runtime import (
        clone_module_state,
        load_high_policy,
        make_environment,
        module_state_equal,
        policy_module,
        prepare_output_directory,
        runner_config,
        save_high_policy,
    )

    _validate_training_dimensions(
        args.num_envs,
        args.steps_per_env,
        args.iterations,
    )
    output = prepare_output_directory(args.output_dir)

    env, adapter, low_policy = make_environment(
        low_checkpoint=args.low_checkpoint,
        low_checkpoint_sha256=args.low_checkpoint_sha256,
        num_envs=args.num_envs,
        seed=args.seed,
        device=args.device,
        task=args.task,
    )
    torch.manual_seed(args.seed)
    low_before = clone_module_state(low_policy)
    runner = OnPolicyRunner(
        adapter,
        runner_config(device=args.device, num_steps=args.steps_per_env, seed=args.seed),
        log_dir=None,
        device=args.device,
    )
    policy = policy_module(runner)
    optimizer = runner.alg.optimizer
    if optimizer.state:
        raise RuntimeError("new optimizer unexpectedly contains state")
    load_high_policy(
        policy,
        args.initial_checkpoint,
        expected_checkpoint_sha256=args.initial_checkpoint_sha256,
        require_pre_action_initial=True,
    )
    if optimizer.state:
        raise RuntimeError("loading high weights populated the fresh optimizer")
    policy_before = clone_module_state(policy)

    optimizer_steps = 0
    original_optimizer_step = optimizer.step

    def counted_step(*values: Any, **kwargs: Any):
        nonlocal optimizer_steps
        optimizer_steps += 1
        return original_optimizer_step(*values, **kwargs)

    optimizer.step = counted_step
    observation = adapter.get_observations().to(args.device)
    policy.train()
    metrics = TrainingMetrics(output, tensorboard=not args.no_tensorboard)
    checkpoints: list[str] = []

    try:
        for iteration in range(1, args.iterations + 1):
            events: list[dict[str, Any]] = []
            optimizer_before = optimizer_steps
            with torch.inference_mode():
                for _ in range(args.steps_per_env):
                    action = runner.alg.act(observation)
                    if not torch.isfinite(action).all():
                        raise RuntimeError("policy produced a non-finite action")
                    observation, reward, done, extras = adapter.step(action.to(adapter.device))
                    observation = observation.to(args.device)
                    reward = reward.to(args.device)
                    done = done.to(args.device)
                    runner.alg.process_env_step(observation, reward, done, extras)
                    events.extend(extras["terminal_events"])

            actor_hidden = clone_hidden_state(policy.memory_a.hidden_state)
            critic_hidden = clone_hidden_state(policy.memory_c.hidden_state)
            if actor_hidden is None or critic_hidden is None:
                raise RuntimeError("recurrent hidden state is absent")
            # compute_returns rebinds advantages; no_grad keeps a normal tensor
            # with a version counter, unlike inference_mode.
            with torch.no_grad():
                runner.alg.compute_returns(observation)
            restore_hidden_state(policy.memory_a, actor_hidden)
            restore_hidden_state(policy.memory_c, critic_hidden)

            losses = runner.alg.update()
            if not _losses_finite(losses):
                raise RuntimeError("PPO returned a missing or non-finite loss")
            if optimizer_steps - optimizer_before != 20:
                raise RuntimeError("PPO update must perform exactly 20 optimizer steps")

            record = metrics.write(iteration=iteration, events=events, losses=losses)
            if iteration in CHECKPOINT_ITERATIONS or iteration == args.iterations:
                name = f"model_iteration_{iteration:02d}.pt"
                save_high_policy(
                    output / name,
                    policy=policy,
                    optimizer=optimizer,
                    iteration=iteration,
                )
                checkpoints.append(name)
            print(
                f"iteration={iteration} episodes={record['completed_episodes']} "
                f"return={record['mean_episode_return']:.4f} "
                f"goal_rate={record['goal_rate']:.4f} "
                f"contact_rate={record['contact_rate']:.4f}",
                flush=True,
            )
    finally:
        metrics.close()

    policy_after = clone_module_state(policy)
    low_after = clone_module_state(low_policy)
    if not module_state_equal(low_before, low_after):
        raise RuntimeError("frozen low policy changed during high-policy training")
    if not _component_changed(policy_before, policy_after, ("memory_a.", "actor.")):
        raise RuntimeError("actor parameters did not update")
    if not _component_changed(policy_before, policy_after, ("memory_c.", "critic.")):
        raise RuntimeError("critic parameters did not update")
    if not all(torch.isfinite(value).all() for value in policy_after.values()):
        raise RuntimeError("trained high policy contains a non-finite value")

    summary = {
        "status": "COMPLETE",
        "iterations": args.iterations,
        "num_envs": args.num_envs,
        "steps_per_env": args.steps_per_env,
        "optimizer_steps": optimizer_steps,
        "stationary_reference_fraction": 0.75,
        "stationary_full_fraction": 0.25,
        "frozen_low_unchanged": True,
        "actor_updated": True,
        "critic_updated": True,
        "checkpoints": checkpoints,
        "metrics": "metrics.jsonl",
    }
    _write_json(output / "training_summary.json", summary)
    print(f"training=complete output={output}", flush=True)
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
            "initial_checkpoint",
            "initial_checkpoint_sha256",
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
