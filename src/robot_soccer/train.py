"""Train the soccer policy with automatic domain randomization (ADR) over
where episodes start.

``--control`` picks what the policy outputs: ``position`` or ``velocity``
targets for the 12 joints every 50 Hz tick (one policy learns to walk and
to play), or ``hierarchical``, a 10 Hz body-velocity command executed by a
frozen low-level policy (needs ``--low-checkpoint``).  Each control has its
own PPO settings (``config.TRAINING_BY_CONTROL``); ``--steps-per-env``,
``--iterations`` and ``--checkpoint-interval`` override them.

Each PPO iteration collects ``--steps-per-env`` policy steps in every
environment, feeds every finished probe episode to ADR as it finishes (so
the knob ranges move during the rollout), then runs one PPO update.  ADR
(``adr.py``) chooses every episode start: most from the current ranges,
``--adr-probe-fraction`` as probes of one range edge, and
``--adr-evaluation-fraction`` at fixed difficulties whose goal rate
compares across runs.

Outputs in ``--output-dir`` (new or empty): ``run_info.json`` and one line
per iteration in ``metrics.jsonl`` (plot them with
``scripts/plot_training.py``), TensorBoard scalars, checkpoints every
``--checkpoint-interval`` iterations and at the end (policy, optimizer,
iteration and ADR state), and ``training_summary.json``.

Start from fresh weights, from ``--initial-checkpoint`` (weights only), or
continue a run with ``--resume`` (weights, optimizer, ADR state and the
iteration count; write to a new output directory).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping

from .config import NUM_ENVS, SEED, TRAINING_BY_CONTROL

# PPO settings that can be overridden from the command line (the control's
# TrainingSettings supplies the rest and the defaults).
PPO_OVERRIDES = (
    ("learning_rate", float),
    ("schedule", str),
    ("desired_kl", float),
    ("lam", float),
    ("entropy_coef", float),
    ("num_learning_epochs", int),
    ("num_mini_batches", int),
    ("clip_param", float),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--control", choices=tuple(TRAINING_BY_CONTROL), default="position")
    parser.add_argument("--low-checkpoint", type=Path, help="hierarchical control only")
    parser.add_argument("--low-checkpoint-sha256", help="hierarchical control only")
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--initial-checkpoint", type=Path, help="start from these high-policy weights")
    start.add_argument("--resume", type=Path, help="continue the run that wrote this checkpoint")
    parser.add_argument("--initial-checkpoint-sha256")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--steps-per-env", type=int, help="default: the control's setting")
    parser.add_argument("--iterations", type=int, help="last iteration to run (default: the control's setting)")
    parser.add_argument("--checkpoint-interval", type=int, help="default: the control's setting")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--adr-buffer-size", type=int, default=256, help="probe results per edge decision")
    parser.add_argument("--gamma", type=float, help="PPO and shaping discount (default: the control's)")
    for name, kind in PPO_OVERRIDES:
        parser.add_argument("--" + name.replace("_", "-"), type=kind, help="default: the control's setting")
    parser.add_argument("--adr-low-threshold", type=float, default=0.2)
    parser.add_argument("--adr-high-threshold", type=float, default=0.4)
    parser.add_argument(
        "--adr-lead-knob",
        default="robot_ball_distance_m",
        help="widen this knob alone until its upper edge reaches --adr-lead-until ('none': no lead)",
    )
    parser.add_argument("--adr-lead-until", type=float, default=4.0)
    parser.add_argument("--adr-probe-fraction", type=float, default=0.2)
    parser.add_argument("--adr-evaluation-fraction", type=float, default=0.05)
    parser.add_argument(
        "--adr-evaluation-difficulties",
        default="0.25,0.5,1.0",
        help="comma-separated fixed difficulties in [0,1]",
    )
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

    from .adr import AutomaticDomainRandomization
    from .metrics import TrainingMetrics, concatenate_episodes
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

    settings = TRAINING_BY_CONTROL[args.control]
    overrides = {name: getattr(args, name) for name in ("gamma", *(name for name, _ in PPO_OVERRIDES))}
    settings = dataclasses.replace(settings, **{name: value for name, value in overrides.items() if value is not None})
    if settings.schedule not in ("fixed", "adaptive"):
        raise ValueError("schedule must be fixed or adaptive")
    for name in ("steps_per_env", "iterations", "checkpoint_interval"):
        if getattr(args, name) is None:
            setattr(args, name, getattr(settings, name))
    if args.num_envs < settings.num_mini_batches or args.num_envs % settings.num_mini_batches != 0:
        raise ValueError(f"num-envs must be a positive multiple of {settings.num_mini_batches}")
    if args.steps_per_env < 2 or args.iterations < 1 or args.checkpoint_interval < 1:
        raise ValueError("steps-per-env must be >= 2 and iterations, checkpoint-interval >= 1")
    output = prepare_output_directory(args.output_dir)

    adr = AutomaticDomainRandomization(
        buffer_size=args.adr_buffer_size,
        low_threshold=args.adr_low_threshold,
        high_threshold=args.adr_high_threshold,
        probe_fraction=args.adr_probe_fraction,
        evaluation_fraction=args.adr_evaluation_fraction,
        evaluation_difficulties=[float(value) for value in args.adr_evaluation_difficulties.split(",")],
        seed=args.seed,
        lead_parameter=None if args.adr_lead_knob == "none" else args.adr_lead_knob,
        lead_until=args.adr_lead_until,
    )
    resumed: Mapping[str, Any] | None = None
    if args.resume is not None:
        # ADR must hold the resumed ranges before the environment draws its
        # first starts, which happens when it is constructed.
        resumed = torch.load(args.resume, map_location="cpu", weights_only=True)
        adr.load_state_dict(resumed["adr_state"])
    first_iteration = 1 if resumed is None else int(resumed["iter"]) + 1
    if first_iteration > args.iterations:
        raise ValueError(f"the resumed run is already at iteration {first_iteration - 1}")

    env, adapter, low_policy = make_environment(
        control=args.control,
        low_checkpoint=args.low_checkpoint,
        low_checkpoint_sha256=args.low_checkpoint_sha256,
        num_envs=args.num_envs,
        seed=args.seed,
        device=args.device,
        reset_sampler=adr,
        gamma=settings.gamma,
    )
    torch.manual_seed(args.seed)
    low_before = clone_module_state(low_policy) if low_policy is not None else None
    runner = OnPolicyRunner(
        adapter,
        runner_config(
            settings=settings,
            device=args.device,
            iterations=args.iterations,
            num_steps=args.steps_per_env,
            seed=args.seed,
        ),
        log_dir=None,
        device=args.device,
    )
    policy = policy_module(runner)
    optimizer = runner.alg.optimizer
    if args.resume is not None:
        load_high_policy(policy, args.resume)
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
    elif args.initial_checkpoint is not None:
        load_high_policy(
            policy,
            args.initial_checkpoint,
            expected_checkpoint_sha256=args.initial_checkpoint_sha256,
        )
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
    metrics = TrainingMetrics(
        output,
        run_info={
            "arguments": {name: str(value) for name, value in vars(args).items()},
            "first_iteration": first_iteration,
            "ppo_gamma": settings.gamma,
            "ppo_settings": dataclasses.asdict(settings),
            "transitions_per_iteration": args.num_envs * args.steps_per_env,
        },
        tensorboard=not args.no_tensorboard,
    )
    checkpoints: list[str] = []
    updates_per_iteration = settings.num_learning_epochs * settings.num_mini_batches

    try:
        for iteration in range(first_iteration, args.iterations + 1):
            episode_batches: list[dict[str, Any]] = []
            optimizer_before = optimizer_steps
            env_seconds = 0.0
            rollout_started = time.perf_counter()
            with torch.inference_mode():
                for _ in range(args.steps_per_env):
                    action = runner.alg.act(observation)
                    step_started = time.perf_counter()
                    observation, reward, done, extras = adapter.step(action.to(adapter.device))
                    env_seconds += time.perf_counter() - step_started
                    observation = observation.to(args.device)
                    reward = reward.to(args.device)
                    done = done.to(args.device)
                    runner.alg.process_env_step(observation, reward, done, extras)
                    episode_batches.append(extras["episodes"])
                    adr.record(extras["episodes"])
            rollout_seconds = time.perf_counter() - rollout_started

            update_started = time.perf_counter()
            if settings.rnn_hidden_dim:
                # compute_returns runs the critic once more, which advances a
                # recurrent policy's hidden state; keep the rollout's.
                actor_hidden = clone_hidden_state(policy.memory_a.hidden_state)
                critic_hidden = clone_hidden_state(policy.memory_c.hidden_state)
                if actor_hidden is None or critic_hidden is None:
                    raise RuntimeError("recurrent hidden state is absent")
            # compute_returns rebinds advantages; no_grad keeps a normal tensor
            # with a version counter, unlike inference_mode.
            with torch.no_grad():
                runner.alg.compute_returns(observation)
            if settings.rnn_hidden_dim:
                restore_hidden_state(policy.memory_a, actor_hidden)
                restore_hidden_state(policy.memory_c, critic_hidden)

            losses = runner.alg.update()
            if not _losses_finite(losses):
                raise RuntimeError("PPO returned a missing or non-finite loss")
            if optimizer_steps - optimizer_before != updates_per_iteration:
                raise RuntimeError(f"PPO update must perform exactly {updates_per_iteration} optimizer steps")
            update_seconds = time.perf_counter() - update_started

            transitions = args.num_envs * args.steps_per_env
            record = metrics.write(
                iteration=iteration,
                episodes=concatenate_episodes(episode_batches),
                losses=losses,
                adr=adr.take_period_metrics(),
                timing={
                    "rollout_seconds": rollout_seconds,
                    "env_step_seconds": env_seconds,
                    "policy_seconds": rollout_seconds - env_seconds,
                    "update_seconds": update_seconds,
                    "transitions_per_second": transitions / (rollout_seconds + update_seconds),
                },
            )
            if iteration % args.checkpoint_interval == 0 or iteration == args.iterations:
                name = f"model_iteration_{iteration:04d}.pt"
                save_high_policy(
                    output / name,
                    policy=policy,
                    optimizer=optimizer,
                    iteration=iteration,
                    adr_state=adr.state_dict(),
                )
                checkpoints.append(name)
            knobs = " ".join(
                f"{name}=[{values['low']:.2f},{values['high']:.2f}]"
                for name, values in record["adr"]["knobs"].items()
            )
            evaluation = " ".join(
                f"d{difficulty}={values['goal_rate']:.3f}"
                for difficulty, values in record.get("evaluation", {}).items()
            )
            print(
                f"iteration={iteration} episodes={record['completed_episodes']} "
                f"goal_rate={record.get('goal_rate', 0.0):.4f} {evaluation} {knobs} "
                f"sps={record['timing']['transitions_per_second']:.0f}",
                flush=True,
            )
    finally:
        metrics.close()

    policy_after = clone_module_state(policy)
    if low_policy is not None and not module_state_equal(low_before, clone_module_state(low_policy)):
        raise RuntimeError("frozen low policy changed during high-policy training")
    if not _component_changed(policy_before, policy_after, ("memory_a.", "actor.", "std")):
        raise RuntimeError("actor parameters did not update")
    if not _component_changed(policy_before, policy_after, ("memory_c.", "critic.")):
        raise RuntimeError("critic parameters did not update")
    if not all(torch.isfinite(value).all() for value in policy_after.values()):
        raise RuntimeError("trained high policy contains a non-finite value")

    summary = {
        "status": "COMPLETE",
        "control": args.control,
        "first_iteration": first_iteration,
        "last_iteration": args.iterations,
        "num_envs": args.num_envs,
        "steps_per_env": args.steps_per_env,
        "optimizer_steps": optimizer_steps,
        "adr_final_ranges": adr.ranges().tolist(),
        "frozen_low_unchanged": True,
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
    required = ["output_dir"]
    if args.control == "hierarchical":
        required += ["low_checkpoint", "low_checkpoint_sha256"]
    missing = [name for name in required if getattr(args, name) is None]
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
