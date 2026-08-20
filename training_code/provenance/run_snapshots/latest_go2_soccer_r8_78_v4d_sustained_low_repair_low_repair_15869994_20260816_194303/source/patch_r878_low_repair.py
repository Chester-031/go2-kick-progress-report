#!/usr/bin/env python3
"""Generate the R8.78 low-only repair trainer from the frozen R8.76 source."""

from __future__ import annotations

import argparse
from pathlib import Path


REPAIR_FUNCTIONS = r'''
R878_PRIMITIVE_NAMES = (
    "stand", "forward", "backward", "left",
    "right", "turn_left", "turn_right", "diagonal",
)


def _r878_actions(
    env: HighLevelSoccerVecEnv,
    global_high_step: int,
    phase: str,
    *,
    fixed_groups: bool = False,
) -> torch.Tensor:
    vectors = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.85, 0.0, 0.0],
            [-0.75, 0.0, 0.0],
            [0.0, 0.85, 0.0],
            [0.0, -0.85, 0.0],
            [0.0, 0.0, 0.85],
            [0.0, 0.0, -0.85],
            [0.70, 0.45, 0.45],
        ],
        device=env.device,
    )
    groups = torch.arange(env.num_envs, device=env.device) % len(vectors)
    if not fixed_groups:
        block = global_high_step // 30
        if phase == "chained":
            groups = (groups + block) % len(vectors)
        elif phase == "mixed":
            groups = (groups + 3 * block) % len(vectors)
            sign = torch.where(
                ((torch.arange(env.num_envs, device=env.device) + block) % 2) == 0,
                1.0,
                -1.0,
            )
            vectors = vectors.clone()
            vectors[7, 1] = 0.55 * sign[0]
            vectors[7, 2] = 0.55 * sign[0]
    return vectors[groups]


def _r878_average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _r878_sustained_evaluation(
    *,
    low_runner: Any,
    env: HighLevelSoccerVecEnv,
    low_adapter: PoseLowLevelEnvAdapter,
) -> dict[str, Any]:
    policy_module(low_runner).eval()
    accum: dict[str, dict[str, dict[str, list[float]]]] = {
        name: {
            "early": {
                "speed": [], "vx": [], "vy": [], "yaw_rate": [],
                "upright": [], "base_done": [],
            },
            "late": {
                "speed": [], "vx": [], "vy": [], "yaw_rate": [],
                "upright": [], "base_done": [],
            },
        }
        for name in R878_PRIMITIVE_NAMES
    }
    # The preceding rollout may leave inference tensors in environment
    # diagnostics. Reset them in the same mode to avoid PyTorch's protected
    # in-place-update failure at the train/eval boundary.
    with torch.inference_mode():
        env.reset()
        low_obs = low_adapter.get_observations().to(low_runner.device)
    with torch.no_grad():
        for high_step in range(240):
            within = high_step % 60
            if high_step > 0 and within == 0:
                with torch.inference_mode():
                    env.reset()
                    low_obs = low_adapter.get_observations().to(low_runner.device)
            high_actions = _r878_actions(
                env, high_step, "evaluation", fixed_groups=True
            )
            env.begin_high_action(high_actions)
            for low_index in range(env.low_level_steps_per_action):
                low_actions = policy_module(low_runner).act_inference(low_obs)
                low_obs, _, low_dones, _, diag = env.low_step(
                    low_actions.to(env.device),
                    is_last_low_step=(
                        low_index == env.low_level_steps_per_action - 1
                    ),
                )
                low_obs = low_obs.to(low_runner.device)
                window = "early" if within < 10 else ("late" if within >= 50 else "")
                if not window:
                    continue
                state = env._state()
                robot_speed = torch.linalg.vector_norm(
                    state["robot_velocity_body"][:, :2], dim=1
                )
                velocity = state["robot_velocity_body"][:, :2]
                yaw_rate = state["robot_angular_velocity_body"][:, 2]
                upright = env._native_low_obs[:, 6:8].square().sum(dim=1)
                groups = torch.arange(env.num_envs, device=env.device) % len(R878_PRIMITIVE_NAMES)
                for index, name in enumerate(R878_PRIMITIVE_NAMES):
                    mask = groups == index
                    accum[name][window]["speed"].append(
                        float(robot_speed[mask].mean())
                    )
                    accum[name][window]["vx"].append(
                        float(velocity[mask, 0].mean())
                    )
                    accum[name][window]["vy"].append(
                        float(velocity[mask, 1].mean())
                    )
                    accum[name][window]["yaw_rate"].append(
                        float(yaw_rate[mask].mean())
                    )
                    accum[name][window]["upright"].append(
                        float(upright[mask].mean())
                    )
                    accum[name][window]["base_done"].append(
                        float(low_dones[mask].float().mean())
                    )
            env.end_high_action()
            low_obs = low_adapter.get_observations().to(low_runner.device)

    result: dict[str, Any] = {"cases": {}}
    pass_count = 0
    for name in R878_PRIMITIVE_NAMES:
        windows: dict[str, Any] = {}
        for window in ("early", "late"):
            windows[window] = {
                key: _r878_average(values)
                for key, values in accum[name][window].items()
            }
        early = windows["early"]
        late = windows["late"]
        if name == "stand":
            early_response = early["speed"]
            late_response = late["speed"]
            passed = late_response < 0.04 and late["upright"] < 0.03
        else:
            response_key, sign, threshold = {
                "forward": ("vx", 1.0, 0.08),
                "backward": ("vx", -1.0, 0.06),
                "left": ("vy", 1.0, 0.06),
                "right": ("vy", -1.0, 0.06),
                "turn_left": ("yaw_rate", 1.0, 0.20),
                "turn_right": ("yaw_rate", -1.0, 0.20),
                "diagonal": ("vx", 1.0, 0.06),
            }[name]
            early_response = sign * early[response_key]
            late_response = sign * late[response_key]
            retained = late_response >= 0.35 * max(early_response, threshold)
            passed = (
                late_response > threshold
                and retained
                and late["upright"] < 0.03
            )
        passed = bool(passed and late["base_done"] < 0.01)
        pass_count += int(passed)
        result["cases"][name] = {
            "early": windows["early"],
            "late": windows["late"],
            "early_response": early_response,
            "late_response": late_response,
            "passed": passed,
        }
    result["pass_count"] = pass_count
    result["total"] = len(R878_PRIMITIVE_NAMES)
    result["accepted"] = pass_count == len(R878_PRIMITIVE_NAMES)
    return result


def run_r878_low_repair(
    *,
    high_runner: Any,
    low_runner: Any,
    env: HighLevelSoccerVecEnv,
    high_adapter: V4DHighLevelEnvAdapter,
    low_adapter: PoseLowLevelEnvAdapter,
    iterations: int,
    log_dir: Path,
) -> None:
    del high_adapter
    high_before = snapshot_policy(high_runner)
    low_before = snapshot_policy(low_runner)
    pre_eval = _r878_sustained_evaluation(
        low_runner=low_runner, env=env, low_adapter=low_adapter
    )
    print("[R878_PRE_SUSTAINED] " + json.dumps(pre_eval, sort_keys=True), flush=True)

    set_optimizer_learning_rate(
        low_runner, float(os.environ.get("R878_LOW_LR", "0.00002"))
    )
    policy_module(high_runner).eval()
    policy_module(low_runner).train()
    low_obs = low_adapter.get_observations().to(low_runner.device)
    rollout_steps = int(low_runner.cfg["num_steps_per_env"])
    if rollout_steps % env.low_level_steps_per_action != 0:
        raise RuntimeError("R8.78 low rollout is not divisible by low steps per high action.")
    high_cycles = rollout_steps // env.low_level_steps_per_action
    global_high_step = 0
    records: list[dict[str, Any]] = []

    for iteration in range(iterations):
        phase = "basic" if iteration < 80 else ("chained" if iteration < 200 else "mixed")
        aggregate: dict[str, list[float]] = {
            "reward": [], "speed": [], "pose_error": [], "upright": [],
            "action_rate": [], "stall": [], "base_done": [],
        }
        with torch.inference_mode():
            for _ in range(high_cycles):
                actions = _r878_actions(env, global_high_step, phase)
                env.begin_high_action(actions)
                for low_index in range(env.low_level_steps_per_action):
                    low_actions = low_runner.alg.act(low_obs)
                    low_obs, reward, dones, extras, diag = env.low_step(
                        low_actions.to(env.device),
                        is_last_low_step=(
                            low_index == env.low_level_steps_per_action - 1
                        ),
                    )
                    low_obs = low_obs.to(low_runner.device)
                    reward = reward.to(low_runner.device)
                    dones = dones.to(low_runner.device)
                    low_runner.alg.process_env_step(low_obs, reward, dones, extras)
                    state = env._state()
                    robot_speed = torch.linalg.vector_norm(
                        state["robot_velocity_body"][:, :2], dim=1
                    )
                    pose_error = env._pose_error()
                    pose_error_xy = torch.sqrt(
                        pose_error[0].square() + pose_error[1].square() + 1.0e-8
                    )
                    upright = env._native_low_obs[:, 6:8].square().sum(dim=1)
                    moving = env.skill_id != env.SKILL_STAND
                    stalled = moving & (pose_error_xy > 0.08) & (robot_speed < 0.04)
                    moving_speed = robot_speed[moving]
                    moving_pose_error = pose_error_xy[moving]
                    aggregate["reward"].append(float(reward.mean()))
                    aggregate["speed"].append(
                        float(moving_speed.mean()) if moving_speed.numel() else 0.0
                    )
                    aggregate["pose_error"].append(
                        float(moving_pose_error.mean()) if moving_pose_error.numel() else 0.0
                    )
                    aggregate["upright"].append(float(upright.mean()))
                    aggregate["action_rate"].append(float(diag["action_rate"]))
                    aggregate["stall"].append(float(stalled.float().mean()))
                    aggregate["base_done"].append(float(dones.float().mean()))
                env.end_high_action()
                low_obs = low_adapter.get_observations().to(low_runner.device)
                global_high_step += 1

        low_runner.alg.compute_returns(low_obs)
        losses = low_runner.alg.update()
        low_runner.current_learning_iteration = iteration
        record = {
            "iteration": iteration,
            "phase": phase,
            **{key: _r878_average(values) for key, values in aggregate.items()},
            "losses": losses,
        }
        records.append(record)
        if iteration == 0 or (iteration + 1) % 10 == 0:
            print("[R878_LOW_ITER] " + json.dumps(record, sort_keys=True), flush=True)
        if iteration == 0 or (iteration + 1) % 50 == 0:
            save_checkpoint(
                runner=low_runner,
                path=log_dir / f"low_model_{iteration}.pt",
                iteration=iteration,
                role=f"r878_sustained_low_{phase}",
            )

    policy_module(low_runner).eval()
    post_eval = _r878_sustained_evaluation(
        low_runner=low_runner, env=env, low_adapter=low_adapter
    )
    high_delta = parameter_delta(high_before, high_runner)
    low_delta = parameter_delta(low_before, low_runner)
    low_final = log_dir / "low_model_final.pt"
    high_unchanged = log_dir / "high_model_unchanged.pt"
    save_checkpoint(
        runner=low_runner, path=low_final,
        iteration=iterations - 1, role="r878_sustained_low_final",
    )
    save_checkpoint(
        runner=high_runner, path=high_unchanged,
        iteration=iterations - 1, role="r878_high_unchanged_audit",
    )
    accepted = bool(
        post_eval["accepted"]
        and post_eval["pass_count"] > pre_eval["pass_count"]
        and low_delta["max_abs"] > 0.0
        and high_delta["max_abs"] == 0.0
    )
    summary = {
        "version": "r8_78_v4d_sustained_low_repair_v1",
        "training_target": "low_only",
        "high_trainable": False,
        "low_trainable": True,
        "stall_penalty": 2.0,
        "pre_sustained": pre_eval,
        "post_sustained": post_eval,
        "high_parameter_delta": high_delta,
        "low_parameter_delta": low_delta,
        "records": records,
        "accepted": accepted,
    }
    summary_path = log_dir / "r878_low_repair_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("[R878_POST_SUSTAINED] " + json.dumps(post_eval, sort_keys=True), flush=True)
    print(
        "[R878_PARAMETER_DELTA] "
        f"high_max_abs={high_delta['max_abs']:.9f} "
        f"low_max_abs={low_delta['max_abs']:.9f}",
        flush=True,
    )
    print(f"R878_LOW_CHECKPOINT={low_final}", flush=True)
    print(f"R878_SUMMARY={summary_path}", flush=True)
    if accepted:
        (log_dir / "R878_LOW_REPAIR_ACCEPTED").touch()
        print("R878_LOW_REPAIR_ACCEPTED", flush=True)
    else:
        print("R878_LOW_REPAIR_REJECTED", flush=True)
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = args.source.read_text(encoding="utf-8")

    reward_old = '''        reward = torch.clamp(
            reward,
            -3.0,
            4.0,
        )
        reward[base_done] = -2.0

        robot_speed = torch.linalg.vector_norm(
            after_state["robot_velocity_body"][:, :2],
            dim=1,
        )
        ball_speed = torch.linalg.vector_norm(
            after_state["ball_velocity_world"][:, :2],
            dim=1,
        )
'''
    reward_new = '''        robot_speed = torch.linalg.vector_norm(
            after_state["robot_velocity_body"][:, :2],
            dim=1,
        )
        ball_speed = torch.linalg.vector_norm(
            after_state["ball_velocity_world"][:, :2],
            dim=1,
        )

        # R8.78: a stable pose is not success while a non-stand Delta Pose
        # remains unresolved. Make the absorbing zero-speed solution negative.
        motion_requested = (~stand_mask) & (after_xy > 0.08)
        sustained_stall = motion_requested & (robot_speed < 0.04)
        reward = reward - 2.0 * sustained_stall.float()

        reward = torch.clamp(
            reward,
            -3.0,
            4.0,
        )
        reward[base_done] = -2.0
'''
    if text.count(reward_old) != 1:
        raise SystemExit(
            f"R878_PATCH_FATAL=reward_target_count:{text.count(reward_old)}"
        )
    text = text.replace(reward_old, reward_new, 1)

    insertion = "\ndef run_v4d_training(\n"
    if text.count(insertion) != 1:
        raise SystemExit(
            f"R878_PATCH_FATAL=training_insertion_count:{text.count(insertion)}"
        )
    text = text.replace(insertion, REPAIR_FUNCTIONS + insertion, 1)

    call_old = '''        run_v4d_training(
            high_runner=high_runner,
            low_runner=low_runner,
            env=env,
            high_adapter=high_adapter,
            low_adapter=low_adapter,
            iterations=args_cli.max_iterations,
            log_dir=log_dir,
        )
'''
    call_new = '''        if os.environ.get("R878_SUSTAINED_LOW_REPAIR") == "1":
            run_r878_low_repair(
                high_runner=high_runner,
                low_runner=low_runner,
                env=env,
                high_adapter=high_adapter,
                low_adapter=low_adapter,
                iterations=args_cli.max_iterations,
                log_dir=log_dir,
            )
        else:
            run_v4d_training(
                high_runner=high_runner,
                low_runner=low_runner,
                env=env,
                high_adapter=high_adapter,
                low_adapter=low_adapter,
                iterations=args_cli.max_iterations,
                log_dir=log_dir,
            )
'''
    if text.count(call_old) != 1:
        raise SystemExit(
            f"R878_PATCH_FATAL=training_call_count:{text.count(call_old)}"
        )
    text = text.replace(call_old, call_new, 1)
    text = text.replace(
        f"checkpoint={{source_low_checkpoint}} frozen=true",
        f"checkpoint={{source_low_checkpoint}} frozen=false repair_mode=true",
        1,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print("R878_LOW_REPAIR_PATCH_OK")


if __name__ == "__main__":
    main()
