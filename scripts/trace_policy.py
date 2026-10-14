"""Play a trained checkpoint deterministically and check its physics.

Fast goals can be real skill or a simulator exploit (a ball launched by
interpenetration, a robot flinging itself).  This runs the policy's mean
action at fixed ADR difficulties and records, per finished episode, the
outcome, duration, start geometry, the peak ball speed, peak robot base
speed and peak robot-ball contact force, then prints their distribution.
For reference, a Go2 runs at most about 3.5-5 m/s and a hard kick from a
small quadruped sends a size-4 ball at a few m/s.

    python scripts/trace_policy.py --headless --control velocity \
        --checkpoint RUN/model_iteration_1200.pt --difficulty 0.5 --difficulty 1 --out trace.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", choices=("position", "velocity"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--difficulty", type=float, action="append", required=True)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app

    import torch
    from rsl_rl.runners import OnPolicyRunner

    from robot_soccer.adr import PARAMETERS, AutomaticDomainRandomization
    from robot_soccer.config import TRAINING_BY_CONTROL
    from robot_soccer.environment import OUTCOMES
    from robot_soccer.metrics import initial_geometry
    from robot_soccer.runtime import load_high_policy, make_environment, policy_module, runner_config

    adr = AutomaticDomainRandomization(seed=1, probe_fraction=0.0, evaluation_fraction=0.0)
    env, adapter, _ = make_environment(
        control=args.control, num_envs=args.num_envs, seed=1, device=args.device, reset_sampler=adr
    )
    settings = TRAINING_BY_CONTROL[args.control]
    runner = OnPolicyRunner(
        adapter,
        runner_config(settings=settings, device=args.device, iterations=1, num_steps=settings.steps_per_env),
        log_dir=None,
        device=args.device,
    )
    policy = policy_module(runner)
    load_high_policy(policy, args.checkpoint)
    policy.eval()

    report = {}
    for difficulty in args.difficulty:
        ranges = adr.difficulty_ranges(difficulty)
        for parameter, (low, high) in zip(PARAMETERS, ranges.tolist()):
            adr.set_range(parameter.name, low, high)
        observation = adapter.reset()
        peak_ball = torch.zeros(env.num_envs, device=env.device)
        peak_robot = torch.zeros(env.num_envs, device=env.device)
        rows = []
        with torch.inference_mode():
            for _ in range(args.steps):
                action = policy.act_inference(observation)
                observation, _, done, extras = adapter.step(action)
                peak_ball = torch.maximum(peak_ball, torch.linalg.vector_norm(env.ball.data.root_lin_vel_w[:, :2], dim=1))
                peak_robot = torch.maximum(peak_robot, torch.linalg.vector_norm(env.robot.data.root_lin_vel_w[:, :2], dim=1))
                done_ids = torch.nonzero(done).flatten()
                episodes = extras["episodes"]
                if done_ids.numel():
                    geometry = initial_geometry(episodes)
                    for index, env_id in enumerate(done_ids.tolist()):
                        rows.append({
                            "outcome": OUTCOMES[int(episodes["outcome"][index])],
                            "seconds": int(episodes["episode_native_ticks"][index]) * 0.02,
                            "ball_goal_m": float(geometry["ball_goal_m"][index]),
                            "robot_ball_m": float(geometry["robot_ball_m"][index]),
                            "facing_deg": float(geometry["facing_deg"][index]),
                            "bearing_deg": float(geometry["bearing_deg"][index]),
                            "peak_ball_speed": float(peak_ball[env_id]),
                            "peak_robot_speed": float(peak_robot[env_id]),
                            "peak_contact_n": float(episodes["maximum_contact_force_n"][index]),
                        })
                    peak_ball[done_ids] = 0.0
                    peak_robot[done_ids] = 0.0
        goals = [row for row in rows if row["outcome"] == "goal"]

        def quantiles(values: list[float]) -> list[float]:
            ordered = sorted(values)
            return [round(ordered[int(q * (len(ordered) - 1))], 3) for q in (0.1, 0.5, 0.9, 1.0)] if ordered else []

        summary = {
            "episodes": len(rows),
            "outcomes": {name: sum(row["outcome"] == name for row in rows) for name in OUTCOMES},
            "goal_seconds_q10_50_90_max": quantiles([row["seconds"] for row in goals]),
            "goal_ball_goal_m_q": quantiles([row["ball_goal_m"] for row in goals]),
            "goal_mean_ball_speed_needed_q": quantiles([row["ball_goal_m"] / max(row["seconds"], 0.02) for row in goals]),
            "peak_ball_speed_q": quantiles([row["peak_ball_speed"] for row in rows]),
            "peak_robot_speed_q": quantiles([row["peak_robot_speed"] for row in rows]),
            "peak_contact_n_q": quantiles([row["peak_contact_n"] for row in rows]),
        }
        report[str(difficulty)] = {"summary": summary, "episodes": rows}
        print(f"[trace] difficulty={difficulty} {json.dumps(summary)}", flush=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    env.close()
    app.close()


if __name__ == "__main__":
    main()
