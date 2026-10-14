"""Record what a trained checkpoint does, without Isaac's renderer.

Plays the policy's mean action from ADR's fixed-difficulty starts and saves,
for the first ``--episodes`` finished episodes per difficulty, every robot
body position and the ball position at every 50 Hz tick, plus the start
knobs and how the episode ended.  ``scripts/render_trajectories.py`` turns
the file into a video anywhere matplotlib runs.  Use this when Isaac
Sim's camera rendering is unavailable (it needs a working RTX/Vulkan
setup; this only needs the physics).

    python scripts/record_trajectories.py --headless --control velocity \
        --checkpoint RUN/model_iteration_3000.pt --difficulty 0.25 --difficulty 1 --out traj.npz
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
    parser.add_argument("--episodes", type=int, default=6, help="finished episodes kept per difficulty")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app

    import numpy as np
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from robot_soccer.adr import PARAMETERS, AutomaticDomainRandomization
    from robot_soccer.config import TRAINING_BY_CONTROL
    from robot_soccer.environment import OUTCOMES
    from robot_soccer.runtime import load_high_policy, make_environment, policy_module, runner_config

    adr = AutomaticDomainRandomization(seed=args.seed, probe_fraction=0.0, evaluation_fraction=0.0)
    env, adapter, _ = make_environment(
        control=args.control, num_envs=args.num_envs, seed=args.seed, device=args.device, reset_sampler=adr
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
    origins = env.env_origins

    saved: dict[str, np.ndarray] = {}
    index = []
    for difficulty in args.difficulty:
        for parameter, (low, high) in zip(PARAMETERS, adr.difficulty_ranges(difficulty).tolist()):
            adr.set_range(parameter.name, low, high)
        observation = adapter.reset()
        bodies: list[list[np.ndarray]] = [[] for _ in range(env.num_envs)]
        balls: list[list[np.ndarray]] = [[] for _ in range(env.num_envs)]
        kept = 0
        with torch.inference_mode():
            while kept < args.episodes:
                # The state before the step: the environment resets a
                # finished slot inside step, so the post-step state of a
                # terminal slot is already the next episode's start.
                body_xyz = (env.robot.data.body_pos_w - origins[:, None, :]).cpu().numpy()
                ball_xyz = (env.ball.data.root_pos_w - origins).cpu().numpy()
                values = env.reset_tags["adr_values"].cpu().numpy()
                sides = env.goal_side.cpu().numpy()
                for slot in range(env.num_envs):
                    bodies[slot].append(body_xyz[slot])
                    balls[slot].append(ball_xyz[slot])
                observation, _, done, extras = adapter.step(policy.act_inference(observation))
                done_ids = torch.nonzero(done).flatten().tolist()
                for row, slot in enumerate(done_ids):
                    if kept < args.episodes:
                        key = f"{args.control}_d{difficulty:g}_e{kept}"
                        saved[key + "_bodies"] = np.stack(bodies[slot]).astype(np.float32)
                        saved[key + "_ball"] = np.stack(balls[slot]).astype(np.float32)
                        index.append({
                            "key": key,
                            "control": args.control,
                            "difficulty": difficulty,
                            "knobs": dict(zip([p.name for p in PARAMETERS], values[slot].tolist())),
                            "goal_side": int(sides[slot]),
                            "outcome": OUTCOMES[int(extras["episodes"]["outcome"][row])],
                            "seconds": int(extras["episodes"]["episode_native_ticks"][row]) * 0.02,
                        })
                        kept += 1
                    bodies[slot] = []
                    balls[slot] = []
        print(f"[trajectories] difficulty={difficulty} kept={kept}", flush=True)
    saved["index_json"] = np.frombuffer(
        json.dumps({"body_names": list(env.robot.body_names), "episodes": index}).encode(), dtype=np.uint8
    )
    np.savez_compressed(args.out, **saved)
    print(f"[trajectories] wrote {args.out} episodes={len(index)}", flush=True)
    env.close()
    app.close()


if __name__ == "__main__":
    main()
