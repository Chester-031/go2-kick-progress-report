"""Record a trained checkpoint playing, as an MP4 per difficulty.

Plays the policy's mean action in one environment, starting every episode
from ADR's fixed-difficulty ranges, and records the scene camera (aimed
at the whole field by ``direct_env_cfg``) at 25 frames per second.  Each
frame is captioned with the difficulty, the episode's start knobs and,
held for half a second at its end, how the episode ended.  Needs a GPU
node and ``--enable_cameras``.

    python scripts/record_policy.py --headless --enable_cameras --control velocity \
        --checkpoint RUN/model_iteration_3000.pt --difficulty 0.25 --difficulty 1 --out-dir videos
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", choices=("position", "velocity"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--difficulty", type=float, action="append", required=True)
    parser.add_argument("--episodes", type=int, default=5, help="episodes per difficulty")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app

    import cv2
    import imageio
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from robot_soccer.adr import PARAMETERS, AutomaticDomainRandomization
    from robot_soccer.config import TRAINING_BY_CONTROL
    from robot_soccer.environment import OUTCOMES
    from robot_soccer.runtime import load_high_policy, make_environment, policy_module, runner_config

    adr = AutomaticDomainRandomization(seed=args.seed, probe_fraction=0.0, evaluation_fraction=0.0)
    env, adapter, _ = make_environment(
        control=args.control,
        num_envs=1,
        seed=args.seed,
        device=args.device,
        reset_sampler=adr,
        render_mode="rgb_array",
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
    scene = env.base_env
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def caption(frame, lines, color=(255, 255, 255)):
        frame = frame.copy()
        for row, text in enumerate(lines):
            position = (16, 36 + 30 * row)
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        return frame

    for difficulty in args.difficulty:
        for parameter, (low, high) in zip(PARAMETERS, adr.difficulty_ranges(difficulty).tolist()):
            adr.set_range(parameter.name, low, high)
        observation = adapter.reset()
        path = args.out_dir / f"{args.control}_difficulty_{difficulty:g}.mp4"
        writer = imageio.get_writer(path, fps=25, codec="libx264", quality=8, macro_block_size=8)
        outcomes = []
        tick = 0
        # The environment resets a finished slot inside step, so the frame
        # after the terminal step already shows the next start: hold the
        # last frame of the episode instead.
        last_frame = scene.render()
        with torch.inference_mode():
            while len(outcomes) < args.episodes:
                values = env.reset_tags["adr_values"][0].tolist()
                knobs = (f"ball-goal {values[0]:.2f} m  robot-ball {values[1]:.2f} m  "
                         f"facing {values[2]:.0f} deg  around {values[3]:.0f} deg")
                header = f"{args.control} control  difficulty {difficulty:g}  episode {len(outcomes) + 1}"
                action = policy.act_inference(observation)
                observation, _, done, extras = adapter.step(action)
                tick += 1
                frame = scene.render()
                if bool(done[0]):
                    outcome = OUTCOMES[int(extras["episodes"]["outcome"][0])]
                    seconds = int(extras["episodes"]["episode_native_ticks"][0]) * 0.02
                    outcomes.append(outcome)
                    color = (80, 255, 80) if outcome == "goal" else (255, 90, 90)
                    final = caption(last_frame, [header, knobs, f"{outcome.upper()} after {seconds:.2f} s"], color)
                    for _ in range(12):
                        writer.append_data(final)
                elif tick % 2 == 0:
                    writer.append_data(caption(frame, [header, knobs]))
                last_frame = frame
        writer.close()
        print(f"[record] {path} outcomes={outcomes}", flush=True)
    env.close()
    app.close()


if __name__ == "__main__":
    main()
