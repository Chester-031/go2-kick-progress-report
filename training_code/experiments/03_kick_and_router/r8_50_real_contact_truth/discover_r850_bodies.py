#!/usr/bin/env python3
"""Discover exact Go2 rigid-body names before constructing filtered sensors."""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Go2-Soccer-Milestone2-v0")
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import isaac_go2_low_level_rl.tasks  # noqa: E402,F401
import isaac_go2_low_level_rl.tasks.manager_based.soccer  # noqa: E402,F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def main() -> None:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    try:
        cfg.observations.policy.enable_corruption = False
    except AttributeError:
        pass
    env = gym.make(args.task, cfg=cfg)
    try:
        scene = env.unwrapped.scene
        names = list(scene["robot"].data.body_names)
        if not names or "FL_foot" not in names:
            raise RuntimeError(f"Unexpected robot body list: {names}")
        result = {
            "task": args.task,
            "robot_prim": "{ENV_REGEX_NS}/Robot",
            "ball_prim": "{ENV_REGEX_NS}/SoccerBall",
            "body_names": names,
            "front_feet": ["FL_foot", "FR_foot"],
            "all_feet": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        }
        out = Path(args.output).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"R850_BODY_DISCOVERY_OK count={len(names)} output={out}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
