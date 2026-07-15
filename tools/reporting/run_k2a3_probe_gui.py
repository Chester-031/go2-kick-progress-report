#!/usr/bin/env python3
from __future__ import annotations

import argparse
import runpy
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import isaaclab.app as app_module


def parse_triplet(value: str) -> tuple[float, float, float]:
    values = tuple(float(item.strip()) for item in value.split(","))
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected X,Y,Z.")
    return values


def disable_debug_visualization(cfg: Any) -> None:
    for group_name in (
        "commands",
        "scene",
        "actions",
        "observations",
        "events",
        "rewards",
        "terminations",
        "curriculum",
    ):
        group = getattr(cfg, group_name, None)
        if group is None:
            continue
        for name in dir(group):
            if name.startswith("_"):
                continue
            try:
                value = getattr(group, name)
            except Exception:
                continue
            if hasattr(value, "debug_vis"):
                try:
                    value.debug_vis = False
                except Exception:
                    pass


class CameraPaceWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        eye_offset: tuple[float, float, float],
        target_offset: tuple[float, float, float],
        min_step_seconds: float,
    ) -> None:
        super().__init__(env)
        self.eye_offset = eye_offset
        self.target_offset = target_offset
        self.min_step_seconds = max(0.0, min_step_seconds)
        self.step_index = 0

    def _update_camera(self) -> None:
        base_env = self.env.unwrapped
        scene = getattr(base_env, "scene", None)
        sim = getattr(base_env, "sim", None)
        if scene is None or sim is None:
            return
        try:
            robot = scene["robot"]
            root = robot.data.root_pos_w[0].detach().cpu().tolist()
            eye = [
                float(root[0]) + self.eye_offset[0],
                float(root[1]) + self.eye_offset[1],
                float(root[2]) + self.eye_offset[2],
            ]
            target = [
                float(root[0]) + self.target_offset[0],
                float(root[1]) + self.target_offset[1],
                max(0.08, float(root[2]) + self.target_offset[2]),
            ]
            sim.set_camera_view(eye=eye, target=target)
        except Exception as error:
            if self.step_index == 0:
                print(f"[K2A3 CAMERA] warning={error}", flush=True)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        result = self.env.reset(*args, **kwargs)
        self._update_camera()
        print("[K2A3 CAMERA] reset_complete", flush=True)
        return result

    def step(self, action: Any) -> Any:
        started = time.perf_counter()
        result = self.env.step(action)
        self.step_index += 1
        self._update_camera()
        remaining = self.min_step_seconds - (time.perf_counter() - started)
        if remaining > 0.0:
            time.sleep(remaining)
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-script", required=True)
    parser.add_argument("--eye-offset", type=parse_triplet, default=(0.82, -0.82, 0.10))
    parser.add_argument("--target-offset", type=parse_triplet, default=(0.08, -0.03, -0.20))
    parser.add_argument("--resolution", default="1280,720")
    parser.add_argument("--min-step-seconds", type=float, default=0.05)
    args, target_args = parser.parse_known_args()

    if target_args and target_args[0] == "--":
        target_args = target_args[1:]

    target = Path(args.target_script).expanduser().resolve()
    if not target.is_file():
        raise SystemExit(f"Probe not found: {target}")

    width_text, height_text = args.resolution.split(",", 1)
    resolution = (int(width_text), int(height_text))

    original_launcher = app_module.AppLauncher
    original_make = gym.make

    class ForcedGuiAppLauncher(original_launcher):
        def __init__(self, launcher_args: Any = None, **kwargs: Any) -> None:
            if isinstance(launcher_args, dict):
                launcher_args = dict(launcher_args)
                launcher_args["headless"] = False
                launcher_args["enable_cameras"] = False
            elif launcher_args is not None:
                try:
                    setattr(launcher_args, "headless", False)
                    setattr(launcher_args, "enable_cameras", False)
                except Exception:
                    pass
            if "headless" in kwargs:
                kwargs["headless"] = False
            super().__init__(launcher_args, **kwargs)

    def patched_make(env_id: str, *make_args: Any, **make_kwargs: Any) -> gym.Env:
        cfg = make_kwargs.get("cfg")
        if cfg is not None:
            disable_debug_visualization(cfg)
            viewer = getattr(cfg, "viewer", None)
            if viewer is not None:
                viewer.eye = args.eye_offset
                viewer.lookat = args.target_offset
                viewer.origin_type = "env"
                viewer.env_index = 0
                viewer.resolution = resolution

        env = original_make(env_id, *make_args, **make_kwargs)
        print(f"[K2A3 CAMERA] wrapped_env={env_id}", flush=True)
        return CameraPaceWrapper(
            env,
            eye_offset=args.eye_offset,
            target_offset=args.target_offset,
            min_step_seconds=args.min_step_seconds,
        )

    app_module.AppLauncher = ForcedGuiAppLauncher
    gym.make = patched_make
    sys.argv = [str(target), *target_args]

    print(f"[K2A3 CAMERA] target={target}", flush=True)
    print(f"[K2A3 CAMERA] target_args={target_args}", flush=True)

    try:
        runpy.run_path(str(target), run_name="__main__")
    finally:
        app_module.AppLauncher = original_launcher
        gym.make = original_make

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
