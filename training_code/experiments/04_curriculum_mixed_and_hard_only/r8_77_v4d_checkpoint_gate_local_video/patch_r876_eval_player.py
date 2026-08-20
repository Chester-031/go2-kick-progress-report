#!/usr/bin/env python3
"""Force R8.76 local video evaluation onto stationary full-support resets."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    text = args.source.read_text(encoding="utf-8")
    marker = "[R876_EVAL_DISTRIBUTION]"
    if marker in text:
        print("R877_PLAYER_FULL_SUPPORT_PATCH=ALREADY_PRESENT")
        return

    old = """    env.set_teacher_scale(0.0)
    env.reset()

    high_adapter = V4DHighLevelEnvAdapter(env)
"""
    new = """    env.set_teacher_scale(0.0)
    if os.environ.get(\"R876_EVAL_FULL_SUPPORT\") == \"1\":
        env.r876_iteration = int(
            os.environ.get(\"R876_EVAL_CURRICULUM_ITERATION\", \"599\")
        )
        phase, difficulty = env._r876_phase()
        if phase != \"full_support\" or difficulty != 1.0:
            raise RuntimeError(
                \"R8.76 evaluation did not select full-support reset.\"
            )
        print(
            \"[R876_EVAL_DISTRIBUTION] \"
            \"phase=full_support difficulty=1.0 \"
            \"robot_ball_sampling=independent_uniform_full_yaw\",
            flush=True,
        )
    env.reset()

    high_adapter = V4DHighLevelEnvAdapter(env)
"""
    if text.count(old) != 1:
        raise SystemExit(
            f"R877_PLAYER_PATCH_FATAL=target_count:{text.count(old)}"
        )
    args.source.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("R877_PLAYER_FULL_SUPPORT_PATCH=OK")


if __name__ == "__main__":
    main()
