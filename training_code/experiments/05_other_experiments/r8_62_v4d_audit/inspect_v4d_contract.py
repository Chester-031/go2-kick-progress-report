#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def sites(lines, pattern):
    rx = re.compile(pattern, re.I)
    return [{"line": i, "text": line.strip()} for i, line in enumerate(lines, 1) if rx.search(line)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--contract", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = Path(a.source).resolve()
    text = src.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    c = json.loads(Path(a.contract).read_text(encoding="utf-8"))

    checks = {
        "high_17d": c["interfaces"]["high"]["obs"] == 17,
        "high_3d": c["interfaces"]["high"]["action"] == 3,
        "low_62d": c["interfaces"]["delta_pose_low"]["obs"] == 62,
        "low_12d": c["interfaces"]["delta_pose_low"]["action"] == 12,
        "sparse_zero_sum": "+1" in text or "own_goal" in text,
        "low_env_present": "low_env" in text,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V4D contract/source checks failed: {checks}")

    result = {
        "version": "r8_62_v4d_audit",
        "training": False,
        "parent_mutation": False,
        "source": str(src),
        "contract_checks": checks,
        "sites": {
            "low_env_step": sites(lines, r"\blow_env\.step\s*\("),
            "high_env_step": sites(lines, r"\bhigh_env\.step\s*\("),
            "reset": sites(lines, r"reset|reset_idx|reset_robot|reset_base"),
            "ball_init": sites(lines, r"ball.*(pos|pose|spawn|reset)|ball_pos|ball_xy"),
            "goal_init": sites(lines, r"goal.*(pos|pose)|goal_pos|target_goal"),
            "curriculum": sites(lines, r"curriculum|difficulty|stage.*reset|progressive"),
            "reward": sites(lines, r"reward|goal_rate|own_goal|scoring"),
            "agent_row": sites(lines, r"env_ids|agent_ids|num_envs|episode"),
            "command_mapping": sites(lines, r"terminal|velocity_commands|delta_pose|pose_error"),
        },
        "required_followup": [
            "verify reset distribution is full-support rather than progressive curriculum",
            "verify ball and goal are represented in the 17D high observation",
            "verify single-agent rows are not mixed in replay/rollout storage",
            "verify only valid simulator states are masked, not all no-contact failures",
            "run coverage over robot pose, ball pose, distance, and yaw before training",
        ],
    }
    out = Path(a.out)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("R862_V4D_CONTRACT_AUDIT_OK")
    print("TRAINING=FALSE")
    print("PARENT_MUTATION=FALSE")
    print("LOW_STEP_COUNT=" + str(len(result["sites"]["low_env_step"])))
    print("RESET_SITE_COUNT=" + str(len(result["sites"]["reset"])))
    print("CURRICULUM_SITE_COUNT=" + str(len(result["sites"]["curriculum"])))
    print("AUDIT_JSON=" + str(out))


if __name__ == "__main__":
    main()
