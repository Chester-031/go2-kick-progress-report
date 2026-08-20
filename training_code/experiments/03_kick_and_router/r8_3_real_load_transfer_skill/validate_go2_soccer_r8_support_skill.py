#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1]).resolve()
required = (
    "train_go2_soccer_r8_support_skill.py",
    "run_go2_soccer_r8_support_skill.sbatch",
    "install_submit_go2_soccer_r8_support_skill.sh",
    "check_go2_soccer_r8_support_skill_result.sh",
    "README.md",
)

errors = []
passed = []

for name in required:
    path = root / name
    if path.is_file() and path.stat().st_size > 0:
        passed.append("file:" + name)
    else:
        errors.append("missing:" + name)

train = root / required[0]
if train.is_file():
    text = train.read_text(encoding="utf-8")
    try:
        ast.parse(text)
        passed.append("ast:train")
    except SyntaxError as exc:
        errors.append("ast:" + repr(exc))

    for token in (
        "# R8_3_REAL_LOAD_TRANSFER_PPO",
        'self.base_env.scene["contact_forces"]',
        "net_forces_w",
        '"unload_24pct"',
        '"unload_16pct"',
        '"unload_10pct"',
        "load_ready_seen_r83",
        "selected_load_ratio",
        "support3_fz",
        "R8_3_LOAD_BASELINE",
        "R8_3_STAGE_GATE",
        "R8_3_STAGE_TIMEOUT",
        "actor_observation=70D_teacher_free",
        "low_action=12D_joint_position_residuals",
        "high_level_router_interface=unchanged",
        "soccer_goal_reward=not_used",
        "self.max_selected_foot_lift >= 0.040",
        "self.max_airborne_foot_forward >= 0.060",
    ):
        if token in text:
            passed.append("token:" + token)
        else:
            errors.append("token_missing:" + token)

    for token in (
        "R8 rollout completion guard failed",
        "teacher_action_labels=full_12D",
    ):
        if token not in text:
            passed.append(
                "forbidden_absent:" + token
            )
        else:
            errors.append(
                "forbidden_present:" + token
            )

for name in required[1:4]:
    path = root / name
    if path.is_file():
        proc = subprocess.run(
            ["bash", "-n", str(path)],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            passed.append("shell:" + name)
        else:
            errors.append(
                "shell:" + name + ":" + proc.stderr
            )

payload = {
    "counts": {
        "errors": len(errors),
        "warnings": 0,
        "passed": len(passed),
    },
    "errors": errors,
    "passed": passed,
}
print(json.dumps(payload, indent=2, sort_keys=True))
print(
    "R8_3_SUPPORT_PACKAGE_AUDIT_COUNTS "
    f"errors={len(errors)} "
    f"warnings=0 passed={len(passed)}"
)
if errors:
    print("R8_3_SUPPORT_PACKAGE_AUDIT_FAILED")
    raise SystemExit(1)
print("R8_3_SUPPORT_PACKAGE_AUDIT_CLEAN")
