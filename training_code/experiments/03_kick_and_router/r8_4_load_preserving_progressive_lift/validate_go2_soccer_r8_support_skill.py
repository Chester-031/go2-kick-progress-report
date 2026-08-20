#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import pathlib
import subprocess
import sys

root=pathlib.Path(sys.argv[1]).resolve()
required=(
 "train_go2_soccer_r8_support_skill.py",
 "run_go2_soccer_r8_support_skill.sbatch",
 "install_submit_go2_soccer_r8_support_skill.sh",
 "check_go2_soccer_r8_support_skill_result.sh",
 "README.md",
)
errors=[]
passed=[]

for name in required:
    p=root/name
    if p.is_file() and p.stat().st_size>0:
        passed.append("file:"+name)
    else:
        errors.append("missing:"+name)

train=root/required[0]
if train.is_file():
    text=train.read_text(encoding="utf-8")
    try:
        ast.parse(text)
        passed.append("ast:train")
    except SyntaxError as exc:
        errors.append("ast:"+repr(exc))

    tokens=(
      "# R8_4_LOAD_PRESERVING_PROGRESSIVE_LIFT_PPO",
      'self.base_env.scene["contact_forces"]',
      "net_forces_w",
      '"unload_10pct"',
      "self.curriculum_stage = self.R8_STAGES[0]",
      '"lift_1cm"',
      '"lift_2cm"',
      '"lift_3cm"',
      '"lift_4cm"',
      "prelift_unload_seen_r84",
      "prelift_window_r84",
      "ordered_lift_gate_r84",
      "self.load_ready_seen_r83",
      "self.max_selected_foot_lift >= 0.040",
      "self.max_airborne_foot_forward >= 0.060",
      "self.max_base_displacement_r8 <= 0.060",
      "self.max_base_height_error_r8 <= 0.030",
      "self.max_upright_error_r8 <= 0.020",
      "actor_observation=70D_teacher_free",
      "low_action=12D_joint_position_residuals",
      "high_level_router_interface=unchanged",
      "soccer_goal_reward=not_used",
      "obs.to(runner.device)",
      "evaluation_stage = env.R8_STAGES[-1]",
    )
    for token in tokens:
        if token in text:
            passed.append("token:"+token)
        else:
            errors.append("token_missing:"+token)

    for token in (
      'obs["policy"].to(runner.device)',
      'self.curriculum_stage = "unload_1cm"',
      "R8 rollout completion guard failed",
      "teacher_action_labels=full_12D",
    ):
        if token not in text:
            passed.append("forbidden_absent:"+token)
        else:
            errors.append("forbidden_present:"+token)

for name in required[1:4]:
    p=root/name
    if p.is_file():
        proc=subprocess.run(
            ["bash","-n",str(p)],
            capture_output=True,
            text=True,
        )
        if proc.returncode==0:
            passed.append("shell:"+name)
        else:
            errors.append("shell:"+name+":"+proc.stderr)

payload={
 "counts":{
   "errors":len(errors),
   "warnings":0,
   "passed":len(passed),
 },
 "errors":errors,
 "passed":passed,
}
print(json.dumps(payload,indent=2,sort_keys=True))
print(
 f"R8_4_SUPPORT_PACKAGE_AUDIT_COUNTS "
 f"errors={len(errors)} warnings=0 passed={len(passed)}"
)
if errors:
    print("R8_4_SUPPORT_PACKAGE_AUDIT_FAILED")
    raise SystemExit(1)
print("R8_4_SUPPORT_PACKAGE_AUDIT_CLEAN")
