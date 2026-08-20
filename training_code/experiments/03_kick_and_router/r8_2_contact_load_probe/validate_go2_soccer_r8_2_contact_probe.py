#!/usr/bin/env python3
from __future__ import annotations
import ast
import json
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1]).resolve()
required = (
    "train_go2_soccer_r8_2_contact_probe.py",
    "run_go2_soccer_r8_2_contact_probe.sbatch",
    "install_submit_go2_soccer_r8_2_contact_probe.sh",
    "check_go2_soccer_r8_2_contact_probe_result.sh",
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
        tree = ast.parse(text)
        passed.append("ast:train")
    except SyntaxError as exc:
        errors.append("ast:" + repr(exc))
        tree = None

    tokens = (
        "# R8_2_CONTACT_LOAD_SIGNAL_PROBE",
        "--contact_probe_json",
        "run_r82_contact_load_probe",
        "net_forces_w",
        "FL_foot",
        "FR_foot",
        "RL_foot",
        "RR_foot",
        "R8_2_CONTACT_SOURCE",
        "R8_2_CONTACT_MAPPING",
        "R8_2_CONTACT_PHASE",
        "R8_2_CONTACT_SIGNAL_READY",
        "R8_2_CONTACT_PROBE_COMPLETE",
        "ppo_training_started",
        "reward_changed",
        "actor_observation_teacher_free",
        "high_level_router_interface",
    )
    for token in tokens:
        if token in text:
            passed.append("token:" + token)
        else:
            errors.append("token_missing:" + token)

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
            errors.append("shell:" + name + ":" + proc.stderr)

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
    f"R8_2_CONTACT_PROBE_AUDIT_COUNTS "
    f"errors={len(errors)} warnings=0 passed={len(passed)}"
)
if errors:
    print("R8_2_CONTACT_PROBE_AUDIT_FAILED")
    raise SystemExit(1)
print("R8_2_CONTACT_PROBE_AUDIT_CLEAN")
