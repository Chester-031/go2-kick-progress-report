#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def indent_block(block: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line if line else line for line in block.splitlines())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    text = Path(args.source).read_text(encoding="utf-8")

    # The R8.63 parent already has the stationary full-support reset and
    # difficulty=1.0 observation. Refuse a source that still contains the
    # old progress-dependent difficulty schedule.
    if "difficulty_tensor = torch.ones(" not in text:
        raise RuntimeError("fresh no-curriculum source is not R8.63 full-support")
    if re.search(r"difficulty\s*=\s*min\(", text):
        raise RuntimeError("progress-dependent difficulty schedule remains in source")

    fresh_block = re.compile(
        r"    high_checkpoint_raw = os\.environ\.get\(\n"
        r".*?"
        r"    if not high_checkpoint\.is_file\(\):\n"
        r".*?\n        \)\n",
        re.DOTALL,
    )
    match = fresh_block.search(text)
    if not match:
        raise RuntimeError("high checkpoint validation block not found")
    original = match.group(0)
    replacement = (
        '    fresh_no_curriculum = os.environ.get("R874B_FRESH_NO_CURRICULUM") == "1"\n'
        "    if fresh_no_curriculum:\n"
        "        high_checkpoint = None\n"
        '        print("[R874B_MARKER] fresh_high_policy_random_init", flush=True)\n'
        "    else:\n"
        + indent_block(original)
        + "\n"
    )
    text = text[: match.start()] + replacement + text[match.end() :]

    high_warm = re.compile(
        r"    warm_start_policy_weights\(\n"
        r"        runner=high_runner,.*?\n    \)\n",
        re.DOTALL,
    )
    m = high_warm.search(text)
    if not m:
        raise RuntimeError("high warm-start call not found")
    high_call = m.group(0)
    high_replacement = (
        "    if not fresh_no_curriculum:\n"
        + indent_block(high_call)
        + "\n"
        + "    else:\n"
        '        print("[R874B_MARKER] high_checkpoint_skipped", flush=True)\n'
    )
    text = text[: m.start()] + high_replacement + text[m.end() :]

    # Keep the validated delta-pose low-level policy as the locomotion layer.
    # This isolates the no-curriculum question to the V4D high-level policy.
    low_trainable_needle = "        low_trainable = bool(stage[\"low_trainable\"])\n"
    if text.count(low_trainable_needle) != 1:
        raise RuntimeError("low trainable stage site count mismatch")
    text = text.replace(
        low_trainable_needle,
        low_trainable_needle
        + '        if os.environ.get("R874B_FREEZE_LOW") == "1":\n'
        + "            low_trainable = False\n"
        + '            print("[R874B_MARKER] low_policy_frozen", flush=True)\n',
        1,
    )

    stage2_low_needle = (
        '            "low_lr": float(os.environ.get("V4D_LOW_LR_STAGE2", "0.000001")),\n'
        '            "low_trainable": True,\n'
    )
    if text.count(stage2_low_needle) != 1:
        raise RuntimeError("stage2 low-trainable site count mismatch")
    stage2_low_replacement = (
        '            "low_lr": (\n'
        '                0.0\n'
        '                if os.environ.get("R874B_FREEZE_LOW") == "1"\n'
        '                else float(os.environ.get("V4D_LOW_LR_STAGE2", "0.000001"))\n'
        '            ),\n'
        '            "low_trainable": (\n'
        '                os.environ.get("R874B_FREEZE_LOW") != "1"\n'
        '            ),\n'
    )
    text = text.replace(stage2_low_needle, stage2_low_replacement, 1)

    low_validation_needle = (
        '    if stage2_delta["low"]["max_abs"] <= 0.0:\n'
        '        raise RuntimeError("V4D stage2 did not update the low policy.")\n'
    )
    if text.count(low_validation_needle) != 1:
        raise RuntimeError("stage2 low validation site count mismatch")
    low_validation_replacement = (
        '    if (\n'
        '        os.environ.get("R874B_FREEZE_LOW") != "1"\n'
        '        and stage2_delta["low"]["max_abs"] <= 0.0\n'
        '    ):\n'
        '        raise RuntimeError("V4D stage2 did not update the low policy.")\n'
    )
    text = text.replace(low_validation_needle, low_validation_replacement, 1)

    report = {
        "version": "r8_74b_v4d_fresh_no_curriculum_fix",
        "training": True,
        "fresh_high_policy": True,
        "low_policy_warm_started_and_frozen": True,
        "progressive_curriculum": False,
        "difficulty_observation": 1.0,
        "reset_distribution": "stationary_full_support",
        "targeted_failure_replay": False,
        "parent_mutation": False,
    }
    out = Path(args.out)
    out.write_text(text, encoding="utf-8")
    out.with_suffix(".patch_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("R874B_FRESH_NO_CURRICULUM_PATCH_OK")
    print("FRESH_HIGH_POLICY=TRUE")
    print("LOW_POLICY=FROZEN_WARM_START")
    print("PROGRESSIVE_CURRICULUM=FALSE")


if __name__ == "__main__":
    main()
