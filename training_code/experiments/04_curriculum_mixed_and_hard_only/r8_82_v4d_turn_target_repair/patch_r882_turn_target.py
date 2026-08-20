#!/usr/bin/env python3
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

text = args.source.read_text(encoding="utf-8")
old = "        terminal_velocity[:, 2] = 0.15 * action[:, 2]\n"
new = "        terminal_velocity[:, 2] = 0.25 * action[:, 2]\n"

if text.count(old) != 1:
    raise SystemExit(
        f"R882_PATCH_FATAL=turn_target_scale_count:{text.count(old)}"
    )

text = text.replace(old, new, 1)
text = text.replace("R881_", "R882_")
text = text.replace("r881_", "r882_")
text = text.replace("r8_81", "r8_82")
text = text.replace("R8.81", "R8.82")

args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(text, encoding="utf-8")
print("R882_TURN_TARGET_PATCH_OK")
