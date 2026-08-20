#!/usr/bin/env python3
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

text = args.source.read_text(encoding="utf-8")

replacements = [
    (
        "        terminal_velocity[:, 2] = 0.15 * action[:, 2]\n",
        "        terminal_velocity[:, 2] = 0.25 * action[:, 2]\n",
        "turn_target_scale",
    ),
    (
        "        requested_velocity = self.last_reference_command\n",
        "        requested_velocity = self.target_terminal_velocity\n",
        "axis_reference_source",
    ),
]

for old, new, label in replacements:
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"R883_PATCH_FATAL={label}_count:{count}"
        )
    text = text.replace(old, new, 1)

text = text.replace("R881_", "R883_")
text = text.replace("r881_", "r883_")
text = text.replace("r8_81", "r8_83")
text = text.replace("R8.81", "R8.83")

args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(text, encoding="utf-8")
print("R883_REFERENCE_VELOCITY_PATCH_OK")
