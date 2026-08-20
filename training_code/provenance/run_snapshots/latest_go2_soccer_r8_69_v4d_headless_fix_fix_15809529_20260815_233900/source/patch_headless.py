#!/usr/bin/env python3
from pathlib import Path
import argparse
import json


HELPER = r'''
def _r869_strip_visual_materials(config):
    """Remove PreviewSurface/MDL materials for headless metric smoke only."""
    seen = set()
    changed = 0

    def visit(obj, depth=0):
        nonlocal changed
        if obj is None or depth > 8 or id(obj) in seen:
            return
        seen.add(id(obj))
        if hasattr(obj, "visual_material"):
            try:
                if getattr(obj, "visual_material") is not None:
                    setattr(obj, "visual_material", None)
                    changed += 1
            except Exception:
                pass
        fields = getattr(obj, "__dataclass_fields__", {})
        for name in fields:
            try:
                visit(getattr(obj, name), depth + 1)
            except Exception:
                pass
        if depth == 0:
            scene = getattr(obj, "scene", None)
            if scene is not None:
                visit(scene, depth + 1)

    visit(config)
    print(f"[R869_MARKER] visual_materials_removed={changed}", flush=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    text = Path(args.source).read_text(encoding="utf-8")
    marker = "def build_parser"
    pos = text.find(marker)
    if pos < 0:
        raise RuntimeError("build_parser definition not found")
    text = text[:pos] + HELPER + "\n" + text[pos:]
    needle = "    gym_env = gym.make(\n"
    if text.count(needle) != 1:
        raise RuntimeError(f"gym.make site count={text.count(needle)}")
    replacement = (
        "    if os.environ.get(\"R865_METRIC_ONLY\") == \"1\":\n"
        "        _r869_strip_visual_materials(env_cfg)\n"
        "        print(\"[R869_MARKER] headless_cfg_applied\", flush=True)\n"
        "\n"
        + needle
    )
    text = text.replace(needle, replacement, 1)
    out = Path(args.out)
    out.write_text(text, encoding="utf-8")
    Path(str(out) + ".patch_report.json").write_text(
        json.dumps(
            {
                "version": "r8_69_v4d_headless_fix",
                "training": False,
                "metric_only_visual_materials_removed": True,
                "video_path_unchanged": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("R869_HEADLESS_FIX_PATCH_OK")


if __name__ == "__main__":
    main()
