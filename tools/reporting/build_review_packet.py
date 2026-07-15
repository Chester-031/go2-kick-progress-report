#!/usr/bin/env python3
from pathlib import Path
import argparse, json, math, shutil
import matplotlib.pyplot as plt

KEYS = {
    "reported_body_relative_forward_m": ["reported_body_relative_forward", "body_relative_forward"],
    "forward_foot_displacement_m": ["max_foot_forward_m", "max_forward_foot_displacement", "forward_foot_displacement", "max_actual_forward_offset"],
    "foot_height_m": ["max_foot_height_m", "max_foot_height", "foot_height_m"],
    "base_drop_m": ["max_base_drop_m", "base_drop_m", "max_base_drop"],
    "base_xy_drift_m": ["max_base_xy_drift_m", "base_xy_drift_m", "horizontal_drift"],
    "root_displacement_m": ["max_root_delta_m", "max_root_displacement", "root_displacement"],
    "foot_speed_mps": ["max_foot_speed_mps", "peak_foot_speed", "foot_speed_mps"],
    "joint_delta_rad": ["max_joint_delta_rad", "joint_delta_rad"],
    "max_action_abs": ["max_action_abs", "maximum_action_magnitude"],
}

def flatten(value, prefix=""):
    out = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(item, name))
    elif isinstance(value, list):
        out[prefix] = value
    else:
        out[prefix] = value
    return out

def load(path_value):
    if not path_value:
        return {}
    path = Path(path_value).expanduser()
    if not path.is_file():
        print(f"[review] JSON not found; fallback metrics used: {path}")
        return {}
    return json.loads(path.read_text())

def extract(section):
    raw = load(section.get("summary_json"))
    flat = {k.lower(): v for k, v in flatten(raw).items()}
    metrics = {}
    for name, candidates in KEYS.items():
        value = None
        for candidate in candidates:
            for key, item in flat.items():
                if candidate.lower() in key:
                    try:
                        number = float(item)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(number):
                        value = number
                        break
            if value is not None:
                break
        metrics[name] = value if value is not None else section.get("fallback_metrics", {}).get(name)
    return metrics, raw

def bar(path, title, labels, values, ylabel):
    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

def latest_event(path_value):
    if not path_value:
        return None
    root = Path(path_value).expanduser()
    if root.is_file() and "tfevents" in root.name:
        return root
    files = list(root.rglob("events.out.tfevents.*")) if root.exists() else []
    return max(files, key=lambda p: p.stat().st_mtime) if files else None

def tensorboard_plot(path, event_file, requested, title):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return []
    acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    acc.Reload()
    available = set(acc.Tags().get("scalars", []))
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    matched = []
    for request in requested:
        options = sorted(tag for tag in available if request == tag or request in tag)
        if not options:
            continue
        tag = options[0]
        points = acc.Scalars(tag)
        if not points:
            continue
        ax.plot([p.step for p in points], [p.value for p in points], label=tag)
        matched.append(tag)
    if not matched:
        plt.close(fig)
        return []
    ax.set_title(title)
    ax.set_xlabel("Training step / iteration")
    ax.set_ylabel("Logged value")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return matched

def fmt(value, unit=""):
    return "Not recorded" if value is None else f"{value:.4f}{unit}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.output)
    plots = out / "assets" / "plots"
    videos = out / "assets" / "videos"
    plots.mkdir(parents=True, exist_ok=True)
    videos.mkdir(parents=True, exist_ok=True)

    k2a2, _ = extract(cfg["k2a2"])
    k2a3, _ = extract(cfg["k2a3"])
    made_plots = []

    forward = [k2a2.get("forward_foot_displacement_m"), k2a3.get("forward_foot_displacement_m")]
    if all(v is not None for v in forward):
        name = "01_world_frame_forward_displacement.png"
        bar(plots / name, "Actual world-frame forward foot displacement", [cfg["k2a2"]["label"], cfg["k2a3"]["label"]], [float(v) for v in forward], "Forward displacement (m)")
        made_plots.append(name)

    labels, values = [], []
    for label, value in [
        ("K2A2 base drop", k2a2.get("base_drop_m")),
        ("K2A3 root displacement", k2a3.get("root_displacement_m")),
        ("K2A3 base drop", k2a3.get("base_drop_m")),
        ("K2A3 base XY drift", k2a3.get("base_xy_drift_m")),
    ]:
        if value is not None:
            labels.append(label)
            values.append(float(value))
    if values:
        name = "02_body_motion_diagnostics.png"
        bar(plots / name, "Body motion diagnostics", labels, values, "Displacement (m)")
        made_plots.append(name)

    for key in ("k2a2", "k2a3"):
        event = latest_event(cfg[key].get("tensorboard_dir"))
        if event:
            name = f"03_{key}_training_curves.png"
            tags = tensorboard_plot(plots / name, event, cfg.get("selected_tensorboard_tags", []), f"{cfg[key]['label']} training curves")
            if tags:
                made_plots.append(name)
                print(f"[review] {key} TensorBoard tags: {tags}")

    made_videos = []
    for key, filename in [("k2a2", "k2a2_reward_hacking.mp4"), ("k2a3", "k2a3_world_anchored_progress.mp4")]:
        source_value = cfg[key].get("video")
        if not source_value:
            continue
        source = Path(source_value).expanduser()
        if source.is_file():
            shutil.copy2(source, videos / filename)
            made_videos.append(filename)
        else:
            print(f"[review] Optional video not found: {source}")

    links = [f"- [Plot: {name}](assets/plots/{name})" for name in made_plots]
    links += [f"- [Video: {name}](assets/videos/{name})" for name in made_videos]
    quick = "\n".join(links) if links else "- No media generated."

    report = f"""# {cfg['title']}

## Quick links

{quick}

## Main finding

K2A2 achieved a high body-relative tracking result while producing very little actual world-frame foot motion. K2A3 replaced the moving body-relative reference with an initial-world-pose anchor and now produces genuine foot lift and forward swing.

## Evidence

| Diagnostic | K2A2 | K2A3 |
|---|---:|---:|
| Reported body-relative forward motion | {fmt(k2a2.get('reported_body_relative_forward_m'), ' m')} | Not applicable |
| Actual world-frame forward motion | {fmt(k2a2.get('forward_foot_displacement_m'), ' m')} | {fmt(k2a3.get('forward_foot_displacement_m'), ' m')} |
| Foot lift height | {fmt(k2a2.get('foot_height_m'), ' m')} | {fmt(k2a3.get('foot_height_m'), ' m')} |
| Base drop | {fmt(k2a2.get('base_drop_m'), ' m')} | {fmt(k2a3.get('base_drop_m'), ' m')} |
| Base XY drift | {fmt(k2a2.get('base_xy_drift_m'), ' m')} | {fmt(k2a3.get('base_xy_drift_m'), ' m')} |
| Total root displacement | {fmt(k2a2.get('root_displacement_m'), ' m')} | {fmt(k2a3.get('root_displacement_m'), ' m')} |
| Peak foot speed | {fmt(k2a2.get('foot_speed_mps'), ' m/s')} | {fmt(k2a3.get('foot_speed_mps'), ' m/s')} |

## Interpretation

K2A2 could improve its tracking score by moving the robot body around the selected foot. K2A3 corrects this reference failure, but base drop, horizontal drift, foot speed, support behavior, and recovery still require complete measurements before ball-contact training resumes.

## Minimal evidence still needed

- One current K2A3 playback video from a fixed side view.
- One world-frame foot/base diagnostic plot.
- One training plot containing mean reward and the key anchored/stability terms.
- A K2A2 failure video only if the old checkpoint can be reproduced quickly.
"""
    (out / "README.md").write_text(report)
    (out / "manifest.json").write_text(json.dumps({"plots": made_plots, "videos": made_videos, "k2a2_metrics": k2a2, "k2a3_metrics": k2a3}, indent=2))
    print(f"[review] Packet created: {out.resolve()}")

if __name__ == "__main__":
    main()
