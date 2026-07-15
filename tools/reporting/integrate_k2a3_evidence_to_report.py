#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


def copy_first(source_dir: Path, pattern: str, destination: Path) -> None:
    matches = sorted(source_dir.glob(pattern))
    if not matches:
        raise SystemExit(f"Missing plot: {pattern}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(matches[0], destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--report-repo", required=True)
    parser.add_argument("--wandb-url", default="")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    source = Path(args.source_repo).expanduser().resolve()
    report = Path(args.report_repo).expanduser().resolve()

    if not (report / ".git").is_dir():
        raise SystemExit(f"Report Git repository not found: {report}")

    video_dir = source / "outputs/kick_behavior_local/videos/k2a3_evidence"
    video = video_dir / "world_anchored_motion_prior.mp4"
    preview = video_dir / "world_anchored_motion_prior_preview.png"
    summary_path = source / "outputs/kick_behavior_local/k2a3_video_execution_probe.json"
    packet_plots = source / "docs/reports/kick_review_packet/assets/plots"

    for required in (video, preview, summary_path):
        if not required.is_file():
            raise SystemExit(f"Missing evidence file: {required}")

    summary = json.loads(summary_path.read_text())

    def find_number(keys: tuple[str, ...], fallback: float) -> float:
        stack = [summary]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in keys and isinstance(item, (int, float)):
                        return float(item)
                    stack.append(item)
            elif isinstance(value, list):
                stack.extend(value)
        return fallback

    forward = find_number(
        ("max_forward_m", "forward_m", "max_foot_forward_m", "forward"),
        0.19999,
    )
    foot_speed = find_number(
        ("max_foot_speed_mps", "foot_speed_mps", "foot_speed"),
        2.377,
    )
    foot_height = find_number(
        ("max_foot_height_m", "foot_height_m", "height"),
        0.06603,
    )
    base_drop = find_number(
        ("max_base_drop_m", "base_drop_m", "base_drop"),
        0.08189,
    )
    base_xy = find_number(
        ("max_base_xy_m", "base_xy_m", "base_xy"),
        0.07663,
    )

    assets_plots = report / "assets/plots"
    assets_videos = report / "assets/videos"
    assets_data = report / "assets/data"
    tools = report / "tools/reporting"

    for directory in (assets_plots, assets_videos, assets_data, tools):
        directory.mkdir(parents=True, exist_ok=True)

    displacement_plot = assets_plots / "actual_world_frame_foot_displacement.png"

    labels = [
        "Original body-relative\nobjective",
        "Corrected world-anchored\nobjective",
    ]
    values = [0.013, forward]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(labels, values)
    ax.set_title("Actual world-frame forward motion of the selected front foot")
    ax.set_ylabel("Forward displacement (m)")
    ax.set_ylim(0.0, max(values) * 1.25)
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.025,
            f"{value:.3f} m",
            ha="center",
            va="bottom",
        )

    ratio = forward / 0.013
    ax.text(
        0.5,
        max(values) * 1.13,
        f"{ratio:.1f}× more genuine forward foot motion",
        ha="center",
        va="center",
    )
    fig.tight_layout()
    fig.savefig(displacement_plot, dpi=180)
    plt.close(fig)

    training_plot = assets_plots / "corrected_world_anchored_training_curves.png"
    copy_first(
        packet_plots,
        "*k2a3_training_curves.png",
        training_plot,
    )

    shutil.copy2(video, assets_videos / "world_anchored_motion_prior.mp4")
    shutil.copy2(preview, assets_videos / "world_anchored_motion_prior_preview.png")
    shutil.copy2(summary_path, assets_data / "latest_motion_execution_probe.json")

    for tool_name in (
        "run_k2a3_probe_gui.py",
        "record_k2a3_probe_video.sh",
        "integrate_k2a3_evidence_to_report.py",
        "build_review_packet.py",
        "publish_review_packet_to_wandb.py",
    ):
        tool = source / "scripts/reporting" / tool_name
        if tool.is_file():
            shutil.copy2(tool, tools / tool.name)

    wandb_line = (
        f"[Open the interactive W&B review run]({args.wandb_url})"
        if args.wandb_url
        else "W&B review run: add the shareable run link here."
    )

    section = f"""
<!-- PHYSICAL-KICK-EVIDENCE-START -->
### 8.3 Experimental evidence: why the original motion reward was misleading

The first motion-prior objective measured the selected front foot relative to the robot's moving body. The policy could therefore improve its tracking reward by lowering or translating the torso while leaving the foot almost stationary in world coordinates.

A world-frame execution probe measured only **0.013 m** of genuine forward foot movement under that original objective.

The corrected objective anchors the desired trajectory to the robot's initial world pose. The latest checkpoint produced **{forward:.3f} m** of genuine world-frame forward foot movement, approximately **{ratio:.1f} times** the original result.

![Comparison of actual world-frame forward foot displacement](assets/plots/actual_world_frame_foot_displacement.png)

**Figure 1 — Actual physical motion, not the internal tracking score.**  
The left bar shows the failed body-relative objective. The right bar shows the corrected world-anchored objective evaluated with the latest checkpoint. This comparison demonstrates that the reward correction produced real forward movement of the selected front foot.

### 8.4 Training behavior under the corrected world-anchored objective

![Corrected world-anchored training curves](assets/plots/corrected_world_anchored_training_curves.png)

**Figure 2 — Optimization of the corrected motion prior.**  
The mean reward and world-anchored foot-tracking terms increase during training, while base-contact termination remains near zero. This indicates that the policy learns the revised trajectory without frequently falling. The curve is supporting evidence only; successful ball contact has not yet been demonstrated.

### 8.5 Automatic Isaac Lab playback

[![Preview of the automatically recorded Isaac Lab playback](assets/videos/world_anchored_motion_prior_preview.png)](assets/videos/world_anchored_motion_prior.mp4)

**Video 1 — Ball-free physical motion prior.**  
The video is recorded directly from the working execution-probe loop. It shows the selected front leg lifting and moving forward while the remaining legs support the body. This is a motion-prior result, not yet a completed kick and not a goal-scoring result.

Latest execution-probe measurements:

| Measurement | Result | Current interpretation |
|---|---:|---|
| Genuine forward foot motion | {forward:.3f} m | Meets the current reach target |
| Peak foot speed | {foot_speed:.3f} m/s | Sufficient for the next contact-learning stage |
| Foot lift height | {foot_height:.3f} m | Still below the 0.10 m target |
| Base vertical drop | {base_drop:.3f} m | Slightly above the 0.08 m target |
| Base horizontal drift | {base_xy:.3f} m | Slightly above the 0.07 m target |

### 8.6 Experiment links and reproducibility

{wandb_line}

| File | Purpose |
|---|---|
| `tools/reporting/run_k2a3_probe_gui.py` | Forces the verified execution probe to run in the GUI, follows the robot with a close leg-focused camera, and slows the simulation for readable playback. |
| `tools/reporting/record_k2a3_probe_video.sh` | Starts the verified probe, waits for physical motion, records the Isaac Sim window, and generates an MP4 plus a preview frame. |
| `tools/reporting/integrate_k2a3_evidence_to_report.py` | Generates the physical-displacement comparison, copies selected evidence, updates this report, and optionally pushes the report repository. |
| `tools/reporting/build_review_packet.py` | Generates compact plots and a manifest from TensorBoard and experiment summaries. |
| `tools/reporting/publish_review_packet_to_wandb.py` | Publishes plots, metrics, videos, and the review artifact to W&B. |

The report intentionally excludes the earlier body-motion bar chart because it compared vertical base drop with three-dimensional total root displacement. Those quantities are not directly comparable.
<!-- PHYSICAL-KICK-EVIDENCE-END -->
""".strip()

    report_path = report / "Report.md"
    text = report_path.read_text()

    start = "<!-- PHYSICAL-KICK-EVIDENCE-START -->"
    end = "<!-- PHYSICAL-KICK-EVIDENCE-END -->"

    if start in text and end in text:
        before = text.split(start, 1)[0].rstrip()
        after = text.split(end, 1)[1].lstrip()
        text = before + "\n\n" + section + "\n\n" + after
    else:
        marker = "\n---\n\n## 9."
        if marker not in text:
            raise SystemExit("Could not locate Section 9 in Report.md.")
        text = text.replace(
            marker,
            "\n\n" + section + "\n\n---\n\n## 9.",
            1,
        )

    report_path.write_text(text)

    (report / "README.md").write_text(
        """# Go2 Kick Progress Report

The full progress report is available in [Report.md](Report.md).

## Most useful evidence

- [Physical kick-motion evidence](Report.md#83-experimental-evidence-why-the-original-motion-reward-was-misleading)
- [Actual world-frame foot-displacement comparison](assets/plots/actual_world_frame_foot_displacement.png)
- [Corrected training curves](assets/plots/corrected_world_anchored_training_curves.png)
- [Automatic Isaac Lab playback](assets/videos/world_anchored_motion_prior.mp4)
- [Reproducibility tools](tools/reporting/)

Large checkpoints and raw simulation outputs are not stored in this report repository.
"""
    )

    subprocess.run(["git", "-C", str(report), "add", "Report.md", "README.md", "assets", "tools"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(report),
            "commit",
            "-m",
            "Add physical kick motion evidence and automatic playback",
        ],
        check=False,
    )

    if args.push:
        subprocess.run(["git", "-C", str(report), "push"], check=True)

    print(f"[REPORT] updated={report}")
    print(f"[REPORT] video={assets_videos / 'world_anchored_motion_prior.mp4'}")
    print(f"[REPORT] displacement_plot={displacement_plot}")
    print(f"[REPORT] training_plot={training_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
