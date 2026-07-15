#!/usr/bin/env python3
from pathlib import Path
import argparse, json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity")
    parser.add_argument("--name", default="kick-progress-review")
    args = parser.parse_args()

    try:
        import wandb
    except ImportError as error:
        raise SystemExit("Install W&B in the active environment with: python -m pip install wandb") from error

    packet = Path(args.packet).resolve()
    manifest = json.loads((packet / "manifest.json").read_text())
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.name,
        job_type="teacher-review",
        config={
            "k2a2_metrics": manifest.get("k2a2_metrics", {}),
            "k2a3_metrics": manifest.get("k2a3_metrics", {})
        }
    )

    media = {}
    for name in manifest.get("plots", []):
        path = packet / "assets" / "plots" / name
        if path.is_file():
            media[f"plots/{path.stem}"] = wandb.Image(str(path))
    for name in manifest.get("videos", []):
        path = packet / "assets" / "videos" / name
        if path.is_file():
            media[f"videos/{path.stem}"] = wandb.Video(str(path), fps=30, format="mp4")
    if media:
        run.log(media)

    artifact = wandb.Artifact(args.name, type="review-packet")
    artifact.add_dir(str(packet))
    run.log_artifact(artifact)
    run.finish()

if __name__ == "__main__":
    main()
