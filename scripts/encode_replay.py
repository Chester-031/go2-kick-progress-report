"""Turn ``replay_isaac6.py`` frames into one captioned highlight reel.

Picks the most impressive goals of a ``record_trajectories.py`` file (the
hardest fixed difficulty first, then the longest start: robot-ball plus
ball-goal distance), captions every frame with the difficulty, the start
knobs and the elapsed time, holds each episode's last frame for a second
with its outcome, and writes them back to back as one 25 fps MP4.

    python scripts/encode_replay.py --trajectories traj.npz --frames frames --out reel.mp4 [--episodes 4]
    python scripts/encode_replay.py ... --list    # print the chosen keys only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FPS = 25
HOLD_SECONDS = 1.0


def choose(episodes: list[dict], count: int) -> list[dict]:
    """The reel's episodes: goals before other outcomes, then the hardest
    difficulty, then the longest start distance."""

    def rank(episode: dict) -> tuple:
        knobs = episode["knobs"]
        return (episode["outcome"] == "goal", episode["difficulty"],
                knobs["robot_ball_distance_m"] + knobs["ball_goal_distance_m"])

    return sorted(episodes, key=rank, reverse=True)[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--frames", type=Path, help="replay_isaac6.py --out-dir")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--title", default="")
    parser.add_argument("--list", action="store_true", help="print the chosen episode keys and stop")
    args = parser.parse_args()
    data = np.load(args.trajectories)
    index = json.loads(bytes(data["index_json"]).decode())
    chosen = choose(index["episodes"], args.episodes)
    if args.list:
        print(" ".join(episode["key"] for episode in chosen))
        return

    import cv2
    import imageio.v2 as imageio

    def caption(image: np.ndarray, lines: list[str], scale: float = 0.8, corner: str = "top") -> None:
        height = image.shape[0]
        for number, text in enumerate(lines):
            y = 40 + 34 * number if corner == "top" else height - 30 - 34 * (len(lines) - 1 - number)
            cv2.putText(image, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(image, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)

    writer = imageio.get_writer(args.out, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
    for number, episode in enumerate(chosen, start=1):
        knobs = episode["knobs"]
        header = [
            f"{args.title}  episode {number}/{len(chosen)}  difficulty {episode['difficulty']:g}".strip(),
            f"robot-ball {knobs['robot_ball_distance_m']:.1f} m  ball-goal {knobs['ball_goal_distance_m']:.1f} m  "
            f"facing off {knobs['facing_error_deg']:.0f} deg",
        ]
        frames = sorted((args.frames / episode["key"]).glob("*.jpg"))
        if not frames:
            raise SystemExit(f"no frames for {episode['key']} in {args.frames}")
        for frame, path in enumerate(frames):
            raw = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            image = raw.copy()
            caption(image, header)
            caption(image, [f"t = {frame / FPS:4.1f} s"], corner="bottom")
            writer.append_data(image)
        # Hold the last frame with the outcome in place of the clock.
        caption(raw, header)
        ending = episode["outcome"].replace("_", " ").upper()
        caption(raw, [f"{ending} after {episode['seconds']:.1f} s"], scale=1.4, corner="bottom")
        for _ in range(int(HOLD_SECONDS * FPS)):
            writer.append_data(raw)
        print(f"[encode] {episode['key']}: {len(frames)} frames, {episode['outcome']}", flush=True)
    writer.close()
    print(f"[encode] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
