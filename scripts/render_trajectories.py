"""Turn recorded trajectories into an MP4 (no simulator needed).

Reads the ``.npz`` written by ``scripts/record_trajectories.py`` and draws,
at 25 frames per second, a 3D close-up that follows the robot (trunk, the
four legs as hip-thigh-calf-foot chains, the head and the ball) beside a
top-down view of the whole field (both goals, the attacked one
highlighted, and the robot's and ball's paths so far).  Each episode is
captioned with its difficulty and start knobs; its last frame is held
with how it ended.

    python scripts/render_trajectories.py traj.npz --out video.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIELD_HALF_LENGTH, FIELD_HALF_WIDTH, GOAL_HALF_WIDTH = 4.0, 2.5, 0.6
LEGS = ("FL", "FR", "RL", "RR")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
ROBOT, BALL, GOAL_COLOR, OWN_GOAL = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"
OUTCOME_COLOR = {"goal": "#008300"}


def draw_frame(figure, body, ball, names, episode, robot_trail, ball_trail, footer):
    figure.clf()
    index = {name: position for position, name in enumerate(names)}
    side = episode["goal_side"]
    knobs = episode["knobs"]
    figure.text(
        0.01, 0.97,
        f"{episode['control']} control   difficulty {episode['difficulty']:g}   "
        f"start: ball {knobs['ball_goal_distance_m']:.2f} m from goal, robot {knobs['robot_ball_distance_m']:.2f} m "
        f"from ball, facing {knobs['facing_error_deg']:.0f}° off, {knobs['robot_bearing_deg']:.0f}° around",
        fontsize=10, color=INK, va="top",
    )
    if footer:
        text, color = footer
        figure.text(0.01, 0.91, text, fontsize=15, fontweight="bold", color=color, va="top")

    close = figure.add_subplot(1, 2, 1, projection="3d")
    base = body[index["base"]]
    for leg in LEGS:
        chain = np.stack([body[index[f"{leg}_{link}"]] for link in ("hip", "thigh", "calf", "foot")])
        close.plot(chain[:, 0], chain[:, 1], chain[:, 2], color=ROBOT, linewidth=3)
        close.scatter(*chain[-1], color=INK, s=10)
    trunk = np.stack([body[index[f"{leg}_hip"]] for leg in ("FL", "FR", "RR", "RL", "FL")])
    close.plot(trunk[:, 0], trunk[:, 1], trunk[:, 2], color=INK, linewidth=2.5)
    if "Head_upper" in index:
        head = np.stack([base, body[index["Head_upper"]]])
        close.plot(head[:, 0], head[:, 1], head[:, 2], color=INK, linewidth=2)
    close.scatter(*ball, color=BALL, s=180, depthshade=False)
    centre = (base[:2] + ball[:2]) / 2.0 if np.linalg.norm(base[:2] - ball[:2]) < 2.0 else base[:2]
    half = 1.2
    close.set_xlim(centre[0] - half, centre[0] + half)
    close.set_ylim(centre[1] - half, centre[1] + half)
    close.set_zlim(0.0, 0.8)
    close.set_box_aspect((1.0, 1.0, 0.8 / (2 * half)))
    goal_x = side * FIELD_HALF_LENGTH
    if abs(goal_x - centre[0]) < half + 0.5:
        for y in (-GOAL_HALF_WIDTH, GOAL_HALF_WIDTH):
            close.plot([goal_x, goal_x], [y, y], [0.0, 0.5], color=GOAL_COLOR, linewidth=3)
        close.plot([goal_x, goal_x], [-GOAL_HALF_WIDTH, GOAL_HALF_WIDTH], [0.5, 0.5], color=GOAL_COLOR, linewidth=3)
    close.view_init(elev=22, azim=-60)
    close.set_title("close-up (follows the robot)", fontsize=10)
    close.tick_params(labelsize=6)

    top = figure.add_subplot(1, 2, 2)
    top.set_facecolor("#eef3e8")
    top.plot([-4, 4, 4, -4, -4], [-2.5, -2.5, 2.5, 2.5, -2.5], color=MUTED, linewidth=1)
    top.axvline(0, color=GRID, linewidth=1)
    for x, color, width in ((goal_x, GOAL_COLOR, 6), (-goal_x, OWN_GOAL, 3)):
        top.plot([x, x], [-GOAL_HALF_WIDTH, GOAL_HALF_WIDTH], color=color, linewidth=width, solid_capstyle="butt")
    if len(robot_trail) > 1:
        trail = np.asarray(robot_trail)
        top.plot(trail[:, 0], trail[:, 1], color=ROBOT, linewidth=1.2, alpha=0.6)
        trail = np.asarray(ball_trail)
        top.plot(trail[:, 0], trail[:, 1], color=BALL, linewidth=1.2, alpha=0.6)
    heading = body[index["Head_upper"]][:2] - base[:2] if "Head_upper" in index else np.array([0.3, 0.0])
    heading = 0.45 * heading / max(np.linalg.norm(heading), 1e-6)
    top.arrow(base[0], base[1], heading[0], heading[1], width=0.04, color=ROBOT, length_includes_head=True)
    top.scatter(ball[0], ball[1], color=BALL, s=40, zorder=3)
    top.set_xlim(-4.6, 4.6)
    top.set_ylim(-2.9, 2.9)
    top.set_aspect("equal")
    top.set_title("whole field (green: goal to score in)", fontsize=10)
    top.tick_params(labelsize=7)
    figure.canvas.draw()
    return np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectories", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--difficulty", type=float, action="append", help="only these difficulties")
    args = parser.parse_args()
    data = np.load(args.trajectories)
    meta = json.loads(bytes(data["index_json"]).decode())
    names = meta["body_names"]
    figure = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor=SURFACE)
    writer = imageio.get_writer(args.out, fps=25, codec="libx264", quality=8, macro_block_size=8)
    for episode in meta["episodes"]:
        if args.difficulty and episode["difficulty"] not in args.difficulty:
            continue
        bodies = data[episode["key"] + "_bodies"]
        balls = data[episode["key"] + "_ball"]
        robot_trail, ball_trail = [], []
        frame = None
        for tick in range(len(bodies)):
            robot_trail.append(bodies[tick][names.index("base")][:2])
            ball_trail.append(balls[tick][:2])
            if tick % 2 == 0 or tick == len(bodies) - 1:
                frame = draw_frame(figure, bodies[tick], balls[tick], names, episode, robot_trail, ball_trail, None)
                if tick != len(bodies) - 1:
                    writer.append_data(frame)
        outcome = episode["outcome"]
        footer = (f"{outcome.replace('_', ' ').upper()} after {episode['seconds']:.2f} s",
                  OUTCOME_COLOR.get(outcome, "#e34948"))
        last = len(bodies) - 1
        frame = draw_frame(figure, bodies[last], balls[last], names, episode, robot_trail, ball_trail, footer)
        for _ in range(15):
            writer.append_data(frame)
        print(f"rendered {episode['key']} {outcome} {episode['seconds']:.2f}s", flush=True)
    writer.close()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
