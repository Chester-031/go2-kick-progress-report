"""Standard training plots for ADR soccer runs.

    python scripts/plot_training.py RUN_DIR [RUN_DIR ...] --out PLOT_DIR
        [--labels a,b] [--smooth 5] [--window 0.2]

Each RUN_DIR is a training output directory; only its ``run_info.json`` and
``metrics.jsonl`` are read (``robot_soccer.metrics`` writes both), so the
plots can be made on a laptop from copied files and need only matplotlib
and numpy.  Several runs (seeds, arms) are overlaid, one colour per run in
a fixed order; figures that need many episodes (heatmaps, goal rate along a
knob) pool the runs.  x is the PPO iteration everywhere.

Figures:
  01_goal_rates      headline: goal rate at each fixed difficulty (the only
                     goal rate that compares across runs and over training,
                     since the ADR training task keeps changing), and by
                     episode kind
  02_adr_ranges      each knob's [low, high] over training, with its limits
  03_adr_probes      probe success at every range edge against the move
                     thresholds, and when the edge moved
  04_adr_health      ADR entropy, share of each knob's limits covered,
                     probe results and edge moves per iteration
  05_outcomes        how episodes end (share of finished episodes)
  06_knob_heatmaps   goal rate and where episodes started, over each pair of
                     knobs, pooled over the last --window of training
  07_goal_by_knob    goal rate along each knob, early vs late in training
  08_episode_timing  episode length, time to goal and to first ball contact,
                     ball progress toward the goal
  09_ppo             PPO losses, episode return, shaping return per potential
  10_throughput      transitions per second and where each iteration's wall
                     time goes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402

# Reference categorical palette (dataviz palette.md), fixed slot order.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
# Blue sequential ramp, 100 -> 700, for magnitudes (heatmaps); ordinal
# marks (training windows) use steps 250 and darker.
BLUE_RAMP = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b")
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
SEQUENTIAL.set_bad(SURFACE)
KNOB_LABEL = {
    "ball_goal_distance_m": "ball to goal (m)",
    "robot_ball_distance_m": "robot to ball (m)",
    "facing_error_deg": "facing error to ball (deg)",
    "robot_bearing_deg": "robot around ball (deg, 0 = behind)",
}
KIND_LABEL = {"train": "training (current ranges)", "probe": "probes (one edge pinned)",
              "evaluation": "fixed difficulty (all)"}
OUTCOME_LABEL = {"goal": "goal", "own_goal": "own goal", "fall": "fall", "boundary": "out of bounds",
                 "joint_hard_limit": "joint limit", "timeout": "timeout (20 s)"}


def style() -> None:
    """Recessive chrome: hairline grid, muted axes, light surface."""

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "semibold",
        "legend.frameon": False, "lines.linewidth": 2,
    })


def load_run(path: Path, label: str) -> dict:
    """A run's static description and its per-iteration records."""

    info = json.loads((path / "run_info.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if not records:
        raise SystemExit(f"{path} has no metrics records yet")
    return {"label": label, "info": info, "records": records,
            "iteration": np.array([record["iteration"] for record in records], float)}


def field(run: dict, *keys: str) -> np.ndarray:
    """One numeric field of every record (NaN where absent or None)."""

    values = []
    for record in run["records"]:
        value = record
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
            if value is None:
                break
        values.append(np.nan if value is None else float(value))
    return np.array(values, float)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average that skips NaN and keeps the length."""

    if window <= 1:
        return values
    out = np.full_like(values, np.nan)
    half = window // 2
    for index in range(len(values)):
        piece = values[max(0, index - half): index + window - half]
        piece = piece[np.isfinite(piece)]
        if piece.size:
            out[index] = piece.mean()
    return out


def save(fig: plt.Figure, out: Path, name: str) -> None:
    path = out / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def run_legend(fig: plt.Figure, runs: list[dict], y: float = -0.02) -> None:
    if len(runs) < 2:
        return
    handles = [plt.Line2D([], [], color=SERIES[index % len(SERIES)]) for index in range(len(runs))]
    fig.legend(handles, [run["label"] for run in runs], loc="upper center", ncol=min(len(runs), 4),
               bbox_to_anchor=(0.5, y))


def direct_label(axis: plt.Axes, x: np.ndarray, y: np.ndarray, text: str, color: str) -> None:
    """Label a line at its last finite point, in text ink with a colour tick."""

    finite = np.flatnonzero(np.isfinite(y))
    if finite.size:
        last = finite[-1]
        axis.annotate(text, (x[last], y[last]), xytext=(4, 0), textcoords="offset points",
                      va="center", fontsize=7.5, color=INK_2)
        axis.plot([x[last]], [y[last]], "o", ms=4, color=color)


def plot_goal_rates(runs: list[dict], out: Path, window: int) -> None:
    difficulties = sorted({key for run in runs for record in run["records"]
                           for key in record.get("evaluation", {})}, key=float)
    panels = [(f"difficulty {difficulty}", ("evaluation", difficulty, "goal_rate")) for difficulty in difficulties]
    panels += [(KIND_LABEL[kind], ("by_kind", kind, "goal_rate")) for kind in ("train", "probe")]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.4), sharey=True, squeeze=False)
    for axis, (title, keys) in zip(axes[0], panels):
        for index, run in enumerate(runs):
            color = SERIES[index % len(SERIES)]
            raw = field(run, *keys)
            axis.plot(run["iteration"], raw, color=color, lw=0.6, alpha=0.25)
            axis.plot(run["iteration"], smooth(raw, window), color=color)
        axis.set_title(title)
        axis.set_xlabel("PPO iteration")
        axis.set_ylim(-0.02, 1.02)
    axes[0][0].set_ylabel("goal rate of finished episodes")
    fig.suptitle("Goal rate.  Fixed difficulty d: every knob drawn from its start value out to d of the way to "
                 "its limits (same task in every run).  Thin = per iteration, thick = moving average", y=1.04)
    run_legend(fig, runs, y=-0.04)
    fig.tight_layout()
    save(fig, out, "01_goal_rates.png")


def plot_adr_ranges(runs: list[dict], out: Path) -> None:
    parameters = runs[0]["info"]["adr_parameters"]
    fig, axes = plt.subplots(1, len(parameters), figsize=(4.4 * len(parameters), 3.6), squeeze=False)
    for axis, parameter in zip(axes[0], parameters):
        name = parameter["name"]
        for index, run in enumerate(runs):
            color = SERIES[index % len(SERIES)]
            low = field(run, "adr", "knobs", name, "low")
            high = field(run, "adr", "knobs", name, "high")
            axis.fill_between(run["iteration"], low, high, color=color, alpha=0.12, lw=0, step="post")
            axis.step(run["iteration"], low, where="post", color=color, lw=1.4)
            axis.step(run["iteration"], high, where="post", color=color, lw=1.4)
        for limit in (parameter["minimum"], parameter["maximum"]):
            axis.axhline(limit, color=INK_2, lw=0.8)
        axis.axhline(parameter["start"], color=MUTED, lw=0.8, ls=":")
        pad = 0.04 * (parameter["maximum"] - parameter["minimum"])
        axis.set_ylim(parameter["minimum"] - pad, parameter["maximum"] + pad)
        axis.set_title(KNOB_LABEL.get(name, name))
        axis.set_xlabel("PPO iteration")
    fig.suptitle("ADR ranges: the shaded band is where training episodes start.  "
                 "Solid grey = the knob's limits, dotted = its start value", y=1.03)
    run_legend(fig, runs, y=-0.04)
    fig.tight_layout()
    save(fig, out, "02_adr_ranges.png")


def plot_adr_probes(runs: list[dict], out: Path, window: int) -> None:
    info = runs[0]["info"]
    parameters = info["adr_parameters"]
    arguments = info.get("arguments", {})
    low_threshold = float(arguments.get("adr_low_threshold", 0.3))
    high_threshold = float(arguments.get("adr_high_threshold", 0.6))
    fig, axes = plt.subplots(2, len(parameters), figsize=(4.4 * len(parameters), 6.0), sharex=True,
                             sharey=True, squeeze=False)
    for column, parameter in enumerate(parameters):
        name = parameter["name"]
        for row, (edge, edge_label) in enumerate((("high_edge", "upper edge"), ("low_edge", "lower edge"))):
            axis = axes[row][column]
            axis.axhspan(low_threshold, high_threshold, color=GRID, alpha=0.5, lw=0)
            for index, run in enumerate(runs):
                color = SERIES[index % len(SERIES)]
                rate = field(run, "adr", "knobs", name, edge, "probe_success_rate")
                axis.plot(run["iteration"], smooth(rate, window), color=color, lw=1.4)
                moves = field(run, "adr", "knobs", name, edge, "moves")
                outward = np.flatnonzero(moves > 0)
                inward = np.flatnonzero(moves < 0)
                axis.plot(run["iteration"][outward], np.full(outward.size, 1.04), "^", ms=5, color=color,
                          clip_on=False)
                axis.plot(run["iteration"][inward], np.full(inward.size, -0.04), "v", ms=5, color=color,
                          clip_on=False)
            axis.set_title(f"{KNOB_LABEL.get(name, name)}: {edge_label}", fontsize=9)
            axis.set_ylim(-0.08, 1.08)
            if row == 1:
                axis.set_xlabel("PPO iteration")
        axes[0][column].set_ylabel("probe goal rate" if column == 0 else "")
        axes[1][column].set_ylabel("probe goal rate" if column == 0 else "")
    fig.suptitle(f"Probes: goal rate with one knob pinned at an edge.  Grey band = hold "
                 f"({low_threshold:g}-{high_threshold:g}); above it the edge moves out (▲), "
                 f"below it moves in (▼)", y=1.02)
    run_legend(fig, runs, y=-0.02)
    fig.tight_layout()
    save(fig, out, "03_adr_probes.png")


def plot_adr_health(runs: list[dict], out: Path) -> None:
    parameters = runs[0]["info"]["adr_parameters"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for index, run in enumerate(runs):
        color = SERIES[index % len(SERIES)]
        axes[0].plot(run["iteration"], field(run, "adr", "entropy"), color=color)
        coverage = np.mean([
            (field(run, "adr", "knobs", p["name"], "high") - field(run, "adr", "knobs", p["name"], "low"))
            / (p["maximum"] - p["minimum"]) for p in parameters], axis=0)
        axes[1].plot(run["iteration"], coverage, color=color)
        moves = sum(np.abs(field(run, "adr", "knobs", p["name"], edge, "moves"))
                    for p in parameters for edge in ("low_edge", "high_edge"))
        axes[2].plot(run["iteration"], np.cumsum(moves), color=color)
    axes[0].set_title("ADR entropy (mean log range width, nats)")
    axes[1].set_title("range width / limit width, mean of knobs")
    axes[1].set_ylim(-0.02, 1.02)
    axes[2].set_title("edge moves so far, all knobs")
    for axis in axes:
        axis.set_xlabel("PPO iteration")
    fig.suptitle("Is ADR still widening the task?  Flat lines mean the ranges stopped moving", y=1.03)
    run_legend(fig, runs, y=-0.06)
    fig.tight_layout()
    save(fig, out, "04_adr_health.png")


def plot_outcomes(runs: list[dict], out: Path, window: int) -> None:
    names = runs[0]["info"]["outcome_names"]
    fig, axes = plt.subplots(1, len(runs), figsize=(6.5 * len(runs), 3.8), squeeze=False, sharey=True)
    for axis, run in zip(axes[0], runs):
        counts = np.stack([field(run, "outcomes", name) for name in names])
        counts = np.nan_to_num(counts)
        totals = counts.sum(axis=0)
        share = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
        share = np.stack([smooth(row, window) for row in share])
        colors = [SERIES[index] for index in range(len(names))]
        axis.stackplot(run["iteration"], share, colors=colors, edgecolor=SURFACE, linewidth=0.8)
        tops = np.cumsum(share[:, -1])
        for index, name in enumerate(names):
            middle = tops[index] - share[index, -1] / 2.0
            if share[index, -1] > 0.03:
                axis.annotate(OUTCOME_LABEL.get(name, name), (run["iteration"][-1], middle), xytext=(6, 0),
                              textcoords="offset points", va="center", fontsize=7.5, color=INK_2)
        axis.set_ylim(0, 1)
        axis.set_xlim(run["iteration"][0], run["iteration"][-1])
        axis.set_title(run["label"] if len(runs) > 1 else "how finished episodes end")
        axis.set_xlabel("PPO iteration")
    axes[0][0].set_ylabel("share of finished episodes")
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[index]) for index in range(len(names))]
    fig.legend(handles, [OUTCOME_LABEL.get(name, name) for name in names], loc="upper center",
               ncol=len(names), bbox_to_anchor=(0.5, -0.0))
    fig.suptitle("Episode outcomes (all episode kinds).  A goal on the same tick as anything unsafe "
                 "counts as the unsafe outcome", y=1.03)
    fig.tight_layout()
    save(fig, out, "05_outcomes.png")


def knob_grid(runs: list[dict], first_fraction: float, last_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Episodes and goals per knob-grid cell, pooled over runs and over the
    records between the given fractions of each run."""

    edges = runs[0]["info"]["knob_grid_edges"]
    shape = tuple(len(axis) - 1 for axis in edges)
    episodes = np.zeros(shape)
    goals = np.zeros(shape)
    for run in runs:
        records = run["records"]
        start = int(np.floor(first_fraction * len(records)))
        stop = max(start + 1, int(np.ceil(last_fraction * len(records))))
        for record in records[start:stop]:
            for row in record.get("knob_grid", []):
                cell = tuple(row[:len(shape)])
                episodes[cell] += row[len(shape)]
                goals[cell] += row[len(shape) + 1]
    return episodes, goals


def plot_knob_heatmaps(runs: list[dict], out: Path, window_fraction: float) -> None:
    edges = [np.asarray(axis) for axis in runs[0]["info"]["knob_grid_edges"]]
    names = [parameter["name"] for parameter in runs[0]["info"]["adr_parameters"]]
    episodes, goals = knob_grid(runs, 1.0 - window_fraction, 1.0)
    knobs = range(len(names))
    pairs = [(x_axis, y_axis) for x_axis in knobs for y_axis in knobs if x_axis < y_axis]
    fig, axes = plt.subplots(2, len(pairs), figsize=(4.3 * len(pairs), 7.4), squeeze=False)
    for column, (x_axis, y_axis) in enumerate(pairs):
        other = tuple(index for index in knobs if index not in (x_axis, y_axis))
        count = episodes.sum(axis=other).T
        goal = goals.sum(axis=other).T
        rate = np.where(count >= 10, goal / np.maximum(count, 1), np.nan)
        for row, (values, limits, title) in enumerate((
            (rate, (0.0, 1.0), "goal rate (cells with >= 10 episodes)"),
            (np.where(count > 0, count, np.nan), (None, None), "episodes started there"),
        )):
            axis = axes[row][column]
            # Cells are drawn at equal size (the grid is finer near the
            # start) with the cell edges as tick labels.
            image = axis.pcolormesh(np.arange(len(edges[x_axis])), np.arange(len(edges[y_axis])), values,
                                    cmap=SEQUENTIAL, vmin=limits[0], vmax=limits[1], edgecolors=SURFACE,
                                    linewidth=1.0)
            axis.grid(False)
            axis.set_xticks(np.arange(len(edges[x_axis])), [f"{edge:g}" for edge in edges[x_axis]], fontsize=7)
            axis.set_yticks(np.arange(len(edges[y_axis])), [f"{edge:g}" for edge in edges[y_axis]], fontsize=7)
            axis.set_xlabel(KNOB_LABEL.get(names[x_axis], names[x_axis]))
            axis.set_ylabel(KNOB_LABEL.get(names[y_axis], names[y_axis]))
            axis.set_title(title, fontsize=9)
            colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
            colorbar.outline.set_visible(False)
    fig.suptitle(f"Where the policy scores, over each pair of knobs (other knobs pooled), last "
                 f"{window_fraction:.0%} of training, all runs and episode kinds pooled", y=1.01)
    fig.tight_layout()
    save(fig, out, "06_knob_heatmaps.png")


def plot_goal_by_knob(runs: list[dict], out: Path) -> None:
    edges = [np.asarray(axis) for axis in runs[0]["info"]["knob_grid_edges"]]
    names = [parameter["name"] for parameter in runs[0]["info"]["adr_parameters"]]
    windows = ((0.0, 0.2, "first 20%"), (0.4, 0.6, "middle 20%"), (0.8, 1.0, "last 20%"))
    shades = (BLUE_RAMP[3], BLUE_RAMP[7], BLUE_RAMP[11])
    fig, axes = plt.subplots(1, len(names), figsize=(4.3 * len(names), 3.6), sharey=True)
    for (first, last, label), shade in zip(windows, shades):
        episodes, goals = knob_grid(runs, first, last)
        for knob, axis in enumerate(axes):
            other = tuple(index for index in range(len(names)) if index != knob)
            count = episodes.sum(axis=other)
            rate = np.where(count >= 10, goals.sum(axis=other) / np.maximum(count, 1), np.nan)
            centres = (edges[knob][:-1] + edges[knob][1:]) / 2
            axis.plot(centres, rate, "-o", color=shade, ms=4, lw=1.6, label=label)
    for knob, axis in enumerate(axes):
        axis.set_xlabel(KNOB_LABEL.get(names[knob], names[knob]))
        axis.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("goal rate (bins with >= 10 episodes)")
    fig.legend([plt.Line2D([], [], color=shade, marker="o", ms=4) for shade in shades],
               [f"{label} of training" for _, _, label in windows], loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, -0.0))
    fig.suptitle("Goal rate along each knob (other knobs pooled over what was sampled), "
                 "early vs late in training", y=1.03)
    fig.tight_layout()
    save(fig, out, "07_goal_by_knob.png")


def plot_episode_timing(runs: list[dict], out: Path, window: int) -> None:
    panels = (
        ("mean episode length (s)", ("episode_seconds", "mean")),
        ("mean time to goal (s)", ("episode_seconds", "mean_goal")),
        ("mean time to first ball contact (s)", ("episode_seconds", "mean_first_contact")),
        ("mean ball progress toward goal (m)", ("mean_goal_axis_ball_displacement_m",)),
        ("ball-contact rate", ("contact_rate",)),
        ("unsafe-end rate", ("unsafe_rate",)),
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharex=True)
    for axis, (title, keys) in zip(axes.flat, panels):
        for index, run in enumerate(runs):
            axis.plot(run["iteration"], smooth(field(run, *keys), window), color=SERIES[index % len(SERIES)])
        axis.set_title(title)
    for axis in axes[-1]:
        axis.set_xlabel("PPO iteration")
    fig.suptitle("Episodes: how long they last and what the robot does with the ball (moving average)", y=1.02)
    run_legend(fig, runs, y=-0.02)
    fig.tight_layout()
    save(fig, out, "08_episode_timing.png")


def plot_ppo(runs: list[dict], out: Path, window: int) -> None:
    losses = sorted({name for run in runs for record in run["records"] for name in record.get("losses", {})})
    potentials = sorted({name for run in runs for record in run["records"]
                         for name in record.get("mean_shaping_by_potential", {})})
    panels = [(f"loss: {name.replace('_', ' ')}", ("losses", name)) for name in losses]
    panels += [("mean episode return", ("mean_episode_return",)),
               ("mean shaping return", ("mean_shaping_return",))]
    panels += [(f"shaping: {name.replace('_', ' ')}", ("mean_shaping_by_potential", name)) for name in potentials]
    columns = min(len(panels), 4)
    rows = int(np.ceil(len(panels) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 3.0 * rows), squeeze=False, sharex=True)
    for axis, (title, keys) in zip(axes.flat, panels):
        for index, run in enumerate(runs):
            axis.plot(run["iteration"], smooth(field(run, *keys), window), color=SERIES[index % len(SERIES)])
        axis.set_title(title)
    for axis in list(axes.flat)[len(panels):]:
        axis.set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("PPO iteration")
    fig.suptitle("PPO losses, episode return, and the undiscounted shaping return per potential "
                 "(moving average)", y=1.02)
    run_legend(fig, runs, y=-0.02)
    fig.tight_layout()
    save(fig, out, "09_ppo.png")


def plot_throughput(runs: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    for index, run in enumerate(runs):
        axes[0].plot(run["iteration"], field(run, "timing", "transitions_per_second"),
                     color=SERIES[index % len(SERIES)])
    axes[0].set_title("transitions per second (rollout + update)")
    axes[0].set_xlabel("PPO iteration")
    run = runs[0]
    # The parts take the last palette slots so they never share a colour
    # with a run in the left panel.
    parts = (("env_step_seconds", "environment steps", SERIES[5]),
             ("policy_seconds", "policy inference", SERIES[6]),
             ("update_seconds", "PPO update", SERIES[7]))
    for key, label, color in parts:
        values = field(run, "timing", key)
        axes[1].plot(run["iteration"], values, color=color)
        direct_label(axes[1], run["iteration"], values, label, color)
    axes[1].set_title(f"seconds per iteration by part ({run['label']})")
    axes[1].set_xlabel("PPO iteration")
    axes[1].set_ylim(bottom=0)
    axes[1].legend([plt.Line2D([], [], color=color) for _, _, color in parts], [label for _, label, _ in parts],
                   loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    axes[0].set_ylim(bottom=0)
    if len(runs) > 1:
        axes[0].legend([plt.Line2D([], [], color=SERIES[i % len(SERIES)]) for i in range(len(runs))],
                       [item["label"] for item in runs], loc="upper center", bbox_to_anchor=(0.5, -0.16),
                       ncol=min(len(runs), 4))
    fig.suptitle("Throughput: where each iteration's wall time goes", y=1.03)
    fig.tight_layout()
    save(fig, out, "10_throughput.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--labels", help="comma-separated run labels (default: directory names)")
    parser.add_argument("--smooth", type=int, default=5, help="moving-average window in iterations")
    parser.add_argument("--window", type=float, default=0.2, help="share of training pooled for heatmaps")
    args = parser.parse_args()
    labels = args.labels.split(",") if args.labels else [path.name for path in args.runs]
    if len(labels) != len(args.runs):
        raise SystemExit("--labels needs one label per run")
    runs = [load_run(path, label) for path, label in zip(args.runs, labels)]
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    plot_goal_rates(runs, args.out, args.smooth)
    plot_adr_ranges(runs, args.out)
    plot_adr_probes(runs, args.out, args.smooth)
    plot_adr_health(runs, args.out)
    plot_outcomes(runs, args.out, args.smooth)
    plot_knob_heatmaps(runs, args.out, args.window)
    plot_goal_by_knob(runs, args.out)
    plot_episode_timing(runs, args.out, args.smooth)
    plot_ppo(runs, args.out, args.smooth)
    plot_throughput(runs, args.out)


if __name__ == "__main__":
    main()
