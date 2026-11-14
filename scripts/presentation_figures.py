"""Presentation figures and CSV tables from one or more sit-racecar run folders.

python scripts/presentation_figures.py results/Spielberg/fresh25 --out results/figures/fresh25
python scripts/presentation_figures.py results/Spielberg/fresh25 \
    --timing "CPU=results/Spielberg/cpu-full1-20261113" "GPU=results/Spielberg/fresh25" \
    --steering "action_scale 3.2=results/Spielberg/cpu-full1-20261113" "action_scale 1.0=results/Spielberg/fresh25"
"""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import LineCollection

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
TRACK_EDGE = "#8a8984"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SPEED_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": TEXT, "axes.labelcolor": TEXT_SECONDARY, "axes.edgecolor": GRID,
        "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlelocation": "left",
        "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.frameon": False, "legend.fontsize": 10, "lines.linewidth": 2,
        "font.family": "DejaVu Sans",
    })


def save(fig, out, name):
    for extension in ("png", "svg"):
        fig.savefig(out / f"{name}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def labeled(spec):
    label, _, path = spec.partition("=")
    return label, Path(path)


def load_attempts(run):
    attempts = []
    for path in sorted((run / "trajectories").glob("*/attempt-*.npz")):
        data = np.load(path, allow_pickle=True)
        attempts.append({
            "id": int(path.stem.split("-")[1]),
            "outcome": path.parent.name,
            "cartesian": data["cartesian"],
            "frenet": data["frenet"],
            "controls": data["controls"],
            "times": data["times"],
            "progress": data["progress"],
            "controllers": np.asarray(data["controllers"]).astype(str),
        })
    attempts.sort(key=lambda attempt: attempt["id"])
    return attempts


def controller(attempt):
    names, counts = np.unique(attempt["controllers"], return_counts=True)
    return str(names[np.argmax(counts)]) if len(names) else "?"


def track_for(run):
    from sit_lmpc.track import Track, map_path

    config = json.loads(json.dumps(__import__("yaml").safe_load((run / "config.yaml").read_text())))
    return Track.from_track_path(map_path(config["simulation"]["map"])), config


def lap_table(run, out):
    attempts = {attempt["id"]: attempt for attempt in load_attempts(run)}
    metrics = json.loads((run / "lap_metrics.json").read_text())
    rows = []
    for m in metrics:
        attempt = attempts.get(m["attempt_id"])
        rows.append({
            "lap": m["lap"],
            "controller": controller(attempt) if attempt else "?",
            "outcome": m["outcome"],
            "lap_time_s": m["lap_time"],
            "mean_speed_mps": m["mean_speed"],
            "max_speed_mps": float(attempt["cartesian"][:, 3].max()) if attempt else None,
            "max_abs_ey_m": m["max_abs_lateral_offset"],
            "out_of_bounds_steps": m["out_of_bounds_steps"],
            "safe_set_violation_max": m.get("safe_set_max"),
        })
    with (out / "laps.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_lap_times(rows, out):
    laps = [row for row in rows if row["outcome"] == "success"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    warmup = [row for row in laps if row["controller"] == "mppi"]
    learned = [row for row in laps if row["controller"] == "lmppi"]
    for group, color, name in ((warmup, SERIES[0], "Warm-up (MPPI tracking)"),
                               (learned, SERIES[1], "SIT-LMPC (LMPPI)")):
        if group:
            ax.plot([row["lap"] for row in group], [row["lap_time_s"] for row in group],
                    color=color, marker="o", markersize=7, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, label=name)
    if warmup:
        reference = float(np.mean([row["lap_time_s"] for row in warmup]))
        ax.axhline(reference, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(f"warm-up {reference:.1f} s", (laps[-1]["lap"], reference), xytext=(0, 5),
                    textcoords="offset points", ha="right", color=TEXT_SECONDARY, fontsize=9)
    if learned:
        best = min(learned, key=lambda row: row["lap_time_s"])
        ax.annotate(f"best {best['lap_time_s']:.1f} s", (best["lap"], best["lap_time_s"]),
                    xytext=(-6, -16), textcoords="offset points", ha="right", color=TEXT, fontsize=9)
    ax.margins(y=0.12)
    ax.set_xlabel("Lap")
    ax.set_ylabel("Lap time (s)")
    ax.set_title("Lap time per lap")
    ax.legend(loc="upper right")
    ax.xaxis.get_major_locator().set_params(integer=True)
    save(fig, out, "lap_times")


def best_and_warmup(attempts):
    laps = [attempt for attempt in attempts if attempt["outcome"] == "success"]
    warmup = [attempt for attempt in laps if controller(attempt) == "mppi"]
    learned = [attempt for attempt in laps if controller(attempt) == "lmppi"]
    selected = []
    if warmup:
        selected.append(("Warm-up lap", warmup[0], SERIES[0]))
    if learned:
        best = min(learned, key=lambda attempt: attempt["times"][-1])
        selected.append((f"Best SIT-LMPC lap (lap {best['id']})", best, SERIES[1]))
    return selected


def plot_track_speed(run, attempts, out):
    selected = best_and_warmup(attempts)
    if not selected:
        return
    track, config = track_for(run)
    half_width = config["planner"]["half_width"]
    s = np.linspace(0, track.s_frame_max, 1500)
    edges = [np.array([track.frenet_to_cartesian(value, side * half_width, 0)[:2] for value in s])
             for side in (-1, 1)]
    peak = max(attempt["cartesian"][:, 3].max() for _, attempt, _ in selected)
    norm = Normalize(0, np.ceil(peak))
    cmap = LinearSegmentedColormap.from_list("speed", SPEED_RAMP)
    fig, axes = plt.subplots(1, len(selected), figsize=(6.5 * len(selected), 6.4), squeeze=False,
                             layout="constrained")
    for ax, (name, attempt, _) in zip(axes[0], selected):
        for edge in edges:
            ax.plot(edge[:, 0], edge[:, 1], color=TRACK_EDGE, linewidth=1)
        points = attempt["cartesian"][:, :2]
        segments = np.stack([points[:-1], points[1:]], axis=1)
        line = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.5)
        line.set_array(attempt["cartesian"][1:, 3])
        ax.add_collection(line)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{name}\n{attempt['times'][-1]:.1f} s, mean {attempt['cartesian'][:, 3].mean():.2f} m/s",
                     fontsize=12)
    bar = fig.colorbar(line, ax=axes[0].tolist(), orientation="horizontal", shrink=0.5, aspect=30)
    bar.set_label("Speed (m/s)")
    bar.outline.set_visible(False)
    save(fig, out, "track_speed")


def plot_profiles(run, attempts, out):
    selected = best_and_warmup(attempts)
    if not selected:
        return
    _, config = track_for(run)
    half_width = config["planner"]["half_width"]
    limit = half_width - config["planner"]["boundary_buffer"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for name, attempt, color in selected:
        ax.plot(attempt["progress"], attempt["cartesian"][:, 3], color=color, label=name)
    ax.set_xlabel("Distance along the lap (m)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Speed along the lap")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2, borderaxespad=0)
    save(fig, out, "speed_profile")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for value, style_, name in ((half_width, "-", "track edge"), (limit, (0, (4, 3)), "planner limit")):
        for sign in (-1, 1):
            ax.axhline(sign * value, color=TRACK_EDGE, linewidth=1, linestyle=style_)
        ax.annotate(f"{name} ±{value:.2f} m", (0, value), xytext=(2, 3), textcoords="offset points",
                    color=TEXT_SECONDARY, fontsize=9)
    for name, attempt, color in selected:
        ax.plot(attempt["progress"], attempt["frenet"][:, 1], color=color, label=name)
    ax.set_xlabel("Distance along the lap (m)")
    ax.set_ylabel("Lateral offset from centerline (m)")
    ax.set_title("Lateral offset along the lap")
    ax.set_ylim(-half_width * 1.15, half_width * 1.3)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2, borderaxespad=0)
    save(fig, out, "lateral_offset")


def planning_steps(run):
    timing = run / "control_timing.npz"
    if not timing.is_file():
        return None
    planning = np.load(timing)["planning"]
    modes = np.concatenate([attempt["controllers"] for attempt in load_attempts(run)])
    count = min(len(planning), len(modes))
    return planning[:count] * 1e3, modes[:count]


def window_summaries(run):
    """Planning time from the 100-step control_step_timing events, for runs stopped before close."""
    events = [json.loads(line) for line in (run / "events.jsonl").open()]
    started = next((event["step"] for event in events if event["event"] == "learned_controller_started"), None)
    windows = {"mppi": [], "lmppi": []}
    for event in events:
        if event["event"] != "control_step_timing" or event.get("planning") is None:
            continue
        first = event["steps"] - event["planning"]["count"]
        if first == 0:
            continue
        mode = "lmppi" if started is not None and first >= started else "mppi"
        if started is not None and first < started < event["steps"]:
            continue
        windows[mode].append(event["planning"])
    return windows


def plot_timing(runs, out):
    rows = []
    data = []
    for label, run in runs:
        steps = planning_steps(run)
        for mode, name in (("mppi", "Warm-up MPPI"), ("lmppi", "SIT-LMPC LMPPI")):
            if steps is not None:
                planning, modes = steps
                values = planning[1:][modes[1:] == mode]
                if not len(values):
                    continue
                p5, p50, p95, p99 = np.percentile(values, [5, 50, 95, 99])
                row = {"run": label, "controller": name, "steps": len(values), "source": "every step",
                       "p50_ms": float(p50), "p95_ms": float(p95), "p99_ms": float(p99),
                       "max_ms": float(values.max()), "over_100ms_pct": float(100 * np.mean(values > 100))}
                data.append((f"{label}: {name}", p5, p50, p95, False))
            else:
                windows = window_summaries(run)[mode]
                if not windows:
                    continue
                p50s = 1e3 * np.array([window["p50"] for window in windows])
                p95s = 1e3 * np.array([window["p95"] for window in windows])
                row = {"run": label, "controller": name, "steps": int(sum(window["count"] for window in windows)),
                       "source": "100-step summaries", "p50_ms": float(np.median(p50s)),
                       "p95_ms": float(np.median(p95s)), "p99_ms": None,
                       "max_ms": float(1e3 * max(window["max"] for window in windows)), "over_100ms_pct": None}
                data.append((f"{label}: {name}*", float(p50s.min()), float(np.median(p50s)), float(np.median(p95s)), True))
            rows.append(row)
    if not rows:
        return
    with (out / "planning_time.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(8, 0.75 * len(data) + 1.6))
    positions = np.arange(len(data))[::-1]
    for position, (name, p5, p50, p95, summarized) in zip(positions, data):
        color = SERIES[1] if "LMPPI" in name else SERIES[0]
        ax.plot([p5, p95], [position, position], color=color, linewidth=6, solid_capstyle="round", alpha=0.35)
        ax.plot([p50], [position], marker="o", markersize=9, color=color, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(f"p50 {p50:.0f} ms · p95 {p95:.0f} ms", (p95, position), xytext=(8, -4),
                    textcoords="offset points", color=TEXT, fontsize=9)
    ax.axvline(100, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("100 ms control period", (100, positions[0] + 0.45), xytext=(4, 0), textcoords="offset points",
                color=TEXT_SECONDARY, fontsize=9)
    ax.set_xscale("log")
    ticks = [value for value in (5, 10, 20, 50, 100, 200, 500, 1000)
             if min(item[1] for item in data) / 1.5 <= value <= max(item[3] for item in data) * 1.5]
    ax.set_xticks(ticks, [str(value) for value in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(positions, [item[0] for item in data])
    label = "Planning time per control step (ms, log scale); bar = 5th–95th percentile, dot = median"
    if any(item[4] for item in data):
        label += "\n* run stopped early: from 100-step summaries (median of window p50 / p95)"
    ax.set_xlabel(label)
    ax.set_title("Time per control step")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.7, len(data) - 0.2)
    save(fig, out, "planning_time")


def plot_steering(runs, out, seconds=30.0):
    selected = []
    for label, run in runs:
        laps = [attempt for attempt in load_attempts(run) if controller(attempt) == "mppi"]
        if laps:
            selected.append((label, laps[0]))
    if not selected:
        return
    fig, axes = plt.subplots(len(selected), 1, figsize=(10, 2.2 * len(selected) + 0.6), sharex=True, sharey=True,
                             squeeze=False)
    for ax, (label, attempt) in zip(axes[:, 0], selected):
        mask = attempt["times"] <= seconds
        delta = attempt["cartesian"][mask, 2]
        change = np.abs(np.diff(delta)).mean()
        ax.plot(attempt["times"][mask], delta, color=SERIES[0])
        ax.set_title(f"{label}: mean change {change:.3f} rad per step, max |δ| {np.abs(delta).max():.2f} rad",
                     fontsize=11)
        ax.set_ylabel("Steering (rad)")
    axes[-1, 0].set_xlabel("Time into the warm-up lap (s)")
    fig.suptitle("Warm-up steering angle", x=0.01, ha="left", fontweight="bold", fontsize=13)
    save(fig, out, "warmup_steering")


def plot_value_loss(run, out):
    events = [json.loads(line) for line in (run / "events.jsonl").open()]
    training = [event for event in events if event["event"] == "training"]
    if not training:
        return
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot([event["generation"] for event in training], [event["loss"] for event in training],
            color=SERIES[0], marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_xlabel("Value-function generation (one per completed lap)")
    ax.set_ylabel("Final training loss (NF negative log-likelihood)")
    ax.set_title("Value-function training")
    ax.xaxis.get_major_locator().set_params(integer=True)
    save(fig, out, "value_training")


def successful_laps(run):
    metrics = json.loads((run / "lap_metrics.json").read_text())
    attempts = {attempt["id"]: attempt for attempt in load_attempts(run)}
    laps = []
    for m in metrics:
        if m["outcome"] == "success":
            laps.append((m["lap"], m["lap_time"], controller(attempts[m["attempt_id"]])))
    return laps


def plot_compare(runs, out):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    rows = []
    warmup_end = 0
    for (label, run), color in zip(runs, SERIES):
        laps = successful_laps(run)
        warmup_end = max(warmup_end, sum(mode == "mppi" for _, _, mode in laps))
        ax.plot([lap for lap, _, _ in laps], [time for _, time, _ in laps], color=color, marker="o",
                markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
        best = min(laps, key=lambda item: item[1])
        ax.annotate(f"{label}: best {best[1]:.1f} s", (best[0], best[1]), xytext=(-6, -16),
                    textcoords="offset points", ha="right", color=TEXT, fontsize=9)
        rows += [{"run": label, "lap": lap, "controller": mode, "lap_time_s": time} for lap, time, mode in laps]
    ax.axvspan(0.5, warmup_end + 0.5, color=GRID, alpha=0.6, linewidth=0)
    ax.annotate("warm-up", (0.6, 1.0), xycoords=("data", "axes fraction"), xytext=(2, -14),
                textcoords="offset points", color=TEXT_SECONDARY, fontsize=9)
    ax.margins(y=0.12)
    ax.set_xlabel("Lap")
    ax.set_ylabel("Lap time (s)")
    ax.set_title("Lap time per lap")
    ax.legend(loc="upper right")
    ax.xaxis.get_major_locator().set_params(integer=True)
    save(fig, out, "compare_lap_times")
    with (out / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_authors_ratio(known_run, bnn_run, out):
    root = Path(__file__).resolve().parents[1] / "examples" / "racecar" / "data"

    def recorded(name):
        rows = csv.DictReader((root / name).open())
        return [float(row["lap_time"]) for row in rows if row["completed"] == "True"]

    authors_known, authors_bnn = recorded("known-dynamics-laps.csv"), recorded("bnn-unknown-laps.csv")
    ours_known = [time for _, time, mode in successful_laps(known_run) if mode == "lmppi"]
    ours_bnn = [time for _, time, mode in successful_laps(bnn_run) if mode == "lmppi"]
    count = min(len(ours_known), len(ours_bnn))
    authors_count = min(len(authors_known), len(authors_bnn)) - 5
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axhline(1.0, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
    ax.plot(range(1, authors_count + 1), np.asarray(authors_bnn[5:5 + authors_count]) / np.asarray(authors_known[5:5 + authors_count]),
            color=SERIES[0], label="Authors (full-scale car)")
    ax.plot(range(1, count + 1), np.asarray(ours_bnn[:count]) / np.asarray(ours_known[:count]), color=SERIES[1],
            marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2, label="Ours (F1TENTH, ROS sim)")
    ax.set_xscale("log")
    ticks = [1, 2, 5, 10, 20, 50, 100, 200]
    ax.set_xticks(ticks, [str(value) for value in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Learned (LMPPI) lap, log scale")
    ax.set_ylabel("BNN lap time ÷ known-model lap time")
    ax.set_title("BNN + chance constraints vs known dynamics")
    ax.legend(loc="upper left")
    save(fig, out, "bnn_vs_known_ratio")


def plot_belief(run, out):
    events = [json.loads(line) for line in (run / "events.jsonl").open()]
    updates = [event for event in events if event["event"] == "dynamics_posterior_training"]
    if not updates:
        return
    index = np.arange(1, len(updates) + 1)
    error = np.array([1e3 * event["heldout_xy10_rmse"] for event in updates])
    coverage = np.array([100 * event["heldout_coverage10"] for event in updates])
    held = np.array([event["status"] != "ready" for event in updates])
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True)
    for ax, values, label in ((axes[0], error, "Held-out 1 s position error (mm)"),
                              (axes[1], coverage, "Held-out coverage (%)")):
        ax.plot(index, values, color=SERIES[0])
        ax.plot(index[~held], values[~held], "o", color=SERIES[0], markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=1.2, label="update published")
        ax.plot(index[held], values[held], "o", color=SURFACE, markersize=7, markeredgecolor=SERIES[0],
                markeredgewidth=1.6, label="update held (failed calibration)")
        ax.set_ylabel(label)
    axes[1].axhline(95, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
    axes[1].annotate("95% target", (index[-1], 95), xytext=(0, 4), textcoords="offset points", ha="right",
                     color=TEXT_SECONDARY, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.0))
    axes[1].set_xlabel("Belief update (one per completed lap)")
    axes[1].xaxis.get_major_locator().set_params(integer=True)
    fig.suptitle("BNN belief between laps", x=0.01, ha="left", fontweight="bold", fontsize=13)
    save(fig, out, "bnn_belief")


def plot_fallbacks(run, out):
    events = [json.loads(line) for line in (run / "events.jsonl").open()]
    fallback_steps = [event["step"] for event in events if event["event"] == "probabilistic_safety_fallback"]
    if not any(event["event"] == "probabilistic_safety_status" for event in events):
        return
    attempts = load_attempts(run)
    starts = []
    total = 0
    for attempt in attempts:
        starts.append((attempt["id"], total, total + len(attempt["controls"]), attempt))
        total += len(attempt["controls"])
    labels, counts, fractions = [], [], []
    for number, (attempt_id, first, last, attempt) in enumerate(starts, 1):
        if controller(attempt) != "lmppi" or attempt["outcome"] != "success":
            continue
        count = sum(first <= step < last for step in fallback_steps)
        labels.append(number)
        counts.append(count)
        fractions.append(100 * count / max(last - first, 1))
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.set_axisbelow(True)
    ax.bar(labels, fractions, color=SERIES[1], width=0.6)
    for x, fraction, count in zip(labels, fractions, counts):
        ax.annotate(str(count), (x, fraction), xytext=(0, 3), textcoords="offset points", ha="center",
                    color=TEXT_SECONDARY, fontsize=8)
    ax.set_xlabel("Lap")
    ax.set_ylabel("Steps using the fallback (%)")
    ax.set_title("Chance-constraint fallbacks per learned lap (labels: step count)")
    ax.xaxis.get_major_locator().set_params(integer=True)
    save(fig, out, "chance_fallbacks")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="Run folder for lap, track, profile and value figures")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timing", nargs="*", default=None, help="LABEL=RUN pairs to compare planning time")
    parser.add_argument("--steering", nargs="*", default=None, help="LABEL=RUN pairs to compare warm-up steering")
    parser.add_argument("--compare", nargs="*", default=None, help="LABEL=RUN pairs for one lap-time chart")
    parser.add_argument("--authors", nargs=2, type=Path, default=None, metavar=("KNOWN_RUN", "BNN_RUN"),
                        help="Compare our BNN/known lap-time ratio with the authors' recorded runs")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    attempts = load_attempts(args.run)
    rows = lap_table(args.run, args.out)
    plot_lap_times(rows, args.out)
    plot_track_speed(args.run, attempts, args.out)
    plot_profiles(args.run, attempts, args.out)
    plot_value_loss(args.run, args.out)
    plot_timing([labeled(spec) for spec in args.timing] if args.timing else [("this run", args.run)], args.out)
    if args.steering:
        plot_steering([labeled(spec) for spec in args.steering], args.out)
    if args.compare:
        plot_compare([labeled(spec) for spec in args.compare], args.out)
    if args.authors:
        plot_authors_ratio(*args.authors, args.out)
    plot_belief(args.run, args.out)
    plot_fallbacks(args.run, args.out)
    print(f"figures and tables in {args.out}")


if __name__ == "__main__":
    main()
