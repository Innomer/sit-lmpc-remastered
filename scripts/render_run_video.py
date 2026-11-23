"""Render a smooth video of a sim run from its saved laps (sim time, not wall-clock lag).

    python scripts/render_run_video.py results/Spielberg/world-nochange --speedup 2
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection

from sit_lmpc.track.resources import map_path
from sit_lmpc.track.track import Track

TIMESTEP = 0.1
OURS = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#52514e"


def load_attempts(run):
    attempts = []
    for outcome in ("success", "failure", "incomplete"):
        for path in sorted((run / "trajectories" / outcome).glob("attempt-*.npz")):
            with np.load(path, allow_pickle=True) as data:
                attempts.append({"id": int(path.stem.split("-")[1]), "outcome": outcome,
                                 "cartesian": data["cartesian"], "frenet": data["frenet"],
                                 "controllers": data["controllers"]})
    return sorted(attempts, key=lambda a: a["id"])


def true_friction_events(run):
    """(lap, track-wide mu or None, patches or None) for each applied scheduler event, in order."""
    path = Path(str(run) + ".conditions.jsonl")
    events = []
    if path.exists():
        for line in path.read_text().splitlines():
            event = json.loads(line)["event"]
            mu = event.get("condition", {}).get("mu")
            events.append((event["lap"], mu, event.get("patches")))
    return events


def true_mu(events, finished_before, s):
    """The scheduler counts finished laps (crashed attempts don't count), so an event "after lap N" applies to every
    attempt that starts after the N-th finished lap."""
    mu = 1.0489
    patches = []
    for lap, value, new_patches in events:
        if finished_before >= lap:
            if value is not None:
                mu = value
            if new_patches is not None:
                patches = new_patches
    for patch in patches:
        if patch["s_start"] <= s <= patch["s_end"]:
            return patch["mu"]
    return mu


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--speedup", type=float, default=2.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--from-attempt", type=int, default=1)
    parser.add_argument("--trail", type=float, default=4.0, help="seconds of path drawn behind the car")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run = Path(args.run)
    if not run.is_dir():
        names = sorted(p.name for p in run.parent.glob("*/") if (p / "trajectories").is_dir())
        raise SystemExit(f"no run folder {run}\nruns with saved laps: {', '.join(names)}")
    attempts = [a for a in load_attempts(run) if a["id"] >= args.from_attempt]
    if not attempts:
        raise SystemExit("no saved attempts yet")
    track = Track.from_track_path(map_path("Spielberg"))
    friction_path = run / "encoder_grip.npy" if (run / "encoder_grip.npy").exists() else run / "encoder_friction.npy"
    friction = np.load(friction_path) if friction_path.exists() else None
    events = true_friction_events(run)

    states, labels, true_values = [], [], []
    step = 0
    finished = sum(1 for a in load_attempts(run) if a["id"] < attempts[0]["id"] and a["outcome"] == "success")
    for attempt in attempts:
        controllers = list(attempt["controllers"]) + [attempt["controllers"][-1]]
        for index, state in enumerate(attempt["cartesian"]):
            states.append(state)
            true_values.append(true_mu(events, finished, float(attempt["frenet"][index, 0])))
            labels.append((attempt["id"], attempt["outcome"], controllers[min(index, len(controllers) - 1)],
                           index * TIMESTEP, step))
            step += 1
        finished += attempt["outcome"] == "success"
    states = np.asarray(states)
    stride = max(1, int(round(args.speedup / (args.fps * TIMESTEP))))
    frames = range(0, len(states), stride)

    fig = plt.figure(figsize=(12, 7.5), facecolor="#fcfcfb")
    ax = fig.add_axes([0.02, 0.05, 0.62, 0.9])
    ax.set_facecolor("#fcfcfb")
    xs = np.asarray(track.centerline.xs)
    ys = np.asarray(track.centerline.ys)
    ax.plot(xs, ys, color="#d6dde3", linewidth=14, solid_capstyle="round", zorder=1)
    ax.plot(xs, ys, color="#a9a8a2", linewidth=0.8, linestyle="--", zorder=2)
    arc = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))))
    for lap, _, patches in events:
        for patch in patches or []:
            inside = (arc >= patch["s_start"]) & (arc <= patch["s_end"])
            ax.plot(xs[inside], ys[inside], color="#e0a96d", linewidth=14, solid_capstyle="butt", zorder=1.5)
            middle = np.flatnonzero(inside)[len(np.flatnonzero(inside)) // 2]
            ax.annotate(f"μ {patch['mu']:g} after {lap} finished laps", (xs[middle], ys[middle]), xytext=(12, 12),
                        textcoords="offset points", fontsize=9, color=INK)
    pad = 5
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    trail = LineCollection([], cmap="viridis", linewidths=3, zorder=3)
    trail.set_clim(0, max(15.0, float(states[:, 3].max())))
    ax.add_collection(trail)
    car, = ax.plot([], [], marker="o", markersize=9, color=OURS, zorder=5)
    heading, = ax.plot([], [], color=OURS, linewidth=2.5, zorder=5)
    info = fig.text(0.67, 0.92, "", fontsize=13, color=INK, va="top", family="monospace")
    side = fig.add_axes([0.68, 0.12, 0.29, 0.3])
    if friction is not None and np.isfinite(friction).any():
        side.set_title("Grip: model's estimate vs true", fontsize=10, loc="left", color=INK)
        side.set_xlabel("sim time [s]", fontsize=9, color=MUTED)
        side.set_ylabel("friction μ", fontsize=9, color=MUTED)
        side.tick_params(labelsize=8, colors=MUTED)
        side.grid(color="#e4e3df")
        raw_line, = side.plot([], [], color=OURS, linewidth=0.6, alpha=0.25)
        estimate_line, = side.plot([], [], color=OURS, linewidth=1.8, label="model estimate (2 s average)")
        true_line, = side.plot([], [], color=INK, linewidth=2, label="true")
        side.legend(fontsize=8, frameon=False, loc="lower left")
        side.set_ylim(0.4, 1.4)
    else:
        side.axis("off")
        estimate_line = true_line = raw_line = None
    fig.colorbar(trail, ax=ax, orientation="horizontal", fraction=0.04, pad=0.0, shrink=0.5, label="speed [m/s]")

    truth = np.asarray(true_values)
    if friction is not None:
        window = int(2.0 / TIMESTEP)
        smooth = np.convolve(np.nan_to_num(friction, nan=1.0), np.ones(window) / window, mode="full")[:len(friction)]
        smooth[:window] = np.nan

    out = Path(args.out) if args.out else run.parent / f"{run.name}.mp4"
    writer = FFMpegWriter(fps=args.fps, codec="libx264", extra_args=["-crf", "30", "-preset", "slow", "-pix_fmt", "yuv420p"])
    trail_steps = int(args.trail / TIMESTEP)
    with writer.saving(fig, str(out), dpi=112):
        for frame in frames:
            start = max(0, frame - trail_steps)
            points = states[start:frame + 1, :2]
            segments = np.stack((points[:-1], points[1:]), axis=1) if len(points) > 1 else np.zeros((0, 2, 2))
            trail.set_segments(segments)
            trail.set_array(states[start + 1:frame + 1, 3])
            x, y, _, speed, yaw = states[frame, :5]
            car.set_data([x], [y])
            heading.set_data([x, x + 2.5 * np.cos(yaw)], [y, y + 2.5 * np.sin(yaw)])
            attempt_id, outcome, controller, lap_time, global_step = labels[frame]
            mode = "learned (LMPPI)" if controller == "lmppi" else "warm-up (no learning)"
            text = (f"{run.name}\n\nattempt  {attempt_id}\ncontroller {mode}\nlap time {lap_time:5.1f} s\n"
                    f"speed    {speed:5.2f} m/s\ntrue μ   {truth[frame]:.2f}")
            if outcome != "success" and frame == len(states) - 1:
                text += f"\n\nattempt ended: {outcome}"
            info.set_text(text)
            if estimate_line is not None:
                upto = min(global_step + 1, len(friction))
                t = np.arange(upto) * TIMESTEP
                raw_line.set_data(t, friction[:upto])
                estimate_line.set_data(t, smooth[:upto])
                true_line.set_data(t, truth[:upto])
                side.set_xlim(0, max(10.0, t[-1] if upto else 10.0))
            writer.grab_frame()
    print(f"saved {out}  ({len(list(frames))} frames, {args.speedup}x sim speed)")


if __name__ == "__main__":
    main()
