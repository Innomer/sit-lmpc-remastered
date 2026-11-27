"""Cumulative (CDF) plots from scripts/eval_world_models.py results.

    python scripts/plot_world_eval.py results/wm-eval --out results/wm-eval/figures
    python scripts/plot_world_eval.py results/wm-eval results/wm-eval-v5 --labels rssm-v3 rssm-v5 --out results/wm-eval-compare

One eval folder: every model in it. Several folders: the bicycle and v3 encoder rows from the first, plus the RSSM row
of each folder under its label.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"bicycle_ekf": "#9a9a95", "bicycle_truth": "#c9c7bf", "v3_encoder": "#2a78d6", "rssm": "#d6612a",
          "rssm-v3": "#d6612a", "rssm-v5": "#7a3fb8", "ekf": "#9a9a95", "v3_readout": "#2a78d6"}
NICE = {"bicycle_ekf": "bicycle from EKF", "bicycle_truth": "bicycle from true state (reference)",
        "v3_encoder": "v3 encoder", "rssm": "RSSM", "ekf": "EKF", "v3_readout": "v3 read-out"}
ESTIMATES = ("steering [rad]", "speed [m/s]", "yaw rate [rad/s]", "slip [rad]")


def cdf(ax, values, label, color, dashed=False):
    values = np.sort(np.asarray(values)[np.isfinite(values)])
    ax.plot(values, np.arange(1, len(values) + 1) / len(values), color=color, linewidth=2.0,
            linestyle="--" if dashed else "-", label=f"{label} (median {np.median(values):.2f})")


def style(ax, xlabel, title, xmax):
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("share of predictions with error ≤ x", fontsize=10)
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False, loc="lower right")


def series(folders, labels, key):
    first = np.load(Path(folders[0]) / "raw.npz")
    out = []
    for name in ("bicycle_truth", "bicycle_ekf", "v3_encoder"):
        out.append((NICE[name], COLORS[name], name == "bicycle_truth", first, f"{key}_{name}"))
    if len(folders) == 1:
        out.append((NICE["rssm"], COLORS["rssm"], False, first, f"{key}_rssm"))
    else:
        for folder, label in zip(folders, labels):
            out.append((label, COLORS.get(label, "#d6612a"), False, np.load(Path(folder) / "raw.npz"), f"{key}_rssm"))
    return out, first


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    labels = args.labels or [Path(f).name for f in args.folders]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    position, first = series(args.folders, labels, "position")
    changed, maps = first["changed"], first["map"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for label, color, dashed, data, key in position:
        cdf(axes[0], data[key], label, color, dashed)
        cdf(axes[1], data[key][changed], label, color, dashed)
    style(axes[0], "position error after 1 s [m]", f"1 s prediction, all test predictions (n={len(changed)})", 2.0)
    style(axes[1], "position error after 1 s [m]", f"within 2 s after a grip change (n={int(changed.sum())})", 2.0)
    fig.tight_layout()
    fig.savefig(out / "cdf_position_1s.png", dpi=110)
    plt.close(fig)

    names = sorted(set(maps))
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5.2))
    for ax, name in zip(np.atleast_1d(axes), names):
        mask = maps == name
        for label, color, dashed, data, key in position:
            cdf(ax, data[key][mask], label, color, dashed)
        style(ax, "position error after 1 s [m]", f"{name} (n={int(mask.sum())})", 2.0)
    fig.tight_layout()
    fig.savefig(out / "cdf_position_1s_by_map.png", dpi=110)
    plt.close(fig)

    speed, _ = series(args.folders, labels, "speed")
    fig, ax = plt.subplots(figsize=(7, 5.2))
    for label, color, dashed, data, key in speed:
        if "true state" not in label:
            cdf(ax, data[key], label, color, dashed)
    style(ax, "speed error after 1 s [m/s]", "1 s prediction, speed", 1.0)
    fig.tight_layout()
    fig.savefig(out / "cdf_speed_1s.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    xmax = (0.08, 1.0, 0.4, 0.12)
    estimators = [("EKF", COLORS["ekf"], first["estimation_ekf"]), ("v3 encoder", COLORS["v3_encoder"],
                                                                    first["estimation_v3_encoder"])]
    if len(args.folders) == 1:
        estimators.append(("RSSM", COLORS["rssm"], first["estimation_rssm"]))
    else:
        for folder, label in zip(args.folders, labels):
            estimators.append((label, COLORS.get(label, "#d6612a"), np.load(Path(folder) / "raw.npz")["estimation_rssm"]))
    for i, ax in enumerate(axes):
        for label, color, values in estimators:
            cdf(ax, np.abs(values[:, i]), label, color)
        style(ax, f"absolute error, {ESTIMATES[i]}", f"state estimation: {ESTIMATES[i].split(' [')[0]}", xmax[i])
    fig.tight_layout()
    fig.savefig(out / "cdf_state_estimation.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    grips = [("v3 read-out", COLORS["v3_readout"], first["grip_v3_readout"], first["grip_window_v3_readout"])]
    if len(args.folders) == 1:
        grips.append(("RSSM condition head", COLORS["rssm"], first["grip_rssm"], first["grip_window_rssm"]))
    else:
        for folder, label in zip(args.folders, labels):
            data = np.load(Path(folder) / "raw.npz")
            grips.append((f"{label} condition head", COLORS.get(label, "#d6612a"), data["grip_rssm"], data["grip_window_rssm"]))
    for label, color, per_step, per_window in grips:
        cdf(axes[0], per_step, label, color)
        cdf(axes[1], per_window, label, color)
    style(axes[0], "absolute grip (friction) error", "grip estimate, every step", 0.5)
    style(axes[1], "absolute grip (friction) error", "grip estimate, averaged over each 8 s window", 0.5)
    fig.tight_layout()
    fig.savefig(out / "cdf_grip.png", dpi=110)
    plt.close(fig)
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
