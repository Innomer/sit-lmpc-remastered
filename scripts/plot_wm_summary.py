"""Two summary figures from data already on disk (no reruns): RSSM training curves and the offline evaluation.

    python scripts/plot_wm_summary.py --runs rssm-v1 rssm-v2 rssm-v3 rssm-v4 rssm-v5 --eval results/wm-eval --out docs/figures
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
GRAY = "#9a9a95"
LIGHT_GRAY = "#c9c7bf"


def base(ax):
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.grid(color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def training(runs, out):
    notes = {"rssm-v1": "v1 (blew up at 9.8k)", "rssm-v2": "v2 (drifted from 15k)", "rssm-v3": "v3 (stable, used)",
             "rssm-v4": "v4 (looser c walk, stopped)", "rssm-v5": "v5 (speed-weighted)"}
    panels = (("position_1s", "1 s position error [m]", 1.6), ("speed", "speed estimate error [m/s]", 1.2),
              ("condition", "grip head (1.0 = no skill)", 1.1))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), facecolor=SURFACE)
    for color, run in zip(SERIES, runs):
        history = json.loads((Path("runs") / run / "history.json").read_text())
        steps = np.array([r["step"] for r in history]) / 1000
        values = {"position_1s": [r["val"]["position_1s"] for r in history],
                  "speed": [r["val"]["anchor_rmse"][1] for r in history],
                  "condition": [r["val"]["condition"] for r in history]}
        for ax, (key, _, top) in zip(axes, panels):
            ax.plot(steps, np.minimum(values[key], top), color=color, linewidth=2.0, label=notes.get(run, run))
    for ax, (key, title, top) in zip(axes, panels):
        base(ax)
        ax.set_ylim(0, top)
        ax.set_xlim(0, 30)
        ax.set_xlabel("training step [thousands]", fontsize=10, color=MUTED)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
    axes[0].text(0.98, 0.97, "values above the top are cut off", transform=axes[0].transAxes, ha="right", va="top",
                 fontsize=8, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(runs), frameon=False, fontsize=9)
    fig.suptitle("RSSM training runs, validation set (lower is better)", fontsize=13, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(out / "rssm_training_curves.png", dpi=120, facecolor=SURFACE)
    plt.close(fig)


def evaluation(folder, out, label="rssm-v3"):
    summary = json.loads((Path(folder) / "summary.json").read_text())
    models = [("bicycle_ekf", "bicycle from EKF\n(what the car has)", GRAY),
              ("v3_encoder", "v3 encoder", SERIES[0]),
              ("rssm", f"RSSM ({label})", SERIES[1]),
              ("bicycle_truth", "bicycle from true state\n(reference only)", LIGHT_GRAY)]
    fig = plt.figure(figsize=(16, 9), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 4, height_ratios=(1.1, 1), hspace=0.45, wspace=0.35)

    ax = fig.add_subplot(grid[0, :2])
    base(ax)
    for i, (key, label, color) in enumerate(models):
        s = summary["prediction"][key]["all"]
        median, p90 = s["median"][0], s["p90"][0]
        ax.barh(i, median, color=color, height=0.55)
        ax.plot([median, p90], [i, i], color=MUTED, linewidth=1.5)
        ax.plot(p90, i, marker="|", color=MUTED, markersize=12)
        ax.text(p90 + 0.03, i, f"median {median:.2f} m · 90% under {p90:.2f} m", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(models)), [m[1] for m in models], fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 3.4)
    ax.set_xlabel("position error after 1 s [m]", fontsize=10, color=MUTED)
    ax.set_title(f"1 s prediction on the test set ({summary['prediction']['rssm']['all']['count']} predictions per model)",
                 fontsize=11, color=INK, loc="left")

    ax = fig.add_subplot(grid[0, 2:])
    base(ax)
    groups = list(summary["prediction_by_map"]["rssm"].keys()) + ["after a grip change"]
    width = 0.2
    for j, (key, label, color) in enumerate(models):
        values = [summary["prediction_by_map"][key][g]["median"][0] for g in groups[:-1]]
        values.append(summary["prediction"][key]["after_change"]["median"][0])
        ax.bar(np.arange(len(groups)) + (j - 1.5) * width, values, width=width * 0.9, color=color,
               label=label.replace("\n", " "))
    ax.set_xticks(range(len(groups)), groups, fontsize=9, color=INK)
    ax.set_ylabel("median position error after 1 s [m]", fontsize=10, color=MUTED)
    ax.set_title("by map, and right after a grip change", fontsize=11, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    names = (("steering", "steering [rad]"), ("speed", "speed [m/s]"), ("yaw_rate", "yaw rate [rad/s]"), ("slip", "slip [rad]"))
    estimators = (("ekf", "EKF", GRAY), ("v3_encoder", "v3 encoder", SERIES[0]), ("rssm", f"RSSM ({label})", SERIES[1]))
    for k, (key, label) in enumerate(names):
        ax = fig.add_subplot(grid[1, k])
        base(ax)
        values = [summary["estimation_rmse"][e][key] for e, _, _ in estimators]
        ax.bar(range(3), values, color=[c for _, _, c in estimators], width=0.6)
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_xticks(range(3), [n for _, n, _ in estimators], fontsize=8, color=INK)
        ax.set_ylim(0, max(values) * 1.25)
        ax.set_title(f"state estimate error (RMSE): {label}", fontsize=10, color=INK, loc="left")
    fig.suptitle("Offline evaluation on the on-track test split (lower is better)", fontsize=13, color=INK, x=0.01,
                 ha="left")
    fig.savefig(out / "rssm_offline_eval.png", dpi=120, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=["rssm-v1", "rssm-v2", "rssm-v3", "rssm-v4", "rssm-v5"])
    parser.add_argument("--eval", default="results/wm-eval")
    parser.add_argument("--out", default="docs/figures")
    parser.add_argument("--label", default="rssm-v3")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    training([r for r in args.runs if (Path("runs") / r / "history.json").exists()], out)
    evaluation(args.eval, out, args.label)
    print(f"saved {out}/rssm_training_curves.png and {out}/rssm_offline_eval.png")


if __name__ == "__main__":
    main()
