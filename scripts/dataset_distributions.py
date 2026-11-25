"""Distributions of the world-model training data: histograms and summary statistics per variable and data source.

    python scripts/dataset_distributions.py --out docs/figures --table results/data-distributions.json

On-track data (data/f1tenth-track) is cleaned as for training: episodes end before the first wall contact
(contacts.json) and before slip > 1.2 rad or yaw rate > 10 rad/s. Sources: collector episodes, the planner's own
closed-loop laps (cl-*), spin-onset rollouts (onset-*), and the map-free dataset (data/f1tenth-world, train split).
For each variable: range, percentiles, mean, standard deviation, skewness, excess kurtosis, and the Kolmogorov-Smirnov
distance to a normal fit and to a uniform distribution over the 0.5-99.5% range (smaller = closer).
"""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sit_lmpc.world.rssm_data import MapFreeData, contacts

SOURCES = ["collector", "closed-loop laps", "spin onsets", "map-free"]
COLORS = {"collector": "#2a78d6", "closed-loop laps": "#eb6834", "spin onsets": "#1baf7a", "map-free": "#eda100"}
STYLES = {"collector": "-", "closed-loop laps": "--", "spin onsets": "-.", "map-free": ":"}
VARIABLES = [
    ("speed", "Speed", "m/s"), ("steering", "Steering angle", "rad"), ("yaw_rate", "Yaw rate", "rad/s"),
    ("slip", "Slip angle", "rad"), ("heading", "Heading (absolute yaw)", "rad"),
    ("heading_error", "Heading relative to the track", "rad"), ("lateral", "Lateral offset e_y", "m"),
    ("steering_rate", "Steering-rate command", "rad/s"), ("acceleration", "Acceleration command", "m/s²"),
    ("grip", "Grip (friction under the car)", "-"), ("mass", "Mass ratio", "-"),
]
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def on_track(root, max_slip=1.2, max_yaw_rate=10.0):
    manifest = json.loads((Path(root) / "manifest.json").read_text())
    marked = contacts(root)
    out = {s: {k: [] for k, _, _ in VARIABLES} for s in SOURCES[:3]}
    meta = {s: {"episodes": 0, "maps": {}} for s in SOURCES[:3]}
    for record in manifest["episodes"]:
        source = "closed-loop laps" if "/cl-" in record["file"] else "spin onsets" if "/onset-" in record["file"] \
            else "collector"
        data = np.load(Path(root) / record["file"])
        s = data["states"].astype(np.float64)
        bad = np.flatnonzero((np.abs(s[:, 6]) > max_slip) | (np.abs(s[:, 5]) > max_yaw_rate))
        count = int(bad[0]) if len(bad) else len(s)
        contact = marked.get(record["file"], {}).get("contact")
        if contact is not None:
            count = min(count, contact)
        if count < 2:
            continue
        s, f, c = s[:count], data["frenet"][:count], data["conditions"][:count]
        u = data["controls"][:count - 1]
        d = out[source]
        d["speed"].append(s[:, 3]); d["steering"].append(s[:, 2]); d["yaw_rate"].append(s[:, 5])
        d["slip"].append(s[:, 6]); d["heading"].append((s[:, 4] + np.pi) % (2 * np.pi) - np.pi)
        heading_error = f[:, 4] if f.shape[1] == 7 else f[:, 2]
        d["heading_error"].append((heading_error + np.pi) % (2 * np.pi) - np.pi); d["lateral"].append(f[:, 1])
        d["steering_rate"].append(u[:, 0]); d["acceleration"].append(u[:, 1])
        d["grip"].append(c[:, 0]); d["mass"].append(c[:, 3])
        meta[source]["episodes"] += 1
        meta[source]["maps"][record["map"]] = meta[source]["maps"].get(record["map"], 0) + count - 1
    return {s: {k: np.concatenate(v) for k, v in d.items() if v} for s, d in out.items()}, meta


def map_free(root, max_slip=1.2, max_yaw_rate=10.0):
    data = MapFreeData(root, "train", 64, max_slip=max_slip, max_yaw_rate=max_yaw_rate)
    d = {k: [] for k, _, _ in VARIABLES}
    for e in data.episodes:
        s = e["states"].astype(np.float64)
        d["speed"].append(s[:, 3]); d["steering"].append(s[:, 2]); d["yaw_rate"].append(s[:, 5])
        d["slip"].append(s[:, 6]); d["heading"].append((s[:, 4] + np.pi) % (2 * np.pi) - np.pi)
        d["steering_rate"].append(e["controls"][:, 0]); d["acceleration"].append(e["controls"][:, 1])
        d["grip"].append(e["conditions"][:, 0]); d["mass"].append(e["conditions"][:, 1])
    return {k: np.concatenate(v) for k, v in d.items() if v}, {"episodes": len(data.episodes)}


def describe(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    lo, hi = np.percentile(x, [0.5, 99.5])
    sample = np.random.default_rng(0).choice(x, min(len(x), 200000), replace=False)
    mu, sd = sample.mean(), sample.std()
    ks_normal = stats.kstest((sample - mu) / sd, "norm").statistic if sd > 0 else 1.0
    ks_uniform = stats.kstest((np.clip(sample, lo, hi) - lo) / (hi - lo), "uniform").statistic if hi > lo else 1.0
    skew, kurt = float(stats.skew(sample)), float(stats.kurtosis(sample))
    if min(ks_normal, ks_uniform) > 0.1:
        shape = "neither normal nor uniform"
    else:
        shape = "close to normal" if ks_normal < ks_uniform else "close to uniform"
    notes = []
    if abs(skew) > 1:
        notes.append("right-skewed" if skew > 0 else "left-skewed")
    if kurt > 3:
        notes.append("peaked, heavy tails")
    elif kurt < -1:
        notes.append("flat")
    return {"n": int(len(x)), "min": float(x.min()), "p1": float(np.percentile(x, 1)), "p5": float(np.percentile(x, 5)),
            "median": float(np.median(x)), "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)),
            "max": float(x.max()), "mean": float(x.mean()), "std": float(x.std()), "skew": skew, "excess_kurtosis": kurt,
            "ks_normal": float(ks_normal), "ks_uniform": float(ks_uniform), "shape": shape, "notes": notes}


def style(ax, title, unit):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlabel(unit, color=MUTED, fontsize=8)
    ax.tick_params(colors=MUTED, labelsize=7, length=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d9d8d4")
    ax.grid(axis="y", color="#ecebe8", linewidth=0.6)


def figure(datasets, path, log):
    fig, axes = plt.subplots(4, 3, figsize=(13, 13), facecolor=SURFACE)
    for ax, (key, title, unit) in zip(axes.flat, VARIABLES):
        values = [datasets[s][key] for s in SOURCES if key in datasets[s]]
        allv = np.concatenate(values)
        lo, hi = np.percentile(allv, [0.1, 99.9])
        if key == "heading":
            lo, hi = -np.pi, np.pi
        bins = np.linspace(lo, hi, 80)
        peaks = []
        for s in SOURCES:
            if key not in datasets[s]:
                continue
            v = datasets[s][key]
            density, edges = np.histogram(v[(v >= lo) & (v <= hi)], bins=bins, density=True)
            ax.step(edges[:-1], density, where="post", color=COLORS[s], linestyle=STYLES[s], linewidth=1.6, label=s)
            if np.std(v) > 0.06 * (hi - lo):
                peaks.append(density.max())
        style(ax, title, unit)
        if key in ("grip", "mass") and peaks and not log:
            ax.set_ylim(0, 1.3 * max(peaks))
            ax.text(0.98, 0.95, "fixed-value spikes cut off", transform=ax.transAxes, ha="right", va="top",
                    color=MUTED, fontsize=7)
        ax.set_ylabel("density (log)" if log else "density", color=MUTED, fontsize=8)
        if log:
            ax.set_yscale("log")
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False, fontsize=10, title="data source",
                         title_fontsize=10, labelcolor=INK)
    fig.suptitle("World-model training data: distribution per variable" + (" (log scale, shows the tails)" if log else ""),
                 color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def joint(datasets, path):
    """Speed vs slip and speed vs yaw rate, all on-track data together: where the data actually is."""
    s = {k: np.concatenate([datasets[x][k] for x in SOURCES[:3]]) for k in ("speed", "slip", "yaw_rate")}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    for ax, key, unit in ((axes[0], "slip", "slip angle (rad)"), (axes[1], "yaw_rate", "yaw rate (rad/s)")):
        lo, hi = np.percentile(s[key], [0.1, 99.9])
        h, xe, ye = np.histogram2d(s["speed"], s[key], bins=(70, 70),
                                   range=((0, np.percentile(s["speed"], 99.9)), (lo, hi)))
        mesh = ax.pcolormesh(xe, ye, np.log10(h.T + 1), cmap="Blues", shading="auto")
        style(ax, f"Speed vs {key.replace('_', ' ')}, on-track data", "speed (m/s)")
        ax.set_ylabel(unit, color=MUTED, fontsize=8)
        cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
        cbar.set_label("log10(steps + 1)", color=MUTED, fontsize=8)
        cbar.ax.tick_params(colors=MUTED, labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track", default="data/f1tenth-track")
    parser.add_argument("--world", default="data/f1tenth-world")
    parser.add_argument("--out", default="docs/figures")
    parser.add_argument("--table", default="results/data-distributions.json")
    args = parser.parse_args()
    datasets, meta = on_track(args.track)
    datasets["map-free"], meta["map-free"] = map_free(args.world)
    datasets["all on-track"] = {k: np.concatenate([datasets[s][k] for s in SOURCES[:3]]) for k in datasets["collector"]}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    figure(datasets, out / "data_distributions.png", log=False)
    figure(datasets, out / "data_distributions_log.png", log=True)
    joint(datasets, out / "data_speed_vs_slip.png")
    table = {"meta": meta, "stats": {s: {k: describe(v) for k, v in d.items()} for s, d in datasets.items()}}
    Path(args.table).parent.mkdir(parents=True, exist_ok=True)
    Path(args.table).write_text(json.dumps(table, indent=1) + "\n")
    for source in ["all on-track"] + SOURCES:
        print(f"\n{source}: " + (json.dumps(meta.get(source, {})) if source in meta else ""))
        for key, title, unit in VARIABLES:
            if key not in table["stats"][source]:
                continue
            r = table["stats"][source][key]
            print(f"  {title:32s} n {r['n']:7d}  range {r['min']:+7.2f} .. {r['max']:+7.2f}  p1-p99 {r['p1']:+6.2f} .. "
                  f"{r['p99']:+6.2f}  mean {r['mean']:+6.2f} sd {r['std']:5.2f}  skew {r['skew']:+5.2f} kurt "
                  f"{r['excess_kurtosis']:+6.2f}  KS normal {r['ks_normal']:.2f} uniform {r['ks_uniform']:.2f}  "
                  f"{r['shape']}{(' (' + ', '.join(r['notes']) + ')') if r['notes'] else ''}")


if __name__ == "__main__":
    main()
