"""Condition-change runs: lap times per lap, laps to recover after the change, crashes, and the RSSM's grip/mass
estimate over time against the truth.

    python scripts/change_report.py --runs change-grip-drop-v11 change-grip-drop-bicycle ... --rssm runs/rssm-v11

Change lap and true conditions come from each run's scenario (configs/scenarios/spielberg-*-lap8.yaml copy in the run's
scheduler log). Recovery: the first lap after the change whose time is within 1% of the median of the run's last three
laps (counted from the first full lap after the change); "not recovered" if no such lap.
"""

import argparse
import glob
import json
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sit_lmpc.world.rssm_risk import LatentFilter
from sit_lmpc.world.rssm import RSSM
from sit_lmpc.world.rssm_train import CONDITION_MEAN, CONDITION_SCALE

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def laps(run):
    log = (Path(run).parent / f"{Path(run).name}.log").read_text().splitlines()
    out = []
    for line in log:
        if line.startswith("[lap]"):
            out.append(("lap", float(line.split("|")[1].strip().split()[0]), "lmppi" in line))
        elif line.startswith("[attempt]") and "failure" in line:
            out.append(("crash", None, True))
    return out


def scenario_events(run):
    path = Path(run).parent / f"{Path(run).name}.conditions.jsonl"
    return [json.loads(l)["event"] for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def recovery(times, change_lap):
    after = [t for i, t in enumerate(times) if i >= change_lap and t is not None]
    if len(after) < 4:
        return None
    target = np.median(after[-3:]) * 1.01
    for k, t in enumerate(after):
        if t <= target:
            return k + 1
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--root", default="results/Spielberg")
    parser.add_argument("--rssm", default="runs/rssm-v11")
    parser.add_argument("--change-lap", type=int, default=8)
    parser.add_argument("--out", default="results/change-scenarios")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    filt = LatentFilter(args.rssm)
    report = {}
    for name in args.runs:
        run = Path(args.root) / name
        entries = laps(run)
        times = [t for kind, t, _ in entries if kind == "lap"]
        crashes = sum(kind == "crash" for kind, _, _ in entries)
        entry = {"lap_times": times, "crashes": crashes, "laps_completed": len(times),
                 "laps_to_recover": recovery(times, args.change_lap), "events": scenario_events(run)}
        latent_file = run / "encoder_z.npy"
        if latent_file.exists() and (run / "rssm_memory.npy").exists():
            s = jnp.asarray(np.load(latent_file), jnp.float32)
            readout = np.asarray(filt.model.apply(filt.network, s, method=RSSM.condition_readout))
            estimate = readout * np.asarray(CONDITION_SCALE) + np.asarray(CONDITION_MEAN)
            files = sorted(glob.glob(str(run / "trajectories" / "*" / "attempt-*.npz")), key=lambda p: Path(p).name)
            starts, start, frenet = [], 0, []
            for f in files:
                a = np.load(f, allow_pickle=True)
                starts.append(start)
                frenet.append(a["frenet"][:, 0] if start == 0 else a["frenet"][1:, 0])
                start += len(a["frenet"]) - 1
            s_track = np.concatenate(frenet)[:len(estimate)]
            entry["grip_estimate"] = {"per_lap_mean": [float(estimate[a:b, 0].mean()) for a, b in
                                                       zip(starts, starts[1:] + [len(estimate)])],
                                      "mass_per_lap_mean": [float(estimate[a:b, 1].mean()) for a, b in
                                                            zip(starts, starts[1:] + [len(estimate)])]}
            fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), facecolor=SURFACE, sharex=True)
            t = np.arange(len(estimate)) * 0.1
            for ax, k, label in ((axes[0], 0, "grip (friction)"), (axes[1], 1, "mass ratio")):
                ax.set_facecolor(SURFACE)
                ax.plot(t, estimate[:, k], color="#2a78d6", linewidth=1.0, label="v11 estimate (condition head on c)")
                for b in starts[1:]:
                    ax.axvline(b * 0.1, color="#d9d8d4", linewidth=0.6)
                ax.axvline(starts[args.change_lap] * 0.1 if len(starts) > args.change_lap else t[-1], color="#e34948",
                           linewidth=1.2, linestyle="--", label="change")
                ax.set_ylabel(label, color=MUTED, fontsize=9)
                ax.tick_params(colors=MUTED, labelsize=8)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
            truth_mu = np.full(len(t), 1.0489)
            truth_mass = np.ones(len(t))
            change_step = starts[args.change_lap] if len(starts) > args.change_lap else len(t)
            for event in entry["events"]:
                condition = event.get("condition") or {}
                if "mu" in condition:
                    truth_mu[change_step:] = condition["mu"]
                if "mass_ratio" in condition:
                    truth_mass[change_step:] = condition["mass_ratio"]
                for patch in event.get("patches") or []:
                    inside = (s_track >= patch["s_start"]) & (s_track <= patch["s_end"])
                    truth_mu[change_step:][inside[change_step:]] = patch["mu"]
            axes[0].plot(t, truth_mu, color=INK, linewidth=1.2, label="true grip")
            axes[1].plot(t, truth_mass, color=INK, linewidth=1.2, label="true mass ratio")
            axes[0].legend(frameon=False, fontsize=8, loc="lower left")
            axes[1].set_xlabel("time (s); grey lines: lap boundaries", color=MUTED, fontsize=9)
            axes[0].set_title(f"{name}: condition estimate over time", color=INK, fontsize=11, loc="left")
            fig.tight_layout()
            fig.savefig(out / f"{name}-condition.png", dpi=110, facecolor=SURFACE)
            plt.close(fig)
        report[name] = entry
        print(name, "laps", len(times), "crashes", crashes, "recover after", entry["laps_to_recover"],
              "times", [round(x, 1) for x in times],
              ("grip per lap " + str([round(x, 2) for x in entry["grip_estimate"]["per_lap_mean"]])) if "grip_estimate" in entry else "")
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
