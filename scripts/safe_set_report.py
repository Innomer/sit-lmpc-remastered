"""Condition-aware safe-set runs: lap times, crashes, safe-set size per step, how often the fallback triggered, and
the condition tag over time.

    python scripts/safe_set_report.py --runs change-patch-v11-condition change-grip-drop-v11-condition

Reads <run>/safe_set_trace.npy (per learned step: step, laps in the window, laps kept, fallback, points, reference tag,
current tag) and <run>/condition_tag.npy (per step tag) written by the runner.
"""

import argparse
import glob
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--root", default="results/Spielberg")
    parser.add_argument("--out", default="results/condition-safe-set")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in args.runs:
        run = Path(args.root) / name
        log = (run.parent / f"{name}.log").read_text().splitlines()
        times = [float(l.split("|")[1].split()[0]) for l in log if l.startswith("[lap]")]
        crashes = [l for l in log if l.startswith("[attempt]") and "failure" in l]
        trace = np.load(run / "safe_set_trace.npy") if (run / "safe_set_trace.npy").exists() else np.zeros((0, 7))
        tags = np.load(run / "condition_tag.npy") if (run / "condition_tag.npy").exists() else np.zeros(0)
        files = sorted(glob.glob(str(run / "trajectories" / "*" / "attempt-*.npz")), key=lambda p: Path(p).name)
        starts, start = [], 0
        for f in files:
            starts.append(start)
            start += len(np.load(f, allow_pickle=True)["cartesian"]) - 1
        per_lap = []
        for k, (a, b) in enumerate(zip(starts, starts[1:] + [10 ** 9])):
            rows = trace[(trace[:, 0] >= a) & (trace[:, 0] < b)] if len(trace) else trace
            if len(rows):
                per_lap.append({"attempt": k + 1, "steps": int(len(rows)), "laps_in_window_mean": float(rows[:, 1].mean()),
                                "laps_kept_mean": float(rows[:, 2].mean()), "laps_kept_min": int(rows[:, 2].min()),
                                "points_mean": float(rows[:, 4].mean()), "fallback_share": float(rows[:, 3].mean())})
        crash_detail = None
        if crashes:
            failed = sorted(glob.glob(str(run / "trajectories" / "failure" / "*.npz")))[-1]
            a = np.load(failed, allow_pickle=True)
            contact = np.flatnonzero(a["contacts"])
            k = int(contact[0]) if len(contact) else len(a["frenet"]) - 1
            true = np.load(run / "encoder_true_states.npy")[-len(a["frenet"]):]
            crash_detail = {"lap": len(times) + 1, "seconds_into_lap": k * 0.1, "s": float(a["frenet"][k, 0]),
                            "max_abs_slip_before": float(np.abs(true[:k, 6]).max()),
                            "max_abs_yaw_rate_last_2s": float(np.abs(true[max(0, k - 20):k, 5]).max())}
        report[name] = {"lap_times": times, "crashes": len(crashes), "crash": crash_detail,
                        "fallback_share_all_learned_steps": float(trace[:, 3].mean()) if len(trace) else None,
                        "per_lap": per_lap}
        print(f"{name}: laps {[round(t, 1) for t in times]}, crashes {len(crashes)} {crash_detail or ''}")
        for row in per_lap:
            print(f"   attempt {row['attempt']:2d}: laps in window {row['laps_in_window_mean']:.1f}, kept {row['laps_kept_mean']:.1f} "
                  f"(min {row['laps_kept_min']}), points {row['points_mean']:.0f}, fallback {row['fallback_share']:.0%}")
        if len(tags):
            fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), facecolor=SURFACE, sharex=True)
            t = np.arange(len(tags)) * 0.1
            axes[0].plot(t, tags, color="#eb6834", linewidth=0.7, label="tag: bicycle 1.0 s position error")
            if len(trace):
                axes[0].plot(trace[:, 0] * 0.1, trace[:, 6], color=INK, linewidth=1.0, label="current tag (2 s median)")
                axes[1].plot(trace[:, 0] * 0.1, trace[:, 2], color="#2a78d6", linewidth=0.8, label="laps kept")
                axes[1].plot(trace[:, 0] * 0.1, trace[:, 1], color="#9ec5f4", linewidth=0.8, label="laps in the window")
                fb = trace[trace[:, 3] > 0]
                axes[1].scatter(fb[:, 0] * 0.1, fb[:, 2], s=4, color="#e34948", label="fallback (newest laps)")
            axes[0].set_ylabel("m", color=MUTED, fontsize=9)
            axes[1].set_ylabel("laps", color=MUTED, fontsize=9)
            axes[1].set_xlabel("time (s); grey lines: lap boundaries", color=MUTED, fontsize=9)
            for ax in axes:
                ax.set_facecolor(SURFACE)
                for s0 in starts[1:]:
                    ax.axvline(s0 * 0.1, color="#d9d8d4", linewidth=0.6)
                ax.tick_params(colors=MUTED, labelsize=8)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                ax.legend(frameon=False, fontsize=8, loc="upper left")
            axes[0].set_title(f"{name}: condition tag and safe-set selection", color=INK, fontsize=11, loc="left")
            fig.tight_layout()
            fig.savefig(out / f"{name}.png", dpi=110, facecolor=SURFACE)
            plt.close(fig)
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
