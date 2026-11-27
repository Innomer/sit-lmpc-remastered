"""Why v11 does not slow down after a change: the chosen plan vs a slower plan before each crash.

    python scripts/dg_report.py rt-grip055-v11 rt-patch045-v11 --out results/retry/diagnosis.json

Reads candidate_breakdown.npz (planner.change_breakdown, one row per learned step). Rows are split into attempts by
the jumps in s: a lap end wraps from the end of the track to its start, a crash reset jumps back to the start from
mid-track. For the last `window` learned steps before each crash after the change, per step: the chosen candidate
and the slowest candidate that stays inside the track limits (by mean planned speed), with mean planned speed,
predicted max |slip|, the peak LiDAR clearance cost (predicted nearest range = 0.35 m x (1 - cost / weight) when
under 0.35 m), terminal value, safe-set distance, track-limit sum and final-pick score. The final pick ranks by
1e7 x (track limit + safe-set violation) + 1000 x safety terms; the value only acts through the sampling weights.
"""

import argparse
import json
from pathlib import Path
import numpy as np

TRACK = 343.32


def split(d):
    s = d["s"]
    starts, kinds = [0], []
    for i in range(1, len(s)):
        if s[i] < s[i - 1] - 10:
            kinds.append("lap" if (s[i - 1] % TRACK) > TRACK - 20 else "crash")
            starts.append(i)
    kinds.append("open")
    return starts, kinds


def describe(d, r, c, weight):
    clearance_cost = float(d["max_clearance_cost"][r, c]) if "max_clearance_cost" in d else 0.0
    nearest = 0.35 * (1 - clearance_cost / weight) if clearance_cost > 0 else None
    return {"candidate": int(c), "mean_speed": round(float(d["mean_speed"][r, c]), 2),
            "max_speed": round(float(d["max_speed"][r, c]), 2),
            "max_abs_slip": round(float(d["max_abs_slip"][r, c]), 3) if "max_abs_slip" in d else None,
            "predicted_nearest_m": None if nearest is None else round(nearest, 2),
            "value": round(float(d["value"][r, c]), 2), "safe_set_distance": round(float(d["hull"][r, c]), 3),
            "boundary": round(float(d["boundary"][r, c]), 3), "score": round(float(d["score"][r, c]), 1)}


def analyse(run, change_lap, window, weight):
    d = dict(np.load(run / "candidate_breakdown.npz"))
    index = d["index"].astype(int)
    starts, kinds = split(d)
    completed, out = 0, []
    for a, kind in enumerate(kinds):
        lo, hi = starts[a], starts[a + 1] if a + 1 < len(starts) else len(index)
        after = completed >= change_lap
        if kind == "lap":
            completed += 1
        if kind != "crash" or not after:
            continue
        rows = []
        for r in range(max(lo, hi - window), hi):
            feasible = np.flatnonzero((d["boundary"][r] == 0) & np.isfinite(d["score"][r]))
            slow = feasible[np.argmin(d["mean_speed"][r, feasible])] if len(feasible) else None
            rows.append({"s": round(float(d["s"][r] % TRACK), 1), "speed": round(float(d["speed"][r]), 2),
                         "in_bounds_candidates": int(len(feasible)),
                         "chosen": describe(d, r, index[r], weight),
                         "slower": None if slow is None else describe(d, r, slow, weight)})
        out.append({"crash_after_laps": completed, "steps": rows})
    return out


def verdict(rows):
    pairs = [r for r in rows if r["slower"] is not None and r["slower"]["candidate"] != r["chosen"]["candidate"]]
    if not pairs:
        return {}
    gap = [r["chosen"]["mean_speed"] - r["slower"]["mean_speed"] for r in pairs]
    worse_hull = [r["slower"]["safe_set_distance"] > r["chosen"]["safe_set_distance"] + 1e-6 for r in pairs]
    lower_value = [r["slower"]["value"] > r["chosen"]["value"] + 1e-6 for r in pairs]
    slip_gap = [r["chosen"]["max_abs_slip"] - r["slower"]["max_abs_slip"] for r in pairs if r["chosen"]["max_abs_slip"] is not None]
    flagged = [r["chosen"]["predicted_nearest_m"] is not None for r in pairs]
    return {"steps": len(pairs), "mean_speed_gap": round(float(np.mean(gap)), 2),
            "slower_plan_has_larger_safe_set_distance": f"{sum(worse_hull)}/{len(pairs)}",
            "slower_plan_has_worse_value": f"{sum(lower_value)}/{len(pairs)}",
            "chosen_minus_slower_predicted_slip": round(float(np.mean(slip_gap)), 3) if slip_gap else None,
            "chosen_plan_predicted_within_0.35m": f"{sum(flagged)}/{len(pairs)}",
            "chosen_max_predicted_slip": round(float(max(r["chosen"]["max_abs_slip"] for r in pairs)), 3) if slip_gap else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--root", default="results/Spielberg")
    parser.add_argument("--change-lap", type=int, default=8)
    parser.add_argument("--warmup-laps", type=int, default=5, help="laps driven before learned control (not in the log)")
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--clearance-weight", type=float, default=1.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    report = {}
    for name in args.runs:
        crashes = analyse(Path(args.root) / name, args.change_lap - args.warmup_laps, args.window, args.clearance_weight)
        for crash in crashes:
            crash["verdict"] = verdict(crash["steps"])
        report[name] = crashes
        for crash in crashes:
            print(f"== {name}: crash after {crash['crash_after_laps']} completed learned laps: {crash['verdict']}")
            for r in crash["steps"][::5]:
                c, w = r["chosen"], r["slower"] or {}
                print(f"   s={r['s']:6.1f} v={r['speed']:4.1f} in-bounds {r['in_bounds_candidates']:3d} | chosen v {c['mean_speed']:4.1f} "
                      f"slip {c['max_abs_slip']} near {c['predicted_nearest_m']} value {c['value']} ss {c['safe_set_distance']} "
                      f"| slower v {w.get('mean_speed')} slip {w.get('max_abs_slip')} near {w.get('predicted_nearest_m')} "
                      f"value {w.get('value')} ss {w.get('safe_set_distance')}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
