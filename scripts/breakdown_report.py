"""Why the car drives under the speed cap: final-pick cost breakdown of the chosen plan vs the fastest feasible one.

    python scripts/breakdown_report.py rx-check-grip-drop-s1 rx-check-patch-s1

Reads candidate_breakdown.npz (planner.change_breakdown). On learned steps inside a capped stretch: the chosen
candidate vs the fastest candidate that stays inside the track limits (body footprint), by mean planned speed.
The final pick takes the lowest score = 1000 x boundary + hull violation + 1000 x safety (safety = sum of the
ranked safety terms), ties going to the first lambda. For each term: mean for both plans and how often the term
alone makes the fastest plan score worse than the chosen one.
"""

import argparse
import json
from pathlib import Path
import numpy as np


def analyse(run):
    d = dict(np.load(run / "candidate_breakdown.npz"))
    index = d["index"].astype(int)
    rows = np.arange(len(index))
    capped = np.isfinite(d["cap_here"]) & (d["cap_here"] < 100)
    feasible = (d["boundary"] == 0) & np.isfinite(d["score"])
    speed = np.where(feasible, d["mean_speed"], -np.inf)
    fastest = np.argmax(speed, axis=1)
    has = feasible.any(axis=1)
    sel = capped & has
    terms = sorted(k for k in d if k.startswith("term_"))
    out = {"learned_steps": int(len(index)), "capped_steps": int(capped.sum()),
           "chosen_is_fastest_feasible": round(float(np.mean(index[sel] == fastest[sel])), 3) if sel.any() else None,
           "actual_speed": round(float(np.nanmean(d["speed"][sel])), 2) if sel.any() else None,
           "cap_here": round(float(np.nanmean(d["cap_here"][sel])), 2) if sel.any() else None,
           "chosen_mean_planned_speed": round(float(np.nanmean(d["mean_speed"][rows, index][sel])), 2) if sel.any() else None,
           "fastest_mean_planned_speed": round(float(np.nanmean(d["mean_speed"][rows, fastest][sel])), 2) if sel.any() else None,
           "fastest_candidate_any_speed": round(float(np.nanmean(d["mean_speed"].max(axis=1)[sel])), 2) if sel.any() else None,
           "terms": {}}
    if "max_over_cap" in d and sel.any():
        out["chosen_max_over_cap"] = round(float(np.nanmean(d["max_over_cap"][rows, index][sel])), 2)
        out["fastest_max_over_cap"] = round(float(np.nanmean(d["max_over_cap"][rows, fastest][sel])), 2)
    differ = sel & (index != fastest)
    for name, weight in [("hull", 1.0), ("boundary", 1000.0), ("value", 0.0)] + [(t, 1000.0) for t in terms]:
        if name not in d or not sel.any():
            continue
        values = np.nan_to_num(d[name])
        chosen, fast = values[rows, index], values[rows, fastest]
        out["terms"][name] = {"chosen_mean": round(float(chosen[sel].mean()), 4), "fastest_mean": round(float(fast[sel].mean()), 4),
                              "fastest_worse_share": round(float(np.mean(weight * (fast[differ] - chosen[differ]) > 1e-9)), 3)
                              if differ.any() and weight else None}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--root", default="results/Spielberg")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    report = {name: analyse(Path(args.root) / name) for name in args.runs}
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
