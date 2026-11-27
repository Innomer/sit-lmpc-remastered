"""Retry-mode runs: per attempt and a summary.

    python scripts/rt_report.py rt-grip055-v11 rt-patch045-v11 ... --change-lap 8 --out results/retry/report.json

Per run: every attempt (outcome, lap time, mean speed, minimum body clearance to the wall, crash type and place),
crashes before and after the change, crashes by attempt number after the change, mean speed of the first attempt
after the change vs the last three completed laps before it, laps until lap times settle (first completed lap within
1% of the median of the last three), laps under 0.2 m clearance after the change, how often the LiDAR clearance check
changed the chosen plan, how often the shifted previous plan was chosen, and planning time on learned steps.
Crash type: spin if |heading error| > 0.6 rad in the 2 s before contact (heading comes from the true pose; the
logged slip of the v11 runs is the model's estimate), else wall.
The change applies once `change-lap` laps have been completed (the scheduler counts completed laps across resets).
"""

import argparse
import glob
import json
import re
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harsh_change_report import wall_distance_field, outline, clearance


def nearest_hit(hits, xy):
    if not hits:
        return None
    points = np.asarray(hits, float)
    j = int(np.argmin(np.hypot(points[:, 0] - xy[0], points[:, 1] - xy[1])))
    return [float(v) for v in points[j]] if np.hypot(*(points[j] - xy)) < 5.0 else None


def attempts(run):
    rows = []
    for path in glob.glob(str(run / "trajectories" / "*" / "attempt-*.npz")):
        a = np.load(path, allow_pickle=True)
        meta = json.loads(str(a["metadata"]))
        rows.append((meta["attempt_id"], meta["outcome"], meta["termination_reason"], a))
    return sorted(rows, key=lambda r: r[0])


def analyse(run, change_lap, field, resolution, origin, height, body):
    hits = re.findall(r"hit something at \(([-\d.]+), ([-\d.]+)\)", (run.parent / f"{run.name}.bridge.log").read_text())
    out, completed, hit_index = [], 0, 0
    change_attempt = None
    for attempt_id, outcome, reason, a in attempts(run):
        c, fr = a["cartesian"], a["frenet"]
        contact = np.flatnonzero(a["contacts"]) if "contacts" in a.files else np.zeros(0, int)
        end = int(contact[0]) if len(contact) else len(c)
        row = {"attempt": int(attempt_id), "outcome": outcome, "after_change": change_attempt is not None,
               "mean_speed": round(float(c[: max(end, 1), 3].mean()), 2),
               "min_clearance_m": round(float(clearance(c[: max(end, 1)], field, resolution, origin, height, body).min()), 3),
               "time_s": round(float(a["times"][-1] - a["times"][0]), 2)}
        if outcome == "failure":
            k = min(end, len(c) - 1)
            heading = float(np.abs(fr[max(0, k - 20):k + 1, 4]).max())
            row["crash"] = {"type": "spin" if heading > 0.6 else "wall", "max_heading_error_last_2s": round(heading, 2),
                            "reason": reason,
                            "s": round(float(fr[k, 0]) % 318.0, 1),
                            "hit": nearest_hit(hits, c[k, :2])}
        else:
            completed += 1
        out.append(row)
        if change_attempt is None and completed >= change_lap:
            change_attempt = attempt_id
    return out, change_attempt


def learned_planning(run):
    timing = run / "control_timing.npz"
    if not timing.exists():
        return None
    plan = np.load(timing)["planning"] * 1000
    log = (run.parent / f"{run.name}.log").read_text().splitlines()
    steps = [int(l.split("|")[1].split()[0]) for l in log if l.startswith("[attempt]")]
    start = sum(steps[:5])
    p = plan[start:]
    return {"p50": round(float(np.percentile(p, 50)), 1), "p95": round(float(np.percentile(p, 95)), 1)} if len(p) else None


def summarize(name, rows, change_attempt, run):
    before = [r for r in rows if not r["after_change"]]
    after = [r for r in rows if r["after_change"]]
    done_before = [r for r in before if r["outcome"] == "success"]
    done_after = [r for r in after if r["outcome"] == "success"]
    entry = {"attempts": len(rows), "crashes_before": sum(r["outcome"] == "failure" for r in before),
             "attempts_before": len(before), "crashes_after": sum(r["outcome"] == "failure" for r in after),
             "attempts_after": len(after),
             "crash_attempt_numbers_after_change": [i + 1 for i, r in enumerate(after) if r["outcome"] == "failure"],
             "crashes": [dict(r["crash"], attempt=r["attempt"]) for r in rows if "crash" in r],
             "lap_times": [(r["attempt"], r["time_s"]) for r in rows if r["outcome"] == "success"],
             "laps_under_0.2m_after": f"{sum(r['min_clearance_m'] < 0.2 for r in after)}/{len(after)}",
             "laps_under_0.2m_before": f"{sum(r['min_clearance_m'] < 0.2 for r in before[5:])}/{len(before[5:])}"}
    if done_before and after:
        entry["speed_before"] = round(float(np.mean([r["mean_speed"] for r in done_before[-3:]])), 2)
        entry["speed_first_after"] = after[0]["mean_speed"]
        entry["first_after_outcome"] = after[0]["outcome"]
    times = [r["time_s"] for r in done_after]
    if len(times) >= 4:
        target = np.median(times[-3:]) * 1.01
        entry["laps_to_settle"] = next(i + 1 for i, t in enumerate(times) if t <= target)
        entry["settled_lap_s"] = round(float(np.median(times[-3:])), 2)
    if done_before:
        entry["pre_change_lap_s"] = round(float(np.median([r["time_s"] for r in done_before[-3:]])), 2)
    counts = run / "planner_counts.json"
    if counts.exists():
        info = json.loads(counts.read_text())
        check = info.get("clearance_check", {})
        if check.get("steps"):
            entry["clearance_check_changed_pick"] = f"{check['changed']}/{check['steps']} ({100 * check['changed'] / check['steps']:.1f}%)"
        cand = info.get("candidates", {})
        entry["previous_plan_chosen"] = cand.get("previous_chosen")
        if cand.get("steps"):
            entry["brake_plans"] = {"steps": cand["steps"], "chosen": cand["chosen"]}
    entry["planning_ms"] = learned_planning(run)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--root", default="results/Spielberg")
    parser.add_argument("--change-lap", type=int, default=8)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    field, resolution, origin, height = wall_distance_field()
    body = outline()
    report = {}
    for name in args.runs:
        run = Path(args.root) / name
        change_lap = args.change_lap if "-none-" not in name else 10 ** 6
        rows, change_attempt = analyse(run, change_lap, field, resolution, origin, height, body)
        entry = summarize(name, rows, change_attempt, run)
        entry["per_attempt"] = rows
        report[name] = entry
        print(f"== {name}: attempts {entry['attempts']}, crashes before {entry['crashes_before']}/{entry['attempts_before']}, "
              f"after {entry['crashes_after']}/{entry['attempts_after']} at attempt no. {entry['crash_attempt_numbers_after_change']}")
        for c in entry["crashes"]:
            print(f"   crash attempt {c['attempt']}: {c['type']} at s={c['s']} hit {c['hit']}")
        print(f"   lap times {[(a, round(t, 1)) for a, t in entry['lap_times']]}")
        print("   " + ", ".join(f"{k} {entry.get(k)}" for k in ("laps_under_0.2m_before", "laps_under_0.2m_after", "speed_before",
                                                             "speed_first_after", "first_after_outcome", "pre_change_lap_s",
                                                             "laps_to_settle", "settled_lap_s", "clearance_check_changed_pick",
                                                             "previous_plan_chosen", "brake_plans", "planning_ms")))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
