"""Footprint track-limit runs: crashes, lap times, maximum body offset and heading error, planning time.

    python scripts/footprint_report.py --prefix fp-none --controllers bicycle v11 v3 --seeds 1 2 3 4 5

Per run: first crash (type: spin if |slip| > 0.4 rad in the 2 s before contact, else wall; track position s; bridge
hit point), all crashes and learned attempts (runs with sit_lmpc_mode false keep going after a crash), learned lap
times, maximum |e_y| of the body corner furthest from the centre line and maximum |heading error| on learned laps,
and planning time p50 / p95 on learned steps.
"""

import argparse
import glob
import json
import re
from pathlib import Path
import numpy as np

CORNERS = np.array([[0.2835, 0.155], [0.2835, -0.155], [-0.2965, 0.155], [-0.2965, -0.155]])


def analyse(run):
    log = (run.parent / f"{run.name}.log").read_text().splitlines()
    lap_lines = [l for l in log if l.startswith("[lap]")]
    learned_laps = [float(l.split("|")[1].split()[0]) for l in lap_lines if "lmppi" in l]
    files = sorted(glob.glob(str(run / "trajectories" / "*" / "attempt-*.npz")), key=lambda p: Path(p).name)
    hits = re.findall(r"hit something at \(([-\d.]+), ([-\d.]+)\)", (run.parent / f"{run.name}.bridge.log").read_text())
    crashes, learned_attempts, body, heading = [], 0, [], []
    for path in files:
        a = np.load(path, allow_pickle=True)
        learned = len(a["controllers"]) and str(a["controllers"][0]) == "lmppi"
        fr = a["frenet"]
        contact = np.flatnonzero(a["contacts"]) if "contacts" in a.files else np.zeros(0, int)
        end = int(contact[0]) if len(contact) else len(fr)
        if learned:
            learned_attempts += 1
            psi = fr[:end, 4]
            ey = fr[:end, 1, None] + CORNERS[:, 0] * np.sin(psi)[:, None] + CORNERS[:, 1] * np.cos(psi)[:, None]
            if end:
                body.append(np.abs(ey).max(axis=1))
                heading.append(np.abs(psi))
        if "/failure/" in path:
            k = end if end < len(fr) else len(fr) - 1
            slip = float(np.abs(a["cartesian"][max(0, k - 20):k, 6]).max()) if k else 0.0
            crashes.append({"attempt": int(Path(path).stem.split("-")[-1]), "s": round(float(fr[k, 0]), 1),
                            "type": "spin" if slip > 0.4 else "wall", "slip": round(slip, 2)})
    for crash, hit in zip(crashes, hits):
        crash["hit"] = [float(hit[0]), float(hit[1])]
    timing = np.load(run / "control_timing.npz")["planning"] * 1e3 if (run / "control_timing.npz").exists() else np.zeros(0)
    first = 0
    for path in files:
        a = np.load(path, allow_pickle=True)
        if len(a["controllers"]) and str(a["controllers"][0]) == "lmppi":
            break
        first += len(a["controls"])
    learned_timing = timing[first + 1:] if len(timing) > first + 1 else timing
    return {"learned_laps": learned_laps, "crashes": crashes, "learned_attempts": learned_attempts,
            "max_body_abs_ey": round(float(np.concatenate(body).max()), 2) if body else None,
            "p99_body_abs_ey": round(float(np.percentile(np.concatenate(body), 99)), 2) if body else None,
            "max_abs_heading_error": round(float(np.concatenate(heading).max()), 2) if heading else None,
            "p99_abs_heading_error": round(float(np.percentile(np.concatenate(heading), 99)), 2) if heading else None,
            "planning_p50_ms": round(float(np.percentile(learned_timing, 50)), 1) if len(learned_timing) else None,
            "planning_p95_ms": round(float(np.percentile(learned_timing, 95)), 1) if len(learned_timing) else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefix", default="fp-none")
    parser.add_argument("--controllers", nargs="+", default=["bicycle", "v11", "v3"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--root", default="results/Spielberg")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    report, summary = {}, {}
    for controller in args.controllers:
        rows = []
        for seed in args.seeds:
            name = f"{args.prefix}-{controller}-s{seed}"
            if not (Path(args.root) / f"{name}.log").exists():
                continue
            entry = analyse(Path(args.root) / name)
            report[name] = entry
            rows.append(entry)
            first = entry["crashes"][0] if entry["crashes"] else None
            print(f"{name:26s} crash {first['type'] + ' s=' + str(first['s']) + ' at attempt ' + str(first['attempt']) if first else 'no':32s} "
                  f"crashes {len(entry['crashes'])}/{entry['learned_attempts']} learned attempts  max body |e_y| "
                  f"{entry['max_body_abs_ey']} (p99 {entry['p99_body_abs_ey']})  max |heading err| {entry['max_abs_heading_error']} (p99 {entry['p99_abs_heading_error']})  planning "
                  f"{entry['planning_p50_ms']}/{entry['planning_p95_ms']} ms  laps {[round(t, 1) for t in entry['learned_laps']]}")
        if rows:
            crashes = sum(len(r["crashes"]) for r in rows)
            attempts = sum(r["learned_attempts"] for r in rows)
            summary[controller] = {"runs": len(rows), "runs_with_a_crash": sum(bool(r["crashes"]) for r in rows),
                                   "crashes_per_learned_attempt": f"{crashes}/{attempts} ({100 * crashes / max(attempts, 1):.0f}%)",
                                   "crash_spots_s": sorted(c["s"] for r in rows for c in r["crashes"])}
    print(json.dumps(summary, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"runs": report, "summary": summary}, indent=1) + "\n")


if __name__ == "__main__":
    main()
