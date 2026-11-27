"""Root-cause runs: every crash per run (lap, track position, type, bridge hit point) and the lap times.

    python scripts/rootcause_report.py --runs rootcause/old-v3-s1-r ... change-none-bicycle-s1 ...

Runs with sit_lmpc_mode false keep going after a crash (each crash is one failed attempt); runs with it true stop at
the first crash. Crash type: spin if |slip| > 0.4 rad in the 2 s before contact, else wall.
"""

import argparse
import glob
import json
import re
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--root", default="results/Spielberg")
    parser.add_argument("--out", default="results/rootcause/report.json")
    args = parser.parse_args()
    report = {}
    for name in args.runs:
        run = Path(args.root) / name
        log = (run.parent / f"{run.name}.log").read_text().splitlines()
        laps = [float(l.split("|")[1].split()[0]) for l in log if l.startswith("[lap]")]
        attempts = [l for l in log if l.startswith("[attempt]")]
        hits = re.findall(r"hit something at \(([-\d.]+), ([-\d.]+)\)",
                          (run.parent / f"{run.name}.bridge.log").read_text())
        crashes = []
        for path in sorted(glob.glob(str(run / "trajectories" / "failure" / "*.npz"))):
            a = np.load(path, allow_pickle=True)
            contact = np.flatnonzero(a["contacts"]) if "contacts" in a.files else []
            k = int(contact[0]) if len(contact) else len(a["frenet"]) - 1
            slip = float(np.abs(a["cartesian"][max(0, k - 20):k, 6]).max()) if k else 0.0
            attempt = int(Path(path).stem.split("-")[-1])
            crashes.append({"attempt": attempt, "seconds_into_lap": round(k * 0.1, 1), "s": round(float(a["frenet"][k, 0]), 1),
                            "type": "spin" if slip > 0.4 else "wall", "max_slip_last_2s": round(slip, 2)})
        for crash, hit in zip(crashes, hits):
            crash["hit"] = [float(hit[0]), float(hit[1])]
        finish = next((l for l in log if l.startswith("[finish]")), "")
        report[name] = {"lap_times": laps, "attempts": len(attempts), "crashes": crashes, "finish": finish[:90]}
        learned = laps[5:]
        print(f"{name:34s} attempts {len(attempts):2d}, crashes {len(crashes)}: "
              + ("; ".join(f"attempt {c['attempt']} {c['type']} s={c['s']} hit {c.get('hit')}" for c in crashes) or "none")
              + f" | learned laps {[round(t, 1) for t in learned]}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
