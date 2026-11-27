"""Print one run's results: lap times, failures, grip estimate per lap and planning time.

    python scripts/run_report.py results/Spielberg/change-friction-known
"""

import json
import sys
from pathlib import Path
import numpy as np


def main():
    run = Path(sys.argv[1])
    log = Path(str(run) + ".log").read_text().splitlines()
    print(f"== {run.name}")
    attempts = sorted((run / "trajectories").glob("*/attempt-*.npz"), key=lambda p: p.name)
    friction_path = run / "encoder_grip.npy" if (run / "encoder_grip.npy").exists() else run / "encoder_friction.npy"
    friction = np.load(friction_path) if friction_path.exists() else None
    position = 0
    for path in attempts:
        data = np.load(path, allow_pickle=True)
        steps = len(data["cartesian"]) - 1
        controller = str(data["controllers"][0]) if len(data["controllers"]) else "?"
        outcome = path.parent.name
        line = f"attempt {int(path.stem[-6:]):2d}  {outcome:10s} {controller:6s} {steps * 0.1:5.1f} s"
        if outcome != "success":
            frenet = data["frenet"]
            line += f"  ended at s {frenet[-1, 0]:.0f} m, max |slip| {np.abs(data['cartesian'][:, 6]).max():.2f}"
        if friction is not None:
            segment = friction[position:position + steps]
            line += f"  grip guess {np.nanmean(segment):.2f}"
        position += steps
        print(line)
    for row in log:
        if row.startswith("[finish]"):
            print(row[:110])
    timing = run / "control_timing.npz"
    if timing.exists():
        planning = np.load(timing)["planning"] * 1000
        learned = planning[planning > 20]
        if len(learned):
            print(f"planning (learned laps) p50 {np.median(learned):.0f} ms, p95 {np.percentile(learned, 95):.0f} ms")
    conditions = Path(str(run) + ".conditions.jsonl")
    if conditions.exists():
        for row in conditions.read_text().splitlines():
            print("condition:", json.loads(row)["event"])


if __name__ == "__main__":
    main()
