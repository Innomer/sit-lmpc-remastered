"""Change-scenario runs with the footprint track limits: per seed and a summary.

    python scripts/fp_change_report.py --out results/footprint/change.json

Run names fp-<scenario>-<controller>-s<seed> (controllers bicycle, v11, v11-react). Per run (via
scripts/change_v2_report.analyse): crash and type, lap times after the change, laps to recover, detection delay and
detections before the change. For runs with change detection: mean speed over the 2 s before and after the first
detection, and over the same track stretch on the previous lap (to separate the reaction from the track's own speed
profile).
"""

import argparse
import glob
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from change_v2_report import analyse

CONTROLLERS = {"bicycle": "bicycle + footprint", "v11": "v11 + footprint, no reaction",
               "v11-react": "v11 + footprint + full reaction"}
SCENARIOS = {"grip-drop": "grip drop (μ 0.7)", "patch": "patch (μ 0.6)"}


def speeds_around_detection(run):
    events_file = run / "change_events.json"
    if not events_file.exists():
        return None
    events = json.loads(events_file.read_text())["events"]
    first = next((e for e in events if e["event"].startswith("detected")), None)
    if first is None:
        return None
    true = np.load(run / "encoder_true_states.npy")
    files = sorted(glob.glob(str(run / "trajectories" / "*" / "attempt-*.npz")), key=lambda p: Path(p).name)
    frenet = np.concatenate([np.load(f, allow_pickle=True)["frenet"][(0 if i == 0 else 1):, 0] for i, f in enumerate(files)])
    d = first["step"]
    n = min(len(true), len(frenet))
    before, after = true[max(0, d - 20):d, 3], true[d:min(n, d + 20), 3]
    s_after = frenet[d:min(n, d + 20)] % frenet.max()
    lap_start = max(0, d - 400)
    reference = []
    for s in s_after:
        near = np.flatnonzero(np.abs((frenet[lap_start:d - 50] % frenet.max()) - s) < 1.0)
        if len(near):
            reference.append(true[lap_start + near[-1], 3])
    return {"speed_2s_before": round(float(before.mean()), 2), "speed_2s_after": round(float(after.mean()), 2),
            "same_stretch_previous_lap": round(float(np.mean(reference)), 2) if reference else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="results/Spielberg")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--change-lap", type=int, default=8)
    parser.add_argument("--out", default="results/footprint/change.json")
    args = parser.parse_args()
    report, summary = {}, []
    for scenario in SCENARIOS:
        for controller, label in CONTROLLERS.items():
            rows = []
            for seed in args.seeds:
                name = f"fp-{scenario}-{controller}-s{seed}"
                run = Path(args.root) / name
                if not (Path(args.root) / f"{name}.log").exists():
                    continue
                entry = analyse(run, args.change_lap)
                entry["speed_at_detection"] = speeds_around_detection(run)
                report[name] = entry
                rows.append(entry)
                crash = entry["crash"]
                print(f"{name:28s} after change {[round(t, 1) for t in entry['after_change']]}  "
                      f"crash {crash['type'] + ' lap ' + str(crash['lap']) + ' s=' + str(crash['s']) if crash else 'no'}  "
                      f"recover {entry['laps_to_recover']}"
                      + (f"  detection delay {entry.get('detection_delay_s')} s, before change {entry.get('detections_before_change')}, "
                         f"speed {entry['speed_at_detection']}" if "detections" in entry else ""))
            if rows:
                crashed = [r for r in rows if r["crash"]]
                summary.append({"scenario": SCENARIOS[scenario], "controller": label, "seeds": len(rows),
                                "crashes": len(crashed), "crash_types": sorted(r["crash"]["type"] for r in crashed),
                                "crashes_before_change": sum(r["crash"]["lap"] <= args.change_lap for r in crashed),
                                "laps_to_recover": [r["laps_to_recover"] for r in rows if not r["crash"]],
                                "mean_lap_after_change_survivors": round(float(np.mean([np.mean(r["after_change"]) for r in rows
                                                                                        if not r["crash"] and r["after_change"]])), 2)
                                if any(not r["crash"] and r["after_change"] for r in rows) else None,
                                **({"detection_delay_s": [r.get("detection_delay_s") for r in rows],
                                    "false_detections": [r.get("detections_before_change") for r in rows]}
                                   if any("detections" in r for r in rows) else {})})
    print("\nsummary")
    for row in summary:
        print(json.dumps(row, ensure_ascii=False))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"runs": report, "summary": summary}, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
