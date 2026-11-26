"""Report for the safe-set v2 / change-aware value runs: per seed and a summary across seeds.

    python scripts/change_v2_report.py --out results/change-v2

Per run: lap times, crash and its type (spin if |slip| > 0.4 rad in the last 2 s before contact, else wall), laps to
recover (first lap after the change within 1% of the median of the last three laps), detection delay after the
change, detections in no-change runs, and how long each track stretch was conservative after a detection.
"""

import argparse
import glob
import json
from pathlib import Path
import numpy as np

SCENARIOS = {"grip-drop": "grip drop (μ 0.7)", "patch": "patch (μ 0.6, s 85-115 m)", "none": "no change"}
CONTROLLERS = {"bicycle": "bicycle (nominal)", "v11": "v11, safe set off", "v11-v2": "v11, safe set v2 + value B"}


def run_name(scenario, controller, seed):
    if scenario == "none":
        return f"change-none-v11-v2-s{seed}" if controller == "v11-v2" else None
    if controller == "v11-v2":
        return f"change-{scenario}-v11-v2-s{seed}"
    return f"change-{scenario}-{controller}" + ("" if seed == 1 else f"-s{seed}")


def analyse(run, change_lap):
    log = (run.parent / f"{run.name}.log").read_text().splitlines()
    times = [float(l.split("|")[1].split()[0]) for l in log if l.startswith("[lap]")]
    crashed = any(l.startswith("[attempt]") and "failure" in l for l in log)
    files = sorted(glob.glob(str(run / "trajectories" / "*" / "attempt-*.npz")), key=lambda p: Path(p).name)
    starts, start = [], 0
    for f in files:
        starts.append(start)
        start += len(np.load(f, allow_pickle=True)["cartesian"]) - 1
    entry = {"laps": times, "after_change": times[change_lap:], "crash": None}
    if crashed:
        a = np.load(sorted(glob.glob(str(run / "trajectories" / "failure" / "*.npz")))[-1], allow_pickle=True)
        contact = np.flatnonzero(a["contacts"])
        k = int(contact[0]) if len(contact) else len(a["frenet"]) - 1
        states = np.load(run / "encoder_true_states.npy")[-len(a["frenet"]):] if (run / "encoder_true_states.npy").exists() \
            else a["cartesian"]
        slip = float(np.abs(states[max(0, k - 20):k, 6]).max()) if k else 0.0
        entry["crash"] = {"lap": len(times) + 1, "seconds_into_lap": round(k * 0.1, 1), "s": round(float(a["frenet"][k, 0]), 1),
                          "type": "spin" if slip > 0.4 else "wall", "max_slip_last_2s": round(slip, 2)}
    after = times[change_lap:]
    recover = None
    if len(after) >= 4 and not crashed:
        target = np.median(after[-3:]) * 1.01
        recover = next((i + 1 for i, t in enumerate(after) if t <= target), None)
    entry["laps_to_recover"] = recover
    events_file = run / "change_events.json"
    if events_file.exists():
        info = json.loads(events_file.read_text())
        events = info["events"]
        change_step = starts[change_lap] if len(starts) > change_lap else None
        detections = [e for e in events if e["event"].startswith("detected")]
        entry["detections"] = [{"step": e["step"], "lap": e["lap"] + 1, "stretch": e["stretch"]} for e in detections]
        if change_step is not None:
            after_change = [e for e in detections if e["step"] >= change_step]
            entry["detection_delay_s"] = round((after_change[0]["step"] - change_step) * 0.1, 1) if after_change else None
            entry["detections_before_change"] = sum(e["step"] < change_step for e in detections)
        else:
            entry["detection_delay_s"] = None
            entry["detections_before_change"] = len(detections)
        count = info["stretches"]
        end = start
        conservative = np.zeros(count)
        span = 0
        if detections:
            first = detections[0]["step"]
            span = end - first
            since = {k: first for k in range(count)}
            for e in events:
                if e["event"] in ("detected again",):
                    for k in range(count):
                        since.setdefault(k, e["step"])
                if e["event"] == "relaxed" and e["stretch"] in since:
                    conservative[e["stretch"]] += e["step"] - since.pop(e["stretch"])
                if e["event"] == "new-condition laps in use":
                    for k in e["stretches"]:
                        if k in since:
                            conservative[k] += e["step"] - since.pop(k)
            for k, s0 in since.items():
                conservative[k] += end - s0
        share = conservative / span if span > 0 else conservative
        entry["conservative_share_per_stretch"] = [round(float(x), 2) for x in share]
        entry["stretches_mostly_conservative"] = [int(k) for k in np.flatnonzero(share > 0.5)]
        entry["relax_events"] = sum(e["event"] == "relaxed" for e in events)
        entry["changed_stretches"] = sorted({e["stretch"] for e in events if e["event"] == "changed"})
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="results/Spielberg")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--change-lap", type=int, default=8)
    parser.add_argument("--out", default="results/change-v2")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for scenario in SCENARIOS:
        for controller in CONTROLLERS:
            for seed in args.seeds:
                name = run_name(scenario, controller, seed)
                if name is None or not (Path(args.root) / f"{name}.log").exists():
                    continue
                entry = analyse(Path(args.root) / name, args.change_lap)
                report[name] = dict(entry, scenario=scenario, controller=controller, seed=seed)
                crash = entry["crash"]
                print(f"{name:32s} after change {[round(t, 1) for t in entry['after_change']]}  "
                      f"crash {crash['type'] + ' lap ' + str(crash['lap']) + ' at s=' + str(crash['s']) if crash else 'no'}  "
                      f"recover {entry['laps_to_recover']}  "
                      + (f"detection delay {entry.get('detection_delay_s')} s, detections {len(entry['detections'])} "
                         f"(before change {entry.get('detections_before_change')}), mostly conservative stretches "
                         f"{entry.get('stretches_mostly_conservative')}, changed {entry.get('changed_stretches')}"
                         if "detections" in entry else ""))
    summary = []
    for scenario, label in SCENARIOS.items():
        for controller, clabel in CONTROLLERS.items():
            rows = [r for r in report.values() if r["scenario"] == scenario and r["controller"] == controller]
            if not rows:
                continue
            crashes = [r for r in rows if r["crash"]]
            survived = [r for r in rows if not r["crash"]]
            row = {"scenario": label, "controller": clabel, "seeds": len(rows), "crashes": len(crashes),
                   "crash_types": sorted(r["crash"]["type"] for r in crashes),
                   "mean_lap_after_change_survivors": round(float(np.mean([np.mean(r["after_change"]) for r in survived])), 2)
                   if survived and all(r["after_change"] for r in survived) else None,
                   "laps_to_recover": [r["laps_to_recover"] for r in survived]}
            if any("detections" in r for r in rows):
                row["detection_delay_s"] = [r.get("detection_delay_s") for r in rows]
                row["false_detections"] = [r.get("detections_before_change") if scenario != "none" else len(r["detections"])
                                           for r in rows]
            summary.append(row)
    print("\nsummary")
    for row in summary:
        print(json.dumps(row, ensure_ascii=False))
    (out / "report.json").write_text(json.dumps({"runs": report, "summary": summary}, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
