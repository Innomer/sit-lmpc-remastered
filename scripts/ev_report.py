"""Evidence-based speed reaction: per seed and a summary.

    python scripts/ev_report.py --out results/footprint/evidence-reaction.json

Run names ev-<scenario>-<controller>-s<seed>: scenarios grip055, patch045 (change after lap 8) and none; controllers
bicycle, v11, v11-react. Per run: crash (type, lap, s, hit point), laps driven after the change, all lap times, laps
under 0.2 m body clearance to the wall after the change (scripts/harsh_change_report.py measure), mean speed vs cap
on capped steps (change_trace.npy, per lap), detection delay (from the start of lap 9, or entry into the patch),
false detections (before the change, or any in a no-change run) and every cap event (set / raised / removed).
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from change_v2_report import analyse
from fp_change_report import speeds_around_detection
from harsh_change_report import wall_distance_field, outline, per_lap, patch_delay
import re

CONTROLLERS = {"bicycle": "bicycle + footprint", "v11": "v11 + footprint, no reaction",
               "v11-react": "v11 + footprint + evidence reaction"}
SCENARIOS = {"grip055": ("grip drop (μ 0.55)", None), "patch045": ("patch (μ 0.45, s 85–115 m)", 85.0),
             "none": ("no change", None)}


def lap_starts(run):
    log = (run.parent / f"{run.name}.log").read_text().splitlines()
    steps = [int(l.split("|")[1].split()[0]) for l in log if l.startswith("[attempt]")]
    return np.cumsum([0] + steps)


def cap_speed(run):
    trace_file = run / "change_trace.npy"
    if not trace_file.exists():
        return None
    t = np.load(trace_file)
    if not len(t):
        return None
    starts = lap_starts(run)
    cap, speed = t[:, 9], t[:, 10]
    capped = np.isfinite(cap) & (cap < 100)
    out = {"capped_steps": int(capped.sum())}
    if capped.any():
        out["mean_speed"] = round(float(speed[capped].mean()), 2)
        out["mean_cap"] = round(float(cap[capped].mean()), 2)
        out["mean_gap"] = round(float((cap[capped] - speed[capped]).mean()), 2)
        per = []
        for lap in range(len(starts) - 1):
            m = capped & (t[:, 0] >= starts[lap]) & (t[:, 0] < starts[lap + 1])
            if m.any():
                per.append({"lap": lap + 1, "capped_share": round(float(m.sum() / max(1, ((t[:, 0] >= starts[lap]) & (t[:, 0] < starts[lap + 1])).sum())), 2),
                            "speed": round(float(speed[m].mean()), 2), "cap": round(float(cap[m].mean()), 2)})
        out["per_lap"] = per
    return out


def cap_events(run):
    events_file = run / "change_events.json"
    if not events_file.exists():
        return None
    events = json.loads(events_file.read_text())["events"]
    keep = [e for e in events if e["event"] in ("cap raised", "cap removed", "detected", "detected again")]
    sets = [e for e in events if e["event"] == "cap set"]
    out = [{"lap": e["lap"] + 1, "event": e["event"], "stretch": e.get("stretch"),
            **({"cap": round(e["cap"], 2)} if "cap" in e else {}),
            **({"reason": e["reason"]} if "reason" in e else {})} for e in keep]
    return {"caps_set": len(sets), "caps_set_laps": sorted({e["lap"] + 1 for e in sets}), "events": out}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="results/Spielberg")
    parser.add_argument("--prefix", default="ev")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--change-lap", type=int, default=8)
    parser.add_argument("--reuse", default="hc", help="prefix of earlier bicycle / v11 runs with the same configs")
    parser.add_argument("--extra", nargs="*", default=[], help="other run names to include (e.g. check runs)")
    parser.add_argument("--out", default="results/footprint/evidence-reaction.json")
    args = parser.parse_args()
    field, resolution, origin, height = wall_distance_field()
    body = outline()
    report, summary = {}, []
    names = [(f"{args.prefix}-{sc}-{c}-s{s}", sc, c) for sc in SCENARIOS for c in CONTROLLERS for s in args.seeds]
    names += [(n, next(sc for sc in SCENARIOS if sc in n), "v11-react") for n in args.extra]
    for name, scenario, controller in names:
        run = Path(args.root) / name
        if not (Path(args.root) / f"{name}.log").exists() and controller != "v11-react" and scenario != "none":
            reused = name.replace(f"{args.prefix}-", f"{args.reuse}-", 1)
            if (Path(args.root) / f"{reused}.log").exists():
                name, run = reused, Path(args.root) / reused
        if not (Path(args.root) / f"{name}.log").exists():
            continue
        label, s_start = SCENARIOS[scenario]
        change = args.change_lap if scenario != "none" else None
        entry = analyse(run, change if change else 10 ** 6)
        log = (run.parent / f"{name}.log").read_text().splitlines()
        entry["lap_times"] = [float(l.split("|")[1].split()[0]) for l in log if l.startswith("[lap]")]
        hits = re.findall(r"hit something at \(([-\d.]+), ([-\d.]+)\)", (run.parent / f"{name}.bridge.log").read_text())
        if entry["crash"] and hits:
            entry["crash"]["hit"] = [float(hits[0][0]), float(hits[0][1])]
        laps = per_lap(run, field, resolution, origin, height, body)
        after = [l for l in laps if change is None or l["attempt"] > change]
        entry["laps_after_change"] = None if change is None else sum(1 for l in after if not l["contact"])
        entry["clearance_after"] = [l["min_body_clearance_m"] for l in after]
        entry["laps_under_0.2m"] = f"{sum(c < 0.2 for c in entry['clearance_after'])}/{len(after)}"
        entry["cap_speed"] = cap_speed(run)
        entry["cap_events"] = cap_events(run)
        if s_start is not None and "detections" in entry:
            entry["detection_delay_s"] = patch_delay(run, None, s_start, args.change_lap)
        if change is None and "detections" in entry:
            entry["false_detections"] = len(entry["detections"])
        elif "detections" in entry:
            entry["false_detections"] = entry.get("detections_before_change")
        entry["speed_at_detection"] = speeds_around_detection(run) if "detections" in entry and entry["detections"] else None
        entry["scenario"], entry["controller"] = scenario, controller
        report[name] = entry
        crash = entry["crash"]
        cs = entry["cap_speed"] or {}
        print(f"{name:28s} crash {(crash['type'] + ' lap ' + str(crash['lap']) + ' s=' + str(crash['s']) + ' hit ' + str(crash.get('hit'))) if crash else 'no'}"
              f" | after-change laps {entry['laps_after_change']} | under 0.2 m {entry['laps_under_0.2m']}"
              f" | laps {[round(t, 1) for t in entry['lap_times']]}"
              + (f" | speed/cap {cs.get('mean_speed')}/{cs.get('mean_cap')} gap {cs.get('mean_gap')}" if cs.get("capped_steps") else "")
              + (f" | det delay {entry.get('detection_delay_s')} false {entry.get('false_detections')} speed {entry['speed_at_detection']}" if "detections" in entry else ""))
    for scenario, (label, _) in SCENARIOS.items():
        for controller, clabel in CONTROLLERS.items():
            rows = [r for n, r in report.items() if r["scenario"] == scenario and r["controller"] == controller
                    and (n.startswith(args.prefix + "-") or n.startswith(args.reuse + "-"))]
            if not rows:
                continue
            under = [r["laps_under_0.2m"].split("/") for r in rows]
            caps = [r["cap_speed"] for r in rows if r["cap_speed"] and r["cap_speed"].get("capped_steps")]
            summary.append({
                "scenario": label, "controller": clabel, "seeds": len(rows), "crashes": sum(bool(r["crash"]) for r in rows),
                "crash_details": [f"{r['crash']['type']} lap {r['crash']['lap']} s={r['crash']['s']}" for r in rows if r["crash"]],
                "laps_after_change": [r["laps_after_change"] for r in rows],
                "laps_under_0.2m": f"{sum(int(u[0]) for u in under)}/{sum(int(u[1]) for u in under)}",
                **({"mean_speed_vs_cap": [f"{c['mean_speed']}/{c['mean_cap']}" for c in caps]} if caps else {}),
                **({"detection_delay_s": [r.get("detection_delay_s") for r in rows],
                    "false_detections": [r.get("false_detections") for r in rows]} if any("detections" in r for r in rows) else {})})
    print("\nsummary")
    for row in summary:
        print(json.dumps(row, ensure_ascii=False))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"runs": report, "summary": summary}, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
