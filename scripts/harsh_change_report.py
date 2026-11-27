"""Harsher change scenarios: per seed and a summary, with per-lap near-misses.

    python scripts/harsh_change_report.py --out results/footprint/harsh-change.json

Run names hc-<scenario>-<controller>-s<seed> (scenarios grip055, patch045; controllers bicycle, v11, v11-react).
Per run (via scripts/change_v2_report.analyse): crash and type, lap times after the change, laps to recover,
detections. Crash location: track position s and the sim's hit point. Per lap: minimum body clearance to the wall
(distance from the body outline, 0.58 x 0.31 m about the centre of gravity, to the nearest occupied pixel of the sim's
Spielberg map) and maximum |slip angle|. Detection delay: from the change (start of lap 9 for the grip drop, entry into
the patch for the patch). Speed: mean over 2 s before and after the first detection.
"""

import argparse
import glob
import json
import re
from pathlib import Path
import sys
import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from change_v2_report import analyse
from fp_change_report import speeds_around_detection

CONTROLLERS = {"bicycle": "bicycle + footprint", "v11": "v11 + footprint, no reaction",
               "v11-react": "v11 + footprint + fixed reaction"}
SCENARIOS = {"grip055": ("grip drop (μ 0.55)", None), "patch045": ("patch (μ 0.45, s 85–115 m)", 85.0)}
MAP = Path(__file__).resolve().parent.parent / "src/sit_lmpc/track/maps/Spielberg/Spielberg_map.yaml"


def wall_distance_field(map_yaml=MAP):
    map_yaml = Path(map_yaml)
    info = yaml.safe_load(map_yaml.read_text())
    image = np.asarray(Image.open(map_yaml.parent / info["image"]).convert("L"), dtype=float) / 255.0
    occupied = (1.0 - image) > info["occupied_thresh"]
    field = distance_transform_edt(~occupied) * info["resolution"]
    return field, info["resolution"], np.asarray(info["origin"][:2]), image.shape[0]


def outline():
    xs = np.linspace(-0.2965, 0.2835, 9)
    ys = np.linspace(-0.155, 0.155, 5)
    points = [(x, y) for x in xs for y in (-0.155, 0.155)] + [(x, y) for x in (-0.2965, 0.2835) for y in ys]
    return np.asarray(points)


def clearance(cartesian, field, resolution, origin, height, body):
    x, y, yaw = cartesian[:, 0], cartesian[:, 1], cartesian[:, 4]
    px = x[:, None] + body[:, 0] * np.cos(yaw)[:, None] - body[:, 1] * np.sin(yaw)[:, None]
    py = y[:, None] + body[:, 0] * np.sin(yaw)[:, None] + body[:, 1] * np.cos(yaw)[:, None]
    col = np.clip(((px - origin[0]) / resolution).astype(int), 0, field.shape[1] - 1)
    row = np.clip(height - 1 - ((py - origin[1]) / resolution).astype(int), 0, field.shape[0] - 1)
    return field[row, col].min(axis=1)


def per_lap(run, field, resolution, origin, height, body):
    files = sorted(glob.glob(str(run / "trajectories" / "*" / "attempt-*.npz")), key=lambda p: Path(p).name)
    laps = []
    for path in files:
        a = np.load(path, allow_pickle=True)
        c = a["cartesian"]
        contact = np.flatnonzero(a["contacts"]) if "contacts" in a.files else np.zeros(0, int)
        end = int(contact[0]) if len(contact) else len(c)
        if end == 0:
            continue
        laps.append({"attempt": int(Path(path).stem.split("-")[-1]),
                     "min_body_clearance_m": round(float(clearance(c[:end], field, resolution, origin, height, body).min()), 3),
                     "max_abs_slip_rad": round(float(np.abs(c[:end, 6]).max()), 3),
                     "contact": bool(len(contact))})
    return laps


def patch_delay(run, entry, s_start, change_lap):
    events_file = run / "change_events.json"
    if not events_file.exists():
        return None
    detections = [e for e in json.loads(events_file.read_text())["events"] if e["event"].startswith("detected")]
    log = (run.parent / f"{run.name}.log").read_text().splitlines()
    steps = [int(l.split("|")[1].split()[0]) for l in log if l.startswith("[attempt]")]
    start = sum(steps[:change_lap])
    files = sorted(glob.glob(str(run / "trajectories" / "*" / "attempt-*.npz")), key=lambda p: Path(p).name)
    s = np.concatenate([np.load(f, allow_pickle=True)["frenet"][(0 if i == 0 else 1):, 0] for i, f in enumerate(files)])
    inside = np.flatnonzero(s[start:start + 800] >= s_start)
    if not len(inside):
        return None
    enter = start + int(inside[0])
    after = [e for e in detections if e["step"] >= enter]
    return round((after[0]["step"] - enter) * 0.1, 1) if after else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="results/Spielberg")
    parser.add_argument("--prefix", default="hc")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--change-lap", type=int, default=8)
    parser.add_argument("--out", default="results/footprint/harsh-change.json")
    args = parser.parse_args()
    field, resolution, origin, height = wall_distance_field()
    body = outline()
    report, summary = {}, []
    for scenario, (label, s_start) in SCENARIOS.items():
        for controller, clabel in CONTROLLERS.items():
            rows = []
            for seed in args.seeds:
                name = f"{args.prefix}-{scenario}-{controller}-s{seed}"
                run = Path(args.root) / name
                if not (Path(args.root) / f"{name}.log").exists():
                    continue
                entry = analyse(run, args.change_lap)
                hits = re.findall(r"hit something at \(([-\d.]+), ([-\d.]+)\)", (run.parent / f"{name}.bridge.log").read_text())
                if entry["crash"] and hits:
                    entry["crash"]["hit"] = [float(hits[0][0]), float(hits[0][1])]
                laps = per_lap(run, field, resolution, origin, height, body)
                entry["laps"] = laps
                entry["after_change_laps"] = [l for l in laps if l["attempt"] > args.change_lap]
                entry["before_change_laps"] = [l for l in laps if args.change_lap - 2 < l["attempt"] <= args.change_lap]
                if s_start is not None and "detections" in entry:
                    entry["detection_delay_s"] = patch_delay(run, None, s_start, args.change_lap)
                entry["speed_at_detection"] = speeds_around_detection(run)
                report[name] = entry
                rows.append(entry)
                crash = entry["crash"]
                after = entry["after_change_laps"]
                print(f"{name:26s} crash {(crash['type'] + ' lap ' + str(crash['lap']) + ' s=' + str(crash['s']) + ' hit ' + str(crash.get('hit'))) if crash else 'no'}\n"
                      f"    after change {[round(t, 1) for t in entry['after_change']]}  recover {entry['laps_to_recover']}\n"
                      f"    min clearance {[l['min_body_clearance_m'] for l in after]}  (before {[l['min_body_clearance_m'] for l in entry['before_change_laps']]})\n"
                      f"    max |slip| {[l['max_abs_slip_rad'] for l in after]}  (before {[l['max_abs_slip_rad'] for l in entry['before_change_laps']]})"
                      + (f"\n    detection delay {entry.get('detection_delay_s')} s, false {entry.get('detections_before_change')}, "
                         f"speed {entry['speed_at_detection']}" if "detections" in entry else ""))
            if not rows:
                continue
            crashed = [r for r in rows if r["crash"]]
            survivors = [r for r in rows if not r["crash"] and r["after_change"]]
            after = [l for r in rows for l in r["after_change_laps"]]
            summary.append({
                "scenario": label, "controller": clabel, "seeds": len(rows), "crashes": len(crashed),
                "crash_details": [f"{r['crash']['type']} lap {r['crash']['lap']} s={r['crash']['s']}" for r in crashed],
                "mean_lap_after_change_survivors": round(float(np.mean([np.mean(r["after_change"]) for r in survivors])), 2) if survivors else None,
                "laps_to_recover": [r["laps_to_recover"] for r in rows],
                "laps_under_0.2m_clearance": f"{sum(l['min_body_clearance_m'] < 0.2 for l in after)}/{len(after)}",
                "median_min_clearance_m": round(float(np.median([l["min_body_clearance_m"] for l in after])), 3) if after else None,
                "laps_slip_over_0.2rad": f"{sum(l['max_abs_slip_rad'] > 0.2 for l in after)}/{len(after)}",
                "median_max_slip_rad": round(float(np.median([l["max_abs_slip_rad"] for l in after])), 3) if after else None,
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
