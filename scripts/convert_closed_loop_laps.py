"""Add the planner's own driving to the on-track dataset: learned (LMPPI) laps from earlier closed-loop Spielberg runs,
with simulated IMU / wheel / PF pose (dyna-dist sensor model) and LiDAR computed from the recorded poses.

    python scripts/convert_closed_loop_laps.py --out data/f1tenth-track

Uses ROS sim runs whose saved states are true states (not a model's estimate) and only learned laps (warm-up laps
apply pure pursuit ten times per step). True grip per step from each run's scheduler log (track-wide changes after N
finished laps, patches by track position); mass ratio and cornering stiffness are nominal.
"""

import argparse
import glob
import json
import sys
import zlib
from pathlib import Path
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_track_data import LIDAR_OFFSET, build_vehicle, vector

from dyna_dist.adapters.gym_st import GymST
from dyna_dist.data.sensors import sample_sensor_parameters, simulate_sensors
from sit_lmpc.track.resources import map_path
from sit_lmpc.track.track import Track

NOMINAL_MU = 1.0489


def usable_runs(root):
    runs = []
    for folder in sorted(Path(root).glob("*/")):
        config_file = folder / "config.yaml"
        if not config_file.exists() or not (folder / "trajectories").exists():
            continue
        config = yaml.safe_load(config_file.read_text())
        dynamics = config.get("planner", {}).get("dynamics", {})
        replaced = dynamics.get("backend") == "rssm" or (dynamics.get("belief_update") == "encoder"
                                                          and dynamics.get("encoder_state", True))
        if config.get("simulation", {}).get("backend") == "ros" and not replaced and \
                config.get("simulation", {}).get("map", "Spielberg") == "Spielberg":
            runs.append(folder)
    return runs


def events(folder):
    path = Path(str(folder) + ".conditions.jsonl")
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            event = json.loads(line)["event"]
            out.append((event["lap"], event.get("condition", {}).get("mu"), event.get("patches")))
    return out


def friction(evts, finished, s, length):
    mu, patches = NOMINAL_MU, []
    for lap, value, new in evts:
        if finished >= lap:
            mu = value if value is not None else mu
            patches = new if new is not None else patches
    for patch in patches:
        lo, hi = patch["s_start"] % length, patch["s_end"] % length
        if (lo <= s <= hi) if lo <= hi else (s >= lo or s <= hi):
            return patch["mu"]
    return mu


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", default="results/Spielberg")
    parser.add_argument("--out", default="data/f1tenth-track")
    parser.add_argument("--seed", type=int, default=20261123)
    args = parser.parse_args()

    import jax
    from f1tenth_gym.envs.lidar.segment_scan import SegmentScanSimulator2D

    out = Path(args.out)
    manifest = json.loads((out / "manifest.json").read_text())
    known = {e["file"] for e in manifest["episodes"]}
    track = Track.from_track_path(map_path("Spielberg"))
    length = float(track.centerline.ss[-1])
    scanner = SegmentScanSimulator2D(manifest["beams"], np.deg2rad(manifest["fov_deg"]), std_dev=0.0, min_range=0.05,
                                     max_range=30.0, device="gpu" if any(d.platform == "gpu" for d in jax.devices()) else "cpu")
    scanner.set_map(track)
    batch_scan = jax.jit(lambda poses: jax.lax.map(scanner._scan, poses, batch_size=16))
    nominal = build_vehicle(NOMINAL_MU, 1.0, 1.0, 1.0)
    simulator = GymST(nominal, 0.1, 0.01)
    lr = nominal["lr"]
    added = 0
    added_steps = 0
    for folder in usable_runs(args.runs_root):
        evts = events(folder)
        files = sorted(glob.glob(str(folder / "trajectories" / "*" / "attempt-*.npz")), key=lambda p: Path(p).name)
        finished = 0
        for path in files:
            data = np.load(path, allow_pickle=True)
            outcome = Path(path).parent.name
            controllers = data["controllers"]
            learned = len(controllers) and str(controllers[0]) == "lmppi"
            finished_before = finished
            finished += outcome == "success"
            if not learned:
                continue
            states = data["cartesian"].astype(np.float64)
            controls = data["controls"].astype(np.float64)
            count = min(len(states), len(controls) + 1)
            states, controls, frenet = states[:count], controls[:count - 1], data["frenet"][:count]
            if count < 41:
                continue
            name = f"cl-{folder.name}-{Path(path).stem[-6:]}"
            key = zlib.crc32(name.encode()) % 10
            split = "test" if key == 9 else "val" if key == 8 else "train"
            filename = f"{split}/{name}.npz"
            if filename in known:
                continue
            mus = np.array([friction(evts, finished_before, float(s) % length, length) for s in frenet[:, 0]])
            conditions = np.column_stack((mus, np.ones(count), np.ones(count), np.ones(count)))
            vehicles = np.stack([vector(build_vehicle(m, 1.0, 1.0, 1.0)) for m in mus])
            rng = np.random.default_rng([args.seed, zlib.crc32(name.encode())])
            sensor_parameters = sample_sensor_parameters(rng, None)
            readings = simulate_sensors(states, controls, simulator.rhs, vehicles, 0.1, rng, sensor_parameters)
            yaw = states[:, 4]
            rear = states[:, :2] - lr * np.stack([np.cos(yaw), np.sin(yaw)], 1)
            lidar_xy = rear + LIDAR_OFFSET * np.stack([np.cos(yaw), np.sin(yaw)], 1)
            poses = np.column_stack([lidar_xy, yaw]).astype(np.float32)
            scans = []
            for start in range(0, len(poses), 600):
                chunk = poses[start:start + 600]
                padded = np.concatenate([chunk, np.repeat(chunk[-1:], 600 - len(chunk), axis=0)])
                scans.append(np.asarray(batch_scan(padded))[:len(chunk)])
            scans = np.concatenate(scans)
            (out / split).mkdir(exist_ok=True)
            names = sorted(sensor_parameters)
            np.savez_compressed(out / filename, states=states, controls=controls, conditions=conditions, frenet=frenet,
                                lidar=scans.astype(np.float16), sensors=readings,
                                sensor_parameter_names=np.asarray(names),
                                sensor_parameter_values=np.asarray([sensor_parameters[n] for n in names]))
            kind = "patches" if any(p for _, _, p in evts) else "step" if any(m is not None for _, m, _ in evts) else "none"
            manifest["episodes"].append({"file": filename, "map": "Spielberg", "split": split, "steps": int(count - 1),
                                         "reason": "crash" if outcome != "success" else "lap",
                                         "base": [NOMINAL_MU, 1.0, 1.0, 1.0], "change": {"kind": kind},
                                         "driver": {"source": "closed_loop", "run": folder.name},
                                         "max_slip": float(np.abs(states[:, 6]).max()),
                                         "max_speed": float(states[:, 3].max())})
            added += 1
            added_steps += count - 1
        print(f"{folder.name:44s} total added so far: {added} laps, {added_steps} steps", flush=True)
    manifest.setdefault("additions", []).append({"source": "closed_loop", "laps": added, "steps": added_steps})
    episodes = manifest["episodes"]
    manifest["summary"] = {"episodes": len(episodes), "transitions": int(sum(e["steps"] for e in episodes)),
                           "crashes": int(sum(e["reason"] == "crash" for e in episodes))}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["summary"]))


if __name__ == "__main__":
    main()
