"""On-track world-model dataset: random vehicles driven around real maps, with grip/mass changes, low-grip patches,
simulated IMU / wheel speed / PF pose (dyna-dist sensor model) and LiDAR scans (the bridge gym's segment scanner).

    python scripts/collect_track_data.py --out data/f1tenth-track --episodes 120

Per episode (one .npz): states (T+1, 7, CoG frame as dyna-dist), controls (T, 2: steering rate, acceleration),
conditions (T+1, 4: friction under the car, front/rear cornering scale, mass ratio), frenet (T+1, 3), lidar
(T+1, beams, float16, clean; add noise at training time), sensors (T+1, 7) and their parameters. Episodes end at a
crash (LiDAR clearance < --crash-clearance), at the slip/yaw-rate envelope, or after --steps.
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np

from dyna_dist.adapters.gym_st import GymST
from dyna_dist.data.sensors import SENSOR_FIELDS, sample_sensor_parameters, simulate_sensors
from dyna_dist.models.profiles import PARAMETER_FIELDS, vehicle_parameters

from sit_lmpc.track.resources import map_path
from sit_lmpc.track.track import Track

MAPS = {"Spielberg": "Spielberg", "custom1": "custom1", "levine": "tracks/levine_sim/levine_sim"}
SPEED_CAPS = {"Spielberg": (4.0, 10.0), "custom1": (5.0, 14.0), "levine": (3.0, 7.0)}
EXCITE = {"probability": 0.0}
LIDAR_OFFSET = 0.275
G = 9.81


def build_vehicle(mu, front, rear, mass):
    parameters = vehicle_parameters("f1tenth", mu, mass, "fixed")
    parameters["C_Sf"] *= front
    parameters["C_Sr"] *= rear
    return parameters


def vector(parameters):
    return np.array([parameters[k] for k in PARAMETER_FIELDS], dtype=np.float64)


def sample_episode(rng, steps, path_length, caps):
    base = np.array([rng.uniform(0.4, 1.2), rng.uniform(0.85, 1.15), rng.uniform(0.85, 1.15), rng.uniform(0.8, 1.3)])
    kind = str(rng.choice(["none", "step", "patches"], p=[0.3, 0.3, 0.4]))
    change = {"kind": kind}
    if kind == "step":
        what = str(rng.choice(["friction", "mass", "both"]))
        after = base.copy()
        if what in ("friction", "both"):
            after[0] = float(np.clip(base[0] * (rng.uniform(0.5, 0.85) if rng.random() < 0.7 else rng.uniform(1.15, 1.4)),
                                     0.3, 1.3))
        if what in ("mass", "both"):
            after[3] = rng.uniform(0.8, 1.3)
        change.update(what=what, step=int(rng.integers(steps // 4, 3 * steps // 4)), after=after.tolist())
    elif kind == "patches":
        patches = []
        for _ in range(int(rng.integers(1, 4))):
            start = rng.uniform(0, path_length)
            patches.append({"s_start": float(start), "s_end": float(start + rng.uniform(10, 40)),
                            "mu": float(max(0.25, base[0] * rng.uniform(0.35, 0.7)))})
        change.update(patches=patches, step=int(rng.integers(0, steps // 2)) if rng.random() < 0.5 else 0)
    driver = {"aggression": float(rng.uniform(0.5, 1.3)), "speed_cap": float(rng.uniform(*caps)),
              "belief_error": float(rng.uniform(0.8, 1.2)), "steer_noise": float(rng.uniform(0.0, 0.08)),
              "accel_noise": float(rng.uniform(0.0, 0.12)), "lookahead_gain": float(rng.uniform(0.12, 0.25)),
              "excite": bool(rng.random() < EXCITE["probability"])}
    return base, change, driver


def speed_profile(path, believed, aggression, cap, brake):
    window = 15
    curvature = np.convolve(np.pad(np.abs(path.ks), window, mode="wrap"), np.ones(2 * window + 1) / (2 * window + 1),
                            mode="same")[window:-window]
    corner = np.minimum(cap, np.sqrt(aggression * believed * G / np.maximum(curvature, 1e-3)))
    ds = np.diff(path.ss, append=path.ss[-1] + (path.ss[1] - path.ss[0]))
    limit = corner.copy()
    for _ in range(2):
        for i in range(len(limit) - 1, -1, -1):
            following = limit[(i + 1) % len(limit)]
            limit[i] = min(corner[i], np.sqrt(following ** 2 + 2 * brake * ds[i]))
    return limit


def condition_at(base, change, step, s, length):
    current = base.copy()
    if change["kind"] == "step" and step >= change["step"]:
        current = np.asarray(change["after"], dtype=float)
    if change["kind"] == "patches" and step >= change["step"]:
        for patch in change["patches"]:
            lo, hi = patch["s_start"] % length, patch["s_end"] % length
            inside = lo <= s <= hi if lo <= hi else (s >= lo or s <= hi)
            if inside:
                current = current.copy()
                current[0] = patch["mu"]
    return current


def drive(track, path, rng, steps, timestep, integrator, max_slip, max_yaw, caps):
    length = float(path.ss[-1])
    base, change, driver = sample_episode(rng, steps, length, caps)
    parameters = build_vehicle(*base)
    simulator = GymST(parameters, timestep, integrator)
    lf, lr = parameters["lf"], parameters["lr"]
    index = int(rng.integers(0, len(path.xs)))
    yaw = float(path.yaws[index])
    offset = rng.uniform(-0.2, 0.2)
    x = path.xs[index] - offset * np.sin(yaw) + lr * np.cos(yaw)
    y = path.ys[index] + offset * np.cos(yaw) + lr * np.sin(yaw)
    state = np.array([x, y, 0.0, rng.uniform(1.0, 5.0), yaw, 0.0, 0.0])
    states, controls, conditions, vehicles, frenet = [state], [], [], [], []
    believed = base[0] * driver["belief_error"]
    profile = speed_profile(path, believed, driver["aggression"], driver["speed_cap"], 0.5 * believed * G)
    noise = np.zeros(2)
    burst = {"left": 0, "accel": 0.0, "steer": 0.0}
    reason = "step_limit"
    guess = float(path.ss[index])
    for step in range(steps):
        rear = state[:2] - lr * np.array([np.cos(state[4]), np.sin(state[4])])
        s, ey, ephi = track.cartesian_to_frenet(rear[0], rear[1], state[4], s_guess=guess)
        guess = float(s)
        s = float(s) % length
        frenet.append([s, float(ey), float(ephi)])
        current = condition_at(base, change, step, s, length)
        conditions.append(current)
        live = build_vehicle(*current)
        simulator.params = vector(live)
        vehicles.append(simulator.params.copy())
        speed = float(state[3])
        target_speed = float(np.interp((s + speed * timestep) % length, path.ss, profile))
        lookahead = float(np.clip(0.5 + driver["lookahead_gain"] * speed, 0.6, 2.5))
        target_s = (s + lookahead) % length
        tx, ty = np.interp(target_s, path.ss, path.xs), np.interp(target_s, path.ss, path.ys)
        alpha = np.arctan2(ty - rear[1], tx - rear[0]) - state[4]
        target_steer = float(np.clip(np.arctan2(2 * (lf + lr) * np.sin(alpha), lookahead),
                                     parameters["s_min"], parameters["s_max"]))
        noise = noise + 0.3 * (-noise) + rng.normal(0.0, 1.0, 2) * np.sqrt(2 * 0.3) * np.array(
            [driver["steer_noise"], driver["accel_noise"]])
        steering_speed = np.sign(target_steer - state[2]) * min(abs(target_steer - state[2]) / 0.1, parameters["sv_max"])
        acceleration = 2.0 * (target_speed - speed)
        steering_speed = float(np.clip(steering_speed + noise[0] * parameters["sv_max"], parameters["sv_min"],
                                       parameters["sv_max"]))
        acceleration = float(np.clip(acceleration + noise[1] * parameters["a_max"], -parameters["a_max"],
                                     min(parameters["a_max"], (20.0 - speed) / timestep)))
        if driver["excite"]:
            if burst["left"] == 0 and rng.random() < 0.04:
                kinds = ["steer"] + (["throttle"] if speed < driver["speed_cap"] else []) + (["brake"] if speed > 2.0 else [])
                kind = rng.choice(kinds)
                burst["left"] = int(rng.integers(5, 11))
                burst["accel"] = {"throttle": 1.0, "brake": -1.0, "steer": 0.6}[kind] * parameters["a_max"]
                burst["steer"] = float(rng.choice([-1, 1]) * rng.uniform(0.15, 0.4)) if kind == "steer" else 0.0
            if burst["left"] > 0 and ((burst["accel"] > 0 and speed > driver["speed_cap"] + 1.0) or
                                      (burst["accel"] < 0 and speed < 1.5)):
                burst["left"] = 0
            if burst["left"] > 0:
                burst["left"] -= 1
                kicked = float(np.clip(target_steer + burst["steer"], parameters["s_min"], parameters["s_max"]))
                steering_speed = float(np.clip(np.sign(kicked - state[2]) * min(abs(kicked - state[2]) / 0.1,
                                                                                 parameters["sv_max"]),
                                               parameters["sv_min"], parameters["sv_max"]))
                acceleration = float(np.clip(burst["accel"], -parameters["a_max"],
                                             min(parameters["a_max"], (20.0 - speed) / timestep)))
        control = np.array([steering_speed, acceleration])
        state = simulator.step(state, control)
        controls.append(control)
        states.append(state)
        if not np.isfinite(state).all() or abs(state[6]) > max_slip or abs(state[5]) > max_yaw or state[3] < 0:
            reason = "envelope_limit"
            break
    rear = state[:2] - lr * np.array([np.cos(state[4]), np.sin(state[4])])
    s, ey, ephi = track.cartesian_to_frenet(rear[0], rear[1], state[4], s_guess=guess)
    frenet.append([float(s) % length, float(ey), float(ephi)])
    current = condition_at(base, change, len(controls), float(s) % length, length)
    conditions.append(current)
    vehicles.append(vector(build_vehicle(*current)))
    episode = {"states": np.asarray(states), "controls": np.asarray(controls), "conditions": np.asarray(conditions),
               "vehicles": np.asarray(vehicles), "frenet": np.asarray(frenet)}
    meta = {"base": base.tolist(), "change": change, "driver": driver, "reason": reason, "lr": lr}
    return episode, meta, simulator


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/f1tenth-track")
    parser.add_argument("--maps", nargs="+", default=list(MAPS))
    parser.add_argument("--episodes", nargs="+", default=["Spielberg=360", "custom1=120", "levine=150"],
                        help="episodes per map as name=count, or one count for every map")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--beams", type=int, default=1080)
    parser.add_argument("--fov", type=float, default=270.0)
    parser.add_argument("--crash-clearance", type=float, default=0.18)
    parser.add_argument("--start-index", type=int, default=0, help="first episode index (new seeds, new files)")
    parser.add_argument("--speed-cap", nargs="+", default=[], help="name=low,high overrides of the per-map speed cap")
    parser.add_argument("--append", action="store_true", help="add to an existing dataset's manifest")
    parser.add_argument("--excite", type=float, default=0.0,
                        help="share of episodes with random 0.5-1 s bursts of full throttle, full brake or steering kicks")
    args = parser.parse_args()

    import jax
    from f1tenth_gym.envs.lidar.segment_scan import SegmentScanSimulator2D

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    timestep, integrator = 0.1, 0.01
    manifest = {"seed": args.seed, "maps": args.maps, "steps": args.steps, "timestep": timestep,
                "beams": args.beams, "fov_deg": args.fov, "lidar_offset": LIDAR_OFFSET, "sensor_fields": list(SENSOR_FIELDS),
                "condition_fields": ["friction", "front_cornering_scale", "rear_cornering_scale", "mass_ratio"],
                "episodes": []}
    counts = ({k: int(v) for k, v in (item.split("=") for item in args.episodes)} if "=" in args.episodes[0]
              else {name: int(args.episodes[0]) for name in args.maps})
    EXCITE["probability"] = args.excite
    for item in args.speed_cap:
        key, values = item.split("=")
        SPEED_CAPS[key] = tuple(float(v) for v in values.split(","))
    previous = json.loads((out / "manifest.json").read_text()) if args.append and (out / "manifest.json").exists() else None
    if previous is not None:
        manifest["episodes"] = previous["episodes"]
        manifest["maps"] = sorted(set(previous["maps"]) | set(args.maps))
        manifest["additions"] = previous.get("additions", []) + [{"maps": args.maps, "start_index": args.start_index,
                                                                    "episodes": args.episodes, "speed_caps": args.speed_cap}]
    for map_index, name in enumerate(args.maps):
        map_index = list(MAPS).index(name)
        track = Track.from_track_path(map_path(MAPS[name]))
        path = track.centerline if track.centerline is not None else track.raceline
        scanner = SegmentScanSimulator2D(args.beams, np.deg2rad(args.fov), std_dev=0.0, min_range=0.05, max_range=30.0,
                                         device="gpu" if any(d.platform == "gpu" for d in jax.devices()) else "cpu")
        scanner.set_map(track)
        batch_scan = jax.jit(lambda poses: jax.lax.map(scanner._scan, poses, batch_size=16))
        for episode_index in range(args.start_index, args.start_index + counts[name]):
            rng = np.random.default_rng([args.seed, map_index, episode_index])
            episode, meta, simulator = drive(track, path, rng, args.steps, timestep, integrator, 1.2, 10.0, SPEED_CAPS[name])
            meta["speed_caps"] = list(SPEED_CAPS[name])
            states = episode["states"]
            yaw = states[:, 4]
            rear = states[:, :2] - meta["lr"] * np.stack([np.cos(yaw), np.sin(yaw)], 1)
            lidar_xy = rear + LIDAR_OFFSET * np.stack([np.cos(yaw), np.sin(yaw)], 1)
            poses = np.column_stack([lidar_xy, yaw]).astype(np.float32)
            padded = np.concatenate([poses, np.repeat(poses[-1:], args.steps + 1 - len(poses), axis=0)])
            scans = np.asarray(batch_scan(padded))[:len(poses)]
            crashed = np.flatnonzero(scans.min(axis=1) < args.crash_clearance)
            count = len(states)
            if len(crashed):
                count = max(int(crashed[0]), 1)
                meta["reason"] = "crash"
            for key in ("states", "frenet", "conditions", "vehicles"):
                episode[key] = episode[key][:count]
            episode["controls"] = episode["controls"][:count - 1]
            scans = scans[:count]
            if count < 20:
                continue
            sensor_parameters = sample_sensor_parameters(rng, None)
            readings = simulate_sensors(episode["states"], episode["controls"], simulator.rhs, episode["vehicles"],
                                        timestep, rng, sensor_parameters)
            split = "test" if episode_index % 10 == 9 else "val" if episode_index % 10 == 8 else "train"
            filename = f"{split}/{name}-{episode_index:04d}.npz"
            (out / split).mkdir(exist_ok=True)
            names = sorted(sensor_parameters)
            np.savez_compressed(out / filename, states=episode["states"], controls=episode["controls"],
                                conditions=episode["conditions"], frenet=episode["frenet"],
                                lidar=scans.astype(np.float16), sensors=readings,
                                sensor_parameter_names=np.asarray(names),
                                sensor_parameter_values=np.asarray([sensor_parameters[n] for n in names]))
            slip = np.abs(episode["states"][:, 6])
            manifest["episodes"].append({"file": filename, "map": name, "split": split, "steps": int(count - 1),
                                         "reason": meta["reason"], "base": meta["base"], "change": meta["change"],
                                         "driver": meta["driver"], "speed_caps": meta["speed_caps"],
                                         "max_slip": float(slip.max()),
                                         "max_speed": float(episode["states"][:, 3].max())})
            print(f"{name:9s} {episode_index:4d} {split:5s} {count - 1:4d} steps  {meta['reason']:14s} "
                  f"change {meta['change']['kind']:7s} max slip {slip.max():.2f} max speed {episode['states'][:, 3].max():5.1f}",
                  flush=True)
    episodes = manifest["episodes"]
    all_slip = [e["max_slip"] for e in episodes]
    manifest["summary"] = {"episodes": len(episodes), "transitions": int(sum(e["steps"] for e in episodes)),
                           "crashes": int(sum(e["reason"] == "crash" for e in episodes)),
                           "episodes_with_slip_over_0.3": int(sum(s > 0.3 for s in all_slip))}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
