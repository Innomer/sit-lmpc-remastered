"""Spin-onset data: groups of near-identical starts near the grip limit, so the same situation ends in a spin, a wall
contact or a recovery depending on small differences.

    python scripts/collect_spin_onsets.py --out data/f1tenth-track --groups 120 --rollouts 8

Each group picks a map position, a random vehicle (grip 0.4-1.2, cornering stiffness +-15%, mass 0.8-1.3, as the main
collector) and one of three onset kinds:
  corner     enters a corner above its grip-limited speed
  patch      a low-grip patch (35-70% of base grip) the driver does not know about, on the corner
  kick       a sudden steering burst (up to 0.4 rad for 0.3-0.6 s) at speed
The driver is the main collector's pure pursuit with a speed profile at 70% of the true grip limit (+-5%), so the
onset is the only thing that makes a group hard.
The severity of each group (corner entry speed factor, patch grip ratio or kick angle) is first set by bisection on
unperturbed rollouts to the point where the outcome flips between crash and recovery. Every rollout of the group then
starts from the same state at that severity with small perturbations (speed 2%, lateral 3 cm, heading
0.01 rad, yaw rate 0.05 rad/s) and its own small control noise, and runs up to --steps steps. A rollout ends at the
first wall contact (car body outline vs the map, as scripts/mark_contacts.py) or past slip 1.2 rad / yaw rate 10 rad/s.
Outcome: contact, spin (|slip| > 0.5 rad or past the envelope) or recovery. All rollouts of a group go to the same
split. Episodes are appended to the dataset manifest with an "onset" record; run scripts/mark_contacts.py afterwards.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_track_data import G, LIDAR_OFFSET, MAPS, build_vehicle, vector
from mark_contacts import Occupancy

from dyna_dist.adapters.gym_st import GymST
from dyna_dist.data.sensors import sample_sensor_parameters, simulate_sensors
from sit_lmpc.track.resources import map_path
from sit_lmpc.track.track import Track

SPIN_SLIP = 0.5
DRIVER_AGGRESSION = 0.7
SPEED_LIMIT = {"Spielberg": 15.0, "custom1": 15.0, "levine": 9.0}


def smoothed_curvature(path, window=15):
    return np.convolve(np.pad(np.abs(path.ks), window, mode="wrap"), np.ones(2 * window + 1) / (2 * window + 1),
                       mode="same")[window:-window]


SEVERITY = {"corner": (0.9, 1.7), "patch": (1.0, 0.25), "kick": (0.02, 0.4)}


def with_severity(setup, severity):
    """corner: entry speed = factor x grip-limited apex speed; patch: patch grip = ratio x base grip; kick: angle."""
    setup = dict(setup, severity=float(severity))
    if setup["kind"] == "corner":
        setup["speed"] = float(max(2.0, min(SPEED_LIMIT_CAP, setup["apex_limit"] * severity)))
    elif setup["kind"] == "patch":
        setup["patch"] = dict(setup["patch"], mu=float(max(0.15, setup["base"][0] * severity)))
    else:
        setup["kick"] = dict(setup["kick"], angle=float(np.sign(setup["kick"]["angle"]) * severity))
    return setup


def tipping_point(track, path, curvature, setup, steps, occupancy, iterations=6):
    """Bisect the severity between its mild and harsh ends until an unperturbed rollout flips crash/recovery."""
    mild, harsh = SEVERITY[setup["kind"]]
    crash = lambda sev: rollout(track, path, curvature, with_severity(setup, sev), None, steps, occupancy)[1] != "recovery"
    if crash(mild):
        return mild, "crash at the mild end"
    if not crash(harsh):
        return harsh, "no crash at the harsh end"
    for _ in range(iterations):
        middle = 0.5 * (mild + harsh)
        if crash(middle):
            harsh = middle
        else:
            mild = middle
    return 0.5 * (mild + harsh), "bisected"


def group_setup(rng, path, curvature, name):
    """Start index, start speed, vehicle, patch and steering kick for one onset group."""
    base = np.array([rng.uniform(0.4, 1.2), rng.uniform(0.85, 1.15), rng.uniform(0.85, 1.15), rng.uniform(0.8, 1.3)])
    kind = str(rng.choice(["corner", "patch", "kick"]))
    length = float(path.ss[-1])
    spacing = float(np.mean(np.diff(path.ss)))
    corners = np.flatnonzero(curvature > np.percentile(curvature, 85))
    setup = {"kind": kind, "base": base, "patch": None, "kick": None, "aggression": DRIVER_AGGRESSION, "belief": 1.0}
    if kind == "kick":
        start = int(rng.integers(0, len(path.xs)))
        limit = np.sqrt(base[0] * G / max(curvature[start], 1e-3))
        speed = float(min(SPEED_LIMIT[name], limit) * rng.uniform(0.6, 1.0))
        setup["kick"] = {"step": int(rng.integers(3, 9)), "steps": int(rng.integers(3, 7)),
                         "angle": float(rng.choice([-1, 1]) * rng.uniform(0.15, 0.4))}
    else:
        apex = int(rng.choice(corners))
        apex_limit = np.sqrt(base[0] * G / max(curvature[apex], 1e-3))
        lead = rng.uniform(0.8, 1.5)
        factor = rng.uniform(1.05, 1.45) if kind == "corner" else rng.uniform(0.85, 1.1)
        speed = float(min(SPEED_LIMIT[name], apex_limit * factor))
        setup["apex_limit"] = float(apex_limit)
        start = int((apex - round(speed * lead / spacing)) % len(path.xs))
        if kind != "corner":
            centre = float(path.ss[apex])
            half = rng.uniform(5.0, 15.0)
            setup["patch"] = {"s_start": (centre - half) % length, "s_end": (centre + half) % length,
                              "mu": float(max(0.2, base[0] * rng.uniform(0.35, 0.7)))}
    setup["belief"] = float(rng.uniform(0.95, 1.05))
    setup.update(start=start, speed=max(speed, 2.0))
    return setup


def grip_at(setup, s, length):
    current = setup["base"].copy()
    patch = setup["patch"]
    if patch is not None:
        lo, hi = patch["s_start"], patch["s_end"]
        if (lo <= s <= hi) if lo <= hi else (s >= lo or s <= hi):
            current[0] = patch["mu"]
    return current


def rollout(track, path, curvature, setup, rng, steps, occupancy):
    base = setup["base"]
    parameters = build_vehicle(*base)
    simulator = GymST(parameters, 0.1, 0.01)
    lf, lr = parameters["lf"], parameters["lr"]
    length = float(path.ss[-1])
    i = setup["start"]
    quiet = rng is None
    rng = np.random.default_rng(0) if quiet else rng
    noise = 0.0 if quiet else 1.0
    yaw = float(path.yaws[i]) + noise * rng.normal(0.0, 0.01)
    offset = noise * rng.normal(0.0, 0.03)
    x = path.xs[i] - offset * np.sin(yaw) + lr * np.cos(yaw)
    y = path.ys[i] + offset * np.cos(yaw) + lr * np.sin(yaw)
    steer0 = float(np.clip(np.arctan((lf + lr) * path.ks[i]), parameters["s_min"], parameters["s_max"]))
    speed0 = setup["speed"] * (1.0 + noise * rng.normal(0.0, 0.02))
    state = np.array([x, y, steer0, speed0, yaw, speed0 * path.ks[i] + noise * rng.normal(0.0, 0.05), 0.0])
    believed = base[0] * setup["belief"]
    window = 15
    corner = np.sqrt(setup["aggression"] * believed * G / np.maximum(curvature, 1e-3))
    ds = np.diff(path.ss, append=path.ss[-1] + (path.ss[1] - path.ss[0]))
    profile = np.minimum(corner, SPEED_LIMIT_CAP)
    for _ in range(2):
        for k in range(len(profile) - 1, -1, -1):
            profile[k] = min(corner[k], np.sqrt(profile[(k + 1) % len(profile)] ** 2 + 2 * 0.5 * believed * G * ds[k]))
    states, controls, conditions, vehicles, frenet = [state], [], [], [], []
    guess = float(path.ss[i])
    steer_noise, accel_noise = noise * rng.uniform(0.005, 0.02), noise * rng.uniform(0.01, 0.03)
    kick = dict(setup["kick"]) if setup["kick"] else None
    if kick:
        kick["angle"] *= 1.0 + noise * rng.normal(0.0, 0.05)
    outcome = "recovery"
    for step in range(steps):
        rear = state[:2] - lr * np.array([np.cos(state[4]), np.sin(state[4])])
        s, ey, ephi = track.cartesian_to_frenet(rear[0], rear[1], state[4], s_guess=guess)
        guess = float(s)
        s = float(s) % length
        frenet.append([s, float(ey), float(ephi)])
        current = grip_at(setup, s, length)
        conditions.append(current)
        simulator.params = vector(build_vehicle(*current))
        vehicles.append(simulator.params.copy())
        speed = float(state[3])
        target_speed = max(float(np.interp((s + speed * 0.1) % length, path.ss, profile)), 0.0)
        if step < 5 and setup["kind"] == "corner":
            target_speed = max(target_speed, speed0)
        lookahead = float(np.clip(0.5 + 0.18 * speed, 0.6, 2.5))
        target_s = (s + lookahead) % length
        tx, ty = np.interp(target_s, path.ss, path.xs), np.interp(target_s, path.ss, path.ys)
        alpha = np.arctan2(ty - rear[1], tx - rear[0]) - state[4]
        target_steer = float(np.clip(np.arctan2(2 * (lf + lr) * np.sin(alpha), lookahead),
                                     parameters["s_min"], parameters["s_max"]))
        if kick and kick["step"] <= step < kick["step"] + kick["steps"]:
            target_steer = float(np.clip(target_steer + kick["angle"], parameters["s_min"], parameters["s_max"]))
        steering_speed = np.sign(target_steer - state[2]) * min(abs(target_steer - state[2]) / 0.1, parameters["sv_max"])
        steering_speed = float(np.clip(steering_speed + rng.normal(0.0, steer_noise) * parameters["sv_max"],
                                       parameters["sv_min"], parameters["sv_max"]))
        acceleration = float(np.clip(2.0 * (target_speed - speed) + rng.normal(0.0, accel_noise) * parameters["a_max"],
                                     -parameters["a_max"], min(parameters["a_max"], (20.0 - speed) / 0.1)))
        control = np.array([steering_speed, acceleration])
        state = simulator.step(state, control)
        controls.append(control)
        states.append(state)
        if not np.isfinite(state).all() or abs(state[6]) > 1.2 or abs(state[5]) > 10.0 or state[3] < 0:
            outcome = "spin"
            break
        if occupancy.contacts(np.asarray(state)[None])[0]:
            outcome = "contact"
            break
    rear = state[:2] - lr * np.array([np.cos(state[4]), np.sin(state[4])])
    s, ey, ephi = track.cartesian_to_frenet(rear[0], rear[1], state[4], s_guess=guess)
    frenet.append([float(s) % length, float(ey), float(ephi)])
    conditions.append(grip_at(setup, float(s) % length, length))
    vehicles.append(vector(build_vehicle(*conditions[-1])))
    states = np.asarray(states)
    if outcome == "recovery" and np.abs(states[:, 6]).max() > SPIN_SLIP:
        outcome = "spin"
    episode = {"states": states, "controls": np.asarray(controls), "conditions": np.asarray(conditions),
               "vehicles": np.asarray(vehicles), "frenet": np.asarray(frenet)}
    return episode, outcome, simulator, lr


SPEED_LIMIT_CAP = 15.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/f1tenth-track")
    parser.add_argument("--maps", nargs="+", default=list(MAPS))
    parser.add_argument("--groups", type=int, default=120, help="onset groups per map")
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--beams", type=int, default=1080)
    parser.add_argument("--fov", type=float, default=270.0)
    args = parser.parse_args()

    import jax
    from f1tenth_gym.envs.lidar.segment_scan import SegmentScanSimulator2D

    out = Path(args.out)
    manifest = json.loads((out / "manifest.json").read_text())
    known = {e["file"] for e in manifest["episodes"]}
    summary = []
    for name in args.maps:
        global SPEED_LIMIT_CAP
        SPEED_LIMIT_CAP = SPEED_LIMIT[name]
        map_index = list(MAPS).index(name)
        track = Track.from_track_path(map_path(MAPS[name]))
        path = track.centerline if track.centerline is not None else track.raceline
        curvature = smoothed_curvature(path)
        occupancy = Occupancy(name)
        scanner = SegmentScanSimulator2D(args.beams, np.deg2rad(args.fov), std_dev=0.0, min_range=0.05, max_range=30.0,
                                         device="gpu" if any(d.platform == "gpu" for d in jax.devices()) else "cpu")
        scanner.set_map(track)
        batch_scan = jax.jit(lambda poses: jax.lax.map(scanner._scan, poses, batch_size=16))
        for group in range(args.groups):
            group_rng = np.random.default_rng([args.seed, map_index, group])
            setup = group_setup(group_rng, path, curvature, name)
            severity, how = tipping_point(track, path, curvature, setup, args.steps, occupancy)
            setup = with_severity(setup, severity)
            split = "test" if group % 10 == 9 else "val" if group % 10 == 8 else "train"
            outcomes = []
            for r in range(args.rollouts):
                filename = f"{split}/onset-{name}-{group:04d}-{r}.npz"
                rng = np.random.default_rng([args.seed, map_index, group, r])
                episode, outcome, simulator, lr = rollout(track, path, curvature, setup, rng, args.steps, occupancy)
                outcomes.append(outcome)
                if filename in known or len(episode["states"]) < 3:
                    continue
                states = episode["states"]
                yaw = states[:, 4]
                rear = states[:, :2] - lr * np.stack([np.cos(yaw), np.sin(yaw)], 1)
                lidar_xy = rear + LIDAR_OFFSET * np.stack([np.cos(yaw), np.sin(yaw)], 1)
                poses = np.column_stack([lidar_xy, yaw]).astype(np.float32)
                padded = np.concatenate([poses, np.repeat(poses[-1:], args.steps + 1 - len(poses), axis=0)])
                scans = np.asarray(batch_scan(padded))[:len(poses)]
                sensor_parameters = sample_sensor_parameters(rng, None)
                readings = simulate_sensors(states, episode["controls"], simulator.rhs, episode["vehicles"], 0.1, rng,
                                            sensor_parameters)
                names = sorted(sensor_parameters)
                (out / split).mkdir(exist_ok=True)
                np.savez_compressed(out / filename, states=states, controls=episode["controls"],
                                    conditions=episode["conditions"], frenet=episode["frenet"],
                                    lidar=scans.astype(np.float16), sensors=readings,
                                    sensor_parameter_names=np.asarray(names),
                                    sensor_parameter_values=np.asarray([sensor_parameters[n] for n in names]))
                change = {"kind": "patches", "patches": [setup["patch"]], "step": 0} if setup["patch"] else {"kind": "none"}
                manifest["episodes"].append({
                    "file": filename, "map": name, "split": split, "steps": int(len(states) - 1),
                    "reason": {"contact": "crash", "spin": "envelope_limit" if np.abs(states[:, 6]).max() > 1.2
                               else "step_limit", "recovery": "step_limit"}[outcome],
                    "base": setup["base"].tolist(), "change": change,
                    "driver": {"aggression": setup["aggression"], "belief_error": setup["belief"], "onset": True},
                    "max_slip": float(np.abs(states[:, 6]).max()), "max_speed": float(states[:, 3].max()),
                    "onset": {"group": f"{name}-{group:04d}", "kind": setup["kind"], "rollout": r, "outcome": outcome,
                              "start_speed": setup["speed"], "kick": setup["kick"], "severity": severity,
                              "severity_search": how}})
            summary.append({"map": name, "group": group, "kind": setup["kind"], "severity": severity,
                            "severity_search": how, "outcomes": outcomes})
            print(f"{name:9s} group {group:4d} {setup['kind']:6s} start {setup['speed']:5.1f} m/s grip "
                  f"{setup['base'][0]:.2f} severity {severity:.3f} ({how}): " + " ".join(o[0] for o in outcomes),
                  flush=True)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "onset_summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
