"""Open-space dynamics data with coverage feedback: target-driven excitation that fills the joint
(speed x steering x slip x yaw rate) space evenly, up to the grip limit.

    python scripts/collect_coverage.py --out data/f1tenth-coverage --episodes 1500

Each episode: a random vehicle (grip 0.4-1.2, front/rear cornering stiffness x0.85-1.15, mass ratio 0.8-1.3, as the
on-track collector; 30% get one step change of grip and/or mass), starting at 1-12 m/s in open space. Driving is a
chain of segments held 0.3-2 s: either a speed/steering target tracked by a simple loop (steering rate towards the
target angle, acceleration towards the target speed), or a full-throttle / full-brake burst with a steering target that stops at a random speed (above / below the
current one).
Coverage feedback: before every segment, --candidates random segments are simulated from the current state and the
one whose visited states fall in the least-filled cells of a running 4-D histogram (speed 0-15 m/s x steering x slip +-0.6 rad x
yaw rate +-8 rad/s, values beyond in the edge cells) is driven; candidates that lose control score x0.25. Speeds up to ~15 m/s. Episodes end at loss of control (|slip| > 1.2 rad or |yaw rate| > 10 rad/s)
or after --steps steps.

Stored in the on-track dataset format (sit_lmpc.world.rssm_data.TrackData) with map "open": LiDAR is constant maximum
range (64 beams) and the manifest marks every episode "lidar": false so LiDAR losses can be masked.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_track_data import build_vehicle, vector

from dyna_dist.adapters.gym_st import GymST
from dyna_dist.data.sensors import SENSOR_FIELDS, sample_sensor_parameters, simulate_sensors

SPEED_EDGES = np.linspace(0.0, 15.0, 11)
STEER_EDGES = np.linspace(-0.42, 0.42, 8)
SLIP_EDGES = np.linspace(-0.6, 0.6, 13)
YAW_EDGES = np.linspace(-8.0, 8.0, 11)
MAX_SPEED = 15.0
LIDAR_RANGE = 30.0


def cells(states):
    return (np.clip(np.digitize(states[:, 3], SPEED_EDGES) - 1, 0, len(SPEED_EDGES) - 2),
            np.clip(np.digitize(states[:, 2], STEER_EDGES) - 1, 0, len(STEER_EDGES) - 2),
            np.clip(np.digitize(states[:, 6], SLIP_EDGES) - 1, 0, len(SLIP_EDGES) - 2),
            np.clip(np.digitize(states[:, 5], YAW_EDGES) - 1, 0, len(YAW_EDGES) - 2))


def random_segment(rng, parameters, speed):
    kind = rng.choice(["target", "throttle", "brake"], p=[0.7, 0.15, 0.15])
    steer = float(rng.uniform(parameters["s_min"], parameters["s_max"]) * rng.choice([1.0, rng.uniform(0, 1)]))
    limit = float(rng.uniform(min(speed, MAX_SPEED), MAX_SPEED)) if kind == "throttle" else float(rng.uniform(0.5, max(speed, 0.6))) \
        if kind == "brake" else float(rng.uniform(0.5, MAX_SPEED))
    return {"kind": str(kind), "speed": limit, "steer": steer,
            "steps": int(rng.integers(3, 21))}


def control(state, segment, parameters):
    delta, speed = state[2], state[3]
    rate = np.clip((segment["steer"] - delta) / 0.1, parameters["sv_min"], parameters["sv_max"])
    if segment["kind"] == "throttle":
        accel = parameters["a_max"] if speed < segment["speed"] else 0.0
    elif segment["kind"] == "brake":
        accel = -parameters["a_max"] if speed > segment["speed"] else 0.0
    else:
        accel = np.clip(2.0 * (segment["speed"] - speed), -parameters["a_max"], parameters["a_max"])
    accel = min(accel, (20.0 - speed) / 0.1)
    return np.array([rate, accel])


def run_segment(simulator, state, segment, parameters):
    states, controls = [], []
    for _ in range(segment["steps"]):
        u = control(state, segment, parameters)
        state = simulator.step(state, u)
        states.append(state)
        controls.append(u)
        if not np.isfinite(state).all() or abs(state[6]) > 1.2 or abs(state[5]) > 10.0 or state[3] < 0:
            return np.asarray(states), np.asarray(controls), True
    return np.asarray(states), np.asarray(controls), False


def episode(rng, histogram, steps, candidates):
    base = np.array([rng.uniform(0.4, 1.2), rng.uniform(0.85, 1.15), rng.uniform(0.85, 1.15), rng.uniform(0.8, 1.3)])
    change = None
    if rng.random() < 0.3:
        after = base.copy()
        what = str(rng.choice(["friction", "mass", "both"]))
        if what in ("friction", "both"):
            after[0] = float(np.clip(base[0] * rng.uniform(0.5, 1.4), 0.3, 1.3))
        if what in ("mass", "both"):
            after[3] = rng.uniform(0.8, 1.3)
        change = {"kind": "step", "what": what, "step": int(rng.integers(steps // 4, 3 * steps // 4)),
                  "after": after.tolist()}
    parameters = build_vehicle(*base)
    simulator = GymST(parameters, 0.1, 0.01)
    state = np.array([0.0, 0.0, 0.0, rng.uniform(1.0, 12.0), rng.uniform(-np.pi, np.pi), 0.0, 0.0])
    states, controls, conditions, vehicles = [state], [], [base.copy()], [vector(parameters)]
    reason = "step_limit"
    while len(controls) < steps:
        current = np.asarray(change["after"]) if change and len(controls) >= change["step"] else base
        live = build_vehicle(*current)
        simulator.params = vector(live)
        best, best_score = None, -1.0
        for _ in range(candidates):
            segment = random_segment(rng, live, state[3])
            trial, _, lost = run_segment(simulator, state.copy(), segment, live)
            if not len(trial):
                continue
            score = np.mean(1.0 / np.sqrt(histogram[cells(trial)] + 1.0)) * (0.25 if lost else 1.0)
            if score > best_score:
                best, best_score = segment, score
        segment = dict(best, steps=min(best["steps"], steps - len(controls)))
        new_states, new_controls, lost = run_segment(simulator, state, segment, live)
        for s, u in zip(new_states, new_controls):
            states.append(s)
            controls.append(u)
            conditions.append(current.copy())
            vehicles.append(simulator.params.copy())
        np.add.at(histogram, cells(new_states), 1.0)
        state = new_states[-1]
        if lost:
            reason = "envelope_limit"
            break
    return ({"states": np.asarray(states), "controls": np.asarray(controls), "conditions": np.asarray(conditions),
             "vehicles": np.asarray(vehicles)}, {"base": base.tolist(), "change": change or {"kind": "none"},
                                                 "reason": reason}, simulator)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/f1tenth-coverage")
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--beams", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261125)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    histogram = np.zeros((len(SPEED_EDGES) - 1, len(STEER_EDGES) - 1, len(SLIP_EDGES) - 1, len(YAW_EDGES) - 1))
    manifest = {"seed": args.seed, "maps": ["open"], "timestep": 0.1, "beams": args.beams, "lidar": False,
                "sensor_fields": list(SENSOR_FIELDS),
                "condition_fields": ["friction", "front_cornering_scale", "rear_cornering_scale", "mass_ratio"],
                "episodes": []}
    for index in range(args.episodes):
        rng = np.random.default_rng([args.seed, index])
        data, meta, simulator = episode(rng, histogram, args.steps, args.candidates)
        count = len(data["states"])
        if count < 20:
            continue
        sensor_parameters = sample_sensor_parameters(rng, None)
        readings = simulate_sensors(data["states"], data["controls"], simulator.rhs, data["vehicles"], 0.1, rng,
                                    sensor_parameters)
        split = "test" if index % 10 == 9 else "val" if index % 10 == 8 else "train"
        filename = f"{split}/open-{index:05d}.npz"
        (out / split).mkdir(exist_ok=True)
        names = sorted(sensor_parameters)
        np.savez_compressed(out / filename, states=data["states"], controls=data["controls"],
                            conditions=data["conditions"], frenet=np.zeros((count, 3)),
                            lidar=np.full((count, args.beams), LIDAR_RANGE, np.float16), sensors=readings,
                            sensor_parameter_names=np.asarray(names),
                            sensor_parameter_values=np.asarray([sensor_parameters[n] for n in names]))
        slip = np.abs(data["states"][:, 6])
        manifest["episodes"].append({"file": filename, "map": "open", "split": split, "steps": int(count - 1),
                                     "reason": meta["reason"], "base": meta["base"], "change": meta["change"],
                                     "lidar": False, "max_slip": float(slip.max()),
                                     "max_speed": float(data["states"][:, 3].max())})
        if index % 100 == 0:
            filled = np.mean(histogram > 0)
            print(f"episode {index:5d}  steps {int(histogram.sum()):7d}  filled cells {filled:.1%}  "
                  f"{meta['reason']}", flush=True)
    np.save(out / "coverage_histogram.npy", histogram)
    episodes = manifest["episodes"]
    manifest["summary"] = {"episodes": len(episodes), "transitions": int(sum(e["steps"] for e in episodes)),
                           "lost_control": int(sum(e["reason"] == "envelope_limit" for e in episodes))}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
