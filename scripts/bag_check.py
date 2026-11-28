"""Offline check of the world model on a recorded car bag: v11 vs the bicycle model, on measured quantities.

    source /opt/ros/jazzy/setup.bash
    python scripts/bag_check.py --bag results/<track>/<run>/bag --config configs/car/levine-v11.yaml

Reads the particle-filter pose, wheel speed, IMU, LiDAR and drive commands from the bag and builds a 10 Hz timeline on
the pose stamps. The world model is filtered along it with the same inputs a car run gives it (RosCar
encoder_observation / lidar_scan). Controls are in the planner's units, rebuilt from the drive commands: steering rate
= change of the steering command / 0.1 s, acceleration = (speed command - measured speed) / simulation.car.speed_horizon.

From every start (every --stride steps after a 3 s warm-up), both models predict 0.5, 1.0 and 1.5 s ahead with the
controls that were actually sent:
- v11: from its filtered state (PF pose + its estimate; steering from the commands when rssm_command_steering);
- bicycle: the nominal single-track model from the measured state (PF pose, steering command, wheel speed, IMU yaw
  rate, kinematic slip).
Starts are skipped when the window has a stop (wheel speed under 0.5 m/s) or a pose jump (car carried back to the
line); the world model's filter restarts after every jump, as in a run. Errors are against what was measured: position against the PF pose (centre of gravity), yaw rate against the IMU,
speed against the wheel speed. Prints median / 90th percentile per horizon and writes JSON next to the bag.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
HORIZONS = (5, 10, 15)


def read_bag(path, topics):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id=""),
                rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    missing = [t for t in topics.values() if t not in types]
    if missing:
        raise SystemExit(f"topics missing from the bag: {missing}; recorded: {sorted(types)}")
    wanted = {v: k for k, v in topics.items()}
    out = {k: [] for k in topics}
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topic in wanted:
            out[wanted[topic]].append((stamp * 1e-9, deserialize_message(data, get_message(types[topic]))))
    return out


def latest(series, times, t):
    i = int(np.searchsorted(times, t, side="right")) - 1
    return series[max(i, 0)][1]


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def imu_arrays(messages, car):
    from sit_lmpc.adapters.imu_frame import axis_matrix, level_rotation
    matrix = axis_matrix(car.imu_axes)
    times = np.asarray([t for t, _ in messages["imu"]])
    accel = np.asarray([[m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z]
                        for _, m in messages["imu"]]) @ matrix.T
    gyro = np.asarray([[m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z]
                       for _, m in messages["imu"]]) @ matrix.T
    if car.imu_level:
        speed_times = np.asarray([t for t, _ in messages["speed"]])
        speeds = np.asarray([m.twist.twist.linear.x for _, m in messages["speed"]])
        still = np.abs(np.interp(times, speed_times, speeds)) < 0.05
        rotation = level_rotation(accel[still].mean(axis=0) if still.sum() > 20 else accel[:200].mean(axis=0))
        accel, gyro = accel @ rotation.T, gyro @ rotation.T
    return times, accel, gyro


def timeline(messages, car, timestep, lr):
    pose_times = np.asarray([t for t, _ in messages["pose"]])
    imu_times, imu_accel, imu_gyro = imu_arrays(messages, car)
    ticks = np.arange(pose_times[0] + 0.5, pose_times[-1], timestep)
    index = {k: np.asarray([t for t, _ in v]) for k, v in messages.items()}
    rows = []
    previous_pose, previous_steer = None, None
    for t in ticks:
        pose = latest(messages["pose"], index["pose"], t)
        inside = (imu_times > t - timestep) & (imu_times <= t)
        last = max(int(np.searchsorted(imu_times, t, side="right")) - 1, 0)
        accel = imu_accel[inside].mean(axis=0) if inside.any() else imu_accel[last]
        gyro = imu_gyro[inside].mean(axis=0) if inside.any() else imu_gyro[last]
        wheel = latest(messages["speed"], index["speed"], t)
        scan = latest(messages["scan"], index["scan"], t)
        drive = latest(messages["drive"], index["drive"], t)
        rear = (pose.pose.pose.position.x, pose.pose.pose.position.y, yaw_of(pose.pose.pose.orientation))
        previous = rear if previous_pose is None else previous_pose
        dx, dy = rear[0] - previous[0], rear[1] - previous[1]
        body = (math.cos(previous[2]) * dx + math.sin(previous[2]) * dy,
                -math.sin(previous[2]) * dx + math.cos(previous[2]) * dy, wrap(rear[2] - previous[2]))
        steer = float(drive.drive.steering_angle)
        speed = float(wheel.twist.twist.linear.x)
        control = (0.0 if previous_steer is None else (steer - previous_steer) / timestep,
                   (float(drive.drive.speed) - speed) / car.speed_horizon)
        yaw_rate = car.imu_yaw_scale * gyro[2]
        jump = previous_pose is not None and math.hypot(dx, dy) > 3 * max(speed, 1.0) * timestep + 0.3
        rows.append({
            "t": t, "rear": rear, "jump": jump, "steer": steer, "speed": speed, "yaw_rate": yaw_rate, "control": control,
            "observation": np.array([car.imu_accel_scale[0] * accel[0], car.imu_accel_scale[1] * accel[1], yaw_rate, speed,
                                     body[0] / timestep, body[1] / timestep, body[2] / timestep, 0.0, 0.0], np.float32),
            "scan": (scan.ranges, scan.angle_min, scan.angle_increment),
            "cog": (rear[0] + lr * math.cos(rear[2]), rear[1] + lr * math.sin(rear[2])),
        })
        previous_pose, previous_steer = rear, steer
    if car.imu_position is not None:
        from sit_lmpc.adapters.imu_frame import to_center_of_gravity
        for i in range(len(rows)):
            rate = rows[i]["yaw_rate"]
            rate_dot = 0.0 if i == 0 or rows[i]["jump"] else (rate - rows[i - 1]["yaw_rate"]) / timestep
            observation = rows[i]["observation"]
            observation[0], observation[1] = to_center_of_gravity(observation[0], observation[1], rate, rate_dot,
                                                                  car.imu_position[0] - lr, car.imu_position[1])
    for i in range(1, len(rows)):
        rows[i]["observation"][7:9] = rows[i - 1]["control"]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bag", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs/car/levine-v11.yaml"))
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--warmup", type=float, default=3.0, help="seconds of filtering before the first start")
    for name, default in (("pose", None), ("speed", None), ("imu", None), ("scan", None), ("drive", None)):
        parser.add_argument(f"--{name}-topic", default=default)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import jax.numpy as jnp
    from sit_lmpc.config import load_config
    from sit_lmpc.dynamics.st import rollout as st_rollout
    from sit_lmpc.experiments.runner import build_vehicle_dynamics
    from sit_lmpc.world.rssm_runtime import RSSMFilter, resample_scan

    config = load_config(args.config)
    car, planner, simulation = config.simulation.car, config.planner, config.simulation
    params, bicycle = build_vehicle_dynamics(config)
    lr, wheelbase = params["lr"], params["lf"] + params["lr"]
    topics = {"pose": args.pose_topic or car.odom_topic, "speed": args.speed_topic or car.speed_topic,
              "imu": args.imu_topic or car.imu_topic, "scan": args.scan_topic or car.scan_topic,
              "drive": args.drive_topic or car.drive_topic}
    messages = read_bag(args.bag, topics)
    rows = timeline(messages, car, simulation.timestep, lr)
    scale = np.asarray(simulation.action_scale, float)
    rssm = RSSMFilter(planner.dynamics.rssm, bicycle.params, bicycle.action_scale, planner.dynamics.rssm_variant,
                      planner.dynamics.rssm_model_timestep, bicycle.integrator_timestep)
    filtered = []
    since_jump = []
    count = 0
    for row in rows:
        if row["jump"]:
            rssm.reset()
            count = 0
        count += 1
        since_jump.append(count)
        anchor, _, _ = rssm.step(row["observation"], resample_scan(*row["scan"]), row["rear"],
                                 np.asarray(row["observation"][7:9]))
        filtered.append((rssm.predictor(), np.asarray(anchor)))
    errors = {model: {h: {"position_m": [], "yaw_rate": [], "speed": []} for h in HORIZONS} for model in ("v11", "bicycle")}
    first = int(round(args.warmup / simulation.timestep))
    for i in range(first, len(rows) - max(HORIZONS) - 1, args.stride):
        row = rows[i]
        window = rows[i:i + max(HORIZONS) + 1]
        if since_jump[i] < first or any(r["jump"] or r["speed"] < 0.5 for r in window):
            continue
        controls = np.clip(np.asarray([r["control"] for r in rows[i + 1:i + 1 + max(HORIZONS)]]), -scale, scale)
        actions = controls / scale
        yaw = row["rear"][2]
        slip = math.atan(lr / wheelbase * math.tan(row["steer"]))
        measured = np.array([row["cog"][0], row["cog"][1], row["steer"], row["speed"], yaw, row["yaw_rate"], slip])
        predictor, anchor = filtered[i]
        start = measured.copy()
        if planner.dynamics.rssm_state:
            fields = [3, 5, 6] if planner.dynamics.rssm_command_steering else [2, 3, 5, 6]
            start[fields] = anchor[1:] if planner.dynamics.rssm_command_steering else anchor
        v11 = np.asarray(predictor.rollout_members(jnp.asarray(start, jnp.float32), jnp.asarray(actions, jnp.float32))[0])
        bike = np.asarray(st_rollout(bicycle, measured.astype(simulation.dtype), actions.astype(simulation.dtype)))
        for h in HORIZONS:
            truth = rows[i + h]
            for model, path in (("v11", v11), ("bicycle", bike)):
                state = path[h - 1]
                errors[model][h]["position_m"].append(math.hypot(state[0] - truth["cog"][0], state[1] - truth["cog"][1]))
                errors[model][h]["yaw_rate"].append(abs(state[5] - truth["yaw_rate"]))
                errors[model][h]["speed"].append(abs(state[3] - truth["speed"]))
    report = {"bag": str(args.bag), "config": args.config, "starts": len(errors["v11"][HORIZONS[0]]["position_m"]),
              "seconds": round(rows[-1]["t"] - rows[0]["t"], 1), "errors": {}}
    print(f"{report['starts']} starts over {report['seconds']} s")
    print(f"{'horizon':>8s} {'model':>8s} {'position m (median / p90)':>27s} {'yaw rate rad/s':>17s} {'speed m/s':>15s}")
    for h in HORIZONS:
        for model in ("v11", "bicycle"):
            e = errors[model][h]
            summary = {k: (round(float(np.median(v)), 3), round(float(np.percentile(v, 90)), 3)) for k, v in e.items() if v}
            report["errors"][f"{model}_{h * simulation.timestep:.1f}s"] = summary
            if summary:
                print(f"{h * simulation.timestep:7.1f}s {model:>8s} {summary['position_m'][0]:12.3f} / {summary['position_m'][1]:.3f}"
                      f" {summary['yaw_rate'][0]:9.3f} / {summary['yaw_rate'][1]:.3f} {summary['speed'][0]:7.3f} / {summary['speed'][1]:.3f}")
    out = Path(args.out) if args.out else Path(args.bag).with_name("bag_check.json")
    out.write_text(json.dumps(report, indent=1) + "\n")
    print("written", out)


if __name__ == "__main__":
    main()
