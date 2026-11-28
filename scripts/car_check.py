"""Check the car's inputs before driving SIT-LMPC: topic rates, IMU yaw-rate scale and sign against the PF
heading, VESC speed against PF speed, LiDAR, joystick buttons and the car's Frenet pose on the track.
For world-model configs (planner.dynamics.backend rssm) also v11's inputs: IMU acceleration units, gravity and signs
(suggests simulation.car.imu_accel_scale), the LiDAR layout against training (1080 beams over 270 degrees), and the
9 encoder inputs against the training statistics. Leave the car still for the first 3 s, then drive or push it
around (accelerating, braking, turning both ways). It never publishes drive commands.

    python scripts/car_check.py --config configs/car-levine-known.yaml --seconds 20
"""

import argparse
import math
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/car-levine-known.yaml")
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args()

    import rclpy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu, Joy, LaserScan
    from sit_lmpc.adapters import vehicle_parameters_for
    from sit_lmpc.config.schema import load_config
    from sit_lmpc.track.resources import map_path
    from sit_lmpc.track.track import Track

    config = load_config(args.config)
    car = config.simulation.car
    track = Track.from_track_path(map_path(config.simulation.map))
    lr = vehicle_parameters_for(config.simulation, config.simulation.friction)["lr"]
    rclpy.init()
    node = rclpy.create_node("sit_lmpc_car_check")
    records = {"odom": [], "speed": [], "imu": [], "scan": [], "joy": []}
    buttons = set()
    layout = {}

    def odom(msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        records["odom"].append((time.perf_counter(), msg.pose.pose.position.x, msg.pose.pose.position.y, yaw,
                                msg.twist.twist.linear.x, msg.twist.twist.angular.z))

    def speed(msg):
        records["speed"].append((time.perf_counter(), msg.twist.twist.linear.x))

    from sit_lmpc.adapters.imu_frame import axis_matrix, level_rotation, tilt_degrees
    imu_matrix = axis_matrix(car.imu_axes)

    def imu(msg):
        a = msg.linear_acceleration
        w = msg.angular_velocity
        accel = imu_matrix @ np.array([a.x, a.y, a.z])
        gyro = imu_matrix @ np.array([w.x, w.y, w.z])
        records["imu"].append((time.perf_counter(), gyro[2], accel[0], accel[1], accel[2], gyro[0], gyro[1]))

    def scan(msg):
        layout.update(count=len(msg.ranges), angle_min=msg.angle_min, angle_max=msg.angle_max,
                      range_max=msg.range_max)
        ranges = np.asarray(msg.ranges, dtype=float)
        angles = msg.angle_min + msg.angle_increment * np.arange(ranges.size)
        valid = (np.abs(angles) < car.scan_half_angle) & np.isfinite(ranges) & (ranges > 0.05)
        records["scan"].append((time.perf_counter(), float(ranges[valid].min()) if valid.any() else math.inf))

    def joy(msg):
        records["joy"].append((time.perf_counter(),))
        for index, value in enumerate(msg.buttons):
            if value:
                buttons.add(index)

    node.create_subscription(Odometry, car.odom_topic, odom, 10)
    if car.speed_topic:
        node.create_subscription(Odometry, car.speed_topic, speed, 10)
    if car.imu_topic:
        node.create_subscription(Imu, car.imu_topic, imu, 10)
    if car.scan_topic:
        node.create_subscription(LaserScan, car.scan_topic, scan, 10)
    if car.joy_topic:
        node.create_subscription(Joy, car.joy_topic, joy, 10)
    print(f"listening for {args.seconds:.0f} s; drive or push the car through left and right turns")
    end = time.perf_counter() + args.seconds
    while time.perf_counter() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    rclpy.shutdown()
    if records["imu"] and car.imu_level and np.any(np.asarray(records["imu"])[:, 1:]):
        raw = np.asarray(records["imu"])
        still = raw[:, 0] < raw[0, 0] + 3.0
        gravity = raw[still, 2:5].mean(axis=0)
        rotation = level_rotation(gravity)
        accel = raw[:, 2:5] @ rotation.T
        gyro = raw[:, [5, 6, 1]] @ rotation.T
        records["imu"] = [tuple(row) for row in np.column_stack([raw[:, 0], gyro[:, 2], accel, gyro[:, 0], gyro[:, 1]])]
        print(f"IMU levelled (imu_level): mount tilt {tilt_degrees(gravity):.1f} deg from the first 3 s (car still)")

    names = {"odom": car.odom_topic, "speed": car.speed_topic, "imu": car.imu_topic, "scan": car.scan_topic,
             "joy": car.joy_topic}
    problems = []
    for key, topic in names.items():
        if not topic:
            continue
        count = len(records[key])
        rate = count / args.seconds
        print(f"{topic:24s} {count:6d} msgs  {rate:6.1f} Hz")
        if count == 0 and key != "joy":
            problems.append(f"no messages on {topic}")
    if records["odom"]:
        odom_array = np.asarray(records["odom"])
        gaps = np.diff(odom_array[:, 0])
        if gaps.size:
            print(f"PF odom largest gap {gaps.max() * 1000:.0f} ms (odom_timeout {car.odom_timeout * 1000:.0f} ms)")
            if gaps.max() > car.odom_timeout:
                problems.append("PF odom gaps exceed simulation.car.odom_timeout")
        last = odom_array[-1]
        s, ey, ephi = track.cartesian_to_frenet(last[1] + lr * math.cos(last[3]),
                                                last[2] + lr * math.sin(last[3]), last[3], use_s_guess=False)
        length = float(track.s_frame_max)
        print(f"car on track: s {s:.2f} / {length:.2f} m ({length - s:.2f} m behind the start line), "
              f"ey {ey:+.2f} m, heading error {ephi:+.2f} rad (planner.half_width {config.planner.half_width})")
        if abs(ey) > config.planner.half_width:
            problems.append("the car is outside planner.half_width; wrong map/track or frame?")
        if abs(ephi) > math.pi / 2:
            problems.append("the car faces against the track direction")
        yaw = np.unwrap(odom_array[:, 3])
        times = odom_array[:, 0]
        imu_dead = bool(records["imu"]) and not np.any(np.asarray(records["imu"])[:, 1:])
        if imu_dead:
            problems.append(f"the IMU ({car.imu_topic}) reports only zeros: enable the VESC IMU or set imu_topic to the "
                            "topic that has data; v11 and the yaw rate need it")
        if records["imu"] and times.size > 10 and not imu_dead:
            pf_rate = np.gradient(yaw, times)
            imu_array = np.asarray(records["imu"])
            imu_rate = np.interp(times, imu_array[:, 0], imu_array[:, 1])
            moving = np.abs(pf_rate) > 0.2
            if moving.sum() > 10:
                scale = float(np.dot(imu_rate[moving], pf_rate[moving]) / np.dot(imu_rate[moving], imu_rate[moving]))
                residual = float(np.sqrt(np.mean((scale * imu_rate[moving] - pf_rate[moving]) ** 2)))
                print(f"IMU yaw rate vs PF heading rate: suggested imu_yaw_scale {scale:+.4f} "
                      f"(config {car.imu_yaw_scale:+.4f}), rms error after scaling {residual:.3f} rad/s")
                if abs(scale - car.imu_yaw_scale) > 0.2 * abs(scale):
                    problems.append(f"set simulation.car.imu_yaw_scale to about {scale:+.3f}")
            else:
                print("IMU check skipped: the car did not turn (|PF yaw rate| > 0.2 rad/s for < 10 samples)")
            still = np.abs(imu_array[:, 1]) < 0.5
            if still.any():
                print(f"IMU gyro z bias while near still: {imu_array[still, 1].mean():+.4f} (raw units)")
        if records["speed"]:
            speed_array = np.asarray(records["speed"])
            wheel = np.interp(times, speed_array[:, 0], speed_array[:, 1])
            moving = np.abs(odom_array[:, 4]) > 0.3
            if moving.sum() > 10:
                ratio = float(np.median(wheel[moving] / odom_array[moving, 4]))
                print(f"VESC speed / PF speed median ratio {ratio:.3f}")
                if not 0.85 < ratio < 1.15:
                    problems.append("VESC speed disagrees with PF speed; check speed_to_erpm_gain or use speed_topic \"\"")
    if records["imu"] and records["speed"] and records["odom"] and np.any(np.asarray(records["imu"])[:, 1:]):
        imu_array = np.asarray(records["imu"])
        still = imu_array[:, 0] < imu_array[0, 0] + 3.0
        gravity = imu_array[still, 2:5].mean(axis=0) if still.any() else imu_array[:, 2:5].mean(axis=0)
        print(f"IMU acceleration while still (x, y, z): {gravity[0]:+.3f} {gravity[1]:+.3f} {gravity[2]:+.3f} (raw units)")
        unit = 1.0
        if 0.7 < abs(gravity[2]) < 1.3:
            unit = 9.81
            problems.append("IMU acceleration looks like g units (|z| about 1 at rest); multiply imu_accel_scale by 9.81")
        elif not 8.0 < abs(gravity[2]) < 11.5:
            problems.append(f"IMU z acceleration at rest is {gravity[2]:+.2f}; expected about 9.81 m/s^2 or 1 g "
                            "(check simulation.car.imu_axes: which IMU axis points up)")
        speed_array = np.asarray(records["speed"])
        t = imu_array[:, 0]
        v = np.interp(t, speed_array[:, 0], speed_array[:, 1])
        dvdt = np.gradient(v, t)
        odom_array = np.asarray(records["odom"])
        r = np.interp(t, odom_array[:, 0], np.gradient(np.unwrap(odom_array[:, 3]), odom_array[:, 0]))
        lateral = v * r
        ax = (imu_array[:, 2] - gravity[0]) * unit
        ay = (imu_array[:, 3] - gravity[1]) * unit
        scale = []
        for measured, reference, name in ((ax, dvdt, "x"), (ay, lateral, "y")):
            active = np.abs(reference) > 0.5
            if active.sum() > 20:
                k = float(np.dot(measured[active], reference[active]) / np.dot(measured[active], measured[active]))
                scale.append(round(k * unit, 3))
                print(f"IMU {name}: reference / IMU slope {k:+.3f} (correlation "
                      f"{np.corrcoef(measured[active], reference[active])[0, 1]:+.2f})")
            else:
                scale.append(None)
                print(f"IMU {name}: not enough {'acceleration' if name == 'x' else 'cornering'} to check")
        print(f"suggested simulation.car.imu_accel_scale: {scale} (config {list(car.imu_accel_scale)}); "
              "a value near +1/-1 (or +-9.81 for g units) only fixes the sign and units, keep it at exactly that")
        for value, current, name in zip(scale, car.imu_accel_scale, "xy"):
            if value is not None and np.sign(value) != np.sign(current):
                problems.append(f"IMU {name} acceleration sign looks flipped; set imu_accel_scale {name} to {np.sign(value) * unit:+.2f}")
    if layout:
        span = math.degrees(layout["angle_max"] - layout["angle_min"])
        print(f"LiDAR: {layout['count']} beams over {span:.0f} deg ({math.degrees(layout['angle_min']):+.0f} to "
              f"{math.degrees(layout['angle_max']):+.0f}), range max {layout['range_max']:.1f} m; training layout "
              f"1080 beams over 270 deg (-135 to +135), resampled by nearest beam")
        if span < 240:
            problems.append("the LiDAR covers less than 240 degrees; the missing angles read as 30 m for the world model")
    if config.planner.dynamics.backend == "rssm" and records["imu"] and records["speed"] and len(records["odom"]) > 20:
        from pathlib import Path
        scale = np.load(Path(config.planner.dynamics.rssm) / "observation_scale.npz")
        odom_array = np.asarray(records["odom"])
        dt = config.simulation.timestep
        idx = np.arange(0, len(odom_array), max(1, int(round(dt / max(np.median(np.diff(odom_array[:, 0])), 1e-3)))))
        pose = odom_array[idx]
        imu_array = np.asarray(records["imu"])
        speed_array = np.asarray(records["speed"])
        rows = []
        for a, b in zip(pose[:-1], pose[1:]):
            dx, dy = b[1] - a[1], b[2] - a[2]
            body = (math.cos(a[3]) * dx + math.sin(a[3]) * dy, -math.sin(a[3]) * dx + math.cos(a[3]) * dy)
            inside = (imu_array[:, 0] > b[0] - dt) & (imu_array[:, 0] <= b[0])
            j = min(np.searchsorted(imu_array[:, 0], b[0]), len(imu_array) - 1)
            sample = imu_array[inside].mean(axis=0) if inside.any() else imu_array[j]
            w = np.interp(b[0], speed_array[:, 0], speed_array[:, 1])
            rows.append([car.imu_accel_scale[0] * sample[2], car.imu_accel_scale[1] * sample[3],
                         car.imu_yaw_scale * sample[1], w, body[0] / dt, body[1] / dt,
                         (b[3] - a[3] + math.pi) % (2 * math.pi) - math.pi])
        rows = np.asarray(rows)
        if car.imu_position is not None:
            from sit_lmpc.adapters.imu_frame import to_center_of_gravity
            rate_dot = np.concatenate([[0.0], np.diff(rows[:, 2]) / dt])
            rows[:, 0], rows[:, 1] = to_center_of_gravity(rows[:, 0], rows[:, 1], rows[:, 2], rate_dot,
                                                          car.imu_position[0] - lr, car.imu_position[1])
        names = ["imu ax", "imu ay", "gyro z", "wheel speed", "PF dx/dt", "PF dy/dt", "PF dyaw/dt"]
        print("world-model inputs (car) vs training: mean / std")
        for k, name in enumerate(names):
            z = (rows[:, k].mean() - float(scale["mean"][k])) / max(float(scale["std"][k]), 1e-6)
            print(f"  {name:12s} car {rows[:, k].mean():+7.3f} / {rows[:, k].std():6.3f}   training "
                  f"{float(scale['mean'][k]):+7.3f} / {float(scale['std'][k]):6.3f}" + ("   <- far off" if abs(z) > 3 else ""))
            if abs(z) > 3:
                problems.append(f"world-model input '{name}' is {z:+.1f} training std away; check units, sign and frame")
    if records["scan"]:
        scan_array = np.asarray(records["scan"])
        print(f"closest LiDAR return ahead (±{car.scan_half_angle:.2f} rad): min {scan_array[:, 1].min():.2f} m "
              f"(stop_distance {car.stop_distance} m)")
    if car.joy_topic:
        print(f"joystick buttons pressed: {sorted(buttons)} (go_button {car.go_button})")
    print("OK" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))


if __name__ == "__main__":
    main()
