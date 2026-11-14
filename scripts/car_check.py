"""Check the car's inputs before driving SIT-LMPC: topic rates, IMU yaw-rate scale and sign against the PF
heading, VESC speed against PF speed, LiDAR, joystick buttons and the car's Frenet pose on the track.
Drive or push the car around (turning both ways) while it runs. It never publishes drive commands.

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

    def odom(msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        records["odom"].append((time.perf_counter(), msg.pose.pose.position.x, msg.pose.pose.position.y, yaw,
                                msg.twist.twist.linear.x, msg.twist.twist.angular.z))

    def speed(msg):
        records["speed"].append((time.perf_counter(), msg.twist.twist.linear.x))

    def imu(msg):
        records["imu"].append((time.perf_counter(), msg.angular_velocity.z))

    def scan(msg):
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
        if records["imu"] and times.size > 10:
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
    if records["scan"]:
        scan_array = np.asarray(records["scan"])
        print(f"closest LiDAR return ahead (±{car.scan_half_angle:.2f} rad): min {scan_array[:, 1].min():.2f} m "
              f"(stop_distance {car.stop_distance} m)")
    if car.joy_topic:
        print(f"joystick buttons pressed: {sorted(buttons)} (go_button {car.go_button})")
    print("OK" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))


if __name__ == "__main__":
    main()
