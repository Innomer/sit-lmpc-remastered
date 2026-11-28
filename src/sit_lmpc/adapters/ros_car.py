"""Real F1TENTH car boundary: PF pose, VESC speed and IMU yaw rate in, speed and steering angle out, in real time."""

import json
import math
import signal
import subprocess
import threading
import time
from collections import deque
import numpy as np

from sit_lmpc.adapters.imu_frame import axis_matrix, level_rotation, tilt_degrees, to_center_of_gravity
from sit_lmpc.adapters.ros_f1tenth import percentile_summary, wrap_angle


def quaternion_yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class RosCar:
    restore_tolerance = 1e9

    def __init__(self, simulation, track, parameters, half_width, emit=None, directory=None):
        import rclpy
        from ackermann_msgs.msg import AckermannDriveStamped
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Imu, Joy, LaserScan
        from std_msgs.msg import Empty

        self.simulation, self.track, self.params = simulation, track, parameters
        self.car = simulation.car
        self.half_width = half_width
        self.emit = emit if emit is not None else (lambda *args, **kwargs: None)
        self.directory = directory
        self.dtype = simulation.dtype
        self.timestep = simulation.timestep
        self.action_scale = np.asarray(simulation.action_scale, dtype=float)
        self.wheelbase = parameters["lf"] + parameters["lr"]
        self.delay_steps = round(simulation.ros.actuation_delay / simulation.timestep)
        self.pending = deque(maxlen=max(self.delay_steps, 1))
        self.delta_history = deque(maxlen=self.delay_steps + 1)
        self.predictor = None
        if self.delay_steps:
            from sit_lmpc.dynamics.st import STModel

            from sit_lmpc.dynamics.st import rollout

            self.predictor = STModel.from_vehicle(parameters, simulation)
            rollout(self.predictor, np.zeros(7, dtype=self.dtype), np.zeros((self.delay_steps, 2), dtype=self.dtype))
        self.rclpy = rclpy
        from rclpy.signals import SignalHandlerOptions

        self.spinning = False
        self.interrupt_pending = False
        signal.signal(signal.SIGINT, self._interrupt)
        signal.signal(signal.SIGTERM, self._interrupt)
        if not rclpy.ok():
            rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
        self.node = rclpy.create_node("sit_lmpc_car")
        self.drive_type = AckermannDriveStamped
        self.odom = None
        self.odom_time = None
        self.odom_count = 0
        self.speed = None
        self.yaw_rate = None
        self.odom_yaw_rate = 0.0
        self.close_scan = False
        self.closest = math.inf
        self.imu = None
        self.imu_matrix = axis_matrix(simulation.car.imu_axes)
        self.imu_rotation = np.eye(3)
        self.imu_window = deque()
        self.imu_accel = np.zeros(3)
        self.imu_gyro = np.zeros(3)
        self.imu_leveled = not simulation.car.imu_level
        self.imu_previous_rate = None
        self.scan = None
        self.encoder_pose = None
        self.applied_control = np.zeros(2)
        self.go = False
        self.joy_pressed = False
        self.delta = 0.0
        self.delta_target = 0.0
        self.delta_time = None
        self.speed_target = 0.0
        self.yaw = None
        self.tick = None
        self.started = None
        self.launch_speed = None
        self.returned_at = None
        self.steps = 0
        self.overruns = 0
        self.watchdog_stops = 0
        self.watchdog_active = False
        self.substep_policy = None
        self.plan_times, self.step_times = [], []
        self.window_plan, self.window_step = [], []
        self.log = []
        self.last_command = None
        self.publish_error = None
        self.lock = threading.Lock()
        self.running = True
        car = self.car
        self.subscriptions = [self.node.create_subscription(Odometry, car.odom_topic, self._odom_callback, 10)]
        if car.speed_topic:
            self.subscriptions.append(
                self.node.create_subscription(Odometry, car.speed_topic, self._speed_callback, 10))
        if car.imu_topic:
            self.subscriptions.append(self.node.create_subscription(Imu, car.imu_topic, self._imu_callback, 10))
        if car.scan_topic and (car.stop_distance > 0 or car.world_model_inputs):
            self.subscriptions.append(
                self.node.create_subscription(LaserScan, car.scan_topic, self._scan_callback, 10))
        if car.joy_topic:
            self.subscriptions.append(self.node.create_subscription(Joy, car.joy_topic, self._joy_callback, 10))
        if car.go_topic:
            self.subscriptions.append(self.node.create_subscription(Empty, car.go_topic, self._go_callback, 10))
        self.drive_pub = self.node.create_publisher(AckermannDriveStamped, car.drive_topic, 10)
        self.bag = self._start_bag()
        self.watchdog = threading.Thread(target=self._watchdog, daemon=True)
        self.watchdog.start()

    def _start_bag(self):
        if not self.car.record_bag or self.directory is None:
            return None
        try:
            process = subprocess.Popen(
                ["ros2", "bag", "record", "-o", str(self.directory / "bag"), *self.car.bag_topics],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            self.emit("car_bag_unavailable", reason="ros2 CLI not found; source ROS before running")
            return None
        self.emit("car_bag_recording", directory=str(self.directory / "bag"), topics=list(self.car.bag_topics))
        return process

    def _odom_callback(self, msg):
        self.odom = msg
        self.odom_time = time.perf_counter()
        self.odom_count += 1
        self.odom_yaw_rate = msg.twist.twist.angular.z
        if not self.car.speed_topic:
            self.speed = msg.twist.twist.linear.x
        if not self.car.imu_topic:
            self.yaw_rate = msg.twist.twist.angular.z

    def _speed_callback(self, msg):
        self.speed = msg.twist.twist.linear.x

    def _imu_callback(self, msg):
        self.imu = msg
        now = time.perf_counter()
        a = msg.linear_acceleration
        w = msg.angular_velocity
        self.imu_window.append((now, self.imu_matrix @ np.array([a.x, a.y, a.z]),
                                self.imu_matrix @ np.array([w.x, w.y, w.z])))
        while self.imu_window and self.imu_window[0][0] < now - 1.0:
            self.imu_window.popleft()
        recent = [entry for entry in self.imu_window if entry[0] >= now - self.timestep]
        self.imu_accel = self.imu_rotation @ np.mean([entry[1] for entry in recent], axis=0)
        self.imu_gyro = self.imu_rotation @ np.mean([entry[2] for entry in recent], axis=0)
        self.yaw_rate = self.car.imu_yaw_scale * self.imu_gyro[2]

    def _level_imu(self):
        if self.imu_leveled or len(self.imu_window) < 20:
            return
        gravity = np.mean([entry[1] for entry in self.imu_window], axis=0)
        self.imu_leveled = True
        if not 5.0 < np.linalg.norm(gravity) < 15.0:
            self.emit("imu_level_skipped", gravity=[float(v) for v in gravity],
                      message="no gravity in the IMU acceleration (expected about 9.81 m/s^2); IMU not levelled")
            return
        self.imu_rotation = level_rotation(gravity)
        self.emit("imu_leveled", tilt_degrees=tilt_degrees(gravity), gravity=[float(v) for v in gravity])

    def _scan_callback(self, msg):
        self.scan = msg
        if self.car.stop_distance <= 0:
            return
        ranges = np.asarray(msg.ranges, dtype=float)
        angles = msg.angle_min + msg.angle_increment * np.arange(ranges.size)
        valid = (np.abs(angles) < self.car.scan_half_angle) & np.isfinite(ranges) & (ranges > 0.05)
        if not valid.any():
            return
        self.closest = float(ranges[valid].min())
        if self.closest < self.car.stop_distance:
            self.close_scan = True

    def _joy_callback(self, msg):
        pressed = len(msg.buttons) > self.car.go_button and msg.buttons[self.car.go_button] == 1
        if pressed and not self.joy_pressed:
            self.go = True
        self.joy_pressed = pressed

    def _go_callback(self, msg):
        self.go = True

    def _interrupt(self, signum, frame):
        if self.spinning:
            self.interrupt_pending = True
            return
        raise KeyboardInterrupt

    def _spin_guarded(self, timeout):
        self.spinning = True
        try:
            self.rclpy.spin_once(self.node, timeout_sec=timeout)
        except Exception:
            if not self.interrupt_pending:
                raise
        finally:
            self.spinning = False
        if self.interrupt_pending:
            self.interrupt_pending = False
            raise KeyboardInterrupt

    def _spin(self, timeout):
        self._spin_guarded(timeout)

    def _publish(self, steering, speed):
        msg = self.drive_type()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.drive.steering_angle = float(steering)
        msg.drive.speed = float(speed)
        with self.lock:
            try:
                self.drive_pub.publish(msg)
            except Exception as error:
                self.publish_error = str(error)
            self.last_command = time.perf_counter()

    def _watchdog(self):
        while self.running:
            time.sleep(0.05)
            last = self.last_command
            if last is None or time.perf_counter() - last < self.car.command_timeout:
                continue
            if not self.watchdog_active and self.speed_target > 0:
                self.watchdog_stops += 1
            self.watchdog_active = True
            self.speed_target = 0.0
            self._publish(self.delta_target, 0.0)

    def _advance_delta(self):
        now = time.perf_counter()
        if self.delta_time is not None:
            change = self.params["sv_max"] * (now - self.delta_time)
            self.delta += min(max(self.delta_target - self.delta, -change), change)
        self.delta_time = now

    def _drive(self, steering, speed):
        self._advance_delta()
        self.delta_target = min(max(float(steering), self.params["s_min"]), self.params["s_max"])
        self.speed_target = min(max(float(speed), 0.0), self.car.max_speed)
        self.watchdog_active = False
        self._publish(self.delta_target, self.speed_target)

    def _stop(self, reason=None):
        self._drive(self.delta_target, 0.0)
        if reason is not None:
            self.emit("car_stop", reason=reason, speed=self.speed, closest_scan=self.closest)

    def _missing_topics(self):
        missing = []
        if self.odom is None:
            missing.append(self.car.odom_topic)
        if self.speed is None:
            missing.append(self.car.speed_topic or self.car.odom_topic)
        if self.yaw_rate is None:
            missing.append(self.car.imu_topic or self.car.odom_topic)
        if self.drive_pub.get_subscription_count() == 0:
            missing.append(f"{self.car.drive_topic} (no subscriber)")
        return missing

    def _wait_for_topics(self):
        deadline = time.perf_counter() + self.car.startup_timeout
        while self._missing_topics():
            if time.perf_counter() > deadline:
                raise RuntimeError(f"car topics missing after {self.car.startup_timeout} s: "
                                   + ", ".join(self._missing_topics()))
            self._stop()
            self._spin(0.05)

    def _drain(self):
        for _ in range(50):
            self._spin_guarded(0.0)

    def _check_odom(self):
        self._drain()
        age = time.perf_counter() - self.odom_time
        if age > self.car.odom_timeout:
            self._stop("stale_odometry")
            raise RuntimeError(f"odometry from {self.car.odom_topic} is {age:.2f} s old; stopped the car")

    def encoder_observation(self):
        """The world model's 9 sensor inputs from the car, in the layout and units of the training data
        (sit_lmpc.world.data, as RosSimulator.encoder_observation): IMU acceleration x, y (m/s^2, body frame, gravity
        removed) and gyro yaw rate (rad/s), wheel speed (m/s), particle-filter pose change in the previous body frame
        divided by the timestep (m/s, m/s, rad/s), and the last control in planner units (steering rate rad/s,
        acceleration m/s^2). Signs and units come from simulation.car (scripts/car_check.py suggests them)."""
        if self.imu is None or self.odom is None or self.speed is None:
            raise RuntimeError("world-model inputs missing: no IMU, odometry or wheel speed yet")
        car = self.car
        accel = self.imu_accel
        msg = self.odom
        current = (msg.pose.pose.position.x, msg.pose.pose.position.y, quaternion_yaw(msg.pose.pose.orientation))
        previous = current if self.encoder_pose is None else self.encoder_pose
        dx, dy = current[0] - previous[0], current[1] - previous[1]
        body_dx = math.cos(previous[2]) * dx + math.sin(previous[2]) * dy
        body_dy = -math.sin(previous[2]) * dx + math.cos(previous[2]) * dy
        dyaw = wrap_angle(current[2] - previous[2])
        self.encoder_pose = current
        dt = self.timestep
        rate = car.imu_yaw_scale * self.imu_gyro[2]
        accel_x = car.imu_accel_scale[0] * accel[0]
        accel_y = car.imu_accel_scale[1] * accel[1]
        if car.imu_position is not None:
            rate_dot = 0.0 if self.imu_previous_rate is None else (rate - self.imu_previous_rate) / dt
            accel_x, accel_y = to_center_of_gravity(accel_x, accel_y, rate, rate_dot,
                                                    car.imu_position[0] - self.params["lr"], car.imu_position[1])
        self.imu_previous_rate = rate
        observation = np.array([accel_x, accel_y,
                                rate, self.speed,
                                body_dx / dt, body_dy / dt, dyaw / dt,
                                self.applied_control[0], self.applied_control[1]], dtype=np.float32)
        return observation, np.array(current)

    def lidar_scan(self):
        """The latest LiDAR scan resampled onto the training layout (1080 beams over 270 degrees; returns under
        0.05 m dropped)."""
        from sit_lmpc.world.rssm_runtime import resample_scan
        if self.scan is None:
            raise RuntimeError(f"no LiDAR scan on {self.car.scan_topic}")
        return resample_scan(self.scan.ranges, self.scan.angle_min, self.scan.angle_increment)

    def _measured(self):
        msg = self.odom
        position = msg.pose.pose.position
        yaw = quaternion_yaw(msg.pose.pose.orientation)
        if self.yaw is not None:
            yaw = self.yaw + wrap_angle(yaw - self.yaw)
        self.yaw = yaw
        lr = self.params["lr"]
        x = position.x + lr * math.cos(yaw)
        y = position.y + lr * math.sin(yaw)
        self._advance_delta()
        delta = self.delta_history[0] if self.delay_steps and self.delta_history else self.delta
        slip = math.atan(lr / self.wheelbase * math.tan(delta))
        return np.asarray([x, y, delta, self.speed, yaw, self.yaw_rate, slip], dtype=self.dtype)

    def _observe(self):
        cartesian = self._measured()
        x, y, yaw = float(cartesian[0]), float(cartesian[1]), float(cartesian[4])
        if self.predictor is not None and len(self.pending) == self.delay_steps:
            from sit_lmpc.dynamics.st import rollout

            actions = np.asarray(self.pending) / self.action_scale
            cartesian = np.asarray(rollout(self.predictor, cartesian, actions)[-1], dtype=self.dtype)
            x, y, yaw = float(cartesian[0]), float(cartesian[1]), float(cartesian[4])
        s, ey, ephi = self.track.cartesian_to_frenet(x, y, yaw, use_s_guess=self.track.s_guess is not None)
        frenet = cartesian.copy()
        frenet[[0, 1, 4]] = (s, ey, ephi)
        return cartesian, frenet

    def _record(self):
        msg = self.odom
        position = msg.pose.pose.position
        self.log.append([
            time.perf_counter() - self.started, position.x, position.y,
            quaternion_yaw(msg.pose.pose.orientation), self.speed, self.yaw_rate, self.odom_yaw_rate,
            self.delta, self.delta_target, self.speed_target, self.closest,
        ])

    def _wait_until(self, deadline, new_odom=False):
        count = self.odom_count
        while time.perf_counter() < deadline:
            self._spin(min(max(deadline - time.perf_counter(), 0.0), 0.01))
            if new_odom and self.odom_count > count:
                return

    def _seed_delay(self):
        self.pending.clear()
        self.delta_history.clear()
        if not self.delay_steps:
            return
        for _ in range(self.delay_steps):
            self.pending.append(np.zeros(2))
        for _ in range(self.delay_steps + 1):
            self.delta_history.append(self.delta)

    def _settle(self):
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            self._stop()
            self._spin(0.05)
            if self.speed is not None and abs(self.speed) < 0.05:
                self._spin_for(1.0)
                self._level_imu()
                return

    def _spin_for(self, seconds):
        if self.imu_leveled:
            return
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            self._stop()
            self._spin(0.02)

    def reset(self, state):
        state = np.asarray(state, dtype=np.float64)
        if self.started is None:
            self.started = time.perf_counter()
        self._settle()
        self._wait_for_topics()
        length = float(self.track.s_frame_max)
        message = (f"Car stopped. Put it on the track 0.3-{self.car.start_zone:.1f} m behind the start line, facing "
                   f"forward, then press joystick button {self.car.go_button} "
                   f"(or: ros2 topic pub --once {self.car.go_topic} std_msgs/msg/Empty).")
        self.emit("car_waiting", message=message)
        while True:
            self.go = False
            while not self.go:
                self._stop()
                self._spin(0.05)
            self.yaw = None
            cartesian = self._measured()
            s, ey, ephi = self.track.cartesian_to_frenet(
                float(cartesian[0]), float(cartesian[1]), float(cartesian[4]), use_s_guess=False)
            behind = length - s
            if 0.3 < behind < self.car.start_zone and abs(ey) < self.half_width and abs(ephi) < self.car.start_heading:
                break
            self.emit("car_start_rejected", behind=float(behind), ey=float(ey), ephi=float(ephi),
                      message="reposition the car and press go again")
        self.launch_speed = min(float(state[3]), self.car.max_speed)
        state[3] = self.launch_speed
        cartesian = state.astype(self.dtype)
        s, ey, ephi = self.track.cartesian_to_frenet(
            float(state[0]), float(state[1]), float(state[4]), use_s_guess=False)
        self.track.s_guess = None
        frenet = cartesian.copy()
        frenet[[0, 1, 4]] = (s, ey, ephi)
        self.applied_control = np.zeros(2)
        self.encoder_pose = None
        self.imu_previous_rate = None
        self.close_scan = False
        self.emit("car_start_accepted", behind=float(behind), launch_speed=self.launch_speed)
        self.returned_at = None
        return (cartesian, frenet), {"lap_times": [0.0], "lap_counts": [0], "sim_time": 0.0}

    def _launch(self):
        length = float(self.track.s_frame_max)
        started = time.perf_counter()
        previous = None
        self.yaw = None
        self.track.s_guess = None
        while True:
            self._check_odom()
            cartesian = self._measured()
            x, y, yaw = float(cartesian[0]), float(cartesian[1]), float(cartesian[4])
            s, ey, _ = self.track.cartesian_to_frenet(x, y, yaw, use_s_guess=self.track.s_guess is not None)
            if previous is not None and previous > length / 2 and s < length / 2:
                break
            previous = s
            if abs(ey) > self.half_width or self.close_scan:
                self._stop("launch_out_of_bounds" if abs(ey) > self.half_width else "launch_obstacle")
                raise RuntimeError("the car left the track or saw an obstacle while driving to the start line")
            goal = self.track.frenet_to_cartesian(s + self.car.launch_lookahead, 0.0, 0.0)
            rear_x = x - self.params["lr"] * math.cos(yaw)
            rear_y = y - self.params["lr"] * math.sin(yaw)
            dx, dy = goal[0] - rear_x, goal[1] - rear_y
            lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)
            curvature = 2.0 * lateral / max(dx * dx + dy * dy, 1e-6)
            self._drive(math.atan(self.wheelbase * curvature), self.launch_speed)
            self._record()
            self._wait_until(time.perf_counter() + 0.05, new_odom=True)
            if time.perf_counter() - started > self.car.launch_timeout:
                self._stop("launch_timeout")
                raise RuntimeError(f"the car did not reach the start line within {self.car.launch_timeout} s")
        self.emit("car_launch", seconds=time.perf_counter() - started, speed=self.speed)
        self._seed_delay()
        self.launch_speed = None

    def brake_to_stop(self):
        started = time.perf_counter()
        start_speed = max(float(self.speed or 0.0), 0.0)
        self.track.s_guess = None
        while True:
            self._check_odom()
            cartesian = self._measured()
            if float(cartesian[3]) < 0.1:
                break
            x, y, yaw = float(cartesian[0]), float(cartesian[1]), float(cartesian[4])
            s, ey, _ = self.track.cartesian_to_frenet(x, y, yaw, use_s_guess=self.track.s_guess is not None)
            if abs(ey) > self.half_width or self.close_scan:
                self._stop("stop_out_of_bounds" if abs(ey) > self.half_width else "stop_obstacle")
                break
            goal = self.track.frenet_to_cartesian(s + self.car.launch_lookahead, 0.0, 0.0)
            rear_x = x - self.params["lr"] * math.cos(yaw)
            rear_y = y - self.params["lr"] * math.sin(yaw)
            dx, dy = goal[0] - rear_x, goal[1] - rear_y
            lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)
            curvature = 2.0 * lateral / max(dx * dx + dy * dy, 1e-6)
            elapsed = time.perf_counter() - started
            self._drive(math.atan(self.wheelbase * curvature), max(start_speed - self.car.stop_decel * elapsed, 0.0))
            self._record()
            self._wait_until(time.perf_counter() + 0.05, new_odom=True)
            if elapsed > start_speed / self.car.stop_decel + 3.0:
                break
        self._stop()
        self.emit("car_lap_stop", seconds=time.perf_counter() - started, speed=self.speed)

    def _follow_targets(self, targets, deadline):
        start_delta, start_speed = self.delta, self.speed
        while time.perf_counter() < deadline:
            self._check_odom()
            steering, speed = targets(self._measured())
            self._drive(steering, speed)
            self._record()
            self._wait_until(deadline, new_odom=True)
        self._advance_delta()
        control = np.array([(self.delta - start_delta) / self.timestep, (self.speed - start_speed) / self.timestep])
        return np.clip(control, -self.action_scale, self.action_scale)

    def step(self, normalized_action, noise):
        started = time.perf_counter()
        if self.returned_at is not None:
            self.plan_times.append(started - self.returned_at)
            self.window_plan.append(self.plan_times[-1])
        if self.launch_speed is not None:
            self._launch()
            self.tick = time.perf_counter()
        deadline = self.tick + self.timestep
        if time.perf_counter() >= deadline:
            self.overruns += 1
            deadline = time.perf_counter() + self.timestep / 2
        control = np.asarray(normalized_action) * self.action_scale + noise
        targets = getattr(getattr(self.substep_policy, "__self__", None), "targets", None)
        if targets is not None:
            control = self._follow_targets(targets, deadline)
        else:
            self._check_odom()
            self._advance_delta()
            rate = min(max(float(control[0]), -self.params["sv_max"]), self.params["sv_max"])
            acceleration = min(max(float(control[1]), -self.params["a_max"]), self.params["a_max"])
            self._drive(self.delta + rate * self.timestep, self.speed + acceleration * self.car.speed_horizon)
            self._record()
            self._wait_until(deadline)
        self._check_odom()
        if self.delay_steps:
            self.pending.append(np.asarray(control, dtype=float))
            self.delta_history.append(self.delta)
        self.applied_control = np.asarray(control, dtype=float)
        observation = self._observe()
        self.tick = deadline
        collision = self.close_scan
        self.close_scan = False
        if collision:
            self._stop("obstacle")
        elif abs(float(observation[1][1])) > self.half_width:
            self._stop("out_of_bounds")
        self.step_times.append(time.perf_counter() - started)
        self.window_step.append(self.step_times[-1])
        self.steps += 1
        if self.simulation.ros.timing_report_steps and self.steps % self.simulation.ros.timing_report_steps == 0:
            self.emit("control_step_timing", steps=self.steps, overruns=self.overruns,
                      watchdog_stops=self.watchdog_stops,
                      planning=percentile_summary(self.window_plan),
                      simulator_step=percentile_summary(self.window_step))
            self.window_plan, self.window_step = [], []
        info = {"collision": collision, "sim_time": time.perf_counter() - self.started}
        self.returned_at = time.perf_counter()
        return observation, control, False, False, info

    def render(self):
        pass

    def display_observation(self, observation):
        pass

    def window_open(self):
        return False

    def timing_summary(self):
        return {
            "steps": self.steps,
            "planning_seconds": percentile_summary(self.plan_times),
            "control_step_seconds": percentile_summary(self.step_times),
            "control_period_seconds": self.timestep,
            "overruns": self.overruns,
            "watchdog_stops": self.watchdog_stops,
            "stepping": "car",
        }

    def close(self):
        for _ in range(5):
            self._publish(self.delta_target, 0.0)
            time.sleep(0.02)
        self.running = False
        summary = self.timing_summary()
        if self.directory is not None and self.steps:
            (self.directory / "control_timing.json").write_text(json.dumps(summary, indent=2) + "\n")
            np.savez(self.directory / "control_timing.npz",
                     planning=np.asarray(self.plan_times), control_step=np.asarray(self.step_times))
        if self.directory is not None and self.log:
            np.savez(self.directory / "car_log.npz", log=np.asarray(self.log, dtype=float),
                     columns=np.asarray(["time", "x", "y", "yaw", "speed", "imu_yaw_rate", "odom_yaw_rate",
                                         "delta_estimate", "steering_command", "speed_command", "closest_scan"]))
        self.emit("control_timing_summary", **summary)
        if self.publish_error is not None:
            self.emit("car_publish_error", message=self.publish_error)
        if self.bag is not None:
            self.bag.send_signal(signal.SIGINT)
            try:
                self.bag.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.bag.kill()
        try:
            self.node.destroy_node()
            if self.rclpy.ok():
                self.rclpy.shutdown()
        except Exception:
            pass
