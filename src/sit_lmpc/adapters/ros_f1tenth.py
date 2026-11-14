"""f1tenth_gym_ros boundary: odometry in, Ackermann drive out, in lockstep or real time."""

import json
import math
import time
from collections import deque
import numpy as np

from sit_lmpc.dynamics.st import PARAMETER_FIELDS


def f1tenth_vehicle_parameters(friction, mass_ratio=1.0):
    from f1tenth_gym.envs.dynamic_models import get_f1tenth_vehicle_parameters

    source = get_f1tenth_vehicle_parameters()
    params = {}
    for name in PARAMETER_FIELDS:
        params[name] = float(getattr(source, name))
    params["mu"] = friction
    params["m"] *= mass_ratio
    return params


def steering_gain(params):
    return 10.0 * params["sv_max"] / (params["s_max"] - params["s_min"])


def speed_gain(params, speed, acceleration):
    if speed > 0.0:
        scale = 10.0
    else:
        scale = 2.0
    if acceleration > 0:
        return scale * params["a_max"] / params["v_max"]
    return scale * params["a_max"] / (-params["v_min"])


def constrained_steering_rate(delta, rate, params):
    if delta <= params["s_min"] and rate <= 0:
        return 0.0
    if delta >= params["s_max"] and rate >= 0:
        return 0.0
    return min(max(rate, params["sv_min"]), params["sv_max"])


def integrate_steering(delta, rate, params, dt):
    k1 = constrained_steering_rate(delta, rate, params)
    k2 = constrained_steering_rate(delta + k1 * dt / 2, rate, params)
    k3 = constrained_steering_rate(delta + k2 * dt / 2, rate, params)
    k4 = constrained_steering_rate(delta + k3 * dt, rate, params)
    return delta + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def percentile_summary(values):
    if not values:
        return None
    array = np.asarray(values)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


class RosSimulator:
    restore_tolerance = 0.25

    def __init__(self, simulation, track, parameters, emit=None, directory=None):
        import rclpy
        from ackermann_msgs.msg import AckermannDriveStamped
        from geometry_msgs.msg import PoseWithCovarianceStamped
        from nav_msgs.msg import Odometry
        from std_msgs.msg import Bool

        self.simulation, self.track, self.params = simulation, track, parameters
        self.ros = simulation.ros
        self.emit = emit if emit is not None else (lambda *args, **kwargs: None)
        self.directory = directory
        self.dtype = simulation.dtype
        self.substeps = round(simulation.timestep / self.ros.bridge_timestep)
        self.message_dt = self.ros.bridge_substeps * self.ros.bridge_timestep
        self.angle_mode = self.ros.drive_mode == "speed_steering_angle"
        self.delay_steps = round(self.ros.actuation_delay / simulation.timestep)
        self.pending = deque(maxlen=max(self.delay_steps, 1))
        self.delta_history = deque(maxlen=self.delay_steps + 1)
        self.predictor = None
        if self.delay_steps:
            from sit_lmpc.dynamics.st import STModel

            self.predictor = STModel.from_vehicle(parameters, simulation)
        self.rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node("sit_lmpc")
        self.drive_type = AckermannDriveStamped
        self.reset_type = PoseWithCovarianceStamped
        self.odom = None
        self.odom_count = 0
        self.collision = False
        self.delta = 0.0
        self.yaw = None
        self.stamp = None
        self.returned_at = None
        self.steps = 0
        self.substep_policy = None
        self.plan_times, self.step_times = [], []
        self.window_plan, self.window_step = [], []
        self.odom_sub = self.node.create_subscription(
            Odometry, self._topic(self.ros.odom_topic), self._odom_callback, 10
        )
        self.collision_sub = self.node.create_subscription(
            Bool, self._topic(self.ros.collision_topic), self._collision_callback, 10
        )
        self.drive_pub = self.node.create_publisher(
            AckermannDriveStamped, self._topic(self.ros.drive_topic), 10
        )
        self.reset_pub = self.node.create_publisher(
            PoseWithCovarianceStamped, self._topic(self.ros.reset_topic), 10
        )

    def _topic(self, name):
        if name.startswith("/"):
            return name
        return f"/{self.ros.namespace}/{name}"

    def _odom_callback(self, msg):
        self.odom = msg
        self.odom_count += 1

    def _collision_callback(self, msg):
        if msg.data:
            self.collision = True

    def _spin(self, timeout):
        self.rclpy.spin_once(self.node, timeout_sec=timeout)

    def _wait_for_odom(self, predicate, timeout, message):
        deadline = time.perf_counter() + timeout
        while not predicate():
            if time.perf_counter() > deadline:
                raise RuntimeError(message)
            self._spin(0.02)
        self.stamp = stamp_seconds(self.odom.header.stamp)

    def _wait_for_advance(self, dt):
        previous = self.stamp
        target = self.stamp + dt - 1e-6
        self._wait_for_odom(
            lambda: self.odom is not None and stamp_seconds(self.odom.header.stamp) >= target,
            self.ros.wait_timeout,
            f"odometry did not advance {dt:.3f} s within {self.ros.wait_timeout} s; "
            "is the bridge running (lockstep needs async_mode False and use_sim_time_bridge True)?",
        )
        advance = self.stamp - previous
        if self.ros.stepping == "lockstep" and advance > dt + self.ros.bridge_timestep / 2:
            raise RuntimeError(
                f"the bridge advanced {advance:.3f} s for one drive message, expected {dt:.3f} s; "
                "set simulation.ros.bridge_substeps to the bridge's substeps_per_drive"
            )

    def _wait_for_bridge(self):
        deadline = time.perf_counter() + self.ros.wait_timeout
        while (self.reset_pub.get_subscription_count() == 0
               or self.drive_pub.get_subscription_count() == 0
               or self.odom is None):
            if time.perf_counter() > deadline:
                raise RuntimeError(
                    f"no bridge found on {self._topic(self.ros.drive_topic)} / "
                    f"{self._topic(self.ros.reset_topic)} / {self._topic(self.ros.odom_topic)} "
                    f"within {self.ros.wait_timeout} s"
                )
            self._spin(0.1)

    def _publish_drive(self, steering, longitudinal):
        msg = self.drive_type()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        if self.angle_mode:
            msg.drive.steering_angle = float(steering)
            msg.drive.speed = float(longitudinal)
        else:
            msg.drive.steering_angle_velocity = float(steering)
            msg.drive.acceleration = float(longitudinal)
        self.drive_pub.publish(msg)

    def _measured_speed(self):
        twist = self.odom.twist.twist.linear
        return math.hypot(twist.x, twist.y)

    def _command(self, steering_speed, acceleration):
        """Send one physical (steering speed, acceleration) command and advance the steering estimate."""
        params = self.params
        if self.angle_mode:
            target = self.delta + steering_speed / steering_gain(params)
            target = min(max(target, params["s_min"]), params["s_max"])
            speed = self._measured_speed()
            self._publish_drive(target, speed + acceleration / speed_gain(params, speed, acceleration))
            rate = steering_gain(params) * (target - self.delta)
            rate = min(max(rate, -params["sv_max"]), params["sv_max"])
            sent = (steering_speed, acceleration)
        else:
            rate = min(max(steering_speed, params["sv_min"]), params["sv_max"])
            duration = self.substeps * self.ros.bridge_timestep if self.ros.stepping == "realtime" else self.message_dt
            rate = min(max(rate, (params["s_min"] - self.delta) / duration), (params["s_max"] - self.delta) / duration)
            sent = (rate, min(max(acceleration, -params["a_max"]), params["a_max"]))
            self._publish_drive(*sent)
        count = self.substeps if self.ros.stepping == "realtime" else self.ros.bridge_substeps
        for _ in range(count):
            self.delta = integrate_steering(self.delta, rate, params, self.ros.bridge_timestep)
        return sent

    def _measured(self):
        msg = self.odom
        position = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        if self.yaw is not None:
            yaw = self.yaw + wrap_angle(yaw - self.yaw)
        self.yaw = yaw
        lr = self.params["lr"]
        x = position.x + lr * math.cos(yaw)
        y = position.y + lr * math.sin(yaw)
        twist = msg.twist.twist
        speed = math.hypot(twist.linear.x, twist.linear.y)
        slip = math.atan2(twist.linear.y, twist.linear.x) if speed > 0.05 else 0.0
        delta = self.delta_history[0] if self.delay_steps and self.delta_history else self.delta
        return np.asarray([x, y, delta, speed, yaw, twist.angular.z, slip], dtype=self.dtype)

    def _observe(self):
        cartesian = self._measured()
        x, y, yaw = float(cartesian[0]), float(cartesian[1]), float(cartesian[4])
        if self.predictor is not None and len(self.pending) == self.delay_steps:
            from sit_lmpc.dynamics.st import rollout

            actions = np.asarray(self.pending) / np.asarray(self.simulation.action_scale)
            cartesian = np.asarray(rollout(self.predictor, cartesian, actions)[-1], dtype=self.dtype)
            x, y, yaw = float(cartesian[0]), float(cartesian[1]), float(cartesian[4])
        s, ey, ephi = self.track.cartesian_to_frenet(
            x, y, yaw, use_s_guess=self.track.s_guess is not None
        )
        frenet = cartesian.copy()
        frenet[[0, 1, 4]] = (s, ey, ephi)
        return cartesian, frenet

    def _launch_distance(self, target_speed):
        """Distance covered by _launch, less a margin so the car stops just past the requested pose."""
        margin = 0.05
        if self.angle_mode:
            gain = speed_gain(self.params, 1.0, 1.0)
            duration = math.log(max(target_speed / 0.02, 1.0)) / gain
            distance = target_speed * duration - target_speed / gain * (1 - math.exp(-gain * duration))
        else:
            acceleration = min(float(self.simulation.action_scale[1]), self.params["a_max"])
            distance = target_speed ** 2 / (2 * acceleration)
        distance += target_speed * self.ros.actuation_delay
        return max(distance - margin, 0.0)

    def reset(self, state):
        state = np.asarray(state, dtype=np.float64)
        lr = self.params["lr"]
        run_up = self._launch_distance(float(state[3]))
        base_x = state[0] - (lr + run_up) * math.cos(state[4])
        base_y = state[1] - (lr + run_up) * math.sin(state[4])
        msg = self.reset_type()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.pose.position.x = float(base_x)
        msg.pose.pose.position.y = float(base_y)
        msg.pose.pose.orientation.z = math.sin(state[4] / 2)
        msg.pose.pose.orientation.w = math.cos(state[4] / 2)
        self._wait_for_bridge()
        self.delta = 0.0
        self.yaw = None
        self.track.s_guess = None
        self.collision = False
        deadline = time.perf_counter() + self.ros.wait_timeout
        while True:
            self.reset_pub.publish(msg)
            self._spin(0.05)
            count = self.odom_count
            self._publish_drive(0.0, 0.0)
            self._wait_for_odom(
                lambda: self.odom_count > count, self.ros.wait_timeout,
                "no odometry received; is the bridge running?",
            )
            position = self.odom.pose.pose.position
            q = self.odom.pose.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            if (math.hypot(position.x - base_x, position.y - base_y) < 0.05
                    and abs(wrap_angle(yaw - state[4])) < 0.05):
                break
            if time.perf_counter() > deadline:
                raise RuntimeError(
                    f"the simulator did not move to the requested reset pose "
                    f"({base_x:.3f}, {base_y:.3f}, {state[4]:.3f}); it reports "
                    f"({position.x:.3f}, {position.y:.3f}, {yaw:.3f})"
                )
        self._launch(float(state[3]))
        self._flush_delay(float(state[3]))
        self.collision = False
        observation = self._observe()
        self.emit("ros_reset", run_up=run_up,
                  requested=[float(value) for value in state],
                  reached=[float(value) for value in observation[0]])
        self.returned_at = time.perf_counter()
        return observation, {"lap_times": [0.0], "lap_counts": [0], "sim_time": self.stamp}

    def _launch(self, target_speed):
        """Bring the car from rest to the runner's starting speed along its heading."""
        acceleration = min(float(self.simulation.action_scale[1]), self.params["a_max"])
        started = self.stamp
        in_flight = deque(maxlen=max(round(self.ros.actuation_delay / self.message_dt), 1))
        while self._measured_speed() + (sum(in_flight) * self.message_dt if self.delay_steps else 0.0) < target_speed - 0.02:
            if self.angle_mode:
                self._publish_drive(0.0, target_speed)
            else:
                pending = sum(in_flight) * self.message_dt if self.delay_steps else 0.0
                remaining = (target_speed - self._measured_speed() - pending) / self.message_dt
                command = min(acceleration, remaining)
                self._publish_drive(0.0, command)
                in_flight.append(command)
            self._wait_for_advance(self.message_dt)
            if self.stamp - started > self.ros.launch_timeout:
                raise RuntimeError(f"launch did not reach {target_speed} m/s within {self.ros.launch_timeout} s")
        self.emit("ros_launch", target_speed=target_speed, seconds=self.stamp - started,
                  speed=self._measured_speed())

    def _flush_delay(self, target_speed):
        """With a delay, hold steady until the launch commands have taken effect, then seed the queues."""
        if not self.delay_steps:
            return
        for _ in range(round(self.ros.actuation_delay / self.message_dt)):
            self._publish_drive(0.0, target_speed if self.angle_mode else 0.0)
            self._wait_for_advance(self.message_dt)
        self.pending.clear()
        for _ in range(self.delay_steps):
            self.pending.append(np.zeros(2))
        self.delta_history.clear()
        for _ in range(self.delay_steps + 1):
            self.delta_history.append(self.delta)

    def step(self, normalized_action, noise):
        started = time.perf_counter()
        if self.returned_at is not None:
            self.plan_times.append(started - self.returned_at)
            self.window_plan.append(self.plan_times[-1])
        control = np.asarray(normalized_action) * np.asarray(self.simulation.action_scale) + noise
        self.collision = False
        if self.ros.stepping == "lockstep" and self.substep_policy is not None:
            sent = []
            for _ in range(self.substeps // self.ros.bridge_substeps):
                command = np.asarray(self.substep_policy(self._measured()), dtype=float) + noise
                sent.append(self._command(float(command[0]), float(command[1])))
                self._wait_for_advance(self.message_dt)
            control = np.mean(np.asarray(sent), axis=0)
        elif self.ros.stepping == "lockstep":
            for _ in range(self.substeps // self.ros.bridge_substeps):
                self._command(float(control[0]), float(control[1]))
                self._wait_for_advance(self.message_dt)
        else:
            self._command(float(control[0]), float(control[1]))
            self._wait_for_advance(self.simulation.timestep)
        if self.delay_steps:
            self.pending.append(np.asarray(control, dtype=float))
            self.delta_history.append(self.delta)
        observation = self._observe()
        self.step_times.append(time.perf_counter() - started)
        self.window_step.append(self.step_times[-1])
        self.steps += 1
        if self.ros.timing_report_steps and self.steps % self.ros.timing_report_steps == 0:
            self.emit("control_step_timing", steps=self.steps,
                      planning=percentile_summary(self.window_plan),
                      simulator_step=percentile_summary(self.window_step))
            self.window_plan, self.window_step = [], []
        info = {"collision": self.collision, "sim_time": self.stamp}
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
            "simulator_step_seconds": percentile_summary(self.step_times),
            "control_period_seconds": self.simulation.timestep,
            "stepping": self.ros.stepping,
        }

    def close(self):
        summary = self.timing_summary()
        if self.directory is not None and self.steps:
            (self.directory / "control_timing.json").write_text(json.dumps(summary, indent=2) + "\n")
            np.savez(self.directory / "control_timing.npz",
                     planning=np.asarray(self.plan_times), simulator_step=np.asarray(self.step_times))
        self.emit("control_timing_summary", **summary)
        try:
            self.node.destroy_node()
            if self.rclpy.ok():
                self.rclpy.shutdown()
        except Exception:
            pass
