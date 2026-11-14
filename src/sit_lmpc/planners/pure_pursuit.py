"""Warm-up controller ported from the ESE 6150 lab 5 submission's pure pursuit (tag submission, 4e5e5ce).

Same speed profile, lookahead, corner cut, latency projection and dynamic steering law; it runs at the
SIT-LMPC control rate and turns the steering-angle and speed targets into steering-speed and
acceleration commands the way the simulator's actuator loops would.
"""

import numpy as np
from .common import PlannerResult


def load_path(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values = line.replace(";", ",").split(",")
        rows.append((float(values[0]), float(values[1])))
    return np.asarray(rows)


class PurePursuit:
    def __init__(self, planner, simulation, vehicle, model):
        self.config = planner.warmup
        self.model = model
        self.horizon = planner.horizon
        self.timestep = simulation.timestep
        self.action_scale = np.asarray(simulation.action_scale, dtype=float)
        self.vehicle = vehicle
        self.wheelbase = vehicle["lf"] + vehicle["lr"]
        self.max_steering = vehicle["s_max"]
        self.accel_gain = 10.0 * vehicle["a_max"] / vehicle["v_max"]
        self.brake_gain = min(10.0 * vehicle["a_max"] / -vehicle["v_min"], 1.0 / simulation.timestep)
        self.path = load_path(self.config.path)
        self.actions = np.zeros((planner.horizon, 2))
        self.last_index = 0
        self.found = False
        self.build_speed_profile()

    def build_speed_profile(self):
        config = self.config
        count = len(self.path)
        following = np.roll(self.path, -1, axis=0)
        self.arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(following - self.path, axis=1))))
        self.length = self.arc[-1]
        self.curvature = np.zeros(count)
        for i in range(count):
            a = self.path[(i - 3) % count]
            b = self.path[i]
            c = self.path[(i + 3) % count]
            cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            sides = np.linalg.norm(b - a) * np.linalg.norm(c - b) * np.linalg.norm(c - a)
            self.curvature[i] = 2.0 * cross / sides if sides > 1e-9 else 0.0
        self.speed = np.zeros(count)
        for i in range(count):
            k = max(self.max_curvature(self.arc[i] - 0.2, 0.7), 1e-5)
            v = min(config.max_speed, np.sqrt(config.max_lat_accel / k))
            steering_left = max(0.0, self.max_steering - np.arctan(self.wheelbase * k))
            if config.understeer_gradient > 0.0:
                v = min(v, np.sqrt(steering_left / (config.understeer_gradient * k)))
            self.speed[i] = v
        for j in range(2 * count, 0, -1):
            i = (j - 1) % count
            following_index = j % count
            ds = self.arc[i + 1] - self.arc[i]
            self.speed[i] = min(self.speed[i], np.sqrt(self.speed[following_index] ** 2 + 2.0 * config.brake * ds))
        for j in range(2 * count):
            i = j % count
            following_index = (j + 1) % count
            ds = self.arc[i + 1] - self.arc[i]
            self.speed[following_index] = min(self.speed[following_index], np.sqrt(self.speed[i] ** 2 + 2.0 * config.accel * ds))
        self.speed *= config.speed_scale

    def index_at(self, s):
        s = s % self.length
        return min(int(np.searchsorted(self.arc, s, side="right")) - 1, len(self.path) - 1)

    def point_at(self, s):
        s = s % self.length
        i = self.index_at(s)
        a = self.path[i]
        b = self.path[(i + 1) % len(self.path)]
        t = (s - self.arc[i]) / (self.arc[i + 1] - self.arc[i])
        return a + t * (b - a)

    def max_curvature(self, s, distance):
        k = 0.0
        for step in range(int(round(distance / 0.1)) + 1):
            k = max(k, abs(self.curvature[self.index_at(s + 0.1 * step)]))
        return k

    def nearest_waypoint(self, x, y):
        count = len(self.path)
        position = np.array([x, y])
        best = self.last_index
        if self.found:
            candidates = (self.last_index - 20 + np.arange(100)) % count
            distances = np.sum((self.path[candidates] - position) ** 2, axis=1)
            if distances.min() < np.sum((self.path[best] - position) ** 2):
                best = int(candidates[np.argmin(distances)])
        if not self.found or np.sum((self.path[best] - position) ** 2) > 4.0:
            best = int(np.argmin(np.sum((self.path - position) ** 2, axis=1)))
        self.found = True
        self.last_index = best
        return best

    def dynamic_steering(self, v, curvature, slip, yaw_rate):
        p = self.vehicle
        lf, lr, mass = p["lf"], p["lr"], p["m"]
        cf = p["mu"] * mass / (lf + lr) * p["C_Sf"] * 9.81 * lr
        cr = p["mu"] * mass / (lf + lr) * p["C_Sr"] * 9.81 * lf
        steady_state = (mass * v * v * curvature + (cf + cr) * slip - (cr * lr - cf * lf) * yaw_rate / v) / cf
        return steady_state + self.config.yaw_rate_gain * (v * curvature - yaw_rate)

    def targets(self, state):
        config = self.config
        x, y, _, v, yaw, yaw_rate, slip = (float(value) for value in state)
        x -= self.vehicle["lr"] * np.cos(yaw)
        y -= self.vehicle["lr"] * np.sin(yaw)
        slip = slip if v > 1.0 else 0.0
        heading = yaw + slip
        x += v * np.cos(heading) * config.latency
        y += v * np.sin(heading) * config.latency
        heading += yaw_rate * config.latency
        i = self.nearest_waypoint(x, y)
        s = self.arc[i]
        lookahead = config.lookahead_base + config.lookahead_gain * v
        k = max(self.max_curvature(s, lookahead), 1e-3)
        lookahead = min(lookahead, np.sqrt(8.0 * config.corner_cut / k))
        lookahead = np.clip(lookahead, config.lookahead_min, config.lookahead_max)
        goal = self.point_at(s + lookahead)
        dx, dy = goal[0] - x, goal[1] - y
        goal_y = -dx * np.sin(heading) + dy * np.cos(heading)
        curvature = 2.0 * goal_y / (dx * dx + dy * dy)
        if v > 1.0:
            steering = self.dynamic_steering(v, curvature, slip, yaw_rate)
        else:
            steering = np.arctan(self.wheelbase * curvature)
        return float(np.clip(steering, -self.max_steering, self.max_steering)), float(self.speed[i])

    def substep(self, state):
        """Physical (steering speed, acceleration) at the simulator's rate, through its actuator loops."""
        p = self.vehicle
        steering, speed = self.targets(state)
        rate = 10.0 * p["sv_max"] / (p["s_max"] - p["s_min"]) * (steering - state[2])
        if state[3] > 0.0:
            gain = 10.0 * p["a_max"] / (p["v_max"] if speed > state[3] else -p["v_min"])
        else:
            gain = 2.0 * p["a_max"] / (p["v_max"] if speed > state[3] else -p["v_min"])
        control = np.array([rate, gain * (speed - state[3])])
        return np.clip(control, -self.action_scale, self.action_scale)

    def update(self, state, target_speed):
        state = np.asarray(state, dtype=float)
        steering, speed = self.targets(state)
        gain = self.accel_gain if speed > state[3] else self.brake_gain
        action = np.array([(steering - state[2]) / self.timestep, gain * (speed - state[3])])
        action = np.clip(action / self.action_scale, -1.0, 1.0).astype(np.float32)
        trajectory = np.repeat(state[None], self.horizon, axis=0)
        return PlannerResult(action, trajectory, None)
