"""Replays a recorded lap as the plant, so the planner can be timed without a simulator or ROS."""

import json
import time
import numpy as np

from sit_lmpc.adapters.ros_f1tenth import percentile_summary


class ReplaySimulator:
    restore_tolerance = 1e9

    def __init__(self, simulation, track, emit=None, directory=None):
        self.simulation, self.track = simulation, track
        self.emit = emit if emit is not None else (lambda *args, **kwargs: None)
        self.directory = directory
        with np.load(simulation.replay_path, allow_pickle=True) as recorded:
            self.cartesian = np.asarray(recorded["cartesian"], dtype=simulation.dtype)
            self.frenet = np.asarray(recorded["frenet"], dtype=simulation.dtype)
        self.index = 0
        self.steps = 0
        self.returned_at = None
        self.plan_times, self.step_times = [], []
        self.window_plan = []

    def _observation(self):
        return self.cartesian[self.index].copy(), self.frenet[self.index].copy()

    def reset(self, state):
        self.index = 0
        self.returned_at = time.perf_counter()
        return self._observation(), {"lap_times": [0.0], "lap_counts": [0], "sim_time": 0.0}

    def step(self, normalized_action, noise):
        started = time.perf_counter()
        if self.returned_at is not None:
            self.plan_times.append(started - self.returned_at)
            self.window_plan.append(self.plan_times[-1])
        control = np.asarray(normalized_action) * np.asarray(self.simulation.action_scale) + noise
        self.index += 1
        if self.index >= len(self.cartesian):
            self.index = 1
        self.steps += 1
        if self.simulation.ros.timing_report_steps and self.steps % self.simulation.ros.timing_report_steps == 0:
            self.emit("control_step_timing", steps=self.steps, planning=percentile_summary(self.window_plan))
            self.window_plan = []
        self.step_times.append(time.perf_counter() - started)
        self.returned_at = time.perf_counter()
        return self._observation(), control, False, False, {"collision": False, "sim_time": self.steps * self.simulation.timestep}

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
            "control_period_seconds": self.simulation.timestep,
            "stepping": "replay",
        }

    def close(self):
        summary = self.timing_summary()
        if self.directory is not None and self.steps:
            (self.directory / "control_timing.json").write_text(json.dumps(summary, indent=2) + "\n")
            np.savez(self.directory / "control_timing.npz", planning=np.asarray(self.plan_times))
        self.emit("control_timing_summary", **summary)
