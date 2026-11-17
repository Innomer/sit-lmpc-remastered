"""World-model training windows from a dyna-dist world dataset: true states, controls, simulated sensors."""

from pathlib import Path
import numpy as np

from dyna_dist.io.dataset import Dataset

LR = 0.17145
TIMESTEP = 0.1
OBSERVATION_FIELDS = ("accel_x", "accel_y", "gyro_z", "wheel_speed", "pose_dx", "pose_dy", "pose_dyaw",
                      "control_steer", "control_accel")


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def observations(sensors, controls):
    """Per-step encoder inputs: IMU, wheel speed, PF pose change in the previous body frame, previous control."""
    count = len(sensors)
    pose = sensors[:, 4:7]
    yaw = np.unwrap(pose[:, 2])
    dx = np.diff(pose[:, 0], prepend=pose[0, 0])
    dy = np.diff(pose[:, 1], prepend=pose[0, 1])
    previous_yaw = np.concatenate(([yaw[0]], yaw[:-1]))
    body_dx = np.cos(previous_yaw) * dx + np.sin(previous_yaw) * dy
    body_dy = -np.sin(previous_yaw) * dx + np.cos(previous_yaw) * dy
    dyaw = np.diff(yaw, prepend=yaw[0])
    previous_control = np.zeros((count, 2))
    previous_control[1:] = controls[: count - 1]
    return np.column_stack((sensors[:, :4], body_dx / TIMESTEP, body_dy / TIMESTEP, dyaw / TIMESTEP, previous_control))


def pf_cog_pose(sensors):
    yaw = sensors[:, 6]
    return np.column_stack((sensors[:, 4] + LR * np.cos(yaw), sensors[:, 5] + LR * np.sin(yaw), yaw))


class WorldData:
    def __init__(self, path, split, include_changes=True):
        self.path = Path(path)
        self.dataset = Dataset(self.path)
        self.items = []
        for member in self.dataset.members(split):
            for filename, sensor_file in zip(member["files"], member["sensor_files"]):
                self._add(filename, sensor_file, "instance", member["instance_id"])
        if include_changes:
            for episode in self.dataset.manifest.get("change_episodes", []):
                if episode["split"] == split:
                    self._add(episode["files"][0], episode["sensor_files"][0], "change", episode["episode_id"])

    def _add(self, filename, sensor_file, kind, source):
        trajectory = self.dataset.trajectory(filename)
        valid = trajectory["valid"]
        count = len(valid) if valid.all() else int(np.argmin(valid))
        if count < 2:
            return
        sensors = np.load(self.path / sensor_file)["readings"][: count + 1]
        states = trajectory["states"][: count + 1]
        controls = trajectory["controls"][:count]
        item = {
            "states": states.astype(np.float32),
            "controls": controls.astype(np.float32),
            "observations": observations(sensors, controls).astype(np.float32),
            "pose": pf_cog_pose(sensors).astype(np.float32),
            "kind": kind,
            "source": source,
            "file": filename,
        }
        if "conditions" in trajectory:
            item["conditions"] = trajectory["conditions"][:count].astype(np.float32)
        self.items.append(item)

    def starts(self, length):
        result = []
        for index, item in enumerate(self.items):
            last = len(item["controls"]) - length
            for start in range(0, last + 1):
                result.append((index, start))
        return result

    def batch(self, rng, size, length, starts=None):
        starts = self.starts(length) if starts is None else starts
        chosen = [starts[i] for i in rng.integers(0, len(starts), size)]
        return self.gather(chosen, length)

    def gather(self, chosen, length):
        fields = {"states": [], "controls": [], "observations": [], "pose": []}
        for index, start in chosen:
            item = self.items[index]
            fields["states"].append(item["states"][start:start + length + 1])
            fields["controls"].append(item["controls"][start:start + length])
            fields["observations"].append(item["observations"][start:start + length + 1])
            fields["pose"].append(item["pose"][start:start + length + 1])
        return {name: np.stack(values) for name, values in fields.items()}

    def observation_scale(self):
        stacked = np.concatenate([item["observations"] for item in self.items])
        return stacked.mean(axis=0).astype(np.float32), (stacked.std(axis=0) + 1e-3).astype(np.float32)
