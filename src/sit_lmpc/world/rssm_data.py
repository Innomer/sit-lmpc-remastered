"""Training windows from the on-track dataset (scripts/collect_track_data.py) for the RSSM."""

import json
from pathlib import Path
import numpy as np

from sit_lmpc.world.data import observations, pf_cog_pose

MIN_RANGE = 0.05
MAX_RANGE = 10.0
MAX_SLIP = 1.0
MAX_YAW_RATE = 8.0


def downsample(lidar, beams):
    """Closest return in each of `beams` equal groups of the full scan (keeps the nearest wall)."""
    groups = np.array_split(np.arange(lidar.shape[-1]), beams)
    return np.stack([lidar[..., g].min(axis=-1) for g in groups], axis=-1)


def log_range(ranges):
    return np.log(np.clip(ranges, MIN_RANGE, MAX_RANGE))


class TrackData:
    def __init__(self, root, split, beams=64, maps=None, max_slip=MAX_SLIP, max_yaw_rate=MAX_YAW_RATE):
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        self.episodes = []
        for record in manifest["episodes"]:
            if record["split"] != split or (maps is not None and record["map"] not in maps):
                continue
            data = np.load(self.root / record["file"])
            states = data["states"].astype(np.float32)
            outside = np.flatnonzero((np.abs(states[:, 6]) > max_slip) | (np.abs(states[:, 5]) > max_yaw_rate))
            count = int(outside[0]) if len(outside) else len(states)
            if count < 2:
                continue
            states = states[:count]
            controls = data["controls"][:count - 1].astype(np.float32)
            sensors = data["sensors"][:count]
            self.episodes.append({
                "states": states,
                "controls": controls,
                "sensors": observations(sensors, controls).astype(np.float32),
                "pose": pf_cog_pose(sensors).astype(np.float32),
                "lidar": downsample(data["lidar"][:count].astype(np.float32), beams),
                "conditions": data["conditions"][:count, [0, 3]].astype(np.float32),
                "map": record["map"],
                "change": record["change"]["kind"],
            })

    def observation_scale(self):
        values = np.concatenate([e["sensors"] for e in self.episodes])
        return values.mean(axis=0), values.std(axis=0) + 1e-6

    def starts(self, window):
        return [(i, t) for i, e in enumerate(self.episodes) for t in range(0, len(e["controls"]) - window + 1)]

    def high_slip_windows(self, window, threshold):
        """(episode, start) of every window whose states include |slip| above threshold."""
        items = []
        for i, e in enumerate(self.episodes):
            count = len(e["controls"]) - window + 1
            if count <= 0:
                continue
            high = np.abs(e["states"][:, 6]) > threshold
            running = np.concatenate(([0], np.cumsum(high)))
            hits = running[window + 1:window + 1 + count] - running[:count]
            items.extend((i, int(t)) for t in np.flatnonzero(hits > 0))
        return items

    def sample(self, rng, size, window, noise=True, high_slip=None, high_slip_fraction=0.0):
        pool = [i for i, e in enumerate(self.episodes) if len(e["controls"]) >= window]
        special = int(round(size * high_slip_fraction)) if high_slip else 0
        chosen = rng.choice(pool, size - special)
        items = [(int(i), int(rng.integers(0, len(self.episodes[i]["controls"]) - window + 1))) for i in chosen]
        if special:
            items += [high_slip[j] for j in rng.integers(0, len(high_slip), special)]
        return self.gather(items, window, rng if noise else None)

    def gather(self, items, window, rng=None):
        batch = {key: [] for key in ("states", "controls", "sensors", "pose", "lidar", "conditions")}
        for index, start in items:
            e = self.episodes[index]
            batch["states"].append(e["states"][start:start + window + 1])
            batch["controls"].append(e["controls"][start:start + window])
            batch["sensors"].append(e["sensors"][start:start + window + 1])
            batch["pose"].append(e["pose"][start:start + window + 1])
            batch["conditions"].append(e["conditions"][start:start + window + 1])
            lidar = e["lidar"][start:start + window + 1]
            if rng is not None:
                lidar = lidar + rng.normal(0.0, rng.uniform(0.0, 0.03), lidar.shape)
                lidar = np.where(rng.random(lidar.shape) < 0.01, MAX_RANGE, lidar)
            batch["lidar"].append(log_range(lidar))
        return {key: np.asarray(value, np.float32) for key, value in batch.items()}


class MapFreeData(TrackData):
    """The map-free excitation dataset (sit_lmpc.world.data.WorldData) as RSSM windows. Driven in open space, so the
    LiDAR it would have seen is maximum range on every beam."""

    def __init__(self, root, split, beams=64, max_slip=MAX_SLIP, max_yaw_rate=MAX_YAW_RATE):
        from sit_lmpc.world.data import WorldData
        data = WorldData(root, split)
        instances = {m["instance_id"]: m for m in data.dataset.instances}
        self.episodes = []
        for item in data.items:
            states = item["states"].astype(np.float32)
            outside = np.flatnonzero((np.abs(states[:, 6]) > max_slip) | (np.abs(states[:, 5]) > max_yaw_rate))
            count = int(outside[0]) if len(outside) else len(states)
            if count < 2:
                continue
            if "conditions" in item:
                conditions = item["conditions"][:, [0, 3]]
                conditions = np.concatenate([conditions, conditions[-1:]])[:count]
            else:
                member = instances[item["source"]]
                conditions = np.tile([member["friction"], member["mass_ratio"]], (count, 1))
            self.episodes.append({
                "states": states[:count],
                "controls": item["controls"][:count - 1].astype(np.float32),
                "sensors": item["observations"][:count].astype(np.float32),
                "pose": item["pose"][:count].astype(np.float32),
                "lidar": np.full((count, beams), MAX_RANGE, np.float32),
                "conditions": conditions.astype(np.float32),
                "map": "map-free",
                "change": item["kind"],
            })
