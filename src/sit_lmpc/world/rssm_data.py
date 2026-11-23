"""Training windows from the on-track dataset (scripts/collect_track_data.py) for the RSSM."""

import json
from pathlib import Path
import numpy as np

from sit_lmpc.world.data import observations, pf_cog_pose

MIN_RANGE = 0.05
MAX_RANGE = 10.0


def downsample(lidar, beams):
    """Closest return in each of `beams` equal groups of the full scan (keeps the nearest wall)."""
    groups = np.array_split(np.arange(lidar.shape[-1]), beams)
    return np.stack([lidar[..., g].min(axis=-1) for g in groups], axis=-1)


def log_range(ranges):
    return np.log(np.clip(ranges, MIN_RANGE, MAX_RANGE))


class TrackData:
    def __init__(self, root, split, beams=64, maps=None):
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        self.episodes = []
        for record in manifest["episodes"]:
            if record["split"] != split or (maps is not None and record["map"] not in maps):
                continue
            data = np.load(self.root / record["file"])
            states = data["states"].astype(np.float32)
            controls = data["controls"].astype(np.float32)
            sensors = data["sensors"]
            self.episodes.append({
                "states": states,
                "controls": controls,
                "sensors": observations(sensors, controls).astype(np.float32),
                "pose": pf_cog_pose(sensors).astype(np.float32),
                "lidar": downsample(data["lidar"].astype(np.float32), beams),
                "conditions": data["conditions"][:, [0, 3]].astype(np.float32),
                "map": record["map"],
                "change": record["change"]["kind"],
            })

    def observation_scale(self):
        values = np.concatenate([e["sensors"] for e in self.episodes])
        return values.mean(axis=0), values.std(axis=0) + 1e-6

    def starts(self, window):
        return [(i, t) for i, e in enumerate(self.episodes) for t in range(0, len(e["controls"]) - window + 1)]

    def sample(self, rng, size, window, noise=True):
        pool = [i for i, e in enumerate(self.episodes) if len(e["controls"]) >= window]
        chosen = rng.choice(pool, size)
        starts = [int(rng.integers(0, len(self.episodes[i]["controls"]) - window + 1)) for i in chosen]
        return self.gather(list(zip(chosen, starts)), window, rng if noise else None)

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
