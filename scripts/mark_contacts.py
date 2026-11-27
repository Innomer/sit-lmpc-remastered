"""Find the first wall contact of every episode in an on-track dataset and store it next to the manifest.

    python scripts/mark_contacts.py --data data/f1tenth-track

Contact = any point on the car body outline touches an occupied cell of the map. The body in the rear-axle frame
spans x -0.125..0.455 m, y +-0.155 m (same as the sim's collision body). Dataset states are at the centre of gravity,
so the rear axle is lr = 0.17145 m behind. Closed-loop episodes (cl-*) are also checked against the sim's own
collision flag in their source trajectory when it is available.

Writes <data>/contacts.json: {"rule": ..., "episodes": {file: {"contact": first contact step or null,
"crash_step": step to use as the crash time or null, "source": ...}}}. TrackData ends every episode before its
contact step, in training and evaluation alike.
"""

import argparse
import glob
import json
from pathlib import Path
import numpy as np
import yaml
from PIL import Image

from sit_lmpc.track.resources import map_path

LR = 0.17145
BODY_X = (-0.125, 0.455)
BODY_Y = 0.155
MAPS = {"Spielberg": "Spielberg", "custom1": "custom1", "levine": "tracks/levine_sim/levine_sim"}


def outline(points_long=14, points_short=6):
    xs = np.linspace(*BODY_X, points_long)
    ys = np.linspace(-BODY_Y, BODY_Y, points_short)
    return np.concatenate([np.column_stack((xs, np.full_like(xs, -BODY_Y))),
                           np.column_stack((xs, np.full_like(xs, BODY_Y))),
                           np.column_stack((np.full_like(ys, BODY_X[0]), ys)),
                           np.column_stack((np.full_like(ys, BODY_X[1]), ys))])


class Occupancy:
    def __init__(self, name):
        base = Path(map_path(MAPS[name]))
        meta = yaml.safe_load(Path(f"{base}_map.yaml").read_text())
        image = np.asarray(Image.open(Path(f"{base}_map.yaml").parent / meta["image"]).convert("L"), np.float32) / 255.0
        occupancy = image if meta.get("negate", 0) else 1.0 - image
        self.occupied = occupancy > meta.get("occupied_thresh", 0.65)
        self.resolution = float(meta["resolution"])
        self.origin = np.asarray(meta["origin"][:2], np.float64)
        self.body = outline()

    def contacts(self, states):
        yaw = states[:, 4]
        c, s = np.cos(yaw), np.sin(yaw)
        rear = states[:, :2] - LR * np.column_stack((c, s))
        px = rear[:, None, 0] + c[:, None] * self.body[None, :, 0] - s[:, None] * self.body[None, :, 1]
        py = rear[:, None, 1] + s[:, None] * self.body[None, :, 0] + c[:, None] * self.body[None, :, 1]
        col = np.floor((px - self.origin[0]) / self.resolution).astype(int)
        row = self.occupied.shape[0] - 1 - np.floor((py - self.origin[1]) / self.resolution).astype(int)
        inside = (row >= 0) & (row < self.occupied.shape[0]) & (col >= 0) & (col < self.occupied.shape[1])
        hit = np.where(inside, self.occupied[np.clip(row, 0, self.occupied.shape[0] - 1),
                                             np.clip(col, 0, self.occupied.shape[1] - 1)], True)
        return hit.any(axis=1)


def sim_flag(results, name):
    """First step flagged by the sim's collision sensor in the source trajectory of a cl-<run>-<attempt> episode."""
    stem = Path(name).stem[3:]
    run, attempt = stem.rsplit("-", 1)
    files = glob.glob(str(Path(results) / run / "trajectories" / "*" / f"attempt-{attempt}.npz"))
    if not files:
        return "missing"
    data = np.load(files[0], allow_pickle=True)
    if "contacts" not in data.files:
        return "missing"
    hit = np.flatnonzero(np.asarray(data["contacts"], bool))
    return int(hit[0]) if len(hit) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/f1tenth-track")
    parser.add_argument("--results", default="results/Spielberg")
    args = parser.parse_args()
    root = Path(args.data)
    manifest = json.loads((root / "manifest.json").read_text())
    maps, out, agree = {}, {}, []
    for record in manifest["episodes"]:
        name = record["map"]
        if name not in maps:
            maps[name] = Occupancy(name)
        data = np.load(root / record["file"])
        states = data["states"].astype(np.float64)
        hit = np.flatnonzero(maps[name].contacts(states))
        contact = int(hit[0]) if len(hit) else None
        entry = {"contact": contact, "source": "body outline vs map"}
        if "/cl-" in record["file"] or record["file"].startswith("cl-"):
            flag = sim_flag(args.results, record["file"])
            if flag != "missing":
                agree.append((record["file"], contact, flag))
                if flag is not None and (contact is None or flag < contact):
                    entry = {"contact": flag, "source": "sim collision flag"}
        crash_step = entry["contact"]
        if crash_step is None and record["reason"] == "crash":
            crash_step = len(states) - 1
        entry["crash_step"] = crash_step
        out[record["file"]] = entry
    (root / "contacts.json").write_text(json.dumps({
        "rule": "first step where the car body outline (rear-axle frame x -0.125..0.455 m, y +-0.155 m) touches an "
                "occupied map cell; for closed-loop episodes the sim collision flag if earlier. crash_step = contact, "
                "or the last step of episodes the collector ended for LiDAR clearance < 0.18 m",
        "episodes": out}, indent=1) + "\n")
    by_reason = {}
    for record in manifest["episodes"]:
        key = ("closed-loop" if "/cl-" in record["file"] else "collector", record["reason"])
        total, hits, cut = by_reason.get(key, (0, 0, 0))
        entry = out[record["file"]]
        frames = 0 if entry["contact"] is None else record["steps"] + 1 - entry["contact"]
        by_reason[key] = (total + 1, hits + (entry["contact"] is not None), cut + frames)
    print("source / end reason: episodes, with a wall contact, frames removed (contact frame and after)")
    for key, (total, hits, cut) in sorted(by_reason.items()):
        print(f"  {key[0]:11s} {key[1]:14s} {total:5d}  {hits:5d}  {cut:6d}")
    if agree:
        both = [(f, c, s) for f, c, s in agree if c is not None or s is not None]
        print(f"closed-loop episodes checked against the sim flag: {len(agree)}; with a contact by either: {len(both)}")
        for f, c, s in both:
            print(f"  {f}: body outline {c}, sim flag {s}")


if __name__ == "__main__":
    main()
