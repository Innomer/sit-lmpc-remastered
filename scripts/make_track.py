"""Build a SIT-LMPC map prefix (map yaml/png, centerline with widths, raceline) from a pure pursuit raceline CSV.

With --map (the car's particle-filter map yaml), the centerline is the midpoint between the walls along the
raceline normals and the widths are the measured wall distances. Without it, the raceline itself is the
centerline, with --half-width on both sides and a synthetic map drawn from that band.

--plot (with --map) saves <out>/<name>_check.png: the map with the raceline, the measured centreline and the two
measured walls. Orange circles mark spots that measure much narrower than their neighbourhood (often a blob of
noise in the corridor), red circles spots where a wall was not found within --max-wall outside the outer side of a corner (often a gap in the wall:
glass, an open door, missed scans). Fix those in the map image if the printed widths disagree with a tape measure.
"""

import argparse
import math
from pathlib import Path
import numpy as np
import yaml
from PIL import Image
from scipy.spatial import cKDTree


def load_raceline(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values = [float(value) for value in line.replace(";", ",").split(",")]
        rows.append(values[:3] if len(values) >= 3 else values[:2] + [1.0])
    points = np.asarray(rows)
    if np.linalg.norm(points[0, :2] - points[-1, :2]) < 1e-6:
        points = points[:-1]
    return points


def resample(points, spacing):
    closed = np.vstack((points, points[:1]))
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(closed[:, :2], axis=0), axis=1))))
    samples = np.arange(0.0, arc[-1], spacing)
    columns = []
    for column in range(closed.shape[1]):
        columns.append(np.interp(samples, arc, closed[:, column]))
    return np.stack(columns, axis=1)


def smooth(points, window):
    count = len(points)
    result = np.zeros_like(points)
    for i in range(count):
        indices = np.arange(i - window, i + window + 1) % count
        result[i] = points[indices].mean(axis=0)
    return result


def tangents(points):
    following = np.roll(points[:, :2], -1, axis=0)
    previous = np.roll(points[:, :2], 1, axis=0)
    direction = following - previous
    return direction / np.linalg.norm(direction, axis=1)[:, None]


class OccupancyMap:
    def __init__(self, path):
        path = Path(path)
        spec = yaml.safe_load(open(path))
        image = np.asarray(Image.open(path.parent / spec["image"]).convert("L"), dtype=float)
        occupancy = image / 255.0 if spec.get("negate", 0) else (255.0 - image) / 255.0
        self.free = occupancy < spec.get("free_thresh", 0.196)
        self.resolution = float(spec["resolution"])
        self.origin = spec["origin"]
        self.image_path = path.parent / spec["image"]
        self.spec = spec

    def is_free(self, x, y):
        row = self.free.shape[0] - 1 - int(math.floor((y - self.origin[1]) / self.resolution))
        column = int(math.floor((x - self.origin[0]) / self.resolution))
        if row < 0 or column < 0 or row >= self.free.shape[0] or column >= self.free.shape[1]:
            return False
        return bool(self.free[row, column])

    def wall_distance(self, point, direction, limit=4.0, step=0.02):
        distance = 0.0
        while distance < limit and self.is_free(point[0] + direction[0] * distance, point[1] + direction[1] * distance):
            distance += step
        return distance


def measured_widths(points, occupancy, limit):
    normals = tangents(points) @ np.array([[0.0, 1.0], [-1.0, 0.0]])
    left, right = [], []
    for point, normal in zip(points[:, :2], normals):
        left.append(occupancy.wall_distance(point, normal, limit))
        right.append(occupancy.wall_distance(point, -normal, limit))
    return np.asarray(left), np.asarray(right), normals


def synthetic_map(centerline, half_width, resolution, margin):
    low = centerline[:, :2].min(axis=0) - half_width - margin
    high = centerline[:, :2].max(axis=0) + half_width + margin
    columns = int(math.ceil((high[0] - low[0]) / resolution))
    rows = int(math.ceil((high[1] - low[1]) / resolution))
    dense = resample(centerline[:, :2], resolution / 2)
    xs = low[0] + (np.arange(columns) + 0.5) * resolution
    ys = low[1] + (np.arange(rows) + 0.5) * resolution
    grid_x, grid_y = np.meshgrid(xs, ys[::-1])
    distances, _ = cKDTree(dense).query(np.stack((grid_x.ravel(), grid_y.ravel()), axis=1))
    image = np.where(distances.reshape(rows, columns) <= half_width, 254, 0).astype(np.uint8)
    spec = {"image": None, "resolution": resolution, "origin": [float(low[0]), float(low[1]), 0.0],
            "negate": 0, "occupied_thresh": 0.65, "free_thresh": 0.196}
    return image, spec


def save_check_plot(path, image, spec, raceline, centerline, max_wall):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points, right, left = centerline[:, :2], centerline[:, 2], centerline[:, 3]
    normals = tangents(points) @ np.array([[0.0, 1.0], [-1.0, 0.0]])
    left_wall = points + normals * left[:, None]
    right_wall = points - normals * right[:, None]
    width = np.minimum(left, right)
    window = 15
    padded = np.concatenate([width[-window:], width, width[:window]])
    local = np.array([np.median(padded[i:i + 2 * window + 1]) for i in range(len(width))])
    narrow = width < 0.75 * local
    heading = np.unwrap(np.arctan2(np.gradient(points[:, 1]), np.gradient(points[:, 0])))
    step = np.maximum(np.hypot(np.gradient(points[:, 0]), np.gradient(points[:, 1])), 1e-6)
    curvature = np.convolve(np.gradient(heading) / step, np.ones(9) / 9, mode="same")
    outer_right = curvature > 0.3
    outer_left = curvature < -0.3
    gap = ((left >= max_wall - 1e-6) & ~outer_left) | ((right >= max_wall - 1e-6) & ~outer_right)
    rows, columns = image.shape[:2]
    resolution, origin = float(spec["resolution"]), spec["origin"]
    extent = (origin[0], origin[0] + columns * resolution, origin[1], origin[1] + rows * resolution)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(image, cmap="gray", extent=extent, origin="upper")
    ax.plot(raceline[:, 0], raceline[:, 1], color="tab:blue", lw=1, label="raceline")
    ax.plot(points[:, 0], points[:, 1], color="tab:green", lw=1.2, label="measured centreline")
    ax.plot(left_wall[:, 0], left_wall[:, 1], ".", color="tab:purple", ms=2, label="measured left wall")
    ax.plot(right_wall[:, 0], right_wall[:, 1], ".", color="tab:cyan", ms=2, label="measured right wall")
    if narrow.any():
        ax.scatter(points[narrow, 0], points[narrow, 1], s=80, facecolors="none", edgecolors="tab:orange",
                   linewidths=1.5, label=f"much narrower than nearby ({narrow.sum()})")
    if gap.any():
        ax.scatter(points[gap, 0], points[gap, 1], s=80, facecolors="none", edgecolors="tab:red",
                   linewidths=1.5, label=f"no wall within {max_wall:.1f} m, not a corner ({gap.sum()})")
    low, high = points.min(axis=0) - 1.5, points.max(axis=0) + 1.5
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"wall distance from the centreline: min {width.min():.2f} m, 5th percentile "
                 f"{np.percentile(width, 5):.2f} m, median {np.median(width):.2f} m")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return int(narrow.sum()), int(gap.sum())


def write_raceline(path, points):
    closed = np.vstack((points, points[:1]))
    segment = np.diff(closed[:, :2], axis=0)
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(segment, axis=1))))[:-1]
    heading = np.unwrap(np.arctan2(segment[:, 1], segment[:, 0]))
    change = np.diff(np.concatenate((heading, heading[:1] + (heading[-1] - heading[0]) + (heading[1] - heading[0]))))
    curvature = change / np.maximum(np.linalg.norm(segment, axis=1), 1e-9)
    with open(path, "w") as stream:
        stream.write("# generated by scripts/make_track.py\n# raceline\n# s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2\n")
        for i in range(len(points)):
            stream.write(f"{arc[i]:.7f};{points[i, 0]:.7f};{points[i, 1]:.7f};{heading[i]:.7f};"
                         f"{curvature[i]:.7f};{points[i, 2]:.7f};0.0000000\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raceline", required=True, help="x,y[,v] CSV, as used by the lab pure pursuit")
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", default=None, help="output directory (default tracks/<name>)")
    parser.add_argument("--map", default=None, help="particle-filter map yaml in the raceline's frame")
    parser.add_argument("--half-width", type=float, default=0.4,
                        help="assumed wall distance each side of the raceline when there is no --map")
    parser.add_argument("--spacing", type=float, default=0.1)
    parser.add_argument("--smooth", type=float, default=1.0, help="centerline smoothing window in m (with --map)")
    parser.add_argument("--max-wall", type=float, default=1.0,
                        help="cap on each wall ray, so side openings and doorways barely shift the centerline")
    parser.add_argument("--car-half-width", type=float, default=0.155)
    parser.add_argument("--plot", action="store_true", help="with --map: save <out>/<name>_check.png (map, walls, suspect spots)")
    args = parser.parse_args()

    out = Path(args.out or f"tracks/{args.name}")
    out.mkdir(parents=True, exist_ok=True)
    raceline = resample(load_raceline(args.raceline), args.spacing)
    if args.map:
        occupancy = OccupancyMap(args.map)
        inside = np.mean([occupancy.is_free(x, y) for x, y in raceline[:, :2]])
        if inside < 0.95:
            raise SystemExit(f"only {inside:.0%} of the raceline lies in free space of {args.map}; "
                             "the map and the raceline are not in the same frame")
        left, right, normals = measured_widths(raceline, occupancy, args.max_wall)
        center = raceline[:, :2] + normals * ((left - right) / 2)[:, None]
        center = smooth(center, max(int(round(args.smooth / 2 / args.spacing)), 1))
        center = resample(center, args.spacing)
        left, right, _ = measured_widths(center, occupancy, args.max_wall)
        centerline = np.column_stack((center, right, left))
        image = np.asarray(Image.open(occupancy.image_path).convert("L"))
        spec = {key: occupancy.spec[key] for key in ("resolution", "origin", "negate", "occupied_thresh", "free_thresh")
                if key in occupancy.spec}
    else:
        centerline = np.column_stack((raceline[:, :2], np.full(len(raceline), args.half_width),
                                      np.full(len(raceline), args.half_width)))
        image, spec = synthetic_map(centerline, args.half_width, 0.05, 1.0)
    spec["image"] = f"{args.name}_map.png"
    Image.fromarray(image).save(out / spec["image"])
    (out / f"{args.name}_map.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    with open(out / f"{args.name}_centerline.csv", "w") as stream:
        stream.write("# x_m, y_m, w_tr_right_m, w_tr_left_m\n")
        for row in centerline:
            stream.write(f"{row[0]:.5f}, {row[1]:.5f}, {row[2]:.4f}, {row[3]:.4f}\n")
    write_raceline(out / f"{args.name}_raceline.csv", raceline)
    walls = np.minimum(centerline[:, 2], centerline[:, 3])
    length = np.sum(np.linalg.norm(np.diff(np.vstack((centerline[:, :2], centerline[:1, :2])), axis=0), axis=1))
    print(f"wrote {out}/{args.name}_*  length {length:.2f} m, {len(centerline)} centerline points")
    print(f"wall distance from centerline: min {walls.min():.2f} m, 5th percentile {np.percentile(walls, 5):.2f} m, "
          f"median {np.median(walls):.2f} m")
    print(f"suggested planner.half_width (body-corner track limit, footprint_limits on; the planner subtracts "
          f"footprint_buffer): {np.percentile(walls, 5):.2f}")
    print(f"  narrowest wall distance {np.min(walls):.2f} m; old centre-point setting (footprint_limits off, "
          f"car half-width {args.car_half_width} m): {np.percentile(walls, 5) - args.car_half_width:.2f}")
    print(f"simulation.map: {out}/{args.name}")
    if args.plot and args.map:
        narrow, gap = save_check_plot(out / f"{args.name}_check.png", image, spec, raceline, centerline, args.max_wall)
        print(f"check plot: {out}/{args.name}_check.png ({narrow} spots much narrower than nearby, {gap} with no wall "
              f"within {args.max_wall:.1f} m)")


if __name__ == "__main__":
    main()
