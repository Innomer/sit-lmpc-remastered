"""Interactive viewer for the world-model datasets (data/f1tenth-track, data/f1tenth-coverage).

    python scripts/view_dataset.py                                   # browse data/f1tenth-track
    python scripts/view_dataset.py --map Spielberg --reason crash    # only Spielberg episodes that ended in a crash
    python scripts/view_dataset.py --dataset data/f1tenth-coverage   # open-space coverage episodes
    python scripts/view_dataset.py --change patches --episode 3      # 4th episode with low-grip patches
    python scripts/view_dataset.py --overview                        # dataset summary from the manifest
    python scripts/view_dataset.py --map levine --episode 10 --step 120 --export episode.png

Episode view: left, the map with the episode's path coloured by speed, the car at the current step and its LiDAR
returns (1080 beams over 270 degrees, mounted 0.275 m ahead of the rear axle); right, speed, steering, yaw rate,
slip, grip and mass ratio under the car, and the controls, with a cursor at the current step. A slider moves through
the episode; buttons step through episodes (< Prev, Next >, << 10, 10 >>) and the "ep" box jumps to an episode number
(type it, press Enter). Keys (click the plot first): right / left or n / p = next / previous episode, up / down =
+10 / -10 steps, home / end = first / last step.
Runs on the CPU only.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MAPS = {"Spielberg": ROOT / "src/sit_lmpc/track/maps/Spielberg/Spielberg_map.yaml",
        "custom1": ROOT / "src/sit_lmpc/track/maps/custom1/custom1_map.yaml",
        "levine": ROOT / "tracks/levine_sim/levine_sim_map.yaml"}
LIDAR_AHEAD_OF_COG = 0.275 - 0.17145
STATE = ("x", "y", "steering", "speed", "heading", "yaw rate", "slip")


def load_manifest(dataset):
    info = json.loads((Path(dataset) / "manifest.json").read_text())
    return info["episodes"] if isinstance(info, dict) and "episodes" in info else info


def select(episodes, args):
    out = []
    for e in episodes:
        if args.map and e["map"] != args.map:
            continue
        if args.split and e["split"] != args.split:
            continue
        if args.reason and e["reason"] != args.reason:
            continue
        if args.change and e.get("change", {}).get("kind") != args.change:
            continue
        out.append(e)
    return out


def load_map(name):
    if name not in MAPS or not MAPS[name].exists():
        return None
    info = yaml.safe_load(MAPS[name].read_text())
    image = np.asarray(Image.open(MAPS[name].parent / info["image"]).convert("L"))
    h, w = image.shape
    x0, y0 = info["origin"][0], info["origin"][1]
    extent = (x0, x0 + w * info["resolution"], y0, y0 + h * info["resolution"])
    return image, extent


def lidar_points(state, scan):
    if scan is None or scan.shape[-1] != 1080:
        return None
    ranges = np.asarray(scan, float)
    angles = np.linspace(-0.75 * np.pi, 0.75 * np.pi, len(ranges))
    yaw = state[4]
    ox = state[0] + LIDAR_AHEAD_OF_COG * np.cos(yaw)
    oy = state[1] + LIDAR_AHEAD_OF_COG * np.sin(yaw)
    keep = ranges < 29.0
    return ox + ranges[keep] * np.cos(yaw + angles[keep]), oy + ranges[keep] * np.sin(yaw + angles[keep])


def describe(e):
    change = e.get("change", {})
    text = f"{e['file']}   map {e['map']}   split {e['split']}   {e['steps']} steps   ended: {e['reason']}"
    text += f"\nchange: {change.get('kind', 'none')}"
    if change.get("kind") == "step":
        text += f" ({change.get('what')}) at step {change.get('step')}"
    if change.get("kind") == "patches":
        text += f" ({len(change.get('patches', []))} patches)"
    base = e.get("base")
    if base:
        text += f"   base grip {base[0]:.2f}, mass ratio {base[3]:.2f}"
    text += f"   max speed {e.get('max_speed', float('nan')):.1f} m/s   max |slip| {e.get('max_slip', float('nan')):.2f} rad"
    driver = e.get("driver")
    if isinstance(driver, dict) and "speed_cap" in driver:
        text += f"   speed cap {driver['speed_cap']:.1f} m/s"
    return text


class Viewer:
    def __init__(self, dataset, episodes, index, step, plt):
        self.dataset, self.episodes, self.plt = Path(dataset), episodes, plt
        self.index = int(np.clip(index, 0, len(episodes) - 1))
        self.fig = plt.figure(figsize=(16, 9))
        grid = self.fig.add_gridspec(6, 2, width_ratios=(1.25, 1), left=0.05, right=0.98, top=0.88, bottom=0.1,
                                     hspace=0.35, wspace=0.15)
        self.ax_map = self.fig.add_subplot(grid[:, 0])
        self.ax_series = [self.fig.add_subplot(grid[i, 1]) for i in range(6)]
        for ax in self.ax_series[:-1]:
            ax.tick_params(labelbottom=False)
        self.title = self.fig.text(0.05, 0.94, "", fontsize=9, va="center", family="monospace")
        self.slider_ax = self.fig.add_axes((0.05, 0.03, 0.62, 0.025))
        self.slider = None
        self.maps = {}
        from matplotlib.widgets import Button, TextBox
        self.buttons = []
        for x, label, delta in ((0.70, "<< 10", -10), (0.75, "< Prev", -1), (0.80, "Next >", 1), (0.85, "10 >>", 10)):
            button = Button(self.fig.add_axes((x, 0.02, 0.045, 0.04)), label)
            button.on_clicked(lambda _, d=delta: self.move(d))
            self.buttons.append(button)
        self.goto = TextBox(self.fig.add_axes((0.935, 0.02, 0.045, 0.04)), "ep ", initial=str(self.index + 1))
        self.goto.on_submit(self.jump)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.show_episode(step)

    def move(self, delta):
        self.index = (self.index + delta) % len(self.episodes)
        self.show_episode(0)

    def jump(self, text):
        try:
            number = int(text)
        except ValueError:
            return
        self.index = int(np.clip(number - 1, 0, len(self.episodes) - 1))
        self.show_episode(0)

    def episode(self):
        e = self.episodes[self.index]
        return e, np.load(self.dataset / e["file"])

    def show_episode(self, step=0):
        from matplotlib.widgets import Slider
        e, data = self.episode()
        self.data, self.entry = data, e
        states = data["states"]
        self.steps = len(states)
        step = int(np.clip(step, 0, self.steps - 1))
        self.ax_map.clear()
        if e["map"] not in self.maps:
            self.maps[e["map"]] = load_map(e["map"])
        loaded = self.maps[e["map"]]
        if loaded is not None:
            image, extent = loaded
            self.ax_map.imshow(image, cmap="gray", extent=extent, origin="upper", vmin=0, vmax=255)
        path = self.ax_map.scatter(states[:, 0], states[:, 1], c=states[:, 3], s=4, cmap="viridis", vmin=0,
                                   vmax=max(10.0, float(states[:, 3].max())))
        if not hasattr(self, "colorbar"):
            self.colorbar = self.fig.colorbar(path, ax=self.ax_map, fraction=0.03, pad=0.01, label="speed (m/s)")
        else:
            self.colorbar.update_normal(path)
        grip = data["conditions"][:, 0]
        low = grip < grip[0] - 0.02
        if e.get("base"):
            low = grip < e["base"][0] - 0.02
        if low.any():
            self.ax_map.scatter(states[low, 0], states[low, 1], s=22, facecolors="none", edgecolors="tab:red",
                                linewidths=0.8, label="reduced grip")
        change = e.get("change", {})
        if change.get("kind") == "step" and change.get("step") is not None and change["step"] < self.steps:
            k = change["step"]
            self.ax_map.plot(states[k, 0], states[k, 1], "x", color="tab:red", ms=10, mew=2, label="condition change")
        self.ax_map.plot(states[0, 0], states[0, 1], "o", color="white", mec="black", ms=6, label="start")
        self.ax_map.plot(states[-1, 0], states[-1, 1], "s", color="tab:red" if e["reason"] == "crash" else "white",
                         mec="black", ms=6, label=f"end ({e['reason']})")
        self.lidar_plot, = self.ax_map.plot([], [], ".", color="tab:orange", ms=1.5, label="LiDAR")
        self.car_plot, = self.ax_map.plot([], [], color="tab:red", lw=2.5)
        margin = 6.0
        self.ax_map.set_xlim(states[:, 0].min() - margin, states[:, 0].max() + margin)
        self.ax_map.set_ylim(states[:, 1].min() - margin, states[:, 1].max() + margin)
        self.ax_map.set_aspect("equal")
        self.ax_map.legend(loc="upper right", fontsize=7, markerscale=2)
        self.ax_map.set_xlabel("x (m)")
        self.ax_map.set_ylabel("y (m)")
        t = np.arange(self.steps) * 0.1
        conditions = data["conditions"]
        controls = data["controls"]
        series = [("speed (m/s)", [(states[:, 3], "speed")]),
                  ("steering (rad)", [(states[:, 2], "steering")]),
                  ("yaw rate (rad/s)", [(states[:, 5], "yaw rate")]),
                  ("slip (rad)", [(states[:, 6], "slip")]),
                  ("condition", [(conditions[:, 0], "grip under the car"), (conditions[:, 3], "mass ratio")]),
                  ("controls", [(np.r_[controls[:, 0], np.nan] / 3.2, "steering rate / 3.2"),
                                (np.r_[controls[:, 1], np.nan] / 9.51, "acceleration / 9.51")])]
        self.cursors = []
        for ax, (label, lines) in zip(self.ax_series, series):
            ax.clear()
            for values, name in lines:
                ax.plot(t, values, lw=1, label=name)
            ax.set_ylabel(label, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
            if len(lines) > 1:
                ax.legend(fontsize=6, loc="upper right")
            self.cursors.append(ax.axvline(step * 0.1, color="tab:red", lw=1))
        self.ax_series[-1].set_xlabel("time (s)")
        self.title.set_text(f"[{self.index + 1}/{len(self.episodes)}]  " + describe(e))
        if hasattr(self, "goto") and self.goto.text != str(self.index + 1):
            self.goto.set_val(str(self.index + 1))
        self.slider_ax.clear()
        self.slider = Slider(self.slider_ax, "step", 0, self.steps - 1, valinit=step, valstep=1)
        self.slider.on_changed(lambda value: self.show_step(int(value)))
        self.show_step(step)

    def show_step(self, step):
        states = self.data["states"]
        s = states[step]
        scan = self.data["lidar"][step] if "lidar" in self.data.files else None
        points = lidar_points(s, scan)
        self.lidar_plot.set_data(*(points if points is not None else ([], [])))
        yaw = s[4]
        corners = np.array([[0.2835, 0.155], [0.2835, -0.155], [-0.2965, -0.155], [-0.2965, 0.155], [0.2835, 0.155]])
        x = s[0] + corners[:, 0] * np.cos(yaw) - corners[:, 1] * np.sin(yaw)
        y = s[1] + corners[:, 0] * np.sin(yaw) + corners[:, 1] * np.cos(yaw)
        self.car_plot.set_data(x, y)
        for cursor in self.cursors:
            cursor.set_xdata([step * 0.1, step * 0.1])
        self.ax_map.set_title(f"step {step}  t = {step * 0.1:.1f} s  speed {s[3]:.2f} m/s  slip {s[6]:+.3f} rad  "
                              f"grip {self.data['conditions'][step, 0]:.2f}", fontsize=9)
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        step = int(self.slider.val)
        if event.key in ("right", "left", "n", "p"):
            self.move(1 if event.key in ("right", "n") else -1)
        elif event.key in ("up", "down"):
            self.slider.set_val(int(np.clip(step + (10 if event.key == "up" else -10), 0, self.steps - 1)))
        elif event.key == "home":
            self.slider.set_val(0)
        elif event.key == "end":
            self.slider.set_val(self.steps - 1)


def overview(episodes, dataset, plt, export):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    maps = sorted({e["map"] for e in episodes})
    reasons = sorted({e["reason"] for e in episodes})
    counts = np.array([[sum(1 for e in episodes if e["map"] == m and e["reason"] == r) for r in reasons] for m in maps])
    bottom = np.zeros(len(maps))
    for j, r in enumerate(reasons):
        axes[0, 0].bar(maps, counts[:, j], bottom=bottom, label=r)
        bottom += counts[:, j]
    axes[0, 0].set_title("episodes by map and end reason")
    axes[0, 0].legend(fontsize=8)
    steps = [sum(e["steps"] for e in episodes if e["map"] == m) / 1000 for m in maps]
    axes[0, 1].bar(maps, steps)
    axes[0, 1].set_title("steps of 0.1 s (thousands)")
    kinds = sorted({e.get("change", {}).get("kind", "none") for e in episodes})
    axes[0, 2].bar(kinds, [sum(1 for e in episodes if e.get("change", {}).get("kind", "none") == k) for k in kinds])
    axes[0, 2].set_title("condition change per episode")
    for m in maps:
        sub = [e for e in episodes if e["map"] == m]
        axes[1, 0].hist([e.get("max_speed", np.nan) for e in sub], bins=30, alpha=0.5, label=m)
        axes[1, 1].hist([min(e.get("max_slip", np.nan), 2.0) for e in sub], bins=30, alpha=0.5, label=m)
        axes[1, 2].hist([e["base"][0] for e in sub if e.get("base")], bins=30, alpha=0.5, label=m)
    axes[1, 0].set_title("max speed per episode (m/s)")
    axes[1, 1].set_title("max |slip| per episode (rad, clipped at 2)")
    axes[1, 2].set_title("base grip per episode")
    for ax in axes[1]:
        ax.legend(fontsize=8)
    fig.suptitle(f"{dataset}: {len(episodes)} episodes, {sum(e['steps'] for e in episodes):,} steps")
    fig.tight_layout()
    if export:
        fig.savefig(export, dpi=120)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=str(ROOT / "data/f1tenth-track"))
    parser.add_argument("--map", help="Spielberg, custom1, levine or open")
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--reason", help="crash, lap, step_limit or envelope_limit")
    parser.add_argument("--change", help="none, step or patches")
    parser.add_argument("--episode", type=int, default=0, help="index among the selected episodes")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--overview", action="store_true")
    parser.add_argument("--export", help="save a PNG instead of opening a window")
    args = parser.parse_args()
    import matplotlib
    if args.export:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name in ("keymap.back", "keymap.forward", "keymap.home", "keymap.pan", "keymap.zoom"):
        plt.rcParams[name] = [k for k in plt.rcParams[name] if k not in ("left", "right", "home", "p", "n")]
    episodes = select(load_manifest(args.dataset), args)
    if not episodes:
        raise SystemExit("no episodes match the filters")
    if args.overview:
        overview(episodes, args.dataset, plt, args.export)
        return
    viewer = Viewer(args.dataset, episodes, args.episode, args.step, plt)
    if args.export:
        viewer.fig.savefig(args.export, dpi=120)
    else:
        plt.show()


if __name__ == "__main__":
    main()
