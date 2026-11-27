"""TensorBoard figures for the RSSM: reconstructions, predictions and condition read-out on fixed windows."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRUE = "#0b0b0b"
MODEL = "#2a78d6"
PRED = "#d6612a"
NAMES = ("steering [rad]", "speed [m/s]", "yaw rate [rad/s]", "slip [rad]")
LR = 0.17145
LIDAR_OFFSET = 0.275


def scan_points(state, ranges, angles_deg, limit=9.5):
    yaw = state[4]
    origin = state[:2] + (LIDAR_OFFSET - LR) * np.array([np.cos(yaw), np.sin(yaw)])
    keep = ranges < limit
    bearing = yaw + np.deg2rad(angles_deg[keep])
    return origin[0] + ranges[keep] * np.cos(bearing), origin[1] + ranges[keep] * np.sin(bearing)


def figure_image(fig):
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return image


def signals(batch, aux, dt=0.1):
    states, conditions = batch["states"], batch["conditions"]
    filtered = aux["filtered"]
    t = np.arange(1, len(states)) * dt
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for i, (ax, name, index) in enumerate(zip(axes.ravel()[:4], NAMES, (2, 3, 5, 6))):
        ax.plot(t, states[1:, index], color=TRUE, linewidth=1.8, label="true")
        ax.plot(t, filtered["anchor"][:, i], color=MODEL, linewidth=1.4, label="model (anchor)")
        ax.set_title(name, fontsize=10, loc="left")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, frameon=False)
    for ax, column, name in ((axes[1, 1], 0, "grip (friction)"), (axes[1, 2], 1, "mass ratio")):
        ax.plot(t, conditions[1:, column], color=TRUE, linewidth=1.8, label="true")
        ax.plot(t, filtered["condition"][:, column], color=MODEL, linewidth=1.4, label="read from c")
        ax.set_title(name, fontsize=10, loc="left")
        ax.grid(alpha=0.3)
    axes[1, 1].legend(fontsize=8, frameon=False)
    for ax in axes[1]:
        ax.set_xlabel("time [s]", fontsize=9)
    fig.tight_layout()
    return figure_image(fig)


def lidar_and_paths(batch, aux, fov_deg=270.0):
    lidar = np.exp(batch["lidar"])
    recon = np.exp(aux["filtered"]["lidar_mean"])
    beams = lidar.shape[1]
    angles = np.linspace(-fov_deg / 2, fov_deg / 2, beams)
    fig = plt.figure(figsize=(14, 7.5))
    grid = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(grid[0, 0])
    last = len(recon) - 1
    ax.plot(angles, lidar[last + 1], color=TRUE, linewidth=1.8, label="true scan")
    ax.plot(angles, recon[last], color=MODEL, linewidth=1.4, label="reconstruction (filter)")
    starts, rollout_lidar = aux["starts"], np.exp(aux["rollout_lidar"])
    k = rollout_lidar.shape[1]
    s0 = int(starts[-1])
    ax.plot(angles, rollout_lidar[-1, -1], color=PRED, linewidth=1.2, linestyle="--",
            label=f"predicted {k * 0.1:.0f} s ahead")
    ax.plot(angles, lidar[s0 + k], color=PRED, linewidth=0.8, alpha=0.4)
    ax.set_yscale("log")
    ax.set_xlabel("beam angle [deg]", fontsize=9)
    ax.set_ylabel("range [m]", fontsize=9)
    ax.set_title("LiDAR, last step", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    vmax = np.log(10.0)
    for column, (data, title) in enumerate(((np.log(lidar[1:]), "true LiDAR over time"),
                                            (np.log(recon), "reconstructed LiDAR over time"))):
        ax = fig.add_subplot(grid[0, 1 + column])
        ax.imshow(data.T, aspect="auto", origin="lower", cmap="viridis", vmin=np.log(0.05), vmax=vmax,
                  extent=[0, len(data) * 0.1, angles[0], angles[-1]])
        ax.set_title(title + " (log range)", fontsize=10, loc="left")
        ax.set_xlabel("time [s]", fontsize=9)
        ax.set_ylabel("beam angle [deg]", fontsize=9)
    ax = fig.add_subplot(grid[1, :2])
    states = batch["states"]
    ax.plot(states[:, 0], states[:, 1], color=TRUE, linewidth=2.0, label="true path")
    ax.plot(aux["x_filtered"][:, 0], aux["x_filtered"][:, 1], color=MODEL, linewidth=1.0, alpha=0.7,
            label="filtered pose (PF)")
    for i, start in enumerate(starts):
        xs = aux["rollout_x"][i]
        path = np.concatenate([aux["x_filtered"][int(start)][None, :2], xs[:, :2]])
        ax.plot(path[:, 0], path[:, 1], color=PRED, linewidth=1.4, label="1 s predictions" if i == 0 else None)
    for n, t in enumerate(np.linspace(10, len(recon) - 1, 3).astype(int)):
        tx, ty = scan_points(states[t + 1], lidar[t + 1], angles)
        rx, ry = scan_points(states[t + 1], recon[t], angles)
        ax.scatter(tx, ty, s=7, color=TRUE, label="true scan points" if n == 0 else None, zorder=3)
        ax.scatter(rx, ry, s=7, color=MODEL, alpha=0.7, label="reconstructed scan points" if n == 0 else None, zorder=4)
        ax.plot(*states[t + 1, :2], marker="o", color=PRED, markersize=6, zorder=5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("top-down: paths, 1 s predictions, and LiDAR points at 3 moments (dots = car)", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    ax = fig.add_subplot(grid[1, 2])
    errors = []
    for i, start in enumerate(starts):
        true = states[int(start) + 1:int(start) + 1 + k, :2]
        errors.append(np.hypot(*(aux["rollout_x"][i][:, :2] - true).T))
    errors = np.asarray(errors)
    for row in errors:
        ax.plot(np.arange(1, k + 1) * 0.1, row, color=PRED, alpha=0.5)
    ax.plot(np.arange(1, k + 1) * 0.1, errors.mean(axis=0), color=PRED, linewidth=2.2, label="mean")
    ax.set_xlabel("prediction time [s]", fontsize=9)
    ax.set_ylabel("position error [m]", fontsize=9)
    ax.set_title("open-loop position error", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return figure_image(fig)
