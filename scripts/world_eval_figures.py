"""Figures for the world-model offline evaluation (sit_lmpc.world.evaluate output).

    python scripts/world_eval_figures.py --eval results/world-eval-v2 --compare results/world-eval-v1
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
OURS = "#2a78d6"
BASE = "#a9a8a2"
SERIES = {"ours": "#2a78d6", "ekf": "#eb6834", "kinematic": "#1baf7a", "bnn_prior_est": "#eb6834",
          "kf_estimated": "#1baf7a", "bicycle_est": "#eda100", "kf_ekf": "#e87ba4"}
LABELS = {
    "ours": "Ours (encoder, sensors only)",
    "bnn_prior_est": "TAs' BNN, no adaptation",
    "residual_prior_est": "Our transition, no adaptation",
    "bicycle_est": "Bicycle model",
    "kf_estimated": "Per-step closed-form z (hand-built estimate)",
    "kf_ekf": "Per-step closed-form z (EKF estimate)",
    "kf_oracle": "Per-step closed-form z (TRUE state, oracle)",
    "ours_true_state": "Ours' z from the TRUE state",
}
REALISTIC = ["ours", "bnn_prior_est", "residual_prior_est", "bicycle_est", "kf_estimated", "kf_ekf"]


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def prediction_bars(raw, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False, constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, kind, title in zip(axes, ("instance", "change"), ("Constant conditions", "Grip/mass change episodes")):
        names = [n for n in REALISTIC if f"prediction/{n}/{kind}" in raw]
        medians = [np.nanmedian(raw[f"prediction/{n}/{kind}"][:, 0]) for n in names]
        means = [np.nanmean(raw[f"prediction/{n}/{kind}"][:, 0]) for n in names]
        order = np.argsort(medians)[::-1]
        y = np.arange(len(names))
        colors = [OURS if names[i] == "ours" else BASE for i in order]
        ax.barh(y, [medians[i] for i in order], color=colors, height=0.6)
        for row, i in enumerate(order):
            count = len(raw[f"prediction/{names[i]}/{kind}"])
            ax.text(medians[i], row, f"  {medians[i]:.2f} m (mean {means[i]:.2f}, n={count})", va="center",
                    fontsize=8.5, color=INK_2)
        ax.set_yticks(y, [LABELS[names[i]] for i in order], fontsize=9, color=INK)
        oracle = raw.get(f"prediction/kf_oracle/{kind}")
        if oracle is not None:
            value = np.nanmedian(oracle[:, 0])
            ax.axvline(value, color=INK_2, linestyle="--", linewidth=1)
            ax.text(value, -0.75, f" oracle, true state: {value:.3f} m", fontsize=8, color=INK_2, va="center")
        ax.set_xlabel("1 s position error, median [m]", color=INK_2)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlim(0, max(medians) * 1.7)
        ax.set_ylim(-1.2, len(names) - 0.5)
        style(ax)
    fig.suptitle("Where will the car be in 1 s? Realistic inputs (sensors / estimates, no true state)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    save(fig, out, "1_prediction_realistic")


def adaptation_curves(adaptation, out):
    bins = np.array([-5, -2, -1, 0, 0.5, 1, 1.5, 2, 3, 5, 8, 12])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True, constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    models = ["ours", "bnn_prior_est", "bicycle_est"]
    for ax, kind, title in zip(axes, ("step", "patch"), ("Step change (friction and/or mass)", "Low-grip patch")):
        rows = [r for r in adaptation if r["kind"] == kind]
        centers = (bins[:-1] + bins[1:]) / 2
        for model in models:
            values = []
            for low, high in zip(bins[:-1], bins[1:]):
                chosen = [r[model] for r in rows if low <= r["since"] < high and model in r]
                values.append(np.nanmedian(chosen) if len(chosen) >= 3 else np.nan)
            ax.plot(centers, values, color=SERIES.get(model, BASE), linewidth=2, marker="o", markersize=4,
                    label=LABELS[model])
        ax.axvline(0, color=INK_2, linestyle="--", linewidth=1)
        ax.annotate("change", (0, 1), xycoords=("data", "axes fraction"), xytext=(4, -4), textcoords="offset points",
                    fontsize=8, color=INK_2, va="top")
        ax.set_xlabel("time since the change [s]", color=INK_2)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        style(ax)
    axes[0].set_ylabel("1 s position error, median [m]", color=INK_2)
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle("Adaptation: prediction error before and after a mid-run change (per-step closed-form z is off this "
                 "scale; see figure 1)", x=0.01, ha="left", fontsize=12, color=INK)
    save(fig, out, "2_adaptation")


def estimation_bars(summary, out):
    quantities = [("slip", "slip [rad]"), ("lateral_velocity", "lateral velocity [m/s]"),
                  ("speed", "speed [m/s]"), ("yaw_rate", "yaw rate [rad/s]"), ("steering", "steering [rad]")]
    estimators = [("encoder", "Encoder (ours)", OURS), ("ekf", "EKF (bicycle model)", SERIES["ekf"]),
                  ("kinematic", "Kinematic slip formula", SERIES["kinematic"])]
    fig, axes = plt.subplots(1, len(quantities), figsize=(14, 3.6), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    rmse = summary["state_estimation_rmse"]
    for ax, (key, label) in zip(axes, quantities):
        present = [(name, text, color) for name, text, color in estimators if key in rmse.get(name, {})]
        values = [rmse[name][key] for name, _, _ in present]
        x = np.arange(len(present))
        ax.bar(x, values, color=[c for _, _, c in present], width=0.6)
        for xi, value in zip(x, values):
            ax.text(xi, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8, color=INK_2)
        ax.set_xticks(x, [text.split(" (")[0] for _, text, _ in present], rotation=20, fontsize=8)
        ax.set_title(f"RMSE: {label}", loc="left", fontsize=9.5, color=INK)
        ax.set_ylim(0, max(values) * 1.25)
        style(ax)
    fig.suptitle("State estimation from sensors (held-out test trajectories)", x=0.01, ha="left", fontsize=12, color=INK)
    save(fig, out, "3_state_estimation")


def z_separation(raw, summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    truth, predicted = raw["z/test_y"], raw["z/test_predicted"]
    for ax, column, name in zip(axes, (0, 1), ("friction", "mass_ratio")):
        ax.scatter(truth[:, column], predicted[:, column], s=10, color=OURS, alpha=0.45, edgecolors="none")
        low, high = truth[:, column].min(), truth[:, column].max()
        ax.plot([low, high], [low, high], color=INK_2, linestyle="--", linewidth=1)
        stats = summary["z_separation"][name]
        ax.set_title(f"{name.replace('_', ' ')}: R² = {stats['r2_test']:.2f}, RMSE {stats['rmse_test']:.3f}",
                     loc="left", fontsize=10, color=INK)
        ax.set_xlabel(f"true {name.replace('_', ' ')}", color=INK_2)
        ax.set_ylabel("linear read-out of encoder z", color=INK_2)
        style(ax)
    fig.suptitle("Does z encode the car's conditions? (test cars; fit on train cars)", x=0.01, ha="left",
                 fontsize=12, color=INK)
    save(fig, out, "4_z_separation")


def change_episode_from_encoder(run):
    import jax
    import yaml
    from dyna_dist.io.checkpoint import load_model
    from dyna_dist.models.features import prior_moments_and_variance
    from sit_lmpc.world.data import WorldData
    from sit_lmpc.world.evaluate import load_encoder

    config = yaml.safe_load((Path(run) / "config.yaml").read_text())
    transition = load_model(config["transition"])
    prior_mean, prior_factor, _ = (np.asarray(x) for x in prior_moments_and_variance(transition.parameters, transition.config))
    data = WorldData(config["dataset"], "test")
    item = next((i for i in data.items if i["kind"] == "change" and np.ptp(i["conditions"][:, 0]) > 0.1
                 and len(i["controls"]) >= 150), None)
    if item is None:
        return None
    encoder, params, mean, std, _ = load_encoder(run, transition.config.latent_dimension, item["observations"].shape[-1])
    _, _, w_mean, _, _ = encoder.apply(params["encoder"], ((item["observations"] - mean) / std)[None])
    z = prior_mean + np.asarray(w_mean)[0] @ prior_factor.T
    return {"z": z, "friction": item["conditions"][:, 0]}


def examples(path, raw, out, encoder_run=None):
    data = np.load(path / "examples.npz")
    meta = json.loads((path / "examples.json").read_text())
    weights = raw["z/weights"]
    constant = next((i for i, m in enumerate(meta) if m["kind"] == "instance"), None)
    change = None
    for i, m in enumerate(meta):
        if m["kind"] == "change" and f"{i}/friction" in data.files and np.ptp(data[f"{i}/friction"]) > 0.1:
            change = i
            break
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    if constant is not None:
        ax = axes[0]
        t = np.arange(len(data[f"{constant}/true_slip"])) * 0.1
        ax.plot(t, data[f"{constant}/true_slip"], color=INK, linewidth=2, label="true")
        ax.plot(t, data[f"{constant}/encoder_slip"], color=OURS, linewidth=1.6, label="encoder (ours)")
        ax.plot(t, data[f"{constant}/ekf_slip"], color=SERIES["ekf"], linewidth=1.3, label="EKF")
        ax.plot(t, data[f"{constant}/kinematic_slip"], color=SERIES["kinematic"], linewidth=1.3, label="kinematic formula")
        ax.set_xlabel("time [s]", color=INK_2)
        ax.set_ylabel("slip angle [rad]", color=INK_2)
        ax.set_title("Slip estimate on one test trajectory", loc="left", fontsize=10.5, color=INK)
        ax.legend(frameon=False, fontsize=8.5)
        style(ax)
    episode = None
    if change is not None:
        episode = {"z": data[f"{change}/z_mean"], "friction": data[f"{change}/friction"]}
    elif encoder_run is not None:
        episode = change_episode_from_encoder(encoder_run)
    if episode is not None:
        ax = axes[1]
        z = episode["z"]
        estimate = np.column_stack((z, np.ones(len(z)))) @ weights
        t = np.arange(len(z)) * 0.1
        friction = episode["friction"]
        ax.step(np.arange(len(friction)) * 0.1, friction, where="post", color=INK, linewidth=2, label="true friction")
        ax.axvspan(0, 1.0, color=GRID, alpha=0.8, linewidth=0)
        ax.annotate("encoder warm-up", (0.5, 0.02), xycoords=("data", "axes fraction"), ha="center", fontsize=8, color=INK_2)
        ax.plot(t, estimate[:, 0], color=OURS, linewidth=1.6, label="friction read out of encoder z")
        ax.set_ylim(min(friction.min(), estimate[10:, 0].min()) - 0.15, max(friction.max(), estimate[10:, 0].max()) + 0.15)
        ax.set_xlabel("time [s]", color=INK_2)
        ax.set_ylabel("friction μ", color=INK_2)
        ax.set_title("A change episode: does z follow the change?", loc="left", fontsize=10.5, color=INK)
        ax.legend(frameon=False, fontsize=8.5)
        style(ax)
    save(fig, out, "5_examples")


def flaw_check(current, previous, out, current_label, previous_label):
    fig, ax = plt.subplots(figsize=(7.5, 3.6), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    groups = [("residual_prior", "prior z, true state"), ("ours_true_state", "encoder z, true state"),
              ("ours", "encoder z + encoder state")]
    width = 0.36
    x = np.arange(len(groups))
    for offset, (raw, label, color) in zip((-width / 2, width / 2),
                                           ((previous, previous_label, BASE), (current, current_label, OURS))):
        if raw is None:
            continue
        values = [np.nanmedian(np.concatenate([raw[f"prediction/{g}/{k}"][:, 0] for k in ("instance", "change")
                                               if f"prediction/{g}/{k}" in raw])) for g, _ in groups]
        bars = ax.bar(x + offset, values, width=width * 0.92, color=color, label=label)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(x, [text for _, text in groups], fontsize=9)
    ax.set_ylabel("1 s position error, median [m]", color=INK_2)
    ax.set_title("Flaw check: from the true state, the encoder's z must not be worse than the prior's",
                 loc="left", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    save(fig, out, "6_flaw_check")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", default="results/world-eval-v2")
    parser.add_argument("--compare", default=None, help="an earlier evaluation folder for the flaw check")
    parser.add_argument("--encoder", default=None, help="encoder run, to draw a change episode when none was saved")
    args = parser.parse_args()
    path = Path(args.eval)
    out = path / "figures"
    out.mkdir(exist_ok=True)
    raw = dict(np.load(path / "raw.npz"))
    summary = json.loads((path / "summary.json").read_text())
    adaptation = json.loads((path / "adaptation.json").read_text())
    prediction_bars(raw, out)
    adaptation_curves(adaptation, out)
    estimation_bars(summary, out)
    z_separation(raw, summary, out)
    if "z/weights" in raw:
        examples(path, raw, out, args.encoder)
    previous = dict(np.load(Path(args.compare) / "raw.npz")) if args.compare and (Path(args.compare) / "raw.npz").exists() else None
    flaw_check(raw, previous, out, path.name, Path(args.compare).name if args.compare else "")
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
