"""Offline evaluation of the world model on held-out sim data (plan step 3).

State estimation (encoder vs EKF vs kinematic slip), 1 s prediction error (bicycle, TAs' BNN prior, residual prior,
ours), adaptation after mid-run changes, and whether z separates friction and mass.

    python -m sit_lmpc.world.evaluate --encoder runs/world-encoder --out results/world-eval
"""

import argparse
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import yaml
from flax import serialization

from dyna_dist.data.transforms import increments as state_increments
from dyna_dist.inference.posterior import _function_step, baseline_increments, components
from dyna_dist.io.checkpoint import load_model
from dyna_dist.models.features import prior_moments_and_variance
from dyna_dist.models.st import step as st_step
from sit_lmpc.world.data import WorldData
from sit_lmpc.world.ekf import nominal_parameters, run_ekf
from sit_lmpc.world.model import Decoder, Encoder, full_state, hidden_from_raw

HORIZON = 10
BURN = 10
STRIDE = 5
WHEELBASE = 0.15875 + 0.17145
LR = 0.17145


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def make_rollout(transition):
    @jax.jit
    def rollout(states, controls, z):
        def one(state, segment, coefficients):
            def advance(x, u):
                y = _function_step(transition, x, u, coefficients)
                return y, y
            return jax.lax.scan(advance, state, segment)[1]
        return jax.vmap(one)(states, controls, z)
    return rollout


def make_bicycle():
    parameters = nominal_parameters()

    @jax.jit
    def rollout(states, controls):
        def one(state, segment):
            def advance(x, u):
                y = st_step(x, u, parameters)
                return y, y
            return jax.lax.scan(advance, state, segment)[1]
        return jax.vmap(one)(states, controls)
    return rollout


def make_kalman(transition, walk_std=0.05, noise_scale=1.0):
    """Per-step closed-form z: the TAs' Gaussian latent update run every step, with random-walk forgetting."""
    prior_mean, prior_factor, variance = prior_moments_and_variance(transition.parameters, transition.config)
    dtype = prior_mean.dtype
    prior_cov = prior_factor @ prior_factor.T
    walk = jnp.asarray(walk_std ** 2, dtype) * jnp.diag(jnp.diag(prior_cov))
    noise = jnp.diag(variance).astype(dtype) * jnp.asarray(noise_scale, dtype)
    eye = jnp.eye(len(prior_mean), dtype=dtype)

    @jax.jit
    def run(states, controls):
        states, controls = jnp.asarray(states, dtype), jnp.asarray(controls, dtype)
        before = states[:-1]
        b, basis = components(transition, before, controls)
        observed = transition.normalizer.normalize_targets(
            state_increments(before, states[1:]) - baseline_increments(transition, before, controls))
        residual = observed - b

        def update(carry, inputs):
            mean, covariance = carry
            row_basis, row_residual = inputs
            covariance = covariance + walk
            innovation_cov = row_basis @ covariance @ row_basis.T + noise
            gain = covariance @ row_basis.T @ jnp.linalg.inv(innovation_cov)
            mean = mean + gain @ (row_residual - row_basis @ mean)
            covariance = (eye - gain @ row_basis) @ covariance
            return (mean, covariance), mean

        _, means = jax.lax.scan(update, (prior_mean, prior_cov), (basis, residual))
        return jnp.concatenate((prior_mean[None], means))

    return run


def hand_built_state(item):
    """What the TAs' pipeline has on the car: PF pose, wheel speed, gyro, commanded steering, kinematic slip."""
    states, observations, pose = item["states"], item["observations"], item["pose"]
    steering = states[:, 2]
    slip = np.arctan(LR / WHEELBASE * np.tan(steering))
    return np.column_stack((pose[:, 0], pose[:, 1], steering, observations[:, 3], pose[:, 2], observations[:, 2],
                            slip)).astype(np.float32)


def errors(predicted, true):
    predicted = np.where(np.isfinite(predicted), predicted, np.nan)
    position = np.linalg.norm(predicted[:, -1, :2] - true[:, -1, :2], axis=-1)
    heading = np.abs(wrap(predicted[:, -1, 4] - true[:, -1, 4]))
    speed = np.abs(predicted[:, -1, 3] - true[:, -1, 3])
    rate = np.sqrt(np.mean((predicted[..., 5] - true[..., 5]) ** 2, axis=-1))
    return np.stack((position, heading, speed, rate), axis=-1)


def load_encoder(run, latent, observation_dim):
    config = yaml.safe_load((Path(run) / "config.yaml").read_text())
    encoder = Encoder(config["hidden"], latent)
    decoder = Decoder(config["hidden"])
    template = {
        "encoder": encoder.init(jax.random.key(0), jnp.zeros((1, 2, observation_dim))),
        "decoder": decoder.init(jax.random.key(1), jnp.zeros((1, 4)), jnp.zeros((1, latent)), jnp.zeros((1, 2)),
                                jnp.zeros((1, 4))),
        "sensor_log_std": jnp.zeros(4),
    }
    params = serialization.from_bytes(template, (Path(run) / "best.msgpack").read_bytes())
    scale = np.load(Path(run) / "observation_scale.npz")
    return encoder, params, scale["mean"], scale["std"], config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", default="runs/world-encoder")
    parser.add_argument("--bnn", default="runs/dynamics-world/best_refit")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="results/world-eval")
    parser.add_argument("--ekf-items", type=int, default=60, help="trajectories to run the (slow, per-step) EKF on")
    parser.add_argument("--max-items", type=int, default=0, help="evaluate only this many trajectories (0 = all)")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load((Path(args.encoder) / "config.yaml").read_text())
    transition = load_model(config["transition"])
    bnn = load_model(args.bnn)
    latent = transition.config.latent_dimension
    residual_prior = np.asarray(prior_moments_and_variance(transition.parameters, transition.config)[0])
    bnn_prior = np.asarray(prior_moments_and_variance(bnn.parameters, bnn.config)[0])
    data = WorldData(config["dataset"], args.split)
    if args.max_items:
        data.items = data.items[: args.max_items // 2] + [i for i in data.items if i["kind"] == "change"][: args.max_items // 2]
    train = WorldData(config["dataset"], "train", include_changes=False)
    encoder, params, mean, std, _ = load_encoder(args.encoder, latent, data.items[0]["observations"].shape[-1])
    apply = jax.jit(lambda o: encoder.apply(params["encoder"], o))
    residual_rollout = make_rollout(transition)
    bnn_rollout = make_rollout(bnn)
    bicycle_rollout = make_bicycle()
    kalman = make_kalman(transition)
    episodes = {e["episode_id"]: e for e in data.dataset.manifest.get("change_episodes", [])}
    instances = {m["instance_id"]: m for m in data.dataset.instances}

    prior_factor = np.asarray(prior_moments_and_variance(transition.parameters, transition.config)[1])

    def encode(item):
        observations = (item["observations"] - mean) / std
        _, raw, w_mean, w_log_std, _ = apply(observations[None])
        w_mean = np.asarray(w_mean)[0]
        z_mean = residual_prior + w_mean @ prior_factor.T
        return np.asarray(hidden_from_raw(raw))[0], z_mean, np.asarray(w_log_std)[0]

    estimation = {"encoder": [], "ekf": [], "kinematic": []}
    prediction = {name: {"instance": [], "change": []} for name in
                  ("bicycle", "bnn_prior", "residual_prior", "ours", "ours_true_state", "bicycle_est",
                   "bnn_prior_est", "residual_prior_est", "kf_oracle", "kf_estimated", "kf_ekf", "ours_on_ekf_items")}
    adaptation = []
    examples = []
    for index, item in enumerate(data.items):
        states, controls = item["states"], item["controls"]
        count = len(controls)
        hidden, z_mean, _ = encode(item)
        truth = states[:, [2, 3, 5, 6]]
        lateral_true = states[:, 3] * np.sin(states[:, 6])
        steps = slice(BURN, None)
        error = hidden[steps] - truth[steps]
        error[:, 3] = wrap(error[:, 3])
        estimation["encoder"].append(np.column_stack((error, hidden[steps, 1] * np.sin(hidden[steps, 3]) - lateral_true[steps])))
        kinematic_slip = np.arctan(LR / WHEELBASE * np.tan(states[:, 2]))
        estimation["kinematic"].append(np.column_stack((wrap(kinematic_slip[steps] - states[steps, 6]),
                                                        states[steps, 3] * np.sin(kinematic_slip[steps]) - lateral_true[steps])))
        ekf = None
        if index < args.ekf_items:
            sensors = item["observations"]
            rear = np.column_stack((item["pose"][:, 0] - LR * np.cos(item["pose"][:, 2]),
                                    item["pose"][:, 1] - LR * np.sin(item["pose"][:, 2]), item["pose"][:, 2]))
            ekf = run_ekf(rear, sensors[:, 2], sensors[:, 3], controls)
            ekf_hidden = ekf[:, [2, 3, 5, 6]]
            ekf_error = ekf_hidden[steps] - truth[steps]
            ekf_error[:, 3] = wrap(ekf_error[:, 3])
            estimation["ekf"].append(np.column_stack((ekf_error, ekf[steps, 3] * np.sin(ekf[steps, 6]) - lateral_true[steps])))
            if len(examples) < 12 and (item["kind"] == "change" or index % 7 == 0):
                record = {"file": item["file"], "kind": item["kind"], "true_slip": states[:, 6], "encoder_slip": hidden[:, 3],
                          "ekf_slip": ekf[:, 6], "kinematic_slip": kinematic_slip, "speed": states[:, 3],
                          "z_mean": z_mean}
                if "conditions" in item:
                    record["friction"] = item["conditions"][:, 0]
                    record["mass_ratio"] = item["conditions"][:, 3]
                examples.append(record)

        starts = np.arange(BURN, count - HORIZON + 1, STRIDE)
        if not len(starts):
            continue
        segments = np.stack([controls[s:s + HORIZON] for s in starts])
        targets = np.stack([states[s + 1:s + 1 + HORIZON] for s in starts])
        true_start = states[starts]
        our_start = np.asarray(full_state(jnp.asarray(item["pose"][starts]), jnp.asarray(hidden[starts])))
        z_at = z_mean[starts]
        estimated = hand_built_state(item)
        est_start = estimated[starts]
        kf_true = np.asarray(kalman(jnp.asarray(states), jnp.asarray(controls)))[starts]
        kf_est = np.asarray(kalman(jnp.asarray(estimated), jnp.asarray(controls)))[starts]
        kind = item["kind"]
        results = {
            "bicycle": np.asarray(bicycle_rollout(jnp.asarray(true_start), jnp.asarray(segments))),
            "bnn_prior": np.asarray(bnn_rollout(jnp.asarray(true_start), jnp.asarray(segments),
                                                jnp.broadcast_to(bnn_prior, (len(starts), len(bnn_prior))))),
            "residual_prior": np.asarray(residual_rollout(jnp.asarray(true_start), jnp.asarray(segments),
                                                          jnp.broadcast_to(residual_prior, (len(starts), latent)))),
            "ours": np.asarray(residual_rollout(jnp.asarray(our_start), jnp.asarray(segments), jnp.asarray(z_at))),
            "ours_true_state": np.asarray(residual_rollout(jnp.asarray(true_start), jnp.asarray(segments), jnp.asarray(z_at))),
            "bicycle_est": np.asarray(bicycle_rollout(jnp.asarray(est_start), jnp.asarray(segments))),
            "bnn_prior_est": np.asarray(bnn_rollout(jnp.asarray(est_start), jnp.asarray(segments),
                                                    jnp.broadcast_to(bnn_prior, (len(starts), len(bnn_prior))))),
            "residual_prior_est": np.asarray(residual_rollout(jnp.asarray(est_start), jnp.asarray(segments),
                                                              jnp.broadcast_to(residual_prior, (len(starts), latent)))),
            "kf_oracle": np.asarray(residual_rollout(jnp.asarray(true_start), jnp.asarray(segments), jnp.asarray(kf_true))),
            "kf_estimated": np.asarray(residual_rollout(jnp.asarray(est_start), jnp.asarray(segments), jnp.asarray(kf_est))),
        }
        if ekf is not None:
            ekf_states = ekf.astype(np.float32)
            kf_ekf = np.asarray(kalman(jnp.asarray(ekf_states), jnp.asarray(controls)))[starts]
            results["kf_ekf"] = np.asarray(residual_rollout(jnp.asarray(ekf_states[starts]), jnp.asarray(segments),
                                                            jnp.asarray(kf_ekf)))
            results["ours_on_ekf_items"] = results["ours"]
        per_model = {name: errors(value, targets) for name, value in results.items()}
        for name, value in per_model.items():
            prediction[name][kind].append(value)
        if kind == "change":
            episode = episodes[item["source"]]
            since = (starts - episode["change_step"]) * 0.1
            for row in range(len(starts)):
                adaptation.append({"kind": episode["kind"], "since": float(since[row]),
                                   **{name: float(per_model[name][row, 0]) for name in per_model}})

    names = ["steering", "speed", "yaw_rate", "slip", "lateral_velocity"]
    summary = {"state_estimation_rmse": {}, "prediction_1s": {}, "adaptation": {}, "z_separation": {}}
    for name, rows in estimation.items():
        if not rows:
            continue
        stacked = np.concatenate(rows)
        if name == "kinematic":
            summary["state_estimation_rmse"][name] = {"slip": float(np.sqrt(np.mean(stacked[:, 0] ** 2))),
                                                      "lateral_velocity": float(np.sqrt(np.mean(stacked[:, 1] ** 2)))}
        else:
            summary["state_estimation_rmse"][name] = {n: float(np.sqrt(np.mean(stacked[:, i] ** 2))) for i, n in enumerate(names)}
    for name, kinds in prediction.items():
        summary["prediction_1s"][name] = {}
        for kind, rows in kinds.items():
            if rows:
                stacked = np.concatenate(rows)
                summary["prediction_1s"][name][kind] = {
                    "position_m": float(np.nanmean(stacked[:, 0])), "heading_rad": float(np.nanmean(stacked[:, 1])),
                    "speed_mps": float(np.nanmean(stacked[:, 2])), "yaw_rate_rmse": float(np.nanmean(stacked[:, 3])),
                    "position_median_m": float(np.nanmedian(stacked[:, 0])),
                    "nonfinite_rollouts": int(np.sum(~np.isfinite(stacked[:, 0]))), "starts": int(len(stacked))}
    bins = [(-20.0, 0.0), (0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 30.0)]
    for kind in ("step", "patch"):
        summary["adaptation"][kind] = {}
        for low, high in bins:
            rows = [r for r in adaptation if r["kind"] == kind and low <= r["since"] < high]
            if rows:
                summary["adaptation"][kind][f"{low:g}..{high:g} s"] = {
                    name: float(np.nanmean([r[name] for r in rows if name in r])) for name in prediction
                    if any(name in r for r in rows)} | {"starts": len(rows)}

    def z_features(dataset):
        features, labels = [], []
        for item in dataset.items:
            if item["kind"] != "instance" or len(item["controls"]) < 30:
                continue
            _, z_mean, _ = encode(item)
            member = instances[item["source"]]
            features.append(z_mean[-20:].mean(axis=0))
            labels.append([member["friction"], member["mass_ratio"]])
        return np.asarray(features), np.asarray(labels)

    train_x, train_y = z_features(train)
    test_x, test_y = z_features(data)
    design = np.column_stack((train_x, np.ones(len(train_x))))
    weights, *_ = np.linalg.lstsq(design, train_y, rcond=None)
    predicted = np.column_stack((test_x, np.ones(len(test_x)))) @ weights
    for column, label in enumerate(("friction", "mass_ratio")):
        residual = test_y[:, column] - predicted[:, column]
        total = test_y[:, column] - test_y[:, column].mean()
        summary["z_separation"][label] = {"r2_test": float(1 - np.sum(residual ** 2) / np.sum(total ** 2)),
                                          "rmse_test": float(np.sqrt(np.mean(residual ** 2))),
                                          "trajectories_train": int(len(train_x)), "trajectories_test": int(len(test_x))}

    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    raw = {f"prediction/{name}/{kind}": np.concatenate(rows) for name, kinds in prediction.items()
           for kind, rows in kinds.items() if rows}
    raw.update({f"estimation/{name}": np.concatenate(rows) for name, rows in estimation.items() if rows})
    raw.update({"z/test_x": test_x, "z/test_y": test_y, "z/test_predicted": predicted, "z/weights": weights})
    np.savez_compressed(out / "raw.npz", **raw)
    (out / "adaptation.json").write_text(json.dumps(adaptation) + "\n")
    np.savez_compressed(out / "examples.npz", **{f"{i}/{k}": v for i, e in enumerate(examples)
                                                  for k, v in e.items() if not isinstance(v, str)})
    (out / "examples.json").write_text(json.dumps([{"file": e["file"], "kind": e["kind"]} for e in examples]) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
