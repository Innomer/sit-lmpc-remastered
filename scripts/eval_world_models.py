"""Offline comparison on the on-track test split: bicycle model, v3 encoder model and the RSSM world model.

    python scripts/eval_world_models.py --rssm runs/rssm-v3 --encoder runs/world-encoder-v3 --out results/wm-eval

Windows of --window steps from every test episode. Each estimator (EKF, v3 encoder, RSSM filter) starts fresh at the
window start and gets --burn steps to settle; from every 1 s after that, each model predicts 1 s ahead with the true
controls, starting from its own estimate:
  bicycle (EKF)    nominal single-track model from the EKF estimate (what the car would have)
  bicycle (truth)  nominal single-track model from the true state (reference: model error only)
  v3 encoder       residual transition with the encoder's z, from PF pose + encoder state
  RSSM             latent rollout (prior mean) from the filtered state (posterior mean, no sampling)
Also state estimation (EKF vs v3 encoder vs RSSM anchor) and grip tracking (v3 read-out vs RSSM condition head).
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")

import jax
import jax.numpy as jnp
import yaml
from flax import serialization

from dyna_dist.io.checkpoint import load_model
from dyna_dist.models.features import prior_moments_and_variance
from sit_lmpc.config import load_config
from sit_lmpc.experiments.runner import build_vehicle_dynamics
from sit_lmpc.world.ekf import run_ekf
from sit_lmpc.world.evaluate import load_encoder, make_bicycle, make_rollout
from sit_lmpc.world.model import full_state, hidden_from_raw
from sit_lmpc.world.rssm import RSSMConfig, RSSM, init_params
from sit_lmpc.world.rssm_data import TrackData
from sit_lmpc.world.rssm_train import run_sequence

LR = 0.17145
HORIZON = 10
NAMES = ("steering", "speed", "yaw_rate", "slip")


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def prediction_errors(predicted, true):
    position = np.linalg.norm(predicted[:, -1, :2] - true[:, -1, :2], axis=-1)
    heading = np.abs(wrap(predicted[:, -1, 4] - true[:, -1, 4]))
    speed = np.abs(predicted[:, -1, 3] - true[:, -1, 3])
    return np.stack((position, heading, speed), axis=-1)


def summarise(values):
    values = np.asarray(values)
    finite = values[np.all(np.isfinite(values), axis=-1)]
    return {"count": int(len(values)), "nonfinite": int(len(values) - len(finite)),
            "mean": finite.mean(axis=0).tolist(), "median": np.median(finite, axis=0).tolist(),
            "p90": np.percentile(finite, 90, axis=0).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rssm", default="runs/rssm-v3")
    parser.add_argument("--encoder", default="runs/world-encoder-v3")
    parser.add_argument("--split", default="test")
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--burn", type=int, default=20)
    parser.add_argument("--out", default="results/wm-eval")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rssm_config_file = yaml.safe_load((Path(args.rssm) / "config.yaml").read_text())
    rssm_config = RSSMConfig(**rssm_config_file["model"])
    model, template = init_params(rssm_config, jax.random.key(0))
    params = serialization.from_bytes(template, (Path(args.rssm) / "best.msgpack").read_bytes())
    scale = np.load(Path(args.rssm) / "observation_scale.npz")
    obs_mean, obs_std = jnp.asarray(scale["mean"]), jnp.asarray(scale["std"])
    _, st = build_vehicle_dynamics(load_config(rssm_config_file["vehicle_config"]))
    st_params = jnp.asarray(st.params, jnp.float32)
    settings = dict(rssm_config_file["settings"], prediction_steps=HORIZON, prediction_stride=HORIZON)
    dt, integrator = float(rssm_config_file["timestep"]), float(rssm_config_file["integrator_timestep"])
    rssm_run = jax.jit(jax.vmap(lambda b: run_sequence(params, model, st_params, dt, integrator, b, jax.random.key(0),
                                                       settings, obs_mean, obs_std, sample=False)[1]))

    encoder_config = yaml.safe_load((Path(args.encoder) / "config.yaml").read_text())
    transition = load_model(encoder_config["transition"])
    latent = transition.config.latent_dimension
    residual_prior = np.asarray(prior_moments_and_variance(transition.parameters, transition.config)[0])
    prior_factor = np.asarray(prior_moments_and_variance(transition.parameters, transition.config)[1])
    encoder, encoder_params, enc_mean, enc_std, _ = load_encoder(args.encoder, latent, 9)
    encode = jax.jit(jax.vmap(lambda o: encoder.apply(encoder_params["encoder"], o[None])))
    readout = np.load(Path(args.encoder) / "condition_readout.npz")
    residual_rollout = make_rollout(transition)
    bicycle_rollout = make_bicycle()

    data = TrackData(rssm_config_file["dataset"], args.split, rssm_config.beams)
    items = [(i, s) for i, e in enumerate(data.episodes) for s in range(0, len(e["controls"]) - args.window + 1, args.window)]
    batch = data.gather(items, args.window)
    print(f"{len(items)} windows of {args.window} steps from {len(data.episodes)} {args.split} episodes", flush=True)

    rssm_aux = {}
    for start in range(0, len(items), 64):
        chunk = {k: jnp.asarray(v[start:start + 64]) for k, v in batch.items()}
        aux = jax.tree_util.tree_map(np.asarray, rssm_run(chunk))
        for key in ("x_filtered", "rollout_x", "starts"):
            rssm_aux.setdefault(key, []).append(aux[key])
        rssm_aux.setdefault("anchor", []).append(aux["filtered"]["anchor"])
        rssm_aux.setdefault("condition", []).append(aux["filtered"]["condition"])
    rssm_aux = {k: np.concatenate(v) if k != "starts" else v[0][0] for k, v in rssm_aux.items()}

    normalized = (batch["sensors"] - enc_mean) / enc_std
    _, raw, w_mean, _, _ = encode(jnp.asarray(normalized))
    v3_hidden = np.asarray(hidden_from_raw(raw))[:, 0]
    w_mean = np.asarray(w_mean)[:, 0]
    v3_z = residual_prior + w_mean @ prior_factor.T
    v3_grip = w_mean @ readout["weights"] + readout["bias"]

    starts = [t for t in range(args.burn, args.window - HORIZON + 1, HORIZON)]
    rssm_starts = {int(s): j for j, s in enumerate(rssm_aux["starts"])}
    rows = {name: {"all": [], "steady": [], "after_change": []} for name in
            ("bicycle_ekf", "bicycle_truth", "v3_encoder", "rssm")}
    by_map = {name: {} for name in rows}
    raw = {"position": {name: [] for name in rows}, "speed": {name: [] for name in rows}, "changed": [], "map": []}
    estimation = {"ekf": [], "v3_encoder": [], "rssm": []}
    grip = {"v3_readout": [], "rssm": []}
    grip_window = {"v3_readout": [], "rssm": []}
    for n, (index, window_start) in enumerate(items):
        states, controls = batch["states"][n], batch["controls"][n]
        sensors, pose, conditions = batch["sensors"][n], batch["pose"][n], batch["conditions"][n]
        name = data.episodes[index]["map"]
        rear = np.column_stack((pose[:, 0] - LR * np.cos(pose[:, 2]), pose[:, 1] - LR * np.sin(pose[:, 2]), pose[:, 2]))
        ekf = np.asarray(run_ekf(rear, sensors[:, 2], sensors[:, 3], controls))
        truth = states[:, [2, 3, 5, 6]]
        steps = slice(args.burn, None)
        for label, estimate in (("ekf", ekf[:, [2, 3, 5, 6]]), ("v3_encoder", v3_hidden[n])):
            error = estimate[steps] - truth[steps]
            error[:, 3] = wrap(error[:, 3])
            estimation[label].append(error)
        error = rssm_aux["anchor"][n][args.burn - 1:] - truth[args.burn:]
        estimation["rssm"].append(error)
        grip["v3_readout"].append(np.abs(v3_grip[n][steps] - conditions[steps, 0]))
        grip["rssm"].append(np.abs(rssm_aux["condition"][n][args.burn - 1:, 0] - conditions[args.burn:, 0]))
        grip_window["v3_readout"].append(abs(v3_grip[n][steps].mean() - conditions[steps, 0].mean()))
        grip_window["rssm"].append(abs(rssm_aux["condition"][n][args.burn - 1:, 0].mean() - conditions[args.burn:, 0].mean()))
        ekf_starts = np.stack([ekf[t] for t in starts])
        truth_starts = np.stack([states[t] for t in starts])
        v3_starts = np.stack([np.asarray(full_state(pose[t], v3_hidden[n][t])) for t in starts])
        segments = np.stack([controls[t:t + HORIZON] for t in starts])
        targets = np.stack([states[t + 1:t + 1 + HORIZON] for t in starts])
        predictions = {
            "bicycle_ekf": np.asarray(bicycle_rollout(jnp.asarray(ekf_starts), jnp.asarray(segments))),
            "bicycle_truth": np.asarray(bicycle_rollout(jnp.asarray(truth_starts), jnp.asarray(segments))),
            "v3_encoder": np.asarray(residual_rollout(jnp.asarray(v3_starts), jnp.asarray(segments),
                                                      jnp.asarray(np.stack([v3_z[n][t] for t in starts])))),
            "rssm": np.stack([rssm_aux["rollout_x"][n][rssm_starts[t]] for t in starts]),
        }
        changed = np.array([np.ptp(conditions[max(0, t - 20):t + 1 + HORIZON, 0]) > 0.05 for t in starts])
        raw["changed"].append(changed)
        raw["map"].append(np.full(len(starts), name))
        for model_name, predicted in predictions.items():
            err = prediction_errors(predicted, targets)
            rows[model_name]["all"].append(err)
            rows[model_name]["steady"].append(err[~changed])
            rows[model_name]["after_change"].append(err[changed])
            by_map[model_name].setdefault(name, []).append(err)
            raw["position"][model_name].append(err[:, 0])
            raw["speed"][model_name].append(err[:, 2])
        if n % 50 == 0:
            print(f"  window {n}/{len(items)}", flush=True)

    summary = {"windows": len(items), "window": args.window, "burn": args.burn, "horizon_s": HORIZON * 0.1,
               "prediction_columns": ["position_m", "heading_rad", "speed_mps"], "prediction": {}, "prediction_by_map": {},
               "estimation_rmse": {}, "grip_abs_error": {}}
    for model_name, groups in rows.items():
        summary["prediction"][model_name] = {g: summarise(np.concatenate(v)) for g, v in groups.items() if
                                             sum(len(x) for x in v)}
        summary["prediction_by_map"][model_name] = {m: summarise(np.concatenate(v)) for m, v in by_map[model_name].items()}
    for label, errs in estimation.items():
        e = np.concatenate(errs)
        e = e[np.all(np.isfinite(e), axis=1)]
        summary["estimation_rmse"][label] = dict(zip(NAMES, np.sqrt(np.mean(e ** 2, axis=0)).tolist()))
    for label in grip:
        summary["grip_abs_error"][label] = {"per_step_mean": float(np.mean(np.concatenate(grip[label]))),
                                            "window_average_mean": float(np.mean(grip_window[label]))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    arrays = {"changed": np.concatenate(raw["changed"]), "map": np.concatenate(raw["map"])}
    for name in rows:
        arrays[f"position_{name}"] = np.concatenate(raw["position"][name])
        arrays[f"speed_{name}"] = np.concatenate(raw["speed"][name])
    for label in estimation:
        arrays[f"estimation_{label}"] = np.concatenate(estimation[label])
    for label in grip:
        arrays[f"grip_{label}"] = np.concatenate(grip[label])
        arrays[f"grip_window_{label}"] = np.asarray(grip_window[label])
    np.savez_compressed(out / "raw.npz", **arrays)

    print("\n1 s prediction, position error at 1 s [m]: mean / median / p90")
    for model_name in rows:
        for group, s in summary["prediction"][model_name].items():
            print(f"  {model_name:14s} {group:13s} n={s['count']:5d}  {s['mean'][0]:6.3f} / {s['median'][0]:6.3f} / "
                  f"{s['p90'][0]:6.3f}   heading {s['median'][1]:.3f} rad  speed {s['median'][2]:.3f} m/s"
                  f"{'  non-finite ' + str(s['nonfinite']) if s['nonfinite'] else ''}")
    print("\nby map (median position error at 1 s [m])")
    for model_name in rows:
        print(f"  {model_name:14s} " + "  ".join(f"{m} {s['median'][0]:.3f}" for m, s in
                                                summary["prediction_by_map"][model_name].items()))
    print("\nstate estimation RMSE (steering rad, speed m/s, yaw rate rad/s, slip rad)")
    for label, r in summary["estimation_rmse"].items():
        print(f"  {label:11s} " + "  ".join(f"{k} {v:.3f}" for k, v in r.items()))
    print("\ngrip absolute error (per step / averaged over each window)")
    for label, r in summary["grip_abs_error"].items():
        print(f"  {label:11s} {r['per_step_mean']:.3f} / {r['window_average_mean']:.3f}")


if __name__ == "__main__":
    main()
