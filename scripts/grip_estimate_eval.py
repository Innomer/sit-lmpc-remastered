"""Compare ways to turn the encoder's per-step z into a steady grip (friction) estimate, offline on held-out sim data.

    python scripts/grip_estimate_eval.py --encoder runs/world-encoder-v3 --out results/grip-estimate

Estimators: raw read-out (as in the planner loop), 2 s moving average, confidence-weighted exponential forgetting
(weights = 1 / read-out variance from the encoder's z std), and the same with a read-out refit on per-step w.
Metrics: per-step error on constant cars, and seconds to catch a friction change in change episodes.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax import serialization

from sit_lmpc.world.data import WorldData
from sit_lmpc.world.model import Decoder, Encoder

TIMESTEP = 0.1
BURN = 10


def load(run):
    config = yaml.safe_load((run / "config.yaml").read_text())
    scale = np.load(run / "observation_scale.npz")
    readout = dict(np.load(run / "condition_readout.npz"))
    from dyna_dist.io.checkpoint import load_model
    transition = load_model(config["transition"])
    latent = transition.config.latent_dimension
    encoder = Encoder(config["hidden"], latent)
    decoder = Decoder(config["hidden"])
    observation_dim = len(scale["mean"])
    template = {
        "encoder": encoder.init(jax.random.key(0), jnp.zeros((1, 2, observation_dim))),
        "decoder": decoder.init(jax.random.key(1), jnp.zeros((1, 4)), jnp.zeros((1, latent)), jnp.zeros((1, 2)),
                                jnp.zeros((1, 4))),
        "sensor_log_std": jnp.zeros(4),
    }
    params = serialization.from_bytes(template, (run / "best.msgpack").read_bytes())["encoder"]

    @jax.jit
    def encode(observations):
        _, _, w_mean, w_log_std, _ = encoder.apply(params, observations[None])
        return w_mean[0], jnp.clip(w_log_std[0], -6.0, 2.0)

    def run_encoder(observations):
        normalized = (observations - scale["mean"]) / scale["std"]
        w_mean, w_log_std = encode(jnp.asarray(normalized, jnp.float32))
        return np.asarray(w_mean), np.exp(np.asarray(w_log_std))

    return config, readout, run_encoder


def per_step(data, run_encoder):
    out = []
    instances = {member["instance_id"]: member for member in data.dataset.instances}
    for item in data.items:
        if len(item["controls"]) < 40:
            continue
        w, s = run_encoder(item["observations"])
        steps = len(item["controls"])
        w, s = w[:steps], s[:steps]
        if "conditions" in item:
            truth = item["conditions"][:steps, 0]
        else:
            truth = np.full(steps, instances[item["source"]]["friction"], np.float32)
        out.append({"kind": item["kind"], "w": w, "s": s, "truth": truth})
    return out


def moving_average(x, seconds):
    window = max(1, int(seconds / TIMESTEP))
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty_like(x)
    for t in range(len(x)):
        start = max(0, t + 1 - window)
        out[t] = (c[t + 1] - c[start]) / (t + 1 - start)
    return out


def confidence_filter(x, variance, tau):
    keep = np.exp(-TIMESTEP / tau)
    numerator = 0.0
    denominator = 0.0
    out = np.empty_like(x)
    for t in range(len(x)):
        weight = 1.0 / max(variance[t], 1e-6)
        numerator = keep * numerator + weight * x[t]
        denominator = keep * denominator + weight
        out[t] = numerator / denominator
    return out


def estimators(track, weights, bias, refit):
    raw = track["w"] @ weights + bias
    variance = (track["s"] ** 2) @ (weights ** 2)
    refit_raw = track["w"] @ refit[:-1] + refit[-1]
    refit_variance = (track["s"] ** 2) @ (refit[:-1] ** 2)
    out = {"A raw": raw, "B 2 s average": moving_average(raw, 2.0)}
    for tau in (1.0, 2.0, 4.0):
        out[f"C confidence, tau {tau:g} s"] = confidence_filter(raw, variance, tau)
    out["D refit, raw"] = refit_raw
    for tau in (1.0, 2.0, 4.0):
        out[f"D refit + confidence, tau {tau:g} s"] = confidence_filter(refit_raw, refit_variance, tau)
    return out


def catch_time(estimate, truth, start, tolerance=0.1, hold=1.0):
    target = truth[start]
    inside = np.abs(estimate[start:] - target) <= tolerance
    need = int(hold / TIMESTEP)
    for t in range(len(inside) - need + 1):
        if inside[t:t + need].all():
            return t * TIMESTEP
    return np.nan


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", default="runs/world-encoder-v3")
    parser.add_argument("--out", default="results/grip-estimate")
    args = parser.parse_args()
    run = Path(args.encoder)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config, readout, run_encoder = load(run)
    weights, bias = readout["weights"], float(readout["bias"])

    train = per_step(WorldData(config["dataset"], "train"), run_encoder)
    x = np.concatenate([t["w"][BURN:] for t in train])
    y = np.concatenate([t["truth"][BURN:] for t in train])
    refit, *_ = np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)

    test = per_step(WorldData(config["dataset"], "test"), run_encoder)
    names = list(estimators(test[0], weights, bias, refit))
    errors = {name: [] for name in names}
    swings = {name: [] for name in names}
    catches = {name: [] for name in names}
    changes = 0
    for track in test:
        estimates = estimators(track, weights, bias, refit)
        truth = track["truth"]
        if track["kind"] == "instance":
            for name in names:
                errors[name].append(estimates[name][BURN:] - truth[BURN:])
                swings[name].append(np.std(estimates[name][BURN:]))
            continue
        jumps = np.flatnonzero(np.abs(np.diff(truth)) > 0.05)
        if not len(jumps):
            continue
        start = int(jumps[0]) + 1
        if len(truth) - start < 20:
            continue
        changes += 1
        for name in names:
            catches[name].append(catch_time(estimates[name], truth, start))

    rows = []
    for name in names:
        error = np.concatenate(errors[name])
        times = np.asarray(catches[name])
        caught = np.isfinite(times)
        rows.append({
            "estimator": name,
            "constant_rmse": float(np.sqrt(np.mean(error ** 2))),
            "constant_bias": float(np.mean(error)),
            "within_trajectory_std": float(np.mean(swings[name])),
            "changes": int(len(times)),
            "caught": int(caught.sum()),
            "catch_median_s": float(np.median(times[caught])) if caught.any() else None,
            "catch_p90_s": float(np.percentile(times[caught], 90)) if caught.any() else None,
        })
    (out / "summary.json").write_text(json.dumps({"encoder": str(run), "change_episodes": changes, "rows": rows,
                                                  "refit_weights": refit.tolist()}, indent=2) + "\n")
    np.savez(out / "refit_readout.npz", weights=refit[:-1], bias=refit[-1])
    np.savez(run / "grip_readout.npz", weights=refit[:-1], bias=refit[-1])
    print(f"{'estimator':34s} {'RMSE':>6s} {'bias':>6s} {'swing':>6s} {'caught':>8s} {'median s':>8s} {'p90 s':>6s}")
    for r in rows:
        median = "-" if r["catch_median_s"] is None else f"{r['catch_median_s']:.1f}"
        p90 = "-" if r["catch_p90_s"] is None else f"{r['catch_p90_s']:.1f}"
        print(f"{r['estimator']:34s} {r['constant_rmse']:6.3f} {r['constant_bias']:6.3f} {r['within_trajectory_std']:6.3f} "
              f"{r['caught']:3d}/{r['changes']:<4d} {median:>8s} {p90:>6s}")


if __name__ == "__main__":
    main()
