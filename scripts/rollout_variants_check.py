"""Lateral (e_y) error of different rollout variants on the learned laps of closed-loop runs, at 0.5 / 1.0 / 1.5 s.

    python scripts/rollout_variants_check.py runs/rssm-v8 --runs rssm-latent-slow-logged rssm-latent-v6-smoke ...

Same starting points and controls as scripts/lateral_error_check.py (every learned-lap step with 1.5 s of applied
controls ahead, stopping before the sim contact). Variants:
  v8 planner        the planner's rollout: filtered (h, s), prior mean for the fast part d, slow part c held at its
                    filtered value; starts from the true pose with steering/speed/yaw rate/slip from the anchor
  v8 sampled prior  as above but d sampled from the prior every step (one sample)
  v8 true state     the planner's rollout started from the full true state
  v3                v3 residual transition with z from the v3 encoder on the run's sensor features, from the true state
  bicycle           single-track model with the sim's own parameters (= nominal here), from the true state
  v8 true <part>    the planner's rollout with one part of the starting state (steering, speed, yaw rate or slip)
                    replaced by its true value
Also prints the starting-state estimate error (anchor vs truth) per part.
"""
PARTS = {"steering": 2, "speed": 3, "yaw rate": 5, "slip": 6}

import argparse
import glob
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import yaml

from dyna_dist.io.checkpoint import load_model
from dyna_dist.models.features import prior_moments_and_variance
from sit_lmpc.track import Track, map_path
from sit_lmpc.world.evaluate import load_encoder, make_rollout
from sit_lmpc.world.rssm import RSSM, bicycle, latent_rollout
from sit_lmpc.world.rssm_risk import LatentFilter, run_episode

sys_path = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(sys_path))
from lateral_error_check import frenet_of

HORIZON = 15
STEPS = {"0.5 s": 5, "1.0 s": 10, "1.5 s": 15}


def learned_start(run, meta):
    files = sorted(glob.glob(str(run / "trajectories" / "*" / "attempt-*.npz")), key=lambda f: Path(f).name)
    return next((a["start"] for a, f in zip(meta["attempts"], files)
                 if str(np.load(f, allow_pickle=True)["controllers"][0]) == "lmppi"), None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rssm")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--runs-root", default="results/Spielberg")
    parser.add_argument("--encoder", default="runs/world-encoder-v3")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    filt = LatentFilter(args.rssm)
    model, network, hidden = filt.model, filt.network, filt.config.hidden
    ones = jnp.ones(2, jnp.float32)
    track = Track.from_track_path(map_path("Spielberg"))

    encoder_config = yaml.safe_load((Path(args.encoder) / "config.yaml").read_text())
    transition = load_model(encoder_config["transition"])
    prior_mean, prior_factor, _ = prior_moments_and_variance(transition.parameters, transition.config)
    prior_mean, prior_factor = np.asarray(prior_mean), np.asarray(prior_factor)
    encoder, enc_params, enc_mean, enc_std, _ = load_encoder(args.encoder, transition.config.latent_dimension, 9)
    v3_rollout = make_rollout(transition)

    def v8(h, s, x, u, key, sample):
        return jax.vmap(lambda h_, s_, x_, u_, k_: latent_rollout(model, network, filt.st_params, ones, filt.dt,
                                                                 filt.integrator, h_, s_, x_, u_, k_,
                                                                 sample=sample)[0])(h, s, x, u, key)

    v8_mean = jax.jit(lambda h, s, x, u, k: v8(h, s, x, u, k, False))
    v8_sampled = jax.jit(lambda h, s, x, u, k: v8(h, s, x, u, k, True))

    @jax.jit
    def bike(x, u):
        def one(state, seq):
            def advance(y, a):
                y = bicycle(filt.st_params, y, a, filt.dt, filt.integrator)
                return y, y
            return jax.lax.scan(advance, state, seq)[1]
        return jax.vmap(one)(x, u)

    names = ("v8 planner", "v8 sampled prior", "v8 true state", "v3", "bicycle") + tuple(f"v8 true {p}" for p in PARTS)
    errors = {name: {k: [] for k in STEPS} for name in names}
    estimate_error = {part: [] for part in PARTS}
    for name in args.runs:
        run = Path(args.runs_root) / name
        episode, _, meta = run_episode(run, HORIZON)
        features = filt.features([episode])[0]
        first = learned_start(run, meta)
        if first is None:
            continue
        true, controls = meta["true"], episode["controls"]
        starts = np.arange(first, len(controls) - HORIZON + 1)
        if not len(starts):
            continue
        h, s = jnp.asarray(features[starts, :hidden]), jnp.asarray(features[starts, hidden:])
        anchors = np.asarray(model.apply(network, h, s, method=RSSM.anchor))
        x_plan = np.column_stack((true[starts, 0], true[starts, 1], anchors[:, 0], anchors[:, 1], true[starts, 4],
                                  anchors[:, 2], anchors[:, 3])).astype(np.float32)
        x_true = true[starts].astype(np.float32)
        u = np.stack([controls[t:t + HORIZON] for t in starts]).astype(np.float32)
        keys = jax.random.split(jax.random.key(0), len(starts))
        normalized = (np.asarray(episode["sensors"]) - enc_mean) / enc_std
        _, _, w_mean, _, _ = encoder.apply(enc_params["encoder"], jnp.asarray(normalized, jnp.float32)[None])
        z = prior_mean + np.asarray(w_mean)[0] @ prior_factor.T
        predictions = {
            "v8 planner": np.asarray(v8_mean(h, s, jnp.asarray(x_plan), jnp.asarray(u), keys)),
            "v8 sampled prior": np.asarray(v8_sampled(h, s, jnp.asarray(x_plan), jnp.asarray(u), keys)),
            "v8 true state": np.asarray(v8_mean(h, s, jnp.asarray(x_true), jnp.asarray(u), keys)),
            "v3": np.asarray(v3_rollout(jnp.asarray(x_true), jnp.asarray(u), jnp.asarray(z[starts], jnp.float32))),
            "bicycle": np.asarray(bike(jnp.asarray(x_true), jnp.asarray(u))),
        }
        for part, index in PARTS.items():
            x_part = x_plan.copy()
            x_part[:, index] = x_true[:, index]
            predictions[f"v8 true {part}"] = np.asarray(v8_mean(h, s, jnp.asarray(x_part), jnp.asarray(u), keys))
            estimate_error[part].append(x_plan[:, index] - x_true[:, index])
        hint = frenet_of(track, true[starts][:, None], np.zeros(len(starts)))[:, 0, 0]
        actual = frenet_of(track, np.stack([true[t + 1:t + 1 + HORIZON] for t in starts]), hint)[..., 1]
        for variant, predicted in predictions.items():
            p = frenet_of(track, predicted, hint)[..., 1]
            for label, k in STEPS.items():
                errors[variant][label].append(np.abs(actual[:, k - 1] - p[:, k - 1]))
        print(f"{name}: {len(starts)} starts", flush=True)
    report = {}
    for variant, by_step in errors.items():
        report[variant] = {label: {"median_m": float(np.median(np.concatenate(v))),
                                   "p95_m": float(np.percentile(np.concatenate(v), 95)),
                                   "n": int(sum(map(len, v)))} for label, v in by_step.items()}
    report["start_estimate_error"] = {}
    for part, values in estimate_error.items():
        v = np.concatenate(values)
        report["start_estimate_error"][part] = {"rmse": float(np.sqrt(np.mean(v ** 2))), "mean": float(v.mean()),
                                                "p95_abs": float(np.percentile(np.abs(v), 95))}
        print(f"start estimate {part:9s} rmse {report['start_estimate_error'][part]['rmse']:.3f}  mean "
              f"{v.mean():+.3f}  p95 |err| {report['start_estimate_error'][part]['p95_abs']:.3f}")
    print("variant              " + "   ".join(f"{label}: median / p95" for label in STEPS))
    for variant, r in report.items():
        if variant == "start_estimate_error":
            continue
        print(f"{variant:20s} " + "   ".join(f"{r[l]['median_m']:5.2f} / {r[l]['p95_m']:5.2f} m   " for l in STEPS))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
