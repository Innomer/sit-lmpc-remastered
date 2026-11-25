"""Replay an RSSM crash through the v3 encoder model: does v3's residual transition predict the spin the RSSM missed?

    JAX_PLATFORMS=cpu python scripts/replay_v3_on_crash.py results/Spielberg/rssm-latent-v7-smoke

At every step of the crashed attempt, roll 1.5 s ahead with the applied controls, from the true state:
  v3 (encoder z)   residual transition with z from the v3 encoder run on the attempt's sensor features
  v3 (prior z)     residual transition with the population-prior z (no adaptation)
  rssm             RSSM from its logged memory and latent (as in replay_rssm_crash.py)
  bicycle          nominal single-track model
"""

import argparse
import glob
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import yaml

from dyna_dist.io.checkpoint import load_model
from dyna_dist.models.features import prior_moments_and_variance
from sit_lmpc.config import load_config
from sit_lmpc.experiments.runner import build_vehicle_dynamics
from sit_lmpc.world.evaluate import load_encoder, make_bicycle, make_rollout
from sit_lmpc.world.rssm_data import attempt_contact
from sit_lmpc.world.rssm_runtime import RSSMFilter


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--encoder", default="runs/world-encoder-v3")
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--rssm", default=None,
                        help="evaluate this RSSM run instead of the one that drove (re-filters the logged sensors and LiDAR)")
    args = parser.parse_args()
    run = Path(args.run)
    config = yaml.safe_load((run / "config.yaml").read_text())
    _, st = build_vehicle_dynamics(load_config(str(run / "config.yaml")))
    rssm = RSSMFilter(args.rssm or config["planner"]["dynamics"]["rssm"], st.params, st.action_scale)
    encoder_config = yaml.safe_load((Path(args.encoder) / "config.yaml").read_text())
    transition = load_model(encoder_config["transition"])
    mean, factor, _ = prior_moments_and_variance(transition.parameters, transition.config)
    prior_mean, prior_factor = np.asarray(mean), np.asarray(factor)
    encoder, params, enc_mean, enc_std, _ = load_encoder(args.encoder, transition.config.latent_dimension, 9)
    v3_rollout = make_rollout(transition)
    bicycle_rollout = make_bicycle()

    true = np.load(run / "encoder_true_states.npy")
    observations = np.load(run / "encoder_observations.npy")
    memory = np.load(run / "rssm_memory.npy")
    latent = np.load(run / "encoder_z.npy")
    if args.rssm:
        lidar = np.load(run / "rssm_lidar.npy")
        memory, latent = memory.copy(), latent.copy()
    attempt = np.load(sorted(glob.glob(str(run / "trajectories" / "failure" / "*.npz")))[-1], allow_pickle=True)
    controls = attempt["controls"]
    n = len(attempt["cartesian"])
    offset = len(true) - n
    contact = attempt_contact(attempt)
    valid = n if contact is None else contact
    if args.rssm:
        yaw = true[:, 4]
        rear = np.column_stack((true[:, 0] - 0.17145 * np.cos(yaw), true[:, 1] - 0.17145 * np.sin(yaw), yaw))
        rssm.reset()
        for k in range(n):
            rssm.step_features(observations[offset + k], lidar[offset + k], rear[offset + k],
                               controls[k - 1] if k else np.zeros(2))
            memory[offset + k] = np.asarray(rssm.h)
            latent[offset + k] = np.asarray(rssm.s)
    normalized = (observations[offset:offset + n] - enc_mean) / enc_std
    _, _, w_mean, _, _ = encoder.apply(params["encoder"], jnp.asarray(normalized, jnp.float32)[None])
    z_encoder = prior_mean + np.asarray(w_mean)[0] @ prior_factor.T
    scale = np.asarray(st.action_scale)

    rows = []
    starts = list(range(0, valid - args.horizon))
    for t in starts:
        state = true[offset + t]
        future = true[offset + t + 1:offset + t + 1 + args.horizon]
        u = controls[t:t + args.horizon]
        preds = {
            "v3 (encoder z)": np.asarray(v3_rollout(jnp.asarray(state[None]), jnp.asarray(u[None]),
                                                    jnp.asarray(z_encoder[t][None])))[0],
            "v3 (prior z)": np.asarray(v3_rollout(jnp.asarray(state[None]), jnp.asarray(u[None]),
                                                  jnp.asarray(prior_mean[None])))[0],
            "rssm": np.asarray(rssm.predictor().replace(h=jnp.asarray(memory[offset + t]), s=jnp.asarray(latent[offset + t]))
                               .rollout_members(jnp.asarray(state, jnp.float32), jnp.asarray(u / scale, jnp.float32))[0]),
            "bicycle": np.asarray(bicycle_rollout(jnp.asarray(state[None]), jnp.asarray(u[None])))[0],
        }
        rows.append({k: (float(np.hypot(*(p[-1, :2] - future[-1, :2]))), float(p[-1, 6])) for k, p in preds.items()}
                    | {"true_slip": float(future[-1, 6]), "t": t})
    names = ["v3 (encoder z)", "v3 (prior z)", "rssm", "bicycle"]
    print(f"{run.name} (RSSM: {args.rssm or config['planner']['dynamics']['rssm']}): {n} steps, first sim contact at "
          f"step {contact} (predictions end before it), all rollouts from the TRUE state, horizon {args.horizon * 0.1:.1f} s")
    print(" t   true slip   " + "   ".join(f"{k:>14s}" for k in names) + "    (predicted slip at 1.5 s)")
    for r in rows[-8:]:
        print(f"{r['t']:3d}   {r['true_slip']:+.2f}      " + "   ".join(f"{r[k][1]:+14.2f}" for k in names))
    print("1.5 s position error, last 10 starts: " + "  ".join(
        f"{k} {np.mean([r[k][0] for r in rows[-10:]]):.2f} m" for k in names))
    print("1.5 s position error, median over the attempt: " + "  ".join(
        f"{k} {np.median([r[k][0] for r in rows]):.2f} m" for k in names))


if __name__ == "__main__":
    main()
