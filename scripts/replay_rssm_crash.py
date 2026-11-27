"""Replay an RSSM closed-loop run: at every step of one attempt, predict 1.5 s ahead with the controls that were
actually applied and compare with what happened.

    JAX_PLATFORMS=cpu python scripts/replay_rssm_crash.py results/Spielberg/rssm-latent-slow-logged

Needs rssm_memory.npy, encoder_z.npy and encoder_true_states.npy (saved by RSSM runs). Rows:
  rssm (own estimate)   the planner's view: RSSM rollout from its memory, latent and state estimate
  rssm (true state)     RSSM rollout from the same memory and latent but the true state (dynamics error only)
  bicycle (true state)  known single-track model from the true state (the known-model planner's view)
"""

import argparse
import glob
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import yaml

from sit_lmpc.config import load_config
from sit_lmpc.experiments.runner import build_vehicle_dynamics
from sit_lmpc.world.rssm_data import attempt_contact
from sit_lmpc.world.rssm_runtime import RSSMFilter
from sit_lmpc.world.rssm import bicycle, BODY_INDEX


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--outcome", default="failure")
    parser.add_argument("--horizon", type=int, default=15)
    args = parser.parse_args()
    run = Path(args.run)
    config = yaml.safe_load((run / "config.yaml").read_text())
    planner_config = load_config(str(run / "config.yaml"))
    _, st = build_vehicle_dynamics(planner_config)
    rssm_run = config["planner"]["dynamics"]["rssm"]
    filt = RSSMFilter(rssm_run, st.params, st.action_scale)
    memory = np.load(run / "rssm_memory.npy")
    latent = np.load(run / "encoder_z.npy")
    true = np.load(run / "encoder_true_states.npy")
    attempt = np.load(sorted(glob.glob(str(run / "trajectories" / args.outcome / "*.npz")))[-1], allow_pickle=True)
    planned = attempt["cartesian"]
    controls = attempt["controls"]
    n = len(planned)
    offset = len(true) - n
    contact = attempt_contact(attempt)
    valid = n if contact is None else contact
    scale = np.asarray(st.action_scale)
    st_params = jnp.asarray(st.params, jnp.float32)
    print(f"attempt with {n} steps; first sim contact at step {contact}; trace offset {offset}; "
          f"horizon {args.horizon * 0.1:.1f} s (predictions end before the contact)")
    print(" t     s    true v   1.5 s position error [m]: rssm(own) rssm(true) bicycle(true) | slip at end: true rssm(own)")
    rows = []
    for t in range(0, valid - args.horizon):
        u = controls[t:t + args.horizon] / scale
        future = true[offset + t + 1:offset + t + 1 + args.horizon]
        predictor = filt.predictor().replace(h=jnp.asarray(memory[offset + t]), s=jnp.asarray(latent[offset + t]))
        own = np.asarray(predictor.rollout_members(jnp.asarray(planned[t], jnp.float32), jnp.asarray(u, jnp.float32))[0])
        from_true = np.asarray(predictor.rollout_members(jnp.asarray(true[offset + t], jnp.float32),
                                                         jnp.asarray(u, jnp.float32))[0])
        x = jnp.asarray(true[offset + t], jnp.float32)
        bike = []
        for k in range(args.horizon):
            x = bicycle(st_params, x, jnp.asarray(controls[t + k], jnp.float32), 0.1, 0.02)
            bike.append(np.asarray(x))
        bike = np.asarray(bike)
        errors = [float(np.hypot(*(p[-1, :2] - future[-1, :2]))) for p in (own, from_true, bike)]
        rows.append(errors)
        if t % 5 == 0 or t >= valid - args.horizon - 6:
            print(f"{t:3d} {attempt['frenet'][t, 0]:6.1f}  {true[offset + t, 3]:5.2f}       {errors[0]:6.2f}     {errors[1]:6.2f}      "
                  f"{errors[2]:6.2f}        |  {future[-1, 6]:+.2f}  {own[-1, 6]:+.2f}")
    rows = np.asarray(rows)
    print(f"\nmedian 1.5 s position error over the attempt: rssm(own) {np.median(rows[:, 0]):.2f} m, "
          f"rssm(true state) {np.median(rows[:, 1]):.2f} m, bicycle(true state) {np.median(rows[:, 2]):.2f} m")
    tail = rows[-10:]
    print(f"last 10 starts before the end: rssm(own) {tail[:, 0].mean():.2f} m, rssm(true) {tail[:, 1].mean():.2f} m, "
          f"bicycle(true) {tail[:, 2].mean():.2f} m")


if __name__ == "__main__":
    main()
