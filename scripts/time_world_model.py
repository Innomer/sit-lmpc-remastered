"""Time the world-model parts of planning, with random weights (timing does not depend on training).

    python scripts/time_world_model.py --out results/wm-timing/laptop.json

Measures, per control step (p50 / p95 over --repeats calls, each forced to finish on the device):
  st         bicycle-only rollout of all candidates (what the known-model planner already pays)
  cheap      bicycle + residual(x, u, h, s held fixed) for all candidates
  latent     full world-model rollout (h, s, x, prior sampling) for all candidates
  stage2     top-N selection + full latent rollout of N + LiDAR decode at every step + clearance penalty
  filter     one posterior update from a LiDAR scan + sensors (once per control step)
Planner total = known-model planner time (scripts/benchmark_planning.py, same samples/horizon, capped safe set)
                - st + (cheap | latent | cheap + stage2) + filter.
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

from sit_lmpc.config import load_config
from sit_lmpc.experiments.runner import build_vehicle_dynamics
from sit_lmpc.world.rssm import BODY_INDEX, RSSM, RSSMConfig, bicycle, cheap_rollout, init_params, latent_rollout


def timed(fn, repeats, *args):
    out = fn(*args)
    jax.block_until_ready(out)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append((time.perf_counter() - start) * 1000)
    return float(np.percentile(times, 50)), float(np.percentile(times, 95))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/world/world-nochance.yaml")
    parser.add_argument("--hidden", nargs="+", type=int, default=[128, 64])
    parser.add_argument("--samples", nargs="+", type=int, default=[1024, 512, 256])
    parser.add_argument("--top", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    _, st = build_vehicle_dynamics(config)
    st_params = st.params.astype(jnp.float32)
    action_scale = jnp.asarray(st.action_scale, jnp.float32)
    rows = []
    print(f"device {jax.devices()[0]}")
    print(f"{'hidden':>6s} {'steps x dt':>10s} {'part':>7s} {'N':>5s} {'p50 ms':>7s} {'p95 ms':>7s}")
    for hidden in args.hidden:
        rssm_config = RSSMConfig(hidden=hidden)
        model, params = init_params(rssm_config, jax.random.key(0))
        h0 = jnp.zeros(hidden)
        s0 = jnp.zeros(rssm_config.latent)
        x0 = jnp.asarray([0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0])
        for steps, dt in ((15, 0.1), (10, 0.15)):
            integrator = 0.02 if dt == 0.1 else 0.025

            @jax.jit
            def st_all(actions):
                def one(seq):
                    return jax.lax.scan(lambda x, a: (bicycle(st_params, x, a * action_scale, dt, integrator),) * 2, x0, seq)[1]
                return jax.vmap(one)(actions)

            @jax.jit
            def cheap_all(actions):
                return jax.vmap(lambda seq: cheap_rollout(model, params, st_params, action_scale, dt, integrator,
                                                          h0, s0, x0, seq))(actions)

            @jax.jit
            def latent_all(actions, key):
                keys = jax.random.split(key, len(actions))
                return jax.vmap(lambda seq, k: latent_rollout(model, params, st_params, action_scale, dt, integrator,
                                                              h0, s0, x0, seq, k)[0])(actions, keys)

            def stage2(top):
                @jax.jit
                def run(actions, scores, key):
                    best = jax.lax.top_k(scores, top)[1]
                    chosen = actions[best]
                    keys = jax.random.split(key, top)
                    xs, hs, ss = jax.vmap(lambda seq, k: latent_rollout(model, params, st_params, action_scale, dt,
                                                                        integrator, h0, s0, x0, seq, k))(chosen, keys)
                    mean, _ = model.apply(params, hs, ss, method=RSSM.decode_lidar)
                    clearance = jnp.exp(mean).min(axis=-1)
                    penalty = jnp.maximum(0.3 - clearance, 0.0).sum(axis=-1)
                    refined = scores[best] - 10.0 * penalty
                    weights = jax.nn.softmax(refined / 0.1)
                    return (weights[:, None, None] * chosen).sum(axis=0), xs
                return run

            @jax.jit
            def filter_step(h, s, lidar, sensors):
                x = jnp.zeros((1, 7)).at[:, BODY_INDEX].set(model.apply(params, h, s, method=RSSM.anchor))
                return model.apply(params, h, s, jnp.zeros((1, 2)), x, lidar, sensors)

            for samples in args.samples:
                actions = jax.random.uniform(jax.random.key(1), (samples, steps, 2), minval=-1.0, maxval=1.0)
                scores = jax.random.normal(jax.random.key(2), (samples,))
                parts = {"st": timed(st_all, args.repeats, actions),
                         "cheap": timed(cheap_all, args.repeats, actions),
                         "latent": timed(latent_all, args.repeats, actions, jax.random.key(3))}
                for top in args.top:
                    if top < samples:
                        parts[f"stage2_top{top}"] = timed(stage2(top), args.repeats, actions, scores, jax.random.key(4))
                for part, (p50, p95) in parts.items():
                    rows.append({"hidden": hidden, "steps": steps, "dt": dt, "samples": samples, "part": part,
                                 "p50_ms": p50, "p95_ms": p95})
                    print(f"{hidden:6d} {steps:4d} x {dt:.2f} {part:>14s} {samples:5d} {p50:7.2f} {p95:7.2f}")
        p50, p95 = timed(filter_step, args.repeats, h0[None], s0[None], jnp.ones((1, rssm_config.beams)),
                         jnp.zeros((1, rssm_config.sensors)))
        rows.append({"hidden": hidden, "part": "filter", "p50_ms": p50, "p95_ms": p95})
        print(f"{hidden:6d} {'':10s} {'filter':>14s} {1:5d} {p50:7.2f} {p95:7.2f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"device": str(jax.devices()[0]), "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
