"""Time the planner's safety checks with a trained RSSM, per control step (p50 / p95, forced to finish on the device).

    python scripts/time_safety_checks.py --config configs/world/rssm-latent-v8.yaml

  latent     world-model rollout of all sampled candidates (the planner's prediction, for reference)
  clearance  LiDAR clearance check on the samples (second latent rollout + LiDAR decode)
  candidate  the same checks again on the final candidates (one per lambda pair) for the final ranking
"""

import argparse
import time
import numpy as np
import jax
import jax.numpy as jnp

from sit_lmpc.config import load_config
from sit_lmpc.experiments.runner import build_vehicle_dynamics
from sit_lmpc.track import Track, map_path
from sit_lmpc.world.rssm_runtime import RSSMFilter


def timed(fn, repeats):
    jax.block_until_ready(fn())
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        times.append((time.perf_counter() - start) * 1000)
    return np.percentile(times, 50), np.percentile(times, 95)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/world/rssm-latent-v8.yaml")
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    config = load_config(args.config)
    planner, dynamics = config.planner, config.planner.dynamics
    _, st = build_vehicle_dynamics(config)
    track = Track.from_track_path(map_path(config.simulation.map))
    rssm = RSSMFilter(dynamics.rssm, st.params, st.action_scale, dynamics.rssm_variant, dynamics.rssm_model_timestep,
                      st.integrator_timestep, dynamics.rssm_clearance_margin, max(dynamics.rssm_clearance_weight, 1.0))
    model = rssm.predictor()
    start = np.asarray(track.waypoints[10]) if hasattr(track, "waypoints") else np.zeros(2)
    state = jnp.asarray([start[0], start[1], 0.0, 8.0, 0.0, 0.0, 0.0], jnp.float32)
    candidates = planner.lambda_count ** 2
    rng = np.random.default_rng(0)
    results = {}
    for name, count in (("samples", planner.samples), ("candidates", candidates)):
        actions = jnp.asarray(rng.uniform(-1, 1, (count, planner.horizon, 2)), jnp.float32)
        rollout = jax.jit(jax.vmap(lambda a: model.rollout_members(state, a)[0]))
        results[f"latent ({name})"] = timed(lambda: rollout(actions), args.repeats)
        results[f"clearance ({name})"] = timed(lambda: model.clearance_cost(state, actions), args.repeats)
    print(f"{jax.devices()[0].device_kind}: {planner.samples} samples, {candidates} candidates, horizon {planner.horizon}")
    for name, (p50, p95) in results.items():
        print(f"  {name:24s} p50 {p50:6.1f} ms   p95 {p95:6.1f} ms")


if __name__ == "__main__":
    main()
