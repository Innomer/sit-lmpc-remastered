"""Does the RSSM encoder under-read the current slip on high-slip steps? Filtered (posterior-mean) slip estimate vs true
slip, binned by |true slip|, on held-out windows of the on-track and map-free data, and on logged sim crashes.

    python scripts/slip_estimate_check.py runs/rssm-v8 runs/rssm-v9 --crash results/Spielberg/rssm-latent-v8-varied-long
"""

import argparse
import glob
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import yaml
from flax import serialization

from sit_lmpc.config import load_config
from sit_lmpc.experiments.runner import build_vehicle_dynamics
from sit_lmpc.world.rssm import RSSMConfig, init_params
from sit_lmpc.world.rssm_data import MapFreeData, TrackData
from sit_lmpc.world.rssm_runtime import RSSMFilter
from sit_lmpc.world.rssm_train import run_sequence

BINS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.0)]


def windows(data, window, stride):
    return [(i, t) for i, e in enumerate(data.episodes) for t in range(0, len(e["controls"]) - window + 1, stride)]


def filtered_slip(run, data, window=100, burn=20, stride=100, limit=3000):
    config = yaml.safe_load((Path(run) / "config.yaml").read_text())
    model, template = init_params(RSSMConfig(**config["model"]), jax.random.key(0))
    params = serialization.from_bytes(template, (Path(run) / "best.msgpack").read_bytes())
    scale = np.load(Path(run) / "observation_scale.npz")
    _, st = build_vehicle_dynamics(load_config(config["vehicle_config"]))
    st_params = jnp.asarray(st.params, jnp.float32)
    settings = dict(config["settings"], prediction_steps=1, prediction_stride=window)
    fn = jax.jit(jax.vmap(lambda b: run_sequence(params, model, st_params, config["timestep"],
                                                 config["integrator_timestep"], b, jax.random.key(0), settings,
                                                 jnp.asarray(scale["mean"]), jnp.asarray(scale["std"]),
                                                 sample=False)[1]["filtered"]["anchor"]))
    items = windows(data, window, stride)
    rng = np.random.default_rng(0)
    if len(items) > limit:
        items = [items[i] for i in rng.choice(len(items), limit, replace=False)]
    high = data.high_slip_windows(window, 0.3)
    items += [high[i] for i in rng.choice(len(high), min(len(high), limit // 3), replace=False)] if high else []
    true, est = [], []
    for k in range(0, len(items), 64):
        batch = data.gather(items[k:k + 64], window)
        anchor = np.asarray(fn({key: jnp.asarray(v) for key, v in batch.items()}))
        true.append(batch["states"][:, 1 + burn:, 6].ravel())
        est.append(anchor[:, burn:, 3].ravel())
    return np.concatenate(true), np.concatenate(est)


def report(name, true, est):
    print(f"  {name}")
    print("    |true slip| bin   steps    mean |true|  mean est (same sign)  ratio   abs error")
    for lo, hi in BINS:
        m = (np.abs(true) >= lo) & (np.abs(true) < hi)
        if m.sum() < 10:
            continue
        sign = np.sign(true[m])
        along = est[m] * sign
        print(f"    {lo:.1f}-{hi:.1f}          {m.sum():7d}   {np.abs(true[m]).mean():.3f}        {along.mean():+.3f}"
              f"               {along.mean() / np.abs(true[m]).mean():.2f}    {np.abs(est[m] - true[m]).mean():.3f}")


def crash(run_model, crash_run, top=8):
    crash_run = Path(crash_run)
    config = yaml.safe_load((crash_run / "config.yaml").read_text())
    _, st = build_vehicle_dynamics(load_config(str(crash_run / "config.yaml")))
    rssm = RSSMFilter(run_model, st.params, st.action_scale)
    true = np.load(crash_run / "encoder_true_states.npy")
    observations = np.load(crash_run / "encoder_observations.npy")
    lidar = np.load(crash_run / "rssm_lidar.npy")
    attempt = np.load(sorted(glob.glob(str(crash_run / "trajectories" / "failure" / "*.npz")))[-1], allow_pickle=True)
    controls = attempt["controls"]
    n = len(attempt["cartesian"])
    offset = len(true) - n
    yaw = true[:, 4]
    rear = np.column_stack((true[:, 0] - 0.17145 * np.cos(yaw), true[:, 1] - 0.17145 * np.sin(yaw), yaw))
    rssm.reset()
    est = []
    for k in range(n):
        est.append(rssm.step_features(observations[offset + k], lidar[offset + k], rear[offset + k],
                                      controls[k - 1] if k else np.zeros(2))[3])
    est = np.asarray(est)
    t = true[offset:offset + n, 6]
    print(f"  crash {crash_run.name}, last {top} steps before contact (true -> estimated slip):")
    print("    " + "  ".join(f"{a:+.2f}->{b:+.2f}" for a, b in zip(t[-top - 1:-1], est[-top - 1:-1])))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--crash", nargs="*", default=[])
    args = parser.parse_args()
    first = yaml.safe_load((Path(args.runs[0]) / "config.yaml").read_text())
    track = TrackData(first["dataset"], "val", first["model"]["beams"])
    free = MapFreeData(first["map_free_dataset"], "validation", first["model"]["beams"]) if first.get("map_free_dataset") else None
    for run in args.runs:
        print(run)
        report("on-track validation", *filtered_slip(run, track))
        if free is not None:
            report("map-free validation", *filtered_slip(run, free, window=40, burn=10, stride=40))
        for crash_run in args.crash:
            crash(run, crash_run)


if __name__ == "__main__":
    main()
