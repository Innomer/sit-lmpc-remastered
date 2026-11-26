"""Would a correct condition latent c improve the RSSM's predictions? Oracle-c test on a grip-change run.

    python scripts/oracle_condition_check.py runs/rssm-v11 --run change-grip-drop-v11 --grip 0.70

Oracle c for a grip value = mean filtered c over training steps whose true grip is within +-0.05 of it (track and
coverage data, after a 2 s burn-in per episode). On the run's starts after the change (and before it, as a control),
roll out 1.5 s from the true state with the applied controls, once with the run's own filtered (h, s) and once with
the slow part c replaced by the oracle c for the true grip. Reports position and lateral (e_y) errors.
"""

import argparse
import glob
import json
from pathlib import Path
import sys
import numpy as np
import jax
import jax.numpy as jnp
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lateral_error_check import frenet_of

from sit_lmpc.track import Track, map_path
from sit_lmpc.world.rssm import RSSM, latent_rollout
from sit_lmpc.world.rssm_data import TrackData, log_range
from sit_lmpc.world.rssm_risk import LatentFilter, run_episode
from sit_lmpc.world.rssm_train import CONDITION_MEAN, CONDITION_SCALE

HORIZON = 15


def oracle_c(filt, config, grip, width=0.05, burn=20, limit=400):
    cut = dict(zip(("max_slip", "max_yaw_rate"), config.get("data_cut", [1.0, 8.0])))
    values = []
    for root in [config["dataset"]] + ([config["coverage_dataset"]] if config.get("coverage_dataset") else []):
        data = TrackData(root, "train", config["model"]["beams"], **cut)
        episodes = [e for e in data.episodes if len(e["states"]) > burn + 5
                    and np.any(np.abs(e["conditions"][burn:, 0] - grip) <= width)][:limit]
        inputs = [{"sensors": e["sensors"], "lidar": log_range(e["lidar"]), "pose": e["pose"], "controls": e["controls"]}
                  for e in episodes]
        for e, f in zip(episodes, filt.features(inputs)):
            mask = np.abs(e["conditions"][:, 0] - grip) <= width
            mask[:burn] = False
            values.append(f[mask, filt.config.hidden:filt.config.hidden + filt.config.condition])
    c = np.concatenate(values)
    return c.mean(axis=0), len(c)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rssm")
    parser.add_argument("--run", default="change-grip-drop-v11")
    parser.add_argument("--root", default="results/Spielberg")
    parser.add_argument("--grip", type=float, default=0.70)
    parser.add_argument("--nominal", type=float, default=1.0489)
    parser.add_argument("--change-lap", type=int, default=8)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    filt = LatentFilter(args.rssm)
    config = yaml.safe_load((Path(args.rssm) / "config.yaml").read_text())
    hidden, cdim = filt.config.hidden, filt.config.condition
    readout = lambda c: np.asarray(filt.model.apply(filt.network, jnp.asarray(np.concatenate(
        [c, np.zeros(filt.config.fast)])[None], jnp.float32), method=RSSM.condition_readout))[0] * np.asarray(
        CONDITION_SCALE) + np.asarray(CONDITION_MEAN)
    oracle = {}
    for g in (args.grip, args.nominal):
        oracle[g], n = oracle_c(filt, config, g)
        print(f"oracle c for grip {g:.2f}: from {n} training steps; its grip read-out {readout(oracle[g])[0]:.2f}", flush=True)

    run = Path(args.root) / args.run
    episode, _, meta = run_episode(run, HORIZON)
    features = filt.features([episode])[0]
    files = sorted(glob.glob(str(run / "trajectories" / "*" / "attempt-*.npz")), key=lambda p: Path(p).name)
    change = meta["attempts"][args.change_lap]["start"]
    true, controls = meta["true"], episode["controls"]
    track = Track.from_track_path(map_path("Spielberg"))
    ones = jnp.ones(2, jnp.float32)
    roll = jax.jit(jax.vmap(lambda h, s, x, u: latent_rollout(filt.model, filt.network, filt.st_params, ones, filt.dt,
                                                              filt.integrator, h, s, x, u, jax.random.key(0),
                                                              sample=False)[0]))
    report = {"oracle_c_grip_readout": {str(g): float(readout(c)[0]) for g, c in oracle.items()}}
    for label, (lo, hi, grip) in {"after the change": (change + 20, len(controls) - HORIZON, args.grip),
                                   "before the change (control)": (meta["attempts"][5]["start"], change - HORIZON,
                                                                    args.nominal)}.items():
        starts = np.arange(lo, hi, 2)
        h = jnp.asarray(features[starts, :hidden])
        s = jnp.asarray(features[starts, hidden:])
        s_oracle = s.at[:, :cdim].set(jnp.asarray(oracle[grip], jnp.float32))
        x = jnp.asarray(true[starts], jnp.float32)
        u = jnp.asarray(np.stack([controls[t:t + HORIZON] for t in starts]), jnp.float32)
        own, forced = np.asarray(roll(h, s, x, u)), np.asarray(roll(h, s_oracle, x, u))
        future = np.stack([true[t + 1:t + 1 + HORIZON] for t in starts])
        hint = frenet_of(track, true[starts][:, None], np.zeros(len(starts)))[:, 0, 0]
        actual = frenet_of(track, future, hint)[..., 1]
        entry = {"starts": int(len(starts)), "own_c_grip_readout_mean": float(np.mean(
            [readout(np.asarray(c))[0] for c in np.asarray(s[:, :cdim])[::10]]))}
        for name, pred in (("own c", own), ("oracle c", forced)):
            lateral = np.abs(frenet_of(track, pred, hint)[..., 1] - actual)
            position = np.hypot(*(pred[..., :2] - future[..., :2]).transpose(2, 0, 1))
            entry[name] = {k: {"position_median": float(np.median(position[:, i])),
                               "position_p95": float(np.percentile(position[:, i], 95)),
                               "lateral_median": float(np.median(lateral[:, i])),
                               "lateral_p95": float(np.percentile(lateral[:, i], 95))}
                           for k, i in (("0.5 s", 4), ("1.0 s", 9), ("1.5 s", 14))}
        report[label] = entry
        print(f"\n{label}: {len(starts)} starts, own c reads grip {entry['own_c_grip_readout_mean']:.2f}, true {grip:.2f}")
        for name in ("own c", "oracle c"):
            print(f"  {name:9s} " + "   ".join(f"{k}: pos {v['position_median']:.2f}/{v['position_p95']:.2f} m, "
                                               f"e_y {v['lateral_median']:.2f}/{v['lateral_p95']:.2f} m"
                                               for k, v in entry[name].items()))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
