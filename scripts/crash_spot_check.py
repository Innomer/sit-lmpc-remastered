"""Why does v11 crash at the same spots on normal track? Compare the approach to each crash with clean passes over
the same stretch: speed, lateral offset, slip, and v11's 1.5 s lateral prediction error from the true state.

    python scripts/crash_spot_check.py runs/rssm-v11 --spot 34 --runs ... --clean ...

Crash approach: the 2 s before the sim contact, if the contact is within --radius m of --spot. Clean passes: every
learned-lap step of the --clean runs (and of the crashed runs' earlier learned laps) whose track position is within
the same 20 m before the spot.
"""

import argparse
import glob
import json
from pathlib import Path
import sys
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lateral_error_check import frenet_of

from sit_lmpc.track import Track, map_path
from sit_lmpc.world.rssm import latent_rollout
from sit_lmpc.world.rssm_risk import LatentFilter, run_episode

HORIZON = 15


def steps_of(run, filt, track, roll):
    episode, _, meta = run_episode(run, HORIZON)
    features = filt.features([episode])[0]
    files = sorted(glob.glob(str(run / "trajectories" / "*" / "attempt-*.npz")), key=lambda p: Path(p).name)
    frenet = np.concatenate([np.load(f, allow_pickle=True)["frenet"][(0 if i == 0 else 1):] for i, f in enumerate(files)])
    learned = next((a["start"] for a, f in zip(meta["attempts"], files)
                    if str(np.load(f, allow_pickle=True)["controllers"][0]) == "lmppi"), None)
    true, controls = meta["true"], episode["controls"]
    n = min(len(true), len(frenet))
    starts = np.arange(learned, n - HORIZON) if learned is not None else np.zeros(0, int)
    hidden = filt.config.hidden
    pred = np.asarray(roll(jnp.asarray(features[starts, :hidden]), jnp.asarray(features[starts, hidden:]),
                           jnp.asarray(true[starts], jnp.float32),
                           jnp.asarray(np.stack([controls[t:t + HORIZON] for t in starts]), jnp.float32))) if len(starts) else None
    error = np.full(n, np.nan)
    if pred is not None:
        hint = frenet_of(track, true[starts][:, None], np.zeros(len(starts)))[:, 0, 0]
        actual = frenet_of(track, true[starts + HORIZON][:, None], hint)[:, 0, 1]
        predicted = frenet_of(track, pred[:, -1][:, None], hint)[:, 0, 1]
        error[starts] = np.abs(actual - predicted)
    return {"s": frenet[:n, 0] % track.s_frame_max, "ey": frenet[:n, 1], "speed": true[:n, 3], "slip": true[:n, 6],
            "error": error, "learned": learned or n, "crash": meta["crash"]}


def summary(rows, mask):
    if not mask.any():
        return None
    return {"steps": int(mask.sum()), "speed_mean": float(rows["speed"][mask].mean()),
            "speed_max": float(rows["speed"][mask].max()),
            "abs_ey_mean": float(np.abs(rows["ey"][mask]).mean()), "abs_ey_max": float(np.abs(rows["ey"][mask]).max()),
            "abs_slip_max": float(np.abs(rows["slip"][mask]).max()),
            "pred_error_1p5s_median": float(np.nanmedian(rows["error"][mask])),
            "pred_error_1p5s_p90": float(np.nanpercentile(rows["error"][mask], 90))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rssm")
    parser.add_argument("--spot", type=float, required=True)
    parser.add_argument("--radius", type=float, default=8.0)
    parser.add_argument("--runs", nargs="+", required=True, help="runs whose crash may be at this spot")
    parser.add_argument("--clean", nargs="*", default=[], help="runs with clean passes over the spot")
    parser.add_argument("--root", default="results/Spielberg")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    filt = LatentFilter(args.rssm)
    track = Track.from_track_path(map_path("Spielberg"))
    ones = jnp.ones(2, jnp.float32)
    roll = jax.jit(jax.vmap(lambda h, s, x, u: latent_rollout(filt.model, filt.network, filt.st_params, ones, filt.dt,
                                                              filt.integrator, h, s, x, u, jax.random.key(0),
                                                              sample=False)[0][-1]))
    roll_last = lambda h, s, x, u: roll(h, s, x, u)[:, None]
    lo, hi = args.spot - 20.0, args.spot
    report = {"spot_m": args.spot, "crashes": {}, "clean": None}
    clean_rows = []
    for name in dict.fromkeys(args.runs + args.clean):
        rows = steps_of(Path(args.root) / name, filt, track, lambda h, s, x, u: roll(h, s, x, u)[:, None])
        idx = np.arange(len(rows["s"]))
        crash = rows["crash"]
        lap_end = crash if crash is not None and abs(rows["s"][crash - 1] - args.spot) <= args.radius else None
        approach = np.zeros(len(idx), bool)
        if lap_end is not None and name in args.runs:
            approach[max(0, lap_end - 20):lap_end] = True
            report["crashes"][name] = summary(rows, approach)
        near = (rows["s"] >= lo) & (rows["s"] <= hi) & (idx >= rows["learned"]) & ~approach
        if lap_end is not None:
            near &= idx < lap_end - 40
        clean_rows.append({k: v[near] for k, v in rows.items() if isinstance(v, np.ndarray)})
    merged = {k: np.concatenate([r[k] for r in clean_rows]) for k in clean_rows[0]}
    report["clean"] = summary(merged, np.ones(len(merged["s"]), bool))
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
