"""Lateral (e_y) prediction error of an RSSM near the track edges, on closed-loop runs and on validation data.

    python scripts/lateral_error_check.py runs/rssm-v8 --runs rssm-latent-slow-logged rssm-latent-v6-smoke ...
    python scripts/lateral_error_check.py runs/rssm-v8 --validation --out results/lateral-error/v8-validation.json

At every step the model is filtered (posterior mean, as the planner's filter) and rolled out with the controls that
were actually applied next, from the state the planner starts from in the sim: true position and heading, steering,
speed, yaw rate and slip from the model's anchor. Predicted and actual
positions are projected onto the track the planner uses (Frenet e_y). Errors are signed towards the nearer wall:
outward error = (actual e_y - predicted e_y) * sign(actual e_y), so positive means the car ended up further out than
predicted (the model placed it further inside).
"""

import argparse
import glob
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import yaml

from sit_lmpc.planners.common import to_frenet
from sit_lmpc.track import Track, map_path
from sit_lmpc.world.rssm import RSSM, latent_rollout
from sit_lmpc.world.rssm_data import TrackData, log_range
from sit_lmpc.world.rssm_risk import LatentFilter, run_episode

HORIZONS = {"0.5 s": 5, "1.0 s": 10, "1.5 s": 15}
TRACKS = {"Spielberg": "Spielberg", "custom1": "custom1", "levine": "tracks/levine_sim/levine_sim"}


def rollouts(filt, features, pose, controls, horizon):
    """Predicted states (starts, horizon, 7) from every step with a full horizon of applied controls ahead."""
    model, network = filt.model, filt.network
    hidden = filt.config.hidden
    st_params = filt.st_params
    ones = jnp.ones(2, jnp.float32)
    anchors = model.apply(network, jnp.asarray(features[:, :hidden]), jnp.asarray(features[:, hidden:]),
                          method=RSSM.anchor)
    anchors = np.asarray(anchors)
    starts = np.arange(0, len(controls) - horizon + 1)
    x0 = np.column_stack((pose[starts, 0], pose[starts, 1], anchors[starts, 0], anchors[starts, 1], pose[starts, 2],
                          anchors[starts, 2], anchors[starts, 3])).astype(np.float32)
    u = np.stack([controls[s:s + horizon] for s in starts]).astype(np.float32)

    @jax.jit
    def run(h, s, x, a):
        return jax.vmap(lambda h_, s_, x_, a_: latent_rollout(model, network, st_params, ones, filt.dt, filt.integrator,
                                                             h_, s_, x_, a_, jax.random.key(0), sample=False)[0])(h, s, x, a)

    out = []
    for k in range(0, len(starts), 512):
        sl = slice(k, k + 512)
        out.append(np.asarray(run(jnp.asarray(features[starts[sl], :hidden]), jnp.asarray(features[starts[sl], hidden:]),
                                  jnp.asarray(x0[sl]), jnp.asarray(u[sl]))))
    return starts, np.concatenate(out) if out else np.zeros((0, horizon, 7), np.float32)


def frenet_of(track, states, s_hint):
    """Frenet (s, e_y) of states (n, ..., 7) with the planner's projection, searched around each start's progress."""
    flat = states.reshape(len(states), -1, 7)
    out = np.zeros(flat.shape[:2] + (2,), np.float32)
    for i in range(len(flat)):
        indices = jnp.asarray(track.get_s_search_range(float(s_hint[i]), 400))
        f = np.asarray(to_frenet(track, jnp.asarray(flat[i]), indices, float(s_hint[i])))
        out[i] = f[:, :2]
    return out.reshape(states.shape[:-1] + (2,))


def collect(track, starts, predicted, true, speed, horizon):
    """Rows: actual e_y and speed at the start and at each horizon, predicted e_y, outward error."""
    actual_future = np.stack([true[s + 1:s + 1 + horizon] for s in starts])
    s_hint = np.asarray([0.0] * len(starts))
    start_frenet = frenet_of(track, true[starts][:, None], s_hint)[:, 0]
    s_hint = start_frenet[:, 0]
    actual_f = frenet_of(track, actual_future, s_hint)
    predicted_f = frenet_of(track, predicted, s_hint)
    rows = {}
    for name, k in HORIZONS.items():
        a, p = actual_f[:, k - 1, 1], predicted_f[:, k - 1, 1]
        path = track.centerline if track.centerline is not None else track.raceline
        curvature = np.interp(actual_f[:, k - 1, 0] % path.ss[-1], path.ss, path.ks)
        rows[name] = {"actual": a, "predicted": p, "outward": (a - p) * np.sign(a), "abs": np.abs(a - p),
                      "start_ey": start_frenet[:, 1], "speed": speed[starts],
                      "corner_outside": (a - p) * -np.sign(curvature), "curvature": np.abs(curvature)}
    return rows


def summarize(rows, edge_bins=(0.0, 0.3, 0.5, 0.7, 0.85, 2.0), speed_bins=(0, 4, 7, 9, 11, 30)):
    out = {}
    for name, r in rows.items():
        entry = {"n": int(len(r["abs"])), "mean_outward_m": float(np.mean(r["outward"])),
                 "median_outward_m": float(np.median(r["outward"])), "mean_abs_m": float(np.mean(r["abs"])),
                 "p95_abs_m": float(np.percentile(r["abs"], 95)),
                 "share_further_out_than_predicted": float(np.mean(r["outward"] > 0)), "by_edge": [], "by_speed": []}
        edge = np.abs(r["actual"])
        for lo, hi in zip(edge_bins[:-1], edge_bins[1:]):
            m = (edge >= lo) & (edge < hi)
            if m.sum() >= 10:
                entry["by_edge"].append({"abs_ey": f"{lo}-{hi}", "n": int(m.sum()),
                                         "mean_outward_m": float(r["outward"][m].mean()),
                                         "p95_abs_m": float(np.percentile(r["abs"][m], 95))})
        bend = r["curvature"] > 0.05
        entry["corner_outside_mean_m"] = float(np.mean(r["corner_outside"][bend]))
        entry["corner_outside_share"] = float(np.mean(r["corner_outside"][bend] > 0))
        entry["corner_outside_n"] = int(bend.sum())
        entry["corner_outside_by_speed"] = []
        for lo, hi in zip(speed_bins[:-1], speed_bins[1:]):
            m = bend & (r["speed"] >= lo) & (r["speed"] < hi)
            if m.sum() >= 10:
                entry["corner_outside_by_speed"].append({"speed": f"{lo}-{hi}", "n": int(m.sum()),
                                                         "mean_m": float(r["corner_outside"][m].mean()),
                                                         "share": float(np.mean(r["corner_outside"][m] > 0))})
        entry["corner_outside_by_start_edge"] = []
        start_edge = np.abs(r["start_ey"])
        for lo, hi in zip(edge_bins[:-1], edge_bins[1:]):
            m = bend & (start_edge >= lo) & (start_edge < hi)
            if m.sum() >= 10:
                entry["corner_outside_by_start_edge"].append({"abs_start_ey": f"{lo}-{hi}", "n": int(m.sum()),
                                                              "mean_m": float(r["corner_outside"][m].mean()),
                                                              "p95_abs_m": float(np.percentile(r["abs"][m], 95))})
        entry["by_predicted_edge"] = []
        predicted_edge = np.abs(r["predicted"])
        outward_vs_predicted = (r["actual"] - r["predicted"]) * np.sign(r["predicted"])
        for lo, hi in zip(edge_bins[:-1], edge_bins[1:]):
            m = (predicted_edge >= lo) & (predicted_edge < hi)
            if m.sum() >= 10:
                entry["by_predicted_edge"].append({"abs_predicted_ey": f"{lo}-{hi}", "n": int(m.sum()),
                                                   "mean_outward_m": float(outward_vs_predicted[m].mean()),
                                                   "share_further_out": float(np.mean(outward_vs_predicted[m] > 0)),
                                                   "p95_abs_m": float(np.percentile(r["abs"][m], 95))})
        for lo, hi in zip(speed_bins[:-1], speed_bins[1:]):
            m = (r["speed"] >= lo) & (r["speed"] < hi)
            if m.sum() >= 10:
                entry["by_speed"].append({"speed": f"{lo}-{hi}", "n": int(m.sum()),
                                          "mean_outward_m": float(r["outward"][m].mean()),
                                          "p95_abs_m": float(np.percentile(r["abs"][m], 95))})
        out[name] = entry
    return out


def merge_rows(parts):
    return {name: {key: np.concatenate([p[name][key] for p in parts]) for key in parts[0][name]} for name in HORIZONS}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rssm")
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--runs-root", default="results/Spielberg")
    parser.add_argument("--learned-only", action="store_true", help="only steps of learned (LMPPI) laps")
    parser.add_argument("--validation", action="store_true", help="per-step margins on the dataset's val split")
    parser.add_argument("--edge", type=float, default=0.5, help="|e_y| at the predicted step counted as near the edge")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    filt = LatentFilter(args.rssm)
    config = yaml.safe_load((Path(args.rssm) / "config.yaml").read_text())
    horizon = 15
    report = {"rssm": args.rssm, "runs": {}}
    if args.runs:
        track = Track.from_track_path(map_path("Spielberg"))
        parts = []
        for name in args.runs:
            episode, _, meta = run_episode(Path(args.runs_root) / name, horizon)
            features = filt.features([episode])[0]
            starts, predicted = rollouts(filt, features, episode["pose"], episode["controls"], horizon)
            if args.learned_only:
                files = sorted(glob.glob(str(Path(args.runs_root) / name / "trajectories" / "*" / "attempt-*.npz")),
                               key=lambda f: Path(f).name)
                learned = next((att["start"] for att, f in zip(meta["attempts"], files)
                                if str(np.load(f, allow_pickle=True)["controllers"][0]) == "lmppi"), None)
                keep = starts >= (learned if learned is not None else len(meta["true"]))
                starts, predicted = starts[keep], predicted[keep]
                if not len(starts):
                    continue
            rows = collect(track, starts, predicted, meta["true"], meta["true"][:, 3], horizon)
            parts.append(rows)
            report["runs"][name] = summarize(rows)
            last = {k: float(np.mean(v["outward"][-20:])) for k, v in rows.items()}
            report["runs"][name]["last_2s_mean_outward_m"] = last
            print(name, {k: (round(v["mean_outward_m"], 3), round(v["p95_abs_m"], 2)) for k, v in report["runs"][name].items()
                         if k in HORIZONS}, "last 2 s outward", {k: round(v, 2) for k, v in last.items()}, flush=True)
        report["all_runs"] = summarize(merge_rows(parts))
    if args.validation:
        cut = dict(zip(("max_slip", "max_yaw_rate"), config.get("data_cut", [1.0, 8.0])))
        data = TrackData(config["dataset"], "val", config["model"]["beams"], **cut)
        episodes = [e for e in data.episodes if "/onset-" not in e["file"] and len(e["controls"]) > horizon + 1]
        tracks = {name: Track.from_track_path(map_path(path)) for name, path in TRACKS.items()}
        inputs = [{"sensors": e["sensors"], "lidar": log_range(e["lidar"]), "pose": e["pose"], "controls": e["controls"]}
                  for e in episodes]
        features = filt.features(inputs)
        per_step = {k: [] for k in range(1, horizon + 1)}
        per_step_edge = {k: [] for k in range(1, horizon + 1)}
        for e, x, inp in zip(episodes, features, inputs):
            true_pose = e["states"][:, [0, 1, 4]].astype(np.float32)
            starts, predicted = rollouts(filt, x, true_pose, inp["controls"], horizon)
            starts, predicted = starts[::3], predicted[::3]
            track = tracks[e["map"]]
            actual_future = np.stack([e["states"][s + 1:s + 1 + horizon] for s in starts])
            hint = frenet_of(track, e["states"][starts][:, None], np.zeros(len(starts)))[:, 0, 0]
            a = frenet_of(track, actual_future, hint)[..., 1]
            p = frenet_of(track, predicted, hint)[..., 1]
            for k in range(1, horizon + 1):
                err = np.abs(a[:, k - 1] - p[:, k - 1])
                per_step[k].append(err)
                per_step_edge[k].append(err[np.abs(a[:, k - 1]) >= args.edge])
        report["validation"] = {
            "episodes": len(episodes), "edge_threshold_m": args.edge,
            "p95_abs_ey_error_m": [float(np.percentile(np.concatenate(per_step[k]), 95)) for k in per_step],
            "p95_abs_ey_error_near_edge_m": [float(np.percentile(np.concatenate(per_step_edge[k]), 95)) for k in per_step_edge],
            "n_near_edge": [int(sum(len(v) for v in per_step_edge[k])) for k in per_step_edge]}
        print(json.dumps(report["validation"], indent=1), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
