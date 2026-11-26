"""Condition tags over time on closed-loop change runs: the grip probe on (h, d) and the model-agnostic fallback tag
(the nominal bicycle model's 1.0 s position error), against the true grip.

    python scripts/condition_tag_check.py --probe runs/rssm-v11-grip --runs change-grip-drop-v11 change-patch-v11

Grip probe: raw per step and filtered (first-order, time constant --tau). Fallback tag at step t: distance between
the actual position and the nominal single-track rollout from the true state at t-10 with the applied controls.
Target for the probe: within 0.1 of the true grip within ~2 s of a change.
"""

import argparse
import glob
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sit_lmpc.world.rssm import bicycle
from sit_lmpc.world.rssm_grip import GripProbe
from sit_lmpc.world.rssm_risk import LatentFilter, run_episode

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
NOMINAL = 1.0489


def true_grip(meta, events, s_track, change):
    mu = np.full(len(s_track), NOMINAL)
    for event in events:
        condition = event.get("condition") or {}
        if "mu" in condition:
            mu[change:] = condition["mu"]
        for patch in event.get("patches") or []:
            inside = (s_track >= patch["s_start"]) & (s_track <= patch["s_end"])
            mu[change:][inside[change:]] = patch["mu"]
    return mu


def bicycle_error(filt, true, controls, lag=10):
    @jax.jit
    def roll(x, u):
        def one(x0, seq):
            return jax.lax.scan(lambda y, a: (bicycle(filt.st_params, y, a, filt.dt, filt.integrator), None), x0, seq)[0]
        return jax.vmap(one)(x, u)
    starts = np.arange(0, len(true) - lag)
    end = np.asarray(roll(jnp.asarray(true[starts], jnp.float32),
                          jnp.asarray(np.stack([controls[t:t + lag] for t in starts]), jnp.float32)))
    error = np.full(len(true), np.nan)
    error[starts + lag] = np.hypot(*(end[:, :2] - true[starts + lag, :2]).T)
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rssm", default="runs/rssm-v11")
    parser.add_argument("--probe", default="runs/rssm-v11-grip")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--root", default="results/Spielberg")
    parser.add_argument("--change-lap", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--out", default="results/condition-tags")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    filt = LatentFilter(args.rssm)
    probe = GripProbe(args.probe)
    report = {}
    for name in args.runs:
        run = Path(args.root) / name
        episode, _, meta = run_episode(run, 15)
        features = filt.features([episode])[0]
        hidden = filt.config.hidden
        raw = np.asarray(probe(jnp.asarray(features[:, :hidden]), jnp.asarray(features[:, hidden:])))
        keep = np.exp(-0.1 / args.tau)
        filtered = np.empty_like(raw)
        filtered[0] = raw[0]
        for t in range(1, len(raw)):
            filtered[t] = keep * filtered[t - 1] + (1 - keep) * raw[t]
        files = sorted(glob.glob(str(run / "trajectories" / "*" / "attempt-*.npz")), key=lambda p: Path(p).name)
        s_track = np.concatenate([np.load(f, allow_pickle=True)["frenet"][(0 if i == 0 else 1):, 0]
                                  for i, f in enumerate(files)])[:len(raw)]
        events = [json.loads(l)["event"] for l in (run.parent / f"{name}.conditions.jsonl").read_text().splitlines()
                  if l.strip()]
        change = meta["attempts"][args.change_lap]["start"] if len(meta["attempts"]) > args.change_lap else len(raw)
        mu = true_grip(meta, events, s_track, change)
        fallback = bicycle_error(filt, meta["true"], episode["controls"])[:len(raw)]
        before = np.arange(len(raw)) < change
        entry = {"change_step": int(change),
                 "probe_before_change": {"mean": float(filtered[before].mean()),
                                         "within_0.1": float(np.mean(np.abs(filtered[before] - mu[before]) <= 0.1))},
                 "fallback_before_change_m": {"median": float(np.nanmedian(fallback[before])),
                                              "p95": float(np.nanpercentile(fallback[before], 95))}}
        if any((e.get("condition") or {}).get("mu") for e in events):
            after = ~before
            within = np.abs(filtered - mu) <= 0.1
            settle = None
            for t in range(change, len(raw) - 20):
                if within[t:t + 20].all():
                    settle = (t - change) * 0.1
                    break
            entry.update(probe_after_change={"mean": float(filtered[after].mean()), "true": float(mu[after].mean()),
                                             "within_0.1_after_2s": float(np.mean(within[change + 20:])),
                                             "seconds_to_within_0.1_for_2s": settle},
                         fallback_after_change_m={"median": float(np.nanmedian(fallback[after])),
                                                  "p95": float(np.nanpercentile(fallback[after], 95))})
        patches = [p for e in events for p in (e.get("patches") or [])]
        if patches:
            inside = (~before) & (mu < NOMINAL - 1e-6)
            outside = (~before) & ~inside
            passes = np.split(np.flatnonzero(inside), np.flatnonzero(np.diff(np.flatnonzero(inside)) > 1) + 1)
            entry.update(patch={"passes": len(passes), "steps_per_pass": [len(p) for p in passes],
                                "probe_inside_mean": float(filtered[inside].mean()),
                                "probe_outside_mean": float(filtered[outside].mean()),
                                "probe_inside_within_0.1": float(np.mean(np.abs(filtered[inside] - mu[inside]) <= 0.1)),
                                "probe_min_per_pass": [float(filtered[p].min()) for p in passes],
                                "fallback_inside_median_m": float(np.nanmedian(fallback[inside])),
                                "fallback_outside_median_m": float(np.nanmedian(fallback[outside]))})
        report[name] = entry
        print(name, json.dumps(entry, indent=1), flush=True)
        t = np.arange(len(raw)) * 0.1
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), facecolor=SURFACE, sharex=True)
        axes[0].plot(t, raw, color="#9ec5f4", linewidth=0.6, label="grip probe, per step")
        axes[0].plot(t, filtered, color="#2a78d6", linewidth=1.2, label=f"grip probe, filtered ({args.tau} s)")
        axes[0].plot(t, mu, color=INK, linewidth=1.2, label="true grip")
        axes[0].set_ylabel("grip", color=MUTED, fontsize=9)
        axes[1].plot(t, fallback, color="#eb6834", linewidth=0.8, label="bicycle 1.0 s position error")
        axes[1].set_ylabel("m", color=MUTED, fontsize=9)
        axes[1].set_xlabel("time (s); grey lines: lap boundaries", color=MUTED, fontsize=9)
        for ax in axes:
            ax.set_facecolor(SURFACE)
            for a in meta["attempts"][1:]:
                ax.axvline(a["start"] * 0.1, color="#d9d8d4", linewidth=0.6)
            ax.axvline(change * 0.1, color="#e34948", linestyle="--", linewidth=1.2, label="change")
            ax.tick_params(colors=MUTED, labelsize=8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.legend(frameon=False, fontsize=8, loc="lower left")
        axes[0].set_title(f"{name}: condition tags over time", color=INK, fontsize=11, loc="left")
        fig.tight_layout()
        fig.savefig(out / f"{name}-tags.png", dpi=110, facecolor=SURFACE)
        plt.close(fig)
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
