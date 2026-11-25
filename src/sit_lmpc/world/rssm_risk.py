"""Crash-risk head on a frozen RSSM: predicts "crash within the next 1.5 s" from the filtered memory and latent (h, s).

    python -m sit_lmpc.world.rssm_risk --config configs/train-rssm-risk.yaml --output runs/rssm-v8-risk

Crash = first wall contact (data/f1tenth-track/contacts.json, or the sim's collision flag in closed-loop runs), or a
spin past the model envelope (slip > 1.2 rad or yaw rate > 10 rad/s, where episodes are cut). A step is labelled 1
when the crash comes within `horizon` steps. Every episode is filtered from its start with the frozen RSSM (posterior
mean), exactly as the planner's filter runs. Training data: the dataset's train split plus the closed-loop sim runs
listed under `train_runs` (crashes under our own planner); `heldout_runs` are only used for the warning-time check.
The network and the class-balanced loss are SIT-LMPC's risk classifier (sit_lmpc.training.risk).
"""

import argparse
import glob
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import optax
import yaml
from flax import serialization
from flax.training.train_state import TrainState

from sit_lmpc.config import load_config
from sit_lmpc.experiments.runner import build_vehicle_dynamics
from sit_lmpc.training.risk import RiskNetwork, calibration_metrics
from sit_lmpc.world.rssm import RSSM, RSSMConfig, bicycle, clamp, init_params
from sit_lmpc.world.rssm_data import TrackData, attempt_contact, log_range

LR = 0.17145


def auc(scores, labels):
    """Area under the ROC curve (probability a random crash step scores above a random safe step)."""
    scores, labels = np.asarray(scores, np.float64), np.asarray(labels, bool)
    positives, negatives = labels.sum(), (~labels).sum()
    if not positives or not negatives:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    edges = np.flatnonzero(np.diff(sorted_scores)) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [len(scores)]))
    for a, b in zip(starts, ends):
        ranks[order[a:b]] = 0.5 * (a + b - 1) + 1
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def labels_for(length, crash_step, horizon):
    """1 where the crash happens within the next `horizon` steps (steps 0..length-1 of a cleaned episode)."""
    y = np.zeros(length, np.float32)
    if crash_step is not None:
        steps = np.arange(length)
        y[(crash_step - steps <= horizon) & (steps < crash_step)] = 1.0
    return y


class LatentFilter:
    """Batched posterior-mean filtering of whole episodes with a trained RSSM (same equations as RSSMFilter)."""

    def __init__(self, run):
        run = Path(run)
        config = yaml.safe_load((run / "config.yaml").read_text())
        self.config = RSSMConfig(**config["model"])
        self.model, template = init_params(self.config, jax.random.key(0))
        self.network = serialization.from_bytes(template, (run / "best.msgpack").read_bytes())
        scale = np.load(run / "observation_scale.npz")
        mean, std = jnp.asarray(scale["mean"], jnp.float32), jnp.asarray(scale["std"], jnp.float32)
        _, st = build_vehicle_dynamics(load_config(config["vehicle_config"]))
        st_params = jnp.asarray(st.params, jnp.float32)
        dt, integrator = float(config["timestep"]), float(config["integrator_timestep"])
        model, network = self.model, self.network

        def apply(method, *args):
            return model.apply(network, *args, method=method)

        def merge(pose, body_state):
            return jnp.stack([pose[0], pose[1], body_state[0], body_state[1], pose[2], body_state[2], body_state[3]])

        def one(sensors, lidar, pose, controls):
            sensors = (sensors - mean) / std
            h0 = jnp.zeros(self.config.hidden, jnp.float32)
            s0, _ = apply(RSSM.posterior, h0, apply(RSSM.embed, lidar[0], sensors[0]))
            x0 = merge(pose[0], apply(RSSM.anchor, h0, s0))

            def step(carry, inputs):
                h, s, x = carry
                u, l, o, p = inputs
                x_pred = clamp(bicycle(st_params, x, u, dt, integrator) + apply(RSSM.residual, x, u, h, s))
                h = apply(RSSM.advance, h, s, u, x_pred)
                s, _ = apply(RSSM.posterior, h, apply(RSSM.embed, l, o))
                return (h, s, merge(p, apply(RSSM.anchor, h, s))), (h, s)

            _, (hs, ss) = jax.lax.scan(step, (h0, s0, x0), (controls, lidar[1:], sensors[1:], pose[1:]))
            return jnp.concatenate([jnp.concatenate([h0[None], hs]), jnp.concatenate([s0[None], ss])], -1)

        self._batch = jax.jit(jax.vmap(one))

    def features(self, episodes, bucket=64, batch=64):
        """(h, s) per step for every episode: dicts with sensors (T, 9), lidar (T, beams, log range), pose (T, 3, CoG),
        controls (T-1, 2)."""
        out = [None] * len(episodes)
        by_length = {}
        for i, e in enumerate(episodes):
            by_length.setdefault(int(np.ceil(len(e["sensors"]) / bucket) * bucket), []).append(i)
        for length, indices in by_length.items():
            for k in range(0, len(indices), batch):
                chunk = indices[k:k + batch]
                pad = lambda a, n: np.concatenate([a, np.repeat(a[-1:], n - len(a), 0)]) if len(a) < n else a
                stack = {key: np.stack([pad(np.asarray(episodes[i][key], np.float32), length - (key == "controls"))
                                        for i in chunk]) for key in ("sensors", "lidar", "pose", "controls")}
                stack = {key: np.concatenate([v, np.repeat(v[-1:], batch - len(chunk), 0)]) for key, v in stack.items()}
                result = np.asarray(self._batch(*(jnp.asarray(stack[key]) for key in ("sensors", "lidar", "pose",
                                                                                        "controls"))))
                for j, i in enumerate(chunk):
                    out[i] = result[j, :len(episodes[i]["sensors"])].astype(np.float32)
        return out


def dataset_episodes(data, horizon):
    episodes, labels, meta = [], [], []
    for e in data.episodes:
        crash = e["crash_step"] if e["crash_step"] is not None and e["crash_step"] <= len(e["states"]) else None
        if crash is None and e.get("spin_step") is not None:
            crash = e["spin_step"]
        episodes.append({"sensors": e["sensors"], "lidar": log_range(e["lidar"]), "pose": e["pose"],
                         "controls": e["controls"]})
        labels.append(labels_for(len(e["states"]), crash, horizon))
        meta.append({"file": e["file"], "map": e["map"], "onset": "/onset-" in e["file"], "crash": crash})
    return episodes, labels, meta


def run_episode(run, horizon):
    """A closed-loop sim run as one continuous episode (its per-step logs), with the crash at the sim contact step."""
    run = Path(run)
    true = np.load(run / "encoder_true_states.npy")
    files = sorted(glob.glob(str(run / "trajectories" / "*" / "attempt-*.npz")), key=lambda p: Path(p).name)
    controls, crash, start, attempts = [], None, 0, []
    for path in files:
        attempt = np.load(path, allow_pickle=True)
        n = len(attempt["cartesian"])
        controls.append(np.asarray(attempt["controls"][:n - 1], np.float32))
        contact = attempt_contact(attempt)
        if contact is not None and crash is None:
            crash = start + contact
        attempts.append({"start": start, "length": n, "outcome": Path(path).parent.name, "contact": contact})
        start += n - 1
    controls = np.concatenate(controls)
    end = len(true) if crash is None else crash
    yaw = true[:, 4]
    pose = np.column_stack((true[:, 0], true[:, 1], yaw)).astype(np.float32)
    episode = {"sensors": np.load(run / "encoder_observations.npy")[:end].astype(np.float32),
               "lidar": np.load(run / "rssm_lidar.npy")[:end].astype(np.float32),
               "pose": pose[:end], "controls": controls[:end - 1]}
    return episode, labels_for(end, crash, horizon), {"run": run.name, "crash": crash, "attempts": attempts,
                                                     "true": true[:end]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config))
    base = Path(config["base"])
    base_config = yaml.safe_load((base / "config.yaml").read_text())
    horizon = int(config["horizon"])
    cut = dict(zip(("max_slip", "max_yaw_rate"), base_config.get("data_cut", [1.0, 8.0])))
    beams = base_config["model"]["beams"]
    filt = LatentFilter(base)

    sets = {}
    for split in ("train", "val", "test"):
        data = TrackData(base_config["dataset"], split, beams, **cut)
        episodes, labels, meta = dataset_episodes(data, horizon)
        sets[split] = (filt.features(episodes), labels, meta)
        print(f"{split}: {len(episodes)} episodes, {sum(map(len, labels))} steps, "
              f"{int(sum(l.sum() for l in labels))} crash-within-{horizon * 0.1:.1f}-s steps", flush=True)
    runs = {}
    for name in config["train_runs"] + config["heldout_runs"]:
        episode, label, meta = run_episode(Path(config["runs_root"]) / name, horizon)
        runs[name] = (filt.features([episode])[0], label, meta)
        print(f"run {name}: {len(label)} steps, crash at {meta['crash']}", flush=True)

    train_x = np.concatenate(sets["train"][0] + [runs[n][0] for n in config["train_runs"]])
    train_y = np.concatenate(sets["train"][1] + [runs[n][1] for n in config["train_runs"]])
    mean, std = train_x.mean(0), train_x.std(0) + 1e-6
    np.savez(output / "normalization.npz", mean=mean, std=std)
    positive = train_y.mean()
    weights = np.where(train_y > 0, 0.5 / positive, 0.5 / (1 - positive)).astype(np.float32)
    print(f"training steps {len(train_y)}, crash-label share {positive:.2%}", flush=True)

    network = RiskNetwork(config["width"], config["layers"])
    params = network.init(jax.random.key(config["seed"]), jnp.zeros((1, train_x.shape[1])))
    state = TrainState.create(apply_fn=network.apply, params=params,
                              tx=optax.chain(optax.clip_by_global_norm(8), optax.adamw(config["learning_rate"], weight_decay=1e-4)))

    @jax.jit
    def step(state, x, y, w):
        def loss(p):
            return jnp.mean(optax.sigmoid_binary_cross_entropy(state.apply_fn(p, x), y) * w)
        value, grads = jax.value_and_grad(loss)(state.params)
        return state.apply_gradients(grads=grads), value

    predict = jax.jit(lambda p, x: jax.nn.sigmoid(network.apply(p, x)))
    normalize = lambda x: (x - mean) / std
    score = lambda xs: np.asarray(predict(state.params, jnp.asarray(normalize(xs))))
    val_x, val_y = np.concatenate(sets["val"][0]), np.concatenate(sets["val"][1])
    rng = np.random.default_rng(config["seed"])
    best, history = -1.0, []
    for epoch in range(config["epochs"]):
        order = rng.permutation(len(train_y))
        losses = []
        for k in range(0, len(order) - config["batch"] + 1, config["batch"]):
            idx = order[k:k + config["batch"]]
            state, value = step(state, jnp.asarray(normalize(train_x[idx])), jnp.asarray(train_y[idx]),
                                jnp.asarray(weights[idx]))
            losses.append(float(value))
        val_auc = auc(score(val_x), val_y)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_auc": val_auc})
        print(f"epoch {epoch + 1:3d}  loss {np.mean(losses):.4f}  val AUC {val_auc:.3f}", flush=True)
        if val_auc > best:
            best = val_auc
            (output / "risk_head.msgpack").write_bytes(serialization.to_bytes(state.params))
    state = state.replace(params=serialization.from_bytes(state.params, (output / "risk_head.msgpack").read_bytes()))

    val_scores = score(val_x)
    threshold = float(np.quantile(val_scores[val_y == 0], 1 - config["false_alarm_rate"]))
    report = {"horizon_s": horizon * 0.1, "threshold": threshold,
              "threshold_rule": f"{config['false_alarm_rate']:.0%} of safe validation steps above it",
              "history": history, "auc": {}, "calibration": {}, "warning": {}}
    for split in ("val", "test"):
        xs, ys, meta = sets[split]
        scores = [score(x) for x in xs]
        flat_s, flat_y = np.concatenate(scores), np.concatenate(ys)
        onset = np.concatenate([np.full(len(y), m["onset"]) for y, m in zip(ys, meta)])
        maps = np.concatenate([np.full(len(y), m["map"]) for y, m in zip(ys, meta)])
        report["auc"][split] = {"all": auc(flat_s, flat_y), "onset groups": auc(flat_s[onset], flat_y[onset]),
                                "other episodes": auc(flat_s[~onset], flat_y[~onset]),
                                **{f"map {m}": auc(flat_s[maps == m], flat_y[maps == m]) for m in sorted(set(maps))},
                                "steps": int(len(flat_y)), "crash steps": int(flat_y.sum())}
        report["calibration"][split] = calibration_metrics(flat_s, flat_y)
    for name in config["train_runs"] + config["heldout_runs"]:
        x, y, meta = runs[name]
        s = score(x)
        entry = {"held_out": name in config["heldout_runs"], "crash_step": meta["crash"], "auc": auc(s, y),
                 "share_of_safe_steps_above_threshold": float(np.mean(s[y == 0] > threshold))}
        if meta["crash"] is not None:
            crash = meta["crash"]
            above = s >= threshold
            last_below = np.flatnonzero(~above[:crash])
            continuous = crash - (last_below[-1] + 1) if len(last_below) else crash
            window = above[max(0, crash - 30):crash]
            first = int(np.argmax(window)) if window.any() else None
            entry.update(warning_continuous_s=round(continuous * 0.1, 1),
                         first_alarm_in_last_3s_s=None if first is None else round((len(window) - first) * 0.1, 1),
                         risk_last_20_steps=np.round(s[max(0, crash - 20):crash], 2).tolist(),
                         true_slip_last_20_steps=np.round(meta["true"][max(0, crash - 20):crash, 6], 2).tolist())
        report["warning"][name] = entry
    (output / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("threshold", "auc")}, indent=1))
    for name, entry in report["warning"].items():
        print(name, {k: v for k, v in entry.items() if k not in ("risk_last_20_steps", "true_slip_last_20_steps")})


if __name__ == "__main__":
    main()
