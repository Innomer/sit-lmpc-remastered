"""Grip probe on a frozen RSSM: predicts the true friction under the car from the filtered memory h and the fast
latent d (the slow part c is left out: it did not turn into a grip variable, entry 32 of EXPERIMENTS.md).

    python -m sit_lmpc.world.rssm_grip --config configs/train-grip-probe.yaml --output runs/rssm-v11-grip

Training data: the on-track and coverage datasets' train splits, every step labelled with its true friction; every
episode is filtered from its start with the frozen RSSM (posterior mean), as in the planner. Validation: their val
splits. Network: SIT-LMPC's RiskNetwork (MLP) with a single output, mean squared error on the standardised grip.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import optax
import yaml
from flax import serialization
from flax.training.train_state import TrainState

from sit_lmpc.training.risk import RiskNetwork
from sit_lmpc.world.rssm_data import TrackData, log_range
from sit_lmpc.world.rssm_risk import LatentFilter

GRIP_MEAN, GRIP_SCALE = 0.8, 0.25


def probe_inputs(features, hidden, condition):
    """(h, d) from LatentFilter features [h, c, d]."""
    return np.concatenate([features[:, :hidden], features[:, hidden + condition:]], axis=-1)


class GripProbe:
    def __init__(self, folder):
        folder = Path(folder)
        config = yaml.safe_load((folder / "config.yaml").read_text())
        scale = np.load(folder / "normalization.npz")
        self.mean, self.std = jnp.asarray(scale["mean"]), jnp.asarray(scale["std"])
        self.network = RiskNetwork(config["width"], config["layers"])
        template = self.network.init(jax.random.key(0), jnp.zeros((1, len(scale["mean"]))))
        self.params = serialization.from_bytes(template, (folder / "probe.msgpack").read_bytes())
        self.hidden, self.condition = config["hidden"], config["condition"]

    def __call__(self, h, s):
        x = jnp.concatenate([h, s[..., self.condition:]], -1)
        out = self.network.apply(self.params, ((x - self.mean) / self.std).reshape(-1, x.shape[-1]))
        return (out * GRIP_SCALE + GRIP_MEAN).reshape(x.shape[:-1])


def episodes_of(roots, split, config, cut):
    out = []
    for root in roots:
        data = TrackData(root, split, config["model"]["beams"], **cut)
        out += [e for e in data.episodes if len(e["states"]) > 5]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    base = Path(config["base"])
    base_config = yaml.safe_load((base / "config.yaml").read_text())
    filt = LatentFilter(base)
    hidden, condition = filt.config.hidden, filt.config.condition
    (output / "config.yaml").write_text(yaml.safe_dump(dict(config, hidden=hidden, condition=condition)))
    cut = dict(zip(("max_slip", "max_yaw_rate"), base_config.get("data_cut", [1.0, 8.0])))
    roots = [base_config["dataset"]] + ([base_config["coverage_dataset"]] if base_config.get("coverage_dataset") else [])
    sets = {}
    for split in ("train", "val"):
        episodes = episodes_of(roots, split, base_config, cut)
        inputs = [{"sensors": e["sensors"], "lidar": log_range(e["lidar"]), "pose": e["pose"], "controls": e["controls"]}
                  for e in episodes]
        features = filt.features(inputs)
        burn = config["burn_in"]
        x = np.concatenate([probe_inputs(f, hidden, condition)[burn:] for f in features])
        y = np.concatenate([e["conditions"][burn:, 0] for e in episodes]).astype(np.float32)
        source = np.concatenate([np.full(max(len(e["states"]) - burn, 0), e["map"] == "open") for e in episodes])
        sets[split] = (x, y, source)
        print(f"{split}: {len(episodes)} episodes, {len(y)} steps, grip {y.min():.2f}..{y.max():.2f}", flush=True)
    x, y, _ = sets["train"]
    mean, std = x.mean(0), x.std(0) + 1e-6
    np.savez(output / "normalization.npz", mean=mean, std=std)
    target = (y - GRIP_MEAN) / GRIP_SCALE
    network = RiskNetwork(config["width"], config["layers"])
    state = TrainState.create(apply_fn=network.apply, params=network.init(jax.random.key(config["seed"]), jnp.zeros((1, x.shape[1]))),
                              tx=optax.chain(optax.clip_by_global_norm(8), optax.adamw(config["learning_rate"], weight_decay=1e-4)))

    @jax.jit
    def step(state, xb, yb):
        def loss(p):
            return jnp.mean((state.apply_fn(p, xb) - yb) ** 2)
        value, grads = jax.value_and_grad(loss)(state.params)
        return state.apply_gradients(grads=grads), value

    predict = jax.jit(lambda p, xb: network.apply(p, xb) * GRIP_SCALE + GRIP_MEAN)
    score = lambda p, xs: np.concatenate([np.asarray(predict(p, jnp.asarray((xs[k:k + 65536] - mean) / std)))
                                          for k in range(0, len(xs), 65536)])
    vx, vy, vsource = sets["val"]
    rng = np.random.default_rng(config["seed"])
    best, history = np.inf, []
    for epoch in range(config["epochs"]):
        order = rng.permutation(len(y))
        for k in range(0, len(order) - config["batch"] + 1, config["batch"]):
            idx = order[k:k + config["batch"]]
            state, _ = step(state, jnp.asarray((x[idx] - mean) / std), jnp.asarray(target[idx]))
        p = score(state.params, vx)
        mae = float(np.mean(np.abs(p - vy)))
        history.append({"epoch": epoch + 1, "val_mae": mae})
        print(f"epoch {epoch + 1:3d}  val MAE {mae:.3f}", flush=True)
        if mae < best:
            best = mae
            (output / "probe.msgpack").write_bytes(serialization.to_bytes(state.params))
    params = serialization.from_bytes(state.params, (output / "probe.msgpack").read_bytes())
    p = score(params, vx)
    report = {"history": history}
    for name, mask in (("all", np.ones(len(vy), bool)), ("track", ~vsource.astype(bool)), ("coverage", vsource.astype(bool))):
        err = p[mask] - vy[mask]
        report[name] = {"steps": int(mask.sum()), "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
                        "r2": float(1 - np.mean(err ** 2) / np.var(vy[mask])), "within_0.1": float(np.mean(np.abs(err) <= 0.1))}
    (output / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=1))


if __name__ == "__main__":
    main()
