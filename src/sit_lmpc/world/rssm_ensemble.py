"""Train ensemble transition heads on a frozen RSSM: shared encoder, memory and main transition stay as trained; each
head is a separate residual network trained on its own random half of the training windows.

    python -m sit_lmpc.world.rssm_ensemble --config configs/train-rssm-ensemble.yaml --output runs/rssm-v8-ensemble

Loss per head: log(1 + mean squared state error) per rollout, so a few spin windows cannot dominate, over latent
rollouts of `prediction_steps` from the true state, along the main model's own
memory/latent path (the planner rolls the heads out the same way), counted only on the head's half of the windows.
The output folder is a complete RSSM run (config, best.msgpack, observation_scale.npz) with model.heads set.
"""

import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import optax
import yaml
from flax import serialization, traverse_util

from sit_lmpc.config import load_config
from sit_lmpc.experiments.runner import build_vehicle_dynamics
from sit_lmpc.world.model import state_error
from sit_lmpc.world.rssm import RSSMConfig, ensemble_rollout, init_params, latent_rollout
from sit_lmpc.world.rssm_data import MapFreeData, TrackData
from sit_lmpc.world.rssm_train import run_sequence


def membership(source, items, heads):
    """Fixed random half of the windows for each head: the same window always goes to the same heads."""
    return np.stack([np.random.default_rng([source, i, t]).random(heads) < 0.5 for i, t in items]).astype(np.float32)


def draw(data, rng, size, window, high_slip=None, high_slip_fraction=0.0):
    pool = [i for i, e in enumerate(data.episodes) if len(e["controls"]) >= window]
    special = int(round(size * high_slip_fraction)) if high_slip else 0
    items = [(int(i), int(rng.integers(0, len(data.episodes[i]["controls"]) - window + 1)))
             for i in rng.choice(pool, size - special)]
    if special:
        items += [high_slip[j] for j in rng.integers(0, len(high_slip), special)]
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    base = Path(config["base"])
    base_config = yaml.safe_load((base / "config.yaml").read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    heads = int(config["heads"])
    run_config = dict(base_config, model=dict(base_config["model"], heads=heads), ensemble=config)
    (output / "config.yaml").write_text(yaml.safe_dump(run_config))
    shutil.copy(base / "observation_scale.npz", output / "observation_scale.npz")

    _, st = build_vehicle_dynamics(load_config(base_config["vehicle_config"]))
    st_params = jnp.asarray(st.params, jnp.float32)
    dt, integrator = float(base_config["timestep"]), float(base_config["integrator_timestep"])
    model, params = init_params(RSSMConfig(**run_config["model"]), jax.random.key(config["seed"]))
    trained = serialization.msgpack_restore((base / "best.msgpack").read_bytes())
    flat = traverse_util.flatten_dict(serialization.to_state_dict(params))
    for key, value in traverse_util.flatten_dict(trained).items():
        flat[key] = jnp.asarray(value)
    init_rng = np.random.default_rng(config["seed"])
    for key in list(flat):
        if key[1].startswith("ensemble_") and config.get("init") == "perturbed_residual":
            value = np.asarray(flat[(key[0], "residual", *key[2:])])
            flat[key] = jnp.asarray(value + config["init_noise"] * (value.std() + 1e-6) * init_rng.normal(size=value.shape),
                                    jnp.float32)
    params = serialization.from_state_dict(params, traverse_util.unflatten_dict(flat))
    trainable = traverse_util.unflatten_dict({k: k[1].startswith("ensemble_") for k in flat})
    scale = np.load(base / "observation_scale.npz")
    obs_mean, obs_std = jnp.asarray(scale["mean"]), jnp.asarray(scale["std"])

    cut = dict(zip(("max_slip", "max_yaw_rate"), base_config.get("data_cut", [1.0, 8.0])))
    beams, window = base_config["model"]["beams"], config["window"]
    train = TrackData(base_config["dataset"], "train", beams, **cut)
    val = TrackData(base_config["dataset"], "val", beams, **cut)
    free = MapFreeData(base_config["map_free_dataset"], "train", beams, **cut)
    free_val = MapFreeData(base_config["map_free_dataset"], "validation", beams, **cut)
    high_slip = train.high_slip_windows(window, config["high_slip_threshold"])
    settings = dict(base_config["settings"], prediction_steps=1, prediction_stride=window)
    horizon, stride = config["prediction_steps"], config["prediction_stride"]
    ones = jnp.ones(2, jnp.float32)

    def window_rollouts(p, batch):
        _, aux = run_sequence(p, model, st_params, dt, integrator, batch, jax.random.key(0), settings, obs_mean,
                              obs_std, sample=False)
        hs, ss = jax.lax.stop_gradient(aux["hs"]), jax.lax.stop_gradient(aux["ss"])
        states, controls = batch["states"], batch["controls"]
        starts = jnp.arange(0, len(controls) - horizon + 1, stride)

        def one(start):
            x0 = states[start]
            u = jax.lax.dynamic_slice_in_dim(controls, start, horizon)
            _, h_out, s_out = latent_rollout(model, p, st_params, ones, dt, integrator, hs[start], ss[start], x0, u,
                                             jax.random.key(0), sample=False)
            h_in = jax.lax.stop_gradient(jnp.concatenate([hs[start][None], h_out[:-1]]))
            s_in = jax.lax.stop_gradient(jnp.concatenate([ss[start][None], s_out[:-1]]))
            xs = ensemble_rollout(model, p, st_params, ones, dt, integrator, h_in, s_in, x0, u)
            target = jax.lax.dynamic_slice_in_dim(states, start + 1, horizon)
            return xs, target

        return jax.vmap(one)(starts)

    def loss(p, batch, mask):
        xs, target = jax.vmap(lambda b: window_rollouts(p, b))(batch)
        error = jnp.log1p(jnp.mean(state_error(xs, target[:, :, :, None, :]) ** 2, axis=(2, 4))).mean(axis=1)
        return jnp.sum(error * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    schedule = optax.warmup_cosine_decay_schedule(0.0, config["learning_rate"], min(300, config["steps"] // 2), config["steps"],
                                                  config["learning_rate"] * 0.05)
    optimizer = optax.chain(optax.masked(optax.set_to_zero(), jax.tree_util.tree_map(lambda m: not m, trainable)),
                            optax.zero_nans(), optax.clip_by_global_norm(1.0),
                            optax.masked(optax.adamw(schedule, weight_decay=1e-5), trainable))
    opt_state = optimizer.init(params)

    @jax.jit
    def update(p, opt_state, batch, mask):
        value, grads = jax.value_and_grad(loss)(p, batch, mask)
        updates, opt_state = optimizer.update(grads, opt_state, p)
        return optax.apply_updates(p, updates), opt_state, value

    @jax.jit
    def evaluate(p, batch):
        xs, target = jax.vmap(lambda b: window_rollouts(p, b))(batch)
        final, true = xs[:, :, -1], target[:, :, -1]
        position = jnp.hypot(final[..., 0] - true[..., None, 0], final[..., 1] - true[..., None, 1])
        spread_slip = final[..., 6].std(axis=-1)
        centre = final[..., :2].mean(axis=-2, keepdims=True)
        spread_position = jnp.sqrt(jnp.sum((final[..., :2] - centre) ** 2, -1).mean(-1))
        mean_error = jnp.hypot(centre[..., 0, 0] - true[..., 0], centre[..., 0, 1] - true[..., 1])
        return {"position": position, "spread_slip": spread_slip, "spread_position": spread_position,
                "mean_error": mean_error, "true_slip": jnp.abs(target[:, :, :, 6]).max(axis=-1)}

    rng = np.random.default_rng(config["seed"])
    val_rng = np.random.default_rng(config["seed"] + 1)
    val_sets = {"on-track": [val.gather(draw(val, val_rng, config["batch"], window), window)
                             for _ in range(config["val_batches"])],
                "map-free": [free_val.gather(draw(free_val, val_rng, config["batch"], window), window)
                             for _ in range(config["val_batches"])],
                "high-slip": [val.gather([w for w in (val.high_slip_windows(window, 0.4))[k::97]][:config["batch"]], window)
                              for k in range(config["val_batches"])]}
    free_count = int(round(config["batch"] * config["map_free_fraction"]))
    history, best, started = [], np.inf, time.time()
    for step in range(1, config["steps"] + 1):
        items = draw(train, rng, config["batch"] - free_count, window, high_slip, config["high_slip_fraction"])
        extra = draw(free, rng, free_count, window)
        batch = train.gather(items, window, rng)
        more = free.gather(extra, window, rng)
        batch = {k: jnp.asarray(np.concatenate([batch[k], more[k]])) for k in batch}
        mask = jnp.asarray(np.concatenate([membership(0, items, heads), membership(1, extra, heads)]))
        params, opt_state, value = update(params, opt_state, batch, mask)
        if step % config["eval_every"] == 0 or step == config["steps"]:
            record = {"step": step, "train": float(value), "elapsed_min": (time.time() - started) / 60}
            for name, batches in val_sets.items():
                out = [jax.tree_util.tree_map(np.asarray, evaluate(params, {k: jnp.asarray(v) for k, v in b.items()}))
                       for b in batches]
                out = {k: np.concatenate([o[k].reshape(-1, *o[k].shape[2:]) for o in out]) for k in out[0]}
                record[name] = {"position_per_head": out["position"].mean(axis=0).round(3).tolist(),
                                "mean_of_heads_error": float(out["mean_error"].mean()),
                                "spread_slip_p50": float(np.median(out["spread_slip"])),
                                "spread_slip_p95": float(np.percentile(out["spread_slip"], 95)),
                                "spread_position_p50": float(np.median(out["spread_position"])),
                                "spread_position_p95": float(np.percentile(out["spread_position"], 95)),
                                "error_spread_correlation": float(np.corrcoef(out["mean_error"],
                                                                              out["spread_position"])[0, 1])}
            history.append(record)
            selection = float(np.mean(record["on-track"]["position_per_head"]))
            if selection < best:
                best = selection
                (output / "best.msgpack").write_bytes(serialization.to_bytes(params))
            (output / "history.json").write_text(json.dumps(history, indent=1) + "\n")
            print(f"step {step:6d}  train {float(value):.3f}  " + "  ".join(
                f"{name}: heads 1.5 s pos {np.mean(r['position_per_head']):.2f} m, spread pos p50/p95 "
                f"{r['spread_position_p50']:.2f}/{r['spread_position_p95']:.2f} m, slip p50/p95 "
                f"{r['spread_slip_p50']:.3f}/{r['spread_slip_p95']:.3f}, corr {r['error_spread_correlation']:.2f}"
                for name, r in record.items() if isinstance(r, dict)) + f"  {record['elapsed_min']:.1f} min",
                flush=True)


if __name__ == "__main__":
    main()
