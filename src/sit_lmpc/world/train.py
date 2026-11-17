"""Train the world-model encoder and sensor decoder against a frozen residual transition (plan stage 2).

    python -m sit_lmpc.world.train --config configs/train-world-encoder.yaml
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import optax
import yaml
from flax import serialization

from dyna_dist.io.checkpoint import load_model
from sit_lmpc.world.data import WorldData
from sit_lmpc.world.model import Decoder, Encoder, world_loss


def load_transition(path):
    return load_model(path)


def initialize(config, transition, data, key):
    encoder = Encoder(config["hidden"], transition.config.latent_dimension)
    decoder = Decoder(config["hidden"])
    length = config["window"]
    dummy_obs = jnp.zeros((1, length + 1, data.items[0]["observations"].shape[-1]))
    encoder_params = encoder.init(key, dummy_obs)
    decoder_params = decoder.init(jax.random.fold_in(key, 1), jnp.zeros((1, 4)),
                                  jnp.zeros((1, transition.config.latent_dimension)), jnp.zeros((1, 2)), jnp.zeros((1, 4)))
    params = {"encoder": encoder_params, "decoder": decoder_params, "sensor_log_std": jnp.zeros(4)}
    return encoder, decoder, params


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/train-world-encoder.yaml")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.steps:
        config["steps"] = args.steps
    if args.output:
        config["output"] = args.output
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    transition = load_transition(config["transition"])
    train = WorldData(config["dataset"], "train")
    validation = WorldData(config["dataset"], "validation")
    observation_mean, observation_std = train.observation_scale()
    np.savez(output / "observation_scale.npz", mean=observation_mean, std=observation_std)
    key = jax.random.key(config["seed"])
    encoder, decoder, params = initialize(config, transition, train, key)
    optimizer = optax.chain(optax.clip_by_global_norm(config["gradient_clip"]), optax.adam(config["learning_rate"]))
    state = optimizer.init(params)
    length = config["window"]
    train_starts = train.starts(length)
    validation_starts = validation.starts(length)
    rng = np.random.default_rng(config["seed"])

    def loss_fn(p, batch, k, kl_scale=1.0):
        return world_loss(p, transition, encoder, decoder, batch, k, config, observation_mean, observation_std, kl_scale)

    @jax.jit
    def update(p, s, batch, k, kl_scale):
        (value, terms), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, batch, k, kl_scale)
        changes, s = optimizer.update(grads, s, p)
        return optax.apply_updates(p, changes), s, value, terms

    evaluate = jax.jit(loss_fn)
    validation_rng = np.random.default_rng(config["seed"] + 1)
    validation_batches = [validation.batch(validation_rng, config["batch"], length, validation_starts)
                          for _ in range(config["validation_batches"])]
    history = []
    best = np.inf
    started = time.time()
    print(f"[world] train windows {len(train_starts)} | validation windows {len(validation_starts)} | "
          f"items {len(train.items)} / {len(validation.items)}", flush=True)
    for step in range(1, config["steps"] + 1):
        batch = train.batch(rng, config["batch"], length, train_starts)
        kl_scale = min(1.0, step / config["kl_warmup"])
        params, state, value, terms = update(params, state, batch, jax.random.fold_in(key, step), kl_scale)
        if step % config["validation_every"] == 0 or step == 1:
            values = [evaluate(params, b, jax.random.key(i)) for i, b in enumerate(validation_batches)]
            val_total = float(np.mean([v[0] for v in values]))
            val_terms = {name: float(np.mean([v[1][name] for v in values])) for name in values[0][1]}
            record = {"step": step, "train": float(value), "validation": val_total,
                      **{f"train_{k}": float(v) for k, v in terms.items()},
                      **{f"validation_{k}": v for k, v in val_terms.items()}, "seconds": time.time() - started}
            history.append(record)
            print(f"[world] step={step} | train={float(value):.4f} | validation={val_total:.4f} | "
                  + " | ".join(f"{k}={v:.4f}" for k, v in val_terms.items()), flush=True)
            selection = val_terms["anchor"] + val_terms["prediction"] + val_terms["true_state_prediction"]
            record["selection"] = selection
            if selection < best:
                best = selection
                (output / "best.msgpack").write_bytes(serialization.to_bytes(params))
            (output / "history.json").write_text(json.dumps(history, indent=1) + "\n")
    (output / "last.msgpack").write_bytes(serialization.to_bytes(params))
    (output / "summary.json").write_text(json.dumps({"best_validation": best, "steps": config["steps"],
                                                     "seconds": time.time() - started}, indent=2) + "\n")
    print(f"[world] saved {output}", flush=True)


if __name__ == "__main__":
    main()
