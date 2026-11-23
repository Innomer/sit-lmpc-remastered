"""Train the RSSM world model on the on-track dataset.

    python -m sit_lmpc.world.rssm_train --config configs/train-rssm.yaml --output runs/rssm-v1
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

from sit_lmpc.config import load_config
from sit_lmpc.experiments.runner import build_vehicle_dynamics
from sit_lmpc.world.model import kl_diag, state_error
from sit_lmpc.world.rssm import BODY_INDEX, BODY_SCALE, RSSM, RSSMConfig, bicycle, clamp, init_params
from sit_lmpc.world.rssm_data import TrackData
from sit_lmpc.world.rssm_plots import lidar_and_paths, signals

CONDITION_MEAN = jnp.array([0.8, 1.05])
CONDITION_SCALE = jnp.array([0.25, 0.15])


def gaussian_nll(target, mean, log_std):
    return 0.5 * ((target - mean) / jnp.exp(log_std)) ** 2 + log_std


def merge(pose, body_state):
    return jnp.stack([pose[0], pose[1], body_state[0], body_state[1], pose[2], body_state[2], body_state[3]])


def sequence_loss(params, model, st_params, dt, integrator, batch, key, settings, obs_mean, obs_std):
    return run_sequence(params, model, st_params, dt, integrator, batch, key, settings, obs_mean, obs_std)[0]


def run_sequence(params, model, st_params, dt, integrator, batch, key, settings, obs_mean, obs_std, sample=True):
    states, controls, pose = batch["states"], batch["controls"], batch["pose"]
    sensors = (batch["sensors"] - obs_mean) / obs_std
    lidar, conditions = batch["lidar"], batch["conditions"]
    c = model.config.condition
    walk = settings["condition_walk_std"]
    anchor_weights = jnp.asarray(settings.get("anchor_weights", [1.0, 1.0, 1.0, 1.0]))
    length = len(controls)
    keys = jax.random.split(key, length + 1)

    def apply(method, *args):
        return model.apply(params, *args, method=method)

    h0 = jnp.zeros(model.config.hidden)
    q_mean, q_log_std = apply(RSSM.posterior, h0, apply(RSSM.embed, lidar[0], sensors[0]))
    s0 = q_mean + jnp.exp(q_log_std) * jax.random.normal(keys[0], q_mean.shape) * sample
    kl0 = kl_diag(q_mean, q_log_std, 0.0, 1.0)
    x0 = merge(pose[0], apply(RSSM.anchor, h0, s0))

    def step(carry, inputs):
        h, s, x, c_mean, c_var = carry
        t, k = inputs
        u = controls[t - 1]
        x_pred = clamp(jax.lax.stop_gradient(bicycle(st_params, x, u, dt, integrator)) + apply(RSSM.residual, x, u, h, s))
        h_next = apply(RSSM.advance, h, s, u, x_pred)
        p_mean, p_log_std = apply(RSSM.prior, h_next)
        q_mean, q_log_std = apply(RSSM.posterior, h_next, apply(RSSM.embed, lidar[t], sensors[t]))
        s_next = q_mean + jnp.exp(q_log_std) * jax.random.normal(k, q_mean.shape) * sample
        kl_c = kl_diag(q_mean[:c], q_log_std[:c], c_mean, c_var + walk ** 2)
        kl_d = kl_diag(q_mean[c:], q_log_std[c:], p_mean, jnp.exp(2 * p_log_std))
        anchor = apply(RSSM.anchor, h_next, s_next)
        x_next = merge(pose[t], anchor)
        teacher = jax.lax.stop_gradient(bicycle(st_params, states[t - 1], u, dt, integrator)) + apply(RSSM.residual, states[t - 1], u, h, s)
        l_mean, l_log_std = apply(RSSM.decode_lidar, h_next, s_next)
        o_mean, o_log_std = apply(RSSM.decode_sensors, h_next, s_next, u)
        condition = apply(RSSM.condition_readout, s_next)
        out = {
            "anchor": jnp.sum(anchor_weights * ((anchor - states[t, BODY_INDEX]) / (0.1 * BODY_SCALE)) ** 2)
                      / jnp.sum(anchor_weights),
            "residual": jnp.mean(state_error(teacher, states[t]) ** 2),
            "lidar": jnp.mean(gaussian_nll(lidar[t], l_mean, l_log_std)),
            "sensors": jnp.mean(gaussian_nll(sensors[t, :4], o_mean, o_log_std)),
            "kl_c": kl_c,
            "kl_d": kl_d,
            "condition": jnp.mean(((condition - (conditions[t] - CONDITION_MEAN) / CONDITION_SCALE)) ** 2),
            "anchor_error": anchor - states[t, BODY_INDEX],
        }
        aux = {"anchor": anchor, "lidar_mean": l_mean, "lidar_log_std": l_log_std,
               "condition": condition * CONDITION_SCALE + CONDITION_MEAN}
        carry = (h_next, s_next, jax.lax.stop_gradient(x_next), q_mean[:c], jnp.exp(2 * q_log_std[:c]))
        return carry, (out, aux, h_next, s_next, x_next)

    init = (h0, s0, jax.lax.stop_gradient(x0), q_mean[:c], jnp.exp(2 * q_log_std[:c]))
    _, (terms, filtered, hs, ss, xs) = jax.lax.scan(step, init, (jnp.arange(1, length + 1), keys[1:]))
    hs = jnp.concatenate([h0[None], hs])
    ss = jnp.concatenate([s0[None], ss])
    xs = jnp.concatenate([x0[None], xs])

    horizon = settings["prediction_steps"]
    starts = jnp.arange(0, length - horizon + 1, settings["prediction_stride"])

    def predict(start):
        def roll(carry, i):
            h, s, x = carry
            u = controls[start + i]
            x_next = clamp(jax.lax.stop_gradient(bicycle(st_params, x, u, dt, integrator)) + apply(RSSM.residual, x, u, h, s))
            h_next = apply(RSSM.advance, h, s, u, x_next)
            p_mean, _ = apply(RSSM.prior, h_next)
            s_next = jnp.concatenate([s[:c], p_mean])
            l_mean, l_log_std = apply(RSSM.decode_lidar, h_next, s_next)
            target = start + i + 1
            return (h_next, s_next, x_next), {
                "state": jnp.mean(state_error(x_next, states[target]) ** 2),
                "lidar": jnp.mean(gaussian_nll(lidar[target], l_mean, l_log_std)),
                "consistency": jnp.mean(((apply(RSSM.anchor, h_next, s_next) - x_next[BODY_INDEX]) / BODY_SCALE) ** 2),
                "position": jnp.hypot(x_next[0] - states[target, 0], x_next[1] - states[target, 1]),
                "x": x_next, "lidar_mean": l_mean,
            }
        _, out = jax.lax.scan(roll, (hs[start], ss[start], xs[start]), jnp.arange(horizon))
        return out

    prediction = jax.vmap(predict)(starts)
    free = settings["free_nats"]
    kl = jnp.maximum(terms["kl_d"].mean(), free) + terms["kl_c"].mean() + kl0 / length
    aux = {"filtered": filtered, "x_filtered": xs, "rollout_x": prediction["x"],
           "rollout_lidar": prediction["lidar_mean"], "starts": starts}
    return {
        "anchor": terms["anchor"].mean(),
        "residual": terms["residual"].mean(),
        "lidar": terms["lidar"].mean(),
        "sensors": terms["sensors"].mean(),
        "kl": kl,
        "kl_c": terms["kl_c"].mean(),
        "kl_d": terms["kl_d"].mean(),
        "condition": terms["condition"].mean(),
        "prediction": prediction["state"].mean(),
        "prediction_lidar": prediction["lidar"].mean(),
        "consistency": prediction["consistency"].mean(),
        "position_1s": prediction["position"][:, -1].mean(),
        "anchor_rmse": jnp.sqrt(jnp.mean(terms["anchor_error"] ** 2, axis=0)),
    }, aux


def diagnostic_windows(data, window, rng):
    chosen = []
    for name in sorted({e["map"] for e in data.episodes}):
        for kinds in (("patches", "step"), ("none", "patches", "step")):
            pool = [i for i, e in enumerate(data.episodes)
                    if e["map"] == name and e["change"] in kinds and len(e["controls"]) >= window + 20]
            if pool:
                index = int(rng.choice(pool))
                start = int(rng.integers(10, len(data.episodes[index]["controls"]) - window + 1))
                chosen.append((f"{name}-{data.episodes[index]['change']}", index, start))
                break
    return chosen


def scalars(writer, prefix, terms, step):
    for name, value in terms.items():
        if name == "anchor_rmse":
            for label, v in zip(("steering", "speed", "yaw_rate", "slip"), np.asarray(value)):
                writer.add_scalar(f"{prefix}/anchor_rmse_{label}", float(v), step)
        else:
            writer.add_scalar(f"{prefix}/{name}", float(value), step)


SIGNALS = (("steering", 2, 0), ("speed", 3, 1), ("yaw_rate", 5, 2), ("slip", 6, 3))


def comparison_layout(labels):
    layout = {}
    for label in labels:
        charts = {name: ["Multiline", [f"^{label}/{name}/(true|model)$"]] for name, _, _ in SIGNALS}
        charts["grip"] = ["Multiline", [f"^{label}/grip/(true|model)$"]]
        charts["mass_ratio"] = ["Multiline", [f"^{label}/mass_ratio/(true|model)$"]]
        charts["lidar_last_scan"] = ["Multiline", [f"^{label}/lidar_last_scan/(true|reconstructed|predicted_1s)$"]]
        charts["prediction_error_m"] = ["Multiline", [f"^{label}/prediction_error_m/(mean|worst)$"]]
        layout[label] = charts
    return layout


def log_comparisons(writer, label, batch, aux):
    states, conditions = batch["states"], batch["conditions"]
    anchor, condition = aux["filtered"]["anchor"], aux["filtered"]["condition"]
    for t in range(len(anchor)):
        for name, index, column in SIGNALS:
            writer.add_scalar(f"{label}/{name}/true", float(states[t + 1, index]), t + 1)
            writer.add_scalar(f"{label}/{name}/model", float(anchor[t, column]), t + 1)
        for name, column in (("grip", 0), ("mass_ratio", 1)):
            writer.add_scalar(f"{label}/{name}/true", float(conditions[t + 1, column]), t + 1)
            writer.add_scalar(f"{label}/{name}/model", float(condition[t, column]), t + 1)
    lidar, recon = np.exp(batch["lidar"]), np.exp(aux["filtered"]["lidar_mean"])
    last = len(recon) - 1
    predicted = np.exp(aux["rollout_lidar"][-1, -1])
    for beam in range(lidar.shape[1]):
        writer.add_scalar(f"{label}/lidar_last_scan/true", float(lidar[last + 1, beam]), beam)
        writer.add_scalar(f"{label}/lidar_last_scan/reconstructed", float(recon[last, beam]), beam)
        writer.add_scalar(f"{label}/lidar_last_scan/predicted_1s", float(predicted[beam]), beam)
    starts, k = aux["starts"], aux["rollout_x"].shape[1]
    errors = np.asarray([np.hypot(*(aux["rollout_x"][i][:, :2] - states[int(s) + 1:int(s) + 1 + k, :2]).T)
                         for i, s in enumerate(starts)])
    for j in range(k):
        writer.add_scalar(f"{label}/prediction_error_m/mean", float(errors[:, j].mean()), j + 1)
        writer.add_scalar(f"{label}/prediction_error_m/worst", float(errors[:, j].max()), j + 1)


def total(terms, weights, kl_scale):
    return sum(weights[name] * terms[name] for name in weights if name != "kl") + kl_scale * weights["kl"] * terms["kl"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config))
    steps = args.steps or config["steps"]

    planner_config = load_config(config["vehicle_config"])
    _, st = build_vehicle_dynamics(planner_config)
    st_params = jnp.asarray(st.params, jnp.float32)
    dt, integrator = float(config["timestep"]), float(config["integrator_timestep"])
    rssm_config = RSSMConfig(**config["model"])
    model, params = init_params(rssm_config, jax.random.key(config["seed"]))

    train = TrackData(config["dataset"], "train", rssm_config.beams)
    val = TrackData(config["dataset"], "val", rssm_config.beams)
    obs_mean, obs_std = train.observation_scale()
    np.savez(output / "observation_scale.npz", mean=obs_mean, std=obs_std)
    obs_mean, obs_std = jnp.asarray(obs_mean), jnp.asarray(obs_std)
    rng = np.random.default_rng(config["seed"])
    val_rng = np.random.default_rng(config["seed"] + 1)
    val_batches = [val.sample(val_rng, config["batch"], config["window"], noise=False) for _ in range(config["val_batches"])]

    weights = config["weights"]
    settings = config["settings"]
    schedule = optax.warmup_cosine_decay_schedule(0.0, config["learning_rate"], min(500, steps // 2), steps, config["learning_rate"] * 0.05)
    optimizer = optax.chain(optax.zero_nans(), optax.clip_by_global_norm(config["clip"]),
                            optax.adamw(schedule, weight_decay=1e-5))
    opt_state = optimizer.init(params)

    def batch_terms(params, batch, key):
        keys = jax.random.split(key, len(batch["controls"]))
        terms = jax.vmap(lambda b, k: sequence_loss(params, model, st_params, dt, integrator, b, k, settings,
                                                    obs_mean, obs_std))(batch, keys)
        return jax.tree_util.tree_map(lambda v: v.mean(axis=0), terms)

    @jax.jit
    def update(params, opt_state, batch, key, kl_scale):
        def loss(p):
            terms = batch_terms(p, batch, key)
            return total(terms, weights, kl_scale), terms
        (value, terms), grads = jax.value_and_grad(loss, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, value, terms

    evaluate = jax.jit(batch_terms)
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(str(output / "tb"))
    windows = diagnostic_windows(val, config["diagnostic_window"], np.random.default_rng(config["seed"] + 2))
    diagnostic_batches = [(label, {k: v[0] for k, v in val.gather([(index, start)], config["diagnostic_window"]).items()})
                          for label, index, start in windows]
    writer.add_custom_scalars(comparison_layout([label for label, _ in diagnostic_batches]))
    run_one = jax.jit(lambda p, b, k: run_sequence(p, model, st_params, dt, integrator, b, k, settings, obs_mean, obs_std))
    history = []
    best = np.inf
    running = None
    skipped = 0
    worse = 0
    started = time.time()
    key = jax.random.key(config["seed"])
    for step in range(1, steps + 1):
        key, sub = jax.random.split(key)
        batch = train.sample(rng, config["batch"], config["window"])
        kl_scale = min(1.0, step / config["kl_warmup"])
        new_params, new_opt_state, value, terms = update(params, opt_state, batch, sub, kl_scale)
        value_f = float(value)
        if running is not None and step > 500 and (not np.isfinite(value_f) or value_f > config["skip_factor"] * running):
            skipped += 1
            writer.add_scalar("train/skipped_batches", skipped, step)
            continue
        params, opt_state = new_params, new_opt_state
        running = value_f if running is None else 0.98 * running + 0.02 * value_f
        if step % 50 == 0:
            scalars(writer, "train", terms, step)
            writer.add_scalar("train/total", float(value), step)
            writer.add_scalar("train/kl_scale", kl_scale, step)
            writer.add_scalar("train/learning_rate", float(schedule(step)), step)
        if step % config["eval_every"] == 0 or step == steps:
            val_terms = [evaluate(params, b, jax.random.key(i)) for i, b in enumerate(val_batches)]
            mean = {k: float(np.mean([np.asarray(t[k]) for t in val_terms])) for k in val_terms[0] if k != "anchor_rmse"}
            mean["anchor_rmse"] = np.mean([np.asarray(t["anchor_rmse"]) for t in val_terms], axis=0).tolist()
            selection = sum(mean[k] for k in config["selection"])
            record = {"step": step, "train_total": float(value), "selection": selection, "skipped": skipped,
                      "elapsed_min": (time.time() - started) / 60, "val": mean}
            history.append(record)
            scalars(writer, "val", {k: v for k, v in mean.items()}, step)
            writer.add_scalar("val/selection", selection, step)
            for label, b in diagnostic_batches:
                jb = {k: jnp.asarray(v) for k, v in b.items()}
                _, aux = run_one(params, jb, jax.random.key(0))
                aux = jax.tree_util.tree_map(np.asarray, aux)
                writer.add_image(f"{label}/signals", signals(b, aux), step, dataformats="HWC")
                writer.add_image(f"{label}/lidar_and_paths", lidar_and_paths(b, aux), step, dataformats="HWC")
                if step % config["comparison_every"] == 0 or step == steps:
                    comparison = SummaryWriter(str(output / "tb" / f"diag-{step:06d}"))
                    log_comparisons(comparison, label, b, aux)
                    comparison.close()
            writer.flush()
            worse = worse + 1 if selection > config["stop_factor"] * best else 0
            if selection < best:
                best = selection
                (output / "best.msgpack").write_bytes(serialization.to_bytes(params))
            (output / "last.msgpack").write_bytes(serialization.to_bytes(params))
            (output / "history.json").write_text(json.dumps(history, indent=1) + "\n")
            rmse = mean["anchor_rmse"]
            print(f"step {step:6d}  total {float(value):9.3f}  val anchor {mean['anchor']:.3f} (steer {rmse[0]:.3f} "
                  f"v {rmse[1]:.2f} r {rmse[2]:.2f} slip {rmse[3]:.3f})  pred {mean['prediction']:.2f}  "
                  f"1s pos {mean['position_1s']:.2f} m  lidar {mean['lidar']:.3f}  kl_c {mean['kl_c']:.2f} "
                  f"kl_d {mean['kl_d']:.2f}  cond {mean['condition']:.3f}  {record['elapsed_min']:.1f} min", flush=True)
            if worse >= config["stop_patience"]:
                print(f"early stop at step {step}: validation {config['stop_factor']}x worse than best for "
                      f"{worse} evaluations", flush=True)
                break
    (output / "summary.json").write_text(json.dumps({"best_selection": best, "steps": steps,
                                                     "final": history[-1]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
