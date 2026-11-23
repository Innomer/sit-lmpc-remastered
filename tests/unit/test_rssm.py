"""RSSM world model: bounds, rollout consistency, condition carry-over, finite losses, LiDAR downsampling."""

import numpy as np
import jax
import jax.numpy as jnp

from sit_lmpc.world.rssm import (BODY_HIGH, BODY_INDEX, BODY_LOW, RSSM, RSSMConfig, bicycle, clamp, init_params,
                                 latent_rollout, prior_sample)
from sit_lmpc.world.rssm_data import downsample
from sit_lmpc.world.rssm_train import sequence_loss

def st_params():
    from sit_lmpc.config import load_config
    from sit_lmpc.experiments.runner import build_vehicle_dynamics
    _, st = build_vehicle_dynamics(load_config("configs/world/world-nochance.yaml"))
    return jnp.asarray(st.params, jnp.float32), jnp.asarray(st.action_scale, jnp.float32)


def small():
    config = RSSMConfig(hidden=16, condition=2, fast=4, beams=16, head=16, decoder=16)
    model, params = init_params(config, jax.random.key(0))
    return config, model, params


def test_anchor_and_clamp_stay_physical():
    config, model, params = small()
    h = jax.random.normal(jax.random.key(1), (50, config.hidden)) * 10
    s = jax.random.normal(jax.random.key(2), (50, config.latent)) * 10
    anchor = model.apply(params, h, s, method=RSSM.anchor)
    assert bool(jnp.all(anchor >= BODY_LOW)) and bool(jnp.all(anchor <= BODY_HIGH))
    x = jnp.full((7,), 100.0)
    assert bool(jnp.all(clamp(x)[BODY_INDEX] == BODY_HIGH))


def test_prior_sample_carries_the_condition_part():
    config, model, params = small()
    s = jnp.arange(config.latent, dtype=jnp.float32)
    nxt = prior_sample(model, params, jnp.zeros(config.hidden), s, jax.random.key(3))
    assert np.allclose(nxt[:config.condition], s[:config.condition])


def test_latent_rollout_matches_a_manual_loop():
    config, model, params = small()
    st, scale = st_params()
    h, s = jnp.zeros(config.hidden), jnp.zeros(config.latent)
    x = jnp.asarray([0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0])
    actions = jax.random.uniform(jax.random.key(4), (6, 2), minval=-1, maxval=1)
    xs, hs, ss = latent_rollout(model, params, st, scale, 0.1, 0.02, h, s, x, actions, jax.random.key(5), sample=False)
    for a in actions:
        u = a * scale
        x = clamp(bicycle(st, x, u, 0.1, 0.02) + model.apply(params, x, u, h, s, method=RSSM.residual))
        h = model.apply(params, h, s, u, x, method=RSSM.advance)
        s = prior_sample(model, params, h, s, jax.random.key(0), sample=False)
    assert np.allclose(xs[-1], x, atol=1e-5) and np.allclose(hs[-1], h, atol=1e-5)


def test_sequence_loss_is_finite_and_kl_nonnegative():
    config, model, params = small()
    st, _ = st_params()
    length = 12
    rng = np.random.default_rng(0)
    states = np.zeros((length + 1, 7), np.float32)
    states[:, 3] = 5.0
    states[:, 0] = np.arange(length + 1) * 0.5
    batch = {"states": states, "controls": rng.normal(0, 0.5, (length, 2)).astype(np.float32),
             "sensors": rng.normal(0, 1, (length + 1, 9)).astype(np.float32),
             "pose": states[:, [0, 1, 4]], "lidar": np.log(rng.uniform(0.5, 5, (length + 1, 16))).astype(np.float32),
             "conditions": np.tile([0.9, 1.0], (length + 1, 1)).astype(np.float32)}
    batch = {k: jnp.asarray(v) for k, v in batch.items()}
    settings = {"condition_walk_std": 0.05, "free_nats": 1.0, "prediction_steps": 4, "prediction_stride": 4}
    terms = sequence_loss(params, model, st, 0.1, 0.02, batch, jax.random.key(6), settings, jnp.zeros(9), jnp.ones(9))
    for name, value in terms.items():
        assert bool(jnp.all(jnp.isfinite(value))), name
    assert float(terms["kl_c"]) >= 0 and float(terms["kl_d"]) >= 0


def test_downsample_keeps_the_nearest_return():
    scan = np.full((2, 1080), 5.0)
    scan[0, 500] = 0.3
    small_scan = downsample(scan, 64)
    assert small_scan.shape == (2, 64) and small_scan[0].min() == 0.3 and small_scan[1].min() == 5.0


def test_runtime_filter_matches_training_filter(tmp_path):
    import yaml
    from flax import serialization
    from sit_lmpc.world.rssm_runtime import RSSMFilter, resample_scan
    from sit_lmpc.world.rssm_train import run_sequence
    config, model, params = small()
    st, scale = st_params()
    model_config = {"hidden": 16, "condition": 2, "fast": 4, "beams": 16, "head": 16, "decoder": 16}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"model": model_config, "timestep": 0.1,
                                                         "integrator_timestep": 0.02}))
    (tmp_path / "best.msgpack").write_bytes(serialization.to_bytes(params))
    np.savez(tmp_path / "observation_scale.npz", mean=np.zeros(9), std=np.ones(9))
    runtime = RSSMFilter(tmp_path, st, scale)
    rng = np.random.default_rng(1)
    length = 8
    scans = rng.uniform(0.5, 8.0, (length + 1, 1080)).astype(np.float32)
    sensors = rng.normal(0, 1, (length + 1, 9)).astype(np.float32)
    controls = rng.normal(0, 0.5, (length, 2)).astype(np.float32)
    pose = np.column_stack((np.arange(length + 1) * 0.5, np.zeros(length + 1), np.zeros(length + 1))).astype(np.float32)
    rear = pose.copy()
    rear[:, 0] -= 0.17145
    anchors = [runtime.step(sensors[t], scans[t], rear[t], controls[t - 1] if t else np.zeros(2))[0]
               for t in range(length + 1)]
    states = np.zeros((length + 1, 7), np.float32)
    states[:, 3] = 5.0
    batch = {"states": states, "controls": controls, "sensors": sensors, "pose": pose,
             "lidar": np.asarray(runtime.lidar_features(scans[0]))[None].repeat(length + 1, 0),
             "conditions": np.tile([0.9, 1.0], (length + 1, 1)).astype(np.float32)}
    batch["lidar"] = np.stack([np.asarray(runtime.lidar_features(s)) for s in scans])
    batch = {k: jnp.asarray(v) for k, v in batch.items()}
    settings = {"condition_walk_std": 0.05, "free_nats": 1.0, "prediction_steps": 4, "prediction_stride": 4}
    _, aux = run_sequence(params, model, st, 0.1, 0.02, batch, jax.random.key(0), settings, jnp.zeros(9), jnp.ones(9),
                          sample=False)
    assert np.allclose(np.asarray(anchors[1:]), np.asarray(aux["filtered"]["anchor"]), atol=1e-4)
    predictor = runtime.predictor()
    xs = predictor.rollout_members(jnp.asarray(runtime.x), jnp.zeros((5, 2)))
    assert xs.shape == (1, 5, 7) and bool(jnp.all(jnp.isfinite(xs)))
    assert resample_scan(np.full(819, 3.0), -np.deg2rad(135), np.deg2rad(270) / 818).shape == (1080,)
