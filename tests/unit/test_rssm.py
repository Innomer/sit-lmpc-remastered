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


def test_high_slip_windows_find_exactly_the_windows_with_high_slip():
    from sit_lmpc.world.rssm_data import TrackData
    data = TrackData.__new__(TrackData)
    states = np.zeros((31, 7), np.float32)
    states[20, 6] = 0.6
    data.episodes = [{"states": states, "controls": np.zeros((30, 2), np.float32)}]
    windows = data.high_slip_windows(10, 0.4)
    starts = sorted(t for _, t in windows)
    assert starts == list(range(10, 21))


def test_clearance_cost_shape_and_range():
    from sit_lmpc.world.rssm_runtime import RSSMPredictor
    config, model, params = small()
    st, scale = st_params()
    predictor = RSSMPredictor(params, st, scale, jnp.zeros(config.hidden, jnp.float32), jnp.zeros(config.latent, jnp.float32),
                              model, "latent", 0.1, 0.02, 0.35, 2.0)
    state = jnp.asarray([0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0], jnp.float32)
    actions = jax.random.uniform(jax.random.key(7), (6, 5, 2), minval=-1, maxval=1)
    cost = predictor.clearance_cost(state, actions)
    assert cost.shape == (6, 5)
    assert bool(jnp.all(cost >= 0)) and bool(jnp.all(cost <= 2.0))
    off = predictor.replace(clearance_weight=0.0).clearance_cost(state, actions)
    assert float(jnp.abs(off).max()) == 0.0


def test_safety_costs_speed_cap_and_slip():
    from types import SimpleNamespace
    from sit_lmpc.planners.lmppi import safety_step_costs, total_safety
    config = SimpleNamespace(safety_speed_cap=10.0, safety_speed_weight=2.0, safety_slip_limit=0.35,
                             safety_slip_weight=1.0)
    states = jnp.zeros((3, 4, 7))
    states = states.at[0, :, 3].set(9.0).at[1, 2, 3].set(10.25).at[2, 1, 6].set(-0.5)
    terms = safety_step_costs(states, config)
    assert float(terms["speed"][0].max()) == 0.0 and np.isclose(float(terms["speed"][1, 2]), 1.0)
    assert np.isclose(float(terms["slip"][2, 1]), 1.0) and float(terms["slip"][:2].max()) == 0.0
    off = SimpleNamespace(safety_speed_cap=None, safety_speed_weight=1.0, safety_slip_limit=0.35, safety_slip_weight=0.0)
    assert float(jnp.abs(total_safety(states, off)).max()) == 0.0


def test_track_data_ends_episodes_before_the_first_contact(tmp_path):
    import json
    from sit_lmpc.world.rssm_data import TrackData, attempt_contact
    length = 30
    states = np.zeros((length + 1, 7), np.float32)
    states[:, 3] = 5.0
    (tmp_path / "train").mkdir()
    np.savez(tmp_path / "train" / "a.npz", states=states, controls=np.zeros((length, 2), np.float32),
             sensors=np.zeros((length + 1, 7), np.float32), lidar=np.full((length + 1, 1080), 5.0, np.float16),
             conditions=np.ones((length + 1, 4), np.float32))
    record = {"file": "train/a.npz", "map": "Spielberg", "split": "train", "change": {"kind": "none"}}
    (tmp_path / "manifest.json").write_text(json.dumps({"episodes": [record]}))
    assert len(TrackData(tmp_path, "train", 16).episodes[0]["states"]) == length + 1
    (tmp_path / "contacts.json").write_text(json.dumps({"episodes": {"train/a.npz": {"contact": 12, "crash_step": 12}}}))
    episode = TrackData(tmp_path, "train", 16).episodes[0]
    assert len(episode["states"]) == 12 and len(episode["controls"]) == 11 and episode["crash_step"] == 12
    np.savez(tmp_path / "attempt.npz", contacts=np.array([False, False, True, True]))
    assert attempt_contact(np.load(tmp_path / "attempt.npz")) == 2


def test_latent_costs_clearance_and_risk_head():
    from sit_lmpc.training.risk import RiskNetwork
    from sit_lmpc.world.rssm_runtime import RSSMPredictor
    config, model, params = small()
    st, scale = st_params()
    network = RiskNetwork(8, 1)
    risk_params = network.init(jax.random.key(1), jnp.zeros((1, config.hidden + config.latent)))
    base = RSSMPredictor(params, st, scale, jnp.zeros(config.hidden, jnp.float32), jnp.zeros(config.latent, jnp.float32),
                         model, "latent", 0.1, 0.02, 0.35, 2.0, risk_params,
                         jnp.zeros(config.hidden + config.latent, jnp.float32),
                         jnp.ones(config.hidden + config.latent, jnp.float32), network, 3.0)
    state = jnp.asarray([0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0], jnp.float32)
    actions = jax.random.uniform(jax.random.key(7), (6, 5, 2), jnp.float32, minval=-1, maxval=1)
    costs = base.latent_costs(state, actions)
    assert costs["risk"].shape == (6, 5) and bool(jnp.all((costs["risk"] >= 0) & (costs["risk"] <= 3.0)))
    assert np.allclose(costs["clearance"], base.clearance_cost(state, actions), atol=1e-6)
    assert "risk" not in base.replace(risk_weight=0.0).latent_costs(state, actions)


def test_boundary_margin_widens_lateral_offset_per_step():
    from types import SimpleNamespace
    from sit_lmpc.planners.lmppi import boundary_margins, with_margin
    ramp = boundary_margins(SimpleNamespace(boundary_margin=0.3, horizon=3))
    assert np.allclose(ramp, [0.1, 0.2, 0.3])
    listed = boundary_margins(SimpleNamespace(boundary_margin=(0.0, 0.1, 0.4), horizon=3))
    frenet = jnp.zeros((2, 3, 7)).at[0, :, 1].set(0.5).at[1, :, 1].set(-0.5)
    widened = with_margin(frenet, listed)
    assert np.allclose(widened[0, :, 1], [0.5, 0.6, 0.9]) and np.allclose(widened[1, :, 1], [-0.5, -0.6, -0.9])
    assert with_margin(frenet, None) is frenet
    assert boundary_margins(SimpleNamespace(boundary_margin=None, horizon=3)) is None


def test_body_frame_residual_rotates_position_correction_with_heading():
    config = RSSMConfig(hidden=16, condition=2, fast=4, beams=16, head=16, decoder=16, body_frame_residual=True)
    model, params = init_params(config, jax.random.key(0))
    plain = RSSMConfig(hidden=16, condition=2, fast=4, beams=16, head=16, decoder=16)
    plain_model = RSSM(plain)
    h, s = jnp.zeros(16, jnp.float32), jnp.zeros(6, jnp.float32)
    u = jnp.asarray([0.5, 1.0], jnp.float32)
    x0 = jnp.asarray([0.0, 0.0, 0.1, 5.0, 0.0, 0.2, 0.05], jnp.float32)
    raw = plain_model.apply(params, x0, u, h, s, method=RSSM.residual)
    for yaw in (0.0, 0.7, -2.0):
        r = model.apply(params, x0.at[4].set(yaw), u, h, s, method=RSSM.residual)
        assert np.allclose(r[0], np.cos(yaw) * raw[0] - np.sin(yaw) * raw[1], atol=1e-6)
        assert np.allclose(r[1], np.sin(yaw) * raw[0] + np.cos(yaw) * raw[1], atol=1e-6)
        assert np.allclose(r[2:], raw[2:], atol=1e-6)
