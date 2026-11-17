"""Reference checks for the world-model math before any training run."""

from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import pytest

ROOT = Path(__file__).resolve().parents[2]
TRANSITION = ROOT / "runs/residual-world/best_refit"
DATASET = ROOT / "data/f1tenth-world"
pytestmark = pytest.mark.skipif(not TRANSITION.exists() or not DATASET.exists(),
                                reason="needs the trained residual transition and the world dataset")


def numpy_kl_diag(mean_q, std_q, mean_p, std_p):
    var_q, var_p = std_q ** 2, std_p ** 2
    return 0.5 * np.sum(var_q / var_p + (mean_q - mean_p) ** 2 / var_p - 1 + np.log(var_p) - np.log(var_q), axis=-1)


@pytest.fixture(scope="module")
def transition():
    from dyna_dist.io.checkpoint import load_model

    return load_model(TRANSITION)


@pytest.fixture(scope="module")
def trajectory():
    from sit_lmpc.world.data import WorldData

    data = WorldData(DATASET, "test")
    return next(item for item in data.items if item["kind"] == "instance" and len(item["controls"]) >= 60)


def test_kl_matches_float64_reference_and_is_nonnegative():
    from sit_lmpc.world.model import kl_diag

    rng = np.random.default_rng(0)
    mean_q, mean_p = rng.normal(size=(50, 16)), rng.normal(size=(50, 16))
    log_std_q = rng.normal(-1, 1.5, size=(50, 16))
    std_p = np.exp(rng.normal(-1, 1.5, size=(50, 16)))
    ours = np.asarray(kl_diag(jnp.asarray(mean_q, jnp.float32), jnp.asarray(log_std_q, jnp.float32),
                              jnp.asarray(mean_p, jnp.float32), jnp.asarray(std_p ** 2, jnp.float32)))
    reference = numpy_kl_diag(mean_q, np.exp(log_std_q), mean_p, std_p)
    np.testing.assert_allclose(ours, reference, rtol=1e-4)
    assert np.all(ours >= 0)
    same = np.asarray(kl_diag(jnp.zeros((1, 16)), jnp.zeros((1, 16)), 0.0, 1.0))
    np.testing.assert_allclose(same, 0.0, atol=1e-6)


def test_whitened_latent_has_the_population_prior_moments(transition):
    from dyna_dist.models.features import prior_and_variance, prior_moments_and_variance

    mean, factor, _ = prior_moments_and_variance(transition.parameters, transition.config)
    w = jax.random.normal(jax.random.key(0), (200000, 16))
    z = np.asarray(mean + w @ factor.T, np.float64)
    reference_mean = np.asarray(mean, np.float64)
    reference_cov = np.asarray(factor, np.float64) @ np.asarray(factor, np.float64).T
    np.testing.assert_allclose(z.mean(axis=0), reference_mean, atol=5e-3 * np.sqrt(np.diag(reference_cov)).max())
    np.testing.assert_allclose(np.cov(z.T), reference_cov, atol=2e-2 * np.abs(reference_cov).max())


def test_rollout_at_the_prior_mean_matches_dyna_dist_prediction(transition, trajectory):
    from dyna_dist.inference.posterior import _predict_prior
    from dyna_dist.models.features import prior_moments_and_variance
    from sit_lmpc.world.model import rollout

    mean, _, _ = prior_moments_and_variance(transition.parameters, transition.config)
    state = jnp.asarray(trajectory["states"][20])
    control = jnp.asarray(trajectory["controls"][20:21])
    ours = np.asarray(rollout(transition, state, control, mean))[0]
    reference = np.asarray(_predict_prior(transition, state[None], control).mean)[0]
    np.testing.assert_allclose(ours, reference, rtol=1e-4, atol=1e-5)


def test_per_step_update_without_drift_equals_dyna_dist_gaussian_posterior(transition, trajectory):
    from dyna_dist.data.transforms import increments
    from dyna_dist.inference.online import GaussianEvidence
    from dyna_dist.inference.posterior import baseline_increments, components
    from dyna_dist.models.features import prior_moments_and_variance
    from sit_lmpc.world.evaluate import make_kalman

    states = jnp.asarray(trajectory["states"][:41])
    controls = jnp.asarray(trajectory["controls"][:40])
    ours = np.asarray(make_kalman(transition, walk_std=0.0)(states, controls))[-1]
    mean, factor, variance = prior_moments_and_variance(transition.parameters, transition.config)
    b, basis = components(transition, states[:-1], controls)
    observed = transition.normalizer.normalize_targets(
        increments(states[:-1], states[1:]) - baseline_increments(transition, states[:-1], controls))
    evidence = GaussianEvidence.create(np.asarray(mean), np.asarray(factor), np.asarray(variance))
    evidence.add(np.asarray(basis), np.asarray(observed - b))
    reference, _ = evidence.moments()
    scale = np.sqrt(np.diag(np.asarray(factor) @ np.asarray(factor).T))
    np.testing.assert_allclose(ours / scale, reference / scale, atol=2e-2)


def test_ekf_tracks_noise_free_measurements(trajectory):
    from sit_lmpc.world.ekf import LR, run_ekf

    states = trajectory["states"].astype(np.float64)
    rear = np.column_stack((states[:, 0] - LR * np.cos(states[:, 4]), states[:, 1] - LR * np.sin(states[:, 4]), states[:, 4]))
    wheel = states[:, 3] * np.cos(states[:, 6])
    estimate = run_ekf(rear, states[:, 5], wheel, trajectory["controls"])
    slip_error = np.sqrt(np.mean((estimate[10:, 6] - states[10:, 6]) ** 2))
    speed_error = np.sqrt(np.mean((estimate[10:, 3] - states[10:, 3]) ** 2))
    assert slip_error < 0.03 and speed_error < 0.1, (slip_error, speed_error)


def test_world_loss_terms_are_finite_and_kl_nonnegative(transition):
    import yaml
    from sit_lmpc.world.data import WorldData
    from sit_lmpc.world.model import world_loss
    from sit_lmpc.world.train import initialize

    config = yaml.safe_load((ROOT / "configs/train-world-encoder-v2.yaml").read_text())
    data = WorldData(DATASET, "validation")
    mean, std = data.observation_scale()
    encoder, decoder, params = initialize(config, transition, data, jax.random.key(0))
    batch = data.batch(np.random.default_rng(0), 4, config["window"])
    total, terms = world_loss(params, transition, encoder, decoder, batch, jax.random.key(1), config, mean, std)
    assert np.isfinite(float(total))
    for name, value in terms.items():
        assert np.isfinite(float(value)), name
    assert float(terms["kl"]) >= 0


def test_runtime_stepping_matches_the_whole_sequence_encoder(transition):
    from dyna_dist.models.features import prior_moments_and_variance
    from sit_lmpc.world.data import WorldData
    from sit_lmpc.world.evaluate import load_encoder
    from sit_lmpc.world.model import hidden_from_raw
    from sit_lmpc.world.runtime import EncoderBelief

    run = ROOT / "runs/world-encoder-v2-fixed"
    if not (run / "best.msgpack").exists():
        pytest.skip("needs a trained encoder run")
    item = WorldData(DATASET, "test").items[0]
    belief = EncoderBelief(run)
    stepped = [belief.step(observation) for observation in item["observations"]]
    encoder, params, mean, std, _ = load_encoder(run, 16, item["observations"].shape[-1])
    _, raw, w_mean, w_log_std, _ = encoder.apply(params["encoder"], ((item["observations"] - mean) / std)[None])
    prior_mean, prior_factor, _ = prior_moments_and_variance(transition.parameters, transition.config)
    hidden = np.asarray(hidden_from_raw(raw))[0]
    z_mean = np.asarray(prior_mean) + np.asarray(w_mean)[0] @ np.asarray(prior_factor).T
    np.testing.assert_allclose(np.stack([s[0] for s in stepped]), hidden, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.stack([s[3] for s in stepped]), np.asarray(w_mean)[0], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.stack([s[1] for s in stepped]), z_mean, rtol=1e-4, atol=1e-5)
    std_z = np.stack([np.sqrt(np.sum(s[2] ** 2, axis=1)) for s in stepped])
    reference_std = np.sqrt((np.asarray(prior_factor)[None] ** 2 * np.exp(2 * np.clip(np.asarray(w_log_std)[0], -6, 2))[:, None, :]).sum(-1))
    np.testing.assert_allclose(std_z, reference_std, rtol=1e-4, atol=1e-6)
