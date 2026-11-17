"""Physics-grounded world model: GRU encoder (state + z belief), frozen bicycle+residual transition, sensor decoder."""

import jax
import jax.numpy as jnp
import flax.linen as nn

import tensorflow_probability.substrates.jax as tfp

from dyna_dist.inference.posterior import _function_step
from dyna_dist.models.features import prior_moments_and_variance

tfd = tfp.distributions

LR = 0.17145
STATE_SCALE = jnp.array([0.1, 0.1, 0.02, 0.2, 0.02, 0.2, 0.02])
HIDDEN_SCALE = jnp.array([0.02, 0.2, 0.2, 0.02])
HIDDEN_INDEX = jnp.array([2, 3, 5, 6])


class Encoder(nn.Module):
    hidden: int
    latent: int

    @nn.compact
    def __call__(self, observations, carry=None):
        observations = jnp.asarray(observations, jnp.float32)
        cell = nn.GRUCell(features=self.hidden)
        if carry is None:
            carry = cell.initialize_carry(jax.random.key(0), observations[..., 0, :].shape)
        rnn = nn.RNN(cell, return_carry=True)
        carry, features = rnn(observations, initial_carry=carry)
        h = nn.relu(nn.Dense(self.hidden)(features))
        hidden_state = nn.Dense(4)(h)
        z_mean = nn.Dense(self.latent)(h)
        z_log_std = nn.Dense(self.latent)(h)
        calibration = nn.Dense(4)(h)
        return carry, hidden_state, z_mean, z_log_std, calibration


class Decoder(nn.Module):
    hidden: int

    @nn.compact
    def __call__(self, hidden_state, z, control, calibration):
        h = jnp.concatenate((hidden_state, z, control), axis=-1)
        h = nn.relu(nn.Dense(self.hidden)(h))
        h = nn.relu(nn.Dense(self.hidden)(h))
        sensors = nn.Dense(4)(h)
        sensors = sensors.at[..., 0].add(calibration[..., 0])
        sensors = sensors.at[..., 1].add(calibration[..., 1])
        sensors = sensors.at[..., 2].add(calibration[..., 2])
        sensors = sensors.at[..., 3].multiply(1.0 + 0.1 * jnp.tanh(calibration[..., 3]))
        return sensors


def full_state(pose, hidden):
    return jnp.concatenate(
        (pose[..., 0:1], pose[..., 1:2], hidden[..., 0:1], hidden[..., 1:2], pose[..., 2:3], hidden[..., 2:3], hidden[..., 3:4]),
        axis=-1,
    )


def rollout(transition, state, controls, z):
    def advance(x, u):
        y = _function_step(transition, x, u, z)
        return y, y

    return jax.lax.scan(advance, state, controls)[1]


def wrap(angle):
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi


def state_error(predicted, true):
    error = predicted - true
    error = error.at[..., 4].set(wrap(error[..., 4])).at[..., 6].set(wrap(error[..., 6]))
    return error / STATE_SCALE


def kl_diag(mean_q, log_std_q, mean_p, var_p):
    """KL between diagonal Gaussians, from TensorFlow Probability."""
    q = tfd.MultivariateNormalDiag(mean_q, jnp.exp(log_std_q))
    p = tfd.MultivariateNormalDiag(jnp.broadcast_to(mean_p, mean_q.shape),
                                   jnp.broadcast_to(jnp.sqrt(var_p), mean_q.shape))
    return tfd.kl_divergence(q, p)


def hidden_from_raw(raw):
    return raw * HIDDEN_SCALE * 10.0


def world_loss(params, transition, encoder, decoder, batch, key, config, observation_mean, observation_std, kl_scale=1.0):
    batch = {name: jnp.asarray(value, jnp.float32) for name, value in batch.items()}
    observations = (batch["observations"] - observation_mean) / observation_std
    _, raw_hidden, w_mean, w_log_std, calibration = encoder.apply(params["encoder"], observations)
    w_log_std = jnp.clip(w_log_std, -6.0, 2.0)
    hidden = hidden_from_raw(raw_hidden)
    states = batch["states"]
    true_hidden = states[..., HIDDEN_INDEX]
    hidden_error = (hidden - true_hidden).at[..., 3].set(wrap(hidden[..., 3] - true_hidden[..., 3])) / HIDDEN_SCALE
    burn = config["burn"]
    anchor = jnp.mean(hidden_error[:, burn:] ** 2)

    prior_mean, prior_factor, _ = prior_moments_and_variance(transition.parameters, transition.config)
    walk_var = config["walk_std"] ** 2
    kl_start = kl_diag(w_mean[:, 0], w_log_std[:, 0], 0.0, 1.0)
    kl_walk = kl_diag(w_mean[:, 1:], w_log_std[:, 1:], w_mean[:, :-1], jnp.exp(2 * w_log_std[:, :-1]) + walk_var)
    kl = jnp.mean(kl_start) / states.shape[1] + jnp.mean(kl_walk)
    to_z = lambda w: prior_mean + jnp.einsum("ij,...j->...i", prior_factor, w)

    controls = batch["controls"]
    pose = batch["pose"]
    horizon = config["horizon"]
    start_steps = jnp.arange(burn, controls.shape[1] - horizon + 1, config["start_stride"])
    noise = jax.random.normal(key, (states.shape[0], len(start_steps), w_mean.shape[-1]))

    def predict(window, start, eps, from_truth):
        estimated = full_state(pose[window, start], hidden[window, start])
        start_state = jnp.where(from_truth, states[window, start], estimated)
        z = to_z(w_mean[window, start] + jnp.exp(w_log_std[window, start]) * eps)
        segment = jax.lax.dynamic_slice_in_dim(controls[window], start, horizon)
        predicted = rollout(transition, start_state, segment, z)
        target = jax.lax.dynamic_slice_in_dim(states[window], start + 1, horizon)
        return jnp.mean(state_error(predicted, target) ** 2)

    windows = jnp.arange(states.shape[0])
    prediction = jnp.mean(jax.vmap(lambda w: jax.vmap(lambda s, e: predict(w, s, e, False))(start_steps, noise[w]))(windows))
    truth_prediction = jnp.mean(jax.vmap(lambda w: jax.vmap(lambda s, e: predict(w, s, e, True))(start_steps, noise[w]))(windows))

    z_sample = w_mean + jnp.exp(w_log_std) * jax.random.normal(jax.random.fold_in(key, 1), w_mean.shape)
    applied = jnp.concatenate((controls, controls[:, -1:]), axis=1)
    sensors = decoder.apply(params["decoder"], hidden / HIDDEN_SCALE / 10.0, z_sample, applied, calibration)
    sensor_true = batch["observations"][..., :4]
    log_std = params["sensor_log_std"]
    sensor_nll = jnp.mean(0.5 * ((sensors - sensor_true) / jnp.exp(log_std)) ** 2 + log_std)

    total = (config["anchor_weight"] * anchor + config["prediction_weight"] * prediction
             + config.get("true_state_prediction_weight", 0.0) * truth_prediction
             + config["observation_weight"] * sensor_nll + config["kl_weight"] * kl_scale * kl)
    return total, {"anchor": anchor, "prediction": prediction, "true_state_prediction": truth_prediction,
                   "observation": sensor_nll, "kl": kl}
