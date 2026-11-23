"""Recurrent state-space world model (RSSM) with a physics-anchored state.

Per step: memory h (GRU), stochastic latent s = (c slow condition, d fast), physical state x (7, CoG frame).
  x' = bicycle(x, u) + residual(x, u, h, s)
  h' = GRU(h, [s, u, x'])
  s' ~ prior(h')            (planning)   or   posterior(h', obs)   (filtering)
Heads: anchor A(h, s) -> (steering, speed, yaw rate, slip), decoders for LiDAR / IMU-wheel, condition head c ->
(friction, mass ratio). The networks see only body-frame quantities (steering, speed, yaw rate, slip), never map
position or absolute heading.
"""

from dataclasses import dataclass
import jax
import jax.numpy as jnp
import flax.linen as nn

from sit_lmpc.dynamics.st import rk4

BODY_INDEX = jnp.array([2, 3, 5, 6])
BODY_SCALE = jnp.array([0.4, 10.0, 2.0, 0.3], jnp.float32)
CONTROL_SCALE = jnp.array([3.2, 9.51], jnp.float32)
DECODER_MIN_LOG_STD = -3.0
BODY_LOW = jnp.array([-0.42, 0.0, -10.0, -1.2], jnp.float32)
BODY_HIGH = jnp.array([0.42, 20.0, 10.0, 1.2], jnp.float32)


@dataclass(frozen=True)
class RSSMConfig:
    hidden: int = 128
    condition: int = 8
    fast: int = 24
    beams: int = 64
    sensors: int = 9
    decoded_sensors: int = 4
    head: int = 128
    decoder: int = 256

    @property
    def latent(self):
        return self.condition + self.fast


def mlp(features, out, name):
    return nn.Sequential([nn.Dense(features), nn.elu, nn.Dense(features), nn.elu, nn.Dense(out)], name=name)


def body(x):
    return x[..., BODY_INDEX] / BODY_SCALE


def clamp(x):
    return x.at[..., BODY_INDEX].set(jnp.clip(x[..., BODY_INDEX], BODY_LOW, BODY_HIGH))


class RSSM(nn.Module):
    config: RSSMConfig

    def setup(self):
        c = self.config
        self.cell = nn.GRUCell(features=c.hidden, name="cell")
        self.prior_head = mlp(c.head, 2 * c.fast, "prior")
        self.posterior_head = mlp(c.head, 2 * c.latent, "posterior")
        self.residual_head = mlp(c.head, 7, "residual")
        self.anchor_head = mlp(c.head, 4, "anchor")
        self.lidar_encoder = nn.Sequential([
            nn.Conv(16, (5,), strides=(2,)), nn.elu, nn.Conv(32, (5,), strides=(2,)), nn.elu,
        ], name="lidar_encoder")
        self.obs_encoder = mlp(c.head, c.head, "obs_encoder")
        self.lidar_decoder = mlp(c.decoder, 2 * c.beams, "lidar_decoder")
        self.sensor_decoder = mlp(c.head, 2 * c.decoded_sensors, "sensor_decoder")
        self.condition_head = mlp(64, 2, "condition_head")

    def gaussian(self, out):
        mean, log_std = jnp.split(out, 2, axis=-1)
        return mean, jnp.clip(log_std, -6.0, 2.0)

    def prior(self, h):
        return self.gaussian(self.prior_head(h))

    def posterior(self, h, embedding):
        return self.gaussian(self.posterior_head(jnp.concatenate([h, embedding], -1)))

    def embed(self, lidar, sensors):
        features = self.lidar_encoder(lidar[..., None]).reshape(*lidar.shape[:-1], -1)
        return self.obs_encoder(jnp.concatenate([features, sensors], -1))

    def residual(self, x, u, h, s):
        return self.residual_head(jnp.concatenate([body(x), u / CONTROL_SCALE, h, s], -1)) * 0.1

    def advance(self, h, s, u, x_next):
        h_next, _ = self.cell(h, jnp.concatenate([s, u / CONTROL_SCALE, body(x_next)], -1))
        return h_next

    def anchor(self, h, s):
        raw = self.anchor_head(jnp.concatenate([h, s], -1))
        return BODY_LOW + (BODY_HIGH - BODY_LOW) * jax.nn.sigmoid(raw)

    def decode_lidar(self, h, s):
        mean, log_std = self.gaussian(self.lidar_decoder(jnp.concatenate([h, s], -1)))
        return mean, jnp.maximum(log_std, DECODER_MIN_LOG_STD)

    def decode_sensors(self, h, s, u):
        mean, log_std = self.gaussian(self.sensor_decoder(jnp.concatenate([h, s, u / CONTROL_SCALE], -1)))
        return mean, jnp.maximum(log_std, DECODER_MIN_LOG_STD)

    def condition_readout(self, s):
        return self.condition_head(s[..., :self.config.condition])

    def __call__(self, h, s, u, x, lidar, sensors):
        x_next = x + self.residual(x, u, h, s)
        h_next = self.advance(h, s, u, x_next)
        mean, _ = self.posterior(h_next, self.embed(lidar, sensors))
        self.prior(h_next)
        self.anchor(h_next, mean)
        self.decode_lidar(h_next, mean)
        self.decode_sensors(h_next, mean, u)
        self.condition_readout(mean)
        return h_next, mean


def init_params(config, key):
    model = RSSM(config)
    zeros = lambda n: jnp.zeros((1, n))
    params = model.init(key, zeros(config.hidden), zeros(config.latent), zeros(2), zeros(7), zeros(config.beams),
                        zeros(config.sensors))
    return model, params


def bicycle(st_params, x, u, dt, integrator):
    count = max(1, round(dt / integrator))
    x, u = x.astype(st_params.dtype), u.astype(st_params.dtype)
    return jax.lax.fori_loop(0, count, lambda _, y: rk4(y, u, st_params, dt / count), x)


def prior_sample(model, params, h, s, key, sample=True):
    """Next latent from the prior: the condition part c is carried over (random walk, mean), d comes from p(d | h)."""
    mean, log_std = model.apply(params, h, method=RSSM.prior)
    d = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape) if sample else mean
    return jnp.concatenate([s[..., :model.config.condition], d], -1)


def latent_rollout(model, params, st_params, action_scale, dt, integrator, h, s, x, actions, key, sample=True):
    """Full world-model rollout of one action sequence: (h, s, x) with d sampled from the prior each step."""
    def step(carry, inputs):
        h, s, x = carry
        a, k = inputs
        u = a * action_scale
        x_next = clamp(bicycle(st_params, x, u, dt, integrator) + model.apply(params, x, u, h, s, method=RSSM.residual))
        h_next = model.apply(params, h, s, u, x_next, method=RSSM.advance)
        s_next = prior_sample(model, params, h_next, s, k, sample)
        return (h_next, s_next, x_next), (x_next, h_next, s_next)
    keys = jax.random.split(key, len(actions))
    _, (xs, hs, ss) = jax.lax.scan(step, (h, s, x), (actions, keys))
    return xs, hs, ss


def cheap_rollout(model, params, st_params, action_scale, dt, integrator, h, s, x, actions):
    """Planning model without the latent dynamics: bicycle + residual with h and s held fixed (c carries the condition)."""
    def step(x, a):
        u = a * action_scale
        x_next = clamp(bicycle(st_params, x, u, dt, integrator) + model.apply(params, x, u, h, s, method=RSSM.residual))
        return x_next, x_next
    return jax.lax.scan(step, x, actions)[1]
