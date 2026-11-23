"""Recurrent state-space world model (RSSM) with a physics-anchored state.

Per step: memory h (GRU), stochastic latent s = (c slow condition, d fast), physical state x (7).
  x' = bicycle(x, u) + residual(x, u, h, s)
  h' = GRU(h, [s, u, x'])
  s' ~ prior(h')            (planning)   or   posterior(h', obs)   (filtering)
Heads: anchor A(h, s) -> x, decoders for LiDAR / IMU / wheel speed, auxiliary condition head c -> (friction, mass).
"""

from dataclasses import dataclass
import jax
import jax.numpy as jnp
import flax.linen as nn

from sit_lmpc.dynamics.st import rk4


@dataclass(frozen=True)
class RSSMConfig:
    hidden: int = 128
    condition: int = 8
    fast: int = 24
    beams: int = 64
    head: int = 128
    decoder: int = 256

    @property
    def latent(self):
        return self.condition + self.fast


def mlp(features, out, name):
    return nn.Sequential([nn.Dense(features), nn.elu, nn.Dense(features), nn.elu, nn.Dense(out)], name=name)


class RSSM(nn.Module):
    config: RSSMConfig

    def setup(self):
        c = self.config
        self.cell = nn.GRUCell(features=c.hidden, name="cell")
        self.prior_head = mlp(c.head, 2 * c.latent, "prior")
        self.posterior_head = mlp(c.head, 2 * c.latent, "posterior")
        self.residual_head = mlp(c.head, 7, "residual")
        self.anchor_head = mlp(c.head, 7, "anchor")
        self.lidar_encoder = nn.Sequential([
            nn.Conv(16, (5,), strides=(2,)), nn.elu, nn.Conv(32, (5,), strides=(2,)), nn.elu,
        ], name="lidar_encoder")
        self.obs_encoder = mlp(c.head, c.head, "obs_encoder")
        self.lidar_decoder = mlp(c.decoder, 2 * c.beams, "lidar_decoder")
        self.sensor_decoder = mlp(c.head, 8, "sensor_decoder")
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
        return self.residual_head(jnp.concatenate([x[..., 2:], u, h, s], -1))

    def advance(self, h, s, u, x_next):
        h_next, _ = self.cell(h, jnp.concatenate([s, u, x_next], -1))
        return h_next

    def anchor(self, h, s):
        return self.anchor_head(jnp.concatenate([h, s], -1))

    def decode_lidar(self, h, s):
        return self.gaussian(self.lidar_decoder(jnp.concatenate([h, s], -1)))

    def decode_sensors(self, h, s, u):
        return self.sensor_decoder(jnp.concatenate([h, s, u], -1))

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
    batch = 1
    zeros = lambda n: jnp.zeros((batch, n))
    params = model.init(key, zeros(config.hidden), zeros(config.latent), zeros(2), zeros(7), zeros(config.beams), zeros(5))
    return model, params


def bicycle(st_params, x, u, dt, integrator):
    count = max(1, round(dt / integrator))
    return jax.lax.fori_loop(0, count, lambda _, y: rk4(y, u, st_params, dt / count), x)


def latent_rollout(model, params, st_params, action_scale, dt, integrator, h, s, x, actions, key):
    """Full world-model rollout of one action sequence: (h, s, x) with s sampled from the prior each step."""
    def step(carry, inputs):
        h, s, x = carry
        a, k = inputs
        u = a * action_scale
        x_next = bicycle(st_params, x, u, dt, integrator)
        x_next = x_next + model.apply(params, x, u, h, s, method=RSSM.residual)
        h_next = model.apply(params, h, s, u, x_next, method=RSSM.advance)
        mean, log_std = model.apply(params, h_next, method=RSSM.prior)
        s_next = mean + jnp.exp(log_std) * jax.random.normal(k, mean.shape)
        return (h_next, s_next, x_next), (x_next, h_next, s_next)
    keys = jax.random.split(key, len(actions))
    _, (xs, hs, ss) = jax.lax.scan(step, (h, s, x), (actions, keys))
    return xs, hs, ss


def cheap_rollout(model, params, st_params, action_scale, dt, integrator, h, s, x, actions):
    """Planning model without the latent dynamics: bicycle + residual with h and s held fixed (c carries the condition)."""
    def step(x, a):
        u = a * action_scale
        x_next = bicycle(st_params, x, u, dt, integrator) + model.apply(params, x, u, h, s, method=RSSM.residual)
        return x_next, x_next
    return jax.lax.scan(step, x, actions)[1]
