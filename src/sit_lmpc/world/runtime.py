"""Run the trained world-model encoder one control step at a time inside the planner loop."""

from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import yaml
from flax import serialization

from dyna_dist.io.checkpoint import load_model
from dyna_dist.models.features import prior_moments_and_variance
from sit_lmpc.world.model import Decoder, Encoder, hidden_from_raw


class EncoderBelief:
    def __init__(self, run):
        run = Path(run)
        config = yaml.safe_load((run / "config.yaml").read_text())
        transition = load_model(config["transition"])
        mean, factor, _ = prior_moments_and_variance(transition.parameters, transition.config)
        self.prior_mean = np.asarray(mean, np.float32)
        self.prior_factor = np.asarray(factor, np.float32)
        latent = transition.config.latent_dimension
        scale = np.load(run / "observation_scale.npz")
        self.observation_mean = scale["mean"].astype(np.float32)
        self.observation_std = scale["std"].astype(np.float32)
        self.encoder = Encoder(config["hidden"], latent)
        decoder = Decoder(config["hidden"])
        observation_dim = len(self.observation_mean)
        template = {
            "encoder": self.encoder.init(jax.random.key(0), jnp.zeros((1, 2, observation_dim))),
            "decoder": decoder.init(jax.random.key(1), jnp.zeros((1, 4)), jnp.zeros((1, latent)), jnp.zeros((1, 2)),
                                    jnp.zeros((1, 4))),
            "sensor_log_std": jnp.zeros(4),
        }
        params = serialization.from_bytes(template, (run / "best.msgpack").read_bytes())["encoder"]
        self.hidden_size = config["hidden"]

        @jax.jit
        def step(carry, observation):
            carry, raw, w_mean, w_log_std, _ = self.encoder.apply(params, observation[None, None], carry)
            return carry, hidden_from_raw(raw)[0, 0], w_mean[0, 0], jnp.clip(w_log_std[0, 0], -6.0, 2.0)

        self._step = step
        readout = run / "condition_readout.npz"
        self.readout = dict(np.load(readout)) if readout.exists() else None
        grip = run / "grip_readout.npz"
        self.grip_readout = dict(np.load(grip)) if grip.exists() else None
        self.last_w_std = None
        self.reset()

    def friction(self, w_mean):
        """Estimated friction from the whitened z (sit_lmpc.world.readout), or None without a fitted read-out."""
        if self.readout is None:
            return None
        return float(np.asarray(w_mean) @ self.readout["weights"] + self.readout["bias"])

    def grip(self, w_mean, w_std):
        """Per-step grip estimate from the read-out refit on per-step w (scripts/grip_estimate_eval.py) and its
        variance from the encoder's own w std, or None without grip_readout.npz."""
        if self.grip_readout is None:
            return None
        weights = self.grip_readout["weights"]
        value = float(np.asarray(w_mean) @ weights + self.grip_readout["bias"])
        variance = float((np.asarray(w_std) ** 2) @ (weights ** 2))
        return value, variance

    def reset(self):
        self.carry = jnp.zeros((1, self.hidden_size), jnp.float32)

    def step(self, observation):
        """observation: the 9 encoder inputs. Returns (steering, speed, yaw rate, slip), z mean, z covariance factor and
        the whitened mean w (z = prior mean + prior factor @ w)."""
        normalized = (np.asarray(observation, np.float32) - self.observation_mean) / self.observation_std
        self.carry, hidden, w_mean, w_log_std = self._step(self.carry, jnp.asarray(normalized))
        w_mean, w_std = np.asarray(w_mean), np.exp(np.asarray(w_log_std))
        self.last_w_std = w_std
        z_mean = self.prior_mean + self.prior_factor @ w_mean
        z_factor = self.prior_factor * w_std[None, :]
        return np.asarray(hidden), z_mean, z_factor, w_mean
