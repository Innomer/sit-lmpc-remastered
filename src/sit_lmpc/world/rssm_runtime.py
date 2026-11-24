"""The RSSM world model inside the planner loop: a per-step filter and a predictor the LMPPI planner rolls out.

RSSMFilter.step runs one filtering step (posterior mean, no sampling) from the 9 sensor features, a LiDAR scan and the
PF pose; RSSMPredictor has the planner's model interface (rollout_members, action_scale, timestep) and rolls the
current (h, s) forward:
  latent  full world-model rollout (h, s, x), prior mean for the fast part, condition part held      (variant A)
  cheap   bicycle + residual with h and s held fixed                                                (variant C)
"""

from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import yaml
from flax import serialization, struct

from sit_lmpc.world.rssm import (BODY_INDEX, RSSM, RSSMConfig, bicycle, cheap_rollout, clamp, ensemble_rollout,
                                 init_params, latent_rollout)
from sit_lmpc.world.rssm_data import downsample, log_range

LR = 0.17145
TRAINING_BEAMS = 1080
TRAINING_FOV = np.deg2rad(270.0)


def resample_scan(ranges, angle_min, angle_increment, beams=TRAINING_BEAMS, fov=TRAINING_FOV):
    """Any LiDAR scan resampled onto the training beam layout (1080 beams over 270 degrees, nearest beam)."""
    ranges = np.asarray(ranges, np.float32)
    angles = angle_min + np.arange(len(ranges)) * angle_increment
    target = np.linspace(-fov / 2, fov / 2, beams)
    index = np.clip(np.round((target - angle_min) / angle_increment).astype(int), 0, len(ranges) - 1)
    out = ranges[index]
    out[(target < angles[0]) | (target > angles[-1])] = np.inf
    return np.where(np.isfinite(out) & (out > 0.05), out, 30.0)


@struct.dataclass
class RSSMPredictor:
    network: dict
    st_params: jax.Array
    action_scale: jax.Array
    h: jax.Array
    s: jax.Array
    model: RSSM = struct.field(pytree_node=False)
    variant: str = struct.field(pytree_node=False, default="latent")
    timestep: float = struct.field(pytree_node=False, default=0.1)
    integrator_timestep: float = struct.field(pytree_node=False, default=0.02)
    clearance_margin: float = struct.field(pytree_node=False, default=0.35)
    clearance_weight: float = struct.field(pytree_node=False, default=0.0)
    grip: jax.Array = 1.0
    physics_weight: float = struct.field(pytree_node=False, default=0.0)
    ensemble_weight: float = struct.field(pytree_node=False, default=0.0)
    ensemble_slip_scale: float = struct.field(pytree_node=False, default=0.1)
    ensemble_position_scale: float = struct.field(pytree_node=False, default=0.3)

    @property
    def params(self):
        return self.st_params

    def rollout_members(self, state, actions):
        state = jnp.asarray(state, self.st_params.dtype)
        actions = jnp.asarray(actions, self.st_params.dtype)
        if self.variant == "cheap":
            xs = cheap_rollout(self.model, self.network, self.st_params, self.action_scale, self.timestep,
                               self.integrator_timestep, self.h, self.s, state, actions)
        else:
            xs = latent_rollout(self.model, self.network, self.st_params, self.action_scale, self.timestep,
                                self.integrator_timestep, self.h, self.s, state, actions, jax.random.key(0),
                                sample=False)[0]
        return xs[None]

    @jax.jit
    def clearance_cost(self, state, actions):
        """Per-step penalty (candidates, horizon) from the LiDAR the world model predicts along each candidate:
        weight * max(margin - nearest predicted range, 0) / margin."""
        state = jnp.asarray(state, self.st_params.dtype)

        def one(sequence):
            _, hs, ss = latent_rollout(self.model, self.network, self.st_params, self.action_scale, self.timestep,
                                       self.integrator_timestep, self.h, self.s, state,
                                       jnp.asarray(sequence, self.st_params.dtype), jax.random.key(0), sample=False)
            mean, _ = self.model.apply(self.network, hs, ss, method=RSSM.decode_lidar)
            return jnp.exp(mean).min(axis=-1)

        nearest = jax.vmap(one)(actions)
        return self.clearance_weight * jnp.clip((self.clearance_margin - nearest) / self.clearance_margin, 0.0, 1.0)

    @jax.jit
    def physics_rollout(self, state, actions):
        """Bicycle-model rollouts (candidates, horizon, 7) of the same candidates from the same state, with the
        friction set to the world model's current grip estimate: the physics safety check."""
        params = self.st_params.at[0].set(jnp.clip(jnp.asarray(self.grip, self.st_params.dtype), 0.3, 1.5))
        state = jnp.asarray(state, self.st_params.dtype)

        def one(sequence):
            def advance(x, a):
                y = bicycle(params, x, a * self.action_scale, self.timestep, self.integrator_timestep)
                return y, y
            return jax.lax.scan(advance, state, jnp.asarray(sequence, self.st_params.dtype))[1]

        return jax.vmap(one)(actions)

    @jax.jit
    def ensemble_cost(self, state, actions):
        """Per-step penalty (candidates, horizon) from the ensemble heads' disagreement along each candidate: the
        heads roll out along the main rollout's memory/latent path; spread in slip (std) and position (rms distance
        from the heads' mean) above its scale starts the penalty, twice the scale gives the full weight."""
        state = jnp.asarray(state, self.st_params.dtype)

        def one(sequence):
            sequence = jnp.asarray(sequence, self.st_params.dtype)
            _, hs, ss = latent_rollout(self.model, self.network, self.st_params, self.action_scale, self.timestep,
                                       self.integrator_timestep, self.h, self.s, state, sequence, jax.random.key(0),
                                       sample=False)
            h_in = jnp.concatenate([self.h[None], hs[:-1]])
            s_in = jnp.concatenate([self.s[None], ss[:-1]])
            xs = ensemble_rollout(self.model, self.network, self.st_params, self.action_scale, self.timestep,
                                  self.integrator_timestep, h_in, s_in, state, sequence)
            slip = xs[..., 6].std(axis=-1)
            centre = xs[..., :2].mean(axis=-2, keepdims=True)
            position = jnp.sqrt(jnp.sum((xs[..., :2] - centre) ** 2, -1).mean(-1))
            return jnp.maximum(jnp.clip(slip / self.ensemble_slip_scale - 1.0, 0.0, 1.0),
                               jnp.clip(position / self.ensemble_position_scale - 1.0, 0.0, 1.0))

        return self.ensemble_weight * jax.vmap(one)(actions)

    def step_members(self, state, action):
        state = jnp.asarray(state, self.st_params.dtype)
        action = jnp.asarray(action, self.st_params.dtype)
        u = action * self.action_scale
        x = bicycle(self.st_params, state, u, self.timestep, self.integrator_timestep)
        x = clamp(x + self.model.apply(self.network, state, u, self.h, self.s, method=RSSM.residual))
        return x[None]

    def context(self):
        return {"kind": "rssm", "variant": self.variant, "model_timestep": self.timestep,
                "hidden": self.model.config.hidden, "latent": self.model.config.latent}


class RSSMFilter:
    def __init__(self, run, st_params, action_scale, variant="latent", model_timestep=0.1, integrator=0.02,
                 clearance_margin=0.35, clearance_weight=0.0, physics_weight=0.0, ensemble_weight=0.0,
                 ensemble_slip_scale=0.1, ensemble_position_scale=0.3):
        run = Path(run)
        config = yaml.safe_load((run / "config.yaml").read_text())
        self.config = RSSMConfig(**config["model"])
        self.model, template = init_params(self.config, jax.random.key(0))
        self.network = serialization.from_bytes(template, (run / "best.msgpack").read_bytes())
        scale = np.load(run / "observation_scale.npz")
        self.obs_mean = jnp.asarray(scale["mean"], jnp.float32)
        self.obs_std = jnp.asarray(scale["std"], jnp.float32)
        self.st_params = jnp.asarray(st_params, jnp.float32)
        self.action_scale = jnp.asarray(action_scale, jnp.float32)
        self.dt = float(config["timestep"])
        self.integrator = float(config["integrator_timestep"])
        self.variant = variant
        self.model_timestep = model_timestep
        self.planner_integrator = integrator
        self.clearance_margin = clearance_margin
        self.clearance_weight = clearance_weight if variant == "latent" else 0.0
        self.physics_weight = physics_weight
        if ensemble_weight > 0 and not self.config.heads:
            raise ValueError(f"rssm_ensemble_weight > 0 needs an RSSM with ensemble heads; {run} has none")
        self.ensemble = (ensemble_weight, ensemble_slip_scale, ensemble_position_scale)
        self.grip = 1.0
        model, network, st = self.model, self.network, self.st_params

        def embed(lidar, sensors):
            return model.apply(network, lidar, (sensors - self.obs_mean) / self.obs_std, method=RSSM.embed)

        def merge(pose, body_state):
            return jnp.stack([pose[0], pose[1], body_state[0], body_state[1], pose[2], body_state[2], body_state[3]])

        @jax.jit
        def first(lidar, sensors, pose):
            h = jnp.zeros(self.config.hidden, jnp.float32)
            s, _ = model.apply(network, h, embed(lidar, sensors), method=RSSM.posterior)
            anchor = model.apply(network, h, s, method=RSSM.anchor)
            return h, s, merge(pose, anchor), anchor, model.apply(network, s, method=RSSM.condition_readout)

        @jax.jit
        def step(h, s, x, u, lidar, sensors, pose):
            x_pred = clamp(bicycle(st, x, u, self.dt, self.integrator) + model.apply(network, x, u, h, s,
                                                                                      method=RSSM.residual))
            h_next = model.apply(network, h, s, u, x_pred, method=RSSM.advance)
            s_next, _ = model.apply(network, h_next, embed(lidar, sensors), method=RSSM.posterior)
            anchor = model.apply(network, h_next, s_next, method=RSSM.anchor)
            return h_next, s_next, merge(pose, anchor), anchor, model.apply(network, s_next, method=RSSM.condition_readout)

        self._first, self._step = first, step
        self.h = jnp.zeros(self.config.hidden, jnp.float32)
        self.s = jnp.zeros(self.config.latent, jnp.float32)
        self.x = jnp.zeros(7, jnp.float32)
        self.reset()

    def reset(self):
        self.started = False

    def lidar_features(self, scan):
        return jnp.asarray(log_range(downsample(np.asarray(scan, np.float32)[None], self.config.beams))[0])

    def step_features(self, sensors, lidar, pose_rear, control):
        """As step, with the LiDAR already as the model's 64 log-range features (e.g. rssm_lidar.npy)."""
        yaw = float(pose_rear[2])
        pose = jnp.asarray([pose_rear[0] + LR * np.cos(yaw), pose_rear[1] + LR * np.sin(yaw), yaw], jnp.float32)
        lidar = jnp.asarray(lidar, jnp.float32)
        sensors = jnp.asarray(sensors, jnp.float32)
        if not self.started:
            self.h, self.s, self.x, anchor, condition = self._first(lidar, sensors, pose)
            self.started = True
        else:
            self.h, self.s, self.x, anchor, condition = self._step(self.h, self.s, self.x,
                                                                   jnp.asarray(control, jnp.float32), lidar, sensors,
                                                                   pose)
        return np.asarray(anchor)

    def step(self, sensors, scan, pose_rear, control):
        """sensors: the 9 encoder features (raw); scan: LiDAR ranges on the training layout (1080 beams, 270 deg);
        pose_rear: PF pose (x, y, yaw) at the rear axle; control: the control applied since the last call."""
        yaw = float(pose_rear[2])
        pose = jnp.asarray([pose_rear[0] + LR * np.cos(yaw), pose_rear[1] + LR * np.sin(yaw), yaw], jnp.float32)
        lidar = self.lidar_features(scan)
        sensors = jnp.asarray(sensors, jnp.float32)
        if not self.started:
            self.h, self.s, self.x, anchor, condition = self._first(lidar, sensors, pose)
            self.started = True
        else:
            self.h, self.s, self.x, anchor, condition = self._step(self.h, self.s, self.x,
                                                                   jnp.asarray(control, jnp.float32), lidar, sensors,
                                                                   pose)
        from sit_lmpc.world.rssm_train import CONDITION_MEAN, CONDITION_SCALE
        grip = float(np.asarray(condition)[0] * CONDITION_SCALE[0] + CONDITION_MEAN[0])
        self.grip = grip
        return np.asarray(anchor), np.asarray(self.x), grip

    def predictor(self):
        return RSSMPredictor(self.network, self.st_params, self.action_scale, self.h, self.s, self.model,
                             self.variant, self.model_timestep, self.planner_integrator, self.clearance_margin,
                             self.clearance_weight, jnp.float32(self.grip), self.physics_weight, *self.ensemble)
