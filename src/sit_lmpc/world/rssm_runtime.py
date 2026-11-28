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

from sit_lmpc.world.rssm import BODY_INDEX, RSSM, RSSMConfig, bicycle, cheap_rollout, clamp, init_params, latent_rollout
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
    risk_params: dict | None = None
    risk_mean: jax.Array | None = None
    risk_std: jax.Array | None = None
    risk_network: object = struct.field(pytree_node=False, default=None)
    risk_weight: float = struct.field(pytree_node=False, default=0.0)

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
    def latent_costs(self, state, actions):
        """Per-step costs (candidates, horizon) from one latent rollout of every candidate: LiDAR clearance (as
        clearance_cost) and the crash-risk head's probability of a crash within 1.5 s at each predicted (h, s),
        times its weight."""
        state = jnp.asarray(state, self.st_params.dtype)

        def one(sequence):
            _, hs, ss = latent_rollout(self.model, self.network, self.st_params, self.action_scale, self.timestep,
                                       self.integrator_timestep, self.h, self.s, state,
                                       jnp.asarray(sequence, self.st_params.dtype), jax.random.key(0), sample=False)
            out = {}
            if self.clearance_weight > 0:
                mean, _ = self.model.apply(self.network, hs, ss, method=RSSM.decode_lidar)
                nearest = jnp.exp(mean).min(axis=-1)
                out["clearance"] = self.clearance_weight * jnp.clip(
                    (self.clearance_margin - nearest) / self.clearance_margin, 0.0, 1.0)
            if self.risk_weight > 0:
                features = (jnp.concatenate([hs, ss], -1) - self.risk_mean) / self.risk_std
                out["risk"] = self.risk_weight * jax.nn.sigmoid(self.risk_network.apply(self.risk_params, features))
            return out

        return jax.vmap(one)(actions)

    @jax.jit
    def rollout_with_costs(self, state, actions):
        state = jnp.asarray(state, self.st_params.dtype)

        def one(sequence):
            xs, hs, ss = latent_rollout(self.model, self.network, self.st_params, self.action_scale, self.timestep,
                                        self.integrator_timestep, self.h, self.s, state,
                                        jnp.asarray(sequence, self.st_params.dtype), jax.random.key(0), sample=False)
            out = {}
            if self.clearance_weight > 0:
                mean, _ = self.model.apply(self.network, hs, ss, method=RSSM.decode_lidar)
                nearest = jnp.exp(mean).min(axis=-1)
                out["clearance"] = self.clearance_weight * jnp.clip(
                    (self.clearance_margin - nearest) / self.clearance_margin, 0.0, 1.0)
            if self.risk_weight > 0:
                features = (jnp.concatenate([hs, ss], -1) - self.risk_mean) / self.risk_std
                out["risk"] = self.risk_weight * jax.nn.sigmoid(self.risk_network.apply(self.risk_params, features))
            return xs, out

        xs, costs = jax.vmap(one)(actions)
        return xs, xs[:, None], costs

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
                 clearance_margin=0.35, clearance_weight=0.0, risk=None, risk_weight=0.0, device="default"):
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
        self.risk = {"risk_weight": 0.0}
        if risk is not None and risk_weight > 0:
            from sit_lmpc.training.risk import RiskNetwork
            risk_config = yaml.safe_load((Path(risk) / "config.yaml").read_text())
            normalization = np.load(Path(risk) / "normalization.npz")
            network = RiskNetwork(risk_config["width"], risk_config["layers"])
            template = network.init(jax.random.key(0), jnp.zeros((1, self.config.hidden + self.config.latent)))
            self.risk = {"risk_params": serialization.from_bytes(template, (Path(risk) / "risk_head.msgpack").read_bytes()),
                         "risk_mean": jnp.asarray(normalization["mean"], jnp.float32),
                         "risk_std": jnp.asarray(normalization["std"], jnp.float32),
                         "risk_network": network, "risk_weight": float(risk_weight)}
        self.filter_device = jax.devices("cpu")[0] if device == "cpu" else None
        model, network, st = self.model, self.network, self.st_params
        obs_mean, obs_std = self.obs_mean, self.obs_std
        if self.filter_device is not None:
            network, st, obs_mean, obs_std = jax.device_put((network, st, obs_mean, obs_std), self.filter_device)

        def embed(lidar, sensors):
            return model.apply(network, lidar, (sensors - obs_mean) / obs_std, method=RSSM.embed)

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
        self._state = jax.device_put((self.h, self.s, self.x), self.filter_device)
        self.reset()

    def reset(self):
        self.started = False

    def lidar_features(self, scan):
        return jnp.asarray(self.lidar_features_host(scan))

    def lidar_features_host(self, scan):
        return log_range(downsample(np.asarray(scan, np.float32)[None], self.config.beams))[0]

    def _keep(self, h, s, x):
        self._state = (h, s, x)
        if self.filter_device is None:
            self.h, self.s, self.x = h, s, x
        else:
            self.h, self.s, self.x = jax.device_put((h, s, x), jax.devices()[0])

    def step_features(self, sensors, lidar, pose_rear, control):
        """As step, with the LiDAR already as the model's 64 log-range features (e.g. rssm_lidar.npy)."""
        yaw = float(pose_rear[2])
        pose, lidar, sensors, control = jax.device_put((
            np.asarray([pose_rear[0] + LR * np.cos(yaw), pose_rear[1] + LR * np.sin(yaw), yaw], np.float32),
            np.asarray(lidar, np.float32), np.asarray(sensors, np.float32), np.asarray(control, np.float32)),
            self.filter_device)
        if not self.started:
            h, s, x, anchor, condition = self._first(lidar, sensors, pose)
            self.started = True
        else:
            h, s, x, anchor, condition = self._step(*self._state, control, lidar, sensors, pose)
        self._keep(h, s, x)
        return np.asarray(anchor)

    def step(self, sensors, scan, pose_rear, control):
        """sensors: the 9 encoder features (raw); scan: LiDAR ranges on the training layout (1080 beams, 270 deg);
        pose_rear: PF pose (x, y, yaw) at the rear axle; control: the control applied since the last call."""
        yaw = float(pose_rear[2])
        pose, lidar, sensors, control = jax.device_put((
            np.asarray([pose_rear[0] + LR * np.cos(yaw), pose_rear[1] + LR * np.sin(yaw), yaw], np.float32),
            np.asarray(self.lidar_features_host(scan), np.float32), np.asarray(sensors, np.float32),
            np.asarray(control, np.float32)), self.filter_device)
        if not self.started:
            h, s, x, anchor, condition = self._first(lidar, sensors, pose)
            self.started = True
        else:
            h, s, x, anchor, condition = self._step(*self._state, control, lidar, sensors, pose)
        self._keep(h, s, x)
        from sit_lmpc.world.rssm_train import CONDITION_MEAN, CONDITION_SCALE
        anchor, x, condition = jax.device_get((anchor, x, condition))
        grip = float(np.asarray(condition)[0] * CONDITION_SCALE[0] + CONDITION_MEAN[0])
        return np.asarray(anchor), np.asarray(x), grip

    def predictor(self):
        return RSSMPredictor(self.network, self.st_params, self.action_scale, self.h, self.s, self.model,
                             self.variant, self.model_timestep, self.planner_integrator, self.clearance_margin,
                             self.clearance_weight, **self.risk)
