"""EKF on the nominal bicycle (single-track) model: baseline and fallback state estimator for the world model.

Measurements per 0.1 s step: PF pose (rear axle x, y, yaw), gyro yaw rate, wheel speed (longitudinal velocity).
"""

from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

from dyna_dist.models.profiles import PARAMETER_FIELDS, vehicle_parameters
from dyna_dist.models.st import derivative as st_derivative
from dyna_dist.models.st import step as st_step

LR = 0.17145


def nominal_parameters(friction=1.0489):
    values = vehicle_parameters("f1tenth", friction, 1.0, "fixed")
    return jnp.asarray([values[name] for name in PARAMETER_FIELDS], dtype=jnp.float32)


def measure(state, control, parameters):
    x, y, delta, speed, yaw, rate, slip = state
    rates = st_derivative(state, control, parameters)
    turn = rates[6] + rate
    accel_x = rates[3] * jnp.cos(slip) - speed * jnp.sin(slip) * turn
    accel_y = rates[3] * jnp.sin(slip) + speed * jnp.cos(slip) * turn
    return jnp.stack((x - LR * jnp.cos(yaw), y - LR * jnp.sin(yaw), yaw, rate, speed * jnp.cos(slip), accel_x, accel_y))


def wrap(angle):
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi


def numerical_jacobian(function, state, epsilon=1e-3):
    offsets = jnp.eye(state.shape[0], dtype=state.dtype) * epsilon
    plus = jax.vmap(lambda d: function(state + d))(offsets)
    minus = jax.vmap(lambda d: function(state - d))(offsets)
    return ((plus - minus) / (2 * epsilon)).T


def transition_with_bias(state, control, parameters):
    return jnp.concatenate((st_step(state[:7], control, parameters), state[7:]))


def measure_with_bias(state, control, parameters):
    return measure(state[:7], control, parameters).at[5:7].add(state[7:9])


@partial(jax.jit, static_argnames=())
def ekf_step(state, covariance, control, next_control, measurement, parameters, process, noise):
    predicted = transition_with_bias(state, control, parameters)
    jacobian = numerical_jacobian(lambda s: transition_with_bias(s, control, parameters), state)
    covariance = jacobian @ covariance @ jacobian.T + process
    innovation = jnp.nan_to_num(measurement - measure_with_bias(predicted, next_control, parameters), nan=0.0,
                                posinf=0.0, neginf=0.0)
    innovation = innovation.at[2].set(wrap(innovation[2]))
    observation = jnp.nan_to_num(numerical_jacobian(lambda s: measure_with_bias(s, next_control, parameters), predicted),
                                 nan=0.0, posinf=0.0, neginf=0.0)
    gain_input = observation @ covariance @ observation.T + noise
    gain = covariance @ observation.T @ jnp.linalg.inv(gain_input)
    updated = predicted + gain @ innovation
    updated = updated.at[4].set(wrap(updated[4])).at[6].set(wrap(updated[6]))
    updated = (updated.at[2].set(jnp.clip(updated[2], -0.42, 0.42)).at[3].set(jnp.maximum(updated[3], 0.0))
               .at[6].set(jnp.clip(updated[6], -1.2, 1.2)))
    covariance = (jnp.eye(9) - gain @ observation) @ covariance
    distance = innovation @ jnp.linalg.solve(gain_input, innovation)
    healthy = (jnp.all(jnp.isfinite(updated)) & jnp.all(jnp.isfinite(covariance)) & (distance < 200.0)
               & (jnp.abs(updated[5]) < 10.0) & (updated[3] < 25.0) & (jnp.trace(covariance) < 100.0)
               & jnp.all(jnp.abs(updated[7:9]) < 3.0))
    yaw = measurement[2]
    reset = jnp.array([measurement[0] + LR * jnp.cos(yaw), measurement[1] + LR * jnp.sin(yaw), predicted[2],
                       jnp.maximum(measurement[4], 0.0), yaw, measurement[3], 0.0, predicted[7], predicted[8]])
    reset = jnp.where(jnp.isfinite(reset[2]), reset, reset.at[2].set(0.0))
    initial = jnp.diag(jnp.array([0.05, 0.05, 0.1, 0.5, 0.05, 0.5, 0.2, 0.5, 0.5]) ** 2)
    updated = jnp.where(healthy, updated, reset)
    covariance = jnp.where(healthy, covariance, initial)
    return updated, covariance


def run_ekf(pose, gyro, wheel, controls, parameters=None, accel=None, accel_std=2.0):
    """pose: (n+1, 3) rear-axle PF pose; gyro, wheel: (n+1,); accel: (n+1, 2) IMU a_x, a_y or None; controls: (n, 2).
    Returns (n+1, 7) estimates."""
    parameters = nominal_parameters() if parameters is None else parameters
    process = jnp.diag(jnp.array([0.02, 0.02, 0.02, 0.3, 0.01, 0.5, 0.05, 0.01, 0.01]) ** 2)
    use_accel = accel is not None
    accel = np.zeros((len(pose), 2)) if accel is None else accel
    accel_std = accel_std if use_accel else 1e6
    noise = jnp.diag(jnp.array([0.03, 0.03, 0.02, 0.03, 0.1, accel_std, accel_std]) ** 2)
    yaw = pose[0, 2]
    state = jnp.array([pose[0, 0] + LR * np.cos(yaw), pose[0, 1] + LR * np.sin(yaw), 0.0, wheel[0], yaw, gyro[0], 0.0,
                       0.0, 0.0], dtype=jnp.float32)
    covariance = jnp.diag(jnp.array([0.05, 0.05, 0.1, 0.5, 0.05, 0.5, 0.2, 0.5, 0.5]) ** 2)
    estimates = [np.asarray(state)[:7]]
    for index in range(len(controls)):
        measurement = jnp.array([pose[index + 1, 0], pose[index + 1, 1], pose[index + 1, 2], gyro[index + 1],
                                 wheel[index + 1], accel[index + 1, 0], accel[index + 1, 1]], dtype=jnp.float32)
        next_control = controls[min(index + 1, len(controls) - 1)]
        state, covariance = ekf_step(state, covariance, jnp.asarray(controls[index]), jnp.asarray(next_control),
                                     measurement, parameters, process, noise)
        estimates.append(np.asarray(state)[:7])
    return np.asarray(estimates)
