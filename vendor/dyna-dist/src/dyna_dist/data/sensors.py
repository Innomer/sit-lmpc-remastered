"""Simulated car sensors from ST trajectories: VESC IMU, wheel speed and a particle-filter pose."""

import numpy as np


DEFAULT_SENSOR_RANGES = {
    "imu_offset_x": (-0.20, 0.15),
    "imu_offset_y": (-0.05, 0.05),
    "imu_yaw_misalignment": (-0.05, 0.05),
    "accel_bias": (-0.5, 0.5),
    "accel_noise": (0.05, 1.0),
    "gyro_bias": (-0.05, 0.05),
    "gyro_noise": (0.002, 0.05),
    "gyro_scale": (0.95, 1.05),
    "wheel_scale": (0.93, 1.07),
    "wheel_noise": (0.01, 0.2),
    "pose_noise": (0.005, 0.05),
    "yaw_noise": (0.002, 0.03),
    "pose_lag": (0.0, 0.1),
}

SENSOR_FIELDS = ("accel_x", "accel_y", "gyro_z", "wheel_speed", "pose_x", "pose_y", "pose_yaw")


def sample_sensor_parameters(rng, ranges=None):
    ranges = dict(DEFAULT_SENSOR_RANGES if ranges is None else ranges)
    parameters = {}
    for name in DEFAULT_SENSOR_RANGES:
        low, high = ranges[name]
        if name in ("accel_bias",):
            parameters["accel_bias_x"] = float(rng.uniform(low, high))
            parameters["accel_bias_y"] = float(rng.uniform(low, high))
        else:
            parameters[name] = float(rng.uniform(low, high))
    return parameters


def derivatives(rhs, states, controls, vehicle):
    count = len(controls)
    result = np.zeros((count + 1, states.shape[1]))
    vehicle = np.asarray(vehicle, dtype=np.float64)
    for index in range(count + 1):
        control = controls[max(index - 1, 0)]
        parameters = vehicle[min(index, len(vehicle) - 1)] if vehicle.ndim == 2 else vehicle
        result[index] = rhs(states[index], np.asarray(control, dtype=np.float64), parameters)
    return result


def body_accelerations(states, rates, offset_x, offset_y):
    speed = states[:, 3]
    yaw_rate = states[:, 5]
    slip = states[:, 6]
    speed_rate = rates[:, 3]
    yaw_acceleration = rates[:, 5]
    slip_rate = rates[:, 6]
    accel_x = speed_rate * np.cos(slip) - speed * np.sin(slip) * (slip_rate + yaw_rate)
    accel_y = speed_rate * np.sin(slip) + speed * np.cos(slip) * (slip_rate + yaw_rate)
    accel_x = accel_x - yaw_acceleration * offset_y - yaw_rate**2 * offset_x
    accel_y = accel_y + yaw_acceleration * offset_x - yaw_rate**2 * offset_y
    return accel_x, accel_y


def lagged(values, times, lag):
    return np.interp(times - lag, times, values)


def simulate_sensors(states, controls, rhs, vehicle, timestep, rng, parameters):
    """Sensor readings at every recorded state; accelerations use the control acting just before that state
    (controls[t-1]), as a real IMU would measure them; the first state uses controls[0]."""
    rates = derivatives(rhs, states, controls, vehicle)
    accel_x, accel_y = body_accelerations(states, rates, parameters["imu_offset_x"], parameters["imu_offset_y"])
    angle = parameters["imu_yaw_misalignment"]
    imu_x = np.cos(angle) * accel_x + np.sin(angle) * accel_y
    imu_y = -np.sin(angle) * accel_x + np.cos(angle) * accel_y
    count = len(states)
    imu_x = imu_x + parameters["accel_bias_x"] + rng.normal(0.0, parameters["accel_noise"], count)
    imu_y = imu_y + parameters["accel_bias_y"] + rng.normal(0.0, parameters["accel_noise"], count)
    gyro = (parameters["gyro_scale"] * states[:, 5] + parameters["gyro_bias"]
            + rng.normal(0.0, parameters["gyro_noise"], count))
    longitudinal = states[:, 3] * np.cos(states[:, 6])
    wheel = parameters["wheel_scale"] * longitudinal + rng.normal(0.0, parameters["wheel_noise"], count)
    lr = vehicle_lr(vehicle)
    yaw = np.unwrap(states[:, 4])
    rear_x = states[:, 0] - lr * np.cos(yaw)
    rear_y = states[:, 1] - lr * np.sin(yaw)
    times = np.arange(count) * timestep
    lag = parameters["pose_lag"]
    pose_x = lagged(rear_x, times, lag) + rng.normal(0.0, parameters["pose_noise"], count)
    pose_y = lagged(rear_y, times, lag) + rng.normal(0.0, parameters["pose_noise"], count)
    pose_yaw = lagged(yaw, times, lag) + rng.normal(0.0, parameters["yaw_noise"], count)
    pose_yaw = (pose_yaw + np.pi) % (2 * np.pi) - np.pi
    return np.stack((imu_x, imu_y, gyro, wheel, pose_x, pose_y, pose_yaw), axis=1)


def vehicle_lr(vehicle):
    from dyna_dist.models.profiles import PARAMETER_FIELDS

    vehicle = np.asarray(vehicle)
    row = vehicle[0] if vehicle.ndim == 2 else vehicle
    return float(row[list(PARAMETER_FIELDS).index("lr")])
