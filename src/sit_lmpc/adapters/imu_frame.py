import numpy as np


def axis_matrix(axes):
    matrix = np.zeros((3, 3))
    for row in range(3):
        name = str(axes[row]).strip()
        sign = -1.0 if name.startswith("-") else 1.0
        matrix[row, "xyz".index(name[-1])] = sign
    return matrix


def level_rotation(gravity):
    norm = float(np.linalg.norm(gravity))
    if not 5.0 < norm < 15.0:
        return np.eye(3)
    g = np.asarray(gravity, float) / norm
    up = np.array([0.0, 0.0, 1.0])
    axis = np.cross(g, up)
    sine = float(np.linalg.norm(axis))
    cosine = float(g @ up)
    if sine < 1e-9:
        return np.eye(3)
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + k + k @ k * ((1.0 - cosine) / sine ** 2)


def tilt_degrees(gravity):
    g = np.asarray(gravity, float)
    return float(np.degrees(np.arccos(np.clip(g[2] / np.linalg.norm(g), -1.0, 1.0))))


def to_center_of_gravity(accel_x, accel_y, rate, rate_dot, offset_x, offset_y):
    return (accel_x + rate_dot * offset_y + rate ** 2 * offset_x,
            accel_y - rate_dot * offset_x + rate ** 2 * offset_y)
