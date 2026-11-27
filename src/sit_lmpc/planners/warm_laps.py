import numpy as np


def synthetic_laps(track, laps, max_speed, lateral_accel=4.0):
    cl = track.centerline
    ss, xs, ys, yaws = (np.asarray(v, float) for v in (cl.ss, cl.xs, cl.ys, cl.yaws))
    ks = np.asarray(getattr(cl, "ks", np.zeros_like(ss)), float)
    length = float(track.s_frame_max)
    rng = np.random.default_rng(0)
    out_f, out_c = [], []
    for i in range(laps):
        scale = 1.0 + 0.05 * rng.standard_normal()
        v = np.minimum(max_speed, np.sqrt(lateral_accel / np.maximum(np.abs(ks), 1e-3))) * scale
        v = np.clip(v, 1.0, max_speed)
        offset = 0.05 * rng.standard_normal()
        step = 0.1
        times = np.concatenate([[0.0], np.cumsum(np.diff(ss) / v[:-1])])
        grid = np.arange(0.0, times[-1], step)
        s = np.interp(grid, times, ss)
        speed = np.interp(s, ss, v)
        yaw = np.interp(s, ss, np.unwrap(yaws))
        k = np.interp(s, ss, ks)
        n = len(grid)
        ey = offset + 0.08 * np.sin(2 * np.pi * s / max(length, 1.0) * 3 + i) + 0.02 * rng.standard_normal(n)
        heading = 0.03 * rng.standard_normal(n)
        speed = speed * (1 + 0.03 * rng.standard_normal(n))
        x = np.interp(s, ss, xs) - ey * np.sin(yaw)
        y = np.interp(s, ss, ys) + ey * np.cos(yaw)
        steer = np.arctan(0.33 * k) + 0.01 * rng.standard_normal(n)
        ttg = (n - 1 - np.arange(n)) * step
        frenet = np.column_stack([s, ey, steer, speed, heading, speed * k, np.zeros_like(s), ttg])
        cart = np.column_stack([x, y, steer, speed, yaw, speed * k, np.zeros_like(s), ttg])
        out_f.append(frenet)
        out_c.append(cart)
    return out_f, out_c


def warm_safe_set(planner, timestep, track, seed, corners, max_speed):
    from sit_lmpc.experiments.randomness import ExperimentRandomness
    from sit_lmpc.planners.safe_set import SafeSet

    frenet_laps, cart_laps = synthetic_laps(track, max(planner.safe_set_size, planner.safe_set_cap_laps, 4) + 1, max_speed)
    safe = SafeSet(planner, timestep, float(track.s_frame_max), seed, randomness=ExperimentRandomness(seed))
    safe.frenet, safe.cartesian = frenet_laps, cart_laps
    if planner.safe_set_footprint_filter and corners is not None:
        safe.footprint = (corners, planner.boundary_limit)
    return safe, frenet_laps[-1], cart_laps[-1]
