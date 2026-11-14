"""External simulator boundary; no shared source patches are applied."""


def vehicle_parameters_for(simulation, friction, mass_ratio=1.0):
    if simulation.backend in ("ros", "car"):
        from .ros_f1tenth import f1tenth_vehicle_parameters

        params = f1tenth_vehicle_parameters(friction, mass_ratio)
    else:
        from .f1tenth import vehicle_parameters

        params = vehicle_parameters(friction, mass_ratio)
    if simulation.steering_limit is not None:
        params["s_min"] = -float(simulation.steering_limit)
        params["s_max"] = float(simulation.steering_limit)
    return params
