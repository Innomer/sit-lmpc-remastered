from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from sit_lmpc.config import load_config
from sit_lmpc.planners.lmppi import candidate_scores, footprint_corners, sample_actions, with_previous_plan
from sit_lmpc.planners.safe_set import SafeSet, within_footprint_limit


def test_previous_plan_is_sampled_and_kept_as_candidate():
    planner = load_config().planner
    previous = jnp.asarray(np.linspace(-0.5, 0.5, planner.horizon * 2).reshape(planner.horizon, 2), jnp.float32)
    da, actions = sample_actions(previous, jax.random.key(3), planner)
    assert np.allclose(da[0], 0.0)
    assert np.allclose(actions[0], previous)
    assert not np.allclose(actions[1], previous)
    candidates = jnp.zeros((8, planner.horizon, 2), jnp.float32)
    kept = with_previous_plan(candidates, previous)
    assert np.allclose(kept[-1], previous)
    assert np.allclose(kept[:-1], 0.0)


def test_safe_set_violation_is_in_the_hard_tier():
    frenet = jnp.zeros((2, 5, 7), jnp.float32)
    hull = jnp.asarray([0.5, 0.0], jnp.float32)
    safety = jnp.asarray([0.0, 1.0], jnp.float32)
    score, _, _, _ = candidate_scores(frenet, hull, 0.85, 0.03, None, safety)
    assert int(jnp.argmin(score)) == 1
    off_track = frenet.at[1, 2, 1].set(1.2)
    score, _, _, _ = candidate_scores(off_track, jnp.asarray([0.0, 0.0], jnp.float32), 0.85, 0.03, None,
                                      jnp.asarray([5.0, 0.0], jnp.float32))
    assert int(jnp.argmin(score)) == 0


def test_body_corner_limit_does_not_subtract_half_width_twice():
    planner = replace(load_config().planner, half_width=1.1, boundary_buffer=0.25)
    assert np.isclose(replace(planner, footprint_limits=True).boundary_limit, 1.0)
    assert np.isclose(replace(planner, footprint_limits=True, footprint_buffer=0.05).boundary_limit, 1.05)
    assert np.isclose(replace(planner, footprint_limits=False).boundary_limit, 0.85)


def test_safe_set_drops_states_outside_the_body_corner_limit():
    corners = footprint_corners(0.58, 0.31, -0.0065)
    frenet = np.zeros((4, 7))
    frenet[:, 0] = [10.0, 11.0, 12.0, 13.0]
    frenet[:, 1] = [0.0, 0.80, 0.86, -0.90]
    keep = within_footprint_limit(frenet, corners, 1.0)
    assert keep.tolist() == [True, True, False, False]
    planner = load_config().planner
    safe = SafeSet(planner, 0.1, 100.0)
    lap = np.zeros((40, 7))
    lap[:, 0] = np.linspace(0.0, 40.0, 40)
    lap[:, 1] = np.where(np.arange(40) % 2, 0.9, 0.1)
    safe.frenet = [lap]
    safe.cartesian = [lap.copy()]
    safe.footprint = (corners, 1.0)
    selected, _ = safe.select_window(0.0, 40.0)
    assert len(selected) and np.all(np.abs(selected[:, 1]) <= 0.2)
    safe.footprint = None
    selected, _ = safe.select_window(0.0, 40.0)
    assert np.any(np.abs(selected[:, 1]) > 0.8)


def test_tag_matched_safe_set_and_tag_value_inputs():
    from sit_lmpc.planners.change_monitor import ChangeMonitor
    from sit_lmpc.training.features import feature_names, features

    planner = replace(load_config().planner, change_safe_set="tag", change_new_laps=2)
    monitor = ChangeMonitor(100.0, stretch=10)
    safe = SafeSet(planner, 0.1, 100.0)
    safe.change_monitor = monitor
    normal = np.full(10, 0.01)
    changed = normal.copy()
    changed[4] = 0.5
    safe.frenet = [np.zeros((5, 7)) for _ in range(5)]
    safe.record_info = [{"lap": i, "laptime": 40.0 + i, "warmup": i == 0, "tags": normal} for i in range(4)]
    safe.record_info.append({"lap": 4, "laptime": 50.0, "warmup": False, "tags": changed})
    monitor.history = [[0.01] * 5 for _ in range(10)]
    monitor.active, monitor.detection_lap = True, 4
    monitor.pre_limit = np.asarray([monitor.limit(k) for k in range(10)])
    monitor.changed[4] = True
    chosen, used = safe.tag_records(40.0, 45.0)
    assert used == 1
    safe.frenet.append(np.zeros((5, 7)))
    safe.record_info.append({"lap": 5, "laptime": 52.0, "warmup": False, "tags": changed})
    chosen, used = safe.tag_records(40.0, 45.0)
    assert used == 2 and chosen == [4, 5]
    chosen, used = safe.tag_records(10.0, 15.0)
    assert used == 3 and chosen == [0, 1, 2, 3, 4, 5]
    assert feature_names("state_tag")[-1] == "tag"
    states = np.arange(16, dtype=float).reshape(2, 8)
    assert features(states, "state_tag").tolist() == [[0, 1, 2, 3, 4, 7], [8, 9, 10, 11, 12, 15]]
    monitor.last_pass_tag[4] = 0.4
    tags = monitor.query_tags()
    assert np.isclose(tags[4], 0.4) and np.isclose(tags[0], 0.01)


def test_reduced_speed_plans_scale_the_speed_profile():
    from sit_lmpc.planners.lmppi import reduced_speed_plans

    previous = np.column_stack([np.linspace(-0.2, 0.2, 15), np.zeros(15)])
    plans = reduced_speed_plans(previous, 8.0, (0.9, 0.75, 0.6), 0.1, 9.51, 15)
    assert plans.shape == (3, 15, 2)
    assert np.allclose(plans[:, :, 0], previous[:, 0])
    assert np.all(np.abs(plans[:, :, 1]) <= 1.0)
    for plan, factor in zip(plans, (0.9, 0.75, 0.6)):
        speed = 8.0 + np.cumsum(plan[:, 1] * 9.51 * 0.1)
        assert np.isclose(speed[-1], factor * 8.0, atol=1e-6)
        assert np.all(speed[1:] <= speed[:-1] + 1e-9)


def test_safe_set_speeds_scaled_after_detection_only():
    from sit_lmpc.planners.change_monitor import ChangeMonitor, CONSERVATIVE, NORMAL

    planner = load_config().planner
    monitor = ChangeMonitor(100.0, stretch=10, speed_factor=0.8)
    safe = SafeSet(planner, 0.1, 100.0)
    safe.change_monitor = monitor
    lap = np.zeros((100, 7))
    lap[:, 0] = np.linspace(0.0, 99.0, 100)
    lap[:, 1] = np.where(np.arange(100) % 2, 0.2, -0.2)
    lap[:, 3] = 9.0
    after = lap.copy()
    after[:, 3] = 6.0
    safe.frenet = [lap, after]
    safe.cartesian = [lap.copy(), after.copy()]
    safe.record_info = [{"lap": 3, "laptime": 38.0, "warmup": False}, {"lap": 9, "laptime": 50.0, "warmup": False}]
    selected, _ = safe.select_window(10.0, 40.0)
    assert np.allclose(np.sort(np.unique(selected[:, 3])), [6.0, 9.0])
    monitor.active, monitor.detection_lap = True, 8
    monitor.mode[:] = NORMAL
    monitor.mode[2] = CONSERVATIVE
    monitor.caps[2], monitor.cap_factor[2] = 7.2, 0.8
    selected, _ = safe.select_window(10.0, 40.0)
    s = selected[:, 0]
    pre = selected[:, 3] != 6.0
    on_flagged = (s >= 20.0) & (s < 30.0)
    assert np.allclose(selected[pre & on_flagged, 3], 7.2)
    assert np.allclose(selected[pre & ~on_flagged, 3], 9.0)
    assert np.all(selected[~pre, 3] == 6.0)
    assert np.isclose(safe.frenet[0][25, 3], 9.0)
    monitor.caps[2], monitor.cap_factor[2] = np.inf, 1.0
    monitor.changed[3] = True
    assert np.allclose(monitor.safe_set_factors()[[2, 3, 4]], [0.8, 0.8, 1.0])
