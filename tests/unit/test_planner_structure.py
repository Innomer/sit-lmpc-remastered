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
