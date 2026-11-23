"""Condition-aware safe-set selection."""

from types import SimpleNamespace
import numpy as np

from sit_lmpc.planners.safe_set import SafeSet


def make(threshold, minimum, size=2):
    config = SimpleNamespace(safe_set_size=size, safe_set_min_states=10 ** 6, condition_threshold=threshold,
                             condition_min_laps=minimum, safe_set_max_points=10 ** 6, hull_selection_points=10 ** 6,
                             reference_interval=3)
    safe_set = SafeSet(config, 0.1, 100.0)
    for index, tag in enumerate(([0.0, 0.0], [0.1, 0.0], [3.0, 0.0], [3.1, 0.0], None)):
        record = np.column_stack((np.linspace(0, 99, 50), np.full(50, float(index))))
        safe_set.frenet.append(record)
        safe_set.cartesian.append(record)
        safe_set.tags.append(None if tag is None else np.array(tag))
    return safe_set


def test_no_condition_uses_every_lap():
    safe_set = make(0.5, 1)
    assert safe_set.active() == (0, 1, 2, 3, 4)


def test_threshold_keeps_only_similar_laps():
    safe_set = make(0.5, 1)
    safe_set.condition = np.array([3.05, 0.0])
    assert safe_set.active() == (2, 3)
    frenet, _ = safe_set.arrays()
    assert set(np.unique(frenet[:, 1])) == {2.0, 3.0}


def test_minimum_laps_fall_back_to_the_closest():
    safe_set = make(0.01, 3)
    safe_set.condition = np.array([0.0, 0.0])
    assert safe_set.active() == (0, 1, 2)


def test_trim_keeps_tags_aligned_with_records():
    safe_set = make(0.5, 1, size=3)
    safe_set.config.safe_set_min_states = 0
    safe_set.controls = [np.zeros((50, 2))] * 5
    safe_set.control_masks = [np.zeros(50, bool)] * 5
    safe_set.state_ids = [np.zeros((50, 2), int)] * 5
    safe_set.trim()
    assert len(safe_set.tags) == len(safe_set.frenet) == 3
    assert [None if t is None else float(t[0]) for t in safe_set.tags] == [3.0, 3.1, None]


def make_state_mode(condition):
    config = SimpleNamespace(safe_set_size=3, safe_set_min_states=10 ** 6, condition_threshold=0.1,
                             condition_min_laps=1, condition_mode="state",
                             safe_set_max_points=10 ** 6, hull_selection_points=10 ** 6, reference_interval=3)
    safe_set = SafeSet(config, 0.1, 100.0)
    s = np.linspace(0, 99, 100)
    patch = (s > 40) & (s < 60)
    for index, low in enumerate((False, True, True)):
        record = np.column_stack((s, np.full(100, float(index))))
        safe_set.frenet.append(record)
        safe_set.cartesian.append(record)
        safe_set.tags.append(None)
        safe_set.state_tags.append(np.where(patch & low, 0.6, 1.05))
    safe_set.condition = np.array([condition])
    return safe_set


def test_state_mode_uses_every_record_and_points_with_matching_grip():
    safe_set = make_state_mode(1.05)
    assert safe_set.active() == (0, 1, 2)
    inside, _ = safe_set.select_window(42.0, 58.0)
    assert set(np.unique(inside[:, 1])) == {1.0, 2.0}
    outside, _ = safe_set.select_window(10.0, 30.0)
    assert set(np.unique(outside[:, 1])) == {0.0, 1.0, 2.0}


def test_state_mode_follows_a_much_lower_current_estimate():
    safe_set = make_state_mode(0.6)
    frenet, _ = safe_set.select_window(10.0, 30.0)
    assert len(np.unique(frenet[:, 1])) == 1


def test_state_mode_keeps_whole_records_across_the_window():
    safe_set = make_state_mode(1.05)
    tags = safe_set.state_tags[0].copy()
    tags[45:50] = 0.7
    safe_set.state_tags[0] = tags
    frenet, _ = safe_set.select_window(10.0, 90.0)
    for record in np.unique(frenet[:, 1]):
        rows = frenet[frenet[:, 1] == record, 0]
        assert rows.min() < 12 and rows.max() > 88
