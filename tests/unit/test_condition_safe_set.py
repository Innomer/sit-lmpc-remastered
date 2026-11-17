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
