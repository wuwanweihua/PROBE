from types import SimpleNamespace

import numpy as np

from probe.rollout.collect_week2_env_probe import _inline_original_labels, _parse_task_ids


def test_parse_task_ids():
    assert _parse_task_ids("738, 757,761") == [738, 757, 761]


def test_inline_original_labels():
    labels = _inline_original_labels('{"738": 8, "757": 7}')
    assert labels["738"]["original"]["eval_successes"] == 8
    assert labels["757"]["original"]["eval_trials"] == 8


def test_numpy_initial_states_are_not_treated_as_booleans():
    initial_states = np.zeros((2, 4), dtype=np.float32)
    assert initial_states is not None and len(initial_states) > 0
