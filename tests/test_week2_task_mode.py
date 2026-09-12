from types import SimpleNamespace

from probe.rollout.collect_week2_env_probe import _inline_original_labels, _parse_task_ids


def test_parse_task_ids():
    assert _parse_task_ids("738, 757,761") == [738, 757, 761]


def test_inline_original_labels():
    labels = _inline_original_labels('{"738": 8, "757": 7}')
    assert labels["738"]["original"]["eval_successes"] == 8
    assert labels["757"]["original"]["eval_trials"] == 8
