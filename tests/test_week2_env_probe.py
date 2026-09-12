from probe.rollout.collect_week2_env_probe import _safe_id


def test_safe_id_preserves_simple_ids():
    assert _safe_id("libero_plus_libero_10_task0738") == "libero_plus_libero_10_task0738"


def test_safe_id_removes_path_separators():
    assert _safe_id("task/738 with spaces") == "task_738_with_spaces"
