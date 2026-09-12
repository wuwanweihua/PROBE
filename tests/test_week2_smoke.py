from probe.rollout.collect_week2_smoke import _select_base_ids


def test_select_base_ids_requires_all_conditions():
    grouped = {
        "task0001": {
            "original": [{"record_id": "a"}],
            "better": [{"record_id": "b"}],
            "worse": [{"record_id": "c"}],
        },
        "task0002": {
            "original": [{"record_id": "d"}],
            "better": [{"record_id": "e"}],
        },
    }

    assert _select_base_ids(grouped, ("original", "better", "worse"), None) == [
        "task0001"
    ]


def test_select_base_ids_is_deterministic_and_bounded():
    grouped = {
        "task0002": {"original": [], "better": [], "worse": []},
        "task0001": {"original": [], "better": [], "worse": []},
    }

    assert _select_base_ids(grouped, ("original", "better", "worse"), 1) == [
        "task0001"
    ]
