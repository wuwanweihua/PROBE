from pathlib import Path

import pytest

from probe.data.resplit_week3 import summarize_split


def _records(groups):
    """Build a manifest block: three conditions per group, groups in order."""
    conditions = ("better", "original", "worse")
    out = []
    for group_id, suite in groups:
        for condition in conditions:
            out.append(
                {
                    "record_id": f"{group_id}_{condition}",
                    "group_id": group_id,
                    "condition_id": condition,
                    "suite": suite,
                }
            )
    return out


def test_summarize_split_counts_records_and_groups():
    records = _records([("g0", "camera"), ("g1", "camera"), ("g2", "robot")])
    payload = {
        "split_seed": 7,
        "fractions": {"train": 0.7, "validation": 0.15, "test": 0.15},
        "groups": {"train": ["g0"], "validation": ["g1"], "test": ["g2"]},
        "record_ids": {
            "train": ["g0_better", "g0_original", "g0_worse"],
            "validation": ["g1_better", "g1_original", "g1_worse"],
            "test": ["g2_better", "g2_original", "g2_worse"],
        },
    }

    summary = summarize_split(payload, records)

    assert summary["split_seed"] == 7
    assert summary["group_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert summary["record_counts"] == {"train": 3, "validation": 3, "test": 3}
    assert summary["suite_group_counts"]["camera"] == {"train": 1, "validation": 1}
    assert summary["suite_group_counts"]["robot"] == {"test": 1}


def test_summarize_split_rejects_a_group_spanning_two_splits():
    records = _records([("g0", "camera")])
    payload = {
        "split_seed": 7,
        "fractions": {"train": 0.7, "validation": 0.15, "test": 0.15},
        "groups": {"train": ["g0"], "validation": [], "test": []},
        "record_ids": {
            "train": ["g0_better", "g0_original"],
            "validation": [],
            "test": ["g0_worse"],
        },
    }

    with pytest.raises(ValueError, match="straddles splits"):
        summarize_split(payload, records)


def test_summarize_split_rejects_a_missing_group():
    records = _records([("g0", "camera"), ("g1", "camera")])
    payload = {
        "split_seed": 7,
        "fractions": {"train": 0.7, "validation": 0.15, "test": 0.15},
        "groups": {"train": ["g0"], "validation": [], "test": []},
        "record_ids": {
            "train": ["g0_better", "g0_original", "g0_worse"],
            "validation": [],
            "test": [],
        },
    }

    with pytest.raises(ValueError, match="missing from the split"):
        summarize_split(payload, records)


def test_summarize_split_rejects_a_duplicated_group():
    records = _records([("g0", "camera")])
    payload = {
        "split_seed": 7,
        "fractions": {"train": 0.7, "validation": 0.15, "test": 0.15},
        "groups": {"train": ["g0"], "validation": ["g0"], "test": []},
        "record_ids": {
            "train": ["g0_better", "g0_original", "g0_worse"],
            "validation": [],
            "test": [],
        },
    }

    with pytest.raises(ValueError, match="appears in two splits"):
        summarize_split(payload, records)
