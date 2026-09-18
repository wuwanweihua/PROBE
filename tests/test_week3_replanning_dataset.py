"""Tests for the replanning-level Week 3 dataset builder."""

from __future__ import annotations

import numpy as np
import pytest

from probe.data.build_week3_replanning_dataset import (
    CONDITION_ORDER,
    compute_s_features,
    load_labels,
    make_task_splits,
)


def _write_labels(path, rows):
    import json

    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def test_condition_order_matches_the_trainer():
    # The trainer folds records with a plain reshape and asserts this order, so
    # the builder must emit exactly the same sequence.
    from probe.data.train_week3_predictors import CONDITION_ORDER as TRAINER_ORDER

    assert CONDITION_ORDER == TRAINER_ORDER


def test_load_labels_reads_task_level_success_counts(tmp_path):
    path = tmp_path / "labels.jsonl"
    _write_labels(
        path,
        [
            {
                "suite": "object",
                "base_id": "task0001",
                "original_successes": 7,
                "original_trials": 12,
                "better_successes": 10,
                "better_trials": 12,
                "worse_successes": 3,
                "worse_trials": 12,
            },
            {"suite": "camera", "base_id": "task9999"},
        ],
    )

    labels = load_labels(path, "object")

    assert set(labels) == {"task0001"}
    assert labels["task0001"]["better"].rate == pytest.approx(10 / 12)
    assert labels["task0001"]["original"].trials == 12


def test_load_labels_reads_per_record_success_counts(tmp_path):
    path = tmp_path / "labels.jsonl"
    _write_labels(
        path,
        [
            {
                "suite": "object",
                "base_id": "task0001",
                "condition_id": "original",
                "eval_successes": 6,
                "eval_trials": 12,
            },
            {
                "suite": "object",
                "base_id": "task0001",
                "condition_id": "better",
                "eval_successes": 9,
                "eval_trials": 12,
            },
        ],
    )

    labels = load_labels(path, "object")

    assert labels["task0001"]["original"].successes == 6
    assert labels["task0001"]["better"].successes == 9


def test_task_splits_never_split_a_task_across_splits():
    # Every replanning group of one task must land in the same split, otherwise
    # near-duplicate rows leak between train and test.
    group_to_task = {
        f"task{t:03d}::r{r:02d}": f"task{t:03d}"
        for t in range(10)
        for r in range(4)
    }
    record_ids = {group: [f"{group}__{c}" for c in CONDITION_ORDER] for group in group_to_task}

    payload = make_task_splits(
        group_to_task,
        record_ids,
        seed=7,
        train_fraction=0.6,
        validation_fraction=0.2,
    )

    split_of_task: dict[str, set[str]] = {}
    for split, groups in payload["groups"].items():
        for group in groups:
            split_of_task.setdefault(group_to_task[group], set()).add(split)
    assert all(len(splits) == 1 for splits in split_of_task.values())

    # record_ids must mirror groups so the training loader can resolve records.
    assert sum(len(v) for v in payload["record_ids"].values()) == len(group_to_task) * 3
    for split in ("train", "validation", "test"):
        group_set = set(payload["groups"][split])
        for record_id in payload["record_ids"][split]:
            assert record_id.split("__")[0] in group_set


def test_task_splits_are_reproducible_for_a_seed():
    group_to_task = {f"g{i}": f"t{i // 2}" for i in range(20)}
    record_ids = {group: [group] for group in group_to_task}

    first = make_task_splits(group_to_task, record_ids, 5, 0.7, 0.15)
    second = make_task_splits(group_to_task, record_ids, 5, 0.7, 0.15)

    assert first["groups"] == second["groups"]
    assert first["split_seed"] == 5


def test_compute_s_features_are_finite_for_k_at_least_two():
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(8, 10, 7)).astype(np.float32)

    values = compute_s_features(samples, 4)

    assert values.shape == (8,)
    assert np.isfinite(values).all()


def test_compute_s_features_are_undefined_for_k_equal_one():
    # With a single sample there is no cross-sample variance to measure.
    samples = np.zeros((4, 10, 7), dtype=np.float32)

    values = compute_s_features(samples, 1)

    assert np.isnan(values).all()


def test_compute_s_features_reject_a_prefix_larger_than_available():
    samples = np.zeros((4, 10, 7), dtype=np.float32)

    with pytest.raises(ValueError, match="Requested K=32"):
        compute_s_features(samples, 32)
