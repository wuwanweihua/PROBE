from pathlib import Path

import numpy as np
import pytest

from probe.data.train_week3_predictors import (
    CONDITION_ORDER,
    DatasetArrays,
    FeaturePreprocessor,
    group_ranking_loss,
    infer_records_per_group,
    probability_metrics,
    selection_rows,
    sigmoid_numpy,
    validate_group_layout,
)

torch = pytest.importorskip("torch", reason="group_ranking_loss requires PyTorch")


def test_sigmoid_numpy_is_stable_for_large_logits():
    values = sigmoid_numpy(np.asarray([-1000.0, 0.0, 1000.0]))
    assert np.isfinite(values).all()
    np.testing.assert_allclose(values, [0.0, 0.5, 1.0], atol=1e-6)


def test_preprocessor_is_fit_transform_and_keeps_finite_values():
    train = np.asarray(
        [[0.0, 1.0, 5.0], [1.0, 3.0, 5.0], [2.0, 5.0, 5.0]],
        dtype=np.float32,
    )
    preprocessor = FeaturePreprocessor.fit(train, pca_dim=2)
    transformed = preprocessor.transform(train)

    assert transformed.shape == (3, 2)
    assert np.isfinite(transformed).all()
    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-5)


def test_probability_metrics_use_success_and_trial_counts():
    probabilities = np.asarray([0.5, 0.75], dtype=np.float32)
    successes = np.asarray([4.0, 6.0], dtype=np.float32)
    trials = np.asarray([8.0, 8.0], dtype=np.float32)
    metrics = probability_metrics(probabilities, successes, trials)

    assert metrics["num_records"] == 2
    assert metrics["total_trials"] == 16
    assert metrics["binomial_nll"] > 0
    assert metrics["brier_exec"] >= 0


def test_selection_accepts_any_condition_tied_for_best_success_rate():
    records = [
        {
            "group_id": "task1",
            "condition_id": "original",
            "eval_successes": 2,
            "eval_trials": 8,
        },
        {
            "group_id": "task1",
            "condition_id": "better",
            "eval_successes": 5,
            "eval_trials": 8,
        },
        {
            "group_id": "task1",
            "condition_id": "worse",
            "eval_successes": 5,
            "eval_trials": 8,
        },
    ]
    rows = selection_rows(
        records,
        np.asarray(["test", "test", "test"]),
        np.asarray([0.1, 0.2, 0.9], dtype=np.float32),
        "B/mlp",
    )

    assert len(rows) == 1
    assert rows[0]["predicted_condition"] == "worse"
    assert rows[0]["oracle_condition"] == "better"
    assert rows[0]["oracle_conditions"] == ["better", "worse"]
    assert rows[0]["selection_correct"] is True
    assert rows[0]["empirical_regret"] == 0.0


def _group_records(group_ids, successes):
    """Build a flat manifest block in CONDITION_ORDER for each group."""
    records = []
    for group_id, rates in zip(group_ids, successes):
        for condition, rate in zip(CONDITION_ORDER, rates):
            records.append(
                {
                    "group_id": group_id,
                    "condition_id": condition,
                    "eval_successes": rate,
                    "eval_trials": 12,
                }
            )
    return records


def test_infer_records_per_group_counts_the_first_contiguous_run():
    records = _group_records(["g0", "g1"], [(1, 2, 3), (4, 5, 6)])
    assert infer_records_per_group(records) == 3


def test_interleaved_groups_are_rejected_by_the_contiguity_check():
    # Conditions listed group-major instead of condition-major: g0,g1,g0,g1,...
    records = []
    for condition in CONDITION_ORDER:
        for group_id in ("g0", "g1"):
            records.append(
                {
                    "group_id": group_id,
                    "condition_id": condition,
                    "eval_successes": 1,
                    "eval_trials": 12,
                }
            )
    with pytest.raises(ValueError, match="span multiple groups"):
        validate_group_layout(records, np.asarray(["train"] * 6), 3)


def test_validate_group_layout_rejects_a_split_straddling_group():
    records = _group_records(["g0"], [(1, 2, 3)])
    with pytest.raises(ValueError, match="straddles splits"):
        validate_group_layout(records, np.asarray(["train", "train", "test"]), 3)


def test_validate_group_layout_rejects_a_wrong_condition_order():
    records = _group_records(["g0"], [(1, 2, 3)])
    records[0]["condition_id"], records[1]["condition_id"] = (
        records[1]["condition_id"],
        records[0]["condition_id"],
    )
    with pytest.raises(ValueError, match="expected condition order"):
        validate_group_layout(records, np.asarray(["train"] * 3), 3)


def test_group_split_indices_are_group_level_not_record_level():
    records = _group_records(["g0", "g1"], [(1, 2, 3), (4, 5, 6)])
    dataset = DatasetArrays(
        records=records,
        features=np.zeros((6, 2), dtype=np.float32),
        successes=np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float32),
        trials=np.full(6, 12.0, dtype=np.float32),
        splits=np.asarray(["train", "train", "train", "test", "test", "test"]),
        records_per_group=3,
    )

    grouped = dataset.group_split_indices
    assert dataset.num_groups == 2
    np.testing.assert_array_equal(grouped["train"], [0])
    np.testing.assert_array_equal(grouped["test"], [1])
    # Record-level indices would be [0,1,2] and [3,4,5]; the group-level
    # mapping must not be confused with those.
    np.testing.assert_array_equal(dataset.split_indices["test"], [3, 4, 5])
    assert list(dataset.group_splits) == ["train", "test"]


def test_group_ranking_loss_prefers_correctly_ordered_logits():
    successes = torch.tensor([3.0, 6.0, 9.0])
    trials = torch.full((3,), 12.0)
    good = torch.tensor([0.0, 1.0, 2.0])
    bad = -good

    good_loss = float(group_ranking_loss(good, successes, trials, 3, 0.1))
    bad_loss = float(group_ranking_loss(bad, successes, trials, 3, 0.1))
    assert good_loss < bad_loss


def test_group_ranking_loss_is_minimised_by_matching_the_empirical_rates():
    # Logits proportional to the empirical rates reproduce the soft target, so
    # any perturbation away from that ordering must increase the loss.
    successes = torch.tensor([3.0, 6.0, 9.0])
    trials = torch.full((3,), 12.0)
    rates = successes / trials

    matched = float(group_ranking_loss(rates, successes, trials, 3, 1.0))
    perturbed = float(
        group_ranking_loss(rates + torch.tensor([1.0, 0.0, -1.0]), successes, trials, 3, 1.0)
    )
    assert matched < perturbed


def test_group_ranking_loss_is_invariant_to_a_per_group_shift():
    # Only within-group differences matter; a constant added to every logit of
    # a group must not change the loss.
    successes = torch.tensor([3.0, 6.0, 9.0, 1.0, 2.0, 4.0])
    trials = torch.full((6,), 12.0)
    logits = torch.tensor([0.1, 0.5, 0.9, -0.2, 0.0, 0.3])

    base = float(group_ranking_loss(logits, successes, trials, 3, 0.1))
    shifted = float(group_ranking_loss(logits + 5.0, successes, trials, 3, 0.1))
    assert base == pytest.approx(shifted, abs=1e-6)


def test_group_ranking_loss_rejects_a_non_dividing_record_count():
    successes = torch.tensor([3.0, 6.0, 9.0, 1.0])
    trials = torch.full((4,), 12.0)
    with pytest.raises(ValueError, match="fold into groups"):
        group_ranking_loss(torch.zeros(4), successes, trials, 3, 0.1)
