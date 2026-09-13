from pathlib import Path

import numpy as np

from probe.data.train_week3_predictors import (
    FeaturePreprocessor,
    probability_metrics,
    selection_rows,
    sigmoid_numpy,
)


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
