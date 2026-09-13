from pathlib import Path

import numpy as np

from probe.data.train_week3_predictors import (
    FeaturePreprocessor,
    probability_metrics,
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
