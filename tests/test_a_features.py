from pathlib import Path

import numpy as np

from probe.data.extract_a_features import (
    compose_a_feature,
    load_policy_images,
    resolve_observation_path,
)


def test_compose_a_feature_normalizes_each_modality_before_concatenation():
    feature = compose_a_feature(
        np.asarray([3.0, 4.0]),
        np.asarray([0.0, 2.0]),
        np.asarray([5.0, 12.0]),
    )

    assert feature.shape == (6,)
    np.testing.assert_allclose(feature[:2], [0.6, 0.8])
    np.testing.assert_allclose(feature[2:4], [0.0, 1.0])
    np.testing.assert_allclose(feature[4:], [5 / 13, 12 / 13])


def test_load_policy_images_reads_the_two_policy_views(tmp_path: Path):
    path = tmp_path / "obs.npz"
    agent = np.zeros((4, 5, 3), dtype=np.uint8)
    wrist = np.ones((4, 5, 3), dtype=np.uint8)
    np.savez_compressed(path, policy_image=agent, policy_wrist_image=wrist)

    loaded_agent, loaded_wrist = load_policy_images(path, "record")

    np.testing.assert_array_equal(loaded_agent, agent)
    np.testing.assert_array_equal(loaded_wrist, wrist)


def test_resolve_observation_path_uses_source_dataset_dir(tmp_path: Path):
    source = tmp_path / "source"
    observation = source / "observations" / "state.npz"
    observation.parent.mkdir(parents=True)
    observation.write_bytes(b"placeholder")

    result = resolve_observation_path(
        {
            "record_id": "camera__state_original",
            "source_dataset_dir": str(source),
            "source_observation_path": "observations/state.npz",
        },
        tmp_path / "week3",
    )

    assert result == observation
