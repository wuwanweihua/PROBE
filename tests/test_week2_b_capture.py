import numpy as np

from probe.policies.pi05_client import Pi05Client


class _FakeWebsocketClient:
    def __init__(self):
        self.payloads = []

    def infer(self, payload):
        self.payloads.append(dict(payload))
        result = {"actions": np.zeros((10, 7), dtype=np.float32)}
        if payload.get("return_feature"):
            result["features"] = np.ones((6,), dtype=np.float32)
            result["feature_metadata"] = {
                "feature_type": "prefix_embedding_mean"
            }
        return result


def test_feature_is_requested_only_for_first_sample():
    client = object.__new__(Pi05Client)
    fake = _FakeWebsocketClient()
    client._client = fake

    samples, times, feature, metadata = client.sample_action_chunks_with_feature_timing(
        {"prompt": "test"},
        k=4,
    )

    assert samples.shape == (4, 10, 7)
    assert len(times) == 4
    assert feature.shape == (6,)
    assert metadata["feature_type"] == "prefix_embedding_mean"
    assert [payload["return_feature"] for payload in fake.payloads] == [
        True,
        False,
        False,
        False,
    ]
