"""openpi pi0.5 websocket client wrapper."""

from __future__ import annotations

import time
from typing import Any

import numpy as np


class Pi05Client:
    """Thin wrapper around openpi_client.WebsocketClientPolicy."""

    def __init__(self, host: str = "0.0.0.0", port: int = 18000) -> None:
        from openpi_client import websocket_client_policy

        self.host = host
        self.port = int(port)
        self._client = websocket_client_policy.WebsocketClientPolicy(host, self.port)

    def infer_once(self, element: dict[str, Any]) -> np.ndarray:
        result = self._client.infer(element)
        if isinstance(result, dict) and "actions" in result:
            return np.asarray(result["actions"])
        return np.asarray(result)

    def infer_once_timed(self, element: dict[str, Any]) -> tuple[np.ndarray, float]:
        start = time.perf_counter()
        actions = self.infer_once(element)
        return actions, time.perf_counter() - start

    def infer_once_with_feature(
        self,
        element: dict[str, Any],
        *,
        return_feature: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
        """Infer one chunk and optionally request a server-side B feature."""

        payload = dict(element)
        payload["return_feature"] = bool(return_feature)
        result = self._client.infer(payload)
        if not isinstance(result, dict) or "actions" not in result:
            raise TypeError("Policy response must be an object containing actions")
        actions = np.asarray(result["actions"])
        feature = result.get("features")
        if feature is not None:
            feature = np.asarray(feature)
        metadata = result.get("feature_metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {"raw": metadata}
        return actions, feature, metadata

    def sample_action_chunks(self, element: dict[str, Any], k: int = 32) -> np.ndarray:
        """Call the frozen policy k times for the same policy input."""

        chunks = [self.infer_once(element) for _ in range(int(k))]
        return np.stack(chunks, axis=0)

    def sample_action_chunks_with_timing(
        self, element: dict[str, Any], k: int = 32
    ) -> tuple[np.ndarray, list[float]]:
        """Sample action chunks and return per-request wall-clock seconds."""

        chunks: list[np.ndarray] = []
        elapsed_seconds: list[float] = []
        for _ in range(int(k)):
            chunk, elapsed = self.infer_once_timed(element)
            chunks.append(chunk)
            elapsed_seconds.append(elapsed)
        return np.stack(chunks, axis=0), elapsed_seconds

    def sample_action_chunks_with_feature_timing(
        self, element: dict[str, Any], k: int = 32
    ) -> tuple[np.ndarray, list[float], np.ndarray, dict[str, Any]]:
        """Sample K chunks and capture B from the first server-side request.

        The B feature is requested only for the first sample. All later
        requests use the same fixed input and request actions only.
        """

        if int(k) < 1:
            raise ValueError("k must be positive")
        chunks: list[np.ndarray] = []
        elapsed_seconds: list[float] = []
        feature: np.ndarray | None = None
        feature_metadata: dict[str, Any] = {}
        for sample_index in range(int(k)):
            start = time.perf_counter()
            chunk, sample_feature, metadata = self.infer_once_with_feature(
                element,
                return_feature=sample_index == 0,
            )
            elapsed_seconds.append(time.perf_counter() - start)
            chunks.append(chunk)
            if sample_index == 0:
                if sample_feature is None:
                    raise RuntimeError(
                        "Server did not return B feature for the first sample"
                    )
                feature = np.asarray(sample_feature)
                feature_metadata = metadata
        assert feature is not None
        return np.stack(chunks, axis=0), elapsed_seconds, feature, feature_metadata
