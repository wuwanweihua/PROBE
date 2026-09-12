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
