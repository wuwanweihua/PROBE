"""Local Unix-socket transport for the observation-aware Qwen rewriter."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import socket
import stat
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from probe.instruction_rewrite.online_vlm import QwenVLMRewriteClient, RewriteResult


LOGGER = logging.getLogger(__name__)


class QwenSocketRewriteClient:
    """Runtime-side client for a Qwen rewriter on a Unix domain socket."""

    def __init__(self, socket_path: str, *, timeout: float = 300.0) -> None:
        self.socket_path = str(socket_path)
        self.timeout = float(timeout)

    def rewrite_from_policy_element(
        self,
        *,
        policy_element: dict[str, Any],
        source_instruction: str,
        task_id: int | None = None,
        task_name: str | None = None,
        condition_type: str | None = None,
        episode_id: str | None = None,
        step_idx: int | None = None,
        replan_idx: int | None = None,
    ) -> RewriteResult:
        request = {
            "source_instruction": str(source_instruction),
            "task_id": task_id,
            "task_name": task_name,
            "condition_type": condition_type,
            "episode_id": episode_id,
            "step_idx": step_idx,
            "replan_idx": replan_idx,
            "state": np.asarray(policy_element["observation/state"]).tolist(),
            "agent_image": _encode_image(policy_element["observation/image"]),
            "wrist_image": _encode_image(policy_element["observation/wrist_image"]),
        }
        try:
            response = self._request(request)
            instruction = str(response.get("instruction") or source_instruction)
            return RewriteResult(
                instruction=instruction,
                raw_text=str(response.get("raw_text") or ""),
                model=str(response.get("model") or "qwen-socket"),
                source_instruction=str(source_instruction),
                error=str(response["error"]) if response.get("error") else None,
                request_id=str(response["request_id"]) if response.get("request_id") else None,
                elapsed_ms=float(response["elapsed_ms"]) if response.get("elapsed_ms") is not None else None,
            )
        except Exception as exc:
            LOGGER.warning("Qwen socket rewrite failed: %s", exc)
            return RewriteResult(
                instruction=str(source_instruction),
                source_instruction=str(source_instruction),
                model="qwen-socket",
                error=str(exc),
            )

    def _request(self, request: dict[str, Any]) -> dict[str, Any]:
        payload = (json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
            channel.settimeout(self.timeout)
            channel.connect(self.socket_path)
            channel.sendall(payload)
            with channel.makefile("rb") as stream:
                response_line = stream.readline()
        if not response_line:
            raise RuntimeError("Qwen socket closed without a response")
        response = json.loads(response_line.decode("utf-8"))
        if not isinstance(response, dict):
            raise RuntimeError("Qwen socket returned a non-object response")
        return response


class QwenSocketRewriteServer:
    """Server-side Qwen process. The model is loaded once and never exposed on TCP."""

    def __init__(
        self,
        *,
        model_dir: str,
        socket_path: str,
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        top_p: float = 0.9,
        timeout: float = 300.0,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.rewriter = QwenVLMRewriteClient(
            model_dir,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        self.timeout = float(timeout)

    def serve_forever(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(self.socket_path))
            os.chmod(self.socket_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP)
            listener.listen(1)
            LOGGER.info("Qwen socket listening at %s", self.socket_path)
            try:
                while True:
                    connection, _ = listener.accept()
                    with connection:
                        connection.settimeout(self.timeout)
                        self._serve_connection(connection)
            finally:
                try:
                    self.socket_path.unlink()
                except FileNotFoundError:
                    pass

    def _serve_connection(self, connection: socket.socket) -> None:
        with connection.makefile("rb") as stream:
            request_line = stream.readline()
        if not request_line:
            return
        request: Any = {}
        try:
            request = json.loads(request_line.decode("utf-8"))
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            element = {
                "observation/image": _decode_image(request["agent_image"]),
                "observation/wrist_image": _decode_image(request["wrist_image"]),
                "observation/state": np.asarray(request["state"], dtype=np.float32),
            }
            started = time.perf_counter()
            result = self.rewriter.rewrite_from_policy_element(
                policy_element=element,
                source_instruction=str(request["source_instruction"]),
                task_id=_optional_int(request.get("task_id")),
                task_name=_optional_str(request.get("task_name")),
                condition_type=_optional_str(request.get("condition_type")),
                episode_id=_optional_str(request.get("episode_id")),
                step_idx=_optional_int(request.get("step_idx")),
                replan_idx=_optional_int(request.get("replan_idx")),
            )
            response = {
                "instruction": result.instruction,
                "raw_text": result.raw_text,
                "model": result.model,
                "source_instruction": result.source_instruction,
                "error": result.error,
                "request_id": result.request_id,
                "elapsed_ms": result.elapsed_ms,
                "server_elapsed_ms": (time.perf_counter() - started) * 1000.0,
            }
        except Exception as exc:  # pragma: no cover - model/runtime dependent
            LOGGER.exception("Qwen socket request failed")
            response = {
                "instruction": str(request.get("source_instruction", "")) if isinstance(request, dict) else "",
                "model": Path(self.rewriter.model_dir).name,
                "error": str(exc),
            }
        connection.sendall((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))


def _encode_image(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype != np.uint8:
        array = np.asarray(np.clip(array, 0, 255), dtype=np.uint8)
    buffer = BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode_image(value: str) -> Any:
    with BytesIO(base64.b64decode(value)) as buffer:
        return np.load(buffer, allow_pickle=False)


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _optional_str(value: Any) -> str | None:
    return str(value) if value is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--socket-path", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
    QwenSocketRewriteServer(
        model_dir=args.model_dir,
        socket_path=args.socket_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.timeout,
    ).serve_forever()


if __name__ == "__main__":
    main()
