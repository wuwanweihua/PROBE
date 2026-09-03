"""OpenAI Responses API wrapper for instruction rewrites."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from probe.instruction_rewrite.prompts import (
    CONDITION_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_condition_pair_prompt,
    build_rewrite_prompt,
)


DEFAULT_MODEL = "gpt-5.5"


@dataclass
class InstructionRewrite:
    instruction: str
    rewrite_type: str
    rationale: str


@dataclass
class InstructionConditionPair:
    better_instruction: str
    better_rationale: str
    worse_instruction: str
    worse_rationale: str
    worse_degradation_type: str


class OpenAIRewriteClient:
    """Tiny Responses API client that avoids adding an SDK dependency."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 90.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Put it in .env or export it before running.")
        self.model = model or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.timeout = float(timeout)

    def rewrite_instruction(
        self,
        *,
        original_instruction: str,
        rewrites_per_task: int = 5,
        task_id: int | None = None,
        classification: dict[str, Any] | None = None,
        prompt_style: str = "strict_semantic",
        max_output_tokens: int = 1200,
    ) -> list[InstructionRewrite]:
        prompt = build_rewrite_prompt(
            original_instruction=original_instruction,
            rewrites_per_task=rewrites_per_task,
            task_id=task_id,
            classification=classification,
            prompt_style=prompt_style,
        )
        response = self._post_json(
            "/responses",
            {
                "model": self.model,
                "input": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "instruction_rewrite_response",
                        "strict": True,
                        "schema": _rewrite_schema(rewrites_per_task),
                    }
                },
                "max_output_tokens": max_output_tokens,
            },
        )
        text = extract_response_text(response)
        payload = json.loads(text)
        rewrites = payload.get("rewrites", [])
        if not isinstance(rewrites, list):
            raise ValueError(f"Unexpected rewrite payload: {payload!r}")
        return [
            InstructionRewrite(
                instruction=str(item["instruction"]).strip(),
                rewrite_type=str(item.get("rewrite_type") or "unknown"),
                rationale=str(item.get("rationale") or ""),
            )
            for item in rewrites
            if isinstance(item, dict) and str(item.get("instruction") or "").strip()
        ]

    def generate_condition_pair(
        self,
        *,
        original_instruction: str,
        task_id: int | None = None,
        classification: dict[str, Any] | None = None,
        max_output_tokens: int = 1000,
    ) -> InstructionConditionPair:
        prompt = build_condition_pair_prompt(
            original_instruction=original_instruction,
            task_id=task_id,
            classification=classification,
        )
        response = self._post_json(
            "/responses",
            {
                "model": self.model,
                "input": [
                    {"role": "system", "content": CONDITION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "instruction_condition_pair",
                        "strict": True,
                        "schema": _condition_pair_schema(),
                    }
                },
                "max_output_tokens": max_output_tokens,
            },
        )
        payload = json.loads(extract_response_text(response))
        return InstructionConditionPair(
            better_instruction=str(payload["better"]["instruction"]).strip(),
            better_rationale=str(payload["better"].get("rationale") or ""),
            worse_instruction=str(payload["worse"]["instruction"]).strip(),
            worse_rationale=str(payload["worse"].get("rationale") or ""),
            worse_degradation_type=str(payload["worse"].get("degradation_type") or "unknown"),
        )

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenAI API request failed with HTTP {exc.code} at {self.base_url}: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI API request failed: {exc}") from exc


def load_dotenv(path: str | Path = ".env", override: bool = True) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (override or key not in os.environ):
            os.environ[key] = value


def extract_response_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    pieces: list[str] = []
    for output in response.get("output", []):
        if not isinstance(output, dict):
            continue
        for content in output.get("content", []):
            if isinstance(content, dict):
                if isinstance(content.get("text"), str):
                    pieces.append(content["text"])
                elif isinstance(content.get("json"), (dict, list)):
                    pieces.append(json.dumps(content["json"], ensure_ascii=False))
    text = "\n".join(piece for piece in pieces if piece).strip()
    if not text:
        raise ValueError(f"No text output in OpenAI response: {response!r}")
    return text


def _rewrite_schema(rewrites_per_task: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "rewrites": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "instruction": {"type": "string"},
                        "rewrite_type": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["instruction", "rewrite_type", "rationale"],
                },
            }
        },
        "required": ["rewrites"],
    }


def _condition_pair_schema() -> dict[str, Any]:
    better_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "instruction": {"type": "string"},
            "rationale": {"type": "string"},
        },
        "required": ["instruction", "rationale"],
    }
    worse_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "instruction": {"type": "string"},
            "rationale": {"type": "string"},
            "degradation_type": {"type": "string"},
        },
        "required": ["instruction", "rationale", "degradation_type"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "better": better_schema,
            "worse": worse_schema,
        },
        "required": ["better", "worse"],
    }
