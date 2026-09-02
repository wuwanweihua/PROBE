"""Prompts for semantic instruction rewrites."""

from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """You rewrite robot manipulation instructions for benchmark evaluation.
Your rewrites must preserve the same task goal, objects, spatial relations, and success condition.
Use semantic paraphrases only: synonym substitution, natural word-order changes, concise rephrasing, and active/passive wording changes are allowed.
Do not add new objects, remove required objects, change colors, change target locations, change order constraints, or make the instruction easier by changing the task.
Return only JSON that matches the requested schema."""


def build_rewrite_prompt(
    *,
    original_instruction: str,
    rewrites_per_task: int,
    task_id: int | None = None,
    classification: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "task_id": task_id,
        "original_instruction": original_instruction,
        "num_rewrites": rewrites_per_task,
        "allowed_rewrite_types": [
            "synonym_substitution",
            "word_order_change",
            "concise_rephrase",
            "active_passive_change",
            "natural_language_variant",
        ],
    }
    if classification:
        payload["classification"] = classification
    return (
        "Create semantically equivalent instruction rewrites for this robot benchmark task.\n"
        "Keep each rewrite executable as a single instruction.\n"
        "Input:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
