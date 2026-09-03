"""Prompts for semantic instruction rewrites."""

from __future__ import annotations

import json
import re
from typing import Any


SYSTEM_PROMPT = """You rewrite robot manipulation instructions for benchmark evaluation.
Your rewrites must preserve the same task goal, objects, spatial relations, action order, and success condition.
You may use synonym substitution, natural word-order changes, concise rephrasing, and active/passive wording changes.
When explicitly requested, you may make logically necessary implicit manipulation steps explicit, such as opening a drawer before placing an object inside it.
Do not add new objects, remove required objects, change colors, change target locations, change order constraints, or make the instruction easier by changing the task.
Write instructions in a direct robot-command style that a vision-language-action policy can execute.
Prefer short commands with explicit object names and explicit receptacle names.
Avoid ambiguous pronouns such as it, this, that, them, or there.
If an object must be closed, name that object explicitly, for example "close the drawer" instead of "close it".
Do not include benchmark metadata such as view numbers, camera settings, seeds, or initstate ids.
Return only JSON that matches the requested schema."""


CONDITION_SYSTEM_PROMPT = """You create two controlled language conditions for a robot manipulation benchmark.
Both conditions must preserve the same task goal, objects, colors, receptacles, spatial relations, action order, and success condition.
The better condition should be clearer for a vision-language-action robot policy.
The worse condition should be less friendly for a vision-language-action robot policy while remaining semantically equivalent and executable by a human.
Do not change the target object, target receptacle, target location, quantity, color, or required final state.
Do not introduce a false goal, a distractor object, impossible actions, negation, or benchmark metadata.
Return only JSON that matches the requested schema."""


LIBERO_METADATA_RE = re.compile(
    r"\s+view\s+[-+]?\d+\s+[-+]?\d+\s+[-+]?\d+\s+[-+]?\d+\s+[-+]?\d+\s+initstate\s+\d+\s*$",
    re.IGNORECASE,
)


def clean_robot_instruction(instruction: str) -> str:
    """Remove LIBERO-Plus metadata suffixes from a natural-language command."""

    cleaned = LIBERO_METADATA_RE.sub("", str(instruction)).strip()
    return " ".join(cleaned.split())


def build_rewrite_prompt(
    *,
    original_instruction: str,
    rewrites_per_task: int,
    task_id: int | None = None,
    classification: dict[str, Any] | None = None,
    prompt_style: str = "strict_semantic",
) -> str:
    clean_instruction = clean_robot_instruction(original_instruction)
    payload: dict[str, Any] = {
        "task_id": task_id,
        "original_instruction": clean_instruction,
        "raw_benchmark_instruction": original_instruction,
        "num_rewrites": rewrites_per_task,
        "prompt_style": prompt_style,
        "allowed_rewrite_types": [
            "synonym_substitution",
            "word_order_change",
            "concise_rephrase",
            "active_passive_change",
            "natural_language_variant",
        ],
        "rewrite_requirements": _style_requirements(prompt_style),
    }
    if classification:
        payload["classification"] = classification
    return (
        "Create semantically equivalent instruction rewrites for this robot benchmark task.\n"
        "Keep each rewrite executable as a single instruction.\n"
        "Input:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def build_condition_pair_prompt(
    *,
    original_instruction: str,
    task_id: int | None = None,
    classification: dict[str, Any] | None = None,
) -> str:
    clean_instruction = clean_robot_instruction(original_instruction)
    payload: dict[str, Any] = {
        "task_id": task_id,
        "original_instruction": clean_instruction,
        "raw_benchmark_instruction": original_instruction,
        "conditions_to_create": ["better", "worse"],
        "better_instruction_requirements": [
            "Use a short direct robot command.",
            "Make the object, color, receptacle, and target location explicit.",
            "Use natural action order and avoid ambiguous pronouns.",
            "If a container must be closed, explicitly name the container to close.",
        ],
        "worse_instruction_requirements": [
            "Keep the exact same task semantics and final success condition.",
            "Make the wording less direct or less natural for a VLA policy.",
            "Allowed changes include passive voice, unusual but grammatical word order, extra descriptive phrasing, or clear but less common synonyms.",
            "Avoid making the instruction objectively wrong or impossible.",
        ],
        "forbidden_changes": [
            "Do not change objects, colors, receptacles, target locations, or action order.",
            "Do not add or remove required manipulation steps.",
            "Do not include view/initstate metadata.",
            "Do not use negation or intentionally contradictory language.",
        ],
    }
    if classification:
        payload["classification"] = classification
    return (
        "Generate one better and one worse instruction condition for the same LIBERO-Plus task.\n"
        "Both must remain semantically equivalent to the original task.\n"
        "Input:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _style_requirements(prompt_style: str) -> list[str]:
    common = [
        "Keep all object colors, object names, receptacles, and spatial relations.",
        "Name the thing being manipulated and the thing being opened or closed explicitly.",
        "Do not use pronouns like it, this, that, them, or there.",
        "Do not include view/initstate metadata.",
    ]
    if prompt_style == "explicit_steps":
        return [
            "Preserve the final task goal and required action order.",
            "Make necessary implicit manipulation steps explicit only when they are logically required by the original task.",
            "For container tasks, mention opening the container before placing an object inside, and closing it afterward when closing is required.",
            "Use two or three short imperative clauses or sentences.",
            *common,
        ]
    if prompt_style == "perception_clear":
        return [
            "Preserve the exact task semantics and action order.",
            "Use simple words that make the target object and target location easy to identify visually.",
            "Prefer 'find/locate the target object' only as a perception cue, without changing the manipulation goal.",
            "Use one or two short imperative sentences.",
            *common,
        ]
    return [
        "Preserve the exact task semantics and action order.",
        "Use one clear imperative sentence.",
        *common,
    ]
