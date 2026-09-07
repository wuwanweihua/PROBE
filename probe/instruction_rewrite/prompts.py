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


ONLINE_REWRITE_SYSTEM_PROMPT = """You are a conservative robot-instruction editor.
The source instruction is authoritative and describes the only task that may be performed.
Keep the overall task, every object identity, color, quantity, receptacle, spatial relation,
action, action order, and final success condition unchanged.

Use the observation only to choose one short-term goal for the next few actions.
Do not identify, rename, or substitute objects from visual appearance. If the image seems to
conflict with the source instruction, trust the source instruction.
The short-term goal must be a necessary subgoal of the source instruction and must not invent
objects, locations, actions, or completion claims. If you are uncertain, set uncertain to true.

Return exactly one JSON object with this schema:
{"short_term_goal":"...","source_spans":["..."],"uncertain":false}

Every source_span must be copied verbatim from the source instruction. Keep the short-term goal
concise and atomic. Do not include explanations, markdown, camera names, view ids, seeds,
initstate ids, or hidden reasoning."""


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


def build_online_rewrite_prompt(
    *,
    source_instruction: str,
    task_id: int | None = None,
    task_name: str | None = None,
    condition_type: str | None = None,
    episode_id: str | None = None,
    step_idx: int | None = None,
    replan_idx: int | None = None,
    state_summary: dict[str, Any] | None = None,
) -> str:
    clean_instruction = clean_robot_instruction(source_instruction)
    payload: dict[str, Any] = {
        "task_id": task_id,
        "task_name": task_name,
        "condition_type": condition_type,
        "episode_id": episode_id,
        "step_idx": step_idx,
        "replan_idx": replan_idx,
        "source_instruction": clean_instruction,
        "raw_benchmark_instruction": source_instruction,
        "rewrite_goal": (
            "Keep the source instruction as the overall task and add one conservative, "
            "observation-grounded short-term goal for the next few actions."
        ),
        "requirements": [
            "The source instruction is authoritative; never rename or substitute an object.",
            "Keep the overall task, object identities, colors, quantities, locations, actions, order, and final state unchanged.",
            "Choose exactly one necessary atomic subgoal for the next few actions.",
            "Copy important object and location phrases into source_spans exactly as they appear in the source.",
            "Do not identify objects from the image when the source already names them.",
            "If uncertain or if no safe subgoal can be selected, set uncertain=true and use an empty short_term_goal.",
            "Return exactly one JSON object and no explanation.",
        ],
        "output_schema": {
            "short_term_goal": "string",
            "source_spans": "array of exact substrings copied from source_instruction",
            "uncertain": "boolean",
        },
    }
    if state_summary is not None:
        payload["robot_state_summary"] = state_summary
    return (
        "Rewrite the robot instruction using the current observation as context.\n"
        "The current observation images are attached separately to this request.\n"
        "Return only the JSON object described by output_schema.\n"
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
