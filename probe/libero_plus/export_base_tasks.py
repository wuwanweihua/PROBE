"""Export LIBERO-Plus base tasks for original/better/worse condition generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from probe.config import parse_task_order
from probe.envs.libero_runner import get_task_suite
from probe.instruction_rewrite.prompts import clean_robot_instruction
from probe.libero_plus.select_tasks import select_tasks


def export_base_tasks(args: argparse.Namespace) -> dict[str, Any]:
    task_suite = get_task_suite(args.task_suite_name)
    task_ids = _resolve_task_ids(args, task_suite.n_tasks)
    classification = _load_classification(args.classification_json)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for task_id in task_ids:
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            classification_item = _lookup_classification(classification, args.task_suite_name, task_id)
            raw_instruction = str(task.language)
            payload = {
                "base_id": f"libero_plus_{args.task_suite_name}_task{task_id:04d}",
                "dataset": "LIBERO-Plus",
                "task_suite": args.task_suite_name,
                "task_id": int(task_id),
                "task_name": raw_instruction,
                "raw_benchmark_instruction": raw_instruction,
                "original_instruction": clean_robot_instruction(raw_instruction),
                "problem_folder": getattr(task, "problem_folder", None),
                "bddl_file": getattr(task, "bddl_file", None),
                "initial_state_count": len(initial_states),
                "classification": classification_item,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
    return {
        "output": str(output_path),
        "task_suite_name": args.task_suite_name,
        "num_exported": written,
        "task_ids": task_ids,
    }


def _resolve_task_ids(args: argparse.Namespace, num_tasks: int) -> list[int]:
    if args.category or args.min_difficulty is not None or args.max_difficulty is not None:
        if not args.classification_json:
            raise ValueError("--classification-json is required when filtering by category or difficulty.")
        selected = select_tasks(
            classification_json=args.classification_json,
            task_suite_name=args.task_suite_name,
            category=args.category,
            min_difficulty=args.min_difficulty,
            max_difficulty=args.max_difficulty,
            limit=args.limit,
        )
        return [int(item["task_id"]) for item in selected]
    task_ids = parse_task_order(args.task_order, num_tasks)
    if args.limit is not None:
        task_ids = task_ids[: args.limit]
    return task_ids


def _load_classification(path: str | None) -> Any:
    if not path:
        return None
    classification_path = Path(path)
    if not classification_path.exists():
        return None
    return json.loads(classification_path.read_text(encoding="utf-8-sig"))


def _lookup_classification(data: Any, suite: str, task_id: int) -> dict[str, Any] | None:
    if data is None:
        return None
    candidates: list[Any] = []
    if isinstance(data, dict):
        for key in (suite, suite.upper(), suite.replace("_", "-"), "tasks"):
            if key in data:
                candidates.append(data[key])
        candidates.append(data)
    else:
        candidates.append(data)
    for candidate in candidates:
        value = _lookup_task(candidate, task_id)
        if isinstance(value, dict):
            return value
    return None


def _lookup_task(candidate: Any, task_id: int) -> Any:
    if isinstance(candidate, list):
        if 0 <= task_id < len(candidate):
            return candidate[task_id]
        return None
    if isinstance(candidate, dict):
        for key in (str(task_id), str(task_id + 1), f"task{task_id:03d}", f"task{task_id:04d}"):
            if key in candidate:
                return candidate[key]
        for value in candidate.values():
            if isinstance(value, dict) and int(value.get("task_id", -1)) in (task_id, task_id + 1):
                return value
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--task-order", default="all")
    parser.add_argument("--classification-json")
    parser.add_argument("--category")
    parser.add_argument("--min-difficulty", type=int)
    parser.add_argument("--max-difficulty", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    report = export_base_tasks(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
