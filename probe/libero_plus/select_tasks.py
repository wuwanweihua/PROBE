"""Select LIBERO-Plus task ids from task_classification.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def select_tasks(
    classification_json: str | Path,
    task_suite_name: str = "libero_10",
    category: str | None = None,
    min_difficulty: int | None = None,
    max_difficulty: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    data = json.loads(Path(classification_json).read_text(encoding="utf-8-sig"))
    entries = _suite_entries(data, task_suite_name)
    selected: list[dict[str, Any]] = []
    category_query = category.lower() if category else None
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        normalized = dict(entry)
        task_id = _task_id_from_entry(normalized, index)
        normalized["task_id"] = task_id
        entry_category = str(normalized.get("category") or normalized.get("type") or "")
        difficulty = _as_int(normalized.get("difficulty_level") or normalized.get("difficulty"))
        if category_query and category_query not in entry_category.lower():
            continue
        if min_difficulty is not None and (difficulty is None or difficulty < min_difficulty):
            continue
        if max_difficulty is not None and (difficulty is None or difficulty > max_difficulty):
            continue
        selected.append(normalized)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def _suite_entries(data: Any, task_suite_name: str) -> list[Any]:
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        raise ValueError("classification JSON must be a list or dict")
    for key in (task_suite_name, task_suite_name.upper(), task_suite_name.replace("_", "-"), "tasks"):
        value = data.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return list(value.values())
    if all(isinstance(value, dict) for value in data.values()):
        return list(data.values())
    raise ValueError(f"Cannot find task entries for suite {task_suite_name!r}")


def _task_id_from_entry(entry: dict[str, Any], fallback_index: int) -> int:
    raw_id = _as_int(entry.get("task_id"))
    if raw_id is not None:
        return raw_id
    raw_id = _as_int(entry.get("id"))
    if raw_id is not None:
        return raw_id - 1 if raw_id > 0 else raw_id
    return fallback_index


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-json", required=True)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--category")
    parser.add_argument("--min-difficulty", type=int)
    parser.add_argument("--max-difficulty", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--format", choices=["comma", "lines", "json"], default="comma")
    parser.add_argument("--output")
    args = parser.parse_args()

    selected = select_tasks(
        classification_json=args.classification_json,
        task_suite_name=args.task_suite_name,
        category=args.category,
        min_difficulty=args.min_difficulty,
        max_difficulty=args.max_difficulty,
        limit=args.limit,
    )
    if args.format == "json":
        text = json.dumps(
            {
                "classification_json": args.classification_json,
                "task_suite_name": args.task_suite_name,
                "category": args.category,
                "min_difficulty": args.min_difficulty,
                "max_difficulty": args.max_difficulty,
                "num_selected": len(selected),
                "task_ids": [item["task_id"] for item in selected],
                "tasks": selected,
            },
            indent=2,
            ensure_ascii=False,
        )
    elif args.format == "lines":
        text = "\n".join(str(item["task_id"]) for item in selected)
    else:
        text = ",".join(str(item["task_id"]) for item in selected)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
