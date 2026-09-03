"""Generate original/better/worse instruction conditions with GPT."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from probe.instruction_rewrite.gpt_rewriter import (
    DEFAULT_MODEL,
    OpenAIRewriteClient,
    load_dotenv,
)
from probe.instruction_rewrite.prompts import build_condition_pair_prompt, clean_robot_instruction


def generate_condition_pairs(args: argparse.Namespace) -> dict[str, Any]:
    base_tasks = _read_jsonl(Path(args.input))
    if args.limit is not None:
        base_tasks = base_tasks[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_base_ids(output_path) if args.resume else set()
    pending = [task for task in base_tasks if str(task.get("base_id")) not in completed]

    if args.dry_run:
        examples = [
            {
                "base_id": task.get("base_id"),
                "task_id": task.get("task_id"),
                "prompt": build_condition_pair_prompt(
                    original_instruction=str(task.get("original_instruction") or task.get("task_name") or ""),
                    task_id=_as_int(task.get("task_id")),
                    classification=task.get("classification") if isinstance(task.get("classification"), dict) else None,
                ),
            }
            for task in pending[: max(1, args.preview)]
        ]
        return {
            "input": args.input,
            "output": args.output,
            "dry_run": True,
            "num_base_tasks": len(base_tasks),
            "num_pending": len(pending),
            "examples": examples,
        }

    load_dotenv(args.env_file)
    client = OpenAIRewriteClient(model=args.model, timeout=args.timeout)
    written = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for index, task in enumerate(pending, start=1):
            original = clean_robot_instruction(
                str(task.get("original_instruction") or task.get("task_name") or "")
            )
            if not original:
                continue
            pair = client.generate_condition_pair(
                original_instruction=original,
                task_id=_as_int(task.get("task_id")),
                classification=task.get("classification") if isinstance(task.get("classification"), dict) else None,
            )
            payload = {
                "condition_batch_id": f"{task.get('base_id')}_{_short_hash(original)}",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model": client.model,
                "base_url": client.base_url,
                "dataset": task.get("dataset", "LIBERO-Plus"),
                "task_suite": task.get("task_suite", args.task_suite_name),
                "base_id": task.get("base_id"),
                "task_id": task.get("task_id"),
                "task_name": task.get("task_name"),
                "raw_benchmark_instruction": task.get("raw_benchmark_instruction") or task.get("task_name"),
                "original_instruction": original,
                "problem_folder": task.get("problem_folder"),
                "bddl_file": task.get("bddl_file"),
                "initial_state_count": task.get("initial_state_count"),
                "classification": task.get("classification"),
                "conditions": [
                    {
                        "condition_id": "original",
                        "condition_type": "original",
                        "instruction": original,
                        "source": "benchmark",
                        "rationale": "Cleaned original benchmark instruction.",
                    },
                    {
                        "condition_id": "better",
                        "condition_type": "better",
                        "instruction": pair.better_instruction,
                        "source": "gpt",
                        "rationale": pair.better_rationale,
                    },
                    {
                        "condition_id": "worse",
                        "condition_type": "worse",
                        "instruction": pair.worse_instruction,
                        "source": "gpt",
                        "rationale": pair.worse_rationale,
                        "degradation_type": pair.worse_degradation_type,
                    },
                ],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
            print(f"[{index}/{len(pending)}] generated conditions for task_id={task.get('task_id')}")
            if args.sleep_seconds > 0 and index < len(pending):
                time.sleep(args.sleep_seconds)

    return {
        "input": args.input,
        "output": str(output_path),
        "model": client.model,
        "base_url": client.base_url,
        "num_base_tasks": len(base_tasks),
        "num_skipped_existing": len(base_tasks) - len(pending),
        "num_written": written,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return records


def _completed_base_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    completed: set[str] = set()
    for payload in _read_jsonl(output_path):
        if payload.get("base_id"):
            completed.add(str(payload["base_id"]))
    return completed


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _short_hash(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Base task JSONL from export_base_tasks.")
    parser.add_argument("--output", required=True, help="Output condition JSONL.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--model", default=None, help=f"Defaults to OPENAI_MODEL or {DEFAULT_MODEL}.")
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview", type=int, default=2)
    args = parser.parse_args()

    report = generate_condition_pairs(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
