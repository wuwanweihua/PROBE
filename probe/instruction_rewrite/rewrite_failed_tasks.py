"""Rewrite failed LIBERO-Plus instructions with GPT."""

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
from probe.instruction_rewrite.prompts import build_rewrite_prompt


def rewrite_failed_tasks(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv(args.env_file)
    records = _load_failed_records(Path(args.input))
    jobs = _dedupe_records(records, args.dedupe)
    if args.limit is not None:
        jobs = jobs[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_keys(output_path) if args.resume else set()
    pending = [job for job in jobs if _job_key(job) not in completed]

    if args.dry_run:
        examples = [
            {
                "task_id": job.get("task_id"),
                "instruction": job.get("instruction"),
                "prompt": build_rewrite_prompt(
                    original_instruction=str(job.get("instruction") or job.get("task_name") or ""),
                    rewrites_per_task=args.rewrites_per_task,
                    task_id=_as_int(job.get("task_id")),
                    classification=job.get("classification") if isinstance(job.get("classification"), dict) else None,
                ),
            }
            for job in pending[: max(1, args.preview)]
        ]
        return {
            "input": args.input,
            "output": args.output,
            "dry_run": True,
            "num_failed_records": len(records),
            "num_unique_jobs": len(jobs),
            "num_pending": len(pending),
            "examples": examples,
        }

    client = OpenAIRewriteClient(model=args.model, timeout=args.timeout)
    written = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for index, job in enumerate(pending, start=1):
            instruction = str(job.get("instruction") or job.get("task_name") or "").strip()
            if not instruction:
                continue
            rewrites = client.rewrite_instruction(
                original_instruction=instruction,
                rewrites_per_task=args.rewrites_per_task,
                task_id=_as_int(job.get("task_id")),
                classification=job.get("classification") if isinstance(job.get("classification"), dict) else None,
            )
            payload = {
                "rewrite_batch_id": f"task{int(job.get('task_id') or 0):04d}_{_short_hash(instruction)}",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model": client.model,
                "task_suite": job.get("task_suite"),
                "task_id": job.get("task_id"),
                "task_name": job.get("task_name"),
                "original_instruction": instruction,
                "source_episode_ids": job.get("source_episode_ids") or [job.get("episode_id")],
                "classification": job.get("classification"),
                "rewrites": [
                    {
                        "rewrite_index": rewrite_index,
                        "instruction": rewrite.instruction,
                        "rewrite_type": rewrite.rewrite_type,
                        "rationale": rewrite.rationale,
                    }
                    for rewrite_index, rewrite in enumerate(rewrites)
                ],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
            print(f"[{index}/{len(pending)}] rewrote task_id={job.get('task_id')} -> {len(rewrites)} variants")
            if args.sleep_seconds > 0 and index < len(pending):
                time.sleep(args.sleep_seconds)

    return {
        "input": args.input,
        "output": str(output_path),
        "model": args.model,
        "num_failed_records": len(records),
        "num_unique_jobs": len(jobs),
        "num_skipped_existing": len(jobs) - len(pending),
        "num_written": written,
    }


def _load_failed_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if record.get("final_success") is False:
                records.append(record)
    return records


def _dedupe_records(records: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if mode == "none":
        return list(records)
    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        if mode == "episode":
            key = str(record.get("episode_id"))
        else:
            key = _job_key(record)
        if key not in grouped:
            grouped[key] = dict(record)
            grouped[key]["source_episode_ids"] = []
        grouped[key]["source_episode_ids"].append(record.get("episode_id"))
    return list(grouped.values())


def _completed_keys(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    completed: set[str] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            completed.add(_job_key(payload))
    return completed


def _job_key(record: dict[str, Any]) -> str:
    task_id = record.get("task_id")
    instruction = record.get("instruction") or record.get("original_instruction") or record.get("task_name") or ""
    return f"{task_id}::{instruction}"


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
    parser.add_argument("--input", required=True, help="Failed episode JSONL produced by export_failed_episodes.")
    parser.add_argument("--output", required=True, help="Output rewrite JSONL path.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rewrites-per-task", type=int, default=5)
    parser.add_argument("--dedupe", choices=["task_instruction", "episode", "none"], default="task_instruction")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview", type=int, default=2)
    args = parser.parse_args()

    report = rewrite_failed_tasks(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
