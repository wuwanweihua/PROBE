"""Extract per-task boundary records from the statistical validation report."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


SECTION_RE = re.compile(r"^【(?P<suite>[^】]+)】\s*$")
ROW_RE = re.compile(
    r"^\|\s*(?P<task_id>\d+)\s*\|\s*"
    r"(?P<original>\d+)\s*/\s*(?P<trials_original>\d+)\s*\|\s*"
    r"(?P<better>\d+)\s*/\s*(?P<trials_better>\d+)\s*\|\s*"
    r"(?P<worse>\d+)\s*/\s*(?P<trials_worse>\d+)\s*\|\s*"
    r"(?P<total>\d+)\s*/\s*(?P<trials_total>\d+)\s*\|"
)


def _parse_rate(successes: int, trials: int) -> float:
    return successes / trials if trials else 0.0


def parse_report(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    suite: str | None = None

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        section_match = SECTION_RE.match(line.strip())
        if section_match:
            suite = section_match.group("suite").lower()
            continue

        row_match = ROW_RE.match(line.strip())
        if not row_match or suite is None:
            continue

        values = {key: int(value) for key, value in row_match.groupdict().items()}
        if not (0 < values["total"] < values["trials_total"]):
            continue

        records.append(
            {
                "suite": suite,
                "task_id": values["task_id"],
                "boundary_rule": "0 < total_successes < total_trials",
                "original_successes": values["original"],
                "original_trials": values["trials_original"],
                "original_success_rate": _parse_rate(
                    values["original"], values["trials_original"]
                ),
                "better_successes": values["better"],
                "better_trials": values["trials_better"],
                "better_success_rate": _parse_rate(
                    values["better"], values["trials_better"]
                ),
                "worse_successes": values["worse"],
                "worse_trials": values["trials_worse"],
                "worse_success_rate": _parse_rate(
                    values["worse"], values["trials_worse"]
                ),
                "total_successes": values["total"],
                "total_trials": values["trials_total"],
                "overall_success_rate": _parse_rate(
                    values["total"], values["trials_total"]
                ),
                "source_report": str(path),
                "source_line": line_number,
            }
        )

    records.sort(key=lambda item: (item["suite"], item["task_id"]))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "suite",
        "task_id",
        "original_successes",
        "better_successes",
        "worse_successes",
        "total_successes",
        "original_success_rate",
        "better_success_rate",
        "worse_success_rate",
        "overall_success_rate",
        "original_trials",
        "better_trials",
        "worse_trials",
        "total_trials",
        "boundary_rule",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: record[field] for field in fields} for record in records)


def write_markdown(path: Path, records: list[dict[str, Any]]) -> None:
    suites = sorted({str(record["suite"]) for record in records})
    lines = [
        "# Boundary Tasks from 8-Retry Results",
        "",
        "Selection rule: `0 < total_successes < 24`.",
        "",
        "| suite | boundary tasks |",
        "|---|---:|",
    ]
    for suite in suites:
        count = sum(record["suite"] == suite for record in records)
        lines.append(f"| {suite} | {count} |")
    lines.extend(
        [
            f"| **total** | **{len(records)}** |",
            "",
            "| suite | task_id | original | better | worse | total |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for record in records:
        lines.append(
            f"| {record['suite']} | {record['task_id']} | "
            f"{record['original_successes']}/{record['original_trials']} | "
            f"{record['better_successes']}/{record['better_trials']} | "
            f"{record['worse_successes']}/{record['worse_trials']} | "
            f"{record['total_successes']}/{record['total_trials']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    records = parse_report(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "boundary_tasks_8retry.jsonl", records)
    write_csv(args.output_dir / "boundary_tasks_8retry.csv", records)
    write_markdown(args.output_dir / "boundary_tasks_8retry.md", records)

    counts: dict[str, int] = {}
    for record in records:
        counts[record["suite"]] = counts.get(record["suite"], 0) + 1
    print(json.dumps({"total": len(records), "by_suite": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
