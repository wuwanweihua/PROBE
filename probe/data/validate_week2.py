"""Validate a Week 2 non-executing probe dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def validate_dataset(
    dataset_dir: str | Path,
    expected_k: int = 4,
    expected_conditions: tuple[str, ...] = ("original", "better", "worse"),
) -> tuple[bool, dict[str, Any]]:
    root = Path(dataset_dir)
    manifest_path = root / "manifest.jsonl"
    report: dict[str, Any] = {"dataset_dir": str(root), "errors": [], "warnings": []}
    if not manifest_path.exists():
        report["errors"].append(f"Manifest not found: {manifest_path}")
        return False, report

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                report["errors"].append(f"Line {line_number}: invalid JSON: {exc}")
                continue
            record_id = str(record.get("record_id", ""))
            if not record_id or record_id in seen_ids:
                report["errors"].append(f"Line {line_number}: duplicate or empty record_id")
            seen_ids.add(record_id)
            records.append(record)

    by_base: dict[str, set[str]] = {}
    shapes: set[tuple[int, ...]] = set()
    for index, record in enumerate(records, start=1):
        if record.get("non_executing") is not True:
            report["errors"].append(f"Record {index}: non_executing is not true")
        if record.get("probe_rng_control") != "server_default_internal_split":
            report["warnings"].append(
                f"Record {index}: unexpected probe_rng_control value"
            )
        if int(record.get("probe_sample_count", -1)) != expected_k:
            report["errors"].append(f"Record {index}: expected K={expected_k}")
        condition = str(record.get("condition_id", ""))
        base_id = str(record.get("base_id", ""))
        by_base.setdefault(base_id, set()).add(condition)

        action_path = root / str(record.get("action_samples_path", ""))
        observation_path = root / str(record.get("observation_path", ""))
        if not action_path.exists():
            report["errors"].append(f"Record {index}: missing {action_path}")
            continue
        if not observation_path.exists():
            report["errors"].append(f"Record {index}: missing {observation_path}")
        try:
            with np.load(action_path, allow_pickle=False) as data:
                samples = np.asarray(data["action_samples"])
        except Exception as exc:
            report["errors"].append(f"Record {index}: cannot read actions: {exc}")
            continue
        shapes.add(tuple(int(value) for value in samples.shape))
        if samples.ndim != 3 or samples.shape[0] != expected_k:
            report["errors"].append(
                f"Record {index}: invalid action shape {samples.shape}"
            )
        if not np.isfinite(samples).all():
            report["errors"].append(f"Record {index}: action samples are not finite")

    for base_id, conditions in by_base.items():
        missing = set(expected_conditions) - conditions
        if missing:
            report["errors"].append(f"{base_id}: missing conditions {sorted(missing)}")

    report.update(
        {
            "num_records": len(records),
            "num_base_states": len(by_base),
            "action_shapes": [list(shape) for shape in sorted(shapes)],
            "expected_k": expected_k,
            "expected_conditions": list(expected_conditions),
        }
    )
    return not report["errors"], report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected-k", type=int, default=4)
    parser.add_argument("--expected-conditions", default="original,better,worse")
    args = parser.parse_args()
    conditions = tuple(
        value.strip() for value in args.expected_conditions.split(",") if value.strip()
    )
    ok, report = validate_dataset(args.dataset, args.expected_k, conditions)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
