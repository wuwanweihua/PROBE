"""Validate a week-1 PROBE dataset manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from probe.data.record_schema import missing_required_fields


def validate_dataset(
    dataset_dir: str | Path,
    manifest_name: str = "records.jsonl",
    min_records: int = 500,
    min_failure_fraction: float = 0.2,
    expected_k: int | None = 32,
) -> tuple[bool, dict[str, Any]]:
    dataset_path = Path(dataset_dir)
    manifest_path = dataset_path / manifest_name
    report: dict[str, Any] = {
        "dataset_dir": str(dataset_path),
        "manifest": str(manifest_path),
        "errors": [],
        "warnings": [],
    }

    if not manifest_path.exists():
        report["errors"].append(f"Manifest not found: {manifest_path}")
        return False, report

    records = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                report["errors"].append(f"Line {line_no}: invalid JSON: {exc}")
                continue
            missing = missing_required_fields(payload)
            if missing:
                report["errors"].append(f"Line {line_no}: missing fields {sorted(missing)}")
            records.append(payload)

    successes = sum(1 for item in records if item.get("final_success") is True)
    failures = sum(1 for item in records if item.get("final_success") is False)
    total = len(records)
    failure_fraction = failures / total if total else 0.0

    report.update(
        {
            "num_records": total,
            "num_success": successes,
            "num_failure": failures,
            "failure_fraction": failure_fraction,
        }
    )

    if total < min_records:
        report["errors"].append(f"Need at least {min_records} records, found {total}.")
    if min_failure_fraction > 0.0 and (successes == 0 or failures == 0):
        report["errors"].append("Need both success and failure records.")
    if failure_fraction < min_failure_fraction:
        report["errors"].append(
            f"Failure fraction must be >= {min_failure_fraction:.2f}, found {failure_fraction:.3f}."
        )

    _validate_paths_and_actions(dataset_path, records, expected_k, report)
    return len(report["errors"]) == 0, report


def _validate_paths_and_actions(
    dataset_path: Path,
    records: list[dict[str, Any]],
    expected_k: int | None,
    report: dict[str, Any],
) -> None:
    try:
        import numpy as np
    except ImportError:
        report["warnings"].append("NumPy not installed; skipped action .npz validation.")
        return

    for idx, record in enumerate(records[:], start=1):
        record_dataset_path = Path(str(record.get("source_dataset_dir") or dataset_path))
        obs_path = record_dataset_path / str(record.get("obs_path", ""))
        actions_path = record_dataset_path / str(record.get("action_samples_path", ""))
        if not obs_path.exists():
            report["errors"].append(f"Record {idx}: missing obs file {obs_path}")
        if not actions_path.exists():
            report["errors"].append(f"Record {idx}: missing action file {actions_path}")
            continue
        try:
            with np.load(actions_path, allow_pickle=False) as data:
                samples = data["action_samples"]
                selected = int(data["selected_sample_index"])
        except Exception as exc:
            report["errors"].append(f"Record {idx}: cannot read action file {actions_path}: {exc}")
            continue
        if expected_k is not None and samples.shape[0] != expected_k:
            report["errors"].append(
                f"Record {idx}: expected K={expected_k}, found shape {samples.shape}."
            )
        if selected < 0 or selected >= samples.shape[0]:
            report["errors"].append(
                f"Record {idx}: selected_sample_index {selected} outside K={samples.shape[0]}."
            )
        if bool(np.isnan(samples).any()):
            report["errors"].append(f"Record {idx}: action samples contain NaN.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset root directory.")
    parser.add_argument("--manifest-name", default="records.jsonl")
    parser.add_argument("--min-records", type=int, default=500)
    parser.add_argument("--min-failure-fraction", type=float, default=0.2)
    parser.add_argument("--expected-k", type=int, default=32)
    args = parser.parse_args()

    ok, report = validate_dataset(
        dataset_dir=args.dataset,
        manifest_name=args.manifest_name,
        min_records=args.min_records,
        min_failure_fraction=args.min_failure_fraction,
        expected_k=args.expected_k,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
