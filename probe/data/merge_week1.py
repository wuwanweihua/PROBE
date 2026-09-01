"""Merge multiple week-1 dataset manifests into one index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def merge_manifests(
    dataset_dirs: list[str | Path],
    output_dir: str | Path,
    output_manifest: str = "records_merged.jsonl",
    input_manifest: str = "records.jsonl",
) -> dict[str, Any]:
    """Create a merged manifest without copying observation/action .npz files."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    merged_path = output_path / output_manifest
    report: dict[str, Any] = {
        "output_manifest": str(merged_path),
        "sources": [],
        "num_records": 0,
        "num_success": 0,
        "num_failure": 0,
    }

    with merged_path.open("w", encoding="utf-8") as out:
        for source_index, dataset_dir in enumerate(dataset_dirs):
            source_dir = Path(dataset_dir)
            manifest_path = source_dir / input_manifest
            source_count = 0
            if not manifest_path.exists():
                raise FileNotFoundError(f"Manifest not found: {manifest_path}")

            with manifest_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    record["source_dataset_dir"] = str(source_dir)
                    record["source_dataset_name"] = source_dir.name
                    record["source_index"] = source_index
                    record["source_record_id"] = record.get("record_id")
                    record["record_id"] = f"merged_{report['num_records']:08d}"

                    report["num_records"] += 1
                    source_count += 1
                    if record.get("final_success") is True:
                        report["num_success"] += 1
                    elif record.get("final_success") is False:
                        report["num_failure"] += 1
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")

            report["sources"].append(
                {
                    "dataset_dir": str(source_dir),
                    "manifest": str(manifest_path),
                    "num_records": source_count,
                }
            )

    total = int(report["num_records"])
    failures = int(report["num_failure"])
    report["failure_fraction"] = failures / total if total else 0.0
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-manifest", default="records_merged.jsonl")
    parser.add_argument("--input-manifest", default="records.jsonl")
    parser.add_argument("datasets", nargs="+", help="Dataset directories to merge.")
    args = parser.parse_args()

    report = merge_manifests(
        dataset_dirs=args.datasets,
        output_dir=args.output_dir,
        output_manifest=args.output_manifest,
        input_manifest=args.input_manifest,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
