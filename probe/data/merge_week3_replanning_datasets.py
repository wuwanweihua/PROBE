"""Merge several replanning-level Week 3 datasets into one training set.

Why this exists
---------------
The replanning datasets are built per suite, and a single suite has too few
independent tasks for a stable comparison (robot 83, camera 58, sensor 28).  A
task-level split leaves only a handful of independent test units, so the metric
noise stays large even though the row count looks impressive.

Training on the union of suites raises the number of *independent tasks*, which
is the quantity that actually limits the noise.  Merging does not change how
many rows any one task contributes.

Feature files are never copied.  Paths are rewritten to absolute so the merged
manifest references each source dataset's existing feature directory; copying
would multiply the on-disk footprint for no benefit.

Split handling
--------------
Each source dataset already has a task-level split (no task straddles two
splits).  The merged split is the per-split union of the sources, so a task that
was test in its own suite stays test here.  Group ids and record ids must be
globally unique across sources; that is asserted rather than assumed.

Usage
-----
    python -m probe.data.merge_week3_replanning_datasets \\
        --dataset-dir data/replan_dataset/robot \\
        --dataset-dir data/replan_dataset/camera \\
        --dataset-dir data/replan_dataset/sensor \\
        --output-dir data/replan_dataset/merged_3suites
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

LOGGER = logging.getLogger(__name__)
SPLITS = ("train", "validation", "test")
CONDITION_ORDER = ("better", "original", "worse")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def absolutize(value: str | None, dataset_dir: Path) -> str | None:
    """Return an absolute feature path so the merged manifest needs no copying."""

    if not value:
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((dataset_dir / path).resolve())


def merge_dataset_dirs(dataset_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    merged_rows: list[dict[str, Any]] = []
    merged_group_splits: dict[str, dict[str, str]] = {s: {} for s in SPLITS}
    seen_records: set[str] = set()
    seen_groups: dict[str, str] = {}
    label_rows: list[dict[str, Any]] = []
    per_source: list[dict[str, Any]] = []

    for dataset_dir in dataset_dirs:
        manifest_path = dataset_dir / "manifest.jsonl"
        splits_path = dataset_dir / "splits.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")
        if not splits_path.exists():
            raise FileNotFoundError(f"splits not found: {splits_path}")

        rows = read_jsonl(manifest_path)
        splits = json.loads(splits_path.read_text(encoding="utf-8"))
        group_split = {
            str(group_id): split
            for split in SPLITS
            for group_id in splits["groups"].get(split, [])
        }

        for row in rows:
            record_id = str(row["record_id"])
            group_id = str(row["group_id"])
            if record_id in seen_records:
                raise ValueError(
                    f"duplicate record_id {record_id!r} between sources; "
                    "suite prefixes are required to merge safely"
                )
            if group_id in seen_groups and seen_groups[group_id] not in (None,):
                raise ValueError(f"duplicate group_id {group_id!r} between sources")
            split = group_split.get(group_id)
            if split is None:
                raise ValueError(f"{record_id}: group {group_id} missing from splits")
            seen_records.add(record_id)
            seen_groups[group_id] = split
            merged_group_splits[split][group_id] = split

            merged = dict(row)
            merged["source_dataset_dir"] = str(dataset_dir)
            merged["obs_path"] = absolutize(row.get("obs_path"), dataset_dir)
            merged["a_feature_path"] = absolutize(row.get("a_feature_path"), dataset_dir)
            merged["b_feature_path"] = absolutize(row.get("b_feature_path"), dataset_dir)
            s_paths = row.get("s_feature_paths") or {}
            merged["s_feature_paths"] = {
                key: absolutize(value, dataset_dir) for key, value in s_paths.items()
            }
            merged_rows.append(merged)

        labels_path = dataset_dir / "labels.csv"
        if labels_path.exists():
            with labels_path.open("r", encoding="utf-8-sig") as handle:
                label_rows.extend(list(csv.DictReader(handle)))

        per_source.append(
            {
                "dataset_dir": str(dataset_dir),
                "records": len(rows),
                "groups": len(group_split),
                "group_counts": {
                    split: sum(1 for s in group_split.values() if s == split)
                    for split in SPLITS
                },
            }
        )

    # A group's records must all carry the same split; assert rather than trust.
    group_seen_split: dict[str, set[str]] = defaultdict(set)
    for row in merged_rows:
        group_seen_split[str(row["group_id"])].add(seen_groups[str(row["group_id"])])
    bad = {g: s for g, s in group_seen_split.items() if len(s) != 1}
    if bad:
        raise ValueError(f"{len(bad)} group(s) span multiple splits, e.g. {list(bad)[:3]}")

    # The trainer folds records into contiguous condition blocks, so every group
    # must appear as exactly three consecutive rows in CONDITION_ORDER.
    per_group_rows: dict[str, list[str]] = defaultdict(list)
    for row in merged_rows:
        per_group_rows[str(row["group_id"])].append(str(row["condition_id"]))
    for group_id, conditions in per_group_rows.items():
        if conditions != list(CONDITION_ORDER):
            raise ValueError(
                f"group {group_id}: expected {list(CONDITION_ORDER)}, got {conditions}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "manifest.jsonl", merged_rows)

    record_ids = {split: [] for split in SPLITS}
    for row in merged_rows:
        record_ids[seen_groups[str(row["group_id"])]].append(str(row["record_id"]))

    split_payload = {
        "split_seed": None,
        "fractions": None,
        "group_key": (
            "merged from per-suite task-level splits; a task stays in the split "
            "its source dataset assigned"
        ),
        "groups": {
            split: sorted(merged_group_splits[split]) for split in SPLITS
        },
        "record_ids": record_ids,
        "sources": per_source,
    }
    (output_dir / "splits.json").write_text(
        json.dumps(split_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if label_rows:
        fieldnames = list(label_rows[0])
        with (output_dir / "labels.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(label_rows)

    summary = {
        "dataset_type": "week3_replanning_merged",
        "sources": per_source,
        "num_records": len(merged_rows),
        "num_groups": len(seen_groups),
        "num_tasks": len({str(row.get("base_id")) for row in merged_rows}),
        "split_group_counts": {
            split: len(split_payload["groups"][split]) for split in SPLITS
        },
        "split_record_counts": {
            split: len(record_ids[split]) for split in SPLITS
        },
        "split_task_counts": {
            split: len(
                {str(row.get("base_id")) for row in merged_rows
                 if seen_groups[str(row["group_id"])] == split}
            )
            for split in SPLITS
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, nargs="+", required=True,
        help="replanning dataset dirs to merge (each with manifest.jsonl + splits.json)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    dataset_dirs = [path.expanduser().resolve() for path in args.dataset_dir]
    output_dir = args.output_dir.expanduser().resolve()

    summary = merge_dataset_dirs(dataset_dirs, output_dir)

    LOGGER.info(
        "merged %d source(s): records=%d groups=%d tasks=%d",
        len(dataset_dirs),
        summary["num_records"],
        summary["num_groups"],
        summary["num_tasks"],
    )
    LOGGER.info(
        "split groups=%s records=%s tasks=%s",
        summary["split_group_counts"],
        summary["split_record_counts"],
        summary["split_task_counts"],
    )
    LOGGER.info("wrote merged dataset to %s", output_dir)


if __name__ == "__main__":
    main()
