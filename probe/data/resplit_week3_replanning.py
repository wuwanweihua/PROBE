"""Refilter and rebalance a replanning-level Week 3 dataset.

Why this exists
---------------
The replanning dataset labels every row of a task with that task's 12-retry
success rate per condition, so a task is only usable as a comparison if one
condition is *clearly* better than the others.  With 12 retries a gap of one
success (1/12 = 8.3pp) is inside the noise of the measurement itself, and a gap
of zero means the task cannot separate the conditions at all.  Filtering on the
gap keeps only tasks whose ordering is credible, and raises the probability that
the observed best condition is the true best condition from ~0.43 to ~0.87.

Filtering removes whole tasks, so the surviving tasks must be dealt out to
train/validation/test again.  A plain random draw leaves the three splits with
different mixes of "which condition is best", which shifts the baseline between
splits and makes them incomparable.  This script stratifies by the best
condition so the three splits have a similar original/better/worse mix.

Feature files are never copied: paths are rewritten to absolute so the new
dataset references the source dataset's existing features.

Usage
-----
    python -m probe.data.resplit_week3_replanning \\
        --dataset-dir data/replan_dataset/merged_3suites \\
        --output-dir  data/replan_dataset/merged_gap2 \\
        --min-gap 2
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

LOGGER = logging.getLogger(__name__)
CONDITION_ORDER = ("better", "original", "worse")
SPLITS = ("train", "validation", "test")


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
    if not value:
        return value
    path = Path(value)
    return str(path) if path.is_absolute() else str((dataset_dir / path).resolve())


def task_success_counts(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, tuple[int, int]]]:
    counts: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
    for row in rows:
        counts[str(row["base_id"])][str(row["condition_id"])] = (
            int(row["eval_successes"]),
            int(row["eval_trials"]),
        )
    return dict(counts)


def gap_and_best(
    per_condition: dict[str, tuple[int, int]],
) -> tuple[int, str | None]:
    """Return (gap, unique_best_condition_or_None)."""

    successes = {c: per_condition[c][0] for c in CONDITION_ORDER}
    ranked = sorted(successes.values(), reverse=True)
    gap = ranked[0] - ranked[1]
    best = max(successes, key=lambda c: successes[c])
    tied = sum(1 for c in CONDITION_ORDER if successes[c] == ranked[0]) > 1
    return gap, (None if tied else best)


def all_conditions_boundary(per_condition: dict[str, tuple[int, int]]) -> bool:
    return all(0 < k < n for k, n in (per_condition[c] for c in CONDITION_ORDER))


def select_tasks(
    per_task: dict[str, dict[str, tuple[int, int]]],
    min_gap: int,
    require_all_boundary: bool,
) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    dropped: list[str] = []
    for task in sorted(per_task):
        condition_counts = per_task[task]
        if set(condition_counts) != set(CONDITION_ORDER):
            dropped.append(task)
            continue
        gap, best = gap_and_best(condition_counts)
        if gap < min_gap or best is None:
            dropped.append(task)
            continue
        if require_all_boundary and not all_conditions_boundary(condition_counts):
            dropped.append(task)
            continue
        kept.append(task)
    return kept, dropped


def balanced_split(
    kept: list[str],
    per_task: dict[str, dict[str, tuple[int, int]]],
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    tries: int = 4000,
) -> dict[str, list[str]]:
    """Assign whole tasks to splits, keeping the best-condition mix similar.

    Tasks are grouped by which condition wins, then each group is dealt out in
    the target proportions.  Among random deals the one with the smallest spread
    in per-split condition shares wins, which keeps the baseline comparable
    across splits.
    """

    fractions = (train_fraction, validation_fraction,
                 1.0 - train_fraction - validation_fraction)
    strata: dict[str, list[str]] = defaultdict(list)
    for task in kept:
        _, best = gap_and_best(per_task[task])
        strata[best].append(task)

    total = len(kept)
    target = {
        split: round(total * fractions[index])
        for index, split in enumerate(SPLITS)
    }
    target["test"] = total - target["train"] - target["validation"]

    best_assign: dict[str, str] | None = None
    best_cost: float | None = None
    for attempt in range(tries):
        rng = random.Random(seed + attempt)
        assign: dict[str, str] = {}
        for label in sorted(strata):
            members = sorted(strata[label])
            rng.shuffle(members)
            count = len(members)
            raw = {s: count * fractions[i] for i, s in enumerate(SPLITS)}
            alloc = {s: int(raw[s]) for s in SPLITS}
            remainder = count - sum(alloc.values())
            for split in sorted(SPLITS, key=lambda s: -(raw[s] - alloc[s]))[:remainder]:
                alloc[split] += 1
            index = 0
            for split in SPLITS:
                for task in members[index:index + alloc[split]]:
                    assign[task] = split
                index += alloc[split]

        shares = {
            split: Counter(
                gap_and_best(per_task[t])[1] for t in kept if assign[t] == split
            )
            for split in SPLITS
        }
        cost = 0.0
        for label in CONDITION_ORDER:
            values = [
                shares[split][label] / (sum(shares[split].values()) or 1)
                for split in SPLITS
            ]
            cost += max(values) - min(values)
        count_penalty = sum(
            abs(sum(1 for t in kept if assign[t] == split) - target[split])
            for split in SPLITS
        )
        cost += 0.15 * count_penalty
        if best_cost is None or cost < best_cost:
            best_cost, best_assign = cost, assign

    assert best_assign is not None
    out: dict[str, list[str]] = {split: [] for split in SPLITS}
    for task, split in best_assign.items():
        out[split].append(task)
    for split in SPLITS:
        out[split].sort()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--min-gap", type=int, default=2,
        help="keep a task only if best minus second-best successes >= this",
    )
    parser.add_argument(
        "--require-all-boundary", action="store_true",
        help="also require 0 < successes < trials for all three conditions",
    )
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = dataset_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")

    rows = read_jsonl(manifest_path)
    per_task = task_success_counts(rows)
    kept, dropped = select_tasks(per_task, args.min_gap, args.require_all_boundary)
    if not kept:
        raise SystemExit("no task survived the filter; relax --min-gap")

    LOGGER.info(
        "tasks: kept=%d dropped=%d (min_gap=%d, require_all_boundary=%s)",
        len(kept), len(dropped), args.min_gap, args.require_all_boundary,
    )
    if len(kept) < 3:
        raise SystemExit("fewer than 3 tasks survive; cannot form three splits")

    assignment = balanced_split(
        kept, per_task, args.seed, args.train_fraction, args.validation_fraction
    )

    kept_set = set(kept)
    task_split = {
        task: split for split, tasks in assignment.items() for task in tasks
    }
    kept_rows: list[dict[str, Any]] = []
    for row in rows:
        base_id = str(row["base_id"])
        if base_id not in kept_set:
            continue
        merged = dict(row)
        merged["source_dataset_dir"] = str(dataset_dir)
        for key in ("obs_path", "a_feature_path", "b_feature_path"):
            merged[key] = absolutize(row.get(key), dataset_dir)
        merged["s_feature_paths"] = {
            k: absolutize(v, dataset_dir)
            for k, v in (row.get("s_feature_paths") or {}).items()
        }
        merged["split"] = task_split[base_id]
        kept_rows.append(merged)

    # Keep the trainer's layout contract: each group is a contiguous block of
    # the three conditions, in CONDITION_ORDER.
    per_group: dict[str, list[str]] = defaultdict(list)
    for row in kept_rows:
        per_group[str(row["group_id"])].append(str(row["condition_id"]))
    for group_id, conditions in per_group.items():
        if conditions != list(CONDITION_ORDER):
            raise ValueError(
                f"group {group_id}: expected {list(CONDITION_ORDER)}, got {conditions}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "manifest.jsonl", kept_rows)

    record_ids = {split: [] for split in SPLITS}
    group_ids = {split: [] for split in SPLITS}
    for row in kept_rows:
        split = task_split[str(row["base_id"])]
        record_ids[split].append(str(row["record_id"]))
    for group_id in per_group:
        task = group_id.split("::")[0]
        group_ids[task_split[task]].append(group_id)

    (output_dir / "splits.json").write_text(
        json.dumps(
            {
                "split_seed": args.seed,
                "fractions": {
                    "train": args.train_fraction,
                    "validation": args.validation_fraction,
                    "test": 1.0 - args.train_fraction - args.validation_fraction,
                },
                "group_key": "base_id (task-level; stratified by best condition)",
                "filter": {
                    "min_gap": args.min_gap,
                    "require_all_boundary": args.require_all_boundary,
                    "kept_tasks": len(kept),
                    "dropped_tasks": len(dropped),
                },
                "groups": {s: sorted(group_ids[s]) for s in SPLITS},
                "record_ids": record_ids,
            },
            indent=2, ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    label_rows = []
    for task in kept:
        for condition in CONDITION_ORDER:
            successes, trials = per_task[task][condition]
            label_rows.append(
                {
                    "group_id": task,
                    "condition_id": condition,
                    "base_id": task,
                    "eval_successes": successes,
                    "eval_trials": trials,
                    "eval_success_rate": successes / trials,
                }
            )
    with (output_dir / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(label_rows[0]))
        writer.writeheader()
        writer.writerows(label_rows)

    summary: dict[str, Any] = {
        "dataset_type": "week3_replanning_refiltered",
        "source_dataset_dir": str(dataset_dir),
        "min_gap": args.min_gap,
        "require_all_boundary": args.require_all_boundary,
        "num_tasks": len(kept),
        "num_dropped_tasks": len(dropped),
        "num_records": len(kept_rows),
        "num_groups": len(per_group),
        "split_task_counts": {s: len(assignment[s]) for s in SPLITS},
        "split_group_counts": {s: len(group_ids[s]) for s in SPLITS},
        "split_record_counts": {s: len(record_ids[s]) for s in SPLITS},
        "best_condition_by_split": {
            split: dict(
                Counter(gap_and_best(per_task[t])[1] for t in assignment[split])
            )
            for split in SPLITS
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    LOGGER.info(
        "records=%d groups=%d tasks=%d", len(kept_rows), len(per_group), len(kept)
    )
    LOGGER.info("split tasks=%s", summary["split_task_counts"])
    LOGGER.info("best condition by split:")
    for split in SPLITS:
        n = len(assignment[split]) or 1
        shares = summary["best_condition_by_split"][split]
        LOGGER.info(
            "  %-11s orig=%.2f better=%.2f worse=%.2f",
            split,
            shares.get("original", 0) / n,
            shares.get("better", 0) / n,
            shares.get("worse", 0) / n,
        )
    LOGGER.info("wrote %s", output_dir)


if __name__ == "__main__":
    main()
