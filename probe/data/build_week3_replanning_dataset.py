"""Build a replanning-level Week 3 dataset from condition rollouts.

Motivation
----------
The original Week 3 dataset has one row per state-condition pair, so a task
contributes exactly three rows.  That makes the test split tiny (36-48 groups)
and the metric very noisy.  This script instead emits **one row per replanning
step**: every time the policy is queried during a rollout, that observation,
its K action chunks and its B feature become a training row.  A task therefore
contributes hundreds of rows, and the evaluation set grows by roughly two
orders of magnitude.

Label and alignment
-------------------
The success label is **not** the outcome of the newly collected rollout.  It is
the success rate measured by the earlier 12-retry sweep for the same
``task x condition`` (successes/trials), so every replanning row of one
condition carries that same rate.  This keeps labels comparable to the previous
dataset and much less noisy than a single boolean episode outcome.

Within a task the three conditions usually stop at different replanning counts
(a failing episode runs longer).  To keep the three conditions of a task
comparable, each group is truncated to the **minimum** replanning count over its
conditions, and rows beyond that are dropped.  A group is one task; its three
conditions stay together, so the group is never split across train/val/test.

Output layout (mirrors the Week 3 dataset so training code is reused)
--------------------------------------------------------------------
    <output-dir>/
      manifest.jsonl          one row per replanning step
      labels.csv              per-group condition success rates
      splits.json             task-level split (seed recorded)
      summary.json
      features/A/<record_id>.npz     (optional; built by a separate extractor)
      features/B/<record_id>.npz     copied from rollout features/
      features/S/K<k>/<record_id>.npz

Usage
-----
    python -m probe.data.build_week3_replanning_dataset \\
        --rollout-dir  data/rollout_replan/object \\
        --labels       data/week3_12retry_7suite_final/labels/all_7suites_12retry_labels.jsonl \\
        --suite        object \\
        --output-dir   data/week3_replan_7suite/object \\
        --s-prefixes   4 8 16 32 \\
        --split-seed   20260916 \\
        --train-fraction 0.7 --validation-fraction 0.15
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

LOGGER = logging.getLogger(__name__)

# Emission order of the three conditions inside one group.  This must match
# probe.data.train_week3_predictors.CONDITION_ORDER exactly, because the trainer
# folds records with a plain reshape and asserts this order.
CONDITION_ORDER = ("better", "original", "worse")

S_FEATURE_NAMES = (
    "mean_disagreement",
    "max_disagreement",
    "early_disagreement",
    "late_disagreement",
    "pairwise_rms_median",
    "pairwise_rms_p90",
    "first_step_disagreement",
    "last_step_disagreement",
)


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


# --------------------------------------------------------------------------- #
# S features (same definition as probe.data.build_week3_dataset)
# --------------------------------------------------------------------------- #


def _pairwise_rms(flat_samples: np.ndarray) -> np.ndarray:
    if flat_samples.shape[0] < 2:
        return np.empty(0, dtype=np.float64)
    distances = flat_samples[:, None, :] - flat_samples[None, :, :]
    rms = np.sqrt(np.mean(distances * distances, axis=2))
    return rms[np.triu_indices(flat_samples.shape[0], k=1)]


def compute_s_features(
    samples: np.ndarray,
    prefix_k: int,
) -> np.ndarray:
    """Eight action-distribution statistics over the first ``prefix_k`` samples.

    ``samples`` is ``[K, T, D]``; the prefix is taken along the sampling axis,
    matching the Week 3 definition.  K=1 is undefined (no cross-sample variance)
    and returns NaNs so the caller can skip it.
    """

    if prefix_k < 1 or prefix_k > samples.shape[0]:
        raise ValueError(
            f"Requested K={prefix_k} for samples with K={samples.shape[0]}"
        )
    prefix = samples[:prefix_k]
    if prefix_k == 1:
        return np.full(len(S_FEATURE_NAMES), np.nan, dtype=np.float32)

    variance = np.var(prefix, axis=0, ddof=1)
    disagreement_curve = np.sqrt(np.mean(variance, axis=1))
    split = max(1, math.ceil(prefix.shape[1] / 2))
    early = float(np.mean(disagreement_curve[:split]))
    late_values = disagreement_curve[split:]
    late = float(np.mean(late_values)) if late_values.size else float("nan")
    pairwise = _pairwise_rms(prefix.reshape(prefix.shape[0], -1))
    return np.asarray(
        [
            np.mean(disagreement_curve),
            np.max(disagreement_curve),
            early,
            late,
            np.median(pairwise),
            np.quantile(pairwise, 0.90),
            disagreement_curve[0],
            disagreement_curve[-1],
        ],
        dtype=np.float32,
    )


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #


@dataclass
class ConditionLabel:
    successes: int
    trials: int

    @property
    def rate(self) -> float:
        return self.successes / self.trials if self.trials else float("nan")


def load_labels(
    path: Path,
    suite: str,
) -> dict[str, dict[str, ConditionLabel]]:
    """Map ``base_id -> condition_id -> 12-retry success counts`` for one suite.

    Accepts both the task-level format written by the rollout summarizer
    (``original_successes``/``original_trials`` per row) and the per-record
    format (``eval_successes``/``eval_trials``), so either label file works.
    """

    labels: dict[str, dict[str, ConditionLabel]] = defaultdict(dict)
    for row in read_jsonl(path):
        if row.get("suite") != suite:
            continue
        base_id = str(row.get("base_id") or "")
        if not base_id:
            raise ValueError(f"{path}: label row without base_id")

        if "original_successes" in row:
            for condition in CONDITION_ORDER:
                successes = row.get(f"{condition}_successes")
                trials = row.get(f"{condition}_trials")
                if successes is None or trials is None:
                    raise ValueError(
                        f"{path}: {base_id} missing {condition} success counts"
                    )
                labels[base_id][condition] = ConditionLabel(int(successes), int(trials))
        else:
            condition = str(row.get("condition_id") or "")
            if condition not in CONDITION_ORDER:
                continue
            labels[base_id][condition] = ConditionLabel(
                int(row["eval_successes"]), int(row["eval_trials"])
            )
    return dict(labels)


# --------------------------------------------------------------------------- #
# Rollout indexing
# --------------------------------------------------------------------------- #


@dataclass
class EpisodeRows:
    episode_id: str
    base_id: str
    condition_id: str
    task_id: str
    task_name: str
    rows_by_replan: dict[int, dict[str, Any]]


def index_episodes(rollout_dir: Path, manifest_name: str) -> list[EpisodeRows]:
    manifest_path = rollout_dir / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(f"rollout manifest not found: {manifest_path}")

    grouped: dict[str, EpisodeRows] = {}
    for row in read_jsonl(manifest_path):
        condition_id = str(row.get("condition_id") or "")
        if condition_id not in CONDITION_ORDER:
            raise ValueError(
                f"{manifest_path}: unexpected condition_id {condition_id!r}"
            )
        episode_id = str(row["episode_id"])
        episode = grouped.get(episode_id)
        if episode is None:
            episode = EpisodeRows(
                episode_id=episode_id,
                base_id=str(row.get("base_id") or ""),
                condition_id=condition_id,
                task_id=str(row.get("task_id") or ""),
                task_name=str(row.get("task_name") or ""),
                rows_by_replan={},
            )
            grouped[episode_id] = episode
        elif episode.condition_id != condition_id:
            raise ValueError(
                f"episode {episode_id} mixes conditions "
                f"{episode.condition_id!r} and {condition_id!r}"
            )
        episode.rows_by_replan[int(row.get("replan_idx") or 0)] = row
    return list(grouped.values())


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #


def resolve(dataset_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else dataset_dir / path


def build_group_rows(
    episodes: list[EpisodeRows],
    labels: dict[str, dict[str, ConditionLabel]],
    rollout_dir: Path,
    s_prefixes: tuple[int, ...],
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Fold episodes into selection groups, truncated to the minimum replan count.

    A selection group is ``(task, replan_idx)``: the three conditions at the same
    replanning step, emitted contiguously in CONDITION_ORDER.  That is exactly the
    layout the existing Week 3 trainer and tie-aware metric expect, so a task with
    ``keep`` usable replanning steps simply yields ``keep`` groups instead of one.
    Every group of a task is tagged with the same ``task_id`` so the split can
    keep a whole task together.
    """

    by_task: dict[str, dict[str, EpisodeRows]] = defaultdict(dict)
    for episode in episodes:
        by_task[episode.base_id][episode.condition_id] = episode

    group_meta: dict[str, dict[str, Any]] = {}
    manifest_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    dropped_tasks: list[str] = []

    for base_id in sorted(by_task):
        conditions = by_task[base_id]
        if set(conditions) != set(CONDITION_ORDER):
            dropped_tasks.append(base_id)
            LOGGER.warning(
                "skip %s: has conditions %s, need all of %s",
                base_id, sorted(conditions), list(CONDITION_ORDER),
            )
            continue
        task_labels = labels.get(base_id)
        if not task_labels or set(task_labels) != set(CONDITION_ORDER):
            dropped_tasks.append(base_id)
            LOGGER.warning("skip %s: missing 12-retry labels", base_id)
            continue

        # Truncate to the minimum replanning count so the three conditions are
        # compared over the same horizon.
        per_condition_counts = {
            condition: len(conditions[condition].rows_by_replan)
            for condition in CONDITION_ORDER
        }
        keep = min(per_condition_counts.values())
        if keep <= 0:
            dropped_tasks.append(base_id)
            LOGGER.warning("skip %s: empty episode", base_id)
            continue

        task_id = conditions[CONDITION_ORDER[0]].task_id
        task_name = conditions[CONDITION_ORDER[0]].task_name
        for condition in CONDITION_ORDER:
            label = task_labels[condition]
            label_rows.append(
                {
                    "group_id": base_id,
                    "condition_id": condition,
                    "base_id": base_id,
                    "eval_successes": label.successes,
                    "eval_trials": label.trials,
                    "eval_success_rate": label.rate,
                }
            )

        for replan_idx in range(keep):
            group_id = f"{base_id}::r{replan_idx:04d}"
            group_meta[group_id] = {
                "group_id": group_id,
                "base_id": base_id,
                "task_id": task_id,
                "task_name": task_name,
                "replan_idx": replan_idx,
                "replans_kept": keep,
                "replans_available": per_condition_counts,
                "dropped_rows": {
                    condition: per_condition_counts[condition] - keep
                    for condition in CONDITION_ORDER
                },
            }
            # Condition-major inner loop keeps each group a contiguous 3-block in
            # CONDITION_ORDER, which the trainer's layout validation requires.
            for condition in CONDITION_ORDER:
                label = task_labels[condition]
                episode = conditions[condition]
                source = episode.rows_by_replan[replan_idx]
                record_id = f"{_safe(base_id)}__{condition}__r{replan_idx:04d}"
                manifest_rows.append(
                    {
                        "record_id": record_id,
                        "group_id": group_id,
                        "base_id": base_id,
                        "condition_id": condition,
                        "task_id": task_id,
                        "replan_idx": replan_idx,
                        "episode_id": episode.episode_id,
                        "step_idx": int(source.get("step_idx") or 0),
                        "instruction": source.get("instruction"),
                        "eval_successes": label.successes,
                        "eval_trials": label.trials,
                        "eval_success_rate": label.rate,
                        "obs_path": source.get("obs_path"),
                        "action_samples_path": source.get("action_samples_path"),
                        "b_feature_path": source.get("b_feature_path"),
                        "rollout_dir": str(rollout_dir),
                        "s_feature_paths": {},
                        "s_feature_metadata": {},
                    }
                )

    if dropped_tasks:
        LOGGER.warning("dropped %d task(s)", len(dropped_tasks))
    return group_meta, manifest_rows, label_rows


def _safe(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")


def write_features(
    output_dir: Path,
    manifest_rows: list[dict[str, Any]],
    rollout_dir: Path,
    s_prefixes: tuple[int, ...],
) -> None:
    """Materialize B (copied) and S (computed) features next to the manifest."""

    b_dir = output_dir / "features" / "B"
    b_dir.mkdir(parents=True, exist_ok=True)
    s_dirs = {k: output_dir / "features" / "S" / f"K{k}" for k in s_prefixes}
    for path in s_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    written_b = 0
    missing_b = 0
    written_s = 0

    for row in manifest_rows:
        record_id = row["record_id"]

        b_source = resolve(rollout_dir, row.get("b_feature_path"))
        if b_source is not None and b_source.exists():
            with np.load(b_source, allow_pickle=False) as data:
                feature = np.asarray(data["b_feature"], dtype=np.float32)
            np.savez_compressed(b_dir / f"{record_id}.npz", b_feature=feature)
            row["b_feature_path"] = f"features/B/{record_id}.npz"
            written_b += 1
        else:
            row["b_feature_path"] = None
            missing_b += 1

        action_source = resolve(rollout_dir, row.get("action_samples_path"))
        if action_source is None or not action_source.exists():
            continue
        with np.load(action_source, allow_pickle=False) as data:
            samples = np.asarray(data["action_samples"], dtype=np.float32)
        for prefix_k, directory in s_dirs.items():
            if prefix_k > samples.shape[0]:
                raise ValueError(
                    f"{record_id}: requested K={prefix_k} but only "
                    f"{samples.shape[0]} samples were stored"
                )
            values = compute_s_features(samples, prefix_k)
            np.savez_compressed(
                directory / f"{record_id}.npz",
                s_feature=values,
                feature_names=np.asarray(S_FEATURE_NAMES),
                prefix_k=np.asarray(prefix_k, dtype=np.int64),
            )
            row["s_feature_paths"][str(prefix_k)] = f"features/S/K{prefix_k}/{record_id}.npz"
            row["s_feature_metadata"][str(prefix_k)] = {
                "prefix_k": prefix_k,
                "defined": bool(np.isfinite(values).all()),
                "normalization": "identity",
            }
            written_s += 1

    LOGGER.info(
        "features: B written=%d missing=%d, S files=%d",
        written_b, missing_b, written_s,
    )


def make_task_splits(
    group_to_task: dict[str, str],
    record_ids_by_group: dict[str, list[str]],
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, Any]:
    """Split whole tasks into train/val/test; no task is ever split.

    ``group_to_task`` maps each selection group to its owning task.  Tasks are
    shuffled by ``seed`` and cut by count; every group of a task follows its task
    into the same split.  Both ``groups`` and ``record_ids`` are emitted: the
    training code resolves records through ``record_ids``, while the reporting
    tools read ``groups``.
    """

    if not 0 < train_fraction < 1:
        raise ValueError("train fraction must be in (0, 1)")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation fraction must be in [0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation fractions must be < 1")

    task_to_groups: dict[str, list[str]] = defaultdict(list)
    for group_id, task_id in group_to_task.items():
        task_to_groups[task_id].append(group_id)

    task_ids = sorted(task_to_groups)
    random.Random(seed).shuffle(task_ids)
    num_tasks = len(task_ids)
    num_train = max(1, int(round(num_tasks * train_fraction))) if num_tasks else 0
    num_validation = (
        max(1, int(round(num_tasks * validation_fraction)))
        if num_tasks >= 3 and validation_fraction > 0
        else 0
    )
    if num_train + num_validation >= num_tasks and num_tasks >= 3:
        num_train = max(1, num_tasks - num_validation - 1)

    task_splits = {
        "train": task_ids[:num_train],
        "validation": task_ids[num_train : num_train + num_validation],
        "test": task_ids[num_train + num_validation :],
    }
    split_groups = {
        split: sorted(
            group_id
            for task_id in task_ids_in_split
            for group_id in task_to_groups[task_id]
        )
        for split, task_ids_in_split in task_splits.items()
    }
    return {
        "split_seed": seed,
        "fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": 1.0 - train_fraction - validation_fraction,
        },
        "group_key": "base_id (one task = one unit; all its replanning groups stay together)",
        "groups": split_groups,
        "record_ids": {
            split: [
                record_id
                for group_id in split_groups[split]
                for record_id in record_ids_by_group.get(group_id, [])
            ]
            for split in ("train", "validation", "test")
        },
        "tasks": task_splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-name", default="records.jsonl")
    parser.add_argument("--s-prefixes", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--split-seed", type=int, default=20260916)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="only write manifest/labels/splits (fast structural check)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    rollout_dir = args.rollout_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(args.labels.expanduser().resolve(), args.suite)
    LOGGER.info("loaded labels for %d task(s) in suite %s", len(labels), args.suite)

    episodes = index_episodes(rollout_dir, args.manifest_name)
    LOGGER.info("indexed %d episode(s)", len(episodes))

    group_meta, manifest_rows, label_rows = build_group_rows(
        episodes, labels, rollout_dir, tuple(sorted(set(args.s_prefixes)))
    )
    LOGGER.info(
        "built %d group(s), %d replanning row(s)",
        len(group_meta), len(manifest_rows),
    )
    if not manifest_rows:
        raise SystemExit("no rows produced; check rollout/label alignment")

    if not args.skip_features:
        write_features(
            output_dir, manifest_rows, rollout_dir, tuple(sorted(set(args.s_prefixes)))
        )

    write_jsonl(output_dir / "manifest.jsonl", manifest_rows)

    with (output_dir / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group_id", "condition_id", "base_id",
                "eval_successes", "eval_trials", "eval_success_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(label_rows)

    records_by_group: dict[str, list[str]] = defaultdict(list)
    for row in manifest_rows:
        records_by_group[row["group_id"]].append(row["record_id"])

    split_payload = make_task_splits(
        {group_id: meta["base_id"] for group_id, meta in group_meta.items()},
        records_by_group,
        seed=args.split_seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    (output_dir / "splits.json").write_text(
        json.dumps(split_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    rows_per_group = defaultdict(int)
    for row in manifest_rows:
        rows_per_group[row["group_id"]] += 1
    summary = {
        "dataset_type": "week3_replanning",
        "suite": args.suite,
        "rollout_dir": str(rollout_dir),
        "labels": str(args.labels),
        "num_records": len(manifest_rows),
        "num_groups": len(group_meta),
        "records_per_group": sorted(set(rows_per_group.values())),
        "split_group_counts": {
            split: len(split_payload["groups"][split])
            for split in ("train", "validation", "test")
        },
        "split_record_counts": {
            split: len(split_payload["record_ids"][split])
            for split in ("train", "validation", "test")
        },
        "s_prefixes": sorted(set(args.s_prefixes)),
        "split_seed": args.split_seed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("wrote dataset to %s", output_dir)
    LOGGER.info(
        "records=%d groups=%d split_groups=%s split_records=%s",
        summary["num_records"], summary["num_groups"],
        summary["split_group_counts"], summary["split_record_counts"],
    )


if __name__ == "__main__":
    main()
