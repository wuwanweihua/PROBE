from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from probe.data.build_week3_dataset import (
    S_FEATURE_NAMES,
    build_dataset,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _make_dataset(root: Path) -> Path:
    (root / "actions").mkdir(parents=True)
    (root / "features" / "B").mkdir(parents=True)
    rows = []
    for condition_index, condition in enumerate(("original", "better", "worse")):
        record_id = f"task0001_{condition}"
        samples = np.arange(32 * 10 * 7, dtype=np.float32).reshape(32, 10, 7)
        samples += condition_index
        np.savez_compressed(
            root / "actions" / f"{record_id}.npz",
            action_samples=samples,
        )
        np.savez_compressed(
            root / "features" / "B" / f"{record_id}.npz",
            b_feature=np.ones(2048, dtype=np.float32) * condition_index,
        )
        rows.append(
            {
                "record_id": record_id,
                "task_id": 1,
                "base_id": "task0001",
                "condition_id": condition,
                "instruction": "do the task",
                "action_samples_path": f"actions/{record_id}.npz",
                "b_feature_path": f"features/B/{record_id}.npz",
            }
        )
    _write_jsonl(root / "manifest.jsonl", rows)
    return root


def test_build_week3_dataset_joins_labels_and_keeps_group_together(tmp_path: Path):
    source = _make_dataset(tmp_path / "source")
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(
        labels_path,
        [
            {
                "suite": "camera",
                "task_id": 1,
                "original_successes": 4,
                "original_trials": 8,
                "better_successes": 5,
                "better_trials": 8,
                "worse_successes": 3,
                "worse_trials": 8,
            }
        ],
    )

    class Args:
        dataset = [f"camera={source}"]
        labels = str(labels_path)
        output_dir = str(tmp_path / "output")
        manifest_name = "manifest.jsonl"
        k_prefixes = [1, 4, 8, 16, 32]
        split_seed = 7
        train_fraction = 0.70
        validation_fraction = 0.15
        require_labels = True
        require_b = True

    summary = build_dataset(Args())
    assert summary["num_records"] == 3
    assert summary["missing_labels"] == 0
    assert summary["missing_b_features"] == 0

    records = [
        json.loads(line)
        for line in (tmp_path / "output" / "manifest.jsonl").read_text().splitlines()
    ]
    assert all(record["eval_trials"] == 8 for record in records)
    assert all(len(record["s_feature_paths"]) == 5 for record in records)

    with np.load(
        tmp_path / "output" / "features" / "S" / "K1" / "camera__task0001_original.npz",
        allow_pickle=False,
    ) as data:
        assert data["s_feature"].shape == (len(S_FEATURE_NAMES),)
        assert np.isnan(data["s_feature"]).all()

    splits = json.loads((tmp_path / "output" / "splits.json").read_text())
    assert len(splits["groups"]["train"]) + len(splits["groups"]["validation"]) + len(
        splits["groups"]["test"]
    ) == 1
