"""Summarize diversity among sampled action chunks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def summarize_action_chunks(
    dataset_dir: str | Path,
    manifest_name: str = "records.jsonl",
    limit: int | None = None,
    csv_out: str | Path | None = None,
) -> dict[str, Any]:
    dataset_path = Path(dataset_dir)
    records = _read_records(dataset_path / manifest_name, limit=limit)
    rows: list[dict[str, Any]] = []

    for record in records:
        record_dataset_path = Path(str(record.get("source_dataset_dir") or dataset_path))
        action_path = record_dataset_path / str(record["action_samples_path"])
        with np.load(action_path, allow_pickle=False) as data:
            samples = np.asarray(data["action_samples"], dtype=np.float64)
        rows.append(_summarize_one_record(record, samples))

    if csv_out is not None:
        _write_csv(rows, Path(csv_out))

    return {
        "dataset_dir": str(dataset_path),
        "manifest": str(dataset_path / manifest_name),
        "num_records": len(rows),
        "overall": _aggregate(rows),
        "success": _aggregate([row for row in rows if row["final_success"] is True]),
        "failure": _aggregate([row for row in rows if row["final_success"] is False]),
        "top_diverse_records": sorted(
            rows,
            key=lambda row: row["flat_pairwise_rms_mean"],
            reverse=True,
        )[:5],
    }


def _read_records(manifest_path: Path, limit: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def _summarize_one_record(record: dict[str, Any], samples: np.ndarray) -> dict[str, Any]:
    if samples.ndim < 3:
        raise ValueError(f"Expected action_samples as [K, T, D], got shape {samples.shape}.")

    k, horizon, action_dim = samples.shape[:3]
    flat = samples.reshape(k, -1)
    pairwise_rms = _pairwise_rms(flat)
    first_step = samples[:, 0, :].reshape(k, -1)
    last_step = samples[:, -1, :].reshape(k, -1)
    endpoints = samples[:, -1, :3] if action_dim >= 3 else last_step
    gripper = samples[:, :, -1] if action_dim >= 1 else samples.reshape(k, -1)

    return {
        "record_id": record.get("record_id"),
        "source_record_id": record.get("source_record_id"),
        "source_dataset_name": record.get("source_dataset_name"),
        "task_id": record.get("task_id"),
        "step_idx": record.get("step_idx"),
        "final_success": record.get("final_success"),
        "perturbation_type": record.get("perturbation_type"),
        "instruction": record.get("instruction"),
        "shape_k": int(k),
        "shape_horizon": int(horizon),
        "shape_action_dim": int(action_dim),
        "flat_pairwise_rms_mean": float(pairwise_rms.mean()) if pairwise_rms.size else 0.0,
        "flat_pairwise_rms_p90": float(np.quantile(pairwise_rms, 0.9)) if pairwise_rms.size else 0.0,
        "first_step_rms_std": float(_mean_feature_std(first_step)),
        "last_step_rms_std": float(_mean_feature_std(last_step)),
        "endpoint_xyz_rms_std": float(_mean_feature_std(endpoints)),
        "gripper_std_mean": float(np.std(gripper, axis=0).mean()),
    }


def _pairwise_rms(flat: np.ndarray) -> np.ndarray:
    distances = []
    for i in range(flat.shape[0]):
        diff = flat[i + 1 :] - flat[i]
        if diff.size:
            distances.extend(np.sqrt(np.mean(diff * diff, axis=1)).tolist())
    return np.asarray(distances, dtype=np.float64)


def _mean_feature_std(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.var(values, axis=0))))


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"num_records": 0}
    keys = [
        "flat_pairwise_rms_mean",
        "flat_pairwise_rms_p90",
        "first_step_rms_std",
        "last_step_rms_std",
        "endpoint_xyz_rms_std",
        "gripper_std_mean",
    ]
    output: dict[str, Any] = {"num_records": len(rows)}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        output[f"{key}_mean"] = float(values.mean())
        output[f"{key}_median"] = float(np.median(values))
    return output


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest-name", default="records.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--csv-out")
    args = parser.parse_args()

    report = summarize_action_chunks(
        dataset_dir=args.dataset,
        manifest_name=args.manifest_name,
        limit=args.limit,
        csv_out=args.csv_out,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
