"""Storage helpers for the non-executing Week 2 probe smoke dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class Week2ProbeWriter:
    """Write one state-condition record and its fixed input/action samples."""

    def __init__(self, dataset_dir: str | Path, manifest_name: str = "manifest.jsonl") -> None:
        self.dataset_dir = Path(dataset_dir)
        self.manifest_path = self.dataset_dir / manifest_name
        self.observations_dir = self.dataset_dir / "observations"
        self.actions_dir = self.dataset_dir / "actions"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.observations_dir.mkdir(parents=True, exist_ok=True)
        self.actions_dir.mkdir(parents=True, exist_ok=True)

    def completed_record_ids(self) -> set[str]:
        if not self.manifest_path.exists():
            return set()
        completed: set[str] = set()
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    if payload.get("record_id"):
                        completed.add(str(payload["record_id"]))
        return completed

    def write_observation(self, record_id: str, arrays: dict[str, Any]) -> str:
        path = self.observations_dir / f"{record_id}.npz"
        np.savez_compressed(
            path,
            **{
                str(key): np.asarray(value)
                for key, value in arrays.items()
                if value is not None
            },
        )
        return path.relative_to(self.dataset_dir).as_posix()

    def write_actions(self, record_id: str, action_samples: Any) -> str:
        path = self.actions_dir / f"{record_id}.npz"
        samples = np.asarray(action_samples)
        np.savez_compressed(
            path,
            action_samples=samples,
            action_sample_shape=np.asarray(samples.shape, dtype=np.int64),
        )
        return path.relative_to(self.dataset_dir).as_posix()

    def append(self, payload: dict[str, Any]) -> None:
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
