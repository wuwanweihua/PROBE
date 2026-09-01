"""Dataset writer for PROBE week-1 call records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from probe.data.record_schema import ProbeCallRecord


class ProbeDatasetWriter:
    """Write records, observations, and action chunks under one dataset root."""

    def __init__(self, dataset_dir: str | Path, manifest_name: str = "records.jsonl") -> None:
        self.dataset_dir = Path(dataset_dir)
        self.manifest_path = self.dataset_dir / manifest_name
        self.obs_dir = self.dataset_dir / "observations"
        self.actions_dir = self.dataset_dir / "actions"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.obs_dir.mkdir(parents=True, exist_ok=True)
        self.actions_dir.mkdir(parents=True, exist_ok=True)
        self._next_index = self._count_existing_records()

    @property
    def num_records(self) -> int:
        return self._next_index

    def allocate_record_id(self) -> str:
        record_id = f"call_{self._next_index:08d}"
        self._next_index += 1
        return record_id

    def write_observation(self, record_id: str, arrays: dict[str, Any]) -> str:
        path = self.obs_dir / f"{record_id}.npz"
        payload = {
            _sanitize_key(key): np.asarray(value)
            for key, value in arrays.items()
            if _is_array_like(value)
        }
        np.savez_compressed(path, **payload)
        return self._relative(path)

    def write_action_samples(
        self,
        record_id: str,
        action_samples: Any,
        selected_sample_index: int,
    ) -> str:
        path = self.actions_dir / f"{record_id}.npz"
        samples = np.asarray(action_samples)
        selected = samples[selected_sample_index]
        np.savez_compressed(
            path,
            action_samples=samples,
            selected_action_chunk=selected,
            selected_sample_index=np.asarray(selected_sample_index, dtype=np.int64),
        )
        return self._relative(path)

    def append_record(self, record: ProbeCallRecord) -> None:
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_json_dict(), ensure_ascii=False) + "\n")

    def _count_existing_records(self) -> int:
        if not self.manifest_path.exists():
            return 0
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.dataset_dir).as_posix()


def _sanitize_key(key: str) -> str:
    return key.replace("/", "__").replace(":", "_")


def _is_array_like(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, bytes, dict)):
        return False
    return hasattr(value, "__array__") or isinstance(value, (list, tuple, int, float, bool))
