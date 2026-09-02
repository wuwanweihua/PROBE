"""Export failed LIBERO-Plus episodes from a PROBE records manifest."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EpisodeSummary:
    episode_id: str
    source_dataset_dir: str
    task_suite: str
    task_id: int
    task_name: str
    trial_idx: int
    seed: int
    final_success: bool
    episode_done: bool
    episode_steps: int
    reward_sum: float
    num_replanning_records: int = 0
    first_step_idx: int | None = None
    last_step_idx: int | None = None
    perturbation_types: set[str] = field(default_factory=set)
    instructions: set[str] = field(default_factory=set)
    errors: set[str] = field(default_factory=set)

    def update(self, record: dict[str, Any]) -> None:
        self.num_replanning_records += 1
        step_idx = _maybe_int(record.get("step_idx"))
        if step_idx is not None:
            self.first_step_idx = step_idx if self.first_step_idx is None else min(self.first_step_idx, step_idx)
            self.last_step_idx = step_idx if self.last_step_idx is None else max(self.last_step_idx, step_idx)
        if record.get("instruction"):
            self.instructions.add(str(record["instruction"]))
        if record.get("perturbation_type"):
            self.perturbation_types.add(str(record["perturbation_type"]))
        if record.get("error"):
            self.errors.add(str(record["error"]))
        if record.get("final_success") is True:
            self.final_success = True
        if bool(record.get("episode_done")):
            self.episode_done = True
        self.episode_steps = max(self.episode_steps, int(record.get("episode_steps") or 0))
        self.reward_sum = max(self.reward_sum, float(record.get("reward_sum") or 0.0))

    def to_json_dict(self, classification: dict[str, Any] | None = None) -> dict[str, Any]:
        primary_instruction = self.task_name
        if len(self.instructions) == 1:
            primary_instruction = next(iter(self.instructions))
        payload: dict[str, Any] = {
            "episode_id": self.episode_id,
            "source_dataset_dir": self.source_dataset_dir,
            "task_suite": self.task_suite,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "instruction": primary_instruction,
            "all_instructions": sorted(self.instructions),
            "trial_idx": self.trial_idx,
            "seed": self.seed,
            "final_success": self.final_success,
            "episode_done": self.episode_done,
            "episode_steps": self.episode_steps,
            "reward_sum": self.reward_sum,
            "num_replanning_records": self.num_replanning_records,
            "first_step_idx": self.first_step_idx,
            "last_step_idx": self.last_step_idx,
            "perturbation_types": sorted(self.perturbation_types),
            "errors": sorted(self.errors),
        }
        if classification:
            payload["classification"] = classification
            for key in ("difficulty_level", "difficulty", "category", "perturbation_type", "type"):
                if key in classification:
                    payload[key] = classification[key]
        return payload


def export_failed_episodes(
    dataset: str | Path,
    output: str | Path,
    manifest_name: str = "records.jsonl",
    task_suite_name: str | None = None,
    classification_json: str | Path | None = None,
    include_success: bool = False,
) -> dict[str, Any]:
    dataset_path = Path(dataset)
    manifest_path = dataset_path / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    classification = _load_classification(classification_json)
    episodes: dict[str, EpisodeSummary] = {}
    num_records = 0
    for record in _read_jsonl(manifest_path):
        num_records += 1
        episode_id = str(record.get("episode_id") or f"task{record.get('task_id')}_trial{record.get('trial_idx')}")
        if episode_id not in episodes:
            episodes[episode_id] = _new_episode(dataset_path, episode_id, record)
        episodes[episode_id].update(record)

    selected = [
        episode
        for episode in episodes.values()
        if include_success or not episode.final_success
    ]
    selected.sort(key=lambda item: (item.task_id, item.trial_idx, item.episode_id))

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for episode in selected:
            suite = task_suite_name or episode.task_suite
            meta = _lookup_classification(classification, suite, episode.task_id)
            handle.write(json.dumps(episode.to_json_dict(meta), ensure_ascii=False) + "\n")

    successes = sum(1 for item in episodes.values() if item.final_success)
    failures = len(episodes) - successes
    return {
        "dataset": str(dataset_path),
        "manifest": str(manifest_path),
        "output": str(output_path),
        "num_records": num_records,
        "num_episodes": len(episodes),
        "num_success_episodes": successes,
        "num_failure_episodes": failures,
        "num_exported": len(selected),
    }


def _new_episode(dataset_path: Path, episode_id: str, record: dict[str, Any]) -> EpisodeSummary:
    return EpisodeSummary(
        episode_id=episode_id,
        source_dataset_dir=str(record.get("source_dataset_dir") or dataset_path),
        task_suite=str(record.get("task_suite") or ""),
        task_id=int(record.get("task_id") or 0),
        task_name=str(record.get("task_name") or record.get("instruction") or ""),
        trial_idx=int(record.get("trial_idx") or 0),
        seed=int(record.get("seed") or 0),
        final_success=bool(record.get("final_success")),
        episode_done=bool(record.get("episode_done")),
        episode_steps=int(record.get("episode_steps") or 0),
        reward_sum=float(record.get("reward_sum") or 0.0),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return records


def _load_classification(path: str | Path | None) -> Any:
    if not path:
        return None
    classification_path = Path(path)
    if not classification_path.exists():
        return None
    return json.loads(classification_path.read_text(encoding="utf-8"))


def _lookup_classification(data: Any, suite: str, task_id: int) -> dict[str, Any] | None:
    if data is None:
        return None
    candidates: list[Any] = []
    if isinstance(data, dict):
        for key in (suite, suite.upper(), suite.replace("_", "-"), "tasks"):
            if key in data:
                candidates.append(data[key])
        candidates.append(data)
    else:
        candidates.append(data)
    for candidate in candidates:
        value = _lookup_task(candidate, task_id)
        if value is not None:
            return value if isinstance(value, dict) else {"value": value}
    return None


def _lookup_task(candidate: Any, task_id: int) -> Any:
    if isinstance(candidate, list):
        if 0 <= task_id < len(candidate):
            return candidate[task_id]
        return None
    if isinstance(candidate, dict):
        for key in (str(task_id), str(task_id + 1), f"task{task_id:03d}", f"task{task_id:04d}"):
            if key in candidate:
                return candidate[key]
        for value in candidate.values():
            if isinstance(value, dict) and int(value.get("task_id", -1)) in (task_id, task_id + 1):
                return value
    return None


def _maybe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset directory containing the records manifest.")
    parser.add_argument("--manifest-name", default="records.jsonl")
    parser.add_argument("--output", required=True, help="Output failed episode JSONL path.")
    parser.add_argument("--task-suite-name")
    parser.add_argument("--classification-json")
    parser.add_argument("--include-success", action="store_true")
    args = parser.parse_args()

    report = export_failed_episodes(
        dataset=args.dataset,
        output=args.output,
        manifest_name=args.manifest_name,
        task_suite_name=args.task_suite_name,
        classification_json=args.classification_json,
        include_success=args.include_success,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
