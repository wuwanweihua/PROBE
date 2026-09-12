"""Collect a small, non-executing Week 2 K-sample smoke dataset.

The source manifest must come from an existing rollout dataset that saved the
first policy-call observation. This collector reuses that fixed input, varies
only the instruction condition, and never calls env.step().
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from probe.data.week2_writer import Week2ProbeWriter
from probe.policies.pi05_client import Pi05Client


LOGGER = logging.getLogger(__name__)


def _condition_id(record: dict[str, Any]) -> str:
    return str(record.get("condition_type") or record.get("condition_id") or "")


def _base_id(record: dict[str, Any]) -> str:
    return str(record.get("base_id") or f"task{int(record['task_id']):04d}")


def load_first_call_records(
    manifest_path: str | Path,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Return base_id -> condition_id -> unique first-call episode records."""

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seen_episodes: set[str] = set()
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if int(payload.get("replan_idx", -1)) != 0:
                continue
            episode_id = str(payload.get("episode_id") or "")
            condition_id = _condition_id(payload)
            if not episode_id or not condition_id:
                continue
            episode_key = f"{episode_id}\0{condition_id}"
            if episode_key in seen_episodes:
                continue
            seen_episodes.add(episode_key)
            grouped[_base_id(payload)][condition_id].append(payload)
    if not grouped:
        raise ValueError(
            f"No records with replan_idx=0 and condition_id found in {path}"
        )
    LOGGER.info(
        "Loaded %d base states with first-call records from %s",
        len(grouped),
        path,
    )
    return {base_id: dict(conditions) for base_id, conditions in grouped.items()}


def _resolve_source_path(source_manifest: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else source_manifest.parent / path


def _load_policy_input(source_manifest: Path, record: dict[str, Any]) -> dict[str, np.ndarray]:
    obs_path = _resolve_source_path(source_manifest, str(record["obs_path"]))
    if not obs_path.exists():
        raise FileNotFoundError(f"Observation file not found: {obs_path}")
    with np.load(obs_path, allow_pickle=False) as data:
        required = ("policy_image", "policy_wrist_image", "policy_state")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"{obs_path} is missing required arrays: {missing}")
        return {key: np.asarray(data[key]).copy() for key in required}


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _source_label(records: list[dict[str, Any]]) -> tuple[int, int]:
    successes = sum(bool(record.get("final_success")) for record in records)
    return successes, len(records)


def _select_base_ids(
    grouped: dict[str, dict[str, list[dict[str, Any]]]],
    condition_types: tuple[str, ...],
    max_base_states: int | None,
) -> list[str]:
    selected = [
        base_id
        for base_id in sorted(grouped)
        if all(condition in grouped[base_id] for condition in condition_types)
    ]
    return selected if max_base_states is None else selected[:max_base_states]


def collect_smoke(args: argparse.Namespace) -> dict[str, Any]:
    source_manifest = Path(args.source_manifest)
    grouped = load_first_call_records(source_manifest)
    condition_types = tuple(
        value.strip() for value in args.condition_types.split(",") if value.strip()
    )
    if not condition_types:
        raise ValueError("At least one condition type is required")
    if args.reference_condition not in condition_types:
        raise ValueError("reference-condition must be included in condition-types")

    base_ids = _select_base_ids(grouped, condition_types, args.max_base_states)
    if not base_ids:
        raise ValueError(
            f"No base states contain all requested conditions: {condition_types}"
        )

    writer = Week2ProbeWriter(args.output_dir)
    completed = writer.completed_record_ids() if args.resume else set()
    client = Pi05Client(args.host, args.port)
    written = 0
    skipped = 0
    failed = 0
    shapes: set[tuple[int, ...]] = set()
    start = time.perf_counter()

    for base_id in base_ids:
        conditions = grouped[base_id]
        reference_record = conditions[args.reference_condition][0]
        reference_input = _load_policy_input(source_manifest, reference_record)
        reference_hashes = {
            key: _sha256_array(value) for key, value in reference_input.items()
        }

        for condition in condition_types:
            condition_records = conditions[condition]
            first_condition_record = condition_records[0]
            record_id = f"{base_id}_{condition}"
            if record_id in completed:
                skipped += 1
                continue

            instruction = str(first_condition_record.get("instruction") or "")
            if not instruction:
                raise ValueError(f"Empty instruction for {record_id}")
            policy_element = {
                "observation/image": reference_input["policy_image"],
                "observation/wrist_image": reference_input["policy_wrist_image"],
                "observation/state": reference_input["policy_state"],
                "prompt": instruction,
            }
            sample_started = time.perf_counter()
            try:
                samples, sample_times = client.sample_action_chunks_with_timing(
                    policy_element, k=args.k_samples
                )
            except Exception:
                failed += 1
                LOGGER.exception("Probe failed for %s", record_id)
                continue

            total_seconds = time.perf_counter() - sample_started
            samples = np.asarray(samples)
            shapes.add(tuple(int(value) for value in samples.shape))
            if samples.ndim != 3 or samples.shape[0] != args.k_samples:
                raise ValueError(
                    f"{record_id}: expected [K,H,D], got {samples.shape}"
                )
            if not np.isfinite(samples).all():
                raise ValueError(f"{record_id}: action samples contain NaN or Inf")

            observation_path = writer.write_observation(
                record_id,
                {
                    "policy_image": reference_input["policy_image"],
                    "policy_wrist_image": reference_input["policy_wrist_image"],
                    "policy_state": reference_input["policy_state"],
                },
            )
            actions_path = writer.write_actions(record_id, samples)
            successes, trials = _source_label(condition_records)
            payload = {
                "record_id": record_id,
                "task_id": int(first_condition_record["task_id"]),
                "base_id": base_id,
                "condition_id": condition,
                "instruction": instruction,
                "source_manifest": str(source_manifest),
                "source_record_ids": [
                    str(record.get("record_id", "")) for record in condition_records
                ],
                "reference_condition": args.reference_condition,
                "reference_source_record_id": str(reference_record.get("record_id", "")),
                "reference_observation_source": str(reference_record["obs_path"]),
                "observation_path": observation_path,
                "action_samples_path": actions_path,
                "policy_call_index": 0,
                "step": 1,
                "source_step_idx": int(reference_record.get("step_idx", -1)),
                "init_state_index": reference_record.get("metadata", {}).get(
                    "init_state_slot"
                ),
                "exec_seed": reference_record.get("exec_seed"),
                "probe_sample_count": int(samples.shape[0]),
                "action_sample_shape": list(samples.shape),
                "probe_rng_control": "server_default_internal_split",
                "probe_sample_indices": list(range(int(samples.shape[0]))),
                "sample_times_seconds": sample_times,
                "probe_total_seconds": total_seconds,
                "eval_successes": successes,
                "eval_trials": trials,
                "eval_success_rate": successes / trials if trials else None,
                "input_hashes": reference_hashes,
                "non_executing": True,
            }
            writer.append(payload)
            written += 1
            LOGGER.info(
                "Wrote %s shape=%s label=%d/%d total=%.3fs",
                record_id,
                tuple(samples.shape),
                successes,
                trials,
                total_seconds,
            )

    return {
        "source_manifest": str(source_manifest),
        "output_dir": str(args.output_dir),
        "condition_types": list(condition_types),
        "reference_condition": args.reference_condition,
        "max_base_states": args.max_base_states,
        "num_selected_base_states": len(base_ids),
        "num_attempted_records": len(base_ids) * len(condition_types),
        "num_written_records": written,
        "num_skipped_records": skipped,
        "num_failed_records": failed,
        "action_shapes": [list(shape) for shape in sorted(shapes)],
        "k_samples": args.k_samples,
        "non_executing": True,
        "elapsed_seconds": time.perf_counter() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--condition-types", default="original,better,worse")
    parser.add_argument("--reference-condition", default="original")
    parser.add_argument("--k-samples", type=int, default=4)
    parser.add_argument("--max-base-states", type=int, default=2)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    report = collect_smoke(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
