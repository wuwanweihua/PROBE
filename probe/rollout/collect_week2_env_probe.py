"""Collect Week 2 samples from a fresh LIBERO environment without executing actions.

For each selected condition batch this collector:

1. restores the task initial state;
2. runs the same dummy stabilization steps as the normal rollout;
3. captures the observation immediately before the first VLA call;
4. sends the fixed observation with each condition instruction K times;
5. never executes a sampled action and never calls env.step after capture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import tqdm

from probe.data.week2_writer import Week2ProbeWriter
from probe.envs.libero_runner import (
    LIBERO_DUMMY_ACTION,
    build_policy_element,
    get_task_suite,
    make_libero_env,
    max_steps_for_suite,
    observation_arrays_for_record,
)
from probe.policies.pi05_client import Pi05Client
from probe.rollout.collect_condition_calls import (
    _read_condition_batches,
    _selected_conditions,
)


LOGGER = logging.getLogger(__name__)


def _sha256_array(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _first_label(labels: dict[str, dict[str, dict[str, Any]]], base_id: str, condition_id: str):
    return labels.get(base_id, {}).get(condition_id, {})


def _load_optional_labels(path: str | None) -> dict[str, dict[str, dict[str, Any]]]:
    if not path:
        return {}
    result: dict[str, dict[str, dict[str, Any]]] = {}
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            base_id = str(
                payload.get("base_id")
                or payload.get("task_id")
                or f"task{int(payload['task_id']):04d}"
            )
            condition_id = str(
                payload.get("condition_id")
                or payload.get("condition_type")
                or "all"
            )
            result.setdefault(base_id, {})[condition_id] = payload
    return result


def _label_for(
    labels: dict[str, dict[str, dict[str, Any]]],
    base_id: str,
    task_id: int,
    condition_id: str,
) -> dict[str, Any]:
    candidates = (
        labels.get(base_id, {}).get(condition_id),
        labels.get(str(task_id), {}).get(condition_id),
        labels.get(base_id, {}).get("all"),
        labels.get(str(task_id), {}).get("all"),
    )
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return {}


def _parse_task_ids(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _build_original_batches(
    task_suite: Any,
    task_ids: list[int],
) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        instruction = str(task.language)
        batches.append(
            {
                "task_id": task_id,
                "base_id": f"task{task_id:04d}",
                "task_name": instruction,
                "original_instruction": instruction,
                "conditions": [
                    {
                        "condition_id": "original",
                        "condition_type": "original",
                        "instruction": instruction,
                    }
                ],
            }
        )
    return batches


def _inline_original_labels(value: str | None) -> dict[str, dict[str, dict[str, Any]]]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--original-success-json must be a JSON object")
    labels: dict[str, dict[str, dict[str, Any]]] = {}
    for task_id, successes in parsed.items():
        count = int(successes)
        labels[str(task_id)] = {
            "original": {
                "eval_successes": count,
                "eval_trials": 8,
                "eval_success_rate": count / 8.0,
            }
        }
    return labels


def collect(args: argparse.Namespace) -> dict[str, Any]:
    task_suite = get_task_suite(args.task_suite_name)
    task_ids = _parse_task_ids(args.task_ids)
    if task_ids:
        if tuple(value.strip() for value in args.condition_types.split(",") if value.strip()) != (
            "original",
        ):
            raise ValueError("--task-ids mode supports only --condition-types original")
        batches = _build_original_batches(task_suite, task_ids)
        condition_path = None
    else:
        if not args.conditions:
            raise ValueError("provide either --conditions or --task-ids")
        condition_path = Path(args.conditions)
        batches = _read_condition_batches(condition_path, args.max_base_states)

    if args.max_base_states is not None:
        batches = batches[: args.max_base_states]
    labels = _load_optional_labels(args.labels)
    labels_inline = _inline_original_labels(args.original_success_json)
    for base_id, condition_labels in labels_inline.items():
        labels.setdefault(base_id, {}).update(condition_labels)
    wanted = tuple(value.strip() for value in args.condition_types.split(",") if value.strip())
    if not wanted:
        raise ValueError("condition-types must not be empty")

    writer = Week2ProbeWriter(args.output_dir)
    completed = writer.completed_record_ids() if args.resume else set()
    client = Pi05Client(args.host, args.port)

    written = 0
    skipped = 0
    failed = 0
    captured = 0
    env_step_calls_after_capture = 0
    shapes: set[tuple[int, ...]] = set()
    started = time.perf_counter()

    for base_index, batch in enumerate(tqdm.tqdm(batches, desc="week2 base states")):
        task_id = int(batch["task_id"])
        base_id = str(batch.get("base_id") or f"task{task_id:04d}")
        selected_conditions = _selected_conditions(batch, wanted)
        selected_ids = {
            str(condition.get("condition_type") or condition["condition_id"])
            for condition in selected_conditions
        }
        if not all(condition in selected_ids for condition in wanted):
            LOGGER.warning(
                "Skipping task %s: requested conditions missing; found=%s",
                task_id,
                sorted(selected_ids),
            )
            continue

        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        if initial_states is None or len(initial_states) == 0:
            raise ValueError(f"Task {task_id} has no initial states")
        init_state_slot = int(args.init_state_index) % len(initial_states)
        exec_seed = int(args.exec_seed_start) + base_index
        env, canonical_instruction = make_libero_env(
            task,
            resolution=args.env_resolution,
            seed=exec_seed,
        )

        try:
            # This is intentionally the only environment progression in the probe.
            env.seed(exec_seed)
            env.reset()
            obs = env.set_init_state(initial_states[init_state_slot])
            for _ in range(args.num_steps_wait):
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                if done:
                    raise RuntimeError(
                        f"Task {task_id} finished during stabilization before policy call"
                    )
            captured += 1
            captured_obs = observation_arrays_for_record(
                obs,
                build_policy_element(obs, canonical_instruction, resize_size=args.resize_size),
            )
            input_hashes = {
                key: _sha256_array(value)
                for key, value in captured_obs.items()
                if key.startswith("policy_")
            }

            for condition in selected_conditions:
                condition_id = str(
                    condition.get("condition_type") or condition["condition_id"]
                )
                if condition_id not in wanted:
                    continue
                record_id = f"{_safe_id(base_id)}_{_safe_id(condition_id)}"
                if record_id in completed:
                    skipped += 1
                    continue

                instruction = str(condition["instruction"])
                policy_element = build_policy_element(
                    obs,
                    instruction,
                    resize_size=args.resize_size,
                )
                sample_started = time.perf_counter()
                try:
                    if args.capture_b:
                        (
                            samples,
                            sample_times,
                            b_feature,
                            b_feature_metadata,
                        ) = client.sample_action_chunks_with_feature_timing(
                            policy_element,
                            k=args.k_samples,
                        )
                    else:
                        samples, sample_times = client.sample_action_chunks_with_timing(
                            policy_element,
                            k=args.k_samples,
                        )
                        b_feature = None
                        b_feature_metadata = {}
                except Exception:
                    failed += 1
                    LOGGER.exception("Probe failed for %s", record_id)
                    continue

                samples = np.asarray(samples)
                total_seconds = time.perf_counter() - sample_started
                shapes.add(tuple(int(value) for value in samples.shape))
                if samples.ndim != 3 or samples.shape[0] != args.k_samples:
                    raise ValueError(
                        f"{record_id}: expected [K,H,D], got {samples.shape}"
                    )
                if not np.isfinite(samples).all():
                    raise ValueError(f"{record_id}: action samples contain NaN or Inf")

                observation_path = writer.write_observation(
                    f"{_safe_id(base_id)}_state",
                    captured_obs,
                )
                actions_path = writer.write_actions(record_id, samples)
                b_feature_path = (
                    writer.write_b_feature(record_id, b_feature)
                    if b_feature is not None
                    else None
                )
                label = _label_for(labels, base_id, task_id, condition_id)
                successes = label.get("eval_successes", label.get("successes"))
                trials = label.get("eval_trials", label.get("trials"))
                if successes is not None:
                    successes = int(successes)
                if trials is not None:
                    trials = int(trials)

                writer.append(
                    {
                        "record_id": record_id,
                        "task_id": task_id,
                        "base_id": base_id,
                        "task_name": str(batch.get("task_name") or canonical_instruction),
                        "category": batch.get("classification", {}).get("category"),
                        "difficulty": batch.get("classification", {}).get(
                            "difficulty_level"
                        ),
                        "condition_id": condition_id,
                        "instruction": instruction,
                "conditions_source": str(condition_path) if condition_path else "libero_task_definition",
                        "observation_path": observation_path,
                        "action_samples_path": actions_path,
                        "b_feature_path": b_feature_path,
                        "b_feature_type": (
                            b_feature_metadata.get("feature_type")
                            if b_feature is not None
                            else None
                        ),
                        "b_feature_shape": (
                            list(np.asarray(b_feature).shape)
                            if b_feature is not None
                            else None
                        ),
                        "b_feature_dtype": (
                            str(np.asarray(b_feature).dtype)
                            if b_feature is not None
                            else None
                        ),
                        "b_feature_metadata": b_feature_metadata,
                        "policy_call_index": 0,
                        "step": 1,
                        "source_step_idx": args.num_steps_wait,
                        "init_state_index": init_state_slot,
                        "exec_seed": exec_seed,
                        "num_steps_wait": args.num_steps_wait,
                        "probe_sample_count": int(samples.shape[0]),
                        "action_sample_shape": list(samples.shape),
                        "probe_rng_control": "server_default_internal_split",
                        "probe_sample_indices": list(range(int(samples.shape[0]))),
                        "sample_times_seconds": sample_times,
                        "probe_total_seconds": total_seconds,
                        "input_hashes": input_hashes,
                        "eval_successes": successes,
                        "eval_trials": trials,
                        "eval_success_rate": (
                            successes / trials
                            if successes is not None and trials
                            else None
                        ),
                        "non_executing_after_capture": True,
                        "environment_steps_after_capture": 0,
                        "checkpoint_uri": args.checkpoint_uri,
                        "policy_name": args.policy_name,
                        "resize_size": args.resize_size,
                        "env_resolution": args.env_resolution,
                        "host": args.host,
                        "port": args.port,
                    }
                )
                written += 1
                LOGGER.info(
                    "Wrote %s shape=%s total=%.3fs",
                    record_id,
                    tuple(samples.shape),
                    total_seconds,
                )
        finally:
            try:
                env.close()
            except Exception:
                LOGGER.exception("Failed to close environment for task %s", task_id)

    return {
        "conditions": str(condition_path) if condition_path else "libero_task_definition",
        "output_dir": str(args.output_dir),
        "num_condition_batches": len(batches),
        "num_captured_states": captured,
        "num_written_records": written,
        "num_skipped_records": skipped,
        "num_failed_records": failed,
        "action_shapes": [list(shape) for shape in sorted(shapes)],
        "k_samples": args.k_samples,
        "capture_b": bool(args.capture_b),
        "step": 1,
        "num_steps_wait": args.num_steps_wait,
        "non_executing_after_capture": True,
        "environment_steps_after_capture": env_step_calls_after_capture,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions")
    parser.add_argument(
        "--task-ids",
        help="Comma-separated LIBERO task IDs. In this mode only original is collected.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--condition-types", default="original,better,worse")
    parser.add_argument("--labels")
    parser.add_argument(
        "--original-success-json",
        help='Inline JSON object mapping task_id to success count out of 8.',
    )
    parser.add_argument("--init-state-index", type=int, default=0)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--env-resolution", type=int, default=256)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--k-samples", type=int, default=32)
    parser.add_argument(
        "--capture-b",
        action="store_true",
        help="Request a server-side B feature on the first action sample.",
    )
    parser.add_argument("--max-base-states", type=int)
    parser.add_argument("--exec-seed-start", type=int, default=700_000)
    parser.add_argument("--checkpoint-uri", default="gs://openpi-assets/checkpoints/pi05_libero")
    parser.add_argument("--policy-name", default="pi05_libero")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    print(json.dumps(collect(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
