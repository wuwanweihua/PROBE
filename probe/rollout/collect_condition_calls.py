"""Collect LIBERO-Plus original/better/worse condition rollouts."""

from __future__ import annotations

import argparse
import collections
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tqdm

from probe.data.record_schema import ProbeCallRecord
from probe.data.writer import ProbeDatasetWriter
from probe.envs.libero_runner import (
    LIBERO_DUMMY_ACTION,
    build_policy_element,
    get_task_suite,
    make_libero_env,
    max_steps_for_suite,
    observation_arrays_for_record,
)
from probe.policies.pi05_client import Pi05Client


@dataclass
class ConditionCollectorArgs:
    conditions: str
    output_dir: str
    host: str = "0.0.0.0"
    port: int = 18000
    task_suite_name: str = "libero_10"
    condition_types: tuple[str, ...] = ("original", "better", "worse")
    num_trials_per_condition: int = 8
    init_state_index: int = 0
    num_steps_wait: int = 10
    resize_size: int = 224
    replan_steps: int = 5
    k_samples: int = 32
    max_base_states: int | None = None
    max_episodes: int | None = None
    action_selection: str = "random"
    exec_seed_start: int = 700_000
    probe_seed_start: int = 100_000
    manifest_name: str = "records.jsonl"
    checkpoint_uri: str = "gs://openpi-assets/checkpoints/pi05_libero"
    policy_name: str = "pi05_libero"
    resume: bool = True


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    report = collect_conditions(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def collect_conditions(args: ConditionCollectorArgs) -> dict[str, Any]:
    condition_batches = _read_condition_batches(Path(args.conditions), args.max_base_states)
    writer = ProbeDatasetWriter(args.output_dir, manifest_name=args.manifest_name)
    completed = _completed_episode_ids(writer.manifest_path) if args.resume else set()
    task_suite = get_task_suite(args.task_suite_name)
    max_steps = max_steps_for_suite(args.task_suite_name)
    client = Pi05Client(args.host, args.port)

    logging.info("Writing dataset to %s", args.output_dir)
    logging.info("Loaded %d condition batches from %s", len(condition_batches), args.conditions)
    logging.info("Condition types: %s", ",".join(args.condition_types))

    attempted = 0
    written_episodes = 0
    success_episodes = 0
    failure_episodes = 0

    for base_index, batch in enumerate(tqdm.tqdm(condition_batches, desc="base states")):
        task_id = int(batch["task_id"])
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, canonical_instruction = make_libero_env(task, seed=args.exec_seed_start + base_index)
        try:
            for condition_index, condition in enumerate(_selected_conditions(batch, args.condition_types)):
                for trial_idx in tqdm.tqdm(
                    range(args.num_trials_per_condition),
                    desc=f"task {task_id} {condition['condition_id']}",
                    leave=False,
                ):
                    if args.max_episodes is not None and attempted >= args.max_episodes:
                        return _report(
                            args,
                            attempted,
                            written_episodes,
                            success_episodes,
                            failure_episodes,
                            len(condition_batches),
                        )
                    exec_seed = _exec_seed(args.exec_seed_start, base_index, condition_index, trial_idx)
                    episode_id = _episode_id(batch, condition, trial_idx, exec_seed)
                    if episode_id in completed:
                        continue
                    attempted += 1
                    success = _run_condition_episode(
                        env=env,
                        initial_states=initial_states,
                        canonical_instruction=canonical_instruction,
                        batch=batch,
                        condition=condition,
                        base_index=base_index,
                        condition_index=condition_index,
                        task_suite_name=args.task_suite_name,
                        task_id=task_id,
                        trial_idx=trial_idx,
                        exec_seed=exec_seed,
                        max_steps=max_steps,
                        args=args,
                        client=client,
                        writer=writer,
                    )
                    written_episodes += 1
                    if success:
                        success_episodes += 1
                    else:
                        failure_episodes += 1
        finally:
            try:
                env.close()
            except Exception:
                pass

    return _report(
        args,
        attempted,
        written_episodes,
        success_episodes,
        failure_episodes,
        len(condition_batches),
    )


def _run_condition_episode(
    *,
    env: Any,
    initial_states: Any,
    canonical_instruction: str,
    batch: dict[str, Any],
    condition: dict[str, Any],
    base_index: int,
    condition_index: int,
    task_suite_name: str,
    task_id: int,
    trial_idx: int,
    exec_seed: int,
    max_steps: int,
    args: ConditionCollectorArgs,
    client: Pi05Client,
    writer: ProbeDatasetWriter,
) -> bool:
    random.seed(exec_seed)
    np.random.seed(exec_seed)
    exec_rng = np.random.default_rng(exec_seed)
    env.seed(exec_seed)
    env.reset()
    init_state_slot = int(args.init_state_index) % len(initial_states)
    obs = env.set_init_state(initial_states[init_state_slot])
    action_plan: collections.deque[np.ndarray] = collections.deque()
    pending: list[ProbeCallRecord] = []
    rewards: list[float] = []
    done = False
    error: str | None = None
    t = 0
    replan_idx = 0
    episode_id = _episode_id(batch, condition, trial_idx, exec_seed)
    instruction = str(condition["instruction"])

    try:
        while t < max_steps + args.num_steps_wait:
            if t < args.num_steps_wait:
                obs, reward, done, _ = env.step(LIBERO_DUMMY_ACTION)
                rewards.append(float(reward))
                t += 1
                continue

            if not action_plan:
                probe_seed = _probe_seed(
                    args.probe_seed_start,
                    base_index,
                    condition_index,
                    trial_idx,
                    replan_idx,
                )
                record = _sample_and_enqueue_action(
                    obs=obs,
                    task_suite_name=task_suite_name,
                    task_id=task_id,
                    task_name=str(batch.get("task_name") or canonical_instruction),
                    trial_idx=trial_idx,
                    step_idx=t,
                    episode_id=episode_id,
                    instruction=instruction,
                    batch=batch,
                    condition=condition,
                    base_index=base_index,
                    init_state_slot=init_state_slot,
                    exec_seed=exec_seed,
                    probe_seed=probe_seed,
                    replan_idx=replan_idx,
                    args=args,
                    client=client,
                    writer=writer,
                    exec_rng=exec_rng,
                    action_plan=action_plan,
                )
                pending.append(record)
                replan_idx += 1

            action = np.asarray(action_plan.popleft())
            obs, reward, done, _ = env.step(action.tolist())
            rewards.append(float(reward))
            if done:
                break
            t += 1
    except Exception as exc:  # pragma: no cover - simulator/runtime dependent
        error = repr(exc)
        logging.exception("Episode failed: %s", episode_id)

    success = bool(done)
    reward_sum = float(np.sum(rewards)) if rewards else 0.0
    for record in pending:
        record.final_success = success
        record.episode_done = bool(done)
        record.episode_steps = int(t)
        record.reward_sum = reward_sum
        record.error = error
        writer.append_record(record)
    logging.info(
        "Episode %s success=%s calls=%d total_records=%d",
        episode_id,
        success,
        len(pending),
        writer.num_records,
    )
    return success


def _sample_and_enqueue_action(
    *,
    obs: dict[str, Any],
    task_suite_name: str,
    task_id: int,
    task_name: str,
    trial_idx: int,
    step_idx: int,
    episode_id: str,
    instruction: str,
    batch: dict[str, Any],
    condition: dict[str, Any],
    base_index: int,
    init_state_slot: int,
    exec_seed: int,
    probe_seed: int,
    replan_idx: int,
    args: ConditionCollectorArgs,
    client: Pi05Client,
    writer: ProbeDatasetWriter,
    exec_rng: np.random.Generator,
    action_plan: collections.deque[np.ndarray],
) -> ProbeCallRecord:
    random.seed(probe_seed)
    np.random.seed(probe_seed)
    element = build_policy_element(obs, instruction, resize_size=args.resize_size)
    samples = client.sample_action_chunks(element, k=args.k_samples)
    selected = _select_action_index(samples, args.action_selection, exec_rng)
    selected_chunk = samples[selected]
    if len(selected_chunk) < args.replan_steps:
        raise ValueError(
            f"Policy returned chunk length {len(selected_chunk)}, smaller than replan_steps={args.replan_steps}."
        )
    action_plan.extend(np.asarray(action) for action in selected_chunk[: args.replan_steps])

    record_id = writer.allocate_record_id()
    obs_path = writer.write_observation(record_id, observation_arrays_for_record(obs, element))
    actions_path = writer.write_action_samples(record_id, samples, selected)
    condition_id = str(condition["condition_id"])
    condition_type = str(condition.get("condition_type") or condition_id)
    return ProbeCallRecord(
        record_id=record_id,
        episode_id=episode_id,
        task_suite=task_suite_name,
        task_id=task_id,
        task_name=task_name,
        trial_idx=trial_idx,
        step_idx=step_idx,
        instruction=instruction,
        policy_name=args.policy_name,
        checkpoint_uri=args.checkpoint_uri,
        seed=exec_seed,
        k_samples=args.k_samples,
        action_selection=args.action_selection,
        selected_sample_index=int(selected),
        obs_path=obs_path,
        action_samples_path=actions_path,
        final_success=False,
        episode_done=False,
        episode_steps=0,
        reward_sum=0.0,
        base_id=str(batch.get("base_id") or f"task{task_id:04d}"),
        condition_id=condition_id,
        condition_type=condition_type,
        exec_seed=exec_seed,
        probe_seed=probe_seed,
        replan_idx=replan_idx,
        perturbation_type=condition_type,
        metadata={
            "dataset": batch.get("dataset", "LIBERO-Plus"),
            "condition_batch_id": batch.get("condition_batch_id"),
            "condition_source": condition.get("source"),
            "condition_rationale": condition.get("rationale"),
            "degradation_type": condition.get("degradation_type"),
            "original_instruction": batch.get("original_instruction"),
            "raw_benchmark_instruction": batch.get("raw_benchmark_instruction"),
            "classification": batch.get("classification"),
            "base_index": base_index,
            "condition_index": condition.get("condition_index"),
            "condition_order_index": condition.get("_order_index"),
            "init_state_index": init_state_slot,
            "probe_seed_stream": args.probe_seed_start,
            "exec_seed_stream": args.exec_seed_start,
            "probe_seed_applied_locally": True,
            "policy_server_seed_control": "not_exposed_by_client",
            "action_sample_shape": list(samples.shape),
            "host": args.host,
            "port": args.port,
        },
    )


def _read_condition_batches(path: Path, max_base_states: int | None) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            _validate_condition_batch(payload, path, line_no)
            batches.append(payload)
            if max_base_states is not None and len(batches) >= max_base_states:
                break
    return batches


def _validate_condition_batch(payload: dict[str, Any], path: Path, line_no: int) -> None:
    if "task_id" not in payload:
        raise ValueError(f"{path}:{line_no}: missing task_id")
    conditions = payload.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError(f"{path}:{line_no}: missing non-empty conditions list")
    for condition in conditions:
        if not isinstance(condition, dict) or not condition.get("condition_id") or not condition.get("instruction"):
            raise ValueError(f"{path}:{line_no}: each condition needs condition_id and instruction")


def _selected_conditions(batch: dict[str, Any], wanted: tuple[str, ...]) -> list[dict[str, Any]]:
    wanted_set = set(wanted)
    selected: list[dict[str, Any]] = []
    for order_index, condition in enumerate(batch["conditions"]):
        condition_type = str(condition.get("condition_type") or condition.get("condition_id"))
        condition_id = str(condition.get("condition_id"))
        if condition_type in wanted_set or condition_id in wanted_set:
            item = dict(condition)
            item["_order_index"] = order_index
            selected.append(item)
    return selected


def _completed_episode_ids(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    completed: set[str] = set()
    with manifest_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("episode_id"):
                completed.add(str(payload["episode_id"]))
    return completed


def _episode_id(batch: dict[str, Any], condition: dict[str, Any], trial_idx: int, exec_seed: int) -> str:
    base_id = str(batch.get("base_id") or f"task{int(batch['task_id']):04d}")
    condition_id = str(condition["condition_id"])
    return f"{base_id}_{condition_id}_trial{trial_idx:04d}_exec{exec_seed}"


def _exec_seed(seed_start: int, base_index: int, condition_index: int, trial_idx: int) -> int:
    return int(seed_start) + base_index * 10_000 + condition_index * 100 + trial_idx


def _probe_seed(seed_start: int, base_index: int, condition_index: int, trial_idx: int, replan_idx: int) -> int:
    return int(seed_start) + base_index * 1_000_000 + condition_index * 100_000 + trial_idx * 1_000 + replan_idx


def _select_action_index(samples: np.ndarray, strategy: str, rng: np.random.Generator) -> int:
    if strategy == "first":
        return 0
    if strategy == "random":
        return int(rng.integers(0, samples.shape[0]))
    raise ValueError(f"Unknown action_selection strategy: {strategy}")


def _parse_condition_types(value: str | None) -> tuple[str, ...]:
    if not value:
        return ("original", "better", "worse")
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _report(
    args: ConditionCollectorArgs,
    attempted: int,
    written_episodes: int,
    success_episodes: int,
    failure_episodes: int,
    num_condition_batches: int,
) -> dict[str, Any]:
    return {
        "conditions": args.conditions,
        "output_dir": args.output_dir,
        "manifest": str(Path(args.output_dir) / args.manifest_name),
        "task_suite_name": args.task_suite_name,
        "condition_types": list(args.condition_types),
        "num_condition_batches": num_condition_batches,
        "num_attempted_episodes": attempted,
        "num_written_episodes": written_episodes,
        "num_success_episodes": success_episodes,
        "num_failure_episodes": failure_episodes,
        "episode_success_fraction": success_episodes / written_episodes if written_episodes else 0.0,
        "exec_seed_start": args.exec_seed_start,
        "probe_seed_start": args.probe_seed_start,
        "k_samples": args.k_samples,
        "num_trials_per_condition": args.num_trials_per_condition,
    }


def _parse_args() -> ConditionCollectorArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--condition-types", default="original,better,worse")
    parser.add_argument("--num-trials-per-condition", type=int, default=8)
    parser.add_argument("--init-state-index", type=int, default=0)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--k-samples", type=int, default=32)
    parser.add_argument("--max-base-states", type=int)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--action-selection", choices=["random", "first"], default="random")
    parser.add_argument("--exec-seed-start", type=int, default=700_000)
    parser.add_argument("--probe-seed-start", type=int, default=100_000)
    parser.add_argument("--manifest-name", default="records.jsonl")
    parser.add_argument("--checkpoint-uri", default="gs://openpi-assets/checkpoints/pi05_libero")
    parser.add_argument("--policy-name", default="pi05_libero")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    namespace = parser.parse_args()
    return ConditionCollectorArgs(
        conditions=namespace.conditions,
        output_dir=namespace.output_dir,
        host=namespace.host,
        port=namespace.port,
        task_suite_name=namespace.task_suite_name,
        condition_types=_parse_condition_types(namespace.condition_types),
        num_trials_per_condition=namespace.num_trials_per_condition,
        init_state_index=namespace.init_state_index,
        num_steps_wait=namespace.num_steps_wait,
        resize_size=namespace.resize_size,
        replan_steps=namespace.replan_steps,
        k_samples=namespace.k_samples,
        max_base_states=namespace.max_base_states,
        max_episodes=namespace.max_episodes,
        action_selection=namespace.action_selection,
        exec_seed_start=namespace.exec_seed_start,
        probe_seed_start=namespace.probe_seed_start,
        manifest_name=namespace.manifest_name,
        checkpoint_uri=namespace.checkpoint_uri,
        policy_name=namespace.policy_name,
        resume=namespace.resume,
    )


if __name__ == "__main__":
    main()
