"""Collect labeled pi0.5 action-chunk samples on LIBERO."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import tqdm

from probe.config import cfg_get, load_config, parse_task_order
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
from probe.perturbations import maybe_perturb_instruction
from probe.policies.pi05_client import Pi05Client


@dataclasses.dataclass
class CollectorArgs:
    config: str | None = None
    host: str = "0.0.0.0"
    port: int = 18000
    task_suite_name: str = "libero_10"
    task_order: str = "all"
    num_trials_per_task: int = 50
    num_steps_wait: int = 10
    resize_size: int = 224
    replan_steps: int = 5
    target_records: int = 500
    k_samples: int = 32
    seed: int = 7
    action_selection: str = "random"
    perturbation_rate: float = 0.0
    perturbation_modes: tuple[str, ...] = ("none",)
    output_dir: str = "/data/week1/pi05_libero"
    manifest_name: str = "records.jsonl"
    checkpoint_uri: str = "gs://openpi-assets/checkpoints/pi05_libero"
    policy_name: str = "pi05_libero"


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    collect(args)


def collect(args: CollectorArgs) -> None:
    np.random.seed(args.seed)
    py_rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    writer = ProbeDatasetWriter(args.output_dir, manifest_name=args.manifest_name)
    task_suite = get_task_suite(args.task_suite_name)
    task_ids = parse_task_order(args.task_order, task_suite.n_tasks)
    max_steps = max_steps_for_suite(args.task_suite_name)
    client = Pi05Client(args.host, args.port)

    logging.info("Writing dataset to %s", args.output_dir)
    logging.info("Collecting until %d records exist.", args.target_records)

    for task_id in tqdm.tqdm(task_ids, desc="tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = make_libero_env(task, seed=args.seed)
        try:
            for trial_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=f"task {task_id}", leave=False):
                if writer.num_records >= args.target_records:
                    logging.info("Target reached: %d records.", writer.num_records)
                    return
                _run_episode(
                    env=env,
                    initial_states=initial_states,
                    task_suite_name=args.task_suite_name,
                    task_id=task_id,
                    task_name=task_description,
                    trial_idx=trial_idx,
                    max_steps=max_steps,
                    args=args,
                    client=client,
                    writer=writer,
                    py_rng=py_rng,
                    np_rng=np_rng,
                )
        finally:
            try:
                env.close()
            except Exception:
                pass

    logging.info("Collection ended with %d records.", writer.num_records)


def _run_episode(
    *,
    env: Any,
    initial_states: Any,
    task_suite_name: str,
    task_id: int,
    task_name: str,
    trial_idx: int,
    max_steps: int,
    args: CollectorArgs,
    client: Pi05Client,
    writer: ProbeDatasetWriter,
    py_rng: random.Random,
    np_rng: np.random.Generator,
) -> None:
    env.reset()
    obs = env.set_init_state(initial_states[trial_idx % len(initial_states)])
    action_plan: collections.deque[np.ndarray] = collections.deque()
    pending: list[ProbeCallRecord] = []
    rewards: list[float] = []
    done = False
    error: str | None = None
    t = 0
    episode_id = f"task{task_id:03d}_trial{trial_idx:04d}_seed{args.seed}"

    try:
        while t < max_steps + args.num_steps_wait:
            if t < args.num_steps_wait:
                obs, reward, done, _ = env.step(LIBERO_DUMMY_ACTION)
                rewards.append(float(reward))
                t += 1
                continue

            if not action_plan:
                record = _sample_and_enqueue_action(
                    obs=obs,
                    task_suite_name=task_suite_name,
                    task_id=task_id,
                    task_name=task_name,
                    trial_idx=trial_idx,
                    step_idx=t,
                    episode_id=episode_id,
                    args=args,
                    client=client,
                    writer=writer,
                    py_rng=py_rng,
                    np_rng=np_rng,
                    action_plan=action_plan,
                )
                pending.append(record)

            action = np.asarray(action_plan.popleft())
            obs, reward, done, _ = env.step(action.tolist())
            rewards.append(float(reward))
            if done:
                break
            t += 1
    except Exception as exc:  # pragma: no cover - depends on simulator/runtime
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


def _sample_and_enqueue_action(
    *,
    obs: dict[str, Any],
    task_suite_name: str,
    task_id: int,
    task_name: str,
    trial_idx: int,
    step_idx: int,
    episode_id: str,
    args: CollectorArgs,
    client: Pi05Client,
    writer: ProbeDatasetWriter,
    py_rng: random.Random,
    np_rng: np.random.Generator,
    action_plan: collections.deque[np.ndarray],
) -> ProbeCallRecord:
    instruction, perturbation_type = maybe_perturb_instruction(
        task_name,
        py_rng,
        rate=args.perturbation_rate,
        modes=args.perturbation_modes,
    )
    element = build_policy_element(obs, instruction, resize_size=args.resize_size)
    samples = client.sample_action_chunks(element, k=args.k_samples)
    selected = _select_action_index(samples, args.action_selection, np_rng)
    selected_chunk = samples[selected]
    if len(selected_chunk) < args.replan_steps:
        raise ValueError(
            f"Policy returned chunk length {len(selected_chunk)}, smaller than replan_steps={args.replan_steps}."
        )
    action_plan.extend(np.asarray(action) for action in selected_chunk[: args.replan_steps])

    record_id = writer.allocate_record_id()
    obs_path = writer.write_observation(record_id, observation_arrays_for_record(obs, element))
    actions_path = writer.write_action_samples(record_id, samples, selected)
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
        seed=args.seed,
        k_samples=args.k_samples,
        action_selection=args.action_selection,
        selected_sample_index=int(selected),
        obs_path=obs_path,
        action_samples_path=actions_path,
        final_success=False,
        episode_done=False,
        episode_steps=0,
        reward_sum=0.0,
        perturbation_type=perturbation_type,
        metadata={
            "action_sample_shape": list(samples.shape),
            "original_instruction": task_name,
            "host": args.host,
            "port": args.port,
        },
    )


def _select_action_index(samples: np.ndarray, strategy: str, rng: np.random.Generator) -> int:
    if strategy == "first":
        return 0
    if strategy == "random":
        return int(rng.integers(0, samples.shape[0]))
    raise ValueError(f"Unknown action_selection strategy: {strategy}")


def _parse_args() -> CollectorArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--task-suite-name")
    parser.add_argument("--task-order")
    parser.add_argument("--num-trials-per-task", type=int)
    parser.add_argument("--num-steps-wait", type=int)
    parser.add_argument("--resize-size", type=int)
    parser.add_argument("--replan-steps", type=int)
    parser.add_argument("--target-records", type=int)
    parser.add_argument("--k-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--action-selection", choices=["random", "first"])
    parser.add_argument("--perturbation-rate", type=float)
    parser.add_argument("--perturbation-modes")
    parser.add_argument("--output-dir")
    parser.add_argument("--manifest-name")
    parser.add_argument("--checkpoint-uri")
    parser.add_argument("--policy-name")
    namespace = parser.parse_args()

    config: dict[str, Any] = load_config(namespace.config) if namespace.config else {}
    args = CollectorArgs(config=namespace.config)
    mappings = {
        "host": "runtime.host",
        "port": "runtime.port",
        "task_suite_name": "libero.task_suite_name",
        "task_order": "libero.task_order",
        "num_trials_per_task": "libero.num_trials_per_task",
        "num_steps_wait": "libero.num_steps_wait",
        "resize_size": "libero.resize_size",
        "replan_steps": "libero.replan_steps",
        "target_records": "collection.target_records",
        "k_samples": "collection.k_samples",
        "seed": "collection.seed",
        "action_selection": "collection.action_selection",
        "perturbation_rate": "collection.perturbation_rate",
        "perturbation_modes": "collection.perturbation_modes",
        "output_dir": "output.dataset_dir",
        "manifest_name": "output.manifest_name",
        "checkpoint_uri": "runtime.checkpoint_uri",
        "policy_name": "runtime.policy_config",
    }
    for field_name, dotted_key in mappings.items():
        cli_value = getattr(namespace, _arg_name(field_name), None)
        value = cli_value if cli_value is not None else cfg_get(config, dotted_key, getattr(args, field_name))
        if field_name == "perturbation_modes":
            value = _parse_modes(value)
        setattr(args, field_name, value)
    return args


def _arg_name(field_name: str) -> str:
    return field_name


def _parse_modes(value: Any) -> tuple[str, ...]:
    if value is None:
        return ("none",)
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    raise ValueError(f"Unsupported perturbation_modes: {value!r}")


if __name__ == "__main__":
    main()
