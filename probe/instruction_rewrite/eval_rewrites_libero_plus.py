"""Evaluate rewritten instructions on LIBERO-Plus with the pi0.5 server."""

from __future__ import annotations

import argparse
import collections
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import tqdm

from probe.envs.libero_runner import (
    LIBERO_DUMMY_ACTION,
    build_policy_element,
    get_task_suite,
    make_libero_env,
    max_steps_for_suite,
)
from probe.policies.pi05_client import Pi05Client


def evaluate_rewrites(args: argparse.Namespace) -> dict[str, Any]:
    rewrites = _load_rewrite_jobs(
        Path(args.rewrites),
        max_tasks=args.max_tasks,
        max_rewrites_per_task=args.max_rewrites_per_task,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_attempt_keys(output_path) if args.resume else set()

    task_suite = get_task_suite(args.task_suite_name)
    max_steps = max_steps_for_suite(args.task_suite_name)
    client = Pi05Client(args.host, args.port)
    written = 0
    recovered = 0
    attempted = 0

    with output_path.open("a", encoding="utf-8") as handle:
        for job in tqdm.tqdm(rewrites, desc="rewrite jobs"):
            task_id = int(job["task_id"])
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, canonical_instruction = make_libero_env(task, seed=args.seed)
            try:
                for trial_offset in range(args.num_trials_per_rewrite):
                    attempt_key = _attempt_key(job, trial_offset, args.seed)
                    if attempt_key in completed:
                        continue
                    result = _run_one_rewrite_episode(
                        env=env,
                        initial_states=initial_states,
                        task_suite_name=args.task_suite_name,
                        task_id=task_id,
                        trial_idx=trial_offset,
                        seed=args.seed,
                        prompt=str(job["rewritten_instruction"]),
                        canonical_instruction=canonical_instruction,
                        client=client,
                        max_steps=max_steps,
                        num_steps_wait=args.num_steps_wait,
                        replan_steps=args.replan_steps,
                        k_samples=args.k_samples,
                        resize_size=args.resize_size,
                    )
                    result.update(job)
                    result["attempt_key"] = attempt_key
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                    written += 1
                    attempted += 1
                    if result.get("final_success") is True:
                        recovered += 1
            finally:
                try:
                    env.close()
                except Exception:
                    pass

    return {
        "rewrites": args.rewrites,
        "output": str(output_path),
        "num_rewrite_jobs": len(rewrites),
        "num_attempted": attempted,
        "num_written": written,
        "num_success": recovered,
        "success_fraction": recovered / attempted if attempted else 0.0,
    }


def _run_one_rewrite_episode(
    *,
    env: Any,
    initial_states: Any,
    task_suite_name: str,
    task_id: int,
    trial_idx: int,
    seed: int,
    prompt: str,
    canonical_instruction: str,
    client: Pi05Client,
    max_steps: int,
    num_steps_wait: int,
    replan_steps: int,
    k_samples: int,
    resize_size: int,
) -> dict[str, Any]:
    env.reset()
    obs = env.set_init_state(initial_states[trial_idx % len(initial_states)])
    action_plan: collections.deque[np.ndarray] = collections.deque()
    rewards: list[float] = []
    done = False
    error: str | None = None
    t = 0
    policy_calls = 0

    try:
        while t < max_steps + num_steps_wait:
            if t < num_steps_wait:
                obs, reward, done, _ = env.step(LIBERO_DUMMY_ACTION)
                rewards.append(float(reward))
                t += 1
                continue
            if not action_plan:
                element = build_policy_element(obs, prompt, resize_size=resize_size)
                samples = client.sample_action_chunks(element, k=k_samples)
                selected_chunk = np.asarray(samples[0])
                if len(selected_chunk) < replan_steps:
                    raise ValueError(
                        f"Policy returned chunk length {len(selected_chunk)}, smaller than replan_steps={replan_steps}."
                    )
                action_plan.extend(np.asarray(action) for action in selected_chunk[:replan_steps])
                policy_calls += 1
            action = np.asarray(action_plan.popleft())
            obs, reward, done, _ = env.step(action.tolist())
            rewards.append(float(reward))
            if done:
                break
            t += 1
    except Exception as exc:  # pragma: no cover - simulator/runtime dependent
        error = repr(exc)
        logging.exception("Rewrite episode failed for task_id=%s", task_id)

    return {
        "task_suite": task_suite_name,
        "task_id": task_id,
        "trial_idx": trial_idx,
        "seed": seed,
        "canonical_instruction": canonical_instruction,
        "evaluated_instruction": prompt,
        "final_success": bool(done),
        "episode_done": bool(done),
        "episode_steps": int(t),
        "reward_sum": float(np.sum(rewards)) if rewards else 0.0,
        "num_policy_calls": policy_calls,
        "error": error,
    }


def _load_rewrite_jobs(
    path: Path,
    max_tasks: int | None,
    max_rewrites_per_task: int | None,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    seen_tasks: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            task_id = int(payload["task_id"])
            if task_id not in seen_tasks:
                if max_tasks is not None and len(seen_tasks) >= max_tasks:
                    continue
                seen_tasks.add(task_id)
            rewrites = payload.get("rewrites") or []
            if max_rewrites_per_task is not None:
                rewrites = rewrites[: max_rewrites_per_task]
            for rewrite in rewrites:
                if not isinstance(rewrite, dict) or not rewrite.get("instruction"):
                    continue
                jobs.append(
                    {
                        "rewrite_batch_id": payload.get("rewrite_batch_id"),
                        "source_model": payload.get("model"),
                        "task_id": task_id,
                        "task_name": payload.get("task_name"),
                        "original_instruction": payload.get("original_instruction"),
                        "source_episode_ids": payload.get("source_episode_ids"),
                        "classification": payload.get("classification"),
                        "rewrite_index": rewrite.get("rewrite_index"),
                        "rewrite_type": rewrite.get("rewrite_type"),
                        "rewrite_rationale": rewrite.get("rationale"),
                        "rewritten_instruction": rewrite["instruction"],
                        "source_line_no": line_no,
                    }
                )
    return jobs


def _completed_attempt_keys(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    keys: set[str] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("attempt_key"):
                keys.add(str(payload["attempt_key"]))
    return keys


def _attempt_key(job: dict[str, Any], trial_offset: int, seed: int) -> str:
    return "|".join(
        [
            str(job.get("task_id")),
            str(job.get("rewrite_batch_id")),
            str(job.get("rewrite_index")),
            str(trial_offset),
            str(seed),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rewrites", required=True, help="Rewrite JSONL produced by rewrite_failed_tasks.")
    parser.add_argument("--output", required=True, help="Output evaluation JSONL path.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--max-rewrites-per-task", type=int)
    parser.add_argument("--num-trials-per-rewrite", type=int, default=1)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--k-samples", type=int, default=1)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    report = evaluate_rewrites(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
