"""Report Week3 condition-selection quality for a set of training runs.

Each run selects a seed by validation NLL; this script reads that seed's
per-group selection results and reports three metrics on the test split:

    accuracy   fraction of groups whose predicted condition is (tied) best
    sel_rate   mean empirical success rate of the condition the model picked
    regret     mean (oracle rate - selected rate), i.e. the success lost by
               picking a suboptimal condition

Accuracy alone says how often the model was right; ``sel_rate`` and ``regret``
say how much that was worth.  A model can win on accuracy yet lose on regret,
because a wrong pick next to a near-tied best costs little while a confident
wrong pick costs a lot.

The trivial baselines (always pick one condition, uniform random) are printed
alongside, together with an exact McNemar test against "always original".
With 36-48 test groups a single group is worth 2-3 accuracy points, so an
unpaired difference of a few points is not evidence on its own.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Any

CONDITIONS = ("original", "better", "worse")
TIE_TOLERANCE = 1e-12


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def load_groups(
    dataset_dir: Path,
    splits_path: Path | None = None,
) -> tuple[dict[str, dict[str, tuple[int, int]]], dict[str, str]]:
    groups: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
    for record in read_jsonl(dataset_dir / "manifest.jsonl"):
        groups[str(record["group_id"])][str(record["condition_id"])] = (
            int(record["eval_successes"]),
            int(record["eval_trials"]),
        )
    splits = read_json(splits_path or (dataset_dir / "splits.json"))
    split_of: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        for group_id in splits.get("groups", {}).get(split, []):
            split_of[str(group_id)] = split
    return dict(groups), split_of


def success_rate(group: dict[str, tuple[int, int]], condition: str) -> float:
    successes, trials = group[condition]
    return successes / trials


def oracle_rate(group: dict[str, tuple[int, int]]) -> float:
    return max(success_rate(group, condition) for condition in CONDITIONS)


def tied_best(group: dict[str, tuple[int, int]]) -> set[str]:
    best = oracle_rate(group)
    return {
        condition
        for condition in CONDITIONS
        if abs(success_rate(group, condition) - best) <= TIE_TOLERANCE
    }


def is_auto_correct(group: dict[str, tuple[int, int]]) -> bool:
    """True when every condition ties at the best rate, so any pick scores."""
    rates = {success_rate(group, condition) for condition in CONDITIONS}
    return len(rates) == 1


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def mcnemar_exact(both_model_only: int, both_baseline_only: int) -> float:
    discordant = both_model_only + both_baseline_only
    if discordant == 0:
        return 1.0
    smaller = min(both_model_only, both_baseline_only)
    tail = sum(comb(discordant, i) for i in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def collect_model_rows(
    run_dirs: list[Path],
    model_kind: str,
) -> tuple[dict[str, list[float]], dict[str, float], dict[str, list[dict]]]:
    """Return (per-method metric lists, per-method per-run accuracy, rows)."""
    accuracy: dict[str, list[float]] = defaultdict(list)
    sel_rate: dict[str, list[float]] = defaultdict(list)
    regret: dict[str, list[float]] = defaultdict(list)
    per_run_accuracy: dict[str, list[float]] = defaultdict(list)
    per_group: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        selection_path = run_dir / "selection_results_by_seed.jsonl"
        if not summary_path.exists() or not selection_path.exists():
            print(f"skipping {run_dir.name}: missing summary/selection results")
            continue
        summary = read_json(summary_path)
        for method, method_payload in summary.get("methods", {}).items():
            payload = method_payload.get(model_kind)
            if not payload:
                continue
            selected_seed = str(payload["selected_seed"])
            tag = f"{method}/{model_kind}"
            hits: list[float] = []
            rates: list[float] = []
            regrets: list[float] = []
            for row in read_jsonl(selection_path):
                if row.get("method") != tag:
                    continue
                if str(row.get("seed")) != selected_seed:
                    continue
                if row.get("split") != "test":
                    continue
                hit = 1.0 if row.get("selection_correct") else 0.0
                hits.append(hit)
                rates.append(float(row["selected_empirical_rate"]))
                regrets.append(float(row["empirical_regret"]))
                per_group[method][str(row["group_id"])].append(hit)
            if hits:
                accuracy[method].append(mean(hits))
                sel_rate[method].append(mean(rates))
                regret[method].append(mean(regrets))
                per_run_accuracy[method].append(mean(hits))
    return accuracy, per_group, {"sel_rate": sel_rate, "regret": regret}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--splits-path",
        type=Path,
        default=None,
        help=(
            "override the split file; defaults to <dataset-dir>/splits.json. "
            "Required when evaluating runs trained on an alternative group "
            "assignment, otherwise the wrong groups are scored as test."
        ),
    )
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--model-kind", default="mlp")
    parser.add_argument(
        "--exclude-tied",
        action="store_true",
        help=(
            "drop groups where all three conditions tie at the best rate; every "
            "method scores on those automatically, which dilutes differences"
        ),
    )
    args = parser.parse_args()

    groups, split_of = load_groups(args.dataset_dir, args.splits_path)
    test_groups = sorted(g for g in groups if split_of.get(g) == "test")
    if args.exclude_tied:
        excluded = sum(1 for g in test_groups if is_auto_correct(groups[g]))
        test_groups = [g for g in test_groups if not is_auto_correct(groups[g])]
        if excluded:
            print(f"excluded {excluded} all-tied test groups (--exclude-tied)")
            print()
    if not test_groups:
        raise SystemExit("no test groups to evaluate")

    run_dirs = sorted(args.run_dir)
    accuracy, per_group, extras = collect_model_rows(run_dirs, args.model_kind)
    if not accuracy:
        raise SystemExit("no model results found in the supplied run dirs")

    print(
        f"=== test groups = {len(test_groups)} | "
        f"oracle mean rate = {mean([oracle_rate(groups[g]) for g in test_groups]):.4f} | "
        f"runs = {len(run_dirs)} ==="
    )
    print()
    print("%-24s %10s %10s %10s" % ("method", "accuracy", "sel_rate", "regret"))
    print("-" * 58)

    for condition in CONDITIONS:
        acc = mean(
            [1.0 if abs(success_rate(groups[g], condition) - oracle_rate(groups[g])) <= TIE_TOLERANCE else 0.0
             for g in test_groups]
        )
        rate = mean([success_rate(groups[g], condition) for g in test_groups])
        reg = mean([oracle_rate(groups[g]) - success_rate(groups[g], condition) for g in test_groups])
        print("%-24s %10.4f %10.4f %10.4f" % (f"fixed {condition}", acc, rate, reg))

    acc = mean([len(tied_best(groups[g])) / 3.0 for g in test_groups])
    rate = mean([mean([success_rate(groups[g], c) for c in CONDITIONS]) for g in test_groups])
    reg = mean([mean([oracle_rate(groups[g]) - success_rate(groups[g], c) for c in CONDITIONS]) for g in test_groups])
    print("%-24s %10.4f %10.4f %10.4f" % ("uniform random", acc, rate, reg))
    print("-" * 58)

    ordered = sorted(accuracy, key=lambda m: -mean(accuracy[m]))
    for method in ordered:
        print(
            "%-24s %10.4f %10.4f %10.4f"
            % (
                f"{method}/{args.model_kind}",
                mean(accuracy[method]),
                mean(extras["sel_rate"][method]),
                mean(extras["regret"][method]),
            )
        )

    print()
    print("=== paired vs fixed original (majority over runs, exact McNemar) ===")
    print("%-24s %10s %10s %9s %10s" % ("method", "model", "orig", "delta", "p"))
    print("-" * 68)
    original_correct = {
        g: 1.0 if abs(success_rate(groups[g], "original") - oracle_rate(groups[g])) <= TIE_TOLERANCE else 0.0
        for g in test_groups
    }
    baseline_accuracy = mean([original_correct[g] for g in test_groups])
    for method in ordered:
        model_only = 0
        baseline_only = 0
        model_hits: list[float] = []
        for group_id in test_groups:
            hits = per_group[method].get(group_id)
            if not hits:
                continue
            majority = 1.0 if mean(hits) >= 0.5 else 0.0
            model_hits.append(majority)
            if majority == 1.0 and original_correct[group_id] == 0.0:
                model_only += 1
            elif majority == 0.0 and original_correct[group_id] == 1.0:
                baseline_only += 1
        model_accuracy = mean(model_hits)
        print(
            "%-24s %10.4f %10.4f %+9.4f %10.4f"
            % (
                f"{method}/{args.model_kind}",
                model_accuracy,
                baseline_accuracy,
                model_accuracy - baseline_accuracy,
                mcnemar_exact(model_only, baseline_only),
            )
        )


if __name__ == "__main__":
    main()
