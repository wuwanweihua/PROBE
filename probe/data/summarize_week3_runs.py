"""Summarize Week 3 results across independent training seeds.

Each supplied run directory may contain several seeds. The script prints the
per-seed probability metrics and tie-aware condition-selection accuracy, then
prints the arithmetic mean across all supplied seed runs.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _selection_metrics(rows: list[dict[str, Any]]) -> dict[str, float] | None:
    if not rows:
        return None
    return {
        "tie_aware_accuracy": sum(
            bool(row.get("selection_correct")) for row in rows
        )
        / len(rows),
        "groups": float(len(rows)),
    }


def collect_rows(run_dirs: list[Path]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        selection_path = run_dir / "selection_results_by_seed.jsonl"
        summary = _read_json(summary_path)
        if not selection_path.exists():
            raise FileNotFoundError(
                f"{selection_path} is missing; rerun training with the updated code"
            )

        selection_by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in _read_jsonl(selection_path):
            selection_by_key[
                (str(row["method"]), int(row["seed"]))
            ].append(row)

        for method, method_payload in summary.get("methods", {}).items():
            mlp = method_payload.get("mlp")
            if not mlp:
                continue
            for seed_text, seed_payload in mlp.get("seeds", {}).items():
                seed = int(seed_text)
                metrics = seed_payload["metrics"]
                selection = _selection_metrics(
                    selection_by_key.get((f"{method}/mlp", seed), [])
                )
                output.append(
                    {
                        "run": run_dir.name,
                        "seed": seed,
                        "method": f"{method}/mlp",
                        "train_nll": float(metrics["train"]["binomial_nll"]),
                        "val_nll": float(metrics["validation"]["binomial_nll"]),
                        "test_nll": float(metrics["test"]["binomial_nll"]),
                        "test_brier": float(metrics["test"]["brier_exec"]),
                        "test_rate_mse": float(metrics["test"]["rate_mse"]),
                        "tie_aware_accuracy": (
                            None
                            if selection is None
                            else selection["tie_aware_accuracy"]
                        ),
                        "selection_groups": (
                            0 if selection is None else int(selection["groups"])
                        ),
                    }
                )
    return output


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row[key] is not None]
    return sum(values) / len(values) if values else None


def _fmt(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def render_report(rows: list[dict[str, Any]], expected_seeds: int | None) -> str:
    run_seed_keys = {(row["run"], row["seed"]) for row in rows}
    if expected_seeds is not None and len(run_seed_keys) != expected_seeds:
        raise ValueError(
            f"expected {expected_seeds} run-seed groups, found {len(run_seed_keys)}"
        )

    lines = [
        "# Week 3 Seed Summary",
        "",
        f"- Independent run-seed groups: **{len(run_seed_keys)}**",
        "- Selection metric: **tie-aware accuracy**",
        "",
        "## Per-Seed Loss Metrics",
        "",
        "| run | seed | method | train NLL | val NLL | test NLL | test Brier | test rate MSE |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item["method"], item["run"], item["seed"])):
        lines.append(
            f"| {row['run']} | {row['seed']} | {row['method']} | "
            f"{_fmt(row['train_nll'])} | {_fmt(row['val_nll'])} | "
            f"{_fmt(row['test_nll'])} | {_fmt(row['test_brier'])} | "
            f"{_fmt(row['test_rate_mse'])} |"
        )

    lines.extend(
        [
            "",
            "## Per-Seed Tie-Aware Accuracy",
            "",
            "| run | seed | method | groups | tie-aware accuracy |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda item: (item["method"], item["run"], item["seed"])):
        lines.append(
            f"| {row['run']} | {row['seed']} | {row['method']} | "
            f"{row['selection_groups']} | {_fmt(row['tie_aware_accuracy'])} |"
        )

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    lines.extend(
        [
            "",
            "## Mean Across All Seed Runs",
            "",
            "| method | seed runs | mean train NLL | mean val NLL | mean test NLL | mean test Brier | mean test rate MSE | mean tie-aware accuracy |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in sorted(by_method):
        method_rows = by_method[method]
        lines.append(
            f"| {method} | {len(method_rows)} | "
            f"{_fmt(_mean(method_rows, 'train_nll'))} | "
            f"{_fmt(_mean(method_rows, 'val_nll'))} | "
            f"{_fmt(_mean(method_rows, 'test_nll'))} | "
            f"{_fmt(_mean(method_rows, 'test_brier'))} | "
            f"{_fmt(_mean(method_rows, 'test_rate_mse'))} | "
            f"{_fmt(_mean(method_rows, 'tie_aware_accuracy'))} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--expected-seeds",
        type=int,
        default=9,
        help="Expected number of unique run-seed groups; use 0 to disable.",
    )
    args = parser.parse_args()

    rows = collect_rows([path.expanduser().resolve() for path in args.run_dir])
    report = render_report(rows, None if args.expected_seeds == 0 else args.expected_seeds)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"report written to: {args.output}")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
