"""Plot Week 3 train/validation/test NLL curves from selected MLP seeds.

Each run directory is one three-seed experiment.  For every method, this
script reads the selected seed recorded in ``summary.json`` and loads its
per-epoch ``history.json``.  The first three run directories are plotted
individually; the average plot averages the selected-seed curves across all
supplied run directories at each epoch.  Because early stopping can produce
different history lengths, each average uses only runs that reached that
epoch.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SPLITS = (
    ("train_nll", "train"),
    ("validation_nll", "validation"),
    ("test_nll", "test"),
)
DEFAULT_METHODS = (
    "A",
    "B",
    "S_K4",
    "S_K8",
    "S_K16",
    "S_K32",
    "B+S_K4",
    "B+S_K8",
    "B+S_K16",
    "B+S_K32",
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def read_history(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, list):
        raise ValueError(f"{path}: expected a JSON list")
    history: list[dict[str, float]] = []
    for row in values:
        if not isinstance(row, dict):
            raise ValueError(f"{path}: history row is not an object")
        required = {"epoch", "train_nll", "validation_nll", "test_nll"}
        missing = required - set(row)
        if missing:
            raise ValueError(
                f"{path}: missing {sorted(missing)}; "
                "rerun training with the updated train_week3_predictors.py"
            )
        history.append(
            {
                "epoch": int(row["epoch"]),
                "train_nll": float(row["train_nll"]),
                "validation_nll": float(row["validation_nll"]),
                "test_nll": float(row["test_nll"]),
            }
        )
    if not history:
        raise ValueError(f"{path}: empty history")
    return history


def selected_histories(
    run_dirs: list[Path],
    methods: tuple[str, ...],
    model_kind: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for run_dir in run_dirs:
        summary = read_json(run_dir / "summary.json")
        run_name = run_dir.name
        run_result: dict[str, dict[str, Any]] = {}
        summary_methods = summary.get("methods", {})
        if not isinstance(summary_methods, dict):
            raise ValueError(f"{run_dir / 'summary.json'}: missing methods object")
        for method in methods:
            method_payload = summary_methods.get(method)
            if not isinstance(method_payload, dict):
                raise ValueError(f"{run_dir}: method {method!r} is missing")
            model_payload = method_payload.get(model_kind)
            if not isinstance(model_payload, dict):
                raise ValueError(f"{run_dir}: {method}/{model_kind} is missing")
            selected_seed = int(model_payload["selected_seed"])
            history_path = (
                run_dir
                / "models"
                / method
                / model_kind
                / f"seed{selected_seed}"
                / "history.json"
            )
            run_result[method] = {
                "selected_seed": selected_seed,
                "history": read_history(history_path),
            }
        result[run_name] = run_result
    return result


def average_histories(
    histories: dict[str, dict[str, dict[str, Any]]],
    methods: tuple[str, ...],
) -> dict[str, list[dict[str, float]]]:
    averages: dict[str, list[dict[str, float]]] = {}
    for method in methods:
        by_epoch: dict[int, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for run_result in histories.values():
            for row in run_result[method]["history"]:
                epoch = int(row["epoch"])
                for key, _ in SPLITS:
                    by_epoch[epoch][key].append(float(row[key]))
        averages[method] = [
            {
                "epoch": float(epoch),
                **{
                    key: sum(values) / len(values)
                    for key, values in sorted(metrics.items())
                },
            }
            for epoch, metrics in sorted(by_epoch.items())
        ]
    return averages


def write_curve_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scope",
        "run",
        "method",
        "selected_seed",
        "epoch",
        "train_nll",
        "validation_nll",
        "test_nll",
        "num_runs_at_epoch",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def flatten_curve_data(
    histories: dict[str, dict[str, dict[str, Any]]],
    averages: dict[str, list[dict[str, float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_name, run_result in histories.items():
        for method, payload in run_result.items():
            for row in payload["history"]:
                rows.append(
                    {
                        "scope": "run",
                        "run": run_name,
                        "method": method,
                        "selected_seed": payload["selected_seed"],
                        "epoch": int(row["epoch"]),
                        "train_nll": row["train_nll"],
                        "validation_nll": row["validation_nll"],
                        "test_nll": row["test_nll"],
                        "num_runs_at_epoch": 1,
                    }
                )
    for method, rows_for_method in averages.items():
        for row in rows_for_method:
            epoch = int(row["epoch"])
            counts = []
            for run_result in histories.values():
                counts.append(
                    any(int(item["epoch"]) == epoch for item in run_result[method]["history"])
                )
            rows.append(
                {
                    "scope": "average",
                    "run": "average_all_runs",
                    "method": method,
                    "selected_seed": "",
                    "epoch": epoch,
                    "train_nll": row["train_nll"],
                    "validation_nll": row["validation_nll"],
                    "test_nll": row["test_nll"],
                    "num_runs_at_epoch": sum(counts),
                }
            )
    return rows


def load_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib. Install it in the active environment, "
            "then rerun this command."
        ) from exc
    return plt


def plot_run(
    plt: Any,
    run_name: str,
    run_result: dict[str, dict[str, Any]],
    methods: tuple[str, ...],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=True)
    colors = {
        method: color
        for method, color in zip(
            methods,
            plt.get_cmap("tab10").colors,
        )
    }
    for axis, (key, label) in zip(axes, SPLITS):
        for method in methods:
            payload = run_result[method]
            history = payload["history"]
            axis.plot(
                [row["epoch"] for row in history],
                [row[key] for row in history],
                label=f"{method} (s{payload['selected_seed']})",
                color=colors[method],
                linewidth=1.6,
            )
        axis.set_title(label)
        axis.set_xlabel("epoch")
        axis.set_ylabel("NLL")
        axis.grid(True, alpha=0.25)
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle(f"{run_name}: selected-seed loss curves", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_average(
    plt: Any,
    averages: dict[str, list[dict[str, float]]],
    methods: tuple[str, ...],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=True)
    colors = {
        method: color
        for method, color in zip(
            methods,
            plt.get_cmap("tab10").colors,
        )
    }
    for axis, (key, label) in zip(axes, SPLITS):
        for method in methods:
            rows = averages[method]
            axis.plot(
                [row["epoch"] for row in rows],
                [row[key] for row in rows],
                label=method,
                color=colors[method],
                linewidth=1.8,
            )
        axis.set_title(label)
        axis.set_xlabel("epoch")
        axis.set_ylabel("NLL")
        axis.grid(True, alpha=0.25)
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle("Average loss curves across all supplied experiments", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-kind", default="mlp")
    parser.add_argument("--first-n", type=int, default=3)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
    )
    args = parser.parse_args()

    run_dirs = [path.expanduser().resolve() for path in args.run_dir]
    if args.first_n < 1:
        raise ValueError("--first-n must be positive")
    if len(run_dirs) < args.first_n:
        raise ValueError(
            f"received {len(run_dirs)} run directories but first-n={args.first_n}"
        )
    methods = tuple(args.methods)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    histories = selected_histories(run_dirs, methods, args.model_kind)
    averages = average_histories(histories, methods)

    plt = load_matplotlib()
    first_dir = output_dir / "first3"
    first_dir.mkdir(parents=True, exist_ok=True)
    for run_name in list(histories)[: args.first_n]:
        plot_run(
            plt,
            run_name,
            histories[run_name],
            methods,
            first_dir / f"{run_name}_loss_curves.png",
        )
    plot_average(
        plt,
        averages,
        methods,
        output_dir / "average_9_runs_loss_curves.png",
    )

    rows = flatten_curve_data(histories, averages)
    write_curve_csv(output_dir / "loss_curves.csv", rows)
    (output_dir / "loss_curves.json").write_text(
        json.dumps(
            {
                "run_dirs": [str(path) for path in run_dirs],
                "model_kind": args.model_kind,
                "methods": list(methods),
                "first_n": args.first_n,
                "averaging": (
                    "selected seed per run, arithmetic mean at each epoch; "
                    "only runs with that epoch are included"
                ),
                "histories": histories,
                "averages": averages,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"plots written to: {output_dir}")
    print(f"first-{args.first_n} plots: {first_dir}")
    print(f"average plot: {output_dir / 'average_9_runs_loss_curves.png'}")
    print(f"curve data: {output_dir / 'loss_curves.csv'}")


if __name__ == "__main__":
    main()
