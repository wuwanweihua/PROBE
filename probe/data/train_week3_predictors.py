"""Train Week 3 success-probability predictors.

The input is the post-processed Week 3 dataset produced by
``probe.data.build_week3_dataset``.  Each row represents one
state-condition pair and contains an observed success count ``s`` out of
``N`` executions.  The predictors return one logit; the sigmoid of that logit
is the estimated episode success probability.

The command trains both a regularized linear head and a two-hidden-layer MLP
for A, B, S, and B+S feature families.  Preprocessing is fitted on the fixed
training split only.  The implementation intentionally uses full-batch
training because the current dataset is small and the objective is an
aggregate binomial NLL.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:  # Keep feature loading and documentation tools usable without PyTorch.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised on lightweight local envs.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)
DEFAULT_METHODS = ("A", "B", "S", "B+S")
DEFAULT_S_K = (4, 8, 16, 32)
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_MODEL_TYPE = "mlp"
DEFAULT_PATIENCE = 40
EPSILON = 1e-6


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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def require_torch() -> Any:
    if torch is None:
        raise RuntimeError(
            "Training requires PyTorch. Install it in the active environment "
            "before running probe.data.train_week3_predictors."
        )
    return torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def resolve_feature_path(dataset_dir: Path, value: str | None) -> Path:
    if not value:
        raise KeyError("record has no feature path")
    path = Path(value)
    return path if path.is_absolute() else dataset_dir / path


def load_npz_feature(
    path: Path,
    key: str,
    record_id: str,
    *,
    allow_nan: bool = False,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"{record_id}: feature file not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        if key not in data:
            raise KeyError(f"{record_id}: {path} is missing {key}")
        feature = np.asarray(data[key], dtype=np.float32)
    if feature.ndim != 1:
        raise ValueError(f"{record_id}: expected 1-D {key}, got {feature.shape}")
    if not allow_nan and not np.isfinite(feature).all():
        raise ValueError(f"{record_id}: {key} contains NaN or Inf")
    return feature


def load_s_feature(
    dataset_dir: Path,
    record: dict[str, Any],
    k: int,
) -> np.ndarray:
    record_id = str(record["record_id"])
    paths = record.get("s_feature_paths") or {}
    path = resolve_feature_path(dataset_dir, paths.get(str(k)))
    return load_npz_feature(path, "s_feature", record_id, allow_nan=False)


def load_raw_feature(
    dataset_dir: Path,
    record: dict[str, Any],
    method: str,
    s_k: int | None,
) -> np.ndarray:
    record_id = str(record["record_id"])
    if method == "A":
        return load_npz_feature(
            resolve_feature_path(dataset_dir, record.get("a_feature_path")),
            "a_feature",
            record_id,
        )
    if method == "B":
        return load_npz_feature(
            resolve_feature_path(dataset_dir, record.get("b_feature_path")),
            "b_feature",
            record_id,
        )
    if s_k is None:
        raise ValueError(f"{method} requires --s-k")
    s_feature = load_s_feature(dataset_dir, record, s_k)
    if method == "S":
        return s_feature
    if method == "B+S":
        b_feature = load_npz_feature(
            resolve_feature_path(dataset_dir, record.get("b_feature_path")),
            "b_feature",
            record_id,
        )
        return np.concatenate([b_feature, s_feature]).astype(np.float32)
    raise ValueError(f"Unknown method {method!r}")


def load_split_map(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        for record_id in payload.get("record_ids", {}).get(split, []):
            if str(record_id) in result:
                raise ValueError(f"record appears in multiple splits: {record_id}")
            result[str(record_id)] = split
    return result


@dataclass
class DatasetArrays:
    records: list[dict[str, Any]]
    features: np.ndarray
    successes: np.ndarray
    trials: np.ndarray
    splits: np.ndarray

    @property
    def split_indices(self) -> dict[str, np.ndarray]:
        return {
            split: np.flatnonzero(self.splits == split)
            for split in ("train", "validation", "test")
        }


def load_dataset_arrays(
    dataset_dir: Path,
    method: str,
    s_k: int | None,
) -> DatasetArrays:
    manifest_path = dataset_dir / "manifest.jsonl"
    splits_path = dataset_dir / "splits.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Week 3 manifest not found: {manifest_path}")
    if not splits_path.exists():
        raise FileNotFoundError(f"Week 3 splits not found: {splits_path}")

    records = read_jsonl(manifest_path)
    split_map = load_split_map(splits_path)
    features: list[np.ndarray] = []
    successes: list[float] = []
    trials: list[float] = []
    split_names: list[str] = []
    feature_dim: int | None = None

    for record in records:
        record_id = str(record.get("record_id") or "")
        if not record_id:
            raise ValueError("manifest record has empty record_id")
        if record_id not in split_map:
            raise ValueError(f"{record_id}: missing from splits.json")
        success = record.get("eval_successes")
        trial = record.get("eval_trials")
        if success is None or trial is None:
            raise ValueError(f"{record_id}: missing eval_successes/eval_trials")
        success = int(success)
        trial = int(trial)
        if trial <= 0 or success < 0 or success > trial:
            raise ValueError(f"{record_id}: invalid label successes={success}, trials={trial}")
        feature = load_raw_feature(dataset_dir, record, method, s_k)
        if feature_dim is None:
            feature_dim = int(feature.shape[0])
        elif int(feature.shape[0]) != feature_dim:
            raise ValueError(
                f"{record_id}: feature dimension {feature.shape[0]} != {feature_dim}"
            )
        features.append(feature)
        successes.append(float(success))
        trials.append(float(trial))
        split_names.append(split_map[record_id])

    if not features:
        raise ValueError(f"{dataset_dir}: empty Week 3 manifest")
    return DatasetArrays(
        records=records,
        features=np.stack(features).astype(np.float32),
        successes=np.asarray(successes, dtype=np.float32),
        trials=np.asarray(trials, dtype=np.float32),
        splits=np.asarray(split_names),
    )


@dataclass
class FeaturePreprocessor:
    """Train-only standardization followed by optional train-only PCA."""

    raw_mean: np.ndarray
    raw_scale: np.ndarray
    pca_mean: np.ndarray | None
    components: np.ndarray | None
    output_mean: np.ndarray
    output_scale: np.ndarray

    @staticmethod
    def _fit_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = values.mean(axis=0).astype(np.float32)
        scale = values.std(axis=0).astype(np.float32)
        scale[scale < EPSILON] = 1.0
        return mean, scale

    @classmethod
    def fit(cls, values: np.ndarray, pca_dim: int | None) -> "FeaturePreprocessor":
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError(f"expected non-empty [N,D] matrix, got {values.shape}")
        raw_mean, raw_scale = cls._fit_scale(values)
        scaled = (values - raw_mean) / raw_scale
        pca_mean: np.ndarray | None = None
        components: np.ndarray | None = None

        if pca_dim is not None and pca_dim > 0 and pca_dim < scaled.shape[1]:
            pca_mean = scaled.mean(axis=0).astype(np.float32)
            centered = scaled - pca_mean
            _, singular_values, right_vectors = np.linalg.svd(
                centered, full_matrices=False
            )
            effective_dim = min(int(pca_dim), right_vectors.shape[0])
            if effective_dim < 1:
                raise ValueError("PCA produced no components")
            components = right_vectors[:effective_dim].astype(np.float32)
            projected = centered @ components.T
        else:
            projected = scaled

        output_mean, output_scale = cls._fit_scale(projected)
        return cls(
            raw_mean=raw_mean,
            raw_scale=raw_scale,
            pca_mean=pca_mean,
            components=components,
            output_mean=output_mean,
            output_scale=output_scale,
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        scaled = (values - self.raw_mean) / self.raw_scale
        if self.components is not None:
            assert self.pca_mean is not None
            scaled = (scaled - self.pca_mean) @ self.components.T
        return ((scaled - self.output_mean) / self.output_scale).astype(np.float32)

    @property
    def output_dim(self) -> int:
        return int(self.output_mean.shape[0])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            raw_mean=self.raw_mean,
            raw_scale=self.raw_scale,
            pca_mean=(
                self.pca_mean
                if self.pca_mean is not None
                else np.empty(0, dtype=np.float32)
            ),
            components=(
                self.components
                if self.components is not None
                else np.empty((0, 0), dtype=np.float32)
            ),
            output_mean=self.output_mean,
            output_scale=self.output_scale,
        )


def prepare_features(
    dataset: DatasetArrays,
    method: str,
    pca_dim: int | None,
) -> tuple[np.ndarray, FeaturePreprocessor, dict[str, Any]]:
    indices = dataset.split_indices
    train_features = dataset.features[indices["train"]]
    if method in ("A", "B"):
        preprocessor = FeaturePreprocessor.fit(train_features, pca_dim)
        transformed = preprocessor.transform(dataset.features)
        metadata = {
            "raw_dim": int(dataset.features.shape[1]),
            "output_dim": preprocessor.output_dim,
            "pca_dim_requested": pca_dim,
            "pca_used": preprocessor.components is not None,
        }
        return transformed, preprocessor, metadata

    if method == "S":
        preprocessor = FeaturePreprocessor.fit(
            train_features,
            None,
        )
        return (
            preprocessor.transform(dataset.features),
            preprocessor,
            {
                "raw_dim": int(dataset.features.shape[1]),
                "output_dim": preprocessor.output_dim,
                "pca_dim_requested": None,
                "pca_used": False,
            },
        )

    if method == "B+S":
        if pca_dim is None or pca_dim <= 0:
            raise ValueError("B+S requires a positive --pca-dim for the B block")
        # B occupies the leading 2048 dimensions in the post-processor output.
        # Infer its boundary from the B feature file instead of hard-coding 2048.
        b_dim = dataset.features.shape[1] - 8
        if b_dim <= 0:
            raise ValueError(f"B+S expected B plus S features, got {dataset.features.shape}")
        b_values = dataset.features[:, :b_dim]
        s_values = dataset.features[:, b_dim:]
        b_preprocessor = FeaturePreprocessor.fit(
            b_values[indices["train"]], pca_dim
        )
        s_preprocessor = FeaturePreprocessor.fit(
            s_values[indices["train"]], None
        )
        transformed = np.concatenate(
            [
                b_preprocessor.transform(b_values),
                s_preprocessor.transform(s_values),
            ],
            axis=1,
        )
        # Save both blocks in one object-like JSON metadata; the caller writes
        # the actual arrays separately to keep the format simple and portable.
        metadata = {
            "raw_dim": int(dataset.features.shape[1]),
            "b_raw_dim": int(b_dim),
            "s_raw_dim": int(s_values.shape[1]),
            "output_dim": int(transformed.shape[1]),
            "pca_dim_requested": pca_dim,
            "pca_used": b_preprocessor.components is not None,
        }
        return transformed, _CombinedPreprocessor(b_preprocessor, s_preprocessor, b_dim), metadata

    raise ValueError(f"Unknown method {method!r}")


@dataclass
class _CombinedPreprocessor:
    b: FeaturePreprocessor
    s: FeaturePreprocessor
    b_dim: int

    @property
    def output_dim(self) -> int:
        return self.b.output_dim + self.s.output_dim

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            b_raw_mean=self.b.raw_mean,
            b_raw_scale=self.b.raw_scale,
            b_pca_mean=(
                self.b.pca_mean
                if self.b.pca_mean is not None
                else np.empty(0, dtype=np.float32)
            ),
            b_components=(
                self.b.components
                if self.b.components is not None
                else np.empty((0, 0), dtype=np.float32)
            ),
            b_output_mean=self.b.output_mean,
            b_output_scale=self.b.output_scale,
            s_raw_mean=self.s.raw_mean,
            s_raw_scale=self.s.raw_scale,
            s_output_mean=self.s.output_mean,
            s_output_scale=self.s.output_scale,
            b_dim=np.asarray(self.b_dim, dtype=np.int64),
        )


if nn is not None:

    class SuccessLinearHead(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)

        def forward(self, features: Any) -> Any:
            return self.linear(features).squeeze(-1)


    class SuccessMLP(nn.Module):
        """Two-hidden-layer nonlinear success-probability head."""

        def __init__(
            self,
            input_dim: int,
            hidden_dims: tuple[int, int] = (64, 32),
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            hidden1, hidden2 = hidden_dims
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden1, hidden2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden2, 1),
            )

        def forward(self, features: Any) -> Any:
            return self.net(features).squeeze(-1)


else:

    class SuccessLinearHead:  # type: ignore[no-redef]
        def __init__(self, input_dim: int) -> None:
            require_torch()


    class SuccessMLP:  # type: ignore[no-redef]
        def __init__(
            self,
            input_dim: int,
            hidden_dims: tuple[int, int] = (64, 32),
            dropout: float = 0.1,
        ) -> None:
            require_torch()


def binomial_nll_from_logits(logits: Any, successes: Any, trials: Any) -> Any:
    """Aggregate binomial NLL, normalized by the total number of trials."""

    require_torch()
    return -(
        successes * F.logsigmoid(logits)
        + (trials - successes) * F.logsigmoid(-logits)
    ).sum() / trials.sum()


def sigmoid_numpy(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    result = np.empty_like(logits)
    positive = logits >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    result[~positive] = exp_logits / (1.0 + exp_logits)
    return result.astype(np.float32)


def probability_metrics(
    probabilities: np.ndarray,
    successes: np.ndarray,
    trials: np.ndarray,
) -> dict[str, float]:
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), EPSILON, 1 - EPSILON)
    successes = np.asarray(successes, dtype=np.float64)
    trials = np.asarray(trials, dtype=np.float64)
    nll = -(
        successes * np.log(probabilities)
        + (trials - successes) * np.log1p(-probabilities)
    ).sum() / trials.sum()
    brier = (
        successes * (1.0 - probabilities) ** 2
        + (trials - successes) * probabilities**2
    ).sum() / trials.sum()
    empirical_rate = successes / trials
    return {
        "binomial_nll": float(nll),
        "brier_exec": float(brier),
        "rate_mse": float(np.mean((probabilities - empirical_rate) ** 2)),
        "mean_predicted_probability": float(np.mean(probabilities)),
        "mean_empirical_rate": float(np.mean(empirical_rate)),
        "num_records": int(len(probabilities)),
        "total_trials": int(trials.sum()),
    }


def train_one_seed(
    features: np.ndarray,
    dataset: DatasetArrays,
    model_kind: str,
    seed: int,
    device: str,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    hidden_dims: tuple[int, int],
    dropout: float,
) -> dict[str, Any]:
    require_torch()
    seed_everything(seed)
    resolved_device = (
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else ("cpu" if device == "auto" else device)
    )
    model: Any
    if model_kind == "linear":
        model = SuccessLinearHead(features.shape[1])
    elif model_kind == "mlp":
        model = SuccessMLP(features.shape[1], hidden_dims, dropout)
    else:
        raise ValueError(f"Unknown model kind {model_kind!r}")
    model.to(resolved_device)

    split_indices = dataset.split_indices
    tensors = {
        "features": torch.from_numpy(features).to(resolved_device),
        "successes": torch.from_numpy(dataset.successes).to(resolved_device),
        "trials": torch.from_numpy(dataset.trials).to(resolved_device),
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    best_state: dict[str, Any] | None = None
    best_val = math.inf
    best_epoch = 0
    history: list[dict[str, float]] = []
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(tensors["features"])
        train_index = torch.as_tensor(split_indices["train"], device=resolved_device)
        train_loss = binomial_nll_from_logits(
            logits[train_index],
            tensors["successes"][train_index],
            tensors["trials"][train_index],
        )
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            all_logits = model(tensors["features"])
            validation_index = torch.as_tensor(
                split_indices["validation"], device=resolved_device
            )
            validation_loss = binomial_nll_from_logits(
                all_logits[validation_index],
                tensors["successes"][validation_index],
                tensors["trials"][validation_index],
            )
        train_value = float(train_loss.detach().cpu())
        validation_value = float(validation_loss.detach().cpu())
        history.append(
            {
                "epoch": epoch,
                "train_nll": train_value,
                "validation_nll": validation_value,
            }
        )
        if validation_value < best_val - 1e-8:
            best_val = validation_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        logits = model(tensors["features"]).detach().cpu().numpy()
    probabilities = sigmoid_numpy(logits)
    metrics = {
        split: probability_metrics(
            probabilities[indices],
            dataset.successes[indices],
            dataset.trials[indices],
        )
        for split, indices in split_indices.items()
    }
    return {
        "model": model,
        "state_dict": best_state,
        "probabilities": probabilities,
        "metrics": metrics,
        "history": history,
        "best_epoch": best_epoch,
        "device": resolved_device,
    }


def selection_rows(
    records: list[dict[str, Any]],
    splits: np.ndarray,
    probabilities: np.ndarray,
    method: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = {}
    for index, record in enumerate(records):
        key = (str(record["group_id"]), str(splits[index]))
        grouped.setdefault(key, []).append((index, record))
    rows: list[dict[str, Any]] = []
    for (group_id, split), group_records in sorted(grouped.items()):
        by_condition = {
            str(record["condition_id"]): (index, record)
            for index, record in group_records
        }
        if not {"original", "better", "worse"}.issubset(by_condition):
            continue
        predicted_condition = max(
            ("original", "better", "worse"),
            key=lambda condition: (
                float(probabilities[by_condition[condition][0]]),
                -("original", "better", "worse").index(condition),
            ),
        )
        oracle_condition = max(
            ("original", "better", "worse"),
            key=lambda condition: (
                float(
                    by_condition[condition][1].get("eval_success_rate")
                    or (
                        by_condition[condition][1]["eval_successes"]
                        / by_condition[condition][1]["eval_trials"]
                    )
                ),
                -("original", "better", "worse").index(condition),
            ),
        )
        selected_record = by_condition[predicted_condition][1]
        oracle_record = by_condition[oracle_condition][1]
        selected_rate = float(
            selected_record["eval_successes"] / selected_record["eval_trials"]
        )
        oracle_rate = float(
            oracle_record["eval_successes"] / oracle_record["eval_trials"]
        )
        rows.append(
            {
                "method": method,
                "group_id": group_id,
                "split": split,
                "predicted_condition": predicted_condition,
                "oracle_condition": oracle_condition,
                "selection_correct": predicted_condition == oracle_condition,
                "selected_empirical_rate": selected_rate,
                "oracle_empirical_rate": oracle_rate,
                "empirical_regret": oracle_rate - selected_rate,
            }
        )
    return rows


def serialize_preprocessor_metadata(
    preprocessor: FeaturePreprocessor | _CombinedPreprocessor,
) -> dict[str, Any]:
    if isinstance(preprocessor, _CombinedPreprocessor):
        return {
            "type": "B+S",
            "b_output_dim": preprocessor.b.output_dim,
            "s_output_dim": preprocessor.s.output_dim,
            "b_raw_dim": preprocessor.b_dim,
        }
    return {
        "type": "standardize_pca",
        "raw_dim": int(preprocessor.raw_mean.shape[0]),
        "output_dim": preprocessor.output_dim,
        "pca_output_dim": (
            int(preprocessor.components.shape[0])
            if preprocessor.components is not None
            else None
        ),
    }


def parse_hidden_dims(value: str) -> tuple[int, int]:
    parts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 2 or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError("hidden dims must be two positive integers, e.g. 64,32")
    return parts[0], parts[1]


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    require_torch()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = tuple(args.methods)
    invalid = set(methods) - set(DEFAULT_METHODS)
    if invalid:
        raise ValueError(f"Unknown methods: {sorted(invalid)}")

    model_kinds = (
        ("linear", "mlp")
        if args.model_type == "both"
        else (args.model_type,)
    )
    all_prediction_rows: list[dict[str, Any]] = []
    all_selection_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "methods": {},
        "config": {
            "methods": list(methods),
            "model_type": args.model_type,
            "s_k": list(args.s_k),
            "pca_dim": args.pca_dim,
            "hidden_dims": list(args.hidden_dims),
            "dropout": args.dropout,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seeds": list(args.seeds),
            "device": args.device,
        },
    }

    for method in methods:
        method_variants = (
            [method]
            if method in ("A", "B")
            else [f"{method}_K{k}" for k in args.s_k]
        )
        for method_name in method_variants:
            actual_method = method_name.split("_K", 1)[0]
            s_k = int(method_name.split("_K", 1)[1]) if "_K" in method_name else None
            dataset = load_dataset_arrays(dataset_dir, actual_method, s_k)
            transformed, preprocessor, feature_metadata = prepare_features(
                dataset,
                actual_method,
                args.pca_dim if actual_method in ("A", "B", "B+S") else None,
            )
            preprocessor_path = output_dir / "preprocessors" / f"{method_name}.npz"
            preprocessor.save(preprocessor_path)
            (output_dir / "preprocessors" / f"{method_name}.json").write_text(
                json.dumps(
                    {
                        "method": method_name,
                        "feature_metadata": {
                            key: value
                            for key, value in feature_metadata.items()
                            if not isinstance(value, (FeaturePreprocessor, _CombinedPreprocessor))
                        },
                        "preprocessor": serialize_preprocessor_metadata(preprocessor),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            seed_results: dict[str, Any] = {}
            for model_kind in model_kinds:
                run_results: dict[str, Any] = {}
                for seed in args.seeds:
                    LOGGER.info("Training %s/%s seed=%s", method_name, model_kind, seed)
                    result = train_one_seed(
                        transformed,
                        dataset,
                        model_kind=model_kind,
                        seed=int(seed),
                        device=args.device,
                        max_epochs=args.max_epochs,
                        patience=args.patience,
                        learning_rate=args.learning_rate,
                        weight_decay=args.weight_decay,
                        hidden_dims=args.hidden_dims,
                        dropout=args.dropout,
                    )
                    run_dir = output_dir / "models" / method_name / model_kind / f"seed{seed}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(result["state_dict"], run_dir / "model.pt")
                    (run_dir / "history.json").write_text(
                        json.dumps(result["history"], indent=2) + "\n",
                        encoding="utf-8",
                    )
                    (run_dir / "metrics.json").write_text(
                        json.dumps(
                            {
                                "method": method_name,
                                "model_kind": model_kind,
                                "seed": int(seed),
                                "best_epoch": result["best_epoch"],
                                "device": result["device"],
                                "metrics": result["metrics"],
                            },
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    run_results[str(seed)] = {
                        "best_epoch": result["best_epoch"],
                        "metrics": result["metrics"],
                        "probabilities": result["probabilities"],
                    }
                selected_seed = min(
                    run_results,
                    key=lambda seed: run_results[seed]["metrics"]["validation"]["binomial_nll"],
                )
                selected = run_results[selected_seed]
                for index, record in enumerate(dataset.records):
                    split = str(dataset.splits[index])
                    all_prediction_rows.append(
                        {
                            "record_id": str(record["record_id"]),
                            "group_id": str(record["group_id"]),
                            "suite": record.get("suite"),
                            "task_id": record.get("task_id"),
                            "condition_id": record.get("condition_id"),
                            "split": split,
                            "method": method_name,
                            "model_kind": model_kind,
                            "selected_seed": int(selected_seed),
                            "predicted_probability": float(selected["probabilities"][index]),
                            "successes": int(record["eval_successes"]),
                            "trials": int(record["eval_trials"]),
                        }
                    )
                all_selection_rows.extend(
                    selection_rows(
                        dataset.records,
                        dataset.splits,
                        selected["probabilities"],
                        f"{method_name}/{model_kind}",
                    )
                )
                seed_results[model_kind] = {
                    "selected_seed": int(selected_seed),
                    "seeds": {
                        seed: {
                            "best_epoch": value["best_epoch"],
                            "metrics": value["metrics"],
                        }
                        for seed, value in run_results.items()
                    },
                    "feature_metadata": feature_metadata,
                }
            summary["methods"][method_name] = seed_results

    write_jsonl(output_dir / "predictions.jsonl", all_prediction_rows)
    write_jsonl(output_dir / "selection_results.jsonl", all_selection_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        choices=list(DEFAULT_METHODS),
    )
    parser.add_argument("--s-k", type=int, nargs="+", default=list(DEFAULT_S_K))
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument(
        "--model-type",
        choices=("linear", "mlp", "both"),
        default=DEFAULT_MODEL_TYPE,
    )
    parser.add_argument("--hidden-dims", type=parse_hidden_dims, default=(64, 32))
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    summary = run_training(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
