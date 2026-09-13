"""Extract frozen observation-and-instruction features for the Week 3 A baseline.

The extractor reads a Week 3 postprocessed dataset, loads the original
policy-ready observation referenced by each record, and encodes:

* the agent-view image;
* the wrist image; and
* the record's true instruction.

The three L2-normalized embeddings are concatenated into one A feature.  No
labels, actions, B features, or VLA policy calls are used.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Protocol

import numpy as np


LOGGER = logging.getLogger(__name__)
A_FEATURE_SCHEMA_VERSION = 1
A_FEATURE_NAMES = ("agent_image", "wrist_image", "instruction")


class VisionTextEncoder(Protocol):
    """Small interface that keeps the file-format logic testable."""

    model_id: str
    revision: str | None
    preprocessing: dict[str, Any]

    def encode(
        self,
        agent_image: np.ndarray,
        wrist_image: np.ndarray,
        instruction: str,
    ) -> np.ndarray:
        ...


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(payload)
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def find_manifest(dataset_dir: Path) -> Path:
    manifest = dataset_dir / "manifest.jsonl"
    if manifest.exists():
        return manifest
    fallback = dataset_dir / "records.jsonl"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"No manifest found in {dataset_dir}; checked {manifest} and {fallback}"
    )


def resolve_observation_path(
    record: dict[str, Any],
    dataset_dir: Path,
    source_root: Path | None = None,
) -> Path:
    raw_reference = (
        record.get("source_observation_path")
        or record.get("observation_path")
        or record.get("obs_path")
    )
    if not raw_reference:
        raise KeyError(f"{record.get('record_id', '<unknown>')}: missing observation path")

    reference = Path(str(raw_reference)).expanduser()
    candidates: list[Path] = []
    if reference.is_absolute():
        candidates.append(reference)
        if source_root is not None:
            candidates.append(source_root / reference.name)
    else:
        source_dataset = record.get("source_dataset_dir")
        if source_dataset:
            candidates.append(Path(str(source_dataset)).expanduser() / reference)
        candidates.append(dataset_dir / reference)
        if source_root is not None:
            candidates.append(source_root / reference)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"{record.get('record_id', '<unknown>')}: observation not found; checked {checked}"
    )


def load_policy_images(path: Path, record_id: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = ("policy_image", "policy_wrist_image")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"{record_id}: {path} is missing arrays {missing}")
        agent_image = np.asarray(data["policy_image"])
        wrist_image = np.asarray(data["policy_wrist_image"])

    for name, image in (("policy_image", agent_image), ("policy_wrist_image", wrist_image)):
        if image.ndim != 3 or image.shape[2] not in (1, 3, 4):
            raise ValueError(
                f"{record_id}: {name} must be HxWxC with 1, 3, or 4 channels, "
                f"got {image.shape}"
            )
        if not np.isfinite(image).all():
            raise ValueError(f"{record_id}: {name} contains NaN or Inf")
    return agent_image, wrist_image


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("encoder returned a zero or non-finite embedding")
    return vector / norm


def compose_a_feature(
    agent_embedding: np.ndarray,
    wrist_embedding: np.ndarray,
    text_embedding: np.ndarray,
) -> np.ndarray:
    """Normalize each modality independently, then concatenate."""

    parts = [
        _l2_normalize(agent_embedding),
        _l2_normalize(wrist_embedding),
        _l2_normalize(text_embedding),
    ]
    feature = np.concatenate(parts).astype(np.float32, copy=False)
    if feature.ndim != 1 or not np.isfinite(feature).all():
        raise ValueError(f"invalid composed A feature with shape {feature.shape}")
    return feature


def write_a_feature(
    path: Path,
    feature: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        a_feature=np.asarray(feature, dtype=np.float32),
        feature_shape=np.asarray(feature.shape, dtype=np.int64),
        feature_names=np.asarray(A_FEATURE_NAMES),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )


def load_a_feature(path: Path, record_id: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "a_feature" not in data:
            raise KeyError(f"{record_id}: {path} is missing a_feature")
        feature = np.asarray(data["a_feature"], dtype=np.float32)
    if feature.ndim != 1 or not np.isfinite(feature).all():
        raise ValueError(f"{record_id}: invalid A feature with shape {feature.shape}")
    return feature


class HuggingFaceVisionTextEncoder:
    """Frozen CLIP/SigLIP encoder using the model's projected embeddings."""

    def __init__(
        self,
        model_id: str,
        revision: str | None,
        device: str,
        local_files_only: bool,
    ) -> None:
        try:
            import torch
            from PIL import Image
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "A feature extraction requires torch, Pillow, and transformers "
                "in the active environment."
            ) from exc

        self._torch = torch
        self._image_type = Image
        self.model_id = model_id
        self.revision = revision
        self.device = self._resolve_device(device)
        load_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
        }
        if revision:
            load_kwargs["revision"] = revision

        self.processor = AutoProcessor.from_pretrained(model_id, **load_kwargs)
        self.model = AutoModel.from_pretrained(model_id, **load_kwargs)
        self.model.eval().to(self.device)
        self.preprocessing = {
            "processor_class": self.processor.__class__.__name__,
            "model_class": self.model.__class__.__name__,
            "image_input": "policy_image and policy_wrist_image",
            "text_input": "manifest.instruction",
            "embedding_source": "model.get_image_features/get_text_features",
            "pooling": "model_projected_embedding",
            "per_modality_normalization": "l2",
            "feature_order": list(A_FEATURE_NAMES),
        }

    def _resolve_device(self, requested: str) -> str:
        if requested != "auto":
            return requested
        if self._torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _to_device(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in payload.items()
        }

    @staticmethod
    def _to_pil(image: np.ndarray, image_type: Any) -> Any:
        array = np.asarray(image)
        if array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        elif array.shape[2] == 4:
            array = array[:, :, :3]
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating):
                if float(np.nanmax(array)) <= 1.0:
                    array = array * 255.0
                array = np.clip(array, 0.0, 255.0)
            array = array.astype(np.uint8)
        return image_type.fromarray(np.ascontiguousarray(array), mode="RGB")

    def _projected_features(
        self,
        method_name: str,
        payload: dict[str, Any],
    ) -> np.ndarray:
        method = getattr(self.model, method_name, None)
        if method is None:
            raise RuntimeError(
                f"{self.model.__class__.__name__} has no {method_name}; "
                "use a CLIP/SigLIP model exposing projected image/text features."
            )
        with self._torch.inference_mode():
            output = method(**self._to_device(payload))
        if hasattr(output, "pooler_output"):
            output = output.pooler_output
        if not hasattr(output, "detach"):
            raise RuntimeError(f"{method_name} returned unsupported output {type(output)!r}")
        return output.detach().float().cpu().numpy()

    def encode(
        self,
        agent_image: np.ndarray,
        wrist_image: np.ndarray,
        instruction: str,
    ) -> np.ndarray:
        agent_pil = self._to_pil(agent_image, self._image_type)
        wrist_pil = self._to_pil(wrist_image, self._image_type)
        image_inputs = self.processor(
            images=[agent_pil, wrist_pil],
            return_tensors="pt",
        )
        text_inputs = self.processor(
            text=[str(instruction)],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        image_embeddings = self._projected_features("get_image_features", image_inputs)
        text_embeddings = self._projected_features("get_text_features", text_inputs)
        if image_embeddings.shape[0] != 2 or text_embeddings.shape[0] != 1:
            raise RuntimeError(
                "Expected two image embeddings and one text embedding, got "
                f"{image_embeddings.shape} and {text_embeddings.shape}"
            )
        return compose_a_feature(
            image_embeddings[0],
            image_embeddings[1],
            text_embeddings[0],
        )


def extract_dataset(
    dataset_dir: Path,
    model_id: str,
    revision: str | None,
    device: str,
    local_files_only: bool,
    source_root: Path | None = None,
    skip_existing: bool = False,
) -> dict[str, Any]:
    manifest_path = find_manifest(dataset_dir)
    records = read_jsonl(manifest_path)
    encoder = HuggingFaceVisionTextEncoder(
        model_id=model_id,
        revision=revision,
        device=device,
        local_files_only=local_files_only,
    )
    feature_dir = dataset_dir / "features" / "A"
    feature_dir.mkdir(parents=True, exist_ok=True)

    feature_records: list[dict[str, Any]] = []
    feature_dim: int | None = None
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index, record in enumerate(records, start=1):
        record_id = str(record.get("record_id") or f"record_{index}")
        instruction = str(record.get("instruction") or "").strip()
        if not instruction:
            raise ValueError(f"{record_id}: empty instruction")
        observation_path = resolve_observation_path(record, dataset_dir, source_root)
        observation_key = str(observation_path.resolve())
        if observation_key not in cache:
            cache[observation_key] = load_policy_images(observation_path, record_id)
        agent_image, wrist_image = cache[observation_key]

        output_path = feature_dir / f"{record_id}.npz"
        if skip_existing and output_path.exists():
            feature = load_a_feature(output_path, record_id)
        else:
            feature = encoder.encode(agent_image, wrist_image, instruction)
            write_a_feature(
                output_path,
                feature,
                {
                    "schema_version": A_FEATURE_SCHEMA_VERSION,
                    "model_id": encoder.model_id,
                    "revision": encoder.revision,
                    "preprocessing": encoder.preprocessing,
                },
            )

        if feature_dim is None:
            feature_dim = int(feature.shape[0])
        elif int(feature.shape[0]) != feature_dim:
            raise ValueError(
                f"{record_id}: feature dimension {feature.shape[0]} differs from "
                f"previous dimension {feature_dim}"
            )
        relative_path = output_path.relative_to(dataset_dir).as_posix()
        record["a_feature_path"] = relative_path
        record["a_feature_shape"] = [int(feature.shape[0])]
        record["a_feature_dtype"] = "float32"
        record["a_feature_metadata"] = {
            "schema_version": A_FEATURE_SCHEMA_VERSION,
            "model_id": encoder.model_id,
            "revision": encoder.revision,
            "preprocessing": encoder.preprocessing,
        }
        feature_records.append(record)
        if index == 1 or index % 50 == 0 or index == len(records):
            LOGGER.info("A features: %d/%d", index, len(records))

    write_jsonl_atomic(manifest_path, feature_records)
    matrix = np.stack(
        [
            load_a_feature(
                feature_dir / f"{record['record_id']}.npz",
                str(record["record_id"]),
            )
            for record in feature_records
        ]
    )
    matrix_path = dataset_dir / "features" / "A_matrix.npz"
    np.savez_compressed(
        matrix_path,
        record_ids=np.asarray([str(record["record_id"]) for record in feature_records]),
        a_features=matrix.astype(np.float32, copy=False),
        feature_names=np.asarray(A_FEATURE_NAMES),
    )

    summary = {
        "dataset_dir": str(dataset_dir),
        "manifest": str(manifest_path),
        "num_records": len(feature_records),
        "num_unique_observations": len(cache),
        "feature_shape": [int(feature_dim)] if feature_dim is not None else [0],
        "feature_dtype": "float32",
        "feature_names": list(A_FEATURE_NAMES),
        "model_id": encoder.model_id,
        "revision": encoder.revision,
        "device": encoder.device,
        "local_files_only": local_files_only,
        "schema_version": A_FEATURE_SCHEMA_VERSION,
        "feature_matrix": str(matrix_path.relative_to(dataset_dir)),
    }
    (dataset_dir / "a_feature_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument(
        "--model-id",
        required=True,
        help="Local model directory or pinned Hugging Face model id.",
    )
    parser.add_argument("--revision")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow transformers to download missing model files.",
    )
    parser.add_argument(
        "--source-root",
        help="Optional fallback root for remapping observation references.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    summary = extract_dataset(
        dataset_dir=Path(args.dataset_dir).expanduser().resolve(),
        model_id=args.model_id,
        revision=args.revision,
        device=args.device,
        local_files_only=not args.allow_download,
        source_root=Path(args.source_root).expanduser().resolve()
        if args.source_root
        else None,
        skip_existing=args.skip_existing,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
