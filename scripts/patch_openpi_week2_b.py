"""Patch openpi Policy with an optional Week 2 B feature capture path.

The feature is the mean of valid multimodal prefix-token embeddings returned
by the model's existing ``embed_prefix`` method. Normal inference is unchanged.
The patch creates a backup and refuses to modify an unexpected source layout.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _backup(path: Path) -> None:
    backup = path.with_name(path.name + ".bak.week2-b")
    if not backup.exists():
        shutil.copy2(path, backup)


def patch_policy(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if "week2_b_feature" in text:
        return False

    init_marker = "        self._sample_actions = nnx_utils.module_jit(model.sample_actions)\n"
    if init_marker not in text:
        raise RuntimeError("Could not locate JAX sample_actions setup in " + str(path))
    init_insert = """        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._week2_b_feature = None
        if hasattr(model, "embed_prefix"):
            self._week2_b_feature = nnx_utils.module_jit(model.embed_prefix)
"""
    text = text.replace(init_marker, init_insert, 1)

    infer_markers = (
        "    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:",
        "    def infer(self, obs: dict) -> dict:",
    )
    infer_start = next((marker for marker in infer_markers if marker in text), None)
    if infer_start is None:
        raise RuntimeError("Could not locate Policy.infer in " + str(path))
    request_insert = infer_start + """
        week2_b_feature = bool(obs.get("return_feature", False))
        obs = {key: value for key, value in obs.items() if key != "return_feature"}
"""
    text = text.replace(infer_start, request_insert, 1)

    observation_marker = "        observation = _model.Observation.from_dict(inputs)\n"
    if observation_marker not in text:
        raise RuntimeError("Could not locate Observation construction in " + str(path))
    feature_insert = observation_marker + """        b_feature = None
        if week2_b_feature:
            if getattr(self, "_is_pytorch_model", False) or self._week2_b_feature is None:
                raise RuntimeError(
                    "Week2 B feature capture requires a JAX model with embed_prefix"
                )
            prefix_tokens, prefix_mask, _ = self._week2_b_feature(observation)
            prefix_mask_f = prefix_mask.astype(prefix_tokens.dtype)[..., None]
            b_feature = jnp.sum(prefix_tokens * prefix_mask_f, axis=1) / jnp.maximum(
                jnp.sum(prefix_mask_f, axis=1), 1.0
            )
"""
    text = text.replace(observation_marker, feature_insert, 1)

    output_marker = "        outputs = self._output_transform(outputs)\n"
    if output_marker not in text:
        raise RuntimeError("Could not locate output transform in " + str(path))
    output_insert = output_marker + """        if b_feature is not None:
            outputs["features"] = np.asarray(b_feature[0, ...], dtype=np.float32)
            outputs["feature_metadata"] = {
                "feature_type": "prefix_embedding_mean",
                "pooling": "mean_valid_prefix_tokens",
                "source": "model.embed_prefix",
                "dtype": "float32",
            }
"""
    text = text.replace(output_marker, output_insert, 1)

    _backup(path)
    path.write_text(text, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpi-root", required=True)
    args = parser.parse_args()
    root = Path(args.openpi_root)
    policy = root / "src" / "openpi" / "policies" / "policy.py"
    if not policy.exists():
        raise FileNotFoundError(policy)
    changed = patch_policy(policy)
    print("policy patched:", changed)
    print("path:", policy)
    print("backup:", policy.with_name(policy.name + ".bak.week2-b"))


if __name__ == "__main__":
    main()
