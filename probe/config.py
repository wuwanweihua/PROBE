"""Small config helpers for PROBE scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON config file.

    YAML support requires PyYAML in the runtime environment. The openpi LIBERO
    image usually has it through robosuite/LIBERO dependencies.
    """

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return json.loads(text)

    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime environment detail
        raise RuntimeError(
            f"Cannot read YAML config {config_path}; install PyYAML or use JSON."
        ) from exc
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level config must be a mapping: {config_path}")
    return data


def cfg_get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Read a nested config value with a dotted key."""

    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def parse_task_order(value: Any, num_tasks: int) -> list[int]:
    """Parse 'all', a comma-separated string, or a sequence into task ids."""

    if value in (None, "all"):
        return list(range(num_tasks))
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    raise ValueError(f"Unsupported task_order value: {value!r}")
