"""Manifest schema for one PROBE call record."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ProbeCallRecord:
    """One labeled call made by a frozen VLA policy.

    A record is created at a replanning point. It stores the observation and all
    sampled action chunks, while the label is the final episode success/failure.
    """

    record_id: str
    episode_id: str
    task_suite: str
    task_id: int
    task_name: str
    trial_idx: int
    step_idx: int
    instruction: str
    policy_name: str
    checkpoint_uri: str
    seed: int
    k_samples: int
    action_selection: str
    selected_sample_index: int
    obs_path: str
    action_samples_path: str
    final_success: bool
    episode_done: bool
    episode_steps: int
    reward_sum: float
    perturbation_type: str = "none"
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "ProbeCallRecord":
        return cls(**payload)


REQUIRED_FIELDS = {
    "record_id",
    "episode_id",
    "task_suite",
    "task_id",
    "task_name",
    "trial_idx",
    "step_idx",
    "instruction",
    "policy_name",
    "checkpoint_uri",
    "seed",
    "k_samples",
    "action_selection",
    "selected_sample_index",
    "obs_path",
    "action_samples_path",
    "final_success",
    "episode_done",
    "episode_steps",
    "reward_sum",
}


def missing_required_fields(payload: dict[str, Any]) -> set[str]:
    return REQUIRED_FIELDS.difference(payload)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value
