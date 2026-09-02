"""LIBERO helpers used by the week-1 collector."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def max_steps_for_suite(task_suite_name: str) -> int:
    limits = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    try:
        return limits[task_suite_name]
    except KeyError as exc:
        raise ValueError(f"Unknown LIBERO task suite: {task_suite_name}") from exc


def get_task_suite(task_suite_name: str) -> Any:
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    return benchmark_dict[task_suite_name]()


def make_libero_env(task: Any, resolution: int = LIBERO_ENV_RESOLUTION, seed: int = 7) -> tuple[Any, str]:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, str(task_description)


def build_policy_element(obs: dict[str, Any], prompt: str, resize_size: int = 224) -> dict[str, Any]:
    """Convert a LIBERO observation into the openpi policy input format."""

    from openpi_client import image_tools

    agent_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    agent_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(agent_img, resize_size, resize_size)
    )
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
    )
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat_to_axis_angle(np.asarray(obs["robot0_eef_quat"]).copy()),
            obs["robot0_gripper_qpos"],
        )
    )
    return {
        "observation/image": agent_img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": str(prompt),
    }


def observation_arrays_for_record(
    raw_obs: dict[str, Any],
    policy_element: dict[str, Any],
) -> dict[str, Any]:
    """Select numeric observation arrays worth saving for later analysis."""

    arrays: dict[str, Any] = {
        "policy_image": policy_element["observation/image"],
        "policy_wrist_image": policy_element["observation/wrist_image"],
        "policy_state": policy_element["observation/state"],
    }
    for key in (
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    ):
        if key in raw_obs:
            arrays[f"raw_{key}"] = raw_obs[key]
    return arrays


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denom = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(denom), 0.0):
        return np.zeros(3, dtype=np.float64)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / denom
