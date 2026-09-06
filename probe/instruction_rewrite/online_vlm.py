"""Local Qwen-VL helpers for observation-aware instruction rewriting."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from probe.instruction_rewrite.prompts import (
    ONLINE_REWRITE_SYSTEM_PROMPT,
    build_online_rewrite_prompt,
)


@dataclass
class RewriteResult:
    """Result returned by the local rewrite model."""

    instruction: str
    raw_text: str = ""
    model: str = ""
    source_instruction: str = ""
    error: str | None = None
    request_id: str | None = None
    elapsed_ms: float | None = None

    def to_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "rewrite_model": self.model,
            "rewrite_instruction": self.instruction,
            "rewrite_raw_text": self.raw_text,
            "rewrite_source_instruction": self.source_instruction,
        }
        if self.error:
            metadata["rewrite_error"] = self.error
        if self.request_id:
            metadata["rewrite_request_id"] = self.request_id
        if self.elapsed_ms is not None:
            metadata["rewrite_elapsed_ms"] = round(float(self.elapsed_ms), 3)
        return {key: value for key, value in metadata.items() if value not in (None, "")}


class QwenVLMRewriteClient:
    """Thin local wrapper around a Qwen2.5-VL model."""

    def __init__(
        self,
        model_dir: str,
        *,
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        top_p: float = 0.9,
        torch_dtype: str = "float16",
        device_map: str = "auto",
    ) -> None:
        self.model_dir = str(model_dir)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.torch_dtype = str(torch_dtype)
        self.device_map = str(device_map)
        self._model: Any | None = None
        self._processor: Any | None = None

    def rewrite_from_policy_element(
        self,
        *,
        policy_element: dict[str, Any],
        source_instruction: str,
        task_id: int | None = None,
        task_name: str | None = None,
        condition_type: str | None = None,
        episode_id: str | None = None,
        step_idx: int | None = None,
        replan_idx: int | None = None,
    ) -> RewriteResult:
        try:
            self._ensure_loaded()
            assert self._model is not None and self._processor is not None
            return self._rewrite(
                policy_element=policy_element,
                source_instruction=source_instruction,
                task_id=task_id,
                task_name=task_name,
                condition_type=condition_type,
                episode_id=episode_id,
                step_idx=step_idx,
                replan_idx=replan_idx,
                model=self._model,
                processor=self._processor,
            )
        except Exception as exc:
            return RewriteResult(
                instruction=str(source_instruction),
                source_instruction=str(source_instruction),
                model=Path(self.model_dir).name,
                error=str(exc),
            )

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        torch = _import_torch()
        transformers = _import_transformers()
        dtype = getattr(torch, self.torch_dtype)
        self._processor = transformers.AutoProcessor.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
        )
        self._model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_dir,
            torch_dtype=dtype,
            device_map=self.device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self._model.eval()

    def _rewrite(
        self,
        *,
        policy_element: dict[str, Any],
        source_instruction: str,
        task_id: int | None,
        task_name: str | None,
        condition_type: str | None,
        episode_id: str | None,
        step_idx: int | None,
        replan_idx: int | None,
        model: Any,
        processor: Any,
    ) -> RewriteResult:
        torch = _import_torch()
        request_id = _request_id(task_id=task_id, episode_id=episode_id, step_idx=step_idx, replan_idx=replan_idx)

        agent_image = _to_pil_image(policy_element["observation/image"])
        wrist_image = _to_pil_image(policy_element["observation/wrist_image"])
        state = np.asarray(policy_element["observation/state"], dtype=np.float32)
        prompt_text = build_online_rewrite_prompt(
            source_instruction=source_instruction,
            task_id=task_id,
            task_name=task_name,
            condition_type=condition_type,
            episode_id=episode_id,
            step_idx=step_idx,
            replan_idx=replan_idx,
            state_summary=_summarize_state(state),
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": ONLINE_REWRITE_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": agent_image},
                    {"type": "image", "image": wrist_image},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[agent_image, wrist_image],
            padding=True,
            return_tensors="pt",
        )
        target_device = next(model.parameters()).device
        inputs = inputs.to(target_device)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": processor.tokenizer.eos_token_id,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p

        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, **generation_kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        prompt_length = int(inputs["input_ids"].shape[-1])
        decoded = processor.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        instruction = _extract_instruction_text(decoded) or str(source_instruction)
        return RewriteResult(
            instruction=instruction,
            raw_text=decoded,
            model=Path(self.model_dir).name,
            source_instruction=str(source_instruction),
            request_id=request_id,
            elapsed_ms=elapsed_ms,
        )


def _import_torch():
    import torch

    return torch


def _import_transformers():
    import transformers

    return transformers


def _to_pil_image(value: Any):
    from PIL import Image

    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype != np.uint8:
        array = np.asarray(np.clip(array, 0, 255), dtype=np.uint8)
    if array.ndim != 3:
        raise ValueError(f"expected 3D image array, got shape {array.shape}")
    return Image.fromarray(array)


def _summarize_state(state: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(state, dtype=np.float32).reshape(-1)
    summary: dict[str, Any] = {"state_vector": flat.tolist()}
    if flat.size >= 7:
        summary["eef_pos"] = flat[:3].tolist()
        summary["eef_axis_angle"] = flat[3:6].tolist()
        summary["gripper_qpos"] = flat[6:7].tolist()
    return summary


def _extract_instruction_text(text: str) -> str:
    cleaned = str(text).strip()
    if not cleaned:
        return ""
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""
    first = lines[0]
    for prefix in ("rewritten instruction:", "instruction:", "rewrite:"):
        if first.lower().startswith(prefix):
            first = first[len(prefix) :].strip()
            break
    first = first.strip("\"' ")
    return " ".join(first.split())


def _request_id(
    *,
    task_id: int | None,
    episode_id: str | None,
    step_idx: int | None,
    replan_idx: int | None,
) -> str:
    parts = [
        f"task{task_id}" if task_id is not None else "task",
        str(episode_id or "episode"),
        f"step{step_idx}" if step_idx is not None else "step",
        f"replan{replan_idx}" if replan_idx is not None else "replan",
        str(int(time.time() * 1000)),
    ]
    return "_".join(parts)
