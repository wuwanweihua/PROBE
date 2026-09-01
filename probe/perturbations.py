"""Lightweight perturbations for collecting enough failure examples."""

from __future__ import annotations

import random


def maybe_perturb_instruction(
    instruction: str,
    rng: random.Random,
    rate: float = 0.0,
    modes: list[str] | tuple[str, ...] = ("none",),
) -> tuple[str, str]:
    """Return an instruction and the perturbation label used.

    Use this only when the plain policy is too successful and the dataset lacks
    failure labels. The original task remains unchanged; only the policy prompt
    is perturbed.
    """

    if rate <= 0.0 or rng.random() >= rate:
        return instruction, "none"

    candidates = [mode for mode in modes if mode != "none"] or [
        "drop_object_words",
        "shuffle_words",
        "generic_instruction",
    ]
    mode = rng.choice(candidates)
    words = instruction.split()

    if mode == "drop_object_words" and len(words) > 4:
        keep = [word for index, word in enumerate(words) if index % 3 != 1]
        return " ".join(keep), mode
    if mode == "shuffle_words" and len(words) > 2:
        shuffled = words[:]
        rng.shuffle(shuffled)
        return " ".join(shuffled), mode
    if mode == "generic_instruction":
        return "complete the manipulation task", mode

    return instruction, "none"
