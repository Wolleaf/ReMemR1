"""Paper-aligned reward primitives shared by training and evaluation.

The functions here are intentionally tensor-free.  Trainer adapters may batch
the scalar results on any device, while the equation-level behavior remains
testable on a CPU-only development machine.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Optional, Sequence

from recurrent.protocol import (
    FinalAction,
    IntermediateAction,
    parse_final_action,
    parse_intermediate_action,
    word_recall,
)


_PROMPT_TAG_TEMPLATE = r"<{name}>(.*?)</{name}>"


def _validated_answers(answers: Sequence[str]) -> tuple[str, ...]:
    if isinstance(answers, str):
        answers = (answers,)
    normalized = tuple(answers)
    if not normalized:
        raise ValueError("answers must contain at least one non-empty string")
    if any(not isinstance(answer, str) for answer in normalized):
        raise TypeError("every answer must be a string")
    if any(not answer.strip() for answer in normalized):
        raise ValueError("answers must not contain empty strings")
    return normalized


def extract_prompt_payload(prompt: str, name: str) -> str:
    """Return one prompt tag payload, or an empty string if it is ambiguous."""

    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be str, got {type(prompt).__name__}")
    if not name or not name.replace("_", "").isalnum():
        raise ValueError(f"invalid prompt tag name: {name!r}")
    matches = re.findall(
        _PROMPT_TAG_TEMPLATE.format(name=re.escape(name)),
        prompt,
        flags=re.DOTALL,
    )
    if len(matches) != 1:
        return ""
    return matches[0].strip()


def _join_evidence(parts: Iterable[Optional[str]]) -> str:
    return " ".join(part.strip() for part in parts if isinstance(part, str) and part.strip())


def memory_gain_reward(
    previous_memory: str,
    current_memory: str,
    answers: Sequence[str],
) -> float:
    """Implement paper equation (5), preserving its argument direction."""

    golds = _validated_answers(answers)
    current = max(word_recall(current_memory, answer) for answer in golds)
    previous = max(word_recall(previous_memory, answer) for answer in golds)
    return current - previous


def callback_gain_reward(
    current_memory: str,
    current_context: str,
    recalled_memory: Optional[str],
    answers: Sequence[str],
) -> float:
    """Implement paper equation (6) using all valid gold answers."""

    golds = _validated_answers(answers)
    baseline = _join_evidence((current_memory, current_context))
    augmented = _join_evidence((recalled_memory, current_memory, current_context))
    augmented_recall = max(word_recall(answer, augmented) for answer in golds)
    baseline_recall = max(word_recall(answer, baseline) for answer in golds)
    return augmented_recall - baseline_recall


def format_reward(action: IntermediateAction | FinalAction) -> float:
    """Return the binary protocol reward without rewarding thinking tags."""

    if not isinstance(action, (IntermediateAction, FinalAction)):
        raise TypeError("action must be an IntermediateAction or FinalAction")
    return float(action.format_valid)


@dataclass(frozen=True, slots=True)
class StateReward:
    """Equation-level reward components for one recurrent action."""

    memory: float
    callback: float
    format: float

    @property
    def total(self) -> float:
        return self.memory + self.callback + self.format


def compute_state_reward(
    *,
    prompt: str,
    response: str,
    recalled_memory: Optional[str],
    answers: Sequence[str],
    is_final: bool,
) -> StateReward:
    """Parse one action and compute the paper's state-reward components."""

    if is_final:
        final_action = parse_final_action(response)
        return StateReward(memory=0.0, callback=0.0, format=format_reward(final_action))

    action = parse_intermediate_action(response)
    current_memory = action.update or ""
    previous_memory = extract_prompt_payload(prompt, "memory")
    current_context = extract_prompt_payload(prompt, "section")
    return StateReward(
        memory=memory_gain_reward(previous_memory, current_memory, answers),
        callback=callback_gain_reward(
            current_memory,
            current_context,
            recalled_memory,
            answers,
        ),
        format=format_reward(action),
    )


__all__ = [
    "StateReward",
    "callback_gain_reward",
    "compute_state_reward",
    "extract_prompt_payload",
    "format_reward",
    "memory_gain_reward",
]
