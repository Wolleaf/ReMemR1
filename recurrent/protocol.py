"""Shared parsing and retrieval primitives for recurrent-memory trajectories.

This module intentionally has no dependencies outside the Python standard
library so training, evaluation, and CPU-only tests can share one contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable, Optional, Sequence, Tuple, Union


Identifier = Union[str, int]
CALLBACK_MODES = frozenset({"learned", "none", "fixed_question"})


@dataclass(frozen=True, slots=True)
class TagOccurrences:
    """All observable occurrences of one XML-like protocol tag."""

    name: str
    payloads: Tuple[str, ...]
    opening_count: int
    closing_count: int

    @property
    def pair_count(self) -> int:
        return len(self.payloads)

    @property
    def occurrence_count(self) -> int:
        return max(self.opening_count, self.closing_count)

    @property
    def is_absent(self) -> bool:
        return self.opening_count == self.closing_count == 0

    @property
    def is_well_formed(self) -> bool:
        return self.opening_count == self.closing_count == self.pair_count

    @property
    def has_duplicate(self) -> bool:
        return self.occurrence_count > 1

    @property
    def has_empty(self) -> bool:
        return any(not payload for payload in self.payloads)

    @property
    def is_single(self) -> bool:
        return self.is_well_formed and self.pair_count == 1

    @property
    def is_single_non_empty(self) -> bool:
        return self.is_single and bool(self.payloads[0])

    @property
    def is_valid_optional_non_empty(self) -> bool:
        return self.is_absent or self.is_single_non_empty

    @property
    def single_payload(self) -> Optional[str]:
        return self.payloads[0] if self.is_single else None

    @property
    def single_non_empty_payload(self) -> Optional[str]:
        return self.payloads[0] if self.is_single_non_empty else None


@dataclass(frozen=True, slots=True)
class IntermediateAction:
    """Parsed intermediate response plus evidence needed for format metrics."""

    thinking_occurrences: TagOccurrences
    update_occurrences: TagOccurrences
    recall_occurrences: TagOccurrences

    @property
    def thinking(self) -> Optional[str]:
        # Thinking is recorded but deliberately excluded from format validity.
        return self.thinking_occurrences.single_payload

    @property
    def update(self) -> Optional[str]:
        return self.update_occurrences.single_non_empty_payload

    @property
    def recall(self) -> Optional[str]:
        return self.recall_occurrences.single_non_empty_payload

    @property
    def update_count(self) -> int:
        return self.update_occurrences.occurrence_count

    @property
    def recall_count(self) -> int:
        return self.recall_occurrences.occurrence_count

    @property
    def has_duplicate_update(self) -> bool:
        return self.update_occurrences.has_duplicate

    @property
    def has_duplicate_recall(self) -> bool:
        return self.recall_occurrences.has_duplicate

    @property
    def is_valid(self) -> bool:
        return (
            self.update_occurrences.is_single_non_empty
            and self.recall_occurrences.is_valid_optional_non_empty
        )

    @property
    def format_valid(self) -> bool:
        return self.is_valid


@dataclass(frozen=True, slots=True)
class BoxedOccurrences:
    """All complete and malformed ``\\boxed`` commands in a response."""

    payloads: Tuple[str, ...]
    command_count: int
    malformed_count: int

    @property
    def pair_count(self) -> int:
        return len(self.payloads)

    @property
    def occurrence_count(self) -> int:
        return self.command_count

    @property
    def is_absent(self) -> bool:
        return self.command_count == 0

    @property
    def is_well_formed(self) -> bool:
        return self.malformed_count == 0 and self.command_count == self.pair_count

    @property
    def has_duplicate(self) -> bool:
        return self.command_count > 1

    @property
    def has_empty(self) -> bool:
        return any(not payload for payload in self.payloads)

    @property
    def is_single_non_empty(self) -> bool:
        return (
            self.is_well_formed
            and self.command_count == 1
            and bool(self.payloads[0])
        )

    @property
    def single_non_empty_payload(self) -> Optional[str]:
        return self.payloads[0] if self.is_single_non_empty else None


@dataclass(frozen=True, slots=True)
class FinalAction:
    """Parsed final response with a strict, observable boxed-answer contract."""

    boxed_occurrences: BoxedOccurrences

    @property
    def boxed_answer(self) -> Optional[str]:
        return self.boxed_occurrences.single_non_empty_payload

    @property
    def boxed_count(self) -> int:
        return self.boxed_occurrences.occurrence_count

    @property
    def has_duplicate_boxed(self) -> bool:
        return self.boxed_occurrences.has_duplicate

    @property
    def is_valid(self) -> bool:
        return self.boxed_occurrences.is_single_non_empty

    @property
    def format_valid(self) -> bool:
        return self.is_valid


def _require_text(text: str) -> None:
    if not isinstance(text, str):
        raise TypeError(f"protocol response must be str, got {type(text).__name__}")


def _parse_tag(text: str, name: str) -> TagOccurrences:
    opening = f"<{name}>"
    closing = f"</{name}>"
    pattern = re.compile(
        rf"{re.escape(opening)}(.*?){re.escape(closing)}",
        flags=re.DOTALL,
    )
    payloads = tuple(match.strip() for match in pattern.findall(text))
    return TagOccurrences(
        name=name,
        payloads=payloads,
        opening_count=text.count(opening),
        closing_count=text.count(closing),
    )


def parse_intermediate_action(text: str) -> IntermediateAction:
    """Parse an intermediate response without silently accepting bad counts."""

    _require_text(text)
    return IntermediateAction(
        thinking_occurrences=_parse_tag(text, "thinking"),
        update_occurrences=_parse_tag(text, "update"),
        recall_occurrences=_parse_tag(text, "recall"),
    )


_BOXED_COMMAND = re.compile(r"\\boxed\b")


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def _parse_boxed_occurrences(text: str) -> BoxedOccurrences:
    payloads = []
    malformed_count = 0
    matches = tuple(_BOXED_COMMAND.finditer(text))

    for match in matches:
        cursor = match.end()
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor == len(text) or text[cursor] != "{":
            malformed_count += 1
            continue

        payload_start = cursor + 1
        depth = 1
        cursor = payload_start
        while cursor < len(text):
            char = text[cursor]
            if char == "{" and not _is_escaped(text, cursor):
                depth += 1
            elif char == "}" and not _is_escaped(text, cursor):
                depth -= 1
                if depth == 0:
                    payloads.append(text[payload_start:cursor].strip())
                    break
            cursor += 1
        else:
            malformed_count += 1

    return BoxedOccurrences(
        payloads=tuple(payloads),
        command_count=len(matches),
        malformed_count=malformed_count,
    )


def parse_final_action(text: str) -> FinalAction:
    """Parse a final response, including multiline and nested-brace answers."""

    _require_text(text)
    return FinalAction(boxed_occurrences=_parse_boxed_occurrences(text))


def resolve_callback_query(
    mode: str,
    *,
    learned_query: Optional[str],
    question: str,
) -> Optional[str]:
    """Resolve the actual retrieval query for one callback ablation mode."""

    if mode not in CALLBACK_MODES:
        raise ValueError(
            f"callback mode must be one of {sorted(CALLBACK_MODES)}, got {mode!r}"
        )
    if not isinstance(question, str):
        raise TypeError("question must be a str")
    if learned_query is not None and not isinstance(learned_query, str):
        raise TypeError("learned_query must be a str or None")
    if mode == "none":
        return None
    query = question if mode == "fixed_question" else learned_query
    if query is None or not query.strip():
        return None
    return query.strip()


def _freeze_identifiers(value: Iterable[Identifier]) -> Tuple[Identifier, ...]:
    if isinstance(value, (str, int)):
        return (value,)
    return tuple(value)


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """One ordered memory state and the chunks/documents that produced it."""

    step_id: int
    update_text: str
    source_chunk_ids: Tuple[Identifier, ...] = field(default_factory=tuple)
    source_doc_ids: Tuple[Identifier, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if isinstance(self.step_id, bool) or not isinstance(self.step_id, int):
            raise TypeError("step_id must be an int")
        if self.step_id < 0:
            raise ValueError("step_id must be non-negative")
        if not isinstance(self.update_text, str):
            raise TypeError("update_text must be a str")
        if not self.update_text.strip():
            raise ValueError("update_text must be non-empty")
        object.__setattr__(
            self, "source_chunk_ids", _freeze_identifiers(self.source_chunk_ids)
        )
        object.__setattr__(
            self, "source_doc_ids", _freeze_identifiers(self.source_doc_ids)
        )


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """The selected full memory record and its directional recall score."""

    query: str
    record: MemoryRecord
    score: float


_WORD = re.compile(r"\w+", flags=re.UNICODE)


def _word_tokens(text: str) -> Tuple[str, ...]:
    if not isinstance(text, str):
        raise TypeError(f"word-recall arguments must be str, got {type(text).__name__}")
    return tuple(_WORD.findall(text.casefold()))


def word_recall(a: str, b: str) -> float:
    """Return the proportion of words in ``a`` that also occur in ``b``.

    The direction is part of the paper's contract: words from ``a`` form the
    denominator. Repeated word occurrences in ``a`` are each counted, while
    membership in ``b`` is binary. An empty ``a`` has recall 0.0.
    """

    words_a = _word_tokens(a)
    words_b = frozenset(_word_tokens(b))
    if not words_a:
        return 0.0
    return sum(word in words_b for word in words_a) / len(words_a)


def retrieve_top1(
    query: str, records: Sequence[MemoryRecord]
) -> Optional[RetrievalResult]:
    """Select ``argmax_x word_recall(query, x.update_text)`` deterministically.

    Records are consumed in order and never deduplicated. Equal scores are
    resolved by the smallest ``step_id``; equal score and step IDs retain input
    order. A query with no words does not trigger retrieval.
    """

    if not _word_tokens(query):
        return None

    best_record: Optional[MemoryRecord] = None
    best_score = -1.0
    for record in records:
        if not isinstance(record, MemoryRecord):
            raise TypeError("records must contain only MemoryRecord instances")
        score = word_recall(query, record.update_text)
        if (
            best_record is None
            or score > best_score
            or (score == best_score and record.step_id < best_record.step_id)
        ):
            best_record = record
            best_score = score

    if best_record is None:
        return None
    return RetrievalResult(query=query, record=best_record, score=best_score)


# Keep both natural spellings available to callers while sharing one contract.
top1_retrieve = retrieve_top1


__all__ = [
    "BoxedOccurrences",
    "CALLBACK_MODES",
    "FinalAction",
    "IntermediateAction",
    "MemoryRecord",
    "RetrievalResult",
    "TagOccurrences",
    "parse_final_action",
    "parse_intermediate_action",
    "retrieve_top1",
    "resolve_callback_query",
    "top1_retrieve",
    "word_recall",
]
