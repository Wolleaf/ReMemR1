"""Strict, dependency-free manifests for the reproduction data pipeline.

The evaluation contract is deliberately represented as data, rather than as
assumptions in a generator.  An 800-document pool is materialized first and
the 200-document variant is derived from its prefix.  Every record carries the
ordered pool and QA-order hashes so that independently written JSONL files can
still be paired and audited.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Integral
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2
EVAL_QA_COUNT = 64
EVAL_PREFIX_DOCUMENT_COUNT = 200
EVAL_POOL_DOCUMENT_COUNT = 800
EVAL_CHUNK_SIZE = 5000

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ManifestValidationError(ValueError):
    """Raised when a manifest cannot prove the reproduction contract."""


def _error(path: str, message: str) -> ManifestValidationError:
    return ManifestValidationError(f"{path}: {message}")


def normalize_text(text: str) -> str:
    """Return the stable hash form of source text.

    NFKC removes compatibility-only differences, and splitting/joining makes
    line endings and runs of Unicode whitespace platform independent.
    """

    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    return " ".join(unicodedata.normalize("NFKC", text).split())


def normalized_text_sha256(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a value using the one canonical JSON representation."""

    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def stable_document_id(title: str, text: str) -> str:
    """Create a source-independent ID from normalized title and body text."""

    payload = {
        "kind": "reproduction-document-v1",
        "text": normalize_text(text),
        "title": normalize_text(title),
    }
    return f"doc_{_sha256_json(payload)}"


def stable_supporting_fact_id(
    document_id: str,
    sentence_index: int,
    text: str,
) -> str:
    payload = {
        "document_id": document_id,
        "kind": "reproduction-supporting-fact-v1",
        "sentence_index": sentence_index,
        "text": normalize_text(text),
    }
    return f"fact_{_sha256_json(payload)}"


def ordered_values_sha256(values: Sequence[str]) -> str:
    """Hash a sequence without discarding its order."""

    return _sha256_json(list(values))


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    """Hash the exact tokenizer output committed to provenance boundaries."""

    normalized = []
    for index, token_id in enumerate(token_ids):
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, Integral)
            or int(token_id) < 0
        ):
            raise ManifestValidationError(
                f"token_ids[{index}] must be a non-negative integer"
            )
        normalized.append(int(token_id))
    if not normalized:
        raise ManifestValidationError("token_ids must not be empty")
    return _sha256_json(
        {"kind": "reproduction-context-token-ids-v1", "token_ids": normalized}
    )


def _require_nonempty_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not normalize_text(value):
        raise _error(path, "must be a non-empty string")
    return value


def _require_sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise _error(path, "must be a lowercase SHA256 hex digest")
    return value


def _require_int(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _error(path, f"must be an integer >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class ManifestMetadata:
    source_name: str
    source_revision: str
    source_sha256: str
    tokenizer_name: str
    tokenizer_revision: str
    seed: int
    schema_version: int = SCHEMA_VERSION

    def validate(self, path: str = "metadata") -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise _error(path, f"unsupported schema_version {self.schema_version}")
        _require_nonempty_text(self.source_name, f"{path}.source_name")
        _require_nonempty_text(self.source_revision, f"{path}.source_revision")
        _require_sha256(self.source_sha256, f"{path}.source_sha256")
        _require_nonempty_text(self.tokenizer_name, f"{path}.tokenizer_name")
        _require_nonempty_text(
            self.tokenizer_revision,
            f"{path}.tokenizer_revision",
        )
        _require_int(self.seed, f"{path}.seed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "source_name": self.source_name,
            "source_revision": self.source_revision,
            "source_sha256": self.source_sha256,
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_revision": self.tokenizer_revision,
        }


@dataclass(frozen=True, slots=True)
class GoldAnswer:
    text: str
    text_sha256: str

    @classmethod
    def create(cls, text: str) -> "GoldAnswer":
        _require_nonempty_text(text, "gold_answer.text")
        return cls(text=text, text_sha256=normalized_text_sha256(text))

    def validate(self, path: str = "gold_answer") -> None:
        _require_nonempty_text(self.text, f"{path}.text")
        if self.text_sha256 != normalized_text_sha256(self.text):
            raise _error(path, "text_sha256 does not match normalized answer text")

    def to_dict(self) -> dict[str, str]:
        return {"text": self.text, "text_sha256": self.text_sha256}


@dataclass(frozen=True, slots=True)
class QARecord:
    qa_id: str
    question: str
    question_sha256: str
    gold_answers: tuple[GoldAnswer, ...]

    @classmethod
    def create(
        cls,
        qa_id: str,
        question: str,
        gold_answers: Sequence[str | GoldAnswer],
    ) -> "QARecord":
        _require_nonempty_text(qa_id, "qa.qa_id")
        _require_nonempty_text(question, "qa.question")
        answers = tuple(
            answer if isinstance(answer, GoldAnswer) else GoldAnswer.create(answer)
            for answer in gold_answers
        )
        record = cls(
            qa_id=qa_id,
            question=question,
            question_sha256=normalized_text_sha256(question),
            gold_answers=answers,
        )
        record.validate()
        return record

    def validate(self, path: str = "qa") -> None:
        _require_nonempty_text(self.qa_id, f"{path}.qa_id")
        _require_nonempty_text(self.question, f"{path}.question")
        if self.question_sha256 != normalized_text_sha256(self.question):
            raise _error(path, "question_sha256 does not match normalized question")
        if not isinstance(self.gold_answers, tuple) or not self.gold_answers:
            raise _error(f"{path}.gold_answers", "must contain every legal answer")
        seen: set[str] = set()
        for index, answer in enumerate(self.gold_answers):
            if not isinstance(answer, GoldAnswer):
                raise _error(f"{path}.gold_answers[{index}]", "invalid answer schema")
            answer.validate(f"{path}.gold_answers[{index}]")
            if answer.text_sha256 in seen:
                raise _error(f"{path}.gold_answers", "contains duplicate normalized answers")
            seen.add(answer.text_sha256)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gold_answers": [answer.to_dict() for answer in self.gold_answers],
            "qa_id": self.qa_id,
            "question": self.question,
            "question_sha256": self.question_sha256,
        }


@dataclass(frozen=True, slots=True)
class DocumentInput:
    title: str
    text: str
    source_document_id: str | None = None

    def validate(self, path: str = "document_input") -> None:
        _require_nonempty_text(self.title, f"{path}.title")
        _require_nonempty_text(self.text, f"{path}.text")
        if self.source_document_id is not None:
            _require_nonempty_text(
                self.source_document_id,
                f"{path}.source_document_id",
            )

    @property
    def document_id(self) -> str:
        return stable_document_id(self.title, self.text)


@dataclass(frozen=True, slots=True)
class SupportingFactInput:
    document_index: int
    sentence_index: int
    text: str
    start_char: int | None = None
    end_char: int | None = None

    def validate(self, path: str = "supporting_fact_input") -> None:
        _require_int(self.document_index, f"{path}.document_index")
        _require_int(self.sentence_index, f"{path}.sentence_index")
        _require_nonempty_text(self.text, f"{path}.text")
        if (self.start_char is None) != (self.end_char is None):
            raise _error(path, "start_char and end_char must be provided together")
        if self.start_char is not None:
            _require_int(self.start_char, f"{path}.start_char")
            _require_int(self.end_char, f"{path}.end_char", minimum=1)
            if self.end_char <= self.start_char:
                raise _error(path, "end_char must be greater than start_char")


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    document_id: str
    position: int
    title: str
    text: str
    title_sha256: str
    text_sha256: str
    source_document_id: str | None
    token_start: int
    token_end: int
    supporting_fact_ids: tuple[str, ...]

    def validate(self, path: str = "document") -> None:
        _require_nonempty_text(self.document_id, f"{path}.document_id")
        _require_int(self.position, f"{path}.position")
        _require_nonempty_text(self.title, f"{path}.title")
        _require_nonempty_text(self.text, f"{path}.text")
        if self.document_id != stable_document_id(self.title, self.text):
            raise _error(path, "document_id does not match normalized title/text")
        if self.title_sha256 != normalized_text_sha256(self.title):
            raise _error(path, "title_sha256 does not match normalized title")
        if self.text_sha256 != normalized_text_sha256(self.text):
            raise _error(path, "text_sha256 does not match normalized text")
        if self.source_document_id is not None:
            _require_nonempty_text(
                self.source_document_id,
                f"{path}.source_document_id",
            )
        _require_int(self.token_start, f"{path}.token_start")
        _require_int(self.token_end, f"{path}.token_end", minimum=1)
        if self.token_end <= self.token_start:
            raise _error(path, "token span must be non-empty")
        if not isinstance(self.supporting_fact_ids, tuple):
            raise _error(f"{path}.supporting_fact_ids", "must be a tuple")
        if len(set(self.supporting_fact_ids)) != len(self.supporting_fact_ids):
            raise _error(f"{path}.supporting_fact_ids", "contains duplicates")

    def identity_dict(self) -> dict[str, Any]:
        """Return fields that must remain identical across 200/800 variants."""

        return {
            "document_id": self.document_id,
            "position": self.position,
            "source_document_id": self.source_document_id,
            "text": self.text,
            "text_sha256": self.text_sha256,
            "title": self.title,
            "title_sha256": self.title_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.identity_dict(),
            "supporting_fact_ids": list(self.supporting_fact_ids),
            "token_end": self.token_end,
            "token_start": self.token_start,
        }


@dataclass(frozen=True, slots=True)
class SupportingFactRecord:
    fact_id: str
    document_id: str
    document_position: int
    sentence_index: int
    text: str
    text_sha256: str
    token_start: int
    token_end: int

    def validate(self, path: str = "supporting_fact") -> None:
        _require_nonempty_text(self.fact_id, f"{path}.fact_id")
        _require_nonempty_text(self.document_id, f"{path}.document_id")
        _require_int(self.document_position, f"{path}.document_position")
        _require_int(self.sentence_index, f"{path}.sentence_index")
        _require_nonempty_text(self.text, f"{path}.text")
        expected_id = stable_supporting_fact_id(
            self.document_id,
            self.sentence_index,
            self.text,
        )
        if self.fact_id != expected_id:
            raise _error(path, "fact_id does not match document/sentence/text")
        if self.text_sha256 != normalized_text_sha256(self.text):
            raise _error(path, "text_sha256 does not match normalized text")
        _require_int(self.token_start, f"{path}.token_start")
        _require_int(self.token_end, f"{path}.token_end", minimum=1)
        if self.token_end <= self.token_start:
            raise _error(path, "token span must be non-empty")

    def identity_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "document_position": self.document_position,
            "fact_id": self.fact_id,
            "sentence_index": self.sentence_index,
            "text": self.text,
            "text_sha256": self.text_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.identity_dict(),
            "token_end": self.token_end,
            "token_start": self.token_start,
        }


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    chunk_id: str
    chunk_index: int
    token_start: int
    token_end: int
    document_ids: tuple[str, ...]
    supporting_fact_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "document_ids": list(self.document_ids),
            "supporting_fact_ids": list(self.supporting_fact_ids),
            "token_end": self.token_end,
            "token_start": self.token_start,
        }


@dataclass(frozen=True, slots=True)
class TokenProvenance:
    chunk: ChunkRecord
    documents: tuple[DocumentRecord, ...]
    supporting_facts: tuple[SupportingFactRecord, ...]


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    metadata: ManifestMetadata
    qa_index: int
    qa_order_sha256: str
    qa: QARecord
    document_count: int
    pool_document_count: int
    document_pool_sha256: str
    documents: tuple[DocumentRecord, ...]
    supporting_facts: tuple[SupportingFactRecord, ...]
    chunks: tuple[ChunkRecord, ...]
    context: str
    context_sha256: str
    context_token_ids_sha256: str
    context_token_count: int
    consumed_token_count: int
    chunk_size: int
    truncation: str
    record_sha256: str = ""

    def to_payload_dict(self) -> dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "consumed_token_count": self.consumed_token_count,
            "context": self.context,
            "context_sha256": self.context_sha256,
            "context_token_ids_sha256": self.context_token_ids_sha256,
            "context_token_count": self.context_token_count,
            "document_count": self.document_count,
            "document_pool_sha256": self.document_pool_sha256,
            "documents": [document.to_dict() for document in self.documents],
            "metadata": self.metadata.to_dict(),
            "pool_document_count": self.pool_document_count,
            "qa": self.qa.to_dict(),
            "qa_index": self.qa_index,
            "qa_order_sha256": self.qa_order_sha256,
            "schema_version": SCHEMA_VERSION,
            "supporting_facts": [fact.to_dict() for fact in self.supporting_facts],
            "truncation": self.truncation,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_payload_dict(), "record_sha256": self.record_sha256}

    def compute_sha256(self) -> str:
        return _sha256_json(self.to_payload_dict())

    def document_by_id(self, document_id: str) -> DocumentRecord:
        for document in self.documents:
            if document.document_id == document_id:
                return document
        raise KeyError(document_id)

    def supporting_fact_by_id(self, fact_id: str) -> SupportingFactRecord:
        for fact in self.supporting_facts:
            if fact.fact_id == fact_id:
                return fact
        raise KeyError(fact_id)

    def chunk_for_token(self, token_index: int) -> ChunkRecord:
        if isinstance(token_index, bool) or not isinstance(token_index, int):
            raise TypeError("token_index must be an integer")
        if token_index < 0 or token_index >= self.context_token_count:
            raise IndexError(token_index)
        return self.chunks[token_index // self.chunk_size]

    def provenance_for_token(self, token_index: int) -> TokenProvenance:
        chunk = self.chunk_for_token(token_index)
        return TokenProvenance(
            chunk=chunk,
            documents=tuple(self.document_by_id(value) for value in chunk.document_ids),
            supporting_facts=tuple(
                self.supporting_fact_by_id(value)
                for value in chunk.supporting_fact_ids
            ),
        )

    def chunks_for_document(self, document_id: str) -> tuple[ChunkRecord, ...]:
        self.document_by_id(document_id)
        return tuple(
            chunk for chunk in self.chunks if document_id in chunk.document_ids
        )

    def chunks_for_supporting_fact(self, fact_id: str) -> tuple[ChunkRecord, ...]:
        self.supporting_fact_by_id(fact_id)
        return tuple(
            chunk for chunk in self.chunks if fact_id in chunk.supporting_fact_ids
        )


def seal_manifest_record(record: ManifestRecord) -> ManifestRecord:
    """Return a record carrying the canonical hash of all other fields."""

    return replace(record, record_sha256=record.compute_sha256())


@dataclass(frozen=True, slots=True)
class EvalExampleInput:
    qa: QARecord
    document_pool: tuple[DocumentInput, ...]
    supporting_facts: tuple[SupportingFactInput, ...]


@dataclass(frozen=True, slots=True)
class EvalManifestContract:
    qa_count: int = EVAL_QA_COUNT
    prefix_document_count: int = EVAL_PREFIX_DOCUMENT_COUNT
    pool_document_count: int = EVAL_POOL_DOCUMENT_COUNT
    chunk_size: int = EVAL_CHUNK_SIZE

    def validate(self) -> None:
        _require_int(self.qa_count, "contract.qa_count", minimum=1)
        _require_int(
            self.prefix_document_count,
            "contract.prefix_document_count",
            minimum=1,
        )
        _require_int(
            self.pool_document_count,
            "contract.pool_document_count",
            minimum=2,
        )
        _require_int(self.chunk_size, "contract.chunk_size", minimum=1)
        if self.prefix_document_count >= self.pool_document_count:
            raise _error(
                "contract",
                "prefix_document_count must be smaller than pool_document_count",
            )


@dataclass(frozen=True, slots=True)
class EvalManifestPair:
    prefix_records: tuple[ManifestRecord, ...]
    pool_records: tuple[ManifestRecord, ...]

    @property
    def prefix_sha256(self) -> str:
        return canonical_jsonl_sha256(self.prefix_records)

    @property
    def pool_sha256(self) -> str:
        return canonical_jsonl_sha256(self.pool_records)


def render_context(documents: Sequence[DocumentInput | DocumentRecord]) -> str:
    """Render documents in the exact order committed by the manifest."""

    blocks = [
        f"Document {index + 1}:\n{document.title}\n{document.text}"
        for index, document in enumerate(documents)
    ]
    return "\n\n".join(blocks)


@dataclass(frozen=True, slots=True)
class _Tokenized:
    count: int
    input_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...] | None


def _tokenize(encode: Callable[[str], Any], text: str) -> _Tokenized:
    result = encode(text)
    offsets: Any = None
    if isinstance(result, Mapping):
        if "input_ids" not in result:
            raise ManifestValidationError("tokenizer output is missing input_ids")
        input_ids = result["input_ids"]
        offsets = result.get("offset_mapping")
    elif hasattr(result, "input_ids"):
        input_ids = result.input_ids
        offsets = getattr(result, "offset_mapping", None)
    else:
        input_ids = result
    if isinstance(input_ids, (str, bytes)) or not hasattr(input_ids, "__len__"):
        raise ManifestValidationError("tokenizer input_ids must be a sized sequence")
    normalized_input_ids = []
    for index, token_id in enumerate(input_ids):
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, Integral)
            or int(token_id) < 0
        ):
            raise ManifestValidationError(
                f"tokenizer input_ids[{index}] must be a non-negative integer"
            )
        normalized_input_ids.append(int(token_id))
    frozen_input_ids = tuple(normalized_input_ids)
    count = len(frozen_input_ids)
    if offsets is None:
        return _Tokenized(count=count, input_ids=frozen_input_ids, offsets=None)
    if len(offsets) != count:
        raise ManifestValidationError("offset_mapping length differs from input_ids")
    parsed: list[tuple[int, int]] = []
    previous_start = -1
    for index, offset in enumerate(offsets):
        if not isinstance(offset, Sequence) or len(offset) != 2:
            raise ManifestValidationError(f"offset_mapping[{index}] is invalid")
        start, end = offset
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
            or end > len(text)
            or start < previous_start
        ):
            raise ManifestValidationError(
                "offset_mapping must contain ordered, non-empty character spans; "
                "encode without special tokens"
            )
        parsed.append((start, end))
        previous_start = start
    return _Tokenized(
        count=count,
        input_ids=frozen_input_ids,
        offsets=tuple(parsed),
    )


def _block_char_spans(documents: Sequence[DocumentInput]) -> tuple[tuple[int, int], ...]:
    blocks = [
        f"Document {index + 1}:\n{document.title}\n{document.text}"
        for index, document in enumerate(documents)
    ]
    spans: list[tuple[int, int]] = []
    cursor = 0
    for index, block in enumerate(blocks):
        start = cursor
        cursor += len(block)
        if index + 1 < len(blocks):
            cursor += 2
        spans.append((start, cursor))
    return tuple(spans)


def _token_boundaries(
    context: str,
    char_spans: Sequence[tuple[int, int]],
    tokenized: _Tokenized,
    encode: Callable[[str], Any],
) -> tuple[tuple[int, int], ...]:
    if tokenized.offsets is not None:
        starts = [start for start, _ in tokenized.offsets]
        boundaries = [0]
        boundaries.extend(
            bisect.bisect_left(starts, end)
            for _, end in char_spans[:-1]
        )
        boundaries.append(tokenized.count)
    else:
        # Prefix tokenization is a dependency-free fallback for tiny fixtures.
        # Production tokenizers should return offset_mapping to avoid O(n^2).
        boundaries = [0]
        boundaries.extend(
            _tokenize(encode, context[:end]).count
            for _, end in char_spans[:-1]
        )
        boundaries.append(tokenized.count)
    return tuple(zip(boundaries, boundaries[1:]))


def _support_token_span(
    context: str,
    char_start: int,
    char_end: int,
    tokenized: _Tokenized,
    encode: Callable[[str], Any],
) -> tuple[int, int]:
    if tokenized.offsets is None:
        start = _tokenize(encode, context[:char_start]).count
        end = _tokenize(encode, context[:char_end]).count
    else:
        overlapping = [
            index
            for index, (start, end) in enumerate(tokenized.offsets)
            if end > char_start and start < char_end
        ]
        if not overlapping:
            raise ManifestValidationError(
                "supporting fact does not overlap a tokenizer token"
            )
        start, end = overlapping[0], overlapping[-1] + 1
    if end <= start:
        raise ManifestValidationError("supporting fact has an empty token span")
    return start, end


def _find_fact_char_span(
    document: DocumentInput,
    fact: SupportingFactInput,
    document_block_start: int,
    document_position: int,
) -> tuple[int, int]:
    body_offset = len(f"Document {document_position + 1}:\n{document.title}\n")
    if fact.start_char is None:
        start = document.text.find(fact.text)
        if start < 0:
            raise ManifestValidationError(
                f"supporting fact is absent from document {document_position}"
            )
        if document.text.find(fact.text, start + 1) >= 0:
            raise ManifestValidationError(
                "repeated supporting fact text requires explicit character offsets"
            )
        end = start + len(fact.text)
    else:
        start, end = fact.start_char, fact.end_char
        if end > len(document.text) or document.text[start:end] != fact.text:
            raise ManifestValidationError(
                f"supporting fact offsets do not match document {document_position}"
            )
    global_start = document_block_start + body_offset + start
    return global_start, global_start + (end - start)


def _spans_overlap(start: int, end: int, other_start: int, other_end: int) -> bool:
    return end > other_start and start < other_end


def build_manifest_record(
    *,
    metadata: ManifestMetadata,
    qa_index: int,
    qa_order_sha256: str,
    qa: QARecord,
    documents: Sequence[DocumentInput],
    supporting_facts: Sequence[SupportingFactInput],
    encode: Callable[[str], Any],
    chunk_size: int,
    pool_document_count: int,
    document_pool_sha256: str,
) -> ManifestRecord:
    """Materialize one untruncated record from ordered source documents."""

    metadata.validate()
    qa.validate()
    _require_int(qa_index, "qa_index")
    _require_sha256(qa_order_sha256, "qa_order_sha256")
    _require_int(chunk_size, "chunk_size", minimum=1)
    _require_int(pool_document_count, "pool_document_count", minimum=1)
    _require_sha256(document_pool_sha256, "document_pool_sha256")
    source_documents = tuple(documents)
    if not source_documents:
        raise ManifestValidationError("documents must not be empty")
    if len(source_documents) > pool_document_count:
        raise ManifestValidationError("document variant is larger than its pool")
    for index, document in enumerate(source_documents):
        if not isinstance(document, DocumentInput):
            raise _error(f"documents[{index}]", "invalid document schema")
        document.validate(f"documents[{index}]")
    document_ids = tuple(document.document_id for document in source_documents)
    if len(set(document_ids)) != len(document_ids):
        raise ManifestValidationError("document pool contains duplicate stable IDs")

    fact_inputs = tuple(supporting_facts)
    for index, fact in enumerate(fact_inputs):
        if not isinstance(fact, SupportingFactInput):
            raise _error(f"supporting_facts[{index}]", "invalid fact schema")
        fact.validate(f"supporting_facts[{index}]")
        if fact.document_index >= len(source_documents):
            raise ManifestValidationError(
                "supporting evidence is outside the materialized document prefix"
            )

    context = render_context(source_documents)
    tokenized = _tokenize(encode, context)
    if tokenized.count <= 0:
        raise ManifestValidationError("tokenized context must not be empty")
    char_spans = _block_char_spans(source_documents)
    document_token_spans = _token_boundaries(
        context,
        char_spans,
        tokenized,
        encode,
    )

    built_facts: list[SupportingFactRecord] = []
    for fact_input in fact_inputs:
        document = source_documents[fact_input.document_index]
        char_start, char_end = _find_fact_char_span(
            document,
            fact_input,
            char_spans[fact_input.document_index][0],
            fact_input.document_index,
        )
        token_start, token_end = _support_token_span(
            context,
            char_start,
            char_end,
            tokenized,
            encode,
        )
        document_id = document_ids[fact_input.document_index]
        built_facts.append(
            SupportingFactRecord(
                fact_id=stable_supporting_fact_id(
                    document_id,
                    fact_input.sentence_index,
                    fact_input.text,
                ),
                document_id=document_id,
                document_position=fact_input.document_index,
                sentence_index=fact_input.sentence_index,
                text=fact_input.text,
                text_sha256=normalized_text_sha256(fact_input.text),
                token_start=token_start,
                token_end=token_end,
            )
        )
    built_facts.sort(
        key=lambda fact: (fact.document_position, fact.sentence_index, fact.fact_id)
    )
    if len({fact.fact_id for fact in built_facts}) != len(built_facts):
        raise ManifestValidationError("supporting facts contain duplicate stable IDs")

    facts_by_document: dict[str, list[str]] = {value: [] for value in document_ids}
    for fact in built_facts:
        facts_by_document[fact.document_id].append(fact.fact_id)
    built_documents = tuple(
        DocumentRecord(
            document_id=document.document_id,
            position=index,
            title=document.title,
            text=document.text,
            title_sha256=normalized_text_sha256(document.title),
            text_sha256=normalized_text_sha256(document.text),
            source_document_id=document.source_document_id,
            token_start=document_token_spans[index][0],
            token_end=document_token_spans[index][1],
            supporting_fact_ids=tuple(facts_by_document[document.document_id]),
        )
        for index, document in enumerate(source_documents)
    )

    chunks: list[ChunkRecord] = []
    for chunk_index, token_start in enumerate(range(0, tokenized.count, chunk_size)):
        token_end = min(token_start + chunk_size, tokenized.count)
        chunks.append(
            ChunkRecord(
                chunk_id=f"chunk-{chunk_index:06d}",
                chunk_index=chunk_index,
                token_start=token_start,
                token_end=token_end,
                document_ids=tuple(
                    document.document_id
                    for document in built_documents
                    if _spans_overlap(
                        document.token_start,
                        document.token_end,
                        token_start,
                        token_end,
                    )
                ),
                supporting_fact_ids=tuple(
                    fact.fact_id
                    for fact in built_facts
                    if _spans_overlap(
                        fact.token_start,
                        fact.token_end,
                        token_start,
                        token_end,
                    )
                ),
            )
        )

    record = ManifestRecord(
        metadata=metadata,
        qa_index=qa_index,
        qa_order_sha256=qa_order_sha256,
        qa=qa,
        document_count=len(built_documents),
        pool_document_count=pool_document_count,
        document_pool_sha256=document_pool_sha256,
        documents=built_documents,
        supporting_facts=tuple(built_facts),
        chunks=tuple(chunks),
        context=context,
        context_sha256=hashlib.sha256(context.encode("utf-8")).hexdigest(),
        context_token_ids_sha256=token_ids_sha256(tokenized.input_ids),
        context_token_count=tokenized.count,
        consumed_token_count=tokenized.count,
        chunk_size=chunk_size,
        truncation="none",
    )
    record = seal_manifest_record(record)
    validate_manifest_record(record)
    return record


def validate_manifest_record(record: ManifestRecord, path: str = "record") -> None:
    """Validate hashes, schemas, complete token coverage, and provenance."""

    if not isinstance(record, ManifestRecord):
        raise _error(path, "invalid manifest record schema")
    record.metadata.validate(f"{path}.metadata")
    record.qa.validate(f"{path}.qa")
    _require_int(record.qa_index, f"{path}.qa_index")
    _require_sha256(record.qa_order_sha256, f"{path}.qa_order_sha256")
    _require_int(record.document_count, f"{path}.document_count", minimum=1)
    _require_int(
        record.pool_document_count,
        f"{path}.pool_document_count",
        minimum=1,
    )
    _require_sha256(
        record.document_pool_sha256,
        f"{path}.document_pool_sha256",
    )
    if not isinstance(record.documents, tuple):
        raise _error(f"{path}.documents", "must be a tuple")
    if not isinstance(record.supporting_facts, tuple):
        raise _error(f"{path}.supporting_facts", "must be a tuple")
    if not isinstance(record.chunks, tuple):
        raise _error(f"{path}.chunks", "must be a tuple")
    if record.document_count != len(record.documents):
        raise _error(path, "document_count differs from documents length")
    if record.document_count > record.pool_document_count:
        raise _error(path, "document_count exceeds pool_document_count")
    if record.document_count == record.pool_document_count:
        expected_pool_hash = ordered_values_sha256(
            [document.document_id for document in record.documents]
        )
        if record.document_pool_sha256 != expected_pool_hash:
            raise _error(path, "full document pool hash mismatch")

    expected_context = render_context(record.documents)
    if record.context != expected_context:
        raise _error(path, "context is inconsistent with document order/content")
    expected_context_hash = hashlib.sha256(record.context.encode("utf-8")).hexdigest()
    if record.context_sha256 != expected_context_hash:
        raise _error(path, "context_sha256 mismatch")
    _require_sha256(
        record.context_token_ids_sha256,
        f"{path}.context_token_ids_sha256",
    )
    _require_int(
        record.context_token_count,
        f"{path}.context_token_count",
        minimum=1,
    )
    _require_int(
        record.consumed_token_count,
        f"{path}.consumed_token_count",
        minimum=1,
    )
    _require_int(record.chunk_size, f"{path}.chunk_size", minimum=1)
    if record.truncation != "none":
        raise _error(path, "evaluation manifests forbid truncation")
    if record.consumed_token_count != record.context_token_count:
        raise _error(path, "token consumption truncates the context")

    document_ids: set[str] = set()
    expected_token_start = 0
    for index, document in enumerate(record.documents):
        if not isinstance(document, DocumentRecord):
            raise _error(f"{path}.documents[{index}]", "invalid document schema")
        document.validate(f"{path}.documents[{index}]")
        if document.position != index:
            raise _error(path, "document positions do not preserve list order")
        if document.document_id in document_ids:
            raise _error(path, "duplicate document ID")
        document_ids.add(document.document_id)
        if document.token_start != expected_token_start:
            raise _error(path, "document token spans are not contiguous")
        expected_token_start = document.token_end
    if expected_token_start != record.context_token_count:
        raise _error(path, "document token spans do not cover the full context")

    fact_ids: set[str] = set()
    source_fact_locations: set[tuple[str, int]] = set()
    facts_by_document: dict[str, list[str]] = {
        document.document_id: [] for document in record.documents
    }
    expected_fact_order = sorted(
        record.supporting_facts,
        key=lambda fact: (fact.document_position, fact.sentence_index, fact.fact_id),
    )
    if list(record.supporting_facts) != expected_fact_order:
        raise _error(path, "supporting facts are not in canonical source order")
    for index, fact in enumerate(record.supporting_facts):
        if not isinstance(fact, SupportingFactRecord):
            raise _error(f"{path}.supporting_facts[{index}]", "invalid fact schema")
        fact.validate(f"{path}.supporting_facts[{index}]")
        if fact.fact_id in fact_ids:
            raise _error(path, "duplicate supporting fact ID")
        fact_ids.add(fact.fact_id)
        if fact.document_id not in document_ids:
            raise _error(path, "supporting fact references a missing document")
        if fact.document_position >= len(record.documents):
            raise _error(path, "supporting fact document position is out of bounds")
        document = record.documents[fact.document_position]
        if document.document_id != fact.document_id:
            raise _error(path, "supporting fact document position is inconsistent")
        if fact.text not in document.text:
            raise _error(path, "supporting fact text is absent from its document")
        source_location = (fact.document_id, fact.sentence_index)
        if source_location in source_fact_locations:
            raise _error(path, "duplicate supporting fact source location")
        source_fact_locations.add(source_location)
        if not (
            document.token_start <= fact.token_start
            and fact.token_end <= document.token_end
        ):
            raise _error(path, "supporting fact token span is outside its document")
        facts_by_document[fact.document_id].append(fact.fact_id)
    for document in record.documents:
        if document.supporting_fact_ids != tuple(
            facts_by_document[document.document_id]
        ):
            raise _error(path, "document supporting-fact mapping is inconsistent")

    expected_chunk_count = (
        record.context_token_count + record.chunk_size - 1
    ) // record.chunk_size
    if len(record.chunks) != expected_chunk_count:
        raise _error(path, "chunk count truncates or extends the context")
    seen_documents: set[str] = set()
    seen_facts: set[str] = set()
    for index, chunk in enumerate(record.chunks):
        if not isinstance(chunk, ChunkRecord):
            raise _error(f"{path}.chunks[{index}]", "invalid chunk schema")
        expected_start = index * record.chunk_size
        expected_end = min(expected_start + record.chunk_size, record.context_token_count)
        if chunk.chunk_index != index or chunk.chunk_id != f"chunk-{index:06d}":
            raise _error(path, "chunk IDs/indices are not canonical")
        if chunk.token_start != expected_start or chunk.token_end != expected_end:
            raise _error(path, "chunk boundaries are not complete and contiguous")
        expected_document_ids = tuple(
            document.document_id
            for document in record.documents
            if _spans_overlap(
                document.token_start,
                document.token_end,
                chunk.token_start,
                chunk.token_end,
            )
        )
        expected_fact_ids = tuple(
            fact.fact_id
            for fact in record.supporting_facts
            if _spans_overlap(
                fact.token_start,
                fact.token_end,
                chunk.token_start,
                chunk.token_end,
            )
        )
        if chunk.document_ids != expected_document_ids:
            raise _error(path, "chunk-to-document provenance mismatch")
        if chunk.supporting_fact_ids != expected_fact_ids:
            raise _error(path, "chunk-to-supporting-fact provenance mismatch")
        seen_documents.update(chunk.document_ids)
        seen_facts.update(chunk.supporting_fact_ids)
    if seen_documents != document_ids:
        raise _error(path, "not every document is reachable from a chunk")
    if seen_facts != fact_ids:
        raise _error(path, "not every supporting fact is reachable from a chunk")

    _require_sha256(record.record_sha256, f"{path}.record_sha256")
    if record.record_sha256 != record.compute_sha256():
        raise _error(path, "record_sha256 mismatch")


def build_eval_manifest_pair(
    examples: Sequence[EvalExampleInput],
    *,
    metadata: ManifestMetadata,
    encode: Callable[[str], Any],
    contract: EvalManifestContract = EvalManifestContract(),
) -> EvalManifestPair:
    """Build pool records first, then derive every prefix record from that pool."""

    contract.validate()
    metadata.validate()
    source_examples = tuple(examples)
    if len(source_examples) != contract.qa_count:
        raise ManifestValidationError(
            f"expected {contract.qa_count} QA examples, got {len(source_examples)}"
        )
    qa_ids = tuple(example.qa.qa_id for example in source_examples)
    if len(set(qa_ids)) != len(qa_ids):
        raise ManifestValidationError("QA order contains duplicate IDs")
    qa_order_sha256 = ordered_values_sha256(qa_ids)

    prefix_records: list[ManifestRecord] = []
    pool_records: list[ManifestRecord] = []
    for qa_index, example in enumerate(source_examples):
        if not isinstance(example, EvalExampleInput):
            raise _error(f"examples[{qa_index}]", "invalid evaluation input schema")
        example.qa.validate(f"examples[{qa_index}].qa")
        pool = tuple(example.document_pool)
        if len(pool) != contract.pool_document_count:
            raise ManifestValidationError(
                f"QA {example.qa.qa_id!r} must materialize exactly "
                f"{contract.pool_document_count} pool documents"
            )
        pool_ids = tuple(document.document_id for document in pool)
        pool_hash = ordered_values_sha256(pool_ids)

        # Building the full record first is intentional and part of the contract.
        pool_record = build_manifest_record(
            metadata=metadata,
            qa_index=qa_index,
            qa_order_sha256=qa_order_sha256,
            qa=example.qa,
            documents=pool,
            supporting_facts=example.supporting_facts,
            encode=encode,
            chunk_size=contract.chunk_size,
            pool_document_count=contract.pool_document_count,
            document_pool_sha256=pool_hash,
        )
        prefix_record = build_manifest_record(
            metadata=metadata,
            qa_index=qa_index,
            qa_order_sha256=qa_order_sha256,
            qa=example.qa,
            documents=pool[: contract.prefix_document_count],
            supporting_facts=example.supporting_facts,
            encode=encode,
            chunk_size=contract.chunk_size,
            pool_document_count=contract.pool_document_count,
            document_pool_sha256=pool_hash,
        )
        pool_records.append(pool_record)
        prefix_records.append(prefix_record)

    pair = EvalManifestPair(
        prefix_records=tuple(prefix_records),
        pool_records=tuple(pool_records),
    )
    validate_eval_manifest_pair(pair, contract=contract)
    return pair


def validate_eval_manifest_pair(
    pair: EvalManifestPair,
    *,
    contract: EvalManifestContract = EvalManifestContract(),
    expected_qa_ids: Sequence[str] | None = None,
    expected_prefix_sha256: str | None = None,
    expected_pool_sha256: str | None = None,
) -> None:
    """Hard-fail any pairing, prefix, order, evidence, hash, or truncation drift."""

    contract.validate()
    if not isinstance(pair, EvalManifestPair):
        raise ManifestValidationError("invalid evaluation manifest pair schema")
    if len(pair.prefix_records) != contract.qa_count:
        raise ManifestValidationError("prefix manifest has the wrong QA count")
    if len(pair.pool_records) != contract.qa_count:
        raise ManifestValidationError("pool manifest has the wrong QA count")

    for index, record in enumerate(pair.prefix_records):
        validate_manifest_record(record, f"prefix_records[{index}]")
    for index, record in enumerate(pair.pool_records):
        validate_manifest_record(record, f"pool_records[{index}]")

    prefix_indices = tuple(record.qa_index for record in pair.prefix_records)
    pool_indices = tuple(record.qa_index for record in pair.pool_records)
    expected_indices = tuple(range(contract.qa_count))
    if prefix_indices != expected_indices or pool_indices != expected_indices:
        raise ManifestValidationError("QA records are not in their fixed order")

    prefix_qa_ids = tuple(record.qa.qa_id for record in pair.prefix_records)
    pool_qa_ids = tuple(record.qa.qa_id for record in pair.pool_records)
    if prefix_qa_ids != pool_qa_ids:
        raise ManifestValidationError("200/800 QA IDs are not exactly paired")
    if len(set(prefix_qa_ids)) != len(prefix_qa_ids):
        raise ManifestValidationError("manifest contains duplicate QA IDs")
    if expected_qa_ids is not None and prefix_qa_ids != tuple(expected_qa_ids):
        raise ManifestValidationError("manifest QA order differs from the locked order")
    qa_order_hash = ordered_values_sha256(prefix_qa_ids)

    baseline_metadata = pair.pool_records[0].metadata
    for index, (prefix, pool) in enumerate(
        zip(pair.prefix_records, pair.pool_records)
    ):
        if prefix.metadata != baseline_metadata or pool.metadata != baseline_metadata:
            raise ManifestValidationError(
                "source/tokenizer revision or seed differs within the paired manifests"
            )
        if prefix.qa_order_sha256 != qa_order_hash or pool.qa_order_sha256 != qa_order_hash:
            raise ManifestValidationError("QA order hash mismatch")
        if prefix.qa != pool.qa:
            raise ManifestValidationError(f"QA/gold pairing mismatch at index {index}")
        if prefix.document_count != contract.prefix_document_count:
            raise ManifestValidationError("prefix variant has the wrong document count")
        if pool.document_count != contract.pool_document_count:
            raise ManifestValidationError("pool variant has the wrong document count")
        if (
            prefix.pool_document_count != contract.pool_document_count
            or pool.pool_document_count != contract.pool_document_count
        ):
            raise ManifestValidationError("declared pool size is inconsistent")
        if prefix.chunk_size != contract.chunk_size or pool.chunk_size != contract.chunk_size:
            raise ManifestValidationError("chunk size differs from the locked contract")
        if prefix.document_pool_sha256 != pool.document_pool_sha256:
            raise ManifestValidationError("200/800 records do not identify the same pool")
        prefix_identities = tuple(
            document.identity_dict() for document in prefix.documents
        )
        pool_prefix_identities = tuple(
            document.identity_dict()
            for document in pool.documents[: contract.prefix_document_count]
        )
        if prefix_identities != pool_prefix_identities:
            raise ManifestValidationError("200 documents are not the strict 800 prefix")
        prefix_facts = tuple(
            fact.identity_dict() for fact in prefix.supporting_facts
        )
        pool_facts = tuple(fact.identity_dict() for fact in pool.supporting_facts)
        if prefix_facts != pool_facts:
            raise ManifestValidationError("supporting evidence was lost across variants")
        if any(
            fact.document_position >= contract.prefix_document_count
            for fact in pool.supporting_facts
        ):
            raise ManifestValidationError(
                "supporting evidence is outside the required 200-document prefix"
            )

    if expected_prefix_sha256 is not None:
        _require_sha256(expected_prefix_sha256, "expected_prefix_sha256")
        if pair.prefix_sha256 != expected_prefix_sha256:
            raise ManifestValidationError("canonical prefix JSONL hash mismatch")
    if expected_pool_sha256 is not None:
        _require_sha256(expected_pool_sha256, "expected_pool_sha256")
        if pair.pool_sha256 != expected_pool_sha256:
            raise ManifestValidationError("canonical pool JSONL hash mismatch")


def canonical_jsonl_bytes(records: Sequence[Any]) -> bytes:
    """Serialize records as canonical one-record-per-line UTF-8 JSONL."""

    materialized = tuple(records)
    if not materialized:
        return b""
    return b"".join(canonical_json_bytes(record) + b"\n" for record in materialized)


def canonical_jsonl_sha256(records: Sequence[Any]) -> str:
    return hashlib.sha256(canonical_jsonl_bytes(records)).hexdigest()


def write_canonical_jsonl(
    path: str | os.PathLike[str],
    records: Sequence[Any],
    *,
    expected_sha256: str | None = None,
) -> str:
    """Write canonical JSONL and return (or verify) its content hash."""

    payload = canonical_jsonl_bytes(records)
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None:
        _require_sha256(expected_sha256, "expected_sha256")
        if digest != expected_sha256:
            raise ManifestValidationError("canonical JSONL hash mismatch before write")
    Path(path).write_bytes(payload)
    return digest


def validate_canonical_jsonl(payload: bytes, expected_sha256: str) -> tuple[Any, ...]:
    """Verify bytes are canonical JSONL and match an externally stored hash."""

    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    _require_sha256(expected_sha256, "expected_sha256")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ManifestValidationError("canonical JSONL content hash mismatch")
    if payload and not payload.endswith(b"\n"):
        raise ManifestValidationError("canonical JSONL must end with a newline")
    try:
        values = tuple(
            json.loads(line.decode("utf-8"))
            for line in payload.splitlines()
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestValidationError("invalid UTF-8 JSONL") from exc
    if canonical_jsonl_bytes(values) != payload:
        raise ManifestValidationError("JSONL bytes are not canonical")
    return values


# A concise alias for checkpoint/config code that stores the data manifest hash.
manifest_sha256 = canonical_jsonl_sha256


__all__ = [
    "SCHEMA_VERSION",
    "EVAL_QA_COUNT",
    "EVAL_PREFIX_DOCUMENT_COUNT",
    "EVAL_POOL_DOCUMENT_COUNT",
    "EVAL_CHUNK_SIZE",
    "ManifestValidationError",
    "ManifestMetadata",
    "GoldAnswer",
    "QARecord",
    "DocumentInput",
    "SupportingFactInput",
    "DocumentRecord",
    "SupportingFactRecord",
    "ChunkRecord",
    "TokenProvenance",
    "ManifestRecord",
    "EvalExampleInput",
    "EvalManifestContract",
    "EvalManifestPair",
    "normalize_text",
    "normalized_text_sha256",
    "stable_document_id",
    "stable_supporting_fact_id",
    "ordered_values_sha256",
    "token_ids_sha256",
    "canonical_json_bytes",
    "canonical_jsonl_bytes",
    "canonical_jsonl_sha256",
    "manifest_sha256",
    "write_canonical_jsonl",
    "validate_canonical_jsonl",
    "render_context",
    "seal_manifest_record",
    "build_manifest_record",
    "validate_manifest_record",
    "build_eval_manifest_pair",
    "validate_eval_manifest_pair",
]
