"""Build deterministic ReMemR1 train/eval artifacts from local QA sources.

Run this module with ``python -m taskutils.data_synthesis.reproduction_builder``.
It never downloads datasets or model weights.  Hugging Face tokenizers are
opened with ``local_files_only=True`` and every source/artifact is content
hashed before the bundle is published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .reproduction_manifest import (
    SCHEMA_VERSION,
    ChunkRecord,
    DocumentInput,
    DocumentRecord,
    EvalExampleInput,
    EvalManifestContract,
    GoldAnswer,
    ManifestMetadata,
    ManifestRecord,
    ManifestValidationError,
    QARecord,
    SupportingFactInput,
    SupportingFactRecord,
    build_eval_manifest_pair,
    build_manifest_record,
    canonical_json_bytes,
    canonical_jsonl_sha256,
    normalize_text,
    normalized_text_sha256,
    ordered_values_sha256,
    stable_document_id,
    validate_canonical_jsonl,
    validate_eval_manifest_pair,
    validate_manifest_record,
    write_canonical_jsonl,
)


BUNDLE_SCHEMA_VERSION = 2
BUNDLE_KIND = "rememr1-reproduction-data-bundle-v2"
FORMAL_BUNDLE_SCHEMA_VERSION = 3
FORMAL_BUNDLE_KIND = "rememr1-reproduction-data-bundle-v3"
CURATION_SCHEMA_VERSION = 1
CURATION_POLICY = "rememr1-source-curation-v1"
REJECTION_LEDGER_NAME = "source-curation-ledger.json"
_OUT_OF_BOUNDS_REASON = "supporting_fact_sentence_index_out_of_bounds"
_AMBIGUOUS_DOCUMENT_REASON = "ambiguous_normalized_document_id"
_MISSING_SUPPORT_TITLE_REASON = "supporting_title_absent_from_context"
_EMPTY_CONTEXT_REASON = "empty_context"
_DUPLICATE_CONTEXT_DOCUMENT_REASON = "duplicate_context_document_id"
_REJECTION_REASONS = (
    _AMBIGUOUS_DOCUMENT_REASON,
    _DUPLICATE_CONTEXT_DOCUMENT_REASON,
    _EMPTY_CONTEXT_REASON,
    _OUT_OF_BOUNDS_REASON,
    _MISSING_SUPPORT_TITLE_REASON,
)
FORMAL_TRAIN_QA_COUNT = 512
FORMAL_TRAIN_DOCUMENT_COUNT = 200
FORMAL_TRAIN_CHUNK_SIZE = 5000
FORMAL_TRAIN_MAX_CHUNKS = 6
FORMAL_TRAIN_MIN_CONTEXT_TOKENS = 25_001
FORMAL_TRAIN_MAX_CONTEXT_TOKENS = 30_000
_FLOATING_REVISIONS = {"", "main", "master", "latest", "head"}
_DATASET_ALIASES = {
    "hotpot": "hotpotqa",
    "hotpotqa": "hotpotqa",
    "2wiki": "2wikimultihopqa",
    "2wikimultihopqa": "2wikimultihopqa",
    "2wikimultihop": "2wikimultihopqa",
}


def _error(path: str, message: str) -> ManifestValidationError:
    return ManifestValidationError(f"{path}: {message}")


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(path, "must be an object")
    return value


def _require_sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _error(path, "must be an array")
    return value


def _require_nonempty(value: Any, path: str) -> str:
    if not isinstance(value, str) or not normalize_text(value):
        raise _error(path, "must be a non-empty string")
    return value


def _require_int(value: Any, path: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _error(path, f"must be an integer >= {minimum}")
    return value


def _require_fixed_revision(value: str, path: str) -> str:
    value = _require_nonempty(value, path).strip()
    if value.lower() in _FLOATING_REVISIONS:
        raise _error(path, "must be an immutable revision, not a floating name")
    return value


def _require_sha256(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _error(path, "must be a lowercase SHA256 digest")
    return value


def _canonical_dataset_name(value: str) -> str:
    key = _require_nonempty(value, "dataset").lower().replace("-", "")
    try:
        return _DATASET_ALIASES[key]
    except KeyError as exc:
        raise _error("dataset", "must be hotpotqa or 2wikimultihopqa") from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def stable_qa_id(dataset: str, source_qa_id: str | None, question: str) -> str:
    """Return a stable, dataset-namespaced QA identifier."""

    dataset = _canonical_dataset_name(dataset)
    if source_qa_id is not None and normalize_text(str(source_qa_id)):
        return f"{dataset}:{normalize_text(str(source_qa_id))}"
    payload = {
        "dataset": dataset,
        "kind": "rememr1-question-qa-v1",
        "question": normalize_text(question),
    }
    return f"{dataset}:qa_{_sha256_bytes(canonical_json_bytes(payload))}"


@dataclass(frozen=True, slots=True)
class SourceDocument:
    title: str
    sentences: tuple[str, ...]
    source_document_id: str

    @property
    def text(self) -> str:
        return "".join(self.sentences)

    @property
    def document_id(self) -> str:
        return stable_document_id(self.title, self.text)

    def to_manifest_input(self) -> DocumentInput:
        return DocumentInput(
            title=self.title,
            text=self.text,
            source_document_id=self.source_document_id,
        )


@dataclass(frozen=True, slots=True)
class ParsedExample:
    source_qa_id: str | None
    qa: QARecord
    documents: tuple[SourceDocument, ...]
    supporting_facts: tuple[SupportingFactInput, ...]


@dataclass(frozen=True, slots=True)
class SourceRejectionReason:
    code: str
    evidence: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "evidence": [dict(value) for value in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class SourceRejection:
    source_index: int
    qa_id: str
    source_record_sha256: str
    reasons: tuple[SourceRejectionReason, ...]
    ambiguous_document_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ambiguous_document_ids": list(self.ambiguous_document_ids),
            "qa_id": self.qa_id,
            "reasons": [value.to_dict() for value in self.reasons],
            "source_index": self.source_index,
            "source_record_sha256": self.source_record_sha256,
        }


@dataclass(frozen=True, slots=True)
class SourceCurationResult:
    examples: tuple[ParsedExample, ...]
    accepted_source_indices: tuple[int, ...]
    rejections: tuple[SourceRejection, ...]
    ambiguous_document_ids: tuple[str, ...]


class _RejectableSourceRecordError(ManifestValidationError):
    def __init__(
        self,
        message: str,
        *,
        reasons: Sequence[SourceRejectionReason],
    ) -> None:
        super().__init__(message)
        self.reasons = tuple(reasons)


@dataclass(frozen=True, slots=True)
class TrainManifestContract:
    qa_count: int = FORMAL_TRAIN_QA_COUNT
    document_count: int = FORMAL_TRAIN_DOCUMENT_COUNT
    chunk_size: int = FORMAL_TRAIN_CHUNK_SIZE
    max_chunks: int = FORMAL_TRAIN_MAX_CHUNKS
    min_context_tokens: int = FORMAL_TRAIN_MIN_CONTEXT_TOKENS
    max_context_tokens: int = FORMAL_TRAIN_MAX_CONTEXT_TOKENS

    def validate(self) -> None:
        _require_int(self.qa_count, "train_contract.qa_count", 1)
        _require_int(self.document_count, "train_contract.document_count", 1)
        _require_int(self.chunk_size, "train_contract.chunk_size", 1)
        _require_int(self.max_chunks, "train_contract.max_chunks", 1)
        _require_int(self.min_context_tokens, "train_contract.min_context_tokens", 1)
        _require_int(self.max_context_tokens, "train_contract.max_context_tokens", 1)
        if self.min_context_tokens > self.max_context_tokens:
            raise _error("train_contract", "minimum context tokens exceeds maximum")
        if self.max_context_tokens != self.chunk_size * self.max_chunks:
            raise _error(
                "train_contract",
                "max_context_tokens must equal chunk_size * max_chunks",
            )
        minimum_for_last_chunk = self.chunk_size * (self.max_chunks - 1) + 1
        if self.min_context_tokens < minimum_for_last_chunk:
            raise _error(
                "train_contract",
                "minimum context tokens must require exactly max_chunks chunks",
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "chunk_size": self.chunk_size,
            "document_count": self.document_count,
            "max_chunks": self.max_chunks,
            "max_context_tokens": self.max_context_tokens,
            "min_context_tokens": self.min_context_tokens,
            "qa_count": self.qa_count,
        }


def _first_present(mapping: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def _extract_question(record: Mapping[str, Any], path: str) -> str:
    value = _first_present(record, ("question", "input"))
    if value is None and "prompt" in record:
        prompt = record["prompt"]
        if isinstance(prompt, Sequence) and not isinstance(prompt, (str, bytes)) and prompt:
            last = prompt[-1]
            if isinstance(last, Mapping):
                value = last.get("content")
    return _require_nonempty(value, f"{path}.question")


def _extract_answers(record: Mapping[str, Any], path: str) -> tuple[str, ...]:
    value = _first_present(
        record,
        ("answers", "golden_answers", "gold_answers", "outputs", "answer"),
    )
    if isinstance(value, Mapping):
        value = _first_present(value, ("text", "answers", "value"))
    if isinstance(value, str):
        candidates: Sequence[Any] = (value,)
    else:
        candidates = _require_sequence(value, f"{path}.answers")
    answers: list[str] = []
    seen: set[str] = set()
    for index, answer in enumerate(candidates):
        if isinstance(answer, Mapping):
            answer = _first_present(answer, ("text", "answer", "value"))
        answer = _require_nonempty(answer, f"{path}.answers[{index}]")
        digest = normalized_text_sha256(answer)
        if digest not in seen:
            answers.append(answer)
            seen.add(digest)
    if not answers:
        raise _error(f"{path}.answers", "must contain at least one legal answer")
    return tuple(answers)


def _sentences_from_value(value: Any, path: str) -> tuple[str, ...]:
    if isinstance(value, str):
        sentences = (value,)
    else:
        sequence = _require_sequence(value, path)
        parsed = []
        for index, sentence in enumerate(sequence):
            if not isinstance(sentence, str):
                raise _error(f"{path}[{index}]", "must be a string")
            parsed.append(sentence)
        sentences = tuple(parsed)
    if not sentences or not normalize_text("".join(sentences)):
        raise _error(path, "must contain non-empty document text")
    return sentences


def _source_document_id(
    dataset: str,
    title: str,
    text: str,
    supplied: Any,
) -> str:
    if supplied is not None and normalize_text(str(supplied)):
        return f"{dataset}:{normalize_text(str(supplied))}"
    return f"{dataset}:{stable_document_id(title, text)}"


def _parse_context_mapping(
    context: Mapping[str, Any],
    dataset: str,
    path: str,
) -> tuple[SourceDocument, ...]:
    titles = _require_sequence(
        _first_present(context, ("title", "titles")),
        f"{path}.title",
    )
    contents = _require_sequence(
        _first_present(context, ("sentences", "content", "contents", "text")),
        f"{path}.sentences",
    )
    if len(titles) != len(contents):
        raise _error(path, "title and sentence arrays have different lengths")
    raw_ids = _first_present(context, ("id", "ids", "document_id", "document_ids"))
    if raw_ids is None:
        identifiers: Sequence[Any] = (None,) * len(titles)
    else:
        identifiers = _require_sequence(raw_ids, f"{path}.document_ids")
        if len(identifiers) != len(titles):
            raise _error(path, "document ID and title arrays have different lengths")
    documents = []
    for index, (title, content, supplied_id) in enumerate(
        zip(titles, contents, identifiers)
    ):
        title = _require_nonempty(title, f"{path}.title[{index}]")
        sentences = _sentences_from_value(content, f"{path}.sentences[{index}]")
        text = "".join(sentences)
        documents.append(
            SourceDocument(
                title=title,
                sentences=sentences,
                source_document_id=_source_document_id(
                    dataset,
                    title,
                    text,
                    supplied_id,
                ),
            )
        )
    return tuple(documents)


def _parse_context_sequence(
    context: Sequence[Any],
    dataset: str,
    path: str,
) -> tuple[SourceDocument, ...]:
    documents = []
    for index, item in enumerate(context):
        item_path = f"{path}[{index}]"
        supplied_id = None
        if isinstance(item, Mapping):
            title = _first_present(item, ("title", "name"))
            content = _first_present(item, ("sentences", "content", "text"))
            supplied_id = _first_present(
                item,
                ("id", "document_id", "doc_id", "source_document_id"),
            )
        else:
            pair = _require_sequence(item, item_path)
            if len(pair) != 2:
                raise _error(item_path, "context entry must be [title, sentences]")
            title, content = pair
        title = _require_nonempty(title, f"{item_path}.title")
        sentences = _sentences_from_value(content, f"{item_path}.sentences")
        text = "".join(sentences)
        documents.append(
            SourceDocument(
                title=title,
                sentences=sentences,
                source_document_id=_source_document_id(
                    dataset,
                    title,
                    text,
                    supplied_id,
                ),
            )
        )
    return tuple(documents)


def _extract_context(
    record: Mapping[str, Any],
    metadata: Mapping[str, Any],
    dataset: str,
    path: str,
) -> tuple[SourceDocument, ...]:
    context = record.get("context")
    if isinstance(context, str) or context is None:
        context = metadata.get("context")
    if isinstance(context, Mapping):
        documents = _parse_context_mapping(context, dataset, f"{path}.context")
    else:
        documents = _parse_context_sequence(
            _require_sequence(context, f"{path}.context"),
            dataset,
            f"{path}.context",
        )
    if not documents:
        raise _error(f"{path}.context", "must contain documents")
    ids = [document.document_id for document in documents]
    if len(set(ids)) != len(ids):
        raise _error(f"{path}.context", "contains duplicate title/text documents")
    normalized_titles = [normalize_text(document.title) for document in documents]
    if len(set(normalized_titles)) != len(normalized_titles):
        raise _error(
            f"{path}.context",
            "contains ambiguous duplicate normalized titles",
        )
    return documents


def _supporting_pairs(value: Any, path: str) -> tuple[tuple[str, int], ...]:
    if isinstance(value, Mapping):
        titles = _require_sequence(
            _first_present(value, ("title", "titles")),
            f"{path}.title",
        )
        indices = _require_sequence(
            _first_present(
                value,
                ("sent_id", "sent_ids", "sentence_id", "sentence_ids"),
            ),
            f"{path}.sent_id",
        )
        if len(titles) != len(indices):
            raise _error(path, "supporting title and sentence arrays differ")
        raw_pairs: Sequence[Any] = tuple(zip(titles, indices))
    else:
        raw_pairs = _require_sequence(value, path)
    pairs: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for index, item in enumerate(raw_pairs):
        item_path = f"{path}[{index}]"
        if isinstance(item, Mapping):
            title = _first_present(item, ("title", "document_title"))
            sentence_index = _first_present(
                item,
                ("sent_id", "sentence_id", "sentence_index"),
            )
        else:
            pair = _require_sequence(item, item_path)
            if len(pair) != 2:
                raise _error(item_path, "must be [title, sentence_index]")
            title, sentence_index = pair
        title = _require_nonempty(title, f"{item_path}.title")
        sentence_index = _require_int(sentence_index, f"{item_path}.sentence_index")
        key = (normalize_text(title), sentence_index)
        if key not in seen:
            pairs.append((title, sentence_index))
            seen.add(key)
    if not pairs:
        raise _error(path, "must contain sentence-level supporting facts")
    return tuple(pairs)


def _extract_supporting_facts(
    record: Mapping[str, Any],
    metadata: Mapping[str, Any],
    documents: tuple[SourceDocument, ...],
    path: str,
) -> tuple[SupportingFactInput, ...]:
    value = record.get("supporting_facts")
    if value is None:
        value = metadata.get("supporting_facts")
    pairs = _supporting_pairs(value, f"{path}.supporting_facts")
    title_positions = {
        normalize_text(document.title): index for index, document in enumerate(documents)
    }
    facts = []
    out_of_bounds: list[Mapping[str, Any]] = []
    missing_titles: list[Mapping[str, Any]] = []
    available_document_ids = sorted(document.document_id for document in documents)
    for fact_index, (title, sentence_index) in enumerate(pairs):
        normalized_title = normalize_text(title)
        if normalized_title not in title_positions:
            missing_titles.append(
                {
                    "available_document_ids": available_document_ids,
                    "normalized_title_sha256": normalized_text_sha256(title),
                    "raw_title_sha256": _sha256_bytes(title.encode("utf-8")),
                    "supporting_fact_index": fact_index,
                }
            )
            continue
        document_index = title_positions[normalized_title]
        document = documents[document_index]
        if sentence_index >= len(document.sentences):
            out_of_bounds.append(
                {
                    "document_id": document.document_id,
                    "sentence_count": len(document.sentences),
                    "sentence_index": sentence_index,
                    "supporting_fact_index": fact_index,
                }
            )
            continue
        text = document.sentences[sentence_index]
        start = sum(len(sentence) for sentence in document.sentences[:sentence_index])
        facts.append(
            SupportingFactInput(
                document_index=document_index,
                sentence_index=sentence_index,
                text=text,
                start_char=start,
                end_char=start + len(text),
            )
        )
    reasons = []
    if out_of_bounds:
        reasons.append(
            SourceRejectionReason(
                code=_OUT_OF_BOUNDS_REASON,
                evidence=tuple(out_of_bounds),
            )
        )
    if missing_titles:
        reasons.append(
            SourceRejectionReason(
                code=_MISSING_SUPPORT_TITLE_REASON,
                evidence=tuple(missing_titles),
            )
        )
    if reasons:
        reasons.sort(key=lambda value: value.code)
        raise _RejectableSourceRecordError(
            f"{path}.supporting_facts: source evidence cannot be verified",
            reasons=reasons,
        )
    return tuple(facts)


def parse_source_record(
    raw_record: Mapping[str, Any],
    *,
    dataset: str,
    source_index: int,
) -> ParsedExample:
    """Parse native HotpotQA/2Wiki or FlashRAG-style nested metadata."""

    dataset = _canonical_dataset_name(dataset)
    path = f"source[{source_index}]"
    record = _require_mapping(raw_record, path)
    metadata_value = record.get("metadata", {})
    metadata = _require_mapping(metadata_value, f"{path}.metadata")
    question = _extract_question(record, path)
    answers = _extract_answers(record, path)
    source_id_value = _first_present(record, ("_id", "id", "qa_id"))
    source_qa_id = None if source_id_value is None else str(source_id_value)
    documents = _extract_context(record, metadata, dataset, path)
    supporting_facts = _extract_supporting_facts(
        record,
        metadata,
        documents,
        path,
    )
    return ParsedExample(
        source_qa_id=source_qa_id,
        qa=QARecord.create(
            stable_qa_id(dataset, source_qa_id, question),
            question,
            answers,
        ),
        documents=documents,
        supporting_facts=supporting_facts,
    )


def _load_json_records(path: Path, split: str | None) -> list[Mapping[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _error("input", f"invalid UTF-8 JSON: {exc}") from exc
    if isinstance(value, list):
        records = value
    elif isinstance(value, Mapping):
        if split is not None and split in value:
            records = value[split]
        elif "data" in value:
            records = value["data"]
        elif "question" in value or "input" in value:
            records = [value]
        else:
            raise _error("input", "JSON object needs --split, data[], or one QA record")
    else:
        raise _error("input", "JSON root must be an object or array")
    return [
        _require_mapping(record, f"input[{index}]")
        for index, record in enumerate(_require_sequence(records, "input records"))
    ]


def _canonical_source_format(
    path: str | os.PathLike[str],
    source_format: str | None,
) -> str:
    value = Path(path).suffix if source_format is None else source_format
    value = value.lower().lstrip(".")
    aliases = {
        "json": "json",
        "jsonl": "jsonl",
        "ndjson": "jsonl",
        "parquet": "parquet",
        "pq": "parquet",
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise _error(
            "input",
            "supported formats are json, jsonl, and parquet",
        ) from exc


def load_local_records(
    path: str | os.PathLike[str],
    *,
    split: str | None = None,
    source_format: str | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Read a local JSON, JSONL, or Parquet source without network access."""

    requested_path = Path(path).expanduser()
    source_path = requested_path.resolve(strict=True)
    source_format = _canonical_source_format(requested_path, source_format)
    if source_format == "json":
        records = _load_json_records(source_path, split)
    elif source_format == "jsonl":
        records = []
        for line_number, line in enumerate(
            source_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                raise _error("input", f"blank JSONL line {line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise _error("input", f"invalid JSONL line {line_number}") from exc
            records.append(_require_mapping(value, f"input line {line_number}"))
    elif source_format == "parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to read Parquet sources") from exc
        records = [
            _require_mapping(value, f"input[{index}]")
            for index, value in enumerate(pq.read_table(source_path).to_pylist())
        ]
    if not records:
        raise _error("input", "contains no QA records")
    return tuple(records)


def _rank(seed: int, qa_id: str, namespace: str, value: str) -> str:
    return _sha256_bytes(
        canonical_json_bytes(
            {
                "namespace": namespace,
                "qa_id": qa_id,
                "seed": seed,
                "value": value,
            }
        )
    )


def _canonical_corpus(
    examples: Sequence[ParsedExample],
) -> dict[str, DocumentInput]:
    candidates: dict[str, list[DocumentInput]] = {}
    for example in examples:
        for source_document in example.documents:
            document = source_document.to_manifest_input()
            candidates.setdefault(document.document_id, []).append(document)
    corpus = {}
    for document_id, versions in candidates.items():
        identities = {(value.title, value.text) for value in versions}
        if len(identities) != 1:
            raise ManifestValidationError(
                f"stable document ID collision for {document_id}"
            )
        corpus[document_id] = min(
            versions,
            key=lambda value: value.source_document_id or "",
        )
    return corpus


def _select_examples(
    examples: Sequence[ParsedExample],
    *,
    qa_count: int,
    seed: int,
) -> tuple[ParsedExample, ...]:
    qa_ids = [example.qa.qa_id for example in examples]
    selected_ids = _select_qa_ids(qa_ids, qa_count=qa_count, seed=seed)
    by_id = {example.qa.qa_id: example for example in examples}
    return tuple(by_id[qa_id] for qa_id in selected_ids)


def _select_qa_ids(
    qa_ids: Sequence[str],
    *,
    qa_count: int,
    seed: int,
) -> tuple[str, ...]:
    if len(set(qa_ids)) != len(qa_ids):
        raise ManifestValidationError("source contains duplicate stable QA IDs")
    if len(qa_ids) < qa_count:
        raise ManifestValidationError(
            f"source has {len(qa_ids)} QAs but contract requires {qa_count}"
        )
    ordered = sorted(
        qa_ids,
        key=lambda qa_id: (
            _rank(seed, qa_id, "qa-selection", qa_id),
            qa_id,
        ),
    )
    return tuple(ordered[:qa_count])


def _materialize_document_pool(
    example: ParsedExample,
    *,
    corpus: Mapping[str, DocumentInput],
    prefix_document_count: int,
    pool_document_count: int,
    seed: int,
) -> tuple[tuple[DocumentInput, ...], tuple[SupportingFactInput, ...]]:
    if prefix_document_count > pool_document_count:
        raise ManifestValidationError("document prefix exceeds pool size")
    core_ids = tuple(document.document_id for document in example.documents)
    if len(core_ids) > prefix_document_count:
        raise ManifestValidationError(
            f"QA {example.qa.qa_id} has {len(core_ids)} source documents, exceeding "
            f"the {prefix_document_count}-document evidence prefix"
        )
    if len(corpus) < pool_document_count:
        raise ManifestValidationError(
            f"document corpus has {len(corpus)} unique documents but contract requires "
            f"{pool_document_count} per QA"
        )
    core_set = set(core_ids)
    distractors = sorted(
        (document_id for document_id in corpus if document_id not in core_set),
        key=lambda document_id: (
            _rank(seed, example.qa.qa_id, "distractor-selection", document_id),
            document_id,
        ),
    )
    needed = pool_document_count - len(core_ids)
    if len(distractors) < needed:
        raise ManifestValidationError(
            f"QA {example.qa.qa_id} does not have enough unique distractors"
        )
    prefix_padding_count = prefix_document_count - len(core_ids)
    prefix_ids = list(core_ids) + distractors[:prefix_padding_count]
    tail_ids = distractors[prefix_padding_count:needed]
    prefix_ids.sort(
        key=lambda document_id: (
            _rank(seed, example.qa.qa_id, "prefix-order", document_id),
            document_id,
        )
    )
    tail_ids.sort(
        key=lambda document_id: (
            _rank(seed, example.qa.qa_id, "pool-tail-order", document_id),
            document_id,
        )
    )
    pool_ids = tuple(prefix_ids + tail_ids)
    if len(pool_ids) != pool_document_count or len(set(pool_ids)) != len(pool_ids):
        raise ManifestValidationError("deterministic pool construction lost documents")
    positions = {document_id: index for index, document_id in enumerate(pool_ids)}
    remapped_facts = tuple(
        SupportingFactInput(
            document_index=positions[core_ids[fact.document_index]],
            sentence_index=fact.sentence_index,
            text=fact.text,
            start_char=fact.start_char,
            end_char=fact.end_char,
        )
        for fact in example.supporting_facts
    )
    if any(fact.document_index >= prefix_document_count for fact in remapped_facts):
        raise ManifestValidationError("supporting evidence escaped the required prefix")
    return tuple(corpus[document_id] for document_id in pool_ids), remapped_facts


def _parse_all_examples(
    records: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
) -> tuple[ParsedExample, ...]:
    return tuple(
        parse_source_record(record, dataset=dataset, source_index=index)
        for index, record in enumerate(records)
    )


def _curation_context(
    record: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    dataset: str,
    path: str,
) -> tuple[tuple[SourceDocument, ...], tuple[SourceRejectionReason, ...]]:
    context = record.get("context")
    if isinstance(context, str) or context is None:
        context = metadata.get("context")
    if isinstance(context, Mapping):
        documents = _parse_context_mapping(context, dataset, f"{path}.context")
    else:
        documents = _parse_context_sequence(
            _require_sequence(context, f"{path}.context"),
            dataset,
            f"{path}.context",
        )
    if not documents:
        return (
            (),
            (
                SourceRejectionReason(
                    code=_EMPTY_CONTEXT_REASON,
                    evidence=({"document_count": 0},),
                ),
            ),
        )

    positions_by_id: dict[str, list[int]] = {}
    for position, document in enumerate(documents):
        positions_by_id.setdefault(document.document_id, []).append(position)
    duplicate_ids = tuple(
        sorted(
            document_id
            for document_id, positions in positions_by_id.items()
            if len(positions) > 1
        )
    )
    normalized_title_ids: dict[str, set[str]] = {}
    normalized_title_counts: dict[str, int] = {}
    for document in documents:
        title = normalize_text(document.title)
        normalized_title_ids.setdefault(title, set()).add(document.document_id)
        normalized_title_counts[title] = normalized_title_counts.get(title, 0) + 1
    unexplained_title_duplicates = [
        title
        for title, count in normalized_title_counts.items()
        if count > 1 and len(normalized_title_ids[title]) > 1
    ]
    if unexplained_title_duplicates:
        raise _error(
            f"{path}.context",
            "contains ambiguous duplicate normalized titles",
        )
    if not duplicate_ids:
        return documents, ()
    evidence = tuple(
        {
            "document_id": document_id,
            "occurrence_positions": positions_by_id[document_id],
            "raw_variant_sha256s": [
                _raw_document_variant_sha256(documents[position])
                for position in positions_by_id[document_id]
            ],
        }
        for document_id in duplicate_ids
    )
    return (
        documents,
        (
            SourceRejectionReason(
                code=_DUPLICATE_CONTEXT_DOCUMENT_REASON,
                evidence=evidence,
            ),
        ),
    )


def _source_record_identity(
    record: Mapping[str, Any],
    *,
    dataset: str,
    source_index: int,
) -> tuple[
    str,
    tuple[SourceDocument, ...],
    tuple[SourceRejectionReason, ...],
]:
    path = f"source[{source_index}]"
    metadata = _require_mapping(record.get("metadata", {}), f"{path}.metadata")
    source_id_value = _first_present(record, ("_id", "id", "qa_id"))
    source_qa_id = None if source_id_value is None else str(source_id_value)
    question = _extract_question(record, path)
    _extract_answers(record, path)
    value = record.get("supporting_facts")
    if value is None:
        value = metadata.get("supporting_facts")
    _supporting_pairs(value, f"{path}.supporting_facts")
    qa_id = stable_qa_id(dataset, source_qa_id, question)
    documents, reasons = _curation_context(
        record,
        metadata,
        dataset=dataset,
        path=path,
    )
    return qa_id, documents, reasons


def _source_record_sha256(record: Mapping[str, Any]) -> str:
    return _sha256_bytes(canonical_json_bytes(record))


def _raw_document_variant_sha256(document: SourceDocument) -> str:
    return _sha256_bytes(
        canonical_json_bytes(
            {
                "text": document.text,
                "title": document.title,
            }
        )
    )


def _source_indices_sha256(indices: Sequence[int]) -> str:
    return _sha256_bytes(canonical_json_bytes(list(indices)))


def curate_source_records(
    records: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
) -> SourceCurationResult:
    """Apply the fixed, label-independent source rejection policy."""

    dataset = _canonical_dataset_name(dataset)
    structured: dict[
        int,
        tuple[Mapping[str, Any], str, tuple[SourceDocument, ...]],
    ] = {}
    parsed_by_index: dict[int, ParsedExample] = {}
    reasons_by_index: dict[int, tuple[SourceRejectionReason, ...]] = {}
    for source_index, raw_record in enumerate(records):
        record = _require_mapping(raw_record, f"source[{source_index}]")
        qa_id, documents, structural_reasons = _source_record_identity(
            record,
            dataset=dataset,
            source_index=source_index,
        )
        structured[source_index] = (record, qa_id, documents)
        if structural_reasons:
            reasons_by_index[source_index] = structural_reasons
            continue
        try:
            example = parse_source_record(
                record,
                dataset=dataset,
                source_index=source_index,
            )
        except _RejectableSourceRecordError as exc:
            if any(
                reason.code
                not in {_OUT_OF_BOUNDS_REASON, _MISSING_SUPPORT_TITLE_REASON}
                for reason in exc.reasons
            ):
                raise
            reasons_by_index[source_index] = exc.reasons
        else:
            if example.qa.qa_id != qa_id or example.documents != documents:
                raise ManifestValidationError(
                    "source identity extraction differs from full parsing"
                )
            parsed_by_index[source_index] = example

    variants_by_document: dict[str, set[str]] = {}
    for _, _, documents in structured.values():
        for document in documents:
            variants_by_document.setdefault(document.document_id, set()).add(
                _raw_document_variant_sha256(document)
            )
    ambiguous_document_ids = tuple(
        sorted(
            document_id
            for document_id, variants in variants_by_document.items()
            if len(variants) > 1
        )
    )
    ambiguous_set = set(ambiguous_document_ids)

    accepted: list[ParsedExample] = []
    accepted_source_indices: list[int] = []
    rejections: list[SourceRejection] = []
    for source_index in range(len(records)):
        record, qa_id, documents = structured[source_index]
        reasons = list(reasons_by_index.get(source_index, ()))
        affected = tuple(
            sorted(
                {
                    document.document_id
                    for document in documents
                    if document.document_id in ambiguous_set
                }
            )
        )
        if affected:
            evidence = tuple(
                {
                    "document_id": document_id,
                    "observed_raw_variant_sha256s": sorted(
                        {
                            _raw_document_variant_sha256(document)
                            for document in documents
                            if document.document_id == document_id
                        }
                    ),
                    "raw_variant_sha256s": sorted(
                        variants_by_document[document_id]
                    ),
                }
                for document_id in affected
            )
            reasons.append(
                SourceRejectionReason(
                    code=_AMBIGUOUS_DOCUMENT_REASON,
                    evidence=evidence,
                )
            )
        if reasons:
            reasons.sort(key=lambda value: value.code)
            rejections.append(
                SourceRejection(
                    source_index=source_index,
                    qa_id=qa_id,
                    source_record_sha256=_source_record_sha256(record),
                    reasons=tuple(reasons),
                    ambiguous_document_ids=affected,
                )
            )
            continue
        accepted.append(parsed_by_index[source_index])
        accepted_source_indices.append(source_index)

    rejections.sort(key=lambda value: value.source_index)
    if len(accepted) + len(rejections) != len(records):
        raise ManifestValidationError("source curation did not account for every record")
    if len({value.source_index for value in rejections}) != len(rejections):
        raise ManifestValidationError("source curation rejected one record more than once")
    _canonical_corpus(accepted)
    return SourceCurationResult(
        examples=tuple(accepted),
        accepted_source_indices=tuple(accepted_source_indices),
        rejections=tuple(rejections),
        ambiguous_document_ids=ambiguous_document_ids,
    )


def build_train_records(
    examples: Sequence[ParsedExample],
    *,
    metadata: ManifestMetadata,
    encode: Callable[[str], Any],
    contract: TrainManifestContract = TrainManifestContract(),
) -> tuple[ManifestRecord, ...]:
    """Build the fixed 512/200 train set and enforce the 6-chunk token window."""

    contract.validate()
    metadata.validate()
    selected = _select_examples(examples, qa_count=contract.qa_count, seed=metadata.seed)
    corpus = _canonical_corpus(examples)
    qa_ids = tuple(example.qa.qa_id for example in selected)
    qa_order_sha256 = ordered_values_sha256(qa_ids)
    records = []
    for qa_index, example in enumerate(selected):
        pool, facts = _materialize_document_pool(
            example,
            corpus=corpus,
            prefix_document_count=contract.document_count,
            pool_document_count=contract.document_count,
            seed=metadata.seed,
        )
        pool_hash = ordered_values_sha256(
            tuple(document.document_id for document in pool)
        )
        record = build_manifest_record(
            metadata=metadata,
            qa_index=qa_index,
            qa_order_sha256=qa_order_sha256,
            qa=example.qa,
            documents=pool,
            supporting_facts=facts,
            encode=encode,
            chunk_size=contract.chunk_size,
            pool_document_count=contract.document_count,
            document_pool_sha256=pool_hash,
        )
        if not (
            contract.min_context_tokens
            <= record.context_token_count
            <= contract.max_context_tokens
        ):
            raise ManifestValidationError(
                f"QA {record.qa.qa_id} context has {record.context_token_count} tokens; "
                f"formal range is [{contract.min_context_tokens}, "
                f"{contract.max_context_tokens}]"
            )
        if len(record.chunks) != contract.max_chunks:
            raise ManifestValidationError(
                f"QA {record.qa.qa_id} must materialize exactly "
                f"{contract.max_chunks} chunks"
            )
        records.append(record)
    result = tuple(records)
    validate_train_records(result, contract=contract)
    return result


def validate_train_records(
    records: Sequence[ManifestRecord],
    *,
    contract: TrainManifestContract = TrainManifestContract(),
) -> None:
    contract.validate()
    records = tuple(records)
    if len(records) != contract.qa_count:
        raise ManifestValidationError("train manifest has the wrong QA count")
    qa_ids = tuple(record.qa.qa_id for record in records)
    if len(set(qa_ids)) != len(qa_ids):
        raise ManifestValidationError("train manifest contains duplicate QA IDs")
    order_hash = ordered_values_sha256(qa_ids)
    metadata = records[0].metadata
    for index, record in enumerate(records):
        validate_manifest_record(record, f"train_records[{index}]")
        if record.qa_index != index or record.qa_order_sha256 != order_hash:
            raise ManifestValidationError("train QA index/order hash mismatch")
        if record.metadata != metadata:
            raise ManifestValidationError("train metadata differs between records")
        if record.document_count != contract.document_count:
            raise ManifestValidationError("train record has the wrong document count")
        if record.pool_document_count != contract.document_count:
            raise ManifestValidationError("train pool count is inconsistent")
        if record.chunk_size != contract.chunk_size:
            raise ManifestValidationError("train chunk size differs from contract")
        if len(record.chunks) != contract.max_chunks:
            raise ManifestValidationError("train record has the wrong chunk count")
        if not (
            contract.min_context_tokens
            <= record.context_token_count
            <= contract.max_context_tokens
        ):
            raise ManifestValidationError("train context token count is outside contract")


def build_eval_pair(
    examples: Sequence[ParsedExample],
    *,
    metadata: ManifestMetadata,
    encode: Callable[[str], Any],
    contract: EvalManifestContract = EvalManifestContract(),
):
    contract.validate()
    selected = _select_examples(examples, qa_count=contract.qa_count, seed=metadata.seed)
    corpus = _canonical_corpus(examples)
    materialized = []
    for example in selected:
        pool, facts = _materialize_document_pool(
            example,
            corpus=corpus,
            prefix_document_count=contract.prefix_document_count,
            pool_document_count=contract.pool_document_count,
            seed=metadata.seed,
        )
        materialized.append(
            EvalExampleInput(
                qa=example.qa,
                document_pool=pool,
                supporting_facts=facts,
            )
        )
    return build_eval_manifest_pair(
        materialized,
        metadata=metadata,
        encode=encode,
        contract=contract,
    )


def _row_from_record(
    record: ManifestRecord,
    *,
    dataset: str,
    variant: str,
    sidecar_name: str,
) -> dict[str, Any]:
    answers = [answer.text for answer in record.qa.gold_answers]
    return {
        "answers": answers,
        "context": record.context,
        "data_source": dataset,
        "extra_info": {
            "chunk_count": len(record.chunks),
            "chunk_size": record.chunk_size,
            "context_sha256": record.context_sha256,
            "context_token_ids_sha256": record.context_token_ids_sha256,
            "context_token_count": record.context_token_count,
            "document_count": record.document_count,
            "document_pool_sha256": record.document_pool_sha256,
            "index": record.qa_index,
            "manifest_record_sha256": record.record_sha256,
            "manifest_sidecar": sidecar_name,
            "qa_id": record.qa.qa_id,
            "qa_order_sha256": record.qa_order_sha256,
            "supporting_fact_ids": [
                fact.fact_id for fact in record.supporting_facts
            ],
            "variant": variant,
        },
        "prompt": [{"content": record.qa.question, "role": "user"}],
        "reward_model": {"ground_truth": answers},
    }


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to write reproduction Parquet") from exc
    table = pa.Table.from_pylist(list(rows))
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
        data_page_version="1.0",
    )


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    keys = set(value)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise _error(path, f"schema keys differ; missing={missing}, extra={extra}")


def _tuple_of_strings(value: Any, path: str) -> tuple[str, ...]:
    return tuple(
        _require_nonempty(item, f"{path}[{index}]")
        for index, item in enumerate(_require_sequence(value, path))
    )


def manifest_record_from_dict(value: Mapping[str, Any]) -> ManifestRecord:
    """Strictly reconstruct a sidecar record for independent readback checks."""

    value = _require_mapping(value, "record")
    _exact_keys(
        value,
        {
            "chunk_size",
            "chunks",
            "consumed_token_count",
            "context",
            "context_sha256",
            "context_token_ids_sha256",
            "context_token_count",
            "document_count",
            "document_pool_sha256",
            "documents",
            "metadata",
            "pool_document_count",
            "qa",
            "qa_index",
            "qa_order_sha256",
            "record_sha256",
            "schema_version",
            "supporting_facts",
            "truncation",
        },
        "record",
    )
    if value["schema_version"] != SCHEMA_VERSION:
        raise _error("record.schema_version", "unsupported value")
    metadata_value = _require_mapping(value["metadata"], "record.metadata")
    _exact_keys(
        metadata_value,
        {
            "schema_version",
            "seed",
            "source_name",
            "source_revision",
            "source_sha256",
            "tokenizer_name",
            "tokenizer_revision",
        },
        "record.metadata",
    )
    metadata = ManifestMetadata(
        source_name=metadata_value["source_name"],
        source_revision=metadata_value["source_revision"],
        source_sha256=metadata_value["source_sha256"],
        tokenizer_name=metadata_value["tokenizer_name"],
        tokenizer_revision=metadata_value["tokenizer_revision"],
        seed=metadata_value["seed"],
        schema_version=metadata_value["schema_version"],
    )
    qa_value = _require_mapping(value["qa"], "record.qa")
    _exact_keys(
        qa_value,
        {"gold_answers", "qa_id", "question", "question_sha256"},
        "record.qa",
    )
    gold_answers = []
    for index, answer_value in enumerate(
        _require_sequence(qa_value["gold_answers"], "record.qa.gold_answers")
    ):
        answer_value = _require_mapping(
            answer_value,
            f"record.qa.gold_answers[{index}]",
        )
        _exact_keys(
            answer_value,
            {"text", "text_sha256"},
            f"record.qa.gold_answers[{index}]",
        )
        gold_answers.append(
            GoldAnswer(
                text=answer_value["text"],
                text_sha256=answer_value["text_sha256"],
            )
        )
    qa = QARecord(
        qa_id=qa_value["qa_id"],
        question=qa_value["question"],
        question_sha256=qa_value["question_sha256"],
        gold_answers=tuple(gold_answers),
    )
    documents = []
    for index, document_value in enumerate(
        _require_sequence(value["documents"], "record.documents")
    ):
        document_value = _require_mapping(document_value, f"record.documents[{index}]")
        _exact_keys(
            document_value,
            {
                "document_id",
                "position",
                "source_document_id",
                "supporting_fact_ids",
                "text",
                "text_sha256",
                "title",
                "title_sha256",
                "token_end",
                "token_start",
            },
            f"record.documents[{index}]",
        )
        documents.append(
            DocumentRecord(
                document_id=document_value["document_id"],
                position=document_value["position"],
                title=document_value["title"],
                text=document_value["text"],
                title_sha256=document_value["title_sha256"],
                text_sha256=document_value["text_sha256"],
                source_document_id=document_value["source_document_id"],
                token_start=document_value["token_start"],
                token_end=document_value["token_end"],
                supporting_fact_ids=_tuple_of_strings(
                    document_value["supporting_fact_ids"],
                    f"record.documents[{index}].supporting_fact_ids",
                ),
            )
        )
    facts = []
    for index, fact_value in enumerate(
        _require_sequence(value["supporting_facts"], "record.supporting_facts")
    ):
        fact_value = _require_mapping(fact_value, f"record.supporting_facts[{index}]")
        _exact_keys(
            fact_value,
            {
                "document_id",
                "document_position",
                "fact_id",
                "sentence_index",
                "text",
                "text_sha256",
                "token_end",
                "token_start",
            },
            f"record.supporting_facts[{index}]",
        )
        facts.append(
            SupportingFactRecord(
                fact_id=fact_value["fact_id"],
                document_id=fact_value["document_id"],
                document_position=fact_value["document_position"],
                sentence_index=fact_value["sentence_index"],
                text=fact_value["text"],
                text_sha256=fact_value["text_sha256"],
                token_start=fact_value["token_start"],
                token_end=fact_value["token_end"],
            )
        )
    chunks = []
    for index, chunk_value in enumerate(
        _require_sequence(value["chunks"], "record.chunks")
    ):
        chunk_value = _require_mapping(chunk_value, f"record.chunks[{index}]")
        _exact_keys(
            chunk_value,
            {
                "chunk_id",
                "chunk_index",
                "document_ids",
                "supporting_fact_ids",
                "token_end",
                "token_start",
            },
            f"record.chunks[{index}]",
        )
        chunks.append(
            ChunkRecord(
                chunk_id=chunk_value["chunk_id"],
                chunk_index=chunk_value["chunk_index"],
                token_start=chunk_value["token_start"],
                token_end=chunk_value["token_end"],
                document_ids=_tuple_of_strings(
                    chunk_value["document_ids"],
                    f"record.chunks[{index}].document_ids",
                ),
                supporting_fact_ids=_tuple_of_strings(
                    chunk_value["supporting_fact_ids"],
                    f"record.chunks[{index}].supporting_fact_ids",
                ),
            )
        )
    record = ManifestRecord(
        metadata=metadata,
        qa_index=value["qa_index"],
        qa_order_sha256=value["qa_order_sha256"],
        qa=qa,
        document_count=value["document_count"],
        pool_document_count=value["pool_document_count"],
        document_pool_sha256=value["document_pool_sha256"],
        documents=tuple(documents),
        supporting_facts=tuple(facts),
        chunks=tuple(chunks),
        context=value["context"],
        context_sha256=value["context_sha256"],
        context_token_ids_sha256=value["context_token_ids_sha256"],
        context_token_count=value["context_token_count"],
        consumed_token_count=value["consumed_token_count"],
        chunk_size=value["chunk_size"],
        truncation=value["truncation"],
        record_sha256=value["record_sha256"],
    )
    validate_manifest_record(record)
    return record


def _artifact_entry(path: Path, *, kind: str, row_count: int, variant: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "path": path.name,
        "row_count": row_count,
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "variant": variant,
    }


def _curation_payload(
    curation: SourceCurationResult,
    *,
    source_sha256: str,
    ledger: Mapping[str, Any],
) -> Mapping[str, Any]:
    accepted_qa_ids = tuple(example.qa.qa_id for example in curation.examples)
    reason_counts = {
        reason: sum(
            any(item.code == reason for item in value.reasons)
            for value in curation.rejections
        )
        for reason in _REJECTION_REASONS
    }
    return {
        "accepted_qa_order_sha256": ordered_values_sha256(accepted_qa_ids),
        "accepted_record_count": len(curation.examples),
        "accepted_source_order_sha256": _source_indices_sha256(
            curation.accepted_source_indices
        ),
        "ambiguous_document_ids": list(curation.ambiguous_document_ids),
        "input_record_count": len(curation.examples) + len(curation.rejections),
        "policy": CURATION_POLICY,
        "rejected_record_count": len(curation.rejections),
        "rejection_ledger": dict(ledger),
        "rejection_reason_counts": reason_counts,
        "schema_version": CURATION_SCHEMA_VERSION,
        "source_sha256": source_sha256,
    }


def _write_curation_ledger(
    output_dir: Path,
    curation: SourceCurationResult,
    *,
    source_sha256: str,
) -> Mapping[str, Any]:
    path = output_dir / REJECTION_LEDGER_NAME
    ledger_payload = _curation_ledger_payload(
        curation,
        source_sha256=source_sha256,
    )
    path.write_bytes(canonical_json_bytes(ledger_payload) + b"\n")
    digest = _sha256_file(path)
    return _curation_payload(
        curation,
        source_sha256=source_sha256,
        ledger={
            "path": path.name,
            "row_count": len(curation.rejections),
            "sha256": digest,
            "size_bytes": path.stat().st_size,
        },
    )


def _curation_ledger_payload(
    curation: SourceCurationResult,
    *,
    source_sha256: str,
) -> Mapping[str, Any]:
    accepted_qa_ids = tuple(example.qa.qa_id for example in curation.examples)
    return {
        "accepted_qa_ids": list(accepted_qa_ids),
        "ambiguous_document_ids": list(curation.ambiguous_document_ids),
        "input_record_count": len(curation.examples) + len(curation.rejections),
        "policy": CURATION_POLICY,
        "rejections": [value.to_dict() for value in curation.rejections],
        "schema_version": CURATION_SCHEMA_VERSION,
        "source_sha256": source_sha256,
    }


def _manifest_payload(
    *,
    mode: str,
    profile: str,
    dataset: str,
    seed: int,
    source_path: Path,
    source_revision: str,
    source_sha256: str,
    source_record_count: int,
    source_format: str,
    source_split: str | None,
    tokenizer_name: str,
    tokenizer_revision: str,
    contract: Mapping[str, Any],
    qa_ids: Sequence[str],
    artifacts: Mapping[str, Mapping[str, Any]],
    curation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source = {
        "path": str(source_path),
        "record_count": source_record_count,
        "revision": source_revision,
        "sha256": source_sha256,
        "split": source_split,
    }
    payload = {
        "artifacts": dict(artifacts),
        "contract": dict(contract),
        "dataset": dataset,
        "kind": FORMAL_BUNDLE_KIND if profile == "formal" else BUNDLE_KIND,
        "mode": mode,
        "profile": profile,
        "qa_ids": list(qa_ids),
        "qa_order_sha256": ordered_values_sha256(tuple(qa_ids)),
        "schema_version": (
            FORMAL_BUNDLE_SCHEMA_VERSION
            if profile == "formal"
            else BUNDLE_SCHEMA_VERSION
        ),
        "seed": seed,
        "source": source,
        "tokenizer": {
            "add_special_tokens": False,
            "local_files_only": True,
            "name": tokenizer_name,
            "revision": tokenizer_revision,
            "return_offsets_mapping": True,
        },
    }
    if profile == "formal":
        if curation is None:
            raise ManifestValidationError("formal bundle is missing source curation")
        source["format"] = source_format
        payload["curation"] = dict(curation)
    elif curation is not None:
        raise ManifestValidationError("fixture bundle cannot publish source curation")
    return payload


def _write_top_manifest(path: Path, payload: Mapping[str, Any]) -> str:
    payload = dict(payload)
    digest = _sha256_bytes(canonical_json_bytes(payload))
    final = {**payload, "manifest_sha256": digest}
    path.write_bytes(canonical_json_bytes(final) + b"\n")
    return digest


def _check_profile(
    mode: str,
    profile: str,
    contract: TrainManifestContract | EvalManifestContract,
) -> None:
    if profile not in {"formal", "fixture"}:
        raise _error("profile", "must be formal or fixture")
    if profile == "formal":
        expected: TrainManifestContract | EvalManifestContract
        expected = TrainManifestContract() if mode == "train" else EvalManifestContract()
        if contract != expected:
            raise ManifestValidationError(
                "formal profile cannot override QA/document/token contracts"
            )


def _build_in_directory(
    output_dir: Path,
    *,
    mode: str,
    profile: str,
    dataset: str,
    source_path: Path,
    source_revision: str,
    tokenizer_name: str,
    tokenizer_revision: str,
    seed: int,
    encode: Callable[[str], Any],
    train_contract: TrainManifestContract,
    eval_contract: EvalManifestContract,
    split: str | None,
    source_format: str | None,
) -> None:
    canonical_source_format = _canonical_source_format(source_path, source_format)
    records = load_local_records(
        source_path,
        split=split,
        source_format=canonical_source_format,
    )
    source_sha256 = _sha256_file(source_path)
    if profile == "formal":
        curation = curate_source_records(records, dataset=dataset)
        parsed = curation.examples
        curation_value: Mapping[str, Any] | None = _write_curation_ledger(
            output_dir,
            curation,
            source_sha256=source_sha256,
        )
    else:
        parsed = _parse_all_examples(records, dataset=dataset)
        curation_value = None
    metadata = ManifestMetadata(
        source_name=dataset,
        source_revision=source_revision,
        source_sha256=source_sha256,
        tokenizer_name=tokenizer_name,
        tokenizer_revision=tokenizer_revision,
        seed=seed,
    )
    artifact_entries: dict[str, Mapping[str, Any]] = {}
    if mode == "train":
        built = build_train_records(
            parsed,
            metadata=metadata,
            encode=encode,
            contract=train_contract,
        )
        variants = (("train", built),)
        contract_value = train_contract.to_dict()
    else:
        pair = build_eval_pair(
            parsed,
            metadata=metadata,
            encode=encode,
            contract=eval_contract,
        )
        variants = (
            (str(eval_contract.prefix_document_count), pair.prefix_records),
            (str(eval_contract.pool_document_count), pair.pool_records),
        )
        contract_value = {
            "chunk_size": eval_contract.chunk_size,
            "pool_document_count": eval_contract.pool_document_count,
            "prefix_document_count": eval_contract.prefix_document_count,
            "qa_count": eval_contract.qa_count,
        }
    qa_ids: tuple[str, ...] | None = None
    for variant, variant_records in variants:
        stem = "train" if mode == "train" else f"eval_{variant}"
        sidecar_name = f"{stem}.sidecar.jsonl"
        parquet_name = f"{stem}.parquet"
        sidecar_path = output_dir / sidecar_name
        parquet_path = output_dir / parquet_name
        sidecar_sha256 = write_canonical_jsonl(sidecar_path, variant_records)
        if sidecar_sha256 != canonical_jsonl_sha256(variant_records):
            raise ManifestValidationError("sidecar hash changed while writing")
        rows = tuple(
            _row_from_record(
                record,
                dataset=dataset,
                variant=variant,
                sidecar_name=sidecar_name,
            )
            for record in variant_records
        )
        _write_parquet(parquet_path, rows)
        current_ids = tuple(record.qa.qa_id for record in variant_records)
        if qa_ids is None:
            qa_ids = current_ids
        elif current_ids != qa_ids:
            raise ManifestValidationError("artifact variants have different QA order")
        artifact_entries[sidecar_name] = _artifact_entry(
            sidecar_path,
            kind="canonical-sidecar-jsonl",
            row_count=len(variant_records),
            variant=variant,
        )
        artifact_entries[parquet_name] = _artifact_entry(
            parquet_path,
            kind="memory-dataset-parquet",
            row_count=len(rows),
            variant=variant,
        )
    assert qa_ids is not None
    payload = _manifest_payload(
        mode=mode,
        profile=profile,
        dataset=dataset,
        seed=seed,
        source_path=source_path,
        source_revision=source_revision,
        source_sha256=source_sha256,
        source_record_count=len(records),
        source_format=canonical_source_format,
        source_split=split,
        tokenizer_name=tokenizer_name,
        tokenizer_revision=tokenizer_revision,
        contract=contract_value,
        qa_ids=qa_ids,
        artifacts=artifact_entries,
        curation=curation_value,
    )
    _write_top_manifest(output_dir / "manifest.json", payload)


def _read_top_manifest(path: Path) -> Mapping[str, Any]:
    payload = path.read_bytes()
    if not payload.endswith(b"\n") or payload.count(b"\n") != 1:
        raise ManifestValidationError("manifest.json must be one canonical JSON line")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestValidationError("manifest.json is invalid UTF-8 JSON") from exc
    value = _require_mapping(value, "manifest")
    base_keys = {
        "artifacts",
        "contract",
        "dataset",
        "kind",
        "manifest_sha256",
        "mode",
        "profile",
        "qa_ids",
        "qa_order_sha256",
        "schema_version",
        "seed",
        "source",
        "tokenizer",
    }
    profile = value.get("profile")
    if profile == "formal":
        expected_keys = base_keys | {"curation"}
        expected_kind = FORMAL_BUNDLE_KIND
        expected_schema = FORMAL_BUNDLE_SCHEMA_VERSION
    elif profile == "fixture":
        expected_keys = base_keys
        expected_kind = BUNDLE_KIND
        expected_schema = BUNDLE_SCHEMA_VERSION
    else:
        raise ManifestValidationError("manifest profile must be formal or fixture")
    _exact_keys(
        value,
        expected_keys,
        "manifest",
    )
    if canonical_json_bytes(value) + b"\n" != payload:
        raise ManifestValidationError("manifest.json is not canonical")
    expected_hash = value["manifest_sha256"]
    without_hash = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if expected_hash != _sha256_bytes(canonical_json_bytes(without_hash)):
        raise ManifestValidationError("manifest self-hash mismatch")
    if value["kind"] != expected_kind or value["schema_version"] != expected_schema:
        raise ManifestValidationError("bundle profile/kind/schema mismatch")
    return value


def _validate_row(
    row: Mapping[str, Any],
    record: ManifestRecord,
    *,
    dataset: str,
    variant: str,
    sidecar_name: str,
) -> None:
    expected = _row_from_record(
        record,
        dataset=dataset,
        variant=variant,
        sidecar_name=sidecar_name,
    )
    if canonical_json_bytes(row) != canonical_json_bytes(expected):
        raise ManifestValidationError(
            f"Parquet row {record.qa_index} differs from its sidecar"
        )


def _validate_curation(
    output_path: Path,
    raw_curation: Any,
    *,
    source: Mapping[str, Any],
    dataset: str,
) -> tuple[str, ...]:
    curation = _require_mapping(raw_curation, "manifest.curation")
    _exact_keys(
        curation,
        {
            "accepted_qa_order_sha256",
            "accepted_record_count",
            "accepted_source_order_sha256",
            "ambiguous_document_ids",
            "input_record_count",
            "policy",
            "rejected_record_count",
            "rejection_ledger",
            "rejection_reason_counts",
            "schema_version",
            "source_sha256",
        },
        "manifest.curation",
    )
    ledger = _require_mapping(
        curation["rejection_ledger"],
        "manifest.curation.rejection_ledger",
    )
    _exact_keys(
        ledger,
        {"path", "row_count", "sha256", "size_bytes"},
        "manifest.curation.rejection_ledger",
    )
    if ledger["path"] != REJECTION_LEDGER_NAME:
        raise ManifestValidationError("curation ledger path changed")
    _require_int(ledger["row_count"], "manifest.curation.rejection_ledger.row_count")
    _require_int(ledger["size_bytes"], "manifest.curation.rejection_ledger.size_bytes")
    _require_sha256(ledger["sha256"], "manifest.curation.rejection_ledger.sha256")
    ledger_path = output_path / REJECTION_LEDGER_NAME
    if ledger_path.is_symlink() or not ledger_path.is_file():
        raise ManifestValidationError("curation ledger is missing or is a symlink")
    if ledger_path.stat().st_size != ledger["size_bytes"]:
        raise ManifestValidationError("curation ledger size mismatch")
    payload = ledger_path.read_bytes()
    if _sha256_bytes(payload) != ledger["sha256"]:
        raise ManifestValidationError("curation ledger hash mismatch")
    if not payload.endswith(b"\n") or payload.count(b"\n") != 1:
        raise ManifestValidationError("curation ledger must be one canonical JSON line")
    try:
        ledger_value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestValidationError("curation ledger is invalid UTF-8 JSON") from exc
    ledger_value = _require_mapping(ledger_value, "curation_ledger")
    _exact_keys(
        ledger_value,
        {
            "accepted_qa_ids",
            "ambiguous_document_ids",
            "input_record_count",
            "policy",
            "rejections",
            "schema_version",
            "source_sha256",
        },
        "curation_ledger",
    )
    if canonical_json_bytes(ledger_value) + b"\n" != payload:
        raise ManifestValidationError("curation ledger is not canonical")
    if (
        ledger_value["schema_version"] != CURATION_SCHEMA_VERSION
        or ledger_value["policy"] != CURATION_POLICY
        or ledger_value["source_sha256"] != source["sha256"]
        or ledger_value["input_record_count"] != source["record_count"]
    ):
        raise ManifestValidationError("curation ledger identity mismatch")

    accepted_qa_ids = _tuple_of_strings(
        ledger_value["accepted_qa_ids"],
        "curation_ledger.accepted_qa_ids",
    )
    if len(set(accepted_qa_ids)) != len(accepted_qa_ids):
        raise ManifestValidationError("curation ledger accepted QA IDs contain duplicates")
    qa_namespace = f"{dataset}:"
    if any(not qa_id.startswith(qa_namespace) for qa_id in accepted_qa_ids):
        raise ManifestValidationError("accepted QA ID has the wrong dataset namespace")
    ambiguous_ids = _tuple_of_strings(
        ledger_value["ambiguous_document_ids"],
        "curation_ledger.ambiguous_document_ids",
    )
    if ambiguous_ids != tuple(sorted(set(ambiguous_ids))):
        raise ManifestValidationError("curation ledger ambiguous IDs are not sorted unique")

    raw_rejections = _require_sequence(
        ledger_value["rejections"],
        "curation_ledger.rejections",
    )
    rejected_indices: list[int] = []
    rejected_qa_ids: list[str] = []
    observed_ambiguous: set[str] = set()
    declared_variants: dict[str, tuple[str, ...]] = {}
    observed_current_variants: dict[str, set[str]] = {}
    reason_counts = {reason: 0 for reason in _REJECTION_REASONS}
    for row_index, raw_rejection in enumerate(raw_rejections):
        path = f"curation_ledger.rejections[{row_index}]"
        rejection = _require_mapping(raw_rejection, path)
        _exact_keys(
            rejection,
            {
                "ambiguous_document_ids",
                "qa_id",
                "reasons",
                "source_index",
                "source_record_sha256",
            },
            path,
        )
        source_index = _require_int(rejection["source_index"], f"{path}.source_index")
        if source_index >= source["record_count"]:
            raise ManifestValidationError("curation rejection source index is out of range")
        rejected_indices.append(source_index)
        rejected_qa_id = _require_nonempty(rejection["qa_id"], f"{path}.qa_id")
        if not rejected_qa_id.startswith(qa_namespace):
            raise ManifestValidationError("rejected QA ID has the wrong dataset namespace")
        rejected_qa_ids.append(rejected_qa_id)
        _require_sha256(rejection["source_record_sha256"], f"{path}.source_record_sha256")
        row_ambiguous = _tuple_of_strings(
            rejection["ambiguous_document_ids"],
            f"{path}.ambiguous_document_ids",
        )
        if row_ambiguous != tuple(sorted(set(row_ambiguous))):
            raise ManifestValidationError("rejection ambiguous IDs are not sorted unique")
        raw_reasons = _require_sequence(rejection["reasons"], f"{path}.reasons")
        if not raw_reasons:
            raise ManifestValidationError("curation rejection must contain reasons")
        reason_codes: list[str] = []
        ambiguity_evidence_ids: list[str] = []
        for reason_index, raw_reason in enumerate(raw_reasons):
            reason_path = f"{path}.reasons[{reason_index}]"
            reason = _require_mapping(raw_reason, reason_path)
            _exact_keys(reason, {"code", "evidence"}, reason_path)
            code = _require_nonempty(reason["code"], f"{reason_path}.code")
            if code not in _REJECTION_REASONS:
                raise ManifestValidationError("curation rejection reason is not allowed")
            reason_codes.append(code)
            reason_counts[code] += 1
            evidence = _require_sequence(reason["evidence"], f"{reason_path}.evidence")
            if not evidence:
                raise ManifestValidationError("curation rejection evidence is empty")
            supporting_fact_indices: list[int] = []
            duplicate_evidence_ids: list[str] = []
            for evidence_index, raw_item in enumerate(evidence):
                evidence_path = f"{reason_path}.evidence[{evidence_index}]"
                item = _require_mapping(raw_item, evidence_path)
                if code == _AMBIGUOUS_DOCUMENT_REASON:
                    _exact_keys(
                        item,
                        {
                            "document_id",
                            "observed_raw_variant_sha256s",
                            "raw_variant_sha256s",
                        },
                        evidence_path,
                    )
                    document_id = _require_nonempty(item["document_id"], f"{evidence_path}.document_id")
                    observed = _tuple_of_strings(
                        item["observed_raw_variant_sha256s"],
                        f"{evidence_path}.observed_raw_variant_sha256s",
                    )
                    variants = _tuple_of_strings(
                        item["raw_variant_sha256s"],
                        f"{evidence_path}.raw_variant_sha256s",
                    )
                    if variants != tuple(sorted(set(variants))) or len(variants) < 2:
                        raise ManifestValidationError("ambiguous raw variants are not sorted unique")
                    for variant_index, variant in enumerate(variants):
                        _require_sha256(variant, f"{evidence_path}.raw_variant_sha256s[{variant_index}]")
                    if observed != tuple(sorted(set(observed))) or not observed:
                        raise ManifestValidationError("observed raw variants are not sorted unique")
                    for variant_index, variant in enumerate(observed):
                        _require_sha256(
                            variant,
                            f"{evidence_path}.observed_raw_variant_sha256s[{variant_index}]",
                        )
                    if not set(observed).issubset(variants):
                        raise ManifestValidationError("observed raw variant is absent from inventory")
                    previous_variants = declared_variants.setdefault(document_id, variants)
                    if previous_variants != variants:
                        raise ManifestValidationError(
                            "ambiguous raw variant inventory differs across rejections"
                        )
                    observed_current_variants.setdefault(document_id, set()).update(observed)
                    ambiguity_evidence_ids.append(document_id)
                    observed_ambiguous.add(document_id)
                elif code == _OUT_OF_BOUNDS_REASON:
                    _exact_keys(
                        item,
                        {"document_id", "sentence_count", "sentence_index", "supporting_fact_index"},
                        evidence_path,
                    )
                    _require_nonempty(item["document_id"], f"{evidence_path}.document_id")
                    sentence_count = _require_int(item["sentence_count"], f"{evidence_path}.sentence_count", 1)
                    sentence_index = _require_int(item["sentence_index"], f"{evidence_path}.sentence_index")
                    supporting_fact_indices.append(
                        _require_int(
                            item["supporting_fact_index"],
                            f"{evidence_path}.supporting_fact_index",
                        )
                    )
                    if sentence_index < sentence_count:
                        raise ManifestValidationError("out-of-bounds evidence is actually in bounds")
                elif code == _MISSING_SUPPORT_TITLE_REASON:
                    _exact_keys(
                        item,
                        {
                            "available_document_ids",
                            "normalized_title_sha256",
                            "raw_title_sha256",
                            "supporting_fact_index",
                        },
                        evidence_path,
                    )
                    available_ids = _tuple_of_strings(
                        item["available_document_ids"],
                        f"{evidence_path}.available_document_ids",
                    )
                    if available_ids != tuple(sorted(set(available_ids))):
                        raise ManifestValidationError("available document IDs are not sorted unique")
                    _require_sha256(
                        item["normalized_title_sha256"],
                        f"{evidence_path}.normalized_title_sha256",
                    )
                    _require_sha256(
                        item["raw_title_sha256"],
                        f"{evidence_path}.raw_title_sha256",
                    )
                    supporting_fact_indices.append(
                        _require_int(
                            item["supporting_fact_index"],
                            f"{evidence_path}.supporting_fact_index",
                        )
                    )
                elif code == _EMPTY_CONTEXT_REASON:
                    _exact_keys(item, {"document_count"}, evidence_path)
                    if item["document_count"] != 0:
                        raise ManifestValidationError("empty context evidence is not empty")
                else:
                    _exact_keys(
                        item,
                        {
                            "document_id",
                            "occurrence_positions",
                            "raw_variant_sha256s",
                        },
                        evidence_path,
                    )
                    duplicate_evidence_ids.append(
                        _require_nonempty(
                            item["document_id"],
                            f"{evidence_path}.document_id",
                        )
                    )
                    positions = tuple(
                        _require_int(value, f"{evidence_path}.occurrence_positions[{index}]")
                        for index, value in enumerate(
                            _require_sequence(
                                item["occurrence_positions"],
                                f"{evidence_path}.occurrence_positions",
                            )
                        )
                    )
                    if positions != tuple(sorted(set(positions))) or len(positions) < 2:
                        raise ManifestValidationError("duplicate occurrence positions are invalid")
                    duplicate_variants = _tuple_of_strings(
                        item["raw_variant_sha256s"],
                        f"{evidence_path}.raw_variant_sha256s",
                    )
                    if len(duplicate_variants) != len(positions):
                        raise ManifestValidationError("duplicate variants do not align with positions")
                    for variant_index, variant in enumerate(duplicate_variants):
                        _require_sha256(
                            variant,
                            f"{evidence_path}.raw_variant_sha256s[{variant_index}]",
                        )
            if code in {
                _OUT_OF_BOUNDS_REASON,
                _MISSING_SUPPORT_TITLE_REASON,
            } and supporting_fact_indices != sorted(set(supporting_fact_indices)):
                raise ManifestValidationError(
                    "supporting fact indices are not strictly increasing"
                )
            if code == _EMPTY_CONTEXT_REASON and len(evidence) != 1:
                raise ManifestValidationError("empty context evidence must have one item")
            if code == _DUPLICATE_CONTEXT_DOCUMENT_REASON and duplicate_evidence_ids != sorted(
                set(duplicate_evidence_ids)
            ):
                raise ManifestValidationError(
                    "duplicate context document IDs are not sorted unique"
                )
        if reason_codes != sorted(set(reason_codes)):
            raise ManifestValidationError("curation rejection reasons are not sorted unique")
        if tuple(ambiguity_evidence_ids) != row_ambiguous:
            raise ManifestValidationError("ambiguous IDs differ from rejection evidence")
    if rejected_indices != sorted(set(rejected_indices)):
        raise ManifestValidationError("curation rejection source indices are not sorted unique")
    if len(set(rejected_qa_ids)) != len(rejected_qa_ids):
        raise ManifestValidationError("curation ledger rejected QA IDs contain duplicates")
    if set(accepted_qa_ids) & set(rejected_qa_ids):
        raise ManifestValidationError("accepted and rejected QA IDs overlap")
    if len(raw_rejections) != ledger["row_count"]:
        raise ManifestValidationError("curation ledger row count mismatch")
    if tuple(sorted(observed_ambiguous)) != ambiguous_ids:
        raise ManifestValidationError("curation ambiguous ID inventory mismatch")
    for document_id in ambiguous_ids:
        if observed_current_variants.get(document_id, set()) != set(
            declared_variants[document_id]
        ):
            raise ManifestValidationError(
                "observed raw variants do not cover the ambiguous inventory"
            )

    rejected_set = set(rejected_indices)
    accepted_indices = tuple(
        index for index in range(source["record_count"]) if index not in rejected_set
    )
    if len(accepted_qa_ids) != len(accepted_indices):
        raise ManifestValidationError("curation accepted QA count mismatch")
    if len(accepted_qa_ids) + len(rejected_qa_ids) != source["record_count"]:
        raise ManifestValidationError("curation QA inventory does not cover the source")
    expected = {
        "accepted_qa_order_sha256": ordered_values_sha256(accepted_qa_ids),
        "accepted_record_count": len(accepted_indices),
        "accepted_source_order_sha256": _source_indices_sha256(accepted_indices),
        "ambiguous_document_ids": list(ambiguous_ids),
        "input_record_count": source["record_count"],
        "policy": CURATION_POLICY,
        "rejected_record_count": len(rejected_indices),
        "rejection_ledger": dict(ledger),
        "rejection_reason_counts": reason_counts,
        "schema_version": CURATION_SCHEMA_VERSION,
        "source_sha256": source["sha256"],
    }
    if curation != expected:
        raise ManifestValidationError(
            "curation manifest differs from its canonical ledger"
        )
    return accepted_qa_ids


def _replay_source_curation(
    output_path: Path,
    *,
    source: Mapping[str, Any],
    dataset: str,
) -> tuple[str, ...]:
    source_path = Path(str(source["path"]))
    if not source_path.is_file():
        raise ManifestValidationError("strict curation replay requires the source file")
    if _sha256_file(source_path) != source["sha256"]:
        raise ManifestValidationError("local source hash changed before curation replay")
    records = load_local_records(
        source_path,
        split=source["split"],
        source_format=source["format"],
    )
    if len(records) != source["record_count"]:
        raise ManifestValidationError("source record count changed before curation replay")
    replayed = curate_source_records(records, dataset=dataset)
    expected_payload = _curation_ledger_payload(
        replayed,
        source_sha256=source["sha256"],
    )
    expected_bytes = canonical_json_bytes(expected_payload) + b"\n"
    ledger_path = output_path / REJECTION_LEDGER_NAME
    if ledger_path.read_bytes() != expected_bytes:
        raise ManifestValidationError(
            "curation ledger differs from strict source replay"
        )
    return tuple(example.qa.qa_id for example in replayed.examples)


def validate_artifact_bundle(
    output_dir: str | os.PathLike[str],
    *,
    replay_source_curation: bool = False,
) -> Mapping[str, Any]:
    """Read every published byte back and cross-check Parquet against sidecars."""

    if type(replay_source_curation) is not bool:
        raise TypeError("replay_source_curation must be a bool")
    output_path = Path(output_dir).expanduser().resolve(strict=True)
    manifest = _read_top_manifest(output_path / "manifest.json")
    is_formal = manifest["profile"] == "formal"
    source = _require_mapping(manifest["source"], "manifest.source")
    source_keys = {"path", "record_count", "revision", "sha256", "split"}
    if is_formal:
        source_keys.add("format")
    _exact_keys(
        source,
        source_keys,
        "manifest.source",
    )
    tokenizer = _require_mapping(manifest["tokenizer"], "manifest.tokenizer")
    _exact_keys(
        tokenizer,
        {
            "add_special_tokens",
            "local_files_only",
            "name",
            "return_offsets_mapping",
            "revision",
        },
        "manifest.tokenizer",
    )
    _require_int(source["record_count"], "manifest.source.record_count", 1)
    _require_sha256(source["sha256"], "manifest.source.sha256")
    _require_nonempty(source["path"], "manifest.source.path")
    if is_formal and source["format"] not in {"json", "jsonl", "parquet"}:
        raise ManifestValidationError("manifest source format is unsupported")
    if source["split"] is not None:
        _require_nonempty(source["split"], "manifest.source.split")
    if (
        tokenizer["add_special_tokens"] is not False
        or tokenizer["local_files_only"] is not True
        or tokenizer["return_offsets_mapping"] is not True
    ):
        raise ManifestValidationError("tokenizer execution contract changed")
    _require_fixed_revision(source["revision"], "manifest.source.revision")
    _require_fixed_revision(tokenizer["revision"], "manifest.tokenizer.revision")
    source_path = Path(str(source["path"]))
    if source_path.is_file() and _sha256_file(source_path) != source["sha256"]:
        raise ManifestValidationError("local source hash changed after build")
    dataset = _canonical_dataset_name(manifest["dataset"])
    if dataset != manifest["dataset"]:
        raise ManifestValidationError("manifest dataset name is not canonical")
    curated = (
        _validate_curation(
            output_path,
            manifest["curation"],
            source=source,
            dataset=dataset,
        )
        if is_formal
        else None
    )
    if replay_source_curation:
        if not is_formal:
            raise ManifestValidationError(
                "strict curation replay is only valid for formal bundles"
            )
        replayed_qa_ids = _replay_source_curation(
            output_path,
            source=source,
            dataset=dataset,
        )
        if replayed_qa_ids != curated:
            raise ManifestValidationError(
                "accepted QA order differs from strict source replay"
            )
    qa_ids = _tuple_of_strings(manifest["qa_ids"], "manifest.qa_ids")
    if len(set(qa_ids)) != len(qa_ids):
        raise ManifestValidationError("top-level QA IDs contain duplicates")
    if manifest["qa_order_sha256"] != ordered_values_sha256(qa_ids):
        raise ManifestValidationError("top-level QA order hash mismatch")
    artifacts = _require_mapping(manifest["artifacts"], "manifest.artifacts")
    expected_files = {"manifest.json"}
    if is_formal:
        expected_files.add(REJECTION_LEDGER_NAME)
    by_variant: dict[str, dict[str, tuple[Path, Mapping[str, Any]]]] = {}
    for name, raw_entry in artifacts.items():
        entry = _require_mapping(raw_entry, f"manifest.artifacts.{name}")
        _exact_keys(
            entry,
            {"kind", "path", "row_count", "sha256", "size_bytes", "variant"},
            f"manifest.artifacts.{name}",
        )
        if name != entry["path"] or Path(name).name != name:
            raise ManifestValidationError("artifact path must be a local basename")
        _require_sha256(entry["sha256"], f"manifest.artifacts.{name}.sha256")
        variant = _require_nonempty(
            entry["variant"],
            f"manifest.artifacts.{name}.variant",
        )
        kind = _require_nonempty(
            entry["kind"],
            f"manifest.artifacts.{name}.kind",
        )
        artifact_path = output_path / name
        if not artifact_path.is_file():
            raise ManifestValidationError(f"missing artifact {name}")
        if artifact_path.stat().st_size != entry["size_bytes"]:
            raise ManifestValidationError(f"artifact size mismatch for {name}")
        if _sha256_file(artifact_path) != entry["sha256"]:
            raise ManifestValidationError(f"artifact hash mismatch for {name}")
        _require_int(entry["row_count"], f"manifest.artifacts.{name}.row_count", 1)
        expected_files.add(name)
        variant_entries = by_variant.setdefault(variant, {})
        if kind in variant_entries:
            raise ManifestValidationError(
                f"variant {variant} repeats artifact kind {kind}"
            )
        variant_entries[kind] = (artifact_path, entry)
    actual_files = {path.name for path in output_path.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise ManifestValidationError("bundle contains unlisted or missing files")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to validate reproduction Parquet") from exc
    mode = manifest["mode"]
    _require_int(manifest["seed"], "manifest.seed")
    expected_metadata = ManifestMetadata(
        source_name=dataset,
        source_revision=source["revision"],
        source_sha256=source["sha256"],
        tokenizer_name=tokenizer["name"],
        tokenizer_revision=tokenizer["revision"],
        seed=manifest["seed"],
    )
    reconstructed_variants: dict[str, tuple[ManifestRecord, ...]] = {}
    for variant, entries in by_variant.items():
        if set(entries) != {"canonical-sidecar-jsonl", "memory-dataset-parquet"}:
            raise ManifestValidationError(f"variant {variant} has incomplete artifacts")
        sidecar_path, sidecar_entry = entries["canonical-sidecar-jsonl"]
        parquet_path, parquet_entry = entries["memory-dataset-parquet"]
        values = validate_canonical_jsonl(
            sidecar_path.read_bytes(),
            sidecar_entry["sha256"],
        )
        records = tuple(manifest_record_from_dict(value) for value in values)
        rows = pq.read_table(parquet_path).to_pylist()
        if len(records) != sidecar_entry["row_count"] or len(rows) != parquet_entry["row_count"]:
            raise ManifestValidationError(f"variant {variant} row count mismatch")
        if tuple(record.qa.qa_id for record in records) != qa_ids:
            raise ManifestValidationError(f"variant {variant} QA order mismatch")
        if any(record.metadata != expected_metadata for record in records):
            raise ManifestValidationError(
                f"variant {variant} metadata differs from the top-level manifest"
            )
        for row, record in zip(rows, records):
            _validate_row(
                row,
                record,
                dataset=dataset,
                variant=variant,
                sidecar_name=sidecar_path.name,
            )
        reconstructed_variants[variant] = records
    contract_value = _require_mapping(manifest["contract"], "manifest.contract")
    if mode == "train":
        try:
            contract = TrainManifestContract(**contract_value)
        except TypeError as exc:
            raise ManifestValidationError("invalid train contract schema") from exc
        if set(reconstructed_variants) != {"train"}:
            raise ManifestValidationError("train bundle has unexpected variants")
        validate_train_records(reconstructed_variants["train"], contract=contract)
    elif mode == "eval":
        try:
            contract = EvalManifestContract(**contract_value)
        except TypeError as exc:
            raise ManifestValidationError("invalid eval contract schema") from exc
        prefix_key = str(contract.prefix_document_count)
        pool_key = str(contract.pool_document_count)
        if set(reconstructed_variants) != {prefix_key, pool_key}:
            raise ManifestValidationError("eval bundle has unexpected variants")
        from .reproduction_manifest import EvalManifestPair

        pair = EvalManifestPair(
            prefix_records=reconstructed_variants[prefix_key],
            pool_records=reconstructed_variants[pool_key],
        )
        validate_eval_manifest_pair(
            pair,
            contract=contract,
            expected_qa_ids=qa_ids,
        )
    else:
        raise ManifestValidationError("manifest mode must be train or eval")
    if curated is not None:
        selected = _select_qa_ids(
            curated,
            qa_count=contract.qa_count,
            seed=manifest["seed"],
        )
        if selected != qa_ids:
            raise ManifestValidationError(
                "QA order differs from deterministic source selection"
            )
    _check_profile(mode, manifest["profile"], contract)
    return manifest


def build_artifact_bundle(
    *,
    input_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    mode: str,
    dataset: str,
    source_revision: str,
    tokenizer_name: str,
    tokenizer_revision: str,
    seed: int,
    encode: Callable[[str], Any],
    profile: str = "formal",
    train_contract: TrainManifestContract = TrainManifestContract(),
    eval_contract: EvalManifestContract = EvalManifestContract(),
    split: str | None = None,
    source_format: str | None = None,
) -> Mapping[str, Any]:
    """Build, atomically publish, then independently read back one bundle."""

    if mode not in {"train", "eval"}:
        raise _error("mode", "must be train or eval")
    dataset = _canonical_dataset_name(dataset)
    source_revision = _require_fixed_revision(source_revision, "source_revision")
    tokenizer_name = _require_nonempty(tokenizer_name, "tokenizer_name")
    tokenizer_revision = _require_fixed_revision(
        tokenizer_revision,
        "tokenizer_revision",
    )
    seed = _require_int(seed, "seed")
    contract: TrainManifestContract | EvalManifestContract
    contract = train_contract if mode == "train" else eval_contract
    contract.validate()
    _check_profile(mode, profile, contract)
    if profile == "formal" and mode == "train" and dataset != "hotpotqa":
        raise ManifestValidationError("formal training is fixed to HotpotQA")
    source_path = Path(input_path).expanduser().resolve(strict=True)
    destination = Path(output_dir).expanduser().resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    try:
        _build_in_directory(
            staging,
            mode=mode,
            profile=profile,
            dataset=dataset,
            source_path=source_path,
            source_revision=source_revision,
            tokenizer_name=tokenizer_name,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            encode=encode,
            train_contract=train_contract,
            eval_contract=eval_contract,
            split=split,
            source_format=source_format,
        )
        validate_artifact_bundle(staging)
        os.replace(staging, destination)
        return validate_artifact_bundle(destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _load_local_tokenizer(name: str, revision: str) -> Callable[[str], Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required to load the pinned tokenizer") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        revision=revision,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("formal provenance requires a fast tokenizer with offsets")

    def encode(text: str) -> Any:
        return tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

    return encode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build offline, hashed ReMemR1 train/eval data bundles",
    )
    parser.add_argument("mode", choices=("train", "eval"))
    parser.add_argument("--input", required=True, help="local JSON/JSONL/Parquet source")
    parser.add_argument("--output", required=True, help="new output directory")
    parser.add_argument("--dataset", required=True, choices=("hotpotqa", "2wikimultihopqa"))
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default=None, help="JSON object split key")
    parser.add_argument("--profile", choices=("formal", "fixture"), default="formal")
    parser.add_argument("--qa-count", type=int, default=None)
    parser.add_argument("--prefix-documents", type=int, default=None)
    parser.add_argument("--pool-documents", type=int, default=None)
    parser.add_argument("--train-documents", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--min-context-tokens", type=int, default=None)
    parser.add_argument("--max-context-tokens", type=int, default=None)
    return parser


def _contracts_from_args(args: argparse.Namespace) -> tuple[TrainManifestContract, EvalManifestContract]:
    train = TrainManifestContract(
        qa_count=args.qa_count or FORMAL_TRAIN_QA_COUNT,
        document_count=args.train_documents or FORMAL_TRAIN_DOCUMENT_COUNT,
        chunk_size=args.chunk_size or FORMAL_TRAIN_CHUNK_SIZE,
        max_chunks=args.max_chunks or FORMAL_TRAIN_MAX_CHUNKS,
        min_context_tokens=args.min_context_tokens or FORMAL_TRAIN_MIN_CONTEXT_TOKENS,
        max_context_tokens=args.max_context_tokens or FORMAL_TRAIN_MAX_CONTEXT_TOKENS,
    )
    eval_contract = EvalManifestContract(
        qa_count=args.qa_count or EvalManifestContract().qa_count,
        prefix_document_count=(
            args.prefix_documents or EvalManifestContract().prefix_document_count
        ),
        pool_document_count=(
            args.pool_documents or EvalManifestContract().pool_document_count
        ),
        chunk_size=args.chunk_size or EvalManifestContract().chunk_size,
    )
    return train, eval_contract


def main(
    argv: Sequence[str] | None = None,
    *,
    tokenizer_loader: Callable[[str, str], Callable[[str], Any]] = _load_local_tokenizer,
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    override_names = (
        "qa_count",
        "prefix_documents",
        "pool_documents",
        "train_documents",
        "chunk_size",
        "max_chunks",
        "min_context_tokens",
        "max_context_tokens",
    )
    if args.profile == "formal" and any(
        getattr(args, name) is not None for name in override_names
    ):
        parser.error("formal profile does not accept contract overrides")
    if args.mode == "train" and any(
        value is not None for value in (args.prefix_documents, args.pool_documents)
    ):
        parser.error("train mode does not accept eval document overrides")
    if args.mode == "eval" and any(
        value is not None
        for value in (
            args.train_documents,
            args.max_chunks,
            args.min_context_tokens,
            args.max_context_tokens,
        )
    ):
        parser.error("eval mode does not accept train contract overrides")
    train_contract, eval_contract = _contracts_from_args(args)
    encode = tokenizer_loader(args.tokenizer, args.tokenizer_revision)
    manifest = build_artifact_bundle(
        input_path=args.input,
        output_dir=args.output,
        mode=args.mode,
        dataset=args.dataset,
        source_revision=args.source_revision,
        tokenizer_name=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
        seed=args.seed,
        encode=encode,
        profile=args.profile,
        train_contract=train_contract,
        eval_contract=eval_contract,
        split=args.split,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "BUNDLE_KIND",
    "CURATION_SCHEMA_VERSION",
    "CURATION_POLICY",
    "REJECTION_LEDGER_NAME",
    "FORMAL_TRAIN_QA_COUNT",
    "FORMAL_TRAIN_DOCUMENT_COUNT",
    "FORMAL_TRAIN_CHUNK_SIZE",
    "FORMAL_TRAIN_MAX_CHUNKS",
    "FORMAL_TRAIN_MIN_CONTEXT_TOKENS",
    "FORMAL_TRAIN_MAX_CONTEXT_TOKENS",
    "SourceDocument",
    "ParsedExample",
    "SourceRejection",
    "SourceCurationResult",
    "TrainManifestContract",
    "stable_qa_id",
    "parse_source_record",
    "load_local_records",
    "curate_source_records",
    "build_train_records",
    "validate_train_records",
    "build_eval_pair",
    "manifest_record_from_dict",
    "build_artifact_bundle",
    "validate_artifact_bundle",
    "main",
]
