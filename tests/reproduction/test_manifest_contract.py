import hashlib
import json
import re
from dataclasses import replace

import pytest

from taskutils.data_synthesis.reproduction_manifest import (
    SCHEMA_VERSION,
    EVAL_CHUNK_SIZE,
    EVAL_POOL_DOCUMENT_COUNT,
    EVAL_PREFIX_DOCUMENT_COUNT,
    EVAL_QA_COUNT,
    DocumentInput,
    EvalExampleInput,
    EvalManifestContract,
    EvalManifestPair,
    ManifestMetadata,
    ManifestValidationError,
    QARecord,
    SupportingFactInput,
    build_eval_manifest_pair,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    canonical_jsonl_sha256,
    normalize_text,
    normalized_text_sha256,
    ordered_values_sha256,
    seal_manifest_record,
    stable_document_id,
    stable_supporting_fact_id,
    validate_canonical_jsonl,
    validate_eval_manifest_pair,
    validate_manifest_record,
    write_canonical_jsonl,
)


def _tokenize_with_offsets(text):
    spans = [match.span() for match in re.finditer(r"\S+", text)]
    return {
        "input_ids": list(range(len(spans))),
        "offset_mapping": spans,
    }


@pytest.fixture
def tiny_contract():
    return EvalManifestContract(
        qa_count=2,
        prefix_document_count=2,
        pool_document_count=4,
        chunk_size=3,
    )


@pytest.fixture
def metadata():
    return ManifestMetadata(
        source_name="toy-hotpot",
        source_revision="source-rev-1",
        source_sha256="a" * 64,
        tokenizer_name="toy-tokenizer",
        tokenizer_revision="tokenizer-rev-1",
        seed=42,
    )


def _examples(*, document_marker="base"):
    examples = []
    for qa_index in range(2):
        documents = tuple(
            DocumentInput(
                title=f"Title {document_marker}-{qa_index}-{doc_index}",
                text=(
                    f"body {qa_index}-{doc_index} has decisive evidence"
                    if doc_index == 0
                    else f"padding body {qa_index}-{doc_index}"
                ),
                source_document_id=f"source-{document_marker}-{qa_index}-{doc_index}",
            )
            for doc_index in range(4)
        )
        examples.append(
            EvalExampleInput(
                qa=QARecord.create(
                    qa_id=f"qa-{qa_index}",
                    question=f"Question {qa_index}?",
                    gold_answers=(f"answer-{qa_index}", f"alias-{qa_index}"),
                ),
                document_pool=documents,
                supporting_facts=(
                    SupportingFactInput(
                        document_index=0,
                        sentence_index=0,
                        text=f"body {qa_index}-0 has decisive evidence",
                    ),
                ),
            )
        )
    return tuple(examples)


@pytest.fixture
def pair(metadata, tiny_contract):
    return build_eval_manifest_pair(
        _examples(),
        metadata=metadata,
        encode=_tokenize_with_offsets,
        contract=tiny_contract,
    )


def _reseal(record, **changes):
    return seal_manifest_record(replace(record, **changes))


def _replace_prefix(pair, index, record):
    records = list(pair.prefix_records)
    records[index] = record
    return EvalManifestPair(tuple(records), pair.pool_records)


def test_formal_eval_contract_defaults_are_locked():
    assert EVAL_QA_COUNT == 64
    assert EVAL_PREFIX_DOCUMENT_COUNT == 200
    assert EVAL_POOL_DOCUMENT_COUNT == 800
    assert EVAL_CHUNK_SIZE == 5000
    assert EvalManifestContract() == EvalManifestContract(
        qa_count=64,
        prefix_document_count=200,
        pool_document_count=800,
        chunk_size=5000,
    )


def test_normalized_text_hash_gives_stable_content_document_ids():
    title_a = "  Fullwidth Ａ title\r\n"
    text_a = "First\tline\r\nsecond   line"
    title_b = "Fullwidth A title"
    text_b = "First line second line"

    assert normalize_text(title_a) == title_b
    assert normalize_text(text_a) == text_b
    assert normalized_text_sha256(text_a) == normalized_text_sha256(text_b)
    assert stable_document_id(title_a, text_a) == stable_document_id(title_b, text_b)
    assert stable_document_id(title_b, text_b).startswith("doc_")


def test_schema_commits_qa_golds_documents_and_revisions(pair):
    record = pair.pool_records[0]

    assert record.metadata.to_dict() == {
        "schema_version": SCHEMA_VERSION,
        "seed": 42,
        "source_name": "toy-hotpot",
        "source_revision": "source-rev-1",
        "source_sha256": "a" * 64,
        "tokenizer_name": "toy-tokenizer",
        "tokenizer_revision": "tokenizer-rev-1",
    }
    assert [answer.text for answer in record.qa.gold_answers] == [
        "answer-0",
        "alias-0",
    ]
    assert record.qa.question_sha256 == normalized_text_sha256("Question 0?")
    assert all(document.title_sha256 for document in record.documents)
    assert all(document.text_sha256 for document in record.documents)
    assert all(document.source_document_id for document in record.documents)


def test_pool_is_built_once_and_short_variant_is_its_strict_prefix(pair, tiny_contract):
    validate_eval_manifest_pair(pair, contract=tiny_contract)

    for prefix, pool in zip(pair.prefix_records, pair.pool_records):
        assert prefix.document_pool_sha256 == pool.document_pool_sha256
        assert pool.document_pool_sha256 == ordered_values_sha256(
            [document.document_id for document in pool.documents]
        )
        assert [document.identity_dict() for document in prefix.documents] == [
            document.identity_dict()
            for document in pool.documents[: tiny_contract.prefix_document_count]
        ]
        assert prefix.qa == pool.qa
        assert prefix.qa_order_sha256 == pool.qa_order_sha256


def test_supporting_facts_survive_both_lengths_and_are_reverse_lookupable(pair):
    for record in (*pair.prefix_records, *pair.pool_records):
        fact = record.supporting_facts[0]
        document = record.document_by_id(fact.document_id)
        provenance = record.provenance_for_token(fact.token_start)

        assert fact.fact_id in document.supporting_fact_ids
        assert document in provenance.documents
        assert fact in provenance.supporting_facts
        assert record.chunks_for_document(document.document_id)
        assert record.chunks_for_supporting_fact(fact.fact_id)
        assert all(
            fact.fact_id in chunk.supporting_fact_ids
            for chunk in record.chunks_for_supporting_fact(fact.fact_id)
        )


def test_chunks_cover_every_token_without_eval_truncation(pair):
    for record in (*pair.prefix_records, *pair.pool_records):
        assert record.truncation == "none"
        assert record.consumed_token_count == record.context_token_count
        assert record.chunks[0].token_start == 0
        assert record.chunks[-1].token_end == record.context_token_count
        assert [chunk.token_start for chunk in record.chunks] == list(
            range(0, record.context_token_count, record.chunk_size)
        )
        assert record.documents[0].token_start == 0
        assert record.documents[-1].token_end == record.context_token_count
        validate_manifest_record(record)


def test_canonical_jsonl_and_hash_are_byte_stable(pair, tmp_path):
    payload = canonical_jsonl_bytes(pair.prefix_records)
    digest = canonical_jsonl_sha256(pair.prefix_records)
    path = tmp_path / "eval-200.jsonl"

    assert digest == hashlib.sha256(payload).hexdigest()
    assert payload.endswith(b"\n")
    assert b"\n  " not in payload
    assert validate_canonical_jsonl(payload, digest)[0]["qa"]["qa_id"] == "qa-0"
    assert write_canonical_jsonl(path, pair.prefix_records) == digest
    assert path.read_bytes() == payload
    assert canonical_json_bytes({"z": 1, "a": 2}) == b'{"a":2,"z":1}'
    assert json.loads(payload.splitlines()[0])["record_sha256"]


def test_canonical_jsonl_rejects_wrong_hash_and_noncanonical_bytes(pair):
    payload = canonical_jsonl_bytes(pair.prefix_records)

    with pytest.raises(ManifestValidationError, match="content hash mismatch"):
        validate_canonical_jsonl(payload, "0" * 64)

    pretty = json.dumps(json.loads(payload.splitlines()[0])).encode() + b"\n"
    with pytest.raises(ManifestValidationError, match="not canonical"):
        validate_canonical_jsonl(pretty, hashlib.sha256(pretty).hexdigest())


def test_validation_hard_fails_unpaired_qa(pair, tiny_contract):
    original = pair.prefix_records[0]
    changed_qa = QARecord.create(
        "different-qa",
        original.qa.question,
        [answer.text for answer in original.qa.gold_answers],
    )
    broken = _replace_prefix(pair, 0, _reseal(original, qa=changed_qa))

    with pytest.raises(ManifestValidationError, match="not exactly paired"):
        validate_eval_manifest_pair(broken, contract=tiny_contract)


def test_validation_hard_fails_gold_pairing_drift(pair, tiny_contract):
    original = pair.prefix_records[0]
    changed_qa = QARecord.create(
        original.qa.qa_id,
        original.qa.question,
        ["silently changed gold"],
    )
    broken = _replace_prefix(pair, 0, _reseal(original, qa=changed_qa))

    with pytest.raises(ManifestValidationError, match="QA/gold pairing mismatch"):
        validate_eval_manifest_pair(broken, contract=tiny_contract)


def test_validation_hard_fails_qa_reordering(pair, tiny_contract):
    broken = EvalManifestPair(
        prefix_records=tuple(reversed(pair.prefix_records)),
        pool_records=pair.pool_records,
    )

    with pytest.raises(ManifestValidationError, match="fixed order"):
        validate_eval_manifest_pair(broken, contract=tiny_contract)


def test_validation_hard_fails_non_prefix_documents(
    pair,
    metadata,
    tiny_contract,
):
    other_pair = build_eval_manifest_pair(
        _examples(document_marker="other"),
        metadata=metadata,
        encode=_tokenize_with_offsets,
        contract=tiny_contract,
    )
    substitute = _reseal(
        other_pair.prefix_records[0],
        document_pool_sha256=pair.pool_records[0].document_pool_sha256,
    )
    broken = _replace_prefix(pair, 0, substitute)

    with pytest.raises(ManifestValidationError, match="strict 800 prefix"):
        validate_eval_manifest_pair(broken, contract=tiny_contract)


def test_validation_hard_fails_if_supporting_evidence_is_dropped(pair, tiny_contract):
    original = pair.prefix_records[0]
    documents = tuple(
        replace(document, supporting_fact_ids=())
        for document in original.documents
    )
    chunks = tuple(
        replace(chunk, supporting_fact_ids=()) for chunk in original.chunks
    )
    changed = _reseal(
        original,
        documents=documents,
        supporting_facts=(),
        chunks=chunks,
    )
    validate_manifest_record(changed)
    broken = _replace_prefix(pair, 0, changed)

    with pytest.raises(ManifestValidationError, match="evidence was lost"):
        validate_eval_manifest_pair(broken, contract=tiny_contract)


def test_validation_hard_fails_invented_supporting_fact_text(pair):
    original = pair.prefix_records[0]
    fact = original.supporting_facts[0]
    invented_text = "an invented supporting claim"
    invented_id = stable_supporting_fact_id(
        fact.document_id,
        fact.sentence_index,
        invented_text,
    )
    invented_fact = replace(
        fact,
        fact_id=invented_id,
        text=invented_text,
        text_sha256=normalized_text_sha256(invented_text),
    )
    documents = tuple(
        replace(
            document,
            supporting_fact_ids=tuple(
                invented_id if value == fact.fact_id else value
                for value in document.supporting_fact_ids
            ),
        )
        for document in original.documents
    )
    chunks = tuple(
        replace(
            chunk,
            supporting_fact_ids=tuple(
                invented_id if value == fact.fact_id else value
                for value in chunk.supporting_fact_ids
            ),
        )
        for chunk in original.chunks
    )
    changed = _reseal(
        original,
        documents=documents,
        supporting_facts=(invented_fact,),
        chunks=chunks,
    )

    with pytest.raises(ManifestValidationError, match="text is absent"):
        validate_manifest_record(changed)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"context_sha256": "0" * 64}, "context_sha256 mismatch"),
        ({"consumed_token_count": 1}, "truncates the context"),
        ({"truncation": "center"}, "forbid truncation"),
    ],
)
def test_validation_hard_fails_hash_or_truncation_drift(
    pair,
    changes,
    message,
):
    changed = _reseal(pair.prefix_records[0], **changes)

    with pytest.raises(ManifestValidationError, match=message):
        validate_manifest_record(changed)


def test_validation_hard_fails_record_hash_tampering(pair):
    changed = replace(pair.prefix_records[0], record_sha256="f" * 64)

    with pytest.raises(ManifestValidationError, match="record_sha256 mismatch"):
        validate_manifest_record(changed)


def test_validation_hard_fails_chunk_boundary_or_provenance_drift(pair):
    original = pair.prefix_records[0]
    wrong_provenance = replace(original.chunks[0], document_ids=())
    changed = _reseal(
        original,
        chunks=(wrong_provenance, *original.chunks[1:]),
    )
    with pytest.raises(ManifestValidationError, match="chunk-to-document"):
        validate_manifest_record(changed)

    missing_tail = _reseal(original, chunks=original.chunks[:-1])
    with pytest.raises(ManifestValidationError, match="chunk count truncates"):
        validate_manifest_record(missing_tail)


def test_validation_hard_fails_pool_hash_or_revision_drift(pair, tiny_contract):
    original = pair.prefix_records[0]
    wrong_pool_hash = _replace_prefix(
        pair,
        0,
        _reseal(original, document_pool_sha256="b" * 64),
    )
    with pytest.raises(ManifestValidationError, match="same pool"):
        validate_eval_manifest_pair(wrong_pool_hash, contract=tiny_contract)

    changed_metadata = replace(original.metadata, tokenizer_revision="other-revision")
    wrong_revision = _replace_prefix(
        pair,
        0,
        _reseal(original, metadata=changed_metadata),
    )
    with pytest.raises(ManifestValidationError, match="revision or seed"):
        validate_eval_manifest_pair(wrong_revision, contract=tiny_contract)


def test_pair_hashes_can_be_locked_externally(pair, tiny_contract):
    validate_eval_manifest_pair(
        pair,
        contract=tiny_contract,
        expected_qa_ids=("qa-0", "qa-1"),
        expected_prefix_sha256=pair.prefix_sha256,
        expected_pool_sha256=pair.pool_sha256,
    )

    with pytest.raises(ManifestValidationError, match="prefix JSONL hash mismatch"):
        validate_eval_manifest_pair(
            pair,
            contract=tiny_contract,
            expected_prefix_sha256="0" * 64,
        )


def test_builder_rejects_support_outside_required_prefix(metadata, tiny_contract):
    examples = list(_examples())
    first = examples[0]
    examples[0] = replace(
        first,
        supporting_facts=(
            SupportingFactInput(
                document_index=3,
                sentence_index=0,
                text="padding body 0-3",
            ),
        ),
    )

    with pytest.raises(ManifestValidationError, match="outside.*prefix"):
        build_eval_manifest_pair(
            examples,
            metadata=metadata,
            encode=_tokenize_with_offsets,
            contract=tiny_contract,
        )
