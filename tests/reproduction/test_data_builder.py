import copy
import hashlib
import json
import re
from pathlib import Path

import pytest

from taskutils.data_synthesis import reproduction_builder as builder
from taskutils.data_synthesis.reproduction_builder import (
    BUNDLE_KIND,
    BUNDLE_SCHEMA_VERSION,
    CURATION_POLICY,
    FORMAL_TRAIN_CHUNK_SIZE,
    FORMAL_TRAIN_DOCUMENT_COUNT,
    FORMAL_TRAIN_MAX_CHUNKS,
    FORMAL_TRAIN_MAX_CONTEXT_TOKENS,
    FORMAL_TRAIN_MIN_CONTEXT_TOKENS,
    FORMAL_TRAIN_QA_COUNT,
    FORMAL_BUNDLE_KIND,
    FORMAL_BUNDLE_SCHEMA_VERSION,
    REJECTION_LEDGER_NAME,
    TrainManifestContract,
    build_artifact_bundle,
    curate_source_records,
    load_local_records,
    main,
    manifest_record_from_dict,
    parse_source_record,
    validate_artifact_bundle,
)
from taskutils.data_synthesis.reproduction_manifest import (
    EvalManifestContract,
    ManifestValidationError,
    canonical_json_bytes,
    ordered_values_sha256,
    stable_document_id,
    stable_supporting_fact_id,
)


def _tokenize_with_offsets(text):
    spans = [match.span() for match in re.finditer(r"\S+", text)]
    return {
        "input_ids": list(range(len(spans))),
        "offset_mapping": spans,
    }


def _hotpot_record(index):
    evidence_title = f"Evidence {index}"
    distractor_title = f"Local distractor {index}"
    return {
        "_id": f"hotpot-{index}",
        "question": f"Who is entity {index}?",
        "answers": [f"answer {index}", f"alias {index}"],
        "context": [
            [
                evidence_title,
                [
                    f"Entity {index} has answer {index}. ",
                    f"A second supporting sentence for {index}.",
                ],
            ],
            [distractor_title, [f"Unrelated local text {index}."]],
        ],
        "supporting_facts": [[evidence_title, 0], [evidence_title, 1]],
        "level": "hard",
        "type": "bridge",
    }


def _flashrag_2wiki_record(index):
    evidence_title = f"Wiki evidence {index}"
    distractor_title = f"Wiki distractor {index}"
    return {
        "id": f"2wiki-{index}",
        "question": f"What connects entities {index}?",
        "golden_answers": [f"connection {index}", f"link {index}"],
        "metadata": {
            "context": {
                "title": [evidence_title, distractor_title],
                "sentences": [
                    [
                        f"Entity {index} connects through connection {index}. ",
                        f"Second wiki support {index}.",
                    ],
                    [f"Unrelated wiki text {index}."],
                ],
                "document_id": [f"wiki-evidence-{index}", f"wiki-noise-{index}"],
            },
            "supporting_facts": {
                "title": [evidence_title, evidence_title],
                "sent_id": [0, 1],
            },
            "level": "hard",
            "type": "compositional",
        },
    }


def _flashrag_hotpot_record(index):
    evidence_title = f"FlashRAG evidence {index}"
    distractor_title = f"FlashRAG distractor {index}"
    return {
        "id": f"flashrag-hotpot-{index}",
        "question": f"Who is entity {index}?",
        "golden_answers": [f"answer {index}", f"alias {index}"],
        "metadata": {
            "context": {
                "title": [evidence_title, distractor_title],
                "sentences": [
                    [
                        f"Entity {index} has answer {index}. ",
                        f"A second supporting sentence for {index}.",
                    ],
                    [f"Unrelated local text {index}."],
                ],
            },
            "supporting_facts": {
                "title": [evidence_title, evidence_title],
                "sent_id": [0, 1],
            },
        },
    }


def _write_json(path, records):
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return path


def test_explicit_source_format_reads_an_extensionless_hf_cache_blob(tmp_path):
    blob = tmp_path / "a81274abafa899ec"
    blob.write_text(json.dumps(_flashrag_hotpot_record(0)) + "\n", encoding="utf-8")

    records = load_local_records(blob, source_format="jsonl")

    assert len(records) == 1
    assert records[0]["id"] == "flashrag-hotpot-0"


def _ambiguous_records_with_oob_overlap():
    first = _flashrag_hotpot_record(100)
    second = copy.deepcopy(first)
    second["id"] = "flashrag-hotpot-100-variant"
    second["question"] = "Which alternate question uses entity 100?"
    second["metadata"]["context"]["title"][0] = " FlashRAG   evidence 100 "
    second["metadata"]["context"]["sentences"][0][0] = (
        "Entity 100 has  answer 100.\n"
    )
    second["metadata"]["supporting_facts"]["title"] = [
        " FlashRAG   evidence 100 ",
        " FlashRAG   evidence 100 ",
    ]
    first["metadata"]["supporting_facts"]["sent_id"][1] = 99
    return first, second


def test_curation_scans_oob_records_for_raw_variant_collisions_and_keeps_all_reasons():
    first, second = _ambiguous_records_with_oob_overlap()
    records = [first, second, *(_flashrag_hotpot_record(index) for index in range(3))]

    result = curate_source_records(records, dataset="hotpotqa")

    assert result.accepted_source_indices == (2, 3, 4)
    assert len(result.ambiguous_document_ids) == 1
    assert [value.source_index for value in result.rejections] == [0, 1]
    assert [reason.code for reason in result.rejections[0].reasons] == [
        "ambiguous_normalized_document_id",
        "supporting_fact_sentence_index_out_of_bounds",
    ]
    assert [reason.code for reason in result.rejections[1].reasons] == [
        "ambiguous_normalized_document_id"
    ]
    assert result.rejections[0].ambiguous_document_ids == result.ambiguous_document_ids
    oob = result.rejections[0].reasons[1].evidence[0]
    assert oob["sentence_index"] == 99
    assert oob["sentence_count"] == 2


def test_curation_does_not_downgrade_unclassified_parse_errors():
    malformed = _flashrag_hotpot_record(0)
    del malformed["metadata"]["supporting_facts"]["sent_id"]

    with pytest.raises(ManifestValidationError, match="sent_id"):
        curate_source_records([malformed], dataset="hotpotqa")


def test_curation_classifies_known_flashrag_structural_rejections_without_repair():
    missing_title = _flashrag_hotpot_record(10)
    missing_title["metadata"]["supporting_facts"]["title"][0] = (
        "title absent from context"
    )
    empty_context = _flashrag_hotpot_record(11)
    empty_context["metadata"]["context"] = {"title": [], "sentences": []}
    duplicate_document = _flashrag_hotpot_record(12)
    duplicate_document["metadata"]["context"]["title"].append(
        duplicate_document["metadata"]["context"]["title"][0]
    )
    duplicate_document["metadata"]["context"]["sentences"].append(
        copy.deepcopy(duplicate_document["metadata"]["context"]["sentences"][0])
    )

    result = curate_source_records(
        [
            missing_title,
            empty_context,
            duplicate_document,
            _flashrag_hotpot_record(13),
        ],
        dataset="hotpotqa",
    )

    assert result.accepted_source_indices == (3,)
    assert [[reason.code for reason in row.reasons] for row in result.rejections] == [
        ["supporting_title_absent_from_context"],
        ["empty_context"],
        ["duplicate_context_document_id"],
    ]
    missing_evidence = result.rejections[0].reasons[0].evidence[0]
    assert set(missing_evidence) == {
        "available_document_ids",
        "normalized_title_sha256",
        "raw_title_sha256",
        "supporting_fact_index",
    }
    assert "title absent" not in json.dumps(missing_evidence)
    assert result.rejections[1].reasons[0].evidence == ({"document_count": 0},)
    duplicate_evidence = result.rejections[2].reasons[0].evidence[0]
    assert duplicate_evidence["occurrence_positions"] == [0, 2]
    assert len(duplicate_evidence["raw_variant_sha256s"]) == 2


def test_single_duplicate_context_row_can_cover_all_ambiguous_raw_variants(tmp_path):
    record = _flashrag_hotpot_record(20)
    record["metadata"]["context"]["title"].append(" FlashRAG   evidence 20 ")
    record["metadata"]["context"]["sentences"].append(
        [
            "Entity 20 has  answer 20.\n",
            "A second supporting sentence for 20.",
        ]
    )

    result = curate_source_records([record], dataset="hotpotqa")

    assert not result.examples
    assert len(result.ambiguous_document_ids) == 1
    assert [reason.code for reason in result.rejections[0].reasons] == [
        "ambiguous_normalized_document_id",
        "duplicate_context_document_id",
    ]
    ambiguity = result.rejections[0].reasons[0].evidence[0]
    assert ambiguity["observed_raw_variant_sha256s"] == ambiguity[
        "raw_variant_sha256s"
    ]
    source_sha256 = "9" * 64
    curation = builder._write_curation_ledger(
        tmp_path,
        result,
        source_sha256=source_sha256,
    )
    assert builder._validate_curation(
        tmp_path,
        curation,
        source=_portable_source_contract(tmp_path, 1, source_sha256),
        dataset="hotpotqa",
    ) == ()


def test_curation_ledger_binds_accepted_qa_order_and_validates_without_source(tmp_path):
    rejected = _flashrag_hotpot_record(9)
    rejected["metadata"]["supporting_facts"]["sent_id"][1] = 99
    records = [
        _flashrag_hotpot_record(0),
        rejected,
        _flashrag_hotpot_record(1),
        _flashrag_hotpot_record(2),
    ]
    result = curate_source_records(records, dataset="hotpotqa")
    source_sha256 = "a" * 64

    curation = builder._write_curation_ledger(
        tmp_path,
        result,
        source_sha256=source_sha256,
    )
    accepted_ids = tuple(example.qa.qa_id for example in result.examples)

    assert curation["accepted_qa_order_sha256"] == ordered_values_sha256(
        accepted_ids
    )
    assert curation["accepted_source_order_sha256"] == hashlib.sha256(
        canonical_json_bytes([0, 2, 3])
    ).hexdigest()
    assert curation["rejection_reason_counts"] == {
        "ambiguous_normalized_document_id": 0,
        "duplicate_context_document_id": 0,
        "empty_context": 0,
        "supporting_fact_sentence_index_out_of_bounds": 1,
        "supporting_title_absent_from_context": 0,
    }
    ledger = json.loads((tmp_path / REJECTION_LEDGER_NAME).read_text(encoding="utf-8"))
    assert ledger["accepted_qa_ids"] == list(accepted_ids)
    assert builder._validate_curation(
        tmp_path,
        curation,
        source={
            "format": "jsonl",
            "path": str(tmp_path / "source-is-deliberately-unavailable"),
            "record_count": len(records),
            "revision": "source-rev",
            "sha256": source_sha256,
            "split": None,
        },
        dataset="hotpotqa",
    ) == accepted_ids


def _rewrite_curation_ledger(tmp_path, curation, mutate):
    path = tmp_path / REJECTION_LEDGER_NAME
    ledger = json.loads(path.read_text(encoding="utf-8"))
    mutate(ledger)
    path.write_bytes(canonical_json_bytes(ledger) + b"\n")
    updated = copy.deepcopy(curation)
    updated["rejection_ledger"]["sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    updated["rejection_ledger"]["size_bytes"] = path.stat().st_size
    return updated


def _portable_source_contract(tmp_path, record_count, source_sha256):
    return {
        "format": "jsonl",
        "path": str(tmp_path / "unavailable-source"),
        "record_count": record_count,
        "revision": "source-rev",
        "sha256": source_sha256,
        "split": None,
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda ledger: ledger["accepted_qa_ids"].__setitem__(
                0, "otherdataset:qa-0"
            ),
            "wrong dataset namespace",
        ),
        (
            lambda ledger: ledger["rejections"][0].__setitem__(
                "qa_id", ledger["accepted_qa_ids"][0]
            ),
            "accepted and rejected QA IDs overlap",
        ),
    ],
)
def test_curation_validator_rejects_invalid_qa_identity_inventory(
    tmp_path,
    mutate,
    message,
):
    rejected = _flashrag_hotpot_record(9)
    rejected["metadata"]["supporting_facts"]["sent_id"][1] = 99
    records = [rejected, *(_flashrag_hotpot_record(index) for index in range(3))]
    result = curate_source_records(records, dataset="hotpotqa")
    source_sha256 = "b" * 64
    curation = builder._write_curation_ledger(
        tmp_path,
        result,
        source_sha256=source_sha256,
    )
    updated = _rewrite_curation_ledger(tmp_path, curation, mutate)

    with pytest.raises(ManifestValidationError, match=message):
        builder._validate_curation(
            tmp_path,
            updated,
            source=_portable_source_contract(
                tmp_path,
                len(records),
                source_sha256,
            ),
            dataset="hotpotqa",
        )


def test_curation_validator_rejects_inconsistent_ambiguous_variant_inventory(tmp_path):
    first, second = _ambiguous_records_with_oob_overlap()
    records = [first, second, *(_flashrag_hotpot_record(index) for index in range(3))]
    result = curate_source_records(records, dataset="hotpotqa")
    source_sha256 = "c" * 64
    curation = builder._write_curation_ledger(
        tmp_path,
        result,
        source_sha256=source_sha256,
    )

    def mutate(ledger):
        evidence = ledger["rejections"][1]["reasons"][0]["evidence"][0]
        evidence["raw_variant_sha256s"] = sorted(
            [*evidence["raw_variant_sha256s"], "f" * 64]
        )

    updated = _rewrite_curation_ledger(tmp_path, curation, mutate)

    with pytest.raises(ManifestValidationError, match="differs across rejections"):
        builder._validate_curation(
            tmp_path,
            updated,
            source=_portable_source_contract(
                tmp_path,
                len(records),
                source_sha256,
            ),
            dataset="hotpotqa",
        )


def test_curation_validator_rejects_nondeterministic_oob_evidence_order(tmp_path):
    rejected = _flashrag_hotpot_record(9)
    rejected["metadata"]["supporting_facts"]["sent_id"] = [99, 98]
    records = [rejected, *(_flashrag_hotpot_record(index) for index in range(3))]
    result = curate_source_records(records, dataset="hotpotqa")
    source_sha256 = "d" * 64
    curation = builder._write_curation_ledger(
        tmp_path,
        result,
        source_sha256=source_sha256,
    )

    def mutate(ledger):
        ledger["rejections"][0]["reasons"][0]["evidence"].reverse()

    updated = _rewrite_curation_ledger(tmp_path, curation, mutate)

    with pytest.raises(ManifestValidationError, match="strictly increasing"):
        builder._validate_curation(
            tmp_path,
            updated,
            source=_portable_source_contract(
                tmp_path,
                len(records),
                source_sha256,
            ),
            dataset="hotpotqa",
        )


def test_curation_ledger_supports_zero_rejections_and_detects_byte_tampering(tmp_path):
    records = [_flashrag_hotpot_record(index) for index in range(3)]
    result = curate_source_records(records, dataset="hotpotqa")
    source_sha256 = "e" * 64
    curation = builder._write_curation_ledger(
        tmp_path,
        result,
        source_sha256=source_sha256,
    )
    assert curation["rejected_record_count"] == 0
    assert curation["rejection_ledger"]["row_count"] == 0
    assert builder._validate_curation(
        tmp_path,
        curation,
        source=_portable_source_contract(tmp_path, len(records), source_sha256),
        dataset="hotpotqa",
    ) == tuple(example.qa.qa_id for example in result.examples)

    ledger_path = tmp_path / REJECTION_LEDGER_NAME
    ledger_path.write_bytes(ledger_path.read_bytes() + b" ")
    with pytest.raises(ManifestValidationError, match="(size|hash) mismatch"):
        builder._validate_curation(
            tmp_path,
            curation,
            source=_portable_source_contract(tmp_path, len(records), source_sha256),
            dataset="hotpotqa",
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda ledger: ledger["rejections"][0].__setitem__("extra", True),
            "schema keys differ",
        ),
        (
            lambda ledger: ledger.__setitem__(
                "input_record_count", ledger["input_record_count"] + 1
            ),
            "identity mismatch",
        ),
        (
            lambda ledger: ledger["rejections"][0]["reasons"][0].__setitem__(
                "code", "unclassified_source_error"
            ),
            "reason is not allowed",
        ),
        (
            lambda ledger: ledger["rejections"].reverse(),
            "source indices are not sorted unique",
        ),
        (
            lambda ledger: ledger["accepted_qa_ids"].reverse(),
            "differs from its canonical ledger",
        ),
    ],
)
def test_resealed_curation_ledger_still_fails_closed(
    tmp_path,
    mutate,
    message,
):
    first = _flashrag_hotpot_record(8)
    second = _flashrag_hotpot_record(9)
    first["metadata"]["supporting_facts"]["sent_id"][1] = 99
    second["metadata"]["supporting_facts"]["sent_id"][1] = 99
    records = [
        first,
        second,
        *(_flashrag_hotpot_record(index) for index in range(3)),
    ]
    result = curate_source_records(records, dataset="hotpotqa")
    source_sha256 = "f" * 64
    curation = builder._write_curation_ledger(
        tmp_path,
        result,
        source_sha256=source_sha256,
    )
    updated = _rewrite_curation_ledger(tmp_path, curation, mutate)

    with pytest.raises(ManifestValidationError, match=message):
        builder._validate_curation(
            tmp_path,
            updated,
            source=_portable_source_contract(
                tmp_path,
                len(records),
                source_sha256,
            ),
            dataset="hotpotqa",
        )


def test_strict_source_replay_rejects_resealed_evidence_hash_tampering(tmp_path):
    rejected = _flashrag_hotpot_record(9)
    rejected["metadata"]["supporting_facts"]["sent_id"][1] = 99
    records = [rejected, *(_flashrag_hotpot_record(index) for index in range(3))]
    source = _write_json(tmp_path / "source.json", records).resolve()
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "formal-bundle"
    output.mkdir()
    result = curate_source_records(records, dataset="hotpotqa")
    curation = builder._write_curation_ledger(
        output,
        result,
        source_sha256=source_sha256,
    )

    def mutate(ledger):
        ledger["rejections"][0]["source_record_sha256"] = "0" * 64

    curation = _rewrite_curation_ledger(output, curation, mutate)
    accepted_ids = tuple(example.qa.qa_id for example in result.examples)
    assert builder._validate_curation(
        output,
        curation,
        source={
            "format": "json",
            "path": str(source),
            "record_count": len(records),
            "revision": "source-rev",
            "sha256": source_sha256,
            "split": None,
        },
        dataset="hotpotqa",
    ) == accepted_ids
    payload = builder._manifest_payload(
        mode="train",
        profile="formal",
        dataset="hotpotqa",
        seed=42,
        source_path=source,
        source_revision="source-rev",
        source_sha256=source_sha256,
        source_record_count=len(records),
        source_format="json",
        source_split=None,
        tokenizer_name="tokenizer",
        tokenizer_revision="tokenizer-rev",
        contract={"qa_count": 2},
        qa_ids=accepted_ids[:2],
        artifacts={},
        curation=curation,
    )
    builder._write_top_manifest(output / "manifest.json", payload)

    with pytest.raises(ManifestValidationError, match="strict source replay"):
        validate_artifact_bundle(output, replay_source_curation=True)


def test_bundle_schema_is_profile_scoped_for_v2_fixture_migration():
    common = {
        "mode": "train",
        "dataset": "hotpotqa",
        "seed": 42,
        "source_path": Path("source.json").resolve(),
        "source_revision": "source-rev",
        "source_sha256": "a" * 64,
        "source_record_count": 4,
        "source_format": "json",
        "source_split": None,
        "tokenizer_name": "tokenizer",
        "tokenizer_revision": "tokenizer-rev",
        "contract": {"qa_count": 2},
        "qa_ids": ("qa-0", "qa-1"),
        "artifacts": {},
    }

    fixture = builder._manifest_payload(profile="fixture", curation=None, **common)
    formal = builder._manifest_payload(
        profile="formal",
        curation={"policy": CURATION_POLICY},
        **common,
    )

    assert fixture["kind"] == BUNDLE_KIND
    assert fixture["schema_version"] == BUNDLE_SCHEMA_VERSION
    assert "curation" not in fixture
    assert "format" not in fixture["source"]
    assert formal["kind"] == FORMAL_BUNDLE_KIND
    assert formal["schema_version"] == FORMAL_BUNDLE_SCHEMA_VERSION
    assert formal["curation"] == {"policy": CURATION_POLICY}
    assert formal["source"]["format"] == "json"


def test_top_manifest_rejects_formal_v2_and_fixture_v3_cross_disguises(tmp_path):
    common = {
        "mode": "train",
        "dataset": "hotpotqa",
        "seed": 42,
        "source_path": Path("source.json").resolve(),
        "source_revision": "source-rev",
        "source_sha256": "a" * 64,
        "source_record_count": 4,
        "source_format": "json",
        "source_split": None,
        "tokenizer_name": "tokenizer",
        "tokenizer_revision": "tokenizer-rev",
        "contract": {"qa_count": 2},
        "qa_ids": ("hotpotqa:qa-0", "hotpotqa:qa-1"),
        "artifacts": {},
    }
    formal_v2 = builder._manifest_payload(
        profile="formal",
        curation={"policy": CURATION_POLICY},
        **common,
    )
    formal_v2["kind"] = BUNDLE_KIND
    formal_v2["schema_version"] = BUNDLE_SCHEMA_VERSION
    builder._write_top_manifest(tmp_path / "formal-v2.json", formal_v2)
    with pytest.raises(ManifestValidationError, match="profile/kind/schema"):
        builder._read_top_manifest(tmp_path / "formal-v2.json")

    fixture_v3 = builder._manifest_payload(
        profile="fixture",
        curation=None,
        **common,
    )
    fixture_v3["kind"] = FORMAL_BUNDLE_KIND
    fixture_v3["schema_version"] = FORMAL_BUNDLE_SCHEMA_VERSION
    builder._write_top_manifest(tmp_path / "fixture-v3.json", fixture_v3)
    with pytest.raises(ManifestValidationError, match="profile/kind/schema"):
        builder._read_top_manifest(tmp_path / "fixture-v3.json")


def test_formal_train_contract_is_locked_to_512_by_200_and_six_chunks():
    assert FORMAL_TRAIN_QA_COUNT == 512
    assert FORMAL_TRAIN_DOCUMENT_COUNT == 200
    assert FORMAL_TRAIN_CHUNK_SIZE == 5000
    assert FORMAL_TRAIN_MAX_CHUNKS == 6
    assert FORMAL_TRAIN_MIN_CONTEXT_TOKENS == 25_001
    assert FORMAL_TRAIN_MAX_CONTEXT_TOKENS == 30_000
    assert TrainManifestContract() == TrainManifestContract(
        qa_count=512,
        document_count=200,
        chunk_size=5000,
        max_chunks=6,
        min_context_tokens=25_001,
        max_context_tokens=30_000,
    )


@pytest.mark.parametrize(
    ("dataset", "factory"),
    [
        ("hotpotqa", _hotpot_record),
        ("hotpotqa", _flashrag_hotpot_record),
        ("2wikimultihopqa", _flashrag_2wiki_record),
    ],
)
def test_native_and_flashrag_schemas_preserve_all_gold_and_sentence_ids(
    dataset,
    factory,
):
    parsed = parse_source_record(factory(0), dataset=dataset, source_index=0)

    assert [answer.text for answer in parsed.qa.gold_answers] == [
        "answer 0" if dataset == "hotpotqa" else "connection 0",
        "alias 0" if dataset == "hotpotqa" else "link 0",
    ]
    assert parsed.qa.qa_id.startswith(f"{dataset}:")
    assert len(parsed.supporting_facts) == 2
    first = parsed.supporting_facts[0]
    document = parsed.documents[first.document_index]
    assert document.text[first.start_char : first.end_char] == first.text
    document_id = stable_document_id(document.title, document.text)
    assert stable_supporting_fact_id(
        document_id,
        first.sentence_index,
        first.text,
    ).startswith("fact_")


@pytest.mark.parametrize(
    ("dataset", "factory"),
    [
        ("hotpotqa", _hotpot_record),
        ("2wikimultihopqa", _flashrag_2wiki_record),
    ],
)
def test_eval_bundle_writes_nested_parquet_sidecars_and_strict_readback(
    tmp_path,
    dataset,
    factory,
):
    pytest.importorskip("pyarrow")
    source = _write_json(
        tmp_path / f"{dataset}.json",
        [factory(index) for index in range(5)],
    )
    output = tmp_path / f"{dataset}-eval"
    contract = EvalManifestContract(
        qa_count=2,
        prefix_document_count=2,
        pool_document_count=4,
        chunk_size=5,
    )

    manifest = build_artifact_bundle(
        input_path=source,
        output_dir=output,
        mode="eval",
        dataset=dataset,
        source_revision="fixture-source-rev",
        tokenizer_name="fixture-tokenizer",
        tokenizer_revision="fixture-tokenizer-rev",
        seed=42,
        encode=_tokenize_with_offsets,
        profile="fixture",
        eval_contract=contract,
    )

    assert manifest["profile"] == "fixture"
    assert set(path.name for path in output.iterdir()) == {
        "manifest.json",
        "eval_2.parquet",
        "eval_2.sidecar.jsonl",
        "eval_4.parquet",
        "eval_4.sidecar.jsonl",
    }
    assert validate_artifact_bundle(output)["manifest_sha256"] == manifest[
        "manifest_sha256"
    ]

    prefix_values = [
        json.loads(line)
        for line in (output / "eval_2.sidecar.jsonl").read_text().splitlines()
    ]
    pool_values = [
        json.loads(line)
        for line in (output / "eval_4.sidecar.jsonl").read_text().splitlines()
    ]
    for prefix, pool in zip(prefix_values, pool_values):
        assert prefix["qa"] == pool["qa"]
        assert prefix["documents"] == pool["documents"][:2]
        assert prefix["document_pool_sha256"] == pool["document_pool_sha256"]
        assert prefix["supporting_facts"] == pool["supporting_facts"]
        assert all(fact["document_position"] < 2 for fact in pool["supporting_facts"])
        manifest_record_from_dict(prefix)
        manifest_record_from_dict(pool)

    import pyarrow.parquet as pq

    rows = pq.read_table(output / "eval_2.parquet").to_pylist()
    assert set(rows[0]) == {
        "answers",
        "context",
        "data_source",
        "extra_info",
        "prompt",
        "reward_model",
    }
    assert rows[0]["answers"] == rows[0]["reward_model"]["ground_truth"]
    assert rows[0]["extra_info"]["qa_id"] == prefix_values[0]["qa"]["qa_id"]
    assert rows[0]["extra_info"]["manifest_record_sha256"] == prefix_values[0][
        "record_sha256"
    ]


def test_same_fixed_input_rebuilds_identical_artifact_hashes(tmp_path):
    pytest.importorskip("pyarrow")
    source = _write_json(
        tmp_path / "source.json",
        [_hotpot_record(index) for index in range(5)],
    )
    contract = EvalManifestContract(
        qa_count=2,
        prefix_document_count=2,
        pool_document_count=4,
        chunk_size=5,
    )

    first = build_artifact_bundle(
        input_path=source,
        output_dir=tmp_path / "first",
        mode="eval",
        dataset="hotpotqa",
        source_revision="source-rev",
        tokenizer_name="tokenizer",
        tokenizer_revision="tokenizer-rev",
        seed=42,
        encode=_tokenize_with_offsets,
        profile="fixture",
        eval_contract=contract,
    )
    second = build_artifact_bundle(
        input_path=source,
        output_dir=tmp_path / "second",
        mode="eval",
        dataset="hotpotqa",
        source_revision="source-rev",
        tokenizer_name="tokenizer",
        tokenizer_revision="tokenizer-rev",
        seed=42,
        encode=_tokenize_with_offsets,
        profile="fixture",
        eval_contract=contract,
    )

    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert {
        name: value["sha256"] for name, value in first["artifacts"].items()
    } == {
        name: value["sha256"] for name, value in second["artifacts"].items()
    }


def test_train_bundle_emits_memory_dataset_columns_and_enforces_token_window(tmp_path):
    pytest.importorskip("pyarrow")
    source = _write_json(
        tmp_path / "train.json",
        [_hotpot_record(index) for index in range(5)],
    )
    contract = TrainManifestContract(
        qa_count=2,
        document_count=3,
        chunk_size=1000,
        max_chunks=1,
        min_context_tokens=1,
        max_context_tokens=1000,
    )
    output = tmp_path / "train-bundle"

    manifest = build_artifact_bundle(
        input_path=source,
        output_dir=output,
        mode="train",
        dataset="hotpotqa",
        source_revision="train-source-rev",
        tokenizer_name="fixture-tokenizer",
        tokenizer_revision="fixture-tokenizer-rev",
        seed=42,
        encode=_tokenize_with_offsets,
        profile="fixture",
        train_contract=contract,
    )

    assert set(manifest["artifacts"]) == {
        "train.parquet",
        "train.sidecar.jsonl",
    }
    values = [
        json.loads(line)
        for line in (output / "train.sidecar.jsonl").read_text().splitlines()
    ]
    assert len(values) == 2
    assert all(value["document_count"] == 3 for value in values)
    assert all(len(value["chunks"]) == 1 for value in values)
    assert all(value["truncation"] == "none" for value in values)
    validate_artifact_bundle(output)

    datasets = pytest.importorskip("datasets")
    loaded = datasets.Dataset.from_parquet(str(output / "train.parquet"))
    assert loaded.column_names == [
        "answers",
        "context",
        "data_source",
        "extra_info",
        "prompt",
        "reward_model",
    ]
    assert loaded[0]["prompt"] == [
        {"content": loaded[0]["prompt"][0]["content"], "role": "user"}
    ]
    assert loaded[0]["reward_model"]["ground_truth"] == loaded[0]["answers"]


def _length_fit_case(*, initial_words, replacement_words):
    example = parse_source_record(
        _hotpot_record(0),
        dataset="hotpotqa",
        source_index=0,
    )
    corpus = builder._canonical_corpus((example,))
    documents = []
    for name, word_count in (
        ("initial", initial_words),
        ("replacement", replacement_words),
    ):
        document = builder.DocumentInput(
            title=f"{name} distractor",
            text=" ".join(f"{name}-{index}" for index in range(word_count)),
            source_document_id=f"fixture:{name}",
        )
        corpus[document.document_id] = document
        documents.append(document)
    ranked = tuple(document.document_id for document in documents)
    initial_pool, facts = builder._materialize_document_pool(
        example,
        corpus=corpus,
        prefix_document_count=3,
        pool_document_count=3,
        seed=42,
        ranked_distractor_ids=ranked,
    )
    initial_count = builder.count_tokens(
        _tokenize_with_offsets,
        builder.render_context(initial_pool),
    )
    return example, corpus, ranked, initial_pool, facts, initial_count


@pytest.mark.parametrize("direction", ["short", "long"])
def test_formal_train_length_fit_is_deterministic_and_preserves_core_facts(direction):
    if direction == "short":
        values = _length_fit_case(initial_words=1, replacement_words=20)
    else:
        values = _length_fit_case(initial_words=20, replacement_words=1)
    example, corpus, ranked, initial_pool, facts, initial_count = values
    if direction == "short":
        minimum, maximum = initial_count + 5, initial_count + 25
    else:
        minimum, maximum = 1, initial_count - 5
    contract = TrainManifestContract(
        qa_count=1,
        document_count=3,
        chunk_size=maximum,
        max_chunks=1,
        min_context_tokens=minimum,
        max_context_tokens=maximum,
    )

    results = [
        builder._fit_formal_train_pool(
            example,
            initial_pool=initial_pool,
            initial_context_token_count=initial_count,
            ranked_distractor_ids=ranked,
            corpus=corpus,
            encode=_tokenize_with_offsets,
            contract=contract,
            body_token_counts={},
        )
        for _ in range(2)
    ]

    assert results[0] == results[1]
    fitted_pool, fitted_count = results[0]
    assert minimum <= fitted_count <= maximum
    initial_ids = [document.document_id for document in initial_pool]
    fitted_ids = [document.document_id for document in fitted_pool]
    changed_slots = [
        index
        for index, values in enumerate(zip(initial_ids, fitted_ids))
        if values[0] != values[1]
    ]
    assert len(changed_slots) == 1
    core_ids = {document.document_id for document in example.documents}
    assert core_ids.issubset(fitted_ids)
    assert all(
        initial_ids[index] == fitted_ids[index]
        for index, document_id in enumerate(initial_ids)
        if document_id in core_ids
    )

    metadata = builder.ManifestMetadata(
        source_name="hotpotqa",
        source_revision="source-rev",
        source_sha256="0" * 64,
        tokenizer_name="tokenizer",
        tokenizer_revision="tokenizer-rev",
        seed=42,
    )
    qa_order_sha256 = ordered_values_sha256((example.qa.qa_id,))
    records = []
    for pool in (initial_pool, fitted_pool):
        records.append(
            builder.build_manifest_record(
                metadata=metadata,
                qa_index=0,
                qa_order_sha256=qa_order_sha256,
                qa=example.qa,
                documents=pool,
                supporting_facts=facts,
                encode=_tokenize_with_offsets,
                chunk_size=maximum,
                pool_document_count=3,
                document_pool_sha256=ordered_values_sha256(
                    tuple(document.document_id for document in pool)
                ),
            )
        )
    assert [
        (
            fact.fact_id,
            fact.document_id,
            fact.document_position,
            fact.sentence_index,
            fact.text,
        )
        for fact in records[0].supporting_facts
    ] == [
        (
            fact.fact_id,
            fact.document_id,
            fact.document_position,
            fact.sentence_index,
            fact.text,
        )
        for fact in records[1].supporting_facts
    ]


def test_formal_train_length_fit_fails_closed_without_an_exact_improvement():
    values = _length_fit_case(initial_words=1, replacement_words=1)
    example, corpus, ranked, initial_pool, _, initial_count = values
    contract = TrainManifestContract(
        qa_count=1,
        document_count=3,
        chunk_size=initial_count + 10,
        max_chunks=1,
        min_context_tokens=initial_count + 5,
        max_context_tokens=initial_count + 10,
    )

    with pytest.raises(ManifestValidationError, match="no exact monotonic"):
        builder._fit_formal_train_pool(
            example,
            initial_pool=initial_pool,
            initial_context_token_count=initial_count,
            ranked_distractor_ids=ranked,
            corpus=corpus,
            encode=_tokenize_with_offsets,
            contract=contract,
            body_token_counts={},
        )


def test_bounded_distractor_ranking_is_the_exact_full_ranking_prefix():
    examples = tuple(
        parse_source_record(_hotpot_record(index), dataset="hotpotqa", source_index=index)
        for index in range(5)
    )
    corpus = builder._canonical_corpus(examples)
    full = builder._ranked_distractor_ids(examples[0], corpus=corpus, seed=42)
    bounded = builder._ranked_distractor_ids(
        examples[0],
        corpus=corpus,
        seed=42,
        limit=3,
    )

    assert bounded == full[:3]


def test_parallel_initial_distractor_ranking_matches_individual_prefixes(monkeypatch):
    examples = tuple(
        parse_source_record(_hotpot_record(index), dataset="hotpotqa", source_index=index)
        for index in range(8)
    )
    corpus = builder._canonical_corpus(examples)
    limits = (3, None, 4, 5, 6, 3, 4, 5)
    monkeypatch.setattr(builder, "_effective_cpu_count", lambda: 4)

    ranked = builder._rank_initial_train_distractors(
        examples,
        corpus=corpus,
        seed=42,
        limits=limits,
    )

    expected = tuple(
        builder._ranked_distractor_ids(
            example,
            corpus=corpus,
            seed=42,
            limit=limit,
        )
        if limit is not None
        else None
        for example, limit in zip(examples, limits)
    )
    assert ranked == expected


def test_cli_fixture_build_runs_end_to_end_without_loading_remote_tokenizer(
    tmp_path,
    capsys,
):
    pytest.importorskip("pyarrow")
    source = _write_json(
        tmp_path / "cli.json",
        [_hotpot_record(index) for index in range(5)],
    )
    output = tmp_path / "cli-output"
    loads = []

    exit_code = main(
        [
            "train",
            "--input",
            str(source),
            "--output",
            str(output),
            "--dataset",
            "hotpotqa",
            "--source-revision",
            "fixture-source-rev",
            "--tokenizer",
            "fixture-tokenizer",
            "--tokenizer-revision",
            "fixture-tokenizer-rev",
            "--profile",
            "fixture",
            "--qa-count",
            "2",
            "--train-documents",
            "3",
            "--chunk-size",
            "1000",
            "--max-chunks",
            "1",
            "--min-context-tokens",
            "1",
            "--max-context-tokens",
            "1000",
        ],
        tokenizer_loader=lambda name, revision: (
            loads.append((name, revision)) or _tokenize_with_offsets
        ),
    )

    assert exit_code == 0
    assert loads == [("fixture-tokenizer", "fixture-tokenizer-rev")]
    assert json.loads(capsys.readouterr().out)["profile"] == "fixture"
    validate_artifact_bundle(output)


def test_formal_profile_refuses_small_fixture_contract(tmp_path):
    source = _write_json(tmp_path / "source.json", [_hotpot_record(0)])
    contract = TrainManifestContract(
        qa_count=1,
        document_count=2,
        chunk_size=1000,
        max_chunks=1,
        min_context_tokens=1,
        max_context_tokens=1000,
    )

    with pytest.raises(ManifestValidationError, match="formal profile cannot override"):
        build_artifact_bundle(
            input_path=source,
            output_dir=tmp_path / "must-not-exist",
            mode="train",
            dataset="hotpotqa",
            source_revision="source-rev",
            tokenizer_name="tokenizer",
            tokenizer_revision="tokenizer-rev",
            seed=42,
            encode=_tokenize_with_offsets,
            profile="formal",
            train_contract=contract,
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_bundle_readback_detects_sidecar_tampering(tmp_path):
    pytest.importorskip("pyarrow")
    source = _write_json(
        tmp_path / "source.json",
        [_hotpot_record(index) for index in range(5)],
    )
    output = tmp_path / "bundle"
    build_artifact_bundle(
        input_path=source,
        output_dir=output,
        mode="eval",
        dataset="hotpotqa",
        source_revision="source-rev",
        tokenizer_name="tokenizer",
        tokenizer_revision="tokenizer-rev",
        seed=42,
        encode=_tokenize_with_offsets,
        profile="fixture",
        eval_contract=EvalManifestContract(
            qa_count=2,
            prefix_document_count=2,
            pool_document_count=4,
            chunk_size=5,
        ),
    )
    sidecar = output / "eval_2.sidecar.jsonl"
    sidecar.write_bytes(sidecar.read_bytes() + b"{}\n")

    with pytest.raises(ManifestValidationError, match="artifact (size|hash) mismatch"):
        validate_artifact_bundle(output)


def test_train_builder_fails_closed_when_context_exceeds_contract(tmp_path):
    source = _write_json(
        tmp_path / "source.json",
        [_hotpot_record(index) for index in range(5)],
    )
    contract = TrainManifestContract(
        qa_count=1,
        document_count=2,
        chunk_size=5,
        max_chunks=1,
        min_context_tokens=1,
        max_context_tokens=5,
    )

    with pytest.raises(ManifestValidationError, match="context has .* formal range"):
        build_artifact_bundle(
            input_path=source,
            output_dir=tmp_path / "too-long",
            mode="train",
            dataset="hotpotqa",
            source_revision="source-rev",
            tokenizer_name="tokenizer",
            tokenizer_revision="tokenizer-rev",
            seed=42,
            encode=_tokenize_with_offsets,
            profile="fixture",
            train_contract=contract,
        )
    assert not (tmp_path / "too-long").exists()


def test_missing_sentence_level_support_is_rejected():
    record = _flashrag_2wiki_record(0)
    del record["metadata"]["supporting_facts"]["sent_id"]

    with pytest.raises(ManifestValidationError, match="sent_id"):
        parse_source_record(record, dataset="2wikimultihopqa", source_index=0)


def test_flattened_context_cannot_claim_document_provenance():
    record = _hotpot_record(0)
    record["context"] = "already flattened without document boundaries"

    with pytest.raises(ManifestValidationError, match="context"):
        parse_source_record(record, dataset="hotpotqa", source_index=0)
