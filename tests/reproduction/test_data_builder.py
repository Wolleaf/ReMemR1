import json
import re
from pathlib import Path

import pytest

from taskutils.data_synthesis.reproduction_builder import (
    FORMAL_TRAIN_CHUNK_SIZE,
    FORMAL_TRAIN_DOCUMENT_COUNT,
    FORMAL_TRAIN_MAX_CHUNKS,
    FORMAL_TRAIN_MAX_CONTEXT_TOKENS,
    FORMAL_TRAIN_MIN_CONTEXT_TOKENS,
    FORMAL_TRAIN_QA_COUNT,
    TrainManifestContract,
    build_artifact_bundle,
    main,
    manifest_record_from_dict,
    parse_source_record,
    validate_artifact_bundle,
)
from taskutils.data_synthesis.reproduction_manifest import (
    EvalManifestContract,
    ManifestValidationError,
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


def _write_json(path, records):
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return path


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
