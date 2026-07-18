from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.cloud import cloud_state
from scripts.reproduction.prefetch_assets import seal_manifest, validate_asset_manifest


COMMIT = "a" * 40
REVISION = "b" * 40
SHA256 = "c" * 64
HANDOFF_KEYS = {
    "asset_manifest",
    "asset_report",
    "bundles",
    "config_files",
    "config_root",
    "config_tree_sha256",
    "data_root",
    "environment_lock",
    "environment_lock_sha256",
    "experiment_profile_id",
    "git_commit",
    "handoff_sha256",
    "kernel_source_root",
    "kernel_sources",
    "pip_freeze",
    "persist_root",
    "schema_version",
    "status",
}


def _write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _asset_manifest(assets):
    return seal_manifest({"assets": assets, "schema_version": 1})


def _unresolved_manifest():
    return _asset_manifest(
        [
            {
                "asset_id": "formal-source",
                "files": [
                    {
                        "metadata_status": "METADATA_UNAVAILABLE",
                        "path": "train.parquet",
                        "required": True,
                    }
                ],
                "kind": "formal_training_source",
                "metadata_resolution": {
                    "command": "resolve metadata",
                    "required_fields": ["size", "sha256"],
                    "status": "METADATA_UNAVAILABLE",
                },
                "repo_id": "owner/formal",
                "repo_type": "dataset",
                "revision": REVISION,
                "training_gate": "BLOCKED",
            }
        ]
    )


def _resolved_asset(asset_id, kind, repo_id, files, revision=REVISION):
    return {
        "asset_id": asset_id,
        "files": files,
        "kind": kind,
        "repo_id": repo_id,
        "repo_type": "model" if kind == "model_and_tokenizer" else "dataset",
        "revision": revision,
    }


def test_resolve_assets_retries_lfs_metadata_and_seals_external_manifest(tmp_path):
    source = _write_json(tmp_path / "tracked.json", _unresolved_manifest())
    output = tmp_path / "runtime" / "assets.json"

    class FakeApi:
        def __init__(self):
            self.calls = []

        def get_paths_info(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise TimeoutError("transient")
            return [
                SimpleNamespace(
                    rfilename="train.parquet",
                    size=123,
                    lfs=SimpleNamespace(size=123, sha256=SHA256),
                )
            ]

    api = FakeApi()
    sleeps = []
    result = cloud_state.resolve_assets(
        source,
        output,
        api=api,
        retries=2,
        sleep=sleeps.append,
    )

    resolved = json.loads(output.read_text(encoding="utf-8"))
    validate_asset_manifest(resolved)
    assert result["resolved_file_count"] == 1
    assert result["manifest_sha256"] == resolved["manifest_sha256"]
    assert resolved["assets"][0]["files"] == [
        {"path": "train.parquet", "sha256": SHA256, "size": 123}
    ]
    assert "metadata_resolution" not in resolved["assets"][0]
    assert "training_gate" not in resolved["assets"][0]
    assert sleeps == [1]
    assert api.calls[-1] == {
        "expand": True,
        "paths": ["train.parquet"],
        "repo_id": "owner/formal",
        "repo_type": "dataset",
        "revision": REVISION,
    }
    assert not list(output.parent.glob(".*.tmp"))


def test_resolve_assets_is_offline_when_already_resolved_and_never_overwrites_input(tmp_path):
    manifest = _asset_manifest(
        [
            _resolved_asset(
                "dataset-source",
                "dataset_source",
                "owner/source",
                [{"path": "dev.jsonl", "sha256": SHA256, "size": 9}],
            )
        ]
    )
    source = _write_json(tmp_path / "tracked.json", manifest)
    output = tmp_path / "runtime.json"

    class ForbiddenApi:
        def get_paths_info(self, **kwargs):  # pragma: no cover - failure explains itself.
            raise AssertionError(kwargs)

    result = cloud_state.resolve_assets(source, output, api=ForbiddenApi())
    assert result["resolved_file_count"] == 0
    assert json.loads(output.read_text(encoding="utf-8")) == manifest
    with pytest.raises(cloud_state.CloudStateError, match="must not overwrite"):
        cloud_state.resolve_assets(source, source, api=ForbiddenApi())


def test_resolve_assets_fails_closed_without_partial_output(tmp_path):
    source = _write_json(tmp_path / "tracked.json", _unresolved_manifest())
    output = tmp_path / "runtime.json"

    class BrokenApi:
        def get_paths_info(self, **kwargs):
            return [
                SimpleNamespace(
                    rfilename="train.parquet",
                    size=123,
                    lfs=None,
                )
            ]

    with pytest.raises(cloud_state.CloudStateError, match="after 2 attempts"):
        cloud_state.resolve_assets(
            source,
            output,
            api=BrokenApi(),
            retries=2,
            sleep=lambda _: None,
        )
    assert not output.exists()


def _whitespace_encoder(text: str):
    offsets = []
    input_ids = []
    cursor = 0
    for token in text.split():
        start = text.index(token, cursor)
        end = start + len(token)
        offsets.append((start, end))
        input_ids.append(len(input_ids) + 1)
        cursor = end
    return {"input_ids": input_ids, "offset_mapping": offsets}


def test_synthetic_gate_source_has_twenty_distinct_qa_and_satisfies_fixture_contract():
    train = cloud_state._synthetic_gate_records("train", 64)
    validation = cloud_state._synthetic_gate_records("validation", 64)
    assert cloud_state._GATE_CONTRACT.qa_count == 20
    assert len(train) == len(validation) == 20
    assert {record["_id"] for record in train}.isdisjoint(
        record["_id"] for record in validation
    )
    document_ids = {
        document["document_id"]
        for record in train
        for document in record["context"]
    }
    assert len(document_ids) == 40
    minimum, maximum = cloud_state._probe_gate_records(
        train,
        repeat_count=64,
        tokenizer_name="owner/tokenizer",
        tokenizer_revision=REVISION,
        encode=_whitespace_encoder,
    )
    assert 1025 <= minimum <= maximum <= 2048


def test_length_stress_source_is_disjoint_non_scientific_fixed_capacity_fixture():
    train = cloud_state._synthetic_length_stress_records("train", 120)
    validation = cloud_state._synthetic_length_stress_records("validation", 120)
    assert cloud_state._LENGTH_STRESS_CONTRACT.qa_count == 2
    assert len(train) == len(validation) == 100
    assert {record["_id"] for record in train}.isdisjoint(
        record["_id"] for record in validation
    )
    assert len(
        {
            document["document_id"]
            for record in train
            for document in record["context"]
        }
    ) == 200
    minimum, maximum = cloud_state._probe_length_stress_records(
        train,
        repeat_count=120,
        tokenizer_name="owner/qwen2",
        tokenizer_revision=REVISION,
        encode=_whitespace_encoder,
    )
    assert 25_001 <= minimum <= maximum <= 30_000
    payload = cloud_state._jsonl_bytes(train)
    examples = tuple(
        cloud_state.parse_source_record(record, dataset="hotpotqa", source_index=index)
        for index, record in enumerate(train)
    )
    metadata = cloud_state.ManifestMetadata(
        source_name="hotpotqa",
        source_revision=(
            f"{cloud_state._LENGTH_STRESS_SOURCE_REVISION_PREFIX}-r120"
        ),
        source_sha256=hashlib.sha256(payload).hexdigest(),
        tokenizer_name="owner/qwen2",
        tokenizer_revision=REVISION,
        seed=cloud_state.SEED,
    )
    first = cloud_state.build_train_records(
        examples,
        metadata=metadata,
        encode=_whitespace_encoder,
        contract=cloud_state._LENGTH_STRESS_CONTRACT,
    )
    second = cloud_state.build_train_records(
        examples,
        metadata=metadata,
        encode=_whitespace_encoder,
        contract=cloud_state._LENGTH_STRESS_CONTRACT,
    )
    assert len(first) == 2
    assert [record.qa.qa_id for record in first] == [
        record.qa.qa_id for record in second
    ]
    assert [record.record_sha256 for record in first] == [
        record.record_sha256 for record in second
    ]
    assert all(record.document_count == 200 for record in first)
    assert all(25_001 <= record.context_token_count <= 30_000 for record in first)


def test_active_profile_constants_and_bundle_identities_are_exact():
    assert cloud_state.SCHEMA_VERSION == 3
    assert cloud_state.EXPERIMENT_PROFILE_ID == "rtx5090-32g-qwen35-2b-v1"
    assert len(cloud_state._CONFIG_IDS) == 33
    assert len(cloud_state._TRAINING_SOURCE_IDS) == 14
    assert all("4b" not in value.casefold() for value in cloud_state._CONFIG_IDS)
    formal_specs = [spec for spec in cloud_state._BUNDLE_SPECS if spec.keys[0] == "formal"]
    assert formal_specs
    assert {spec.tokenizer_asset for spec in formal_specs} == {
        "qwen35-2b-model-tokenizer"
    }
    length_specs = [
        spec for spec in cloud_state._BUNDLE_SPECS if spec.length_stress_split is not None
    ]
    assert {spec.length_stress_split for spec in length_specs} == {
        "train",
        "validation",
    }
    assert all(spec.profile == "fixture" for spec in length_specs)
    assert all(spec.tokenizer_asset == "qwen35-2b-model-tokenizer" for spec in length_specs)
    assert all(spec.relative_path.startswith("capacity/length-stress/") for spec in length_specs)


def test_length_stress_bundle_builds_two_verified_non_scientific_records(tmp_path):
    parquet = pytest.importorskip(
        "pyarrow.parquet", reason="pyarrow is required to verify Parquet bundles"
    )
    repeat_count = 120
    source = tmp_path / "capacity" / "length-stress" / "source" / "train.jsonl"
    source.parent.mkdir(parents=True)
    source.write_bytes(
        cloud_state._jsonl_bytes(
            cloud_state._synthetic_length_stress_records("train", repeat_count)
        )
    )
    bundle = tmp_path / "capacity" / "length-stress" / "train"

    manifest = cloud_state.build_artifact_bundle(
        dataset="hotpotqa",
        encode=_whitespace_encoder,
        input_path=source,
        mode="train",
        output_dir=bundle,
        profile="fixture",
        seed=cloud_state.SEED,
        source_revision=(
            f"{cloud_state._LENGTH_STRESS_SOURCE_REVISION_PREFIX}-r{repeat_count:03d}"
        ),
        split=None,
        tokenizer_name="owner/qwen2",
        tokenizer_revision=REVISION,
        train_contract=cloud_state._LENGTH_STRESS_CONTRACT,
    )
    verified = cloud_state.validate_artifact_bundle(bundle)
    rows = parquet.read_table(bundle / "train.parquet").to_pylist()

    assert verified == manifest
    assert manifest["profile"] == "fixture"
    assert manifest["contract"] == cloud_state._LENGTH_STRESS_CONTRACT.to_dict()
    assert manifest["source"]["record_count"] == 100
    assert manifest["source"]["revision"].startswith(
        cloud_state._LENGTH_STRESS_SOURCE_REVISION_PREFIX
    )
    assert len(rows) == 2
    assert all(row["extra_info"]["document_count"] == 200 for row in rows)
    assert all(
        25_001 <= row["extra_info"]["context_token_count"] <= 30_000
        for row in rows
    )


def test_g1_eval_fixture_builds_two_recurrent_records(tmp_path):
    pytest.importorskip("pyarrow", reason="pyarrow is required to build Parquet fixtures")
    from taskutils.memory_eval.reproduction_runner import load_eval_records

    repeat_count = 64
    source = tmp_path / "validation.jsonl"
    source.write_bytes(
        cloud_state._jsonl_bytes(
            cloud_state._synthetic_gate_records("validation", repeat_count)
        )
    )
    bundle = tmp_path / "g1-eval"
    manifest = cloud_state.build_artifact_bundle(
        dataset="hotpotqa",
        encode=_whitespace_encoder,
        eval_contract=cloud_state._G1_EVAL_CONTRACT,
        input_path=source,
        mode="eval",
        output_dir=bundle,
        profile="fixture",
        seed=cloud_state.SEED,
        source_revision=f"{cloud_state._GATE_SOURCE_REVISION_PREFIX}-r{repeat_count:03d}",
        split=None,
        tokenizer_name="owner/qwen2",
        tokenizer_revision=REVISION,
    )

    records, observed = load_eval_records(
        bundle,
        expected_manifest_sha256=manifest["manifest_sha256"],
        variant=cloud_state._G1_EVAL_CONTRACT.pool_document_count,
        sample_count=cloud_state._G1_EVAL_CONTRACT.qa_count,
    )
    assert observed["profile"] == "fixture"
    assert len(records) == 2
    assert all(record.chunk_size == 1024 for record in records)
    assert all(len(record.chunks) == 2 for record in records)


def _build_data_manifest(tmp_path: Path):
    source_payloads = {
        ("owner/formal", "hotpotqa_train_32k.parquet"): b"formal train",
        ("owner/formal", "hotpotqa_dev.parquet"): b"formal validation",
        ("owner/hotpot", "hotpotqa/train.jsonl"): b"hotpot train",
        ("owner/hotpot", "hotpotqa/dev.jsonl"): b"hotpot eval",
        ("owner/2wiki", "2wikimultihopqa/dev.jsonl"): b"2wiki eval",
    }
    assets = [
        _resolved_asset(
            "qwen35-08b-model-tokenizer",
            "model_and_tokenizer",
            "owner/qwen08",
            [{"path": "tokenizer.json", "sha256": "0" * 64, "size": 0}],
            "1" * 40,
        ),
        _resolved_asset(
            "qwen35-2b-model-tokenizer",
            "model_and_tokenizer",
            "owner/qwen2",
            [{"path": "tokenizer.json", "sha256": "0" * 64, "size": 0}],
            "2" * 40,
        ),
    ]
    for asset_id, repo_id, filenames, kind in (
        (
            "byted-hotpotqa-formal",
            "owner/formal",
            ("hotpotqa_train_32k.parquet", "hotpotqa_dev.parquet"),
            "formal_training_source",
        ),
        (
            "hotpotqa-source",
            "owner/hotpot",
            ("hotpotqa/train.jsonl", "hotpotqa/dev.jsonl"),
            "dataset_source",
        ),
        (
            "2wikimultihopqa-source",
            "owner/2wiki",
            ("2wikimultihopqa/dev.jsonl",),
            "dataset_source",
        ),
    ):
        files = []
        for filename in filenames:
            payload = source_payloads[(repo_id, filename)]
            files.append(
                {
                    "path": filename,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size": len(payload),
                }
            )
        assets.append(_resolved_asset(asset_id, kind, repo_id, files))
    manifest = _asset_manifest(assets)
    manifest_path = _write_json(tmp_path / "assets.json", manifest)
    files = {}
    for (repo_id, filename), payload in source_payloads.items():
        path = tmp_path / "hf" / repo_id.replace("/", "-") / Path(filename).name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        files[(repo_id, filename)] = path
    return manifest_path, files


def _fake_bundle_manifest(kwargs):
    if kwargs["profile"] == "fixture":
        if kwargs["mode"] == "train":
            contract = kwargs["train_contract"].to_dict()
        else:
            value = kwargs["eval_contract"]
            contract = {
                "chunk_size": value.chunk_size,
                "pool_document_count": value.pool_document_count,
                "prefix_document_count": value.prefix_document_count,
                "qa_count": value.qa_count,
            }
    elif kwargs["mode"] == "train":
        contract = kwargs.get(
            "train_contract",
            cloud_state.TrainManifestContract(),
        ).to_dict()
    else:
        from taskutils.data_synthesis.reproduction_manifest import EvalManifestContract

        value = EvalManifestContract()
        contract = {
            "chunk_size": value.chunk_size,
            "pool_document_count": value.pool_document_count,
            "prefix_document_count": value.prefix_document_count,
            "qa_count": value.qa_count,
        }
    source = Path(kwargs["input_path"]).resolve()
    source_record = {
        "path": str(source),
        "revision": kwargs["source_revision"],
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "split": kwargs["split"],
    }
    if kwargs["profile"] == "formal":
        source_record["format"] = kwargs["source_format"].lstrip(".").lower()
    return {
        "contract": contract,
        "dataset": kwargs["dataset"],
        "manifest_sha256": hashlib.sha256(str(kwargs["output_dir"]).encode()).hexdigest(),
        "mode": kwargs["mode"],
        "profile": kwargs["profile"],
        "seed": kwargs["seed"],
        "source": source_record,
        "tokenizer": {
            "name": kwargs["tokenizer_name"],
            "revision": kwargs["tokenizer_revision"],
        },
    }


def test_build_data_is_local_only_and_idempotently_validates_existing_bundles(
    tmp_path, monkeypatch
):
    manifest_path, source_files = _build_data_manifest(tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    data_root = tmp_path / "data"
    calls = {"build": [], "build_kwargs": [], "download": [], "validate": []}
    manifests = {}

    gate_payloads = {
        split: cloud_state._jsonl_bytes(cloud_state._synthetic_gate_records(split, 64))
        for split in ("train", "validation")
    }
    monkeypatch.setattr(
        cloud_state,
        "_select_gate_sources",
        lambda tokenizers: (64, gate_payloads, {"g0": {}, "g1": {}}),
    )
    length_payloads = {
        split: cloud_state._jsonl_bytes(
            cloud_state._synthetic_length_stress_records(split, 120)
        )
        for split in ("train", "validation")
    }
    monkeypatch.setattr(
        cloud_state,
        "_select_length_stress_sources",
        lambda tokenizer: (120, length_payloads, {"train": {}, "validation": {}}),
    )
    monkeypatch.setattr(cloud_state, "_probe_formal_parquet", lambda *args, **kwargs: None)

    def downloader(**kwargs):
        calls["download"].append(kwargs)
        return str(source_files[(kwargs["repo_id"], kwargs["filename"])])

    def builder(**kwargs):
        destination = Path(kwargs["output_dir"])
        destination.mkdir(parents=True)
        (destination / "placeholder").write_text("bundle", encoding="ascii")
        manifests[destination.resolve()] = _fake_bundle_manifest(kwargs)
        calls["build"].append(destination.resolve())
        calls["build_kwargs"].append(kwargs.copy())

    def validator(path, **kwargs):
        resolved = Path(path).resolve()
        calls["validate"].append((resolved, kwargs.copy()))
        return manifests[resolved]

    result = cloud_state.build_data(
        manifest_path,
        cache_dir,
        data_root,
        downloader=downloader,
        tokenizer_loader=lambda name, revision: _whitespace_encoder,
        builder=builder,
        validator=validator,
        tracked_manifest_path=manifest_path,
    )
    assert len(calls["build"]) == 11
    assert len(calls["validate"]) == 11
    assert all(
        kwargs == {
            "replay_source_curation": manifests[path]["profile"] == "formal"
        }
        for path, kwargs in calls["validate"]
    )
    assert len(calls["download"]) == 3
    assert all(call["local_files_only"] is True for call in calls["download"])
    assert all(call["repo_id"] != "owner/formal" for call in calls["download"])
    assert all(
        call["source_format"] == ".jsonl"
        for call in calls["build_kwargs"]
        if call["profile"] == "formal"
    )
    formal_train = Path(result["bundles"]["formal"]["train"]["path"]).resolve()
    formal_validation = Path(
        result["bundles"]["formal"]["validation"]["path"]
    ).resolve()
    assert manifests[formal_train]["source"]["path"] == str(
        source_files[("owner/hotpot", "hotpotqa/train.jsonl")].resolve()
    )
    assert manifests[formal_validation]["source"]["path"] == str(
        source_files[("owner/hotpot", "hotpotqa/dev.jsonl")].resolve()
    )
    assert result["bundles"]["gates"]["g0"]["train"]["action"] == "built"
    assert result["gate_source"]["repeat_count"] == 64
    assert result["length_stress_source"]["non_scientific"] is True
    assert result["length_stress_source"]["generator"] == (
        cloud_state._LENGTH_STRESS_SOURCE_REVISION_PREFIX
    )
    for split in ("train", "validation"):
        record = result["bundles"]["capacity"]["length_stress"][split]
        assert record["profile"] == "fixture"
        assert manifests[Path(record["path"]).resolve()]["contract"] == (
            cloud_state._LENGTH_STRESS_CONTRACT.to_dict()
        )
        assert record["tokenizer_name"] == "owner/qwen2"

    second = cloud_state.build_data(
        manifest_path,
        cache_dir,
        data_root,
        downloader=downloader,
        tokenizer_loader=lambda name, revision: _whitespace_encoder,
        builder=builder,
        validator=validator,
        tracked_manifest_path=manifest_path,
    )
    assert len(calls["build"]) == 11
    assert second["bundles"]["formal"]["train"]["action"] == "verified"


def test_formal_bundle_identity_rejects_wrong_source_format(tmp_path):
    source = tmp_path / "train.jsonl"
    source.write_text("{}\n", encoding="ascii")
    spec = next(
        item for item in cloud_state._BUNDLE_SPECS if item.keys == ("formal", "train")
    )
    kwargs = {
        "dataset": spec.dataset,
        "input_path": source,
        "mode": spec.mode,
        "output_dir": tmp_path / "bundle",
        "profile": spec.profile,
        "seed": cloud_state.SEED,
        "source_format": ".jsonl",
        "source_revision": "source-revision",
        "split": spec.source_split,
        "tokenizer_name": "tokenizer",
        "tokenizer_revision": "tokenizer-revision",
    }
    manifest = _fake_bundle_manifest(kwargs)
    manifest["source"]["format"] = "json"

    with pytest.raises(cloud_state.CloudStateError, match="source format mismatch"):
        cloud_state._validate_bundle_identity(
            manifest,
            spec,
            source_path=source.resolve(),
            source_revision="source-revision",
            tokenizer_name="tokenizer",
            tokenizer_revision="tokenizer-revision",
        )


def test_build_data_cli_atomically_writes_same_summary_it_prints(tmp_path, monkeypatch, capsys):
    summary = {
        "bundles": {"formal": {}},
        "schema_version": cloud_state.SCHEMA_VERSION,
        "status": "complete",
    }
    monkeypatch.setattr(cloud_state, "build_data", lambda *args: summary)
    output = tmp_path / "state" / "data-summary.json"
    result = cloud_state.main(
        [
            "build-data",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--data-root",
            str(tmp_path / "data"),
            "--summary",
            str(output),
        ]
    )
    assert result == 0
    assert json.loads(output.read_text(encoding="ascii")) == summary
    assert json.loads(capsys.readouterr().out) == summary


def test_resolved_config_tree_requires_exact_33_ids_and_five_field_entries(
    tmp_path, monkeypatch
):
    pytest.importorskip("hydra", reason="hydra-core is required to resolve configs")
    from scripts.cloud import resolve_configs

    data_root = tmp_path / "data"
    data_root.mkdir()

    def bundle(root, relative):
        path = root / relative
        path.mkdir(parents=True, exist_ok=True)
        return path, {
            "manifest_sha256": hashlib.sha256(relative.encode("ascii")).hexdigest()
        }

    monkeypatch.setattr(resolve_configs, "_bundle", bundle)
    config_root = tmp_path / "resolved"
    result = resolve_configs.compose_all(data_root, config_root)
    records = cloud_state._tree_records(
        config_root,
        expected_names=cloud_state._CONFIG_TREE_NAMES,
    )

    assert len(result["configs"]) == 33
    assert set(result["configs"]) == cloud_state._CONFIG_IDS
    assert all(len(entry) == 5 for entry in result["configs"].values())
    cloud_state._validate_resolved_config_tree(
        config_root,
        records,
        data_root.resolve(),
    )

    index_path = config_root / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["configs"]["g0_qwen35_08b"]["unexpected"] = True
    _write_json(index_path, index)
    records = cloud_state._tree_records(
        config_root,
        expected_names=cloud_state._CONFIG_TREE_NAMES,
    )
    with pytest.raises(cloud_state.CloudStateError, match="keys mismatch"):
        cloud_state._validate_resolved_config_tree(
            config_root,
            records,
            data_root.resolve(),
        )


def _minimal_handoff(tmp_path: Path):
    files = {}
    for name in ("asset-manifest", "asset-report", "environment", "pip-freeze"):
        path = tmp_path / name
        path.write_text("{}\n", encoding="ascii")
        files[name] = {"path": str(path), "sha256": SHA256, "size": 3}
    unsigned = {
        "asset_manifest": {"file": files["asset-manifest"], "manifest_sha256": SHA256},
        "asset_report": {"file": files["asset-report"]},
        "bundles": {},
        "config_files": {},
        "config_root": str(tmp_path),
        "config_tree_sha256": SHA256,
        "data_root": str(tmp_path),
        "environment_lock": {"file": files["environment"]},
        "environment_lock_sha256": "e" * 64,
        "experiment_profile_id": cloud_state.EXPERIMENT_PROFILE_ID,
        "git_commit": COMMIT,
        "kernel_source_root": str(tmp_path),
        "kernel_sources": {},
        "pip_freeze": files["pip-freeze"],
        "persist_root": str(tmp_path),
        "schema_version": cloud_state.SCHEMA_VERSION,
        "status": "cpu_ready",
    }
    sealed = dict(unsigned)
    sealed["handoff_sha256"] = cloud_state._canonical_sha256(unsigned)
    return sealed


def _patch_handoff_dependencies(monkeypatch, tmp_path):
    runtime_manifest = {"manifest_sha256": SHA256}
    monkeypatch.setattr(cloud_state, "_git_repository_record", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        cloud_state,
        "_verify_file_record",
        lambda record, path: Path(record["path"]),
    )
    monkeypatch.setattr(cloud_state, "load_asset_manifest", lambda path: runtime_manifest)
    monkeypatch.setattr(cloud_state, "_require_resolved_manifest", lambda manifest: None)
    monkeypatch.setattr(cloud_state, "_validate_runtime_against_tracked", lambda *args: None)
    monkeypatch.setattr(cloud_state, "_validate_asset_report", lambda *args: None)
    monkeypatch.setattr(
        cloud_state,
        "load_environment_lock",
        lambda *args, **kwargs: {"kernels": [], "lock_sha256": "e" * 64},
    )
    monkeypatch.setattr(cloud_state, "_verify_tree_records", lambda *args, **kwargs: None)
    monkeypatch.setattr(cloud_state, "_validate_resolved_config_tree", lambda *args: None)
    monkeypatch.setattr(cloud_state, "_iter_bundle_records", lambda bundles: [])
    monkeypatch.setattr(cloud_state, "_TRACKED_ASSET_MANIFEST", tmp_path / "tracked.json")


def test_verify_handoff_checks_self_hash_canonical_form_and_printable_scalar(
    tmp_path, monkeypatch
):
    _patch_handoff_dependencies(monkeypatch, tmp_path)
    sealed = _minimal_handoff(tmp_path)
    handoff_path = tmp_path / "handoff.json"
    cloud_state._atomic_write_json(handoff_path, sealed)

    handoff, value = cloud_state.verify_handoff(
        handoff_path,
        COMMIT,
        field="git_commit",
    )
    assert handoff["status"] == "cpu_ready"
    assert set(handoff) == HANDOFF_KEYS
    assert handoff["experiment_profile_id"] == cloud_state.EXPERIMENT_PROFILE_ID
    assert value == COMMIT
    with pytest.raises(cloud_state.CloudStateError, match="not a scalar"):
        cloud_state.verify_handoff(handoff_path, COMMIT, field="asset_manifest")
    with pytest.raises(cloud_state.CloudStateError, match="does not exist"):
        cloud_state.verify_handoff(handoff_path, COMMIT, field="missing.value")

    extra = dict(sealed)
    extra["unexpected"] = True
    cloud_state._atomic_write_json(handoff_path, extra)
    with pytest.raises(cloud_state.CloudStateError, match="keys mismatch"):
        cloud_state.verify_handoff(handoff_path, COMMIT)

    wrong_profile = dict(sealed)
    wrong_profile["experiment_profile_id"] = "legacy-4b-profile"
    unsigned = dict(wrong_profile)
    unsigned.pop("handoff_sha256")
    wrong_profile["handoff_sha256"] = cloud_state._canonical_sha256(unsigned)
    cloud_state._atomic_write_json(handoff_path, wrong_profile)
    with pytest.raises(cloud_state.CloudStateError, match="active 5090/2B"):
        cloud_state.verify_handoff(handoff_path, COMMIT)

    tampered = dict(sealed)
    tampered["git_commit"] = "d" * 40
    cloud_state._atomic_write_json(handoff_path, tampered)
    with pytest.raises(cloud_state.CloudStateError, match="self-hash mismatch"):
        cloud_state.verify_handoff(handoff_path, COMMIT)

    cloud_state._atomic_write_json(handoff_path, sealed)
    handoff_path.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n", encoding="ascii")
    with pytest.raises(cloud_state.CloudStateError, match="not canonical"):
        cloud_state.verify_handoff(handoff_path, COMMIT)


def test_file_record_detects_byte_tampering_even_when_size_is_unchanged(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"original")
    record = cloud_state._file_record(path)
    path.write_bytes(b"tampered")
    with pytest.raises(cloud_state.CloudStateError, match="hash changed"):
        cloud_state._verify_file_record(record, "artifact")


def test_asset_report_is_bound_to_every_file_digest(tmp_path):
    cached = tmp_path / "dev.jsonl"
    cached.write_bytes(b"123456789")
    digest = hashlib.sha256(cached.read_bytes()).hexdigest()
    manifest = _asset_manifest(
        [
            _resolved_asset(
                "dataset-source",
                "dataset_source",
                "owner/source",
                [{"path": "dev.jsonl", "sha256": digest, "size": 9}],
            )
        ]
    )
    report = {
        "assets": [
            {
                "asset_id": "dataset-source",
                "files": [
                    {
                        "attempts": 1,
                        "cached_path": str(cached.resolve()),
                        "integrity": {"algorithm": "sha256", "digest": digest, "size": 9},
                        "path": "dev.jsonl",
                        "status": "verified",
                    }
                ],
                "repo_id": "owner/source",
                "revision": REVISION,
                "status": "complete",
            }
        ],
        "manifest_sha256": manifest["manifest_sha256"],
        "mode": "download_and_verify",
        "schema_version": 1,
        "status": "complete",
    }
    cloud_state._validate_asset_report(report, manifest)
    report["assets"][0]["files"][0]["integrity"]["digest"] = "f" * 64
    with pytest.raises(cloud_state.CloudStateError, match="integrity evidence mismatch"):
        cloud_state._validate_asset_report(report, manifest)


def test_formal_parquet_probe_rejects_actual_flattened_upstream_schema(tmp_path):
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "flat.parquet"
    parquet.write_table(
        pyarrow.Table.from_pylist(
            [
                {
                    "data_source": "hotpotqa",
                    "prompt": [{"content": "question?", "role": "user"}],
                    "context": "already concatenated and no longer auditable",
                    "reward_model": {
                        "ground_truth": ["answer"],
                        "style": "rule",
                    },
                    "extra_info": {
                        "index": 0,
                        "num_docs": 200,
                        "question": "question?",
                    },
                }
            ]
        ),
        path,
    )

    with pytest.raises(cloud_state.CloudStateError, match="lacks structured"):
        cloud_state._probe_formal_parquet(path, dataset="hotpotqa")


def test_formal_bundles_rebuild_from_structured_raw_hotpotqa_sources():
    formal_specs = {
        spec.keys: (spec.source_asset, spec.source_file)
        for spec in cloud_state._BUNDLE_SPECS
        if spec.keys in {("formal", "train"), ("formal", "validation")}
    }

    assert formal_specs == {
        ("formal", "train"): ("hotpotqa-source", "hotpotqa/train.jsonl"),
        ("formal", "validation"): ("hotpotqa-source", "hotpotqa/dev.jsonl"),
    }
    assert "byted-hotpotqa-formal" in cloud_state._ACTIVE_ASSET_IDS


def test_publish_handoff_writes_canonical_self_hashed_json(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    repository.mkdir()
    asset_manifest = (tmp_path / "runtime-assets.json")
    asset_manifest.write_text("{}\n", encoding="ascii")
    asset_report = tmp_path / "asset-report.json"
    asset_report.write_text("{}\n", encoding="ascii")
    pip_freeze = tmp_path / "pip-freeze.txt"
    pip_freeze.write_text("package==1.0\n", encoding="ascii")
    data_root = tmp_path / "data"
    data_root.mkdir()
    kernel_root = tmp_path / "kernels"
    kernel_root.mkdir()
    config_root = tmp_path / "configs"
    config_root.mkdir()
    tracked_manifest = tmp_path / "tracked-assets.json"
    tracked_manifest.write_text("{}\n", encoding="ascii")
    environment_lock = tmp_path / "environment-lock.json"
    environment_lock.write_text("{}\n", encoding="ascii")
    runtime = {"manifest_sha256": SHA256}

    monkeypatch.setattr(cloud_state, "_git_repository_record", lambda *args, **kwargs: {})
    monkeypatch.setattr(cloud_state, "load_asset_manifest", lambda path: runtime)
    monkeypatch.setattr(cloud_state, "_validate_runtime_against_tracked", lambda *args: None)
    monkeypatch.setattr(cloud_state, "_validate_asset_report", lambda *args: None)
    monkeypatch.setattr(
        cloud_state,
        "load_environment_lock",
        lambda *args, **kwargs: {"kernels": [], "lock_sha256": "e" * 64},
    )
    monkeypatch.setattr(cloud_state, "_load_and_record_bundles", lambda *args: {})
    monkeypatch.setattr(cloud_state, "_kernel_records", lambda *args: {})
    monkeypatch.setattr(
        cloud_state,
        "_tree_records",
        lambda *args, **kwargs: {name: {"sha256": SHA256, "size": 1} for name in cloud_state._CONFIG_TREE_NAMES},
    )
    monkeypatch.setattr(cloud_state, "_validate_resolved_config_tree", lambda *args: None)

    output = tmp_path / "state" / "handoff.json"
    handoff = cloud_state.publish_handoff(
        output,
        COMMIT,
        asset_manifest,
        asset_report,
        pip_freeze,
        data_root,
        kernel_root,
        config_root,
        tmp_path,
        repository_root=repository,
        tracked_manifest_path=tracked_manifest,
        environment_lock_path=environment_lock,
    )

    unsigned = dict(handoff)
    digest = unsigned.pop("handoff_sha256")
    assert digest == cloud_state._canonical_sha256(unsigned)
    assert set(handoff) == HANDOFF_KEYS
    assert handoff["status"] == "cpu_ready"
    assert handoff["data_root"] == str(data_root.resolve())
    assert output.read_bytes() == cloud_state._canonical_bytes(handoff) + b"\n"


def test_main_returns_two_with_structured_error(tmp_path, capsys):
    manifest = _write_json(tmp_path / "assets.json", _unresolved_manifest())
    result = cloud_state.main(
        [
            "resolve-assets",
            "--manifest",
            str(manifest),
            "--output",
            str(manifest),
        ]
    )
    assert result == 2
    error = json.loads(capsys.readouterr().out)
    assert error["status"] == "blocked"
