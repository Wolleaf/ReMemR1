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


def test_g1_eval_fixture_builds_two_recurrent_records(tmp_path):
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
        _resolved_asset(
            "qwen35-4b-model-tokenizer",
            "model_and_tokenizer",
            "owner/qwen4",
            [{"path": "tokenizer.json", "sha256": "0" * 64, "size": 0}],
            "4" * 40,
        ),
    ]
    for asset_id, repo_id, filenames, kind in (
        (
            "byted-hotpotqa-formal",
            "owner/formal",
            ("hotpotqa_train_32k.parquet", "hotpotqa_dev.parquet"),
            "formal_training_source",
        ),
        ("hotpotqa-source", "owner/hotpot", ("hotpotqa/dev.jsonl",), "dataset_source"),
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
        contract = cloud_state.TrainManifestContract().to_dict()
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
    return {
        "contract": contract,
        "dataset": kwargs["dataset"],
        "manifest_sha256": hashlib.sha256(str(kwargs["output_dir"]).encode()).hexdigest(),
        "mode": kwargs["mode"],
        "profile": kwargs["profile"],
        "seed": kwargs["seed"],
        "source": {
            "path": str(source),
            "revision": kwargs["source_revision"],
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "split": kwargs["split"],
        },
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
    calls = {"build": [], "download": []}
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

    def validator(path):
        return manifests[Path(path).resolve()]

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
    assert len(calls["build"]) == 9
    assert len(calls["download"]) == 4
    assert all(call["local_files_only"] is True for call in calls["download"])
    assert result["bundles"]["gates"]["g0"]["train"]["action"] == "built"
    assert result["gate_source"]["repeat_count"] == 64

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
    assert len(calls["build"]) == 9
    assert second["bundles"]["formal"]["train"]["action"] == "verified"


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
    assert value == COMMIT
    with pytest.raises(cloud_state.CloudStateError, match="not a scalar"):
        cloud_state.verify_handoff(handoff_path, COMMIT, field="asset_manifest")
    with pytest.raises(cloud_state.CloudStateError, match="does not exist"):
        cloud_state.verify_handoff(handoff_path, COMMIT, field="missing.value")

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


def test_formal_parquet_probe_rejects_flat_context(tmp_path):
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "flat.parquet"
    parquet.write_table(
        pyarrow.Table.from_pylist(
            [
                {
                    "_id": "flat-1",
                    "answers": ["answer"],
                    "context": "already concatenated and no longer auditable",
                    "question": "question?",
                    "supporting_facts": [
                        {"title": "missing title", "sent_id": 0}
                    ],
                }
            ]
        ),
        path,
    )

    with pytest.raises(cloud_state.CloudStateError, match="lacks structured"):
        cloud_state._probe_formal_parquet(path, dataset="hotpotqa")


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
