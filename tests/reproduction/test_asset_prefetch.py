import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from scripts.reproduction import prefetch_assets as assets


ROOT = Path(__file__).resolve().parents[2]
ASSET_MANIFEST_PATH = ROOT / "environment" / "reproduction-assets.json"
QWEN35_SPEC = importlib.util.spec_from_file_location(
    "qwen35_asset_contract_under_test",
    ROOT / "verl" / "models" / "qwen35.py",
)
assert QWEN35_SPEC is not None and QWEN35_SPEC.loader is not None
qwen35 = importlib.util.module_from_spec(QWEN35_SPEC)
sys.modules[QWEN35_SPEC.name] = qwen35
QWEN35_SPEC.loader.exec_module(qwen35)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob_sha1(data: bytes) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {len(data)}\0".encode("ascii"))
    digest.update(data)
    return digest.hexdigest()


def _manifest_for_files(tmp_path, file_specs, *, kind="dataset_source"):
    payload = {
        "assets": [
            {
                "asset_id": "fixture-asset",
                "files": file_specs,
                "kind": kind,
                "repo_id": (
                    "Qwen/Qwen3.5-2B"
                    if kind == "model_and_tokenizer"
                    else "owner/dataset"
                ),
                "repo_type": "model" if kind == "model_and_tokenizer" else "dataset",
                "revision": (
                    qwen35.QWEN35_PINNED_REVISIONS["Qwen/Qwen3.5-2B"]
                    if kind == "model_and_tokenizer"
                    else "1" * 40
                ),
            }
        ],
        "schema_version": 1,
    }
    return assets.seal_manifest(payload)


def test_repository_asset_manifest_pins_models_and_source_hashes():
    manifest = assets.load_asset_manifest(ASSET_MANIFEST_PATH)
    indexed = {asset["asset_id"]: asset for asset in manifest["assets"]}

    assert set(indexed) == {
        "2wikimultihopqa-source",
        "byted-hotpotqa-formal",
        "hotpotqa-source",
        "qwen35-08b-model-tokenizer",
        "qwen35-2b-model-tokenizer",
    }
    assert all("4b" not in asset_id.casefold() for asset_id in indexed)

    assert {
        asset_id: indexed[asset_id]["revision"]
        for asset_id in (
            "qwen35-08b-model-tokenizer",
            "qwen35-2b-model-tokenizer",
        )
    } == {
        "qwen35-08b-model-tokenizer": "2fc06364715b967f1860aea9cf38778875588b17",
        "qwen35-2b-model-tokenizer": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    }
    assert "qwen35-4b-model-tokenizer" not in indexed
    assert indexed["hotpotqa-source"]["revision"] == "bcafb8dd07d453be3cbeeeb3f78be1841bddf92c"
    assert indexed["2wikimultihopqa-source"]["revision"] == "bcafb8dd07d453be3cbeeeb3f78be1841bddf92c"
    assert {
        item["path"]: item["sha256"] for item in indexed["hotpotqa-source"]["files"]
    } == {
        "hotpotqa/dev.jsonl": "434ec155867019396312f3d466ce3406c71fe8a2917f49f63bb646ec5ad2ff52",
        "hotpotqa/train.jsonl": "a81274abafa899ec0ee073102edbe6bb694a8a1174201b4e43e2bc6c98964d1a",
    }


def test_formal_byted_source_is_pinned_but_fail_closed_until_lfs_metadata_is_available():
    manifest = assets.load_asset_manifest(ASSET_MANIFEST_PATH)
    formal = next(asset for asset in manifest["assets"] if asset["asset_id"] == "byted-hotpotqa-formal")
    calls = []

    report = assets.prefetch_assets(
        manifest,
        asset_ids=[formal["asset_id"]],
        downloader=lambda **kwargs: calls.append(kwargs),
    )

    assert formal["repo_id"] == "BytedTsinghua-SIA/hotpotqa"
    assert formal["revision"] == "27275ff4fee67ac0acb6478e405e7ac07efbdc1a"
    assert formal["training_gate"] == "BLOCKED"
    assert [item["path"] for item in formal["files"]] == [
        "hotpotqa_train_32k.parquet",
        "hotpotqa_dev.parquet",
    ]
    assert all(item["metadata_status"] == "METADATA_UNAVAILABLE" for item in formal["files"])
    assert "get_paths_info" in formal["metadata_resolution"]["command"]
    assert calls == []
    assert report["status"] == "incomplete"
    assert {item["status"] for item in report["assets"][0]["files"]} == {
        "metadata_unavailable"
    }


def test_manifest_self_hash_and_paths_fail_closed_on_tampering():
    manifest = assets.load_asset_manifest(ASSET_MANIFEST_PATH)
    manifest["assets"][0]["revision"] = "0" * 40
    with pytest.raises(assets.AssetManifestError, match="manifest_sha256"):
        assets.validate_asset_manifest(manifest)

    invalid = assets.seal_manifest(
        {
            "assets": [
                {
                    "asset_id": "bad-path",
                    "files": [{"path": "../escape", "sha256": "a" * 64, "size": 1}],
                    "kind": "dataset_source",
                    "repo_id": "owner/repo",
                    "repo_type": "dataset",
                    "revision": "1" * 40,
                }
            ],
            "schema_version": 1,
        }
    )
    with pytest.raises(assets.AssetManifestError, match="safe POSIX-relative"):
        assets.validate_asset_manifest(invalid)


def test_offline_cache_verification_checks_sha256_and_git_blob(tmp_path):
    sha_data = b"large-file-fixture"
    git_data = b"small-git-file"
    sha_path = tmp_path / "sha.bin"
    git_path = tmp_path / "git.txt"
    sha_path.write_bytes(sha_data)
    git_path.write_bytes(git_data)
    manifest = _manifest_for_files(
        tmp_path,
        [
            {"path": "sha.bin", "sha256": _sha256(sha_data), "size": len(sha_data)},
            {"git_blob_sha1": _git_blob_sha1(git_data), "path": "git.txt", "size": len(git_data)},
        ],
    )
    paths = {"sha.bin": sha_path, "git.txt": git_path}
    calls = []

    def downloader(**kwargs):
        calls.append(kwargs)
        return str(paths[kwargs["filename"]])

    report = assets.prefetch_assets(manifest, downloader=downloader, timeout=7)

    assert report["status"] == "complete"
    assert report["mode"] == "offline_verify"
    assert all(call["local_files_only"] is True for call in calls)
    assert all(call["etag_timeout"] == 7 for call in calls)
    algorithms = [item["integrity"]["algorithm"] for item in report["assets"][0]["files"]]
    assert algorithms == ["sha256", "git_blob_sha1"]


def test_download_mode_retries_with_timeout_and_repairs_cache(tmp_path):
    data = b"eventually-valid"
    cached = tmp_path / "asset.bin"
    cached.write_bytes(data)
    manifest = _manifest_for_files(
        tmp_path,
        [{"path": "asset.bin", "sha256": _sha256(data), "size": len(data)}],
    )
    calls = []
    sleeps = []

    def flaky_downloader(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise TimeoutError("transient Hub timeout")
        return str(cached)

    report = assets.prefetch_assets(
        manifest,
        download=True,
        retries=3,
        timeout=11,
        downloader=flaky_downloader,
        sleep=sleeps.append,
    )

    assert report["status"] == "complete"
    assert len(calls) == 3
    assert [call["force_download"] for call in calls] == [False, True, True]
    assert all(call["local_files_only"] is False for call in calls)
    assert all(call["etag_timeout"] == 11 for call in calls)
    assert sleeps == [1, 2]


def test_materialized_qwen_snapshot_manifest_is_loader_compatible(tmp_path):
    config_data = b"{}"
    weight_data = b"fake-weight-bytes"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    config_path = source_dir / "config.json"
    weight_path = source_dir / "model.safetensors-00001-of-00001.safetensors"
    config_path.write_bytes(config_data)
    weight_path.write_bytes(weight_data)
    manifest = _manifest_for_files(
        tmp_path,
        [
            {"path": "config.json", "sha256": _sha256(config_data), "size": len(config_data)},
            {
                "path": weight_path.name,
                "sha256": _sha256(weight_data),
                "size": len(weight_data),
            },
        ],
        kind="model_and_tokenizer",
    )
    source_paths = {"config.json": config_path, weight_path.name: weight_path}

    report = assets.prefetch_assets(
        manifest,
        downloader=lambda **kwargs: str(source_paths[kwargs["filename"]]),
        materialize_dir=tmp_path / "materialized",
    )

    snapshot = Path(report["assets"][0]["materialized_snapshot"]["path"])
    loader_evidence = qwen35.validate_qwen35_revision_source(
        snapshot,
        qwen35.QWEN35_PINNED_REVISIONS["Qwen/Qwen3.5-2B"],
    )
    assert report["status"] == "complete"
    assert loader_evidence.kind == "verified_snapshot_manifest"
    assert loader_evidence.manifest_sha256 == report["assets"][0]["materialized_snapshot"][
        "manifest_sha256"
    ]


def test_json_report_write_is_atomic_and_round_trips(tmp_path):
    report_path = tmp_path / "nested" / "assets.json"
    report = {"schema_version": 1, "status": "complete"}
    assets.write_json_report(report_path, report)
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert list(report_path.parent.glob("*.tmp")) == []
