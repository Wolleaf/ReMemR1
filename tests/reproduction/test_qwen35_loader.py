import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "verl" / "models" / "qwen35.py"
MODULE_SPEC = importlib.util.spec_from_file_location("qwen35_loader_under_test", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
qwen35 = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = qwen35
MODULE_SPEC.loader.exec_module(qwen35)


QWEN35_4B_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def _hf_snapshot_path(tmp_path, revision=QWEN35_4B_REVISION):
    snapshot_path = (
        tmp_path
        / "models--Qwen--Qwen3.5-4B"
        / "snapshots"
        / revision
    )
    snapshot_path.mkdir(parents=True)
    return snapshot_path


def _write_snapshot_manifest(snapshot_path, *, revision=QWEN35_4B_REVISION):
    files = {}
    for path in sorted(snapshot_path.rglob("*")):
        if path.is_file() and path.name != qwen35.QWEN35_SNAPSHOT_MANIFEST_FILENAME:
            content = path.read_bytes()
            files[path.relative_to(snapshot_path).as_posix()] = {
                "sha256": hashlib.sha256(content).hexdigest(),
                "size": len(content),
            }
    unsigned = {
        "schema_version": qwen35.QWEN35_SNAPSHOT_MANIFEST_SCHEMA_VERSION,
        "model_id": "Qwen/Qwen3.5-4B",
        "revision": revision,
        "files": files,
    }
    manifest_sha256 = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    payload = {**unsigned, "manifest_sha256": manifest_sha256}
    manifest_path = snapshot_path / qwen35.QWEN35_SNAPSHOT_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path, payload


def _conditional_config():
    return SimpleNamespace(
        model_type="qwen3_5",
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config=SimpleNamespace(model_type="qwen3_5_text"),
    )


class _FakeAutoConfig:
    config = None
    calls = []

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        cls.calls.append((model_name_or_path, kwargs))
        return cls.config


class _FakeAutoModelForCausalLM:
    calls = []
    loading_info = {
        "missing_keys": [],
        "unexpected_keys": [],
        "mismatched_keys": [],
        "error_msgs": [],
    }

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        cls.calls.append((model_name_or_path, kwargs))
        text_config = kwargs["config"].text_config
        return SimpleNamespace(config=text_config), dict(cls.loading_info)


def _install_fake_transformers(monkeypatch, config, loading_info=None):
    _FakeAutoConfig.config = config
    _FakeAutoConfig.calls = []
    _FakeAutoModelForCausalLM.calls = []
    _FakeAutoModelForCausalLM.loading_info = loading_info or {
        "missing_keys": [],
        "unexpected_keys": [],
        "mismatched_keys": [],
        "error_msgs": [],
    }
    fake_transformers = SimpleNamespace(
        AutoConfig=_FakeAutoConfig,
        AutoModelForCausalLM=_FakeAutoModelForCausalLM,
        __version__="5.14.0",
    )
    monkeypatch.setattr(qwen35.importlib, "import_module", lambda name: fake_transformers)


def test_module_has_no_transformers_or_removed_vision_auto_model_at_import_time():
    source_path = Path(qwen35.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]

    assert all(
        not (
            (isinstance(node, ast.Import) and any(alias.name == "transformers" for alias in node.names))
            or (isinstance(node, ast.ImportFrom) and node.module == "transformers")
        )
        for node in imports
    )
    assert "AutoModelForVision2Seq" not in source_path.read_text(encoding="utf-8")


def test_native_causal_lm_receives_full_conditional_config_and_explicit_options(monkeypatch):
    config = _conditional_config()
    loading_info = {
        "missing_keys": [],
        "unexpected_keys": ["model.visual.blocks.0.weight", "mtp.layers.0.weight"],
        "mismatched_keys": [],
        "error_msgs": [],
    }
    _install_fake_transformers(monkeypatch, config, loading_info)
    dtype = SimpleNamespace(__str__=lambda self: "torch.bfloat16")

    result = qwen35.load_qwen35_text_model(
        "Qwen/Qwen3.5-4B",
        revision=QWEN35_4B_REVISION,
        attn_implementation="sdpa",
        dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    config_path, config_kwargs = _FakeAutoConfig.calls[0]
    assert config_path == "Qwen/Qwen3.5-4B"
    assert config_kwargs == {
        "revision": QWEN35_4B_REVISION,
        "trust_remote_code": False,
        "local_files_only": True,
    }

    model_path, model_kwargs = _FakeAutoModelForCausalLM.calls[0]
    assert model_path == "Qwen/Qwen3.5-4B"
    assert model_kwargs["config"] is config
    assert model_kwargs["revision"] == QWEN35_4B_REVISION
    assert model_kwargs["attn_implementation"] == "sdpa"
    assert model_kwargs["dtype"] is dtype
    assert model_kwargs["trust_remote_code"] is False
    assert model_kwargs["output_loading_info"] is True
    assert model_kwargs["local_files_only"] is True
    assert model_kwargs["low_cpu_mem_usage"] is True

    assert result.metadata.conditional_checkpoint is True
    assert result.metadata.config_passed_to_auto_model == "full_conditional_config"
    assert result.metadata.text_config_path == "text_config"
    assert result.metadata.checkpoint_text_prefix == "model.language_model"
    assert result.metadata.target_text_prefix == "model"
    assert result.metadata.native_prefix_conversion == (
        'PrefixChange(prefix_to_remove="language_model", model_prefix="model")'
    )
    assert result.metadata.target_architecture == "Qwen3_5ForCausalLM"
    assert result.metadata.transformers_version == "5.14.0"
    assert result.metadata.revision_evidence.kind == "hub_revision_request"
    assert result.metadata.revision_evidence.model_id == "Qwen/Qwen3.5-4B"
    assert result.metadata.revision_evidence.revision == QWEN35_4B_REVISION
    assert result.loading_report.allowed_unexpected_keys == (
        "model.visual.blocks.0.weight",
        "mtp.layers.0.weight",
    )


def test_text_config_is_recognized_without_conditional_prefix_mapping(monkeypatch, tmp_path):
    config = SimpleNamespace(
        model_type="qwen3_5_text",
        architectures=["Qwen3_5ForCausalLM"],
    )
    _install_fake_transformers(monkeypatch, config)

    class TextFakeAutoModel(_FakeAutoModelForCausalLM):
        @classmethod
        def from_pretrained(cls, model_name_or_path, **kwargs):
            cls.calls.append((model_name_or_path, kwargs))
            return SimpleNamespace(config=kwargs["config"]), dict(cls.loading_info)

    fake_transformers = SimpleNamespace(
        AutoConfig=_FakeAutoConfig,
        AutoModelForCausalLM=TextFakeAutoModel,
        __version__="5.14.0",
    )
    TextFakeAutoModel.calls = []
    monkeypatch.setattr(qwen35.importlib, "import_module", lambda name: fake_transformers)

    snapshot_path = _hf_snapshot_path(tmp_path)
    result = qwen35.load_qwen35_text_model(
        snapshot_path,
        revision=QWEN35_4B_REVISION,
        attn_implementation="eager",
        dtype="auto",
    )

    assert result.metadata.conditional_checkpoint is False
    assert result.metadata.config_passed_to_auto_model == "text_config"
    assert result.metadata.text_config_path == "$"
    assert result.metadata.checkpoint_text_prefix == "model"
    assert result.metadata.native_prefix_conversion == "identity"
    assert result.metadata.revision_evidence.kind == "hf_cache_snapshot_path"
    assert result.metadata.revision_evidence.model_id == "Qwen/Qwen3.5-4B"


@pytest.mark.parametrize(
    "key",
    [
        "model.language_model.layers.0.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.visualizer.weight",
        "mtp_extra.weight",
    ],
)
def test_unknown_or_unmapped_unexpected_key_fails(key):
    with pytest.raises(qwen35.Qwen35StateDictError, match="unexpected keys"):
        qwen35.validate_qwen35_loading_info(
            {
                "missing_keys": [],
                "unexpected_keys": [key],
                "mismatched_keys": [],
                "error_msgs": [],
            }
        )


def test_any_missing_text_key_fails():
    with pytest.raises(qwen35.Qwen35StateDictError, match="missing keys"):
        qwen35.validate_qwen35_loading_info(
            {
                "missing_keys": ["model.layers.0.self_attn.q_proj.weight"],
                "unexpected_keys": [],
            }
        )


def test_only_exact_visual_and_mtp_namespaces_are_allowed():
    report = qwen35.validate_qwen35_loading_info(
        {
            "missing_keys": [],
            "unexpected_keys": [
                "model.visual.patch_embed.proj.weight",
                "mtp.fc.weight",
                "mtp.layers.0.mlp.down_proj.weight",
            ],
        }
    )

    assert report.allowed_unexpected_keys == (
        "model.visual.patch_embed.proj.weight",
        "mtp.fc.weight",
        "mtp.layers.0.mlp.down_proj.weight",
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mismatched_keys", [("model.norm.weight", (2,), (3,))], "mismatched keys"),
        ("error_msgs", ["safetensors checksum failed"], "loader errors"),
    ],
)
def test_mismatch_and_loader_errors_always_fail(field, value, message):
    loading_info = {
        "missing_keys": [],
        "unexpected_keys": [],
        "mismatched_keys": [],
        "error_msgs": [],
    }
    loading_info[field] = value

    with pytest.raises(qwen35.Qwen35StateDictError, match=message):
        qwen35.validate_qwen35_loading_info(loading_info)


def test_mapping_metadata_is_json_serializable_and_savable(monkeypatch, tmp_path):
    config = _conditional_config()
    _install_fake_transformers(monkeypatch, config)
    result = qwen35.load_qwen35_text_model(
        "Qwen/Qwen3.5-4B",
        revision=QWEN35_4B_REVISION,
        attn_implementation="sdpa",
        dtype="bfloat16",
    )

    output_path = result.save_mapping_metadata(tmp_path / "mapping.json")
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved["schema_version"] == 2
    assert saved["model_name_or_path"] == "Qwen/Qwen3.5-4B"
    assert saved["revision"] == QWEN35_4B_REVISION
    assert saved["revision_evidence"] == {
        "kind": "hub_revision_request",
        "manifest_path": None,
        "manifest_sha256": None,
        "model_id": "Qwen/Qwen3.5-4B",
        "resolved_snapshot_path": None,
        "revision": QWEN35_4B_REVISION,
        "verified_file_count": 0,
        "verified_total_bytes": 0,
    }
    assert saved["dtype"] == "bfloat16"
    assert saved["attention_implementation"] == "sdpa"
    assert saved["native_prefix_conversion"] == (
        'PrefixChange(prefix_to_remove="language_model", model_prefix="model")'
    )
    assert saved["strict_loading"] is True
    assert saved["allowed_missing_key_patterns"] == []
    assert saved["allowed_unexpected_key_patterns"] == [
        r"^model\.visual(?:\.|$)",
        r"^mtp(?:\.|$)",
    ]


@pytest.mark.parametrize("revision", [None, "", "main", "master", "latest"])
def test_revision_must_be_explicit_and_non_floating(monkeypatch, revision):
    _install_fake_transformers(monkeypatch, _conditional_config())
    with pytest.raises(ValueError, match="revision"):
        qwen35.load_qwen35_text_model(
            "Qwen/Qwen3.5-4B",
            revision=revision,
            attn_implementation="sdpa",
            dtype="bfloat16",
        )


def test_non_qwen35_config_fails_before_model_load(monkeypatch):
    config = SimpleNamespace(model_type="qwen2", architectures=["Qwen2ForCausalLM"])
    _install_fake_transformers(monkeypatch, config)

    with pytest.raises(qwen35.Qwen35ConfigError, match="qwen3_5"):
        qwen35.load_qwen35_text_model(
            "not-qwen35",
            revision="fixed-revision",
            attn_implementation="sdpa",
            dtype="bfloat16",
        )
    assert not _FakeAutoModelForCausalLM.calls


def test_official_model_id_requires_its_pinned_revision(monkeypatch):
    _install_fake_transformers(monkeypatch, _conditional_config())
    with pytest.raises(ValueError, match="reproduction-pinned"):
        qwen35.load_qwen35_text_model(
            "Qwen/Qwen3.5-4B",
            revision="wrong-but-non-floating-revision",
            attn_implementation="sdpa",
            dtype="bfloat16",
        )


def test_arbitrary_local_directory_cannot_claim_revision_from_config_metadata(tmp_path):
    local_path = tmp_path / "copied-model"
    local_path.mkdir()
    (local_path / "config.json").write_text(
        json.dumps({"_commit_hash": QWEN35_4B_REVISION}),
        encoding="utf-8",
    )

    with pytest.raises(qwen35.Qwen35RevisionError, match="cannot prove revision"):
        qwen35.validate_qwen35_revision_source(local_path, QWEN35_4B_REVISION)


def test_hf_cache_snapshot_path_proves_exact_pinned_commit_without_weights(tmp_path):
    snapshot_path = _hf_snapshot_path(tmp_path)

    evidence = qwen35.validate_qwen35_revision_source(
        snapshot_path,
        QWEN35_4B_REVISION,
    )

    assert evidence.kind == "hf_cache_snapshot_path"
    assert evidence.model_id == "Qwen/Qwen3.5-4B"
    assert evidence.revision == QWEN35_4B_REVISION
    assert Path(evidence.resolved_snapshot_path) == snapshot_path.resolve()
    assert evidence.manifest_path is None


def test_hf_cache_snapshot_path_rejects_requested_commit_mismatch(tmp_path):
    other_commit = "0" * 40
    snapshot_path = _hf_snapshot_path(tmp_path, revision=other_commit)

    with pytest.raises(qwen35.Qwen35RevisionError, match="does not match requested"):
        qwen35.validate_qwen35_revision_source(snapshot_path, QWEN35_4B_REVISION)


def test_hdfs_style_copy_is_accepted_only_with_fully_verified_manifest(monkeypatch, tmp_path):
    local_path = tmp_path / "hdfs-cache" / "Qwen3.5-4B"
    local_path.mkdir(parents=True)
    (local_path / "config.json").write_text("{}", encoding="utf-8")
    (local_path / "model.safetensors").write_bytes(b"fake-weights-for-contract-test")
    manifest_path, payload = _write_snapshot_manifest(local_path)
    _install_fake_transformers(monkeypatch, _conditional_config())

    result = qwen35.load_qwen35_text_model(
        local_path,
        revision=QWEN35_4B_REVISION,
        attn_implementation="sdpa",
        dtype="bfloat16",
    )

    evidence = result.metadata.revision_evidence
    assert evidence.kind == "verified_snapshot_manifest"
    assert evidence.model_id == "Qwen/Qwen3.5-4B"
    assert evidence.manifest_sha256 == payload["manifest_sha256"]
    assert Path(evidence.manifest_path) == manifest_path.resolve()
    assert evidence.verified_file_count == 2
    assert evidence.verified_total_bytes == sum(
        entry["size"] for entry in payload["files"].values()
    )
    assert _FakeAutoModelForCausalLM.calls[0][0] == str(local_path)


@pytest.mark.parametrize(
    "weight_name",
    (
        "model-00001-of-00002.safetensors",
        "model.safetensors-00001-of-00002.safetensors",
    ),
)
def test_snapshot_manifest_accepts_both_official_shard_filename_forms(tmp_path, weight_name):
    local_path = tmp_path / "exported-snapshot"
    local_path.mkdir()
    (local_path / "config.json").write_text("{}", encoding="utf-8")
    (local_path / weight_name).write_bytes(b"weights")
    _write_snapshot_manifest(local_path)

    evidence = qwen35.validate_qwen35_revision_source(local_path, QWEN35_4B_REVISION)

    assert evidence.kind == "verified_snapshot_manifest"
    assert evidence.verified_file_count == 2


def test_snapshot_manifest_rejects_arbitrary_weight_filename(tmp_path):
    local_path = tmp_path / "exported-snapshot"
    local_path.mkdir()
    (local_path / "config.json").write_text("{}", encoding="utf-8")
    (local_path / "weights-00001-of-00002.safetensors").write_bytes(b"weights")
    _write_snapshot_manifest(local_path)

    with pytest.raises(qwen35.Qwen35RevisionError, match="must include model.safetensors"):
        qwen35.validate_qwen35_revision_source(local_path, QWEN35_4B_REVISION)


def test_snapshot_manifest_rejects_tampered_weight_bytes(tmp_path):
    local_path = tmp_path / "exported-snapshot"
    local_path.mkdir()
    (local_path / "config.json").write_text("{}", encoding="utf-8")
    weight_path = local_path / "model.safetensors"
    weight_path.write_bytes(b"expected")
    _write_snapshot_manifest(local_path)
    weight_path.write_bytes(b"tampered")

    with pytest.raises(qwen35.Qwen35RevisionError, match="SHA-256 mismatch"):
        qwen35.validate_qwen35_revision_source(local_path, QWEN35_4B_REVISION)


def test_snapshot_manifest_rejects_self_hash_or_revision_relabel(tmp_path):
    local_path = tmp_path / "exported-snapshot"
    local_path.mkdir()
    (local_path / "config.json").write_text("{}", encoding="utf-8")
    (local_path / "model.safetensors").write_bytes(b"weights")
    manifest_path, payload = _write_snapshot_manifest(local_path)
    payload["manifest_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(qwen35.Qwen35RevisionError, match="self-hash mismatch"):
        qwen35.validate_qwen35_revision_source(local_path, QWEN35_4B_REVISION)

    _write_snapshot_manifest(local_path, revision="0" * 40)
    with pytest.raises(qwen35.Qwen35RevisionError, match="does not match requested"):
        qwen35.validate_qwen35_revision_source(local_path, QWEN35_4B_REVISION)


def test_snapshot_manifest_rejects_undeclared_local_file(tmp_path):
    local_path = tmp_path / "exported-snapshot"
    local_path.mkdir()
    (local_path / "config.json").write_text("{}", encoding="utf-8")
    (local_path / "model.safetensors").write_bytes(b"weights")
    _write_snapshot_manifest(local_path)
    (local_path / "untracked.bin").write_bytes(b"wrong snapshot")

    with pytest.raises(qwen35.Qwen35RevisionError, match="inventory mismatch"):
        qwen35.validate_qwen35_revision_source(local_path, QWEN35_4B_REVISION)


def test_direct_hdfs_uri_must_be_copied_with_revision_evidence_first():
    with pytest.raises(qwen35.Qwen35RevisionError, match="copy HDFS snapshots locally"):
        qwen35.validate_qwen35_revision_source(
            "hdfs://models/Qwen3.5-4B",
            QWEN35_4B_REVISION,
        )


def test_strict_loading_cannot_be_disabled(monkeypatch):
    _install_fake_transformers(monkeypatch, _conditional_config())
    with pytest.raises(ValueError, match="cannot be disabled"):
        qwen35.load_qwen35_text_model(
            "Qwen/Qwen3.5-4B",
            revision=QWEN35_4B_REVISION,
            attn_implementation="sdpa",
            dtype="bfloat16",
            strict=False,
        )
