import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "verl" / "utils" / "checkpoint" / "reproduction.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_contract_under_test", MODULE_PATH)
checkpoint = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checkpoint
SPEC.loader.exec_module(checkpoint)


SHA_A = "a" * 64
SHA_B = "b" * 64
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


class AdapterFakeTensor:
    dtype = "torch.float32"
    shape = (2,)

    def __init__(self, payload):
        self.payload = payload

    def tobytes(self):
        return self.payload


ADAPTER_STATE = {
    "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": AdapterFakeTensor(
        b"adapter-a"
    ),
    "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": AdapterFakeTensor(
        b"adapter-b"
    ),
}


def _fake_safetensors_loader(unused_path):
    return ADAPTER_STATE


def _sample_extra_state(**overrides):
    values = {
        "global_step": 20,
        "rng_state": {
            "python": [3, [2147483648, 42], None],
            "numpy": {"bit_generator": "MT19937", "position": 7},
            "torch_cpu_b64": "AAEC",
            "torch_cuda_b64": ["AwQ="],
        },
        "dataloader_state": checkpoint.DataloaderProgress(
            position=80,
            state_sha256=SHA_A,
        ),
        "data_manifest_sha256": SHA_B,
        "base_model_id": "Qwen/Qwen3.5-4B",
        "base_model_revision": MODEL_REVISION,
        "text_mapping": {
            "schema_version": 1,
            "loader": "transformers.AutoModelForCausalLM",
            "source_architectures": ("Qwen3_5ForConditionalGeneration",),
            "checkpoint_text_prefix": "model.language_model",
            "target_text_prefix": "model",
        },
        "lora_config": {
            "r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.0,
            "bias": "none",
        },
        "lora_target_manifest": {
            "schema_version": 1,
            "target_modules": [
                "model.layers.0.mlp.up_proj",
                "model.layers.0.self_attn.q_proj",
            ],
            "sha256": checkpoint.canonical_json_sha256(
                {
                    "schema_version": 1,
                    "target_modules": [
                        "model.layers.0.mlp.up_proj",
                        "model.layers.0.self_attn.q_proj",
                    ],
                }
            ),
            "text_model_prefix": "model",
        },
        "adapter_tensor_keys": sorted(ADAPTER_STATE),
        "adapter_state_sha256": checkpoint.canonical_tensor_state_sha256(
            ADAPTER_STATE
        ),
        "model_build_metadata": {
            "role": "actor",
            "qwen35_text_only": True,
            "qwen35_mapping_sha256": SHA_A,
        },
        "resolved_config": {
            "actor_rollout_ref": {"model": {"lora_rank": 32}},
            "trainer": {"total_training_steps": 80},
        },
    }
    values.update(overrides)
    return checkpoint.CheckpointExtraState.create(**values)


def _sample_adapter_metadata(extra_state=None):
    extra_state = extra_state or _sample_extra_state()
    return checkpoint.AdapterExportMetadata.from_checkpoint(
        extra_state,
        tokenizer_id="Qwen/Qwen3.5-4B",
        tokenizer_revision=MODEL_REVISION,
        template_revision="rememr1-template-v1",
    )


def _publish_payload(path, value):
    return checkpoint.atomic_publish_directory(
        path,
        lambda staging: (staging / "state.txt").write_text(value, encoding="utf-8"),
        marker_metadata={"kind": "checkpoint"},
    )


def _publish_reproduction_checkpoint(path, state, marker_metadata):
    return checkpoint.atomic_publish_directory(
        path,
        lambda staging: state.save(staging / checkpoint.EXTRA_STATE_FILENAME),
        marker_metadata=marker_metadata,
    )


def _bound_dataset(
    *,
    digest=SHA_A,
    mode="train",
    profile="formal",
    count=1,
):
    metadata = tuple(
        {
            "manifest_sha256": digest,
            "mode": mode,
            "profile": profile,
        }
        for _ in range(count)
    )
    return SimpleNamespace(
        bundle_manifest_sha256s=tuple(digest for _ in range(count)),
        bundle_manifest_metadata=metadata,
    )


def _sample_merged_metadata(**overrides):
    values = {
        "global_step": 40,
        "source_adapter_metadata_sha256": SHA_A,
        "base_model_id": "Qwen/Qwen3.5-4B",
        "base_model_revision": MODEL_REVISION,
        "tokenizer_id": "Qwen/Qwen3.5-4B",
        "tokenizer_revision": MODEL_REVISION,
        "template_revision": "rememr1-template-v1",
        "text_mapping_sha256": SHA_B,
        "dtype": "bfloat16",
        "model_state_sha256": "c" * 64,
        "verification_prompt": "merge contract",
        "max_new_tokens": 4,
        "rtol": 0.002,
        "atol": 0.002,
    }
    values.update(overrides)
    return checkpoint.build_merged_model_metadata(**values)


def _publish_fake_merged_artifact(path, metadata, *, adapter_file=False):
    def writer(staging):
        (staging / "config.json").write_text("{}\n", encoding="utf-8")
        (staging / "model.safetensors").write_bytes(b"safe-model")
        (staging / checkpoint.MERGED_MODEL_METADATA_FILENAME).write_bytes(
            checkpoint.canonical_json_bytes(metadata) + b"\n"
        )
        if adapter_file:
            (staging / "adapter_config.json").write_text("{}\n", encoding="utf-8")

    return checkpoint.atomic_publish_directory(
        path,
        writer,
        validator=lambda staging: checkpoint.validate_merged_model_artifact(
            staging,
            expected_metadata=metadata,
        ),
        marker_metadata={
            "artifact_type": "merged_causal_lm",
            "metadata_sha256": metadata["metadata_sha256"],
            "model_state_sha256": metadata["model_state_sha256"],
        },
    )


def test_merged_model_artifact_is_self_hashed_complete_and_adapter_free(tmp_path):
    metadata = _sample_merged_metadata()
    path = _publish_fake_merged_artifact(tmp_path / "merged", metadata)

    assert checkpoint.validate_merged_model_artifact(path) == metadata

    metadata_path = path / checkpoint.MERGED_MODEL_METADATA_FILENAME
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["global_step"] = 41
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(checkpoint.CheckpointContractError):
        checkpoint.validate_merged_model_artifact(path)


def test_invalid_merged_model_never_publishes(tmp_path):
    destination = tmp_path / "merged"
    with pytest.raises(checkpoint.CheckpointContractError, match="adapter-only"):
        _publish_fake_merged_artifact(
            destination,
            _sample_merged_metadata(),
            adapter_file=True,
        )

    assert not destination.exists()


def test_manifest_pair_configuration_requires_distinct_matching_profiles():
    checkpoint.validate_reproduction_manifest_configuration(
        train_sha256=SHA_A,
        train_mode="train",
        train_profile="formal",
        validation_sha256=SHA_B,
        validation_mode="train",
        validation_profile="formal",
        formal_data=True,
    )

    with pytest.raises(checkpoint.CheckpointContractError, match="distinct"):
        checkpoint.validate_reproduction_manifest_configuration(
            train_sha256=SHA_A,
            train_mode="train",
            train_profile="formal",
            validation_sha256=SHA_A,
            validation_mode="train",
            validation_profile="formal",
            formal_data=True,
        )

    with pytest.raises(checkpoint.CheckpointContractError, match="40/80-step"):
        checkpoint.validate_reproduction_manifest_configuration(
            train_sha256=SHA_A,
            train_mode="train",
            train_profile="fixture",
            validation_sha256=SHA_B,
            validation_mode="train",
            validation_profile="fixture",
            formal_data=False,
            total_training_steps=40,
        )
    with pytest.raises(checkpoint.CheckpointContractError, match="requires formal"):
        checkpoint.validate_reproduction_manifest_configuration(
            train_sha256=SHA_A,
            train_mode="train",
            train_profile="formal",
            validation_sha256=SHA_B,
            validation_mode="train",
            validation_profile="fixture",
            formal_data=True,
        )


@pytest.mark.parametrize(
    ("dataset", "configured_digest", "configured_profile", "message"),
    [
        (_bound_dataset(), SHA_B, "formal", "exactly"),
        (_bound_dataset(profile="fixture"), SHA_A, "formal", "mode/profile/hash"),
        (_bound_dataset(count=0), SHA_A, "formal", "exactly"),
        (_bound_dataset(count=2), SHA_A, "formal", "exactly"),
    ],
)
def test_loaded_dataset_manifest_binding_rejects_wrong_val_fixture_and_cardinality(
    dataset,
    configured_digest,
    configured_profile,
    message,
):
    with pytest.raises(checkpoint.CheckpointContractError, match=message):
        checkpoint.validate_bound_dataset_manifest(
            dataset,
            configured_sha256=configured_digest,
            configured_mode="train",
            configured_profile=configured_profile,
            label="validation",
        )


def test_loaded_dataset_manifest_binding_accepts_only_exact_mode_profile_and_hash():
    dataset = _bound_dataset()
    checkpoint.validate_bound_dataset_manifest(
        dataset,
        configured_sha256=SHA_A,
        configured_mode="train",
        configured_profile="formal",
        label="train",
    )
    with pytest.raises(checkpoint.CheckpointContractError, match="mode/profile/hash"):
        checkpoint.validate_bound_dataset_manifest(
            dataset,
            configured_sha256=SHA_A,
            configured_mode="eval",
            configured_profile="formal",
            label="train",
        )


def test_extra_state_round_trip_is_canonical_and_self_hashing(tmp_path):
    rng = {"torch": (3, 2, 1), "python": {"seed": 42}}
    state = _sample_extra_state(rng_state=rng)
    rng["torch"] = (99,)

    path = state.save(tmp_path / checkpoint.EXTRA_STATE_FILENAME)
    loaded = checkpoint.CheckpointExtraState.load(path)

    assert loaded == state
    assert loaded.rng_state["torch"] == [3, 2, 1]
    assert loaded.to_dict()["extra_state_sha256"] == loaded.sha256
    assert loaded.to_json_bytes() == state.to_json_bytes()
    assert loaded.text_mapping["source_architectures"] == [
        "Qwen3_5ForConditionalGeneration"
    ]
    assert loaded.lora_target_sha256 == state.lora_target_manifest["sha256"]


def test_component_hash_rejects_tampering_even_if_outer_hash_is_recomputed():
    serialized = _sample_extra_state().to_dict()
    serialized["lora_config"]["r"] = 8
    payload = dict(serialized)
    payload.pop("extra_state_sha256")
    serialized["extra_state_sha256"] = checkpoint.canonical_json_sha256(payload)

    with pytest.raises(checkpoint.CheckpointContractError, match="lora_config_sha256"):
        checkpoint.CheckpointExtraState.from_dict(serialized)


def test_outer_hash_and_exact_schema_reject_tampering_and_unknown_fields():
    serialized = _sample_extra_state().to_dict()
    serialized["global_step"] = 21

    with pytest.raises(checkpoint.CheckpointContractError, match="extra_state_sha256"):
        checkpoint.CheckpointExtraState.from_dict(serialized)

    serialized = _sample_extra_state().to_dict()
    serialized["unexpected"] = True
    with pytest.raises(checkpoint.CheckpointContractError, match="unknown"):
        checkpoint.CheckpointExtraState.from_dict(serialized)

    serialized = _sample_extra_state().to_dict()
    serialized["schema_version"] = True
    with pytest.raises(checkpoint.CheckpointContractError, match="schema"):
        checkpoint.CheckpointExtraState.from_dict(serialized)


def test_schema_rejects_floating_revision_invalid_progress_and_nonfinite_json():
    with pytest.raises(checkpoint.CheckpointContractError, match="floating"):
        _sample_extra_state(base_model_revision="main")
    with pytest.raises(checkpoint.CheckpointContractError, match="requires"):
        checkpoint.DataloaderProgress()
    with pytest.raises(checkpoint.CheckpointContractError, match="non-finite"):
        checkpoint.canonical_json_bytes({"loss": float("nan")})


def test_json_safe_state_handles_tensor_array_and_bytes_without_eager_torch():
    class FakeTensor:
        dtype = "fake-fp32"
        shape = (2,)

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return [1.0, 2.0]

    converted = checkpoint.to_json_safe_state(
        {"tensor": FakeTensor(), "blob": b"\x00\x01"}
    )

    assert converted == {
        "tensor": {
            "type": "tensor",
            "dtype": "fake-fp32",
            "shape": [2],
            "values": [1.0, 2.0],
        },
        "blob": {"type": "bytes", "base64": "AAE="},
    }
    assert checkpoint.json_safe_state_sha256(converted) == checkpoint.canonical_json_sha256(
        converted
    )


def test_resume_compatibility_checks_only_immutable_training_identity():
    state = _sample_extra_state()

    assert checkpoint.validate_checkpoint_compatibility(
        state,
        base_model_id=state.base_model_id,
        base_model_revision=state.base_model_revision,
        text_mapping=state.text_mapping,
        lora_config=state.lora_config,
        lora_target_manifest=state.lora_target_manifest,
        model_build_metadata=state.model_build_metadata,
        resolved_config=state.resolved_config,
        data_manifest_sha256=state.data_manifest_sha256,
    ) is state

    incompatible_lora = dict(state.lora_config)
    incompatible_lora["r"] = 8
    with pytest.raises(checkpoint.CheckpointContractError, match="lora_config_sha256"):
        checkpoint.validate_checkpoint_compatibility(
            state,
            base_model_id=state.base_model_id,
            base_model_revision=state.base_model_revision,
            text_mapping=state.text_mapping,
            lora_config=incompatible_lora,
            lora_target_manifest=state.lora_target_manifest,
            model_build_metadata=state.model_build_metadata,
            resolved_config=state.resolved_config,
            data_manifest_sha256=state.data_manifest_sha256,
        )


def test_resolved_config_resume_allowlist_is_narrow_and_explicit():
    saved = {
        "algorithm": {"alpha": 0.5},
        "data": {"seed": 42, "train_files": ["formal.parquet"]},
        "actor_rollout_ref": {
            "actor": {"optim": {"total_training_steps": 40}},
            "model": {"revision": MODEL_REVISION},
            "rollout": {"temperature": 1.0},
        },
        "recurrent": {"enable": "memory_revisit"},
        "reproduction": {
            "export_adapter_on_save": False,
            "adapter_export_dir": None,
        },
        "trainer": {
            "total_training_steps": 40,
            "resume_mode": "disable",
            "resume_from_path": None,
            "default_local_dir": "checkpoints/b40",
            "experiment_name": "b40",
            "rollout_data_dir": "outputs/b40",
            "validation_data_dir": "validation/b40",
        },
    }
    resumed = json.loads(json.dumps(saved))
    resumed["actor_rollout_ref"]["actor"]["optim"]["total_training_steps"] = 80
    resumed["trainer"].update(
        {
            "total_training_steps": 80,
            "resume_mode": "resume_path",
            "resume_from_path": "C:/checkpoints/global_step_40",
            "default_local_dir": "checkpoints/b80",
            "experiment_name": "b80",
            "rollout_data_dir": "outputs/b80",
            "validation_data_dir": "validation/b80",
        }
    )
    resumed["reproduction"].update(
        {
            "export_adapter_on_save": True,
            "adapter_export_dir": "exports/b80",
        }
    )

    checkpoint.validate_resolved_config_compatibility(saved, resumed)

    for path, mutate in (
        ("algorithm.alpha", lambda value: value["algorithm"].update(alpha=1.0)),
        ("data.seed", lambda value: value["data"].update(seed=7)),
        (
            "actor_rollout_ref.rollout.temperature",
            lambda value: value["actor_rollout_ref"]["rollout"].update(temperature=0.7),
        ),
        ("recurrent.enable", lambda value: value["recurrent"].update(enable="none")),
    ):
        drifted = json.loads(json.dumps(resumed))
        mutate(drifted)
        with pytest.raises(checkpoint.CheckpointContractError, match=path):
            checkpoint.validate_resolved_config_compatibility(saved, drifted)


def test_rank_extra_state_requires_step_scheduler_and_complete_rng():
    raw_state = {
        "schema_version": checkpoint.REPRODUCTION_RANK_EXTRA_SCHEMA_VERSION,
        "global_step": 20,
        "lr_scheduler": {"last_epoch": 20},
        "rng": {
            "torch_cpu": object(),
            "torch_cuda": [],
            "numpy": object(),
            "python": object(),
        },
    }

    assert checkpoint.validate_reproduction_rank_extra_state(
        raw_state,
        expected_global_step=20,
    ) is raw_state
    with pytest.raises(checkpoint.CheckpointContractError, match="global_step"):
        checkpoint.validate_reproduction_rank_extra_state(
            raw_state,
            expected_global_step=21,
        )
    raw_state["lr_scheduler"] = None
    with pytest.raises(checkpoint.CheckpointContractError, match="lr_scheduler"):
        checkpoint.validate_reproduction_rank_extra_state(raw_state)


def test_scheduler_epoch_and_optimizer_lr_must_match_checkpoint_step():
    scheduler = {"last_epoch": 20, "_last_lr": [1e-5]}
    optimizer = {"param_groups": [{"lr": 1e-5}], "state": {}}

    checkpoint.validate_scheduler_optimizer_alignment(
        scheduler,
        optimizer,
        expected_global_step=20,
    )
    scheduler["last_epoch"] = 19
    with pytest.raises(checkpoint.CheckpointContractError, match="global_step"):
        checkpoint.validate_scheduler_optimizer_alignment(
            scheduler,
            optimizer,
            expected_global_step=20,
        )
    scheduler.update(last_epoch=20, _last_lr=[2e-5])
    with pytest.raises(checkpoint.CheckpointContractError, match="optimizer lr"):
        checkpoint.validate_scheduler_optimizer_alignment(
            scheduler,
            optimizer,
            expected_global_step=20,
        )


def test_atomic_publish_writes_marker_and_detects_later_tampering(tmp_path):
    destination = tmp_path / "global_step_20"
    _publish_payload(destination, "valid")

    manifest = checkpoint.verify_complete_directory(destination)
    assert manifest.metadata == {"kind": "checkpoint"}
    assert [record.path for record in manifest.files] == ["state.txt"]
    assert (destination / checkpoint.COMPLETION_MARKER_FILENAME).is_file()
    marker = json.loads(
        (destination / checkpoint.COMPLETION_MARKER_FILENAME).read_text(encoding="utf-8")
    )
    assert marker["manifest_sha256"] == manifest.sha256

    (destination / "state.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(checkpoint.IncompleteCheckpointError, match="do not match"):
        checkpoint.verify_complete_directory(destination)


def test_completion_marker_outer_hash_rejects_marker_tampering(tmp_path):
    destination = _publish_payload(tmp_path / "global_step_20", "valid")
    marker_path = destination / checkpoint.COMPLETION_MARKER_FILENAME
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["metadata"]["kind"] = "tampered"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(checkpoint.CheckpointContractError, match="outer sha256"):
        checkpoint.verify_complete_directory(destination)


def test_reproduction_checkpoint_binds_artifact_directory_marker_and_state_step(tmp_path):
    valid = _publish_reproduction_checkpoint(
        tmp_path / "valid" / "global_step_20",
        _sample_extra_state(global_step=20),
        {
            "artifact_type": "reproduction_training_checkpoint",
            "global_step": 20,
        },
    )
    _, loaded = checkpoint.verify_reproduction_checkpoint_directory(valid)
    assert loaded.global_step == 20

    invalid_cases = (
        (
            "wrong-artifact/global_step_20",
            _sample_extra_state(global_step=20),
            {"artifact_type": "adapter_export", "global_step": 20},
            "artifact_type",
        ),
        (
            "wrong-marker/global_step_20",
            _sample_extra_state(global_step=20),
            {
                "artifact_type": "reproduction_training_checkpoint",
                "global_step": 19,
            },
            "marker global_step",
        ),
        (
            "wrong-directory/checkpoint_20",
            _sample_extra_state(global_step=20),
            {
                "artifact_type": "reproduction_training_checkpoint",
                "global_step": 20,
            },
            "global_step_<N>",
        ),
        (
            "wrong-state/global_step_20",
            _sample_extra_state(global_step=19),
            {
                "artifact_type": "reproduction_training_checkpoint",
                "global_step": 20,
            },
            "extra-state global_step",
        ),
    )
    for relative_path, state, metadata, message in invalid_cases:
        destination = _publish_reproduction_checkpoint(
            tmp_path / relative_path,
            state,
            metadata,
        )
        with pytest.raises(checkpoint.CheckpointContractError, match=message):
            checkpoint.verify_reproduction_checkpoint_directory(destination)


def test_public_checkpoint_contract_exports_strict_helpers():
    expected = {
        "DEFAULT_RESUME_CONFIG_ALLOWLIST",
        "canonical_tensor_state_sha256",
        "normalize_resolved_config",
        "validate_resolved_config_compatibility",
        "validate_scheduler_optimizer_alignment",
        "verify_reproduction_checkpoint_directory",
    }

    assert expected <= set(checkpoint.__all__)


def test_failed_atomic_publish_never_deletes_last_valid_checkpoint(tmp_path):
    previous = tmp_path / "global_step_20"
    failed = tmp_path / "global_step_40"
    _publish_payload(previous, "last-known-good")
    previous_marker = (previous / checkpoint.COMPLETION_MARKER_FILENAME).read_bytes()

    def failing_writer(staging):
        (staging / "actor.pt").write_bytes(b"partial")
        raise RuntimeError("injected disk failure")

    with pytest.raises(RuntimeError, match="injected disk failure"):
        checkpoint.atomic_publish_directory(failed, failing_writer)

    assert not failed.exists()
    assert (previous / "state.txt").read_text(encoding="utf-8") == "last-known-good"
    assert (previous / checkpoint.COMPLETION_MARKER_FILENAME).read_bytes() == previous_marker
    assert checkpoint.verify_complete_directory(previous)
    assert not list(tmp_path.glob(".global_step_40.staging-*"))


def test_atomic_publish_refuses_overwrite_and_validator_mutation(tmp_path):
    existing = tmp_path / "global_step_20"
    _publish_payload(existing, "old")

    with pytest.raises(FileExistsError, match="refusing to replace"):
        _publish_payload(existing, "new")
    assert (existing / "state.txt").read_text(encoding="utf-8") == "old"

    changed = tmp_path / "global_step_40"

    def mutate_after_marker(staging):
        (staging / "unexpected.txt").write_text("mutation", encoding="utf-8")

    with pytest.raises(checkpoint.IncompleteCheckpointError):
        checkpoint.atomic_publish_directory(
            changed,
            lambda staging: (staging / "state.txt").write_text("new", encoding="utf-8"),
            validator=mutate_after_marker,
        )
    assert not changed.exists()


class FakePeftModel:
    def __init__(
        self,
        *,
        include_full_model=False,
        include_binary=False,
        config_overrides=None,
    ):
        self.include_full_model = include_full_model
        self.include_binary = include_binary
        self.config_overrides = config_overrides or {}
        self.save_kwargs = None

    def save_pretrained(self, directory, **kwargs):
        self.save_kwargs = kwargs
        root = Path(directory)
        adapter_config = {
            "peft_type": "LORA",
            "base_model_name_or_path": "Qwen/Qwen3.5-4B",
            "r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.0,
            "bias": "none",
            "target_modules": [
                "model.layers.0.mlp.up_proj",
                "model.layers.0.self_attn.q_proj",
            ],
        }
        adapter_config.update(self.config_overrides)
        (root / "adapter_config.json").write_text(
            json.dumps(adapter_config),
            encoding="utf-8",
        )
        (root / "adapter_model.safetensors").write_bytes(b"tiny-adapter")
        if self.include_full_model:
            (root / "model.safetensors").write_bytes(b"forbidden-full-model")
        if self.include_binary:
            (root / "adapter_model.bin").write_bytes(b"forbidden-pickle")


def test_adapter_only_export_has_metadata_safe_weights_and_completion_hash(tmp_path):
    model = FakePeftModel()
    metadata = _sample_adapter_metadata()
    destination = tmp_path / "adapter"

    result = checkpoint.export_peft_adapter(
        model,
        destination,
        metadata,
        safetensors_loader=_fake_safetensors_loader,
    )
    loaded = checkpoint.validate_adapter_export(
        result,
        expected_metadata=metadata,
        safetensors_loader=_fake_safetensors_loader,
    )

    assert result == destination
    assert loaded == metadata
    assert model.save_kwargs["safe_serialization"] is True
    assert (destination / "adapter_model.safetensors").is_file()
    assert checkpoint.AdapterExportMetadata.load(
        destination / checkpoint.ADAPTER_METADATA_FILENAME
    ) == metadata


def test_adapter_export_rejects_full_model_files_without_publishing(tmp_path):
    destination = tmp_path / "adapter"

    with pytest.raises(
        checkpoint.CheckpointContractError,
        match="non-adapter|full-model",
    ):
        checkpoint.export_peft_adapter(
            FakePeftModel(include_full_model=True),
            destination,
            _sample_adapter_metadata(),
            safetensors_loader=_fake_safetensors_loader,
        )

    assert not destination.exists()
    assert not list(tmp_path.glob(".adapter.staging-*"))


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (FakePeftModel(include_binary=True), "binary weights"),
        (FakePeftModel(config_overrides={"r": 8}), "r does not match"),
        (
            FakePeftModel(config_overrides={"target_modules": ["wrong.target"]}),
            "target_modules",
        ),
    ],
)
def test_adapter_export_rejects_binary_and_config_drift(tmp_path, model, message):
    with pytest.raises(checkpoint.CheckpointContractError, match=message):
        checkpoint.export_peft_adapter(
            model,
            tmp_path / "adapter",
            _sample_adapter_metadata(),
            safetensors_loader=_fake_safetensors_loader,
        )


def test_adapter_export_rejects_tensor_key_or_semantic_hash_mismatch(tmp_path):
    destination = checkpoint.export_peft_adapter(
        FakePeftModel(),
        tmp_path / "adapter",
        _sample_adapter_metadata(),
        safetensors_loader=_fake_safetensors_loader,
    )
    wrong_keys = {"unexpected.lora_A.weight": AdapterFakeTensor(b"adapter-a")}
    with pytest.raises(checkpoint.CheckpointContractError, match="tensor keys"):
        checkpoint.validate_adapter_export(
            destination,
            safetensors_loader=lambda unused: wrong_keys,
        )
    changed_bytes = dict(ADAPTER_STATE)
    first_key = sorted(changed_bytes)[0]
    changed_bytes[first_key] = AdapterFakeTensor(b"different-bytes")
    with pytest.raises(checkpoint.CheckpointContractError, match="semantic"):
        checkpoint.validate_adapter_export(
            destination,
            safetensors_loader=lambda unused: changed_bytes,
        )


def test_adapter_metadata_is_strict_and_bound_to_source_checkpoint():
    extra_state = _sample_extra_state()
    metadata = _sample_adapter_metadata(extra_state)
    serialized = metadata.to_dict()

    assert metadata.source_extra_state_sha256 == extra_state.sha256
    serialized["adapter_state_sha256"] = SHA_B
    with pytest.raises(checkpoint.CheckpointContractError, match="metadata_sha256"):
        checkpoint.AdapterExportMetadata.from_dict(serialized)


def test_lazy_adapter_reload_merge_and_merged_model_reload(tmp_path):
    adapter_directory = checkpoint.export_peft_adapter(
        FakePeftModel(),
        tmp_path / "adapter",
        _sample_adapter_metadata(),
        safetensors_loader=_fake_safetensors_loader,
    )
    calls = []
    base_model = object()
    merged_model = object()

    class LoadedAdapter:
        def merge_and_unload(self, **kwargs):
            calls.append(("merge", kwargs))
            return merged_model

    def peft_loader(received_base, received_path, **kwargs):
        calls.append(("peft_load", received_base, received_path, kwargs))
        return LoadedAdapter()

    result = checkpoint.merge_and_unload_adapter(
        base_model,
        adapter_directory,
        peft_model_loader=peft_loader,
        load_kwargs={"device_map": "cpu"},
        merge_kwargs={"safe_merge": True},
        validate_export=False,
    )

    assert result is merged_model
    assert calls == [
        (
            "peft_load",
            base_model,
            str(adapter_directory),
            {"device_map": "cpu", "is_trainable": False},
        ),
        ("merge", {"safe_merge": True}),
    ]

    merged_directory = tmp_path / "merged"
    merged_directory.mkdir()
    reloaded = object()

    class ModelLoader:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("model_load", path, kwargs))
            return reloaded

    assert checkpoint.reload_merged_model(
        merged_directory,
        model_loader=ModelLoader,
        torch_dtype="bfloat16",
    ) is reloaded
    assert calls[-1] == (
        "model_load",
        str(merged_directory),
        {"torch_dtype": "bfloat16"},
    )


def test_real_tiny_peft_adapter_export_reload_and_merge_without_downloads(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")

    config = transformers.GPT2Config(
        vocab_size=32,
        n_positions=16,
        n_embd=8,
        n_layer=1,
        n_head=1,
    )
    torch.manual_seed(7)
    base = transformers.GPT2LMHeadModel(config)
    base_state = {name: tensor.detach().clone() for name, tensor in base.state_dict().items()}
    peft_model = peft.get_peft_model(
        base,
        peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["c_attn"],
        ),
    )
    with torch.no_grad():
        for name, parameter in peft_model.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.125)
    peft_model.eval()
    input_ids = torch.tensor([[1, 2, 3]])
    expected_logits = peft_model(input_ids=input_ids).logits.detach()

    text_mapping = {"mapping_strategy": "generic_test_causal_lm"}
    lora_config = {
        "r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "bias": "none",
    }
    target_manifest = {
        "schema_version": 1,
        "target_modules": ["c_attn"],
    }
    adapter_state = peft.get_peft_model_state_dict(peft_model)
    metadata = checkpoint.AdapterExportMetadata(
        global_step=1,
        base_model_id="tiny-local-gpt2",
        base_model_revision="tiny-v1",
        tokenizer_id="tiny-local-tokenizer",
        tokenizer_revision="tiny-v1",
        template_revision="tiny-template-v1",
        text_mapping=text_mapping,
        text_mapping_sha256=checkpoint.canonical_json_sha256(text_mapping),
        lora_config=lora_config,
        lora_config_sha256=checkpoint.canonical_json_sha256(lora_config),
        lora_target_manifest=target_manifest,
        lora_target_sha256=checkpoint.canonical_json_sha256(target_manifest),
        adapter_tensor_keys=tuple(sorted(adapter_state)),
        adapter_state_sha256=checkpoint.canonical_tensor_state_sha256(adapter_state),
        source_extra_state_sha256="e" * 64,
    )
    adapter_dir = checkpoint.export_peft_adapter(
        peft_model,
        tmp_path / "adapter",
        metadata,
    )
    assert not (adapter_dir / "model.safetensors").exists()
    assert not (adapter_dir / "pytorch_model.bin").exists()

    reloaded_base = transformers.GPT2LMHeadModel(config)
    reloaded_base.load_state_dict(base_state)
    reloaded_adapter = checkpoint.load_peft_adapter(reloaded_base, adapter_dir)
    reloaded_adapter.eval()
    torch.testing.assert_close(
        reloaded_adapter(input_ids=input_ids).logits,
        expected_logits,
        rtol=1e-5,
        atol=1e-5,
    )

    merge_base = transformers.GPT2LMHeadModel(config)
    merge_base.load_state_dict(base_state)
    merged = checkpoint.merge_and_unload_adapter(merge_base, adapter_dir)
    merged.eval()
    torch.testing.assert_close(
        merged(input_ids=input_ids).logits,
        expected_logits,
        rtol=1e-5,
        atol=1e-5,
    )
    merged_dir = tmp_path / "merged"
    merged.save_pretrained(merged_dir, safe_serialization=True)
    reloaded_merged = checkpoint.reload_merged_model(merged_dir)
    reloaded_merged.eval()
    torch.testing.assert_close(
        reloaded_merged(input_ids=input_ids).logits,
        expected_logits,
        rtol=1e-5,
        atol=1e-5,
    )


def test_merge_reload_validation_failure_never_publishes_target(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    exporter_path = REPO_ROOT / "scripts" / "reproduction" / "export_adapter.py"
    spec = importlib.util.spec_from_file_location(
        "reproduction_export_adapter_under_test",
        exporter_path,
    )
    exporter = importlib.util.module_from_spec(spec)
    for package_name in ("verl", "verl.utils", "verl.utils.checkpoint"):
        package = ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(
        sys.modules,
        "verl.utils.checkpoint.reproduction",
        checkpoint,
    )
    spec.loader.exec_module(exporter)

    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    destination = tmp_path / "merged"
    metadata = SimpleNamespace(
        sha256=SHA_A,
        tokenizer_id="tiny-tokenizer",
        tokenizer_revision=MODEL_REVISION,
        base_model_id="tiny-base",
        base_model_revision=MODEL_REVISION,
        global_step=40,
        template_revision="rememr1-template-v1",
        text_mapping_sha256=SHA_A,
    )

    class FakeBatch(dict):
        def to(self, unused_device):
            return self

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, unused_id, **unused_kwargs):
            return cls()

        def __call__(self, unused_prompt, *, return_tensors):
            assert return_tensors == "pt"
            return FakeBatch(input_ids=torch.tensor([[1]]))

        def save_pretrained(self, directory):
            (Path(directory) / "tokenizer.json").write_text("{}", encoding="utf-8")

    class FakeModel:
        def __init__(self, logit):
            self.logit = logit

        def to(self, *unused_args, **unused_kwargs):
            return self

        def eval(self):
            return self

        def __call__(self, **unused_inputs):
            return SimpleNamespace(logits=torch.tensor([[[self.logit]]]))

        def generate(self, **unused_inputs):
            return torch.tensor([[1, 2]])

        def merge_and_unload(self, *, safe_merge):
            assert safe_merge is True
            return self

        def save_pretrained(self, directory, *, safe_serialization):
            assert safe_serialization is True
            (Path(directory) / "config.json").write_text("{}", encoding="utf-8")
            (Path(directory) / "model.safetensors").write_bytes(b"merged")

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, unused_path, **unused_kwargs):
            return FakeModel(2.0)

    monkeypatch.setattr(transformers, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(transformers, "AutoModelForCausalLM", FakeAutoModel)
    monkeypatch.setattr(exporter, "validate_adapter_export", lambda unused: metadata)
    monkeypatch.setattr(
        exporter,
        "_load_pinned_base_model",
        lambda unused_metadata, unused_torch, **unused_kwargs: object(),
    )
    monkeypatch.setattr(
        exporter,
        "load_peft_adapter",
        lambda unused_base, unused_path: FakeModel(1.0),
    )
    monkeypatch.setattr(
        exporter,
        "_assert_merged_model_contract",
        lambda unused_model, unused_torch: SHA_B,
    )

    with pytest.raises(AssertionError):
        exporter.merge_and_verify_adapter(
            adapter_path,
            destination,
            prompt="verify me",
            device="cpu",
        )

    assert not destination.exists()
    assert not list(tmp_path.glob(".merged.staging-*"))


def test_failed_verification_cannot_prune_previous_checkpoint(tmp_path):
    old = _publish_payload(tmp_path / "global_step_20", "old")
    new = _publish_payload(tmp_path / "global_step_40", "new")

    def failed_validator(unused_path):
        return checkpoint.CheckpointValidationEvidence(
            load_succeeded=True,
            generate_succeeded=False,
            hash_matched=True,
        )

    with pytest.raises(checkpoint.CheckpointVerificationError, match="failed checks"):
        checkpoint.verify_checkpoint_for_pruning(new, failed_validator)

    assert old.is_dir()
    assert new.is_dir()


def test_verified_new_checkpoint_allows_explicit_sibling_prune(tmp_path):
    old = _publish_payload(tmp_path / "global_step_20", "old")
    new = _publish_payload(tmp_path / "global_step_40", "new")
    validated_paths = []

    def validator(path):
        validated_paths.append(path)
        assert (path / "state.txt").read_text(encoding="utf-8") == "new"
        return checkpoint.CheckpointValidationEvidence(
            load_succeeded=True,
            generate_succeeded=True,
            hash_matched=True,
            details={"generated_token_hash": SHA_A},
        )

    receipt = checkpoint.verify_checkpoint_for_pruning(new, validator)
    removed = checkpoint.prune_checkpoints_after_verification(
        [old],
        verification=receipt,
    )

    assert validated_paths == [new.resolve()]
    assert removed == (old.resolve(),)
    assert not old.exists()
    assert checkpoint.verify_complete_directory(new)


def test_prune_rechecks_receipt_and_validates_all_candidates_first(tmp_path):
    old = _publish_payload(tmp_path / "global_step_20", "old")
    another_old = _publish_payload(tmp_path / "global_step_30", "another")
    new = _publish_payload(tmp_path / "global_step_40", "new")
    receipt = checkpoint.verify_checkpoint_for_pruning(
        new,
        lambda unused: checkpoint.CheckpointValidationEvidence(True, True, True),
    )

    (another_old / "state.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(checkpoint.IncompleteCheckpointError):
        checkpoint.prune_checkpoints_after_verification(
            [old, another_old],
            verification=receipt,
        )
    assert old.is_dir()
    assert another_old.is_dir()

    (new / "state.txt").write_text("changed after validation", encoding="utf-8")
    with pytest.raises(checkpoint.IncompleteCheckpointError):
        checkpoint.prune_checkpoints_after_verification(
            [old],
            verification=receipt,
        )
    assert old.is_dir()


def test_verified_checkpoint_receipt_cannot_be_constructed_directly(tmp_path):
    with pytest.raises(TypeError, match="only be created"):
        checkpoint.VerifiedCheckpoint(
            tmp_path,
            SHA_A,
            checkpoint.CheckpointValidationEvidence(True, True, True),
            _token=object(),
        )


def test_verified_checkpoint_receipt_is_immutable(tmp_path):
    new = _publish_payload(tmp_path / "global_step_40", "new")
    receipt = checkpoint.verify_checkpoint_for_pruning(
        new,
        lambda unused: checkpoint.CheckpointValidationEvidence(True, True, True),
    )

    with pytest.raises((AttributeError, TypeError)):
        receipt.path = tmp_path
