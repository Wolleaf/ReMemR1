from copy import deepcopy
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from scripts.cloud import verify_training_artifacts as verifier


TRAIN_SHA = "a" * 64
VALIDATION_SHA = "b" * 64
STATE_SHA = "c" * 64
BASE_MODEL = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"


def _identity(tmp_path: Path, *, resumed: bool):
    train_file = tmp_path / "train" / "train.parquet"
    validation_file = tmp_path / "validation" / "train.parquet"
    train_file.parent.mkdir()
    validation_file.parent.mkdir()
    train_file.write_bytes(b"train")
    validation_file.write_bytes(b"validation")
    predecessor = tmp_path / "checkpoints" / "global_step_1"
    if resumed:
        predecessor.mkdir(parents=True)
    step = 2 if resumed else 1
    resolved_config = {
        "actor_rollout_ref": {
            "actor": {"optim": {"total_training_steps": step}},
            "model": {"path": BASE_MODEL, "revision": REVISION},
        },
        "critic": {"optim": {"total_training_steps": step}},
        "data": {
            "train_files": str(train_file),
            "val_files": str(validation_file),
        },
        "reproduction": {
            "data_manifest_sha256": TRAIN_SHA,
            "template_revision": "rememr1-template-v1",
            "val_data_manifest_sha256": VALIDATION_SHA,
        },
        "trainer": {
            "resume_from_path": str(predecessor) if resumed else None,
            "resume_mode": "resume_path" if resumed else "disable",
            "total_training_steps": step,
        },
    }
    sealed_config = tmp_path / ("resume.yaml" if resumed else "fresh.yaml")
    sealed_config.write_text(yaml.safe_dump(resolved_config), encoding="utf-8")
    semantic_fields = {
        "adapter_state_sha256": "1" * 64,
        "adapter_tensor_keys": ("adapter.weight",),
        "lora_config_sha256": "2" * 64,
        "lora_target_sha256": "3" * 64,
        "text_mapping_sha256": "4" * 64,
    }
    state = SimpleNamespace(
        base_model_id=BASE_MODEL,
        base_model_revision=REVISION,
        data_manifest_sha256=TRAIN_SHA,
        global_step=step,
        resolved_config=resolved_config,
        sealed_config_path=sealed_config,
        sha256=STATE_SHA,
        **semantic_fields,
    )
    adapter = SimpleNamespace(
        base_model_id=BASE_MODEL,
        base_model_revision=REVISION,
        global_step=step,
        sha256="d" * 64,
        source_extra_state_sha256=STATE_SHA,
        template_revision="rememr1-template-v1",
        tokenizer_id=BASE_MODEL,
        tokenizer_revision=REVISION,
        **semantic_fields,
    )
    return state, adapter, train_file, validation_file, predecessor


def _patch_checkpoint_module(monkeypatch, state, adapter):
    modules = {
        "verl": ModuleType("verl"),
        "verl.utils": ModuleType("verl.utils"),
        "verl.utils.checkpoint": ModuleType("verl.utils.checkpoint"),
        "verl.utils.checkpoint.reproduction": ModuleType(
            "verl.utils.checkpoint.reproduction"
        ),
    }
    for name, module in modules.items():
        if name != "verl.utils.checkpoint.reproduction":
            module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    reproduction = modules["verl.utils.checkpoint.reproduction"]
    reproduction.verify_reproduction_checkpoint_directory = lambda path: (
        object(),
        state,
    )
    reproduction.validate_adapter_export = lambda path: adapter

    def validate_config(saved, current, *, allowed_drift_paths):
        saved = deepcopy(saved)
        current = deepcopy(current)
        for dotted_path in allowed_drift_paths:
            for value in (saved, current):
                components = dotted_path.split(".")
                parent = value
                for component in components[:-1]:
                    parent = parent.get(component, {})
                parent.pop(components[-1], None)
        if saved != current:
            raise ValueError("resolved config drift")

    reproduction.validate_resolved_config_compatibility = validate_config


def _verify(
    monkeypatch,
    tmp_path: Path,
    state,
    adapter,
    *,
    resumed: bool,
    expected_train_file: Path | None = None,
    expected_validation_file: Path | None = None,
):
    _patch_checkpoint_module(monkeypatch, state, adapter)
    train_file = expected_train_file or Path(
        state.resolved_config["data"]["train_files"]
    )
    validation_file = expected_validation_file or Path(
        state.resolved_config["data"]["val_files"]
    )
    predecessor = Path(state.resolved_config["trainer"]["resume_from_path"]) if resumed else None
    return verifier.verify_training_artifacts(
        tmp_path / "checkpoint",
        tmp_path / "adapter",
        expected_step=2 if resumed else 1,
        expected_train_file=train_file,
        expected_validation_file=validation_file,
        expected_resolved_config=state.sealed_config_path,
        expected_train_manifest=TRAIN_SHA,
        expected_validation_manifest=VALIDATION_SHA,
        expected_base_model=BASE_MODEL,
        expected_revision=REVISION,
        expected_resume_from=predecessor,
    )


@pytest.mark.parametrize("resumed", [False, True])
def test_verifier_accepts_fresh_and_resumed_gate_identity(tmp_path, monkeypatch, resumed):
    state, adapter, *_ = _identity(tmp_path, resumed=resumed)

    result = _verify(monkeypatch, tmp_path, state, adapter, resumed=resumed)

    assert result == {
        "adapter_metadata_sha256": adapter.sha256,
        "checkpoint_extra_state_sha256": STATE_SHA,
        "global_step": 2 if resumed else 1,
        "status": "verified",
    }


@pytest.mark.parametrize(
    ("target", "bad_value"),
    [
        ("state.global_step", 7),
        ("state.base_model_id", "other/model"),
        ("state.base_model_revision", "e" * 40),
        ("state.data_manifest_sha256", "e" * 64),
        ("config.reproduction.val_data_manifest_sha256", "e" * 64),
        ("config.trainer.total_training_steps", 7),
        ("config.trainer.resume_mode", "resume_path"),
        ("adapter.global_step", 7),
        ("adapter.base_model_id", "other/model"),
        ("adapter.base_model_revision", "e" * 40),
        ("adapter.tokenizer_id", "other/model"),
        ("adapter.tokenizer_revision", "e" * 40),
        ("adapter.template_revision", "e" * 40),
        ("adapter.source_extra_state_sha256", "e" * 64),
        ("adapter.adapter_state_sha256", "e" * 64),
        ("adapter.adapter_tensor_keys", ("different.weight",)),
        ("adapter.lora_config_sha256", "e" * 64),
        ("adapter.lora_target_sha256", "e" * 64),
        ("adapter.text_mapping_sha256", "e" * 64),
    ],
)
def test_verifier_rejects_checkpoint_config_and_adapter_drift(
    tmp_path, monkeypatch, target, bad_value
):
    state, adapter, *_ = _identity(tmp_path, resumed=False)
    state = deepcopy(state)
    adapter = deepcopy(adapter)
    if target.startswith("state."):
        setattr(state, target.removeprefix("state."), bad_value)
    elif target.startswith("adapter."):
        setattr(adapter, target.removeprefix("adapter."), bad_value)
    else:
        current = state.resolved_config
        components = target.removeprefix("config.").split(".")
        for component in components[:-1]:
            current = current[component]
        current[components[-1]] = bad_value

    with pytest.raises(verifier.ArtifactGateError):
        _verify(monkeypatch, tmp_path, state, adapter, resumed=False)


@pytest.mark.parametrize("field", ["train_files", "val_files"])
def test_verifier_rejects_a_different_resolved_data_file(
    tmp_path, monkeypatch, field
):
    state, adapter, train_file, validation_file, _ = _identity(
        tmp_path, resumed=False
    )
    replacement = tmp_path / "other" / f"{field}.parquet"
    replacement.parent.mkdir()
    replacement.write_bytes(b"other")
    state.resolved_config["data"][field] = str(replacement)

    with pytest.raises(verifier.ArtifactGateError, match=f"data.{field}"):
        _verify(
            monkeypatch,
            tmp_path,
            state,
            adapter,
            resumed=False,
            expected_train_file=train_file,
            expected_validation_file=validation_file,
        )


def test_verifier_rejects_a_different_resume_predecessor(tmp_path, monkeypatch):
    state, adapter, train_file, validation_file, _ = _identity(tmp_path, resumed=True)
    other = tmp_path / "checkpoints" / "other" / "global_step_1"
    other.mkdir(parents=True)
    _patch_checkpoint_module(monkeypatch, state, adapter)

    with pytest.raises(verifier.ArtifactGateError, match="resume predecessor"):
        verifier.verify_training_artifacts(
            tmp_path / "checkpoint",
            tmp_path / "adapter",
            expected_step=2,
            expected_train_file=train_file,
            expected_validation_file=validation_file,
            expected_resolved_config=state.sealed_config_path,
            expected_train_manifest=TRAIN_SHA,
            expected_validation_manifest=VALIDATION_SHA,
            expected_base_model=BASE_MODEL,
            expected_revision=REVISION,
            expected_resume_from=other,
        )
