import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from scripts.cloud import resume_endpoint_probe as probe
from scripts.cloud import run_resolved_training as runtime


_CONTRACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "verl"
    / "utils"
    / "checkpoint"
    / "reproduction.py"
)
_CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "_resume_probe_checkpoint_contract", _CONTRACT_PATH
)
checkpoint_contract = importlib.util.module_from_spec(_CONTRACT_SPEC)
sys.modules[_CONTRACT_SPEC.name] = checkpoint_contract
_CONTRACT_SPEC.loader.exec_module(checkpoint_contract)


@pytest.fixture(autouse=True)
def _inject_checkpoint_contract(monkeypatch):
    monkeypatch.setattr(probe, "_checkpoint_module", lambda: checkpoint_contract)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _fixture(tmp_path, config_id="b40_qwen35_2b_5090_r0"):
    attempt = tmp_path / "b40-attempt"
    attempt.mkdir()
    _write(attempt / ".success", b"0\n")
    predecessor_checkpoint = tmp_path / "b20-attempt" / "checkpoints" / "global_step_20"
    predecessor_checkpoint.mkdir(parents=True)
    predecessor_evidence = _write(
        predecessor_checkpoint / "checkpoint-evidence.json",
        b'{"status":"verified"}\n',
    )
    predecessor = {
        "logical_config_id": "b20_qwen35_2b_5090_r0",
        "resolved_config_sha256": "1" * 64,
        "checkpoint_dir": str(predecessor_checkpoint.resolve()),
        "checkpoint_step": 20,
        "evidence_path": str(predecessor_evidence.resolve()),
        "evidence_sha256": _sha256(predecessor_evidence),
    }
    binding_unsigned = {
        "schema_version": 1,
        "config_id": config_id,
        "index_sha256": "2" * 64,
        "resolved_config_sha256": "3" * 64,
        "attempt_id": attempt.name,
        "attempt_root": str(attempt.resolve()),
        "paths": {
            "checkpoint_dir": str((attempt / "checkpoints").resolve()),
            "rollout_dir": str((attempt / "logs" / "rollouts").resolve()),
            "validation_dir": str((attempt / "logs" / "validation").resolve()),
            "adapter_dir": str((attempt / "artifacts" / "adapter").resolve()),
            "telemetry_path": str((attempt / "telemetry.json").resolve()),
            "step_zero_fingerprint_path": {
                "path": str((tmp_path / "b20-fingerprint.json").resolve()),
                "sha256": "4" * 64,
            },
            "step_zero_reference": None,
            "pilot_evidence_path": None,
        },
        "resume_predecessor": predecessor,
    }
    binding = {
        **binding_unsigned,
        "binding_sha256": runtime._canonical_sha256(binding_unsigned),
    }
    binding_path = attempt / runtime.BINDING_SOURCE_FILENAME
    _write(binding_path, runtime._canonical_json_bytes(binding) + b"\n")
    runtime_evidence_path = _write(
        attempt / "runtime-bound" / runtime.EVIDENCE_FILENAME,
        b'{"fixture":true}\n',
    )
    runtime_config_sha = "5" * 64
    runtime_evidence = {
        "config_id": config_id,
        "attempt_id": attempt.name,
        "attempt_root": str(attempt.resolve()),
        "source_binding": {
            "path": str(binding_path.resolve()),
            "file_sha256": _sha256(binding_path),
            "binding_sha256": binding["binding_sha256"],
        },
        "resume_predecessor": predecessor,
        "runtime_bound_config_sha256": runtime_config_sha,
        "evidence_sha256": "6" * 64,
    }

    step = 40
    checkpoint = attempt / "checkpoints" / f"global_step_{step}"
    actor = checkpoint / "actor"
    raw_rng = {
        "torch_cpu": "cpu-state",
        "torch_cuda": [],
        "numpy": "numpy-state",
        "python": "python-state",
    }
    scheduler = {"last_epoch": step, "_last_lr": [0.125]}
    optimizer = {"param_groups": [{"lr": 0.125}], "state": {"slot": 1}}
    rank_extra = {
        "schema_version": 1,
        "global_step": step,
        "lr_scheduler": scheduler,
        "rng": raw_rng,
    }
    driver_rng = {"seed": 11}
    dataloader = {"position": 160}
    driver_json = checkpoint_contract.to_json_safe_state(driver_rng)
    rank_rng_json = checkpoint_contract.to_json_safe_state(raw_rng)
    worker = {
        "schema_version": 1,
        "rank": 0,
        "world_size": 1,
        "global_step": step,
        "rng_state": rank_rng_json,
        "rng_state_sha256": checkpoint_contract.canonical_json_sha256(
            rank_rng_json
        ),
        "lr_scheduler_sha256": checkpoint_contract.json_safe_state_sha256(
            scheduler
        ),
        "adapter_tensor_keys": ["adapter.weight"],
        "adapter_state_sha256": "7" * 64,
    }
    state_sha = "8" * 64
    resolved_config = {
        "reproduction": {
            "offload_profile": "r0",
            "runtime_attempt_id": attempt.name,
            "runtime_binding_sha256": binding["binding_sha256"],
            "runtime_bound_evidence_path": str(runtime_evidence_path.resolve()),
            "sealed_config_id": config_id,
            "sealed_config_sha256": binding["resolved_config_sha256"],
        },
        "trainer": {
            "default_local_dir": str((attempt / "checkpoints").resolve()),
            "resume_from_path": predecessor["checkpoint_dir"],
            "resume_mode": "resume_path",
            "total_training_steps": step,
        },
    }
    state = types.SimpleNamespace(
        global_step=step,
        base_model_id=probe.BASE_MODEL,
        base_model_revision=probe.MODEL_REVISION,
        resolved_config_sha256=runtime_config_sha,
        resolved_config=resolved_config,
        dataloader_state=types.SimpleNamespace(
            state_sha256=checkpoint_contract.json_safe_state_sha256(dataloader)
        ),
        rng_state={
            "schema_version": 1,
            "driver": {
                "rng_state": driver_json,
                "rng_state_sha256": checkpoint_contract.canonical_json_sha256(
                    driver_json
                ),
            },
            "actor_workers": [worker],
        },
        adapter_tensor_keys=("adapter.weight",),
        adapter_state_sha256="7" * 64,
        lora_config_sha256="9" * 64,
        lora_target_sha256="a" * 64,
        text_mapping_sha256="b" * 64,
        sha256=state_sha,
    )
    adapter_metadata_path = (
        attempt
        / "artifacts"
        / "adapter"
        / f"global_step_{step}"
        / "adapter"
        / checkpoint_contract.ADAPTER_METADATA_FILENAME
    )
    _write(adapter_metadata_path, b'{"fixture":"adapter"}\n')
    adapter = types.SimpleNamespace(
        global_step=step,
        base_model_id=probe.BASE_MODEL,
        base_model_revision=probe.MODEL_REVISION,
        source_extra_state_sha256=state_sha,
        adapter_tensor_keys=("adapter.weight",),
        adapter_state_sha256="7" * 64,
        lora_config_sha256="9" * 64,
        lora_target_sha256="a" * 64,
        text_mapping_sha256="b" * 64,
        tokenizer_id=probe.BASE_MODEL,
        tokenizer_revision=probe.MODEL_REVISION,
        template_revision="rememr1-template-v1",
        sha256="c" * 64,
    )

    loaded_values = {}
    files = {
        "data.pt": dataloader,
        "driver_rng.pt": driver_rng,
        "actor/model_world_size_1_rank_0.pt": {"weight": 1},
        "actor/optim_world_size_1_rank_0.pt": optimizer,
        "actor/extra_state_world_size_1_rank_0.pt": rank_extra,
    }
    records = []
    for relative, value in files.items():
        path = _write(checkpoint / relative, f"fixture:{relative}\n".encode("ascii"))
        loaded_values[str(path.resolve())] = value
        records.append(types.SimpleNamespace(path=relative, sha256=_sha256(path)))
    root_extra = _write(
        checkpoint / checkpoint_contract.EXTRA_STATE_FILENAME,
        b'{"fixture":"root-extra"}\n',
    )
    completion = _write(
        checkpoint / checkpoint_contract.COMPLETION_MARKER_FILENAME,
        b'{"fixture":"complete"}\n',
    )
    records.extend(
        (
            types.SimpleNamespace(
                path=checkpoint_contract.EXTRA_STATE_FILENAME,
                sha256=_sha256(root_extra),
            ),
            types.SimpleNamespace(
                path=checkpoint_contract.COMPLETION_MARKER_FILENAME,
                sha256=_sha256(completion),
            ),
        )
    )
    manifest = types.SimpleNamespace(files=tuple(records), sha256="d" * 64)
    calls = []

    def loader(path, *, map_location, weights_only):
        calls.append((Path(path), map_location, weights_only))
        return loaded_values[str(Path(path).resolve())]

    dependencies = {
        "runtime_verifier": lambda unused: runtime_evidence,
        "checkpoint_verifier": lambda unused: (manifest, state),
        "adapter_validator": lambda unused: adapter,
        "state_loader": loader,
    }
    verify_dependencies = {
        "runtime_verifier": dependencies["runtime_verifier"],
        "checkpoint_verifier": dependencies["checkpoint_verifier"],
        "adapter_evidence_verifier": lambda unused: (
            types.SimpleNamespace(),
            adapter,
        ),
    }
    return types.SimpleNamespace(
        attempt=attempt,
        adapter=adapter,
        calls=calls,
        dependencies=dependencies,
        loaded_values=loaded_values,
        manifest=manifest,
        output=tmp_path / "artifact-stage" / "resume-probe.json",
        rank_extra=rank_extra,
        runtime_evidence=runtime_evidence,
        state=state,
        verify_dependencies=verify_dependencies,
    )


def test_probe_deserializes_all_resume_state_on_cpu_without_updates(tmp_path):
    fixture = _fixture(tmp_path)

    evidence = probe.build_resume_endpoint_probe(
        fixture.attempt, **fixture.dependencies
    )

    assert evidence["status"] == "verified"
    assert evidence["probe_mode"] == "deserialize-only"
    assert evidence["updates_applied"] == 0
    assert evidence["endpoint"] == "b40"
    assert evidence["resume_predecessor"] == fixture.runtime_evidence[
        "resume_predecessor"
    ]
    assert len(evidence["loaded_state_files"]) == 5
    assert {item["kind"] for item in evidence["loaded_state_files"]} == {
        "actor_model",
        "actor_optimizer",
        "actor_rank_extra",
        "dataloader",
        "driver_rng",
    }
    assert len(fixture.calls) == 5
    assert all(
        map_location == "cpu" and weights_only is False
        for _, map_location, weights_only in fixture.calls
    )
    unsigned = {
        key: value for key, value in evidence.items() if key != "probe_sha256"
    }
    assert evidence["probe_sha256"] == probe._canonical_sha256(unsigned)

    published = probe.publish_resume_endpoint_probe(
        evidence,
        fixture.output,
        **fixture.verify_dependencies,
    )
    assert published == evidence
    assert fixture.output.read_bytes() == probe._canonical_bytes(evidence) + b"\n"


@pytest.mark.parametrize(
    ("target", "match"),
    [
        ("scheduler", "global_step"),
        ("optimizer", "optimizer shard"),
        ("rng", "RNG state differs"),
        ("predecessor", "predecessor step"),
    ],
)
def test_probe_rejects_incomplete_or_misaligned_resume_state(
    tmp_path, target, match
):
    fixture = _fixture(tmp_path)
    if target == "scheduler":
        fixture.rank_extra["lr_scheduler"]["last_epoch"] = 39
    elif target == "optimizer":
        optimizer_path = next(
            path for path in fixture.loaded_values if "optim_world_size_" in path
        )
        fixture.loaded_values[optimizer_path] = {}
    elif target == "rng":
        fixture.rank_extra["rng"]["python"] = "changed"
    else:
        fixture.runtime_evidence["resume_predecessor"]["checkpoint_step"] = 19

    with pytest.raises(Exception, match=match):
        probe.build_resume_endpoint_probe(fixture.attempt, **fixture.dependencies)


def test_probe_rejects_non_endpoint_attempt(tmp_path):
    fixture = _fixture(tmp_path, config_id="b60_qwen35_2b_5090_r0")

    with pytest.raises(probe.ResumeProbeError, match="not a B40/C40/B80/C80"):
        probe.build_resume_endpoint_probe(fixture.attempt, **fixture.dependencies)


def test_probe_verification_rejects_source_tampering_and_overwrite(tmp_path):
    fixture = _fixture(tmp_path)
    evidence = probe.build_resume_endpoint_probe(
        fixture.attempt, **fixture.dependencies
    )
    probe.publish_resume_endpoint_probe(
        evidence,
        fixture.output,
        **fixture.verify_dependencies,
    )
    original = fixture.output.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        probe.publish_resume_endpoint_probe(
            evidence,
            fixture.output,
            **fixture.verify_dependencies,
        )
    assert fixture.output.read_bytes() == original

    loaded_path = Path(evidence["loaded_state_files"][0]["path"])
    loaded_path.write_bytes(loaded_path.read_bytes() + b"tampered")
    with pytest.raises(probe.ResumeProbeError, match="loaded state file changed"):
        probe.verify_resume_endpoint_probe(
            fixture.output,
            **fixture.verify_dependencies,
        )


def test_probe_module_import_does_not_import_torch():
    command = (
        "import sys; before=set(sys.modules); "
        "import scripts.cloud.resume_endpoint_probe as probe; "
        "probe._checkpoint_module(); "
        "loaded=set(sys.modules)-before; "
        "assert not ({'torch','numpy'} & loaded), sorted(loaded)"
    )
    subprocess.run(
        [sys.executable, "-c", command],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )
