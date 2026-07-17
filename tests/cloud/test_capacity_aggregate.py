from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.cloud import capacity_aggregate as aggregate


GIB = 1024**3
COMMIT = "a" * 40
CONFIG_SHA = "b" * 64
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"


@pytest.fixture
def short_tmp_path(tmp_path):
    if os.name != "nt":
        yield tmp_path
        return
    root = Path(tempfile.mkdtemp(prefix="g2-", dir=Path.cwd().anchor))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _canonical(value):
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _write_json(path: Path, value, *, canonical=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    if canonical:
        path.write_bytes(_canonical(value) + b"\n")
    else:
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return path


def _sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_meta(run: Path, stage: str, pipeline: Path, profile="r0"):
    run.mkdir(parents=True, exist_ok=True)
    (run / ".success").write_bytes(b"0\n")
    (run / "run.meta").write_bytes(
        "\n".join(
            (
                "schema_version=1",
                f"stage={stage}",
                f"git_commit={COMMIT}",
                f"experiment_profile_id={aggregate.PROFILE_ID}",
                "phase=gpu-capacity",
                f"offload_profile={profile}",
                f"budget_projection_sha256={'f' * 64 if profile == 'r1' else ''}",
                f"pipeline_dir={pipeline}",
                "started_at=2026-07-17T00:00:00Z",
                "",
            )
        ).encode("ascii"),
    )


def _trainable(initial_hash="1" * 64):
    return {
        "schema_version": 1,
        "names": ["base.model.layer.lora_A.default.weight"],
        "tensor_count": 1,
        "trainable_numel": 2,
        "total_numel": 10,
        "trainable_ratio": 0.2,
        "manifest_sha256": "2" * 64,
        "state_sha256": initial_hash,
    }


def _checkpoint_state(
    *,
    run: Path,
    profile: str,
    step: int,
    config_id: str,
    config_sha: str,
    resolved_sha: str,
    resumed: bool,
    initial_hash="1" * 64,
    final_hash="3" * 64,
):
    resolved = {
        "reproduction": {
            "adapter_export_dir": str(run / "artifacts" / "adapter"),
            "experiment_profile_id": aggregate.PROFILE_ID,
            "offload_profile": profile,
            "runtime_attempt_id": run.name,
            "sealed_config_id": config_id,
            "sealed_config_sha256": config_sha,
        },
        "trainer": {
            "default_local_dir": str(run / "checkpoints"),
            "resume_mode": "resume_path" if resumed else "disable",
            "resume_from_path": "bound-predecessor" if resumed else None,
            "total_training_steps": step,
        },
    }
    return SimpleNamespace(
        adapter_state_sha256=final_hash,
        adapter_tensor_keys=("base.model.layer.lora_A.default.weight",),
        base_model_id="Qwen/Qwen3.5-2B",
        base_model_revision=MODEL_REVISION,
        data_manifest_sha256="4" * 64,
        global_step=step,
        lora_config_sha256="5" * 64,
        lora_target_sha256="6" * 64,
        model_build_metadata={"trainable_parameters": _trainable(initial_hash)},
        resolved_config=resolved,
        resolved_config_sha256=resolved_sha,
        sha256=("7" if step == 5 else "8") * 64,
        text_mapping_sha256="9" * 64,
    )


def _artifact_fixture(tmp_path: Path, *, initial_hash="1" * 64, final_hash="3" * 64):
    pipeline = tmp_path / "pipeline"
    runs = tmp_path / "runs"
    step1 = runs / "attempt-g2b-step1"
    resume = runs / "attempt-g2b-resume5"
    _run_meta(step1, "g2b-step1", pipeline)
    _run_meta(resume, "g2b-resume5", pipeline)
    step1_checkpoint = step1 / "checkpoints" / "global_step_1"
    resume_checkpoint = resume / "checkpoints" / "global_step_5"
    adapter_path = resume / "artifacts" / "adapter" / "global_step_5" / "adapter"
    for path in (
        step1_checkpoint,
        resume_checkpoint,
        adapter_path,
        resume / "runtime-bound",
    ):
        path.mkdir(parents=True, exist_ok=True)
    step1_config = "g2b_qwen35_2b_5090_step1_r0"
    resume_config = "g2b_qwen35_2b_5090_resume5_r0"
    step1_sha = "a" * 64
    resume_sha = "b" * 64
    runtime_config_sha = "c" * 64
    step1_state = _checkpoint_state(
        run=step1,
        profile="r0",
        step=1,
        config_id=step1_config,
        config_sha=step1_sha,
        resolved_sha="d" * 64,
        resumed=False,
        final_hash="e" * 64,
    )
    final_state = _checkpoint_state(
        run=resume,
        profile="r0",
        step=5,
        config_id=resume_config,
        config_sha=resume_sha,
        resolved_sha=runtime_config_sha,
        resumed=True,
        initial_hash=initial_hash,
        final_hash=final_hash,
    )
    runtime = {
        "attempt_id": resume.name,
        "config_id": resume_config,
        "evidence_sha256": "f" * 64,
        "resume_predecessor": {
            "checkpoint_dir": str(step1_checkpoint),
            "checkpoint_step": 1,
            "logical_config_id": step1_config,
            "resolved_config_sha256": step1_sha,
        },
        "runtime_bound_config_sha256": runtime_config_sha,
    }
    manifests = {
        resume_checkpoint: SimpleNamespace(sha256="0" * 64),
        step1_checkpoint: SimpleNamespace(sha256="1" * 64),
    }
    states = {resume_checkpoint: final_state, step1_checkpoint: step1_state}
    adapter = SimpleNamespace(
        adapter_state_sha256=final_hash,
        adapter_tensor_keys=final_state.adapter_tensor_keys,
        base_model_id=final_state.base_model_id,
        base_model_revision=final_state.base_model_revision,
        global_step=5,
        lora_config_sha256=final_state.lora_config_sha256,
        lora_target_sha256=final_state.lora_target_sha256,
        sha256="2" * 64,
        source_extra_state_sha256=final_state.sha256,
        text_mapping_sha256=final_state.text_mapping_sha256,
    )
    evidence = aggregate.create_artifact_evidence(
        resume_checkpoint,
        adapter_path,
        checkpoint_verifier=lambda path: (manifests[path], states[path]),
        adapter_verifier=lambda path: adapter,
        runtime_verifier=lambda path: runtime,
    )
    return {
        "adapter": adapter_path,
        "evidence": evidence,
        "pipeline": pipeline,
        "resume": resume,
        "resume_checkpoint": resume_checkpoint,
        "step1": step1,
        "step1_checkpoint": step1_checkpoint,
    }


def test_artifact_evidence_binds_resume_checkpoint_adapter_and_nonzero_update(tmp_path):
    fixture = _artifact_fixture(tmp_path)
    evidence = fixture["evidence"]

    assert evidence["adapter_update_nonzero"] is True
    assert evidence["base_model_unchanged"] is True
    assert evidence["checkpoint"]["global_step"] == 5
    assert evidence["predecessor"]["checkpoint_step"] == 1
    unsigned = dict(evidence)
    digest = unsigned.pop("artifact_evidence_sha256")
    assert digest == aggregate._canonical_sha256(unsigned)


def test_artifact_evidence_rejects_zero_adapter_update(tmp_path):
    with pytest.raises(aggregate.CapacityAggregateError, match="adapter update is zero"):
        _artifact_fixture(tmp_path, initial_hash="3" * 64, final_hash="3" * 64)


def _index_fixture(tmp_path: Path):
    root = tmp_path / "resolved-configs"
    data = tmp_path / "data"
    root.mkdir()
    data.mkdir()
    records = {}
    for config_id in sorted(aggregate._ALL_CONFIG_IDS):
        path = root / f"{config_id}.yaml"
        path.write_text(f"name: {config_id}\n", encoding="ascii")
        if config_id in aggregate._GATE_CONFIG_IDS | aggregate._EVAL_CONFIG_IDS:
            profile = None
            source = config_id
        else:
            profile = config_id.rsplit("_", 1)[-1]
            source = config_id.removesuffix(f"_{profile}")
        records[config_id] = {
            "offload_profile": profile,
            "overrides": [],
            "path": str(path),
            "sha256": _sha(path),
            "source_config": source,
        }
    index = {
        "configs": records,
        "data_root": str(data),
        "schema_version": 1,
        "status": "resolved",
    }
    index_path = _write_json(root / "index.json", index, canonical=False)
    return index_path, index


def _handoff_fixture(path: Path, index_path: Path):
    unsigned = {
        "asset_manifest": {},
        "asset_report": {},
        "bundles": {},
        "config_files": {
            "index.json": {"sha256": _sha(index_path), "size": index_path.stat().st_size}
        },
        "config_root": str(index_path.parent),
        "config_tree_sha256": "d" * 64,
        "data_root": str(index_path.parent.parent / "data"),
        "environment_lock": {},
        "environment_lock_sha256": "e" * 64,
        "experiment_profile_id": aggregate.PROFILE_ID,
        "git_commit": COMMIT,
        "kernel_source_root": str(index_path.parent),
        "kernel_sources": {},
        "pip_freeze": {},
        "persist_root": str(path.parent),
        "schema_version": 3,
        "status": "cpu_ready",
    }
    handoff = {**unsigned, "handoff_sha256": aggregate._canonical_sha256(unsigned)}
    _write_json(path, handoff)
    return handoff


def _gpu_fixture(pipeline: Path, profile="r0", host_memory_gib=128):
    evidence_path = pipeline / "gpu-evidence" / "causal-conv1d" / "bf16-forward.json"
    hardware = {
        "capacity_profile": profile.upper(),
        "cuda_runtime": "13.0",
        "cuda_toolkit": "13.0",
        "driver_version": "999.1",
        "gpu_compute_capability": [12, 0],
        "gpu_free_memory_bytes": 30 * GIB,
        "gpu_name": "NVIDIA GeForce RTX 5090",
        "gpu_total_memory_bytes": 32 * GIB,
        "gpu_uuid": "GPU-test-uuid",
        "host_cpu_count": 32,
        "host_total_memory_bytes": host_memory_gib * GIB,
        "minimum_persistent_disk_free_bytes": 128 * GIB,
        "other_compute_process_count": 0,
        "persistent_disk_free_bytes": 250 * GIB,
        "persistent_disk_probe_path": str(pipeline),
        "torch_gpu_total_memory_bytes": 32 * GIB,
    }
    _write_json(
        evidence_path,
        {
            "hardware": hardware,
            "kernel": "causal-conv1d",
            "losses": [],
            "mode": "forward",
            "optimizer_steps": 0,
            "shape": [1, 64, 512],
            "status": "verified",
        },
        canonical=False,
    )
    kernel_record = {
        "bf16_backward_log_sha256": "3" * 64,
        "bf16_forward_log_sha256": "4" * 64,
        "build_log_sha256": "5" * 64,
        "commit": "6" * 40,
        "optimizer_loop_log_sha256": "7" * 64,
        "repository": "https://example.invalid/kernel.git",
        "status": "VERIFIED_SM120",
    }
    causal_record = dict(kernel_record)
    causal_record["bf16_forward_log_sha256"] = _sha(evidence_path)
    unsigned = {
        "environment_lock_sha256": "1" * 64,
        "kernels": {
            "causal-conv1d": causal_record,
            "flash-linear-attention": kernel_record,
        },
        "packages": {},
        "pip_freeze_sha256": "2" * 64,
        "python": "3.12.0",
        "schema_version": 1,
        "status": "VERIFIED_SM120",
        "system": {
            "cuda_runtime": "13.0",
            "driver_version": "999.1",
            "gpu_compute_capability": [12, 0],
            "gpu_name": "NVIDIA GeForce RTX 5090",
            "operating_system": "test-linux",
        },
        "training_gate": "READY",
    }
    build = {**unsigned, "build_info_sha256": aggregate._canonical_sha256(unsigned)}
    _write_json(pipeline / "build-info.json", build, canonical=False)
    return evidence_path


def test_gpu_evidence_enforces_profile_specific_host_ram(tmp_path):
    accepted = _gpu_fixture(tmp_path / "accepted", "r0", host_memory_gib=80)
    _, _, hardware = aggregate._load_gpu_evidence(accepted, "r0")
    assert hardware["host_total_memory_bytes"] == 80 * GIB

    rejected_r0 = _gpu_fixture(tmp_path / "rejected-r0", "r0", host_memory_gib=79)
    with pytest.raises(aggregate.CapacityAggregateError, match="GPU host total memory"):
        aggregate._load_gpu_evidence(rejected_r0, "r0")

    rejected_r1 = _gpu_fixture(tmp_path / "rejected-r1", "r1", host_memory_gib=127)
    with pytest.raises(aggregate.CapacityAggregateError, match="GPU host total memory"):
        aggregate._load_gpu_evidence(rejected_r1, "r1")


def _scientific(**overrides):
    value = {
        "all_outputs_truncated": False,
        "finite_gradients": True,
        "finite_losses": True,
        "high_truncation_rate": False,
        "nonzero_advantage_groups": 1,
        "reward_variance_groups": 1,
        "systematic_format_failure": False,
    }
    value.update(overrides)
    return value


def _step(step, config_id, config_sha, attempt_id, *, scientific):
    actor_logits = {
        "bytes": 96,
        "dtype": "bfloat16",
        "element_size": 2,
        "numel": 48,
        "shape": [1, 3, 16],
    }
    post_allocated = 20 * GIB + step * 16 * 1024**2
    post_reserved = 21 * GIB + step * 16 * 1024**2
    phase_telemetry = [
        {
            "actor_peak_allocated_bytes": 24 * GIB,
            "actor_peak_reserved_bytes": 24 * GIB,
            "actor_post_allocated_bytes": post_allocated,
            "actor_post_reserved_bytes": post_reserved,
            "duration_seconds": float(index + 1),
            "host_peak_used_bytes": 60 * GIB,
            "name": name,
            "nvml_peak_used_bytes": 28 * GIB,
            "reference_peak_allocated_bytes": 0,
            "reference_peak_reserved_bytes": 0,
            "reference_post_allocated_bytes": 0,
            "reference_post_reserved_bytes": 0,
            "swap_peak_used_bytes": 0,
        }
        for index, name in enumerate(aggregate._TELEMETRY_PHASES)
    ]
    return {
        "actor_logits": actor_logits,
        "allocator_retry_count": 0,
        "attempt_id": attempt_id,
        "global_step": step,
        "gpu_uuid": "GPU-test-uuid",
        "host_peak_used_bytes": 60 * GIB,
        "host_total_memory_bytes": 128 * GIB,
        "kind": "rememr1-training-step-telemetry-v2",
        "nvml_peak_used_bytes": 28 * GIB,
        "nvml_total_bytes": 32 * GIB,
        "peak_allocated_bytes": 24 * GIB,
        "peak_reserved_bytes": 24 * GIB,
        "phase_telemetry": phase_telemetry,
        "post_step_allocated_bytes": post_allocated,
        "post_step_nvml_used_bytes": 22 * GIB + step * 16 * 1024**2,
        "post_step_reserved_bytes": post_reserved,
        "record_sha256": "5" * 64,
        "reference_logits": actor_logits,
        "schema_version": 2,
        "scientific_evidence": scientific,
        "sealed_config_id": config_id,
        "sealed_config_sha256": config_sha,
        "step_wall_seconds": 10.0,
        "swap_used_bytes": 0,
        "timing_seconds": {"save_checkpoint": 1.0, "step": 9.0},
        "worker_roles": ["actor", "reference"],
    }


def _aggregate_fixture(tmp_path: Path, profile="r0", host_memory_gib=128):
    pipeline = tmp_path / "pipeline"
    generation = "base" if profile == "r0" else f"approval-{'1' * 64}-budget-{'f' * 64}"
    stages = pipeline / "stages" / "gpu-capacity" / profile / generation
    stages.mkdir(parents=True)
    runs = tmp_path / "runs"
    index_path, index = _index_fixture(tmp_path)
    handoff = _handoff_fixture(pipeline / "cpu-handoff.json", index_path)
    gpu_path = _gpu_fixture(pipeline, profile, host_memory_gib=host_memory_gib)
    specs = {
        "g2a": ("g2a", "g2a.run", (1,)),
        "g2b-step1": ("g2b-step1", "g2b-step1.run", (1,)),
        "g2b-resume5": ("g2b-resume5", "g2b-resume5.run", (2, 3, 4, 5)),
        "length-stress": ("g2-length-stress", "g2-length-stress.run", (1,)),
    }
    run_by_stage = {}
    runtime_by_run = {}
    ledger_by_run = {}
    for logical, (runtime_stage, pointer_name, steps) in specs.items():
        run = runs / f"attempt-{runtime_stage}"
        _run_meta(run, runtime_stage, pipeline, profile)
        (run / "runtime-bound").mkdir()
        config_id = f"{aggregate.capacity_evidence.ATTEMPT_CONFIG_TASK[logical]}_{profile}"
        config_sha = index["configs"][config_id]["sha256"]
        science = (
            _scientific(
                all_outputs_truncated=True,
                finite_gradients=False,
                finite_losses=False,
                high_truncation_rate=True,
                systematic_format_failure=True,
            )
            if logical == "length-stress"
            else _scientific()
        )
        ledger_steps = [
            _step(step, config_id, config_sha, run.name, scientific=science)
            for step in steps
        ]
        ledger_by_run[run] = {
            "identity": {
                "attempt_id": run.name,
                "experiment_profile_id": aggregate.PROFILE_ID,
                "gpu_uuid": "GPU-test-uuid",
                "offload_profile": profile,
                "sealed_config_id": config_id,
                "sealed_config_sha256": config_sha,
            },
            "ledger_sha256": hashlib.sha256(logical.encode()).hexdigest(),
            "steps": ledger_steps,
        }
        runtime_by_run[run] = {
            "attempt_id": run.name,
            "attempt_root": str(run),
            "config_id": config_id,
            "evidence_sha256": hashlib.sha256((logical + "runtime").encode()).hexdigest(),
            "resume_predecessor": None,
            "runtime_bound_config_sha256": "6" * 64,
            "source_config": {
                "path": str(index_path.parent / f"{config_id}.yaml"),
                "sha256": config_sha,
            },
            "source_index": {"path": str(index_path), "sha256": _sha(index_path)},
        }
        _write_json(stages / pointer_name, {})
        (stages / pointer_name).write_bytes((str(run) + "\n").encode("ascii"))
        run_by_stage[logical] = run
    step1 = run_by_stage["g2b-step1"]
    resume = run_by_stage["g2b-resume5"]
    step1_checkpoint = step1 / "checkpoints" / "global_step_1"
    resume_checkpoint = resume / "checkpoints" / "global_step_5"
    adapter_path = resume / "artifacts" / "adapter" / "global_step_5" / "adapter"
    for path in (step1_checkpoint, resume_checkpoint, adapter_path):
        path.mkdir(parents=True)
    step1_id = f"g2b_qwen35_2b_5090_step1_{profile}"
    runtime_by_run[resume]["resume_predecessor"] = {
        "checkpoint_dir": str(step1_checkpoint),
        "checkpoint_step": 1,
        "logical_config_id": step1_id,
        "resolved_config_sha256": index["configs"][step1_id]["sha256"],
    }
    artifact_unsigned = {
        "adapter": {
            "initial_state_sha256": "1" * 64,
            "metadata_sha256": "2" * 64,
            "path": str(adapter_path),
            "state_sha256": "3" * 64,
            "tensor_keys_sha256": "4" * 64,
        },
        "adapter_update_nonzero": True,
        "attempt_id": resume.name,
        "base_model": {"id": "Qwen/Qwen3.5-2B", "revision": MODEL_REVISION},
        "base_model_unchanged": True,
        "budget_projection_sha256": "f" * 64 if profile == "r1" else None,
        "checkpoint": {
            "completion_sha256": "5" * 64,
            "extra_state_sha256": "6" * 64,
            "global_step": 5,
            "path": str(resume_checkpoint),
        },
        "commit": COMMIT,
        "config_id": f"g2b_qwen35_2b_5090_resume5_{profile}",
        "config_sha256": index["configs"][f"g2b_qwen35_2b_5090_resume5_{profile}"]["sha256"],
        "experiment_profile_id": aggregate.PROFILE_ID,
        "kind": aggregate.ARTIFACT_KIND,
        "offload_profile": profile,
        "pipeline_dir": str(pipeline),
        "predecessor": {
            "attempt_id": step1.name,
            "checkpoint_path": str(step1_checkpoint),
            "checkpoint_step": 1,
            "completion_sha256": "7" * 64,
            "config_id": step1_id,
            "config_sha256": index["configs"][step1_id]["sha256"],
            "extra_state_sha256": "8" * 64,
        },
        "runtime_bound": {
            "config_sha256": "9" * 64,
            "evidence_sha256": "a" * 64,
            "path": str(resume / "runtime-bound" / "runtime-bound.json"),
        },
        "schema_version": 1,
        "status": "verified",
    }
    artifact = {
        **artifact_unsigned,
        "artifact_evidence_sha256": aggregate._canonical_sha256(artifact_unsigned),
    }
    artifact_path = _write_json(pipeline / "artifact-evidence.json", artifact)
    kwargs = {
        "handoff_path": pipeline / "cpu-handoff.json",
        "index_path": index_path,
        "gpu_evidence_path": gpu_path,
        "profile": profile,
        "g2a_pointer": stages / "g2a.run",
        "g2b_step1_pointer": stages / "g2b-step1.run",
        "g2b_resume5_pointer": stages / "g2b-resume5.run",
        "length_stress_pointer": stages / "g2-length-stress.run",
        "artifact_evidence_path": artifact_path,
        "handoff_verifier": lambda path, commit: handoff,
        "index_loader": lambda path, digest: index,
        "runtime_verifier": lambda path: runtime_by_run[path.parent],
        "ledger_loader": lambda path: ledger_by_run[path.parent],
        "artifact_revalidator": lambda checkpoint, adapter: artifact,
    }
    return kwargs, ledger_by_run


def test_aggregate_publishes_exact_capacity_inputs_and_excludes_stress_science(tmp_path):
    kwargs, _ = _aggregate_fixture(tmp_path)
    outputs = aggregate.aggregate_capacity(**kwargs)

    assert set(outputs) == {
        "attempt-metadata.json",
        "identity.json",
        "selected-configs.json",
        "telemetry.json",
    }
    telemetry = outputs["telemetry.json"]
    assert len(telemetry["steps"]) == 7
    assert telemetry["finite_gradients"] is True
    assert telemetry["finite_losses"] is True
    assert telemetry["all_outputs_truncated"] is False
    assert telemetry["systematic_format_failure"] is False
    assert telemetry["high_truncation_rate"] is False
    assert telemetry["nonzero_advantage_groups"] == 6
    assert [step["step"] for step in telemetry["steps"] if step["step"] is not None] == [
        2,
        3,
        4,
        5,
    ]
    assert outputs["identity.json"]["selected_profile"] == "R0"
    assert len(outputs["selected-configs.json"]) == 14


def test_aggregate_rejects_missing_logits_and_incomplete_phase_evidence(tmp_path):
    kwargs, ledgers = _aggregate_fixture(tmp_path / "missing-logits")
    next(iter(ledgers.values()))["steps"][0]["reference_logits"] = None
    with pytest.raises(aggregate.CapacityAggregateError, match="reference logits"):
        aggregate.aggregate_capacity(**kwargs)

    kwargs, ledgers = _aggregate_fixture(tmp_path / "missing-phase")
    next(iter(ledgers.values()))["steps"][0]["phase_telemetry"].pop()
    with pytest.raises(aggregate.CapacityAggregateError, match="phase inventory"):
        aggregate.aggregate_capacity(**kwargs)

    kwargs, ledgers = _aggregate_fixture(tmp_path / "reference-resources")
    next(iter(ledgers.values()))["steps"][0]["phase_telemetry"][0][
        "reference_peak_allocated_bytes"
    ] = 1
    with pytest.raises(aggregate.CapacityAggregateError, match="placeholders must be zero"):
        aggregate.aggregate_capacity(**kwargs)


def test_aggregate_fails_on_missing_random_science_and_ambiguous_terminal(tmp_path):
    kwargs, ledgers = _aggregate_fixture(tmp_path)
    g2a_run = Path((Path(kwargs["g2a_pointer"])).read_text(encoding="ascii").strip())
    ledgers[g2a_run]["steps"][0]["scientific_evidence"] = None
    with pytest.raises(aggregate.CapacityAggregateError, match="scientific evidence"):
        aggregate.aggregate_capacity(**kwargs)

    ledgers[g2a_run]["steps"][0]["scientific_evidence"] = _scientific()
    (g2a_run / ".failed").write_bytes(b"1\n")
    with pytest.raises(aggregate.CapacityAggregateError, match="terminal marker"):
        aggregate.aggregate_capacity(**kwargs)


def test_pointer_must_be_one_canonical_line(tmp_path):
    kwargs, _ = _aggregate_fixture(tmp_path)
    pointer = Path(kwargs["g2a_pointer"])
    pointer.write_bytes(pointer.read_bytes() + b"extra\n")
    with pytest.raises(aggregate.CapacityAggregateError, match="exactly one line"):
        aggregate.aggregate_capacity(**kwargs)


def test_target_identity_cli_builds_r1_authority_before_r1_attempts(
    tmp_path, monkeypatch, capsys
):
    kwargs, _ = _aggregate_fixture(tmp_path)
    handoff = json.loads(Path(kwargs["handoff_path"]).read_text(encoding="ascii"))
    index = json.loads(Path(kwargs["index_path"]).read_text(encoding="ascii"))
    monkeypatch.setattr(
        aggregate, "_default_handoff_verifier", lambda path, commit: handoff
    )
    monkeypatch.setattr(
        aggregate, "_default_index_loader", lambda path, digest: index
    )
    output = tmp_path / "r1-target-identity.json"

    assert aggregate.main(
        [
            "target-identity",
            "--handoff",
            str(kwargs["handoff_path"]),
            "--index",
            str(kwargs["index_path"]),
            "--gpu-evidence",
            str(kwargs["gpu_evidence_path"]),
            "--profile",
            "r1",
            "--output",
            str(output),
        ]
    ) == 0
    message = json.loads(capsys.readouterr().out)
    identity = json.loads(output.read_text(encoding="ascii"))
    assert identity["selected_profile"] == "R1"
    assert identity["gpu"]["uuid"] == "GPU-test-uuid"
    assert message["selected_config_set_sha256"] == identity[
        "selected_config_set_sha256"
    ]
    assert aggregate.main(
        [
            "target-identity",
            "--handoff",
            str(kwargs["handoff_path"]),
            "--index",
            str(kwargs["index_path"]),
            "--gpu-evidence",
            str(kwargs["gpu_evidence_path"]),
            "--profile",
            "r1",
            "--output",
            str(output),
        ]
    ) == 0
    assert output.read_bytes() == _canonical(identity) + b"\n"


def _capacity_stop_fixture(
    tmp_path: Path,
    stopped_stage="g2a",
    profile="r0",
    host_memory_gib=128,
):
    kwargs, ledgers = _aggregate_fixture(
        tmp_path,
        profile,
        host_memory_gib=host_memory_gib,
    )
    runtime_stage = (
        "g2-length-stress" if stopped_stage == "length-stress" else stopped_stage
    )
    pointer_key = {
        "g2a": "g2a_pointer",
        "g2b-step1": "g2b_step1_pointer",
        "g2b-resume5": "g2b_resume5_pointer",
        "length-stress": "length_stress_pointer",
    }[stopped_stage]
    run = Path(Path(kwargs[pointer_key]).read_text(encoding="ascii").strip())
    (run / ".success").unlink()
    (run / ".capacity-stop").write_bytes(b"43\n")
    (run / "retryable").write_bytes(b"false\n")
    pipeline = Path(kwargs["handoff_path"]).parent
    stopped_pointer = (
        pipeline
        / "attempts"
        / "gpu-capacity"
        / profile
        / Path(kwargs[pointer_key]).parent.name
        / f"{runtime_stage}-20260717T000000Z-1.run"
    )
    stopped_pointer.parent.mkdir(parents=True, exist_ok=True)
    stopped_pointer.write_bytes((str(run) + "\n").encode("ascii"))

    runtime = kwargs["runtime_verifier"](run / "runtime-bound")
    binding_path = run / "runtime-binding-source.json"
    binding = {"binding_sha256": "c" * 64, "status": "bound"}
    _write_json(binding_path, binding)
    runtime["source_binding"] = {
        "binding_sha256": "c" * 64,
        "file_sha256": _sha(binding_path),
        "path": str(binding_path),
    }
    runtime_path = run / "runtime-bound" / "runtime-bound.json"
    _write_json(runtime_path, runtime)
    marker_unsigned = {
        "attempt_id": run.name,
        "attempt_root": str(run),
        "config_id": runtime["config_id"],
        "exception_type": "torch.OutOfMemoryError",
        "kind": aggregate.CAPACITY_STOP_KIND,
        "reason": "cuda_oom",
        "runtime_binding": {
            "binding_sha256": "c" * 64,
            "file_sha256": _sha(binding_path),
            "path": str(binding_path),
        },
        "runtime_bound_evidence": {
            "evidence_sha256": runtime["evidence_sha256"],
            "file_sha256": _sha(runtime_path),
            "path": str(runtime_path),
        },
        "schema_version": 1,
        "status": "capacity-stop",
    }
    marker = {
        **marker_unsigned,
        "capacity_stop_sha256": aggregate._canonical_sha256(marker_unsigned),
    }
    marker_path = _write_json(run / "evidence" / "capacity-stop.json", marker)
    ordered = ["g2a", "g2b-step1", "g2b-resume5", "length-stress"]
    stop_index = ordered.index(stopped_stage)
    stop_kwargs = {
        "handoff_path": kwargs["handoff_path"],
        "index_path": kwargs["index_path"],
        "gpu_evidence_path": kwargs["gpu_evidence_path"],
        "profile": profile,
        "stopped_stage": stopped_stage,
        "stopped_pointer": stopped_pointer,
        "g2a_pointer": kwargs["g2a_pointer"] if stop_index > 0 else None,
        "g2b_step1_pointer": (
            kwargs["g2b_step1_pointer"] if stop_index > 1 else None
        ),
        "g2b_resume5_pointer": (
            kwargs["g2b_resume5_pointer"] if stop_index > 2 else None
        ),
        "handoff_verifier": kwargs["handoff_verifier"],
        "index_loader": kwargs["index_loader"],
        "runtime_verifier": kwargs["runtime_verifier"],
        "ledger_loader": kwargs["ledger_loader"],
    }
    return stop_kwargs, ledgers, marker_path, run


def test_capacity_stop_finalizes_only_trusted_r0_oom(tmp_path):
    kwargs, _, _, _ = _capacity_stop_fixture(tmp_path)
    outputs = aggregate.aggregate_capacity_stop(**kwargs)

    attempt = outputs["attempt-metadata.json"]["attempts"]["g2a"]
    telemetry = outputs["telemetry.json"]
    assert attempt["status"] == "failed"
    assert attempt["failure_kind"] == "capacity"
    assert telemetry["oom"] is True
    assert telemetry["terminal_reason"] == "oom"
    assert telemetry["non_capacity_failure"] is False
    evidence = aggregate.capacity_evidence.create_capacity_evidence(
        identity=outputs["identity.json"],
        telemetry=telemetry,
        attempt_metadata=outputs["attempt-metadata.json"],
        selected_configs=outputs["selected-configs.json"],
    )
    assert evidence["classification"]["r1_eligible"] is True
    assert evidence["classification"]["eligibility_reason"] == "oom"


def test_capacity_stop_without_step_records_accepts_r0_80_gib_hardware(tmp_path):
    kwargs, _, _, _ = _capacity_stop_fixture(tmp_path, host_memory_gib=80)
    outputs = aggregate.aggregate_capacity_stop(**kwargs)

    telemetry = outputs["telemetry.json"]
    assert telemetry["steps"] == []
    assert telemetry["host_total_memory_bytes"] == 80 * GIB


def test_r1_capacity_stop_is_canonical_but_never_authorizes_another_profile(
    short_tmp_path,
):
    kwargs, _, _, _ = _capacity_stop_fixture(short_tmp_path, profile="r1")
    outputs = aggregate.aggregate_capacity_stop(**kwargs)

    assert outputs["identity.json"]["selected_profile"] == "R1"
    assert outputs["telemetry.json"]["oom"] is True
    evidence = aggregate.capacity_evidence.create_capacity_evidence(
        identity=outputs["identity.json"],
        telemetry=outputs["telemetry.json"],
        attempt_metadata=outputs["attempt-metadata.json"],
        selected_configs=outputs["selected-configs.json"],
    )
    assert evidence["classification"]["r1_eligible"] is False
    assert evidence["classification"]["eligibility_reason"] is None


def test_capacity_stop_rejects_forged_reason_and_ordinary_failure(tmp_path):
    kwargs, _, marker_path, run = _capacity_stop_fixture(tmp_path)
    marker = json.loads(marker_path.read_text(encoding="ascii"))
    marker["exception_type"] = "builtins.RuntimeError"
    unsigned = dict(marker)
    unsigned.pop("capacity_stop_sha256")
    marker["capacity_stop_sha256"] = aggregate._canonical_sha256(unsigned)
    _write_json(marker_path, marker)
    with pytest.raises(aggregate.CapacityAggregateError, match="identity/reason"):
        aggregate.aggregate_capacity_stop(**kwargs)

    marker["exception_type"] = "torch.OutOfMemoryError"
    unsigned = dict(marker)
    unsigned.pop("capacity_stop_sha256")
    marker["capacity_stop_sha256"] = aggregate._canonical_sha256(unsigned)
    _write_json(marker_path, marker)
    (run / ".capacity-stop").unlink()
    (run / ".failed").write_bytes(b"43\n")
    with pytest.raises(aggregate.CapacityAggregateError, match=r"\.capacity-stop"):
        aggregate.aggregate_capacity_stop(**kwargs)


def test_capacity_stop_does_not_authorize_r1_when_prior_science_failed(tmp_path):
    kwargs, ledgers, _, _ = _capacity_stop_fixture(tmp_path, "g2b-step1")
    g2a_run = Path(Path(kwargs["g2a_pointer"]).read_text(encoding="ascii").strip())
    ledgers[g2a_run]["steps"][0]["scientific_evidence"][
        "high_truncation_rate"
    ] = True

    outputs = aggregate.aggregate_capacity_stop(**kwargs)
    telemetry = outputs["telemetry.json"]
    assert telemetry["non_capacity_failure"] is True
    evidence = aggregate.capacity_evidence.create_capacity_evidence(
        identity=outputs["identity.json"],
        telemetry=telemetry,
        attempt_metadata=outputs["attempt-metadata.json"],
        selected_configs=outputs["selected-configs.json"],
    )
    assert evidence["classification"]["r1_eligible"] is False
