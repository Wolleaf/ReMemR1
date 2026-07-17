from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from scripts.cloud import training_telemetry as telemetry
from scripts.cloud.training_telemetry import (
    REQUIRED_PHASES,
    TrainingScientificStop,
    TrainingTelemetryError,
    append_step,
    create_step_record,
    load_ledger,
    verify_success_ledger,
)


def _worker(role="actor", **overrides):
    is_actor = role == "actor"
    peak_allocated = overrides.get(
        "peak_allocated_bytes",
        (18 if is_actor else 0) * 1024**3,
    )
    peak_reserved = overrides.get(
        "peak_reserved_bytes",
        (19 if is_actor else 0) * 1024**3,
    )
    phase_records = [
        {
            "duration_seconds": float(index + 1) if is_actor else 0.0,
            "host_peak_used_bytes": (40 if is_actor else 0) * 1024**3,
            "name": name,
            "nvml_peak_used_bytes": ((20 + index) if is_actor else 0) * 1024**3,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "post_allocated_bytes": (12 if is_actor else 0) * 1024**3,
            "post_reserved_bytes": (13 if is_actor else 0) * 1024**3,
            "swap_peak_used_bytes": 0,
        }
        for index, name in enumerate(REQUIRED_PHASES)
    ]
    logits = {
        "bytes": 96,
        "dtype": "bfloat16",
        "element_size": 2,
        "numel": 48,
        "shape": [1, 3, 16],
    }
    value = {
        "actor_logits": logits if role == "actor" else None,
        "allocator_retry_count": 0,
        "gpu_uuid": "GPU-5090-test",
        "host_peak_used_bytes": (40 if is_actor else 0) * 1024**3,
        "host_total_memory_bytes": 128 * 1024**3,
        "nvml_peak_used_bytes": (20 if is_actor else 0) * 1024**3,
        "nvml_total_bytes": 32 * 1024**3,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "phase_records": phase_records,
        "post_step_allocated_bytes": (12 if is_actor else 0) * 1024**3,
        "post_step_nvml_used_bytes": (14 if is_actor else 0) * 1024**3,
        "post_step_reserved_bytes": (13 if is_actor else 0) * 1024**3,
        "rank": 0,
        "reference_logits": logits if role == "reference" else None,
        "role": role,
        "step_wall_seconds": 123.5 if is_actor else 0.0,
        "swap_used_bytes": 0,
        "world_size": 1,
    }
    value.update(overrides)
    return value


def _record(step=1, scientific=False, **worker_overrides):
    return create_step_record(
        worker_records=[
            _worker("actor", **worker_overrides),
            _worker("reference"),
        ],
        global_step=step,
        attempt_id="g2b-resume5-r0-attempt-1",
        sealed_config_id="g2b_qwen35_2b_5090_resume5_r0",
        sealed_config_sha256="a" * 64,
        offload_profile="r0",
        timing_seconds={
            "gen": 100.0,
            "save_checkpoint": 10.0,
            "update_actor": 20.0,
        },
        scientific_evidence=(
            {
                "all_outputs_truncated": False,
                "finite_gradients": True,
                "finite_losses": True,
                "high_truncation_rate": False,
                "nonzero_advantage_groups": 2,
                "reward_variance_groups": 2,
                "systematic_format_failure": False,
            }
            if scientific
            else None
        ),
    )


def test_step_ledger_is_canonical_self_hashed_and_preserves_actual_logits(tmp_path):
    path = tmp_path / "attempt" / "telemetry.json"
    append_step(path, _record(2))
    append_step(path, _record(3, peak_allocated_bytes=19 * 1024**3))

    ledger = load_ledger(path)
    assert [step["global_step"] for step in ledger["steps"]] == [2, 3]
    assert ledger["steps"][0]["actor_logits"] == {
        "bytes": 96,
        "dtype": "bfloat16",
        "element_size": 2,
        "numel": 48,
        "shape": [1, 3, 16],
    }
    assert ledger["steps"][0]["reference_logits"] == ledger["steps"][0][
        "actor_logits"
    ]
    assert ledger["steps"][0]["peak_allocated_bytes"] == 18 * 1024**3
    assert all(
        phase["reference_peak_allocated_bytes"] == 0
        for phase in ledger["steps"][0]["phase_telemetry"]
    )
    assert [phase["name"] for phase in ledger["steps"][0]["phase_telemetry"]] == list(
        REQUIRED_PHASES
    )
    assert path.read_bytes() == json.dumps(
        ledger,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii") + b"\n"


def test_telemetry_rejects_multi_rank_identity_drift_and_tampering(tmp_path):
    with pytest.raises(TrainingTelemetryError, match="rank/world size mismatch"):
        create_step_record(
            worker_records=[
                _worker("actor"),
                _worker("reference", rank=1, world_size=2),
            ],
            global_step=1,
            attempt_id="attempt",
            sealed_config_id="config_r0",
            sealed_config_sha256="a" * 64,
            offload_profile="r0",
            timing_seconds={"save_checkpoint": 1.0, "step": 1.0},
        )

    path = tmp_path / "telemetry.json"
    append_step(path, _record())
    value = json.loads(path.read_text(encoding="ascii"))
    value["steps"][0]["peak_allocated_bytes"] += 1
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="ascii")
    with pytest.raises(TrainingTelemetryError, match="self-hash mismatch"):
        load_ledger(path)


def test_telemetry_append_requires_increasing_step_and_stable_attempt(tmp_path):
    path = tmp_path / "telemetry.json"
    append_step(path, _record(2))
    with pytest.raises(TrainingTelemetryError, match="strictly increasing"):
        append_step(path, _record(2))
    drift = _record(3)
    drift["attempt_id"] = "other-attempt"
    unsigned = dict(drift)
    unsigned.pop("record_sha256")
    # A semantic edit without a corresponding self-hash is rejected first.
    with pytest.raises(TrainingTelemetryError, match="self-hash mismatch"):
        append_step(path, drift)


def test_scientific_step_evidence_is_bound_into_the_ledger(tmp_path):
    path = tmp_path / "telemetry.json"
    append_step(path, _record(scientific=True))

    evidence = load_ledger(path)["steps"][0]["scientific_evidence"]
    assert evidence["finite_gradients"] is True
    assert evidence["finite_losses"] is True
    assert evidence["nonzero_advantage_groups"] == 2
    assert evidence["reward_variance_groups"] == 2


def test_strict_success_verifies_identity_final_step_and_science(tmp_path):
    attempt = tmp_path / "g2b-resume5-r0-attempt-1"
    append_step(attempt / "telemetry.json", _record(5, scientific=True))

    ledger = verify_success_ledger(
        attempt,
        expected_config_id="g2b_qwen35_2b_5090_resume5_r0",
        expected_config_sha256="a" * 64,
        expected_offload_profile="r0",
        expected_final_step=5,
    )
    assert ledger["steps"][-1]["global_step"] == 5

    with pytest.raises(TrainingTelemetryError, match="differs at sealed_config_id"):
        verify_success_ledger(
            attempt,
            expected_config_id="other-config",
            expected_config_sha256="a" * 64,
            expected_offload_profile="r0",
            expected_final_step=5,
        )


def test_strict_success_rejects_rehashed_false_scientific_evidence(tmp_path):
    attempt = tmp_path / "g2b-resume5-r0-attempt-1"
    path = attempt / "telemetry.json"
    append_step(path, _record(5, scientific=True))
    value = json.loads(path.read_text(encoding="ascii"))
    record = value["steps"][0]
    record["scientific_evidence"]["finite_losses"] = False
    unsigned_record = dict(record)
    unsigned_record.pop("record_sha256")
    record["record_sha256"] = telemetry._canonical_sha256(unsigned_record)
    unsigned_ledger = dict(value)
    unsigned_ledger.pop("ledger_sha256")
    value["ledger_sha256"] = telemetry._canonical_sha256(unsigned_ledger)
    path.write_bytes(telemetry._canonical_bytes(value) + b"\n")

    with pytest.raises(TrainingScientificStop, match="non-finite losses"):
        verify_success_ledger(
            attempt,
            expected_config_id="g2b_qwen35_2b_5090_resume5_r0",
            expected_config_sha256="a" * 64,
            expected_offload_profile="r0",
            expected_final_step=5,
        )
    assert telemetry.main(
        [
            "verify-success",
            "--attempt-dir",
            str(attempt),
            "--expected-config-id",
            "g2b_qwen35_2b_5090_resume5_r0",
            "--expected-config-sha256",
            "a" * 64,
            "--expected-offload-profile",
            "r0",
            "--expected-final-step",
            "5",
        ]
    ) == 42


def test_length_stress_checks_resources_but_excludes_scientific_result(tmp_path):
    attempt = tmp_path / "length-stress-attempt"
    record = create_step_record(
        worker_records=[_worker("actor"), _worker("reference")],
        global_step=1,
        attempt_id=attempt.name,
        sealed_config_id="g2_length_stress_qwen35_2b_5090_r0",
        sealed_config_sha256="b" * 64,
        offload_profile="r0",
        timing_seconds={"save_checkpoint": 0.0, "step": 1.0},
        scientific_evidence={
            "all_outputs_truncated": True,
            "finite_gradients": False,
            "finite_losses": False,
            "high_truncation_rate": True,
            "nonzero_advantage_groups": 0,
            "reward_variance_groups": 0,
            "systematic_format_failure": True,
        },
    )
    append_step(attempt / "telemetry.json", record)

    ledger = verify_success_ledger(
        attempt,
        expected_config_id="g2_length_stress_qwen35_2b_5090_r0",
        expected_config_sha256="b" * 64,
        expected_offload_profile="r0",
        expected_final_step=1,
        length_stress=True,
    )
    assert ledger["steps"][0]["scientific_evidence"]["finite_losses"] is False


def test_worker_and_trainer_wire_reset_collect_around_the_optimizer_loop():
    root = Path(__file__).resolve().parents[2]
    worker_tree = ast.parse(
        (root / "verl" / "workers" / "fsdp_workers.py").read_text(encoding="utf-8")
    )
    worker_class = next(
        node
        for node in worker_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker"
    )
    worker_methods = {
        node.name
        for node in worker_class.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "reset_reproduction_step_telemetry" in worker_methods
    assert "advance_reproduction_step_telemetry" in worker_methods
    assert "collect_reproduction_step_telemetry" in worker_methods
    assert "reset_reproduction_reference_logits" in worker_methods
    assert "collect_reproduction_reference_logits" in worker_methods

    trainer_tree = ast.parse(
        (root / "verl" / "trainer" / "ppo" / "ray_trainer.py").read_text(
            encoding="utf-8"
        )
    )
    trainer = next(
        node
        for node in trainer_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer"
    )
    fit = next(
        node
        for node in trainer.body
        if isinstance(node, ast.FunctionDef) and node.name == "fit"
    )

    def calls(name):
        return [
            node
            for node in ast.walk(fit)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == name
        ]

    resets = calls("reset_reproduction_step_telemetry")
    reference_resets = calls("reset_reproduction_reference_logits")
    advances = calls("advance_reproduction_step_telemetry")
    update = calls("update_actor")[0]
    save = calls("_save_checkpoint")[0]
    collects = calls("collect_reproduction_step_telemetry")
    reference_collects = calls("collect_reproduction_reference_logits")
    reference_forward = calls("compute_ref_log_prob")[0]
    assert len(resets) == 1
    assert len(reference_resets) == 1
    assert len(advances) == 1
    assert len(collects) == 1
    assert len(reference_collects) == 1
    assert all(
        reset.lineno < reference_forward.lineno
        for reset in [*resets, *reference_resets]
    )
    assert reference_forward.lineno < update.lineno < save.lineno
    assert all(
        save.lineno < collect.lineno
        for collect in [*collects, *reference_collects]
    )

    advance_helper = next(
        node
        for node in ast.walk(fit)
        if isinstance(node, ast.FunctionDef)
        and node.name == "advance_runtime_telemetry"
    )
    phase_calls = [
        node
        for node in ast.walk(fit)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "advance_runtime_telemetry"
    ]
    assert {
        call.args[0].value
        for call in phase_calls
        if call.args and isinstance(call.args[0], ast.Constant)
    } == set(REQUIRED_PHASES[1:])
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "advance_reproduction_step_telemetry"
        for node in ast.walk(advance_helper)
    )

    reference_methods = [
        node
        for node in worker_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "reset_reproduction_reference_logits",
            "collect_reproduction_reference_logits",
        }
    ]
    assert not any(
        isinstance(node, ast.Attribute)
        and node.attr in {"reset_peak_memory_stats", "memory_stats"}
        for method in reference_methods
        for node in ast.walk(method)
    )


def test_telemetry_rejects_reference_resource_double_counting():
    reference = _worker("reference")
    reference["peak_allocated_bytes"] = 2 * 1024**3
    with pytest.raises(TrainingTelemetryError, match="zero resource placeholders"):
        create_step_record(
            worker_records=[_worker("actor"), reference],
            global_step=1,
            attempt_id="attempt",
            sealed_config_id="config_r0",
            sealed_config_sha256="a" * 64,
            offload_profile="r0",
            timing_seconds={"save_checkpoint": 1.0},
        )


def test_telemetry_requires_reference_logits_and_complete_phase_inventory():
    reference = _worker("reference", reference_logits=None)
    with pytest.raises(TrainingTelemetryError, match="actual actor and reference logits"):
        create_step_record(
            worker_records=[_worker("actor"), reference],
            global_step=1,
            attempt_id="attempt",
            sealed_config_id="config_r0",
            sealed_config_sha256="a" * 64,
            offload_profile="r0",
            timing_seconds={"save_checkpoint": 1.0},
        )

    actor = _worker("actor")
    actor["phase_records"] = actor["phase_records"][:-1]
    with pytest.raises(TrainingTelemetryError, match="phase inventory is incomplete"):
        create_step_record(
            worker_records=[actor, _worker("reference")],
            global_step=1,
            attempt_id="attempt",
            sealed_config_id="config_r0",
            sealed_config_sha256="a" * 64,
            offload_profile="r0",
            timing_seconds={"save_checkpoint": 1.0},
        )


def test_reward_variance_telemetry_does_not_depend_on_optional_group_filtering():
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse(
        (root / "verl" / "trainer" / "ppo" / "ray_trainer.py").read_text(
            encoding="utf-8"
        )
    )
    reward_variance_value = next(
        value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant) and key.value == "reward_variance_groups"
    )
    referenced_names = {
        node.id for node in ast.walk(reward_variance_value) if isinstance(node, ast.Name)
    }

    assert "grouped_outcome_rewards" in referenced_names
    assert "prompt_uid2metric_std" not in referenced_names


def test_format_telemetry_is_computed_for_the_outcome_only_arm():
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse(
        (root / "verl" / "trainer" / "ppo" / "ray_trainer.py").read_text(
            encoding="utf-8"
        )
    )
    fallback = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "format_rewards"
        and any(isinstance(op, ast.Is) for op in node.test.ops)
        and any(isinstance(value, ast.Constant) and value.value is None for value in node.test.comparators)
    )
    fallback_calls = {
        node.func.id
        for node in ast.walk(fallback)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "compute_format_rewards" in fallback_calls
