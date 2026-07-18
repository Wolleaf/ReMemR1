import copy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from scripts.cloud import capacity_evidence as capacity
from scripts.cloud import cost_gate


GIB = 1024**3
MIB = 1024**2


@pytest.fixture
def short_tmp_path(tmp_path):
    if os.name != "nt":
        yield tmp_path
        return
    root = Path(tempfile.mkdtemp(prefix="r1-", dir=Path.cwd().anchor))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _configs(profile):
    return {
        config_id: capacity.canonical_sha256({"config_id": config_id})
        for config_id in capacity.expected_config_ids(profile)
    }


def _identity(profile):
    configs = _configs(profile)
    identity = capacity.build_identity(
        commit="a" * 40,
        cpu_handoff_sha256="b" * 64,
        active_config_tree_sha256="c" * 64,
        selected_profile=profile,
        selected_configs=configs,
        gpu={
            "name": "NVIDIA GeForce RTX 5090",
            "total_vram_bytes": 32 * GIB,
            "uuid": "GPU-test-uuid",
        },
    )
    return identity, configs


def _telemetry(identity):
    steps = []
    for step in range(1, 6):
        steps.append(
            {
                "nvml_peak_used_bytes": 28 * GIB,
                "nvml_total_bytes": 32 * GIB,
                "peak_allocated_bytes": 26 * GIB,
                "peak_reserved_bytes": 27 * GIB,
                "post_step_allocated_bytes": 20 * GIB + step * 32 * MIB,
                "post_step_nvml_used_bytes": 22 * GIB + step * 32 * MIB,
                "post_step_reserved_bytes": 21 * GIB + step * 32 * MIB,
                "step": step,
            }
        )
    return {
        "adapter_update_nonzero": True,
        "all_outputs_truncated": False,
        "allocator_fragmentation_failure": False,
        "allocator_retry_count": 0,
        "base_model_unchanged": True,
        "finite_gradients": True,
        "finite_losses": True,
        "fresh_process_resume_completed": True,
        "g2a_completed": True,
        "g2b_completed": True,
        "host_peak_used_bytes": 70 * GIB,
        "host_total_memory_bytes": 128 * GIB,
        "identity": identity,
        "length_stress_completed": True,
        "non_capacity_failure": False,
        "nonzero_advantage_groups": 1,
        "nvml_total_bytes": 32 * GIB,
        "oom": False,
        "reward_variance_groups": 1,
        "steps": steps,
        "swap_used_bytes": 0,
        "systematic_format_failure": False,
        "terminal_reason": None,
    }


def _attempt_metadata(identity, configs, *, final_status="success", failure_kind=None):
    attempts = {}
    previous_stage = None
    previous_digest = None
    for index, stage in enumerate(capacity.ATTEMPT_ORDER):
        config_id = (
            f"{capacity.ATTEMPT_CONFIG_TASK[stage]}_"
            f"{identity['selected_profile'].lower()}"
        )
        digest = capacity.canonical_sha256(
            {"attempt": stage, "profile": identity["selected_profile"]}
        )
        status = final_status if index == len(capacity.ATTEMPT_ORDER) - 1 else "success"
        attempts[stage] = {
            "attempt_id": f"attempt-{index}",
            "config_id": config_id,
            "config_sha256": configs[config_id],
            "digest": digest,
            "failure_kind": failure_kind if status == "failed" else None,
            "predecessor": (
                None
                if previous_stage is None
                else {"digest": previous_digest, "stage": previous_stage}
            ),
            "status": status,
        }
        previous_stage, previous_digest = stage, digest
    return {"attempts": attempts, "identity": identity}


def _budget_projection(tmp_path):
    evidence = tmp_path / "budget-evidence.json"
    evidence.write_text("measured\n", encoding="ascii")
    projection = cost_gate.create_projection(
        {
            "decision": "bc40",
            "disk_remaining_rmb": 0.0,
            "evidence": [
                {
                    "path": str(evidence.resolve()),
                    "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
                }
            ],
            "gpu_cost_done_rmb": 0.0,
            "gpu_hourly_rate_rmb": 10.0,
            "measurements": {
                "evaluation_remaining_seconds": 1.0,
                "t_artifacts_seconds": 1.0,
                "t_compute_seconds": 1.0,
                "t_init_seconds": 1.0,
                "t_resume_seconds": 1.0,
                "t_save_seconds": 1.0,
            },
            "non_gpu_cost_done_rmb": 0.0,
            "ops_reserve_rmb": 50.0,
            "schema_version": 1,
        }
    )
    path = tmp_path / "budget.json"
    path.write_bytes(
        json.dumps(
            projection,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    return path, projection


def _approval_context(tmp_path, marker, r1_identity):
    marker_path = tmp_path / "authority" / "r1-approval.json"
    capacity._atomic_json(marker_path, marker)
    identity_text = (
        f"git_commit={r1_identity['commit']}\n"
        f"experiment_profile_id={capacity.PROFILE_ID}\n"
        f"config_tree_sha256={r1_identity['active_config_tree_sha256']}\n"
        f"environment_lock_file_sha256={'2' * 64}\n"
        f"asset_manifest_file_sha256={'3' * 64}\n"
    ).encode("ascii")
    identity_sha = hashlib.sha256(identity_text).hexdigest()
    pipeline = tmp_path / "pipelines" / f"{r1_identity['commit']}-{identity_sha}"
    pipeline.mkdir(parents=True)
    (pipeline / "identity").write_bytes(identity_text)
    launcher = tmp_path / "launchers" / "launcher-r1"
    launcher.mkdir(parents=True)
    request = {
        "schema_version": 1,
        "phase": "gpu-capacity",
        "experiment_profile_id": capacity.PROFILE_ID,
        "expected_commit": r1_identity["commit"],
        "offload_profile": "r1",
        "r1_approval": str(marker_path),
        "r1_approval_file_sha256": hashlib.sha256(marker_path.read_bytes()).hexdigest(),
        "budget_projection": str((tmp_path / "budget.json").resolve()),
        "budget_projection_file_sha256": hashlib.sha256(
            (tmp_path / "budget.json").read_bytes()
        ).hexdigest(),
        "keep_running": "no",
        "retry_failed_stage": "no",
        "dry_run": "no",
        "requested_at": "2026-07-17T00:00:00Z",
    }
    request_path = launcher / "request.json"
    capacity._atomic_json(request_path, request)
    claim = (
        pipeline
        / "capacity"
        / "r1"
        / "approval-consumptions"
        / marker["approval_marker_sha256"]
        / "consumption.json"
    )
    claim.parent.mkdir(parents=True)
    return marker_path, launcher, pipeline, claim


def _approval_consumption(tmp_path, marker, r1_identity):
    marker_path, launcher, pipeline, claim = _approval_context(
        tmp_path, marker, r1_identity
    )
    consumption = capacity.create_r1_approval_consumption(
        approval_marker=marker,
        approval_marker_path=marker_path,
        r1_identity=r1_identity,
        launcher_dir=launcher,
        pipeline_dir=pipeline,
        consumption_path=claim,
        consumed_at="2026-07-17T00:01:00Z",
    )
    capacity._atomic_consumption_json_new(
        claim,
        consumption,
        pipeline_dir=pipeline,
        marker_sha256=marker["approval_marker_sha256"],
    )
    return marker_path, consumption, claim


def _r1_authority(tmp_path):
    r0_identity, r0_configs = _identity("R0")
    telemetry = _telemetry(r0_identity)
    telemetry["steps"][0]["peak_allocated_bytes"] = 28 * GIB
    evidence = capacity.create_capacity_evidence(
        identity=r0_identity,
        telemetry=telemetry,
        attempt_metadata=_attempt_metadata(r0_identity, r0_configs),
        selected_configs=r0_configs,
    )
    r1_identity, _ = _identity("R1")
    _, projection = _budget_projection(tmp_path)
    marker = capacity.create_r1_approval_marker(
        r1_identity=r1_identity,
        r0_capacity_evidence=evidence,
        r0_terminal_sha256="d" * 64,
        budget_projection_sha256=projection["projection_sha256"],
        approval_nonce="approval-consumption-test",
    )
    return r1_identity, marker


def test_green_r0_capacity_profile_is_self_hashed_and_verifiable():
    identity, configs = _identity("R0")
    telemetry = _telemetry(identity)
    attempts = _attempt_metadata(identity, configs)
    profile = capacity.create_capacity_profile(
        identity=identity,
        telemetry=telemetry,
        attempt_metadata=attempts,
        selected_configs=configs,
    )

    assert profile["selected_profile"] == "R0"
    assert profile["approval_marker_sha256"] is None
    assert profile["approval_consumption_sha256"] is None
    assert profile["capacity_evidence"]["classification"]["overall"] == "green"
    assert set(profile["selected_configs"]) == set(capacity.expected_config_ids("R0"))
    verified = capacity.verify_capacity_profile(
        profile,
        expected_identity=identity,
        telemetry=telemetry,
        attempt_metadata=attempts,
    )
    assert verified["self_sha256"] == profile["self_sha256"]

    tampered = copy.deepcopy(profile)
    tampered["gpu"]["uuid"] = "GPU-drifted"
    with pytest.raises(capacity.CapacityEvidenceError, match="self_sha256"):
        capacity.verify_capacity_profile(tampered)


def test_capacity_evidence_enforces_profile_specific_host_ram():
    r0_identity, r0_configs = _identity("R0")
    r0_telemetry = _telemetry(r0_identity)
    r0_telemetry["host_total_memory_bytes"] = 80 * GIB
    evidence = capacity.create_capacity_evidence(
        identity=r0_identity,
        telemetry=r0_telemetry,
        attempt_metadata=_attempt_metadata(r0_identity, r0_configs),
        selected_configs=r0_configs,
    )
    assert evidence["telemetry_sha256"] == capacity.canonical_sha256(r0_telemetry)

    r0_telemetry["host_total_memory_bytes"] = 80 * GIB - 1
    with pytest.raises(capacity.CapacityEvidenceError, match="R0 telemetry"):
        capacity.create_capacity_evidence(
            identity=r0_identity,
            telemetry=r0_telemetry,
            attempt_metadata=_attempt_metadata(r0_identity, r0_configs),
            selected_configs=r0_configs,
        )

    r1_identity, r1_configs = _identity("R1")
    r1_telemetry = _telemetry(r1_identity)
    r1_telemetry["host_total_memory_bytes"] = 128 * GIB - 1
    with pytest.raises(capacity.CapacityEvidenceError, match="R1 telemetry"):
        capacity.create_capacity_evidence(
            identity=r1_identity,
            telemetry=r1_telemetry,
            attempt_metadata=_attempt_metadata(r1_identity, r1_configs),
            selected_configs=r1_configs,
        )


def test_capacity_thresholds_cover_green_yellow_and_red_boundaries():
    identity, _ = _identity("R0")
    telemetry = _telemetry(identity)
    result = capacity.classify_telemetry(telemetry)
    assert result["overall"] == "green"

    yellow = copy.deepcopy(telemetry)
    yellow["steps"][0]["peak_allocated_bytes"] = 27 * GIB + 1
    yellow_result = capacity.classify_telemetry(yellow)
    assert yellow_result["metrics"]["pytorch_peak_allocated"]["status"] == "yellow"
    assert yellow_result["r1_eligible"] is True
    assert yellow_result["eligibility_reason"] == "gpu_memory_headroom"

    red = copy.deepcopy(telemetry)
    red["steps"][0]["peak_reserved_bytes"] = 30 * GIB
    assert capacity.classify_telemetry(red)["metrics"]["pytorch_peak_reserved"][
        "status"
    ] == "red"

    headroom_red = copy.deepcopy(telemetry)
    headroom_red["steps"][0]["nvml_peak_used_bytes"] = 31 * GIB + 1
    assert capacity.classify_telemetry(headroom_red)["metrics"]["nvml_peak_used"][
        "status"
    ] == "red"


def test_host_ram_or_growth_pressure_cannot_authorize_r1():
    identity, _ = _identity("R0")
    host_yellow = _telemetry(identity)
    host_yellow["host_peak_used_bytes"] = 105 * GIB
    result = capacity.classify_telemetry(host_yellow)
    assert result["metrics"]["host_ram"]["status"] == "yellow"
    assert result["r1_eligible"] is False

    growth = _telemetry(identity)
    for step in growth["steps"]:
        if step["step"] >= 3:
            step["post_step_allocated_bytes"] += (step["step"] - 3) * 600 * MIB
    result = capacity.classify_telemetry(growth)
    assert result["metrics"]["resident_growth_step3_to_step5"]["status"] == "red"
    assert result["r1_eligible"] is False


def test_high_truncation_is_yellow_science_and_never_authorizes_r1():
    identity, configs = _identity("R0")
    telemetry = _telemetry(identity)
    telemetry["high_truncation_rate"] = True

    result = capacity.classify_telemetry(telemetry)

    assert result["metrics"]["format"]["status"] == "yellow"
    assert result["overall"] == "yellow"
    assert result["r1_eligible"] is False
    with pytest.raises(capacity.CapacityEvidenceError, match="not all green"):
        capacity.create_capacity_profile(
            identity=identity,
            telemetry=telemetry,
            attempt_metadata=_attempt_metadata(identity, configs),
            selected_configs=configs,
        )


def test_partial_r0_oom_attempt_produces_eligible_evidence_not_a_profile():
    identity, configs = _identity("R0")
    telemetry = _telemetry(identity)
    telemetry.update(
        {
            "fresh_process_resume_completed": False,
            "g2a_completed": False,
            "g2b_completed": False,
            "length_stress_completed": False,
            "oom": True,
            "steps": [],
            "terminal_reason": "oom",
        }
    )
    config_id = f"{capacity.ATTEMPT_CONFIG_TASK['g2a']}_r0"
    attempts = {
        "attempts": {
            "g2a": {
                "attempt_id": "attempt-oom",
                "config_id": config_id,
                "config_sha256": configs[config_id],
                "digest": "d" * 64,
                "failure_kind": "capacity",
                "predecessor": None,
                "status": "failed",
            }
        },
        "identity": identity,
    }
    evidence = capacity.create_capacity_evidence(
        identity=identity,
        telemetry=telemetry,
        attempt_metadata=attempts,
        selected_configs=configs,
    )
    assert evidence["classification"]["overall"] == "red"
    assert evidence["classification"]["r1_eligible"] is True
    assert evidence["classification"]["eligibility_reason"] == "oom"
    with pytest.raises(capacity.CapacityEvidenceError, match="not all green"):
        capacity.create_capacity_profile(
            identity=identity,
            telemetry=telemetry,
            attempt_metadata=attempts,
            selected_configs=configs,
        )

def test_r0_yellow_evidence_authorizes_only_an_all_green_r1(short_tmp_path):
    r0_identity, r0_configs = _identity("R0")
    r0_telemetry = _telemetry(r0_identity)
    r0_telemetry["steps"][0]["peak_allocated_bytes"] = 28 * GIB
    r0_attempts = _attempt_metadata(r0_identity, r0_configs)
    r0_evidence = capacity.create_capacity_evidence(
        identity=r0_identity,
        telemetry=r0_telemetry,
        attempt_metadata=r0_attempts,
        selected_configs=r0_configs,
    )
    assert r0_evidence["classification"]["r1_eligible"] is True
    with pytest.raises(capacity.CapacityEvidenceError, match="not all green"):
        capacity.create_capacity_profile(
            identity=r0_identity,
            telemetry=r0_telemetry,
            attempt_metadata=r0_attempts,
            selected_configs=r0_configs,
        )

    r1_identity, r1_configs = _identity("R1")
    r1_telemetry = _telemetry(r1_identity)
    r1_attempts = _attempt_metadata(r1_identity, r1_configs)
    terminal_sha = "d" * 64
    _, projection = _budget_projection(short_tmp_path)
    budget_sha = projection["projection_sha256"]
    marker = capacity.create_r1_approval_marker(
        r1_identity=r1_identity,
        r0_capacity_evidence=r0_evidence,
        r0_terminal_sha256=terminal_sha,
        budget_projection_sha256=budget_sha,
        approval_nonce="approval-001",
    )
    marker_path, consumption, consumption_path = _approval_consumption(
        short_tmp_path, marker, r1_identity
    )
    with pytest.raises(capacity.CapacityEvidenceError, match="blocked without"):
        capacity.create_capacity_profile(
            identity=r1_identity,
            telemetry=r1_telemetry,
            attempt_metadata=r1_attempts,
            selected_configs=r1_configs,
        )

    profile = capacity.create_capacity_profile(
        identity=r1_identity,
        telemetry=r1_telemetry,
        attempt_metadata=r1_attempts,
        selected_configs=r1_configs,
        approval_marker=marker,
        approval_marker_path=marker_path,
        approval_consumption=consumption,
        approval_consumption_path=consumption_path,
        r0_capacity_evidence=r0_evidence,
        r0_terminal_sha256=terminal_sha,
        budget_projection_sha256=budget_sha,
    )
    assert profile["approval_marker_sha256"] == marker["approval_marker_sha256"]
    capacity.verify_capacity_profile(
        profile,
        approval_marker=marker,
        approval_marker_path=marker_path,
        approval_consumption=consumption,
        approval_consumption_path=consumption_path,
        r0_capacity_evidence=r0_evidence,
        r0_terminal_sha256=terminal_sha,
        budget_projection_sha256=budget_sha,
    )

    with pytest.raises(capacity.CapacityEvidenceError, match="already consumed"):
        capacity.create_capacity_profile(
            identity=r1_identity,
            telemetry=r1_telemetry,
            attempt_metadata=r1_attempts,
            selected_configs=r1_configs,
            approval_marker=marker,
            approval_marker_path=marker_path,
            approval_consumption=consumption,
            approval_consumption_path=consumption_path,
            r0_capacity_evidence=r0_evidence,
            r0_terminal_sha256=terminal_sha,
            budget_projection_sha256=budget_sha,
            used_approval_marker_sha256s={marker["approval_marker_sha256"]},
        )


def test_approve_r1_cli_publishes_once_and_refuses_overwrite(tmp_path, capsys):
    r0_identity, r0_configs = _identity("R0")
    r0_telemetry = _telemetry(r0_identity)
    r0_telemetry["steps"][0]["peak_allocated_bytes"] = 28 * GIB
    r0_evidence = capacity.create_capacity_evidence(
        identity=r0_identity,
        telemetry=r0_telemetry,
        attempt_metadata=_attempt_metadata(r0_identity, r0_configs),
        selected_configs=r0_configs,
    )
    r1_identity, _ = _identity("R1")
    identity_path = tmp_path / "r1-identity.json"
    evidence_path = tmp_path / "r0-capacity-evidence.json"
    output = tmp_path / "r1-approval.json"
    capacity._atomic_json(identity_path, r1_identity)
    capacity._atomic_json(evidence_path, r0_evidence)
    args = [
        "approve-r1",
        "--r1-identity",
        str(identity_path),
        "--r0-capacity-evidence",
        str(evidence_path),
        "--r0-terminal-sha256",
        "d" * 64,
        "--budget-projection-sha256",
        "e" * 64,
        "--approval-nonce",
        "operator-approval-001",
        "--output",
        str(output),
    ]

    assert capacity.main(args) == 0
    message = json.loads(capsys.readouterr().out)
    marker = json.loads(output.read_text(encoding="ascii"))
    assert message == {
        "approval_marker_sha256": marker["approval_marker_sha256"],
        "status": "approved-once",
    }
    capacity.verify_r1_approval_marker(
        marker,
        r1_identity=r1_identity,
        r0_capacity_evidence=r0_evidence,
        r0_terminal_sha256="d" * 64,
        budget_projection_sha256="e" * 64,
    )

    original = output.read_bytes()
    assert capacity.main(args) == 2
    error = json.loads(capsys.readouterr().err)
    assert "refusing to replace" in error["error"]
    assert output.read_bytes() == original


def test_approve_r1_cli_rejects_unsafe_nonce(tmp_path, capsys):
    r0_identity, r0_configs = _identity("R0")
    telemetry = _telemetry(r0_identity)
    telemetry["steps"][0]["peak_allocated_bytes"] = 28 * GIB
    evidence = capacity.create_capacity_evidence(
        identity=r0_identity,
        telemetry=telemetry,
        attempt_metadata=_attempt_metadata(r0_identity, r0_configs),
        selected_configs=r0_configs,
    )
    r1_identity, _ = _identity("R1")
    identity_path = tmp_path / "identity.json"
    evidence_path = tmp_path / "evidence.json"
    capacity._atomic_json(identity_path, r1_identity)
    capacity._atomic_json(evidence_path, evidence)
    output = tmp_path / "approval.json"

    assert capacity.main(
        [
            "approve-r1",
            "--r1-identity",
            str(identity_path),
            "--r0-capacity-evidence",
            str(evidence_path),
            "--r0-terminal-sha256",
            "d" * 64,
            "--budget-projection-sha256",
            "e" * 64,
            "--approval-nonce",
            "not safe/nonce",
            "--output",
            str(output),
        ]
    ) == 2
    assert "unique safe identifier" in json.loads(capsys.readouterr().err)["error"]
    assert not output.exists()


def test_consume_r1_cli_publishes_and_self_verifies_once(short_tmp_path, capsys):
    r1_identity, marker = _r1_authority(short_tmp_path)
    marker_path, launcher, pipeline, output = _approval_context(
        short_tmp_path, marker, r1_identity
    )
    identity_path = short_tmp_path / "r1-identity.json"
    capacity._atomic_json(identity_path, r1_identity)
    args = [
        "consume-r1",
        "--approval-marker",
        str(marker_path),
        "--r1-identity",
        str(identity_path),
        "--launcher-dir",
        str(launcher),
        "--pipeline-dir",
        str(pipeline),
        "--output",
        str(output),
    ]

    assert capacity.main(args) == 0
    message = json.loads(capsys.readouterr().out)
    consumption = json.loads(output.read_text(encoding="ascii"))
    assert message == {
        "consumption_sha256": consumption["consumption_sha256"],
        "status": "consumed",
    }
    assert consumption["launcher_request_sha256"] == capacity._file_sha256(
        launcher / "request.json"
    )
    assert consumption["approval_marker_file_sha256"] == capacity._file_sha256(
        marker_path
    )
    capacity.verify_r1_approval_consumption(
        consumption,
        consumption_path=output,
        approval_marker=marker,
        approval_marker_path=marker_path,
        r1_identity=r1_identity,
    )

    original = output.read_bytes()
    assert capacity.main(args) == 2
    assert "already consumed" in json.loads(capsys.readouterr().err)["error"]
    assert output.read_bytes() == original


def test_consume_r1_rejects_nonempty_claim_and_wrong_output(short_tmp_path, capsys):
    r1_identity, marker = _r1_authority(short_tmp_path)
    marker_path, launcher, pipeline, output = _approval_context(
        short_tmp_path, marker, r1_identity
    )
    identity_path = short_tmp_path / "r1-identity.json"
    capacity._atomic_json(identity_path, r1_identity)
    common = [
        "consume-r1",
        "--approval-marker",
        str(marker_path),
        "--r1-identity",
        str(identity_path),
        "--launcher-dir",
        str(launcher),
        "--pipeline-dir",
        str(pipeline),
    ]
    wrong_output = output.with_name("other.json")
    assert capacity.main([*common, "--output", str(wrong_output)]) == 2
    assert "escaped its marker claim" in json.loads(capsys.readouterr().err)["error"]
    assert not wrong_output.exists()

    (output.parent / "crash-remnant").write_text("claimed\n", encoding="ascii")
    assert capacity.main([*common, "--output", str(output)]) == 2
    assert "already consumed or nonempty" in json.loads(capsys.readouterr().err)[
        "error"
    ]
    assert not output.exists()


def test_consumption_reverification_detects_request_and_pipeline_byte_drift(
    short_tmp_path,
):
    r1_identity, marker = _r1_authority(short_tmp_path)
    marker_path, consumption, consumption_path = _approval_consumption(
        short_tmp_path, marker, r1_identity
    )
    launcher = Path(consumption["launcher_dir"])
    pipeline = Path(consumption["pipeline_dir"])
    request_path = launcher / "request.json"
    request_path.write_bytes(request_path.read_bytes() + b"\n")
    with pytest.raises(capacity.CapacityEvidenceError, match="launcher_request_sha256"):
        capacity.verify_r1_approval_consumption(
            consumption,
            consumption_path=consumption_path,
            approval_marker=marker,
            approval_marker_path=marker_path,
            r1_identity=r1_identity,
        )

    request_path.write_bytes(request_path.read_bytes().rstrip(b"\n") + b"\n")
    identity_path = pipeline / "identity"
    identity_path.write_bytes(identity_path.read_bytes().replace(b"2" * 64, b"4" * 64))
    with pytest.raises(capacity.CapacityEvidenceError, match="pipeline directory identity"):
        capacity.verify_r1_approval_consumption(
            consumption,
            consumption_path=consumption_path,
            approval_marker=marker,
            approval_marker_path=marker_path,
            r1_identity=r1_identity,
        )


def test_consumption_create_and_verify_reject_same_path_marker_byte_replacement(
    short_tmp_path,
):
    r1_identity, marker = _r1_authority(short_tmp_path)
    marker_path, launcher, pipeline, claim = _approval_context(
        short_tmp_path, marker, r1_identity
    )
    original = marker_path.read_bytes()

    # Semantically identical JSON at the admitted path is still a different authority file.
    marker_path.write_bytes(original + b" \n")
    with pytest.raises(capacity.CapacityEvidenceError, match="file hash changed"):
        capacity.create_r1_approval_consumption(
            approval_marker=marker,
            approval_marker_path=marker_path,
            r1_identity=r1_identity,
            launcher_dir=launcher,
            pipeline_dir=pipeline,
            consumption_path=claim,
            consumed_at="2026-07-17T00:01:00Z",
        )

    marker_path.write_bytes(original)
    consumption = capacity.create_r1_approval_consumption(
        approval_marker=marker,
        approval_marker_path=marker_path,
        r1_identity=r1_identity,
        launcher_dir=launcher,
        pipeline_dir=pipeline,
        consumption_path=claim,
        consumed_at="2026-07-17T00:01:00Z",
    )
    capacity._atomic_consumption_json_new(
        claim,
        consumption,
        pipeline_dir=pipeline,
        marker_sha256=marker["approval_marker_sha256"],
    )
    marker_path.write_bytes(original + b" \n")
    with pytest.raises(capacity.CapacityEvidenceError, match="file hash changed"):
        capacity.verify_r1_approval_consumption(
            consumption,
            consumption_path=claim,
            approval_marker=marker,
            approval_marker_path=marker_path,
            r1_identity=r1_identity,
        )

def test_consume_r1_rejects_wrong_marker_path_and_hash(short_tmp_path, capsys):
    r1_identity, marker = _r1_authority(short_tmp_path)
    marker_path, launcher, pipeline, output = _approval_context(
        short_tmp_path, marker, r1_identity
    )
    identity_path = short_tmp_path / "r1-identity.json"
    capacity._atomic_json(identity_path, r1_identity)
    copied_marker = short_tmp_path / "authority" / "copied-approval.json"
    capacity._atomic_json(copied_marker, marker)
    common = [
        "consume-r1",
        "--r1-identity",
        str(identity_path),
        "--launcher-dir",
        str(launcher),
        "--pipeline-dir",
        str(pipeline),
        "--output",
        str(output),
    ]
    assert capacity.main(
        ["consume-r1", "--approval-marker", str(copied_marker), *common[1:]]
    ) == 2
    assert "launcher request" in json.loads(capsys.readouterr().err)["error"]

    tampered = copy.deepcopy(marker)
    tampered["approval_marker_sha256"] = "f" * 64
    capacity._atomic_json(marker_path, tampered)
    assert capacity.main(
        ["consume-r1", "--approval-marker", str(marker_path), *common[1:]]
    ) == 2
    assert "approval_marker_sha256" in json.loads(capsys.readouterr().err)["error"]
    assert not output.exists()


def test_r1_rejects_tampered_approval_and_non_green_telemetry():
    r0_identity, r0_configs = _identity("R0")
    r0_telemetry = _telemetry(r0_identity)
    r0_telemetry["steps"][0]["peak_allocated_bytes"] = 28 * GIB
    evidence = capacity.create_capacity_evidence(
        identity=r0_identity,
        telemetry=r0_telemetry,
        attempt_metadata=_attempt_metadata(r0_identity, r0_configs),
        selected_configs=r0_configs,
    )
    r1_identity, r1_configs = _identity("R1")
    marker = capacity.create_r1_approval_marker(
        r1_identity=r1_identity,
        r0_capacity_evidence=evidence,
        r0_terminal_sha256="d" * 64,
        budget_projection_sha256="e" * 64,
        approval_nonce="approval-002",
    )
    tampered = copy.deepcopy(marker)
    tampered["budget_projection_sha256"] = "f" * 64
    with pytest.raises(capacity.CapacityEvidenceError, match="approval_marker_sha256"):
        capacity.verify_r1_approval_marker(
            tampered,
            r1_identity=r1_identity,
            r0_capacity_evidence=evidence,
            r0_terminal_sha256="d" * 64,
            budget_projection_sha256="e" * 64,
        )

    yellow_r1 = _telemetry(r1_identity)
    yellow_r1["steps"][0]["peak_allocated_bytes"] = 28 * GIB
    with pytest.raises(
        capacity.CapacityEvidenceError, match="R1 capacity evidence is not all green"
    ):
        capacity.create_capacity_profile(
            identity=r1_identity,
            telemetry=yellow_r1,
            attempt_metadata=_attempt_metadata(r1_identity, r1_configs),
            selected_configs=r1_configs,
            approval_marker=marker,
            r0_capacity_evidence=evidence,
            r0_terminal_sha256="d" * 64,
            budget_projection_sha256="e" * 64,
        )


def test_telemetry_and_attempt_identity_drift_fail_closed():
    identity, configs = _identity("R0")
    telemetry = _telemetry(identity)
    attempts = _attempt_metadata(identity, configs)

    drifted_telemetry = copy.deepcopy(telemetry)
    drifted_telemetry["identity"]["gpu"]["uuid"] = "GPU-other"
    with pytest.raises(capacity.CapacityEvidenceError, match="identity drifted"):
        capacity.create_capacity_evidence(
            identity=identity,
            telemetry=drifted_telemetry,
            attempt_metadata=attempts,
            selected_configs=configs,
        )

    drifted_total = copy.deepcopy(telemetry)
    drifted_total["nvml_total_bytes"] = 64 * GIB
    with pytest.raises(capacity.CapacityEvidenceError, match="total VRAM drifted"):
        capacity.create_capacity_evidence(
            identity=identity,
            telemetry=drifted_total,
            attempt_metadata=attempts,
            selected_configs=configs,
        )

    non_capacity_failure = copy.deepcopy(telemetry)
    non_capacity_failure["non_capacity_failure"] = True
    assert capacity.classify_telemetry(non_capacity_failure)["metrics"]["execution"][
        "status"
    ] == "red"

    drifted_attempts = copy.deepcopy(attempts)
    drifted_attempts["attempts"]["g2b-resume5"]["predecessor"]["digest"] = "f" * 64
    with pytest.raises(capacity.CapacityEvidenceError, match="predecessor drifted"):
        capacity.create_capacity_evidence(
            identity=identity,
            telemetry=telemetry,
            attempt_metadata=drifted_attempts,
            selected_configs=configs,
        )
