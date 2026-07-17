from __future__ import annotations

import hashlib
import json

import pytest

from scripts.cloud import cost_gate


def _input(tmp_path, **overrides):
    telemetry = tmp_path / "telemetry.json"
    telemetry.write_text("evidence\n", encoding="ascii")
    value = {
        "decision": "bc40",
        "disk_remaining_rmb": 10.0,
        "evidence": [
            {
                "path": str(telemetry.resolve()),
                "sha256": hashlib.sha256(telemetry.read_bytes()).hexdigest(),
            }
        ],
        "gpu_cost_done_rmb": 20.0,
        "gpu_hourly_rate_rmb": 2.0,
        "measurements": {
            "evaluation_remaining_seconds": 3600.0,
            "t_artifacts_seconds": 600.0,
            "t_compute_seconds": 60.0,
            "t_init_seconds": 120.0,
            "t_resume_seconds": 90.0,
            "t_save_seconds": 30.0,
        },
        "non_gpu_cost_done_rmb": 5.0,
        "ops_reserve_rmb": 35.0,
        "schema_version": 1,
    }
    value.update(overrides)
    return value


def test_bc40_projection_uses_preregistered_counts_and_single_contingency(tmp_path):
    projection = cost_gate.create_projection(_input(tmp_path))

    expected_train = 4 * 120 + 4 * 90 + 86 * 60 + 6 * 30 + 600
    expected_gpu = 20 + 2 * 1.2 * (expected_train + 3600) / 3600
    assert projection["counts"] == {
        "fresh_init": 4,
        "resume": 4,
        "compute": 86,
        "save": 6,
    }
    assert projection["train_raw_seconds"] == expected_train
    assert projection["gpu_target_rmb"] == expected_gpu
    assert projection["status"] == "pass"
    assert cost_gate.verify_projection(projection) == projection


def test_projection_rejects_understated_reserve_and_changed_evidence(tmp_path):
    with pytest.raises(cost_gate.CostGateError, match="50 RMB"):
        cost_gate.create_projection(_input(tmp_path, ops_reserve_rmb=34.99))

    value = _input(tmp_path)
    projection = cost_gate.create_projection(value)
    evidence_path = tmp_path / "telemetry.json"
    evidence_path.write_text("changed\n", encoding="ascii")
    with pytest.raises(cost_gate.CostGateError, match="file hash changed"):
        cost_gate.verify_projection(projection)


def test_verify_recomputes_formula_even_if_projection_is_rehashed(tmp_path):
    projection = cost_gate.create_projection(_input(tmp_path))
    projection["gpu_target_rmb"] -= 1.0
    unsigned = dict(projection)
    unsigned.pop("projection_sha256")
    projection["projection_sha256"] = cost_gate._canonical_sha256(unsigned)

    with pytest.raises(cost_gate.CostGateError, match="GPU formula drifted"):
        cost_gate.verify_projection(projection)


def test_over_budget_cli_publishes_decision_and_returns_scientific_stop(tmp_path, capsys):
    value = _input(
        tmp_path,
        gpu_hourly_rate_rmb=100.0,
        gpu_cost_done_rmb=440.0,
    )
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(value), encoding="utf-8")
    output = tmp_path / "projection.json"

    rc = cost_gate.main(
        ["project", "--input", str(input_path), "--output", str(output)]
    )

    assert rc == 42
    projection = json.loads(output.read_text(encoding="ascii"))
    assert projection["status"] == "blocked"
    assert json.loads(capsys.readouterr().out) == projection
    with pytest.raises(cost_gate.CostGateError, match="exceeds"):
        cost_gate.verify_projection(projection)
