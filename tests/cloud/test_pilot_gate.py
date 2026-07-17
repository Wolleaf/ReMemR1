from __future__ import annotations

import json
import ast
import hashlib
from dataclasses import asdict
import importlib.util
import sys
from pathlib import Path

import pytest

from scripts.cloud import pilot_gate
from scripts.cloud.pilot_evidence import (
    PilotEvidenceError,
    append_step_record,
    create_gate_report,
    create_step_record,
    load_evidence,
    resolve_stable_prompt_group_ids,
)


_FINGERPRINT_SOURCE = (
    Path(__file__).resolve().parents[2] / "verl" / "utils" / "reproduction_fingerprint.py"
)
_FINGERPRINT_SPEC = importlib.util.spec_from_file_location(
    "_pilot_gate_test_fingerprint",
    _FINGERPRINT_SOURCE,
)
assert _FINGERPRINT_SPEC is not None and _FINGERPRINT_SPEC.loader is not None
_FINGERPRINT_MODULE = importlib.util.module_from_spec(_FINGERPRINT_SPEC)
sys.modules[_FINGERPRINT_SPEC.name] = _FINGERPRINT_MODULE
_FINGERPRINT_SPEC.loader.exec_module(_FINGERPRINT_MODULE)
StepZeroFingerprint = _FINGERPRINT_MODULE.StepZeroFingerprint


def test_trainer_builds_pilot_evidence_before_update_and_appends_only_after_success():
    trainer_path = (
        Path(__file__).resolve().parents[2] / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    )
    tree = ast.parse(trainer_path.read_text(encoding="utf-8"))
    trainer = next(
        node
        for node in tree.body
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
            and (
                isinstance(node.func, ast.Name)
                and node.func.id == name
                or isinstance(node.func, ast.Attribute)
                and node.func.attr == name
            )
        ]

    create_call = calls("create_pilot_step_record")[0]
    update_call = calls("update_actor")[0]
    append_call = calls("append_pilot_step_record")[0]
    assert create_call.lineno < update_call.lineno < append_call.lineno
    prompt_groups = next(
        keyword.value
        for keyword in create_call.keywords
        if keyword.arg == "prompt_group_ids"
    )
    assert isinstance(prompt_groups, ast.Call)
    assert isinstance(prompt_groups.func, ast.Name)
    assert prompt_groups.func.id == "resolve_stable_prompt_group_ids"


def test_runtime_uuids_resolve_to_stable_manifest_prompt_groups():
    b_groups = resolve_stable_prompt_group_ids(
        ["b-random-1", "b-random-2"],
        ["qa-001", "qa-002"],
        ["b-random-1", "b-random-1", "b-random-2", "b-random-2"],
    )
    c_groups = resolve_stable_prompt_group_ids(
        ["c-random-1", "c-random-2"],
        ["qa-001", "qa-002"],
        ["c-random-1", "c-random-1", "c-random-2", "c-random-2"],
    )

    assert b_groups == c_groups == ["qa-001", "qa-001", "qa-002", "qa-002"]
    assert resolve_stable_prompt_group_ids(
        ["runtime-1", "runtime-2"],
        ["qa-001", "qa-002"],
        ["runtime-2", "runtime-1"],
    ) != ["qa-001", "qa-002"]


def test_stable_prompt_group_mapping_rejects_duplicate_or_unknown_identity():
    with pytest.raises(PilotEvidenceError, match="must be unique"):
        resolve_stable_prompt_group_ids(
            ["runtime-1", "runtime-2"],
            ["qa-001", "qa-001"],
            ["runtime-1"],
        )
    with pytest.raises(PilotEvidenceError, match="lacks a manifest identity"):
        resolve_stable_prompt_group_ids(
            ["runtime-1", "runtime-2"],
            ["qa-001", "qa-002"],
            ["runtime-3"],
        )


def _fingerprint() -> StepZeroFingerprint:
    return StepZeroFingerprint(
        run_seed=42,
        rollout_global_step=1,
        base_model_id="Qwen/Qwen3.5-2B",
        base_model_revision="1" * 40,
        data_manifest_sha256="2" * 64,
        initial_adapter_sha256="3" * 64,
        first_batch_sha256="4" * 64,
        first_sampled_tokens_sha256="5" * 64,
    )


def _write_arm(
    tmp_path,
    arm: str,
    nonzero_steps: set[int],
    *,
    state_nonzero_steps: set[int] | None = None,
):
    path = tmp_path / f"{arm}.jsonl"
    for step in (1, 2, 3):
        outcome_values = [1.0, -1.0, 0.5, -0.5, 1.0, -1.0, 0.5, -0.5]
        if step not in nonzero_steps:
            outcome_values = [0.0] * 8
        kwargs = {}
        if arm == "c":
            state_steps = nonzero_steps if state_nonzero_steps is None else state_nonzero_steps
            state_values = [0.25, -0.25, 0.5, -0.5, 0.25, -0.25, 0.5, -0.5]
            if step not in state_steps:
                state_values = [0.0] * 8
            total_values = [
                0.8 * outcome + 0.2 * state
                for outcome, state in zip(outcome_values, state_values, strict=True)
            ]
            kwargs = {
                "state_advantage_values": state_values,
            }
        else:
            total_values = outcome_values
        record = create_step_record(
            arm=arm,
            offload_profile="r0",
            global_step=step,
            alpha=1.0 if arm == "b" else 0.8,
            prompt_group_ids=["p0"] * 4 + ["p1"] * 4,
            advantage_values=total_values,
            outcome_advantage_values=outcome_values,
            action_step_ids=[0] * 8,
            action_types=[2] * 6 + [0] * 2,
            **kwargs,
        )
        append_step_record(path, record)
    return path


def _write_fingerprints(tmp_path):
    b_path = tmp_path / "b-step-zero.json"
    c_path = tmp_path / "c-step-zero.json"
    _fingerprint().save(b_path)
    _fingerprint().save(c_path)
    return b_path, c_path


def test_three_step_evidence_is_canonical_self_hashed_and_no_duplicate_step(tmp_path):
    path = _write_arm(tmp_path, "b", {1})
    records = load_evidence(path)

    assert [record["global_step"] for record in records] == [1, 2, 3]
    assert sum(record["nonzero_prompt_group_count"] for record in records) == 2
    assert all(
        line == json.dumps(json.loads(line), sort_keys=True, separators=(",", ":"))
        for line in path.read_text(encoding="ascii").splitlines()
    )
    with pytest.raises(PilotEvidenceError, match="expected step 4"):
        append_step_record(path, records[-1])


def test_gate_passes_only_when_both_arms_reach_two_of_six_groups(tmp_path):
    b_path = _write_arm(tmp_path, "b", {1})
    c_path = _write_arm(tmp_path, "c", {1})
    b_step_zero, c_step_zero = _write_fingerprints(tmp_path)

    report = create_gate_report(
        b_evidence_path=b_path,
        c_evidence_path=c_path,
        b_step_zero_path=b_step_zero,
        c_step_zero_path=c_step_zero,
    )

    assert report["outcome"] == "pass"
    assert report["b_nonzero_prompt_groups"] == 2
    assert report["c_nonzero_prompt_groups"] == 2
    assert report["c_nonzero_state_groups"] == 2
    assert report["expected_prompt_groups_per_arm"] == 6
    assert report["minimum_nonzero_state_groups"] == 1
    assert len(report["shared_step1_action_identity_sha256"]) == 64
    assert len(report["shared_step1_outcome_advantage_values_sha256"]) == 64
    assert report["retryable"] is False


def test_gate_stops_when_c_total_advantage_is_nonzero_but_state_advantage_is_dead(
    tmp_path,
):
    b_path = _write_arm(tmp_path, "b", {1})
    c_path = _write_arm(tmp_path, "c", {1}, state_nonzero_steps=set())
    b_step_zero, c_step_zero = _write_fingerprints(tmp_path)

    report = create_gate_report(
        b_evidence_path=b_path,
        c_evidence_path=c_path,
        b_step_zero_path=b_step_zero,
        c_step_zero_path=c_step_zero,
    )

    assert report["b_nonzero_prompt_groups"] == 2
    assert report["c_nonzero_prompt_groups"] == 2
    assert report["c_nonzero_state_groups"] == 0
    assert report["outcome"] == "scientific-stop"


def test_scientific_stop_returns_42_and_publishes_nonretryable_report(tmp_path, capsys):
    b_path = _write_arm(tmp_path, "b", set())
    c_path = _write_arm(tmp_path, "c", {2})
    b_step_zero, c_step_zero = _write_fingerprints(tmp_path)
    output = tmp_path / "gate.json"

    rc = pilot_gate.main(
        [
            "--b-evidence",
            str(b_path),
            "--c-evidence",
            str(c_path),
            "--b-step-zero",
            str(b_step_zero),
            "--c-step-zero",
            str(c_step_zero),
            "--output",
            str(output),
        ]
    )

    assert rc == 42
    report = json.loads(output.read_text(encoding="ascii"))
    assert report["outcome"] == "scientific-stop"
    assert report["retryable"] is False
    assert json.loads(capsys.readouterr().out) == report


def test_tampering_and_step_zero_drift_fail_closed(tmp_path):
    b_path = _write_arm(tmp_path, "b", {1})
    c_path = _write_arm(tmp_path, "c", {1})
    b_step_zero, c_step_zero = _write_fingerprints(tmp_path)
    lines = b_path.read_text(encoding="ascii").splitlines()
    value = json.loads(lines[0])
    value["nonzero_prompt_group_count"] = 0
    lines[0] = json.dumps(value, sort_keys=True, separators=(",", ":"))
    b_path.write_text("\n".join(lines) + "\n", encoding="ascii")

    with pytest.raises(PilotEvidenceError, match="self-hash mismatch"):
        create_gate_report(
            b_evidence_path=b_path,
            c_evidence_path=c_path,
            b_step_zero_path=b_step_zero,
            c_step_zero_path=c_step_zero,
        )

    b_path = tmp_path / "b-clean.jsonl"
    for record in load_evidence(c_path):
        b_record = create_step_record(
            arm="b",
            offload_profile="r0",
            global_step=record["global_step"],
            alpha=1.0,
            prompt_group_ids=["p0"] * 4 + ["p1"] * 4,
            advantage_values=[1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            outcome_advantage_values=[
                1.0,
                -1.0,
                1.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            action_step_ids=[0] * 8,
            action_types=[2] * 6 + [0] * 2,
        )
        append_step_record(b_path, b_record)
    c_step_zero.unlink()
    StepZeroFingerprint(
        **{
            **asdict(_fingerprint()),
            "first_sampled_tokens_sha256": "6" * 64,
        }
    ).save(c_step_zero)
    with pytest.raises(Exception, match="step-zero fingerprint mismatch"):
        create_gate_report(
            b_evidence_path=b_path,
            c_evidence_path=c_path,
            b_step_zero_path=b_step_zero,
            c_step_zero_path=c_step_zero,
        )


def test_recomputed_hash_cannot_hide_a_broken_advantage_equation(tmp_path):
    path = _write_arm(tmp_path, "c", {1})
    lines = path.read_text(encoding="ascii").splitlines()
    value = json.loads(lines[0])
    value["action_records"][0]["total_advantage"] += 0.5
    unsigned = dict(value)
    unsigned.pop("record_sha256")
    value["record_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    lines[0] = json.dumps(value, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")

    with pytest.raises(PilotEvidenceError, match="advantage equation failed"):
        load_evidence(path)


def test_gate_rejects_step1_action_mapping_drift(tmp_path):
    b_path = _write_arm(tmp_path, "b", {1})
    c_path = _write_arm(tmp_path, "c", {1})
    b_step_zero, c_step_zero = _write_fingerprints(tmp_path)
    lines = c_path.read_text(encoding="ascii").splitlines()
    value = json.loads(lines[0])
    value["action_records"][0]["action_type"] = 1
    value["action_identity_sha256"] = hashlib.sha256(
        json.dumps(
            [
                {
                    "action_type": action["action_type"],
                    "prompt_group_id_sha256": action["prompt_group_id_sha256"],
                    "sample_ordinal": action["sample_ordinal"],
                    "step_id": action["step_id"],
                }
                for action in value["action_records"]
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    unsigned = dict(value)
    unsigned.pop("record_sha256")
    value["record_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    lines[0] = json.dumps(value, sort_keys=True, separators=(",", ":"))
    c_path.write_bytes(("\n".join(lines) + "\n").encode("ascii"))

    with pytest.raises(PilotEvidenceError, match="action mapping differs"):
        create_gate_report(
            b_evidence_path=b_path,
            c_evidence_path=c_path,
            b_step_zero_path=b_step_zero,
            c_step_zero_path=c_step_zero,
        )
