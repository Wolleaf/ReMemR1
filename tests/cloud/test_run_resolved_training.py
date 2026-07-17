import builtins
import copy
import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

from scripts.cloud import run_resolved_training as runtime


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _config(
    name,
    step,
    *,
    resume_from=None,
    adapter=True,
    fingerprint=False,
    reference=False,
    pilot=False,
):
    root = f"/sealed/{name}"
    return {
        "actor_rollout_ref": {
            "actor": {"ppo_mini_batch_size": 2},
            "model": {"path": "/models/qwen35-2b", "revision": "fixed-revision"},
            "rollout": {"n": 4},
        },
        "algorithm": {"alpha": 1.0},
        "data": {
            "train_batch_size": 2,
            "train_files": "/data/train.parquet",
            "val_files": "/data/validation.parquet",
        },
        "reproduction": {
            "adapter_export_dir": f"{root}/artifacts/adapter" if adapter else None,
            "data_manifest_sha256": "a" * 64,
            "experiment_profile_id": "rtx5090-32g-qwen35-2b-v1",
            "offload_profile": "r0",
            "pilot_evidence_path": f"{root}/evidence/pilot.jsonl" if pilot else None,
            "run_seed": 42,
            "step_zero_fingerprint_path": (
                f"{root}/evidence/step_zero_fingerprint.json"
                if fingerprint
                else None
            ),
            "step_zero_reference_path": (
                "/sealed/b20/evidence/step_zero_fingerprint.json"
                if reference
                else None
            ),
            "val_data_manifest_sha256": "b" * 64,
        },
        "trainer": {
            "default_local_dir": f"{root}/checkpoints",
            "resume_from_path": resume_from,
            "resume_mode": "resume_path" if resume_from else "disable",
            "rollout_data_dir": f"{root}/logs/rollouts",
            "total_training_steps": step,
            "validation_data_dir": f"{root}/logs/validation",
        },
    }


def _sealed_index(tmp_path, configs):
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True)
    resolved = tmp_path / "resolved"
    resolved.mkdir(parents=True)
    records = {}
    for config_id, config in configs.items():
        path = resolved / f"{config_id}.yaml"
        path.write_text(
            yaml.safe_dump(config, allow_unicode=False, sort_keys=True),
            encoding="ascii",
        )
        records[config_id] = {
            "offload_profile": "r0",
            "overrides": ["reproduction/offload@_global_=r0"],
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "source_config": config_id.removesuffix("_r0"),
        }
    index = {
        "configs": records,
        "data_root": str(data_root.resolve()),
        "schema_version": 1,
        "status": "resolved",
    }
    index_path = resolved / "index.json"
    _write_index(index_path, index)
    return index_path


def _write_index(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="ascii"))


def _write_manual_binding(
    index_path,
    config_id,
    attempt_dir,
    mutate,
    **kwargs,
):
    resolved_index, original = runtime._build_attempt_binding(
        index_path,
        config_id,
        attempt_dir,
        resume_config_id=kwargs.get("resume_config_id"),
        resume_checkpoint_dir=kwargs.get("resume_checkpoint_dir"),
        resume_evidence_path=kwargs.get("resume_evidence_path"),
        step_zero_reference_path=kwargs.get("step_zero_reference_path"),
    )
    binding = copy.deepcopy(original)
    mutate(binding)
    unsigned = {key: value for key, value in binding.items() if key != "binding_sha256"}
    binding["binding_sha256"] = runtime._canonical_sha256(unsigned)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    binding_path = attempt_dir / runtime.BINDING_SOURCE_FILENAME
    binding_path.write_bytes(runtime._canonical_json_bytes(binding) + b"\n")
    return resolved_index, binding_path


def _fresh_index(tmp_path, **config_kwargs):
    return _sealed_index(
        tmp_path,
        {"fresh_r0": _config("fresh", 20, **config_kwargs)},
    )


def _bind_fresh(tmp_path, **config_kwargs):
    index = _fresh_index(tmp_path, **config_kwargs)
    attempt = tmp_path / "attempt-fresh"
    evidence = runtime.bind_attempt(index, "fresh_r0", attempt)
    return index, attempt, evidence


def _bind_g2(tmp_path, config_id="g2a_qwen35_2b_5090_r0"):
    index = _sealed_index(
        tmp_path,
        {config_id: _config("g2", 1)},
    )
    attempt = tmp_path / "attempt-g2"
    runtime.bind_attempt(index, config_id, attempt)
    return index, attempt


def _fake_torch():
    class TrustedOOM(RuntimeError):
        pass

    module = types.ModuleType("torch")
    module.OutOfMemoryError = TrustedOOM
    module.cuda = types.SimpleNamespace(OutOfMemoryError=TrustedOOM)
    return module, TrustedOOM


def _materialize_checkpoint(attempt, step):
    checkpoint = attempt / "checkpoints" / f"global_step_{step}"
    checkpoint.mkdir(parents=True)
    evidence = checkpoint / "checkpoint-evidence.json"
    evidence.write_text('{"status":"verified"}\n', encoding="ascii")
    return checkpoint, evidence


def _resume_fixture(tmp_path):
    configs = {
        "b20_r0": _config("b20", 20, fingerprint=True),
        "b40_r0": _config(
            "b40",
            40,
            resume_from="/sealed/b20/checkpoints/global_step_20",
            fingerprint=True,
        ),
        "c20_r0": _config("c20", 20),
    }
    index = _sealed_index(tmp_path, configs)
    predecessor_attempt = tmp_path / "predecessor-attempt"
    runtime.bind_attempt(index, "b20_r0", predecessor_attempt)
    checkpoint = predecessor_attempt / "checkpoints" / "global_step_20"
    checkpoint.mkdir(parents=True)
    evidence = checkpoint / "checkpoint-evidence.json"
    evidence.write_text('{"status":"verified"}\n', encoding="ascii")
    fingerprint = predecessor_attempt / "evidence" / "step_zero_fingerprint.json"
    fingerprint.parent.mkdir()
    fingerprint.write_text('{"fingerprint":"b20"}\n', encoding="ascii")
    return index, checkpoint, evidence


def test_fresh_bind_publishes_canonical_round_trippable_artifacts(tmp_path):
    index, attempt, evidence = _bind_fresh(
        tmp_path,
        adapter=True,
        fingerprint=True,
        pilot=True,
    )

    output = attempt / "runtime-bound"
    assert set(path.name for path in output.iterdir()) == {
        runtime.BOUND_CONFIG_FILENAME,
        runtime.BINDING_FILENAME,
        runtime.EVIDENCE_FILENAME,
    }
    assert (attempt / runtime.BINDING_SOURCE_FILENAME).is_file()
    assert runtime.verify_bound_config(output) == evidence
    assert evidence["status"] == "bound"
    assert evidence["source_index"]["sha256"] == _sha256(index)

    binding = _read_json(attempt / runtime.BINDING_SOURCE_FILENAME)
    unsigned = {key: value for key, value in binding.items() if key != "binding_sha256"}
    assert binding["binding_sha256"] == runtime._canonical_sha256(unsigned)
    assert _read_json(output / runtime.BINDING_FILENAME) == binding

    bound_path = output / runtime.BOUND_CONFIG_FILENAME
    bound = yaml.safe_load(bound_path.read_bytes())
    source = yaml.safe_load((index.parent / "fresh_r0.yaml").read_bytes())
    assert bound["actor_rollout_ref"] == source["actor_rollout_ref"]
    assert bound["algorithm"] == source["algorithm"]
    assert bound["data"] == source["data"]
    assert bound["trainer"]["default_local_dir"] == str(
        (attempt / "checkpoints").resolve()
    )
    assert bound["reproduction"]["runtime_telemetry_path"] == str(
        (attempt / "telemetry.json").resolve()
    )
    assert {
        change["path"] for change in evidence["leaf_changes"]
    } <= runtime._ALLOWED_CHANGED_PATHS
    assert runtime._canonical_yaml_bytes(bound) == bound_path.read_bytes()


def test_checkpoint_root_is_attempt_scoped_even_for_legacy_gate_basename(tmp_path):
    config = _config("g1", 1)
    config["trainer"]["default_local_dir"] = (
        "/sealed/gates/g1/seed42/run-20260716-step1"
    )
    index = _sealed_index(tmp_path, {"g1_step1": config})
    attempt = tmp_path / "attempt-g1"

    runtime.bind_attempt(index, "g1_step1", attempt)

    bound = yaml.safe_load(
        (attempt / "runtime-bound" / runtime.BOUND_CONFIG_FILENAME).read_bytes()
    )
    assert bound["trainer"]["default_local_dir"] == str(
        (attempt / "checkpoints").resolve()
    )


def test_resume_bind_requires_and_verifies_the_exact_sealed_predecessor(tmp_path):
    index, checkpoint, checkpoint_evidence = _resume_fixture(tmp_path)
    attempt = tmp_path / "attempt-b40"

    evidence = runtime.bind_attempt(
        index,
        "b40_r0",
        attempt,
        resume_config_id="b20_r0",
        resume_checkpoint_dir=checkpoint,
        resume_evidence_path=checkpoint_evidence,
    )

    predecessor = evidence["resume_predecessor"]
    assert predecessor["logical_config_id"] == "b20_r0"
    assert predecessor["checkpoint_step"] == 20
    assert predecessor["checkpoint_dir"] == str(checkpoint.resolve())
    fingerprint = (
        checkpoint.parent.parent / "evidence" / "step_zero_fingerprint.json"
    )
    binding = _read_json(attempt / runtime.BINDING_SOURCE_FILENAME)
    assert binding["paths"]["step_zero_fingerprint_path"] == {
        "path": str(fingerprint.resolve()),
        "sha256": _sha256(fingerprint),
    }
    bound = yaml.safe_load(
        (attempt / "runtime-bound" / runtime.BOUND_CONFIG_FILENAME).read_bytes()
    )
    assert bound["trainer"]["resume_from_path"] == str(checkpoint.resolve())
    assert bound["reproduction"]["step_zero_fingerprint_path"] == str(
        fingerprint.resolve()
    )
    assert runtime.verify_bound_config(attempt / "runtime-bound") == evidence


def test_resume_requires_and_revalidates_predecessor_step_zero_fingerprint(tmp_path):
    index, checkpoint, checkpoint_evidence = _resume_fixture(tmp_path)
    fingerprint = (
        checkpoint.parent.parent / "evidence" / "step_zero_fingerprint.json"
    )
    fingerprint.unlink()
    with pytest.raises(runtime.RuntimeBindingError, match="step-zero fingerprint"):
        runtime.bind_attempt(
            index,
            "b40_r0",
            tmp_path / "attempt-missing-fingerprint",
            resume_config_id="b20_r0",
            resume_checkpoint_dir=checkpoint,
            resume_evidence_path=checkpoint_evidence,
        )

    fingerprint.write_text('{"fingerprint":"b20"}\n', encoding="ascii")
    attempt = tmp_path / "attempt-fingerprint-drift"
    runtime.bind_attempt(
        index,
        "b40_r0",
        attempt,
        resume_config_id="b20_r0",
        resume_checkpoint_dir=checkpoint,
        resume_evidence_path=checkpoint_evidence,
    )
    fingerprint.write_text('{"fingerprint":"changed"}\n', encoding="ascii")
    with pytest.raises(runtime.RuntimeBindingError, match="bytes changed"):
        runtime.verify_bound_config(attempt / "runtime-bound")


def test_c_resume_binds_predecessor_fingerprint_and_original_b20_reference(tmp_path):
    configs = {
        "b20_r0": _config("b20", 20, fingerprint=True),
        "c20_r0": _config(
            "c20", 20, fingerprint=True, reference=True
        ),
        "c40_r0": _config(
            "c40",
            40,
            resume_from="/sealed/c20/checkpoints/global_step_20",
            fingerprint=True,
            reference=True,
        ),
    }
    index = _sealed_index(tmp_path, configs)
    b20_attempt = tmp_path / "b20-attempt"
    runtime.bind_attempt(index, "b20_r0", b20_attempt)
    b20_fingerprint = b20_attempt / "evidence" / "step_zero_fingerprint.json"
    b20_fingerprint.parent.mkdir()
    b20_fingerprint.write_text('{"fingerprint":"b20"}\n', encoding="ascii")

    c20_attempt = tmp_path / "c20-attempt"
    runtime.bind_attempt(
        index,
        "c20_r0",
        c20_attempt,
        step_zero_reference_path=b20_fingerprint,
    )
    checkpoint = c20_attempt / "checkpoints" / "global_step_20"
    checkpoint.mkdir(parents=True)
    checkpoint_evidence = checkpoint / "checkpoint-evidence.json"
    checkpoint_evidence.write_text('{"status":"verified"}\n', encoding="ascii")
    c20_fingerprint = c20_attempt / "evidence" / "step_zero_fingerprint.json"
    c20_fingerprint.parent.mkdir()
    c20_fingerprint.write_text('{"fingerprint":"c20"}\n', encoding="ascii")

    attempt = tmp_path / "c40-attempt"
    runtime.bind_attempt(
        index,
        "c40_r0",
        attempt,
        resume_config_id="c20_r0",
        resume_checkpoint_dir=checkpoint,
        resume_evidence_path=checkpoint_evidence,
        step_zero_reference_path=b20_fingerprint,
    )

    bound = yaml.safe_load(
        (attempt / "runtime-bound" / runtime.BOUND_CONFIG_FILENAME).read_bytes()
    )
    assert bound["reproduction"]["step_zero_fingerprint_path"] == str(
        c20_fingerprint.resolve()
    )
    assert bound["reproduction"]["step_zero_reference_path"] == str(
        b20_fingerprint.resolve()
    )


def test_b_multihop_resume_keeps_the_original_b20_fingerprint(tmp_path):
    configs = {
        "b20_r0": _config("b20", 20, fingerprint=True),
        "b40_r0": _config(
            "b40",
            40,
            resume_from="/sealed/b20/checkpoints/global_step_20",
            fingerprint=True,
        ),
        "b60_r0": _config(
            "b60",
            60,
            resume_from="/sealed/b40/checkpoints/global_step_40",
            fingerprint=True,
        ),
        "b80_r0": _config(
            "b80",
            80,
            resume_from="/sealed/b60/checkpoints/global_step_60",
            fingerprint=True,
        ),
    }
    index = _sealed_index(tmp_path, configs)
    attempts = {"b20": tmp_path / "b20-attempt"}
    runtime.bind_attempt(index, "b20_r0", attempts["b20"])
    fingerprint = attempts["b20"] / "evidence" / "step_zero_fingerprint.json"
    fingerprint.parent.mkdir()
    fingerprint.write_text('{"fingerprint":"b20"}\n', encoding="ascii")
    fingerprint_record = {"path": str(fingerprint.resolve()), "sha256": _sha256(fingerprint)}
    checkpoint, checkpoint_evidence = _materialize_checkpoint(attempts["b20"], 20)

    for current, step, predecessor, predecessor_step in (
        ("b40", 40, "b20", 20),
        ("b60", 60, "b40", 40),
        ("b80", 80, "b60", 60),
    ):
        attempt = tmp_path / f"{current}-attempt"
        runtime.bind_attempt(
            index,
            f"{current}_r0",
            attempt,
            resume_config_id=f"{predecessor}_r0",
            resume_checkpoint_dir=checkpoint,
            resume_evidence_path=checkpoint_evidence,
        )
        attempts[current] = attempt
        binding = _read_json(attempt / runtime.BINDING_SOURCE_FILENAME)
        bound = yaml.safe_load(
            (attempt / "runtime-bound" / runtime.BOUND_CONFIG_FILENAME).read_bytes()
        )
        assert binding["paths"]["step_zero_fingerprint_path"] == fingerprint_record
        assert bound["reproduction"]["step_zero_fingerprint_path"] == str(
            fingerprint.resolve()
        )
        assert not (
            attempt / "evidence" / "step_zero_fingerprint.json"
        ).exists()
        assert runtime.verify_bound_config(attempt / "runtime-bound")[
            "config_id"
        ] == f"{current}_r0"
        assert binding["resume_predecessor"]["checkpoint_step"] == predecessor_step
        if step < 80:
            checkpoint, checkpoint_evidence = _materialize_checkpoint(attempt, step)

    fingerprint.write_text('{"fingerprint":"tampered"}\n', encoding="ascii")
    with pytest.raises(runtime.RuntimeBindingError, match="fingerprint.*bytes changed"):
        runtime.verify_bound_config(attempts["b80"] / "runtime-bound")


def test_c_multihop_resume_keeps_c20_fingerprint_and_explicit_b20_reference(
    tmp_path,
):
    configs = {
        "b20_r0": _config("b20", 20, fingerprint=True),
        "c20_r0": _config("c20", 20, fingerprint=True, reference=True),
        "c40_r0": _config(
            "c40",
            40,
            resume_from="/sealed/c20/checkpoints/global_step_20",
            fingerprint=True,
            reference=True,
        ),
        "c60_r0": _config(
            "c60",
            60,
            resume_from="/sealed/c40/checkpoints/global_step_40",
            fingerprint=True,
            reference=True,
        ),
        "c80_r0": _config(
            "c80",
            80,
            resume_from="/sealed/c60/checkpoints/global_step_60",
            fingerprint=True,
            reference=True,
        ),
    }
    index = _sealed_index(tmp_path, configs)
    b20_attempt = tmp_path / "b20-attempt"
    runtime.bind_attempt(index, "b20_r0", b20_attempt)
    b20_fingerprint = b20_attempt / "evidence" / "step_zero_fingerprint.json"
    b20_fingerprint.parent.mkdir()
    b20_fingerprint.write_text('{"fingerprint":"b20"}\n', encoding="ascii")
    b20_record = {
        "path": str(b20_fingerprint.resolve()),
        "sha256": _sha256(b20_fingerprint),
    }

    c20_attempt = tmp_path / "c20-attempt"
    runtime.bind_attempt(
        index,
        "c20_r0",
        c20_attempt,
        step_zero_reference_path=b20_fingerprint,
    )
    c20_fingerprint = c20_attempt / "evidence" / "step_zero_fingerprint.json"
    c20_fingerprint.parent.mkdir()
    c20_fingerprint.write_text('{"fingerprint":"c20"}\n', encoding="ascii")
    c20_record = {
        "path": str(c20_fingerprint.resolve()),
        "sha256": _sha256(c20_fingerprint),
    }
    checkpoint, checkpoint_evidence = _materialize_checkpoint(c20_attempt, 20)

    for current, step, predecessor, predecessor_step in (
        ("c40", 40, "c20", 20),
        ("c60", 60, "c40", 40),
        ("c80", 80, "c60", 60),
    ):
        attempt = tmp_path / f"{current}-attempt"
        runtime.bind_attempt(
            index,
            f"{current}_r0",
            attempt,
            resume_config_id=f"{predecessor}_r0",
            resume_checkpoint_dir=checkpoint,
            resume_evidence_path=checkpoint_evidence,
            step_zero_reference_path=b20_fingerprint,
        )
        binding = _read_json(attempt / runtime.BINDING_SOURCE_FILENAME)
        bound = yaml.safe_load(
            (attempt / "runtime-bound" / runtime.BOUND_CONFIG_FILENAME).read_bytes()
        )
        assert binding["paths"]["step_zero_fingerprint_path"] == c20_record
        assert binding["paths"]["step_zero_reference"] == b20_record
        assert bound["reproduction"]["step_zero_fingerprint_path"] == str(
            c20_fingerprint.resolve()
        )
        assert bound["reproduction"]["step_zero_reference_path"] == str(
            b20_fingerprint.resolve()
        )
        assert not (
            attempt / "evidence" / "step_zero_fingerprint.json"
        ).exists()
        assert runtime.verify_bound_config(attempt / "runtime-bound")[
            "config_id"
        ] == f"{current}_r0"
        assert binding["resume_predecessor"]["checkpoint_step"] == predecessor_step
        if step < 80:
            checkpoint, checkpoint_evidence = _materialize_checkpoint(attempt, step)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({}, "resume_predecessor"),
        ({"resume_config_id": "c20_r0"}, "logical config"),
    ],
)
def test_resume_bind_rejects_missing_or_wrong_arm(tmp_path, overrides, match):
    index, checkpoint, checkpoint_evidence = _resume_fixture(tmp_path)
    kwargs = {
        "resume_config_id": "b20_r0",
        "resume_checkpoint_dir": checkpoint,
        "resume_evidence_path": checkpoint_evidence,
        **overrides,
    }
    if not overrides:
        kwargs = {}

    with pytest.raises(runtime.RuntimeBindingError, match=match):
        runtime.bind_attempt(index, "b40_r0", tmp_path / "attempt", **kwargs)


def test_resume_bind_rejects_wrong_step_directory_and_evidence_sha(tmp_path):
    index, checkpoint, checkpoint_evidence = _resume_fixture(tmp_path)
    wrong_step = checkpoint.with_name("global_step_19")
    wrong_step.mkdir()
    wrong_evidence = wrong_step / checkpoint_evidence.name
    wrong_evidence.write_bytes(checkpoint_evidence.read_bytes())
    with pytest.raises(runtime.RuntimeBindingError, match="checkpoint directory"):
        runtime.bind_attempt(
            index,
            "b40_r0",
            tmp_path / "attempt-step",
            resume_config_id="b20_r0",
            resume_checkpoint_dir=wrong_step,
            resume_evidence_path=wrong_evidence,
        )

    attempt = tmp_path / "attempt-sha"
    resolved_index, binding_path = _write_manual_binding(
        index,
        "b40_r0",
        attempt,
        lambda binding: binding["resume_predecessor"].update(
            {"evidence_sha256": "0" * 64}
        ),
        resume_config_id="b20_r0",
        resume_checkpoint_dir=checkpoint,
        resume_evidence_path=checkpoint_evidence,
    )
    with pytest.raises(runtime.RuntimeBindingError, match="evidence bytes changed"):
        runtime.bind_resolved_config(
            resolved_index,
            "b40_r0",
            binding_path,
            attempt / "runtime-bound",
        )


def test_step_zero_reference_and_nullable_outputs_follow_the_sealed_config(tmp_path):
    configs = {
        "b20_r0": _config(
            "b20", 20, adapter=True, fingerprint=True, reference=False, pilot=True
        ),
        "c20_r0": _config(
            "c20", 20, adapter=True, fingerprint=True, reference=True, pilot=True
        ),
        "plain_r0": _config(
            "plain", 1, adapter=False, fingerprint=False, reference=False, pilot=False
        ),
    }
    index = _sealed_index(tmp_path, configs)
    reference = tmp_path / "step-zero.json"
    reference.write_text('{"fingerprint":"fixed"}\n', encoding="ascii")

    with pytest.raises(runtime.RuntimeBindingError, match="requires --step-zero-reference"):
        runtime.bind_attempt(index, "c20_r0", tmp_path / "c-missing")
    with pytest.raises(runtime.RuntimeBindingError, match="does not accept"):
        runtime.bind_attempt(
            index,
            "b20_r0",
            tmp_path / "b-extra",
            step_zero_reference_path=reference,
        )

    runtime.bind_attempt(
        index,
        "c20_r0",
        tmp_path / "c-valid",
        step_zero_reference_path=reference,
    )
    c_binding = _read_json(
        tmp_path / "c-valid" / runtime.BINDING_SOURCE_FILENAME
    )
    assert c_binding["paths"]["step_zero_reference"] == {
        "path": str(reference.resolve()),
        "sha256": _sha256(reference),
    }
    assert c_binding["paths"]["adapter_dir"] is not None
    assert c_binding["paths"]["step_zero_fingerprint_path"] is not None
    assert c_binding["paths"]["pilot_evidence_path"] is not None

    runtime.bind_attempt(index, "plain_r0", tmp_path / "plain-valid")
    plain_binding = _read_json(
        tmp_path / "plain-valid" / runtime.BINDING_SOURCE_FILENAME
    )
    assert plain_binding["paths"]["adapter_dir"] is None
    assert plain_binding["paths"]["step_zero_fingerprint_path"] is None
    assert plain_binding["paths"]["step_zero_reference"] is None
    assert plain_binding["paths"]["pilot_evidence_path"] is None


@pytest.mark.parametrize(
    "path_name",
    ["adapter_dir", "step_zero_fingerprint_path", "pilot_evidence_path"],
)
def test_manual_binding_cannot_change_sealed_output_nullability(tmp_path, path_name):
    index = _fresh_index(
        tmp_path,
        adapter=True,
        fingerprint=True,
        pilot=True,
    )
    attempt = tmp_path / f"attempt-{path_name}"
    resolved_index, binding_path = _write_manual_binding(
        index,
        "fresh_r0",
        attempt,
        lambda binding: binding["paths"].update({path_name: None}),
    )

    with pytest.raises(runtime.RuntimeBindingError, match="nullability"):
        runtime.bind_resolved_config(
            resolved_index,
            "fresh_r0",
            binding_path,
            attempt / "runtime-bound",
        )


def test_index_contract_rejects_old_bindings_field_wrong_sha_and_extra_file(tmp_path):
    index = _fresh_index(tmp_path)
    index_value = _read_json(index)
    index_value["configs"]["fresh_r0"]["bindings"] = {}
    _write_index(index, index_value)
    with pytest.raises(runtime.RuntimeBindingError, match="keys differ"):
        runtime.bind_attempt(index, "fresh_r0", tmp_path / "attempt-bindings")

    index = _fresh_index(tmp_path / "wrong-sha")
    index_value = _read_json(index)
    index_value["configs"]["fresh_r0"]["sha256"] = "0" * 64
    _write_index(index, index_value)
    with pytest.raises(runtime.RuntimeBindingError, match="bytes changed"):
        runtime.bind_attempt(index, "fresh_r0", tmp_path / "attempt-sha")

    index = _fresh_index(tmp_path / "extra")
    (index.parent / "unexpected.txt").write_text("extra\n", encoding="ascii")
    with pytest.raises(runtime.RuntimeBindingError, match="inventory"):
        runtime.bind_attempt(index, "fresh_r0", tmp_path / "attempt-extra")


@pytest.mark.parametrize(
    ("field", "match"),
    [
        ("index_sha256", "index SHA-256 changed"),
        ("resolved_config_sha256", "selected a different config SHA-256"),
    ],
)
def test_runtime_binding_rejects_wrong_index_or_selected_config_sha(
    tmp_path, field, match
):
    index = _fresh_index(tmp_path)
    attempt = tmp_path / f"attempt-{field}"
    resolved_index, binding_path = _write_manual_binding(
        index,
        "fresh_r0",
        attempt,
        lambda binding: binding.update({field: "0" * 64}),
    )

    with pytest.raises(runtime.RuntimeBindingError, match=match):
        runtime.bind_resolved_config(
            resolved_index,
            "fresh_r0",
            binding_path,
            attempt / "runtime-bound",
        )


def test_resume_binding_rejects_wrong_predecessor_config_sha(tmp_path):
    index, checkpoint, checkpoint_evidence = _resume_fixture(tmp_path)
    attempt = tmp_path / "attempt-predecessor-sha"
    resolved_index, binding_path = _write_manual_binding(
        index,
        "b40_r0",
        attempt,
        lambda binding: binding["resume_predecessor"].update(
            {"resolved_config_sha256": "0" * 64}
        ),
        resume_config_id="b20_r0",
        resume_checkpoint_dir=checkpoint,
        resume_evidence_path=checkpoint_evidence,
    )

    with pytest.raises(runtime.RuntimeBindingError, match="selected a different config"):
        runtime.bind_resolved_config(
            resolved_index,
            "b40_r0",
            binding_path,
            attempt / "runtime-bound",
        )


def test_binding_rejects_extra_keys_scientific_override_and_path_escape(tmp_path):
    index = _fresh_index(tmp_path)

    for label, mutate, match in (
        (
            "extra",
            lambda binding: binding.update({"model_path": "/other/model"}),
            "keys differ",
        ),
        (
            "scientific",
            lambda binding: binding["paths"].update({"train_batch_size": 1}),
            "keys differ",
        ),
        (
            "escape",
            lambda binding: binding["paths"].update(
                {"checkpoint_dir": str((tmp_path / "outside").resolve())}
            ),
            "escapes|registered attempt path",
        ),
    ):
        attempt = tmp_path / f"attempt-{label}"
        resolved_index, binding_path = _write_manual_binding(
            index,
            "fresh_r0",
            attempt,
            mutate,
        )
        with pytest.raises(runtime.RuntimeBindingError, match=match):
            runtime.bind_resolved_config(
                resolved_index,
                "fresh_r0",
                binding_path,
                attempt / "runtime-bound",
            )


@pytest.mark.parametrize(
    "target",
    ["yaml", "evidence", "copied_binding", "source_binding", "source_config", "index"],
)
def test_verification_rejects_every_tampered_identity_layer(tmp_path, target):
    index, attempt, _ = _bind_fresh(tmp_path)
    output = attempt / "runtime-bound"
    targets = {
        "yaml": output / runtime.BOUND_CONFIG_FILENAME,
        "evidence": output / runtime.EVIDENCE_FILENAME,
        "copied_binding": output / runtime.BINDING_FILENAME,
        "source_binding": attempt / runtime.BINDING_SOURCE_FILENAME,
        "source_config": index.parent / "fresh_r0.yaml",
        "index": index,
    }
    with targets[target].open("ab") as handle:
        handle.write(b" \n")

    with pytest.raises(runtime.RuntimeBindingError):
        runtime.verify_bound_config(output)


def test_atomic_publication_refuses_overwrite_and_cleans_failed_staging(
    tmp_path, monkeypatch
):
    index, attempt, _ = _bind_fresh(tmp_path / "overwrite")
    output = attempt / "runtime-bound"
    original = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError, match="overwrite"):
        runtime.bind_attempt(index, "fresh_r0", attempt)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == original

    index = _fresh_index(tmp_path / "failure")
    failed_attempt = tmp_path / "attempt-failure"

    def fail_staging(directory, *, prepared=None):
        if prepared is not None:
            raise runtime.RuntimeBindingError("injected staging verification failure")
        return runtime._verify_bound_directory(directory, prepared=prepared)

    monkeypatch.setattr(runtime, "_verify_bound_directory", fail_staging)
    with pytest.raises(runtime.RuntimeBindingError, match="injected"):
        runtime.bind_attempt(index, "fresh_r0", failed_attempt)
    assert not (failed_attempt / "runtime-bound").exists()
    assert not list(failed_attempt.glob(".runtime-bound.staging-*"))
    assert (failed_attempt / runtime.BINDING_SOURCE_FILENAME).is_file()


@pytest.mark.parametrize("kind", ["index", "config", "binding"])
def test_symlinked_identity_files_are_rejected(tmp_path, kind):
    index = _fresh_index(tmp_path)
    if kind == "index":
        alias = tmp_path / "index-link.json"
        try:
            os.symlink(index, alias)
        except (NotImplementedError, OSError) as exc:
            pytest.skip(f"file symlinks are unavailable: {exc}")
        with pytest.raises(runtime.RuntimeBindingError, match="symlink"):
            runtime.bind_attempt(alias, "fresh_r0", tmp_path / "attempt")
        return

    if kind == "config":
        config = index.parent / "fresh_r0.yaml"
        target = tmp_path / "config-target.yaml"
        config.replace(target)
        try:
            os.symlink(target, config)
        except (NotImplementedError, OSError) as exc:
            pytest.skip(f"file symlinks are unavailable: {exc}")
        with pytest.raises(runtime.RuntimeBindingError, match="symlink"):
            runtime.bind_attempt(index, "fresh_r0", tmp_path / "attempt")
        return

    _, attempt, _ = _bind_fresh(tmp_path / "bound")
    binding = attempt / runtime.BINDING_SOURCE_FILENAME
    target = attempt / "binding-target.json"
    binding.replace(target)
    try:
        os.symlink(target, binding)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"file symlinks are unavailable: {exc}")
    with pytest.raises(runtime.RuntimeBindingError, match="symlink"):
        runtime.verify_bound_config(attempt / "runtime-bound")


def test_module_import_and_bind_verify_never_import_training_stack(
    tmp_path, monkeypatch
):
    command = (
        "import sys; before=set(sys.modules); "
        "import scripts.cloud.run_resolved_training; "
        "loaded=set(sys.modules)-before; "
        "blocked={'torch','ray','omegaconf','verl.trainer.main_ppo'}; "
        "assert not (loaded & blocked), sorted(loaded & blocked)"
    )
    subprocess.run(
        [sys.executable, "-c", command],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )

    blocked = ("torch", "ray", "omegaconf", "verl.trainer.main_ppo")
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in blocked or any(name.startswith(f"{value}.") for value in blocked):
            raise AssertionError(f"unexpected training import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    _, attempt, _ = _bind_fresh(tmp_path)
    runtime.verify_bound_config(attempt / "runtime-bound")


def test_run_verifies_before_lazy_import_and_invokes_trainer(tmp_path, monkeypatch):
    _, attempt, evidence = _bind_fresh(tmp_path)
    calls = []

    class FakeOmegaConf:
        @staticmethod
        def load(path):
            calls.append(("load", Path(path)))
            return {"loaded": str(path)}

        @staticmethod
        def resolve(config):
            calls.append(("resolve", config))

    omega = types.ModuleType("omegaconf")
    omega.OmegaConf = FakeOmegaConf
    trainer_main = types.ModuleType("verl.trainer.main_ppo")
    trainer_main.run_ppo = lambda config: calls.append(("run", config))
    verl_package = types.ModuleType("verl")
    verl_package.__path__ = []
    trainer_package = types.ModuleType("verl.trainer")
    trainer_package.__path__ = []
    verl_package.trainer = trainer_package
    trainer_package.main_ppo = trainer_main
    monkeypatch.setitem(sys.modules, "omegaconf", omega)
    monkeypatch.setitem(sys.modules, "verl", verl_package)
    monkeypatch.setitem(sys.modules, "verl.trainer", trainer_package)
    monkeypatch.setitem(sys.modules, "verl.trainer.main_ppo", trainer_main)

    assert runtime.run_bound_config(attempt / "runtime-bound") == evidence
    assert [call[0] for call in calls] == ["load", "resolve", "run"]

    calls.clear()
    with (attempt / "runtime-bound" / runtime.BOUND_CONFIG_FILENAME).open("ab") as handle:
        handle.write(b"tampered\n")
    with pytest.raises(runtime.RuntimeBindingError):
        runtime.run_bound_config(attempt / "runtime-bound")
    assert calls == []


@pytest.mark.parametrize("link_name", ["__cause__", "__context__"])
def test_trusted_cuda_oom_uses_type_identity_across_exception_chain(link_name):
    torch_module, trusted_type = _fake_torch()
    oom = trusted_type("capacity exhausted")
    wrapper = RuntimeError("trainer wrapper")
    setattr(wrapper, link_name, oom)

    matched = runtime._trusted_cuda_oom(wrapper, torch_module=torch_module)

    assert matched == (oom, "torch.OutOfMemoryError")


def test_trusted_cuda_oom_conservatively_unwraps_ray_cause():
    torch_module, trusted_type = _fake_torch()
    oom = trusted_type("capacity exhausted")

    class FakeRayTaskError(RuntimeError):
        __module__ = "ray.exceptions"

        def as_instanceof_cause(self):
            return oom

    wrapper = FakeRayTaskError("remote task failed")

    assert runtime._trusted_cuda_oom(
        wrapper, torch_module=torch_module
    ) == (oom, "torch.OutOfMemoryError")


def test_trusted_cuda_oom_rejects_messages_names_and_non_ray_converters():
    torch_module, trusted_type = _fake_torch()

    class OutOfMemoryError(RuntimeError):
        pass

    class UntrustedWrapper(RuntimeError):
        def as_instanceof_cause(self):
            return trusted_type("hidden behind an untrusted converter")

    errors = (
        RuntimeError("torch.OutOfMemoryError: CUDA out of memory"),
        OutOfMemoryError("same class name"),
        UntrustedWrapper("not a Ray exception"),
    )

    assert all(
        runtime._trusted_cuda_oom(error, torch_module=torch_module) is None
        for error in errors
    )


def test_trusted_cuda_oom_preserves_distinct_cuda_exception_label():
    class TopLevelOOM(RuntimeError):
        pass

    class CudaOOM(RuntimeError):
        pass

    torch_module = types.ModuleType("torch")
    torch_module.OutOfMemoryError = TopLevelOOM
    torch_module.cuda = types.SimpleNamespace(OutOfMemoryError=CudaOOM)
    oom = CudaOOM("capacity exhausted")

    assert runtime._trusted_cuda_oom(
        oom, torch_module=torch_module
    ) == (oom, "torch.cuda.OutOfMemoryError")


def test_trusted_cuda_oom_prefers_exact_cuda_subclass_over_top_level_base():
    class TopLevelOOM(RuntimeError):
        pass

    class CudaOOM(TopLevelOOM):
        pass

    torch_module = types.ModuleType("torch")
    torch_module.OutOfMemoryError = TopLevelOOM
    torch_module.cuda = types.SimpleNamespace(OutOfMemoryError=CudaOOM)
    oom = CudaOOM("capacity exhausted")

    assert runtime._trusted_cuda_oom(
        oom, torch_module=torch_module
    ) == (oom, "torch.cuda.OutOfMemoryError")


def test_capacity_stop_allowlist_is_exactly_the_sealed_g2_matrix():
    expected = {
        f"{name}_{profile}"
        for name in (
            "g2a_qwen35_2b_5090",
            "g2b_qwen35_2b_5090_step1",
            "g2b_qwen35_2b_5090_resume5",
            "g2_length_stress_qwen35_2b_5090",
        )
        for profile in ("r0", "r1")
    }

    assert runtime._G2_CAPACITY_CONFIG_IDS == expected


def test_g2_cuda_oom_publishes_canonical_terminal_evidence_and_exits_43(
    tmp_path, monkeypatch, capsys
):
    _, attempt = _bind_g2(tmp_path)
    torch_module, trusted_type = _fake_torch()
    oom = trusted_type("capacity exhausted")
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    def fail_training(_):
        raise oom

    monkeypatch.setattr(runtime, "run_bound_config", fail_training)

    assert runtime.main(["run", "--attempt-dir", str(attempt)]) == 43
    output = capsys.readouterr()
    assert output.err == ""
    printed = json.loads(output.out)
    marker = attempt / "evidence" / runtime.CAPACITY_STOP_FILENAME
    stop = _read_json(marker)
    assert printed == stop == runtime.verify_capacity_stop(attempt)
    assert marker.read_bytes() == runtime._canonical_json_bytes(stop) + b"\n"
    unsigned = {
        key: value for key, value in stop.items() if key != "capacity_stop_sha256"
    }
    assert stop["capacity_stop_sha256"] == runtime._canonical_sha256(unsigned)
    assert stop["exception_type"] == "torch.OutOfMemoryError"
    assert stop["runtime_bound_evidence"] == {
        "path": str((attempt / "runtime-bound" / runtime.EVIDENCE_FILENAME).resolve()),
        "file_sha256": _sha256(
            attempt / "runtime-bound" / runtime.EVIDENCE_FILENAME
        ),
        "evidence_sha256": _read_json(
            attempt / "runtime-bound" / runtime.EVIDENCE_FILENAME
        )["evidence_sha256"],
    }
    assert stop["runtime_binding"] == {
        "path": str((attempt / runtime.BINDING_SOURCE_FILENAME).resolve()),
        "file_sha256": _sha256(attempt / runtime.BINDING_SOURCE_FILENAME),
        "binding_sha256": _read_json(
            attempt / runtime.BINDING_SOURCE_FILENAME
        )["binding_sha256"],
    }


def test_non_oom_g2_failure_stays_blocked_without_capacity_stop(
    tmp_path, monkeypatch, capsys
):
    _, attempt = _bind_g2(tmp_path)
    torch_module, _ = _fake_torch()
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    def fail_training(_):
        raise RuntimeError("ordinary trainer failure")

    monkeypatch.setattr(runtime, "run_bound_config", fail_training)

    assert runtime.main(["run", "--attempt-dir", str(attempt)]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err)["status"] == "blocked"
    assert not (attempt / "evidence" / runtime.CAPACITY_STOP_FILENAME).exists()


def test_non_g2_oom_stays_blocked_without_capacity_stop(
    tmp_path, monkeypatch, capsys
):
    _, attempt, _ = _bind_fresh(tmp_path)
    torch_module, trusted_type = _fake_torch()
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    def fail_training(_):
        raise trusted_type("capacity exhausted")

    monkeypatch.setattr(runtime, "run_bound_config", fail_training)

    assert runtime.main(["run", "--attempt-dir", str(attempt)]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err)["status"] == "blocked"
    assert not (attempt / "evidence" / runtime.CAPACITY_STOP_FILENAME).exists()


def test_capacity_stop_verification_rejects_rehashed_tampering(tmp_path):
    _, attempt = _bind_g2(tmp_path)
    torch_module, trusted_type = _fake_torch()
    runtime.publish_capacity_stop_for_exception(
        attempt / "runtime-bound",
        trusted_type("capacity exhausted"),
        torch_module=torch_module,
    )
    marker = attempt / "evidence" / runtime.CAPACITY_STOP_FILENAME
    stop = _read_json(marker)
    stop["runtime_binding"]["file_sha256"] = "0" * 64
    unsigned = {
        key: value for key, value in stop.items() if key != "capacity_stop_sha256"
    }
    stop["capacity_stop_sha256"] = runtime._canonical_sha256(unsigned)
    marker.write_bytes(runtime._canonical_json_bytes(stop) + b"\n")

    with pytest.raises(runtime.RuntimeBindingError, match="binding bytes changed"):
        runtime.verify_capacity_stop(attempt)


def test_capacity_stop_publication_refuses_overwrite(tmp_path):
    _, attempt = _bind_g2(tmp_path)
    torch_module, trusted_type = _fake_torch()
    output = attempt / "runtime-bound"
    runtime.publish_capacity_stop_for_exception(
        output,
        trusted_type("first OOM"),
        torch_module=torch_module,
    )
    marker = attempt / "evidence" / runtime.CAPACITY_STOP_FILENAME
    original = marker.read_bytes()

    with pytest.raises(FileExistsError, match="overwrite"):
        runtime.publish_capacity_stop_for_exception(
            output,
            trusted_type("second OOM"),
            torch_module=torch_module,
        )
    assert marker.read_bytes() == original


def test_capacity_stop_publication_refuses_preoccupied_marker(tmp_path):
    _, attempt = _bind_g2(tmp_path)
    torch_module, trusted_type = _fake_torch()
    marker = attempt / "evidence" / runtime.CAPACITY_STOP_FILENAME
    marker.parent.mkdir()
    marker.write_bytes(b"preoccupied\n")

    with pytest.raises(FileExistsError, match="overwrite"):
        runtime.publish_capacity_stop_for_exception(
            attempt / "runtime-bound",
            trusted_type("capacity exhausted"),
            torch_module=torch_module,
        )
    assert marker.read_bytes() == b"preoccupied\n"


def test_failed_capacity_stop_self_verification_never_exits_43(
    tmp_path, monkeypatch, capsys
):
    _, attempt = _bind_g2(tmp_path)
    torch_module, trusted_type = _fake_torch()
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    def fail_training(_):
        raise trusted_type("capacity exhausted")

    def fail_verification(_):
        raise runtime.RuntimeBindingError("injected capacity-stop verification failure")

    monkeypatch.setattr(runtime, "run_bound_config", fail_training)
    monkeypatch.setattr(runtime, "verify_capacity_stop", fail_verification)

    assert runtime.main(["run", "--attempt-dir", str(attempt)]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err)["status"] == "blocked"


def test_failed_capacity_stop_directory_sync_never_exits_43(
    tmp_path, monkeypatch, capsys
):
    _, attempt = _bind_g2(tmp_path)
    torch_module, trusted_type = _fake_torch()
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    def fail_training(_):
        raise trusted_type("capacity exhausted")

    def fail_sync(_):
        raise OSError("injected directory sync failure")

    monkeypatch.setattr(runtime, "run_bound_config", fail_training)
    monkeypatch.setattr(runtime, "_sync_parent", fail_sync)

    assert runtime.main(["run", "--attempt-dir", str(attempt)]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err)["status"] == "blocked"


def test_cli_has_only_runtime_binding_arguments_and_round_trips(tmp_path, capsys):
    help_text = runtime._parser().format_help()
    bind_help = runtime._parser()._subparsers._group_actions[0].choices[
        "bind"
    ].format_help()
    forbidden = (
        "--model",
        "--data",
        "--batch",
        "--group",
        "--offload",
        "--reward",
        "--alpha",
    )
    assert all(option not in bind_help for option in forbidden)
    assert "{bind,verify,run}" in help_text

    index = _fresh_index(tmp_path)
    attempt = tmp_path / "attempt-cli"
    assert runtime.main(
        [
            "bind",
            "--index",
            str(index),
            "--config-id",
            "fresh_r0",
            "--attempt-dir",
            str(attempt),
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "bound"
    assert runtime.main(["verify", "--attempt-dir", str(attempt)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "bound"
