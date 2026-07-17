from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.cloud import eval_matrix
from taskutils.data_synthesis.reproduction_manifest import canonical_json_bytes
from taskutils.memory_eval import reproduction_runner as runner
from taskutils.memory_eval.reproduction_metrics import (
    EvaluationContractError,
    evaluate_output,
    stratified_paired_bootstrap_delta,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _write_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _write_canonical(path: Path, value: dict) -> str:
    return _write_bytes(path, canonical_json_bytes(value) + b"\n")


def _source_config(tmp_path: Path, level: str) -> tuple[Path, dict]:
    step = 40 if level == "l1" else 80
    sample_count = 32 if level == "l1" else 64
    b_name = f"b{step}"
    c_name = f"c{step}"
    bundles = {}
    for dataset in eval_matrix.DATASETS:
        bundle = tmp_path / "bundles" / dataset
        bundle.mkdir(parents=True)
        bundles[dataset] = {
            "bundle_dir": str(bundle.resolve()),
            "manifest_sha256": _digest(f"manifest-{dataset}"),
        }
    artifact_keys = {
        "artifact_kind",
        "artifact_path",
        "training_step",
        "training_config_source_id",
        "training_config_id",
        "training_config_path",
        "training_config_sha256",
        "adapter_metadata_sha256",
        "checkpoint_extra_state_sha256",
    }

    def artifact(name: str, kind: str, training_step: int):
        value = {key: None for key in artifact_keys}
        value.update(
            artifact_kind=kind,
            training_step=training_step,
            training_config_source_id=(
                None if kind == "base" else f"{name}_qwen35_2b_5090"
            ),
        )
        return value

    evaluation = {
        "schema_version": 2,
        "delivery_level": level,
        "evaluator_version": eval_matrix.EVALUATOR_VERSION,
        "runner_module": "taskutils.memory_eval.reproduction_runner",
        "backend": "transformers",
        "base_model_id": eval_matrix.BASE_MODEL,
        "revision": eval_matrix.MODEL_REVISION,
        "tokenizer_id": eval_matrix.BASE_MODEL,
        "tokenizer_revision": eval_matrix.MODEL_REVISION,
        "template_revision": runner.PROMPT_TEMPLATE_REVISION,
        "attention_implementation": "sdpa",
        "dtype": "bfloat16",
        "device": "cuda:0",
        "local_files_only": True,
        "seed": 42,
        "datasets": bundles,
        "document_variants": [200, 800],
        "sample_count_per_stratum": sample_count,
        "required_cell_count": 20,
        "primary_callback_mode": "learned",
        "callback_ablation": {
            "artifact": c_name,
            "modes": ["learned", "none", "fixed_question"],
        },
        "artifacts": {
            "base": artifact("base", "base", 0),
            b_name: artifact(b_name, "adapter", step),
            c_name: artifact(c_name, "adapter", step),
        },
        "decode": {
            "greedy": True,
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "n": 1,
            "chunk_size": 5000,
            "memory_max_tokens": 768,
            "final_max_tokens": 512,
        },
        "timeouts_seconds": {
            "model_load": 1800,
            "sample": 1800,
            "task": 86400,
        },
        "output_root": str((tmp_path / "evaluation").resolve()),
    }
    path = tmp_path / f"eval-{level}.json"
    path.write_text(
        json.dumps({"reproduction_evaluation": evaluation}, sort_keys=True),
        encoding="ascii",
    )
    return path, evaluation


def _runtime_binding(
    tmp_path: Path, config_path: Path, evaluation: dict
) -> tuple[Path, dict, dict[str, dict]]:
    level = evaluation["delivery_level"]
    step = 40 if level == "l1" else 80
    capacity = tmp_path / "capacity-profile.json"
    capacity_sha = _write_bytes(capacity, b"capacity-profile\n")
    metadata: dict[str, dict] = {}
    artifact_bindings = {}
    resume_probes = {}
    for name in (f"b{step}", f"c{step}"):
        adapter_dir = tmp_path / "adapters" / name
        adapter_dir.mkdir(parents=True)
        training_config = tmp_path / "resolved" / f"{name}.yaml"
        config_sha = _write_bytes(
            training_config, f"resolved: {name}\n".encode("ascii")
        )
        metadata_sha = _digest(f"adapter-{name}")
        checkpoint_sha = _digest(f"checkpoint-{name}")
        metadata[str(adapter_dir.resolve())] = {
            "global_step": step,
            "base_model_id": eval_matrix.BASE_MODEL,
            "base_model_revision": eval_matrix.MODEL_REVISION,
            "tokenizer_id": eval_matrix.BASE_MODEL,
            "tokenizer_revision": eval_matrix.MODEL_REVISION,
            "template_revision": runner.PROMPT_TEMPLATE_REVISION,
            "metadata_sha256": metadata_sha,
            "source_extra_state_sha256": checkpoint_sha,
        }
        artifact_bindings[name] = {
            "artifact_path": str(adapter_dir.resolve()),
            "training_config_id": f"selected-r0-{name}",
            "training_config_path": str(training_config.resolve()),
            "training_config_sha256": config_sha,
            "adapter_metadata_sha256": metadata_sha,
            "checkpoint_extra_state_sha256": checkpoint_sha,
        }
        probe_path = tmp_path / "resume-probes" / f"{name}.json"
        probe_file_sha = _write_canonical(probe_path, {"endpoint": name})
        resume_probes[name] = {
            "path": str(probe_path.resolve()),
            "sha256": probe_file_sha,
            "probe_sha256": _digest(f"resume-probe-{name}"),
        }
    unsigned = {
        "schema_version": 1,
        "delivery_level": level,
        "evaluation_config_sha256": hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest(),
        "capacity_profile": {
            "path": str(capacity.resolve()),
            "sha256": capacity_sha,
        },
        "capacity_approval": None,
        "artifacts": artifact_bindings,
        "resume_probes": resume_probes,
    }
    binding = {
        **unsigned,
        "runtime_binding_sha256": hashlib.sha256(
            canonical_json_bytes(unsigned)
        ).hexdigest(),
    }
    path = tmp_path / f"binding-{level}.json"
    _write_canonical(path, binding)
    return path, binding, metadata


def _record_loader(evaluation: dict):
    bundle_to_dataset = {
        str(Path(spec["bundle_dir"]).resolve()): name
        for name, spec in evaluation["datasets"].items()
    }

    def load(bundle_dir, *, expected_manifest_sha256, variant, sample_count):
        dataset = bundle_to_dataset[str(Path(bundle_dir).resolve())]
        records = tuple(
            SimpleNamespace(qa=SimpleNamespace(qa_id=f"{dataset}-qa-{index:03d}"))
            for index in range(sample_count)
        )
        manifest = {
            "manifest_sha256": expected_manifest_sha256,
            "dataset": dataset,
            "mode": "eval",
            "profile": "formal",
            "contract": {"chunk_size": 5000},
        }
        return records, manifest

    return load


def _contract(tmp_path: Path, level: str = "l1"):
    config_path, evaluation = _source_config(tmp_path, level)
    binding_path, binding, metadata = _runtime_binding(
        tmp_path, config_path, evaluation
    )

    def adapter_validator(path):
        return metadata[str(Path(path).resolve())]

    plan = eval_matrix.build_plan(
        config_path,
        binding_path,
        record_loader=_record_loader(evaluation),
        adapter_validator=adapter_validator,
    )
    return {
        "config_path": config_path,
        "evaluation": evaluation,
        "binding_path": binding_path,
        "binding": binding,
        "metadata": metadata,
        "adapter_validator": adapter_validator,
        "record_loader": _record_loader(evaluation),
        "plan": plan,
    }


def _binding_inputs(tmp_path: Path, level: str = "l1", profile: str = "r0"):
    yaml = pytest.importorskip("yaml")
    _, evaluation = _source_config(tmp_path / "source", level)
    step = 40 if level == "l1" else 80
    eval_id = f"eval{step}_qwen35_2b_5090"
    b_name, c_name = f"b{step}", f"c{step}"
    b_id = f"{b_name}_qwen35_2b_5090_{profile}"
    c_id = f"{c_name}_qwen35_2b_5090_{profile}"
    resolved = tmp_path / "resolved"
    resolved.mkdir(parents=True)
    config_values = {
        eval_id: {"reproduction_evaluation": evaluation},
        b_id: {"kind": "training", "name": b_name},
        c_id: {"kind": "training", "name": c_name},
    }
    records = {}
    for config_id, value in config_values.items():
        path = resolved / f"{config_id}.yaml"
        path.write_text(
            yaml.safe_dump(value, allow_unicode=False, sort_keys=True),
            encoding="ascii",
        )
        records[config_id] = {
            "offload_profile": None if config_id == eval_id else profile,
            "overrides": [],
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_config": config_id.removesuffix("_r0"),
        }
    data_root = tmp_path / "data"
    data_root.mkdir()
    index_path = resolved / "index.json"
    index_value = {
        "configs": records,
        "data_root": str(data_root.resolve()),
        "schema_version": 1,
        "status": "resolved",
    }
    index_path.write_text(
        json.dumps(index_value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()

    capacity = {
        "selected_profile": profile.upper(),
        "selected_configs": {
            b_id: records[b_id]["sha256"],
            c_id: records[c_id]["sha256"],
        },
        "self_sha256": _digest("capacity-profile"),
    }
    capacity_path = tmp_path / "capacity-profile.json"
    _write_canonical(capacity_path, capacity)

    runtime_evidence = {}
    checkpoint_states = {}
    adapter_metadata = {}
    resume_probe_evidence = {}
    resume_probe_paths = {}
    attempts = {}
    for name, config_id in ((b_name, b_id), (c_name, c_id)):
        attempt = tmp_path / f"{name}-attempt"
        (attempt / "runtime-bound").mkdir(parents=True)
        (attempt / ".success").write_text("0\n", encoding="ascii")
        checkpoint = attempt / "checkpoints" / f"global_step_{step}"
        checkpoint.mkdir(parents=True)
        adapter = (
            attempt
            / "artifacts"
            / "adapter"
            / f"global_step_{step}"
            / "adapter"
        )
        adapter.mkdir(parents=True)
        runtime_sha = _digest(f"runtime-{name}")
        checkpoint_sha = _digest(f"checkpoint-{name}")
        runtime_evidence[str((attempt / "runtime-bound").resolve())] = {
            "attempt_id": attempt.name,
            "attempt_root": str(attempt.resolve()),
            "config_id": config_id,
            "runtime_bound_config_sha256": runtime_sha,
            "source_config": {
                "path": records[config_id]["path"],
                "sha256": records[config_id]["sha256"],
            },
            "source_index": {
                "path": str(index_path.resolve()),
                "sha256": index_sha,
            },
        }
        checkpoint_states[str(checkpoint.resolve())] = {
            "base_model_id": eval_matrix.BASE_MODEL,
            "base_model_revision": eval_matrix.MODEL_REVISION,
            "extra_state_sha256": checkpoint_sha,
            "global_step": step,
            "resolved_config_sha256": runtime_sha,
        }
        adapter_metadata[str(adapter.resolve())] = {
            "base_model_id": eval_matrix.BASE_MODEL,
            "base_model_revision": eval_matrix.MODEL_REVISION,
            "global_step": step,
            "metadata_sha256": _digest(f"adapter-{name}"),
            "source_extra_state_sha256": checkpoint_sha,
            "template_revision": runner.PROMPT_TEMPLATE_REVISION,
            "tokenizer_id": eval_matrix.BASE_MODEL,
            "tokenizer_revision": eval_matrix.MODEL_REVISION,
        }
        probe_path = tmp_path / "resume-probes" / f"{name}.json"
        _write_canonical(probe_path, {"endpoint": name})
        probe_sha = _digest(f"resume-probe-{name}")
        resume_probe_paths[name] = probe_path
        resume_probe_evidence[str(probe_path.resolve())] = {
            "endpoint": name,
            "config_id": config_id,
            "attempt_root": str(attempt.resolve()),
            "global_step": step,
            "offload_profile": profile,
            "probe_sha256": probe_sha,
            "checkpoint": {
                "root_extra_state": {
                    "extra_state_sha256": checkpoint_sha,
                }
            },
            "adapter": {
                "metadata_sha256": _digest(f"adapter-{name}"),
            },
        }
        attempts[name] = attempt

    return {
        "adapter_metadata": adapter_metadata,
        "attempts": attempts,
        "capacity": capacity,
        "capacity_path": capacity_path,
        "checkpoint_states": checkpoint_states,
        "eval_id": eval_id,
        "evaluation": evaluation,
        "index_path": index_path,
        "records": records,
        "runtime_evidence": runtime_evidence,
        "resume_probe_evidence": resume_probe_evidence,
        "resume_probe_paths": resume_probe_paths,
    }


def _build_binding_from_inputs(inputs):
    level = inputs["evaluation"]["delivery_level"]
    step = 40 if level == "l1" else 80
    return eval_matrix.build_runtime_binding(
        inputs["index_path"],
        inputs["eval_id"],
        inputs["capacity_path"],
        inputs["attempts"][f"b{step}"],
        inputs["attempts"][f"c{step}"],
        b_resume_probe_path=inputs["resume_probe_paths"][f"b{step}"],
        c_resume_probe_path=inputs["resume_probe_paths"][f"c{step}"],
        capacity_validator=lambda value: inputs["capacity"],
        training_binding_verifier=lambda path: inputs["runtime_evidence"][
            str(path.resolve())
        ],
        checkpoint_validator=lambda path: inputs["checkpoint_states"][
            str(path.resolve())
        ],
        adapter_validator=lambda path: inputs["adapter_metadata"][
            str(path.resolve())
        ],
        resume_probe_verifier=lambda path: inputs["resume_probe_evidence"][
            str(path.resolve())
        ],
        approval_marker_path=inputs.get("approval_marker_path"),
        approval_consumption_path=inputs.get("approval_consumption_path"),
    )


@pytest.mark.parametrize("level", ["l1", "l2"])
def test_bind_derives_self_hashed_identity_from_capacity_and_attempts(
    tmp_path, level
):
    inputs = _binding_inputs(tmp_path, level)
    binding = _build_binding_from_inputs(inputs)
    step = 40 if level == "l1" else 80

    assert binding["delivery_level"] == level
    assert set(binding["artifacts"]) == {f"b{step}", f"c{step}"}
    assert binding["evaluation_config_sha256"] == inputs["records"][
        inputs["eval_id"]
    ]["sha256"]
    assert binding["capacity_profile"] == {
        "path": str(inputs["capacity_path"].resolve()),
        "sha256": hashlib.sha256(inputs["capacity_path"].read_bytes()).hexdigest(),
    }
    assert set(binding["resume_probes"]) == {f"b{step}", f"c{step}"}
    unsigned = {
        key: value for key, value in binding.items() if key != "runtime_binding_sha256"
    }
    assert binding["runtime_binding_sha256"] == hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    for name, artifact in binding["artifacts"].items():
        expected_id = f"{name}_qwen35_2b_5090_r0"
        assert artifact["training_config_id"] == expected_id
        assert artifact["training_config_sha256"] == inputs["records"][expected_id][
            "sha256"
        ]

    output = tmp_path / "runtime-binding.json"
    eval_matrix.publish_runtime_binding(binding, output)
    loaded, _ = eval_matrix._load_binding(
        output, binding["evaluation_config_sha256"], level
    )
    assert loaded == binding


def test_bind_cli_publishes_without_shell_constructed_json(
    tmp_path, monkeypatch, capsys
):
    inputs = _binding_inputs(tmp_path)
    monkeypatch.setattr(
        eval_matrix,
        "_verify_capacity_profile",
        lambda value, **kwargs: inputs["capacity"],
    )
    monkeypatch.setattr(
        eval_matrix,
        "_training_binding_verifier",
        lambda: lambda path: inputs["runtime_evidence"][str(path.resolve())],
    )
    monkeypatch.setattr(
        eval_matrix,
        "_checkpoint_validator",
        lambda: lambda path: inputs["checkpoint_states"][str(path.resolve())],
    )
    monkeypatch.setattr(
        eval_matrix,
        "_adapter_validator",
        lambda: lambda path: inputs["adapter_metadata"][str(path.resolve())],
    )
    monkeypatch.setattr(
        eval_matrix,
        "_resume_probe_verifier",
        lambda: lambda path: inputs["resume_probe_evidence"][str(path.resolve())],
    )
    output = tmp_path / "cli-binding.json"

    assert eval_matrix.main(
        [
            "bind",
            "--index",
            str(inputs["index_path"]),
            "--eval-config-id",
            inputs["eval_id"],
            "--capacity-profile",
            str(inputs["capacity_path"]),
            "--b-attempt",
            str(inputs["attempts"]["b40"]),
            "--c-attempt",
            str(inputs["attempts"]["c40"]),
            "--b-resume-probe",
            str(inputs["resume_probe_paths"]["b40"]),
            "--c-resume-probe",
            str(inputs["resume_probe_paths"]["c40"]),
            "--output",
            str(output),
        ]
    ) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads(output.read_text(encoding="ascii"))
    assert printed["runtime_binding_sha256"] == _digest(
        canonical_json_bytes(
            {
                key: value
                for key, value in printed.items()
                if key != "runtime_binding_sha256"
            }
        ).decode("ascii")
    )


def test_r1_binding_persists_and_revalidates_one_time_consumption(tmp_path):
    inputs = _binding_inputs(tmp_path, profile="r1")
    marker = tmp_path / "r1-approval.json"
    consumption = tmp_path / "consumption.json"
    _write_canonical(marker, {"status": "approved-once"})
    _write_canonical(consumption, {"status": "consumed"})
    inputs["approval_marker_path"] = marker
    inputs["approval_consumption_path"] = consumption

    binding = _build_binding_from_inputs(inputs)
    assert binding["capacity_approval"] == {
        "marker": {
            "path": str(marker.resolve()),
            "sha256": hashlib.sha256(marker.read_bytes()).hexdigest(),
        },
        "consumption": {
            "path": str(consumption.resolve()),
            "sha256": hashlib.sha256(consumption.read_bytes()).hexdigest(),
        },
    }

    consumption.write_bytes(consumption.read_bytes() + b"\n")
    with pytest.raises(
        eval_matrix.EvalMatrixError,
        match="capacity_approval.consumption hash changed",
    ):
        eval_matrix.publish_runtime_binding(binding, tmp_path / "binding.json")


def test_bind_requires_endpoint_probe_identity_and_plan_revalidates_bytes(tmp_path):
    inputs = _binding_inputs(tmp_path)
    b_probe = inputs["resume_probe_paths"]["b40"]
    evidence = inputs["resume_probe_evidence"][str(b_probe.resolve())]
    evidence["endpoint"] = "c40"
    with pytest.raises(eval_matrix.EvalMatrixError, match="probe differs at endpoint"):
        _build_binding_from_inputs(inputs)

    inputs = _binding_inputs(tmp_path / "tamper")
    binding = _build_binding_from_inputs(inputs)
    binding_path = tmp_path / "tamper" / "binding.json"
    eval_matrix.publish_runtime_binding(binding, binding_path)
    inputs["resume_probe_paths"]["b40"].write_bytes(b"tampered\n")
    with pytest.raises(eval_matrix.EvalMatrixError, match="resume_probes.b40 hash changed"):
        eval_matrix.build_plan(
            inputs["index_path"].parent / f"{inputs['eval_id']}.yaml",
            binding_path,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("capacity_sha", "capacity profile config hash differs"),
        ("attempt_config", "different config ID"),
        ("checkpoint_step", "checkpoint differs at global_step"),
        ("checkpoint_config", "checkpoint differs at resolved_config_sha256"),
        ("adapter_checkpoint", "adapter differs at source_extra_state_sha256"),
        ("terminal", "success marker"),
    ],
)
def test_bind_rejects_capacity_attempt_checkpoint_and_adapter_drift(
    tmp_path, mutation, match
):
    inputs = _binding_inputs(tmp_path)
    if mutation == "capacity_sha":
        inputs["capacity"]["selected_configs"]["b40_qwen35_2b_5090_r0"] = "0" * 64
    elif mutation == "attempt_config":
        evidence = inputs["runtime_evidence"][
            str((inputs["attempts"]["b40"] / "runtime-bound").resolve())
        ]
        evidence["config_id"] = "c40_qwen35_2b_5090_r0"
    elif mutation == "checkpoint_step":
        state = next(iter(inputs["checkpoint_states"].values()))
        state["global_step"] = 39
    elif mutation == "checkpoint_config":
        state = next(iter(inputs["checkpoint_states"].values()))
        state["resolved_config_sha256"] = "0" * 64
    elif mutation == "adapter_checkpoint":
        metadata = next(iter(inputs["adapter_metadata"].values()))
        metadata["source_extra_state_sha256"] = "0" * 64
    else:
        (inputs["attempts"]["b40"] / ".success").unlink()

    with pytest.raises(eval_matrix.EvalMatrixError, match=match):
        _build_binding_from_inputs(inputs)


def test_bind_rejects_delivery_level_config_id_drift_and_overwrite(tmp_path):
    inputs = _binding_inputs(tmp_path)
    eval_path = inputs["index_path"].parent / f"{inputs['eval_id']}.yaml"
    yaml = pytest.importorskip("yaml")
    _, l2_evaluation = _source_config(tmp_path / "level-drift", "l2")
    root = {"reproduction_evaluation": l2_evaluation}
    eval_path.write_text(
        yaml.safe_dump(root, allow_unicode=False, sort_keys=True), encoding="ascii"
    )
    index = json.loads(inputs["index_path"].read_text(encoding="ascii"))
    index["configs"][inputs["eval_id"]]["sha256"] = hashlib.sha256(
        eval_path.read_bytes()
    ).hexdigest()
    inputs["index_path"].write_text(
        json.dumps(index, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    with pytest.raises(eval_matrix.EvalMatrixError, match="delivery level"):
        _build_binding_from_inputs(inputs)

    inputs = _binding_inputs(tmp_path / "overwrite")
    binding = _build_binding_from_inputs(inputs)
    output = tmp_path / "binding.json"
    eval_matrix.publish_runtime_binding(binding, output)
    with pytest.raises(FileExistsError, match="overwrite"):
        eval_matrix.publish_runtime_binding(binding, output)


def test_plan_uses_attempt_scoped_output_root_and_refuses_retry_reuse(tmp_path):
    contract = _contract(tmp_path)
    output_root = tmp_path / "eval-attempt" / "artifacts" / "results"
    plan = eval_matrix.build_plan(
        contract["config_path"],
        contract["binding_path"],
        output_root=output_root,
        record_loader=contract["record_loader"],
        adapter_validator=contract["adapter_validator"],
    )

    assert plan["output_root"] == str(output_root.resolve())
    assert all(
        Path(cell["output_dir"]).is_relative_to(output_root.resolve())
        for cell in plan["cells"]
    )
    plan_path = tmp_path / "eval-plan.json"
    eval_matrix.publish_plan(plan, plan_path)
    output_root.mkdir(parents=True)
    assert eval_matrix.load_plan(plan_path) == plan

    with pytest.raises(FileExistsError, match="reuse evaluation output root"):
        eval_matrix.build_plan(
            contract["config_path"],
            contract["binding_path"],
            output_root=output_root,
            record_loader=contract["record_loader"],
            adapter_validator=contract["adapter_validator"],
        )

    tampered = copy.deepcopy(plan)
    tampered["output_root"] = str(
        output_root.parent / "noncanonical" / ".." / output_root.name
    )
    unsigned = {key: value for key, value in tampered.items() if key != "plan_sha256"}
    tampered["plan_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    with pytest.raises(eval_matrix.EvalMatrixError, match="not canonical"):
        eval_matrix._validate_plan(tampered)


@pytest.mark.parametrize(
    ("level", "sample_count", "b_name", "c_name"),
    [("l1", 32, "b40", "c40"), ("l2", 64, "b80", "c80")],
)
def test_plan_has_exact_matrix_and_deduplicates_c_learned(
    tmp_path, level, sample_count, b_name, c_name
):
    contract = _contract(tmp_path, level)
    plan = contract["plan"]

    assert plan["delivery_level"] == level
    assert plan["sample_count_per_stratum"] == sample_count
    assert len(plan["cells"]) == len({cell["cell_id"] for cell in plan["cells"]}) == 20
    assert set(plan["artifacts"]) == {"base", b_name, c_name}
    assert sum(
        cell["artifact"] == c_name and cell["callback_mode"] == "learned"
        for cell in plan["cells"]
    ) == 4
    for dataset in eval_matrix.DATASETS:
        ids_200 = {
            tuple(cell["ordered_qa_ids"])
            for cell in plan["cells"]
            if cell["dataset"] == dataset and cell["document_count"] == 200
        }
        ids_800 = {
            tuple(cell["ordered_qa_ids"])
            for cell in plan["cells"]
            if cell["dataset"] == dataset and cell["document_count"] == 800
        }
        assert len(ids_200) == len(ids_800) == 1
        assert ids_200 == ids_800

    plan_path = tmp_path / "plan.json"
    eval_matrix.publish_plan(plan, plan_path)
    assert eval_matrix.load_plan(plan_path) == plan


def test_plan_consumes_resolved_yaml_evaluation_section(tmp_path):
    yaml = pytest.importorskip("yaml")
    _, evaluation = _source_config(tmp_path, "l1")
    config_path = tmp_path / "eval-l1.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {"reproduction_evaluation": evaluation},
            allow_unicode=False,
            sort_keys=False,
        ),
        encoding="ascii",
    )
    binding_path, _, metadata = _runtime_binding(tmp_path, config_path, evaluation)
    plan = eval_matrix.build_plan(
        config_path,
        binding_path,
        record_loader=_record_loader(evaluation),
        adapter_validator=lambda path: metadata[str(Path(path).resolve())],
    )
    assert plan["source_config"]["sha256"] == hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sample_count_per_stratum", 64, "sample_count_per_stratum"),
        ("required_cell_count", 19, "required_cell_count"),
        ("primary_callback_mode", "none", "primary_callback_mode"),
    ],
)
def test_source_contract_fails_closed(tmp_path, field, value, message):
    config_path, evaluation = _source_config(tmp_path, "l1")
    evaluation[field] = value
    config_path.write_text(
        json.dumps({"reproduction_evaluation": evaluation}, sort_keys=True),
        encoding="ascii",
    )
    binding_path, _, metadata = _runtime_binding(tmp_path, config_path, evaluation)

    with pytest.raises(eval_matrix.EvalMatrixError, match=message):
        eval_matrix.build_plan(
            config_path,
            binding_path,
            record_loader=_record_loader(evaluation),
            adapter_validator=lambda path: metadata[str(Path(path).resolve())],
        )


def test_eval40_rejects_extra_eval80_artifact_and_non_greedy_decode(tmp_path):
    config_path, evaluation = _source_config(tmp_path, "l1")
    evaluation["artifacts"]["b80"] = copy.deepcopy(evaluation["artifacts"]["b40"])
    config_path.write_text(
        json.dumps({"reproduction_evaluation": evaluation}, sort_keys=True),
        encoding="ascii",
    )
    binding_path, _, metadata = _runtime_binding(tmp_path, config_path, evaluation)
    with pytest.raises(eval_matrix.EvalMatrixError, match="artifacts keys differ"):
        eval_matrix.build_plan(
            config_path,
            binding_path,
            record_loader=_record_loader(evaluation),
            adapter_validator=lambda path: metadata[str(Path(path).resolve())],
        )

    config_path, evaluation = _source_config(tmp_path / "greedy", "l1")
    evaluation["decode"]["do_sample"] = True
    config_path.write_text(
        json.dumps({"reproduction_evaluation": evaluation}, sort_keys=True),
        encoding="ascii",
    )
    binding_path, _, metadata = _runtime_binding(
        tmp_path / "greedy", config_path, evaluation
    )
    with pytest.raises(eval_matrix.EvalMatrixError, match="fixed greedy identity"):
        eval_matrix.build_plan(
            config_path,
            binding_path,
            record_loader=_record_loader(evaluation),
            adapter_validator=lambda path: metadata[str(Path(path).resolve())],
        )


def test_binding_checks_capacity_adapter_checkpoint_and_training_config(tmp_path):
    contract = _contract(tmp_path)
    binding = copy.deepcopy(contract["binding"])
    binding["artifacts"]["b40"]["checkpoint_extra_state_sha256"] = "f" * 64
    unsigned = {key: value for key, value in binding.items() if key != "runtime_binding_sha256"}
    binding["runtime_binding_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    _write_bytes(
        contract["binding_path"], canonical_json_bytes(binding) + b"\n"
    )
    with pytest.raises(eval_matrix.EvalMatrixError, match="source_extra_state_sha256"):
        eval_matrix.build_plan(
            contract["config_path"],
            contract["binding_path"],
            record_loader=contract["record_loader"],
            adapter_validator=contract["adapter_validator"],
        )

    contract = _contract(tmp_path / "capacity")
    capacity_path = Path(contract["binding"]["capacity_profile"]["path"])
    capacity_path.write_bytes(b"tampered\n")
    with pytest.raises(eval_matrix.EvalMatrixError, match="capacity_profile hash changed"):
        eval_matrix.build_plan(
            contract["config_path"],
            contract["binding_path"],
            record_loader=contract["record_loader"],
            adapter_validator=contract["adapter_validator"],
        )

    contract = _contract(tmp_path / "training")
    training_path = Path(
        contract["binding"]["artifacts"]["c40"]["training_config_path"]
    )
    training_path.write_bytes(b"tampered: true\n")
    with pytest.raises(eval_matrix.EvalMatrixError, match="training_config hash changed"):
        eval_matrix.build_plan(
            contract["config_path"],
            contract["binding_path"],
            record_loader=contract["record_loader"],
            adapter_validator=contract["adapter_validator"],
        )


def test_plan_rejects_cross_variant_qa_order_drift(tmp_path):
    config_path, evaluation = _source_config(tmp_path, "l1")
    binding_path, _, metadata = _runtime_binding(tmp_path, config_path, evaluation)
    ordinary_loader = _record_loader(evaluation)

    def drifted_loader(bundle_dir, **kwargs):
        records, manifest = ordinary_loader(bundle_dir, **kwargs)
        if kwargs["variant"] == 800:
            records = tuple(reversed(records))
        return records, manifest

    with pytest.raises(eval_matrix.EvalMatrixError, match="ordered QA IDs"):
        eval_matrix.build_plan(
            config_path,
            binding_path,
            record_loader=drifted_loader,
            adapter_validator=lambda path: metadata[str(Path(path).resolve())],
        )


def test_dry_run_returns_twenty_shell_free_commands_without_creating_cells(tmp_path):
    contract = _contract(tmp_path)
    plan_path = tmp_path / "plan.json"
    eval_matrix.publish_plan(contract["plan"], plan_path)

    result = eval_matrix.run_matrix(plan_path, dry_run=True)

    assert result["status"] == "dry-run"
    assert len(result["commands"]) == 20
    assert all(isinstance(command, list) for command in result["commands"])
    assert all("run-cell" in command for command in result["commands"])
    assert not (Path(contract["plan"]["output_root"]) / "cells").exists()


def _model_metadata(plan: dict, cell: dict) -> dict:
    runtime = plan["runtime"]
    artifact = plan["artifacts"][cell["artifact"]]
    adapter_metadata = None
    if artifact["artifact_kind"] == "adapter":
        adapter_metadata = {
            "global_step": artifact["training_step"],
            "metadata_sha256": artifact["adapter_metadata_sha256"],
            "source_extra_state_sha256": artifact[
                "checkpoint_extra_state_sha256"
            ],
        }
    return {
        "adapter_metadata": adapter_metadata,
        "merged_metadata": None,
        "artifact_kind": artifact["artifact_kind"],
        "backend": "transformers",
        "base_model_id": runtime["base_model_id"],
        "device": runtime["device"],
        "dtype": runtime["dtype"],
        "greedy": True,
        "revision": runtime["revision"],
        "seed": runtime["seed"],
        "tokenizer_id": runtime["tokenizer_id"],
        "tokenizer_revision": runtime["tokenizer_revision"],
    }


def _cell_score(cell: dict) -> float:
    if cell["artifact"] == "base":
        return 0.125
    if cell["artifact"].startswith("b"):
        return 0.25
    return 0.5


def _publish_fake_cell(plan: dict, cell: dict, *, corrupt_config: bool = False):
    score = _cell_score(cell)
    correct_count = round(score * cell["sample_count"])
    results = []
    for qa_index, qa_id in enumerate(cell["ordered_qa_ids"]):
        gold_answer = f"gold answer {qa_id}"
        answer = gold_answer if qa_index < correct_count else "definitely wrong"
        raw_output = f"\\boxed{{{answer}}}"
        evaluated = evaluate_output(raw_output, [gold_answer])
        results.append({
            "callback_mode": cell["callback_mode"],
            "context_sha256": _digest(f"context-{qa_id}"),
            "context_token_count": 1,
            "context_token_ids_sha256": _digest(f"tokens-{qa_id}"),
            "document_count": cell["document_count"],
            "elapsed_seconds": 0.0,
            "gold_answers": [gold_answer],
            "manifest_record_sha256": _digest(f"record-{qa_id}"),
            "metrics": {
                "answer": evaluated.to_dict(),
                "callback": {
                    "average_lookback_distance": None,
                    "duplicate_retrieved_state_count": 0,
                    "duplicate_retrieved_state_rate": None,
                    "effective_query_count": 0,
                    "effective_query_rate": 0.0,
                    "eligible_steps": 1,
                    "lexical_gold_hit_count": 0,
                    "lexical_gold_hit_rate": None,
                    "lexical_supporting_fact_hit_count": 0,
                    "lexical_supporting_fact_hit_rate": None,
                    "model_query_count": 0,
                    "model_query_rate": 0.0,
                    "model_query_status_counts": {
                        "valid": 0,
                        "empty": 0,
                        "malformed": 0,
                        "duplicate": 0,
                        "absent": 1,
                    },
                    "model_query_status_rates": {
                        "valid": 0.0,
                        "empty": 0.0,
                        "malformed": 0.0,
                        "duplicate": 0.0,
                        "absent": 1.0,
                    },
                    "retrieval_count": 0,
                    "retrieval_empty_count": 0,
                    "retrieval_empty_rate": None,
                    "retrieval_rate": 0.0,
                    "retrieval_success_rate": None,
                    "supporting_doc_hit_count": 0,
                    "supporting_doc_hit_rate": None,
                },
                "format": {
                    "final_valid": True,
                    "intermediate_valid_count": 1,
                    "intermediate_valid_rate": 1.0,
                    "recall_protocol_valid_count": 1,
                    "recall_protocol_valid_rate": 1.0,
                    "thinking_single_non_empty_count": 1,
                    "thinking_single_non_empty_rate": 1.0,
                    "update_single_non_empty_count": 1,
                    "update_single_non_empty_rate": 1.0,
                },
                "scores": evaluated.scores.to_dict(),
                "truncation": {
                    "final_reached_token_limit": False,
                    "memory_reached_token_limit_count": 0,
                    "memory_reached_token_limit_rate": 0.0,
                },
            },
            "parsed_answer": evaluated.extraction.answer,
            "processed_chunk_count": 1,
            "processed_doc_count": cell["document_count"],
            "prompt_template_sha256": runner.PROMPT_TEMPLATE_SHA256,
            "prompt_template_revision": runner.PROMPT_TEMPLATE_REVISION,
            "qa_id": qa_id,
            "qa_index": qa_index,
            "raw_final_output": raw_output,
            "status": "success",
            "trajectory": [{"kind": "final", "raw_output": raw_output}],
        })
    summary = runner._aggregate_summary(results, callback_mode=cell["callback_mode"])
    run_config = eval_matrix.expected_run_config(plan, cell)
    run_config["model_metadata"] = _model_metadata(plan, cell)
    if corrupt_config:
        run_config["input"]["capacity_profile_sha256"] = "0" * 64
    runner._publish_results(
        Path(cell["output_dir"]),
        results=results,
        run_config=run_config,
        summary=summary,
    )


def _load_fake_results(cell: dict) -> list[dict]:
    payload = (Path(cell["output_dir"]) / runner.RESULTS_FILENAME).read_bytes()
    return [json.loads(line) for line in payload.splitlines()]


def _reseal_fake_cell(cell: dict, results: list[dict]) -> None:
    output_dir = Path(cell["output_dir"])
    results_payload = runner._canonical_jsonl_bytes(results)
    results_sha = _write_bytes(output_dir / runner.RESULTS_FILENAME, results_payload)
    summary = runner._aggregate_summary(results, callback_mode=cell["callback_mode"])
    summary_sha = _write_canonical(output_dir / runner.SUMMARY_FILENAME, summary)
    run_config_payload = (output_dir / runner.RUN_CONFIG_FILENAME).read_bytes()
    marker_payload = {
        "failure_count": summary["failure_count"],
        "kind": runner.RUN_KIND,
        "results_sha256": results_sha,
        "run_config_sha256": hashlib.sha256(run_config_payload).hexdigest(),
        "schema_version": runner.RUN_SCHEMA_VERSION,
        "status": summary["status"],
        "success_count": summary["success_count"],
        "summary_sha256": summary_sha,
    }
    marker = {
        **marker_payload,
        "marker_sha256": hashlib.sha256(
            canonical_json_bytes(marker_payload)
        ).hexdigest(),
    }
    _write_canonical(output_dir / runner.COMPLETION_FILENAME, marker)


def _reseal_verified_package(
    package_dir: Path, *, aggregate: dict, package: dict
) -> None:
    aggregate_unsigned = {
        key: value for key, value in aggregate.items() if key != "aggregate_sha256"
    }
    aggregate["aggregate_sha256"] = hashlib.sha256(
        canonical_json_bytes(aggregate_unsigned)
    ).hexdigest()
    package["aggregate_sha256"] = aggregate["aggregate_sha256"]
    package_unsigned = {
        key: value for key, value in package.items() if key != "package_sha256"
    }
    package["package_sha256"] = hashlib.sha256(
        canonical_json_bytes(package_unsigned)
    ).hexdigest()
    aggregate_file_sha = _write_canonical(
        package_dir / eval_matrix.AGGREGATE_FILENAME, aggregate
    )
    package_file_sha = _write_canonical(
        package_dir / eval_matrix.PACKAGE_FILENAME, package
    )
    marker_payload = {
        "kind": eval_matrix.PACKAGE_KIND,
        "schema_version": eval_matrix.PACKAGE_SCHEMA_VERSION,
        "status": "verified",
        "plan_sha256": package["plan_sha256"],
        "cell_count": package["cell_count"],
        "aggregate_sha256": aggregate["aggregate_sha256"],
        "package_sha256": package["package_sha256"],
        "aggregate_file_sha256": aggregate_file_sha,
        "package_file_sha256": package_file_sha,
    }
    marker = {
        **marker_payload,
        "marker_sha256": hashlib.sha256(
            canonical_json_bytes(marker_payload)
        ).hexdigest(),
    }
    _write_canonical(package_dir / eval_matrix.VERIFIED_FILENAME, marker)


def test_verified_package_is_atomic_self_hashed_and_stratified(tmp_path):
    contract = _contract(tmp_path)
    plan = contract["plan"]
    plan_path = tmp_path / "plan.json"
    eval_matrix.publish_plan(plan, plan_path)
    for cell in plan["cells"]:
        _publish_fake_cell(plan, cell)

    package_dir = tmp_path / "verified-package"
    result = eval_matrix.publish_verified_package(plan_path, package_dir)

    assert result["status"] == "verified"
    package_payload = json.loads(
        (package_dir / eval_matrix.PACKAGE_FILENAME).read_text(encoding="ascii")
    )
    assert package_payload["resume_probes"] == plan["resume_probes"]
    assert {path.name for path in package_dir.iterdir()} == {
        eval_matrix.AGGREGATE_FILENAME,
        eval_matrix.PACKAGE_FILENAME,
        eval_matrix.VERIFIED_FILENAME,
    }
    assert eval_matrix.verify_package(plan_path, package_dir) == result
    aggregate = runner._read_canonical_json(
        package_dir / eval_matrix.AGGREGATE_FILENAME
    )
    package = runner._read_canonical_json(package_dir / eval_matrix.PACKAGE_FILENAME)
    assert aggregate["cell_count"] == package["cell_count"] == 20
    assert aggregate["primary"]["contrast"] == "c40-b40"
    assert aggregate["primary"]["paired_bootstrap"]["stratum_count"] == 4
    assert aggregate["primary"]["paired_bootstrap"]["sample_counts"] == [32] * 4
    assert aggregate["primary"]["paired_bootstrap"]["resamples"] == 10_000
    assert aggregate["primary"]["paired_bootstrap"]["observed_delta"] == pytest.approx(
        0.25
    )
    assert package["aggregate_sha256"] == aggregate["aggregate_sha256"]
    assert not tuple(tmp_path.glob(".verified-package.staging-*"))
    with pytest.raises(FileExistsError):
        eval_matrix.publish_verified_package(plan_path, package_dir)


def test_package_refuses_missing_or_identity_drifted_cell_without_publication(tmp_path):
    contract = _contract(tmp_path)
    plan = contract["plan"]
    plan_path = tmp_path / "plan.json"
    eval_matrix.publish_plan(plan, plan_path)
    for cell in plan["cells"][:-1]:
        _publish_fake_cell(plan, cell)
    package_dir = tmp_path / "missing-package"
    with pytest.raises(eval_matrix.EvalMatrixError):
        eval_matrix.publish_verified_package(plan_path, package_dir)
    assert not package_dir.exists()

    contract = _contract(tmp_path / "drift")
    plan = contract["plan"]
    plan_path = tmp_path / "drift" / "plan.json"
    eval_matrix.publish_plan(plan, plan_path)
    for index, cell in enumerate(plan["cells"]):
        _publish_fake_cell(plan, cell, corrupt_config=index == 0)
    package_dir = tmp_path / "drift" / "drift-package"
    with pytest.raises(eval_matrix.EvalMatrixError, match="run config differs at input"):
        eval_matrix.publish_verified_package(plan_path, package_dir)
    assert not package_dir.exists()


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("score", "scores differ from raw prediction"),
        ("prediction", "answer metrics differ from raw prediction"),
    ],
)
def test_cell_reuse_recomputes_scores_after_self_consistent_reseal(
    tmp_path, tamper, message
):
    contract = _contract(tmp_path)
    plan = contract["plan"]
    cell = plan["cells"][0]
    _publish_fake_cell(plan, cell)
    results = _load_fake_results(cell)
    if tamper == "score":
        old_score = results[0]["metrics"]["scores"]["exact_match"]
        results[0]["metrics"]["scores"]["exact_match"] = 1.0 - old_score
    else:
        results[0]["raw_final_output"] = "\\boxed{tampered prediction}"
        results[0]["trajectory"][-1]["raw_output"] = results[0][
            "raw_final_output"
        ]
    _reseal_fake_cell(cell, results)

    with pytest.raises(eval_matrix.EvalMatrixError, match=message):
        eval_matrix.verify_cell_output(plan, cell)


@pytest.mark.parametrize("tamper", ["duplicate", "missing"])
def test_cell_reuse_rejects_duplicate_or_missing_sample_after_reseal(
    tmp_path, tamper
):
    contract = _contract(tmp_path)
    plan = contract["plan"]
    cell = plan["cells"][0]
    _publish_fake_cell(plan, cell)
    results = _load_fake_results(cell)
    if tamper == "duplicate":
        results[-1] = copy.deepcopy(results[0])
    else:
        results.pop()
    _reseal_fake_cell(cell, results)

    with pytest.raises(eval_matrix.EvalMatrixError, match="QA order differs"):
        eval_matrix.verify_cell_output(plan, cell)


@pytest.mark.parametrize("field", ["output_dir", "results_sha256"])
def test_package_rejects_rehashed_cell_path_or_results_identity_drift(
    tmp_path, field
):
    contract = _contract(tmp_path)
    plan = contract["plan"]
    plan_path = tmp_path / "plan.json"
    eval_matrix.publish_plan(plan, plan_path)
    for cell in plan["cells"]:
        _publish_fake_cell(plan, cell)
    package_dir = tmp_path / "verified-package"
    eval_matrix.publish_verified_package(plan_path, package_dir)
    aggregate = json.loads(
        (package_dir / eval_matrix.AGGREGATE_FILENAME).read_text(encoding="ascii")
    )
    package = json.loads(
        (package_dir / eval_matrix.PACKAGE_FILENAME).read_text(encoding="ascii")
    )
    package["cells"][0][field] = (
        str((tmp_path / "drifted-cell").resolve())
        if field == "output_dir"
        else "f" * 64
    )
    _reseal_verified_package(package_dir, aggregate=aggregate, package=package)

    with pytest.raises(
        eval_matrix.EvalMatrixError,
        match="package differs from independently verified cells",
    ):
        eval_matrix.verify_package(plan_path, package_dir)


def test_package_rejects_rehashed_aggregate_score_tamper(tmp_path):
    contract = _contract(tmp_path)
    plan = contract["plan"]
    plan_path = tmp_path / "plan.json"
    eval_matrix.publish_plan(plan, plan_path)
    for cell in plan["cells"]:
        _publish_fake_cell(plan, cell)
    package_dir = tmp_path / "verified-package"
    eval_matrix.publish_verified_package(plan_path, package_dir)
    aggregate = json.loads(
        (package_dir / eval_matrix.AGGREGATE_FILENAME).read_text(encoding="ascii")
    )
    package = json.loads(
        (package_dir / eval_matrix.PACKAGE_FILENAME).read_text(encoding="ascii")
    )
    aggregate["methods"]["base"]["macro"]["exact_match"] = 0.999
    _reseal_verified_package(package_dir, aggregate=aggregate, package=package)

    with pytest.raises(
        eval_matrix.EvalMatrixError,
        match="aggregate differs from independently recomputed cells",
    ):
        eval_matrix.verify_package(plan_path, package_dir)


def test_stratified_bootstrap_equal_weights_strata_and_validates_pairs():
    result = stratified_paired_bootstrap_delta(
        [([0.0, 0.0], [1.0, 1.0]), ([0.0] * 10, [0.0] * 10)],
        resamples=50,
        seed=7,
    )
    assert result.observed_delta == pytest.approx(0.5)
    assert result.sample_counts == (2, 10)
    with pytest.raises(EvaluationContractError, match="same length"):
        stratified_paired_bootstrap_delta([([0.0], [0.0, 1.0])])
