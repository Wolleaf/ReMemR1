import os
from types import SimpleNamespace

import pytest

from scripts.cloud import g1_eval


def test_runtime_rejects_a_fixture_record_without_exactly_two_chunks(monkeypatch):
    records = (SimpleNamespace(chunks=("only-one",)),) * g1_eval.SAMPLE_COUNT
    manifest = {
        "contract": g1_eval.EXPECTED_CONTRACT,
        "dataset": "hotpotqa",
        "mode": "eval",
        "profile": "fixture",
    }
    monkeypatch.setattr(
        g1_eval.runner,
        "load_eval_records",
        lambda *args, **kwargs: (records, manifest),
    )

    with pytest.raises(g1_eval.G1EvalError, match="exactly two chunks"):
        g1_eval._load_inputs(SimpleNamespace(), "a" * 64)


def test_model_loading_evidence_binds_the_actual_transformers_adapter_load():
    adapter = {"global_step": 2, "metadata_sha256": "a" * 64}
    metadata = {
        "adapter_metadata": adapter,
        "artifact_kind": "adapter",
        "backend": "transformers",
        "base_model_id": g1_eval.BASE_MODEL,
        "device": "cuda:0",
        "dtype": "bfloat16",
        "greedy": True,
        "merged_metadata": None,
        "revision": g1_eval.MODEL_REVISION,
        "seed": 42,
        "tokenizer_id": g1_eval.BASE_MODEL,
        "tokenizer_revision": g1_eval.MODEL_REVISION,
        "qwen35_mapping": {
            "attention_implementation": "sdpa",
            "dtype": "bfloat16",
            "loader": "transformers.AutoModelForCausalLM",
            "mapping_strategy": "transformers_native_qwen35_causal_lm",
            "model_name_or_path": g1_eval.BASE_MODEL,
            "revision": g1_eval.MODEL_REVISION,
            "schema_version": 2,
            "strict_loading": True,
            "target_architecture": "Qwen3_5ForCausalLM",
        },
    }

    g1_eval._validate_model_loading_evidence(metadata, adapter)
    metadata["backend"] = "scripted"
    with pytest.raises(g1_eval.G1EvalError, match="backend"):
        g1_eval._validate_model_loading_evidence(metadata, adapter)


def _patch_contract(monkeypatch):
    monkeypatch.setattr(
        g1_eval,
        "_load_inputs",
        lambda *args: ((SimpleNamespace(qa=SimpleNamespace(qa_id="qa-1")),), {}),
    )
    monkeypatch.setattr(g1_eval, "_run_contract", lambda *args: (object(), object(), {}))

    def validate(path, *args):
        status = (path / "status").read_text(encoding="ascii")
        if status != "success":
            raise g1_eval.G1EvalError("failed fixture output")
        return {"failure_count": 0, "sample_count": 2, "status": "completed"}

    monkeypatch.setattr(g1_eval, "_validate_existing", validate)


def test_failed_eval_is_preserved_without_occupying_canonical_output(tmp_path, monkeypatch):
    _patch_contract(monkeypatch)
    bundle = tmp_path / "bundle"
    adapter = tmp_path / "adapter"
    output = tmp_path / "eval"
    bundle.mkdir()
    adapter.mkdir()
    metadata = SimpleNamespace(to_dict=lambda: {"global_step": 2})

    def failed_task(*args, output_dir, **kwargs):
        output_dir.mkdir()
        (output_dir / "status").write_text("failed", encoding="ascii")
        return {}

    with pytest.raises(g1_eval.G1EvalError, match="failed fixture"):
        g1_eval.run_or_verify(
            bundle,
            "a" * 64,
            adapter,
            output,
            verify_existing=False,
            adapter_validator=lambda path: metadata,
            evaluation_task=failed_task,
        )

    assert not output.exists()
    assert len(list(tmp_path.glob(".eval.failed-*"))) == 1

    def successful_task(*args, output_dir, **kwargs):
        output_dir.mkdir()
        (output_dir / "status").write_text("success", encoding="ascii")
        return {}

    result = g1_eval.run_or_verify(
        bundle,
        "a" * 64,
        adapter,
        output,
        verify_existing=False,
        adapter_validator=lambda path: metadata,
        evaluation_task=successful_task,
    )
    assert output.is_dir()
    assert result["status"] == "completed"


def test_successful_interrupted_attempt_is_adopted_without_rerun(tmp_path, monkeypatch):
    _patch_contract(monkeypatch)
    bundle = tmp_path / "bundle"
    adapter = tmp_path / "adapter"
    output = tmp_path / "eval"
    attempt = tmp_path / ".eval.attempt-old"
    bundle.mkdir()
    adapter.mkdir()
    attempt.mkdir()
    (attempt / "status").write_text("success", encoding="ascii")
    metadata = SimpleNamespace(to_dict=lambda: {"global_step": 2})

    result = g1_eval.run_or_verify(
        bundle,
        "a" * 64,
        adapter,
        output,
        verify_existing=False,
        adapter_validator=lambda path: metadata,
        evaluation_task=lambda *args, **kwargs: pytest.fail("eval should not rerun"),
    )

    assert result["status"] == "completed"
    assert output.is_dir()
    assert not attempt.exists()


def test_canonical_output_symlink_is_rejected_before_resolution(tmp_path, monkeypatch):
    _patch_contract(monkeypatch)
    bundle = tmp_path / "bundle"
    adapter = tmp_path / "adapter"
    target = tmp_path / "target"
    output = tmp_path / "eval"
    bundle.mkdir()
    adapter.mkdir()
    target.mkdir()
    try:
        os.symlink(target, output, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    metadata = SimpleNamespace(to_dict=lambda: {"global_step": 2})
    with pytest.raises(g1_eval.G1EvalError, match="output path must not be a symlink"):
        g1_eval.run_or_verify(
            bundle,
            "a" * 64,
            adapter,
            output,
            verify_existing=False,
            adapter_validator=lambda path: metadata,
            evaluation_task=lambda *args, **kwargs: pytest.fail("eval must not run"),
        )
