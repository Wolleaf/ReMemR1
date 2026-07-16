"""Run or re-verify the bounded G1 adapter recurrent-evaluation gate."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from taskutils.memory_eval import reproduction_runner as runner


BASE_MODEL = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
SAMPLE_COUNT = 2
VARIANT = 16
EXPECTED_CONTRACT = {
    "chunk_size": 1024,
    "pool_document_count": 16,
    "prefix_document_count": 8,
    "qa_count": 2,
}
MODEL_LOAD_TIMEOUT = 1800.0
SAMPLE_TIMEOUT = 1800.0
TASK_TIMEOUT = 7200.0


class G1EvalError(RuntimeError):
    """Raised when the bounded G1 eval contract is not satisfied."""


def _sync_parent(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sibling_path(output_dir: Path, kind: str) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return output_dir.with_name(
        f".{output_dir.name}.{kind}-{stamp}-{os.getpid()}-{time.time_ns()}"
    )


def _load_inputs(
    bundle_dir: Path,
    manifest_sha256: str,
) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
    records, manifest = runner.load_eval_records(
        bundle_dir,
        expected_manifest_sha256=manifest_sha256,
        variant=VARIANT,
        sample_count=SAMPLE_COUNT,
    )
    if (
        manifest.get("mode") != "eval"
        or manifest.get("profile") != "fixture"
        or manifest.get("dataset") != "hotpotqa"
        or manifest.get("contract") != EXPECTED_CONTRACT
    ):
        raise G1EvalError("G1 eval bundle does not match the bounded fixture contract")
    if len(records) != SAMPLE_COUNT:
        raise G1EvalError("G1 eval fixture has an unexpected sample count")
    if any(len(record.chunks) != 2 for record in records):
        raise G1EvalError("every G1 eval fixture sample must have exactly two chunks")
    return records, manifest


def _run_contract(
    bundle_dir: Path,
    manifest: Mapping[str, Any],
    adapter_dir: Path,
) -> tuple[runner.ModelLoadSpec, runner.SampleEvaluationConfig, dict[str, Any]]:
    backend = runner.ModelLoadSpec(
        artifact_kind="adapter",
        base_model_id=BASE_MODEL,
        revision=MODEL_REVISION,
        artifact_path=str(adapter_dir),
        tokenizer_id=BASE_MODEL,
        tokenizer_revision=MODEL_REVISION,
        template_revision=runner.PROMPT_TEMPLATE_REVISION,
        attention_implementation="sdpa",
        device="cuda:0",
        seed=42,
        local_files_only=True,
    )
    sample = runner.SampleEvaluationConfig(
        callback_mode="learned",
        chunk_size=EXPECTED_CONTRACT["chunk_size"],
        memory_max_tokens=256,
        final_max_tokens=256,
    )
    input_metadata = {
        "bundle_manifest_sha256": manifest["manifest_sha256"],
        "bundle_path": str(bundle_dir),
        "dataset": "hotpotqa",
        "formal_evaluation": False,
        "gate_kind": "g1-hf-recurrent-eval-v1",
        "sample_count": SAMPLE_COUNT,
        "variant": VARIANT,
    }
    return backend, sample, input_metadata


def _validate_model_loading_evidence(
    model_metadata: Any,
    adapter_metadata: Mapping[str, Any],
) -> None:
    if not isinstance(model_metadata, Mapping):
        raise G1EvalError("G1 eval lacks model-loading evidence")
    expected_model_fields = {
        "adapter_metadata": dict(adapter_metadata),
        "artifact_kind": "adapter",
        "backend": "transformers",
        "base_model_id": BASE_MODEL,
        "device": "cuda:0",
        "dtype": "bfloat16",
        "greedy": True,
        "merged_metadata": None,
        "revision": MODEL_REVISION,
        "seed": 42,
        "tokenizer_id": BASE_MODEL,
        "tokenizer_revision": MODEL_REVISION,
    }
    for key, expected in expected_model_fields.items():
        if model_metadata.get(key) != expected:
            raise G1EvalError(f"G1 eval model-loading evidence changed: {key}")
    mapping = model_metadata.get("qwen35_mapping")
    if not isinstance(mapping, Mapping):
        raise G1EvalError("G1 eval lacks Qwen3.5 text-mapping evidence")
    expected_mapping_fields = {
        "attention_implementation": "sdpa",
        "dtype": "bfloat16",
        "loader": "transformers.AutoModelForCausalLM",
        "mapping_strategy": "transformers_native_qwen35_causal_lm",
        "model_name_or_path": BASE_MODEL,
        "revision": MODEL_REVISION,
        "schema_version": 2,
        "strict_loading": True,
        "target_architecture": "Qwen3_5ForCausalLM",
    }
    for key, expected in expected_mapping_fields.items():
        if mapping.get(key) != expected:
            raise G1EvalError(f"G1 eval Qwen3.5 loading evidence changed: {key}")


def _validate_existing(
    output_dir: Path,
    records: Sequence[Any],
    backend: runner.ModelLoadSpec,
    sample: runner.SampleEvaluationConfig,
    input_metadata: Mapping[str, Any],
    adapter_metadata: Mapping[str, Any],
) -> Mapping[str, Any]:
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise G1EvalError("G1 eval output is missing or unsafe")
    if any(path.is_symlink() or not path.is_file() for path in output_dir.iterdir()):
        raise G1EvalError("G1 eval output contains an unsafe entry")
    qa_ids = [record.qa.qa_id for record in records]
    summary = runner._validate_result_directory(output_dir, expected_qa_ids=qa_ids)
    config = runner._read_canonical_json(output_dir / runner.RUN_CONFIG_FILENAME)
    expected_fields = {
        "backend": backend.public_dict(),
        "callback_mode": sample.callback_mode,
        "input": dict(input_metadata),
        "kind": runner.RUN_KIND,
        "prompt_template_sha256": runner.PROMPT_TEMPLATE_SHA256,
        "prompt_template_revision": runner.PROMPT_TEMPLATE_REVISION,
        "qa_ids": qa_ids,
        "sample": sample.to_dict(),
        "schema_version": runner.RUN_SCHEMA_VERSION,
        "timeouts": {
            "model_load_seconds": MODEL_LOAD_TIMEOUT,
            "sample_seconds": SAMPLE_TIMEOUT,
            "task_seconds": TASK_TIMEOUT,
        },
    }
    if set(config) != {*expected_fields, "model_metadata"}:
        raise G1EvalError("G1 eval run config field set changed")
    for key, expected in expected_fields.items():
        if config.get(key) != expected:
            raise G1EvalError(f"G1 eval run config changed: {key}")
    _validate_model_loading_evidence(config.get("model_metadata"), adapter_metadata)
    if (
        summary.get("status") != "completed"
        or summary.get("failure_count") != 0
        or summary.get("success_count") != SAMPLE_COUNT
        or summary.get("sample_count") != SAMPLE_COUNT
    ):
        raise G1EvalError("G1 recurrent evaluation did not complete every fixture sample")
    return summary


def run_or_verify(
    bundle_dir: Path,
    manifest_sha256: str,
    adapter_dir: Path,
    output_dir: Path,
    *,
    verify_existing: bool,
    adapter_validator: Callable[[Path], Any] | None = None,
    evaluation_task: Callable[..., Mapping[str, Any]] = runner.run_evaluation_task,
) -> Mapping[str, Any]:
    if adapter_validator is None:
        from verl.utils.checkpoint.reproduction import validate_adapter_export

        adapter_validator = validate_adapter_export

    bundle_candidate = bundle_dir.expanduser()
    adapter_candidate = adapter_dir.expanduser()
    output_candidate = output_dir.expanduser()
    for label, candidate in (
        ("bundle", bundle_candidate),
        ("adapter", adapter_candidate),
        ("output", output_candidate),
    ):
        if candidate.is_symlink():
            raise G1EvalError(f"G1 eval {label} path must not be a symlink")
    bundle_dir = bundle_candidate.resolve(strict=True)
    adapter_dir = adapter_candidate.resolve(strict=True)
    output_dir = output_candidate.resolve(strict=False)
    adapter_metadata = adapter_validator(adapter_dir).to_dict()
    if adapter_metadata.get("global_step") != 2:
        raise G1EvalError("G1 eval requires the resumed global_step_2 adapter")
    records, manifest = _load_inputs(bundle_dir, manifest_sha256)
    backend, sample, input_metadata = _run_contract(bundle_dir, manifest, adapter_dir)
    if output_dir.exists():
        try:
            return _validate_existing(
                output_dir,
                records,
                backend,
                sample,
                input_metadata,
                adapter_metadata,
            )
        except Exception:
            if verify_existing or output_dir.is_symlink() or not output_dir.is_dir():
                raise
            archive = _sibling_path(output_dir, "failed")
            os.replace(output_dir, archive)
            _sync_parent(output_dir)
            print(f"archived failed G1 eval evidence: {archive}", file=sys.stderr)
    if verify_existing:
        raise G1EvalError("G1 eval output does not exist")
    for candidate in sorted(
        output_dir.parent.glob(f".{output_dir.name}.attempt-*"), reverse=True
    ):
        try:
            summary = _validate_existing(
                candidate,
                records,
                backend,
                sample,
                input_metadata,
                adapter_metadata,
            )
        except Exception:
            continue
        os.replace(candidate, output_dir)
        _sync_parent(output_dir)
        return summary

    attempt = _sibling_path(output_dir, "attempt")
    try:
        evaluation_task(
            records,
            output_dir=attempt,
            backend_spec=backend,
            sample_config=sample,
            model_load_timeout_s=MODEL_LOAD_TIMEOUT,
            sample_timeout_s=SAMPLE_TIMEOUT,
            task_timeout_s=TASK_TIMEOUT,
            input_metadata=input_metadata,
        )
        _validate_existing(
            attempt,
            records,
            backend,
            sample,
            input_metadata,
            adapter_metadata,
        )
        os.replace(attempt, output_dir)
        _sync_parent(output_dir)
        return _validate_existing(
            output_dir,
            records,
            backend,
            sample,
            input_metadata,
            adapter_metadata,
        )
    except Exception:
        if attempt.exists() and attempt.is_dir() and not attempt.is_symlink():
            archive = _sibling_path(output_dir, "failed")
            os.replace(attempt, archive)
            _sync_parent(output_dir)
            print(f"preserved failed G1 eval evidence: {archive}", file=sys.stderr)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = run_or_verify(
            args.bundle_dir,
            args.expected_manifest_sha256,
            args.adapter_dir,
            args.output_dir,
            verify_existing=args.verify_existing,
        )
    except Exception as exc:
        print(
            json.dumps(
                {"error": f"{type(exc).__name__}: {exc}", "status": "blocked"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
