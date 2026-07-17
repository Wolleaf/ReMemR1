"""Plan, run, and verify the formal 2B evaluation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from taskutils.data_synthesis.reproduction_manifest import canonical_json_bytes
from taskutils.memory_eval import reproduction_runner as runner
from taskutils.memory_eval.reproduction_metrics import (
    evaluate_output,
    stratified_paired_bootstrap_delta,
)


EVALUATOR_VERSION = "rememr1-eval-matrix-v1"
PLAN_KIND = "rememr1-formal-eval-matrix-plan-v1"
PACKAGE_KIND = "rememr1-formal-eval-verified-package-v1"
AGGREGATE_KIND = "rememr1-formal-eval-aggregate-v1"
PLAN_SCHEMA_VERSION = 1
BINDING_SCHEMA_VERSION = 1
PACKAGE_SCHEMA_VERSION = 1
BOOTSTRAP_RESAMPLES = 10_000
BASE_MODEL = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
DATASETS = ("hotpotqa", "2wikimultihopqa")
DOCUMENT_VARIANTS = (200, 800)
CALLBACK_MODES = ("learned", "none", "fixed_question")
AGGREGATE_FILENAME = "aggregate.json"
PACKAGE_FILENAME = "package.json"
VERIFIED_FILENAME = "VERIFIED.json"

_SHA256_LENGTH = 64
_LEVELS = {
    "l1": {"sample_count": 32, "step": 40, "b": "b40", "c": "c40"},
    "l2": {"sample_count": 64, "step": 80, "b": "b80", "c": "c80"},
}
_EVAL_KEYS = {
    "schema_version",
    "delivery_level",
    "evaluator_version",
    "runner_module",
    "backend",
    "base_model_id",
    "revision",
    "tokenizer_id",
    "tokenizer_revision",
    "template_revision",
    "attention_implementation",
    "dtype",
    "device",
    "local_files_only",
    "seed",
    "datasets",
    "document_variants",
    "sample_count_per_stratum",
    "required_cell_count",
    "primary_callback_mode",
    "callback_ablation",
    "artifacts",
    "decode",
    "timeouts_seconds",
    "output_root",
}
_ARTIFACT_KEYS = {
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
_BOUND_ARTIFACT_KEYS = {
    "artifact_path",
    "training_config_id",
    "training_config_path",
    "training_config_sha256",
    "adapter_metadata_sha256",
    "checkpoint_extra_state_sha256",
}
_RESUME_PROBE_RECORD_KEYS = {"path", "sha256", "probe_sha256"}
_CELL_KEYS = {
    "cell_id",
    "dataset",
    "document_count",
    "artifact",
    "callback_mode",
    "sample_count",
    "bundle_dir",
    "manifest_sha256",
    "ordered_qa_ids",
    "ordered_qa_ids_sha256",
    "output_dir",
}
_SUCCESS_RESULT_KEYS = {
    "callback_mode",
    "context_sha256",
    "context_token_count",
    "context_token_ids_sha256",
    "document_count",
    "elapsed_seconds",
    "gold_answers",
    "manifest_record_sha256",
    "metrics",
    "parsed_answer",
    "processed_chunk_count",
    "processed_doc_count",
    "prompt_template_revision",
    "prompt_template_sha256",
    "qa_id",
    "qa_index",
    "raw_final_output",
    "status",
    "trajectory",
}
_METRIC_KEYS = {"answer", "callback", "format", "scores", "truncation"}
_ANSWER_KEYS = {"extraction", "scores"}
_EXTRACTION_KEYS = {
    "answer",
    "fallback_success",
    "mode",
    "strict_boxed_success",
}
_SCORE_KEYS = {
    "exact_match",
    "exact_match_gold",
    "substring_exact_match",
    "substring_exact_match_gold",
    "token_f1",
    "token_f1_gold",
}
_CALLBACK_METRIC_KEYS = {
    "average_lookback_distance",
    "duplicate_retrieved_state_count",
    "duplicate_retrieved_state_rate",
    "effective_query_count",
    "effective_query_rate",
    "eligible_steps",
    "lexical_gold_hit_count",
    "lexical_gold_hit_rate",
    "lexical_supporting_fact_hit_count",
    "lexical_supporting_fact_hit_rate",
    "model_query_count",
    "model_query_rate",
    "model_query_status_counts",
    "model_query_status_rates",
    "retrieval_count",
    "retrieval_empty_count",
    "retrieval_empty_rate",
    "retrieval_rate",
    "retrieval_success_rate",
    "supporting_doc_hit_count",
    "supporting_doc_hit_rate",
}
_QUERY_STATUS_KEYS = {"absent", "duplicate", "empty", "malformed", "valid"}
_FORMAT_METRIC_KEYS = {
    "final_valid",
    "intermediate_valid_count",
    "intermediate_valid_rate",
    "recall_protocol_valid_count",
    "recall_protocol_valid_rate",
    "thinking_single_non_empty_count",
    "thinking_single_non_empty_rate",
    "update_single_non_empty_count",
    "update_single_non_empty_rate",
}
_TRUNCATION_METRIC_KEYS = {
    "final_reached_token_limit",
    "memory_reached_token_limit_count",
    "memory_reached_token_limit_rate",
}
_COMPLETION_MARKER_KEYS = {
    "failure_count",
    "kind",
    "marker_sha256",
    "results_sha256",
    "run_config_sha256",
    "schema_version",
    "status",
    "success_count",
    "summary_sha256",
}
_AGGREGATE_KEYS = {
    "aggregate_sha256",
    "cell_count",
    "delivery_level",
    "kind",
    "methods",
    "plan_sha256",
    "primary",
    "sample_count_per_stratum",
    "schema_version",
    "status",
}
_PACKAGE_KEYS = {
    "aggregate_sha256",
    "artifacts",
    "capacity_approval",
    "capacity_profile",
    "cell_count",
    "cells",
    "delivery_level",
    "evaluation_config",
    "kind",
    "package_sha256",
    "plan_sha256",
    "resume_probes",
    "runtime_binding",
    "schema_version",
    "status",
}
_VERIFIED_MARKER_KEYS = {
    "aggregate_file_sha256",
    "aggregate_sha256",
    "cell_count",
    "kind",
    "marker_sha256",
    "package_file_sha256",
    "package_sha256",
    "plan_sha256",
    "schema_version",
    "status",
}


class EvalMatrixError(RuntimeError):
    """Raised when a matrix identity or result fails closed."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any] | Sequence[Any]) -> str:
    return _sha256_bytes(canonical_json_bytes(value))


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise EvalMatrixError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvalMatrixError(f"{label} must be a mapping")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EvalMatrixError(f"{label} must be a lowercase SHA-256")
    return value


def _require_nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvalMatrixError(f"{label} must be a non-empty string")
    return value


def _absolute_path(value: Any, label: str, *, exists: bool) -> Path:
    raw = os.fspath(value) if isinstance(value, os.PathLike) else _require_nonempty(value, label)
    if not isinstance(raw, str) or not raw.strip():
        raise EvalMatrixError(f"{label} must be a non-empty path")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        raise EvalMatrixError(f"{label} must be absolute")
    for component in [*reversed(candidate.parents), candidate]:
        if component.is_symlink():
            raise EvalMatrixError(f"{label} contains a symlink: {component}")
    try:
        return candidate.resolve(strict=exists)
    except OSError as exc:
        raise EvalMatrixError(f"cannot resolve {label}: {exc}") from exc


def _file_record(path: Any, digest: Any, label: str) -> dict[str, str]:
    resolved = _absolute_path(path, f"{label}.path", exists=True)
    if not resolved.is_file():
        raise EvalMatrixError(f"{label}.path must be a file")
    expected = _require_sha256(digest, f"{label}.sha256")
    if _sha256_file(resolved) != expected:
        raise EvalMatrixError(f"{label} hash changed")
    return {"path": str(resolved), "sha256": expected}


def _verify_file_record(value: Any, label: str) -> None:
    record = _require_mapping(value, label)
    _require_exact_keys(record, {"path", "sha256"}, label)
    _file_record(record["path"], record["sha256"], label)


def _capacity_approval_records(value: Any, label: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    records = _require_mapping(value, label)
    _require_exact_keys(records, {"marker", "consumption"}, label)
    for name in ("marker", "consumption"):
        _verify_file_record(records[name], f"{label}.{name}")
    return records


def _resume_probe_records(
    value: Any,
    label: str,
    *,
    expected_names: set[str],
) -> dict[str, dict[str, str]]:
    records = _require_mapping(value, label)
    _require_exact_keys(records, expected_names, label)
    normalized = {}
    for name in sorted(expected_names):
        record = _require_mapping(records[name], f"{label}.{name}")
        _require_exact_keys(
            record,
            _RESUME_PROBE_RECORD_KEYS,
            f"{label}.{name}",
        )
        file_record = _file_record(
            record["path"], record["sha256"], f"{label}.{name}"
        )
        normalized[name] = {
            **file_record,
            "probe_sha256": _require_sha256(
                record["probe_sha256"], f"{label}.{name}.probe_sha256"
            ),
        }
    return normalized


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvalMatrixError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path, *, canonical: bool) -> Mapping[str, Any]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload, object_pairs_hook=_reject_duplicate_pairs)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise EvalMatrixError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise EvalMatrixError(f"{path} must contain an object")
    if canonical and canonical_json_bytes(value) + b"\n" != payload:
        raise EvalMatrixError(f"{path} is not canonical newline-terminated JSON")
    return value


def _load_eval_source(path: Path) -> tuple[Mapping[str, Any], str]:
    source = _absolute_path(path, "evaluation config", exists=True)
    if not source.is_file():
        raise EvalMatrixError("evaluation config must be a file")
    payload = source.read_bytes()
    try:
        if source.suffix.lower() == ".json":
            raw = json.loads(payload, object_pairs_hook=_reject_duplicate_pairs)
        else:
            import yaml

            class UniqueKeyLoader(yaml.SafeLoader):
                pass

            def construct_unique_mapping(loader, node, deep=False):
                mapping = {}
                for key_node, value_node in node.value:
                    key = loader.construct_object(key_node, deep=deep)
                    if key in mapping:
                        raise EvalMatrixError(f"duplicate YAML key: {key}")
                    mapping[key] = loader.construct_object(value_node, deep=deep)
                return mapping

            UniqueKeyLoader.add_constructor(
                yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                construct_unique_mapping,
            )
            raw = yaml.load(payload, Loader=UniqueKeyLoader)
    except Exception as exc:
        raise EvalMatrixError(f"cannot parse evaluation config: {exc}") from exc
    root = _require_mapping(raw, "evaluation config")
    evaluation = root.get("reproduction_evaluation", root)
    return _require_mapping(evaluation, "reproduction_evaluation"), _sha256_bytes(payload)


def _sync_parent(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _sync_parent(destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _level_contract(level: Any) -> Mapping[str, Any]:
    if level not in _LEVELS:
        raise EvalMatrixError("delivery_level must be l1 or l2")
    return _LEVELS[level]


def _validate_source_config(value: Mapping[str, Any]) -> Mapping[str, Any]:
    _require_exact_keys(value, _EVAL_KEYS, "reproduction_evaluation")
    if value["schema_version"] != 2:
        raise EvalMatrixError("evaluation schema_version must be 2")
    level = value["delivery_level"]
    contract = _level_contract(level)
    expected_scalars = {
        "evaluator_version": EVALUATOR_VERSION,
        "runner_module": "taskutils.memory_eval.reproduction_runner",
        "backend": "transformers",
        "base_model_id": BASE_MODEL,
        "revision": MODEL_REVISION,
        "tokenizer_id": BASE_MODEL,
        "tokenizer_revision": MODEL_REVISION,
        "template_revision": runner.PROMPT_TEMPLATE_REVISION,
        "attention_implementation": "sdpa",
        "dtype": "bfloat16",
        "device": "cuda:0",
        "local_files_only": True,
        "seed": 42,
        "sample_count_per_stratum": contract["sample_count"],
        "required_cell_count": 20,
        "primary_callback_mode": "learned",
    }
    for name, expected in expected_scalars.items():
        if value[name] != expected:
            raise EvalMatrixError(f"evaluation config differs at {name}")
    if list(value["document_variants"]) != list(DOCUMENT_VARIANTS):
        raise EvalMatrixError("document_variants must be [200, 800]")

    datasets = _require_mapping(value["datasets"], "datasets")
    _require_exact_keys(datasets, set(DATASETS), "datasets")
    manifest_hashes = set()
    for name in DATASETS:
        dataset = _require_mapping(datasets[name], f"datasets.{name}")
        _require_exact_keys(dataset, {"bundle_dir", "manifest_sha256"}, f"datasets.{name}")
        _absolute_path(dataset["bundle_dir"], f"datasets.{name}.bundle_dir", exists=True)
        manifest_hashes.add(_require_sha256(dataset["manifest_sha256"], f"datasets.{name}.manifest_sha256"))
    if len(manifest_hashes) != len(DATASETS):
        raise EvalMatrixError("dataset manifests must have distinct identities")

    callback = _require_mapping(value["callback_ablation"], "callback_ablation")
    _require_exact_keys(callback, {"artifact", "modes"}, "callback_ablation")
    if callback["artifact"] != contract["c"] or list(callback["modes"]) != list(CALLBACK_MODES):
        raise EvalMatrixError("callback ablation does not match the delivery level")

    artifacts = _require_mapping(value["artifacts"], "artifacts")
    expected_artifacts = {"base", contract["b"], contract["c"]}
    _require_exact_keys(artifacts, expected_artifacts, "artifacts")
    for name, raw in artifacts.items():
        artifact = _require_mapping(raw, f"artifacts.{name}")
        _require_exact_keys(artifact, _ARTIFACT_KEYS, f"artifacts.{name}")
        if name == "base":
            if artifact["artifact_kind"] != "base" or artifact["training_step"] != 0:
                raise EvalMatrixError("base artifact identity is invalid")
            nullable = _ARTIFACT_KEYS - {"artifact_kind", "training_step"}
            if any(artifact[field] is not None for field in nullable):
                raise EvalMatrixError("base artifact must not carry training identity")
        else:
            if artifact["artifact_kind"] != "adapter":
                raise EvalMatrixError(f"{name} must be an adapter")
            if artifact["training_step"] != contract["step"]:
                raise EvalMatrixError(f"{name} training step differs from level")
            _require_nonempty(
                artifact["training_config_source_id"],
                f"artifacts.{name}.training_config_source_id",
            )
            for field in _BOUND_ARTIFACT_KEYS:
                if artifact[field] is not None:
                    raise EvalMatrixError(
                        f"artifacts.{name}.{field} must be null before runtime binding"
                    )

    decode = _require_mapping(value["decode"], "decode")
    expected_decode = {
        "greedy": True,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "n": 1,
        "chunk_size": 5000,
        "memory_max_tokens": 768,
        "final_max_tokens": 512,
    }
    _require_exact_keys(decode, set(expected_decode), "decode")
    if dict(decode) != expected_decode:
        raise EvalMatrixError("decode must use the fixed greedy identity")
    timeouts = _require_mapping(value["timeouts_seconds"], "timeouts_seconds")
    _require_exact_keys(timeouts, {"model_load", "sample", "task"}, "timeouts_seconds")
    for name, timeout in timeouts.items():
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise EvalMatrixError(f"timeouts_seconds.{name} must be positive")
    _absolute_path(value["output_root"], "output_root", exists=False)
    return value


def _validate_binding_value(
    value: Any,
    *,
    config_sha256: str | None = None,
    level: str | None = None,
) -> Mapping[str, Any]:
    binding = _require_mapping(value, "runtime binding")
    expected_keys = {
        "schema_version",
        "delivery_level",
        "evaluation_config_sha256",
        "capacity_profile",
        "capacity_approval",
        "artifacts",
        "resume_probes",
        "runtime_binding_sha256",
    }
    _require_exact_keys(binding, expected_keys, "runtime binding")
    if binding["schema_version"] != BINDING_SCHEMA_VERSION:
        raise EvalMatrixError("runtime binding schema_version is unsupported")
    binding_level = binding["delivery_level"]
    contract = _level_contract(binding_level)
    if level is not None and binding_level != level:
        raise EvalMatrixError("runtime binding delivery level differs")
    _require_sha256(
        binding["evaluation_config_sha256"], "evaluation_config_sha256"
    )
    if (
        config_sha256 is not None
        and binding["evaluation_config_sha256"] != config_sha256
    ):
        raise EvalMatrixError("runtime binding is for a different evaluation config")
    capacity = _require_mapping(binding["capacity_profile"], "capacity_profile")
    _require_exact_keys(capacity, {"path", "sha256"}, "capacity_profile")
    _capacity_approval_records(binding["capacity_approval"], "capacity_approval")
    _resume_probe_records(
        binding["resume_probes"],
        "resume_probes",
        expected_names={contract["b"], contract["c"]},
    )
    artifacts = _require_mapping(binding["artifacts"], "runtime binding artifacts")
    _require_exact_keys(
        artifacts,
        {contract["b"], contract["c"]},
        "runtime binding artifacts",
    )
    for name, artifact_value in artifacts.items():
        artifact = _require_mapping(
            artifact_value, f"runtime binding artifacts.{name}"
        )
        _require_exact_keys(
            artifact,
            _BOUND_ARTIFACT_KEYS,
            f"runtime binding artifacts.{name}",
        )
    digest = _require_sha256(binding["runtime_binding_sha256"], "runtime_binding_sha256")
    unsigned = {key: value for key, value in binding.items() if key != "runtime_binding_sha256"}
    if digest != _canonical_sha256(unsigned):
        raise EvalMatrixError("runtime binding self-hash mismatch")
    return binding


def _load_binding(path: Path, config_sha256: str, level: str) -> tuple[Mapping[str, Any], str]:
    binding_path = _absolute_path(path, "runtime binding", exists=True)
    binding = _validate_binding_value(
        _load_json(binding_path, canonical=True),
        config_sha256=config_sha256,
        level=level,
    )
    return binding, _sha256_file(binding_path)


def _metadata_dict(value: Any) -> Mapping[str, Any]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    return _require_mapping(value, "adapter metadata")


def _adapter_validator() -> Callable[[Path], Any]:
    from verl.utils.checkpoint.reproduction import validate_adapter_export

    return validate_adapter_export


def _checkpoint_validator() -> Callable[[Path], Any]:
    from verl.utils.checkpoint.reproduction import (
        verify_reproduction_checkpoint_directory,
    )

    return verify_reproduction_checkpoint_directory


def _training_binding_verifier() -> Callable[[Path], Mapping[str, Any]]:
    from scripts.cloud.run_resolved_training import verify_bound_config

    return verify_bound_config


def _resume_probe_verifier() -> Callable[[Path], Mapping[str, Any]]:
    from scripts.cloud.resume_endpoint_probe import verify_resume_endpoint_probe

    return verify_resume_endpoint_probe


def _verify_capacity_profile(
    value: Mapping[str, Any],
    *,
    approval_marker_path: str | os.PathLike[str] | None,
    approval_consumption_path: str | os.PathLike[str] | None,
    r0_capacity_evidence_path: str | os.PathLike[str] | None,
    r0_terminal_sha256: str | None,
    budget_projection_sha256: str | None,
) -> Mapping[str, Any]:
    from scripts.cloud.capacity_evidence import verify_capacity_profile

    def optional_json(
        path: str | os.PathLike[str] | None, label: str
    ) -> Mapping[str, Any] | None:
        if path is None:
            return None
        resolved = _absolute_path(path, label, exists=True)
        if not resolved.is_file():
            raise EvalMatrixError(f"{label} must be a file")
        return _load_json(resolved, canonical=False)

    approval_marker = optional_json(approval_marker_path, "R1 approval marker")
    approval_consumption = optional_json(
        approval_consumption_path, "R1 approval consumption"
    )
    return verify_capacity_profile(
        value,
        approval_marker=approval_marker,
        approval_marker_path=approval_marker_path,
        approval_consumption=approval_consumption,
        approval_consumption_path=approval_consumption_path,
        r0_capacity_evidence=optional_json(
            r0_capacity_evidence_path, "R0 capacity evidence"
        ),
        r0_terminal_sha256=r0_terminal_sha256,
        budget_projection_sha256=budget_projection_sha256,
    )


def _successful_attempt(value: str | os.PathLike[str], label: str) -> Path:
    attempt = _absolute_path(value, label, exists=True)
    if not attempt.is_dir():
        raise EvalMatrixError(f"{label} must be a directory")
    success = attempt / ".success"
    if success.is_symlink() or not success.is_file():
        raise EvalMatrixError(f"{label} lacks an immutable success marker")
    if success.read_text(encoding="ascii").strip() != "0":
        raise EvalMatrixError(f"{label} success marker is invalid")
    for marker in (".running", ".failed", ".scientific-stop"):
        if (attempt / marker).exists() or (attempt / marker).is_symlink():
            raise EvalMatrixError(f"{label} has conflicting terminal state")
    return attempt


def _state_mapping(value: Any, label: str) -> tuple[Mapping[str, Any], str]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise EvalMatrixError(f"{label} validator returned an invalid tuple")
        value = value[1]
    serialized = value.to_dict() if hasattr(value, "to_dict") else value
    state = _require_mapping(serialized, label)
    semantic_sha = getattr(value, "sha256", state.get("extra_state_sha256"))
    return state, _require_sha256(semantic_sha, f"{label}.extra_state_sha256")


def _bind_attempt_artifact(
    *,
    artifact_name: str,
    attempt_path: str | os.PathLike[str],
    expected_config_id: str,
    expected_config_sha256: str,
    expected_config_path: Path,
    expected_index_path: Path,
    expected_index_sha256: str,
    step: int,
    training_binding_verifier: Callable[[Path], Mapping[str, Any]],
    checkpoint_validator: Callable[[Path], Any],
    adapter_validator: Callable[[Path], Any],
) -> dict[str, Any]:
    attempt = _successful_attempt(attempt_path, f"{artifact_name} attempt")
    runtime_dir = attempt / "runtime-bound"
    evidence = _require_mapping(
        training_binding_verifier(runtime_dir),
        f"{artifact_name} runtime-bound evidence",
    )
    if evidence.get("config_id") != expected_config_id:
        raise EvalMatrixError(f"{artifact_name} attempt uses a different config ID")
    if evidence.get("attempt_root") != str(attempt):
        raise EvalMatrixError(f"{artifact_name} runtime attempt root changed")
    if evidence.get("attempt_id") != attempt.name:
        raise EvalMatrixError(f"{artifact_name} runtime attempt ID changed")
    source_index = _require_mapping(
        evidence.get("source_index"), f"{artifact_name} source index"
    )
    if source_index != {
        "path": str(expected_index_path),
        "sha256": expected_index_sha256,
    }:
        raise EvalMatrixError(f"{artifact_name} source index identity changed")
    source_config = _require_mapping(
        evidence.get("source_config"), f"{artifact_name} source config"
    )
    if source_config != {
        "path": str(expected_config_path),
        "sha256": expected_config_sha256,
    }:
        raise EvalMatrixError(f"{artifact_name} source config identity changed")
    runtime_config_sha256 = _require_sha256(
        evidence.get("runtime_bound_config_sha256"),
        f"{artifact_name} runtime config SHA-256",
    )

    checkpoint = _absolute_path(
        attempt / "checkpoints" / f"global_step_{step}",
        f"{artifact_name} checkpoint",
        exists=True,
    )
    if not checkpoint.is_dir():
        raise EvalMatrixError(f"{artifact_name} checkpoint must be a directory")
    state, checkpoint_sha256 = _state_mapping(
        checkpoint_validator(checkpoint), f"{artifact_name} checkpoint"
    )
    expected_state = {
        "global_step": step,
        "base_model_id": BASE_MODEL,
        "base_model_revision": MODEL_REVISION,
        "resolved_config_sha256": runtime_config_sha256,
    }
    for field, expected in expected_state.items():
        if state.get(field) != expected:
            raise EvalMatrixError(f"{artifact_name} checkpoint differs at {field}")

    adapter_path = _absolute_path(
        attempt / "artifacts" / "adapter" / f"global_step_{step}" / "adapter",
        f"{artifact_name} adapter",
        exists=True,
    )
    if not adapter_path.is_dir():
        raise EvalMatrixError(f"{artifact_name} adapter must be a directory")
    metadata = _metadata_dict(adapter_validator(adapter_path))
    expected_metadata = {
        "global_step": step,
        "base_model_id": BASE_MODEL,
        "base_model_revision": MODEL_REVISION,
        "tokenizer_id": BASE_MODEL,
        "tokenizer_revision": MODEL_REVISION,
        "template_revision": runner.PROMPT_TEMPLATE_REVISION,
        "source_extra_state_sha256": checkpoint_sha256,
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise EvalMatrixError(f"{artifact_name} adapter differs at {field}")
    metadata_sha256 = _require_sha256(
        metadata.get("metadata_sha256"),
        f"{artifact_name} adapter metadata SHA-256",
    )
    return {
        "artifact_path": str(adapter_path),
        "training_config_id": expected_config_id,
        "training_config_path": str(expected_config_path),
        "training_config_sha256": expected_config_sha256,
        "adapter_metadata_sha256": metadata_sha256,
        "checkpoint_extra_state_sha256": checkpoint_sha256,
    }


def build_runtime_binding(
    index_path: str | os.PathLike[str],
    evaluation_config_id: str,
    capacity_profile_path: str | os.PathLike[str],
    b_attempt_path: str | os.PathLike[str],
    c_attempt_path: str | os.PathLike[str],
    *,
    b_resume_probe_path: str | os.PathLike[str] | None = None,
    c_resume_probe_path: str | os.PathLike[str] | None = None,
    capacity_validator: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    training_binding_verifier: Callable[[Path], Mapping[str, Any]] | None = None,
    checkpoint_validator: Callable[[Path], Any] | None = None,
    adapter_validator: Callable[[Path], Any] | None = None,
    resume_probe_verifier: Callable[[Path], Mapping[str, Any]] | None = None,
    approval_marker_path: str | os.PathLike[str] | None = None,
    approval_consumption_path: str | os.PathLike[str] | None = None,
    r0_capacity_evidence_path: str | os.PathLike[str] | None = None,
    r0_terminal_sha256: str | None = None,
    budget_projection_sha256: str | None = None,
) -> dict[str, Any]:
    from scripts.cloud import run_resolved_training

    index_file = _absolute_path(index_path, "resolved config index", exists=True)
    if not index_file.is_file():
        raise EvalMatrixError("resolved config index must be a file")
    index_sha256 = _sha256_file(index_file)
    try:
        resolved_index, index = run_resolved_training._load_index(
            index_file, index_sha256
        )
    except Exception as exc:
        raise EvalMatrixError(f"resolved config index is invalid: {exc}") from exc
    configs = _require_mapping(index["configs"], "resolved config index.configs")
    evaluation_config_id = _require_nonempty(
        evaluation_config_id, "evaluation_config_id"
    )
    if evaluation_config_id not in configs:
        raise EvalMatrixError("evaluation config ID is not sealed")
    evaluation_entry = _require_mapping(
        configs[evaluation_config_id], f"configs.{evaluation_config_id}"
    )
    evaluation_path = resolved_index.parent / f"{evaluation_config_id}.yaml"
    evaluation, evaluation_sha256 = _load_eval_source(evaluation_path)
    evaluation = _validate_source_config(evaluation)
    if evaluation_entry["sha256"] != evaluation_sha256:
        raise EvalMatrixError("evaluation config hash differs from the sealed index")
    contract = _level_contract(evaluation["delivery_level"])
    expected_evaluation_id = f"eval{contract['step']}_qwen35_2b_5090"
    if evaluation_config_id != expected_evaluation_id:
        raise EvalMatrixError("evaluation config ID differs from the delivery level")

    capacity_file = _absolute_path(
        capacity_profile_path, "capacity profile", exists=True
    )
    if not capacity_file.is_file():
        raise EvalMatrixError("capacity profile must be a file")
    raw_capacity = _load_json(capacity_file, canonical=False)
    if capacity_validator is None:
        capacity = _verify_capacity_profile(
            raw_capacity,
            approval_marker_path=approval_marker_path,
            approval_consumption_path=approval_consumption_path,
            r0_capacity_evidence_path=r0_capacity_evidence_path,
            r0_terminal_sha256=r0_terminal_sha256,
            budget_projection_sha256=budget_projection_sha256,
        )
    else:
        capacity = capacity_validator(raw_capacity)
    capacity = _require_mapping(capacity, "verified capacity profile")
    selected_profile = _require_nonempty(
        capacity.get("selected_profile"), "capacity selected_profile"
    )
    if selected_profile not in {"R0", "R1"}:
        raise EvalMatrixError("capacity selected_profile must be R0 or R1")
    _require_sha256(capacity.get("self_sha256"), "capacity self_sha256")
    selected_configs = _require_mapping(
        capacity.get("selected_configs"), "capacity selected_configs"
    )

    capacity_approval: Mapping[str, Any] | None = None
    if selected_profile == "R1":
        if approval_marker_path is None or approval_consumption_path is None:
            raise EvalMatrixError(
                "R1 runtime binding requires marker and one-time consumption records"
            )
        marker_file = _absolute_path(
            approval_marker_path, "R1 approval marker", exists=True
        )
        consumption_file = _absolute_path(
            approval_consumption_path, "R1 approval consumption", exists=True
        )
        if not marker_file.is_file() or not consumption_file.is_file():
            raise EvalMatrixError("R1 approval records must be regular files")
        capacity_approval = {
            "marker": {
                "path": str(marker_file),
                "sha256": _sha256_file(marker_file),
            },
            "consumption": {
                "path": str(consumption_file),
                "sha256": _sha256_file(consumption_file),
            },
        }
    elif approval_marker_path is not None or approval_consumption_path is not None:
        raise EvalMatrixError("R0 runtime binding must not carry R1 approval records")

    b_attempt = _absolute_path(b_attempt_path, "B attempt", exists=True)
    c_attempt = _absolute_path(c_attempt_path, "C attempt", exists=True)
    if b_attempt == c_attempt:
        raise EvalMatrixError("B and C attempts must be distinct")
    binding_verifier = training_binding_verifier or _training_binding_verifier()
    checkpoint_check = checkpoint_validator or _checkpoint_validator()
    adapter_check = adapter_validator or _adapter_validator()
    artifacts: dict[str, Any] = {}
    for artifact_name, attempt in (
        (contract["b"], b_attempt),
        (contract["c"], c_attempt),
    ):
        source_id = _require_nonempty(
            evaluation["artifacts"][artifact_name]["training_config_source_id"],
            f"artifacts.{artifact_name}.training_config_source_id",
        )
        expected_config_id = f"{source_id}_{selected_profile.lower()}"
        if expected_config_id not in selected_configs or expected_config_id not in configs:
            raise EvalMatrixError(
                f"capacity profile does not select {expected_config_id}"
            )
        expected_sha256 = _require_sha256(
            selected_configs[expected_config_id],
            f"capacity selected_configs.{expected_config_id}",
        )
        if configs[expected_config_id]["sha256"] != expected_sha256:
            raise EvalMatrixError(
                f"capacity profile config hash differs for {expected_config_id}"
            )
        expected_path = resolved_index.parent / f"{expected_config_id}.yaml"
        artifacts[artifact_name] = _bind_attempt_artifact(
            artifact_name=artifact_name,
            attempt_path=attempt,
            expected_config_id=expected_config_id,
            expected_config_sha256=expected_sha256,
            expected_config_path=expected_path,
            expected_index_path=resolved_index,
            expected_index_sha256=index_sha256,
            step=contract["step"],
            training_binding_verifier=binding_verifier,
            checkpoint_validator=checkpoint_check,
            adapter_validator=adapter_check,
        )

    probe_paths = {
        contract["b"]: b_resume_probe_path,
        contract["c"]: c_resume_probe_path,
    }
    if any(value is None for value in probe_paths.values()):
        raise EvalMatrixError("B and C endpoint resume probes are required")
    probe_check = resume_probe_verifier or _resume_probe_verifier()
    resume_probes: dict[str, Any] = {}
    for artifact_name, attempt in (
        (contract["b"], b_attempt),
        (contract["c"], c_attempt),
    ):
        probe_file = _absolute_path(
            probe_paths[artifact_name],
            f"{artifact_name} resume probe",
            exists=True,
        )
        if not probe_file.is_file():
            raise EvalMatrixError(f"{artifact_name} resume probe must be a file")
        evidence = _require_mapping(
            probe_check(probe_file), f"{artifact_name} resume probe evidence"
        )
        checkpoint = _require_mapping(
            evidence.get("checkpoint"), f"{artifact_name} probe checkpoint"
        )
        root_extra = _require_mapping(
            checkpoint.get("root_extra_state"),
            f"{artifact_name} probe root extra-state",
        )
        adapter = _require_mapping(
            evidence.get("adapter"), f"{artifact_name} probe adapter"
        )
        expected_probe_identity = {
            "endpoint": artifact_name,
            "config_id": artifacts[artifact_name]["training_config_id"],
            "attempt_root": str(attempt),
            "global_step": contract["step"],
            "offload_profile": selected_profile.lower(),
        }
        for field, expected in expected_probe_identity.items():
            if evidence.get(field) != expected:
                raise EvalMatrixError(
                    f"{artifact_name} resume probe differs at {field}"
                )
        if (
            root_extra.get("extra_state_sha256")
            != artifacts[artifact_name]["checkpoint_extra_state_sha256"]
            or adapter.get("metadata_sha256")
            != artifacts[artifact_name]["adapter_metadata_sha256"]
        ):
            raise EvalMatrixError(
                f"{artifact_name} resume probe differs from bound artifacts"
            )
        resume_probes[artifact_name] = {
            "path": str(probe_file),
            "sha256": _sha256_file(probe_file),
            "probe_sha256": _require_sha256(
                evidence.get("probe_sha256"),
                f"{artifact_name} resume probe SHA-256",
            ),
        }
    if Path(resume_probes[contract["b"]]["path"]) == Path(
        resume_probes[contract["c"]]["path"]
    ):
        raise EvalMatrixError("B and C resume probes must be distinct")

    unsigned = {
        "schema_version": BINDING_SCHEMA_VERSION,
        "delivery_level": evaluation["delivery_level"],
        "evaluation_config_sha256": evaluation_sha256,
        "capacity_profile": {
            "path": str(capacity_file),
            "sha256": _sha256_file(capacity_file),
        },
        "capacity_approval": capacity_approval,
        "artifacts": artifacts,
        "resume_probes": resume_probes,
    }
    return {
        **unsigned,
        "runtime_binding_sha256": _canonical_sha256(unsigned),
    }


def publish_runtime_binding(
    binding: Mapping[str, Any], output_path: str | os.PathLike[str]
) -> Path:
    _validate_binding_value(binding)
    destination = Path(output_path)
    _atomic_write_json(destination, binding)
    return destination.resolve(strict=True)


def _bind_artifacts(
    config: Mapping[str, Any],
    binding: Mapping[str, Any],
    *,
    adapter_validator: Callable[[Path], Any],
) -> dict[str, Any]:
    contract = _level_contract(config["delivery_level"])
    expected_names = {contract["b"], contract["c"]}
    bound_values = _require_mapping(binding["artifacts"], "runtime binding artifacts")
    _require_exact_keys(bound_values, expected_names, "runtime binding artifacts")
    result: dict[str, Any] = {"base": dict(config["artifacts"]["base"])}
    for name in sorted(expected_names):
        raw = _require_mapping(bound_values[name], f"runtime binding artifacts.{name}")
        _require_exact_keys(raw, _BOUND_ARTIFACT_KEYS, f"runtime binding artifacts.{name}")
        artifact_path = _absolute_path(raw["artifact_path"], f"artifacts.{name}.artifact_path", exists=True)
        if not artifact_path.is_dir():
            raise EvalMatrixError(f"artifacts.{name}.artifact_path must be a directory")
        training_config = _file_record(
            raw["training_config_path"],
            raw["training_config_sha256"],
            f"artifacts.{name}.training_config",
        )
        metadata = _metadata_dict(adapter_validator(artifact_path))
        expected_metadata = {
            "global_step": contract["step"],
            "base_model_id": BASE_MODEL,
            "base_model_revision": MODEL_REVISION,
            "tokenizer_id": BASE_MODEL,
            "tokenizer_revision": MODEL_REVISION,
            "template_revision": runner.PROMPT_TEMPLATE_REVISION,
            "metadata_sha256": _require_sha256(
                raw["adapter_metadata_sha256"],
                f"artifacts.{name}.adapter_metadata_sha256",
            ),
            "source_extra_state_sha256": _require_sha256(
                raw["checkpoint_extra_state_sha256"],
                f"artifacts.{name}.checkpoint_extra_state_sha256",
            ),
        }
        for field, expected in expected_metadata.items():
            if metadata.get(field) != expected:
                raise EvalMatrixError(f"adapter {name} metadata differs at {field}")
        source = config["artifacts"][name]
        result[name] = {
            **dict(source),
            "artifact_path": str(artifact_path),
            "training_config_id": _require_nonempty(
                raw["training_config_id"], f"artifacts.{name}.training_config_id"
            ),
            "training_config_path": training_config["path"],
            "training_config_sha256": training_config["sha256"],
            "adapter_metadata_sha256": expected_metadata["metadata_sha256"],
            "checkpoint_extra_state_sha256": expected_metadata[
                "source_extra_state_sha256"
            ],
        }
    return result


def _record_qa_id(record: Any) -> str:
    qa = getattr(record, "qa", None)
    qa_id = getattr(qa, "qa_id", None)
    return _require_nonempty(qa_id, "manifest record qa_id")


def _load_strata(
    config: Mapping[str, Any],
    *,
    record_loader: Callable[..., tuple[Sequence[Any], Mapping[str, Any]]],
) -> dict[tuple[str, int], dict[str, Any]]:
    sample_count = config["sample_count_per_stratum"]
    strata: dict[tuple[str, int], dict[str, Any]] = {}
    per_dataset_ids: dict[str, list[str]] = {}
    for dataset_name in DATASETS:
        dataset = config["datasets"][dataset_name]
        bundle_dir = _absolute_path(
            dataset["bundle_dir"], f"datasets.{dataset_name}.bundle_dir", exists=True
        )
        for variant in DOCUMENT_VARIANTS:
            records, manifest = record_loader(
                bundle_dir,
                expected_manifest_sha256=dataset["manifest_sha256"],
                variant=variant,
                sample_count=sample_count,
            )
            if len(records) != sample_count:
                raise EvalMatrixError("eval record count differs from delivery level")
            manifest = _require_mapping(manifest, f"{dataset_name} manifest")
            if (
                manifest.get("manifest_sha256") != dataset["manifest_sha256"]
                or manifest.get("dataset") != dataset_name
                or manifest.get("mode") != "eval"
                or manifest.get("profile") != "formal"
                or _require_mapping(manifest.get("contract"), "manifest contract").get(
                    "chunk_size"
                )
                != 5000
            ):
                raise EvalMatrixError("eval bundle differs from the formal matrix contract")
            qa_ids = [_record_qa_id(record) for record in records]
            if len(set(qa_ids)) != len(qa_ids):
                raise EvalMatrixError("eval stratum contains duplicate QA IDs")
            previous = per_dataset_ids.setdefault(dataset_name, qa_ids)
            if previous != qa_ids:
                raise EvalMatrixError(
                    f"{dataset_name} 200/800 variants do not share ordered QA IDs"
                )
            strata[(dataset_name, variant)] = {
                "bundle_dir": str(bundle_dir),
                "manifest_sha256": dataset["manifest_sha256"],
                "ordered_qa_ids": qa_ids,
                "ordered_qa_ids_sha256": _canonical_sha256(qa_ids),
            }
    if len(strata) != 4:
        raise EvalMatrixError("formal matrix must contain four strata")
    return strata


def _cell_specs(level: str) -> tuple[tuple[str, str], ...]:
    contract = _level_contract(level)
    return (
        ("base", "learned"),
        (contract["b"], "learned"),
        (contract["c"], "learned"),
        (contract["c"], "none"),
        (contract["c"], "fixed_question"),
    )


def build_plan(
    config_path: str | os.PathLike[str],
    binding_path: str | os.PathLike[str],
    *,
    output_root: str | os.PathLike[str] | None = None,
    record_loader: Callable[..., tuple[Sequence[Any], Mapping[str, Any]]] = runner.load_eval_records,
    adapter_validator: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    config_file = _absolute_path(config_path, "evaluation config", exists=True)
    binding_file = _absolute_path(binding_path, "runtime binding", exists=True)
    config, config_sha256 = _load_eval_source(config_file)
    config = _validate_source_config(config)
    binding, binding_file_sha256 = _load_binding(
        binding_file, config_sha256, config["delivery_level"]
    )
    capacity = _require_mapping(binding["capacity_profile"], "capacity_profile")
    _require_exact_keys(capacity, {"path", "sha256"}, "capacity_profile")
    capacity_record = _file_record(
        capacity["path"], capacity["sha256"], "capacity_profile"
    )
    capacity_approval = _capacity_approval_records(
        binding["capacity_approval"], "capacity_approval"
    )
    resume_probes = _resume_probe_records(
        binding["resume_probes"],
        "resume_probes",
        expected_names={
            _level_contract(config["delivery_level"])["b"],
            _level_contract(config["delivery_level"])["c"],
        },
    )
    artifacts = _bind_artifacts(
        config,
        binding,
        adapter_validator=adapter_validator or _adapter_validator(),
    )
    strata = _load_strata(config, record_loader=record_loader)
    output_root = _absolute_path(
        config["output_root"] if output_root is None else output_root,
        "output_root",
        exists=False,
    )
    if output_root.exists():
        raise FileExistsError(
            f"refusing to reuse evaluation output root {output_root}"
        )
    cells = []
    for dataset_name in DATASETS:
        for variant in DOCUMENT_VARIANTS:
            stratum = strata[(dataset_name, variant)]
            for artifact_name, callback_mode in _cell_specs(config["delivery_level"]):
                cell_id = (
                    f"{dataset_name}-docs{variant}-{artifact_name}-{callback_mode}"
                )
                cells.append(
                    {
                        "cell_id": cell_id,
                        "dataset": dataset_name,
                        "document_count": variant,
                        "artifact": artifact_name,
                        "callback_mode": callback_mode,
                        "sample_count": config["sample_count_per_stratum"],
                        "bundle_dir": stratum["bundle_dir"],
                        "manifest_sha256": stratum["manifest_sha256"],
                        "ordered_qa_ids": list(stratum["ordered_qa_ids"]),
                        "ordered_qa_ids_sha256": stratum[
                            "ordered_qa_ids_sha256"
                        ],
                        "output_dir": str(output_root / "cells" / cell_id),
                    }
                )
    if len(cells) != 20 or len({cell["cell_id"] for cell in cells}) != 20:
        raise EvalMatrixError("formal matrix did not produce 20 unique cells")
    c_name = _level_contract(config["delivery_level"])["c"]
    learned_c = [
        cell
        for cell in cells
        if cell["artifact"] == c_name and cell["callback_mode"] == "learned"
    ]
    if len(learned_c) != 4:
        raise EvalMatrixError("C learned must be deduplicated to one cell per stratum")

    unsigned: dict[str, Any] = {
        "kind": PLAN_KIND,
        "schema_version": PLAN_SCHEMA_VERSION,
        "delivery_level": config["delivery_level"],
        "evaluator_version": EVALUATOR_VERSION,
        "source_config": {"path": str(config_file), "sha256": config_sha256},
        "runtime_binding": {
            "path": str(binding_file),
            "sha256": binding_file_sha256,
            "runtime_binding_sha256": binding["runtime_binding_sha256"],
        },
        "capacity_profile": capacity_record,
        "capacity_approval": (
            None
            if capacity_approval is None
            else {
                "marker": dict(capacity_approval["marker"]),
                "consumption": dict(capacity_approval["consumption"]),
            }
        ),
        "resume_probes": resume_probes,
        "output_root": str(output_root),
        "runtime": {
            "base_model_id": config["base_model_id"],
            "revision": config["revision"],
            "tokenizer_id": config["tokenizer_id"],
            "tokenizer_revision": config["tokenizer_revision"],
            "template_revision": config["template_revision"],
            "attention_implementation": config["attention_implementation"],
            "dtype": config["dtype"],
            "device": config["device"],
            "local_files_only": config["local_files_only"],
            "seed": config["seed"],
            "decode": dict(config["decode"]),
            "timeouts_seconds": dict(config["timeouts_seconds"]),
        },
        "artifacts": artifacts,
        "required_cell_count": 20,
        "sample_count_per_stratum": config["sample_count_per_stratum"],
        "cells": cells,
    }
    return {**unsigned, "plan_sha256": _canonical_sha256(unsigned)}


def publish_plan(plan: Mapping[str, Any], output_path: str | os.PathLike[str]) -> Path:
    _validate_plan(plan)
    output = Path(output_path).expanduser().resolve(strict=False)
    _atomic_write_json(output, plan)
    return output


def _validate_plan(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    expected = {
        "kind",
        "schema_version",
        "delivery_level",
        "evaluator_version",
        "source_config",
        "runtime_binding",
        "capacity_profile",
        "capacity_approval",
        "resume_probes",
        "output_root",
        "runtime",
        "artifacts",
        "required_cell_count",
        "sample_count_per_stratum",
        "cells",
        "plan_sha256",
    }
    _require_exact_keys(plan, expected, "plan")
    if (
        plan["kind"] != PLAN_KIND
        or plan["schema_version"] != PLAN_SCHEMA_VERSION
        or plan["evaluator_version"] != EVALUATOR_VERSION
    ):
        raise EvalMatrixError("unsupported evaluation plan")
    contract = _level_contract(plan["delivery_level"])
    if (
        plan["required_cell_count"] != 20
        or plan["sample_count_per_stratum"] != contract["sample_count"]
    ):
        raise EvalMatrixError("plan delivery contract changed")
    digest = _require_sha256(plan["plan_sha256"], "plan_sha256")
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if digest != _canonical_sha256(unsigned):
        raise EvalMatrixError("plan self-hash mismatch")
    _verify_file_record(plan["source_config"], "source_config")
    runtime_binding = _require_mapping(plan["runtime_binding"], "runtime_binding")
    _require_exact_keys(
        runtime_binding, {"path", "sha256", "runtime_binding_sha256"}, "runtime_binding"
    )
    _file_record(runtime_binding["path"], runtime_binding["sha256"], "runtime_binding")
    _require_sha256(runtime_binding["runtime_binding_sha256"], "runtime_binding_sha256")
    _verify_file_record(plan["capacity_profile"], "capacity_profile")
    _capacity_approval_records(plan["capacity_approval"], "capacity_approval")
    _resume_probe_records(
        plan["resume_probes"],
        "resume_probes",
        expected_names={contract["b"], contract["c"]},
    )
    artifacts = _require_mapping(plan["artifacts"], "artifacts")
    expected_artifacts = {"base", contract["b"], contract["c"]}
    _require_exact_keys(artifacts, expected_artifacts, "artifacts")
    for name, raw_artifact in artifacts.items():
        artifact = _require_mapping(raw_artifact, f"artifacts.{name}")
        _require_exact_keys(artifact, _ARTIFACT_KEYS, f"artifacts.{name}")
        if name == "base":
            if artifact["artifact_kind"] != "base" or artifact["training_step"] != 0:
                raise EvalMatrixError("plan base artifact identity changed")
        else:
            if artifact["artifact_kind"] != "adapter" or artifact["training_step"] != contract["step"]:
                raise EvalMatrixError(f"plan artifact identity changed for {name}")
            artifact_path = _absolute_path(
                artifact["artifact_path"], f"artifacts.{name}.artifact_path", exists=True
            )
            if not artifact_path.is_dir():
                raise EvalMatrixError(f"artifacts.{name}.artifact_path must be a directory")
            _file_record(
                artifact["training_config_path"],
                artifact["training_config_sha256"],
                f"artifacts.{name}.training_config",
            )
            _require_sha256(
                artifact["adapter_metadata_sha256"],
                f"artifacts.{name}.adapter_metadata_sha256",
            )
            _require_sha256(
                artifact["checkpoint_extra_state_sha256"],
                f"artifacts.{name}.checkpoint_extra_state_sha256",
            )
    cells = plan["cells"]
    if not isinstance(cells, list) or len(cells) != 20:
        raise EvalMatrixError("plan must contain exactly 20 cells")
    cell_ids = [cell.get("cell_id") for cell in cells if isinstance(cell, Mapping)]
    if len(cell_ids) != 20 or len(set(cell_ids)) != 20:
        raise EvalMatrixError("plan cell IDs are not unique")
    expected_specs = set(_cell_specs(plan["delivery_level"]))
    strata: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    output_root = _absolute_path(plan["output_root"], "output_root", exists=False)
    if plan["output_root"] != str(output_root):
        raise EvalMatrixError("plan output_root is not canonical")
    for raw_cell in cells:
        cell = _require_mapping(raw_cell, "cell")
        _require_exact_keys(cell, _CELL_KEYS, f"cell.{cell.get('cell_id', '<unknown>')}")
        key = (cell.get("dataset"), cell.get("document_count"))
        strata.setdefault(key, []).append(cell)
        expected_cell_id = (
            f"{cell.get('dataset')}-docs{cell.get('document_count')}-"
            f"{cell.get('artifact')}-{cell.get('callback_mode')}"
        )
        if cell.get("cell_id") != expected_cell_id:
            raise EvalMatrixError("cell ID does not match its coordinates")
        if cell.get("sample_count") != contract["sample_count"]:
            raise EvalMatrixError("cell sample count differs from delivery level")
        bundle_dir = _absolute_path(cell.get("bundle_dir"), "cell bundle_dir", exists=True)
        if not bundle_dir.is_dir():
            raise EvalMatrixError("cell bundle_dir must be a directory")
        expected_output = output_root / "cells" / expected_cell_id
        actual_output = _absolute_path(cell.get("output_dir"), "cell output_dir", exists=False)
        if cell.get("output_dir") != str(actual_output):
            raise EvalMatrixError("cell output path is not canonical")
        if actual_output != expected_output:
            raise EvalMatrixError("cell output path escapes its matrix identity")
        _require_sha256(cell.get("manifest_sha256"), "cell manifest_sha256")
        qa_ids = cell.get("ordered_qa_ids")
        if not isinstance(qa_ids, list) or len(qa_ids) != contract["sample_count"]:
            raise EvalMatrixError("cell QA count differs from delivery level")
        if (
            any(not isinstance(qa_id, str) or not qa_id for qa_id in qa_ids)
            or len(set(qa_ids)) != len(qa_ids)
        ):
            raise EvalMatrixError("cell ordered QA IDs must be unique non-empty strings")
        if cell.get("ordered_qa_ids_sha256") != _canonical_sha256(qa_ids):
            raise EvalMatrixError("cell ordered QA hash mismatch")
    expected_strata = {(dataset, variant) for dataset in DATASETS for variant in DOCUMENT_VARIANTS}
    if set(strata) != expected_strata:
        raise EvalMatrixError("plan strata changed")
    for key, stratum_cells in strata.items():
        if {(cell["artifact"], cell["callback_mode"]) for cell in stratum_cells} != expected_specs:
            raise EvalMatrixError(f"cell methods changed for stratum {key}")
        qa_orders = {tuple(cell["ordered_qa_ids"]) for cell in stratum_cells}
        if len(qa_orders) != 1:
            raise EvalMatrixError(f"ordered QA IDs differ across cells for stratum {key}")
    for dataset in DATASETS:
        variant_orders = {
            tuple(strata[(dataset, variant)][0]["ordered_qa_ids"])
            for variant in DOCUMENT_VARIANTS
        }
        if len(variant_orders) != 1:
            raise EvalMatrixError(
                f"{dataset} 200/800 variants do not share ordered QA IDs"
            )
    return plan


def load_plan(path: str | os.PathLike[str]) -> Mapping[str, Any]:
    plan_path = _absolute_path(path, "plan", exists=True)
    plan = _load_json(plan_path, canonical=True)
    return _validate_plan(plan)


def _artifact(plan: Mapping[str, Any], cell: Mapping[str, Any]) -> Mapping[str, Any]:
    return _require_mapping(plan["artifacts"][cell["artifact"]], "cell artifact")


def _backend_spec(plan: Mapping[str, Any], cell: Mapping[str, Any]) -> runner.ModelLoadSpec:
    runtime = plan["runtime"]
    artifact = _artifact(plan, cell)
    return runner.ModelLoadSpec(
        artifact_kind=artifact["artifact_kind"],
        base_model_id=runtime["base_model_id"],
        revision=runtime["revision"],
        artifact_path=artifact["artifact_path"],
        tokenizer_id=runtime["tokenizer_id"],
        tokenizer_revision=runtime["tokenizer_revision"],
        template_revision=runtime["template_revision"],
        attention_implementation=runtime["attention_implementation"],
        dtype=runtime["dtype"],
        device=runtime["device"],
        seed=runtime["seed"],
        local_files_only=runtime["local_files_only"],
    )


def _sample_config(plan: Mapping[str, Any], cell: Mapping[str, Any]) -> runner.SampleEvaluationConfig:
    decode = plan["runtime"]["decode"]
    return runner.SampleEvaluationConfig(
        callback_mode=cell["callback_mode"],
        chunk_size=decode["chunk_size"],
        memory_max_tokens=decode["memory_max_tokens"],
        final_max_tokens=decode["final_max_tokens"],
    )


def _cell_input_metadata(plan: Mapping[str, Any], cell: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _artifact(plan, cell)
    return {
        "formal_evaluation": True,
        "matrix_kind": PLAN_KIND,
        "delivery_level": plan["delivery_level"],
        "evaluator_version": EVALUATOR_VERSION,
        "matrix_plan_sha256": plan["plan_sha256"],
        "cell_id": cell["cell_id"],
        "evaluation_config_sha256": plan["source_config"]["sha256"],
        "runtime_binding_sha256": plan["runtime_binding"][
            "runtime_binding_sha256"
        ],
        "capacity_profile_sha256": plan["capacity_profile"]["sha256"],
        "r1_approval_marker_file_sha256": (
            None
            if plan["capacity_approval"] is None
            else plan["capacity_approval"]["marker"]["sha256"]
        ),
        "r1_approval_consumption_file_sha256": (
            None
            if plan["capacity_approval"] is None
            else plan["capacity_approval"]["consumption"]["sha256"]
        ),
        "resume_probe_sha256": (
            None
            if cell["artifact"] == "base"
            else plan["resume_probes"][cell["artifact"]]["probe_sha256"]
        ),
        "training_config_id": artifact["training_config_id"],
        "training_config_sha256": artifact["training_config_sha256"],
        "checkpoint_extra_state_sha256": artifact[
            "checkpoint_extra_state_sha256"
        ],
        "adapter_metadata_sha256": artifact["adapter_metadata_sha256"],
        "bundle_manifest_sha256": cell["manifest_sha256"],
        "bundle_path": cell["bundle_dir"],
        "dataset": cell["dataset"],
        "sample_count": cell["sample_count"],
        "variant": cell["document_count"],
        "ordered_qa_ids_sha256": cell["ordered_qa_ids_sha256"],
        "greedy_decode": dict(plan["runtime"]["decode"]),
    }


def expected_run_config(
    plan: Mapping[str, Any], cell: Mapping[str, Any]
) -> dict[str, Any]:
    backend = _backend_spec(plan, cell)
    sample = _sample_config(plan, cell)
    timeouts = plan["runtime"]["timeouts_seconds"]
    return {
        "backend": backend.public_dict(),
        "callback_mode": sample.callback_mode,
        "input": _cell_input_metadata(plan, cell),
        "kind": runner.RUN_KIND,
        "prompt_template_sha256": runner.PROMPT_TEMPLATE_SHA256,
        "prompt_template_revision": runner.PROMPT_TEMPLATE_REVISION,
        "qa_ids": list(cell["ordered_qa_ids"]),
        "sample": sample.to_dict(),
        "schema_version": runner.RUN_SCHEMA_VERSION,
        "timeouts": {
            "model_load_seconds": float(timeouts["model_load"]),
            "sample_seconds": float(timeouts["sample"]),
            "task_seconds": float(timeouts["task"]),
        },
    }


def _verify_model_metadata(
    model_metadata: Any,
    plan: Mapping[str, Any],
    cell: Mapping[str, Any],
) -> None:
    metadata = _require_mapping(model_metadata, "model_metadata")
    runtime = plan["runtime"]
    artifact = _artifact(plan, cell)
    expected = {
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
    for name, expected_value in expected.items():
        if metadata.get(name) != expected_value:
            raise EvalMatrixError(f"cell model metadata differs at {name}")
    adapter_metadata = metadata.get("adapter_metadata")
    if artifact["artifact_kind"] == "base":
        if adapter_metadata is not None or metadata.get("merged_metadata") is not None:
            raise EvalMatrixError("base cell unexpectedly loaded an artifact")
    else:
        adapter_metadata = _require_mapping(adapter_metadata, "adapter_metadata")
        expected_adapter = {
            "global_step": artifact["training_step"],
            "metadata_sha256": artifact["adapter_metadata_sha256"],
            "source_extra_state_sha256": artifact[
                "checkpoint_extra_state_sha256"
            ],
        }
        for name, expected_value in expected_adapter.items():
            if adapter_metadata.get(name) != expected_value:
                raise EvalMatrixError(f"cell adapter metadata differs at {name}")


def _cell_by_id(plan: Mapping[str, Any], cell_id: str) -> Mapping[str, Any]:
    matches = [cell for cell in plan["cells"] if cell["cell_id"] == cell_id]
    if len(matches) != 1:
        raise EvalMatrixError(f"unknown or duplicate cell ID: {cell_id}")
    return matches[0]


def _load_cell_results(
    output_dir: Path, *, expected_sha256: str | None = None
) -> list[Mapping[str, Any]]:
    payload = (output_dir / runner.RESULTS_FILENAME).read_bytes()
    if expected_sha256 is not None and _sha256_bytes(payload) != expected_sha256:
        raise EvalMatrixError("cell results hash changed during verification")
    try:
        values = [json.loads(line) for line in payload.splitlines()]
    except json.JSONDecodeError as exc:
        raise EvalMatrixError(f"cannot parse cell results: {exc}") from exc
    if runner._canonical_jsonl_bytes(values) != payload:
        raise EvalMatrixError("cell results are not canonical")
    if any(not isinstance(value, Mapping) for value in values):
        raise EvalMatrixError("cell result entries must be mappings")
    return values


def _validate_success_result(
    result: Mapping[str, Any],
    *,
    cell: Mapping[str, Any],
    expected_qa_id: str,
) -> None:
    label = f"cell {cell['cell_id']} result {expected_qa_id}"
    _require_exact_keys(result, _SUCCESS_RESULT_KEYS, label)
    if result["status"] != "success":
        raise EvalMatrixError(f"{label} is not successful")
    if result["qa_id"] != expected_qa_id:
        raise EvalMatrixError(f"{label} QA identity changed")
    if result["callback_mode"] != cell["callback_mode"]:
        raise EvalMatrixError(f"{label} callback mode changed")
    if result["document_count"] != cell["document_count"]:
        raise EvalMatrixError(f"{label} document count changed")
    if result["processed_doc_count"] != cell["document_count"]:
        raise EvalMatrixError(f"{label} did not consume the full document variant")
    if (
        result["prompt_template_revision"] != runner.PROMPT_TEMPLATE_REVISION
        or result["prompt_template_sha256"] != runner.PROMPT_TEMPLATE_SHA256
    ):
        raise EvalMatrixError(f"{label} prompt template identity changed")
    for name in (
        "context_sha256",
        "context_token_ids_sha256",
        "manifest_record_sha256",
    ):
        _require_sha256(result[name], f"{label}.{name}")
    for name in (
        "context_token_count",
        "processed_chunk_count",
        "processed_doc_count",
        "qa_index",
    ):
        value = result[name]
        minimum = 0 if name == "qa_index" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise EvalMatrixError(f"{label}.{name} must be an integer >= {minimum}")
    elapsed = result["elapsed_seconds"]
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or elapsed < 0
    ):
        raise EvalMatrixError(f"{label}.elapsed_seconds must be finite and non-negative")
    raw_output = result["raw_final_output"]
    gold_answers = result["gold_answers"]
    if not isinstance(raw_output, str):
        raise EvalMatrixError(f"{label}.raw_final_output must be a string")
    if not isinstance(gold_answers, list):
        raise EvalMatrixError(f"{label}.gold_answers must be a list")
    trajectory = result["trajectory"]
    if not isinstance(trajectory, list) or not trajectory:
        raise EvalMatrixError(f"{label}.trajectory must be a non-empty list")
    final_turn = trajectory[-1]
    if (
        not isinstance(final_turn, Mapping)
        or final_turn.get("kind") != "final"
        or final_turn.get("raw_output") != raw_output
    ):
        raise EvalMatrixError(f"{label} final trajectory output changed")

    metrics = _require_mapping(result["metrics"], f"{label}.metrics")
    _require_exact_keys(metrics, _METRIC_KEYS, f"{label}.metrics")
    answer = _require_mapping(metrics["answer"], f"{label}.metrics.answer")
    extraction = _require_mapping(
        answer.get("extraction"), f"{label}.metrics.answer.extraction"
    )
    answer_scores = _require_mapping(
        answer.get("scores"), f"{label}.metrics.answer.scores"
    )
    scores = _require_mapping(metrics["scores"], f"{label}.metrics.scores")
    callback = _require_mapping(metrics["callback"], f"{label}.metrics.callback")
    format_metrics = _require_mapping(metrics["format"], f"{label}.metrics.format")
    truncation = _require_mapping(
        metrics["truncation"], f"{label}.metrics.truncation"
    )
    _require_exact_keys(answer, _ANSWER_KEYS, f"{label}.metrics.answer")
    _require_exact_keys(
        extraction, _EXTRACTION_KEYS, f"{label}.metrics.answer.extraction"
    )
    _require_exact_keys(
        answer_scores, _SCORE_KEYS, f"{label}.metrics.answer.scores"
    )
    _require_exact_keys(scores, _SCORE_KEYS, f"{label}.metrics.scores")
    _require_exact_keys(callback, _CALLBACK_METRIC_KEYS, f"{label}.metrics.callback")
    _require_exact_keys(
        _require_mapping(
            callback["model_query_status_counts"],
            f"{label}.metrics.callback.model_query_status_counts",
        ),
        _QUERY_STATUS_KEYS,
        f"{label}.metrics.callback.model_query_status_counts",
    )
    _require_exact_keys(
        _require_mapping(
            callback["model_query_status_rates"],
            f"{label}.metrics.callback.model_query_status_rates",
        ),
        _QUERY_STATUS_KEYS,
        f"{label}.metrics.callback.model_query_status_rates",
    )
    _require_exact_keys(
        format_metrics, _FORMAT_METRIC_KEYS, f"{label}.metrics.format"
    )
    _require_exact_keys(
        truncation, _TRUNCATION_METRIC_KEYS, f"{label}.metrics.truncation"
    )
    try:
        evaluated = evaluate_output(raw_output, gold_answers)
    except (TypeError, ValueError) as exc:
        raise EvalMatrixError(f"{label} raw answer cannot be evaluated: {exc}") from exc
    expected_answer = evaluated.to_dict()
    if dict(answer) != expected_answer:
        raise EvalMatrixError(f"{label} answer metrics differ from raw prediction")
    if dict(scores) != evaluated.scores.to_dict():
        raise EvalMatrixError(f"{label} scores differ from raw prediction")
    if result["parsed_answer"] != evaluated.extraction.answer:
        raise EvalMatrixError(f"{label} parsed answer differs from raw prediction")


def verify_cell_output(
    plan: Mapping[str, Any], cell: Mapping[str, Any]
) -> dict[str, Any]:
    output_dir = _absolute_path(cell["output_dir"], "cell output", exists=True)
    if not output_dir.is_dir():
        raise EvalMatrixError("cell output must be a directory")
    if any(path.is_symlink() or not path.is_file() for path in output_dir.iterdir()):
        raise EvalMatrixError("cell output contains an unsafe entry")
    try:
        summary = runner._validate_result_directory(
            output_dir, expected_qa_ids=cell["ordered_qa_ids"]
        )
    except runner.EvaluationRunnerError as exc:
        raise EvalMatrixError(f"cell result directory is invalid: {exc}") from exc
    if (
        summary.get("status") != "completed"
        or summary.get("failure_count") != 0
        or summary.get("success_count") != cell["sample_count"]
        or summary.get("sample_count") != cell["sample_count"]
    ):
        raise EvalMatrixError(f"cell {cell['cell_id']} did not fully succeed")
    marker = runner._read_canonical_json(output_dir / runner.COMPLETION_FILENAME)
    _require_exact_keys(marker, _COMPLETION_MARKER_KEYS, "cell completion marker")
    if (
        marker["kind"] != runner.RUN_KIND
        or marker["schema_version"] != runner.RUN_SCHEMA_VERSION
        or marker["status"] != "completed"
        or marker["failure_count"] != 0
        or marker["success_count"] != cell["sample_count"]
    ):
        raise EvalMatrixError("cell completion marker contract changed")
    run_config = runner._read_canonical_json(output_dir / runner.RUN_CONFIG_FILENAME)
    expected = expected_run_config(plan, cell)
    if set(run_config) != {*expected, "model_metadata"}:
        raise EvalMatrixError("cell run config field set changed")
    for name, expected_value in expected.items():
        if run_config.get(name) != expected_value:
            raise EvalMatrixError(f"cell run config differs at {name}")
    _verify_model_metadata(run_config.get("model_metadata"), plan, cell)
    results = _load_cell_results(
        output_dir, expected_sha256=marker["results_sha256"]
    )
    result_qa_ids = [result.get("qa_id") for result in results]
    if result_qa_ids != cell["ordered_qa_ids"]:
        raise EvalMatrixError("cell result QA order changed")
    if len(set(result_qa_ids)) != len(result_qa_ids):
        raise EvalMatrixError("cell contains duplicate QA samples")
    for result, qa_id in zip(results, cell["ordered_qa_ids"]):
        _validate_success_result(result, cell=cell, expected_qa_id=qa_id)
    try:
        recomputed_summary = runner._aggregate_summary(
            results, callback_mode=cell["callback_mode"]
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise EvalMatrixError(f"cell summary cannot be recomputed: {exc}") from exc
    if dict(summary) != recomputed_summary:
        raise EvalMatrixError("cell summary differs from recomputed results")
    return {
        "cell_id": cell["cell_id"],
        "output_dir": str(output_dir),
        "ordered_qa_ids_sha256": cell["ordered_qa_ids_sha256"],
        "results_sha256": marker["results_sha256"],
        "run_config_sha256": marker["run_config_sha256"],
        "summary_sha256": marker["summary_sha256"],
        "completion_marker_sha256": marker["marker_sha256"],
        "summary": dict(summary),
    }


def execute_cell(
    plan: Mapping[str, Any],
    cell_id: str,
    *,
    record_loader: Callable[..., tuple[Sequence[Any], Mapping[str, Any]]] = runner.load_eval_records,
    evaluation_task: Callable[..., Mapping[str, Any]] = runner.run_evaluation_task,
) -> Mapping[str, Any]:
    _validate_plan(plan)
    cell = _cell_by_id(plan, cell_id)
    output_dir = Path(cell["output_dir"])
    if output_dir.exists():
        return verify_cell_output(plan, cell)
    records, manifest = record_loader(
        cell["bundle_dir"],
        expected_manifest_sha256=cell["manifest_sha256"],
        variant=cell["document_count"],
        sample_count=cell["sample_count"],
    )
    if [_record_qa_id(record) for record in records] != cell["ordered_qa_ids"]:
        raise EvalMatrixError("runtime eval inputs differ from the plan")
    if manifest.get("manifest_sha256") != cell["manifest_sha256"]:
        raise EvalMatrixError("runtime manifest differs from the plan")
    timeouts = plan["runtime"]["timeouts_seconds"]
    evaluation_task(
        records,
        output_dir=output_dir,
        backend_spec=_backend_spec(plan, cell),
        sample_config=_sample_config(plan, cell),
        model_load_timeout_s=float(timeouts["model_load"]),
        sample_timeout_s=float(timeouts["sample"]),
        task_timeout_s=float(timeouts["task"]),
        input_metadata=_cell_input_metadata(plan, cell),
    )
    return verify_cell_output(plan, cell)


def dry_run_commands(
    plan_path: str | os.PathLike[str], cell_ids: Sequence[str] | None = None
) -> list[list[str]]:
    plan_file = _absolute_path(plan_path, "plan", exists=True)
    plan = load_plan(plan_file)
    selected = list(cell_ids or [cell["cell_id"] for cell in plan["cells"]])
    for cell_id in selected:
        _cell_by_id(plan, cell_id)
    return [
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "run-cell",
            "--plan",
            str(plan_file),
            "--cell-id",
            cell_id,
        ]
        for cell_id in selected
    ]


def run_matrix(
    plan_path: str | os.PathLike[str],
    *,
    cell_ids: Sequence[str] | None = None,
    dry_run: bool = False,
) -> Mapping[str, Any]:
    plan = load_plan(plan_path)
    selected = list(cell_ids or [cell["cell_id"] for cell in plan["cells"]])
    if dry_run:
        return {
            "status": "dry-run",
            "plan_sha256": plan["plan_sha256"],
            "commands": dry_run_commands(plan_path, selected),
        }
    evidence = [execute_cell(plan, cell_id) for cell_id in selected]
    return {
        "status": "completed",
        "plan_sha256": plan["plan_sha256"],
        "cell_count": len(evidence),
        "cells": evidence,
    }


def _metric_values(results: Sequence[Mapping[str, Any]], name: str) -> list[float]:
    values = []
    for result in results:
        try:
            value = result["metrics"]["scores"][name]
        except (KeyError, TypeError) as exc:
            raise EvalMatrixError(f"result lacks metric {name}") from exc
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EvalMatrixError(f"result metric {name} is not numeric")
        values.append(float(value))
    return values


def _build_aggregate(
    plan: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
    results_by_cell: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    contract = _level_contract(plan["delivery_level"])
    cells_by_key = {
        (
            cell["dataset"],
            cell["document_count"],
            cell["artifact"],
            cell["callback_mode"],
        ): cell
        for cell in plan["cells"]
    }
    learned_artifacts = ("base", contract["b"], contract["c"])
    method_metrics: dict[str, Any] = {}
    for artifact in learned_artifacts:
        per_stratum = []
        for dataset in DATASETS:
            for variant in DOCUMENT_VARIANTS:
                cell = cells_by_key[(dataset, variant, artifact, "learned")]
                results = results_by_cell[cell["cell_id"]]
                per_stratum.append(
                    {
                        "dataset": dataset,
                        "document_count": variant,
                        "exact_match": fmean(_metric_values(results, "exact_match")),
                        "token_f1": fmean(_metric_values(results, "token_f1")),
                        "substring_exact_match": fmean(
                            _metric_values(results, "substring_exact_match")
                        ),
                    }
                )
        method_metrics[artifact] = {
            "strata": per_stratum,
            "macro": {
                metric: fmean(item[metric] for item in per_stratum)
                for metric in (
                    "exact_match",
                    "token_f1",
                    "substring_exact_match",
                )
            },
        }

    primary_strata = []
    primary_inputs = []
    for dataset in DATASETS:
        for variant in DOCUMENT_VARIANTS:
            b_cell = cells_by_key[(dataset, variant, contract["b"], "learned")]
            c_cell = cells_by_key[(dataset, variant, contract["c"], "learned")]
            b_values = _metric_values(results_by_cell[b_cell["cell_id"]], "exact_match")
            c_values = _metric_values(results_by_cell[c_cell["cell_id"]], "exact_match")
            primary_inputs.append((b_values, c_values))
            primary_strata.append(
                {
                    "dataset": dataset,
                    "document_count": variant,
                    "sample_count": len(b_values),
                    "observed_delta": fmean(
                        candidate - baseline
                        for baseline, candidate in zip(b_values, c_values)
                    ),
                }
            )
    bootstrap = stratified_paired_bootstrap_delta(
        primary_inputs, resamples=bootstrap_resamples, seed=42
    )
    payload: dict[str, Any] = {
        "kind": AGGREGATE_KIND,
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "status": "verified",
        "delivery_level": plan["delivery_level"],
        "plan_sha256": plan["plan_sha256"],
        "cell_count": len(evidence),
        "sample_count_per_stratum": plan["sample_count_per_stratum"],
        "methods": method_metrics,
        "primary": {
            "contrast": f"{contract['c']}-{contract['b']}",
            "metric": "exact_match",
            "stratum_weighting": "equal",
            "strata": primary_strata,
            "paired_bootstrap": bootstrap.to_dict(),
        },
    }
    return {**payload, "aggregate_sha256": _canonical_sha256(payload)}


def _write_canonical(path: Path, value: Mapping[str, Any]) -> str:
    payload = canonical_json_bytes(value) + b"\n"
    path.write_bytes(payload)
    return _sha256_bytes(payload)


def _validate_self_hash(
    value: Mapping[str, Any], hash_field: str, label: str
) -> str:
    digest = _require_sha256(value.get(hash_field), f"{label}.{hash_field}")
    unsigned = {key: item for key, item in value.items() if key != hash_field}
    if digest != _canonical_sha256(unsigned):
        raise EvalMatrixError(f"{label} self-hash mismatch")
    return digest


def _collect_verified_cell_data(
    plan: Mapping[str, Any],
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, Sequence[Mapping[str, Any]]],
]:
    evidence: dict[str, Mapping[str, Any]] = {}
    results_by_cell: dict[str, Sequence[Mapping[str, Any]]] = {}
    for cell in plan["cells"]:
        cell_evidence = verify_cell_output(plan, cell)
        cell_id = cell["cell_id"]
        evidence[cell_id] = cell_evidence
        results_by_cell[cell_id] = _load_cell_results(
            Path(cell["output_dir"]),
            expected_sha256=cell_evidence["results_sha256"],
        )
    if len(evidence) != plan["required_cell_count"]:
        raise EvalMatrixError("all 20 cells must verify")
    return evidence, results_by_cell


def _build_package(
    plan: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
    aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": PACKAGE_KIND,
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "status": "verified",
        "delivery_level": plan["delivery_level"],
        "plan_sha256": plan["plan_sha256"],
        "evaluation_config": dict(plan["source_config"]),
        "runtime_binding": dict(plan["runtime_binding"]),
        "capacity_profile": dict(plan["capacity_profile"]),
        "capacity_approval": (
            None
            if plan["capacity_approval"] is None
            else {
                "marker": dict(plan["capacity_approval"]["marker"]),
                "consumption": dict(plan["capacity_approval"]["consumption"]),
            }
        ),
        "resume_probes": {
            name: dict(record) for name, record in plan["resume_probes"].items()
        },
        "artifacts": dict(plan["artifacts"]),
        "cell_count": plan["required_cell_count"],
        "cells": [evidence[cell["cell_id"]] for cell in plan["cells"]],
        "aggregate_sha256": aggregate["aggregate_sha256"],
    }
    return {**payload, "package_sha256": _canonical_sha256(payload)}


def _validate_package_directory(
    directory: Path,
    *,
    plan: Mapping[str, Any],
) -> Mapping[str, Any]:
    _validate_plan(plan)
    if directory.is_symlink() or not directory.is_dir():
        raise EvalMatrixError("verified package directory is missing or unsafe")
    files = {path.name for path in directory.iterdir() if path.is_file()}
    if files != {AGGREGATE_FILENAME, PACKAGE_FILENAME, VERIFIED_FILENAME}:
        raise EvalMatrixError("verified package file inventory mismatch")
    if any(path.is_symlink() or not path.is_file() for path in directory.iterdir()):
        raise EvalMatrixError("verified package contains an unsafe entry")
    aggregate = runner._read_canonical_json(directory / AGGREGATE_FILENAME)
    package = runner._read_canonical_json(directory / PACKAGE_FILENAME)
    marker = runner._read_canonical_json(directory / VERIFIED_FILENAME)
    _require_exact_keys(aggregate, _AGGREGATE_KEYS, "aggregate")
    _require_exact_keys(package, _PACKAGE_KEYS, "package")
    _require_exact_keys(marker, _VERIFIED_MARKER_KEYS, "verified marker")
    aggregate_sha = _validate_self_hash(aggregate, "aggregate_sha256", "aggregate")
    package_sha = _validate_self_hash(package, "package_sha256", "package")
    marker_sha = _validate_self_hash(marker, "marker_sha256", "marker")
    if (
        aggregate.get("kind") != AGGREGATE_KIND
        or package.get("kind") != PACKAGE_KIND
        or marker.get("kind") != PACKAGE_KIND
        or aggregate.get("schema_version") != PACKAGE_SCHEMA_VERSION
        or package.get("schema_version") != PACKAGE_SCHEMA_VERSION
        or marker.get("schema_version") != PACKAGE_SCHEMA_VERSION
        or aggregate.get("status") != "verified"
        or package.get("status") != "verified"
        or marker.get("status") != "verified"
        or aggregate.get("cell_count") != plan["required_cell_count"]
        or package.get("cell_count") != plan["required_cell_count"]
        or marker.get("cell_count") != plan["required_cell_count"]
    ):
        raise EvalMatrixError("verified package contract changed")
    if (
        aggregate.get("plan_sha256") != plan["plan_sha256"]
        or package.get("plan_sha256") != plan["plan_sha256"]
        or marker.get("plan_sha256") != plan["plan_sha256"]
    ):
        raise EvalMatrixError("verified package points at a different plan")
    if package.get("aggregate_sha256") != aggregate_sha:
        raise EvalMatrixError("package does not bind aggregate identity")
    for name, expected in (
        ("delivery_level", plan["delivery_level"]),
        ("evaluation_config", plan["source_config"]),
        ("runtime_binding", plan["runtime_binding"]),
        ("capacity_profile", plan["capacity_profile"]),
        ("capacity_approval", plan["capacity_approval"]),
        ("resume_probes", plan["resume_probes"]),
        ("artifacts", plan["artifacts"]),
    ):
        if package.get(name) != expected:
            raise EvalMatrixError(f"package {name} differs from the plan")

    evidence, results_by_cell = _collect_verified_cell_data(plan)
    recomputed_aggregate = _build_aggregate(
        plan,
        evidence,
        results_by_cell,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
    )
    if dict(aggregate) != recomputed_aggregate:
        raise EvalMatrixError("aggregate differs from independently recomputed cells")
    recomputed_package = _build_package(plan, evidence, recomputed_aggregate)
    if dict(package) != recomputed_package:
        raise EvalMatrixError("package differs from independently verified cells")

    marker_payload = {
        "kind": PACKAGE_KIND,
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "status": "verified",
        "plan_sha256": plan["plan_sha256"],
        "cell_count": plan["required_cell_count"],
        "aggregate_sha256": recomputed_aggregate["aggregate_sha256"],
        "package_sha256": recomputed_package["package_sha256"],
        "aggregate_file_sha256": _sha256_file(directory / AGGREGATE_FILENAME),
        "package_file_sha256": _sha256_file(directory / PACKAGE_FILENAME),
    }
    expected_marker = {
        **marker_payload,
        "marker_sha256": _canonical_sha256(marker_payload),
    }
    if dict(marker) != expected_marker:
        raise EvalMatrixError("verified marker identity mismatch")
    return {
        "aggregate_sha256": recomputed_aggregate["aggregate_sha256"],
        "package_sha256": recomputed_package["package_sha256"],
        "marker_sha256": marker_sha,
        "status": "verified",
    }


def publish_verified_package(
    plan_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
) -> Mapping[str, Any]:
    plan = load_plan(plan_path)
    destination = Path(output_dir).expanduser().resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite verified package {destination}")
    evidence, results_by_cell = _collect_verified_cell_data(plan)
    aggregate = _build_aggregate(
        plan,
        evidence,
        results_by_cell,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
    )
    package = _build_package(plan, evidence, aggregate)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-", dir=destination.parent
        )
    )
    try:
        aggregate_file_sha = _write_canonical(staging / AGGREGATE_FILENAME, aggregate)
        package_file_sha = _write_canonical(staging / PACKAGE_FILENAME, package)
        marker_payload = {
            "kind": PACKAGE_KIND,
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "status": "verified",
            "plan_sha256": plan["plan_sha256"],
            "cell_count": plan["required_cell_count"],
            "aggregate_sha256": aggregate["aggregate_sha256"],
            "package_sha256": package["package_sha256"],
            "aggregate_file_sha256": aggregate_file_sha,
            "package_file_sha256": package_file_sha,
        }
        marker = {
            **marker_payload,
            "marker_sha256": _canonical_sha256(marker_payload),
        }
        _write_canonical(staging / VERIFIED_FILENAME, marker)
        _validate_package_directory(
            staging,
            plan=plan,
        )
        os.replace(staging, destination)
        _sync_parent(destination)
        return _validate_package_directory(
            destination,
            plan=plan,
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def verify_package(
    plan_path: str | os.PathLike[str], package_dir: str | os.PathLike[str]
) -> Mapping[str, Any]:
    plan = load_plan(plan_path)
    directory = _absolute_path(package_dir, "package directory", exists=True)
    return _validate_package_directory(
        directory,
        plan=plan,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    bind = subparsers.add_parser(
        "bind", help="derive a self-hashed runtime binding from immutable attempts"
    )
    bind.add_argument("--index", type=Path, required=True)
    bind.add_argument("--eval-config-id", required=True)
    bind.add_argument("--capacity-profile", type=Path, required=True)
    bind.add_argument("--b-attempt", type=Path, required=True)
    bind.add_argument("--c-attempt", type=Path, required=True)
    bind.add_argument("--b-resume-probe", type=Path, required=True)
    bind.add_argument("--c-resume-probe", type=Path, required=True)
    bind.add_argument("--output", type=Path, required=True)
    bind.add_argument("--r1-approval-marker", type=Path)
    bind.add_argument("--r1-approval-consumption", type=Path)
    bind.add_argument("--r0-capacity-evidence", type=Path)
    bind.add_argument("--r0-terminal-sha256")
    bind.add_argument("--budget-projection-sha256")

    plan = subparsers.add_parser("plan")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--bindings", type=Path, required=True)
    plan.add_argument("--output-root", type=Path, required=True)
    plan.add_argument("--output", type=Path)
    plan.add_argument("--dry-run", action="store_true")

    run = subparsers.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--cell-id", action="append", default=[])
    run.add_argument("--dry-run", action="store_true")

    run_cell = subparsers.add_parser("run-cell")
    run_cell.add_argument("--plan", type=Path, required=True)
    run_cell.add_argument("--cell-id", required=True)

    package = subparsers.add_parser("package")
    package.add_argument("--plan", type=Path, required=True)
    package.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify-package")
    verify.add_argument("--plan", type=Path, required=True)
    verify.add_argument("--package-dir", type=Path, required=True)
    return parser


def _print_json(value: Mapping[str, Any]) -> None:
    print(canonical_json_bytes(value).decode("ascii"))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "bind":
            value = build_runtime_binding(
                args.index,
                args.eval_config_id,
                args.capacity_profile,
                args.b_attempt,
                args.c_attempt,
                b_resume_probe_path=args.b_resume_probe,
                c_resume_probe_path=args.c_resume_probe,
                approval_marker_path=args.r1_approval_marker,
                approval_consumption_path=args.r1_approval_consumption,
                r0_capacity_evidence_path=args.r0_capacity_evidence,
                r0_terminal_sha256=args.r0_terminal_sha256,
                budget_projection_sha256=args.budget_projection_sha256,
            )
            publish_runtime_binding(value, args.output)
            result: Mapping[str, Any] = value
        elif args.command == "plan":
            value = build_plan(
                args.config,
                args.bindings,
                output_root=args.output_root,
            )
            if not args.dry_run:
                if args.output is None:
                    raise EvalMatrixError("plan --output is required without --dry-run")
                publish_plan(value, args.output)
            result = value
        elif args.command == "run":
            result = run_matrix(
                args.plan,
                cell_ids=args.cell_id or None,
                dry_run=args.dry_run,
            )
        elif args.command == "run-cell":
            result = execute_cell(load_plan(args.plan), args.cell_id)
        elif args.command == "package":
            result = publish_verified_package(args.plan, args.output)
        else:
            result = verify_package(args.plan, args.package_dir)
    except Exception as exc:
        print(
            json.dumps(
                {"error": f"{type(exc).__name__}: {exc}", "status": "blocked"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    _print_json(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EvalMatrixError",
    "build_runtime_binding",
    "build_plan",
    "dry_run_commands",
    "execute_cell",
    "expected_run_config",
    "load_plan",
    "publish_plan",
    "publish_runtime_binding",
    "publish_verified_package",
    "run_matrix",
    "verify_cell_output",
    "verify_package",
]
