"""Deserialize and bind endpoint resume state without applying an update."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

KIND = "rememr1-resume-endpoint-probe-v1"
SCHEMA_VERSION = 1
EVIDENCE_FILENAME = "resume-endpoint-probe.json"
CHECKPOINT_EXTRA_FILENAME = "reproduction_extra_state.json"
COMPLETION_MARKER_FILENAME = "_REPRODUCTION_COMPLETE.json"
ADAPTER_METADATA_FILENAME = "reproduction_adapter_metadata.json"
BASE_MODEL = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ENDPOINTS = {
    f"{arm}{step}_qwen35_2b_5090_{profile}": {
        "endpoint": f"{arm}{step}",
        "global_step": step,
        "offload_profile": profile,
        "predecessor_step": 20 if step == 40 else 60,
    }
    for arm in ("b", "c")
    for step in (40, 80)
    for profile in ("r0", "r1")
}
_EVIDENCE_KEYS = {
    "adapter",
    "attempt_id",
    "attempt_root",
    "checkpoint",
    "config_id",
    "endpoint",
    "global_step",
    "kind",
    "loaded_state_files",
    "offload_profile",
    "probe_mode",
    "probe_sha256",
    "resume_predecessor",
    "runtime_binding",
    "runtime_bound_evidence",
    "schema_version",
    "status",
    "updates_applied",
}
_FILE_RECORD_KEYS = {"file_sha256", "path"}
_LOADED_FILE_KEYS = {"file_sha256", "kind", "path", "rank"}
_CHECKPOINT_MODULE: Any | None = None


class ResumeProbeError(RuntimeError):
    """Raised when an endpoint cannot prove safe full-state resume."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResumeProbeError(f"{label} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ResumeProbeError(f"{label} keys differ")


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ResumeProbeError(f"{label} must be a lowercase SHA-256")
    return value


def _safe_path(
    value: str | os.PathLike[str],
    label: str,
    *,
    exists: bool,
) -> Path:
    candidate = Path(os.path.abspath(Path(value).expanduser()))
    for component in [*reversed(candidate.parents), candidate]:
        if component.is_symlink():
            raise ResumeProbeError(f"{label} contains a symlink: {component}")
    try:
        return candidate.resolve(strict=exists)
    except OSError as exc:
        raise ResumeProbeError(f"cannot resolve {label}: {exc}") from exc


def _file_record(path: Path) -> dict[str, str]:
    resolved = _safe_path(path, "probe source file", exists=True)
    if not resolved.is_file():
        raise ResumeProbeError(f"probe source must be a file: {resolved}")
    return {"path": str(resolved), "file_sha256": _sha256_file(resolved)}


def _field(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ResumeProbeError(f"resolved config lacks {dotted_path}")
        current = current[part]
    return current


def _successful_attempt(value: str | os.PathLike[str]) -> Path:
    attempt = _safe_path(value, "endpoint attempt", exists=True)
    if not attempt.is_dir():
        raise ResumeProbeError("endpoint attempt must be a directory")
    success = _safe_path(attempt / ".success", "endpoint success", exists=True)
    if not success.is_file() or success.read_bytes() != b"0\n":
        raise ResumeProbeError("endpoint attempt lacks canonical success state")
    for marker in (".running", ".failed", ".scientific-stop", ".capacity-stop"):
        candidate = attempt / marker
        if candidate.exists() or candidate.is_symlink():
            raise ResumeProbeError("endpoint attempt has conflicting terminal state")
    return attempt


def _checkpoint_module() -> Any:
    global _CHECKPOINT_MODULE
    if _CHECKPOINT_MODULE is None:
        module_name = "_rememr1_resume_probe_checkpoint_contract"
        path = (
            REPOSITORY_ROOT
            / "verl"
            / "utils"
            / "checkpoint"
            / "reproduction.py"
        )
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ResumeProbeError("cannot load checkpoint contract module")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        _CHECKPOINT_MODULE = module
    return _CHECKPOINT_MODULE


def _default_runtime_verifier(path: Path) -> Mapping[str, Any]:
    from scripts.cloud.run_resolved_training import verify_bound_config

    return verify_bound_config(path)


def _default_checkpoint_verifier(path: Path) -> tuple[Any, Any]:
    return _checkpoint_module().verify_reproduction_checkpoint_directory(path)


def _default_adapter_validator(path: Path) -> Any:
    return _checkpoint_module().validate_adapter_export(path)


def _default_adapter_evidence_verifier(path: Path) -> tuple[Any, Any]:
    contract = _checkpoint_module()
    manifest = contract.verify_complete_directory(path)
    metadata = contract.AdapterExportMetadata.load(
        path / ADAPTER_METADATA_FILENAME
    )
    expected_metadata = {
        "artifact_type": metadata.ARTIFACT_TYPE,
        "metadata_sha256": metadata.sha256,
    }
    if manifest.metadata != expected_metadata:
        raise ResumeProbeError("adapter completion metadata differs")
    return manifest, metadata


def _default_state_loader(
    path: Path, *, map_location: str, weights_only: bool
) -> Any:
    import torch

    return torch.load(
        path,
        map_location=map_location,
        weights_only=weights_only,
    )


def _validate_worker_metadata(
    value: Any,
    *,
    global_step: int,
    adapter_tensor_keys: Sequence[str],
    adapter_state_sha256: str,
) -> list[Mapping[str, Any]]:
    contract = _checkpoint_module()
    if not isinstance(value, list) or not value:
        raise ResumeProbeError("checkpoint actor worker metadata is missing")
    expected_keys = {
        "adapter_state_sha256",
        "adapter_tensor_keys",
        "global_step",
        "lr_scheduler_sha256",
        "rank",
        "rng_state",
        "rng_state_sha256",
        "schema_version",
        "world_size",
    }
    ordered = sorted(value, key=lambda item: item.get("rank", -1))
    world_size = len(ordered)
    if [item.get("rank") for item in ordered] != list(range(world_size)):
        raise ResumeProbeError("checkpoint actor ranks are not contiguous")
    for item in ordered:
        item = _mapping(item, "checkpoint actor worker metadata")
        _exact_keys(item, expected_keys, "checkpoint actor worker metadata")
        if (
            item["schema_version"] != 1
            or item["world_size"] != world_size
            or item["global_step"] != global_step
        ):
            raise ResumeProbeError("checkpoint actor worker identity differs")
        if item["adapter_tensor_keys"] != list(adapter_tensor_keys):
            raise ResumeProbeError("checkpoint actor adapter tensor keys differ")
        if item["adapter_state_sha256"] != adapter_state_sha256:
            raise ResumeProbeError("checkpoint actor adapter state differs")
        if (
            contract.canonical_json_sha256(item["rng_state"])
            != _sha256(item["rng_state_sha256"], "actor RNG state SHA-256")
        ):
            raise ResumeProbeError("checkpoint actor JSON-safe RNG hash differs")
        _sha256(item["lr_scheduler_sha256"], "actor scheduler SHA-256")
    return ordered


def _manifest_records(manifest: Any) -> tuple[dict[str, str], str]:
    records: dict[str, str] = {}
    for record in getattr(manifest, "files", ()):
        relative = getattr(record, "path", None)
        digest = getattr(record, "sha256", None)
        if not isinstance(relative, str) or not relative:
            raise ResumeProbeError("checkpoint manifest has an invalid file path")
        records[relative] = _sha256(digest, "checkpoint manifest file SHA-256")
    if not records or len(records) != len(getattr(manifest, "files", ())):
        raise ResumeProbeError("checkpoint manifest file inventory is invalid")
    return records, _sha256(getattr(manifest, "sha256", None), "manifest SHA-256")


def _manifest_loaded_record(
    checkpoint: Path,
    manifest_records: Mapping[str, str],
    *,
    relative: str,
    kind: str,
    rank: int | None,
) -> tuple[Path, dict[str, Any]]:
    if relative not in manifest_records:
        raise ResumeProbeError(f"checkpoint manifest is missing {relative}")
    path = _safe_path(checkpoint / relative, f"{kind} state", exists=True)
    try:
        path.relative_to(checkpoint)
    except ValueError as exc:
        raise ResumeProbeError(f"{kind} state escaped the checkpoint") from exc
    if not path.is_file() or _sha256_file(path) != manifest_records[relative]:
        raise ResumeProbeError(f"{kind} state differs from checkpoint manifest")
    return path, {
        "kind": kind,
        "rank": rank,
        "path": str(path),
        "file_sha256": manifest_records[relative],
    }


def build_resume_endpoint_probe(
    attempt_dir: str | os.PathLike[str],
    *,
    runtime_verifier: Callable[[Path], Mapping[str, Any]] | None = None,
    checkpoint_verifier: Callable[[Path], tuple[Any, Any]] | None = None,
    adapter_validator: Callable[[Path], Any] | None = None,
    state_loader: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    from scripts.cloud import run_resolved_training as runtime

    contract_module = _checkpoint_module()
    attempt = _successful_attempt(attempt_dir)
    runtime_dir = attempt / "runtime-bound"
    runtime_evidence = _mapping(
        (runtime_verifier or _default_runtime_verifier)(runtime_dir),
        "runtime-bound evidence",
    )
    config_id = runtime_evidence.get("config_id")
    if config_id not in _ENDPOINTS:
        raise ResumeProbeError("attempt is not a B40/C40/B80/C80 endpoint")
    contract = _ENDPOINTS[config_id]
    if (
        runtime_evidence.get("attempt_id") != attempt.name
        or runtime_evidence.get("attempt_root") != str(attempt)
    ):
        raise ResumeProbeError("runtime-bound attempt identity differs")

    runtime_evidence_path = _safe_path(
        runtime_dir / runtime.EVIDENCE_FILENAME,
        "runtime-bound evidence file",
        exists=True,
    )
    binding_path, binding = runtime._load_binding(
        attempt / runtime.BINDING_SOURCE_FILENAME
    )
    source_binding = _mapping(
        runtime_evidence.get("source_binding"), "runtime source binding"
    )
    if source_binding != {
        "path": str(binding_path),
        "file_sha256": _sha256_file(binding_path),
        "binding_sha256": binding["binding_sha256"],
    }:
        raise ResumeProbeError("runtime binding identity differs")
    predecessor = _mapping(
        runtime_evidence.get("resume_predecessor"), "resume predecessor"
    )
    if predecessor.get("checkpoint_step") != contract["predecessor_step"]:
        raise ResumeProbeError("resume predecessor step differs from endpoint")

    step = contract["global_step"]
    checkpoint = _safe_path(
        attempt / "checkpoints" / f"global_step_{step}",
        "endpoint checkpoint",
        exists=True,
    )
    manifest, state = (checkpoint_verifier or _default_checkpoint_verifier)(
        checkpoint
    )
    records, manifest_sha256 = _manifest_records(manifest)
    if getattr(state, "global_step", None) != step:
        raise ResumeProbeError("root extra-state global_step differs")
    if getattr(state, "base_model_id", None) != BASE_MODEL:
        raise ResumeProbeError("root extra-state base model differs")
    if getattr(state, "base_model_revision", None) != MODEL_REVISION:
        raise ResumeProbeError("root extra-state model revision differs")
    if (
        getattr(state, "resolved_config_sha256", None)
        != runtime_evidence.get("runtime_bound_config_sha256")
    ):
        raise ResumeProbeError("root extra-state runtime config hash differs")
    resolved_config = _mapping(
        getattr(state, "resolved_config", None), "checkpoint resolved config"
    )
    expected_config_values = {
        "reproduction.offload_profile": contract["offload_profile"],
        "reproduction.runtime_attempt_id": attempt.name,
        "reproduction.runtime_binding_sha256": binding["binding_sha256"],
        "reproduction.runtime_bound_evidence_path": str(runtime_evidence_path),
        "reproduction.sealed_config_id": config_id,
        "reproduction.sealed_config_sha256": binding["resolved_config_sha256"],
        "trainer.default_local_dir": str(attempt / "checkpoints"),
        "trainer.resume_from_path": predecessor["checkpoint_dir"],
        "trainer.resume_mode": "resume_path",
        "trainer.total_training_steps": step,
    }
    for dotted_path, expected in expected_config_values.items():
        if _field(resolved_config, dotted_path) != expected:
            raise ResumeProbeError(
                f"checkpoint resolved config differs at {dotted_path}"
            )

    adapter_path = _safe_path(
        attempt / "artifacts" / "adapter" / f"global_step_{step}" / "adapter",
        "endpoint adapter",
        exists=True,
    )
    adapter = (adapter_validator or _default_adapter_validator)(adapter_path)
    adapter_expectations = {
        "adapter_state_sha256": getattr(state, "adapter_state_sha256", None),
        "adapter_tensor_keys": getattr(state, "adapter_tensor_keys", None),
        "base_model_id": BASE_MODEL,
        "base_model_revision": MODEL_REVISION,
        "global_step": step,
        "lora_config_sha256": getattr(state, "lora_config_sha256", None),
        "lora_target_sha256": getattr(state, "lora_target_sha256", None),
        "source_extra_state_sha256": getattr(state, "sha256", None),
        "template_revision": "rememr1-template-v1",
        "text_mapping_sha256": getattr(state, "text_mapping_sha256", None),
        "tokenizer_id": BASE_MODEL,
        "tokenizer_revision": MODEL_REVISION,
    }
    for field, expected in adapter_expectations.items():
        actual = getattr(adapter, field, None)
        if field == "adapter_tensor_keys":
            actual, expected = tuple(actual or ()), tuple(expected or ())
        if actual != expected:
            raise ResumeProbeError(f"adapter differs from checkpoint at {field}")
    adapter_sha256 = _sha256(getattr(adapter, "sha256", None), "adapter SHA-256")

    loader = state_loader or _default_state_loader
    loaded_files: list[dict[str, Any]] = []
    dataloader_path, record = _manifest_loaded_record(
        checkpoint,
        records,
        relative="data.pt",
        kind="dataloader",
        rank=None,
    )
    loaded_files.append(record)
    driver_path, record = _manifest_loaded_record(
        checkpoint,
        records,
        relative="driver_rng.pt",
        kind="driver_rng",
        rank=None,
    )
    loaded_files.append(record)
    dataloader_state = loader(
        dataloader_path, map_location="cpu", weights_only=False
    )
    driver_rng = loader(driver_path, map_location="cpu", weights_only=False)
    if (
        contract_module.json_safe_state_sha256(dataloader_state)
        != state.dataloader_state.state_sha256
    ):
        raise ResumeProbeError("dataloader state hash differs")

    rng_metadata = _mapping(state.rng_state, "root RNG metadata")
    _exact_keys(
        rng_metadata,
        {"schema_version", "driver", "actor_workers"},
        "root RNG metadata",
    )
    if rng_metadata["schema_version"] != 1:
        raise ResumeProbeError("root RNG metadata schema is unsupported")
    driver_metadata = _mapping(rng_metadata["driver"], "driver RNG metadata")
    _exact_keys(
        driver_metadata,
        {"rng_state", "rng_state_sha256"},
        "driver RNG metadata",
    )
    driver_json = contract_module.to_json_safe_state(driver_rng)
    if (
        driver_json != driver_metadata["rng_state"]
        or contract_module.canonical_json_sha256(driver_json)
        != driver_metadata["rng_state_sha256"]
    ):
        raise ResumeProbeError("driver RNG state differs")

    workers = _validate_worker_metadata(
        rng_metadata["actor_workers"],
        global_step=step,
        adapter_tensor_keys=state.adapter_tensor_keys,
        adapter_state_sha256=state.adapter_state_sha256,
    )
    world_size = len(workers)
    for worker in workers:
        rank = worker["rank"]
        loaded: dict[str, Any] = {}
        for kind, basename in (
            ("actor_model", "model"),
            ("actor_optimizer", "optim"),
            ("actor_rank_extra", "extra_state"),
        ):
            relative = (
                f"actor/{basename}_world_size_{world_size}_rank_{rank}.pt"
            )
            path, record = _manifest_loaded_record(
                checkpoint,
                records,
                relative=relative,
                kind=kind,
                rank=rank,
            )
            loaded_files.append(record)
            loaded[kind] = loader(path, map_location="cpu", weights_only=False)
        if not isinstance(loaded["actor_model"], Mapping) or not loaded[
            "actor_model"
        ]:
            raise ResumeProbeError(f"actor rank {rank} model shard is unreadable")
        optimizer = loaded["actor_optimizer"]
        if not isinstance(optimizer, Mapping) or not optimizer:
            raise ResumeProbeError(
                f"actor rank {rank} optimizer shard is unreadable"
            )
        raw_extra = loaded["actor_rank_extra"]
        contract_module.validate_reproduction_rank_extra_state(
            raw_extra,
            expected_global_step=step,
        )
        contract_module.validate_scheduler_optimizer_alignment(
            raw_extra["lr_scheduler"],
            optimizer,
            expected_global_step=step,
        )
        if (
            contract_module.json_safe_state_sha256(raw_extra["rng"])
            != worker["rng_state_sha256"]
        ):
            raise ResumeProbeError(f"actor rank {rank} RNG state differs")
        if (
            contract_module.json_safe_state_sha256(
                raw_extra["lr_scheduler"]
            )
            != worker["lr_scheduler_sha256"]
        ):
            raise ResumeProbeError(f"actor rank {rank} scheduler state differs")

    root_extra_path = checkpoint / CHECKPOINT_EXTRA_FILENAME
    completion_path = checkpoint / COMPLETION_MARKER_FILENAME
    adapter_metadata_path = (
        adapter_path / ADAPTER_METADATA_FILENAME
    )
    unsigned = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "status": "verified",
        "probe_mode": "deserialize-only",
        "updates_applied": 0,
        "endpoint": contract["endpoint"],
        "config_id": config_id,
        "attempt_id": attempt.name,
        "attempt_root": str(attempt),
        "global_step": step,
        "offload_profile": contract["offload_profile"],
        "runtime_bound_evidence": {
            **_file_record(runtime_evidence_path),
            "evidence_sha256": runtime_evidence["evidence_sha256"],
        },
        "runtime_binding": {
            **_file_record(binding_path),
            "binding_sha256": binding["binding_sha256"],
        },
        "resume_predecessor": dict(predecessor),
        "checkpoint": {
            "path": str(checkpoint),
            "manifest_sha256": manifest_sha256,
            "completion_marker": _file_record(completion_path),
            "root_extra_state": {
                **_file_record(root_extra_path),
                "extra_state_sha256": state.sha256,
            },
        },
        "adapter": {
            "path": str(adapter_path),
            "metadata": _file_record(adapter_metadata_path),
            "metadata_sha256": adapter_sha256,
        },
        "loaded_state_files": loaded_files,
    }
    return {**unsigned, "probe_sha256": _canonical_sha256(unsigned)}


def _verify_file_record(value: Any, label: str) -> Path:
    record = _mapping(value, label)
    _exact_keys(record, _FILE_RECORD_KEYS, label)
    path = _safe_path(record["path"], f"{label}.path", exists=True)
    if (
        record["path"] != str(path)
        or not path.is_file()
        or _sha256_file(path) != _sha256(record["file_sha256"], label)
    ):
        raise ResumeProbeError(f"{label} changed")
    return path


def verify_resume_endpoint_probe(
    evidence_path: str | os.PathLike[str],
    *,
    runtime_verifier: Callable[[Path], Mapping[str, Any]] | None = None,
    checkpoint_verifier: Callable[[Path], tuple[Any, Any]] | None = None,
    adapter_evidence_verifier: Callable[[Path], tuple[Any, Any]] | None = None,
) -> Mapping[str, Any]:
    path = _safe_path(evidence_path, "resume endpoint evidence", exists=True)
    if not path.is_file():
        raise ResumeProbeError("resume endpoint evidence must be a file")
    payload = path.read_bytes()
    try:
        evidence = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ResumeProbeError(f"cannot parse resume endpoint evidence: {exc}") from exc
    evidence = _mapping(evidence, "resume endpoint evidence")
    if _canonical_bytes(evidence) + b"\n" != payload:
        raise ResumeProbeError("resume endpoint evidence is not canonical JSON")
    _exact_keys(evidence, _EVIDENCE_KEYS, "resume endpoint evidence")
    if (
        evidence["kind"] != KIND
        or evidence["schema_version"] != SCHEMA_VERSION
        or evidence["status"] != "verified"
        or evidence["probe_mode"] != "deserialize-only"
        or evidence["updates_applied"] != 0
    ):
        raise ResumeProbeError("resume endpoint evidence contract changed")
    digest = _sha256(evidence["probe_sha256"], "probe_sha256")
    unsigned = {key: value for key, value in evidence.items() if key != "probe_sha256"}
    if digest != _canonical_sha256(unsigned):
        raise ResumeProbeError("resume endpoint evidence self-hash mismatch")
    config_id = evidence["config_id"]
    if config_id not in _ENDPOINTS:
        raise ResumeProbeError("resume endpoint config is not allowlisted")
    contract = _ENDPOINTS[config_id]
    attempt = _safe_path(evidence["attempt_root"], "endpoint attempt", exists=True)
    if (
        evidence["attempt_root"] != str(attempt)
        or evidence["attempt_id"] != attempt.name
        or evidence["endpoint"] != contract["endpoint"]
        or evidence["global_step"] != contract["global_step"]
        or evidence["offload_profile"] != contract["offload_profile"]
    ):
        raise ResumeProbeError("resume endpoint evidence identity changed")

    runtime_record = _mapping(
        evidence["runtime_bound_evidence"], "runtime_bound_evidence"
    )
    _exact_keys(
        runtime_record,
        {"path", "file_sha256", "evidence_sha256"},
        "runtime_bound_evidence",
    )
    runtime_path = _verify_file_record(
        {key: runtime_record[key] for key in _FILE_RECORD_KEYS},
        "runtime_bound_evidence",
    )
    _sha256(runtime_record["evidence_sha256"], "runtime evidence SHA-256")
    if runtime_path != attempt / "runtime-bound" / "runtime-bound.json":
        raise ResumeProbeError("runtime-bound evidence path changed")

    binding_record = _mapping(evidence["runtime_binding"], "runtime_binding")
    _exact_keys(
        binding_record,
        {"path", "file_sha256", "binding_sha256"},
        "runtime_binding",
    )
    binding_path = _verify_file_record(
        {key: binding_record[key] for key in _FILE_RECORD_KEYS},
        "runtime_binding",
    )
    _sha256(binding_record["binding_sha256"], "runtime binding SHA-256")
    if binding_path != attempt / "runtime-binding-source.json":
        raise ResumeProbeError("runtime binding path changed")

    current_runtime = _mapping(
        (runtime_verifier or _default_runtime_verifier)(
            attempt / "runtime-bound"
        ),
        "reverified runtime-bound evidence",
    )
    if (
        current_runtime.get("evidence_sha256")
        != runtime_record["evidence_sha256"]
        or current_runtime.get("config_id") != config_id
        or current_runtime.get("attempt_root") != str(attempt)
        or _mapping(
            current_runtime.get("source_binding"),
            "reverified runtime source binding",
        ).get("binding_sha256")
        != binding_record["binding_sha256"]
    ):
        raise ResumeProbeError("reverified runtime identity differs")

    predecessor = _mapping(evidence["resume_predecessor"], "resume_predecessor")
    _exact_keys(
        predecessor,
        {
            "checkpoint_dir",
            "checkpoint_step",
            "evidence_path",
            "evidence_sha256",
            "logical_config_id",
            "resolved_config_sha256",
        },
        "resume_predecessor",
    )
    if predecessor["checkpoint_step"] != contract["predecessor_step"]:
        raise ResumeProbeError("resume predecessor step changed")
    predecessor_evidence = _safe_path(
        predecessor["evidence_path"], "resume predecessor evidence", exists=True
    )
    if (
        not predecessor_evidence.is_file()
        or _sha256_file(predecessor_evidence)
        != _sha256(predecessor["evidence_sha256"], "predecessor evidence SHA-256")
    ):
        raise ResumeProbeError("resume predecessor evidence changed")
    _sha256(predecessor["resolved_config_sha256"], "predecessor config SHA-256")

    checkpoint = _mapping(evidence["checkpoint"], "checkpoint")
    _exact_keys(
        checkpoint,
        {
            "completion_marker",
            "manifest_sha256",
            "path",
            "root_extra_state",
        },
        "checkpoint",
    )
    checkpoint_path = _safe_path(checkpoint["path"], "checkpoint.path", exists=True)
    expected_checkpoint = attempt / "checkpoints" / f"global_step_{contract['global_step']}"
    if checkpoint_path != expected_checkpoint or not checkpoint_path.is_dir():
        raise ResumeProbeError("checkpoint path changed")
    _sha256(checkpoint["manifest_sha256"], "checkpoint manifest SHA-256")
    completion_path = _verify_file_record(
        checkpoint["completion_marker"], "checkpoint completion marker"
    )
    if completion_path != checkpoint_path / COMPLETION_MARKER_FILENAME:
        raise ResumeProbeError("checkpoint completion marker path changed")
    root_record = _mapping(checkpoint["root_extra_state"], "root_extra_state")
    _exact_keys(
        root_record,
        {"path", "file_sha256", "extra_state_sha256"},
        "root_extra_state",
    )
    root_path = _verify_file_record(
        {key: root_record[key] for key in _FILE_RECORD_KEYS},
        "root_extra_state",
    )
    if root_path != checkpoint_path / CHECKPOINT_EXTRA_FILENAME:
        raise ResumeProbeError("root extra-state path changed")
    _sha256(root_record["extra_state_sha256"], "root extra-state SHA-256")
    current_manifest, current_state = (
        checkpoint_verifier or _default_checkpoint_verifier
    )(checkpoint_path)
    if (
        getattr(current_manifest, "sha256", None)
        != checkpoint["manifest_sha256"]
        or getattr(current_state, "sha256", None)
        != root_record["extra_state_sha256"]
        or getattr(current_state, "global_step", None) != contract["global_step"]
    ):
        raise ResumeProbeError("reverified checkpoint identity differs")

    adapter = _mapping(evidence["adapter"], "adapter")
    _exact_keys(adapter, {"path", "metadata", "metadata_sha256"}, "adapter")
    adapter_path = _safe_path(adapter["path"], "adapter.path", exists=True)
    expected_adapter = (
        attempt
        / "artifacts"
        / "adapter"
        / f"global_step_{contract['global_step']}"
        / "adapter"
    )
    if adapter_path != expected_adapter or not adapter_path.is_dir():
        raise ResumeProbeError("adapter path changed")
    metadata_path = _verify_file_record(adapter["metadata"], "adapter metadata")
    if metadata_path != adapter_path / ADAPTER_METADATA_FILENAME:
        raise ResumeProbeError("adapter metadata path changed")
    _sha256(adapter["metadata_sha256"], "adapter metadata SHA-256")
    _, current_adapter = (
        adapter_evidence_verifier or _default_adapter_evidence_verifier
    )(adapter_path)
    if (
        getattr(current_adapter, "sha256", None) != adapter["metadata_sha256"]
        or getattr(current_adapter, "source_extra_state_sha256", None)
        != root_record["extra_state_sha256"]
        or getattr(current_adapter, "global_step", None)
        != contract["global_step"]
    ):
        raise ResumeProbeError("reverified adapter identity differs")

    loaded = evidence["loaded_state_files"]
    if not isinstance(loaded, list) or not loaded:
        raise ResumeProbeError("loaded_state_files must be a non-empty list")
    identities: set[tuple[str, int | None]] = set()
    ranks: set[int] = set()
    for raw in loaded:
        record = _mapping(raw, "loaded state file")
        _exact_keys(record, _LOADED_FILE_KEYS, "loaded state file")
        kind = record["kind"]
        rank = record["rank"]
        if kind not in {
            "actor_model",
            "actor_optimizer",
            "actor_rank_extra",
            "dataloader",
            "driver_rng",
        }:
            raise ResumeProbeError("loaded state kind is invalid")
        if kind.startswith("actor_"):
            if type(rank) is not int or rank < 0:
                raise ResumeProbeError("actor loaded state rank is invalid")
            ranks.add(rank)
        elif rank is not None:
            raise ResumeProbeError("non-actor loaded state must have null rank")
        identity = (kind, rank)
        if identity in identities:
            raise ResumeProbeError("loaded state inventory contains duplicates")
        identities.add(identity)
        loaded_path = _verify_file_record(
            {key: record[key] for key in _FILE_RECORD_KEYS},
            "loaded state file",
        )
        try:
            loaded_path.relative_to(checkpoint_path)
        except ValueError as exc:
            raise ResumeProbeError("loaded state file escaped checkpoint") from exc
    expected_identities = {("dataloader", None), ("driver_rng", None)}
    expected_identities.update(
        (kind, rank)
        for rank in range(len(ranks))
        for kind in ("actor_model", "actor_optimizer", "actor_rank_extra")
    )
    if ranks != set(range(len(ranks))) or identities != expected_identities:
        raise ResumeProbeError("loaded state inventory is incomplete")
    return evidence


def _atomic_create(path: Path, value: Mapping[str, Any]) -> None:
    destination = _safe_path(path, "resume endpoint output", exists=False)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    parent = _safe_path(
        destination.parent, "resume endpoint output directory", exists=True
    )
    payload = _canonical_bytes(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.staging-", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination, follow_symlinks=False)
        if os.name != "nt":
            parent_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
    finally:
        temporary.unlink(missing_ok=True)


def publish_resume_endpoint_probe(
    evidence: Mapping[str, Any],
    output_path: str | os.PathLike[str],
    *,
    runtime_verifier: Callable[[Path], Mapping[str, Any]] | None = None,
    checkpoint_verifier: Callable[[Path], tuple[Any, Any]] | None = None,
    adapter_evidence_verifier: Callable[[Path], tuple[Any, Any]] | None = None,
) -> Mapping[str, Any]:
    output = Path(output_path)
    _atomic_create(output, evidence)
    return verify_resume_endpoint_probe(
        output,
        runtime_verifier=runtime_verifier,
        checkpoint_verifier=checkpoint_verifier,
        adapter_evidence_verifier=adapter_evidence_verifier,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe = subparsers.add_parser("probe")
    probe.add_argument("--attempt-dir", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--evidence", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "probe":
            evidence = build_resume_endpoint_probe(args.attempt_dir)
            result = publish_resume_endpoint_probe(evidence, args.output)
        else:
            result = verify_resume_endpoint_probe(args.evidence)
    except Exception as exc:
        print(
            json.dumps(
                {"error": f"{type(exc).__name__}: {exc}", "status": "blocked"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(_canonical_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EVIDENCE_FILENAME",
    "ResumeProbeError",
    "build_resume_endpoint_probe",
    "publish_resume_endpoint_probe",
    "verify_resume_endpoint_probe",
]
