"""Bind one sealed resolved config to an attempt, then optionally run it."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
EVIDENCE_KIND = "rememr1-runtime-bound-training-config-v1"
BOUND_CONFIG_FILENAME = "runtime-bound.yaml"
BINDING_FILENAME = "runtime-binding.json"
BINDING_SOURCE_FILENAME = "runtime-binding-source.json"
EVIDENCE_FILENAME = "runtime-bound.json"
CAPACITY_STOP_FILENAME = "capacity-stop.json"
CAPACITY_STOP_KIND = "rememr1-capacity-stop-v1"
CAPACITY_STOP_EXIT_CODE = 43
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_INDEX_KEYS = {"configs", "data_root", "schema_version", "status"}
_INDEX_ENTRY_KEYS = {
    "offload_profile",
    "overrides",
    "path",
    "sha256",
    "source_config",
}
_BINDING_KEYS = {
    "schema_version",
    "config_id",
    "index_sha256",
    "resolved_config_sha256",
    "attempt_id",
    "attempt_root",
    "paths",
    "resume_predecessor",
    "binding_sha256",
}
_PATH_KEYS = {
    "checkpoint_dir",
    "rollout_dir",
    "validation_dir",
    "adapter_dir",
    "telemetry_path",
    "step_zero_fingerprint_path",
    "step_zero_reference",
    "pilot_evidence_path",
}
_PREDECESSOR_KEYS = {
    "logical_config_id",
    "resolved_config_sha256",
    "checkpoint_dir",
    "checkpoint_step",
    "evidence_path",
    "evidence_sha256",
}
_EVIDENCE_KEYS = {
    "kind",
    "schema_version",
    "status",
    "config_id",
    "attempt_id",
    "attempt_root",
    "source_index",
    "source_config",
    "source_binding",
    "resume_predecessor",
    "leaf_changes",
    "runtime_bound_yaml_sha256",
    "runtime_bound_config_sha256",
    "evidence_sha256",
}
_CAPACITY_STOP_KEYS = {
    "kind",
    "schema_version",
    "status",
    "reason",
    "config_id",
    "attempt_id",
    "attempt_root",
    "runtime_bound_evidence",
    "runtime_binding",
    "exception_type",
    "capacity_stop_sha256",
}
_G2_CAPACITY_CONFIG_IDS = frozenset(
    f"{name}_{profile}"
    for name in (
        "g2a_qwen35_2b_5090",
        "g2b_qwen35_2b_5090_step1",
        "g2b_qwen35_2b_5090_resume5",
        "g2_length_stress_qwen35_2b_5090",
    )
    for profile in ("r0", "r1")
)
_CAPACITY_OOM_EXCEPTION_TYPES = frozenset(
    {"torch.OutOfMemoryError", "torch.cuda.OutOfMemoryError"}
)
_IMMUTABLE_PREDECESSOR_PATHS = (
    "data.train_files",
    "data.val_files",
    "reproduction.experiment_profile_id",
    "reproduction.offload_profile",
    "reproduction.run_seed",
    "reproduction.data_manifest_sha256",
    "reproduction.val_data_manifest_sha256",
    "actor_rollout_ref.model.path",
    "actor_rollout_ref.model.revision",
)
_ALLOWED_CHANGED_PATHS = {
    "trainer.default_local_dir",
    "trainer.rollout_data_dir",
    "trainer.validation_data_dir",
    "trainer.resume_from_path",
    "reproduction.adapter_export_dir",
    "reproduction.step_zero_fingerprint_path",
    "reproduction.step_zero_reference_path",
    "reproduction.pilot_evidence_path",
    "reproduction.runtime_attempt_id",
    "reproduction.runtime_binding_sha256",
    "reproduction.runtime_bound_evidence_path",
    "reproduction.runtime_telemetry_path",
    "reproduction.sealed_config_id",
    "reproduction.sealed_config_sha256",
}


class RuntimeBindingError(RuntimeError):
    """Raised when a sealed config or runtime binding fails closed."""


def _construct_unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise RuntimeBindingError(f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeBindingError(f"{label} must be a mapping")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise RuntimeBindingError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RuntimeBindingError(f"{label} must be a lowercase SHA-256")
    return value


def _require_safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise RuntimeBindingError(f"{label} is not a safe identifier")
    return value


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeBindingError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _path_without_symlinks(
    value: str | os.PathLike[str], label: str, *, exists: bool
) -> Path:
    candidate = Path(os.path.abspath(Path(value).expanduser()))
    for component in [*reversed(candidate.parents), candidate]:
        if component.is_symlink():
            raise RuntimeBindingError(f"{label} contains a symlink: {component}")
    try:
        resolved = candidate.resolve(strict=exists)
    except OSError as exc:
        raise RuntimeBindingError(f"cannot resolve {label}: {exc}") from exc
    return resolved


def _require_within(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeBindingError(f"{label} escapes its allowed root") from exc


def _load_json(path: Path, label: str, *, canonical: bool) -> Mapping[str, Any]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload, object_pairs_hook=_reject_duplicate_pairs)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeBindingError(f"cannot parse {label}: {exc}") from exc
    value = _require_mapping(value, label)
    if canonical and _canonical_json_bytes(value) + b"\n" != payload:
        raise RuntimeBindingError(f"{label} is not canonical newline-terminated JSON")
    return value


def _load_yaml(path: Path, label: str) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeBindingError("PyYAML is required to bind resolved configs") from exc

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        _construct_unique_mapping,
    )
    payload = path.read_bytes()
    try:
        value = yaml.load(payload, Loader=UniqueKeyLoader)
    except Exception as exc:
        if isinstance(exc, RuntimeBindingError):
            raise
        raise RuntimeBindingError(f"cannot parse {label}: {exc}") from exc
    value = _require_mapping(value, label)
    if b"${" in payload or b"???" in payload:
        raise RuntimeBindingError(f"{label} is not fully resolved")
    return value


def _canonical_yaml_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeBindingError("PyYAML is required to bind resolved configs") from exc
    return yaml.safe_dump(
        dict(value),
        allow_unicode=False,
        default_flow_style=False,
        sort_keys=True,
        width=4096,
    ).encode("ascii")


def _field(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for name in dotted_path.split("."):
        if not isinstance(current, Mapping) or name not in current:
            return None
        current = current[name]
    return current


def _set_field(value: dict[str, Any], dotted_path: str, replacement: Any) -> None:
    names = dotted_path.split(".")
    current: dict[str, Any] = value
    for name in names[:-1]:
        child = current.get(name)
        if not isinstance(child, dict):
            raise RuntimeBindingError(f"cannot bind missing config container {dotted_path}")
        current = child
    current[names[-1]] = replacement


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(value[key], path))
        return result
    if isinstance(value, list):
        return {prefix: copy.deepcopy(value)}
    return {prefix: value}


def _leaf_changes(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> list[dict[str, Any]]:
    before_flat = _flatten(before)
    after_flat = _flatten(after)
    changes = []
    for path in sorted(set(before_flat) | set(after_flat)):
        before_present = path in before_flat
        after_present = path in after_flat
        old = before_flat.get(path)
        new = after_flat.get(path)
        if before_present != after_present or old != new:
            changes.append(
                {
                    "path": path,
                    "before_present": before_present,
                    "after_present": after_present,
                    "before": old,
                    "after": new,
                }
            )
    unexpected = {change["path"] for change in changes} - _ALLOWED_CHANGED_PATHS
    if unexpected:
        raise RuntimeBindingError(
            f"runtime binding changed non-allowlisted leaves: {sorted(unexpected)}"
        )
    return changes


def _load_index(
    index_path: str | os.PathLike[str], expected_sha256: str
) -> tuple[Path, Mapping[str, Any]]:
    path = _path_without_symlinks(index_path, "resolved config index", exists=True)
    if not path.is_file():
        raise RuntimeBindingError("resolved config index must be a file")
    expected_sha256 = _require_sha256(expected_sha256, "index_sha256")
    if _sha256_file(path) != expected_sha256:
        raise RuntimeBindingError("resolved config index SHA-256 changed")
    index = _load_json(path, "resolved config index", canonical=False)
    _require_exact_keys(index, _INDEX_KEYS, "resolved config index")
    if index["schema_version"] != 1 or index["status"] != "resolved":
        raise RuntimeBindingError("resolved config index has an invalid contract")
    data_root = _path_without_symlinks(index["data_root"], "index data_root", exists=True)
    if not data_root.is_dir():
        raise RuntimeBindingError("index data_root must be a directory")
    configs = _require_mapping(index["configs"], "resolved config index.configs")
    if not configs:
        raise RuntimeBindingError("resolved config index has no configs")
    expected_inventory = {"index.json"}
    for config_id, raw_entry in configs.items():
        _require_safe_id(config_id, "config ID")
        entry = _require_mapping(raw_entry, f"configs.{config_id}")
        _require_exact_keys(entry, _INDEX_ENTRY_KEYS, f"configs.{config_id}")
        expected_path = path.parent / f"{config_id}.yaml"
        configured_path = _path_without_symlinks(
            entry["path"], f"configs.{config_id}.path", exists=True
        )
        if configured_path != expected_path.resolve(strict=True):
            raise RuntimeBindingError(f"config path changed for {config_id}")
        if not configured_path.is_file():
            raise RuntimeBindingError(f"config path is not a file for {config_id}")
        expected_digest = _require_sha256(entry["sha256"], f"configs.{config_id}.sha256")
        if _sha256_file(configured_path) != expected_digest:
            raise RuntimeBindingError(f"resolved config bytes changed for {config_id}")
        if not isinstance(entry["overrides"], list) or not all(
            isinstance(value, str) for value in entry["overrides"]
        ):
            raise RuntimeBindingError(f"config index metadata is malformed for {config_id}")
        if entry["offload_profile"] not in {None, "r0", "r1"}:
            raise RuntimeBindingError(f"offload profile is invalid for {config_id}")
        _require_safe_id(entry["source_config"], f"configs.{config_id}.source_config")
        expected_inventory.add(f"{config_id}.yaml")
    actual_inventory = set()
    for candidate in path.parent.iterdir():
        if candidate.is_symlink() or not candidate.is_file():
            raise RuntimeBindingError("resolved config directory contains an unsafe entry")
        actual_inventory.add(candidate.name)
    if actual_inventory != expected_inventory:
        raise RuntimeBindingError(
            "resolved config directory inventory differs from index"
        )
    return path, index


def _load_binding(path: str | os.PathLike[str]) -> tuple[Path, Mapping[str, Any]]:
    binding_path = _path_without_symlinks(path, "runtime binding", exists=True)
    if not binding_path.is_file():
        raise RuntimeBindingError("runtime binding must be a file")
    binding = _load_json(binding_path, "runtime binding", canonical=True)
    _require_exact_keys(binding, _BINDING_KEYS, "runtime binding")
    if binding["schema_version"] != SCHEMA_VERSION:
        raise RuntimeBindingError("runtime binding schema_version is unsupported")
    _require_safe_id(binding["config_id"], "runtime binding config_id")
    _require_safe_id(binding["attempt_id"], "runtime binding attempt_id")
    _require_sha256(binding["index_sha256"], "runtime binding index_sha256")
    _require_sha256(
        binding["resolved_config_sha256"],
        "runtime binding resolved_config_sha256",
    )
    paths = _require_mapping(binding["paths"], "runtime binding paths")
    _require_exact_keys(paths, _PATH_KEYS, "runtime binding paths")
    predecessor = binding["resume_predecessor"]
    if predecessor is not None:
        predecessor = _require_mapping(predecessor, "resume_predecessor")
        _require_exact_keys(predecessor, _PREDECESSOR_KEYS, "resume_predecessor")
    digest = _require_sha256(binding["binding_sha256"], "binding_sha256")
    unsigned = {key: value for key, value in binding.items() if key != "binding_sha256"}
    if digest != _canonical_sha256(unsigned):
        raise RuntimeBindingError("runtime binding self-hash mismatch")
    return binding_path, binding


def _source_config(
    index_path: Path,
    index: Mapping[str, Any],
    config_id: str,
    expected_sha256: str,
) -> tuple[Path, Mapping[str, Any], Mapping[str, Any]]:
    configs = index["configs"]
    if config_id not in configs:
        raise RuntimeBindingError(f"config ID is not sealed: {config_id}")
    entry = configs[config_id]
    digest = _require_sha256(expected_sha256, "resolved_config_sha256")
    if entry["sha256"] != digest:
        raise RuntimeBindingError("runtime binding selected a different config SHA-256")
    path = index_path.parent / f"{config_id}.yaml"
    if _sha256_file(path) != digest:
        raise RuntimeBindingError("selected resolved config bytes changed")
    config = _load_yaml(path, "resolved config")
    if _field(config, "trainer.total_training_steps") in {None, 0}:
        raise RuntimeBindingError("resolved config is not a training config")
    return path, entry, config


def _same_nullable_shape(source: Any, bound: Any, label: str) -> None:
    if (source is None) != (bound is None):
        raise RuntimeBindingError(f"{label} nullability differs from sealed config")


def _expected_attempt_path(
    attempt_root: Path, source_path: str, category: str
) -> Path:
    basename = Path(source_path).name
    if category == "checkpoint":
        return attempt_root / "checkpoints"
    if category == "rollout":
        return attempt_root / "logs" / "rollouts"
    if category == "validation":
        return attempt_root / "logs" / "validation"
    if category == "adapter":
        return attempt_root / "artifacts" / "adapter"
    if category == "evidence":
        return attempt_root / "evidence" / basename
    raise AssertionError(category)


def _bound_output_path(
    raw: Any,
    expected: Path,
    attempt_root: Path,
    label: str,
) -> str:
    if not isinstance(raw, str):
        raise RuntimeBindingError(f"{label} must be an absolute canonical path")
    path = _path_without_symlinks(raw, label, exists=False)
    if raw != str(path):
        raise RuntimeBindingError(f"{label} must be an absolute canonical path")
    _require_within(path, attempt_root, label)
    if path != expected.resolve(strict=False):
        raise RuntimeBindingError(f"{label} is not the registered attempt path")
    return str(path)


def _validate_reference(value: Any, label: str) -> str:
    record = _require_mapping(value, label)
    _require_exact_keys(record, {"path", "sha256"}, label)
    path = _path_without_symlinks(record["path"], f"{label}.path", exists=True)
    if record["path"] != str(path):
        raise RuntimeBindingError(f"{label}.path must be an absolute canonical path")
    if not path.is_file():
        raise RuntimeBindingError(f"{label}.path must be a file")
    digest = _require_sha256(record["sha256"], f"{label}.sha256")
    if _sha256_file(path) != digest:
        raise RuntimeBindingError(f"{label} bytes changed")
    return str(path)


def _resolve_predecessor_step_zero_record(
    predecessor: Mapping[str, Any],
) -> dict[str, str]:
    checkpoint_dir = _path_without_symlinks(
        predecessor["checkpoint_dir"],
        "resume predecessor checkpoint directory",
        exists=True,
    )
    attempt_root = checkpoint_dir.parent.parent
    runtime_dir = attempt_root / "runtime-bound"
    evidence = verify_bound_config(runtime_dir)
    logical_id = _require_safe_id(
        predecessor["logical_config_id"],
        "resume_predecessor.logical_config_id",
    )
    config_sha256 = _require_sha256(
        predecessor["resolved_config_sha256"],
        "resume_predecessor.resolved_config_sha256",
    )
    checkpoint_step = predecessor["checkpoint_step"]
    expected_checkpoint = (
        attempt_root / "checkpoints" / f"global_step_{checkpoint_step}"
    ).resolve(strict=True)
    if checkpoint_dir != expected_checkpoint:
        raise RuntimeBindingError(
            "resume predecessor checkpoint is not under its runtime-bound attempt"
        )
    if (
        evidence["config_id"] != logical_id
        or evidence["attempt_id"] != attempt_root.name
        or evidence["attempt_root"] != str(attempt_root)
        or evidence["source_config"]["sha256"] != config_sha256
    ):
        raise RuntimeBindingError(
            "resume predecessor runtime-bound identity differs"
        )

    _, predecessor_binding = _load_binding(runtime_dir / BINDING_FILENAME)
    if (
        predecessor_binding["config_id"] != logical_id
        or predecessor_binding["resolved_config_sha256"] != config_sha256
        or predecessor_binding["attempt_id"] != attempt_root.name
        or predecessor_binding["attempt_root"] != str(attempt_root)
    ):
        raise RuntimeBindingError(
            "resume predecessor runtime binding identity differs"
        )
    raw_record = predecessor_binding["paths"]["step_zero_fingerprint_path"]
    if isinstance(raw_record, str):
        fingerprint_path = _path_without_symlinks(
            raw_record,
            "resume predecessor step-zero fingerprint",
            exists=True,
        )
        if raw_record != str(fingerprint_path) or not fingerprint_path.is_file():
            raise RuntimeBindingError(
                "resume predecessor step-zero fingerprint must be a canonical file"
            )
        return {
            "path": str(fingerprint_path),
            "sha256": _sha256_file(fingerprint_path),
        }

    record = _require_mapping(
        raw_record, "resume predecessor step-zero fingerprint"
    )
    _require_exact_keys(
        record,
        {"path", "sha256"},
        "resume predecessor step-zero fingerprint",
    )
    fingerprint = _validate_reference(
        record, "resume predecessor step-zero fingerprint"
    )
    return {
        "path": fingerprint,
        "sha256": _require_sha256(
            record["sha256"],
            "resume predecessor step-zero fingerprint.sha256",
        ),
    }


def _validate_predecessor(
    binding_value: Any,
    *,
    current_config: Mapping[str, Any],
    index_path: Path,
    index: Mapping[str, Any],
) -> tuple[str | None, Mapping[str, Any] | None]:
    resume_mode = _field(current_config, "trainer.resume_mode")
    sealed_resume_path = _field(current_config, "trainer.resume_from_path")
    if resume_mode == "disable" and sealed_resume_path is None:
        if binding_value is not None:
            raise RuntimeBindingError("fresh config must not bind a resume predecessor")
        return None, None
    if resume_mode != "resume_path" or not isinstance(sealed_resume_path, str):
        raise RuntimeBindingError("training config has an unsupported resume contract")
    predecessor = _require_mapping(binding_value, "resume_predecessor")
    logical_id = _require_safe_id(
        predecessor["logical_config_id"], "resume_predecessor.logical_config_id"
    )
    predecessor_path, _, predecessor_config = _source_config(
        index_path,
        index,
        logical_id,
        predecessor["resolved_config_sha256"],
    )
    del predecessor_path
    checkpoint_step = predecessor["checkpoint_step"]
    if (
        isinstance(checkpoint_step, bool)
        or not isinstance(checkpoint_step, int)
        or checkpoint_step <= 0
    ):
        raise RuntimeBindingError("resume_predecessor.checkpoint_step must be positive")
    predecessor_total = _field(predecessor_config, "trainer.total_training_steps")
    if checkpoint_step != predecessor_total:
        raise RuntimeBindingError("resume predecessor step differs from sealed config")
    expected_sealed_path = str(
        Path(_field(predecessor_config, "trainer.default_local_dir"))
        / f"global_step_{checkpoint_step}"
    )
    if os.path.normpath(sealed_resume_path) != os.path.normpath(expected_sealed_path):
        raise RuntimeBindingError(
            "resume predecessor logical config differs from sealed resume path"
        )
    for dotted_path in _IMMUTABLE_PREDECESSOR_PATHS:
        if _field(current_config, dotted_path) != _field(predecessor_config, dotted_path):
            raise RuntimeBindingError(
                f"resume predecessor identity differs at {dotted_path}"
            )
    checkpoint_dir = _path_without_symlinks(
        predecessor["checkpoint_dir"],
        "resume_predecessor.checkpoint_dir",
        exists=True,
    )
    if predecessor["checkpoint_dir"] != str(checkpoint_dir):
        raise RuntimeBindingError(
            "resume_predecessor.checkpoint_dir must be an absolute canonical path"
        )
    if not checkpoint_dir.is_dir() or checkpoint_dir.name != f"global_step_{checkpoint_step}":
        raise RuntimeBindingError("resume predecessor checkpoint directory is invalid")
    evidence_path = _path_without_symlinks(
        predecessor["evidence_path"],
        "resume_predecessor.evidence_path",
        exists=True,
    )
    if predecessor["evidence_path"] != str(evidence_path):
        raise RuntimeBindingError(
            "resume_predecessor.evidence_path must be an absolute canonical path"
        )
    if not evidence_path.is_file():
        raise RuntimeBindingError("resume predecessor evidence must be a file")
    _require_within(evidence_path, checkpoint_dir, "resume predecessor evidence")
    evidence_sha = _require_sha256(
        predecessor["evidence_sha256"], "resume_predecessor.evidence_sha256"
    )
    if _sha256_file(evidence_path) != evidence_sha:
        raise RuntimeBindingError("resume predecessor evidence bytes changed")
    normalized = {
        **dict(predecessor),
        "checkpoint_dir": str(checkpoint_dir),
        "evidence_path": str(evidence_path),
    }
    return str(checkpoint_dir), normalized


def _apply_binding(
    source: Mapping[str, Any],
    binding: Mapping[str, Any],
    *,
    index_path: Path,
    index: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], Mapping[str, Any] | None]:
    bound = copy.deepcopy(dict(source))
    attempt_root = _path_without_symlinks(
        binding["attempt_root"], "attempt_root", exists=False
    )
    paths = binding["paths"]
    checkpoint_source = _field(source, "trainer.default_local_dir")
    rollout_source = _field(source, "trainer.rollout_data_dir")
    validation_source = _field(source, "trainer.validation_data_dir")
    if not all(
        isinstance(value, str) and value
        for value in (checkpoint_source, rollout_source, validation_source)
    ):
        raise RuntimeBindingError("sealed training config lacks output directories")
    checkpoint_dir = _bound_output_path(
        paths["checkpoint_dir"],
        _expected_attempt_path(attempt_root, checkpoint_source, "checkpoint"),
        attempt_root,
        "checkpoint_dir",
    )
    rollout_dir = _bound_output_path(
        paths["rollout_dir"],
        _expected_attempt_path(attempt_root, rollout_source, "rollout"),
        attempt_root,
        "rollout_dir",
    )
    validation_dir = _bound_output_path(
        paths["validation_dir"],
        _expected_attempt_path(attempt_root, validation_source, "validation"),
        attempt_root,
        "validation_dir",
    )
    _set_field(bound, "trainer.default_local_dir", checkpoint_dir)
    _set_field(bound, "trainer.rollout_data_dir", rollout_dir)
    _set_field(bound, "trainer.validation_data_dir", validation_dir)

    source_adapter = _field(source, "reproduction.adapter_export_dir")
    _same_nullable_shape(source_adapter, paths["adapter_dir"], "adapter_dir")
    if source_adapter is not None:
        adapter_dir = _bound_output_path(
            paths["adapter_dir"],
            _expected_attempt_path(attempt_root, source_adapter, "adapter"),
            attempt_root,
            "adapter_dir",
        )
        _set_field(bound, "reproduction.adapter_export_dir", adapter_dir)

    resume_path, predecessor = _validate_predecessor(
        binding["resume_predecessor"],
        current_config=source,
        index_path=index_path,
        index=index,
    )
    if resume_path is not None:
        _set_field(bound, "trainer.resume_from_path", resume_path)

    source_fingerprint = _field(source, "reproduction.step_zero_fingerprint_path")
    _same_nullable_shape(
        source_fingerprint,
        paths["step_zero_fingerprint_path"],
        "step_zero_fingerprint_path",
    )
    if source_fingerprint is not None:
        if resume_path is None:
            fingerprint = _bound_output_path(
                paths["step_zero_fingerprint_path"],
                _expected_attempt_path(
                    attempt_root, source_fingerprint, "evidence"
                ),
                attempt_root,
                "step_zero_fingerprint_path",
            )
        else:
            expected_record = _resolve_predecessor_step_zero_record(predecessor)
            actual_record = _require_mapping(
                paths["step_zero_fingerprint_path"],
                "step_zero_fingerprint_path",
            )
            fingerprint = _validate_reference(
                actual_record, "step_zero_fingerprint_path"
            )
            if dict(actual_record) != expected_record:
                raise RuntimeBindingError(
                    "resume step-zero fingerprint differs from predecessor binding"
                )
        _set_field(bound, "reproduction.step_zero_fingerprint_path", fingerprint)

    source_reference = _field(source, "reproduction.step_zero_reference_path")
    _same_nullable_shape(
        source_reference, paths["step_zero_reference"], "step_zero_reference"
    )
    if source_reference is not None:
        reference = _validate_reference(
            paths["step_zero_reference"], "step_zero_reference"
        )
        _set_field(bound, "reproduction.step_zero_reference_path", reference)

    source_pilot = _field(source, "reproduction.pilot_evidence_path")
    _same_nullable_shape(source_pilot, paths["pilot_evidence_path"], "pilot_evidence_path")
    if source_pilot is not None:
        pilot = _bound_output_path(
            paths["pilot_evidence_path"],
            _expected_attempt_path(attempt_root, source_pilot, "evidence"),
            attempt_root,
            "pilot_evidence_path",
        )
        _set_field(bound, "reproduction.pilot_evidence_path", pilot)

    telemetry_expected = attempt_root / "telemetry.json"
    telemetry_path = _bound_output_path(
        paths["telemetry_path"],
        telemetry_expected,
        attempt_root,
        "telemetry_path",
    )
    output_dir = attempt_root / "runtime-bound"
    reproduction = bound.get("reproduction")
    if not isinstance(reproduction, dict):
        raise RuntimeBindingError("sealed config lacks reproduction mapping")
    reproduction.update(
        {
            "runtime_attempt_id": binding["attempt_id"],
            "runtime_binding_sha256": binding["binding_sha256"],
            "runtime_bound_evidence_path": str(output_dir / EVIDENCE_FILENAME),
            "runtime_telemetry_path": telemetry_path,
            "sealed_config_id": binding["config_id"],
            "sealed_config_sha256": binding["resolved_config_sha256"],
        }
    )
    changes = _leaf_changes(source, bound)
    return bound, changes, predecessor


def _build_attempt_binding(
    index_path: str | os.PathLike[str],
    config_id: str,
    attempt_dir: str | os.PathLike[str],
    *,
    resume_config_id: str | None,
    resume_checkpoint_dir: str | os.PathLike[str] | None,
    resume_evidence_path: str | os.PathLike[str] | None,
    step_zero_reference_path: str | os.PathLike[str] | None,
) -> tuple[Path, Mapping[str, Any]]:
    raw_index = _path_without_symlinks(
        index_path, "resolved config index", exists=True
    )
    index_sha256 = _sha256_file(raw_index)
    resolved_index, index = _load_index(raw_index, index_sha256)
    config_id = _require_safe_id(config_id, "config_id")
    configs = _require_mapping(index["configs"], "resolved config index.configs")
    if config_id not in configs:
        raise RuntimeBindingError(f"config ID is not sealed: {config_id}")
    entry = _require_mapping(configs[config_id], f"configs.{config_id}")
    resolved_config_sha256 = _require_sha256(
        entry["sha256"], f"configs.{config_id}.sha256"
    )
    _, _, source = _source_config(
        resolved_index,
        index,
        config_id,
        resolved_config_sha256,
    )

    attempt_root = _path_without_symlinks(
        attempt_dir, "attempt_dir", exists=False
    )
    try:
        attempt_root.relative_to(resolved_index.parent)
    except ValueError:
        pass
    else:
        raise RuntimeBindingError(
            "attempt_dir must not be inside the sealed resolved-config directory"
        )
    attempt_id = _require_safe_id(attempt_root.name, "attempt_dir basename")

    checkpoint_source = _field(source, "trainer.default_local_dir")
    rollout_source = _field(source, "trainer.rollout_data_dir")
    validation_source = _field(source, "trainer.validation_data_dir")
    if not all(
        isinstance(value, str) and value
        for value in (checkpoint_source, rollout_source, validation_source)
    ):
        raise RuntimeBindingError("sealed training config lacks output directories")
    source_adapter = _field(source, "reproduction.adapter_export_dir")
    source_fingerprint = _field(source, "reproduction.step_zero_fingerprint_path")
    source_reference = _field(source, "reproduction.step_zero_reference_path")
    source_pilot = _field(source, "reproduction.pilot_evidence_path")
    for label, value in (
        ("adapter_export_dir", source_adapter),
        ("step_zero_fingerprint_path", source_fingerprint),
        ("step_zero_reference_path", source_reference),
        ("pilot_evidence_path", source_pilot),
    ):
        if value is not None and (not isinstance(value, str) or not value):
            raise RuntimeBindingError(
                f"sealed reproduction.{label} must be a path or null"
            )

    if source_reference is None:
        if step_zero_reference_path is not None:
            raise RuntimeBindingError(
                "sealed config does not accept a step-zero reference"
            )
        step_zero_reference = None
    else:
        if step_zero_reference_path is None:
            raise RuntimeBindingError(
                "sealed config requires --step-zero-reference"
            )
        reference_path = _path_without_symlinks(
            step_zero_reference_path, "step-zero reference", exists=True
        )
        if not reference_path.is_file():
            raise RuntimeBindingError("step-zero reference must be a file")
        step_zero_reference = {
            "path": str(reference_path),
            "sha256": _sha256_file(reference_path),
        }

    resume_values = (
        resume_config_id,
        resume_checkpoint_dir,
        resume_evidence_path,
    )
    if any(value is not None for value in resume_values) and not all(
        value is not None for value in resume_values
    ):
        raise RuntimeBindingError(
            "--resume-config-id, --resume-checkpoint-dir, and "
            "--resume-evidence must be provided together"
        )
    if all(value is not None for value in resume_values):
        logical_id = _require_safe_id(resume_config_id, "resume_config_id")
        if logical_id not in configs:
            raise RuntimeBindingError(
                f"resume config ID is not sealed: {logical_id}"
            )
        predecessor_entry = _require_mapping(
            configs[logical_id], f"configs.{logical_id}"
        )
        predecessor_sha256 = _require_sha256(
            predecessor_entry["sha256"], f"configs.{logical_id}.sha256"
        )
        _, _, predecessor_config = _source_config(
            resolved_index,
            index,
            logical_id,
            predecessor_sha256,
        )
        checkpoint_step = _field(
            predecessor_config, "trainer.total_training_steps"
        )
        checkpoint_dir = _path_without_symlinks(
            resume_checkpoint_dir,
            "resume checkpoint directory",
            exists=True,
        )
        evidence_path = _path_without_symlinks(
            resume_evidence_path,
            "resume checkpoint evidence",
            exists=True,
        )
        if not checkpoint_dir.is_dir():
            raise RuntimeBindingError("resume checkpoint directory must be a directory")
        if not evidence_path.is_file():
            raise RuntimeBindingError("resume checkpoint evidence must be a file")
        predecessor: Mapping[str, Any] | None = {
            "logical_config_id": logical_id,
            "resolved_config_sha256": predecessor_sha256,
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_step": checkpoint_step,
            "evidence_path": str(evidence_path),
            "evidence_sha256": _sha256_file(evidence_path),
        }
    else:
        predecessor = None

    _, predecessor = _validate_predecessor(
        predecessor,
        current_config=source,
        index_path=resolved_index,
        index=index,
    )

    if source_fingerprint is None:
        step_zero_fingerprint: str | Mapping[str, str] | None = None
    elif (
        _field(source, "trainer.resume_mode") == "resume_path"
        and predecessor is not None
    ):
        step_zero_fingerprint = _resolve_predecessor_step_zero_record(predecessor)
    else:
        step_zero_fingerprint = str(
            _expected_attempt_path(
                attempt_root, source_fingerprint, "evidence"
            ).resolve(strict=False)
        )

    paths = {
        "checkpoint_dir": str(
            _expected_attempt_path(
                attempt_root, checkpoint_source, "checkpoint"
            ).resolve(strict=False)
        ),
        "rollout_dir": str(
            _expected_attempt_path(
                attempt_root, rollout_source, "rollout"
            ).resolve(strict=False)
        ),
        "validation_dir": str(
            _expected_attempt_path(
                attempt_root, validation_source, "validation"
            ).resolve(strict=False)
        ),
        "adapter_dir": (
            str(
                _expected_attempt_path(
                    attempt_root, source_adapter, "adapter"
                ).resolve(strict=False)
            )
            if source_adapter is not None
            else None
        ),
        "telemetry_path": str(
            (attempt_root / "telemetry.json").resolve(strict=False)
        ),
        "step_zero_fingerprint_path": step_zero_fingerprint,
        "step_zero_reference": step_zero_reference,
        "pilot_evidence_path": (
            str(
                _expected_attempt_path(
                    attempt_root, source_pilot, "evidence"
                ).resolve(strict=False)
            )
            if source_pilot is not None
            else None
        ),
    }
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "config_id": config_id,
        "index_sha256": index_sha256,
        "resolved_config_sha256": resolved_config_sha256,
        "attempt_id": attempt_id,
        "attempt_root": str(attempt_root),
        "paths": paths,
        "resume_predecessor": predecessor,
    }
    binding = {**unsigned, "binding_sha256": _canonical_sha256(unsigned)}
    _apply_binding(source, binding, index_path=resolved_index, index=index)
    return resolved_index, binding


def bind_attempt(
    index_path: str | os.PathLike[str],
    config_id: str,
    attempt_dir: str | os.PathLike[str],
    *,
    resume_config_id: str | None = None,
    resume_checkpoint_dir: str | os.PathLike[str] | None = None,
    resume_evidence_path: str | os.PathLike[str] | None = None,
    step_zero_reference_path: str | os.PathLike[str] | None = None,
) -> Mapping[str, Any]:
    resolved_index, binding = _build_attempt_binding(
        index_path,
        config_id,
        attempt_dir,
        resume_config_id=resume_config_id,
        resume_checkpoint_dir=resume_checkpoint_dir,
        resume_evidence_path=resume_evidence_path,
        step_zero_reference_path=step_zero_reference_path,
    )
    attempt_root = Path(binding["attempt_root"])
    output_dir = attempt_root / "runtime-bound"
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    attempt_root.mkdir(parents=True, exist_ok=True)
    attempt_root = _path_without_symlinks(
        attempt_root, "attempt_dir", exists=True
    )
    if not attempt_root.is_dir():
        raise RuntimeBindingError("attempt_dir must be a directory")
    binding_path = attempt_root / BINDING_SOURCE_FILENAME
    payload = _canonical_json_bytes(binding) + b"\n"
    _write_new_bytes(binding_path, payload)
    return bind_resolved_config(
        resolved_index,
        binding["config_id"],
        binding_path,
        output_dir,
    )


def _prepare(
    index_path: str | os.PathLike[str],
    config_id: str,
    binding_path: str | os.PathLike[str],
) -> dict[str, Any]:
    binding_file, binding = _load_binding(binding_path)
    config_id = _require_safe_id(config_id, "config_id")
    if binding["config_id"] != config_id:
        raise RuntimeBindingError("CLI config ID differs from runtime binding")
    attempt_root = _path_without_symlinks(
        binding["attempt_root"], "attempt_root", exists=False
    )
    if binding["attempt_root"] != str(attempt_root):
        raise RuntimeBindingError("attempt_root must be an absolute canonical path")
    if binding["attempt_id"] != attempt_root.name:
        raise RuntimeBindingError("attempt_id must equal the attempt_root basename")
    expected_binding_file = (attempt_root / BINDING_SOURCE_FILENAME).resolve(
        strict=False
    )
    if binding_file != expected_binding_file:
        raise RuntimeBindingError(
            f"runtime binding must be {BINDING_SOURCE_FILENAME} under attempt_root"
        )
    resolved_index, index = _load_index(index_path, binding["index_sha256"])
    source_path, _, source = _source_config(
        resolved_index,
        index,
        config_id,
        binding["resolved_config_sha256"],
    )
    bound, changes, predecessor = _apply_binding(
        source,
        binding,
        index_path=resolved_index,
        index=index,
    )
    output_dir = attempt_root / "runtime-bound"
    return {
        "index_path": resolved_index,
        "index": index,
        "source_path": source_path,
        "source": source,
        "binding_path": binding_file,
        "binding": binding,
        "bound": bound,
        "changes": changes,
        "predecessor": predecessor,
        "attempt_root": attempt_root,
        "output_dir": output_dir,
    }


def _sync_parent(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes(path: Path, payload: bytes) -> str:
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _write_new_bytes(path: Path, payload: bytes) -> str:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _sync_parent(path)
    return hashlib.sha256(payload).hexdigest()


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = _path_without_symlinks(
        path, "capacity stop evidence", exists=False
    )
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    parent = _path_without_symlinks(
        destination.parent, "capacity stop evidence directory", exists=True
    )
    if not parent.is_dir():
        raise RuntimeBindingError("capacity stop evidence parent must be a directory")
    payload = _canonical_json_bytes(value) + b"\n"
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
        _sync_parent(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _trusted_cuda_oom(
    error: BaseException,
    *,
    torch_module: Any | None = None,
) -> tuple[BaseException, str] | None:
    if torch_module is None:
        try:
            import importlib

            torch_module = importlib.import_module("torch")
        except ImportError:
            return None

    candidates: list[tuple[type[BaseException], str]] = []

    def add_candidate(value: Any, label: str) -> None:
        if (
            isinstance(value, type)
            and issubclass(value, BaseException)
            and all(existing is not value for existing, _ in candidates)
        ):
            candidates.append((value, label))

    add_candidate(
        getattr(torch_module, "OutOfMemoryError", None),
        "torch.OutOfMemoryError",
    )
    cuda = getattr(torch_module, "cuda", None)
    add_candidate(
        getattr(cuda, "OutOfMemoryError", None) if cuda is not None else None,
        "torch.cuda.OutOfMemoryError",
    )
    if not candidates:
        return None

    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending and len(seen) < 64:
        current = pending.pop(0)
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        for exception_type, label in candidates:
            if type(current) is exception_type:
                return current, label
        for exception_type, label in candidates:
            if isinstance(current, exception_type):
                return current, label
        for linked in (current.__cause__, current.__context__):
            if isinstance(linked, BaseException) and id(linked) not in seen:
                pending.append(linked)
        module_name = type(current).__module__
        if module_name == "ray" or module_name.startswith("ray."):
            converter = getattr(current, "as_instanceof_cause", None)
            if callable(converter):
                try:
                    converted = converter()
                except Exception:
                    converted = None
                if isinstance(converted, BaseException) and id(converted) not in seen:
                    pending.append(converted)
    return None


def _evidence_payload(prepared: Mapping[str, Any], yaml_sha256: str) -> dict[str, Any]:
    binding_path = prepared["binding_path"]
    binding = prepared["binding"]
    payload = {
        "kind": EVIDENCE_KIND,
        "schema_version": SCHEMA_VERSION,
        "status": "bound",
        "config_id": binding["config_id"],
        "attempt_id": binding["attempt_id"],
        "attempt_root": str(prepared["attempt_root"]),
        "source_index": {
            "path": str(prepared["index_path"]),
            "sha256": binding["index_sha256"],
        },
        "source_config": {
            "path": str(prepared["source_path"]),
            "sha256": binding["resolved_config_sha256"],
        },
        "source_binding": {
            "path": str(binding_path),
            "file_sha256": _sha256_file(binding_path),
            "binding_sha256": binding["binding_sha256"],
        },
        "resume_predecessor": prepared["predecessor"],
        "leaf_changes": prepared["changes"],
        "runtime_bound_yaml_sha256": yaml_sha256,
        "runtime_bound_config_sha256": _canonical_sha256(prepared["bound"]),
    }
    return {**payload, "evidence_sha256": _canonical_sha256(payload)}


def bind_resolved_config(
    index_path: str | os.PathLike[str],
    config_id: str,
    binding_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
) -> Mapping[str, Any]:
    prepared = _prepare(index_path, config_id, binding_path)
    destination = _path_without_symlinks(output_dir, "runtime-bound output", exists=False)
    if destination != prepared["output_dir"].resolve(strict=False):
        raise RuntimeBindingError(
            "runtime-bound output must be <attempt_root>/runtime-bound"
        )
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    for path_name in (
        "checkpoint_dir",
        "rollout_dir",
        "validation_dir",
        "adapter_dir",
        "telemetry_path",
        "step_zero_fingerprint_path",
        "pilot_evidence_path",
    ):
        raw = prepared["binding"]["paths"][path_name]
        if raw is not None and not isinstance(raw, Mapping) and Path(raw).exists():
            raise RuntimeBindingError(f"attempt output already exists: {path_name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-", dir=destination.parent
        )
    )
    try:
        yaml_payload = _canonical_yaml_bytes(prepared["bound"])
        yaml_sha = _write_bytes(staging / BOUND_CONFIG_FILENAME, yaml_payload)
        binding_payload = _canonical_json_bytes(prepared["binding"]) + b"\n"
        _write_bytes(staging / BINDING_FILENAME, binding_payload)
        evidence = _evidence_payload(prepared, yaml_sha)
        _write_bytes(
            staging / EVIDENCE_FILENAME,
            _canonical_json_bytes(evidence) + b"\n",
        )
        _verify_bound_directory(staging, prepared=prepared)
        os.replace(staging, destination)
        _sync_parent(destination)
        return _verify_bound_directory(destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _verify_file_record(record: Any, label: str) -> Path:
    value = _require_mapping(record, label)
    _require_exact_keys(value, {"path", "sha256"}, label)
    path = _path_without_symlinks(value["path"], f"{label}.path", exists=True)
    if not path.is_file() or _sha256_file(path) != _require_sha256(
        value["sha256"], f"{label}.sha256"
    ):
        raise RuntimeBindingError(f"{label} changed")
    return path


def _verify_bound_directory(
    directory: Path,
    *,
    prepared: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    staged_verification = prepared is not None
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeBindingError("runtime-bound output is missing or unsafe")
    entries = tuple(directory.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise RuntimeBindingError("runtime-bound output contains an unsafe entry")
    if {entry.name for entry in entries} != {
        BOUND_CONFIG_FILENAME,
        BINDING_FILENAME,
        EVIDENCE_FILENAME,
    }:
        raise RuntimeBindingError("runtime-bound output inventory changed")
    evidence = _load_json(
        directory / EVIDENCE_FILENAME, "runtime-bound evidence", canonical=True
    )
    _require_exact_keys(evidence, _EVIDENCE_KEYS, "runtime-bound evidence")
    if (
        evidence["kind"] != EVIDENCE_KIND
        or evidence["schema_version"] != SCHEMA_VERSION
        or evidence["status"] != "bound"
    ):
        raise RuntimeBindingError("runtime-bound evidence contract changed")
    digest = _require_sha256(evidence["evidence_sha256"], "evidence_sha256")
    unsigned = {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    if digest != _canonical_sha256(unsigned):
        raise RuntimeBindingError("runtime-bound evidence self-hash mismatch")
    index_path = _verify_file_record(evidence["source_index"], "source_index")
    source_path = _verify_file_record(evidence["source_config"], "source_config")
    source_binding = _require_mapping(evidence["source_binding"], "source_binding")
    _require_exact_keys(
        source_binding,
        {"path", "file_sha256", "binding_sha256"},
        "source_binding",
    )
    external_binding = _path_without_symlinks(
        source_binding["path"], "source_binding.path", exists=True
    )
    if (
        not external_binding.is_file()
        or _sha256_file(external_binding)
        != _require_sha256(source_binding["file_sha256"], "source_binding.file_sha256")
    ):
        raise RuntimeBindingError("source runtime binding changed")
    copied_binding = _load_json(
        directory / BINDING_FILENAME, "copied runtime binding", canonical=True
    )
    external_value = _load_json(external_binding, "source runtime binding", canonical=True)
    if copied_binding != external_value:
        raise RuntimeBindingError("copied runtime binding differs from source")
    if copied_binding.get("binding_sha256") != source_binding["binding_sha256"]:
        raise RuntimeBindingError("runtime binding identity changed")
    if prepared is None:
        prepared = _prepare(
            index_path,
            evidence["config_id"],
            external_binding,
        )
    if prepared["source_path"] != source_path:
        raise RuntimeBindingError("runtime-bound source config path changed")
    binding = prepared["binding"]
    if evidence["config_id"] != binding["config_id"]:
        raise RuntimeBindingError("runtime-bound config ID changed")
    if evidence["attempt_id"] != binding["attempt_id"]:
        raise RuntimeBindingError("runtime-bound attempt ID changed")
    if evidence["attempt_root"] != str(prepared["attempt_root"]):
        raise RuntimeBindingError("runtime-bound attempt root changed")
    expected_directory = prepared["output_dir"].resolve(strict=False)
    if not staged_verification and directory.resolve(strict=True) != expected_directory:
        raise RuntimeBindingError("runtime-bound directory escaped attempt root")
    bound_path = directory / BOUND_CONFIG_FILENAME
    bound = _load_yaml(bound_path, "runtime-bound config")
    if _canonical_yaml_bytes(bound) != bound_path.read_bytes():
        raise RuntimeBindingError("runtime-bound YAML is not canonical")
    if bound != prepared["bound"]:
        raise RuntimeBindingError("runtime-bound config differs from deterministic binding")
    if _sha256_file(bound_path) != evidence["runtime_bound_yaml_sha256"]:
        raise RuntimeBindingError("runtime-bound YAML SHA-256 changed")
    if _canonical_sha256(bound) != evidence["runtime_bound_config_sha256"]:
        raise RuntimeBindingError("runtime-bound config semantic SHA-256 changed")
    if evidence["leaf_changes"] != prepared["changes"]:
        raise RuntimeBindingError("runtime-bound leaf diff changed")
    if evidence["resume_predecessor"] != prepared["predecessor"]:
        raise RuntimeBindingError("runtime-bound resume predecessor changed")
    return evidence


def verify_bound_config(output_dir: str | os.PathLike[str]) -> Mapping[str, Any]:
    directory = _path_without_symlinks(
        output_dir, "runtime-bound output", exists=True
    )
    return _verify_bound_directory(directory)


def verify_capacity_stop(
    attempt_dir: str | os.PathLike[str],
) -> Mapping[str, Any]:
    attempt_root = _path_without_symlinks(
        attempt_dir, "capacity stop attempt", exists=True
    )
    if not attempt_root.is_dir():
        raise RuntimeBindingError("capacity stop attempt must be a directory")
    stop_path = _path_without_symlinks(
        attempt_root / "evidence" / CAPACITY_STOP_FILENAME,
        "capacity stop evidence",
        exists=True,
    )
    if not stop_path.is_file():
        raise RuntimeBindingError("capacity stop evidence must be a file")
    stop = _load_json(stop_path, "capacity stop evidence", canonical=True)
    _require_exact_keys(stop, _CAPACITY_STOP_KEYS, "capacity stop evidence")
    if (
        stop["kind"] != CAPACITY_STOP_KIND
        or stop["schema_version"] != SCHEMA_VERSION
        or stop["status"] != "capacity-stop"
        or stop["reason"] != "cuda_oom"
    ):
        raise RuntimeBindingError("capacity stop evidence contract changed")
    digest = _require_sha256(
        stop["capacity_stop_sha256"], "capacity_stop_sha256"
    )
    unsigned = {
        key: value for key, value in stop.items() if key != "capacity_stop_sha256"
    }
    if digest != _canonical_sha256(unsigned):
        raise RuntimeBindingError("capacity stop evidence self-hash mismatch")
    if stop["exception_type"] not in _CAPACITY_OOM_EXCEPTION_TYPES:
        raise RuntimeBindingError("capacity stop exception type is not allowlisted")
    config_id = _require_safe_id(stop["config_id"], "capacity stop config_id")
    if config_id not in _G2_CAPACITY_CONFIG_IDS:
        raise RuntimeBindingError("capacity stop config is not a G2 capacity config")
    if stop["attempt_root"] != str(attempt_root):
        raise RuntimeBindingError("capacity stop attempt root changed")
    if stop["attempt_id"] != attempt_root.name:
        raise RuntimeBindingError("capacity stop attempt ID changed")

    runtime_dir = attempt_root / "runtime-bound"
    bound_evidence = verify_bound_config(runtime_dir)
    if (
        bound_evidence["config_id"] != config_id
        or bound_evidence["attempt_id"] != stop["attempt_id"]
        or bound_evidence["attempt_root"] != stop["attempt_root"]
    ):
        raise RuntimeBindingError("capacity stop runtime-bound identity changed")

    evidence_record = _require_mapping(
        stop["runtime_bound_evidence"], "runtime_bound_evidence"
    )
    _require_exact_keys(
        evidence_record,
        {"path", "file_sha256", "evidence_sha256"},
        "runtime_bound_evidence",
    )
    evidence_path = _path_without_symlinks(
        evidence_record["path"], "runtime_bound_evidence.path", exists=True
    )
    expected_evidence_path = (runtime_dir / EVIDENCE_FILENAME).resolve(strict=True)
    if evidence_path != expected_evidence_path:
        raise RuntimeBindingError("capacity stop runtime-bound evidence path changed")
    if _sha256_file(evidence_path) != _require_sha256(
        evidence_record["file_sha256"], "runtime_bound_evidence.file_sha256"
    ):
        raise RuntimeBindingError("capacity stop runtime-bound evidence bytes changed")
    if bound_evidence["evidence_sha256"] != _require_sha256(
        evidence_record["evidence_sha256"],
        "runtime_bound_evidence.evidence_sha256",
    ):
        raise RuntimeBindingError("capacity stop runtime-bound evidence hash changed")

    binding_record = _require_mapping(
        stop["runtime_binding"], "runtime_binding"
    )
    _require_exact_keys(
        binding_record,
        {"path", "file_sha256", "binding_sha256"},
        "runtime_binding",
    )
    binding_path, binding = _load_binding(binding_record["path"])
    expected_binding_path = (
        attempt_root / BINDING_SOURCE_FILENAME
    ).resolve(strict=True)
    if binding_path != expected_binding_path:
        raise RuntimeBindingError("capacity stop runtime binding path changed")
    if _sha256_file(binding_path) != _require_sha256(
        binding_record["file_sha256"], "runtime_binding.file_sha256"
    ):
        raise RuntimeBindingError("capacity stop runtime binding bytes changed")
    if binding["binding_sha256"] != _require_sha256(
        binding_record["binding_sha256"], "runtime_binding.binding_sha256"
    ):
        raise RuntimeBindingError("capacity stop runtime binding hash changed")
    return stop


def publish_capacity_stop_for_exception(
    output_dir: str | os.PathLike[str],
    error: BaseException,
    *,
    torch_module: Any | None = None,
) -> Mapping[str, Any] | None:
    matched = _trusted_cuda_oom(error, torch_module=torch_module)
    if matched is None:
        return None
    _, exception_type = matched
    runtime_dir = _path_without_symlinks(
        output_dir, "runtime-bound output", exists=True
    )
    bound_evidence = verify_bound_config(runtime_dir)
    config_id = bound_evidence["config_id"]
    if config_id not in _G2_CAPACITY_CONFIG_IDS:
        return None
    attempt_root = _path_without_symlinks(
        bound_evidence["attempt_root"], "capacity stop attempt", exists=True
    )
    if runtime_dir != (attempt_root / "runtime-bound").resolve(strict=True):
        raise RuntimeBindingError("capacity stop runtime-bound directory changed")
    evidence_path = (runtime_dir / EVIDENCE_FILENAME).resolve(strict=True)
    binding_path, binding = _load_binding(
        attempt_root / BINDING_SOURCE_FILENAME
    )
    unsigned = {
        "kind": CAPACITY_STOP_KIND,
        "schema_version": SCHEMA_VERSION,
        "status": "capacity-stop",
        "reason": "cuda_oom",
        "config_id": config_id,
        "attempt_id": bound_evidence["attempt_id"],
        "attempt_root": str(attempt_root),
        "runtime_bound_evidence": {
            "path": str(evidence_path),
            "file_sha256": _sha256_file(evidence_path),
            "evidence_sha256": bound_evidence["evidence_sha256"],
        },
        "runtime_binding": {
            "path": str(binding_path),
            "file_sha256": _sha256_file(binding_path),
            "binding_sha256": binding["binding_sha256"],
        },
        "exception_type": exception_type,
    }
    stop = {
        **unsigned,
        "capacity_stop_sha256": _canonical_sha256(unsigned),
    }
    _atomic_create_json(
        attempt_root / "evidence" / CAPACITY_STOP_FILENAME,
        stop,
    )
    return verify_capacity_stop(attempt_root)


def run_bound_config(output_dir: str | os.PathLike[str]) -> Mapping[str, Any]:
    evidence = verify_bound_config(output_dir)
    config_path = Path(output_dir).resolve(strict=True) / BOUND_CONFIG_FILENAME

    from omegaconf import OmegaConf

    config = OmegaConf.load(config_path)
    OmegaConf.resolve(config)
    from verl.trainer.main_ppo import run_ppo

    run_ppo(config)
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    bind = subparsers.add_parser(
        "bind",
        help="bind one sealed config to a new immutable attempt",
    )
    bind.add_argument("--index", type=Path, required=True, help="sealed index.json")
    bind.add_argument("--config-id", required=True, help="config ID in index.json")
    bind.add_argument(
        "--attempt-dir",
        type=Path,
        required=True,
        help="unique attempt directory; its basename becomes the attempt ID",
    )
    bind.add_argument(
        "--resume-config-id",
        help="sealed predecessor config ID; requires both resume path options",
    )
    bind.add_argument(
        "--resume-checkpoint-dir",
        type=Path,
        help="immutable predecessor global_step_N directory",
    )
    bind.add_argument(
        "--resume-evidence",
        type=Path,
        help="verified evidence file inside the predecessor checkpoint",
    )
    bind.add_argument(
        "--step-zero-reference",
        type=Path,
        help="immutable B-arm fingerprint required by sealed C configs",
    )

    verify = subparsers.add_parser(
        "verify", help="verify an attempt's published runtime-bound config"
    )
    verify.add_argument("--attempt-dir", type=Path, required=True)

    run = subparsers.add_parser(
        "run", help="verify, lazily load the trainer, and run one attempt"
    )
    run.add_argument("--attempt-dir", type=Path, required=True)
    return parser


def _print_json(value: Mapping[str, Any]) -> None:
    print(_canonical_json_bytes(value).decode("ascii"))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "bind":
            result = bind_attempt(
                args.index,
                args.config_id,
                args.attempt_dir,
                resume_config_id=args.resume_config_id,
                resume_checkpoint_dir=args.resume_checkpoint_dir,
                resume_evidence_path=args.resume_evidence,
                step_zero_reference_path=args.step_zero_reference,
            )
        elif args.command == "verify":
            result = verify_bound_config(args.attempt_dir / "runtime-bound")
        else:
            try:
                result = run_bound_config(args.attempt_dir / "runtime-bound")
            except Exception as run_error:
                capacity_stop = publish_capacity_stop_for_exception(
                    args.attempt_dir / "runtime-bound",
                    run_error,
                )
                if capacity_stop is None:
                    raise
                _print_json(capacity_stop)
                return CAPACITY_STOP_EXIT_CODE
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
    "CAPACITY_STOP_EXIT_CODE",
    "RuntimeBindingError",
    "bind_attempt",
    "bind_resolved_config",
    "publish_capacity_stop_for_exception",
    "run_bound_config",
    "verify_bound_config",
    "verify_capacity_stop",
]
