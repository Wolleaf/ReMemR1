"""Strict Transformers-native text-only loading for Qwen3.5 checkpoints.

Qwen3.5 dense checkpoints use a conditional-generation configuration and
contain text, vision, and multi-token-prediction (MTP) weights. Transformers 5
maps both the conditional ``qwen3_5`` config and the text ``qwen3_5_text``
config to ``Qwen3_5ForCausalLM``. This module keeps that native mapping while
making every discarded key and loading option auditable.

For a conditional checkpoint, Transformers' ``qwen3_5_text`` conversion uses
``PrefixChange(prefix_to_remove="language_model", model_prefix="model")``. In concrete
terms, checkpoint keys under ``model.language_model.*`` become text-model keys
under ``model.*``; this is a native conversion, not a local state-dict rewrite.

Transformers is imported inside the loader so CPU-only utilities can import
this module without importing model or accelerator dependencies.
"""

from __future__ import annotations

import importlib
import hashlib
import json
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any


QWEN35_CONDITIONAL_MODEL_TYPE = "qwen3_5"
QWEN35_TEXT_MODEL_TYPE = "qwen3_5_text"
QWEN35_TEXT_ARCHITECTURE = "Qwen3_5ForCausalLM"
QWEN35_MAPPING_METADATA_FILENAME = "qwen35_text_mapping.json"
QWEN35_SNAPSHOT_MANIFEST_FILENAME = "qwen35_snapshot_manifest.json"
QWEN35_SNAPSHOT_MANIFEST_SCHEMA_VERSION = 1
QWEN35_PINNED_REVISIONS: Mapping[str, str] = MappingProxyType(
    {
        "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
        "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
        "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    }
)

# These are the only non-text checkpoint namespaces ignored by the upstream
# Qwen3_5ForCausalLM implementation. Keep the boundary after each prefix so a
# similarly named text key cannot be silently discarded.
QWEN35_ALLOWED_MISSING_KEY_PATTERNS: tuple[str, ...] = ()
QWEN35_ALLOWED_UNEXPECTED_KEY_PATTERNS: tuple[str, ...] = (
    r"^model\.visual(?:\.|$)",
    r"^mtp(?:\.|$)",
)

_FLOATING_REVISIONS = frozenset({"main", "master", "latest", "dev", "develop"})
_COMMIT_HASH_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_HF_CACHE_REPO_PREFIX = "models--"
_SNAPSHOT_MANIFEST_KEYS = frozenset(
    {"schema_version", "model_id", "revision", "files", "manifest_sha256"}
)
_SNAPSHOT_MANIFEST_FILE_KEYS = frozenset({"sha256", "size"})
_MAX_SNAPSHOT_MANIFEST_BYTES = 16 * 1024 * 1024
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_AUTO_CONFIG_FORWARD_KWARGS = frozenset(
    {
        "cache_dir",
        "force_download",
        "local_files_only",
        "proxies",
        "subfolder",
        "token",
    }
)
_RESERVED_MODEL_KWARGS = frozenset(
    {
        "attn_implementation",
        "config",
        "dtype",
        "ignore_mismatched_sizes",
        "output_loading_info",
        "revision",
        "torch_dtype",
        "trust_remote_code",
    }
)
_MISSING = object()


class Qwen35LoaderError(RuntimeError):
    """Base error for a violated Qwen3.5 text-loader contract."""


class Qwen35ConfigError(Qwen35LoaderError):
    """Raised when a config is not a supported dense Qwen3.5 config."""


class Qwen35StateDictError(Qwen35LoaderError):
    """Raised when loading information contains a non-whitelisted key."""


class Qwen35RevisionError(Qwen35LoaderError):
    """Raised when a local checkpoint cannot prove its requested revision."""


@dataclass(frozen=True)
class Qwen35ConfigInfo:
    """Resolved relationship between a checkpoint config and its text model."""

    config_model_type: str
    text_config_model_type: str
    conditional_checkpoint: bool
    source_architectures: tuple[str, ...]
    config_passed_to_auto_model: str
    text_config_path: str
    checkpoint_text_prefix: str
    target_text_prefix: str
    native_prefix_conversion: str


@dataclass(frozen=True)
class Qwen35LoadingReport:
    """Normalized loading keys after strict whitelist validation."""

    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    allowed_missing_keys: tuple[str, ...]
    allowed_unexpected_keys: tuple[str, ...]
    mismatched_keys: tuple[str, ...]
    error_messages: tuple[str, ...]


@dataclass(frozen=True)
class Qwen35RevisionEvidence:
    """Auditable evidence that the loaded bytes belong to a fixed revision."""

    kind: str
    model_id: str
    revision: str
    resolved_snapshot_path: str | None
    manifest_path: str | None
    manifest_sha256: str | None
    verified_file_count: int
    verified_total_bytes: int


@dataclass(frozen=True)
class Qwen35MappingMetadata:
    """JSON-safe metadata needed to reproduce the text-only mapping."""

    schema_version: int
    loader: str
    mapping_strategy: str
    model_name_or_path: str
    revision: str
    revision_evidence: Qwen35RevisionEvidence
    config_model_type: str
    text_config_model_type: str
    conditional_checkpoint: bool
    source_architectures: tuple[str, ...]
    target_architecture: str
    config_passed_to_auto_model: str
    text_config_path: str
    checkpoint_text_prefix: str
    target_text_prefix: str
    native_prefix_conversion: str
    attention_implementation: str
    dtype: str
    transformers_version: str
    strict_loading: bool
    allowed_missing_key_patterns: tuple[str, ...]
    allowed_unexpected_key_patterns: tuple[str, ...]
    ignored_missing_keys: tuple[str, ...]
    ignored_unexpected_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a representation that can be serialized as JSON."""

        return asdict(self)

    def save(self, path: str | Path) -> Path:
        """Save the mapping contract and return the resulting file path."""

        return save_qwen35_mapping_metadata(self, path)


@dataclass(frozen=True)
class Qwen35LoadResult:
    """A loaded model together with its validated loading evidence."""

    model: Any
    loading_info: Mapping[str, Any]
    loading_report: Qwen35LoadingReport
    metadata: Qwen35MappingMetadata

    def save_mapping_metadata(self, path: str | Path) -> Path:
        return self.metadata.save(path)


def _config_value(config: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(config, Mapping):
        if name in config:
            return config[name]
    elif hasattr(config, name):
        return getattr(config, name)

    if default is _MISSING:
        raise Qwen35ConfigError(f"Qwen3.5 config is missing required field {name!r}")
    return default


def _normalize_architectures(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence):
        raise Qwen35ConfigError("config.architectures must be a sequence of strings")
    if not all(isinstance(item, str) and item for item in value):
        raise Qwen35ConfigError("config.architectures contains an invalid architecture name")
    return tuple(value)


def inspect_qwen35_config(config: Any) -> Qwen35ConfigInfo:
    """Identify a dense Qwen3.5 conditional or text-only configuration.

    The full conditional config is deliberately passed to ``AutoModelForCausalLM``.
    Transformers 5 recognizes its ``text_config`` sub-config, selects
    ``Qwen3_5ForCausalLM``, and remaps ``model.language_model.*`` to ``model.*``.
    Passing the nested config directly would bypass part of that native
    compatibility path.
    """

    model_type = _config_value(config, "model_type")
    if not isinstance(model_type, str):
        raise Qwen35ConfigError("config.model_type must be a string")

    architectures = _normalize_architectures(_config_value(config, "architectures", None))
    if model_type == QWEN35_CONDITIONAL_MODEL_TYPE:
        text_config = _config_value(config, "text_config")
        text_model_type = _config_value(text_config, "model_type")
        if text_model_type != QWEN35_TEXT_MODEL_TYPE:
            raise Qwen35ConfigError(
                "qwen3_5 conditional config must contain text_config.model_type="
                f"{QWEN35_TEXT_MODEL_TYPE!r}, got {text_model_type!r}"
            )
        return Qwen35ConfigInfo(
            config_model_type=model_type,
            text_config_model_type=text_model_type,
            conditional_checkpoint=True,
            source_architectures=architectures,
            config_passed_to_auto_model="full_conditional_config",
            text_config_path="text_config",
            checkpoint_text_prefix="model.language_model",
            target_text_prefix="model",
            native_prefix_conversion=(
                'PrefixChange(prefix_to_remove="language_model", model_prefix="model")'
            ),
        )

    if model_type == QWEN35_TEXT_MODEL_TYPE:
        return Qwen35ConfigInfo(
            config_model_type=model_type,
            text_config_model_type=model_type,
            conditional_checkpoint=False,
            source_architectures=architectures,
            config_passed_to_auto_model="text_config",
            text_config_path="$",
            checkpoint_text_prefix="model",
            target_text_prefix="model",
            native_prefix_conversion="identity",
        )

    raise Qwen35ConfigError(
        "text-only Qwen3.5 loader supports model_type 'qwen3_5' or "
        f"'qwen3_5_text', got {model_type!r}"
    )


def _as_key_tuple(values: Any, field_name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise Qwen35StateDictError(f"loading_info[{field_name!r}] must be a sequence")

    normalized: list[str] = []
    for value in values:
        if isinstance(value, str):
            normalized.append(value)
        elif field_name == "mismatched_keys" and isinstance(value, Sequence) and value:
            normalized.append(str(value[0]))
        else:
            raise Qwen35StateDictError(
                f"loading_info[{field_name!r}] contains a non-string entry: {value!r}"
            )
    return tuple(normalized)


def _partition_allowed(keys: tuple[str, ...], patterns: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    compiled = tuple(re.compile(pattern) for pattern in patterns)
    allowed: list[str] = []
    rejected: list[str] = []
    for key in keys:
        destination = allowed if any(pattern.match(key) for pattern in compiled) else rejected
        destination.append(key)
    return tuple(allowed), tuple(rejected)


def validate_qwen35_loading_info(
    loading_info: Mapping[str, Any],
    *,
    allowed_missing_key_patterns: Sequence[str] = QWEN35_ALLOWED_MISSING_KEY_PATTERNS,
    allowed_unexpected_key_patterns: Sequence[str] = QWEN35_ALLOWED_UNEXPECTED_KEY_PATTERNS,
) -> Qwen35LoadingReport:
    """Validate ``from_pretrained(..., output_loading_info=True)`` output.

    By default, no missing text key is valid. Only the exact vision and MTP
    namespaces present in a unified Qwen3.5 checkpoint may be unexpected.
    Mismatched tensors and loader error messages always fail.
    """

    if not isinstance(loading_info, Mapping):
        raise Qwen35StateDictError("loading_info must be a mapping")

    missing = _as_key_tuple(loading_info.get("missing_keys", ()), "missing_keys")
    unexpected = _as_key_tuple(loading_info.get("unexpected_keys", ()), "unexpected_keys")
    mismatched = _as_key_tuple(loading_info.get("mismatched_keys", ()), "mismatched_keys")
    errors = _as_key_tuple(loading_info.get("error_msgs", ()), "error_msgs")

    allowed_missing, rejected_missing = _partition_allowed(missing, allowed_missing_key_patterns)
    allowed_unexpected, rejected_unexpected = _partition_allowed(unexpected, allowed_unexpected_key_patterns)

    violations: list[str] = []
    if rejected_missing:
        violations.append(f"missing keys: {list(rejected_missing)!r}")
    if rejected_unexpected:
        violations.append(f"unexpected keys: {list(rejected_unexpected)!r}")
    if mismatched:
        violations.append(f"mismatched keys: {list(mismatched)!r}")
    if errors:
        violations.append(f"loader errors: {list(errors)!r}")
    if violations:
        raise Qwen35StateDictError("Qwen3.5 text-only strict load failed; " + "; ".join(violations))

    return Qwen35LoadingReport(
        missing_keys=missing,
        unexpected_keys=unexpected,
        allowed_missing_keys=allowed_missing,
        allowed_unexpected_keys=allowed_unexpected,
        mismatched_keys=mismatched,
        error_messages=errors,
    )


def _require_fixed_revision(model_name_or_path: str, revision: str) -> str:
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("revision is required and must identify a fixed model snapshot")
    normalized = revision.strip()
    if normalized.casefold() in _FLOATING_REVISIONS:
        raise ValueError(
            f"revision {normalized!r} is floating; use an immutable commit hash or release tag"
        )
    pinned_revision = QWEN35_PINNED_REVISIONS.get(model_name_or_path)
    if pinned_revision is not None and normalized != pinned_revision:
        raise ValueError(
            f"revision for {model_name_or_path!r} must be the reproduction-pinned "
            f"commit {pinned_revision!r}, got {normalized!r}"
        )
    return normalized


def _canonical_manifest_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _raise_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _reject_duplicate_json_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_snapshot_manifest(path: Path) -> Mapping[str, Any]:
    try:
        manifest_stat = path.stat()
    except OSError as exc:
        raise Qwen35RevisionError(f"cannot stat snapshot manifest {str(path)!r}") from exc
    if path.is_symlink() or not stat.S_ISREG(manifest_stat.st_mode):
        raise Qwen35RevisionError("snapshot manifest must be a regular, non-symlink file")
    if manifest_stat.st_size > _MAX_SNAPSHOT_MANIFEST_BYTES:
        raise Qwen35RevisionError(
            f"snapshot manifest exceeds {_MAX_SNAPSHOT_MANIFEST_BYTES} bytes"
        )

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_raise_json_constant,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise Qwen35RevisionError(f"invalid snapshot manifest {str(path)!r}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise Qwen35RevisionError("snapshot manifest root must be a JSON object")
    return payload


def _manifest_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise Qwen35RevisionError("snapshot manifest file paths must be non-empty strings")
    if "\\" in value or "\x00" in value or ":" in value:
        raise Qwen35RevisionError(f"unsafe snapshot manifest file path {value!r}")
    normalized = PurePosixPath(value)
    if (
        normalized.is_absolute()
        or normalized.as_posix() != value
        or any(part in {"", ".", ".."} for part in normalized.parts)
    ):
        raise Qwen35RevisionError(f"unsafe snapshot manifest file path {value!r}")
    if value == QWEN35_SNAPSHOT_MANIFEST_FILENAME:
        raise Qwen35RevisionError("snapshot manifest cannot include its own file hash")
    return value


def _snapshot_regular_files(snapshot_root: Path, manifest_path: Path) -> Mapping[str, Path]:
    files: dict[str, Path] = {}
    try:
        candidates = tuple(snapshot_root.rglob("*"))
    except OSError as exc:
        raise Qwen35RevisionError(
            f"cannot enumerate local snapshot {str(snapshot_root)!r}"
        ) from exc

    for candidate in candidates:
        if candidate == manifest_path:
            continue
        if candidate.is_symlink():
            raise Qwen35RevisionError(
                "manifest-backed snapshots must be self-contained; symlink found at "
                f"{str(candidate)!r}"
            )
        try:
            candidate_stat = candidate.stat()
        except OSError as exc:
            raise Qwen35RevisionError(f"cannot stat snapshot entry {str(candidate)!r}") from exc
        if stat.S_ISDIR(candidate_stat.st_mode):
            continue
        if not stat.S_ISREG(candidate_stat.st_mode):
            raise Qwen35RevisionError(
                f"snapshot contains non-regular entry {str(candidate)!r}"
            )
        relative = candidate.relative_to(snapshot_root).as_posix()
        files[relative] = candidate
    return files


def _sha256_file(path: Path, expected_size: int) -> str:
    try:
        before = path.stat()
        if not stat.S_ISREG(before.st_mode) or path.is_symlink():
            raise Qwen35RevisionError(
                f"snapshot file must remain regular and non-symlink: {str(path)!r}"
            )
        if before.st_size != expected_size:
            raise Qwen35RevisionError(
                f"snapshot file size mismatch for {str(path)!r}: "
                f"expected {expected_size}, got {before.st_size}"
            )

        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
        after = path.stat()
    except Qwen35RevisionError:
        raise
    except OSError as exc:
        raise Qwen35RevisionError(f"cannot hash snapshot file {str(path)!r}") from exc

    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise Qwen35RevisionError(f"snapshot file changed while hashing: {str(path)!r}")
    return digest.hexdigest()


def _validate_snapshot_manifest(
    snapshot_root: Path,
    *,
    revision: str,
) -> Qwen35RevisionEvidence:
    manifest_path = snapshot_root / QWEN35_SNAPSHOT_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise Qwen35RevisionError(
            "cannot prove revision for local Qwen3.5 directory; use a resolved Hugging Face "
            f"models--<org>--<repo>/snapshots/<commit> path or include "
            f"{QWEN35_SNAPSHOT_MANIFEST_FILENAME!r}"
        )
    payload = _read_snapshot_manifest(manifest_path)

    actual_keys = frozenset(payload)
    if actual_keys != _SNAPSHOT_MANIFEST_KEYS:
        missing = sorted(_SNAPSHOT_MANIFEST_KEYS - actual_keys)
        unknown = sorted(actual_keys - _SNAPSHOT_MANIFEST_KEYS)
        raise Qwen35RevisionError(
            f"snapshot manifest schema keys mismatch; missing={missing!r}, unknown={unknown!r}"
        )
    if payload["schema_version"] != QWEN35_SNAPSHOT_MANIFEST_SCHEMA_VERSION:
        raise Qwen35RevisionError(
            "unsupported snapshot manifest schema_version "
            f"{payload['schema_version']!r}"
        )

    model_id = payload["model_id"]
    if not isinstance(model_id, str) or model_id not in QWEN35_PINNED_REVISIONS:
        raise Qwen35RevisionError(
            "snapshot manifest model_id must be an official pinned Qwen3.5 model"
        )
    manifest_revision = payload["revision"]
    if manifest_revision != revision:
        raise Qwen35RevisionError(
            "snapshot manifest revision does not match requested revision: "
            f"{manifest_revision!r} != {revision!r}"
        )
    pinned_revision = QWEN35_PINNED_REVISIONS[model_id]
    if manifest_revision != pinned_revision:
        raise Qwen35RevisionError(
            f"snapshot manifest for {model_id!r} must use pinned commit "
            f"{pinned_revision!r}, got {manifest_revision!r}"
        )

    manifest_sha256 = payload["manifest_sha256"]
    if not isinstance(manifest_sha256, str) or not _SHA256_PATTERN.fullmatch(
        manifest_sha256
    ):
        raise Qwen35RevisionError("snapshot manifest manifest_sha256 must be lowercase SHA-256")
    unsigned_payload = dict(payload)
    del unsigned_payload["manifest_sha256"]
    computed_manifest_sha256 = hashlib.sha256(
        _canonical_manifest_bytes(unsigned_payload)
    ).hexdigest()
    if computed_manifest_sha256 != manifest_sha256:
        raise Qwen35RevisionError(
            "snapshot manifest self-hash mismatch: "
            f"expected {manifest_sha256}, computed {computed_manifest_sha256}"
        )

    file_entries = payload["files"]
    if not isinstance(file_entries, Mapping) or not file_entries:
        raise Qwen35RevisionError("snapshot manifest files must be a non-empty object")
    normalized_entries: dict[str, tuple[str, int]] = {}
    for raw_path, raw_entry in file_entries.items():
        relative_path = _manifest_relative_path(raw_path)
        if not isinstance(raw_entry, Mapping):
            raise Qwen35RevisionError(
                f"snapshot manifest entry for {relative_path!r} must be an object"
            )
        entry_keys = frozenset(raw_entry)
        if entry_keys != _SNAPSHOT_MANIFEST_FILE_KEYS:
            raise Qwen35RevisionError(
                f"snapshot manifest entry keys mismatch for {relative_path!r}"
            )
        sha256 = raw_entry["sha256"]
        size = raw_entry["size"]
        if not isinstance(sha256, str) or not _SHA256_PATTERN.fullmatch(sha256):
            raise Qwen35RevisionError(
                f"snapshot manifest sha256 for {relative_path!r} must be lowercase SHA-256"
            )
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise Qwen35RevisionError(
                f"snapshot manifest size for {relative_path!r} must be a non-negative integer"
            )
        normalized_entries[relative_path] = (sha256, size)

    if "config.json" not in normalized_entries:
        raise Qwen35RevisionError("snapshot manifest must include config.json")
    has_model_weights = any(
        PurePosixPath(relative_path).parent == PurePosixPath(".")
        and (
            PurePosixPath(relative_path).name == "model.safetensors"
            or re.fullmatch(
                r"model-\d+-of-\d+\.safetensors", PurePosixPath(relative_path).name
            )
        )
        for relative_path in normalized_entries
    )
    if not has_model_weights:
        raise Qwen35RevisionError(
            "snapshot manifest must include model.safetensors or sharded model safetensors"
        )

    actual_files = _snapshot_regular_files(snapshot_root, manifest_path)
    declared_paths = frozenset(normalized_entries)
    actual_paths = frozenset(actual_files)
    if declared_paths != actual_paths:
        missing = sorted(declared_paths - actual_paths)
        undeclared = sorted(actual_paths - declared_paths)
        raise Qwen35RevisionError(
            "snapshot manifest file inventory mismatch; "
            f"missing={missing!r}, undeclared={undeclared!r}"
        )

    total_bytes = 0
    for relative_path in sorted(normalized_entries):
        expected_sha256, expected_size = normalized_entries[relative_path]
        actual_sha256 = _sha256_file(actual_files[relative_path], expected_size)
        if actual_sha256 != expected_sha256:
            raise Qwen35RevisionError(
                f"snapshot file SHA-256 mismatch for {relative_path!r}: "
                f"expected {expected_sha256}, computed {actual_sha256}"
            )
        total_bytes += expected_size

    return Qwen35RevisionEvidence(
        kind="verified_snapshot_manifest",
        model_id=model_id,
        revision=revision,
        resolved_snapshot_path=str(snapshot_root),
        manifest_path=str(manifest_path),
        manifest_sha256=manifest_sha256,
        verified_file_count=len(normalized_entries),
        verified_total_bytes=total_bytes,
    )


def _hf_cache_snapshot_evidence(
    snapshot_root: Path,
    *,
    revision: str,
) -> Qwen35RevisionEvidence | None:
    if snapshot_root.parent.name != "snapshots":
        return None
    cache_repo_name = snapshot_root.parent.parent.name
    if not cache_repo_name.startswith(_HF_CACHE_REPO_PREFIX):
        return None

    matching_model_ids = [
        model_id
        for model_id in QWEN35_PINNED_REVISIONS
        if cache_repo_name == _HF_CACHE_REPO_PREFIX + model_id.replace("/", "--")
    ]
    if len(matching_model_ids) != 1:
        raise Qwen35RevisionError(
            f"unrecognized Qwen3.5 Hugging Face cache repository {cache_repo_name!r}"
        )
    model_id = matching_model_ids[0]
    path_revision = snapshot_root.name
    if not _COMMIT_HASH_PATTERN.fullmatch(path_revision):
        raise Qwen35RevisionError(
            f"Hugging Face snapshot directory must be a 40-hex commit, got {path_revision!r}"
        )
    if path_revision != revision:
        raise Qwen35RevisionError(
            "resolved Hugging Face snapshot commit does not match requested revision: "
            f"{path_revision!r} != {revision!r}"
        )
    pinned_revision = QWEN35_PINNED_REVISIONS[model_id]
    if revision != pinned_revision:
        raise Qwen35RevisionError(
            f"local snapshot for {model_id!r} must use pinned commit "
            f"{pinned_revision!r}, got {revision!r}"
        )
    return Qwen35RevisionEvidence(
        kind="hf_cache_snapshot_path",
        model_id=model_id,
        revision=revision,
        resolved_snapshot_path=str(snapshot_root),
        manifest_path=None,
        manifest_sha256=None,
        verified_file_count=0,
        verified_total_bytes=0,
    )


def _resolve_local_source(
    model_name_or_path: str | Path,
    model_ref: str,
) -> Path | None:
    candidate = Path(model_ref).expanduser()
    explicitly_local = isinstance(model_name_or_path, Path)
    explicitly_local = explicitly_local or candidate.is_absolute()
    explicitly_local = explicitly_local or model_ref.startswith((".", "~", "\\"))
    explicitly_local = explicitly_local or "\\" in model_ref or model_ref.count("/") > 1
    explicitly_local = explicitly_local or candidate.is_symlink()

    if "://" in model_ref:
        raise Qwen35RevisionError(
            f"Qwen3.5 loader requires a Hub model ID or local directory, got URI {model_ref!r}; "
            "copy HDFS snapshots locally with their manifest first"
        )
    if not candidate.exists():
        if explicitly_local:
            raise Qwen35RevisionError(f"local Qwen3.5 path does not exist: {model_ref!r}")
        return None
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise Qwen35RevisionError(f"cannot resolve local Qwen3.5 path {model_ref!r}") from exc
    if not resolved.is_dir():
        raise Qwen35RevisionError(f"local Qwen3.5 path must be a directory: {model_ref!r}")
    return resolved


def validate_qwen35_revision_source(
    model_name_or_path: str | Path,
    revision: str,
) -> Qwen35RevisionEvidence:
    """Prove that a Hub request or local snapshot resolves to ``revision``.

    Transformers ignores ``revision`` for local directories. Such paths are
    therefore accepted only when their resolved Hugging Face cache path embeds
    the pinned commit, or when every local file is covered by the strict,
    self-hashed snapshot manifest used for HDFS exports.
    """

    model_ref = str(model_name_or_path)
    if not model_ref:
        raise ValueError("model_name_or_path must not be empty")
    fixed_revision = _require_fixed_revision(model_ref, revision)
    local_source = _resolve_local_source(model_name_or_path, model_ref)
    if local_source is not None:
        cache_evidence = _hf_cache_snapshot_evidence(
            local_source,
            revision=fixed_revision,
        )
        if cache_evidence is not None:
            return cache_evidence
        return _validate_snapshot_manifest(local_source, revision=fixed_revision)

    return Qwen35RevisionEvidence(
        kind="hub_revision_request",
        model_id=model_ref,
        revision=fixed_revision,
        resolved_snapshot_path=None,
        manifest_path=None,
        manifest_sha256=None,
        verified_file_count=0,
        verified_total_bytes=0,
    )


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be provided explicitly")
    return value.strip()


def _dtype_name(dtype: Any) -> str:
    value = str(dtype)
    return value.removeprefix("torch.")


def _load_transformers_api() -> tuple[Any, Any, str]:
    try:
        transformers = importlib.import_module("transformers")
        auto_config = getattr(transformers, "AutoConfig")
        auto_model = getattr(transformers, "AutoModelForCausalLM")
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "Qwen3.5 text loading requires a Transformers 5 build with "
            "AutoModelForCausalLM Qwen3.5 support"
        ) from exc
    return auto_config, auto_model, str(getattr(transformers, "__version__", "unknown"))


def load_qwen35_text_model(
    model_name_or_path: str | Path,
    *,
    revision: str,
    attn_implementation: str,
    dtype: Any,
    strict: bool = True,
    config: Any | None = None,
    **from_pretrained_kwargs: Any,
) -> Qwen35LoadResult:
    """Load a Qwen3.5 text CausalLM through the native Transformers 5 path.

    ``revision``, ``attn_implementation``, and ``dtype`` are mandatory and are
    forwarded unchanged. The loader always requests loading information and
    applies the fixed strict whitelist; callers cannot disable this behavior.
    """

    model_ref = str(model_name_or_path)
    revision_evidence = validate_qwen35_revision_source(model_name_or_path, revision)
    fixed_revision = revision_evidence.revision
    attention = _require_nonempty_string(attn_implementation, "attn_implementation")
    if dtype is None:
        raise ValueError("dtype must be provided explicitly")
    if strict is not True:
        raise ValueError("strict Qwen3.5 key validation cannot be disabled")

    conflicting = sorted(_RESERVED_MODEL_KWARGS.intersection(from_pretrained_kwargs))
    if conflicting:
        raise ValueError(f"reserved from_pretrained kwargs cannot be overridden: {conflicting!r}")

    AutoConfig, AutoModelForCausalLM, transformers_version = _load_transformers_api()
    if config is None:
        config_kwargs = {
            key: value
            for key, value in from_pretrained_kwargs.items()
            if key in _AUTO_CONFIG_FORWARD_KWARGS
        }
        config = AutoConfig.from_pretrained(
            model_ref,
            revision=fixed_revision,
            trust_remote_code=False,
            **config_kwargs,
        )

    config_info = inspect_qwen35_config(config)
    loaded = AutoModelForCausalLM.from_pretrained(
        model_ref,
        config=config,
        revision=fixed_revision,
        attn_implementation=attention,
        dtype=dtype,
        trust_remote_code=False,
        output_loading_info=True,
        **from_pretrained_kwargs,
    )
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise Qwen35LoaderError(
            "Transformers did not return (model, loading_info) despite output_loading_info=True"
        )
    model, loading_info = loaded
    report = validate_qwen35_loading_info(loading_info)

    loaded_config = getattr(model, "config", None)
    loaded_model_type = _config_value(loaded_config, "model_type", None) if loaded_config is not None else None
    if loaded_model_type is not None and loaded_model_type != QWEN35_TEXT_MODEL_TYPE:
        raise Qwen35ConfigError(
            "AutoModelForCausalLM resolved a non-text config after loading: "
            f"{loaded_model_type!r}"
        )

    metadata = Qwen35MappingMetadata(
        schema_version=2,
        loader="transformers.AutoModelForCausalLM",
        mapping_strategy="transformers_native_qwen35_causal_lm",
        model_name_or_path=model_ref,
        revision=fixed_revision,
        revision_evidence=revision_evidence,
        config_model_type=config_info.config_model_type,
        text_config_model_type=config_info.text_config_model_type,
        conditional_checkpoint=config_info.conditional_checkpoint,
        source_architectures=config_info.source_architectures,
        target_architecture=QWEN35_TEXT_ARCHITECTURE,
        config_passed_to_auto_model=config_info.config_passed_to_auto_model,
        text_config_path=config_info.text_config_path,
        checkpoint_text_prefix=config_info.checkpoint_text_prefix,
        target_text_prefix=config_info.target_text_prefix,
        native_prefix_conversion=config_info.native_prefix_conversion,
        attention_implementation=attention,
        dtype=_dtype_name(dtype),
        transformers_version=transformers_version,
        strict_loading=True,
        allowed_missing_key_patterns=QWEN35_ALLOWED_MISSING_KEY_PATTERNS,
        allowed_unexpected_key_patterns=QWEN35_ALLOWED_UNEXPECTED_KEY_PATTERNS,
        ignored_missing_keys=report.allowed_missing_keys,
        ignored_unexpected_keys=report.allowed_unexpected_keys,
    )
    return Qwen35LoadResult(
        model=model,
        loading_info=loading_info,
        loading_report=report,
        metadata=metadata,
    )


def save_qwen35_mapping_metadata(metadata: Qwen35MappingMetadata, path: str | Path) -> Path:
    """Persist mapping metadata in a deterministic, human-readable form."""

    if not isinstance(metadata, Qwen35MappingMetadata):
        raise TypeError("metadata must be a Qwen35MappingMetadata instance")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metadata.to_dict(), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return output_path


__all__ = [
    "QWEN35_ALLOWED_MISSING_KEY_PATTERNS",
    "QWEN35_ALLOWED_UNEXPECTED_KEY_PATTERNS",
    "QWEN35_CONDITIONAL_MODEL_TYPE",
    "QWEN35_MAPPING_METADATA_FILENAME",
    "QWEN35_PINNED_REVISIONS",
    "QWEN35_SNAPSHOT_MANIFEST_FILENAME",
    "QWEN35_SNAPSHOT_MANIFEST_SCHEMA_VERSION",
    "QWEN35_TEXT_ARCHITECTURE",
    "QWEN35_TEXT_MODEL_TYPE",
    "Qwen35ConfigError",
    "Qwen35ConfigInfo",
    "Qwen35LoadResult",
    "Qwen35LoaderError",
    "Qwen35LoadingReport",
    "Qwen35MappingMetadata",
    "Qwen35RevisionError",
    "Qwen35RevisionEvidence",
    "Qwen35StateDictError",
    "inspect_qwen35_config",
    "load_qwen35_text_model",
    "save_qwen35_mapping_metadata",
    "validate_qwen35_revision_source",
    "validate_qwen35_loading_info",
]
