"""Fail-closed checkpoint and PEFT export contracts for reproduction runs.

This module intentionally has no eager Torch, Transformers, or PEFT imports.
Training code can therefore use its schema and filesystem helpers during CPU
preflight, while model-dependent operations import their dependencies lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import base64
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any, Callable, ClassVar, Mapping, Optional, Sequence


CHECKPOINT_EXTRA_STATE_SCHEMA_VERSION = 2
ADAPTER_EXPORT_METADATA_SCHEMA_VERSION = 2
COMPLETION_MANIFEST_SCHEMA_VERSION = 2
REPRODUCTION_RANK_EXTRA_SCHEMA_VERSION = 1
MERGED_MODEL_METADATA_SCHEMA_VERSION = 1

EXTRA_STATE_FILENAME = "reproduction_extra_state.json"
ADAPTER_METADATA_FILENAME = "reproduction_adapter_metadata.json"
MERGED_MODEL_METADATA_FILENAME = "reproduction_merge_metadata.json"
COMPLETION_MARKER_FILENAME = "_REPRODUCTION_COMPLETE.json"

_FLOATING_REVISIONS = frozenset({"main", "master", "latest", "head"})
_HEX_DIGITS = frozenset("0123456789abcdef")
_PRUNE_AUTHORIZATION_TOKEN = object()
_MERGED_MODEL_ARTIFACT_TYPE = "merged_causal_lm"
_MERGED_MODEL_METADATA_PAYLOAD_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "global_step",
        "source_adapter_metadata_sha256",
        "base_model_id",
        "base_model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "template_revision",
        "text_mapping_sha256",
        "dtype",
        "model_state_sha256",
        "verification_prompt",
        "max_new_tokens",
        "rtol",
        "atol",
    }
)

DEFAULT_RESUME_CONFIG_ALLOWLIST = (
    "actor_rollout_ref.actor.optim.total_training_steps",
    "critic.optim.total_training_steps",
    "trainer.total_training_steps",
    "trainer.resume_mode",
    "trainer.resume_from_path",
    "trainer.default_local_dir",
    "trainer.experiment_name",
    "trainer.rollout_data_dir",
    "trainer.validation_data_dir",
    "reproduction.export_adapter_on_save",
    "reproduction.adapter_export_dir",
)


class CheckpointContractError(ValueError):
    """Raised when checkpoint metadata violates the reproduction contract."""


class IncompleteCheckpointError(CheckpointContractError):
    """Raised when a directory lacks a valid completion manifest."""


class CheckpointVerificationError(CheckpointContractError):
    """Raised when a checkpoint does not pass the explicit prune gate."""


def _normalize_json(value: Any, *, path: str = "$") -> Any:
    """Return a detached JSON value while rejecting ambiguous encodings."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CheckpointContractError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CheckpointContractError(f"{path} contains a non-string mapping key")
            normalized[key] = _normalize_json(item, path=f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise CheckpointContractError(
        f"{path} contains unsupported JSON value {type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a value using the one canonical representation used for hashes."""

    normalized = _normalize_json(value)
    return json.dumps(
        normalized,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    """Hash a strict canonical JSON value."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def to_json_safe_state(value: Any, *, path: str = "$") -> Any:
    """Convert tensors, arrays, bytes, and RNG tuples into canonical JSON data."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return _normalize_json(value, path=path)
    if isinstance(value, bytes):
        return {"type": "bytes", "base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Mapping):
        converted = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CheckpointContractError(f"{path} contains a non-string mapping key")
            converted[key] = to_json_safe_state(item, path=f"{path}.{key}")
        return converted
    if isinstance(value, (list, tuple)):
        return [
            to_json_safe_state(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    detach = getattr(value, "detach", None)
    cpu = getattr(value, "cpu", None)
    tolist = getattr(value, "tolist", None)
    if callable(detach) and callable(cpu) and callable(tolist):
        tensor = value.detach().cpu()
        return {
            "type": "tensor",
            "dtype": str(getattr(tensor, "dtype", type(tensor).__name__)),
            "shape": [int(dimension) for dimension in getattr(tensor, "shape", ())],
            "values": to_json_safe_state(tensor.tolist(), path=f"{path}.values"),
        }

    if callable(tolist):
        return {
            "type": "array",
            "dtype": str(getattr(value, "dtype", type(value).__name__)),
            "shape": [int(dimension) for dimension in getattr(value, "shape", ())],
            "values": to_json_safe_state(value.tolist(), path=f"{path}.values"),
        }

    item = getattr(value, "item", None)
    if callable(item):
        scalar = item()
        if scalar is not value:
            return to_json_safe_state(scalar, path=path)
    raise CheckpointContractError(
        f"{path} contains unsupported state value {type(value).__name__}"
    )


def json_safe_state_sha256(value: Any) -> str:
    """Hash an arbitrary raw state after deterministic JSON-safe conversion."""

    return canonical_json_sha256(to_json_safe_state(value))


def _require_sorted_unique_strings(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise CheckpointContractError(f"{name} must be a non-empty sequence")
    normalized = tuple(value)
    if any(not isinstance(item, str) or not item for item in normalized):
        raise CheckpointContractError(f"{name} must contain non-empty strings")
    if normalized != tuple(sorted(set(normalized))):
        raise CheckpointContractError(f"{name} must be sorted and unique")
    return normalized


def _tensor_storage_bytes(tensor: Any) -> bytes:
    value = tensor
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    contiguous = getattr(value, "contiguous", None)
    if callable(contiguous):
        value = contiguous()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        try:
            return numpy().tobytes(order="C")
        except (RuntimeError, TypeError):
            try:
                import torch

                return value.view(torch.uint8).numpy().tobytes(order="C")
            except (ImportError, RuntimeError, TypeError, AttributeError):
                pass
    tobytes = getattr(value, "tobytes", None)
    if callable(tobytes):
        return tobytes()
    raise CheckpointContractError(
        f"adapter tensor {type(tensor).__name__} does not expose readable bytes"
    )


def canonical_tensor_state_sha256(state_dict: Mapping[str, Any]) -> str:
    """Hash exact tensor keys, metadata, and storage bytes in canonical order."""

    if not isinstance(state_dict, Mapping) or not state_dict:
        raise CheckpointContractError("adapter tensor state must be a non-empty mapping")
    keys = tuple(sorted(state_dict))
    _require_sorted_unique_strings(keys, name="adapter tensor keys")
    metadata = []
    for key in keys:
        tensor = state_dict[key]
        shape = getattr(tensor, "shape", None)
        if shape is None:
            raise CheckpointContractError(f"adapter tensor {key!r} has no shape")
        metadata.append(
            {
                "key": key,
                "shape": [int(dimension) for dimension in shape],
                "dtype": str(getattr(tensor, "dtype", type(tensor).__name__)),
            }
        )
    hasher = hashlib.sha256()
    hasher.update(
        canonical_json_bytes({"schema_version": 1, "tensors": metadata})
    )
    for key in keys:
        payload = _tensor_storage_bytes(state_dict[key])
        encoded_key = key.encode("utf-8")
        hasher.update(len(encoded_key).to_bytes(8, "big"))
        hasher.update(encoded_key)
        hasher.update(len(payload).to_bytes(8, "big"))
        hasher.update(payload)
    return hasher.hexdigest()


def _remove_config_path(config: dict[str, Any], dotted_path: str) -> None:
    parts = dotted_path.split(".")
    current: Any = config
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return
        current = current[part]
    if isinstance(current, dict):
        current.pop(parts[-1], None)


def normalize_resolved_config(
    config: Mapping[str, Any],
    *,
    allowed_drift_paths: Sequence[str] = DEFAULT_RESUME_CONFIG_ALLOWLIST,
) -> dict[str, Any]:
    """Remove only explicit orchestration fields before resume comparison."""

    normalized = _require_nonempty_mapping(config, name="resolved_config")
    for dotted_path in allowed_drift_paths:
        if not isinstance(dotted_path, str) or not dotted_path.strip():
            raise CheckpointContractError(
                "allowed resolved-config drift paths must be non-empty strings"
            )
        _remove_config_path(normalized, dotted_path)
    return normalized


def _config_difference_paths(saved: Any, current: Any, *, path: str = "$") -> list[str]:
    if isinstance(saved, dict) and isinstance(current, dict):
        differences = []
        for key in sorted(set(saved) | set(current)):
            child_path = f"{path}.{key}"
            if key not in saved or key not in current:
                differences.append(child_path)
            else:
                differences.extend(
                    _config_difference_paths(saved[key], current[key], path=child_path)
                )
        return differences
    if isinstance(saved, list) and isinstance(current, list):
        differences = []
        for index in range(max(len(saved), len(current))):
            child_path = f"{path}[{index}]"
            if index >= len(saved) or index >= len(current):
                differences.append(child_path)
            else:
                differences.extend(
                    _config_difference_paths(
                        saved[index],
                        current[index],
                        path=child_path,
                    )
                )
        return differences
    return [] if saved == current else [path]


def validate_resolved_config_compatibility(
    saved_config: Mapping[str, Any],
    current_config: Mapping[str, Any],
    *,
    allowed_drift_paths: Sequence[str] = DEFAULT_RESUME_CONFIG_ALLOWLIST,
) -> None:
    """Fail closed on every config drift outside the segmented-run allowlist."""

    normalized_saved = normalize_resolved_config(
        saved_config,
        allowed_drift_paths=allowed_drift_paths,
    )
    normalized_current = normalize_resolved_config(
        current_config,
        allowed_drift_paths=allowed_drift_paths,
    )
    if normalized_saved == normalized_current:
        return
    differences = _config_difference_paths(normalized_saved, normalized_current)
    raise CheckpointContractError(
        "resolved config drift is not allowed for resume; changed paths="
        + json.dumps(differences, ensure_ascii=True)
    )


def capture_process_rng_state() -> dict[str, Any]:
    """Capture restorable Python, NumPy, Torch CPU, and all CUDA RNG states."""

    import random

    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("NumPy is required for reproduction RNG checkpoints") from exc
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Torch is required for reproduction RNG checkpoints") from exc

    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_states,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_process_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a state produced by :func:`capture_process_rng_state`."""

    import random

    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("NumPy is required for reproduction RNG checkpoints") from exc
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Torch is required for reproduction RNG checkpoints") from exc

    if not isinstance(state, Mapping):
        raise CheckpointContractError("raw RNG state must be a mapping")
    _require_exact_keys(
        state,
        {"torch_cpu", "torch_cuda", "numpy", "python"},
        label="raw RNG state",
    )
    if state["torch_cpu"] is None or state["numpy"] is None or state["python"] is None:
        raise CheckpointContractError("raw RNG state contains a missing required state")
    cuda_states = state["torch_cuda"]
    if not isinstance(cuda_states, (list, tuple)):
        raise CheckpointContractError("raw RNG torch_cuda state must be a sequence")

    torch.set_rng_state(state["torch_cpu"])
    if cuda_states:
        if not torch.cuda.is_available():
            raise CheckpointContractError(
                "checkpoint contains CUDA RNG state but CUDA is unavailable"
            )
        if len(cuda_states) != torch.cuda.device_count():
            raise CheckpointContractError(
                "checkpoint CUDA RNG state count does not match visible devices"
            )
        torch.cuda.set_rng_state_all(list(cuda_states))
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if any(not isinstance(key, str) for key in value):
        raise CheckpointContractError(f"{label} must contain only string keys")
    keys = set(value)
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        raise CheckpointContractError(
            f"{label} keys do not match schema; missing={missing}, unknown={unknown}"
        )


def _require_schema_version(value: Any, expected: int, *, label: str) -> None:
    if type(value) is not int or value != expected:
        raise CheckpointContractError(f"unsupported {label} schema {value!r}")


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CheckpointContractError(f"{name} must be a non-empty string")
    if value != value.strip():
        raise CheckpointContractError(f"{name} must not contain surrounding whitespace")
    return value


def _require_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CheckpointContractError(f"{name} must be a non-negative int")
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise CheckpointContractError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def validate_reproduction_manifest_configuration(
    *,
    train_sha256: Any,
    train_mode: Any,
    train_profile: Any,
    validation_sha256: Any,
    validation_mode: Any,
    validation_profile: Any,
    formal_data: Any,
    total_training_steps: Any = None,
) -> None:
    """Validate the configured train/validation bundle identities before loading data."""

    if not isinstance(formal_data, bool):
        raise CheckpointContractError("reproduction.formal_data must be boolean")
    train_digest = _require_sha256(
        train_sha256,
        name="reproduction.data_manifest_sha256",
    )
    validation_digest = _require_sha256(
        validation_sha256,
        name="reproduction.val_data_manifest_sha256",
    )
    if train_digest == validation_digest:
        raise CheckpointContractError(
            "reproduction train and validation manifests must be distinct bundles"
        )

    identities = (
        ("train", train_mode, train_profile),
        ("validation", validation_mode, validation_profile),
    )
    for label, mode, profile in identities:
        if mode != "train":
            raise CheckpointContractError(
                f"reproduction {label} manifest mode must be train"
            )
        if profile not in {"fixture", "formal"}:
            raise CheckpointContractError(
                f"reproduction {label} manifest profile must be fixture or formal"
            )
    if formal_data and any(profile != "formal" for _, _, profile in identities):
        raise CheckpointContractError(
            "reproduction.formal_data=true requires formal train and validation bundles"
        )
    if (
        isinstance(total_training_steps, int)
        and not isinstance(total_training_steps, bool)
        and total_training_steps >= 40
        and not formal_data
    ):
        raise CheckpointContractError(
            "40/80-step reproduction jobs require formal_data=true"
        )


def validate_bound_dataset_manifest(
    dataset: Any,
    *,
    configured_sha256: Any,
    configured_mode: Any,
    configured_profile: Any,
    label: str,
) -> None:
    """Bind one loaded dataset to exactly one configured sealed manifest."""

    digest = _require_sha256(
        configured_sha256,
        name=f"reproduction {label} manifest SHA-256",
    )
    mode = _require_nonempty_string(
        configured_mode,
        name=f"reproduction {label} manifest mode",
    )
    profile = _require_nonempty_string(
        configured_profile,
        name=f"reproduction {label} manifest profile",
    )
    bundle_hashes = getattr(dataset, "bundle_manifest_sha256s", None)
    metadata = getattr(dataset, "bundle_manifest_metadata", None)
    if bundle_hashes != (digest,) or not isinstance(metadata, tuple) or len(metadata) != 1:
        raise CheckpointContractError(
            f"reproduction {label} dataset must bind exactly its configured manifest"
        )
    actual = metadata[0]
    if not isinstance(actual, Mapping) or (
        actual.get("manifest_sha256") != digest
        or actual.get("mode") != mode
        or actual.get("profile") != profile
    ):
        raise CheckpointContractError(
            f"reproduction {label} dataset mode/profile/hash contract mismatch"
        )


def _require_pinned_revision(value: Any, *, name: str) -> str:
    revision = _require_nonempty_string(value, name=name)
    if revision.casefold() in _FLOATING_REVISIONS:
        raise CheckpointContractError(
            f"{name} {revision!r} is floating; use an immutable commit or release tag"
        )
    return revision


def validate_merged_model_metadata(value: Any) -> dict[str, Any]:
    """Validate and normalize the self-hashed merged-model identity record."""

    if not isinstance(value, Mapping):
        raise CheckpointContractError("merged model metadata must be a mapping")
    _require_exact_keys(
        value,
        _MERGED_MODEL_METADATA_PAYLOAD_KEYS | {"metadata_sha256"},
        label="merged model metadata",
    )
    _require_schema_version(
        value["schema_version"],
        MERGED_MODEL_METADATA_SCHEMA_VERSION,
        label="merged model metadata",
    )
    if value["artifact_type"] != _MERGED_MODEL_ARTIFACT_TYPE:
        raise CheckpointContractError(
            f"unsupported merged model artifact type {value['artifact_type']!r}"
        )
    _require_nonnegative_int(value["global_step"], name="merged global_step")
    for name in (
        "source_adapter_metadata_sha256",
        "text_mapping_sha256",
        "model_state_sha256",
    ):
        _require_sha256(value[name], name=name)
    for name in ("base_model_id", "tokenizer_id", "template_revision"):
        _require_nonempty_string(value[name], name=name)
    _require_pinned_revision(
        value["base_model_revision"],
        name="base_model_revision",
    )
    _require_pinned_revision(
        value["tokenizer_revision"],
        name="tokenizer_revision",
    )
    if value["dtype"] != "bfloat16":
        raise CheckpointContractError("merged model dtype must be bfloat16")
    _require_nonempty_string(
        value["verification_prompt"],
        name="verification_prompt",
    )
    max_new_tokens = value["max_new_tokens"]
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        raise CheckpointContractError("max_new_tokens must be a positive int")
    for name in ("rtol", "atol"):
        tolerance = value[name]
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(tolerance)
            or tolerance < 0
        ):
            raise CheckpointContractError(
                f"merged model {name} must be finite and non-negative"
            )

    normalized = _normalize_json(value, path="merged_model_metadata")
    unsigned = dict(normalized)
    serialized_hash = _require_sha256(
        unsigned.pop("metadata_sha256"),
        name="metadata_sha256",
    )
    expected_hash = canonical_json_sha256(unsigned)
    if serialized_hash != expected_hash:
        raise CheckpointContractError(
            f"merged model metadata_sha256 mismatch: expected {expected_hash}, "
            f"got {serialized_hash}"
        )
    return normalized


def build_merged_model_metadata(
    *,
    global_step: int,
    source_adapter_metadata_sha256: str,
    base_model_id: str,
    base_model_revision: str,
    tokenizer_id: str,
    tokenizer_revision: str,
    template_revision: str,
    text_mapping_sha256: str,
    dtype: str,
    model_state_sha256: str,
    verification_prompt: str,
    max_new_tokens: int,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Build canonical merged-model metadata and attach its semantic hash."""

    payload = {
        "schema_version": MERGED_MODEL_METADATA_SCHEMA_VERSION,
        "artifact_type": _MERGED_MODEL_ARTIFACT_TYPE,
        "global_step": global_step,
        "source_adapter_metadata_sha256": source_adapter_metadata_sha256,
        "base_model_id": base_model_id,
        "base_model_revision": base_model_revision,
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "template_revision": template_revision,
        "text_mapping_sha256": text_mapping_sha256,
        "dtype": dtype,
        "model_state_sha256": model_state_sha256,
        "verification_prompt": verification_prompt,
        "max_new_tokens": max_new_tokens,
        "rtol": rtol,
        "atol": atol,
    }
    payload["metadata_sha256"] = canonical_json_sha256(payload)
    return validate_merged_model_metadata(payload)


def _require_nonempty_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise CheckpointContractError(f"{name} must be a non-empty mapping")
    normalized = _normalize_json(value, path=name)
    if not isinstance(normalized, dict):  # Defensive: Mapping always normalizes to dict.
        raise CheckpointContractError(f"{name} must be a mapping")
    return normalized


def _reject_json_constant(value: str) -> None:
    raise CheckpointContractError(f"JSON contains non-standard constant {value!r}")


def _reject_duplicate_object_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointContractError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> Any:
    try:
        payload = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise CheckpointContractError(f"Unable to read JSON file {path}: {exc}") from exc
    try:
        return json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_object_pairs,
            parse_constant=_reject_json_constant,
        )
    except CheckpointContractError:
        raise
    except json.JSONDecodeError as exc:
        raise CheckpointContractError(f"Invalid JSON file {path}: {exc}") from exc


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


@dataclass(frozen=True, slots=True)
class DataloaderProgress:
    """Serializable dataloader cursor, external state digest, or both."""

    position: Optional[int] = None
    state_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        if self.position is None and self.state_sha256 is None:
            raise CheckpointContractError(
                "dataloader progress requires position, state_sha256, or both"
            )
        if self.position is not None:
            _require_nonnegative_int(self.position, name="dataloader.position")
        if self.state_sha256 is not None:
            _require_sha256(self.state_sha256, name="dataloader.state_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {"position": self.position, "state_sha256": self.state_sha256}

    @classmethod
    def from_dict(cls, value: Any) -> "DataloaderProgress":
        if not isinstance(value, Mapping):
            raise CheckpointContractError("dataloader_state must be a mapping")
        _require_exact_keys(
            value,
            {"position", "state_sha256"},
            label="dataloader_state",
        )
        return cls(position=value["position"], state_sha256=value["state_sha256"])


def _target_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    schema_version = manifest.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != 1:
        raise CheckpointContractError("lora_target_manifest.schema_version must be 1")
    target_modules = manifest.get("target_modules")
    if not isinstance(target_modules, list) or not target_modules:
        raise CheckpointContractError(
            "lora_target_manifest.target_modules must be a non-empty list"
        )
    if any(not isinstance(name, str) or not name for name in target_modules):
        raise CheckpointContractError(
            "lora_target_manifest.target_modules must contain non-empty strings"
        )
    if target_modules != sorted(set(target_modules)):
        raise CheckpointContractError(
            "lora_target_manifest.target_modules must be sorted and unique"
        )
    digest = canonical_json_sha256(
        {"schema_version": 1, "target_modules": target_modules}
    )
    embedded_digest = manifest.get("sha256")
    if embedded_digest is not None and embedded_digest != digest:
        raise CheckpointContractError(
            "lora_target_manifest embedded sha256 does not match target modules"
        )
    return digest


@dataclass(frozen=True, slots=True)
class CheckpointExtraState:
    """Strict, self-hashing state required for a deterministic resume."""

    SCHEMA_VERSION: ClassVar[int] = CHECKPOINT_EXTRA_STATE_SCHEMA_VERSION

    global_step: int
    rng_state: Mapping[str, Any]
    rng_state_sha256: str
    dataloader_state: DataloaderProgress
    data_manifest_sha256: str
    base_model_id: str
    base_model_revision: str
    text_mapping: Mapping[str, Any]
    text_mapping_sha256: str
    lora_config: Mapping[str, Any]
    lora_config_sha256: str
    lora_target_manifest: Mapping[str, Any]
    lora_target_sha256: str
    adapter_tensor_keys: tuple[str, ...]
    adapter_state_sha256: str
    model_build_metadata: Mapping[str, Any]
    model_build_metadata_sha256: str
    resolved_config: Mapping[str, Any]
    resolved_config_sha256: str

    _PAYLOAD_KEYS: ClassVar[set[str]] = {
        "schema_version",
        "global_step",
        "rng_state",
        "rng_state_sha256",
        "dataloader_state",
        "data_manifest_sha256",
        "base_model_id",
        "base_model_revision",
        "text_mapping",
        "text_mapping_sha256",
        "lora_config",
        "lora_config_sha256",
        "lora_target_manifest",
        "lora_target_sha256",
        "adapter_tensor_keys",
        "adapter_state_sha256",
        "model_build_metadata",
        "model_build_metadata_sha256",
        "resolved_config",
        "resolved_config_sha256",
    }

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.global_step, name="global_step")
        rng_state = _require_nonempty_mapping(self.rng_state, name="rng_state")
        if not isinstance(self.dataloader_state, DataloaderProgress):
            raise CheckpointContractError(
                "dataloader_state must be a DataloaderProgress instance"
            )
        _require_sha256(self.data_manifest_sha256, name="data_manifest_sha256")
        _require_nonempty_string(self.base_model_id, name="base_model_id")
        _require_pinned_revision(self.base_model_revision, name="base_model_revision")

        text_mapping = _require_nonempty_mapping(self.text_mapping, name="text_mapping")
        lora_config = _require_nonempty_mapping(self.lora_config, name="lora_config")
        target_manifest = _require_nonempty_mapping(
            self.lora_target_manifest,
            name="lora_target_manifest",
        )
        adapter_tensor_keys = _require_sorted_unique_strings(
            self.adapter_tensor_keys,
            name="adapter_tensor_keys",
        )
        _require_sha256(self.adapter_state_sha256, name="adapter_state_sha256")
        model_build_metadata = _require_nonempty_mapping(
            self.model_build_metadata,
            name="model_build_metadata",
        )
        resolved_config = _require_nonempty_mapping(
            self.resolved_config,
            name="resolved_config",
        )

        expected_hashes = {
            "rng_state_sha256": canonical_json_sha256(rng_state),
            "text_mapping_sha256": canonical_json_sha256(text_mapping),
            "lora_config_sha256": canonical_json_sha256(lora_config),
            "lora_target_sha256": _target_manifest_sha256(target_manifest),
            "model_build_metadata_sha256": canonical_json_sha256(
                model_build_metadata
            ),
            "resolved_config_sha256": canonical_json_sha256(resolved_config),
        }
        for name, expected in expected_hashes.items():
            actual = _require_sha256(getattr(self, name), name=name)
            if actual != expected:
                raise CheckpointContractError(
                    f"{name} mismatch: expected {expected}, got {actual}"
                )

        object.__setattr__(self, "rng_state", rng_state)
        object.__setattr__(self, "text_mapping", text_mapping)
        object.__setattr__(self, "lora_config", lora_config)
        object.__setattr__(self, "lora_target_manifest", target_manifest)
        object.__setattr__(self, "adapter_tensor_keys", adapter_tensor_keys)
        object.__setattr__(self, "model_build_metadata", model_build_metadata)
        object.__setattr__(self, "resolved_config", resolved_config)

    @classmethod
    def create(
        cls,
        *,
        global_step: int,
        rng_state: Mapping[str, Any],
        dataloader_state: DataloaderProgress,
        data_manifest_sha256: str,
        base_model_id: str,
        base_model_revision: str,
        text_mapping: Mapping[str, Any],
        lora_config: Mapping[str, Any],
        lora_target_manifest: Mapping[str, Any],
        adapter_tensor_keys: Sequence[str],
        adapter_state_sha256: str,
        model_build_metadata: Mapping[str, Any],
        resolved_config: Mapping[str, Any],
    ) -> "CheckpointExtraState":
        """Build state while deriving every component hash from canonical JSON."""

        normalized_rng = _require_nonempty_mapping(rng_state, name="rng_state")
        normalized_mapping = _require_nonempty_mapping(
            text_mapping,
            name="text_mapping",
        )
        normalized_lora = _require_nonempty_mapping(lora_config, name="lora_config")
        normalized_targets = _require_nonempty_mapping(
            lora_target_manifest,
            name="lora_target_manifest",
        )
        normalized_build_metadata = _require_nonempty_mapping(
            model_build_metadata,
            name="model_build_metadata",
        )
        normalized_config = _require_nonempty_mapping(
            resolved_config,
            name="resolved_config",
        )
        return cls(
            global_step=global_step,
            rng_state=normalized_rng,
            rng_state_sha256=canonical_json_sha256(normalized_rng),
            dataloader_state=dataloader_state,
            data_manifest_sha256=data_manifest_sha256,
            base_model_id=base_model_id,
            base_model_revision=base_model_revision,
            text_mapping=normalized_mapping,
            text_mapping_sha256=canonical_json_sha256(normalized_mapping),
            lora_config=normalized_lora,
            lora_config_sha256=canonical_json_sha256(normalized_lora),
            lora_target_manifest=normalized_targets,
            lora_target_sha256=_target_manifest_sha256(normalized_targets),
            adapter_tensor_keys=_require_sorted_unique_strings(
                adapter_tensor_keys,
                name="adapter_tensor_keys",
            ),
            adapter_state_sha256=adapter_state_sha256,
            model_build_metadata=normalized_build_metadata,
            model_build_metadata_sha256=canonical_json_sha256(
                normalized_build_metadata
            ),
            resolved_config=normalized_config,
            resolved_config_sha256=canonical_json_sha256(normalized_config),
        )

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "global_step": self.global_step,
            "rng_state": _normalize_json(self.rng_state),
            "rng_state_sha256": self.rng_state_sha256,
            "dataloader_state": self.dataloader_state.to_dict(),
            "data_manifest_sha256": self.data_manifest_sha256,
            "base_model_id": self.base_model_id,
            "base_model_revision": self.base_model_revision,
            "text_mapping": _normalize_json(self.text_mapping),
            "text_mapping_sha256": self.text_mapping_sha256,
            "lora_config": _normalize_json(self.lora_config),
            "lora_config_sha256": self.lora_config_sha256,
            "lora_target_manifest": _normalize_json(self.lora_target_manifest),
            "lora_target_sha256": self.lora_target_sha256,
            "adapter_tensor_keys": list(self.adapter_tensor_keys),
            "adapter_state_sha256": self.adapter_state_sha256,
            "model_build_metadata": _normalize_json(self.model_build_metadata),
            "model_build_metadata_sha256": self.model_build_metadata_sha256,
            "resolved_config": _normalize_json(self.resolved_config),
            "resolved_config_sha256": self.resolved_config_sha256,
        }

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self._payload_dict())

    @property
    def extra_state_sha256(self) -> str:
        return self.sha256

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_dict()
        payload["extra_state_sha256"] = self.sha256
        return payload

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        _atomic_write_bytes(destination, self.to_json_bytes())
        return destination

    @classmethod
    def from_dict(cls, value: Any) -> "CheckpointExtraState":
        if not isinstance(value, Mapping):
            raise CheckpointContractError("checkpoint extra state must be a mapping")
        expected_keys = cls._PAYLOAD_KEYS | {"extra_state_sha256"}
        _require_exact_keys(value, expected_keys, label="checkpoint extra state")
        _require_schema_version(
            value["schema_version"],
            cls.SCHEMA_VERSION,
            label="checkpoint extra-state",
        )
        state = cls(
            global_step=value["global_step"],
            rng_state=value["rng_state"],
            rng_state_sha256=value["rng_state_sha256"],
            dataloader_state=DataloaderProgress.from_dict(value["dataloader_state"]),
            data_manifest_sha256=value["data_manifest_sha256"],
            base_model_id=value["base_model_id"],
            base_model_revision=value["base_model_revision"],
            text_mapping=value["text_mapping"],
            text_mapping_sha256=value["text_mapping_sha256"],
            lora_config=value["lora_config"],
            lora_config_sha256=value["lora_config_sha256"],
            lora_target_manifest=value["lora_target_manifest"],
            lora_target_sha256=value["lora_target_sha256"],
            adapter_tensor_keys=tuple(value["adapter_tensor_keys"]),
            adapter_state_sha256=value["adapter_state_sha256"],
            model_build_metadata=value["model_build_metadata"],
            model_build_metadata_sha256=value["model_build_metadata_sha256"],
            resolved_config=value["resolved_config"],
            resolved_config_sha256=value["resolved_config_sha256"],
        )
        serialized_hash = _require_sha256(
            value["extra_state_sha256"],
            name="extra_state_sha256",
        )
        if serialized_hash != state.sha256:
            raise CheckpointContractError(
                f"extra_state_sha256 mismatch: expected {state.sha256}, got {serialized_hash}"
            )
        return state

    @classmethod
    def load(cls, path: str | Path) -> "CheckpointExtraState":
        return cls.from_dict(_load_json(Path(path)))


def validate_checkpoint_compatibility(
    checkpoint: CheckpointExtraState,
    *,
    base_model_id: str,
    base_model_revision: str,
    text_mapping: Mapping[str, Any],
    lora_config: Mapping[str, Any],
    lora_target_manifest: Mapping[str, Any],
    model_build_metadata: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    data_manifest_sha256: str,
) -> CheckpointExtraState:
    """Reject a resume whose immutable training identity changed."""

    if not isinstance(checkpoint, CheckpointExtraState):
        raise CheckpointContractError(
            "checkpoint must be a CheckpointExtraState instance"
        )
    expected_model_id = _require_nonempty_string(base_model_id, name="base_model_id")
    expected_revision = _require_pinned_revision(
        base_model_revision,
        name="base_model_revision",
    )
    expected_manifest_hash = _require_sha256(
        data_manifest_sha256,
        name="data_manifest_sha256",
    )
    expected_text_hash = canonical_json_sha256(
        _require_nonempty_mapping(text_mapping, name="text_mapping")
    )
    expected_lora_hash = canonical_json_sha256(
        _require_nonempty_mapping(lora_config, name="lora_config")
    )
    expected_target_hash = _target_manifest_sha256(
        _require_nonempty_mapping(
            lora_target_manifest,
            name="lora_target_manifest",
        )
    )
    expected_build_hash = canonical_json_sha256(
        _require_nonempty_mapping(
            model_build_metadata,
            name="model_build_metadata",
        )
    )

    comparisons = {
        "base_model_id": (checkpoint.base_model_id, expected_model_id),
        "base_model_revision": (
            checkpoint.base_model_revision,
            expected_revision,
        ),
        "data_manifest_sha256": (
            checkpoint.data_manifest_sha256,
            expected_manifest_hash,
        ),
        "text_mapping_sha256": (
            checkpoint.text_mapping_sha256,
            expected_text_hash,
        ),
        "lora_config_sha256": (
            checkpoint.lora_config_sha256,
            expected_lora_hash,
        ),
        "lora_target_sha256": (
            checkpoint.lora_target_sha256,
            expected_target_hash,
        ),
        "model_build_metadata_sha256": (
            checkpoint.model_build_metadata_sha256,
            expected_build_hash,
        ),
    }
    mismatches = {
        name: {"checkpoint": actual, "current": expected}
        for name, (actual, expected) in comparisons.items()
        if actual != expected
    }
    if mismatches:
        raise CheckpointContractError(
            "checkpoint is incompatible with the current resolved config: "
            + json.dumps(mismatches, sort_keys=True)
        )
    validate_resolved_config_compatibility(
        checkpoint.resolved_config,
        resolved_config,
    )
    return checkpoint


def validate_reproduction_rank_extra_state(
    value: Any,
    *,
    expected_global_step: Optional[int] = None,
) -> Mapping[str, Any]:
    """Validate the raw scheduler/RNG payload stored by one FSDP rank."""

    if not isinstance(value, Mapping):
        raise CheckpointContractError("rank extra state must be a mapping")
    _require_exact_keys(
        value,
        {"schema_version", "global_step", "lr_scheduler", "rng"},
        label="rank extra state",
    )
    _require_schema_version(
        value["schema_version"],
        REPRODUCTION_RANK_EXTRA_SCHEMA_VERSION,
        label="rank extra-state",
    )
    global_step = _require_nonnegative_int(
        value["global_step"],
        name="rank extra_state.global_step",
    )
    if expected_global_step is not None:
        expected = _require_nonnegative_int(
            expected_global_step,
            name="expected_global_step",
        )
        if global_step != expected:
            raise CheckpointContractError(
                f"rank extra-state global_step mismatch: expected {expected}, got {global_step}"
            )
    if value["lr_scheduler"] is None:
        raise CheckpointContractError("rank extra state is missing lr_scheduler")
    rng = value["rng"]
    if not isinstance(rng, Mapping):
        raise CheckpointContractError("rank extra state is missing RNG state")
    _require_exact_keys(
        rng,
        {"torch_cpu", "torch_cuda", "numpy", "python"},
        label="rank RNG state",
    )
    if rng["torch_cpu"] is None or rng["numpy"] is None or rng["python"] is None:
        raise CheckpointContractError("rank RNG state contains a missing required state")
    if not isinstance(rng["torch_cuda"], (list, tuple)):
        raise CheckpointContractError("rank torch_cuda RNG state must be a sequence")
    return value


def validate_scheduler_optimizer_alignment(
    scheduler_state: Mapping[str, Any],
    optimizer_state: Mapping[str, Any],
    *,
    expected_global_step: int,
) -> None:
    """Cross-check scheduler epoch and effective optimizer learning rates."""

    expected_step = _require_nonnegative_int(
        expected_global_step,
        name="expected_global_step",
    )
    if not isinstance(scheduler_state, Mapping):
        raise CheckpointContractError("lr_scheduler state must be a mapping")
    last_epoch = scheduler_state.get("last_epoch")
    if type(last_epoch) is not int or last_epoch != expected_step:
        raise CheckpointContractError(
            "lr_scheduler last_epoch does not match checkpoint global_step: "
            f"expected {expected_step}, got {last_epoch!r}"
        )
    last_lrs = scheduler_state.get("_last_lr")
    if not isinstance(last_lrs, (list, tuple)) or not last_lrs:
        raise CheckpointContractError("lr_scheduler state is missing _last_lr")
    if not isinstance(optimizer_state, Mapping):
        raise CheckpointContractError("optimizer state must be a mapping")
    param_groups = optimizer_state.get("param_groups")
    if not isinstance(param_groups, list) or not param_groups:
        raise CheckpointContractError("optimizer state is missing param_groups")
    optimizer_lrs = [group.get("lr") for group in param_groups]
    if len(optimizer_lrs) != len(last_lrs):
        raise CheckpointContractError(
            "lr_scheduler and optimizer parameter-group counts differ"
        )
    for index, (scheduler_lr, optimizer_lr) in enumerate(
        zip(last_lrs, optimizer_lrs)
    ):
        if not isinstance(scheduler_lr, (int, float)) or not isinstance(
            optimizer_lr,
            (int, float),
        ):
            raise CheckpointContractError(
                f"learning rate for parameter group {index} is not numeric"
            )
        if not math.isclose(
            float(scheduler_lr),
            float(optimizer_lr),
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise CheckpointContractError(
                "lr_scheduler _last_lr does not match optimizer lr for "
                f"parameter group {index}"
            )


def atomic_write_text(path: str | Path, value: str) -> Path:
    """Durably replace a small UTF-8 tracker file."""

    if not isinstance(value, str):
        raise TypeError("value must be a string")
    destination = Path(path)
    _atomic_write_bytes(destination, value.encode("utf-8"))
    _fsync_directory(destination.parent)
    return destination


@dataclass(frozen=True, slots=True)
class AdapterExportMetadata:
    """Portable identity metadata stored next to a PEFT adapter export."""

    SCHEMA_VERSION: ClassVar[int] = ADAPTER_EXPORT_METADATA_SCHEMA_VERSION
    ARTIFACT_TYPE: ClassVar[str] = "peft_adapter"

    global_step: int
    base_model_id: str
    base_model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    template_revision: str
    text_mapping: Mapping[str, Any]
    text_mapping_sha256: str
    lora_config: Mapping[str, Any]
    lora_config_sha256: str
    lora_target_manifest: Mapping[str, Any]
    lora_target_sha256: str
    adapter_tensor_keys: tuple[str, ...]
    adapter_state_sha256: str
    source_extra_state_sha256: str

    _PAYLOAD_KEYS: ClassVar[set[str]] = {
        "schema_version",
        "artifact_type",
        "global_step",
        "base_model_id",
        "base_model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "template_revision",
        "text_mapping",
        "text_mapping_sha256",
        "lora_config",
        "lora_config_sha256",
        "lora_target_manifest",
        "lora_target_sha256",
        "adapter_tensor_keys",
        "adapter_state_sha256",
        "source_extra_state_sha256",
    }

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.global_step, name="global_step")
        _require_nonempty_string(self.base_model_id, name="base_model_id")
        _require_pinned_revision(self.base_model_revision, name="base_model_revision")
        _require_nonempty_string(self.tokenizer_id, name="tokenizer_id")
        _require_pinned_revision(self.tokenizer_revision, name="tokenizer_revision")
        _require_pinned_revision(self.template_revision, name="template_revision")
        text_mapping = _require_nonempty_mapping(
            self.text_mapping,
            name="text_mapping",
        )
        expected_mapping_hash = canonical_json_sha256(text_mapping)
        actual_mapping_hash = _require_sha256(
            self.text_mapping_sha256,
            name="text_mapping_sha256",
        )
        if actual_mapping_hash != expected_mapping_hash:
            raise CheckpointContractError(
                "adapter text_mapping_sha256 does not match text_mapping"
            )
        object.__setattr__(self, "text_mapping", text_mapping)
        lora_config = _require_nonempty_mapping(self.lora_config, name="lora_config")
        if canonical_json_sha256(lora_config) != _require_sha256(
            self.lora_config_sha256,
            name="lora_config_sha256",
        ):
            raise CheckpointContractError("adapter LoRA config hash mismatch")
        target_manifest = _require_nonempty_mapping(
            self.lora_target_manifest,
            name="lora_target_manifest",
        )
        if _target_manifest_sha256(target_manifest) != _require_sha256(
            self.lora_target_sha256,
            name="lora_target_sha256",
        ):
            raise CheckpointContractError("adapter LoRA target hash mismatch")
        adapter_tensor_keys = _require_sorted_unique_strings(
            self.adapter_tensor_keys,
            name="adapter_tensor_keys",
        )
        _require_sha256(self.adapter_state_sha256, name="adapter_state_sha256")
        _require_sha256(
            self.source_extra_state_sha256,
            name="source_extra_state_sha256",
        )
        object.__setattr__(self, "lora_config", lora_config)
        object.__setattr__(self, "lora_target_manifest", target_manifest)
        object.__setattr__(self, "adapter_tensor_keys", adapter_tensor_keys)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: CheckpointExtraState,
        *,
        tokenizer_id: str,
        tokenizer_revision: str,
        template_revision: str,
    ) -> "AdapterExportMetadata":
        if not isinstance(checkpoint, CheckpointExtraState):
            raise CheckpointContractError(
                "checkpoint must be a CheckpointExtraState instance"
            )
        return cls(
            global_step=checkpoint.global_step,
            base_model_id=checkpoint.base_model_id,
            base_model_revision=checkpoint.base_model_revision,
            tokenizer_id=tokenizer_id,
            tokenizer_revision=tokenizer_revision,
            template_revision=template_revision,
            text_mapping=checkpoint.text_mapping,
            text_mapping_sha256=checkpoint.text_mapping_sha256,
            lora_config=checkpoint.lora_config,
            lora_config_sha256=checkpoint.lora_config_sha256,
            lora_target_manifest=checkpoint.lora_target_manifest,
            lora_target_sha256=checkpoint.lora_target_sha256,
            adapter_tensor_keys=checkpoint.adapter_tensor_keys,
            adapter_state_sha256=checkpoint.adapter_state_sha256,
            source_extra_state_sha256=checkpoint.sha256,
        )

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "artifact_type": self.ARTIFACT_TYPE,
            "global_step": self.global_step,
            "base_model_id": self.base_model_id,
            "base_model_revision": self.base_model_revision,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "template_revision": self.template_revision,
            "text_mapping": _normalize_json(self.text_mapping),
            "text_mapping_sha256": self.text_mapping_sha256,
            "lora_config": _normalize_json(self.lora_config),
            "lora_config_sha256": self.lora_config_sha256,
            "lora_target_manifest": _normalize_json(self.lora_target_manifest),
            "lora_target_sha256": self.lora_target_sha256,
            "adapter_tensor_keys": list(self.adapter_tensor_keys),
            "adapter_state_sha256": self.adapter_state_sha256,
            "source_extra_state_sha256": self.source_extra_state_sha256,
        }

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self._payload_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_dict()
        payload["metadata_sha256"] = self.sha256
        return payload

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        _atomic_write_bytes(destination, self.to_json_bytes())
        return destination

    @classmethod
    def from_dict(cls, value: Any) -> "AdapterExportMetadata":
        if not isinstance(value, Mapping):
            raise CheckpointContractError("adapter export metadata must be a mapping")
        _require_exact_keys(
            value,
            cls._PAYLOAD_KEYS | {"metadata_sha256"},
            label="adapter export metadata",
        )
        _require_schema_version(
            value["schema_version"],
            cls.SCHEMA_VERSION,
            label="adapter metadata",
        )
        if value["artifact_type"] != cls.ARTIFACT_TYPE:
            raise CheckpointContractError(
                f"unsupported adapter artifact type {value['artifact_type']!r}"
            )
        metadata = cls(
            global_step=value["global_step"],
            base_model_id=value["base_model_id"],
            base_model_revision=value["base_model_revision"],
            tokenizer_id=value["tokenizer_id"],
            tokenizer_revision=value["tokenizer_revision"],
            template_revision=value["template_revision"],
            text_mapping=value["text_mapping"],
            text_mapping_sha256=value["text_mapping_sha256"],
            lora_config=value["lora_config"],
            lora_config_sha256=value["lora_config_sha256"],
            lora_target_manifest=value["lora_target_manifest"],
            lora_target_sha256=value["lora_target_sha256"],
            adapter_tensor_keys=tuple(value["adapter_tensor_keys"]),
            adapter_state_sha256=value["adapter_state_sha256"],
            source_extra_state_sha256=value["source_extra_state_sha256"],
        )
        serialized_hash = _require_sha256(
            value["metadata_sha256"],
            name="metadata_sha256",
        )
        if serialized_hash != metadata.sha256:
            raise CheckpointContractError(
                f"metadata_sha256 mismatch: expected {metadata.sha256}, got {serialized_hash}"
            )
        return metadata

    @classmethod
    def load(cls, path: str | Path) -> "AdapterExportMetadata":
        return cls.from_dict(_load_json(Path(path)))


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class FileRecord:
    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        relative = PurePosixPath(_require_nonempty_string(self.path, name="file.path"))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "." in relative.parts
            or "\\" in self.path
        ):
            raise CheckpointContractError(f"unsafe completion-manifest path {self.path!r}")
        _require_nonnegative_int(self.size, name=f"file[{self.path}].size")
        _require_sha256(self.sha256, name=f"file[{self.path}].sha256")

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: Any) -> "FileRecord":
        if not isinstance(value, Mapping):
            raise CheckpointContractError("completion file record must be a mapping")
        _require_exact_keys(value, {"path", "size", "sha256"}, label="file record")
        return cls(path=value["path"], size=value["size"], sha256=value["sha256"])


@dataclass(frozen=True, slots=True)
class CompletionManifest:
    files: tuple[FileRecord, ...]
    files_sha256: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    SCHEMA_VERSION: ClassVar[int] = COMPLETION_MANIFEST_SCHEMA_VERSION
    STATUS: ClassVar[str] = "complete"

    def __post_init__(self) -> None:
        if not isinstance(self.files, tuple):
            object.__setattr__(self, "files", tuple(self.files))
        if any(not isinstance(record, FileRecord) for record in self.files):
            raise CheckpointContractError("completion files must contain FileRecord values")
        paths = [record.path for record in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise CheckpointContractError(
                "completion-manifest file paths must be sorted and unique"
            )
        expected = canonical_json_sha256([record.to_dict() for record in self.files])
        actual = _require_sha256(self.files_sha256, name="files_sha256")
        if actual != expected:
            raise CheckpointContractError(
                f"files_sha256 mismatch: expected {expected}, got {actual}"
            )
        normalized_metadata = _normalize_json(self.metadata, path="completion.metadata")
        if not isinstance(normalized_metadata, dict):
            raise CheckpointContractError("completion metadata must be a mapping")
        object.__setattr__(self, "metadata", normalized_metadata)

    @classmethod
    def create(
        cls,
        files: Sequence[FileRecord],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "CompletionManifest":
        records = tuple(files)
        return cls(
            files=records,
            files_sha256=canonical_json_sha256(
                [record.to_dict() for record in records]
            ),
            metadata={} if metadata is None else metadata,
        )

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "status": self.STATUS,
            "files": [record.to_dict() for record in self.files],
            "files_sha256": self.files_sha256,
            "metadata": _normalize_json(self.metadata),
        }

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self._payload_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_dict()
        payload["manifest_sha256"] = self.sha256
        return payload

    @classmethod
    def from_dict(cls, value: Any) -> "CompletionManifest":
        if not isinstance(value, Mapping):
            raise CheckpointContractError("completion manifest must be a mapping")
        _require_exact_keys(
            value,
            {
                "schema_version",
                "status",
                "files",
                "files_sha256",
                "metadata",
                "manifest_sha256",
            },
            label="completion manifest",
        )
        _require_schema_version(
            value["schema_version"],
            cls.SCHEMA_VERSION,
            label="completion manifest",
        )
        if value["status"] != cls.STATUS:
            raise IncompleteCheckpointError(
                f"completion manifest status is {value['status']!r}, not 'complete'"
            )
        if not isinstance(value["files"], list):
            raise CheckpointContractError("completion manifest files must be a list")
        if not isinstance(value["metadata"], Mapping):
            raise CheckpointContractError("completion manifest metadata must be a mapping")
        manifest = cls(
            files=tuple(FileRecord.from_dict(record) for record in value["files"]),
            files_sha256=value["files_sha256"],
            metadata=value["metadata"],
        )
        serialized_hash = _require_sha256(
            value["manifest_sha256"],
            name="manifest_sha256",
        )
        if serialized_hash != manifest.sha256:
            raise CheckpointContractError(
                "completion manifest outer sha256 does not match its payload"
            )
        return manifest


def _directory_file_records(directory: Path) -> tuple[FileRecord, ...]:
    records = []
    for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise CheckpointContractError(
                f"checkpoint directories must not contain symlinks: {path}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise CheckpointContractError(f"unsupported checkpoint entry: {path}")
        relative = path.relative_to(directory).as_posix()
        if relative == COMPLETION_MARKER_FILENAME:
            continue
        records.append(
            FileRecord(
                path=relative,
                size=path.stat().st_size,
                sha256=_file_sha256(path),
            )
        )
    return tuple(records)


def write_completion_marker(
    directory: str | Path,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> CompletionManifest:
    """Write the marker last, after hashing every regular artifact file."""

    root = Path(directory)
    if not root.is_dir() or root.is_symlink():
        raise CheckpointContractError(f"checkpoint staging directory is invalid: {root}")
    marker_path = root / COMPLETION_MARKER_FILENAME
    if marker_path.exists() or marker_path.is_symlink():
        raise CheckpointContractError(
            f"writer must not create reserved marker {COMPLETION_MARKER_FILENAME}"
        )
    manifest = CompletionManifest.create(
        _directory_file_records(root),
        metadata=metadata,
    )
    _atomic_write_bytes(marker_path, canonical_json_bytes(manifest.to_dict()) + b"\n")
    return manifest


def _read_completion_manifest(directory: str | Path) -> CompletionManifest:
    root = Path(directory)
    if not root.is_dir() or root.is_symlink():
        raise IncompleteCheckpointError(f"checkpoint directory is missing or invalid: {root}")
    marker_path = root / COMPLETION_MARKER_FILENAME
    if not marker_path.is_file() or marker_path.is_symlink():
        raise IncompleteCheckpointError(
            f"checkpoint has no valid {COMPLETION_MARKER_FILENAME}: {root}"
        )
    return CompletionManifest.from_dict(_load_json(marker_path))


def verify_complete_directory(directory: str | Path) -> CompletionManifest:
    """Verify marker schema and every recorded file hash."""

    root = Path(directory)
    manifest = _read_completion_manifest(root)
    actual_records = _directory_file_records(root)
    if actual_records != manifest.files:
        raise IncompleteCheckpointError(
            f"checkpoint contents do not match completion manifest: {root}"
        )
    return manifest


def validate_merged_model_artifact(
    directory: str | Path,
    *,
    expected_metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Verify a merged model's completion record, identity, and file shape."""

    root = Path(directory)
    completion = verify_complete_directory(root)
    metadata_path = root / MERGED_MODEL_METADATA_FILENAME
    if not metadata_path.is_file() or metadata_path.is_symlink():
        raise CheckpointContractError(
            f"merged artifact is missing {MERGED_MODEL_METADATA_FILENAME}"
        )
    metadata = validate_merged_model_metadata(_load_json(metadata_path))
    if expected_metadata is not None:
        expected = validate_merged_model_metadata(expected_metadata)
        if metadata != expected:
            raise CheckpointContractError(
                "merged model metadata changed while publishing"
            )

    expected_marker = {
        "artifact_type": _MERGED_MODEL_ARTIFACT_TYPE,
        "metadata_sha256": metadata["metadata_sha256"],
        "model_state_sha256": metadata["model_state_sha256"],
    }
    if completion.metadata != expected_marker:
        raise CheckpointContractError(
            "merged model completion marker identity mismatch"
        )

    paths = {record.path for record in completion.files}
    if "config.json" not in paths:
        raise CheckpointContractError("merged artifact is missing config.json")
    if any(path.endswith((".bin", ".pt", ".pth")) for path in paths):
        raise CheckpointContractError(
            "merged artifact contains a forbidden pickle-based model file"
        )
    if {"adapter_config.json", "adapter_model.safetensors"} & paths:
        raise CheckpointContractError(
            "merged artifact still contains adapter-only files"
        )
    has_single_weights = "model.safetensors" in paths
    has_sharded_weights = "model.safetensors.index.json" in paths and any(
        re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", path)
        for path in paths
    )
    if not (has_single_weights or has_sharded_weights):
        raise CheckpointContractError(
            "merged artifact has no complete safetensors model weights"
        )
    return metadata


def verify_reproduction_checkpoint_directory(
    directory: str | Path,
    *,
    verify_files: bool = True,
    allow_atomic_staging_name: bool = False,
) -> tuple[CompletionManifest, CheckpointExtraState]:
    """Bind marker identity, directory step, and strict root extra-state."""

    root = Path(directory)
    manifest = (
        verify_complete_directory(root)
        if verify_files
        else _read_completion_manifest(root)
    )
    match = re.fullmatch(r"global_step_(\d+)", root.name)
    if match is None and allow_atomic_staging_name:
        match = re.fullmatch(r"\.global_step_(\d+)\.staging-.+", root.name)
    if match is None:
        raise CheckpointContractError(
            "reproduction checkpoint directory must be named global_step_<N>"
        )
    directory_step = int(match.group(1))
    if set(manifest.metadata) != {"artifact_type", "global_step"}:
        raise CheckpointContractError(
            "reproduction checkpoint marker metadata does not match schema"
        )
    if manifest.metadata["artifact_type"] != "reproduction_training_checkpoint":
        raise CheckpointContractError(
            "completion marker artifact_type is not a reproduction checkpoint"
        )
    marker_step = _require_nonnegative_int(
        manifest.metadata["global_step"],
        name="completion marker global_step",
    )
    if marker_step != directory_step:
        raise CheckpointContractError(
            "completion marker global_step does not match checkpoint directory"
        )
    extra_state = CheckpointExtraState.load(root / EXTRA_STATE_FILENAME)
    if extra_state.global_step != directory_step:
        raise CheckpointContractError(
            "reproduction extra-state global_step does not match checkpoint directory"
        )
    return manifest, extra_state


def _fsync_tree(directory: Path) -> None:
    entries = tuple(directory.rglob("*"))
    for path in entries:
        if not path.is_file() or path.is_symlink():
            continue
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            try:
                os.fsync(descriptor)
            except OSError:
                # Windows rejects fsync on read-only descriptors. Writers have
                # already closed their files and the completion marker itself
                # is fsynced, so this durability pass remains best-effort.
                pass
        finally:
            os.close(descriptor)
    nested_directories = sorted(
        (
            path
            for path in entries
            if path.is_dir() and not path.is_symlink()
        ),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in nested_directories:
        _fsync_directory(path)
    _fsync_directory(directory)


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_publish_directory(
    destination: str | Path,
    writer: Callable[[Path], Any],
    *,
    validator: Optional[Callable[[Path], Any]] = None,
    marker_metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Build in a sibling temp directory and atomically publish when complete.

    Existing destinations are never replaced. A writer or validator failure
    only removes the unpublished staging directory; no prior checkpoint is
    inspected or pruned by this operation.
    """

    if not callable(writer):
        raise TypeError("writer must be callable")
    if validator is not None and not callable(validator):
        raise TypeError("validator must be callable")

    final_path = Path(destination)
    parent = final_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists() or final_path.is_symlink():
        raise FileExistsError(f"refusing to replace existing directory {final_path}")

    staging = Path(
        tempfile.mkdtemp(prefix=f".{final_path.name}.staging-", dir=str(parent))
    )
    try:
        writer(staging)
        if not staging.is_dir() or staging.is_symlink():
            raise CheckpointContractError("writer removed or replaced its staging directory")
        written_manifest = write_completion_marker(staging, metadata=marker_metadata)
        if validator is not None:
            result = validator(staging)
            if result is False:
                raise CheckpointVerificationError("staging validator returned False")
        after_validation = verify_complete_directory(staging)
        if after_validation.sha256 != written_manifest.sha256:
            raise CheckpointVerificationError(
                "staging validator mutated checkpoint contents"
            )
        _fsync_tree(staging)
        if final_path.exists() or final_path.is_symlink():
            raise FileExistsError(f"refusing to replace existing directory {final_path}")
        os.rename(staging, final_path)
        _fsync_directory(parent)
        return final_path
    except BaseException:
        if staging.exists():
            try:
                shutil.rmtree(staging)
            except OSError:
                pass
        raise


# Integration-friendly alias that states the checkpoint use case explicitly.
atomic_write_checkpoint = atomic_publish_directory


def _load_adapter_tensor_state(
    root: Path,
    adapter_weights: Sequence[Path],
    *,
    safetensors_loader: Optional[Callable[[str], Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    if safetensors_loader is None:
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "safetensors is required to validate adapter tensor bytes"
            ) from exc
        safetensors_loader = load_file
    state_dict = {}
    for path in adapter_weights:
        try:
            shard = safetensors_loader(str(path))
        except Exception as exc:
            raise CheckpointContractError(
                f"cannot read adapter safetensors {path.name!r}: {exc}"
            ) from exc
        if not isinstance(shard, Mapping) or not shard:
            raise CheckpointContractError(
                f"adapter safetensors {path.name!r} contains no tensors"
            )
        duplicate_keys = sorted(set(state_dict).intersection(shard))
        if duplicate_keys:
            raise CheckpointContractError(
                f"adapter safetensors duplicate tensor keys: {duplicate_keys}"
            )
        state_dict.update(shard)
    return state_dict


def _validate_exported_lora_config(
    adapter_config: Mapping[str, Any],
    metadata: AdapterExportMetadata,
) -> None:
    lora_config = metadata.lora_config
    expected = {
        "r": lora_config.get("rank", lora_config.get("r")),
        "lora_alpha": lora_config.get("alpha", lora_config.get("lora_alpha")),
        "lora_dropout": lora_config.get(
            "dropout",
            lora_config.get("lora_dropout"),
        ),
        "bias": lora_config.get("bias"),
    }
    for key, expected_value in expected.items():
        if expected_value is None or adapter_config.get(key) != expected_value:
            raise CheckpointContractError(
                f"adapter_config.json {key} does not match checkpoint LoRA config"
            )
    target_modules = adapter_config.get("target_modules")
    if not isinstance(target_modules, list):
        raise CheckpointContractError(
            "adapter_config.json target_modules must be an exact list"
        )
    expected_targets = metadata.lora_target_manifest["target_modules"]
    if (
        len(target_modules) != len(set(target_modules))
        or sorted(target_modules) != expected_targets
    ):
        raise CheckpointContractError(
            "adapter_config.json target_modules do not match the resolved target manifest"
        )


def validate_adapter_export(
    directory: str | Path,
    *,
    expected_metadata: Optional[AdapterExportMetadata] = None,
    verify_directory: bool = True,
    safetensors_loader: Optional[Callable[[str], Mapping[str, Any]]] = None,
) -> AdapterExportMetadata:
    """Validate that a completed directory contains adapter-only safetensors."""

    root = Path(directory)
    completion = (
        verify_complete_directory(root)
        if verify_directory
        else _read_completion_manifest(root)
    )
    metadata = AdapterExportMetadata.load(root / ADAPTER_METADATA_FILENAME)
    if expected_metadata is not None and metadata.to_dict() != expected_metadata.to_dict():
        raise CheckpointContractError("adapter export metadata changed during export")

    marker_metadata = completion.metadata
    if marker_metadata.get("artifact_type") != AdapterExportMetadata.ARTIFACT_TYPE:
        raise CheckpointContractError("completion marker does not identify a PEFT adapter")
    if marker_metadata.get("metadata_sha256") != metadata.sha256:
        raise CheckpointContractError("completion marker metadata hash mismatch")

    adapter_config_path = root / "adapter_config.json"
    adapter_config = _load_json(adapter_config_path)
    if not isinstance(adapter_config, Mapping) or not adapter_config:
        raise CheckpointContractError("adapter_config.json must contain a JSON object")
    configured_base = adapter_config.get("base_model_name_or_path")
    if configured_base not in (None, "", metadata.base_model_id):
        raise CheckpointContractError(
            "adapter_config.json base_model_name_or_path does not match metadata"
        )
    _validate_exported_lora_config(adapter_config, metadata)

    adapter_weights = tuple(root.glob("adapter_model*.safetensors"))
    if not adapter_weights or any(not path.is_file() for path in adapter_weights):
        raise CheckpointContractError(
            "adapter export must contain adapter_model*.safetensors"
        )

    allowed_weight_paths = set(adapter_weights)
    for path in root.rglob("*.safetensors"):
        if path not in allowed_weight_paths:
            raise CheckpointContractError(
                "adapter-only export contains a non-adapter safetensors file "
                f"{path.relative_to(root).as_posix()!r}"
            )
    binary_weights = tuple(root.rglob("*.bin"))
    if binary_weights:
        names = [path.relative_to(root).as_posix() for path in binary_weights]
        raise CheckpointContractError(
            f"adapter-only export contains forbidden binary weights {names}"
        )

    forbidden_names = {
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
    }
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name in forbidden_names or (
            path.name.startswith("model-") and path.name.endswith(".safetensors")
        ):
            raise CheckpointContractError(
                f"adapter-only export contains full-model weight file {path.name!r}"
            )

    tensor_state = _load_adapter_tensor_state(
        root,
        adapter_weights,
        safetensors_loader=safetensors_loader,
    )
    tensor_keys = tuple(sorted(tensor_state))
    if tensor_keys != metadata.adapter_tensor_keys:
        raise CheckpointContractError(
            "adapter safetensors tensor keys do not match checkpoint metadata"
        )
    state_hash = canonical_tensor_state_sha256(tensor_state)
    if state_hash != metadata.adapter_state_sha256:
        raise CheckpointContractError(
            "adapter safetensors semantic tensor-state hash mismatch"
        )
    return metadata


def export_peft_adapter(
    model: Any,
    destination: str | Path,
    metadata: AdapterExportMetadata,
    *,
    save_pretrained_kwargs: Optional[Mapping[str, Any]] = None,
    safetensors_loader: Optional[Callable[[str], Mapping[str, Any]]] = None,
) -> Path:
    """Atomically export PEFT adapter weights and reproduction metadata."""

    save_pretrained = getattr(model, "save_pretrained", None)
    if not callable(save_pretrained):
        raise TypeError("model must provide save_pretrained()")
    if not isinstance(metadata, AdapterExportMetadata):
        raise TypeError("metadata must be an AdapterExportMetadata instance")
    kwargs = dict(save_pretrained_kwargs or {})
    if kwargs.get("safe_serialization") is False:
        raise CheckpointContractError("adapter export requires safe_serialization=True")
    kwargs["safe_serialization"] = True

    def writer(staging: Path) -> None:
        save_pretrained(str(staging), **kwargs)
        metadata.save(staging / ADAPTER_METADATA_FILENAME)

    def validator(staging: Path) -> None:
        validate_adapter_export(
            staging,
            expected_metadata=metadata,
            verify_directory=False,
            safetensors_loader=safetensors_loader,
        )

    return atomic_publish_directory(
        destination,
        writer,
        validator=validator,
        marker_metadata={
            "artifact_type": AdapterExportMetadata.ARTIFACT_TYPE,
            "metadata_sha256": metadata.sha256,
        },
    )


def _from_pretrained_callable(loader: Any, *, label: str) -> Callable[..., Any]:
    method = getattr(loader, "from_pretrained", None)
    if callable(method):
        return method
    if callable(loader):
        return loader
    raise TypeError(f"{label} must be callable or provide from_pretrained()")


def load_peft_adapter(
    base_model: Any,
    adapter_directory: str | Path,
    *,
    peft_model_loader: Any = None,
    validate_export: bool = True,
    **load_kwargs: Any,
) -> Any:
    """Reload an exported adapter, importing PEFT only when no loader is given."""

    adapter_path = Path(adapter_directory)
    if validate_export:
        validate_adapter_export(adapter_path)
    if peft_model_loader is None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("PEFT is required to reload an adapter") from exc
        peft_model_loader = PeftModel
    loader = _from_pretrained_callable(peft_model_loader, label="peft_model_loader")
    kwargs = dict(load_kwargs)
    kwargs.setdefault("is_trainable", False)
    return loader(base_model, str(adapter_path), **kwargs)


def merge_and_unload_adapter(
    base_model: Any,
    adapter_directory: str | Path,
    *,
    peft_model_loader: Any = None,
    load_kwargs: Optional[Mapping[str, Any]] = None,
    merge_kwargs: Optional[Mapping[str, Any]] = None,
    validate_export: bool = True,
) -> Any:
    """Reload a PEFT adapter and merge it into its base model lazily."""

    adapter_model = load_peft_adapter(
        base_model,
        adapter_directory,
        peft_model_loader=peft_model_loader,
        validate_export=validate_export,
        **dict(load_kwargs or {}),
    )
    merge_and_unload = getattr(adapter_model, "merge_and_unload", None)
    if not callable(merge_and_unload):
        raise TypeError("loaded PEFT model must provide merge_and_unload()")
    return merge_and_unload(**dict(merge_kwargs or {}))


def reload_merged_model(
    model_directory: str | Path,
    *,
    model_loader: Any = None,
    **load_kwargs: Any,
) -> Any:
    """Reload a saved merged CausalLM with an injectable lazy loader."""

    model_path = Path(model_directory)
    if not model_path.is_dir() or model_path.is_symlink():
        raise CheckpointContractError(f"merged model directory is invalid: {model_path}")
    if model_loader is None:
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise ImportError("Transformers is required to reload a merged model") from exc
        model_loader = AutoModelForCausalLM
    loader = _from_pretrained_callable(model_loader, label="model_loader")
    return loader(str(model_path), **load_kwargs)


# Naming aliases used by export-oriented call sites.
merge_and_unload_peft_adapter = merge_and_unload_adapter
reload_peft_adapter = load_peft_adapter


@dataclass(frozen=True, slots=True)
class CheckpointValidationEvidence:
    """The three validations required before any retention deletion."""

    load_succeeded: bool
    generate_succeeded: bool
    hash_matched: bool
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("load_succeeded", "generate_succeeded", "hash_matched"):
            if type(getattr(self, name)) is not bool:
                raise CheckpointVerificationError(f"{name} must be a bool")
        normalized = _normalize_json(self.details, path="verification.details")
        if not isinstance(normalized, dict):
            raise CheckpointVerificationError("verification details must be a mapping")
        object.__setattr__(self, "details", normalized)

    def require_complete(self) -> None:
        failures = [
            name
            for name in ("load_succeeded", "generate_succeeded", "hash_matched")
            if not getattr(self, name)
        ]
        if failures:
            raise CheckpointVerificationError(
                f"checkpoint is not eligible for pruning; failed checks={failures}"
            )


@dataclass(frozen=True, slots=True, init=False)
class VerifiedCheckpoint:
    """Opaque prune authorization issued only by verify_checkpoint_for_pruning."""

    path: Path
    marker_sha256: str
    evidence: CheckpointValidationEvidence
    _token: object = field(repr=False)

    def __init__(
        self,
        path: Path,
        marker_sha256: str,
        evidence: CheckpointValidationEvidence,
        *,
        _token: object,
    ) -> None:
        if _token is not _PRUNE_AUTHORIZATION_TOKEN:
            raise TypeError(
                "VerifiedCheckpoint can only be created by verify_checkpoint_for_pruning()"
            )
        object.__setattr__(self, "_token", _token)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "marker_sha256", marker_sha256)
        object.__setattr__(self, "evidence", evidence)


def verify_checkpoint_for_pruning(
    checkpoint: str | Path,
    validator: Callable[[Path], CheckpointValidationEvidence],
) -> VerifiedCheckpoint:
    """Run load/generate/hash validation and issue a prune authorization."""

    if not callable(validator):
        raise TypeError("validator must be callable")
    supplied_path = Path(checkpoint)
    if supplied_path.is_symlink():
        raise CheckpointVerificationError(
            "checkpoint verification does not accept a symlink path"
        )
    checkpoint_path = supplied_path.resolve(strict=True)
    before = verify_complete_directory(checkpoint_path)
    evidence = validator(checkpoint_path)
    if not isinstance(evidence, CheckpointValidationEvidence):
        raise CheckpointVerificationError(
            "validator must return CheckpointValidationEvidence"
        )
    evidence.require_complete()
    after = verify_complete_directory(checkpoint_path)
    if after.sha256 != before.sha256:
        raise CheckpointVerificationError(
            "checkpoint changed while running the prune validation"
        )
    return VerifiedCheckpoint(
        checkpoint_path,
        after.sha256,
        evidence,
        _token=_PRUNE_AUTHORIZATION_TOKEN,
    )


def prune_checkpoints_after_verification(
    candidates: Sequence[str | Path],
    *,
    verification: VerifiedCheckpoint,
) -> tuple[Path, ...]:
    """Delete completed sibling checkpoints only after a fresh receipt check."""

    if (
        not isinstance(verification, VerifiedCheckpoint)
        or verification._token is not _PRUNE_AUTHORIZATION_TOKEN
    ):
        raise CheckpointVerificationError(
            "pruning requires a receipt from verify_checkpoint_for_pruning()"
        )
    verification.evidence.require_complete()
    current = verify_complete_directory(verification.path)
    if current.sha256 != verification.marker_sha256:
        raise CheckpointVerificationError(
            "verified checkpoint changed after validation; refusing to prune"
        )

    if isinstance(candidates, (str, Path)):
        raise TypeError("candidates must be a sequence of checkpoint paths")
    resolved_candidates = []
    for candidate in candidates:
        supplied_path = Path(candidate)
        if supplied_path.is_symlink():
            raise CheckpointVerificationError("prune candidates must not be symlinks")
        path = supplied_path.resolve(strict=True)
        if path == verification.path:
            raise CheckpointVerificationError(
                "the newly verified checkpoint cannot be pruned"
            )
        if path.parent != verification.path.parent:
            raise CheckpointVerificationError(
                "prune candidates must be siblings of the verified checkpoint"
            )
        resolved_candidates.append(path)
    if len(resolved_candidates) != len(set(resolved_candidates)):
        raise CheckpointVerificationError("prune candidates must be unique")

    # Validate every candidate before the first destructive operation.
    for path in resolved_candidates:
        verify_complete_directory(path)
    for path in resolved_candidates:
        shutil.rmtree(path)
    return tuple(resolved_candidates)


prune_after_verification = prune_checkpoints_after_verification


__all__ = [
    "ADAPTER_EXPORT_METADATA_SCHEMA_VERSION",
    "ADAPTER_METADATA_FILENAME",
    "CHECKPOINT_EXTRA_STATE_SCHEMA_VERSION",
    "COMPLETION_MANIFEST_SCHEMA_VERSION",
    "COMPLETION_MARKER_FILENAME",
    "DEFAULT_RESUME_CONFIG_ALLOWLIST",
    "EXTRA_STATE_FILENAME",
    "MERGED_MODEL_METADATA_FILENAME",
    "MERGED_MODEL_METADATA_SCHEMA_VERSION",
    "REPRODUCTION_RANK_EXTRA_SCHEMA_VERSION",
    "AdapterExportMetadata",
    "CheckpointContractError",
    "CheckpointExtraState",
    "CheckpointValidationEvidence",
    "CheckpointVerificationError",
    "CompletionManifest",
    "DataloaderProgress",
    "FileRecord",
    "IncompleteCheckpointError",
    "VerifiedCheckpoint",
    "atomic_publish_directory",
    "atomic_write_text",
    "atomic_write_checkpoint",
    "build_merged_model_metadata",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "canonical_tensor_state_sha256",
    "capture_process_rng_state",
    "export_peft_adapter",
    "load_peft_adapter",
    "json_safe_state_sha256",
    "merge_and_unload_adapter",
    "merge_and_unload_peft_adapter",
    "normalize_resolved_config",
    "prune_after_verification",
    "prune_checkpoints_after_verification",
    "reload_merged_model",
    "reload_peft_adapter",
    "restore_process_rng_state",
    "to_json_safe_state",
    "validate_adapter_export",
    "validate_bound_dataset_manifest",
    "validate_checkpoint_compatibility",
    "validate_merged_model_artifact",
    "validate_merged_model_metadata",
    "validate_reproduction_manifest_configuration",
    "validate_reproduction_rank_extra_state",
    "validate_resolved_config_compatibility",
    "validate_scheduler_optimizer_alignment",
    "verify_checkpoint_for_pruning",
    "verify_complete_directory",
    "verify_reproduction_checkpoint_directory",
    "write_completion_marker",
]
