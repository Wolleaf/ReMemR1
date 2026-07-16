"""Self-hashed step-zero evidence for paired reproduction runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence


FINGERPRINT_SCHEMA_VERSION = 1
_SHA256_HEX = frozenset("0123456789abcdef")
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_MAX_SEED = 2**32 - 1


class FingerprintContractError(ValueError):
    """Raised when step-zero evidence is missing, malformed, or inconsistent."""


def _normalize_json(value: Any, *, path: str = "$") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FingerprintContractError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise FingerprintContractError(f"{path} contains a non-string key")
            result[key] = _normalize_json(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise FingerprintContractError(
        f"{path} contains unsupported value {type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _normalize_json(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise FingerprintContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FingerprintContractError(f"{name} must be a non-empty string")
    return value


def _require_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FingerprintContractError(f"{name} must be a non-negative int")
    return value


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FingerprintContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def hash_ordered_sample_ids(sample_ids: Sequence[str | int]) -> str:
    """Hash the exact first-batch order without accepting ambiguous IDs."""

    if isinstance(sample_ids, (str, bytes)) or not isinstance(sample_ids, Sequence):
        raise TypeError("sample_ids must be a sequence")
    normalized: list[str | int] = []
    for index, sample_id in enumerate(sample_ids):
        if isinstance(sample_id, bool) or not isinstance(sample_id, (str, int)):
            raise FingerprintContractError(
                f"sample_ids[{index}] must be a string or integer"
            )
        if isinstance(sample_id, str) and not sample_id:
            raise FingerprintContractError(f"sample_ids[{index}] must not be empty")
        if isinstance(sample_id, int) and sample_id < 0:
            raise FingerprintContractError(
                f"sample_ids[{index}] integer must be non-negative"
            )
        normalized.append(sample_id)
    if not normalized:
        raise FingerprintContractError("sample_ids must not be empty")
    return canonical_sha256(
        {"schema_version": 1, "ordered_sample_ids": normalized}
    )


def hash_sampled_token_rows(token_rows: Sequence[Sequence[int]]) -> str:
    """Hash padded sampled-token rows in their exact trajectory/action order."""

    if isinstance(token_rows, (str, bytes)) or not isinstance(token_rows, Sequence):
        raise TypeError("token_rows must be a sequence of integer sequences")
    normalized: list[list[int]] = []
    expected_width: int | None = None
    for row_index, row in enumerate(token_rows):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise FingerprintContractError(f"token_rows[{row_index}] must be a sequence")
        normalized_row: list[int] = []
        for column_index, token in enumerate(row):
            if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                raise FingerprintContractError(
                    f"token_rows[{row_index}][{column_index}] must be a non-negative int"
                )
            normalized_row.append(token)
        if expected_width is None:
            expected_width = len(normalized_row)
        elif len(normalized_row) != expected_width:
            raise FingerprintContractError("sampled token rows must have equal width")
        normalized.append(normalized_row)
    if not normalized or expected_width == 0:
        raise FingerprintContractError("token_rows must not be empty")
    return canonical_sha256(
        {
            "schema_version": 1,
            "shape": [len(normalized), expected_width],
            "token_rows": normalized,
        }
    )


@dataclass(frozen=True, slots=True)
class StepZeroFingerprint:
    run_seed: int
    rollout_global_step: int
    base_model_id: str
    base_model_revision: str
    data_manifest_sha256: str
    initial_adapter_sha256: str
    first_batch_sha256: str
    first_sampled_tokens_sha256: str

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.run_seed, name="run_seed")
        if self.run_seed > _MAX_SEED:
            raise FingerprintContractError(f"run_seed must be at most {_MAX_SEED}")
        _require_nonnegative_int(
            self.rollout_global_step,
            name="rollout_global_step",
        )
        if self.rollout_global_step != 1:
            raise FingerprintContractError(
                "step-zero fingerprint rollout_global_step must be 1"
            )
        _require_nonempty_string(self.base_model_id, name="base_model_id")
        _require_nonempty_string(self.base_model_revision, name="base_model_revision")
        if not _COMMIT_SHA.fullmatch(self.base_model_revision):
            raise FingerprintContractError(
                "base_model_revision must be a lowercase 40-character commit SHA"
            )
        for name in (
            "data_manifest_sha256",
            "initial_adapter_sha256",
            "first_batch_sha256",
            "first_sampled_tokens_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)

    def _payload(self) -> dict[str, Any]:
        return {"schema_version": FINGERPRINT_SCHEMA_VERSION, **asdict(self)}

    @property
    def sha256(self) -> str:
        return canonical_sha256(self._payload())

    def to_dict(self) -> dict[str, Any]:
        result = self._payload()
        result["fingerprint_sha256"] = self.sha256
        return result

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        if not destination.is_absolute():
            raise FingerprintContractError(
                "step-zero fingerprint path must be absolute"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FingerprintContractError(
                f"step-zero fingerprint already exists: {destination}"
            )
        payload = canonical_json_bytes(self.to_dict()) + b"\n"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.tmp-",
            dir=str(destination.parent),
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary_path, destination)
            except FileExistsError as exc:
                raise FingerprintContractError(
                    f"step-zero fingerprint already exists: {destination}"
                ) from exc
        finally:
            temporary_path.unlink(missing_ok=True)
        return destination

    @classmethod
    def from_dict(cls, value: Any) -> "StepZeroFingerprint":
        if not isinstance(value, Mapping):
            raise FingerprintContractError("fingerprint must be a JSON object")
        expected_keys = {
            "schema_version",
            "run_seed",
            "rollout_global_step",
            "base_model_id",
            "base_model_revision",
            "data_manifest_sha256",
            "initial_adapter_sha256",
            "first_batch_sha256",
            "first_sampled_tokens_sha256",
            "fingerprint_sha256",
        }
        if set(value) != expected_keys:
            raise FingerprintContractError("fingerprint keys do not match schema")
        if type(value["schema_version"]) is not int or value["schema_version"] != FINGERPRINT_SCHEMA_VERSION:
            raise FingerprintContractError("unsupported fingerprint schema")
        fingerprint = cls(
            run_seed=value["run_seed"],
            rollout_global_step=value["rollout_global_step"],
            base_model_id=value["base_model_id"],
            base_model_revision=value["base_model_revision"],
            data_manifest_sha256=value["data_manifest_sha256"],
            initial_adapter_sha256=value["initial_adapter_sha256"],
            first_batch_sha256=value["first_batch_sha256"],
            first_sampled_tokens_sha256=value["first_sampled_tokens_sha256"],
        )
        serialized_hash = _require_sha256(
            value["fingerprint_sha256"],
            name="fingerprint_sha256",
        )
        if serialized_hash != fingerprint.sha256:
            raise FingerprintContractError("fingerprint_sha256 mismatch")
        return fingerprint

    @classmethod
    def load(cls, path: str | Path) -> "StepZeroFingerprint":
        source = Path(path)
        if not source.is_absolute():
            raise FingerprintContractError(
                "step-zero fingerprint path must be absolute"
            )
        try:
            value = json.loads(
                source.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_pairs,
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise FingerprintContractError(f"cannot load fingerprint {source}: {exc}") from exc
        return cls.from_dict(value)

    def assert_matches(self, reference: "StepZeroFingerprint") -> None:
        if not isinstance(reference, StepZeroFingerprint):
            raise TypeError("reference must be a StepZeroFingerprint")
        mismatches = [
            name
            for name in asdict(self)
            if getattr(self, name) != getattr(reference, name)
        ]
        if mismatches:
            raise FingerprintContractError(
                f"step-zero fingerprint mismatch in fields: {mismatches}"
            )


def load_and_verify_step_zero_fingerprint(
    path: str | Path,
    *,
    reference_path: str | Path | None = None,
) -> StepZeroFingerprint:
    """Load self-hashed evidence and optionally verify its paired baseline."""

    fingerprint = StepZeroFingerprint.load(path)
    if reference_path is not None:
        fingerprint.assert_matches(StepZeroFingerprint.load(reference_path))
    return fingerprint


def publish_step_zero_fingerprint(
    fingerprint: StepZeroFingerprint,
    path: str | Path,
    *,
    reference_path: str | Path | None = None,
) -> Path:
    """Compare paired evidence before atomically publishing a no-clobber file."""

    if not isinstance(fingerprint, StepZeroFingerprint):
        raise TypeError("fingerprint must be a StepZeroFingerprint")
    if reference_path is not None:
        fingerprint.assert_matches(StepZeroFingerprint.load(reference_path))
    return fingerprint.save(path)


__all__ = [
    "FINGERPRINT_SCHEMA_VERSION",
    "FingerprintContractError",
    "StepZeroFingerprint",
    "canonical_json_bytes",
    "canonical_sha256",
    "hash_ordered_sample_ids",
    "hash_sampled_token_rows",
    "load_and_verify_step_zero_fingerprint",
    "publish_step_zero_fingerprint",
]
