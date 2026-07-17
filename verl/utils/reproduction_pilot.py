"""Immutable evidence for the fixed three-step B/C pilot gate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import tempfile
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PILOT_SCHEMA_VERSION = 2
EXPERIMENT_PROFILE_ID = "rtx5090-32g-qwen35-2b-v1"
EXPECTED_STEPS = (1, 2, 3)
EXPECTED_PROMPT_GROUPS_PER_STEP = 2
MIN_NONZERO_PROMPT_GROUPS = 2
MIN_NONZERO_STATE_GROUPS = 1


class PilotEvidenceError(RuntimeError):
    """Raised when pilot evidence is incomplete, mutable, or inconsistent."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PilotEvidenceError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise PilotEvidenceError(f"pilot evidence is not canonical JSON: {exc}") from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _step_zero_fingerprint_class() -> type:
    module_name = "_rememr1_step_zero_contract_for_pilot"
    module = sys.modules.get(module_name)
    if module is None:
        source = Path(__file__).with_name("reproduction_fingerprint.py")
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise PilotEvidenceError(f"cannot load step-zero contract: {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.StepZeroFingerprint


def _require_absolute_safe_path(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise PilotEvidenceError(f"{label} must be absolute")
    path = Path(os.path.abspath(path))
    for component in [*reversed(path.parents), path]:
        if component.is_symlink():
            raise PilotEvidenceError(f"{label} contains a symlink: {component}")
    return path


def _atomic_write(path: Path, payload: bytes, *, no_replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path = _require_absolute_safe_path(path, "pilot evidence output")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if no_replace:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise PilotEvidenceError(f"pilot evidence already exists: {path}") from exc
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PilotEvidenceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PilotEvidenceError(f"{label} must be finite")
    return result


def _group_summaries(
    group_ids: Sequence[Any],
    values: Sequence[Any],
    *,
    label: str,
) -> list[dict[str, Any]]:
    if isinstance(group_ids, (str, bytes)) or isinstance(values, (str, bytes)):
        raise PilotEvidenceError(f"{label} inputs must be sequences")
    if len(group_ids) != len(values) or not group_ids:
        raise PilotEvidenceError(f"{label} IDs and values must be non-empty and aligned")
    grouped: dict[str, list[float]] = {}
    for index, (group_id, raw_value) in enumerate(zip(group_ids, values, strict=True)):
        if not isinstance(group_id, (str, int)) or isinstance(group_id, bool):
            raise PilotEvidenceError(f"{label} group ID {index} is invalid")
        normalized_id = str(group_id)
        if not normalized_id:
            raise PilotEvidenceError(f"{label} group ID {index} is empty")
        grouped.setdefault(normalized_id, []).append(
            _finite_float(raw_value, f"{label} value {index}")
        )
    summaries = []
    for group_id in sorted(grouped):
        group_values = grouped[group_id]
        summaries.append(
            {
                "group_id_sha256": hashlib.sha256(group_id.encode("utf-8")).hexdigest(),
                "member_count": len(group_values),
                "nonzero": any(abs(value) > 1e-12 for value in group_values),
                "values_sha256": _canonical_sha256(group_values),
            }
        )
    return summaries


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _summaries_from_actions(
    actions: Sequence[Mapping[str, Any]],
    *,
    group_key: str,
    value_key: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = {}
    for action in actions:
        group_id = action[group_key]
        if group_id is None:
            continue
        grouped.setdefault(group_id, []).append(float(action[value_key]))
    return [
        {
            "group_id_sha256": group_id,
            "member_count": len(values),
            "nonzero": any(abs(value) > 1e-12 for value in values),
            "values_sha256": _canonical_sha256(values),
        }
        for group_id, values in sorted(grouped.items())
    ]


def _action_identity(action: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "action_type": action["action_type"],
        "prompt_group_id_sha256": action["prompt_group_id_sha256"],
        "sample_ordinal": action["sample_ordinal"],
        "step_id": action["step_id"],
    }


def _equation_matches(action: Mapping[str, Any], alpha: float, arm: str) -> bool:
    outcome = float(action["outcome_advantage"])
    total = float(action["total_advantage"])
    if arm == "b":
        expected = outcome
    else:
        expected = alpha * outcome + (1.0 - alpha) * float(action["state_advantage"])
    return math.isclose(total, expected, rel_tol=1e-6, abs_tol=1e-7)


def resolve_stable_prompt_group_ids(
    runtime_prompt_group_ids: Sequence[Any],
    manifest_prompt_group_ids: Sequence[Any],
    action_runtime_prompt_group_ids: Sequence[Any],
) -> list[str]:
    """Map per-process UUID groups onto stable ordered manifest QA identities."""

    sequences = (
        runtime_prompt_group_ids,
        manifest_prompt_group_ids,
        action_runtime_prompt_group_ids,
    )
    if any(isinstance(value, (str, bytes)) for value in sequences):
        raise PilotEvidenceError("pilot prompt group identities must be sequences")
    if (
        not runtime_prompt_group_ids
        or len(runtime_prompt_group_ids) != len(manifest_prompt_group_ids)
        or not action_runtime_prompt_group_ids
    ):
        raise PilotEvidenceError("pilot runtime and manifest groups are not aligned")

    mapping: dict[str, str] = {}
    seen_manifest_ids: set[str] = set()
    for index, (runtime_id, manifest_id) in enumerate(
        zip(runtime_prompt_group_ids, manifest_prompt_group_ids, strict=True)
    ):
        if isinstance(runtime_id, bool) or not isinstance(runtime_id, (str, int)):
            raise PilotEvidenceError(f"pilot runtime group {index} is invalid")
        if isinstance(manifest_id, bool) or not isinstance(manifest_id, (str, int)):
            raise PilotEvidenceError(f"pilot manifest group {index} is invalid")
        runtime_key = str(runtime_id)
        manifest_key = str(manifest_id)
        if not runtime_key or not manifest_key:
            raise PilotEvidenceError("pilot prompt group identities must be non-empty")
        if runtime_key in mapping or manifest_key in seen_manifest_ids:
            raise PilotEvidenceError("pilot prompt group identities must be unique")
        mapping[runtime_key] = manifest_key
        seen_manifest_ids.add(manifest_key)

    resolved = []
    for index, runtime_id in enumerate(action_runtime_prompt_group_ids):
        runtime_key = str(runtime_id)
        if runtime_key not in mapping:
            raise PilotEvidenceError(
                f"pilot action runtime group {index} lacks a manifest identity"
            )
        resolved.append(mapping[runtime_key])
    return resolved


def create_step_record(
    *,
    arm: str,
    offload_profile: str,
    global_step: int,
    alpha: float,
    prompt_group_ids: Sequence[Any],
    advantage_values: Sequence[Any],
    outcome_advantage_values: Sequence[Any],
    action_step_ids: Sequence[Any],
    action_types: Sequence[Any],
    state_advantage_values: Sequence[Any] | None = None,
    experiment_profile_id: str = EXPERIMENT_PROFILE_ID,
) -> dict[str, Any]:
    if arm not in {"b", "c"}:
        raise PilotEvidenceError("pilot arm must be b or c")
    if offload_profile not in {"r0", "r1"}:
        raise PilotEvidenceError("pilot offload profile must be r0 or r1")
    if experiment_profile_id != EXPERIMENT_PROFILE_ID:
        raise PilotEvidenceError("pilot experiment profile is not active")
    if isinstance(global_step, bool) or global_step not in EXPECTED_STEPS:
        raise PilotEvidenceError("pilot global step must be 1, 2, or 3")
    alpha = _finite_float(alpha, "alpha")
    expected_alpha = 1.0 if arm == "b" else 0.8
    if alpha != expected_alpha:
        raise PilotEvidenceError(f"pilot {arm.upper()} alpha must be {expected_alpha}")
    sequences = {
        "prompt group IDs": prompt_group_ids,
        "total advantages": advantage_values,
        "outcome advantages": outcome_advantage_values,
        "action step IDs": action_step_ids,
        "action types": action_types,
    }
    if any(isinstance(value, (str, bytes)) for value in sequences.values()):
        raise PilotEvidenceError("pilot action evidence inputs must be sequences")
    lengths = {len(value) for value in sequences.values()}
    if lengths != {len(prompt_group_ids)} or not prompt_group_ids:
        raise PilotEvidenceError("pilot action evidence must be non-empty and aligned")
    if arm == "c":
        if state_advantage_values is None or isinstance(state_advantage_values, (str, bytes)):
            raise PilotEvidenceError("C pilot requires aligned state advantages")
        if len(state_advantage_values) != len(prompt_group_ids):
            raise PilotEvidenceError("C pilot state advantages are not aligned")
    elif state_advantage_values is not None:
        raise PilotEvidenceError("B pilot must not publish state advantages")

    actions: list[dict[str, Any]] = []
    ordinals: dict[tuple[str, int, int], int] = {}
    for index in range(len(prompt_group_ids)):
        raw_prompt_group = prompt_group_ids[index]
        if not isinstance(raw_prompt_group, (str, int)) or isinstance(raw_prompt_group, bool):
            raise PilotEvidenceError(f"prompt group ID {index} is invalid")
        prompt_group = str(raw_prompt_group)
        if not prompt_group:
            raise PilotEvidenceError(f"prompt group ID {index} is empty")
        raw_step_id = action_step_ids[index]
        if isinstance(raw_step_id, bool) or not isinstance(raw_step_id, (int, float)):
            raise PilotEvidenceError(f"action step ID {index} is invalid")
        step_id = int(raw_step_id)
        if float(raw_step_id) != step_id or step_id < 0:
            raise PilotEvidenceError(f"action step ID {index} is invalid")
        raw_action_type = action_types[index]
        if isinstance(raw_action_type, bool) or not isinstance(raw_action_type, (int, float)):
            raise PilotEvidenceError(f"action type {index} is invalid")
        action_type = int(raw_action_type)
        if float(raw_action_type) != action_type or action_type not in {0, 1, 2}:
            raise PilotEvidenceError(f"action type {index} is invalid")
        prompt_hash = _sha256_text(prompt_group)
        state_hash = _canonical_sha256([prompt_hash, step_id]) if arm == "c" else None
        ordinal_key = (prompt_hash, step_id, action_type)
        sample_ordinal = ordinals.get(ordinal_key, 0)
        ordinals[ordinal_key] = sample_ordinal + 1
        action = {
            "action_type": action_type,
            "outcome_advantage": _finite_float(
                outcome_advantage_values[index], f"outcome advantage {index}"
            ),
            "prompt_group_id_sha256": prompt_hash,
            "sample_ordinal": sample_ordinal,
            "state_advantage": (
                _finite_float(state_advantage_values[index], f"state advantage {index}")
                if state_advantage_values is not None
                else None
            ),
            "state_group_id_sha256": state_hash,
            "step_id": step_id,
            "total_advantage": _finite_float(
                advantage_values[index], f"total advantage {index}"
            ),
        }
        if not _equation_matches(action, alpha, arm):
            raise PilotEvidenceError(
                f"pilot {arm.upper()} advantage equation failed at action {index}"
            )
        actions.append(action)

    prompt_groups = _summaries_from_actions(
        actions,
        group_key="prompt_group_id_sha256",
        value_key="total_advantage",
    )
    if len(prompt_groups) != EXPECTED_PROMPT_GROUPS_PER_STEP:
        raise PilotEvidenceError(
            f"pilot step must contain {EXPECTED_PROMPT_GROUPS_PER_STEP} prompt groups"
        )
    state_groups = _summaries_from_actions(
        actions,
        group_key="state_group_id_sha256",
        value_key="state_advantage",
    )
    payload = {
        "action_identity_sha256": _canonical_sha256(
            [_action_identity(action) for action in actions]
        ),
        "action_records": actions,
        "advantage_values_sha256": _canonical_sha256(
            [action["total_advantage"] for action in actions]
        ),
        "alpha": alpha,
        "arm": arm,
        "experiment_profile_id": experiment_profile_id,
        "global_step": global_step,
        "kind": "rememr1-pilot-step-v2",
        "nonzero_prompt_group_count": sum(item["nonzero"] for item in prompt_groups),
        "nonzero_state_group_count": sum(item["nonzero"] for item in state_groups),
        "offload_profile": offload_profile,
        "outcome_advantage_values_sha256": _canonical_sha256(
            [action["outcome_advantage"] for action in actions]
        ),
        "prompt_group_count": len(prompt_groups),
        "prompt_groups": prompt_groups,
        "schema_version": PILOT_SCHEMA_VERSION,
        "state_advantage_values_sha256": (
            _canonical_sha256([action["state_advantage"] for action in actions])
            if arm == "c"
            else None
        ),
        "state_group_count": len(state_groups),
        "state_groups": state_groups,
    }
    return {**payload, "record_sha256": _canonical_sha256(payload)}


def _validate_step_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PilotEvidenceError("pilot step record must be an object")
    expected_keys = {
        "action_identity_sha256",
        "action_records",
        "advantage_values_sha256",
        "alpha",
        "arm",
        "experiment_profile_id",
        "global_step",
        "kind",
        "nonzero_prompt_group_count",
        "nonzero_state_group_count",
        "offload_profile",
        "outcome_advantage_values_sha256",
        "prompt_group_count",
        "prompt_groups",
        "record_sha256",
        "schema_version",
        "state_advantage_values_sha256",
        "state_group_count",
        "state_groups",
    }
    if set(value) != expected_keys:
        raise PilotEvidenceError("pilot step record keys do not match schema")
    unsigned = dict(value)
    observed_sha = unsigned.pop("record_sha256")
    if observed_sha != _canonical_sha256(unsigned):
        raise PilotEvidenceError("pilot step record self-hash mismatch")
    if value["kind"] != "rememr1-pilot-step-v2" or value["schema_version"] != 2:
        raise PilotEvidenceError("pilot step contract changed")
    if value["arm"] not in {"b", "c"}:
        raise PilotEvidenceError("pilot arm must be b or c")
    if value["offload_profile"] not in {"r0", "r1"}:
        raise PilotEvidenceError("pilot offload profile must be r0 or r1")
    if value["experiment_profile_id"] != EXPERIMENT_PROFILE_ID:
        raise PilotEvidenceError("pilot experiment profile is not active")
    if type(value["global_step"]) is not int or value["global_step"] not in EXPECTED_STEPS:
        raise PilotEvidenceError("pilot global step must be 1, 2, or 3")
    alpha = _finite_float(value["alpha"], "alpha")
    expected_alpha = 1.0 if value["arm"] == "b" else 0.8
    if alpha != expected_alpha:
        raise PilotEvidenceError("pilot alpha does not match arm")
    actions = value["action_records"]
    if not isinstance(actions, list) or not actions:
        raise PilotEvidenceError("pilot action records must be a non-empty array")
    action_keys = {
        "action_type",
        "outcome_advantage",
        "prompt_group_id_sha256",
        "sample_ordinal",
        "state_advantage",
        "state_group_id_sha256",
        "step_id",
        "total_advantage",
    }
    seen_identities: set[tuple[Any, ...]] = set()
    for index, action in enumerate(actions):
        if not isinstance(action, Mapping) or set(action) != action_keys:
            raise PilotEvidenceError(f"pilot action record {index} schema mismatch")
        if type(action["action_type"]) is not int or action["action_type"] not in {0, 1, 2}:
            raise PilotEvidenceError(f"pilot action record {index} type is invalid")
        if type(action["step_id"]) is not int or action["step_id"] < 0:
            raise PilotEvidenceError(f"pilot action record {index} step is invalid")
        if type(action["sample_ordinal"]) is not int or action["sample_ordinal"] < 0:
            raise PilotEvidenceError(f"pilot action record {index} ordinal is invalid")
        if not _is_sha256(action["prompt_group_id_sha256"]):
            raise PilotEvidenceError(f"pilot action record {index} prompt digest is invalid")
        state_hash = action["state_group_id_sha256"]
        state_value = action["state_advantage"]
        if value["arm"] == "b":
            if state_hash is not None or state_value is not None:
                raise PilotEvidenceError("B pilot action must not contain state advantage")
        else:
            expected_state_hash = _canonical_sha256(
                [action["prompt_group_id_sha256"], action["step_id"]]
            )
            if state_hash != expected_state_hash or state_value is None:
                raise PilotEvidenceError("C pilot action lacks the expected state mapping")
        _finite_float(action["outcome_advantage"], f"outcome advantage {index}")
        _finite_float(action["total_advantage"], f"total advantage {index}")
        if state_value is not None:
            _finite_float(state_value, f"state advantage {index}")
        if not _equation_matches(action, alpha, value["arm"]):
            raise PilotEvidenceError(f"pilot advantage equation failed at action {index}")
        identity = tuple(_action_identity(action).values())
        if identity in seen_identities:
            raise PilotEvidenceError("pilot action identities must be unique")
        seen_identities.add(identity)

    expected_prompt_groups = _summaries_from_actions(
        actions,
        group_key="prompt_group_id_sha256",
        value_key="total_advantage",
    )
    expected_state_groups = _summaries_from_actions(
        actions,
        group_key="state_group_id_sha256",
        value_key="state_advantage",
    )
    if value["prompt_groups"] != expected_prompt_groups:
        raise PilotEvidenceError("prompt group summaries do not match action records")
    if value["state_groups"] != expected_state_groups:
        raise PilotEvidenceError("state group summaries do not match action records")
    expected_hashes = {
        "action_identity_sha256": _canonical_sha256(
            [_action_identity(action) for action in actions]
        ),
        "advantage_values_sha256": _canonical_sha256(
            [float(action["total_advantage"]) for action in actions]
        ),
        "outcome_advantage_values_sha256": _canonical_sha256(
            [float(action["outcome_advantage"]) for action in actions]
        ),
        "state_advantage_values_sha256": (
            _canonical_sha256([float(action["state_advantage"]) for action in actions])
            if value["arm"] == "c"
            else None
        ),
    }
    for key, expected in expected_hashes.items():
        if value[key] != expected:
            raise PilotEvidenceError(f"pilot {key} mismatch")
    for collection_name in ("prompt_groups", "state_groups"):
        collection = value[collection_name]
        if not isinstance(collection, list):
            raise PilotEvidenceError(f"{collection_name} must be an array")
        for item in collection:
            if not isinstance(item, Mapping) or set(item) != {
                "group_id_sha256",
                "member_count",
                "nonzero",
                "values_sha256",
            }:
                raise PilotEvidenceError(f"{collection_name} item schema mismatch")
            for digest_name in ("group_id_sha256", "values_sha256"):
                digest = item[digest_name]
                if not _is_sha256(digest):
                    raise PilotEvidenceError(f"{collection_name} digest is invalid")
            if type(item["member_count"]) is not int or item["member_count"] < 1:
                raise PilotEvidenceError(f"{collection_name} member count is invalid")
            if type(item["nonzero"]) is not bool:
                raise PilotEvidenceError(f"{collection_name} nonzero flag is invalid")
    if value["prompt_group_count"] != len(value["prompt_groups"]):
        raise PilotEvidenceError("prompt group count mismatch")
    if value["state_group_count"] != len(value["state_groups"]):
        raise PilotEvidenceError("state group count mismatch")
    if value["nonzero_prompt_group_count"] != sum(
        item["nonzero"] for item in value["prompt_groups"]
    ):
        raise PilotEvidenceError("nonzero prompt group count mismatch")
    if value["nonzero_state_group_count"] != sum(
        item["nonzero"] for item in value["state_groups"]
    ):
        raise PilotEvidenceError("nonzero state group count mismatch")
    if value["prompt_group_count"] != EXPECTED_PROMPT_GROUPS_PER_STEP:
        raise PilotEvidenceError("pilot step has the wrong number of prompt groups")
    return dict(value)


def load_evidence(path: str | Path) -> list[dict[str, Any]]:
    source = _require_absolute_safe_path(path, "pilot evidence path")
    try:
        raw_lines = source.read_bytes().splitlines(keepends=True)
    except OSError as exc:
        raise PilotEvidenceError(f"cannot read pilot evidence: {exc}") from exc
    if len(raw_lines) not in {1, 2, 3}:
        raise PilotEvidenceError("pilot evidence must contain one to three records")
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(raw_lines, start=1):
        if not raw_line.endswith(b"\n"):
            raise PilotEvidenceError(f"pilot evidence line {line_number} lacks newline")
        try:
            value = json.loads(
                raw_line,
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=lambda item: (_ for _ in ()).throw(
                    PilotEvidenceError(f"non-finite JSON value: {item}")
                ),
            )
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise PilotEvidenceError(f"invalid pilot JSONL line {line_number}: {exc}") from exc
        record = _validate_step_record(value)
        if raw_line != _canonical_bytes(record) + b"\n":
            raise PilotEvidenceError(f"pilot evidence line {line_number} is not canonical")
        records.append(record)
    steps = [record["global_step"] for record in records]
    if steps != list(range(1, len(records) + 1)):
        raise PilotEvidenceError("pilot evidence steps must be ordered and contiguous")
    identity = {
        (record["arm"], record["offload_profile"], record["experiment_profile_id"])
        for record in records
    }
    if len(identity) != 1:
        raise PilotEvidenceError("pilot evidence identity changed between steps")
    return records


def append_step_record(path: str | Path, record: Mapping[str, Any]) -> Path:
    destination = _require_absolute_safe_path(path, "pilot evidence path")
    validated = _validate_step_record(record)
    existing = load_evidence(destination) if destination.exists() else []
    expected_step = len(existing) + 1
    if validated["global_step"] != expected_step:
        raise PilotEvidenceError(
            f"pilot evidence expected step {expected_step}, got {validated['global_step']}"
        )
    if existing:
        previous = existing[0]
        for key in ("arm", "offload_profile", "experiment_profile_id"):
            if validated[key] != previous[key]:
                raise PilotEvidenceError(f"pilot evidence changed {key}")
    if len(existing) >= len(EXPECTED_STEPS):
        raise PilotEvidenceError("pilot evidence already contains all three steps")
    payload = b"".join(_canonical_bytes(item) + b"\n" for item in [*existing, validated])
    _atomic_write(destination, payload)
    return destination


def create_gate_report(
    *,
    b_evidence_path: str | Path,
    c_evidence_path: str | Path,
    b_step_zero_path: str | Path,
    c_step_zero_path: str | Path,
) -> dict[str, Any]:
    b_path = _require_absolute_safe_path(b_evidence_path, "B pilot evidence")
    c_path = _require_absolute_safe_path(c_evidence_path, "C pilot evidence")
    b_records = load_evidence(b_path)
    c_records = load_evidence(c_path)
    if [record["global_step"] for record in b_records] != list(EXPECTED_STEPS):
        raise PilotEvidenceError("B pilot evidence is not exactly three steps")
    if [record["global_step"] for record in c_records] != list(EXPECTED_STEPS):
        raise PilotEvidenceError("C pilot evidence is not exactly three steps")
    if any(record["arm"] != "b" for record in b_records):
        raise PilotEvidenceError("B evidence contains the wrong arm")
    if any(record["arm"] != "c" for record in c_records):
        raise PilotEvidenceError("C evidence contains the wrong arm")
    if b_records[0]["offload_profile"] != c_records[0]["offload_profile"]:
        raise PilotEvidenceError("B/C pilots use different offload profiles")

    StepZeroFingerprint = _step_zero_fingerprint_class()
    b_step_zero = StepZeroFingerprint.load(
        _require_absolute_safe_path(b_step_zero_path, "B step-zero fingerprint")
    )
    c_step_zero = StepZeroFingerprint.load(
        _require_absolute_safe_path(c_step_zero_path, "C step-zero fingerprint")
    )
    c_step_zero.assert_matches(b_step_zero)
    b_nonzero = sum(record["nonzero_prompt_group_count"] for record in b_records)
    c_nonzero = sum(record["nonzero_prompt_group_count"] for record in c_records)
    c_state_nonzero = sum(record["nonzero_state_group_count"] for record in c_records)
    if b_records[0]["action_identity_sha256"] != c_records[0]["action_identity_sha256"]:
        raise PilotEvidenceError("B/C step1 action mapping differs")
    if (
        b_records[0]["outcome_advantage_values_sha256"]
        != c_records[0]["outcome_advantage_values_sha256"]
    ):
        raise PilotEvidenceError("B/C step1 outcome advantages differ")
    passed = (
        b_nonzero >= MIN_NONZERO_PROMPT_GROUPS
        and c_nonzero >= MIN_NONZERO_PROMPT_GROUPS
        and c_state_nonzero >= MIN_NONZERO_STATE_GROUPS
    )
    payload = {
        "b_evidence_file_sha256": _sha256_file(b_path),
        "b_nonzero_prompt_groups": b_nonzero,
        "b_step_zero_fingerprint_sha256": b_step_zero.sha256,
        "c_evidence_file_sha256": _sha256_file(c_path),
        "c_nonzero_prompt_groups": c_nonzero,
        "c_nonzero_state_groups": c_state_nonzero,
        "c_step_zero_fingerprint_sha256": c_step_zero.sha256,
        "experiment_profile_id": EXPERIMENT_PROFILE_ID,
        "expected_prompt_groups_per_arm": (
            len(EXPECTED_STEPS) * EXPECTED_PROMPT_GROUPS_PER_STEP
        ),
        "kind": "rememr1-pilot-gate-v2",
        "minimum_nonzero_state_groups": MIN_NONZERO_STATE_GROUPS,
        "minimum_nonzero_prompt_groups_per_arm": MIN_NONZERO_PROMPT_GROUPS,
        "offload_profile": b_records[0]["offload_profile"],
        "outcome": "pass" if passed else "scientific-stop",
        "retryable": False,
        "schema_version": PILOT_SCHEMA_VERSION,
        "shared_step1_action_identity_sha256": b_records[0]["action_identity_sha256"],
        "shared_step1_outcome_advantage_values_sha256": b_records[0][
            "outcome_advantage_values_sha256"
        ],
    }
    return {**payload, "gate_sha256": _canonical_sha256(payload)}


def publish_gate_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    destination = _require_absolute_safe_path(path, "pilot gate output")
    if not isinstance(report, Mapping) or report.get("gate_sha256") is None:
        raise PilotEvidenceError("pilot gate report is invalid")
    unsigned = dict(report)
    observed = unsigned.pop("gate_sha256")
    if observed != _canonical_sha256(unsigned):
        raise PilotEvidenceError("pilot gate report self-hash mismatch")
    _atomic_write(destination, _canonical_bytes(dict(report)) + b"\n", no_replace=True)
    return destination


__all__ = [
    "EXPECTED_PROMPT_GROUPS_PER_STEP",
    "EXPECTED_STEPS",
    "EXPERIMENT_PROFILE_ID",
    "MIN_NONZERO_PROMPT_GROUPS",
    "MIN_NONZERO_STATE_GROUPS",
    "PILOT_SCHEMA_VERSION",
    "PilotEvidenceError",
    "append_step_record",
    "create_gate_report",
    "create_step_record",
    "load_evidence",
    "publish_gate_report",
    "resolve_stable_prompt_group_ids",
]
