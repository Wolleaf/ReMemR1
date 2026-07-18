"""Create a canonical per-attempt CUDA/NVML optimizer-step ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2
PROFILE_ID = "rtx5090-32g-qwen35-2b-v1"
VERIFICATION_SCOPES = ("capacity", "engineering", "scientific")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_PHASES = (
    "rollout",
    "reward",
    "actor_log_prob",
    "reference_log_prob",
    "update",
    "save",
)


class TrainingTelemetryError(RuntimeError):
    """Raised when step telemetry cannot prove a stable single-GPU identity."""


class TrainingScientificStop(TrainingTelemetryError):
    """Raised when valid telemetry proves a non-retryable scientific stop."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise TrainingTelemetryError(f"telemetry is not canonical JSON: {exc}") from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _safe_absolute(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise TrainingTelemetryError(f"{label} must be absolute")
    path = Path(os.path.abspath(path))
    for component in [*reversed(path.parents), path]:
        if component.is_symlink():
            raise TrainingTelemetryError(f"{label} contains a symlink: {component}")
    return path


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path = _safe_absolute(path, "telemetry output")
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
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingTelemetryError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise TrainingTelemetryError(f"{label} must be finite and non-negative")
    return result


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrainingTelemetryError(f"{label} must be a non-negative integer")
    return value


def _validate_logits(value: Any, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {
        "bytes",
        "dtype",
        "element_size",
        "numel",
        "shape",
    }:
        raise TrainingTelemetryError(f"{label} logits schema mismatch")
    shape = value["shape"]
    if not isinstance(shape, list) or not shape or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in shape
    ):
        raise TrainingTelemetryError(f"{label} logits shape is invalid")
    numel = _integer(value["numel"], f"{label}.numel")
    element_size = _integer(value["element_size"], f"{label}.element_size")
    byte_count = _integer(value["bytes"], f"{label}.bytes")
    if math.prod(shape) != numel or numel * element_size != byte_count:
        raise TrainingTelemetryError(f"{label} logits size identity mismatch")
    dtype = value["dtype"]
    if not isinstance(dtype, str) or not dtype:
        raise TrainingTelemetryError(f"{label} logits dtype is invalid")
    return {
        "bytes": byte_count,
        "dtype": dtype,
        "element_size": element_size,
        "numel": numel,
        "shape": shape,
    }


def _validate_worker_phases(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(REQUIRED_PHASES):
        raise TrainingTelemetryError(f"{label} phase inventory is incomplete")
    required_keys = {
        "duration_seconds",
        "host_peak_used_bytes",
        "name",
        "nvml_peak_used_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "post_allocated_bytes",
        "post_reserved_bytes",
        "swap_peak_used_bytes",
    }
    phases = []
    for index, (raw_phase, expected_name) in enumerate(
        zip(value, REQUIRED_PHASES, strict=True)
    ):
        if not isinstance(raw_phase, Mapping) or set(raw_phase) != required_keys:
            raise TrainingTelemetryError(f"{label} phase {index} schema mismatch")
        if raw_phase["name"] != expected_name:
            raise TrainingTelemetryError(f"{label} phase order mismatch")
        phase = {"name": expected_name}
        phase["duration_seconds"] = _number(
            raw_phase["duration_seconds"], f"{label}.{expected_name}.duration_seconds"
        )
        for key in required_keys - {"duration_seconds", "name"}:
            phase[key] = _integer(
                raw_phase[key], f"{label}.{expected_name}.{key}"
            )
        phases.append(phase)
    return phases


def create_step_record(
    *,
    worker_records: Sequence[Mapping[str, Any]],
    global_step: int,
    attempt_id: str,
    sealed_config_id: str,
    sealed_config_sha256: str,
    offload_profile: str,
    timing_seconds: Mapping[str, Any],
    scientific_evidence: Mapping[str, Any] | None = None,
    experiment_profile_id: str = PROFILE_ID,
) -> dict[str, Any]:
    if experiment_profile_id != PROFILE_ID:
        raise TrainingTelemetryError("telemetry experiment profile is not active")
    if offload_profile not in {"r0", "r1"}:
        raise TrainingTelemetryError("telemetry offload profile must be r0 or r1")
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 1:
        raise TrainingTelemetryError("telemetry global step must be positive")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise TrainingTelemetryError("telemetry attempt ID must be non-empty")
    if not isinstance(sealed_config_id, str) or not sealed_config_id:
        raise TrainingTelemetryError("telemetry config ID must be non-empty")
    if not isinstance(sealed_config_sha256, str) or _SHA256.fullmatch(
        sealed_config_sha256
    ) is None:
        raise TrainingTelemetryError("telemetry config SHA is invalid")
    if not isinstance(worker_records, Sequence) or isinstance(worker_records, (str, bytes)):
        raise TrainingTelemetryError("worker telemetry must be a sequence")
    workers = [dict(record) for record in worker_records]
    if len(workers) != 2:
        raise TrainingTelemetryError(
            "single-5090 telemetry requires one actor and one reference worker"
        )
    required_worker_keys = {
        "actor_logits",
        "allocator_retry_count",
        "gpu_uuid",
        "host_peak_used_bytes",
        "host_total_memory_bytes",
        "nvml_peak_used_bytes",
        "nvml_total_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "post_step_allocated_bytes",
        "post_step_nvml_used_bytes",
        "post_step_reserved_bytes",
        "phase_records",
        "rank",
        "reference_logits",
        "role",
        "step_wall_seconds",
        "swap_used_bytes",
        "world_size",
    }
    if any(set(worker) != required_worker_keys for worker in workers):
        raise TrainingTelemetryError("worker telemetry keys do not match schema")
    if any(worker["rank"] != 0 or worker["world_size"] != 1 for worker in workers):
        raise TrainingTelemetryError("single-5090 telemetry rank/world size mismatch")
    by_role = {worker["role"]: worker for worker in workers}
    if set(by_role) != {"actor", "reference"}:
        raise TrainingTelemetryError(
            "worker telemetry must contain distinct actor and reference roles"
        )
    actor_worker = by_role["actor"]
    reference_worker = by_role["reference"]
    gpu_uuid = actor_worker["gpu_uuid"]
    if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-"):
        raise TrainingTelemetryError("worker GPU UUID is invalid")
    if reference_worker["gpu_uuid"] != gpu_uuid:
        raise TrainingTelemetryError("actor/reference GPU identity mismatch")
    if (
        actor_worker["nvml_total_bytes"] != reference_worker["nvml_total_bytes"]
        or actor_worker["host_total_memory_bytes"]
        != reference_worker["host_total_memory_bytes"]
    ):
        raise TrainingTelemetryError("actor/reference hardware totals differ")
    integer_fields = (
        "allocator_retry_count",
        "host_peak_used_bytes",
        "host_total_memory_bytes",
        "nvml_peak_used_bytes",
        "nvml_total_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "post_step_allocated_bytes",
        "post_step_nvml_used_bytes",
        "post_step_reserved_bytes",
        "swap_used_bytes",
    )
    normalized_workers = {
        role: {
            field: _integer(worker[field], f"{role}_worker.{field}")
            for field in integer_fields
        }
        for role, worker in by_role.items()
    }
    actor_logits = _validate_logits(actor_worker["actor_logits"], "actor")
    reference_logits = _validate_logits(
        reference_worker["reference_logits"], "reference"
    )
    if actor_logits is None or reference_logits is None:
        raise TrainingTelemetryError(
            "formal telemetry requires actual actor and reference logits evidence"
        )
    if actor_worker["reference_logits"] is not None:
        raise TrainingTelemetryError("actor worker published reference logits")
    if reference_worker["actor_logits"] is not None:
        raise TrainingTelemetryError("reference worker published actor logits")
    actor_phases = _validate_worker_phases(
        actor_worker["phase_records"], "actor_worker"
    )
    reference_phases = _validate_worker_phases(
        reference_worker["phase_records"], "reference_worker"
    )
    reference_resource_fields = (
        "allocator_retry_count",
        "host_peak_used_bytes",
        "nvml_peak_used_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "post_step_allocated_bytes",
        "post_step_nvml_used_bytes",
        "post_step_reserved_bytes",
        "swap_used_bytes",
    )
    if any(normalized_workers["reference"][field] != 0 for field in reference_resource_fields):
        raise TrainingTelemetryError(
            "reference worker must publish logits-only zero resource placeholders"
        )
    if any(
        any(value != 0 for key, value in phase.items() if key != "name")
        for phase in reference_phases
    ):
        raise TrainingTelemetryError(
            "reference worker must not publish independent phase resource evidence"
        )
    phase_telemetry = []
    for actor_phase, reference_phase in zip(
        actor_phases, reference_phases, strict=True
    ):
        phase_telemetry.append(
            {
                "actor_peak_allocated_bytes": actor_phase["peak_allocated_bytes"],
                "actor_peak_reserved_bytes": actor_phase["peak_reserved_bytes"],
                "actor_post_allocated_bytes": actor_phase["post_allocated_bytes"],
                "actor_post_reserved_bytes": actor_phase["post_reserved_bytes"],
                "duration_seconds": actor_phase["duration_seconds"],
                "host_peak_used_bytes": actor_phase["host_peak_used_bytes"],
                "name": actor_phase["name"],
                "nvml_peak_used_bytes": actor_phase["nvml_peak_used_bytes"],
                "reference_peak_allocated_bytes": 0,
                "reference_peak_reserved_bytes": 0,
                "reference_post_allocated_bytes": 0,
                "reference_post_reserved_bytes": 0,
                "swap_peak_used_bytes": actor_phase["swap_peak_used_bytes"],
            }
        )
    peak_allocated_bytes = max(
        phase["actor_peak_allocated_bytes"]
        for phase in phase_telemetry
    )
    peak_reserved_bytes = max(
        phase["actor_peak_reserved_bytes"]
        for phase in phase_telemetry
    )
    normalized_timing = {
        str(name): _number(value, f"timing_seconds.{name}")
        for name, value in sorted(timing_seconds.items())
    }
    expected_scientific_keys = {
        "all_outputs_truncated",
        "finite_gradients",
        "finite_losses",
        "high_truncation_rate",
        "nonzero_advantage_groups",
        "reward_variance_groups",
        "systematic_format_failure",
    }
    if scientific_evidence is None:
        normalized_scientific = {
            key: None for key in sorted(expected_scientific_keys)
        }
    else:
        if not isinstance(scientific_evidence, Mapping) or set(
            scientific_evidence
        ) != expected_scientific_keys:
            raise TrainingTelemetryError("scientific evidence keys do not match schema")
        normalized_scientific = dict(scientific_evidence)
        for key in (
            "all_outputs_truncated",
            "finite_gradients",
            "finite_losses",
            "high_truncation_rate",
            "systematic_format_failure",
        ):
            if type(normalized_scientific[key]) is not bool:
                raise TrainingTelemetryError(f"scientific evidence {key} must be boolean")
        for key in ("nonzero_advantage_groups", "reward_variance_groups"):
            normalized_scientific[key] = _integer(
                normalized_scientific[key],
                f"scientific_evidence.{key}",
            )
    payload = {
        "actor_logits": actor_logits,
        "allocator_retry_count": normalized_workers["actor"]["allocator_retry_count"],
        "attempt_id": attempt_id,
        "experiment_profile_id": experiment_profile_id,
        "global_step": global_step,
        "gpu_uuid": gpu_uuid,
        "host_peak_used_bytes": normalized_workers["actor"]["host_peak_used_bytes"],
        "host_total_memory_bytes": normalized_workers["actor"][
            "host_total_memory_bytes"
        ],
        "kind": "rememr1-training-step-telemetry-v2",
        "nvml_peak_used_bytes": normalized_workers["actor"]["nvml_peak_used_bytes"],
        "nvml_total_bytes": normalized_workers["actor"]["nvml_total_bytes"],
        "offload_profile": offload_profile,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "phase_telemetry": phase_telemetry,
        "post_step_allocated_bytes": normalized_workers["actor"][
            "post_step_allocated_bytes"
        ],
        "post_step_nvml_used_bytes": normalized_workers["actor"][
            "post_step_nvml_used_bytes"
        ],
        "post_step_reserved_bytes": normalized_workers["actor"][
            "post_step_reserved_bytes"
        ],
        "reference_logits": reference_logits,
        "schema_version": SCHEMA_VERSION,
        "scientific_evidence": normalized_scientific,
        "sealed_config_id": sealed_config_id,
        "sealed_config_sha256": sealed_config_sha256,
        "step_wall_seconds": _number(
            actor_worker["step_wall_seconds"], "actor_worker.step_wall_seconds"
        ),
        "swap_used_bytes": normalized_workers["actor"]["swap_used_bytes"],
        "timing_seconds": normalized_timing,
        "worker_roles": ["actor", "reference"],
    }
    return {**payload, "record_sha256": _canonical_sha256(payload)}


def _validate_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or "record_sha256" not in value:
        raise TrainingTelemetryError("telemetry step record is invalid")
    unsigned = dict(value)
    digest = unsigned.pop("record_sha256")
    if digest != _canonical_sha256(unsigned):
        raise TrainingTelemetryError("telemetry step self-hash mismatch")
    expected_keys = {
        "actor_logits",
        "allocator_retry_count",
        "attempt_id",
        "experiment_profile_id",
        "global_step",
        "gpu_uuid",
        "host_peak_used_bytes",
        "host_total_memory_bytes",
        "kind",
        "nvml_peak_used_bytes",
        "nvml_total_bytes",
        "offload_profile",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "phase_telemetry",
        "post_step_allocated_bytes",
        "post_step_nvml_used_bytes",
        "post_step_reserved_bytes",
        "record_sha256",
        "reference_logits",
        "schema_version",
        "scientific_evidence",
        "sealed_config_id",
        "sealed_config_sha256",
        "step_wall_seconds",
        "swap_used_bytes",
        "timing_seconds",
        "worker_roles",
    }
    if set(value) != expected_keys:
        raise TrainingTelemetryError("telemetry record keys do not match schema")
    if value["kind"] != "rememr1-training-step-telemetry-v2" or value[
        "schema_version"
    ] != 2:
        raise TrainingTelemetryError("telemetry record contract changed")
    if value["worker_roles"] != ["actor", "reference"]:
        raise TrainingTelemetryError("telemetry worker roles changed")
    if value["actor_logits"] is None or value["reference_logits"] is None:
        raise TrainingTelemetryError("telemetry logits evidence is incomplete")
    _validate_logits(value["actor_logits"], "actor")
    _validate_logits(value["reference_logits"], "reference")
    phases = value["phase_telemetry"]
    if not isinstance(phases, list) or [phase.get("name") for phase in phases] != list(
        REQUIRED_PHASES
    ):
        raise TrainingTelemetryError("telemetry phase inventory changed")
    if "save_checkpoint" not in value["timing_seconds"]:
        raise TrainingTelemetryError("telemetry does not include checkpoint-save timing")
    return dict(value)


def load_ledger(path: str | Path) -> dict[str, Any]:
    source = _safe_absolute(path, "telemetry ledger")
    try:
        raw = source.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingTelemetryError(f"cannot read telemetry ledger: {exc}") from exc
    if not isinstance(value, Mapping) or set(value) != {
        "identity",
        "kind",
        "ledger_sha256",
        "schema_version",
        "steps",
    }:
        raise TrainingTelemetryError("telemetry ledger keys do not match schema")
    unsigned = dict(value)
    digest = unsigned.pop("ledger_sha256")
    if digest != _canonical_sha256(unsigned):
        raise TrainingTelemetryError("telemetry ledger self-hash mismatch")
    if raw != _canonical_bytes(value) + b"\n":
        raise TrainingTelemetryError("telemetry ledger is not canonical")
    steps = value["steps"]
    if not isinstance(steps, list) or not steps:
        raise TrainingTelemetryError("telemetry ledger has no steps")
    validated = [_validate_record(record) for record in steps]
    if [record["global_step"] for record in validated] != sorted(
        {record["global_step"] for record in validated}
    ):
        raise TrainingTelemetryError("telemetry steps must be unique and increasing")
    identity = value["identity"]
    expected_identity = {
        key: validated[0][key]
        for key in (
            "attempt_id",
            "experiment_profile_id",
            "gpu_uuid",
            "offload_profile",
            "sealed_config_id",
            "sealed_config_sha256",
        )
    }
    if identity != expected_identity or any(
        any(record[key] != expected for key, expected in expected_identity.items())
        for record in validated
    ):
        raise TrainingTelemetryError("telemetry identity changed between steps")
    return dict(value)


def verify_success_ledger(
    attempt_dir: str | Path,
    *,
    expected_config_id: str,
    expected_config_sha256: str,
    expected_offload_profile: str,
    expected_final_step: int,
    verification_scope: str,
    length_stress: bool = False,
) -> dict[str, Any]:
    if verification_scope not in VERIFICATION_SCOPES:
        raise TrainingTelemetryError(
            f"telemetry verification scope must be one of {VERIFICATION_SCOPES}"
        )
    attempt = _safe_absolute(attempt_dir, "telemetry attempt")
    if attempt.is_symlink() or not attempt.is_dir():
        raise TrainingTelemetryError("telemetry attempt must be a regular directory")
    ledger_path = attempt / "telemetry.json"
    if ledger_path.is_symlink() or not ledger_path.is_file():
        raise TrainingTelemetryError("training attempt lacks telemetry.json")
    ledger = load_ledger(ledger_path)
    identity = ledger["identity"]
    expected_identity = {
        "attempt_id": attempt.name,
        "experiment_profile_id": PROFILE_ID,
        "offload_profile": expected_offload_profile,
        "sealed_config_id": expected_config_id,
        "sealed_config_sha256": expected_config_sha256,
    }
    for field, expected in expected_identity.items():
        if identity.get(field) != expected:
            raise TrainingTelemetryError(f"telemetry identity differs at {field}")
    if (
        isinstance(expected_final_step, bool)
        or not isinstance(expected_final_step, int)
        or expected_final_step < 1
        or ledger["steps"][-1]["global_step"] != expected_final_step
    ):
        raise TrainingTelemetryError("telemetry final optimizer step differs")

    scientific: list[Mapping[str, Any]] = []
    for record in ledger["steps"]:
        step = record["global_step"]
        if record["actor_logits"] is None or record["reference_logits"] is None:
            raise TrainingTelemetryError(
                f"telemetry step {step} lacks actor or reference logits evidence"
            )
        nvml_total = record["nvml_total_bytes"]
        host_total = record["host_total_memory_bytes"]
        if nvml_total < 1 or host_total < 1:
            raise TrainingTelemetryError(f"telemetry step {step} lacks resource capacity")
        if any(
            record[field] > nvml_total
            for field in (
                "nvml_peak_used_bytes",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
                "post_step_allocated_bytes",
                "post_step_nvml_used_bytes",
                "post_step_reserved_bytes",
            )
        ) or record["host_peak_used_bytes"] > host_total:
            raise TrainingTelemetryError(
                f"telemetry step {step} resource usage exceeds measured capacity"
            )
        if (
            record["peak_allocated_bytes"] > record["peak_reserved_bytes"]
            or record["post_step_allocated_bytes"]
            > record["post_step_reserved_bytes"]
        ):
            raise TrainingTelemetryError(
                f"telemetry step {step} allocator identity is inconsistent"
            )
        if length_stress:
            continue
        evidence = record["scientific_evidence"]
        if any(value is None for value in evidence.values()):
            raise TrainingTelemetryError(
                f"telemetry step {step} lacks scientific evidence"
            )
        scientific.append(evidence)
        if evidence["finite_losses"] is not True:
            raise TrainingScientificStop(f"telemetry step {step} has non-finite losses")
        if evidence["finite_gradients"] is not True:
            raise TrainingScientificStop(f"telemetry step {step} has non-finite gradients")
        if verification_scope == "capacity" and evidence["all_outputs_truncated"]:
            raise TrainingScientificStop(f"telemetry step {step} truncated every output")
        if verification_scope == "capacity" and evidence["systematic_format_failure"]:
            raise TrainingScientificStop(
                f"telemetry step {step} has systematic format failure"
            )
    if verification_scope == "scientific" and scientific and all(
        item["all_outputs_truncated"] for item in scientific
    ):
        raise TrainingScientificStop(
            "training telemetry truncated every output throughout the attempt"
        )
    if verification_scope == "scientific" and scientific and all(
        item["systematic_format_failure"] for item in scientific
    ):
        raise TrainingScientificStop(
            "training telemetry has systematic format failure throughout the attempt"
        )
    return ledger


def append_step(path: str | Path, record: Mapping[str, Any]) -> Path:
    destination = _safe_absolute(path, "telemetry ledger")
    validated = _validate_record(record)
    existing = load_ledger(destination) if destination.exists() else None
    steps = [] if existing is None else list(existing["steps"])
    if steps and validated["global_step"] <= steps[-1]["global_step"]:
        raise TrainingTelemetryError("telemetry step is not strictly increasing")
    identity = {
        key: validated[key]
        for key in (
            "attempt_id",
            "experiment_profile_id",
            "gpu_uuid",
            "offload_profile",
            "sealed_config_id",
            "sealed_config_sha256",
        )
    }
    if existing is not None and existing["identity"] != identity:
        raise TrainingTelemetryError("telemetry append identity drifted")
    payload = {
        "identity": identity,
        "kind": "rememr1-training-telemetry-ledger-v2",
        "schema_version": SCHEMA_VERSION,
        "steps": [*steps, validated],
    }
    sealed = {**payload, "ledger_sha256": _canonical_sha256(payload)}
    _atomic_write(destination, _canonical_bytes(sealed) + b"\n")
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser(
        "verify-success", help="verify strict success telemetry for one attempt"
    )
    verify.add_argument("--attempt-dir", type=Path, required=True)
    verify.add_argument("--expected-config-id", required=True)
    verify.add_argument("--expected-config-sha256", required=True)
    verify.add_argument("--expected-offload-profile", choices=("r0", "r1"), required=True)
    verify.add_argument("--expected-final-step", type=int, required=True)
    verify.add_argument(
        "--verification-scope",
        choices=VERIFICATION_SCOPES,
        required=True,
    )
    verify.add_argument("--length-stress", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        ledger = verify_success_ledger(
            args.attempt_dir,
            expected_config_id=args.expected_config_id,
            expected_config_sha256=args.expected_config_sha256,
            expected_offload_profile=args.expected_offload_profile,
            expected_final_step=args.expected_final_step,
            verification_scope=args.verification_scope,
            length_stress=args.length_stress,
        )
    except TrainingScientificStop as exc:
        print(json.dumps({"error": str(exc), "status": "scientific-stop"}), file=sys.stderr)
        return 42
    except Exception as exc:
        print(
            json.dumps(
                {"error": f"{type(exc).__name__}: {exc}", "status": "invalid"}
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "final_step": ledger["steps"][-1]["global_step"],
                "ledger_sha256": ledger["ledger_sha256"],
                "status": "verified",
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "PROFILE_ID",
    "REQUIRED_PHASES",
    "VERIFICATION_SCOPES",
    "SCHEMA_VERSION",
    "TrainingTelemetryError",
    "TrainingScientificStop",
    "append_step",
    "create_step_record",
    "load_ledger",
    "verify_success_ledger",
]


if __name__ == "__main__":
    raise SystemExit(main())
