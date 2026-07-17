"""Create and verify self-hashed RTX 5090 B/C budget projections."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
GPU_LIMIT_RMB = 450.0
TOTAL_LIMIT_RMB = 500.0
CONTINGENCY = 1.20
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DECISION_COUNTS = {
    "bc40": {"fresh_init": 4, "resume": 4, "compute": 86, "save": 6},
    "bc80": {"fresh_init": 0, "resume": 6, "compute": 80, "save": 4},
}


class CostGateError(RuntimeError):
    """Raised when a budget projection is incomplete or understates cost."""


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
        raise CostGateError(f"cost projection is not canonical JSON: {exc}") from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CostGateError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise CostGateError(f"{label} must be finite and non-negative")
    return result


def _load_json(path: Path, label: str, *, canonical: bool = False) -> Mapping[str, Any]:
    if path.is_symlink():
        raise CostGateError(f"{label} must not be a symlink")
    try:
        source = path.resolve(strict=True)
        raw = source.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CostGateError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise CostGateError(f"{label} must be an object")
    if canonical and raw != _canonical_bytes(value) + b"\n":
        raise CostGateError(f"{label} is not canonical JSON")
    return value


def _file_record(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise CostGateError(f"{label} must contain path and sha256")
    path = Path(str(value["path"])).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise CostGateError(f"{label}.path must be an absolute regular file")
    path = path.resolve(strict=True)
    if not path.is_file():
        raise CostGateError(f"{label}.path must be a regular file")
    digest = value["sha256"]
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise CostGateError(f"{label}.sha256 is invalid")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != digest:
        raise CostGateError(f"{label} file hash changed")
    return {"path": str(path), "sha256": digest}


def create_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "decision",
        "disk_remaining_rmb",
        "evidence",
        "gpu_cost_done_rmb",
        "gpu_hourly_rate_rmb",
        "measurements",
        "non_gpu_cost_done_rmb",
        "ops_reserve_rmb",
        "schema_version",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise CostGateError("cost input keys do not match schema")
    if value["schema_version"] != SCHEMA_VERSION:
        raise CostGateError("cost input schema version is unsupported")
    decision = value["decision"]
    if decision not in _DECISION_COUNTS:
        raise CostGateError("cost decision must be bc40 or bc80")
    measurements = value["measurements"]
    measurement_keys = {
        "evaluation_remaining_seconds",
        "t_artifacts_seconds",
        "t_compute_seconds",
        "t_init_seconds",
        "t_resume_seconds",
        "t_save_seconds",
    }
    if not isinstance(measurements, Mapping) or set(measurements) != measurement_keys:
        raise CostGateError("cost measurement keys do not match schema")
    normalized_measurements = {
        key: _number(measurements[key], f"measurements.{key}")
        for key in sorted(measurements)
    }
    evidence = value["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise CostGateError("cost input requires measured evidence records")
    normalized_evidence = [
        _file_record(record, f"evidence[{index}]")
        for index, record in enumerate(evidence)
    ]
    counts = _DECISION_COUNTS[decision]
    train_seconds = (
        counts["fresh_init"] * normalized_measurements["t_init_seconds"]
        + counts["resume"] * normalized_measurements["t_resume_seconds"]
        + counts["compute"] * normalized_measurements["t_compute_seconds"]
        + counts["save"] * normalized_measurements["t_save_seconds"]
        + normalized_measurements["t_artifacts_seconds"]
    )
    evaluation_seconds = normalized_measurements["evaluation_remaining_seconds"]
    gpu_rate = _number(value["gpu_hourly_rate_rmb"], "gpu_hourly_rate_rmb")
    gpu_done = _number(value["gpu_cost_done_rmb"], "gpu_cost_done_rmb")
    non_gpu_done = _number(value["non_gpu_cost_done_rmb"], "non_gpu_cost_done_rmb")
    disk_remaining = _number(value["disk_remaining_rmb"], "disk_remaining_rmb")
    ops_reserve = _number(value["ops_reserve_rmb"], "ops_reserve_rmb")
    minimum_ops_reserve = max(0.0, 50.0 - non_gpu_done - disk_remaining)
    if ops_reserve < minimum_ops_reserve:
        raise CostGateError(
            "ops reserve is below the preregistered 50 RMB non-GPU buffer"
        )
    gpu_target = gpu_done + gpu_rate * CONTINGENCY * (
        train_seconds + evaluation_seconds
    ) / 3600.0
    total_target = gpu_target + non_gpu_done + disk_remaining + ops_reserve
    passed = gpu_target <= GPU_LIMIT_RMB and total_target <= TOTAL_LIMIT_RMB
    payload = {
        "contingency": CONTINGENCY,
        "counts": counts,
        "decision": decision,
        "evidence": normalized_evidence,
        "evaluation_remaining_raw_seconds": evaluation_seconds,
        "gpu_cost_done_rmb": gpu_done,
        "gpu_hourly_rate_rmb": gpu_rate,
        "gpu_limit_rmb": GPU_LIMIT_RMB,
        "gpu_target_rmb": gpu_target,
        "input_sha256": _canonical_sha256(dict(value)),
        "measurements": normalized_measurements,
        "non_gpu_cost_done_rmb": non_gpu_done,
        "disk_remaining_rmb": disk_remaining,
        "ops_reserve_rmb": ops_reserve,
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if passed else "blocked",
        "total_limit_rmb": TOTAL_LIMIT_RMB,
        "total_target_rmb": total_target,
        "train_raw_seconds": train_seconds,
    }
    return {**payload, "projection_sha256": _canonical_sha256(payload)}


def verify_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or "projection_sha256" not in value:
        raise CostGateError("cost projection is invalid")
    expected_keys = {
        "contingency",
        "counts",
        "decision",
        "disk_remaining_rmb",
        "evaluation_remaining_raw_seconds",
        "evidence",
        "gpu_cost_done_rmb",
        "gpu_hourly_rate_rmb",
        "gpu_limit_rmb",
        "gpu_target_rmb",
        "input_sha256",
        "measurements",
        "non_gpu_cost_done_rmb",
        "ops_reserve_rmb",
        "projection_sha256",
        "schema_version",
        "status",
        "total_limit_rmb",
        "total_target_rmb",
        "train_raw_seconds",
    }
    if set(value) != expected_keys:
        raise CostGateError("cost projection keys do not match schema")
    unsigned = dict(value)
    digest = unsigned.pop("projection_sha256")
    if digest != _canonical_sha256(unsigned):
        raise CostGateError("cost projection self-hash mismatch")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise CostGateError("cost projection schema version is unsupported")
    decision = value.get("decision")
    if decision not in _DECISION_COUNTS or value.get("counts") != _DECISION_COUNTS[decision]:
        raise CostGateError("cost projection stage counts drifted")
    measurements = value.get("measurements")
    if not isinstance(measurements, Mapping):
        raise CostGateError("cost projection measurements are invalid")
    normalized = {
        key: _number(measurements.get(key), f"measurements.{key}")
        for key in (
            "evaluation_remaining_seconds",
            "t_artifacts_seconds",
            "t_compute_seconds",
            "t_init_seconds",
            "t_resume_seconds",
            "t_save_seconds",
        )
    }
    counts = _DECISION_COUNTS[decision]
    expected_train = (
        counts["fresh_init"] * normalized["t_init_seconds"]
        + counts["resume"] * normalized["t_resume_seconds"]
        + counts["compute"] * normalized["t_compute_seconds"]
        + counts["save"] * normalized["t_save_seconds"]
        + normalized["t_artifacts_seconds"]
    )
    if _number(value.get("train_raw_seconds"), "train_raw_seconds") != expected_train:
        raise CostGateError("cost projection training formula drifted")
    if (
        _number(
            value.get("evaluation_remaining_raw_seconds"),
            "evaluation_remaining_raw_seconds",
        )
        != normalized["evaluation_remaining_seconds"]
    ):
        raise CostGateError("cost projection evaluation seconds drifted")
    gpu_done = _number(value.get("gpu_cost_done_rmb"), "gpu_cost_done_rmb")
    gpu_rate = _number(value.get("gpu_hourly_rate_rmb"), "gpu_hourly_rate_rmb")
    expected_gpu = gpu_done + gpu_rate * CONTINGENCY * (
        expected_train + normalized["evaluation_remaining_seconds"]
    ) / 3600.0
    expected_total = (
        expected_gpu
        + (non_gpu_done := _number(
            value.get("non_gpu_cost_done_rmb"), "non_gpu_cost_done_rmb"
        ))
        + (disk_remaining := _number(
            value.get("disk_remaining_rmb"), "disk_remaining_rmb"
        ))
        + (ops_reserve := _number(value.get("ops_reserve_rmb"), "ops_reserve_rmb"))
    )
    if ops_reserve < max(0.0, 50.0 - non_gpu_done - disk_remaining):
        raise CostGateError("cost projection understates the non-GPU reserve")
    if _number(value.get("gpu_target_rmb"), "gpu_target_rmb") != expected_gpu:
        raise CostGateError("cost projection GPU formula drifted")
    if _number(value.get("total_target_rmb"), "total_target_rmb") != expected_total:
        raise CostGateError("cost projection total formula drifted")
    if value.get("contingency") != CONTINGENCY:
        raise CostGateError("cost contingency drifted")
    if value.get("gpu_limit_rmb") != GPU_LIMIT_RMB or value.get(
        "total_limit_rmb"
    ) != TOTAL_LIMIT_RMB:
        raise CostGateError("cost limits drifted")
    input_sha = value.get("input_sha256")
    if not isinstance(input_sha, str) or _SHA256.fullmatch(input_sha) is None:
        raise CostGateError("cost input SHA is invalid")
    for index, record in enumerate(value.get("evidence", ())):
        _file_record(record, f"evidence[{index}]")
    if value.get("status") != "pass":
        raise CostGateError("cost projection exceeds the preregistered budget")
    if _number(value.get("gpu_target_rmb"), "gpu_target_rmb") > GPU_LIMIT_RMB:
        raise CostGateError("GPU target exceeds 450 RMB")
    if _number(value.get("total_target_rmb"), "total_target_rmb") > TOTAL_LIMIT_RMB:
        raise CostGateError("total target exceeds 500 RMB")
    return dict(value)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(os.path.abspath(path.expanduser()))
    if path.exists() or path.is_symlink():
        raise CostGateError(f"refusing to overwrite cost projection: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    project = commands.add_parser("project")
    project.add_argument("--input", type=Path, required=True)
    project.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--projection", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "project":
            result = create_projection(_load_json(args.input, "cost input"))
            _atomic_json(args.output, result)
            if result["status"] != "pass":
                print(json.dumps(result, sort_keys=True))
                return 42
        else:
            result = verify_projection(
                _load_json(args.projection, "cost projection", canonical=True)
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
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONTINGENCY",
    "CostGateError",
    "GPU_LIMIT_RMB",
    "TOTAL_LIMIT_RMB",
    "create_projection",
    "main",
    "verify_projection",
]
