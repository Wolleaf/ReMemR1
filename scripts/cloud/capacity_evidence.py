"""Classify G2 telemetry and seal or verify an RTX 5090 capacity profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Collection, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
PROFILE_ID = "rtx5090-32g-qwen35-2b-v1"
GIB = 1024**3
MIB = 1024**2
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_RTX_5090_NAME = re.compile(r"(?:^|\s)GEFORCE\s+RTX\s+5090$", re.IGNORECASE)
_UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_CONSUMPTION_KIND = "r1-approval-consumption"
_APPROVAL_MARKER_KEYS = {
    "active_config_tree_sha256",
    "approval_marker_sha256",
    "approval_nonce",
    "budget_projection_sha256",
    "capacity_reason",
    "commit",
    "cpu_handoff_sha256",
    "profile_id",
    "r0_capacity_evidence_sha256",
    "r0_terminal_sha256",
    "requested_profile",
    "schema_version",
    "status",
}
_CONSUMPTION_KEYS = {
    "schema_version",
    "kind",
    "status",
    "approval_marker_sha256",
    "approval_marker_file_sha256",
    "approval_marker_path",
    "launcher_dir",
    "launcher_request_sha256",
    "pipeline_identity_sha256",
    "pipeline_dir",
    "experiment_profile_id",
    "git_commit",
    "phase",
    "offload_profile",
    "consumed_at",
    "consumption_sha256",
}
_LAUNCHER_REQUEST_KEYS = {
    "schema_version",
    "phase",
    "experiment_profile_id",
    "expected_commit",
    "offload_profile",
    "r1_approval",
    "r1_approval_file_sha256",
    "budget_projection",
    "budget_projection_file_sha256",
    "keep_running",
    "retry_failed_stage",
    "dry_run",
    "requested_at",
}

CONFIG_TASKS = (
    "g2a_qwen35_2b_5090",
    "g2b_qwen35_2b_5090_step1",
    "g2b_qwen35_2b_5090_resume5",
    "g2_length_stress_qwen35_2b_5090",
    "b_pilot_qwen35_2b_5090",
    "c_pilot_qwen35_2b_5090",
    "b20_qwen35_2b_5090",
    "c20_qwen35_2b_5090",
    "b40_qwen35_2b_5090",
    "c40_qwen35_2b_5090",
    "b60_qwen35_2b_5090",
    "c60_qwen35_2b_5090",
    "b80_qwen35_2b_5090",
    "c80_qwen35_2b_5090",
)
ATTEMPT_ORDER = ("g2a", "g2b-step1", "g2b-resume5", "length-stress")
ATTEMPT_CONFIG_TASK = {
    "g2a": CONFIG_TASKS[0],
    "g2b-step1": CONFIG_TASKS[1],
    "g2b-resume5": CONFIG_TASKS[2],
    "length-stress": CONFIG_TASKS[3],
}
CAPACITY_TERMINAL_REASONS = {"allocator_pressure", "gpu_memory_headroom", "oom"}
_COLORS = {"green": 0, "yellow": 1, "red": 2}


class CapacityEvidenceError(RuntimeError):
    """Raised when capacity evidence is incomplete, unsafe, or identity-drifted."""


def canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CapacityEvidenceError(f"value is not canonical JSON: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _seal(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    sealed = json.loads(json.dumps(payload, allow_nan=False))
    sealed.pop(field, None)
    sealed[field] = canonical_sha256(sealed)
    return sealed


def _verify_self_hash(payload: Mapping[str, Any], field: str) -> str:
    digest = payload.get(field)
    if not _is_sha256(digest):
        raise CapacityEvidenceError(f"{field} is malformed")
    content = dict(payload)
    content.pop(field, None)
    if digest != canonical_sha256(content):
        raise CapacityEvidenceError(f"{field} does not match the canonical payload")
    return digest


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def _require_sha256(value: Any, label: str) -> str:
    if not _is_sha256(value):
        raise CapacityEvidenceError(f"{label} must be a lowercase SHA-256")
    return value


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CapacityEvidenceError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise CapacityEvidenceError(
            f"{label} keys mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def expected_config_ids(profile: str) -> tuple[str, ...]:
    if profile not in {"R0", "R1"}:
        raise CapacityEvidenceError(f"selected_profile must be R0 or R1, observed {profile!r}")
    suffix = profile.lower()
    return tuple(f"{task}_{suffix}" for task in CONFIG_TASKS)


def _validate_selected_configs(value: Any, profile: str) -> dict[str, str]:
    configs = _require_mapping(value, "selected_configs")
    expected = set(expected_config_ids(profile))
    _exact_keys(configs, expected, "selected_configs")
    normalized: dict[str, str] = {}
    for config_id in sorted(configs):
        if _SAFE_ID.fullmatch(config_id) is None:
            raise CapacityEvidenceError(f"selected config ID is unsafe: {config_id!r}")
        normalized[config_id] = _require_sha256(
            configs[config_id], f"selected_configs.{config_id}"
        )
    return normalized


def _validate_gpu(value: Any) -> dict[str, Any]:
    gpu = _require_mapping(value, "gpu")
    _exact_keys(gpu, {"name", "total_vram_bytes", "uuid"}, "gpu")
    name = gpu["name"]
    uuid = gpu["uuid"]
    total = gpu["total_vram_bytes"]
    normalized_name = " ".join(name.split()) if isinstance(name, str) else ""
    if _RTX_5090_NAME.search(normalized_name) is None:
        raise CapacityEvidenceError("gpu.name is not a GeForce RTX 5090")
    if not isinstance(uuid, str) or not uuid.startswith("GPU-"):
        raise CapacityEvidenceError("gpu.uuid is malformed")
    if isinstance(total, bool) or not isinstance(total, int) or total < 31 * GIB:
        raise CapacityEvidenceError("gpu.total_vram_bytes is below 31 GiB")
    return {"name": name, "total_vram_bytes": total, "uuid": uuid}


def validate_identity(value: Any) -> dict[str, Any]:
    identity = _require_mapping(value, "identity")
    expected = {
        "active_config_tree_sha256",
        "commit",
        "cpu_handoff_sha256",
        "gpu",
        "profile_id",
        "selected_config_set_sha256",
        "selected_profile",
    }
    _exact_keys(identity, expected, "identity")
    if identity["profile_id"] != PROFILE_ID:
        raise CapacityEvidenceError(f"profile_id must be {PROFILE_ID!r}")
    commit = identity["commit"]
    if not isinstance(commit, str) or _HEX40.fullmatch(commit) is None:
        raise CapacityEvidenceError("identity.commit must be a lowercase 40-character Git SHA")
    profile = identity["selected_profile"]
    expected_config_ids(profile)
    return {
        "active_config_tree_sha256": _require_sha256(
            identity["active_config_tree_sha256"], "identity.active_config_tree_sha256"
        ),
        "commit": commit,
        "cpu_handoff_sha256": _require_sha256(
            identity["cpu_handoff_sha256"], "identity.cpu_handoff_sha256"
        ),
        "gpu": _validate_gpu(identity["gpu"]),
        "profile_id": PROFILE_ID,
        "selected_config_set_sha256": _require_sha256(
            identity["selected_config_set_sha256"],
            "identity.selected_config_set_sha256",
        ),
        "selected_profile": profile,
    }


def build_identity(
    *,
    commit: str,
    cpu_handoff_sha256: str,
    active_config_tree_sha256: str,
    selected_profile: str,
    selected_configs: Mapping[str, str],
    gpu: Mapping[str, Any],
) -> dict[str, Any]:
    configs = _validate_selected_configs(selected_configs, selected_profile)
    identity = {
        "active_config_tree_sha256": active_config_tree_sha256,
        "commit": commit,
        "cpu_handoff_sha256": cpu_handoff_sha256,
        "gpu": dict(gpu),
        "profile_id": PROFILE_ID,
        "selected_config_set_sha256": canonical_sha256(configs),
        "selected_profile": selected_profile,
    }
    return validate_identity(identity)


def _same_identity(observed: Any, expected: Mapping[str, Any], label: str) -> None:
    normalized = validate_identity(observed)
    if normalized != expected:
        changed = sorted(key for key in expected if normalized.get(key) != expected.get(key))
        raise CapacityEvidenceError(f"{label} identity drifted in fields: {changed}")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) and converted >= 0 else None


def _metric(status: str, observed: Any, rule: str, *, r1_trigger: bool = False) -> dict[str, Any]:
    if status not in _COLORS:
        raise AssertionError(status)
    return {
        "observed": observed,
        "r1_trigger": r1_trigger,
        "rule": rule,
        "status": status,
    }


def _max_step_value(steps: Sequence[Any], field: str) -> float | None:
    values = [
        value
        for step in steps
        if isinstance(step, Mapping) and (value := _number(step.get(field))) is not None
    ]
    return max(values) if values else None


def _classify_peak(value: float | None, green_max: int, red_min: int, *, oom: bool) -> str:
    if oom or value is None or value >= red_min:
        return "red"
    return "green" if value <= green_max else "yellow"


def _resident_growth(steps: Sequence[Any]) -> tuple[str, dict[str, Any]]:
    fields = (
        "post_step_allocated_bytes",
        "post_step_reserved_bytes",
        "post_step_nvml_used_bytes",
    )
    details: dict[str, Any] = {}
    worst_status = "green"
    found = False
    for field in fields:
        values: list[float] = []
        for step_number in (3, 4, 5):
            matches = [
                _number(step.get(field))
                for step in steps
                if isinstance(step, Mapping) and step.get("step") == step_number
            ]
            matches = [value for value in matches if value is not None]
            if not matches:
                values = []
                break
            values.append(max(matches))
        if not values:
            continue
        found = True
        delta = values[2] - values[0]
        monotonic = values[0] <= values[1] <= values[2]
        if monotonic and delta > GIB:
            status = "red"
        elif delta > 512 * MIB:
            status = "yellow"
        else:
            status = "green"
        details[field] = {"delta_step5_minus_step3": int(delta), "monotonic": monotonic}
        if _COLORS[status] > _COLORS[worst_status]:
            worst_status = status
    if not found:
        return "red", {"missing_steps": [3, 4, 5]}
    return worst_status, details


def classify_telemetry(telemetry: Mapping[str, Any]) -> dict[str, Any]:
    data = _require_mapping(telemetry, "telemetry")
    steps = data.get("steps")
    if not isinstance(steps, list):
        raise CapacityEvidenceError("telemetry.steps must be an array")
    if not isinstance(data.get("oom"), bool):
        raise CapacityEvidenceError("telemetry.oom must be a boolean")
    if not isinstance(data.get("non_capacity_failure"), bool):
        raise CapacityEvidenceError("telemetry.non_capacity_failure must be a boolean")
    oom = data["oom"]
    terminal_reason = data.get("terminal_reason")
    if terminal_reason is not None and terminal_reason not in CAPACITY_TERMINAL_REASONS | {
        "non_capacity"
    }:
        raise CapacityEvidenceError("telemetry.terminal_reason is not preregistered")
    if oom and terminal_reason != "oom":
        raise CapacityEvidenceError("OOM telemetry must use terminal_reason='oom'")

    allocated = _max_step_value(steps, "peak_allocated_bytes")
    reserved = _max_step_value(steps, "peak_reserved_bytes")
    nvml_used = _max_step_value(steps, "nvml_peak_used_bytes")
    nvml_total = _number(data.get("nvml_total_bytes"))
    if nvml_total is None:
        totals = [
            value
            for step in steps
            if isinstance(step, Mapping)
            and (value := _number(step.get("nvml_total_bytes"))) is not None
        ]
        nvml_total = min(totals) if totals else None

    metrics: dict[str, dict[str, Any]] = {}
    metrics["pytorch_peak_allocated"] = _metric(
        _classify_peak(allocated, 27 * GIB, 29 * GIB, oom=oom),
        None if allocated is None else int(allocated),
        "green <=27 GiB; yellow >27 and <29 GiB; red >=29 GiB or OOM",
        r1_trigger=True,
    )
    metrics["pytorch_peak_reserved"] = _metric(
        _classify_peak(reserved, 28 * GIB, 30 * GIB, oom=oom),
        None if reserved is None else int(reserved),
        "green <=28 GiB; yellow >28 and <30 GiB; red >=30 GiB or OOM",
        r1_trigger=True,
    )
    headroom = None if nvml_total is None or nvml_used is None else nvml_total - nvml_used
    if oom or nvml_used is None or headroom is None or headroom < GIB:
        nvml_status = "red"
    elif nvml_used <= 29 * GIB and headroom >= 2 * GIB:
        nvml_status = "green"
    else:
        nvml_status = "yellow"
    metrics["nvml_peak_used"] = _metric(
        nvml_status,
        {
            "headroom_bytes": None if headroom is None else int(headroom),
            "peak_used_bytes": None if nvml_used is None else int(nvml_used),
            "total_bytes": None if nvml_total is None else int(nvml_total),
        },
        "green peak <=29 GiB and headroom >=2 GiB; yellow headroom 1-2 GiB; "
        "red headroom <1 GiB or OOM",
        r1_trigger=True,
    )

    resident_status, resident_details = _resident_growth(steps)
    metrics["resident_growth_step3_to_step5"] = _metric(
        resident_status,
        resident_details,
        "green delta <=512 MiB; yellow >512 MiB; red monotonic delta >1 GiB",
    )

    retries = data.get("allocator_retry_count")
    fragmentation_value = data.get("allocator_fragmentation_failure")
    fragmentation = fragmentation_value is True
    if not isinstance(fragmentation_value, bool):
        allocator_status = "red"
    elif isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        allocator_status = "red"
    elif fragmentation or retries >= 2:
        allocator_status = "red"
    elif retries == 1:
        allocator_status = "yellow"
    else:
        allocator_status = "green"
    metrics["allocator"] = _metric(
        allocator_status,
        {"fragmentation_failure": fragmentation, "retry_count": retries},
        "green zero retries; yellow one pressure signal; "
        "red repeated retries or fragmentation failure",
        r1_trigger=True,
    )

    host_peak = _number(data.get("host_peak_used_bytes"))
    host_total = _number(data.get("host_total_memory_bytes"))
    swap_used = _number(data.get("swap_used_bytes"))
    host_ratio = None if host_peak is None or not host_total else host_peak / host_total
    page_fault_thrash = data.get("sustained_page_fault_thrash") is True
    if (
        swap_used is None
        or swap_used > 0
        or host_ratio is None
        or host_ratio >= 0.9
        or page_fault_thrash
    ):
        host_status = "red"
    elif host_ratio >= 0.8:
        host_status = "yellow"
    else:
        host_status = "green"
    metrics["host_ram"] = _metric(
        host_status,
        {
            "peak_ratio": host_ratio,
            "peak_used_bytes": None if host_peak is None else int(host_peak),
            "sustained_page_fault_thrash": page_fault_thrash,
            "swap_used_bytes": None if swap_used is None else int(swap_used),
            "total_bytes": None if host_total is None else int(host_total),
        },
        "green peak <80%; yellow 80-90%; red >=90%, swap use, or page-fault thrash",
    )

    execution_fields = (
        "fresh_process_resume_completed",
        "g2a_completed",
        "g2b_completed",
        "length_stress_completed",
    )
    execution_ok = (
        all(data.get(field) is True for field in execution_fields)
        and terminal_reason is None
        and data["non_capacity_failure"] is False
    )
    step_time_jitter = data.get("step_time_jitter") is True
    execution_status = "red" if not execution_ok else "yellow" if step_time_jitter else "green"
    metrics["execution"] = _metric(
        execution_status,
        {
            **{field: data.get(field) for field in execution_fields},
            "non_capacity_failure": data["non_capacity_failure"],
            "step_time_jitter": step_time_jitter,
            "terminal_reason": terminal_reason,
        },
        "all G2 stages and fresh-process resume must complete",
    )
    numerics_ok = all(
        data.get(field) is True
        for field in (
            "adapter_update_nonzero",
            "base_model_unchanged",
            "finite_gradients",
            "finite_losses",
        )
    )
    numerical_anomaly = data.get("numerical_anomaly_requires_explanation") is True
    numerics_status = "red" if not numerics_ok else "yellow" if numerical_anomaly else "green"
    metrics["numerics"] = _metric(
        numerics_status,
        {
            field: data.get(field)
            for field in (
                "adapter_update_nonzero",
                "base_model_unchanged",
                "finite_gradients",
                "finite_losses",
            )
        }
        | {"anomaly_requires_explanation": numerical_anomaly},
        "losses/gradients finite, adapter changed, base unchanged",
    )
    variance_groups = data.get("reward_variance_groups")
    advantage_groups = data.get("nonzero_advantage_groups")
    grpo_valid = all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 1
        for value in (variance_groups, advantage_groups)
    )
    low_reward_variance = data.get("low_reward_variance") is True
    grpo_status = "red" if not grpo_valid else "yellow" if low_reward_variance else "green"
    metrics["grpo"] = _metric(
        grpo_status,
        {
            "nonzero_advantage_groups": advantage_groups,
            "reward_variance_groups": variance_groups,
            "low_reward_variance": low_reward_variance,
        },
        "at least one reward-variance group and one nonzero-advantage group",
    )
    format_ok = (
        data.get("systematic_format_failure") is False
        and data.get("all_outputs_truncated") is False
    )
    high_truncation_rate = data.get("high_truncation_rate") is True
    format_status = "red" if not format_ok else "yellow" if high_truncation_rate else "green"
    metrics["format"] = _metric(
        format_status,
        {
            "all_outputs_truncated": data.get("all_outputs_truncated"),
            "systematic_format_failure": data.get("systematic_format_failure"),
            "high_truncation_rate": high_truncation_rate,
        },
        "no systematic parser/format failure and not all outputs truncated",
    )

    overall = max((entry["status"] for entry in metrics.values()), key=_COLORS.__getitem__)
    non_green = [name for name, entry in metrics.items() if entry["status"] != "green"]
    r1_trigger_metrics = [
        name for name in non_green if metrics[name]["r1_trigger"]
    ]
    explicit_capacity_stop = terminal_reason in CAPACITY_TERMINAL_REASONS
    non_capacity_failure = data["non_capacity_failure"]
    if explicit_capacity_stop:
        r1_eligible = not non_capacity_failure
    else:
        disqualifying = [name for name in non_green if not metrics[name]["r1_trigger"]]
        r1_eligible = bool(r1_trigger_metrics) and not disqualifying and not non_capacity_failure
    if terminal_reason == "oom":
        eligibility_reason = "oom"
    elif terminal_reason in {"allocator_pressure", "gpu_memory_headroom"}:
        eligibility_reason = terminal_reason
    elif "allocator" in r1_trigger_metrics:
        eligibility_reason = "allocator_pressure"
    elif r1_trigger_metrics:
        eligibility_reason = "gpu_memory_headroom"
    else:
        eligibility_reason = None
    return {
        "eligibility_reason": eligibility_reason,
        "metrics": metrics,
        "non_green_metrics": non_green,
        "overall": overall,
        "r1_eligible": r1_eligible,
    }


def _attempt_payload(value: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    metadata = _require_mapping(value, "attempt_metadata")
    _exact_keys(metadata, {"attempts", "identity"}, "attempt_metadata")
    return _require_mapping(metadata["identity"], "attempt_metadata.identity"), _require_mapping(
        metadata["attempts"], "attempt_metadata.attempts"
    )


def validate_attempt_metadata(
    value: Any,
    *,
    identity: Mapping[str, Any],
    selected_configs: Mapping[str, str],
    require_complete_success: bool,
) -> dict[str, Any]:
    observed_identity, attempts = _attempt_payload(value)
    _same_identity(observed_identity, identity, "attempt metadata")
    actual = set(attempts)
    allowed = set(ATTEMPT_ORDER)
    if not actual or not actual <= allowed:
        raise CapacityEvidenceError("attempt metadata has missing or unknown G2 attempts")
    if require_complete_success and actual != allowed:
        raise CapacityEvidenceError("a sealed capacity profile requires all four G2 attempts")
    normalized: dict[str, Any] = {}
    previous_stage: str | None = None
    previous_digest: str | None = None
    for stage in ATTEMPT_ORDER:
        if stage not in attempts:
            continue
        if previous_stage is None and stage != "g2a":
            raise CapacityEvidenceError(f"{stage} exists without its predecessor")
        attempt = _require_mapping(attempts[stage], f"attempts.{stage}")
        expected_keys = {
            "attempt_id",
            "config_id",
            "config_sha256",
            "digest",
            "failure_kind",
            "predecessor",
            "status",
        }
        _exact_keys(attempt, expected_keys, f"attempts.{stage}")
        attempt_id = attempt["attempt_id"]
        if not isinstance(attempt_id, str) or _SAFE_ID.fullmatch(attempt_id) is None:
            raise CapacityEvidenceError(f"attempts.{stage}.attempt_id is unsafe")
        config_id = f"{ATTEMPT_CONFIG_TASK[stage]}_{identity['selected_profile'].lower()}"
        if attempt["config_id"] != config_id:
            raise CapacityEvidenceError(f"attempts.{stage}.config_id drifted")
        config_sha = _require_sha256(attempt["config_sha256"], f"attempts.{stage}.config_sha256")
        if selected_configs.get(config_id) != config_sha:
            raise CapacityEvidenceError(f"attempts.{stage} does not bind the sealed config")
        digest = _require_sha256(attempt["digest"], f"attempts.{stage}.digest")
        status = attempt["status"]
        failure_kind = attempt["failure_kind"]
        if status not in {"failed", "success"}:
            raise CapacityEvidenceError(f"attempts.{stage}.status is invalid")
        if status == "success" and failure_kind is not None:
            raise CapacityEvidenceError(f"successful attempt {stage} has a failure kind")
        if status == "failed" and failure_kind not in {"capacity", "non_capacity"}:
            raise CapacityEvidenceError(f"failed attempt {stage} lacks a valid failure kind")
        predecessor = attempt["predecessor"]
        if previous_stage is None:
            if predecessor is not None:
                raise CapacityEvidenceError("g2a must not claim a predecessor")
        else:
            predecessor = _require_mapping(predecessor, f"attempts.{stage}.predecessor")
            _exact_keys(predecessor, {"digest", "stage"}, f"attempts.{stage}.predecessor")
            if (
                predecessor.get("stage") != previous_stage
                or predecessor.get("digest") != previous_digest
            ):
                raise CapacityEvidenceError(f"attempts.{stage}.predecessor drifted")
        if require_complete_success and status != "success":
            raise CapacityEvidenceError(f"attempts.{stage} did not succeed")
        normalized[stage] = json.loads(json.dumps(attempt, allow_nan=False))
        previous_stage, previous_digest = stage, digest
        if status == "failed":
            break
    if set(normalized) != actual:
        raise CapacityEvidenceError(
            "attempts exist after a terminal failure or before a missing predecessor"
        )
    return normalized


def create_capacity_evidence(
    *,
    identity: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    attempt_metadata: Mapping[str, Any],
    selected_configs: Mapping[str, str],
) -> dict[str, Any]:
    normalized_identity = validate_identity(identity)
    profile = normalized_identity["selected_profile"]
    configs = _validate_selected_configs(selected_configs, profile)
    config_set_sha = canonical_sha256(configs)
    if config_set_sha != normalized_identity["selected_config_set_sha256"]:
        raise CapacityEvidenceError("selected config set hash drifted from identity")
    telemetry_mapping = _require_mapping(telemetry, "telemetry")
    _same_identity(telemetry_mapping.get("identity"), normalized_identity, "telemetry")
    nvml_total = telemetry_mapping.get("nvml_total_bytes")
    expected_vram = normalized_identity["gpu"]["total_vram_bytes"]
    if (
        isinstance(nvml_total, bool)
        or not isinstance(nvml_total, int)
        or nvml_total != expected_vram
    ):
        raise CapacityEvidenceError("telemetry NVML total VRAM drifted from GPU identity")
    for index, step in enumerate(telemetry_mapping.get("steps", ())):
        if isinstance(step, Mapping) and "nvml_total_bytes" in step:
            if step["nvml_total_bytes"] != expected_vram:
                raise CapacityEvidenceError(
                    f"telemetry.steps[{index}] total VRAM drifted from GPU identity"
                )
    required_host_ram = (80 if profile == "R0" else 128) * GIB
    observed_host_ram = telemetry_mapping.get("host_total_memory_bytes")
    if (
        isinstance(observed_host_ram, bool)
        or not isinstance(observed_host_ram, int)
        or observed_host_ram < required_host_ram
    ):
        raise CapacityEvidenceError(
            f"{profile} telemetry does not satisfy its host RAM profile"
        )
    attempts = validate_attempt_metadata(
        attempt_metadata,
        identity=normalized_identity,
        selected_configs=configs,
        require_complete_success=False,
    )
    classification = classify_telemetry(telemetry_mapping)
    failed_attempts = [
        attempt for attempt in attempts.values() if attempt["status"] == "failed"
    ]
    terminal_reason = telemetry_mapping.get("terminal_reason")
    if terminal_reason in CAPACITY_TERMINAL_REASONS and (
        len(failed_attempts) != 1 or failed_attempts[0]["failure_kind"] != "capacity"
    ):
        raise CapacityEvidenceError(
            "capacity terminal telemetry does not match a capacity-failed attempt"
        )
    if any(attempt["failure_kind"] != "capacity" for attempt in failed_attempts):
        classification["r1_eligible"] = False
    if failed_attempts and terminal_reason not in CAPACITY_TERMINAL_REASONS:
        classification["r1_eligible"] = False
    if profile != "R0":
        classification["r1_eligible"] = False
        classification["eligibility_reason"] = None
    payload = {
        "attempt_metadata_sha256": canonical_sha256(attempt_metadata),
        "classification": classification,
        "identity": normalized_identity,
        "schema_version": SCHEMA_VERSION,
        "telemetry_sha256": canonical_sha256(telemetry_mapping),
    }
    return _seal(payload, "capacity_evidence_sha256")


def verify_capacity_evidence(value: Any) -> dict[str, Any]:
    evidence = _require_mapping(value, "capacity evidence")
    _exact_keys(
        evidence,
        {
            "attempt_metadata_sha256",
            "capacity_evidence_sha256",
            "classification",
            "identity",
            "schema_version",
            "telemetry_sha256",
        },
        "capacity evidence",
    )
    if evidence["schema_version"] != SCHEMA_VERSION:
        raise CapacityEvidenceError("capacity evidence schema version mismatch")
    _verify_self_hash(evidence, "capacity_evidence_sha256")
    _require_sha256(evidence["attempt_metadata_sha256"], "attempt_metadata_sha256")
    _require_sha256(evidence["telemetry_sha256"], "telemetry_sha256")
    validate_identity(evidence["identity"])
    classification = _require_mapping(evidence["classification"], "classification")
    if classification.get("overall") not in _COLORS or not isinstance(
        classification.get("metrics"), Mapping
    ):
        raise CapacityEvidenceError("capacity classification is invalid")
    return json.loads(json.dumps(evidence, allow_nan=False))


def create_r1_approval_marker(
    *,
    r1_identity: Mapping[str, Any],
    r0_capacity_evidence: Mapping[str, Any],
    r0_terminal_sha256: str,
    budget_projection_sha256: str,
    approval_nonce: str,
) -> dict[str, Any]:
    identity = validate_identity(r1_identity)
    if identity["selected_profile"] != "R1":
        raise CapacityEvidenceError("R1 approval requires an R1 target identity")
    evidence = verify_capacity_evidence(r0_capacity_evidence)
    r0_identity = validate_identity(evidence["identity"])
    if r0_identity["selected_profile"] != "R0":
        raise CapacityEvidenceError("approval evidence must come from R0")
    for key in ("profile_id", "commit", "cpu_handoff_sha256", "active_config_tree_sha256", "gpu"):
        if r0_identity[key] != identity[key]:
            raise CapacityEvidenceError(f"R0/R1 identity drifted in {key}")
    classification = evidence["classification"]
    reason = classification.get("eligibility_reason")
    if classification.get("r1_eligible") is not True or reason not in CAPACITY_TERMINAL_REASONS:
        raise CapacityEvidenceError("R0 evidence is not eligible for R1 approval")
    if not isinstance(approval_nonce, str) or _SAFE_ID.fullmatch(approval_nonce) is None:
        raise CapacityEvidenceError("approval_nonce must be a unique safe identifier")
    payload = {
        "active_config_tree_sha256": identity["active_config_tree_sha256"],
        "approval_nonce": approval_nonce,
        "budget_projection_sha256": _require_sha256(
            budget_projection_sha256, "budget_projection_sha256"
        ),
        "capacity_reason": reason,
        "commit": identity["commit"],
        "cpu_handoff_sha256": identity["cpu_handoff_sha256"],
        "profile_id": identity["profile_id"],
        "r0_capacity_evidence_sha256": evidence["capacity_evidence_sha256"],
        "r0_terminal_sha256": _require_sha256(r0_terminal_sha256, "r0_terminal_sha256"),
        "requested_profile": "R1",
        "schema_version": SCHEMA_VERSION,
        "status": "approved-once",
    }
    return _seal(payload, "approval_marker_sha256")


def verify_r1_approval_marker(
    value: Any,
    *,
    r1_identity: Mapping[str, Any],
    r0_capacity_evidence: Mapping[str, Any],
    r0_terminal_sha256: str,
    budget_projection_sha256: str,
    used_marker_sha256s: Collection[str] = (),
) -> dict[str, Any]:
    marker = _require_mapping(value, "R1 approval marker")
    _exact_keys(marker, _APPROVAL_MARKER_KEYS, "R1 approval marker")
    digest = _verify_self_hash(marker, "approval_marker_sha256")
    if digest in used_marker_sha256s:
        raise CapacityEvidenceError("R1 approval marker was already consumed")
    expected = create_r1_approval_marker(
        r1_identity=r1_identity,
        r0_capacity_evidence=r0_capacity_evidence,
        r0_terminal_sha256=r0_terminal_sha256,
        budget_projection_sha256=budget_projection_sha256,
        approval_nonce=marker.get("approval_nonce"),
    )
    if dict(marker) != expected:
        raise CapacityEvidenceError("R1 approval marker identity or authority drifted")
    return json.loads(json.dumps(marker, allow_nan=False))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_existing_path(value: Any, label: str, *, directory: bool) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise CapacityEvidenceError(f"{label} must be an absolute canonical path")
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise CapacityEvidenceError(f"{label} must be absolute")
    lexical = Path(os.path.abspath(raw))
    for component in [*reversed(lexical.parents), lexical]:
        if component.is_symlink():
            raise CapacityEvidenceError(f"{label} contains a symlink: {component}")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise CapacityEvidenceError(f"cannot resolve {label}: {exc}") from exc
    if str(raw) != str(resolved) or lexical != resolved:
        raise CapacityEvidenceError(f"{label} must be canonical")
    if directory and not resolved.is_dir():
        raise CapacityEvidenceError(f"{label} must be a directory")
    if not directory and not resolved.is_file():
        raise CapacityEvidenceError(f"{label} must be a regular file")
    return resolved


def _validate_consumption_marker(
    value: Any, *, r1_identity: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    marker = _require_mapping(value, "R1 approval marker")
    _exact_keys(marker, _APPROVAL_MARKER_KEYS, "R1 approval marker")
    marker_sha = _verify_self_hash(marker, "approval_marker_sha256")
    identity = validate_identity(r1_identity)
    if identity["selected_profile"] != "R1":
        raise CapacityEvidenceError("approval consumption requires an R1 identity")
    expected = {
        "active_config_tree_sha256": identity["active_config_tree_sha256"],
        "commit": identity["commit"],
        "cpu_handoff_sha256": identity["cpu_handoff_sha256"],
        "profile_id": identity["profile_id"],
        "requested_profile": "R1",
        "schema_version": SCHEMA_VERSION,
        "status": "approved-once",
    }
    for field, expected_value in expected.items():
        if marker[field] != expected_value:
            raise CapacityEvidenceError(f"R1 approval marker drifted in {field}")
    for field in (
        "budget_projection_sha256",
        "r0_capacity_evidence_sha256",
        "r0_terminal_sha256",
    ):
        _require_sha256(marker[field], f"R1 approval marker.{field}")
    if marker["capacity_reason"] not in CAPACITY_TERMINAL_REASONS:
        raise CapacityEvidenceError("R1 approval marker capacity reason is invalid")
    nonce = marker["approval_nonce"]
    if not isinstance(nonce, str) or _SAFE_ID.fullmatch(nonce) is None:
        raise CapacityEvidenceError("R1 approval marker nonce is unsafe")
    return json.loads(json.dumps(marker, allow_nan=False)), marker_sha


def _validate_pipeline_identity(
    pipeline_dir: Path, *, r1_identity: Mapping[str, Any]
) -> tuple[Path, str]:
    identity = validate_identity(r1_identity)
    identity_path = _safe_existing_path(
        pipeline_dir / "identity", "approval pipeline identity", directory=False
    )
    try:
        pipeline_text = identity_path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise CapacityEvidenceError(f"cannot read pipeline identity: {exc}") from exc
    lines = pipeline_text.splitlines(keepends=True)
    expected_names = [
        "git_commit",
        "experiment_profile_id",
        "config_tree_sha256",
        "environment_lock_file_sha256",
        "asset_manifest_file_sha256",
    ]
    if (
        len(lines) != len(expected_names)
        or any(not line.endswith("\n") or "=" not in line for line in lines)
        or [line.split("=", 1)[0] for line in lines] != expected_names
    ):
        raise CapacityEvidenceError("approval pipeline identity schema drifted")
    values = {
        name: line[:-1].split("=", 1)[1]
        for name, line in zip(expected_names, lines, strict=True)
    }
    if (
        values["git_commit"] != identity["commit"]
        or values["experiment_profile_id"] != PROFILE_ID
        or values["config_tree_sha256"] != identity["active_config_tree_sha256"]
    ):
        raise CapacityEvidenceError("approval pipeline identity drifted from R1 target")
    for field in (
        "config_tree_sha256",
        "environment_lock_file_sha256",
        "asset_manifest_file_sha256",
    ):
        _require_sha256(values[field], f"pipeline identity {field}")
    identity_sha = _file_sha256(identity_path)
    if pipeline_dir.name != f"{identity['commit']}-{identity_sha}":
        raise CapacityEvidenceError("approval pipeline directory identity drifted")
    return identity_path, identity_sha


def _validate_launcher_request(
    launcher_dir: Path,
    *,
    marker: Mapping[str, Any],
    marker_path: Path,
    r1_identity: Mapping[str, Any],
) -> tuple[Path, str, str]:
    identity = validate_identity(r1_identity)
    request_path = _safe_existing_path(
        launcher_dir / "request.json", "approval launcher request", directory=False
    )
    try:
        request = json.loads(request_path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CapacityEvidenceError(f"cannot read launcher request: {exc}") from exc
    request = _require_mapping(request, "launcher request")
    _exact_keys(request, _LAUNCHER_REQUEST_KEYS, "launcher request")
    request_marker_file_sha = _require_sha256(
        request["r1_approval_file_sha256"],
        "launcher request r1_approval_file_sha256",
    )
    if (
        request["schema_version"] != SCHEMA_VERSION
        or request["phase"] != "gpu-capacity"
        or request["experiment_profile_id"] != PROFILE_ID
        or request["expected_commit"] != identity["commit"]
        or request["offload_profile"] != "r1"
        or request["r1_approval"] != str(marker_path)
        or not isinstance(request["budget_projection"], str)
        or not request["budget_projection"]
        or request["keep_running"] not in {"yes", "no"}
        or request["retry_failed_stage"] not in {"yes", "no"}
        or request["dry_run"] != "no"
        or not isinstance(request["requested_at"], str)
        or _UTC_TIMESTAMP.fullmatch(request["requested_at"]) is None
    ):
        raise CapacityEvidenceError("launcher request does not authorize this R1 consumption")
    if _file_sha256(marker_path) != request_marker_file_sha:
        raise CapacityEvidenceError("R1 approval marker file hash changed after launch admission")
    projection_path = _safe_existing_path(
        request["budget_projection"], "approval budget projection", directory=False
    )
    request_file_sha = _require_sha256(
        request["budget_projection_file_sha256"],
        "launcher request budget_projection_file_sha256",
    )
    if _file_sha256(projection_path) != request_file_sha:
        raise CapacityEvidenceError("approval budget projection file hash changed")
    try:
        projection_raw = projection_path.read_bytes()
        projection = json.loads(projection_raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CapacityEvidenceError(f"cannot read approval budget projection: {exc}") from exc
    projection = _require_mapping(projection, "approval budget projection")
    try:
        canonical_projection = (
            json.dumps(
                projection,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CapacityEvidenceError(
            f"approval budget projection is not canonical JSON: {exc}"
        ) from exc
    if projection_raw != canonical_projection:
        raise CapacityEvidenceError("approval budget projection is not canonical JSON")
    try:
        from scripts.cloud.cost_gate import verify_projection

        verified_projection = verify_projection(projection)
    except Exception as exc:
        raise CapacityEvidenceError(f"approval budget projection is invalid: {exc}") from exc
    if (
        verified_projection["projection_sha256"]
        != marker["budget_projection_sha256"]
    ):
        raise CapacityEvidenceError(
            "approval budget projection identity differs from the marker"
        )
    return request_path, _file_sha256(request_path), request_marker_file_sha


def _validate_consumption_context(
    *,
    approval_marker: Mapping[str, Any],
    approval_marker_path: str | os.PathLike[str],
    r1_identity: Mapping[str, Any],
    launcher_dir: str | os.PathLike[str],
    pipeline_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    identity = validate_identity(r1_identity)
    marker, marker_sha = _validate_consumption_marker(
        approval_marker, r1_identity=identity
    )
    marker_path = _safe_existing_path(
        approval_marker_path, "R1 approval marker path", directory=False
    )
    if dict(_load_json(marker_path, "R1 approval marker file")) != marker:
        raise CapacityEvidenceError("approval marker file differs from verified marker")
    launcher = _safe_existing_path(
        launcher_dir, "approval launcher directory", directory=True
    )
    pipeline = _safe_existing_path(
        pipeline_dir, "approval pipeline directory", directory=True
    )
    _, pipeline_identity_sha = _validate_pipeline_identity(
        pipeline, r1_identity=identity
    )
    _, launcher_request_sha, marker_file_sha = _validate_launcher_request(
        launcher,
        marker=marker,
        marker_path=marker_path,
        r1_identity=identity,
    )
    return {
        "identity": identity,
        "launcher_dir": launcher,
        "launcher_request_sha256": launcher_request_sha,
        "marker": marker,
        "marker_path": marker_path,
        "marker_file_sha256": marker_file_sha,
        "marker_sha256": marker_sha,
        "pipeline_dir": pipeline,
        "pipeline_identity_sha256": pipeline_identity_sha,
    }


def _expected_consumption_path(pipeline_dir: Path, marker_sha256: str) -> Path:
    return (
        pipeline_dir
        / "capacity"
        / "r1"
        / "approval-consumptions"
        / marker_sha256
        / "consumption.json"
    )


def _validate_new_consumption_path(
    value: str | os.PathLike[str], *, pipeline_dir: Path, marker_sha256: str
) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise CapacityEvidenceError("approval consumption output must be absolute")
    lexical = Path(os.path.abspath(raw))
    if str(raw) != str(lexical):
        raise CapacityEvidenceError("approval consumption output must be canonical")
    expected = _expected_consumption_path(pipeline_dir, marker_sha256)
    if lexical != expected:
        raise CapacityEvidenceError("approval consumption output escaped its marker claim")
    claim_dir = _safe_existing_path(
        expected.parent, "approval consumption claim directory", directory=True
    )
    try:
        entries = list(claim_dir.iterdir())
    except OSError as exc:
        raise CapacityEvidenceError(f"cannot inspect approval consumption claim: {exc}") from exc
    if entries:
        raise CapacityEvidenceError("R1 approval claim is already consumed or nonempty")
    return claim_dir / "consumption.json"


def create_r1_approval_consumption(
    *,
    approval_marker: Mapping[str, Any],
    approval_marker_path: str | os.PathLike[str],
    r1_identity: Mapping[str, Any],
    launcher_dir: str | os.PathLike[str],
    pipeline_dir: str | os.PathLike[str],
    consumption_path: str | os.PathLike[str],
    consumed_at: str | None = None,
) -> dict[str, Any]:
    context = _validate_consumption_context(
        approval_marker=approval_marker,
        approval_marker_path=approval_marker_path,
        r1_identity=r1_identity,
        launcher_dir=launcher_dir,
        pipeline_dir=pipeline_dir,
    )
    _validate_new_consumption_path(
        consumption_path,
        pipeline_dir=context["pipeline_dir"],
        marker_sha256=context["marker_sha256"],
    )
    timestamp = consumed_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not isinstance(timestamp, str) or _UTC_TIMESTAMP.fullmatch(timestamp) is None:
        raise CapacityEvidenceError("approval consumption timestamp is malformed")
    payload = {
        "approval_marker_path": str(context["marker_path"]),
        "approval_marker_file_sha256": context["marker_file_sha256"],
        "approval_marker_sha256": context["marker_sha256"],
        "consumed_at": timestamp,
        "experiment_profile_id": PROFILE_ID,
        "git_commit": context["identity"]["commit"],
        "kind": _CONSUMPTION_KIND,
        "launcher_dir": str(context["launcher_dir"]),
        "launcher_request_sha256": context["launcher_request_sha256"],
        "offload_profile": "r1",
        "phase": "gpu-capacity",
        "pipeline_dir": str(context["pipeline_dir"]),
        "pipeline_identity_sha256": context["pipeline_identity_sha256"],
        "schema_version": SCHEMA_VERSION,
        "status": "consumed",
    }
    return _seal(payload, "consumption_sha256")


def verify_r1_approval_consumption(
    value: Any,
    *,
    consumption_path: str | os.PathLike[str],
    approval_marker: Mapping[str, Any],
    approval_marker_path: str | os.PathLike[str],
    r1_identity: Mapping[str, Any],
) -> dict[str, Any]:
    consumption = _require_mapping(value, "R1 approval consumption")
    _exact_keys(consumption, _CONSUMPTION_KEYS, "R1 approval consumption")
    digest = _verify_self_hash(consumption, "consumption_sha256")
    context = _validate_consumption_context(
        approval_marker=approval_marker,
        approval_marker_path=approval_marker_path,
        r1_identity=r1_identity,
        launcher_dir=consumption["launcher_dir"],
        pipeline_dir=consumption["pipeline_dir"],
    )
    claim_path = _safe_existing_path(
        consumption_path, "approval consumption record", directory=False
    )
    expected_claim = _expected_consumption_path(
        context["pipeline_dir"], context["marker_sha256"]
    )
    if claim_path != expected_claim:
        raise CapacityEvidenceError("approval consumption record escaped its marker claim")
    if dict(_load_json(claim_path, "R1 approval consumption file")) != dict(consumption):
        raise CapacityEvidenceError("approval consumption file differs from verified record")
    expected_scalars = {
        "schema_version": SCHEMA_VERSION,
        "kind": _CONSUMPTION_KIND,
        "status": "consumed",
        "approval_marker_sha256": context["marker_sha256"],
        "approval_marker_file_sha256": context["marker_file_sha256"],
        "approval_marker_path": str(context["marker_path"]),
        "launcher_dir": str(context["launcher_dir"]),
        "launcher_request_sha256": context["launcher_request_sha256"],
        "pipeline_identity_sha256": context["pipeline_identity_sha256"],
        "pipeline_dir": str(context["pipeline_dir"]),
        "experiment_profile_id": PROFILE_ID,
        "git_commit": context["identity"]["commit"],
        "phase": "gpu-capacity",
        "offload_profile": "r1",
    }
    for field, expected in expected_scalars.items():
        if consumption[field] != expected:
            raise CapacityEvidenceError(f"approval consumption drifted in {field}")
    if not isinstance(consumption["consumed_at"], str) or _UTC_TIMESTAMP.fullmatch(
        consumption["consumed_at"]
    ) is None:
        raise CapacityEvidenceError("approval consumption timestamp is malformed")
    return {**json.loads(json.dumps(consumption, allow_nan=False)), "consumption_sha256": digest}


def create_capacity_profile(
    *,
    identity: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    attempt_metadata: Mapping[str, Any],
    selected_configs: Mapping[str, str],
    approval_marker: Mapping[str, Any] | None = None,
    approval_marker_path: str | os.PathLike[str] | None = None,
    approval_consumption: Mapping[str, Any] | None = None,
    approval_consumption_path: str | os.PathLike[str] | None = None,
    r0_capacity_evidence: Mapping[str, Any] | None = None,
    r0_terminal_sha256: str | None = None,
    budget_projection_sha256: str | None = None,
    used_approval_marker_sha256s: Collection[str] = (),
) -> dict[str, Any]:
    normalized_identity = validate_identity(identity)
    profile = normalized_identity["selected_profile"]
    configs = _validate_selected_configs(selected_configs, profile)
    if canonical_sha256(configs) != normalized_identity["selected_config_set_sha256"]:
        raise CapacityEvidenceError("selected config set hash drifted from identity")
    evidence = create_capacity_evidence(
        identity=normalized_identity,
        telemetry=telemetry,
        attempt_metadata=attempt_metadata,
        selected_configs=configs,
    )
    if evidence["classification"]["overall"] != "green":
        raise CapacityEvidenceError(f"{profile} capacity evidence is not all green")
    attempts = validate_attempt_metadata(
        attempt_metadata,
        identity=normalized_identity,
        selected_configs=configs,
        require_complete_success=True,
    )
    approval_sha: str | None = None
    consumption_sha: str | None = None
    if profile == "R0":
        if any(
            value is not None
            for value in (
                approval_marker,
                approval_marker_path,
                approval_consumption,
                approval_consumption_path,
                r0_capacity_evidence,
                r0_terminal_sha256,
                budget_projection_sha256,
            )
        ):
            raise CapacityEvidenceError("R0 must not carry an R1 approval marker")
    else:
        if (
            approval_marker is None
            or approval_marker_path is None
            or approval_consumption is None
            or approval_consumption_path is None
            or r0_capacity_evidence is None
            or r0_terminal_sha256 is None
            or budget_projection_sha256 is None
        ):
            raise CapacityEvidenceError(
                "R1 is blocked without its explicit one-time approval evidence"
            )
        marker = verify_r1_approval_marker(
            approval_marker,
            r1_identity=normalized_identity,
            r0_capacity_evidence=r0_capacity_evidence,
            r0_terminal_sha256=r0_terminal_sha256,
            budget_projection_sha256=budget_projection_sha256,
            used_marker_sha256s=used_approval_marker_sha256s,
        )
        approval_sha = marker["approval_marker_sha256"]
        consumption = verify_r1_approval_consumption(
            approval_consumption,
            consumption_path=approval_consumption_path,
            approval_marker=marker,
            approval_marker_path=approval_marker_path,
            r1_identity=normalized_identity,
        )
        consumption_sha = consumption["consumption_sha256"]
    payload = {
        "active_config_tree_sha256": normalized_identity["active_config_tree_sha256"],
        "approval_marker_sha256": approval_sha,
        "approval_consumption_sha256": consumption_sha,
        "capacity_evidence": evidence,
        "capacity_evidence_sha256": evidence["capacity_evidence_sha256"],
        "commit": normalized_identity["commit"],
        "cpu_handoff_sha256": normalized_identity["cpu_handoff_sha256"],
        "g2_attempts": attempts,
        "g2_attempts_sha256": canonical_sha256(attempts),
        "gpu": normalized_identity["gpu"],
        "profile_id": normalized_identity["profile_id"],
        "schema_version": SCHEMA_VERSION,
        "selected_config_set_sha256": normalized_identity["selected_config_set_sha256"],
        "selected_configs": configs,
        "selected_profile": profile,
        "telemetry_sha256": evidence["telemetry_sha256"],
    }
    return _seal(payload, "self_sha256")


def verify_capacity_profile(
    value: Any,
    *,
    expected_identity: Mapping[str, Any] | None = None,
    telemetry: Mapping[str, Any] | None = None,
    attempt_metadata: Mapping[str, Any] | None = None,
    approval_marker: Mapping[str, Any] | None = None,
    approval_marker_path: str | os.PathLike[str] | None = None,
    approval_consumption: Mapping[str, Any] | None = None,
    approval_consumption_path: str | os.PathLike[str] | None = None,
    r0_capacity_evidence: Mapping[str, Any] | None = None,
    r0_terminal_sha256: str | None = None,
    budget_projection_sha256: str | None = None,
    used_approval_marker_sha256s: Collection[str] = (),
) -> dict[str, Any]:
    profile = _require_mapping(value, "capacity profile")
    expected_keys = {
        "active_config_tree_sha256",
        "approval_marker_sha256",
        "approval_consumption_sha256",
        "capacity_evidence",
        "capacity_evidence_sha256",
        "commit",
        "cpu_handoff_sha256",
        "g2_attempts",
        "g2_attempts_sha256",
        "gpu",
        "profile_id",
        "schema_version",
        "selected_config_set_sha256",
        "selected_configs",
        "selected_profile",
        "self_sha256",
        "telemetry_sha256",
    }
    _exact_keys(profile, expected_keys, "capacity profile")
    if profile["schema_version"] != SCHEMA_VERSION or profile["profile_id"] != PROFILE_ID:
        raise CapacityEvidenceError("capacity profile schema/profile ID mismatch")
    _verify_self_hash(profile, "self_sha256")
    selected_profile = profile["selected_profile"]
    configs = _validate_selected_configs(profile["selected_configs"], selected_profile)
    if canonical_sha256(configs) != profile["selected_config_set_sha256"]:
        raise CapacityEvidenceError("capacity profile selected config set hash is invalid")
    identity = validate_identity(
        {
            "active_config_tree_sha256": profile["active_config_tree_sha256"],
            "commit": profile["commit"],
            "cpu_handoff_sha256": profile["cpu_handoff_sha256"],
            "gpu": profile["gpu"],
            "profile_id": profile["profile_id"],
            "selected_config_set_sha256": profile["selected_config_set_sha256"],
            "selected_profile": selected_profile,
        }
    )
    if expected_identity is not None and validate_identity(expected_identity) != identity:
        raise CapacityEvidenceError("capacity profile identity drifted from expected identity")
    evidence = verify_capacity_evidence(profile["capacity_evidence"])
    if evidence["identity"] != identity:
        raise CapacityEvidenceError("capacity evidence identity drifted from capacity profile")
    if evidence["capacity_evidence_sha256"] != profile["capacity_evidence_sha256"]:
        raise CapacityEvidenceError("capacity evidence hash drifted")
    if evidence["telemetry_sha256"] != profile["telemetry_sha256"]:
        raise CapacityEvidenceError("telemetry hash drifted")
    if evidence["classification"]["overall"] != "green":
        raise CapacityEvidenceError("sealed capacity profile is not all green")
    attempts = validate_attempt_metadata(
        {"attempts": profile["g2_attempts"], "identity": identity},
        identity=identity,
        selected_configs=configs,
        require_complete_success=True,
    )
    if canonical_sha256(attempts) != profile["g2_attempts_sha256"]:
        raise CapacityEvidenceError("G2 attempt hash drifted")
    if telemetry is not None:
        if canonical_sha256(telemetry) != profile["telemetry_sha256"]:
            raise CapacityEvidenceError("external telemetry differs from capacity profile")
        telemetry_mapping = _require_mapping(telemetry, "telemetry")
        _same_identity(telemetry_mapping.get("identity"), identity, "external telemetry")
        if classify_telemetry(telemetry_mapping) != evidence["classification"]:
            raise CapacityEvidenceError("capacity classification differs from external telemetry")
    if attempt_metadata is not None and canonical_sha256(attempt_metadata) != evidence[
        "attempt_metadata_sha256"
    ]:
        raise CapacityEvidenceError("external attempt metadata differs from capacity profile")
    approval_sha = profile["approval_marker_sha256"]
    consumption_sha = profile["approval_consumption_sha256"]
    if selected_profile == "R0":
        if approval_sha is not None or consumption_sha is not None:
            raise CapacityEvidenceError("R0 capacity profile carries R1 approval authority")
    else:
        _require_sha256(approval_sha, "approval_marker_sha256")
        _require_sha256(consumption_sha, "approval_consumption_sha256")
        if (
            approval_marker is None
            or approval_marker_path is None
            or approval_consumption is None
            or approval_consumption_path is None
            or r0_capacity_evidence is None
            or r0_terminal_sha256 is None
            or budget_projection_sha256 is None
        ):
            raise CapacityEvidenceError("R1 verification requires the complete approval evidence")
        marker = verify_r1_approval_marker(
            approval_marker,
            r1_identity=identity,
            r0_capacity_evidence=r0_capacity_evidence,
            r0_terminal_sha256=r0_terminal_sha256,
            budget_projection_sha256=budget_projection_sha256,
            used_marker_sha256s=used_approval_marker_sha256s,
        )
        if marker.get("approval_marker_sha256") != approval_sha:
            raise CapacityEvidenceError("R1 approval marker hash drifted")
        consumption = verify_r1_approval_consumption(
            approval_consumption,
            consumption_path=approval_consumption_path,
            approval_marker=marker,
            approval_marker_path=approval_marker_path,
            r1_identity=identity,
        )
        if consumption["consumption_sha256"] != consumption_sha:
            raise CapacityEvidenceError("R1 approval consumption hash drifted")
    return json.loads(json.dumps(profile, allow_nan=False))


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    if path.is_symlink():
        raise CapacityEvidenceError(f"{label} must not be a symlink")
    try:
        value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CapacityEvidenceError(f"cannot read {label}: {exc}") from exc
    return _require_mapping(value, label)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a one-shot authority marker without replacing any prior bytes."""

    destination = path.expanduser()
    if not destination.is_absolute():
        raise CapacityEvidenceError("approval marker output must be absolute")
    lexical = Path(os.path.abspath(destination))
    if str(destination) != str(lexical):
        raise CapacityEvidenceError("approval marker output must be canonical")
    for component in reversed(lexical.parent.parents):
        if component.is_symlink():
            raise CapacityEvidenceError("approval marker output path contains a symlink")
    lexical.parent.mkdir(parents=True, exist_ok=True)
    for component in [*reversed(lexical.parent.parents), lexical.parent, lexical]:
        if component.is_symlink():
            raise CapacityEvidenceError("approval marker output path contains a symlink")
    parent = lexical.parent.resolve(strict=True)
    if lexical.parent != parent:
        raise CapacityEvidenceError("approval marker output path contains a symlink")
    destination = parent / lexical.name
    if destination.exists() or destination.is_symlink():
        raise CapacityEvidenceError("refusing to replace an existing R1 approval marker")
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-", dir=str(parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # A same-directory hard link is an atomic create-if-absent operation.
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise CapacityEvidenceError(
            "refusing to replace an existing R1 approval marker"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_consumption_json_new(
    path: Path,
    payload: Mapping[str, Any],
    *,
    pipeline_dir: Path,
    marker_sha256: str,
) -> None:
    destination = _validate_new_consumption_path(
        path, pipeline_dir=pipeline_dir, marker_sha256=marker_sha256
    )
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".tmp-", dir=str(destination.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise CapacityEvidenceError("R1 approval claim is already consumed") from exc
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evidence = subparsers.add_parser("evidence", help="write self-hashed G2 capacity evidence")
    generate = subparsers.add_parser("generate", help="write an all-green capacity-profile.json")
    approve = subparsers.add_parser(
        "approve-r1", help="write a one-shot R1 capacity approval marker"
    )
    consume = subparsers.add_parser(
        "consume-r1", help="bind a claimed R1 approval to one launcher and pipeline"
    )
    verify = subparsers.add_parser("verify", help="verify a capacity-profile.json")
    for command in (evidence, generate):
        command.add_argument("--identity", type=Path, required=True)
        command.add_argument("--telemetry", type=Path, required=True)
        command.add_argument("--attempt-metadata", type=Path, required=True)
        command.add_argument("--selected-configs", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    generate.add_argument("--approval-marker", type=Path)
    generate.add_argument("--approval-consumption", type=Path)
    generate.add_argument("--r0-capacity-evidence", type=Path)
    generate.add_argument("--r0-terminal-sha256")
    generate.add_argument("--budget-projection-sha256")
    approve.add_argument("--r1-identity", type=Path, required=True)
    approve.add_argument("--r0-capacity-evidence", type=Path, required=True)
    approve.add_argument("--r0-terminal-sha256", required=True)
    approve.add_argument("--budget-projection-sha256", required=True)
    approve.add_argument("--approval-nonce", required=True)
    approve.add_argument("--output", type=Path, required=True)
    consume.add_argument("--approval-marker", type=Path, required=True)
    consume.add_argument("--r1-identity", type=Path, required=True)
    consume.add_argument("--launcher-dir", type=Path, required=True)
    consume.add_argument("--pipeline-dir", type=Path, required=True)
    consume.add_argument("--output", type=Path, required=True)
    verify.add_argument("--capacity-profile", type=Path, required=True)
    verify.add_argument("--approval-marker", type=Path)
    verify.add_argument("--approval-consumption", type=Path)
    verify.add_argument("--r0-capacity-evidence", type=Path)
    verify.add_argument("--r0-terminal-sha256")
    verify.add_argument("--budget-projection-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "approve-r1":
            identity = _load_json(args.r1_identity, "R1 identity")
            r0_evidence = _load_json(args.r0_capacity_evidence, "R0 capacity evidence")
            result = create_r1_approval_marker(
                r1_identity=identity,
                r0_capacity_evidence=r0_evidence,
                r0_terminal_sha256=args.r0_terminal_sha256,
                budget_projection_sha256=args.budget_projection_sha256,
                approval_nonce=args.approval_nonce,
            )
            verify_r1_approval_marker(
                result,
                r1_identity=identity,
                r0_capacity_evidence=r0_evidence,
                r0_terminal_sha256=args.r0_terminal_sha256,
                budget_projection_sha256=args.budget_projection_sha256,
            )
            _atomic_json_new(args.output, result)
            print(
                json.dumps(
                    {
                        "approval_marker_sha256": result["approval_marker_sha256"],
                        "status": "approved-once",
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "consume-r1":
            identity = _load_json(args.r1_identity, "R1 identity")
            marker = _load_json(args.approval_marker, "R1 approval marker")
            result = create_r1_approval_consumption(
                approval_marker=marker,
                approval_marker_path=args.approval_marker,
                r1_identity=identity,
                launcher_dir=args.launcher_dir,
                pipeline_dir=args.pipeline_dir,
                consumption_path=args.output,
            )
            context = _validate_consumption_context(
                approval_marker=marker,
                approval_marker_path=args.approval_marker,
                r1_identity=identity,
                launcher_dir=args.launcher_dir,
                pipeline_dir=args.pipeline_dir,
            )
            _atomic_consumption_json_new(
                args.output,
                result,
                pipeline_dir=context["pipeline_dir"],
                marker_sha256=context["marker_sha256"],
            )
            verify_r1_approval_consumption(
                _load_json(args.output, "R1 approval consumption"),
                consumption_path=args.output,
                approval_marker=marker,
                approval_marker_path=args.approval_marker,
                r1_identity=identity,
            )
            print(
                json.dumps(
                    {
                        "consumption_sha256": result["consumption_sha256"],
                        "status": "consumed",
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "verify":
            profile = verify_capacity_profile(
                _load_json(args.capacity_profile, "capacity profile"),
                approval_marker=(
                    _load_json(args.approval_marker, "R1 approval marker")
                    if args.approval_marker
                    else None
                ),
                approval_marker_path=args.approval_marker,
                approval_consumption=(
                    _load_json(args.approval_consumption, "R1 approval consumption")
                    if args.approval_consumption
                    else None
                ),
                approval_consumption_path=args.approval_consumption,
                r0_capacity_evidence=(
                    _load_json(args.r0_capacity_evidence, "R0 capacity evidence")
                    if args.r0_capacity_evidence
                    else None
                ),
                r0_terminal_sha256=args.r0_terminal_sha256,
                budget_projection_sha256=args.budget_projection_sha256,
            )
            print(json.dumps({"self_sha256": profile["self_sha256"], "status": "verified"}))
            return 0
        identity = _load_json(args.identity, "identity")
        telemetry = _load_json(args.telemetry, "telemetry")
        attempts = _load_json(args.attempt_metadata, "attempt metadata")
        configs = _load_json(args.selected_configs, "selected configs")
        if args.command == "evidence":
            result = create_capacity_evidence(
                identity=identity,
                telemetry=telemetry,
                attempt_metadata=attempts,
                selected_configs=configs,
            )
        else:
            result = create_capacity_profile(
                identity=identity,
                telemetry=telemetry,
                attempt_metadata=attempts,
                selected_configs=configs,
                approval_marker=(
                    _load_json(args.approval_marker, "R1 approval marker")
                    if args.approval_marker
                    else None
                ),
                approval_marker_path=args.approval_marker,
                approval_consumption=(
                    _load_json(args.approval_consumption, "R1 approval consumption")
                    if args.approval_consumption
                    else None
                ),
                approval_consumption_path=args.approval_consumption,
                r0_capacity_evidence=(
                    _load_json(args.r0_capacity_evidence, "R0 capacity evidence")
                    if args.r0_capacity_evidence
                    else None
                ),
                r0_terminal_sha256=args.r0_terminal_sha256,
                budget_projection_sha256=args.budget_projection_sha256,
            )
        _atomic_json(args.output, result)
    except Exception as exc:
        print(
            json.dumps({"error": f"{type(exc).__name__}: {exc}", "status": "blocked"}),
            file=sys.stderr,
        )
        return 2
    digest = result.get("self_sha256") or result.get("capacity_evidence_sha256")
    print(json.dumps({"sha256": digest, "status": "ready"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
