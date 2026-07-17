"""Verify G2 attempts and publish capacity inputs without sealing a profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from scripts.cloud import capacity_evidence


SCHEMA_VERSION = 1
PROFILE_ID = capacity_evidence.PROFILE_ID
ARTIFACT_KIND = "rememr1-g2-artifact-evidence-v1"
CAPACITY_STOP_KIND = "rememr1-capacity-stop-v1"
CAPACITY_STOP_EXIT_CODE = 43
GIB = 1024**3
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_RTX_5090 = re.compile(r"(?:^|\s)GEFORCE\s+RTX\s+5090$", re.IGNORECASE)
_UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_LORA_PARAMETER = re.compile(
    r"(?:^|\.)(?:lora_[AB]|lora_embedding_[AB])(?:\.|$)", re.IGNORECASE
)

_HANDOFF_KEYS = {
    "asset_manifest",
    "asset_report",
    "bundles",
    "config_files",
    "config_root",
    "config_tree_sha256",
    "data_root",
    "environment_lock",
    "environment_lock_sha256",
    "experiment_profile_id",
    "git_commit",
    "handoff_sha256",
    "kernel_source_root",
    "kernel_sources",
    "pip_freeze",
    "persist_root",
    "schema_version",
    "status",
}
_INDEX_KEYS = {"configs", "data_root", "schema_version", "status"}
_INDEX_ENTRY_KEYS = {
    "offload_profile",
    "overrides",
    "path",
    "sha256",
    "source_config",
}
_RUN_META_KEYS = {
    "schema_version",
    "stage",
    "git_commit",
    "experiment_profile_id",
    "phase",
    "offload_profile",
    "budget_projection_sha256",
    "pipeline_dir",
    "started_at",
}
_GPU_HARDWARE_KEYS = {
    "capacity_profile",
    "cuda_runtime",
    "cuda_toolkit",
    "driver_version",
    "gpu_compute_capability",
    "gpu_free_memory_bytes",
    "gpu_name",
    "gpu_total_memory_bytes",
    "gpu_uuid",
    "host_cpu_count",
    "host_total_memory_bytes",
    "minimum_persistent_disk_free_bytes",
    "other_compute_process_count",
    "persistent_disk_free_bytes",
    "persistent_disk_probe_path",
    "torch_gpu_total_memory_bytes",
}
_BUILD_INFO_KEYS = {
    "build_info_sha256",
    "environment_lock_sha256",
    "kernels",
    "packages",
    "pip_freeze_sha256",
    "python",
    "schema_version",
    "status",
    "system",
    "training_gate",
}
_BUILD_SYSTEM_KEYS = {
    "cuda_runtime",
    "driver_version",
    "gpu_compute_capability",
    "gpu_name",
    "operating_system",
}
_BUILD_KERNEL_KEYS = {
    "bf16_backward_log_sha256",
    "bf16_forward_log_sha256",
    "build_log_sha256",
    "commit",
    "optimizer_loop_log_sha256",
    "repository",
    "status",
}
_SCIENTIFIC_KEYS = {
    "all_outputs_truncated",
    "finite_gradients",
    "finite_losses",
    "high_truncation_rate",
    "nonzero_advantage_groups",
    "reward_variance_groups",
    "systematic_format_failure",
}
_TELEMETRY_PHASES = (
    "rollout",
    "reward",
    "actor_log_prob",
    "reference_log_prob",
    "update",
    "save",
)
_LOGITS_KEYS = {"bytes", "dtype", "element_size", "numel", "shape"}
_PHASE_TELEMETRY_KEYS = {
    "actor_peak_allocated_bytes",
    "actor_peak_reserved_bytes",
    "actor_post_allocated_bytes",
    "actor_post_reserved_bytes",
    "duration_seconds",
    "host_peak_used_bytes",
    "name",
    "nvml_peak_used_bytes",
    "reference_peak_allocated_bytes",
    "reference_peak_reserved_bytes",
    "reference_post_allocated_bytes",
    "reference_post_reserved_bytes",
    "swap_peak_used_bytes",
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
_CAPACITY_STOP_EXCEPTIONS = {
    "torch.OutOfMemoryError",
    "torch.cuda.OutOfMemoryError",
}

_GATE_CONFIG_IDS = {
    "g0_qwen35_08b",
    "g1_qwen35_2b_step1",
    "g1_qwen35_2b_resume2",
}
_EVAL_CONFIG_IDS = {
    "eval40_qwen35_2b_5090",
    "eval80_qwen35_2b_5090",
}
_ALL_CONFIG_IDS = _GATE_CONFIG_IDS | _EVAL_CONFIG_IDS | {
    f"{task}_{profile}"
    for task in capacity_evidence.CONFIG_TASKS
    for profile in ("r0", "r1")
}

_ATTEMPTS = (
    ("g2a", "g2a", "g2a.run", (1,)),
    ("g2b-step1", "g2b-step1", "g2b-step1.run", (1,)),
    ("g2b-resume5", "g2b-resume5", "g2b-resume5.run", (2, 3, 4, 5)),
    ("length-stress", "g2-length-stress", "g2-length-stress.run", (1,)),
)


class CapacityAggregateError(RuntimeError):
    """Raised when durable capacity inputs cannot be proven complete."""


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
        raise CapacityAggregateError(f"value is not canonical JSON: {exc}") from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CapacityAggregateError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CapacityAggregateError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise CapacityAggregateError(
            f"{label} keys mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise CapacityAggregateError(f"{label} must be a lowercase SHA-256")
    return value


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise CapacityAggregateError(f"{label} is not a safe identifier")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CapacityAggregateError(f"{label} must be an integer >= {minimum}")
    return value


def _safe_path(
    value: str | os.PathLike[str],
    label: str,
    *,
    exists: bool,
    kind: str | None = None,
) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise CapacityAggregateError(f"{label} must be absolute")
    lexical = Path(os.path.abspath(raw))
    for component in [*reversed(lexical.parents), lexical]:
        if component.is_symlink():
            raise CapacityAggregateError(f"{label} contains a symlink: {component}")
    try:
        resolved = lexical.resolve(strict=exists)
    except OSError as exc:
        raise CapacityAggregateError(f"cannot resolve {label}: {exc}") from exc
    if resolved != lexical or str(raw) != str(resolved):
        raise CapacityAggregateError(f"{label} must be an absolute canonical path")
    if kind == "file" and not resolved.is_file():
        raise CapacityAggregateError(f"{label} must be a regular file")
    if kind == "directory" and not resolved.is_dir():
        raise CapacityAggregateError(f"{label} must be a directory")
    return resolved


def _within(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CapacityAggregateError(f"{label} escapes its allowed root") from exc


def _load_json(
    value: str | os.PathLike[str],
    label: str,
    *,
    canonical: bool,
) -> tuple[Path, Mapping[str, Any]]:
    path = _safe_path(value, label, exists=True, kind="file")
    try:
        raw = path.read_bytes()
        parsed = json.loads(raw, object_pairs_hook=_reject_duplicate_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CapacityAggregateError(f"cannot read {label}: {exc}") from exc
    result = _mapping(parsed, label)
    if canonical and raw != _canonical_bytes(result) + b"\n":
        raise CapacityAggregateError(f"{label} is not canonical newline-terminated JSON")
    return path, result


def _field(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for component in dotted.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise CapacityAggregateError(f"missing required field {dotted}")
        current = current[component]
    return current


def _object_field(value: Any, name: str, label: str) -> Any:
    if isinstance(value, Mapping):
        if name not in value:
            raise CapacityAggregateError(f"{label} lacks {name}")
        return value[name]
    if not hasattr(value, name):
        raise CapacityAggregateError(f"{label} lacks {name}")
    return getattr(value, name)


def _default_handoff_verifier(path: Path, commit: str) -> Mapping[str, Any]:
    from scripts.cloud.cloud_state import verify_handoff

    verified, _ = verify_handoff(path, commit)
    return verified


def _load_handoff(
    path: str | os.PathLike[str],
    verifier: Callable[[Path, str], Any] | None,
) -> tuple[Path, Mapping[str, Any]]:
    source, handoff = _load_json(path, "CPU handoff", canonical=True)
    _exact_keys(handoff, _HANDOFF_KEYS, "CPU handoff")
    if (
        handoff["schema_version"] != 3
        or handoff["status"] != "cpu_ready"
        or handoff["experiment_profile_id"] != PROFILE_ID
    ):
        raise CapacityAggregateError("CPU handoff is not the active schema-v3 5090/2B handoff")
    commit = handoff["git_commit"]
    if not isinstance(commit, str) or _HEX40.fullmatch(commit) is None:
        raise CapacityAggregateError("CPU handoff Git commit is malformed")
    digest = _sha256(handoff["handoff_sha256"], "handoff.handoff_sha256")
    unsigned = dict(handoff)
    unsigned.pop("handoff_sha256")
    if digest != _canonical_sha256(unsigned):
        raise CapacityAggregateError("CPU handoff self-hash mismatch")
    checked = (verifier or _default_handoff_verifier)(source, commit)
    if isinstance(checked, tuple):
        checked = checked[0]
    if checked is not None and dict(_mapping(checked, "verified CPU handoff")) != dict(handoff):
        raise CapacityAggregateError("CPU handoff changed during verification")
    return source, handoff


def _default_index_loader(path: Path, digest: str) -> Mapping[str, Any]:
    from scripts.cloud.run_resolved_training import _load_index

    _, value = _load_index(path, digest)
    return value


def _load_index(
    path: str | os.PathLike[str],
    handoff: Mapping[str, Any],
    loader: Callable[[Path, str], Any] | None,
) -> tuple[Path, Mapping[str, Any], str]:
    source, raw = _load_json(path, "resolved config index", canonical=False)
    _exact_keys(raw, _INDEX_KEYS, "resolved config index")
    expected = _safe_path(
        Path(str(handoff["config_root"])) / "index.json",
        "handoff config index",
        exists=True,
        kind="file",
    )
    if source != expected:
        raise CapacityAggregateError("resolved config index differs from the CPU handoff")
    digest = _sha256_file(source)
    config_files = _mapping(handoff["config_files"], "handoff.config_files")
    index_record = _mapping(config_files.get("index.json"), "handoff.config_files.index.json")
    if index_record.get("sha256") != digest or index_record.get("size") != source.stat().st_size:
        raise CapacityAggregateError("resolved config index bytes drifted from the CPU handoff")
    checked = (loader or _default_index_loader)(source, digest)
    if isinstance(checked, tuple):
        checked = checked[-1]
    index = _mapping(checked, "verified resolved config index")
    if dict(index) != dict(raw):
        raise CapacityAggregateError("resolved config index changed during verification")
    if index["schema_version"] != 1 or index["status"] != "resolved":
        raise CapacityAggregateError("resolved config index contract is invalid")
    configs = _mapping(index["configs"], "resolved config index.configs")
    if set(configs) != _ALL_CONFIG_IDS:
        raise CapacityAggregateError(
            "resolved config index must contain the exact 33-config active inventory"
        )
    for config_id, raw_entry in configs.items():
        entry = _mapping(raw_entry, f"configs.{config_id}")
        _exact_keys(entry, _INDEX_ENTRY_KEYS, f"configs.{config_id}")
        _sha256(entry["sha256"], f"configs.{config_id}.sha256")
        expected_path = source.parent / f"{config_id}.yaml"
        configured_path = _safe_path(
            str(entry["path"]), f"configs.{config_id}.path", exists=True, kind="file"
        )
        if configured_path != expected_path or _sha256_file(configured_path) != entry["sha256"]:
            raise CapacityAggregateError(f"resolved config bytes drifted for {config_id}")
        if config_id in _GATE_CONFIG_IDS | _EVAL_CONFIG_IDS:
            if entry["offload_profile"] is not None or entry["source_config"] != config_id:
                raise CapacityAggregateError(f"resolved config identity drifted for {config_id}")
        else:
            profile = config_id.rsplit("_", 1)[-1]
            source_id = config_id.removesuffix(f"_{profile}")
            if (
                profile not in {"r0", "r1"}
                or entry["offload_profile"] != profile
                or entry["source_config"] != source_id
            ):
                raise CapacityAggregateError(f"resolved offload identity drifted for {config_id}")
    return source, index, digest


def _load_gpu_evidence(
    value: str | os.PathLike[str],
    profile: str,
    *,
    require_observed_profile: bool = True,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    supplied = _safe_path(value, "GPU evidence", exists=True, kind="file")
    if supplied.name == "build-info.json":
        build_info_path = supplied
        evidence_path = supplied.parent / "gpu-evidence" / "causal-conv1d" / "bf16-forward.json"
    else:
        evidence_path = supplied
        if (
            evidence_path.name != "bf16-forward.json"
            or evidence_path.parent.name != "causal-conv1d"
            or evidence_path.parent.parent.name != "gpu-evidence"
        ):
            raise CapacityAggregateError("GPU evidence must be causal-conv1d/bf16-forward.json")
        build_info_path = evidence_path.parents[2] / "build-info.json"
    evidence_path, evidence = _load_json(evidence_path, "GPU forward evidence", canonical=False)
    build_info_path, build_info = _load_json(build_info_path, "GPU build info", canonical=False)
    del build_info_path
    _exact_keys(
        evidence,
        {"hardware", "kernel", "losses", "mode", "optimizer_steps", "shape", "status"},
        "GPU forward evidence",
    )
    if (
        evidence["kernel"] != "causal-conv1d"
        or evidence["mode"] != "forward"
        or evidence["optimizer_steps"] != 0
        or evidence["status"] != "verified"
        or evidence["losses"] != []
    ):
        raise CapacityAggregateError("GPU forward evidence contract changed")
    _exact_keys(build_info, _BUILD_INFO_KEYS, "GPU build info")
    build_digest = _sha256(build_info.get("build_info_sha256"), "build_info_sha256")
    unsigned_build = dict(build_info)
    unsigned_build.pop("build_info_sha256")
    if build_digest != _canonical_sha256(unsigned_build):
        raise CapacityAggregateError("GPU build info self-hash mismatch")
    if (
        build_info.get("schema_version") != 1
        or build_info.get("status") != "VERIFIED_SM120"
        or build_info.get("training_gate") != "READY"
    ):
        raise CapacityAggregateError("GPU build info is not ready")
    kernels = _mapping(build_info.get("kernels"), "build_info.kernels")
    if set(kernels) != {"causal-conv1d", "flash-linear-attention"}:
        raise CapacityAggregateError("GPU build info kernel inventory drifted")
    for kernel_name, raw_record in kernels.items():
        record = _mapping(raw_record, f"build_info.kernels.{kernel_name}")
        _exact_keys(record, _BUILD_KERNEL_KEYS, f"build_info.kernels.{kernel_name}")
        if record["status"] != "VERIFIED_SM120":
            raise CapacityAggregateError(f"GPU build kernel {kernel_name} is not verified")
        for field in (
            "bf16_backward_log_sha256",
            "bf16_forward_log_sha256",
            "build_log_sha256",
            "optimizer_loop_log_sha256",
        ):
            _sha256(record[field], f"build_info.kernels.{kernel_name}.{field}")
    causal = _mapping(kernels.get("causal-conv1d"), "build_info.kernels.causal-conv1d")
    evidence_digest = _sha256_file(evidence_path)
    if causal.get("bf16_forward_log_sha256") != evidence_digest:
        raise CapacityAggregateError("GPU forward evidence is not bound by build-info")
    hardware = _mapping(evidence["hardware"], "GPU evidence.hardware")
    _exact_keys(hardware, _GPU_HARDWARE_KEYS, "GPU evidence.hardware")
    system = _mapping(build_info["system"], "build_info.system")
    _exact_keys(system, _BUILD_SYSTEM_KEYS, "build_info.system")
    for field in ("cuda_runtime", "driver_version", "gpu_compute_capability", "gpu_name"):
        if system[field] != hardware[field]:
            raise CapacityAggregateError(f"GPU build/evidence identity drifted at {field}")
    observed_profile = hardware["capacity_profile"]
    if observed_profile not in {"R0", "R1"} or (
        require_observed_profile and observed_profile != profile.upper()
    ):
        raise CapacityAggregateError("GPU evidence offload profile drifted")
    required_host_memory = (80 if observed_profile == "R0" else 128) * GIB
    _integer(
        hardware["host_total_memory_bytes"],
        "GPU host total memory",
        minimum=required_host_memory,
    )
    if (
        hardware["cuda_runtime"] != "13.0"
        or hardware["cuda_toolkit"] != "13.0"
        or hardware["gpu_compute_capability"] != [12, 0]
        or hardware["other_compute_process_count"] != 0
    ):
        raise CapacityAggregateError("GPU evidence is not a clean CUDA-13 sm_120 host")
    name = hardware["gpu_name"]
    uuid = hardware["gpu_uuid"]
    total = hardware["gpu_total_memory_bytes"]
    if not isinstance(name, str) or _RTX_5090.search(" ".join(name.split())) is None:
        raise CapacityAggregateError("GPU evidence is not a GeForce RTX 5090")
    if not isinstance(uuid, str) or not uuid.startswith("GPU-"):
        raise CapacityAggregateError("GPU evidence UUID is malformed")
    _integer(total, "GPU total VRAM", minimum=31 * GIB)
    if hardware["torch_gpu_total_memory_bytes"] < 31 * GIB:
        raise CapacityAggregateError("Torch-visible VRAM is below 31 GiB")
    return (
        {"name": name, "total_vram_bytes": total, "uuid": uuid},
        evidence_digest,
        dict(hardware),
    )


def build_target_identity(
    *,
    handoff_path: str | os.PathLike[str],
    index_path: str | os.PathLike[str],
    gpu_evidence_path: str | os.PathLike[str],
    profile: str,
    handoff_verifier: Callable[[Path, str], Any] | None = None,
    index_loader: Callable[[Path, str], Any] | None = None,
) -> dict[str, Any]:
    """Build a pre-run R0/R1 target identity from sealed, immutable inputs."""

    if profile not in {"r0", "r1"}:
        raise CapacityAggregateError("profile must be r0 or r1")
    _, handoff = _load_handoff(handoff_path, handoff_verifier)
    _, index, _ = _load_index(index_path, handoff, index_loader)
    configs = _mapping(index["configs"], "resolved config index.configs")
    selected = {
        config_id: configs[config_id]["sha256"]
        for config_id in capacity_evidence.expected_config_ids(profile.upper())
    }
    # R1 approval is intentionally created before an R1 process exists. The
    # prior R0 evidence supplies the immutable physical GPU identity; R1 host
    # RAM/offload readiness is rechecked by the new launcher's GPU preflight.
    gpu, _, _ = _load_gpu_evidence(
        gpu_evidence_path,
        profile,
        require_observed_profile=profile == "r0",
    )
    return capacity_evidence.build_identity(
        commit=handoff["git_commit"],
        cpu_handoff_sha256=handoff["handoff_sha256"],
        active_config_tree_sha256=handoff["config_tree_sha256"],
        selected_profile=profile.upper(),
        selected_configs=selected,
        gpu=gpu,
    )


def _read_meta(path: Path, expected_stage: str) -> dict[str, str]:
    meta_path = _safe_path(path / "run.meta", "run metadata", exists=True, kind="file")
    try:
        raw = meta_path.read_bytes()
        text = raw.decode("ascii")
    except (OSError, UnicodeError) as exc:
        raise CapacityAggregateError(f"cannot read run metadata: {exc}") from exc
    if not text.endswith("\n") or "\r" in text or not text:
        raise CapacityAggregateError("run metadata is not canonical newline-terminated text")
    result: dict[str, str] = {}
    for line in text[:-1].split("\n"):
        if "=" not in line:
            raise CapacityAggregateError("run metadata contains a malformed line")
        key, item = line.split("=", 1)
        if not key or key in result:
            raise CapacityAggregateError("run metadata contains a duplicate/empty key")
        result[key] = item
    _exact_keys(result, _RUN_META_KEYS, "run metadata")
    if result["schema_version"] != "1" or result["stage"] != expected_stage:
        raise CapacityAggregateError(f"run metadata stage drifted for {expected_stage}")
    if not _UTC_TIMESTAMP.fullmatch(result["started_at"]):
        raise CapacityAggregateError("run metadata timestamp is malformed")
    if not _HEX40.fullmatch(result["git_commit"]):
        raise CapacityAggregateError("run metadata Git commit is malformed")
    if result["experiment_profile_id"] != PROFILE_ID or result["phase"] != "gpu-capacity":
        raise CapacityAggregateError("run metadata profile/phase drifted")
    if result["offload_profile"] not in {"r0", "r1"}:
        raise CapacityAggregateError("run metadata offload profile is invalid")
    budget_sha = result["budget_projection_sha256"]
    if (result["offload_profile"] == "r0" and budget_sha != "") or (
        result["offload_profile"] == "r1" and _HEX64.fullmatch(budget_sha) is None
    ):
        raise CapacityAggregateError("run metadata budget projection identity drifted")
    return result


def _verify_success(run_dir: Path) -> None:
    dot_entries = {entry.name for entry in run_dir.iterdir() if entry.name.startswith(".")}
    if dot_entries != {".success"}:
        raise CapacityAggregateError(
            f"attempt must have exactly one success terminal marker, observed {sorted(dot_entries)}"
        )
    marker = _safe_path(run_dir / ".success", "success marker", exists=True, kind="file")
    if marker.read_bytes() != b"0\n":
        raise CapacityAggregateError("success marker is malformed")


def _resolve_pointer(
    value: str | os.PathLike[str], expected_name: str
) -> tuple[Path, Path, Path]:
    pointer = _safe_path(value, f"{expected_name} pointer", exists=True, kind="file")
    if (
        pointer.name != expected_name
        or pointer.parents[1].name not in {"r0", "r1"}
        or pointer.parents[2].name != "gpu-capacity"
        or pointer.parents[3].name != "stages"
        or not pointer.parent.name
    ):
        raise CapacityAggregateError(
            "pointer must be pipeline/stages/gpu-capacity/<profile>/"
            f"<generation>/{expected_name}"
        )
    try:
        raw = pointer.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise CapacityAggregateError(f"cannot read {expected_name} pointer: {exc}") from exc
    if not text.endswith("\n") or text.count("\n") != 1 or "\r" in text:
        raise CapacityAggregateError(f"{expected_name} pointer must contain exactly one line")
    target = text[:-1]
    if not target or target != target.strip():
        raise CapacityAggregateError(f"{expected_name} pointer target is malformed")
    run_dir = _safe_path(target, f"{expected_name} run", exists=True, kind="directory")
    if str(run_dir) != target:
        raise CapacityAggregateError(f"{expected_name} pointer target is noncanonical")
    return pointer, run_dir, pointer.parents[4]


def _resolve_stopped_pointer(
    value: str | os.PathLike[str], expected_stage: str, profile: str
) -> tuple[Path, Path, Path]:
    pointer = _safe_path(value, "capacity-stop attempt pointer", exists=True, kind="file")
    if (
        pointer.parents[1].name != profile
        or pointer.parents[2].name != "gpu-capacity"
        or pointer.parents[3].name != "attempts"
        or not pointer.parent.name
        or not pointer.name.startswith(f"{expected_stage}-")
        or not pointer.name.endswith(".run")
    ):
        raise CapacityAggregateError(
            f"capacity-stop pointer must be pipeline/attempts/gpu-capacity/{profile}/"
            "<generation>/<stage>-*.run"
        )
    try:
        raw = pointer.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise CapacityAggregateError(f"cannot read capacity-stop pointer: {exc}") from exc
    if not text.endswith("\n") or text.count("\n") != 1 or "\r" in text:
        raise CapacityAggregateError("capacity-stop pointer must contain exactly one line")
    target = text[:-1]
    if not target or target != target.strip():
        raise CapacityAggregateError("capacity-stop pointer target is malformed")
    run_dir = _safe_path(target, "capacity-stop attempt", exists=True, kind="directory")
    if str(run_dir) != target:
        raise CapacityAggregateError("capacity-stop pointer target is noncanonical")
    return pointer, run_dir, pointer.parents[4]


def _verify_capacity_stop_terminal(run_dir: Path) -> None:
    dot_entries = {entry.name for entry in run_dir.iterdir() if entry.name.startswith(".")}
    if dot_entries != {".capacity-stop"}:
        raise CapacityAggregateError(
            "stopped attempt must have exactly one .capacity-stop terminal marker"
        )
    marker = _safe_path(
        run_dir / ".capacity-stop", "capacity-stop terminal", exists=True, kind="file"
    )
    retryable = _safe_path(
        run_dir / "retryable", "capacity-stop retryable marker", exists=True, kind="file"
    )
    if marker.read_bytes() != b"43\n" or retryable.read_bytes() != b"false\n":
        raise CapacityAggregateError("capacity-stop terminal marker is malformed")


def _default_runtime_verifier(path: Path) -> Mapping[str, Any]:
    from scripts.cloud.run_resolved_training import verify_bound_config

    return verify_bound_config(path)


def _default_ledger_loader(path: Path) -> Mapping[str, Any]:
    from scripts.cloud.training_telemetry import load_ledger

    return load_ledger(path)


def _validate_capacity_stop_marker(
    run_dir: Path,
    *,
    config_id: str,
    runtime: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    marker_path, marker = _load_json(
        run_dir / "evidence" / "capacity-stop.json",
        "capacity-stop evidence",
        canonical=True,
    )
    _exact_keys(marker, _CAPACITY_STOP_KEYS, "capacity-stop evidence")
    digest = _sha256(marker["capacity_stop_sha256"], "capacity_stop_sha256")
    unsigned = dict(marker)
    unsigned.pop("capacity_stop_sha256")
    if digest != _canonical_sha256(unsigned):
        raise CapacityAggregateError("capacity-stop evidence self-hash mismatch")
    if (
        marker["kind"] != CAPACITY_STOP_KIND
        or marker["schema_version"] != 1
        or marker["status"] != "capacity-stop"
        or marker["reason"] != "cuda_oom"
        or marker["exception_type"] not in _CAPACITY_STOP_EXCEPTIONS
        or marker["config_id"] != config_id
        or marker["attempt_id"] != run_dir.name
        or marker["attempt_root"] != str(run_dir)
    ):
        raise CapacityAggregateError("capacity-stop evidence identity/reason drifted")
    runtime_record = _mapping(
        marker["runtime_bound_evidence"], "capacity-stop runtime_bound_evidence"
    )
    binding_record = _mapping(marker["runtime_binding"], "capacity-stop runtime_binding")
    _exact_keys(
        runtime_record,
        {"path", "file_sha256", "evidence_sha256"},
        "capacity-stop runtime_bound_evidence",
    )
    _exact_keys(
        binding_record,
        {"path", "file_sha256", "binding_sha256"},
        "capacity-stop runtime_binding",
    )
    runtime_path = _safe_path(
        runtime_record["path"],
        "capacity-stop runtime-bound evidence path",
        exists=True,
        kind="file",
    )
    binding_path = _safe_path(
        binding_record["path"],
        "capacity-stop runtime binding path",
        exists=True,
        kind="file",
    )
    if (
        runtime_path != run_dir / "runtime-bound" / "runtime-bound.json"
        or binding_path != run_dir / "runtime-binding-source.json"
        or runtime_record["file_sha256"] != _sha256_file(runtime_path)
        or binding_record["file_sha256"] != _sha256_file(binding_path)
        or runtime_record["evidence_sha256"] != runtime.get("evidence_sha256")
    ):
        raise CapacityAggregateError("capacity-stop runtime evidence bytes drifted")
    source_binding = _mapping(runtime.get("source_binding"), "runtime source_binding")
    if (
        source_binding.get("path") != str(binding_path)
        or source_binding.get("file_sha256") != binding_record["file_sha256"]
        or source_binding.get("binding_sha256") != binding_record["binding_sha256"]
    ):
        raise CapacityAggregateError("capacity-stop runtime binding identity drifted")
    if marker_path != run_dir / "evidence" / "capacity-stop.json":
        raise CapacityAggregateError("capacity-stop evidence escaped the attempt")
    return dict(marker), digest


def _validate_stopped_attempt(
    *,
    logical_stage: str,
    runtime_stage: str,
    run_dir: Path,
    pipeline_dir: Path,
    config_id: str,
    config_sha256: str,
    index_path: Path,
    index_sha256: str,
    commit: str,
    profile: str,
    gpu_uuid: str,
    expected_steps: tuple[int, ...],
    runtime_verifier: Callable[[Path], Any] | None,
    ledger_loader: Callable[[Path], Any] | None,
) -> dict[str, Any]:
    _verify_capacity_stop_terminal(run_dir)
    meta = _read_meta(run_dir, runtime_stage)
    if (
        meta["git_commit"] != commit
        or meta["offload_profile"] != profile
        or _safe_path(meta["pipeline_dir"], "run pipeline", exists=True, kind="directory")
        != pipeline_dir
    ):
        raise CapacityAggregateError("capacity-stop run identity drifted")
    runtime = _mapping(
        (runtime_verifier or _default_runtime_verifier)(run_dir / "runtime-bound"),
        "capacity-stop runtime evidence",
    )
    if (
        runtime.get("config_id") != config_id
        or runtime.get("attempt_id") != run_dir.name
        or runtime.get("attempt_root") != str(run_dir)
        or runtime.get("source_index") != {"path": str(index_path), "sha256": index_sha256}
        or runtime.get("source_config")
        != {"path": str(index_path.parent / f"{config_id}.yaml"), "sha256": config_sha256}
    ):
        raise CapacityAggregateError("capacity-stop runtime-bound identity drifted")
    runtime_digest = _sha256(runtime.get("evidence_sha256"), "runtime evidence SHA")
    marker, marker_digest = _validate_capacity_stop_marker(
        run_dir,
        config_id=config_id,
        runtime=runtime,
    )
    telemetry_path = run_dir / "telemetry.json"
    ledger: Mapping[str, Any] | None = None
    if telemetry_path.exists() or telemetry_path.is_symlink():
        _safe_path(telemetry_path, "partial telemetry ledger", exists=True, kind="file")
        ledger = _mapping(
            (ledger_loader or _default_ledger_loader)(telemetry_path),
            "partial telemetry ledger",
        )
    if ledger is None:
        steps: list[Mapping[str, Any]] = []
        ledger_digest: str | None = None
    else:
        identity = _mapping(ledger.get("identity"), "partial telemetry identity")
        expected_identity = {
            "attempt_id": run_dir.name,
            "experiment_profile_id": PROFILE_ID,
            "gpu_uuid": gpu_uuid,
            "offload_profile": profile,
            "sealed_config_id": config_id,
            "sealed_config_sha256": config_sha256,
        }
        if dict(identity) != expected_identity:
            raise CapacityAggregateError("partial telemetry identity drifted")
        raw_steps = ledger.get("steps")
        if not isinstance(raw_steps, list):
            raise CapacityAggregateError("partial telemetry steps must be an array")
        steps = [_mapping(step, "partial telemetry step") for step in raw_steps]
        observed = tuple(step.get("global_step") for step in steps)
        if observed != expected_steps[: len(observed)]:
            raise CapacityAggregateError("partial telemetry is not an exact optimizer-step prefix")
        ledger_digest = _sha256(ledger.get("ledger_sha256"), "partial ledger SHA")
    attempt_digest = _canonical_sha256(
        {
            "attempt_id": run_dir.name,
            "capacity_stop_sha256": marker_digest,
            "config_id": config_id,
            "config_sha256": config_sha256,
            "ledger_sha256": ledger_digest,
            "runtime_evidence_sha256": runtime_digest,
            "stage": logical_stage,
            "terminal_exit_code": CAPACITY_STOP_EXIT_CODE,
        }
    )
    return {
        "attempt_id": run_dir.name,
        "capacity_stop": marker,
        "capacity_stop_sha256": marker_digest,
        "digest": attempt_digest,
        "ledger": None if ledger is None else dict(ledger),
        "ledger_sha256": ledger_digest,
        "meta": meta,
        "run_dir": run_dir,
        "runtime": dict(runtime),
        "runtime_evidence_sha256": runtime_digest,
        "steps": [dict(step) for step in steps],
    }


def _validate_attempt(
    *,
    logical_stage: str,
    runtime_stage: str,
    pointer_path: Path,
    run_dir: Path,
    pipeline_dir: Path,
    config_id: str,
    config_sha256: str,
    index_path: Path,
    index_sha256: str,
    commit: str,
    profile: str,
    gpu_uuid: str,
    expected_steps: tuple[int, ...],
    runtime_verifier: Callable[[Path], Any] | None,
    ledger_loader: Callable[[Path], Any] | None,
) -> dict[str, Any]:
    del pointer_path
    _verify_success(run_dir)
    meta = _read_meta(run_dir, runtime_stage)
    if (
        meta["git_commit"] != commit
        or meta["offload_profile"] != profile
        or _safe_path(meta["pipeline_dir"], "run pipeline", exists=True, kind="directory")
        != pipeline_dir
    ):
        raise CapacityAggregateError(f"{logical_stage} run identity drifted")
    runtime = _mapping(
        (runtime_verifier or _default_runtime_verifier)(run_dir / "runtime-bound"),
        f"{logical_stage} runtime evidence",
    )
    if (
        runtime.get("config_id") != config_id
        or runtime.get("attempt_id") != run_dir.name
        or runtime.get("attempt_root") != str(run_dir)
        or runtime.get("source_index") != {"path": str(index_path), "sha256": index_sha256}
        or runtime.get("source_config")
        != {"path": str(index_path.parent / f"{config_id}.yaml"), "sha256": config_sha256}
    ):
        raise CapacityAggregateError(f"{logical_stage} runtime-bound identity drifted")
    runtime_digest = _sha256(runtime.get("evidence_sha256"), f"{logical_stage}.evidence_sha256")
    ledger = _mapping(
        (ledger_loader or _default_ledger_loader)(run_dir / "telemetry.json"),
        f"{logical_stage} telemetry ledger",
    )
    identity = _mapping(ledger.get("identity"), f"{logical_stage} ledger identity")
    expected_identity = {
        "attempt_id": run_dir.name,
        "experiment_profile_id": PROFILE_ID,
        "gpu_uuid": gpu_uuid,
        "offload_profile": profile,
        "sealed_config_id": config_id,
        "sealed_config_sha256": config_sha256,
    }
    if dict(identity) != expected_identity:
        raise CapacityAggregateError(f"{logical_stage} telemetry identity drifted")
    steps = ledger.get("steps")
    if not isinstance(steps, list) or tuple(step.get("global_step") for step in steps) != expected_steps:
        raise CapacityAggregateError(
            f"{logical_stage} telemetry must contain exactly steps {list(expected_steps)}"
        )
    ledger_digest = _sha256(ledger.get("ledger_sha256"), f"{logical_stage}.ledger_sha256")
    attempt_digest = _canonical_sha256(
        {
            "attempt_id": run_dir.name,
            "config_id": config_id,
            "config_sha256": config_sha256,
            "ledger_sha256": ledger_digest,
            "runtime_evidence_sha256": runtime_digest,
            "stage": logical_stage,
        }
    )
    return {
        "attempt_id": run_dir.name,
        "digest": attempt_digest,
        "ledger": dict(ledger),
        "ledger_sha256": ledger_digest,
        "meta": meta,
        "run_dir": run_dir,
        "runtime": dict(runtime),
        "runtime_evidence_sha256": runtime_digest,
    }


def _default_checkpoint_verifier(path: Path) -> Any:
    from verl.utils.checkpoint.reproduction import verify_reproduction_checkpoint_directory

    return verify_reproduction_checkpoint_directory(path)


def _default_adapter_verifier(path: Path) -> Any:
    from verl.utils.checkpoint.reproduction import validate_adapter_export

    return validate_adapter_export(path)


def _checkpoint_pair(result: Any, label: str) -> tuple[Any, Any]:
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise CapacityAggregateError(f"{label} verifier returned an invalid result")
    return result[0], result[1]


def _validate_trainable_manifest(value: Any) -> str:
    manifest = _mapping(value, "checkpoint trainable-parameter manifest")
    _exact_keys(
        manifest,
        {
            "schema_version",
            "names",
            "tensor_count",
            "trainable_numel",
            "total_numel",
            "trainable_ratio",
            "manifest_sha256",
            "state_sha256",
        },
        "checkpoint trainable-parameter manifest",
    )
    names = manifest["names"]
    if (
        manifest["schema_version"] != 1
        or not isinstance(names, list)
        or not names
        or names != sorted(set(names))
        or any(not isinstance(name, str) or _LORA_PARAMETER.search(name) is None for name in names)
        or manifest["tensor_count"] != len(names)
    ):
        raise CapacityAggregateError("checkpoint does not prove a LoRA-only trainable surface")
    trainable = _integer(manifest["trainable_numel"], "trainable_numel", minimum=1)
    total = _integer(manifest["total_numel"], "total_numel", minimum=trainable)
    ratio = manifest["trainable_ratio"]
    if (
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or not math.isfinite(float(ratio))
        or not math.isclose(float(ratio), trainable / total, rel_tol=1e-12, abs_tol=0.0)
    ):
        raise CapacityAggregateError("checkpoint trainable ratio is inconsistent")
    _sha256(manifest["manifest_sha256"], "trainable manifest SHA")
    return _sha256(manifest["state_sha256"], "initial adapter state SHA")


def _state_identity(state: Any, label: str) -> dict[str, Any]:
    resolved = _mapping(_object_field(state, "resolved_config", label), f"{label}.resolved_config")
    return {
        "base_model_id": _object_field(state, "base_model_id", label),
        "base_model_revision": _object_field(state, "base_model_revision", label),
        "data_manifest_sha256": _object_field(state, "data_manifest_sha256", label),
        "experiment_profile_id": _field(resolved, "reproduction.experiment_profile_id"),
        "offload_profile": _field(resolved, "reproduction.offload_profile"),
        "sealed_config_id": _field(resolved, "reproduction.sealed_config_id"),
        "sealed_config_sha256": _field(resolved, "reproduction.sealed_config_sha256"),
        "runtime_attempt_id": _field(resolved, "reproduction.runtime_attempt_id"),
        "resolved_config": resolved,
        "resolved_config_sha256": _object_field(state, "resolved_config_sha256", label),
        "state_sha256": _object_field(state, "sha256", label),
    }


def create_artifact_evidence(
    checkpoint_dir: str | os.PathLike[str],
    adapter_dir: str | os.PathLike[str],
    *,
    checkpoint_verifier: Callable[[Path], Any] | None = None,
    adapter_verifier: Callable[[Path], Any] | None = None,
    runtime_verifier: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    checkpoint = _safe_path(
        checkpoint_dir, "G2b resume5 checkpoint", exists=True, kind="directory"
    )
    adapter_path = _safe_path(adapter_dir, "G2b resume5 adapter", exists=True, kind="directory")
    if checkpoint.name != "global_step_5" or checkpoint.parent.name != "checkpoints":
        raise CapacityAggregateError("G2 artifact checkpoint must be checkpoints/global_step_5")
    run_dir = checkpoint.parent.parent
    expected_adapter = run_dir / "artifacts" / "adapter" / "global_step_5" / "adapter"
    if adapter_path != expected_adapter:
        raise CapacityAggregateError("G2 adapter path does not belong to the resume5 attempt")
    _verify_success(run_dir)
    meta = _read_meta(run_dir, "g2b-resume5")
    final_manifest, final_state = _checkpoint_pair(
        (checkpoint_verifier or _default_checkpoint_verifier)(checkpoint),
        "resume5 checkpoint",
    )
    adapter = (adapter_verifier or _default_adapter_verifier)(adapter_path)
    final = _state_identity(final_state, "resume5 checkpoint")
    if _object_field(final_state, "global_step", "resume5 checkpoint") != 5:
        raise CapacityAggregateError("resume5 checkpoint is not global step 5")
    profile = final["offload_profile"]
    expected_config_id = f"g2b_qwen35_2b_5090_resume5_{profile}"
    if (
        profile not in {"r0", "r1"}
        or final["experiment_profile_id"] != PROFILE_ID
        or final["sealed_config_id"] != expected_config_id
        or final["runtime_attempt_id"] != run_dir.name
        or meta["offload_profile"] != profile
    ):
        raise CapacityAggregateError("resume5 checkpoint profile/config identity drifted")
    config_sha = _sha256(final["sealed_config_sha256"], "resume5 sealed config SHA")
    if final["base_model_id"] != "Qwen/Qwen3.5-2B" or final["base_model_revision"] != (
        "15852e8c16360a2fea060d615a32b45270f8a8fc"
    ):
        raise CapacityAggregateError("resume5 checkpoint base-model identity drifted")
    resolved = final["resolved_config"]
    if (
        _field(resolved, "trainer.total_training_steps") != 5
        or _field(resolved, "trainer.resume_mode") != "resume_path"
        or _safe_path(
            _field(resolved, "trainer.default_local_dir"),
            "resume5 checkpoint output root",
            exists=True,
            kind="directory",
        )
        != checkpoint.parent
        or _safe_path(
            _field(resolved, "reproduction.adapter_export_dir"),
            "resume5 adapter output root",
            exists=True,
            kind="directory",
        )
        != run_dir / "artifacts" / "adapter"
    ):
        raise CapacityAggregateError("resume5 checkpoint runtime paths/step drifted")
    runtime_path = run_dir / "runtime-bound"
    runtime = _mapping(
        (runtime_verifier or _default_runtime_verifier)(runtime_path),
        "resume5 runtime-bound evidence",
    )
    if (
        runtime.get("config_id") != expected_config_id
        or runtime.get("attempt_id") != run_dir.name
        or runtime.get("runtime_bound_config_sha256") != final["resolved_config_sha256"]
    ):
        raise CapacityAggregateError("resume5 checkpoint differs from runtime-bound evidence")
    predecessor = _mapping(runtime.get("resume_predecessor"), "resume5 predecessor")
    predecessor_id = f"g2b_qwen35_2b_5090_step1_{profile}"
    predecessor_sha = _sha256(
        predecessor.get("resolved_config_sha256"), "resume predecessor config SHA"
    )
    predecessor_checkpoint = _safe_path(
        predecessor.get("checkpoint_dir"),
        "resume predecessor checkpoint",
        exists=True,
        kind="directory",
    )
    if (
        predecessor.get("logical_config_id") != predecessor_id
        or predecessor.get("checkpoint_step") != 1
        or predecessor_checkpoint.name != "global_step_1"
    ):
        raise CapacityAggregateError("resume5 predecessor identity drifted")
    predecessor_run = predecessor_checkpoint.parent.parent
    _verify_success(predecessor_run)
    predecessor_meta = _read_meta(predecessor_run, "g2b-step1")
    if (
        predecessor_meta["git_commit"] != meta["git_commit"]
        or predecessor_meta["offload_profile"] != profile
        or predecessor_meta["pipeline_dir"] != meta["pipeline_dir"]
    ):
        raise CapacityAggregateError("resume predecessor run identity drifted")
    predecessor_manifest, predecessor_state = _checkpoint_pair(
        (checkpoint_verifier or _default_checkpoint_verifier)(predecessor_checkpoint),
        "step1 predecessor checkpoint",
    )
    predecessor_identity = _state_identity(predecessor_state, "step1 predecessor checkpoint")
    if (
        _object_field(predecessor_state, "global_step", "step1 predecessor checkpoint") != 1
        or predecessor_identity["sealed_config_id"] != predecessor_id
        or predecessor_identity["sealed_config_sha256"] != predecessor_sha
        or predecessor_identity["runtime_attempt_id"] != predecessor_run.name
        or predecessor_identity["base_model_id"] != final["base_model_id"]
        or predecessor_identity["base_model_revision"] != final["base_model_revision"]
    ):
        raise CapacityAggregateError("resume predecessor checkpoint identity drifted")
    adapter_fields = {
        "global_step": 5,
        "base_model_id": final["base_model_id"],
        "base_model_revision": final["base_model_revision"],
        "source_extra_state_sha256": final["state_sha256"],
        "adapter_state_sha256": _object_field(final_state, "adapter_state_sha256", "resume5 checkpoint"),
        "lora_config_sha256": _object_field(final_state, "lora_config_sha256", "resume5 checkpoint"),
        "lora_target_sha256": _object_field(final_state, "lora_target_sha256", "resume5 checkpoint"),
        "text_mapping_sha256": _object_field(final_state, "text_mapping_sha256", "resume5 checkpoint"),
    }
    for name, expected in adapter_fields.items():
        if _object_field(adapter, name, "resume5 adapter") != expected:
            raise CapacityAggregateError(f"resume5 adapter differs from checkpoint at {name}")
    if tuple(_object_field(adapter, "adapter_tensor_keys", "resume5 adapter")) != tuple(
        _object_field(final_state, "adapter_tensor_keys", "resume5 checkpoint")
    ):
        raise CapacityAggregateError("resume5 adapter tensor keys drifted")
    build = _mapping(
        _object_field(final_state, "model_build_metadata", "resume5 checkpoint"),
        "resume5 model build metadata",
    )
    initial_hash = _validate_trainable_manifest(build.get("trainable_parameters"))
    final_hash = _sha256(adapter_fields["adapter_state_sha256"], "final adapter state SHA")
    if initial_hash == final_hash:
        raise CapacityAggregateError("resume5 adapter update is zero")
    completion_sha = _sha256(
        _object_field(final_manifest, "sha256", "resume5 checkpoint manifest"),
        "resume5 checkpoint completion SHA",
    )
    predecessor_completion_sha = _sha256(
        _object_field(predecessor_manifest, "sha256", "step1 checkpoint manifest"),
        "step1 checkpoint completion SHA",
    )
    adapter_metadata_sha = _sha256(
        _object_field(adapter, "sha256", "resume5 adapter"), "adapter metadata SHA"
    )
    payload = {
        "adapter": {
            "initial_state_sha256": initial_hash,
            "metadata_sha256": adapter_metadata_sha,
            "path": str(adapter_path),
            "state_sha256": final_hash,
            "tensor_keys_sha256": _canonical_sha256(
                list(_object_field(adapter, "adapter_tensor_keys", "resume5 adapter"))
            ),
        },
        "adapter_update_nonzero": True,
        "attempt_id": run_dir.name,
        "base_model": {"id": final["base_model_id"], "revision": final["base_model_revision"]},
        "base_model_unchanged": True,
        "budget_projection_sha256": meta["budget_projection_sha256"] or None,
        "checkpoint": {
            "completion_sha256": completion_sha,
            "extra_state_sha256": _sha256(final["state_sha256"], "checkpoint extra-state SHA"),
            "global_step": 5,
            "path": str(checkpoint),
        },
        "commit": meta["git_commit"],
        "config_id": expected_config_id,
        "config_sha256": config_sha,
        "experiment_profile_id": PROFILE_ID,
        "kind": ARTIFACT_KIND,
        "offload_profile": profile,
        "pipeline_dir": meta["pipeline_dir"],
        "predecessor": {
            "attempt_id": predecessor_run.name,
            "checkpoint_path": str(predecessor_checkpoint),
            "checkpoint_step": 1,
            "completion_sha256": predecessor_completion_sha,
            "config_id": predecessor_id,
            "config_sha256": predecessor_sha,
            "extra_state_sha256": _sha256(
                predecessor_identity["state_sha256"], "predecessor extra-state SHA"
            ),
        },
        "runtime_bound": {
            "config_sha256": _sha256(
                runtime.get("runtime_bound_config_sha256"), "runtime-bound config SHA"
            ),
            "evidence_sha256": _sha256(
                runtime.get("evidence_sha256"), "runtime-bound evidence SHA"
            ),
            "path": str(runtime_path / "runtime-bound.json"),
        },
        "schema_version": SCHEMA_VERSION,
        "status": "verified",
    }
    return {**payload, "artifact_evidence_sha256": _canonical_sha256(payload)}


def _validate_artifact_schema(value: Mapping[str, Any]) -> None:
    expected = {
        "adapter",
        "adapter_update_nonzero",
        "artifact_evidence_sha256",
        "attempt_id",
        "base_model",
        "base_model_unchanged",
        "budget_projection_sha256",
        "checkpoint",
        "commit",
        "config_id",
        "config_sha256",
        "experiment_profile_id",
        "kind",
        "offload_profile",
        "pipeline_dir",
        "predecessor",
        "runtime_bound",
        "schema_version",
        "status",
    }
    _exact_keys(value, expected, "artifact evidence")
    digest = _sha256(value["artifact_evidence_sha256"], "artifact_evidence_sha256")
    unsigned = dict(value)
    unsigned.pop("artifact_evidence_sha256")
    if digest != _canonical_sha256(unsigned):
        raise CapacityAggregateError("artifact evidence self-hash mismatch")
    if (
        value["kind"] != ARTIFACT_KIND
        or value["schema_version"] != SCHEMA_VERSION
        or value["status"] != "verified"
        or value["experiment_profile_id"] != PROFILE_ID
        or value["adapter_update_nonzero"] is not True
        or value["base_model_unchanged"] is not True
    ):
        raise CapacityAggregateError("artifact evidence is not a verified 5090/2B artifact")


def _validate_logits_evidence(value: Any, label: str) -> None:
    logits = _mapping(value, label)
    _exact_keys(logits, _LOGITS_KEYS, label)
    shape = logits["shape"]
    if not isinstance(shape, list) or not shape:
        raise CapacityAggregateError(f"{label} shape is invalid")
    dimensions = [_integer(item, f"{label}.shape", minimum=1) for item in shape]
    numel = _integer(logits["numel"], f"{label}.numel", minimum=1)
    element_size = _integer(
        logits["element_size"], f"{label}.element_size", minimum=1
    )
    byte_count = _integer(logits["bytes"], f"{label}.bytes", minimum=1)
    if math.prod(dimensions) != numel or numel * element_size != byte_count:
        raise CapacityAggregateError(f"{label} size identity mismatch")
    if not isinstance(logits["dtype"], str) or not logits["dtype"]:
        raise CapacityAggregateError(f"{label} dtype is invalid")


def _validate_complete_step_telemetry(record: Mapping[str, Any], label: str) -> None:
    if (
        record.get("kind") != "rememr1-training-step-telemetry-v2"
        or record.get("schema_version") != 2
        or record.get("worker_roles") != ["actor", "reference"]
    ):
        raise CapacityAggregateError(f"{label} telemetry contract changed")
    _validate_logits_evidence(record.get("actor_logits"), f"{label} actor logits")
    _validate_logits_evidence(
        record.get("reference_logits"), f"{label} reference logits"
    )
    phases = record.get("phase_telemetry")
    if not isinstance(phases, list) or len(phases) != len(_TELEMETRY_PHASES):
        raise CapacityAggregateError(f"{label} telemetry phase inventory is incomplete")
    normalized_phases: list[Mapping[str, Any]] = []
    for index, (raw_phase, expected_name) in enumerate(
        zip(phases, _TELEMETRY_PHASES, strict=True)
    ):
        phase = _mapping(raw_phase, f"{label} phase {index}")
        _exact_keys(phase, _PHASE_TELEMETRY_KEYS, f"{label} phase {index}")
        if phase["name"] != expected_name:
            raise CapacityAggregateError(f"{label} telemetry phase order changed")
        duration = phase["duration_seconds"]
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or duration < 0
        ):
            raise CapacityAggregateError(f"{label} phase duration is invalid")
        for key in _PHASE_TELEMETRY_KEYS - {"duration_seconds", "name"}:
            _integer(phase[key], f"{label}.{expected_name}.{key}")
        normalized_phases.append(phase)
    if any(
        phase[key] != 0
        for phase in normalized_phases
        for key in (
            "reference_peak_allocated_bytes",
            "reference_peak_reserved_bytes",
            "reference_post_allocated_bytes",
            "reference_post_reserved_bytes",
        )
    ):
        raise CapacityAggregateError(
            f"{label} reference resource placeholders must be zero"
        )
    expected_allocated = max(
        phase["actor_peak_allocated_bytes"] for phase in normalized_phases
    )
    expected_reserved = max(
        phase["actor_peak_reserved_bytes"] for phase in normalized_phases
    )
    if record.get("peak_allocated_bytes") != expected_allocated:
        raise CapacityAggregateError(f"{label} allocated peak is not phase-derived")
    if record.get("peak_reserved_bytes") != expected_reserved:
        raise CapacityAggregateError(f"{label} reserved peak is not phase-derived")
    if record.get("nvml_peak_used_bytes", -1) < max(
        phase["nvml_peak_used_bytes"] for phase in normalized_phases
    ):
        raise CapacityAggregateError(f"{label} NVML peak is below a phase peak")
    save_phase = normalized_phases[-1]
    if record.get("post_step_allocated_bytes") != save_phase[
        "actor_post_allocated_bytes"
    ]:
        raise CapacityAggregateError(f"{label} post-step allocation is not phase-derived")
    if record.get("post_step_reserved_bytes") != save_phase[
        "actor_post_reserved_bytes"
    ]:
        raise CapacityAggregateError(f"{label} post-step reservation is not phase-derived")
    timings = _mapping(record.get("timing_seconds"), f"{label} timings")
    save_timing = timings.get("save_checkpoint")
    if (
        isinstance(save_timing, bool)
        or not isinstance(save_timing, (int, float))
        or not math.isfinite(float(save_timing))
        or save_timing < 0
    ):
        raise CapacityAggregateError(f"{label} lacks checkpoint-save timing")


def _merge_telemetry(
    attempts: Mapping[str, Mapping[str, Any]],
    *,
    identity: Mapping[str, Any],
    gpu_evidence_sha256: str,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    merged_steps: list[dict[str, Any]] = []
    scientific: list[Mapping[str, Any]] = []
    all_records: list[Mapping[str, Any]] = []
    for logical_stage, _, _, _ in _ATTEMPTS:
        ledger_steps = attempts[logical_stage]["ledger"]["steps"]
        for record in ledger_steps:
            item = dict(_mapping(record, f"{logical_stage} telemetry step"))
            _validate_complete_step_telemetry(
                item,
                f"{logical_stage} step {item.get('global_step')}",
            )
            all_records.append(item)
            item["attempt_stage"] = logical_stage
            item["step"] = item["global_step"] if logical_stage == "g2b-resume5" else None
            merged_steps.append(item)
            if logical_stage != "length-stress":
                evidence = _mapping(
                    item.get("scientific_evidence"),
                    f"{logical_stage} step {item['global_step']} scientific evidence",
                )
                _exact_keys(evidence, _SCIENTIFIC_KEYS, "scientific evidence")
                if any(value is None for value in evidence.values()):
                    raise CapacityAggregateError(
                        f"{logical_stage} step {item['global_step']} lacks scientific evidence"
                    )
                scientific.append(evidence)
    if not scientific:
        raise CapacityAggregateError("random G2 attempts produced no scientific evidence")
    gpu_uuids = {record.get("gpu_uuid") for record in all_records}
    totals = {record.get("nvml_total_bytes") for record in all_records}
    if gpu_uuids != {identity["gpu"]["uuid"]} or totals != {
        identity["gpu"]["total_vram_bytes"]
    }:
        raise CapacityAggregateError("step telemetry GPU identity drifted")
    for evidence in scientific:
        for name in (
            "all_outputs_truncated",
            "finite_gradients",
            "finite_losses",
            "high_truncation_rate",
            "systematic_format_failure",
        ):
            if type(evidence[name]) is not bool:
                raise CapacityAggregateError(f"scientific evidence {name} must be boolean")
        for name in ("nonzero_advantage_groups", "reward_variance_groups"):
            _integer(evidence[name], f"scientific evidence {name}")
    host_totals = [_integer(record["host_total_memory_bytes"], "host total memory", minimum=1) for record in all_records]
    telemetry = {
        "adapter_update_nonzero": artifact["adapter_update_nonzero"],
        "all_outputs_truncated": any(item["all_outputs_truncated"] for item in scientific),
        "allocator_fragmentation_failure": False,
        "allocator_retry_count": max(
            _integer(record["allocator_retry_count"], "allocator retry count")
            for record in all_records
        ),
        "artifact_evidence_sha256": artifact["artifact_evidence_sha256"],
        "base_model_unchanged": artifact["base_model_unchanged"],
        "budget_projection_sha256": artifact["budget_projection_sha256"],
        "finite_gradients": all(item["finite_gradients"] for item in scientific),
        "finite_losses": all(item["finite_losses"] for item in scientific),
        "fresh_process_resume_completed": True,
        "g2a_completed": True,
        "g2b_completed": True,
        "gpu_evidence_sha256": gpu_evidence_sha256,
        "high_truncation_rate": any(item["high_truncation_rate"] for item in scientific),
        "host_peak_used_bytes": max(
            _integer(record["host_peak_used_bytes"], "host peak used bytes")
            for record in all_records
        ),
        "host_total_memory_bytes": min(host_totals),
        "identity": dict(identity),
        "ledger_sha256s": {
            stage: attempts[stage]["ledger_sha256"] for stage, _, _, _ in _ATTEMPTS
        },
        "length_stress_completed": True,
        "non_capacity_failure": False,
        "nonzero_advantage_groups": sum(
            item["nonzero_advantage_groups"] for item in scientific
        ),
        "nvml_total_bytes": identity["gpu"]["total_vram_bytes"],
        "oom": False,
        "reward_variance_groups": sum(item["reward_variance_groups"] for item in scientific),
        "steps": merged_steps,
        "swap_used_bytes": max(
            _integer(record["swap_used_bytes"], "swap used bytes") for record in all_records
        ),
        "systematic_format_failure": any(
            item["systematic_format_failure"] for item in scientific
        ),
        "terminal_reason": None,
    }
    return telemetry


def _merge_capacity_stop_telemetry(
    attempts: Mapping[str, Mapping[str, Any]],
    *,
    stopped_stage: str,
    identity: Mapping[str, Any],
    hardware: Mapping[str, Any],
    gpu_evidence_sha256: str,
    artifact: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged_steps: list[dict[str, Any]] = []
    raw_records: list[Mapping[str, Any]] = []
    scientific: list[Mapping[str, Any]] = []
    scientific_complete = True
    for logical_stage, _, _, _ in _ATTEMPTS:
        if logical_stage not in attempts:
            break
        attempt = attempts[logical_stage]
        source_steps = (
            attempt["steps"]
            if attempt.get("ledger") is None
            else attempt["ledger"]["steps"]
        )
        for raw_record in source_steps:
            record = dict(_mapping(raw_record, f"{logical_stage} telemetry step"))
            raw_records.append(record)
            record["attempt_stage"] = logical_stage
            record["step"] = (
                record["global_step"] if logical_stage == "g2b-resume5" else None
            )
            merged_steps.append(record)
            if logical_stage == "length-stress":
                continue
            evidence = record.get("scientific_evidence")
            if not isinstance(evidence, Mapping) or set(evidence) != _SCIENTIFIC_KEYS:
                scientific_complete = False
                continue
            if any(item is None for item in evidence.values()):
                scientific_complete = False
                continue
            normalized = dict(evidence)
            if any(
                type(normalized[name]) is not bool
                for name in (
                    "all_outputs_truncated",
                    "finite_gradients",
                    "finite_losses",
                    "high_truncation_rate",
                    "systematic_format_failure",
                )
            ):
                scientific_complete = False
                continue
            try:
                for name in ("nonzero_advantage_groups", "reward_variance_groups"):
                    _integer(normalized[name], f"scientific evidence {name}")
            except CapacityAggregateError:
                scientific_complete = False
                continue
            scientific.append(normalized)
    if raw_records:
        if {record.get("gpu_uuid") for record in raw_records} != {
            identity["gpu"]["uuid"]
        } or {record.get("nvml_total_bytes") for record in raw_records} != {
            identity["gpu"]["total_vram_bytes"]
        }:
            raise CapacityAggregateError("capacity-stop telemetry GPU identity drifted")

    if scientific and scientific_complete:
        finite_gradients: bool | None = all(
            item["finite_gradients"] for item in scientific
        )
        finite_losses: bool | None = all(item["finite_losses"] for item in scientific)
        all_truncated: bool | None = any(
            item["all_outputs_truncated"] for item in scientific
        )
        high_truncation: bool | None = any(
            item["high_truncation_rate"] for item in scientific
        )
        format_failure: bool | None = any(
            item["systematic_format_failure"] for item in scientific
        )
        nonzero_groups: int | None = sum(
            item["nonzero_advantage_groups"] for item in scientific
        )
        variance_groups: int | None = sum(
            item["reward_variance_groups"] for item in scientific
        )
        scientific_ok = (
            finite_gradients
            and finite_losses
            and not all_truncated
            and not high_truncation
            and not format_failure
            and nonzero_groups >= 1
            and variance_groups >= 1
        )
    else:
        finite_gradients = None
        finite_losses = None
        all_truncated = None
        high_truncation = None
        format_failure = None
        nonzero_groups = None
        variance_groups = None
        scientific_ok = not scientific and scientific_complete

    if stopped_stage == "length-stress" and artifact is None:
        raise CapacityAggregateError(
            "length-stress capacity stop requires verified resume5 artifacts"
        )
    artifact_ok = artifact is None or (
        artifact.get("adapter_update_nonzero") is True
        and artifact.get("base_model_unchanged") is True
    )
    non_capacity_failure = not (scientific_ok and artifact_ok)

    def maximum(field: str) -> int | None:
        if not raw_records:
            return None
        return max(_integer(record[field], field) for record in raw_records)

    host_totals = [
        _integer(record["host_total_memory_bytes"], "host total memory", minimum=1)
        for record in raw_records
    ]
    stop_attempt = attempts[stopped_stage]
    stage_index = [item[0] for item in _ATTEMPTS].index(stopped_stage)
    budget_hashes = {
        attempt["meta"]["budget_projection_sha256"] for attempt in attempts.values()
    }
    if len(budget_hashes) != 1:
        raise CapacityAggregateError("capacity-stop attempts used different budget projections")
    budget_hash = next(iter(budget_hashes)) or None
    telemetry = {
        "adapter_update_nonzero": (
            artifact["adapter_update_nonzero"] if artifact is not None else None
        ),
        "all_outputs_truncated": all_truncated,
        "allocator_fragmentation_failure": None,
        "allocator_retry_count": maximum("allocator_retry_count"),
        "artifact_evidence": None if artifact is None else dict(artifact),
        "artifact_evidence_sha256": (
            None if artifact is None else artifact["artifact_evidence_sha256"]
        ),
        "base_model_unchanged": (
            artifact["base_model_unchanged"] if artifact is not None else None
        ),
        "budget_projection_sha256": budget_hash,
        "capacity_stop_sha256": stop_attempt["capacity_stop_sha256"],
        "finite_gradients": finite_gradients,
        "finite_losses": finite_losses,
        "fresh_process_resume_completed": stage_index > 2,
        "g2a_completed": stage_index > 0,
        "g2b_completed": stage_index > 2,
        "gpu_evidence_sha256": gpu_evidence_sha256,
        "high_truncation_rate": high_truncation,
        "host_peak_used_bytes": maximum("host_peak_used_bytes"),
        "host_total_memory_bytes": (
            min(host_totals)
            if host_totals
            else _integer(
                hardware["host_total_memory_bytes"],
                "GPU evidence host total memory",
                minimum=(80 if identity["selected_profile"] == "R0" else 128) * GIB,
            )
        ),
        "identity": dict(identity),
        "ledger_sha256s": {
            stage: attempts[stage].get("ledger_sha256") for stage in attempts
        },
        "length_stress_completed": False,
        "non_capacity_failure": non_capacity_failure,
        "nonzero_advantage_groups": nonzero_groups,
        "nvml_total_bytes": identity["gpu"]["total_vram_bytes"],
        "oom": True,
        "reward_variance_groups": variance_groups,
        "steps": merged_steps,
        "swap_used_bytes": maximum("swap_used_bytes"),
        "systematic_format_failure": format_failure,
        "terminal_reason": "oom",
    }
    return telemetry


def aggregate_capacity_stop(
    *,
    handoff_path: str | os.PathLike[str],
    index_path: str | os.PathLike[str],
    gpu_evidence_path: str | os.PathLike[str],
    profile: str,
    stopped_stage: str,
    stopped_pointer: str | os.PathLike[str],
    g2a_pointer: str | os.PathLike[str] | None = None,
    g2b_step1_pointer: str | os.PathLike[str] | None = None,
    g2b_resume5_pointer: str | os.PathLike[str] | None = None,
    handoff_verifier: Callable[[Path, str], Any] | None = None,
    index_loader: Callable[[Path, str], Any] | None = None,
    runtime_verifier: Callable[[Path], Any] | None = None,
    ledger_loader: Callable[[Path], Any] | None = None,
    artifact_builder: Callable[[Path, Path], Mapping[str, Any]] | None = None,
) -> dict[str, Mapping[str, Any]]:
    if profile not in {"r0", "r1"}:
        raise CapacityAggregateError("capacity-stop profile must be r0 or r1")
    ordered_stages = [item[0] for item in _ATTEMPTS]
    if stopped_stage not in ordered_stages:
        raise CapacityAggregateError("stopped_stage is not a G2 capacity stage")
    stop_index = ordered_stages.index(stopped_stage)
    supplied_success = {
        "g2a": g2a_pointer,
        "g2b-step1": g2b_step1_pointer,
        "g2b-resume5": g2b_resume5_pointer,
    }
    expected_previous = set(ordered_stages[:stop_index])
    actual_previous = {stage for stage, pointer in supplied_success.items() if pointer is not None}
    if actual_previous != expected_previous:
        raise CapacityAggregateError(
            "capacity-stop finalization requires exactly the successful predecessor pointers"
        )

    _, handoff = _load_handoff(handoff_path, handoff_verifier)
    index_file, index, index_digest = _load_index(index_path, handoff, index_loader)
    configs = _mapping(index["configs"], "resolved config index.configs")
    selected = {
        config_id: configs[config_id]["sha256"]
        for config_id in capacity_evidence.expected_config_ids(profile.upper())
    }
    gpu, gpu_digest, hardware = _load_gpu_evidence(gpu_evidence_path, profile)
    stop_spec = _ATTEMPTS[stop_index]
    _, stopped_run, pipeline_dir = _resolve_stopped_pointer(
        stopped_pointer, stop_spec[1], profile
    )
    attempts: dict[str, dict[str, Any]] = {}
    run_root = stopped_run.parent
    for logical, runtime_stage, expected_pointer, expected_steps in _ATTEMPTS[:stop_index]:
        pointer_value = supplied_success[logical]
        assert pointer_value is not None
        pointer, run_dir, observed_pipeline = _resolve_pointer(
            pointer_value, expected_pointer
        )
        if (
            pointer.parents[1].name != profile
            or observed_pipeline != pipeline_dir
            or run_dir.parent != run_root
        ):
            raise CapacityAggregateError(
                "capacity-stop predecessor pointers escaped their pipeline/run root"
            )
        config_id = f"{capacity_evidence.ATTEMPT_CONFIG_TASK[logical]}_{profile}"
        attempts[logical] = _validate_attempt(
            logical_stage=logical,
            runtime_stage=runtime_stage,
            pointer_path=pointer,
            run_dir=run_dir,
            pipeline_dir=pipeline_dir,
            config_id=config_id,
            config_sha256=selected[config_id],
            index_path=index_file,
            index_sha256=index_digest,
            commit=handoff["git_commit"],
            profile=profile,
            gpu_uuid=gpu["uuid"],
            expected_steps=expected_steps,
            runtime_verifier=runtime_verifier,
            ledger_loader=ledger_loader,
        )
    logical, runtime_stage, _, expected_steps = stop_spec
    stopped_config_id = f"{capacity_evidence.ATTEMPT_CONFIG_TASK[logical]}_{profile}"
    attempts[logical] = _validate_stopped_attempt(
        logical_stage=logical,
        runtime_stage=runtime_stage,
        run_dir=stopped_run,
        pipeline_dir=pipeline_dir,
        config_id=stopped_config_id,
        config_sha256=selected[stopped_config_id],
        index_path=index_file,
        index_sha256=index_digest,
        commit=handoff["git_commit"],
        profile=profile,
        gpu_uuid=gpu["uuid"],
        expected_steps=expected_steps,
        runtime_verifier=runtime_verifier,
        ledger_loader=ledger_loader,
    )
    if len({attempt["run_dir"] for attempt in attempts.values()}) != len(attempts):
        raise CapacityAggregateError("capacity-stop chain reuses an attempt directory")

    for stage, attempt in attempts.items():
        predecessor = attempt["runtime"].get("resume_predecessor")
        if stage == "g2b-resume5":
            step1 = attempts.get("g2b-step1")
            if step1 is None:
                raise CapacityAggregateError("resume5 capacity stop lacks step1 predecessor")
            predecessor = _mapping(predecessor, "resume5 runtime predecessor")
            predecessor_id = f"g2b_qwen35_2b_5090_step1_{profile}"
            if (
                predecessor.get("logical_config_id") != predecessor_id
                or predecessor.get("resolved_config_sha256") != selected[predecessor_id]
                or predecessor.get("checkpoint_step") != 1
                or predecessor.get("checkpoint_dir")
                != str(step1["run_dir"] / "checkpoints" / "global_step_1")
            ):
                raise CapacityAggregateError(
                    "resume5 capacity-stop predecessor identity drifted"
                )
        elif predecessor is not None:
            raise CapacityAggregateError(
                f"{stage} capacity-stop chain unexpectedly binds a resume predecessor"
            )

    artifact: Mapping[str, Any] | None = None
    if stopped_stage == "length-stress":
        resume = attempts["g2b-resume5"]
        checkpoint = resume["run_dir"] / "checkpoints" / "global_step_5"
        adapter = (
            resume["run_dir"]
            / "artifacts"
            / "adapter"
            / "global_step_5"
            / "adapter"
        )
        artifact = (
            artifact_builder(checkpoint, adapter)
            if artifact_builder is not None
            else create_artifact_evidence(
                checkpoint,
                adapter,
                runtime_verifier=runtime_verifier,
            )
        )
        artifact = _mapping(artifact, "resume5 artifact evidence")
        _validate_artifact_schema(artifact)
        resume_id = f"g2b_qwen35_2b_5090_resume5_{profile}"
        if (
            artifact["attempt_id"] != resume["attempt_id"]
            or artifact["commit"] != handoff["git_commit"]
            or artifact["pipeline_dir"] != str(pipeline_dir)
            or artifact["offload_profile"] != profile
            or artifact["config_id"] != resume_id
            or artifact["config_sha256"] != selected[resume_id]
        ):
            raise CapacityAggregateError(
                "capacity-stop artifact evidence is not bound to resume5"
            )

    identity = capacity_evidence.build_identity(
        commit=handoff["git_commit"],
        cpu_handoff_sha256=handoff["handoff_sha256"],
        active_config_tree_sha256=handoff["config_tree_sha256"],
        selected_profile=profile.upper(),
        selected_configs=selected,
        gpu=gpu,
    )
    attempt_values: dict[str, Any] = {}
    previous_stage: str | None = None
    previous_digest: str | None = None
    for stage in ordered_stages[: stop_index + 1]:
        attempt = attempts[stage]
        config_id = f"{capacity_evidence.ATTEMPT_CONFIG_TASK[stage]}_{profile}"
        is_stopped = stage == stopped_stage
        attempt_values[stage] = {
            "attempt_id": attempt["attempt_id"],
            "config_id": config_id,
            "config_sha256": selected[config_id],
            "digest": attempt["digest"],
            "failure_kind": "capacity" if is_stopped else None,
            "predecessor": (
                None
                if previous_stage is None
                else {"digest": previous_digest, "stage": previous_stage}
            ),
            "status": "failed" if is_stopped else "success",
        }
        previous_stage, previous_digest = stage, attempt["digest"]
    attempt_metadata = {"attempts": attempt_values, "identity": identity}
    capacity_evidence.validate_attempt_metadata(
        attempt_metadata,
        identity=identity,
        selected_configs=selected,
        require_complete_success=False,
    )
    telemetry = _merge_capacity_stop_telemetry(
        attempts,
        stopped_stage=stopped_stage,
        identity=identity,
        hardware=hardware,
        gpu_evidence_sha256=gpu_digest,
        artifact=artifact,
    )
    evidence = capacity_evidence.create_capacity_evidence(
        identity=identity,
        telemetry=telemetry,
        attempt_metadata=attempt_metadata,
        selected_configs=selected,
    )
    if profile == "r0":
        if telemetry["non_capacity_failure"] is False and (
            evidence["classification"]["r1_eligible"] is not True
            or evidence["classification"]["eligibility_reason"] != "oom"
        ):
            raise CapacityAggregateError(
                "trusted R0 capacity stop did not classify as R1-eligible OOM"
            )
        if telemetry["non_capacity_failure"] is True and evidence["classification"][
            "r1_eligible"
        ] is True:
            raise CapacityAggregateError("non-capacity evidence incorrectly authorized R1")
    elif (
        evidence["classification"]["r1_eligible"] is not False
        or evidence["classification"]["eligibility_reason"] is not None
    ):
        raise CapacityAggregateError("R1 capacity stop must never authorize another profile")
    return {
        "attempt-metadata.json": attempt_metadata,
        "identity.json": identity,
        "selected-configs.json": selected,
        "telemetry.json": telemetry,
    }


def aggregate_capacity(
    *,
    handoff_path: str | os.PathLike[str],
    index_path: str | os.PathLike[str],
    gpu_evidence_path: str | os.PathLike[str],
    profile: str,
    g2a_pointer: str | os.PathLike[str],
    g2b_step1_pointer: str | os.PathLike[str],
    g2b_resume5_pointer: str | os.PathLike[str],
    length_stress_pointer: str | os.PathLike[str],
    artifact_evidence_path: str | os.PathLike[str],
    handoff_verifier: Callable[[Path, str], Any] | None = None,
    index_loader: Callable[[Path, str], Any] | None = None,
    runtime_verifier: Callable[[Path], Any] | None = None,
    ledger_loader: Callable[[Path], Any] | None = None,
    artifact_revalidator: Callable[[Path, Path], Mapping[str, Any]] | None = None,
) -> dict[str, Mapping[str, Any]]:
    if profile not in {"r0", "r1"}:
        raise CapacityAggregateError("profile must be r0 or r1")
    handoff_file, handoff = _load_handoff(handoff_path, handoff_verifier)
    index_file, index, index_digest = _load_index(index_path, handoff, index_loader)
    configs = _mapping(index["configs"], "resolved config index.configs")
    selected = {
        config_id: configs[config_id]["sha256"]
        for config_id in capacity_evidence.expected_config_ids(profile.upper())
    }
    gpu, gpu_digest, _ = _load_gpu_evidence(gpu_evidence_path, profile)
    pointers = (g2a_pointer, g2b_step1_pointer, g2b_resume5_pointer, length_stress_pointer)
    resolved: dict[str, tuple[Path, Path]] = {}
    pipeline_dir: Path | None = None
    run_root: Path | None = None
    for (logical, _, expected_pointer, _), pointer_value in zip(_ATTEMPTS, pointers):
        pointer, run_dir, observed_pipeline = _resolve_pointer(pointer_value, expected_pointer)
        if pointer.parents[1].name != profile:
            raise CapacityAggregateError("G2 pointer scope differs from selected profile")
        if pipeline_dir is None:
            pipeline_dir = observed_pipeline
            run_root = run_dir.parent
        elif observed_pipeline != pipeline_dir or run_dir.parent != run_root:
            raise CapacityAggregateError("G2 pointers do not share one pipeline/run root")
        if run_dir in {item[1] for item in resolved.values()}:
            raise CapacityAggregateError("G2 pointers reuse an attempt directory")
        resolved[logical] = (pointer, run_dir)
    assert pipeline_dir is not None
    attempts: dict[str, dict[str, Any]] = {}
    for logical, runtime_stage, _, expected_steps in _ATTEMPTS:
        config_id = f"{capacity_evidence.ATTEMPT_CONFIG_TASK[logical]}_{profile}"
        pointer, run_dir = resolved[logical]
        attempts[logical] = _validate_attempt(
            logical_stage=logical,
            runtime_stage=runtime_stage,
            pointer_path=pointer,
            run_dir=run_dir,
            pipeline_dir=pipeline_dir,
            config_id=config_id,
            config_sha256=selected[config_id],
            index_path=index_file,
            index_sha256=index_digest,
            commit=handoff["git_commit"],
            profile=profile,
            gpu_uuid=gpu["uuid"],
            expected_steps=expected_steps,
            runtime_verifier=runtime_verifier,
            ledger_loader=ledger_loader,
        )
    budget_hashes = {
        attempts[stage]["meta"]["budget_projection_sha256"]
        for stage, _, _, _ in _ATTEMPTS
    }
    if len(budget_hashes) != 1:
        raise CapacityAggregateError("G2 attempts used different budget projections")
    budget_hash = next(iter(budget_hashes)) or None
    resume_predecessor = _mapping(
        attempts["g2b-resume5"]["runtime"].get("resume_predecessor"),
        "resume5 runtime predecessor",
    )
    step1_run = attempts["g2b-step1"]["run_dir"]
    predecessor_id = f"g2b_qwen35_2b_5090_step1_{profile}"
    expected_checkpoint = step1_run / "checkpoints" / "global_step_1"
    if (
        resume_predecessor.get("logical_config_id") != predecessor_id
        or resume_predecessor.get("resolved_config_sha256") != selected[predecessor_id]
        or resume_predecessor.get("checkpoint_step") != 1
        or resume_predecessor.get("checkpoint_dir") != str(expected_checkpoint)
    ):
        raise CapacityAggregateError("resume5 does not bind the verified step1 attempt")
    for stage in ("g2a", "g2b-step1", "length-stress"):
        if attempts[stage]["runtime"].get("resume_predecessor") is not None:
            raise CapacityAggregateError(f"{stage} unexpectedly binds a resume predecessor")
    artifact_file, artifact = _load_json(
        artifact_evidence_path, "G2 artifact evidence", canonical=True
    )
    del artifact_file
    _validate_artifact_schema(artifact)
    resume_run = attempts["g2b-resume5"]["run_dir"]
    artifact_checkpoint = _safe_path(
        _mapping(artifact["checkpoint"], "artifact checkpoint")["path"],
        "artifact checkpoint path",
        exists=True,
        kind="directory",
    )
    artifact_adapter = _safe_path(
        _mapping(artifact["adapter"], "artifact adapter")["path"],
        "artifact adapter path",
        exists=True,
        kind="directory",
    )
    recreated = (
        artifact_revalidator(artifact_checkpoint, artifact_adapter)
        if artifact_revalidator is not None
        else create_artifact_evidence(
            artifact_checkpoint,
            artifact_adapter,
            runtime_verifier=runtime_verifier,
        )
    )
    if dict(recreated) != dict(artifact):
        raise CapacityAggregateError("G2 artifact evidence changed during revalidation")
    predecessor = _mapping(artifact["predecessor"], "artifact predecessor")
    if (
        artifact["attempt_id"] != resume_run.name
        or artifact["commit"] != handoff["git_commit"]
        or artifact["pipeline_dir"] != str(pipeline_dir)
        or artifact["offload_profile"] != profile
        or artifact["budget_projection_sha256"] != budget_hash
        or artifact["config_id"] != f"g2b_qwen35_2b_5090_resume5_{profile}"
        or artifact["config_sha256"]
        != selected[f"g2b_qwen35_2b_5090_resume5_{profile}"]
        or artifact_checkpoint != resume_run / "checkpoints" / "global_step_5"
        or predecessor.get("attempt_id") != step1_run.name
        or predecessor.get("config_id") != predecessor_id
        or predecessor.get("config_sha256") != selected[predecessor_id]
        or predecessor.get("checkpoint_path") != str(expected_checkpoint)
    ):
        raise CapacityAggregateError("G2 artifact evidence is not bound to verified attempts")
    identity = capacity_evidence.build_identity(
        commit=handoff["git_commit"],
        cpu_handoff_sha256=handoff["handoff_sha256"],
        active_config_tree_sha256=handoff["config_tree_sha256"],
        selected_profile=profile.upper(),
        selected_configs=selected,
        gpu=gpu,
    )
    attempt_values: dict[str, Any] = {}
    previous_stage: str | None = None
    previous_digest: str | None = None
    for logical, _, _, _ in _ATTEMPTS:
        config_id = f"{capacity_evidence.ATTEMPT_CONFIG_TASK[logical]}_{profile}"
        current = attempts[logical]
        attempt_values[logical] = {
            "attempt_id": current["attempt_id"],
            "config_id": config_id,
            "config_sha256": selected[config_id],
            "digest": current["digest"],
            "failure_kind": None,
            "predecessor": (
                None
                if previous_stage is None
                else {"digest": previous_digest, "stage": previous_stage}
            ),
            "status": "success",
        }
        previous_stage, previous_digest = logical, current["digest"]
    attempt_metadata = {"attempts": attempt_values, "identity": identity}
    capacity_evidence.validate_attempt_metadata(
        attempt_metadata,
        identity=identity,
        selected_configs=selected,
        require_complete_success=True,
    )
    telemetry = _merge_telemetry(
        attempts,
        identity=identity,
        gpu_evidence_sha256=gpu_digest,
        artifact=artifact,
    )
    # This is a compatibility/self-validation call only; capacity sealing remains
    # the responsibility of capacity_evidence.py in the next pipeline stage.
    capacity_evidence.create_capacity_evidence(
        identity=identity,
        telemetry=telemetry,
        attempt_metadata=attempt_metadata,
        selected_configs=selected,
    )
    del handoff_file
    return {
        "attempt-metadata.json": attempt_metadata,
        "identity.json": identity,
        "selected-configs.json": selected,
        "telemetry.json": telemetry,
    }


def _atomic_publish(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    destination = _safe_path(path, "evidence output", exists=False)
    encoded = _canonical_bytes(payload) + b"\n"
    if destination.exists():
        if destination.is_symlink() or not destination.is_file() or destination.read_bytes() != encoded:
            raise CapacityAggregateError(f"refusing to replace immutable evidence {destination}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-", dir=str(destination.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if destination.exists() or destination.is_symlink():
            raise CapacityAggregateError(f"evidence output appeared concurrently: {destination}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    artifacts = commands.add_parser("artifacts", help="verify the G2b resume5 checkpoint/adapter")
    artifacts.add_argument("--checkpoint-dir", type=Path, required=True)
    artifacts.add_argument("--adapter-dir", type=Path, required=True)
    artifacts.add_argument("--output", type=Path, required=True)
    target = commands.add_parser(
        "target-identity", help="build a pre-run R0/R1 identity from sealed inputs"
    )
    target.add_argument("--handoff", type=Path, required=True)
    target.add_argument("--index", type=Path, required=True)
    target.add_argument("--gpu-evidence", type=Path, required=True)
    target.add_argument("--profile", choices=("r0", "r1"), required=True)
    target.add_argument("--output", type=Path, required=True)
    stopped = commands.add_parser(
        "capacity-stop",
        help="finalize a trusted R0 CUDA-OOM stop without sealing a capacity profile",
    )
    stopped.add_argument("--handoff", type=Path, required=True)
    stopped.add_argument("--index", type=Path, required=True)
    stopped.add_argument("--gpu-evidence", type=Path, required=True)
    stopped.add_argument("--profile", choices=("r0", "r1"), required=True)
    stopped.add_argument(
        "--stopped-stage",
        choices=("g2a", "g2b-step1", "g2b-resume5", "g2-length-stress"),
        required=True,
    )
    stopped.add_argument("--stopped-pointer", type=Path, required=True)
    stopped.add_argument("--g2a-pointer", type=Path)
    stopped.add_argument("--g2b-step1-pointer", type=Path)
    stopped.add_argument("--g2b-resume5-pointer", type=Path)
    stopped.add_argument("--output-dir", type=Path, required=True)
    aggregate = commands.add_parser("aggregate", help="verify and aggregate four G2 attempts")
    aggregate.add_argument("--handoff", type=Path, required=True)
    aggregate.add_argument("--index", type=Path, required=True)
    aggregate.add_argument("--gpu-evidence", type=Path, required=True)
    aggregate.add_argument("--profile", choices=("r0", "r1"), required=True)
    aggregate.add_argument("--g2a-pointer", type=Path, required=True)
    aggregate.add_argument("--g2b-step1-pointer", type=Path, required=True)
    aggregate.add_argument("--g2b-resume5-pointer", type=Path, required=True)
    aggregate.add_argument("--length-stress-pointer", type=Path, required=True)
    aggregate.add_argument("--artifact-evidence", type=Path, required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "artifacts":
            result = create_artifact_evidence(args.checkpoint_dir, args.adapter_dir)
            _atomic_publish(args.output, result)
            message = {"sha256": result["artifact_evidence_sha256"], "status": "verified"}
        elif args.command == "target-identity":
            result = build_target_identity(
                handoff_path=args.handoff,
                index_path=args.index,
                gpu_evidence_path=args.gpu_evidence,
                profile=args.profile,
            )
            _atomic_publish(args.output, result)
            message = {
                "selected_config_set_sha256": result["selected_config_set_sha256"],
                "selected_profile": result["selected_profile"],
                "status": "target-ready",
            }
        elif args.command == "capacity-stop":
            outputs = aggregate_capacity_stop(
                handoff_path=args.handoff,
                index_path=args.index,
                gpu_evidence_path=args.gpu_evidence,
                profile=args.profile,
                stopped_stage=(
                    "length-stress"
                    if args.stopped_stage == "g2-length-stress"
                    else args.stopped_stage
                ),
                stopped_pointer=args.stopped_pointer,
                g2a_pointer=args.g2a_pointer,
                g2b_step1_pointer=args.g2b_step1_pointer,
                g2b_resume5_pointer=args.g2b_resume5_pointer,
            )
            output_dir = args.output_dir.expanduser()
            if not output_dir.is_absolute():
                raise CapacityAggregateError("capacity-stop output directory must be absolute")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_dir = _safe_path(
                output_dir, "capacity-stop output directory", exists=True, kind="directory"
            )
            for filename, payload in outputs.items():
                _atomic_publish(output_dir / filename, payload)
            message = {"files": sorted(outputs), "status": "capacity-stop-finalized"}
        else:
            outputs = aggregate_capacity(
                handoff_path=args.handoff,
                index_path=args.index,
                gpu_evidence_path=args.gpu_evidence,
                profile=args.profile,
                g2a_pointer=args.g2a_pointer,
                g2b_step1_pointer=args.g2b_step1_pointer,
                g2b_resume5_pointer=args.g2b_resume5_pointer,
                length_stress_pointer=args.length_stress_pointer,
                artifact_evidence_path=args.artifact_evidence,
            )
            output_dir = _safe_path(
                args.output_dir, "capacity output directory", exists=True, kind="directory"
            )
            for filename, payload in outputs.items():
                _atomic_publish(output_dir / filename, payload)
            message = {"files": sorted(outputs), "status": "aggregated"}
    except Exception as exc:
        print(
            json.dumps(
                {"error": f"{type(exc).__name__}: {exc}", "status": "blocked"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(message, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_KIND",
    "CapacityAggregateError",
    "aggregate_capacity",
    "aggregate_capacity_stop",
    "build_target_identity",
    "create_artifact_evidence",
    "main",
]
