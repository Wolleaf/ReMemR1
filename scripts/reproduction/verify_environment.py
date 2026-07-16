"""Validate the pinned reproduction environment and fail closed on GPU kernels."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = 1
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_EXACT_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+_-]*$")
_REQUIRED_PACKAGES = {
    "datasets",
    "huggingface-hub",
    "hydra-core",
    "omegaconf",
    "peft",
    "ray",
    "tensordict",
    "torch",
    "torchdata",
    "transformers",
}
_EVIDENCE_FIELDS = {
    "bf16_backward_log_sha256",
    "bf16_forward_log_sha256",
    "build_log_sha256",
    "optimizer_loop_log_sha256",
}


class EnvironmentContractError(RuntimeError):
    """Raised when an environment or kernel gate cannot be proven safe."""


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _without_digest(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    content = dict(payload)
    content.pop(field, None)
    return content


def seal_payload(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    sealed = json.loads(json.dumps(payload))
    sealed[field] = canonical_json_sha256(_without_digest(sealed, field))
    return sealed


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        raise EnvironmentContractError(
            f"{path} keys mismatch; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def _safe_relative_path(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise EnvironmentContractError(f"{path} must be a non-empty POSIX-relative path")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise EnvironmentContractError(f"{path} is unsafe")
    return value


def normalized_text_sha256(path: str | Path) -> str:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise EnvironmentContractError(f"cannot read pinned requirements {path}: {exc}") from exc
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _parse_requirements(path: Path) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    packages: dict[str, str] = {}
    vcs: dict[str, tuple[str, str]] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        if " @ git+" in line:
            name_part, source = line.split(" @ git+", 1)
            distribution = name_part.split("[", 1)[0].strip().lower()
            match = re.fullmatch(r"(.+\.git)@([0-9a-f]{40})", source)
            if not match:
                raise EnvironmentContractError(f"{path}:{line_number} has an unpinned VCS requirement")
            if distribution in vcs:
                raise EnvironmentContractError(f"duplicate VCS requirement {distribution!r}")
            vcs[distribution] = (match.group(1), match.group(2))
            continue
        if "==" not in line:
            raise EnvironmentContractError(f"{path}:{line_number} is not an exact package pin")
        name, version = (part.strip() for part in line.split("==", 1))
        name = name.lower()
        if not name or not _EXACT_VERSION.fullmatch(version):
            raise EnvironmentContractError(f"{path}:{line_number} has a malformed package pin")
        if name in packages:
            raise EnvironmentContractError(f"duplicate package requirement {name!r}")
        packages[name] = version
    return packages, vcs


def validate_environment_lock(lock: Mapping[str, Any], *, repository_root: str | Path) -> None:
    if not isinstance(lock, Mapping):
        raise EnvironmentContractError("environment lock must be a JSON object")
    _exact_keys(
        lock,
        {
            "kernels",
            "lock_sha256",
            "packages",
            "platform",
            "requirements",
            "schema_version",
            "transitive_freeze",
        },
        "lock",
    )
    if lock["schema_version"] != SCHEMA_VERSION:
        raise EnvironmentContractError(f"unsupported lock schema_version={lock['schema_version']!r}")
    if not _is_sha256(lock["lock_sha256"]):
        raise EnvironmentContractError("lock_sha256 is malformed")
    expected_lock_digest = canonical_json_sha256(_without_digest(lock, "lock_sha256"))
    if lock["lock_sha256"] != expected_lock_digest:
        raise EnvironmentContractError("lock_sha256 does not match the canonical lock payload")

    platform_info = lock["platform"]
    if not isinstance(platform_info, Mapping):
        raise EnvironmentContractError("platform must be an object")
    _exact_keys(
        platform_info,
        {"cuda_toolkit", "gpu_compute_capability", "operating_system", "python", "torch_index_url"},
        "platform",
    )
    if not re.fullmatch(r"3\.12\.\d+", str(platform_info["python"])):
        raise EnvironmentContractError("platform.python must be an exact Python 3.12 patch version")
    if platform_info["cuda_toolkit"] != "13.0":
        raise EnvironmentContractError("the formal environment must remain pinned to CUDA 13.0")
    if platform_info["gpu_compute_capability"] != [12, 0]:
        raise EnvironmentContractError("the formal GPU gate must target compute capability [12, 0]")
    if platform_info["torch_index_url"] != "https://download.pytorch.org/whl/cu130":
        raise EnvironmentContractError("the Torch wheel index must remain pinned to cu130")

    package_entries = lock["packages"]
    if not isinstance(package_entries, list):
        raise EnvironmentContractError("packages must be a list")
    package_pins: dict[str, str] = {}
    for index, entry in enumerate(package_entries):
        if not isinstance(entry, Mapping):
            raise EnvironmentContractError(f"packages[{index}] must be an object")
        _exact_keys(entry, {"name", "version"}, f"packages[{index}]")
        name = entry["name"]
        version = entry["version"]
        if not isinstance(name, str) or name.lower() != name or not _EXACT_VERSION.fullmatch(str(version)):
            raise EnvironmentContractError(f"packages[{index}] is not an exact normalized pin")
        if name in package_pins:
            raise EnvironmentContractError(f"duplicate package pin {name!r}")
        package_pins[name] = version
    if set(package_pins) != _REQUIRED_PACKAGES:
        raise EnvironmentContractError(
            f"critical package pins mismatch; missing={sorted(_REQUIRED_PACKAGES - set(package_pins))}, "
            f"extra={sorted(set(package_pins) - _REQUIRED_PACKAGES)}"
        )

    root = Path(repository_root).resolve()
    requirement_entries = lock["requirements"]
    if not isinstance(requirement_entries, list) or len(requirement_entries) != 2:
        raise EnvironmentContractError("requirements must contain the core and kernel pin files")
    parsed_packages: dict[str, str] = {}
    parsed_vcs: dict[str, tuple[str, str]] = {}
    for index, entry in enumerate(requirement_entries):
        if not isinstance(entry, Mapping):
            raise EnvironmentContractError(f"requirements[{index}] must be an object")
        _exact_keys(entry, {"normalized_content_sha256", "path"}, f"requirements[{index}]")
        relative_path = _safe_relative_path(entry["path"], f"requirements[{index}].path")
        expected_digest = entry["normalized_content_sha256"]
        if not _is_sha256(expected_digest):
            raise EnvironmentContractError(f"requirements[{index}].normalized_content_sha256 is malformed")
        requirement_path = (root / Path(*PurePosixPath(relative_path).parts)).resolve()
        if requirement_path != root and root not in requirement_path.parents:
            raise EnvironmentContractError(f"requirements[{index}].path escapes the repository")
        if normalized_text_sha256(requirement_path) != expected_digest:
            raise EnvironmentContractError(f"requirements file hash mismatch: {relative_path}")
        packages, vcs = _parse_requirements(requirement_path)
        overlap = set(parsed_packages) & set(packages)
        vcs_overlap = set(parsed_vcs) & set(vcs)
        if overlap or vcs_overlap:
            raise EnvironmentContractError(f"duplicate requirements across files: {sorted(overlap | vcs_overlap)}")
        parsed_packages.update(packages)
        parsed_vcs.update(vcs)
    if parsed_packages != package_pins:
        raise EnvironmentContractError("requirements package pins do not match the environment lock")

    kernels = lock["kernels"]
    if not isinstance(kernels, list) or len(kernels) != 2:
        raise EnvironmentContractError("exactly two kernel sources must be pinned")
    kernel_names: set[str] = set()
    for index, kernel in enumerate(kernels):
        if not isinstance(kernel, Mapping):
            raise EnvironmentContractError(f"kernels[{index}] must be an object")
        _exact_keys(
            kernel,
            {
                "commit",
                "distribution",
                "name",
                "repository",
                "required_evidence",
                "training_gate",
                "verification_status",
            },
            f"kernels[{index}]",
        )
        name = kernel["name"]
        distribution = kernel["distribution"]
        if name not in {"causal-conv1d", "flash-linear-attention"} or name in kernel_names:
            raise EnvironmentContractError(f"kernels[{index}].name is invalid or duplicated")
        kernel_names.add(name)
        if distribution != name:
            raise EnvironmentContractError(f"kernels[{index}].distribution must match its pinned name")
        if not isinstance(kernel["commit"], str) or not _HEX40.fullmatch(kernel["commit"]):
            raise EnvironmentContractError(f"kernels[{index}].commit is malformed")
        if kernel["verification_status"] != "UNVERIFIED" or kernel["training_gate"] != "BLOCKED":
            raise EnvironmentContractError(
                f"kernels[{index}] must remain explicitly UNVERIFIED/BLOCKED in the pre-GPU lock"
            )
        if set(kernel["required_evidence"]) != _EVIDENCE_FIELDS:
            raise EnvironmentContractError(f"kernels[{index}].required_evidence is incomplete")
        vcs_pin = parsed_vcs.get(distribution)
        expected_vcs = (kernel["repository"], kernel["commit"])
        if vcs_pin != expected_vcs:
            raise EnvironmentContractError(f"kernel requirement does not match lock for {name}")

    freeze = lock["transitive_freeze"]
    if not isinstance(freeze, Mapping):
        raise EnvironmentContractError("transitive_freeze must be an object")
    _exact_keys(freeze, {"artifact", "status"}, "transitive_freeze")
    if freeze != {"artifact": "pip-freeze.txt", "status": "REQUIRED_AFTER_GPU_INSTALL"}:
        raise EnvironmentContractError("transitive freeze must remain explicitly required until GPU installation")


def load_environment_lock(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    try:
        lock = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EnvironmentContractError(f"cannot read environment lock {path}: {exc}") from exc
    validate_environment_lock(lock, repository_root=repository_root)
    return lock


def validate_build_info(build_info: Mapping[str, Any], lock: Mapping[str, Any]) -> None:
    if not isinstance(build_info, Mapping):
        raise EnvironmentContractError("build info must be a JSON object")
    _exact_keys(
        build_info,
        {
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
        },
        "build_info",
    )
    if build_info["schema_version"] != SCHEMA_VERSION:
        raise EnvironmentContractError("build info schema version mismatch")
    if not _is_sha256(build_info["build_info_sha256"]):
        raise EnvironmentContractError("build_info_sha256 is malformed")
    expected_digest = canonical_json_sha256(_without_digest(build_info, "build_info_sha256"))
    if build_info["build_info_sha256"] != expected_digest:
        raise EnvironmentContractError("build_info_sha256 does not match the canonical payload")
    if build_info["environment_lock_sha256"] != lock["lock_sha256"]:
        raise EnvironmentContractError("build info was not produced from the pinned environment lock")

    if build_info["status"] != "VERIFIED_SM120" or build_info["training_gate"] != "READY":
        raise EnvironmentContractError("environment remains UNVERIFIED; long GPU runs are blocked")
    if build_info["python"] != lock["platform"]["python"]:
        raise EnvironmentContractError("build Python version does not match the lock")
    if not _is_sha256(build_info["pip_freeze_sha256"]):
        raise EnvironmentContractError("a hashed pip-freeze.txt artifact is required")

    expected_packages = {entry["name"]: entry["version"] for entry in lock["packages"]}
    if build_info["packages"] != expected_packages:
        raise EnvironmentContractError("build package versions do not match the lock")

    system = build_info["system"]
    if not isinstance(system, Mapping):
        raise EnvironmentContractError("build_info.system must be an object")
    _exact_keys(
        system,
        {"cuda_runtime", "driver_version", "gpu_compute_capability", "gpu_name", "operating_system"},
        "build_info.system",
    )
    if system["cuda_runtime"] != "13.0" or system["gpu_compute_capability"] != [12, 0]:
        raise EnvironmentContractError("the build was not verified on CUDA 13.0 / sm_120")
    for key in ("driver_version", "gpu_name", "operating_system"):
        value = system[key]
        if not isinstance(value, str) or not value.strip() or value == "UNVERIFIED":
            raise EnvironmentContractError(f"build_info.system.{key} has not been captured")

    kernel_info = build_info["kernels"]
    expected_kernels = {kernel["name"]: kernel for kernel in lock["kernels"]}
    if not isinstance(kernel_info, Mapping) or set(kernel_info) != set(expected_kernels):
        raise EnvironmentContractError("build kernel records do not match the lock")
    for name, pinned in expected_kernels.items():
        record = kernel_info[name]
        if not isinstance(record, Mapping):
            raise EnvironmentContractError(f"build_info.kernels.{name} must be an object")
        _exact_keys(
            record,
            {"commit", "repository", "status"} | _EVIDENCE_FIELDS,
            f"build_info.kernels.{name}",
        )
        if record["repository"] != pinned["repository"] or record["commit"] != pinned["commit"]:
            raise EnvironmentContractError(f"build_info.kernels.{name} source does not match the lock")
        if record["status"] != "VERIFIED_SM120":
            raise EnvironmentContractError(f"build_info.kernels.{name} remains UNVERIFIED")
        for evidence_field in _EVIDENCE_FIELDS:
            if not _is_sha256(record[evidence_field]):
                raise EnvironmentContractError(
                    f"build_info.kernels.{name}.{evidence_field} requires a real evidence SHA-256"
                )


def verify_runtime_versions(
    lock: Mapping[str, Any],
    *,
    python_version: str | None = None,
    version_getter: Callable[[str], str] = importlib.metadata.version,
) -> None:
    observed_python = python_version or platform.python_version()
    if observed_python != lock["platform"]["python"]:
        raise EnvironmentContractError(
            f"Python version mismatch: expected {lock['platform']['python']}, observed {observed_python}"
        )
    for package in lock["packages"]:
        try:
            observed = version_getter(package["name"])
        except importlib.metadata.PackageNotFoundError as exc:
            raise EnvironmentContractError(f"required package is not installed: {package['name']}") from exc
        if observed != package["version"]:
            raise EnvironmentContractError(
                f"{package['name']} version mismatch: expected {package['version']}, observed {observed}"
            )


def _installed_direct_url(distribution: str) -> Mapping[str, Any]:
    try:
        direct_url_text = importlib.metadata.distribution(distribution).read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError as exc:
        raise EnvironmentContractError(f"kernel distribution is not installed: {distribution}") from exc
    if not direct_url_text:
        raise EnvironmentContractError(f"kernel distribution lacks direct_url.json: {distribution}")
    try:
        return json.loads(direct_url_text)
    except json.JSONDecodeError as exc:
        raise EnvironmentContractError(f"invalid direct_url.json for {distribution}") from exc


def verify_kernel_install_sources(
    lock: Mapping[str, Any],
    *,
    direct_url_getter: Callable[[str], Mapping[str, Any]] = _installed_direct_url,
) -> None:
    for kernel in lock["kernels"]:
        direct_url = direct_url_getter(kernel["distribution"])
        vcs_info = direct_url.get("vcs_info") if isinstance(direct_url, Mapping) else None
        if not isinstance(vcs_info, Mapping) or vcs_info.get("vcs") != "git":
            raise EnvironmentContractError(f"{kernel['name']} was not installed from the pinned Git source")
        observed_url = str(direct_url.get("url", "")).rstrip("/")
        expected_url = kernel["repository"].rstrip("/")
        if observed_url != expected_url or vcs_info.get("commit_id") != kernel["commit"]:
            raise EnvironmentContractError(f"{kernel['name']} installed source/commit does not match the lock")


def probe_torch_hardware() -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise EnvironmentContractError("Torch is not installed") from exc
    if not torch.cuda.is_available():
        raise EnvironmentContractError("CUDA is unavailable; sm_120 verification cannot pass")
    return {
        "cuda_runtime": str(torch.version.cuda),
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_name": torch.cuda.get_device_name(0),
    }


def verify_runtime_hardware(
    lock: Mapping[str, Any],
    build_info: Mapping[str, Any],
    *,
    hardware_probe: Callable[[], Mapping[str, Any]] = probe_torch_hardware,
) -> None:
    observed = hardware_probe()
    expected_capability = lock["platform"]["gpu_compute_capability"]
    if observed.get("cuda_runtime") != lock["platform"]["cuda_toolkit"]:
        raise EnvironmentContractError("runtime CUDA version does not match the cu130 lock")
    if observed.get("gpu_compute_capability") != expected_capability:
        raise EnvironmentContractError("runtime GPU is not compute capability 12.0")
    system = build_info["system"]
    for key in ("cuda_runtime", "gpu_compute_capability", "gpu_name"):
        if observed.get(key) != system.get(key):
            raise EnvironmentContractError(f"runtime hardware does not match build info field {key}")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = _repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=root / "environment" / "reproduction-cu130.lock.json")
    parser.add_argument("--build-info", type=Path, default=root / "environment" / "build-info.template.json")
    parser.add_argument("--lock-only", action="store_true", help="validate pins without opening the GPU gate")
    parser.add_argument(
        "--seal-build-info",
        action="store_true",
        help="recalculate build_info_sha256 after evidence fields have been filled",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = _repository_root()
    try:
        lock = load_environment_lock(args.lock, repository_root=root)
        if args.lock_only:
            print(json.dumps({"lock_sha256": lock["lock_sha256"], "status": "valid_pins_gpu_unverified"}, sort_keys=True))
            return 0
        try:
            build_info = json.loads(args.build_info.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EnvironmentContractError(f"cannot read build info {args.build_info}: {exc}") from exc
        if args.seal_build_info:
            sealed = seal_payload(build_info, "build_info_sha256")
            _atomic_write_json(args.build_info, sealed)
            print(json.dumps({"build_info_sha256": sealed["build_info_sha256"], "status": "sealed"}, sort_keys=True))
            return 0
        validate_build_info(build_info, lock)
        verify_runtime_versions(lock)
        verify_kernel_install_sources(lock)
        verify_runtime_hardware(lock, build_info)
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}", "status": "blocked"}, sort_keys=True))
        return 2
    print(json.dumps({"environment_lock_sha256": lock["lock_sha256"], "status": "ready"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
