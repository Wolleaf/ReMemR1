"""Generate truthful sm_120 kernel evidence and seal the runtime build info."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.reproduction import verify_environment as environment


class ProbeError(RuntimeError):
    """Raised when a CUDA kernel cannot satisfy the evidence contract."""


GIB = 1024**3
MIN_GPU_MEMORY_BYTES = 31 * GIB
MIN_FREE_GPU_MEMORY_BYTES = 29 * GIB
MIN_DISK_FREE_BYTES = 200 * GIB
MIN_CPU_COUNT = 24
MIN_HOST_MEMORY_BYTES = {"R0": 96 * GIB, "R1": 128 * GIB}
_RTX_5090_NAME = re.compile(r"(?:^|\s)GEFORCE\s+RTX\s+5090$", re.IGNORECASE)
_NVCC_RELEASE = re.compile(r"\brelease\s+([0-9]+\.[0-9]+)\b", re.IGNORECASE)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_finite(torch: Any, label: str, *tensors: Any) -> None:
    for index, tensor in enumerate(tensors):
        if tensor is None or not torch.isfinite(tensor).all().item():
            raise ProbeError(f"{label} tensor {index} is missing or non-finite")


def _causal_case(mode: str, steps: int) -> dict[str, Any]:
    import torch
    from causal_conv1d import causal_conv1d_fn

    torch.manual_seed(42)
    parameters = [
        torch.nn.Parameter(torch.randn(1, 64, 512, device="cuda", dtype=torch.bfloat16)),
        torch.nn.Parameter(torch.randn(64, 4, device="cuda", dtype=torch.bfloat16) * 0.02),
        torch.nn.Parameter(torch.zeros(64, device="cuda", dtype=torch.bfloat16)),
    ]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4) if mode == "optimizer" else None
    losses: list[float] = []
    iterations = steps if optimizer is not None else 1
    for _ in range(iterations):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        output = causal_conv1d_fn(
            parameters[0], parameters[1], parameters[2], activation="silu"
        )
        torch.cuda.synchronize()
        _require_finite(torch, "causal-conv1d output", output)
        if mode != "forward":
            loss = output.float().square().mean()
            loss.backward()
            torch.cuda.synchronize()
            _require_finite(
                torch,
                "causal-conv1d gradients",
                *(parameter.grad for parameter in parameters),
            )
            losses.append(float(loss.detach().cpu()))
            if optimizer is not None:
                optimizer.step()
    return {
        "kernel": "causal-conv1d",
        "mode": mode,
        "optimizer_steps": iterations if optimizer is not None else 0,
        "losses": losses,
        "shape": [1, 64, 512],
        "status": "verified",
    }


def _fla_parameters(torch: Any) -> dict[str, Any]:
    shape = (1, 128, 16, 128)
    gate_shape = shape[:3]
    return {
        name: torch.nn.Parameter(
            torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.02
        )
        for name in ("q", "k", "v")
    } | {
        "g": torch.nn.Parameter(
            torch.full(gate_shape, -0.1, device="cuda", dtype=torch.float32)
        ),
        "beta": torch.nn.Parameter(
            torch.full(gate_shape, 0.5, device="cuda", dtype=torch.bfloat16)
        ),
    }


def _fla_case(mode: str, steps: int) -> dict[str, Any]:
    import torch
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    torch.manual_seed(42)
    inputs = _fla_parameters(torch)
    parameters = list(inputs.values())
    optimizer = torch.optim.AdamW(parameters, lr=1e-4) if mode == "optimizer" else None
    losses: list[float] = []
    iterations = steps if optimizer is not None else 1
    for _ in range(iterations):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        result = chunk_gated_delta_rule(
            **inputs,
            use_qk_l2norm_in_kernel=True,
        )
        output = result[0] if isinstance(result, tuple) else result
        torch.cuda.synchronize()
        _require_finite(torch, "flash-linear-attention output", output)
        if mode != "forward":
            loss = output.float().square().mean()
            loss.backward()
            torch.cuda.synchronize()
            _require_finite(
                torch,
                "flash-linear-attention gradients",
                *(parameter.grad for parameter in parameters),
            )
            losses.append(float(loss.detach().cpu()))
            if optimizer is not None:
                optimizer.step()
    return {
        "kernel": "flash-linear-attention",
        "mode": mode,
        "optimizer_steps": iterations if optimizer is not None else 0,
        "losses": losses,
        "shape": [1, 128, 16, 128],
        "status": "verified",
        "use_qk_l2norm_in_kernel": True,
    }


def _run_text(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProbeError(f"cannot run {command[0]}: {exc}") from exc
    return completed.stdout


def _parse_mib(value: str, label: str) -> int:
    try:
        mib = int(value.strip())
    except ValueError as exc:
        raise ProbeError(f"nvidia-smi returned invalid {label}: {value!r}") from exc
    if mib < 0:
        raise ProbeError(f"nvidia-smi returned negative {label}")
    return mib * 1024**2


def _query_nvidia_smi() -> dict[str, Any]:
    rows = [
        line.strip()
        for line in _run_text(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,uuid,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ]
        ).splitlines()
        if line.strip()
    ]
    gpus: list[dict[str, Any]] = []
    for row in rows:
        fields = [field.strip() for field in row.split(",")]
        if len(fields) != 5 or not all(fields):
            raise ProbeError(f"nvidia-smi returned an invalid GPU row: {row!r}")
        name, driver, gpu_uuid, total, free = fields
        gpus.append(
            {
                "driver_version": driver,
                "gpu_free_memory_bytes": _parse_mib(free, "free GPU memory"),
                "gpu_name": name,
                "gpu_total_memory_bytes": _parse_mib(total, "total GPU memory"),
                "gpu_uuid": gpu_uuid,
            }
        )

    process_rows = [
        line.strip()
        for line in _run_text(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ]
        ).splitlines()
        if line.strip() and "no running processes" not in line.lower()
    ]
    processes: list[dict[str, Any]] = []
    for row in process_rows:
        fields = [field.strip() for field in row.split(",", 3)]
        if len(fields) != 4 or not all(fields):
            raise ProbeError(f"nvidia-smi returned an invalid compute process row: {row!r}")
        gpu_uuid, pid, process_name, used = fields
        try:
            parsed_pid = int(pid)
        except ValueError as exc:
            raise ProbeError(f"nvidia-smi returned an invalid process pid: {pid!r}") from exc
        processes.append(
            {
                "gpu_uuid": gpu_uuid,
                "pid": parsed_pid,
                "process_name": process_name,
                "used_gpu_memory_bytes": _parse_mib(used, "process GPU memory"),
            }
        )
    return {"compute_processes": processes, "gpus": gpus}


def _query_cuda_toolkit() -> str:
    output = _run_text(["nvcc", "--version"])
    match = _NVCC_RELEASE.search(output)
    if match is None:
        raise ProbeError("nvcc did not report a CUDA toolkit release")
    return match.group(1)


def _read_host_total_memory_bytes() -> int:
    path = Path("/proc/meminfo")
    try:
        for line in path.read_text(encoding="ascii").splitlines():
            if line.startswith("MemTotal:"):
                fields = line.split()
                if len(fields) == 3 and fields[2].lower() == "kb":
                    return int(fields[1]) * 1024
    except (OSError, ValueError) as exc:
        raise ProbeError(f"cannot determine host RAM: {exc}") from exc
    raise ProbeError("cannot determine host RAM from /proc/meminfo")


def _host_resources(disk_path: Path) -> dict[str, Any]:
    candidate = disk_path.resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    try:
        disk_free = shutil.disk_usage(candidate).free
    except OSError as exc:
        raise ProbeError(f"cannot determine free disk at {candidate}: {exc}") from exc
    return {
        "host_cpu_count": os.cpu_count() or 0,
        "host_total_memory_bytes": _read_host_total_memory_bytes(),
        "persistent_disk_free_bytes": disk_free,
        "persistent_disk_probe_path": str(candidate),
    }


def _valid_5090_name(name: Any) -> bool:
    return isinstance(name, str) and _RTX_5090_NAME.search(" ".join(name.split())) is not None


def _probe_hardware(
    *,
    profile: str = "R0",
    disk_path: Path = REPOSITORY_ROOT,
    torch_module: Any | None = None,
    nvidia_smi_probe: Callable[[], Mapping[str, Any]] = _query_nvidia_smi,
    toolkit_probe: Callable[[], str] = _query_cuda_toolkit,
    host_resource_probe: Callable[[], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if profile not in MIN_HOST_MEMORY_BYTES:
        raise ProbeError(f"unsupported capacity profile {profile!r}")
    if torch_module is None:
        import torch as torch_module

    smi = nvidia_smi_probe()
    gpus = smi.get("gpus") if isinstance(smi, Mapping) else None
    processes = smi.get("compute_processes") if isinstance(smi, Mapping) else None
    if not isinstance(gpus, list) or len(gpus) != 1:
        observed = len(gpus) if isinstance(gpus, list) else "invalid"
        raise ProbeError(f"expected exactly 1 NVIDIA GPU, observed {observed}")
    if not isinstance(processes, list):
        raise ProbeError("nvidia-smi compute process inventory is invalid")
    other_processes = [
        process
        for process in processes
        if not isinstance(process, Mapping) or process.get("pid") != os.getpid()
    ]
    if other_processes:
        raise ProbeError(f"other compute processes are running: {other_processes!r}")

    cuda = torch_module.cuda
    if not cuda.is_available():
        raise ProbeError("CUDA is unavailable")
    device_count = cuda.device_count()
    if device_count != 1:
        raise ProbeError(f"expected exactly 1 visible CUDA GPU, observed {device_count}")
    capability = list(cuda.get_device_capability(0))
    name = cuda.get_device_name(0)
    memory = int(cuda.get_device_properties(0).total_memory)
    runtime = str(torch_module.version.cuda)
    toolkit = str(toolkit_probe())
    gpu = gpus[0]
    if not isinstance(gpu, Mapping):
        raise ProbeError("nvidia-smi GPU inventory is invalid")
    if capability != [12, 0]:
        raise ProbeError(f"expected sm_120, observed {capability}")
    if not _valid_5090_name(name) or not _valid_5090_name(gpu.get("gpu_name")):
        raise ProbeError(f"expected NVIDIA GeForce RTX 5090, observed {name!r}")
    smi_memory = gpu.get("gpu_total_memory_bytes")
    free_memory = gpu.get("gpu_free_memory_bytes")
    if not isinstance(smi_memory, int) or not isinstance(free_memory, int):
        raise ProbeError("nvidia-smi GPU memory inventory is invalid")
    if min(memory, smi_memory) < MIN_GPU_MEMORY_BYTES:
        raise ProbeError(
            f"expected at least 31 GiB VRAM, observed {min(memory, smi_memory) / GIB:.1f} GiB"
        )
    if free_memory < MIN_FREE_GPU_MEMORY_BYTES:
        raise ProbeError(
            f"expected at least 29 GiB free VRAM, observed {free_memory / GIB:.1f} GiB"
        )
    if runtime != "13.0":
        raise ProbeError(f"expected CUDA runtime 13.0, observed {runtime!r}")
    if toolkit != "13.0":
        raise ProbeError(f"expected CUDA toolkit 13.0, observed {toolkit!r}")

    resources = dict(
        host_resource_probe() if host_resource_probe is not None else _host_resources(disk_path)
    )
    cpu_count = resources.get("host_cpu_count")
    host_memory = resources.get("host_total_memory_bytes")
    disk_free = resources.get("persistent_disk_free_bytes")
    if not isinstance(cpu_count, int) or cpu_count < MIN_CPU_COUNT:
        raise ProbeError(
            f"expected at least {MIN_CPU_COUNT} host CPU cores, observed {cpu_count!r}"
        )
    required_ram = MIN_HOST_MEMORY_BYTES[profile]
    if not isinstance(host_memory, int) or host_memory < required_ram:
        raise ProbeError(
            f"{profile} requires at least {required_ram / GIB:.0f} GiB host RAM, "
            f"observed {host_memory / GIB:.1f} GiB"
            if isinstance(host_memory, int)
            else f"{profile} host RAM inventory is invalid"
        )
    if not isinstance(disk_free, int) or disk_free < MIN_DISK_FREE_BYTES:
        raise ProbeError(
            f"expected at least 200 GiB free persistent disk, observed {disk_free / GIB:.1f} GiB"
            if isinstance(disk_free, int)
            else "persistent disk inventory is invalid"
        )
    gpu_uuid = gpu.get("gpu_uuid")
    driver = gpu.get("driver_version")
    if not isinstance(gpu_uuid, str) or not gpu_uuid or not isinstance(driver, str) or not driver:
        raise ProbeError("nvidia-smi returned an invalid GPU identity")
    return {
        "capacity_profile": profile,
        "cuda_runtime": runtime,
        "cuda_toolkit": toolkit,
        "driver_version": driver,
        "gpu_compute_capability": capability,
        "gpu_free_memory_bytes": free_memory,
        "gpu_name": name,
        "gpu_total_memory_bytes": smi_memory,
        "gpu_uuid": gpu_uuid,
        "host_cpu_count": cpu_count,
        "host_total_memory_bytes": host_memory,
        "other_compute_process_count": 0,
        "persistent_disk_free_bytes": disk_free,
        "persistent_disk_probe_path": resources.get("persistent_disk_probe_path", str(disk_path)),
        "torch_gpu_total_memory_bytes": memory,
    }


def _os_release() -> str:
    values: dict[str, str] = {}
    path = Path("/etc/os-release")
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value.strip().strip('"')
    return values.get("PRETTY_NAME") or platform.platform()


def run_probe(
    evidence_root: Path,
    *,
    optimizer_steps: int,
    profile: str = "R0",
    hardware_probe: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    hardware = (
        hardware_probe()
        if hardware_probe is not None
        else _probe_hardware(profile=profile, disk_path=evidence_root)
    )
    probes: dict[str, dict[str, Path]] = {}
    functions: dict[str, Callable[[str, int], dict[str, Any]]] = {
        "causal-conv1d": _causal_case,
        "flash-linear-attention": _fla_case,
    }
    for kernel, function in functions.items():
        paths: dict[str, Path] = {}
        for mode in ("forward", "backward", "optimizer"):
            result = function(mode, optimizer_steps)
            result["hardware"] = hardware
            path = evidence_root / kernel / f"bf16-{mode}.json"
            _atomic_json(path, result)
            paths[mode] = path
        probes[kernel] = paths
    return {"hardware": hardware, "probes": probes}


def build_info(
    root: Path,
    evidence_root: Path,
    build_log: Path,
    freeze: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    lock = environment.load_environment_lock(
        root / "environment/reproduction-cu130.lock.json",
        repository_root=root,
    )
    environment.verify_runtime_versions(lock)
    environment.verify_kernel_install_sources(lock)
    packages = {
        item["name"]: importlib.metadata.version(item["name"])
        for item in lock["packages"]
    }
    hardware = result["hardware"]
    records: dict[str, Any] = {}
    pinned = {item["name"]: item for item in lock["kernels"]}
    for name, paths in result["probes"].items():
        source = pinned[name]
        records[name] = {
            "bf16_backward_log_sha256": _sha256_file(paths["backward"]),
            "bf16_forward_log_sha256": _sha256_file(paths["forward"]),
            "build_log_sha256": _sha256_file(build_log),
            "commit": source["commit"],
            "optimizer_loop_log_sha256": _sha256_file(paths["optimizer"]),
            "repository": source["repository"],
            "status": "VERIFIED_SM120",
        }
    payload = {
        "build_info_sha256": "0" * 64,
        "environment_lock_sha256": lock["lock_sha256"],
        "kernels": records,
        "packages": packages,
        "pip_freeze_sha256": _sha256_file(freeze),
        "python": platform.python_version(),
        "schema_version": 1,
        "status": "VERIFIED_SM120",
        "system": {
            "cuda_runtime": hardware["cuda_runtime"],
            "driver_version": hardware["driver_version"],
            "gpu_compute_capability": hardware["gpu_compute_capability"],
            "gpu_name": hardware["gpu_name"],
            "operating_system": _os_release(),
        },
        "training_gate": "READY",
    }
    return environment.seal_payload(payload, "build_info_sha256")


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ProbeError(f"{label} must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ProbeError(f"{label} is not a regular file")
    return resolved


def _load_json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(_regular_file(path, label).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ProbeError(f"{label} must contain a JSON object")
    return value


def verify_existing_evidence(
    root: Path,
    evidence_root: Path,
    build_log: Path,
    freeze: Path,
    build_info_path: Path,
    *,
    minimum_optimizer_steps: int,
    profile: str = "R0",
    hardware_probe: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    lock = environment.load_environment_lock(
        root / "environment/reproduction-cu130.lock.json",
        repository_root=root,
    )
    info = _load_json_mapping(build_info_path, "build info")
    environment.validate_build_info(info, lock)
    environment.verify_runtime_versions(lock)
    environment.verify_kernel_install_sources(lock)

    hardware = (
        hardware_probe()
        if hardware_probe is not None
        else _probe_hardware(profile=profile, disk_path=evidence_root)
    )
    system = info["system"]
    for key in ("cuda_runtime", "driver_version", "gpu_compute_capability", "gpu_name"):
        if system[key] != hardware[key]:
            raise ProbeError(f"current host differs from build evidence field {key}")
    if system["operating_system"] != _os_release():
        raise ProbeError("current operating system differs from build evidence")

    build_log = _regular_file(build_log, "kernel build log")
    freeze = _regular_file(freeze, "GPU pip freeze")
    if _sha256_file(freeze) != info["pip_freeze_sha256"]:
        raise ProbeError("GPU pip freeze changed after kernel verification")
    build_digest = _sha256_file(build_log)
    evidence_fields = {
        "forward": "bf16_forward_log_sha256",
        "backward": "bf16_backward_log_sha256",
        "optimizer": "optimizer_loop_log_sha256",
    }
    for kernel, record in info["kernels"].items():
        if record["build_log_sha256"] != build_digest:
            raise ProbeError(f"kernel build log changed for {kernel}")
        for mode, digest_field in evidence_fields.items():
            evidence_path = evidence_root / kernel / f"bf16-{mode}.json"
            evidence = _load_json_mapping(evidence_path, f"{kernel} {mode} evidence")
            if _sha256_file(evidence_path) != record[digest_field]:
                raise ProbeError(f"{kernel} {mode} evidence hash changed")
            evidence_hardware = evidence.get("hardware")
            stable_hardware_fields = {
                "cuda_runtime",
                "cuda_toolkit",
                "driver_version",
                "gpu_compute_capability",
                "gpu_name",
                "gpu_total_memory_bytes",
                "gpu_uuid",
                "torch_gpu_total_memory_bytes",
            }
            hardware_changed = not isinstance(evidence_hardware, Mapping) or any(
                key in evidence_hardware and evidence_hardware.get(key) != hardware.get(key)
                for key in stable_hardware_fields
            )
            if (
                evidence.get("kernel") != kernel
                or evidence.get("mode") != mode
                or evidence.get("status") != "verified"
                or hardware_changed
            ):
                raise ProbeError(f"{kernel} {mode} evidence identity changed")
            steps = evidence.get("optimizer_steps")
            if mode == "optimizer":
                if not isinstance(steps, int) or steps < minimum_optimizer_steps:
                    raise ProbeError(f"{kernel} optimizer evidence has too few steps")
            elif steps != 0:
                raise ProbeError(f"{kernel} {mode} evidence has unexpected optimizer steps")
            losses = evidence.get("losses")
            if not isinstance(losses, list) or any(
                not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in losses
            ):
                raise ProbeError(f"{kernel} {mode} evidence has invalid losses")
    return {
        "build_info_sha256": info["build_info_sha256"],
        "driver_version": hardware["driver_version"],
        "status": "verified-existing",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--build-log", type=Path, required=True)
    parser.add_argument("--pip-freeze", type=Path, required=True)
    parser.add_argument("--build-info", type=Path, required=True)
    parser.add_argument("--optimizer-steps", type=int, default=20)
    parser.add_argument("--profile", choices=sorted(MIN_HOST_MEMORY_BYTES), default="R0")
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.optimizer_steps < 2:
            raise ProbeError("--optimizer-steps must be at least 2")
        root = Path(__file__).resolve().parents[2]
        build_log = args.build_log.resolve(strict=True)
        freeze = args.pip_freeze.resolve(strict=True)
        if args.verify_existing:
            verified = verify_existing_evidence(
                root,
                args.evidence_root.resolve(strict=True),
                build_log,
                freeze,
                args.build_info,
                minimum_optimizer_steps=args.optimizer_steps,
                profile=args.profile,
            )
            print(json.dumps(verified, sort_keys=True))
            return 0
        result = run_probe(
            args.evidence_root.resolve(),
            optimizer_steps=args.optimizer_steps,
            profile=args.profile,
        )
        sealed = build_info(root, args.evidence_root, build_log, freeze, result)
        _atomic_json(args.build_info, sealed)
        environment.validate_build_info(
            sealed,
            environment.load_environment_lock(
                root / "environment/reproduction-cu130.lock.json",
                repository_root=root,
            ),
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
    print(
        json.dumps(
            {
                "build_info": str(args.build_info.resolve()),
                "build_info_sha256": sealed["build_info_sha256"],
                "status": "ready",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
