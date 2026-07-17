import hashlib
import json
from fractions import Fraction
from types import SimpleNamespace

import pytest

from scripts.cloud import gpu_probe


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _FakeCuda:
    def __init__(self, *, count=1, name="NVIDIA GeForce RTX 5090", memory=32 * 1024**3):
        self._count = count
        self._name = name
        self._memory = memory

    def is_available(self):
        return True

    def device_count(self):
        return self._count

    def get_device_capability(self, index):
        assert index == 0
        return (12, 0)

    def get_device_name(self, index):
        assert index == 0
        return self._name

    def get_device_properties(self, index):
        assert index == 0
        return SimpleNamespace(total_memory=self._memory)


def _hardware_inputs():
    smi = {
        "compute_processes": [],
        "gpus": [
            {
                "driver_version": "999.1",
                "gpu_free_memory_bytes": 30 * 1024**3,
                "gpu_name": "NVIDIA GeForce RTX 5090",
                "gpu_total_memory_bytes": 32 * 1024**3,
                "gpu_uuid": "GPU-test-uuid",
            }
        ],
    }
    resources = {
        "host_cpu_count": 32,
        "host_total_memory_bytes": 160 * 1024**3,
        "persistent_disk_free_bytes": 250 * 1024**3,
        "persistent_disk_probe_path": "/persistent",
    }
    torch_module = SimpleNamespace(cuda=_FakeCuda(), version=SimpleNamespace(cuda="13.0"))
    return torch_module, smi, resources


def _probe_with(torch_module, smi, resources, *, profile="R0", toolkit="13.0"):
    return gpu_probe._probe_hardware(
        profile=profile,
        torch_module=torch_module,
        nvidia_smi_probe=lambda: smi,
        toolkit_probe=lambda: toolkit,
        host_resource_probe=lambda: resources,
    )


def test_rtx5090_preflight_accepts_injected_r0_and_r1_profiles():
    torch_module, smi, resources = _hardware_inputs()
    result = _probe_with(torch_module, smi, resources)
    assert result["gpu_name"] == "NVIDIA GeForce RTX 5090"
    assert result["gpu_free_memory_bytes"] == 30 * 1024**3
    assert result["cuda_toolkit"] == "13.0"
    assert result["other_compute_process_count"] == 0

    result = _probe_with(torch_module, smi, resources, profile="R1")
    assert result["capacity_profile"] == "R1"


def test_rtx5090_r0_accepts_autodl_16_core_90_gb_profile():
    torch_module, smi, resources = _hardware_inputs()
    resources.update(
        host_cpu_count=16,
        host_total_memory_bytes=90_000_000_000,
    )

    result = _probe_with(torch_module, smi, resources, profile="R0")

    assert result["host_cpu_count"] == 16
    assert result["host_total_memory_bytes"] == 90_000_000_000


def test_rtx5090_r0_rejects_15_cores_and_less_than_80_gib():
    torch_module, smi, resources = _hardware_inputs()
    resources.update(
        host_cpu_count=15,
        host_total_memory_bytes=90 * 1024**3,
    )
    with pytest.raises(gpu_probe.ProbeError, match="at least 16 host CPU cores"):
        _probe_with(torch_module, smi, resources, profile="R0")

    resources.update(
        host_cpu_count=16,
        host_total_memory_bytes=80 * 1024**3 - 1,
    )
    with pytest.raises(gpu_probe.ProbeError, match="R0 requires at least 80 GiB"):
        _probe_with(torch_module, smi, resources, profile="R0")


def test_gpu_host_inventory_uses_cgroup_effective_resources(tmp_path):
    result = gpu_probe._host_resources(
        tmp_path,
        resource_probe=lambda: SimpleNamespace(
            effective_cpu_cores=Fraction(33, 2),
            effective_memory_bytes=90 * 1024**3,
        ),
    )

    assert result["host_cpu_count"] == 16
    assert result["host_total_memory_bytes"] == 90 * 1024**3


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda torch, smi, resources: setattr(torch.cuda, "_count", 2), "exactly 1 visible"),
        (
            lambda torch, smi, resources: setattr(torch.cuda, "_name", "NVIDIA RTX PRO 6000"),
            "GeForce RTX 5090",
        ),
        (
            lambda torch, smi, resources: setattr(torch.cuda, "_memory", 30 * 1024**3),
            "31 GiB",
        ),
        (
            lambda torch, smi, resources: smi["gpus"][0].update(
                gpu_free_memory_bytes=28 * 1024**3
            ),
            "29 GiB free",
        ),
        (
            lambda torch, smi, resources: smi["compute_processes"].append(
                {"gpu_uuid": "GPU-test-uuid", "pid": 1234, "process_name": "python"}
            ),
            "other compute processes",
        ),
        (
            lambda torch, smi, resources: resources.update(
                persistent_disk_free_bytes=199 * 1024**3
            ),
            "200 GiB",
        ),
    ],
)
def test_rtx5090_preflight_fails_closed(mutate, message):
    torch_module, smi, resources = _hardware_inputs()
    mutate(torch_module, smi, resources)
    with pytest.raises(gpu_probe.ProbeError, match=message):
        _probe_with(torch_module, smi, resources)


def test_rtx5090_preflight_requires_exact_runtime_toolkit_and_r1_ram():
    torch_module, smi, resources = _hardware_inputs()
    with pytest.raises(gpu_probe.ProbeError, match="toolkit 13.0"):
        _probe_with(torch_module, smi, resources, toolkit="12.8")

    torch_module.version.cuda = "12.8"
    with pytest.raises(gpu_probe.ProbeError, match="runtime 13.0"):
        _probe_with(torch_module, smi, resources)
    torch_module.version.cuda = "13.0"

    resources["host_total_memory_bytes"] = 127 * 1024**3
    with pytest.raises(gpu_probe.ProbeError, match="R1 requires at least 128 GiB"):
        _probe_with(torch_module, smi, resources, profile="R1")


def test_nvidia_smi_and_nvcc_parsers_are_injectable(monkeypatch):
    outputs = iter(
        [
            "NVIDIA GeForce RTX 5090, 999.1, GPU-test, 32640, 30720\n",
            "",
            "Cuda compilation tools, release 13.0, V13.0.1\n",
        ]
    )
    monkeypatch.setattr(gpu_probe, "_run_text", lambda command: next(outputs))
    inventory = gpu_probe._query_nvidia_smi()
    assert inventory["gpus"][0]["gpu_free_memory_bytes"] == 30720 * 1024**2
    assert inventory["compute_processes"] == []
    assert gpu_probe._query_cuda_toolkit() == "13.0"


def test_existing_gpu_evidence_rehashes_files_and_matches_driver(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    evidence_root = tmp_path / "evidence"
    build_log = tmp_path / "kernel-build.log"
    freeze = tmp_path / "pip-freeze.txt"
    build_info_path = tmp_path / "build-info.json"
    build_log.write_text("build\n", encoding="ascii")
    freeze.write_text("packages\n", encoding="ascii")
    hardware = {
        "cuda_runtime": "13.0",
        "driver_version": "999.1",
        "gpu_compute_capability": [12, 0],
        "capacity_profile": "R0",
        "cuda_toolkit": "13.0",
        "gpu_free_memory_bytes": 30 * 1024**3,
        "gpu_name": "NVIDIA GeForce RTX 5090",
        "gpu_total_memory_bytes": 32 * 1024**3,
        "gpu_uuid": "GPU-test-uuid",
        "host_cpu_count": 32,
        "host_total_memory_bytes": 128 * 1024**3,
        "other_compute_process_count": 0,
        "persistent_disk_free_bytes": 250 * 1024**3,
        "persistent_disk_probe_path": "/persistent",
    }
    kernels = {}
    for kernel in ("causal-conv1d", "flash-linear-attention"):
        digests = {}
        for mode, field in (
            ("forward", "bf16_forward_log_sha256"),
            ("backward", "bf16_backward_log_sha256"),
            ("optimizer", "optimizer_loop_log_sha256"),
        ):
            path = evidence_root / kernel / f"bf16-{mode}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "hardware": hardware,
                        "kernel": kernel,
                        "losses": [] if mode == "forward" else [1.0],
                        "mode": mode,
                        "optimizer_steps": 20 if mode == "optimizer" else 0,
                        "status": "verified",
                    }
                ),
                encoding="ascii",
            )
            digests[field] = _sha256(path)
        kernels[kernel] = {
            **digests,
            "build_log_sha256": _sha256(build_log),
        }
    build_info = {
        "build_info_sha256": "a" * 64,
        "kernels": kernels,
        "pip_freeze_sha256": _sha256(freeze),
        "system": {
            "cuda_runtime": hardware["cuda_runtime"],
            "driver_version": hardware["driver_version"],
            "gpu_compute_capability": hardware["gpu_compute_capability"],
            "gpu_name": hardware["gpu_name"],
            "operating_system": "test-linux",
        },
    }
    build_info_path.write_text(json.dumps(build_info), encoding="ascii")
    monkeypatch.setattr(gpu_probe.environment, "load_environment_lock", lambda *a, **k: {})
    monkeypatch.setattr(gpu_probe.environment, "validate_build_info", lambda *a, **k: None)
    monkeypatch.setattr(gpu_probe.environment, "verify_runtime_versions", lambda *a, **k: None)
    monkeypatch.setattr(
        gpu_probe.environment, "verify_kernel_install_sources", lambda *a, **k: None
    )
    monkeypatch.setattr(gpu_probe, "_os_release", lambda: "test-linux")

    result = gpu_probe.verify_existing_evidence(
        root,
        evidence_root,
        build_log,
        freeze,
        build_info_path,
        minimum_optimizer_steps=20,
        hardware_probe=lambda: hardware,
    )
    assert result["status"] == "verified-existing"

    r1_hardware = {
        **hardware,
        "capacity_profile": "R1",
        "host_total_memory_bytes": 160 * 1024**3,
    }
    with pytest.raises(gpu_probe.ProbeError, match="evidence identity changed"):
        gpu_probe.verify_existing_evidence(
            root,
            evidence_root,
            build_log,
            freeze,
            build_info_path,
            minimum_optimizer_steps=20,
            profile="R1",
            hardware_probe=lambda: r1_hardware,
        )

    (evidence_root / "causal-conv1d" / "bf16-forward.json").write_text(
        "{}", encoding="ascii"
    )
    with pytest.raises(gpu_probe.ProbeError, match="hash changed|identity changed"):
        gpu_probe.verify_existing_evidence(
            root,
            evidence_root,
            build_log,
            freeze,
            build_info_path,
            minimum_optimizer_steps=20,
            hardware_probe=lambda: hardware,
        )
