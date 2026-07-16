import hashlib
import json

import pytest

from scripts.cloud import gpu_probe


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        "gpu_name": "NVIDIA RTX PRO 6000 Blackwell",
        "gpu_total_memory_bytes": 96 * 1024**3,
        "gpu_uuid": "GPU-test-uuid",
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
