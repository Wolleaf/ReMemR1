import copy
import json
from pathlib import Path

import pytest

from scripts.reproduction import verify_environment as environment


ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "environment" / "reproduction-cu130.lock.json"
BUILD_INFO_TEMPLATE_PATH = ROOT / "environment" / "build-info.template.json"


def _load_lock():
    return environment.load_environment_lock(LOCK_PATH, repository_root=ROOT)


def _verified_build_info(lock):
    template = json.loads(BUILD_INFO_TEMPLATE_PATH.read_text(encoding="utf-8"))
    template["status"] = "VERIFIED_SM120"
    template["training_gate"] = "READY"
    template["python"] = lock["platform"]["python"]
    template["pip_freeze_sha256"] = "a" * 64
    template["packages"] = {item["name"]: item["version"] for item in lock["packages"]}
    template["system"] = {
        "cuda_runtime": "13.0",
        "driver_version": "600.00",
        "gpu_compute_capability": [12, 0],
        "gpu_name": "NVIDIA RTX PRO 6000 Blackwell",
        "operating_system": "Ubuntu 22.04",
    }
    for record in template["kernels"].values():
        record["status"] = "VERIFIED_SM120"
        for field in environment._EVIDENCE_FIELDS:
            record[field] = "b" * 64
    return environment.seal_payload(template, "build_info_sha256")


def test_environment_lock_pins_critical_versions_and_unverified_kernels():
    lock = _load_lock()
    packages = {item["name"]: item["version"] for item in lock["packages"]}

    assert lock["platform"] == {
        "cuda_toolkit": "13.0",
        "gpu_compute_capability": [12, 0],
        "operating_system": "Ubuntu 22.04",
        "python": "3.12.2",
        "torch_index_url": "https://download.pytorch.org/whl/cu130",
    }
    assert packages == {
        "datasets": "5.0.0",
        "huggingface-hub": "1.23.0",
        "hydra-core": "1.3.4",
        "omegaconf": "2.3.1",
        "peft": "0.19.1",
        "ray": "2.56.0",
        "tensordict": "0.13.0",
        "torch": "2.11.0+cu130",
        "torchdata": "0.11.0",
        "transformers": "5.14.0",
    }
    assert {item["name"]: item["commit"] for item in lock["kernels"]} == {
        "causal-conv1d": "4f6ae4e26ae5fe8af9372f8d312ab25cc4595223",
        "flash-linear-attention": "b328e7c611ca205d1908cce9a90b8ca223fc0101",
    }
    assert all(item["verification_status"] == "UNVERIFIED" for item in lock["kernels"])
    assert all(item["training_gate"] == "BLOCKED" for item in lock["kernels"])


def test_lock_rejects_resealed_pin_that_diverges_from_requirements():
    lock = _load_lock()
    tampered = copy.deepcopy(lock)
    next(item for item in tampered["packages"] if item["name"] == "torch")["version"] = "9.9.9"
    tampered = environment.seal_payload(tampered, "lock_sha256")

    with pytest.raises(environment.EnvironmentContractError, match="requirements package pins"):
        environment.validate_environment_lock(tampered, repository_root=ROOT)


def test_build_info_template_is_self_hashed_and_deliberately_blocks_training():
    lock = _load_lock()
    template = json.loads(BUILD_INFO_TEMPLATE_PATH.read_text(encoding="utf-8"))
    assert template["build_info_sha256"] == environment.canonical_json_sha256(
        environment._without_digest(template, "build_info_sha256")
    )
    assert template["status"] == "UNVERIFIED"
    assert template["training_gate"] == "BLOCKED"
    with pytest.raises(environment.EnvironmentContractError, match="long GPU runs are blocked"):
        environment.validate_build_info(template, lock)


def test_fully_evidenced_build_info_passes_but_missing_kernel_evidence_fails():
    lock = _load_lock()
    verified = _verified_build_info(lock)
    environment.validate_build_info(verified, lock)

    missing = copy.deepcopy(verified)
    missing["kernels"]["flash-linear-attention"]["bf16_backward_log_sha256"] = "UNVERIFIED"
    missing = environment.seal_payload(missing, "build_info_sha256")
    with pytest.raises(environment.EnvironmentContractError, match="requires a real evidence"):
        environment.validate_build_info(missing, lock)


def test_runtime_versions_must_match_every_exact_pin():
    lock = _load_lock()
    versions = {item["name"]: item["version"] for item in lock["packages"]}
    environment.verify_runtime_versions(
        lock,
        python_version="3.12.2",
        version_getter=versions.__getitem__,
    )

    versions["transformers"] = "5.13.0"
    with pytest.raises(environment.EnvironmentContractError, match="transformers version mismatch"):
        environment.verify_runtime_versions(
            lock,
            python_version="3.12.2",
            version_getter=versions.__getitem__,
        )


def test_kernel_direct_url_commits_are_checked_not_inferred_from_import_success():
    lock = _load_lock()
    sources = {
        item["distribution"]: {
            "url": item["repository"],
            "vcs_info": {"commit_id": item["commit"], "vcs": "git"},
        }
        for item in lock["kernels"]
    }
    environment.verify_kernel_install_sources(lock, direct_url_getter=sources.__getitem__)

    sources["causal-conv1d"]["vcs_info"]["commit_id"] = "0" * 40
    with pytest.raises(environment.EnvironmentContractError, match="does not match the lock"):
        environment.verify_kernel_install_sources(lock, direct_url_getter=sources.__getitem__)


def test_runtime_hardware_must_be_sm120_and_match_sealed_build_info():
    lock = _load_lock()
    build_info = _verified_build_info(lock)
    hardware = {
        "cuda_runtime": "13.0",
        "gpu_compute_capability": [12, 0],
        "gpu_name": "NVIDIA RTX PRO 6000 Blackwell",
    }
    environment.verify_runtime_hardware(lock, build_info, hardware_probe=lambda: hardware)

    wrong_gpu = {**hardware, "gpu_compute_capability": [10, 0]}
    with pytest.raises(environment.EnvironmentContractError, match="not compute capability 12.0"):
        environment.verify_runtime_hardware(lock, build_info, hardware_probe=lambda: wrong_gpu)
