import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PREPARE_CPU = REPO_ROOT / "scripts" / "cloud" / "prepare_cpu.sh"


def _bash_path(path: Path) -> str:
    path = path.resolve()
    if os.name != "nt":
        return str(path)
    drive, tail = os.path.splitdrive(str(path))
    return f"/mnt/{drive[0].lower()}/{tail.lstrip('\\/').replace(os.sep, '/')}"


def _bash_available() -> bool:
    bash = shutil.which("bash")
    git = shutil.which("git")
    if not bash or not git:
        return False
    result = subprocess.run(
        [bash, "-lc", "command -v git >/dev/null && command -v realpath >/dev/null"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return result.returncode == 0


pytestmark = pytest.mark.skipif(
    not _bash_available(), reason="Linux bash/git/realpath required"
)


def _git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [shutil.which("git"), "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )


def _write_checkout(tmp_path: Path) -> tuple[Path, str]:
    project = tmp_path / "project"
    cloud = project / "scripts" / "cloud"
    cloud.mkdir(parents=True)
    (cloud / "prepare_cpu.sh").write_bytes(PREPARE_CPU.read_bytes())
    (cloud / "init_cloud.sh").write_bytes(
        b"#!/usr/bin/env bash\n"
        b"printf 'init-arg=<%s>\\n' \"$@\"\n"
        b"printf 'init-env=<%s>|<%s>|<%s>\\n' \"${CUDA_VISIBLE_DEVICES-unset}\" "
        b"\"${NVIDIA_VISIBLE_DEVICES-unset}\" "
        b"\"${REMEMR1_ALLOW_GPU_CPU_PHASE-unset}\"\n"
    )
    (cloud / "start_cpu_prep.sh").write_bytes(
        b"#!/usr/bin/env bash\n"
        b"printf 'start-arg=<%s>\\n' \"$@\"\n"
        b"printf 'start-env=<%s>|<%s>|<%s>|<%s>|<%s>\\n' "
        b"\"${CUDA_VISIBLE_DEVICES-unset}\" "
        b"\"${NVIDIA_VISIBLE_DEVICES-unset}\" "
        b"\"${REMEMR1_ALLOW_GPU_CPU_PHASE-unset}\" "
        b"\"${REMEMR1_MIN_CPU_CORES-unset}\" "
        b"\"${REMEMR1_MIN_RAM_GIB-unset}\"\n"
    )
    _git(project, "init")
    _git(project, "config", "user.name", "ReMemR1 Test")
    _git(project, "config", "user.email", "test@example.invalid")
    _git(project, "add", ".")
    _git(project, "commit", "-m", "fixture")
    commit = _git(project, "rev-parse", "HEAD").stdout.strip()
    assert len(commit) == 40
    return project, commit


def _run_prepare(project: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            shutil.which("bash"),
            _bash_path(project / "scripts" / "cloud" / "prepare_cpu.sh"),
            *arguments,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=30,
    )


def test_prepare_cpu_pins_clean_head_and_forces_low_resource_cpu_environment(
    tmp_path,
):
    project, commit = _write_checkout(tmp_path)

    result = _run_prepare(
        project,
        "--keep-running",
        "--dry-run",
        "--retry-failed-stage",
    )

    assert result.returncode == 0, result.stderr
    assert f"init-arg=<{commit}>" in result.stdout
    assert "init-arg=<--expected-commit>" in result.stdout
    assert "init-arg=<--allow-guest-shutdown>" in result.stdout
    assert "init-env=<>|<void>|<yes>" in result.stdout
    assert result.stdout.splitlines()[-4:] == [
        "start-arg=<--keep-running>",
        "start-arg=<--dry-run>",
        "start-arg=<--retry-failed-stage>",
        "start-env=<>|<void>|<yes>|<1/2>|<2>",
    ]


def test_prepare_cpu_rejects_dirty_checkout_before_initialization(tmp_path):
    project, _ = _write_checkout(tmp_path)
    (project / "untracked.txt").write_text("dirty\n", encoding="ascii")

    result = _run_prepare(project)

    assert result.returncode != 0
    assert "refuses a dirty checkout" in result.stderr
    assert "init-arg=" not in result.stdout
    assert "start-arg=" not in result.stdout


def test_prepare_cpu_rejects_unknown_argument(tmp_path):
    project, _ = _write_checkout(tmp_path)

    result = _run_prepare(project, "--unknown")

    assert result.returncode == 2
    assert "unknown CPU preparation argument" in result.stderr
    assert "init-arg=" not in result.stdout
