import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCH = REPO_ROOT / "scripts" / "cloud" / "launch.sh"
BOOTSTRAP = REPO_ROOT / "scripts" / "cloud" / "bootstrap.sh"
RUN_GPU = REPO_ROOT / "scripts" / "cloud" / "run_gpu.sh"
STATUS = REPO_ROOT / "scripts" / "cloud" / "status.sh"
PREPARE_KERNEL_SOURCES = REPO_ROOT / "scripts" / "cloud" / "prepare_kernel_sources.sh"


def _bash_path(path: Path) -> str:
    path = path.resolve()
    if os.name != "nt":
        return str(path)
    drive, tail = os.path.splitdrive(str(path))
    return f"/mnt/{drive[0].lower()}/{tail.lstrip('\\/').replace(os.sep, '/')}"


def _bash_available() -> bool:
    bash = shutil.which("bash")
    if not bash:
        return False
    probe = subprocess.run(
        [bash, "-lc", "command -v flock >/dev/null && command -v setsid >/dev/null"],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(not _bash_available(), reason="Linux bash/flock/setsid required")


@pytest.fixture
def launcher_tmp_path(tmp_path):
    if os.name == "nt":
        yield tmp_path
        return
    # The production cloud env keeps TMPDIR on the persistent volume, while
    # test mode intentionally rejects every /root/autodl-tmp path.
    with tempfile.TemporaryDirectory(prefix="rememr1-cloud-test-", dir="/tmp") as path:
        yield Path(path)


def _write_cloud_fixture(tmp_path: Path, pipeline_rc: int = 0):
    project = tmp_path / "project"
    persist = tmp_path / "persist"
    launcher_root = persist / "launchers"
    pipeline = project / "scripts" / "cloud" / "run_pipeline.sh"
    pipeline.parent.mkdir(parents=True)
    launcher_root.mkdir(parents=True)
    lock_library = shlex.quote(
        _bash_path(REPO_ROOT / "scripts" / "cloud" / "lib" / "lock.sh")
    )
    pipeline.write_bytes(
        f"""#!/usr/bin/env bash
set -u
source {lock_library}
require_cloud_lock
run_dir=\"${{PERSIST_ROOT}}/fake-pipeline-${{FAKE_PIPELINE_RC}}\"
mkdir -p \"${{run_dir}}\"
printf '%s\\n' \"$*\" > \"${{run_dir}}/args\"
printf '%s\\n' \"${{run_dir}}\" > \"${{REMEMR1_RESULT_FILE}}\"
exit \"${{FAKE_PIPELINE_RC}}\"
""".encode("ascii")
    )
    capability = persist / "shutdown.capability"
    capability.write_bytes(b"test-only\n")
    lock_file = persist / "cloud.lock"
    env_file = persist / "cloud.env"
    values = {
        "REMEMR1_PROJECT_DIR": _bash_path(project),
        "PERSIST_ROOT": _bash_path(persist),
        "EXPECTED_COMMIT": "a" * 40,
        "REMEMR1_EXPERIMENT_PROFILE": "rtx5090-32g-qwen35-2b-v1",
        "CAPABILITY_FILE": _bash_path(capability),
        "LOCK_FILE": _bash_path(lock_file),
        "LAUNCHER_ROOT": _bash_path(launcher_root),
    }
    env_file.write_bytes(
        "".join(f"{key}={shlex.quote(value)}\n" for key, value in values.items()).encode(
            "ascii"
        )
    )
    shutdown_log = persist / "shutdown-events.log"
    env = os.environ.copy()
    for name in (
        "REMEMR1_PROJECT_DIR",
        "REMEMR1_PERSIST_ROOT",
        "PERSIST_ROOT",
        "EXPECTED_COMMIT",
        "CAPABILITY_FILE",
        "LOCK_FILE",
        "LAUNCHER_ROOT",
    ):
        env.pop(name, None)
    env.update(
        {
            "REMEMR1_CLOUD_ENV": _bash_path(env_file),
            "REMEMR1_TEST_MODE": "yes",
            "REMEMR1_TEST_SHUTDOWN_LOG": _bash_path(shutdown_log),
            "FAKE_PIPELINE_RC": str(pipeline_rc),
        }
    )
    if os.name == "nt":
        forwarded = [
            "REMEMR1_CLOUD_ENV",
            "REMEMR1_TEST_MODE",
            "REMEMR1_TEST_SHUTDOWN_LOG",
            "REMEMR1_TEST_LAUNCHER_FAIL_AT",
            "FAKE_PIPELINE_RC",
        ]
        existing = env.get("WSLENV", "")
        env["WSLENV"] = ":".join(part for part in [existing, *forwarded] if part)
    return env, launcher_root, shutdown_log, lock_file


def _write_composite_gpu_fixture(
    tmp_path: Path,
    *,
    cpu_finalize_rc: int = 0,
    gpu_gates_rc: int = 0,
    gpu_capacity_rc: int = 0,
    omit_terminal_phase: str = "",
):
    env, launcher_root, shutdown_log, lock_file = _write_cloud_fixture(tmp_path)
    project = tmp_path / "project"
    pipeline = project / "scripts" / "cloud" / "run_pipeline.sh"
    cloud_dir = pipeline.parent
    fixture_library = cloud_dir / "lib"
    fixture_library.mkdir()
    for source in (
        LAUNCH,
        REPO_ROOT / "scripts" / "cloud" / "launcher_worker.sh",
    ):
        shutil.copy2(source, cloud_dir / source.name)
    for name in ("runtime.sh", "lock.sh", "shutdown.sh"):
        shutil.copy2(
            REPO_ROOT / "scripts" / "cloud" / "lib" / name,
            fixture_library / name,
        )
    lock_library = shlex.quote(
        _bash_path(REPO_ROOT / "scripts" / "cloud" / "lib" / "lock.sh")
    )
    pipeline.write_bytes(
        f"""#!/usr/bin/env bash
set -u
source {lock_library}
require_cloud_lock
arguments="$*"
phase=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase) phase="$2"; shift 2 ;;
        *) shift ;;
    esac
done
case "${{phase}}" in
    cpu-finalize) rc="${{FAKE_CPU_FINALIZE_RC}}" ;;
    gpu-gates) rc="${{FAKE_GPU_GATES_RC}}" ;;
    gpu-capacity) rc="${{FAKE_GPU_CAPACITY_RC}}" ;;
    *) exit 91 ;;
esac
run_dir="${{PERSIST_ROOT}}/fake-composite-${{phase}}-${{rc}}"
terminal_dir="${{run_dir}}/terminal"
mkdir -p "${{terminal_dir}}"
printf '%s\n' "${{arguments}}" > "${{run_dir}}/args"
printf '%s\n' "${{phase}}" > "${{run_dir}}/phase"
printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
    "${{phase}}" "${{arguments}}" "${{REMEMR1_RESULT_FILE}}" \
    "${{REMEMR1_LAUNCHER_DIR}}" "${{CUDA_VISIBLE_DEVICES-__unset__}}" \
    "${{NVIDIA_VISIBLE_DEVICES-__unset__}}" \
    "${{REMEMR1_ALLOW_GPU_CPU_PHASE-__unset__}}" \
    "${{HF_HUB_OFFLINE-__unset__}}" "${{HF_DATASETS_OFFLINE-__unset__}}" \
    "${{TRANSFORMERS_OFFLINE-__unset__}}" "${{WANDB_MODE-__unset__}}" \
    "${{rc}}" >> "${{PERSIST_ROOT}}/composite-calls.log"
printf '%s\n' "${{run_dir}}" > "${{REMEMR1_RESULT_FILE}}"
if [[ "${{FAKE_OMIT_TERMINAL_PHASE}}" != "${{phase}}" ]]; then
    printf '%s\n' "${{terminal_dir}}" > "${{REMEMR1_RESULT_FILE}}.terminal"
fi
exit "${{rc}}"
""".encode("ascii")
    )
    env.update(
        {
            "FAKE_CPU_FINALIZE_RC": str(cpu_finalize_rc),
            "FAKE_GPU_GATES_RC": str(gpu_gates_rc),
            "FAKE_GPU_CAPACITY_RC": str(gpu_capacity_rc),
            "FAKE_OMIT_TERMINAL_PHASE": omit_terminal_phase,
            "CUDA_VISIBLE_DEVICES": "base-cuda",
            "NVIDIA_VISIBLE_DEVICES": "base-nvidia",
            "REMEMR1_ALLOW_GPU_CPU_PHASE": "base-allow",
            "HF_HUB_OFFLINE": "base-hf",
            "HF_DATASETS_OFFLINE": "base-datasets",
            "TRANSFORMERS_OFFLINE": "base-transformers",
            "WANDB_MODE": "base-wandb",
        }
    )
    if os.name == "nt":
        forwarded = [
            "FAKE_CPU_FINALIZE_RC",
            "FAKE_GPU_GATES_RC",
            "FAKE_GPU_CAPACITY_RC",
            "FAKE_OMIT_TERMINAL_PHASE",
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "REMEMR1_ALLOW_GPU_CPU_PHASE",
            "HF_HUB_OFFLINE",
            "HF_DATASETS_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "WANDB_MODE",
        ]
        env["WSLENV"] = ":".join(
            part for part in [env.get("WSLENV", ""), *forwarded] if part
        )
    return env, launcher_root, shutdown_log, lock_file


def _launch(env, *args, expected_returncodes=(0,)):
    result = subprocess.run(
        [shutil.which("bash"), _bash_path(LAUNCH), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=15,
    )
    assert result.returncode in expected_returncodes, result.stderr
    fields = dict(
        line.split("=", 1)
        for line in result.stdout.splitlines()
        if "=" in line
    )
    assert "launcher_dir" in fields
    return fields


def _run_public_gpu(env, *args, expected_returncodes=(0,)):
    result = subprocess.run(
        [shutil.which("bash"), _bash_path(RUN_GPU), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
        timeout=15,
    )
    assert result.returncode in expected_returncodes, result.stderr
    fields = dict(
        line.split("=", 1)
        for line in result.stdout.splitlines()
        if "=" in line
    )
    assert "launcher_dir" in fields
    return fields


def _composite_calls(tmp_path: Path):
    rows = (tmp_path / "persist" / "composite-calls.log").read_text(
        encoding="utf-8"
    ).splitlines()
    return [row.split("|") for row in rows]


def _wait_for_terminal(launcher_root: Path, timeout: float = 15.0) -> Path:
    deadline = time.monotonic() + timeout
    launcher = None
    while time.monotonic() < deadline:
        directories = [path for path in launcher_root.iterdir() if path.is_dir()]
        if directories:
            launcher = max(directories, key=lambda path: path.name)
            if (launcher / "exit-code").is_file() and (
                (launcher / ".success").is_file()
                or (launcher / ".failed").is_file()
                or (launcher / ".scientific-stop").is_file()
                or (launcher / ".capacity-stop").is_file()
            ):
                return launcher
        time.sleep(0.05)
    log = ""
    if launcher and (launcher / "launcher.log").exists():
        log = (launcher / "launcher.log").read_text(encoding="utf-8")
    pytest.fail(f"launcher did not finish before timeout; log={log}")


def _events(path: Path):
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def _terminal_json(launcher: Path):
    value = json.loads((launcher / "terminal.json").read_text(encoding="utf-8"))
    assert set(value) == {
        "budget_projection",
        "budget_projection_file_sha256",
        "exit_code",
        "expected_commit",
        "experiment_profile_id",
        "finished_at",
        "offload_profile",
        "outcome",
        "phase",
        "pipeline_result",
        "pipeline_terminal_dir",
        "retry_hint",
        "retryable",
        "schema_version",
        "started_at",
    }
    return value


def test_run_logged_preserves_the_command_exit_status(tmp_path):
    runtime = shlex.quote(
        _bash_path(REPO_ROOT / "scripts" / "cloud" / "lib" / "runtime.sh")
    )
    log_file = shlex.quote(_bash_path(tmp_path / "command.log"))
    probe = tmp_path / "run-logged-probe.sh"
    probe.write_bytes(
        (
            f"source {runtime}\n"
            "set +e\n"
            f"run_logged {log_file} /usr/bin/bash -c 'printf command-output; exit 19'\n"
            "rc=$?\n"
            "[[ ${rc} -eq 19 ]]\n"
        ).encode("ascii")
    )

    result = subprocess.run(
        [shutil.which("bash"), _bash_path(probe)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "command.log").read_text(encoding="ascii") == "command-output"


def test_host_python_selector_falls_back_and_fails_closed(tmp_path):
    runtime = shlex.quote(
        _bash_path(REPO_ROOT / "scripts" / "cloud" / "lib" / "runtime.sh")
    )
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    for name, exit_code in (("python3", 1), ("python", 0)):
        candidate = fake_bin / name
        candidate.write_bytes(f"#!/bin/sh\nexit {exit_code}\n".encode("ascii"))
        candidate.chmod(0o755)

    compatible = shlex.quote(_bash_path(fake_bin / "python"))
    incompatible = shlex.quote(_bash_path(fake_bin / "python3"))
    missing = shlex.quote(_bash_path(tmp_path / "missing-python"))
    missing_prefix = shlex.quote(_bash_path(tmp_path / "missing-prefix"))
    error_file = shlex.quote(_bash_path(tmp_path / "selector.err"))
    fake_path = shlex.quote(_bash_path(fake_bin))
    probe = tmp_path / "host-python-probe.sh"
    probe.write_bytes(
        f"""set -euo pipefail
source {runtime}
[[ "$(rememr1_select_host_python {missing} {compatible})" == {compatible} ]]
[[ "$(rememr1_select_host_python {incompatible} {compatible})" == {compatible} ]]
if rememr1_select_host_python {missing} {incompatible} 2>{error_file}; then
    exit 91
fi
[[ "$(<{error_file})" == "required Python 3.10+ host interpreter is missing" ]]
PATH={fake_path}
[[ "$(rememr1_find_host_python {missing_prefix})" == {compatible} ]]
""".encode("ascii")
    )

    result = subprocess.run(
        [shutil.which("bash"), _bash_path(probe)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr


def test_kernel_source_prefetch_accepts_only_an_empty_unmaterialized_clone():
    source = PREPARE_KERNEL_SOURCES.read_text(encoding="utf-8")

    assert 'git clone --filter=blob:none --no-checkout' in source
    assert '! -e "${destination}/.git/index"' in source
    assert '! -L "${destination}/.git/index"' in source
    assert '! -name .git -print -quit' in source
    assert '"${unmaterialized_checkout}" != yes' in source
    assert "kernel source checkout is dirty after materialization" in source
    assert source.index('git -C "${destination}" checkout --detach') < source.index(
        "kernel source checkout is dirty after materialization"
    )


def test_shutdown_markers_distinguish_skips_from_backend_failure(tmp_path):
    runtime = shlex.quote(
        _bash_path(REPO_ROOT / "scripts" / "cloud" / "lib" / "runtime.sh")
    )
    shutdown = shlex.quote(
        _bash_path(REPO_ROOT / "scripts" / "cloud" / "lib" / "shutdown.sh")
    )
    root = shlex.quote(_bash_path(tmp_path / "shutdown-markers"))
    script = f"""
set -euo pipefail
source {runtime}
source {shutdown}
root={root}
mkdir -p "${{root}}"/authorization "${{root}}"/sync "${{root}}"/revalidation \
    "${{root}}"/backend "${{root}}"/success
declare -gA _REMEMR1_CAPABILITY=([shutdown_backend]=test-backend)
REMEMR1_TEST_MODE=no
unset REMEMR1_TEST_SHUTDOWN_LOG || true

verify_guest_shutdown_authorization() {{ return 1; }}
rememr1_sync_all() {{ return 0; }}
_dispatch_guest_shutdown_backend() {{ : > "${{root}}/authorization/backend-called"; return 0; }}
if request_guest_shutdown "${{root}}/authorization" cpu 23; then exit 91; fi
[[ "$(<"${{root}}/authorization/shutdown-skipped")" == authorization-failed ]]
[[ ! -e "${{root}}/authorization/shutdown-requested" ]]
[[ ! -e "${{root}}/authorization/shutdown-backend" ]]
[[ ! -e "${{root}}/authorization/shutdown-failed" ]]
[[ ! -e "${{root}}/authorization/backend-called" ]]

verify_guest_shutdown_authorization() {{ return 0; }}
rememr1_sync_all() {{ return 1; }}
_dispatch_guest_shutdown_backend() {{ : > "${{root}}/sync/backend-called"; return 0; }}
if request_guest_shutdown "${{root}}/sync" cpu 23; then exit 92; fi
[[ "$(<"${{root}}/sync/shutdown-skipped")" == pre-dispatch-sync-failed ]]
[[ ! -e "${{root}}/sync/shutdown-requested" ]]
[[ ! -e "${{root}}/sync/shutdown-backend" ]]
[[ ! -e "${{root}}/sync/shutdown-failed" ]]
[[ ! -e "${{root}}/sync/backend-called" ]]

rememr1_sync_all() {{ return 0; }}
_dispatch_guest_shutdown_backend() {{ return 2; }}
if request_guest_shutdown "${{root}}/revalidation" gpu-gates 17; then exit 94; fi
[[ "$(<"${{root}}/revalidation/shutdown-skipped")" == backend-revalidation-failed ]]
[[ ! -e "${{root}}/revalidation/shutdown-requested" ]]
[[ ! -e "${{root}}/revalidation/shutdown-backend" ]]
[[ ! -e "${{root}}/revalidation/shutdown-failed" ]]

_dispatch_guest_shutdown_backend() {{
    [[ -f "${{root}}/backend/shutdown-requested" ]] || return 98
    : > "${{root}}/backend/backend-called"
    return 1
}}
if request_guest_shutdown "${{root}}/backend" gpu-gates 17; then exit 93; fi
[[ -f "${{root}}/backend/shutdown-requested" ]]
[[ "$(<"${{root}}/backend/shutdown-backend")" == test-backend ]]
[[ -f "${{root}}/backend/shutdown-failed" ]]
[[ -f "${{root}}/backend/backend-called" ]]
[[ ! -e "${{root}}/backend/shutdown-skipped" ]]

_dispatch_guest_shutdown_backend() {{
    [[ -f "${{root}}/success/shutdown-requested" ]] || return 98
    : > "${{root}}/success/backend-called"
    return 0
}}
request_guest_shutdown "${{root}}/success" gpu-gates 0
[[ -f "${{root}}/success/shutdown-requested" ]]
[[ "$(<"${{root}}/success/shutdown-backend")" == test-backend ]]
[[ -f "${{root}}/success/backend-called" ]]
[[ ! -e "${{root}}/success/shutdown-failed" ]]
[[ ! -e "${{root}}/success/shutdown-skipped" ]]
"""

    probe = tmp_path / "shutdown-marker-probe.sh"
    probe.write_bytes(script.encode("ascii"))
    result = subprocess.run(
        [shutil.which("bash"), _bash_path(probe)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr


def test_autodl_shutdown_backend_is_pinned_and_invoked_without_systemd_arguments(
    tmp_path,
):
    shutdown_source = (
        REPO_ROOT / "scripts" / "cloud" / "lib" / "shutdown.sh"
    ).read_text(encoding="utf-8")
    init_source = (REPO_ROOT / "scripts" / "cloud" / "init_cloud.sh").read_text(
        encoding="utf-8"
    )
    bootstrap_source = BOOTSTRAP.read_text(encoding="utf-8")

    dispatch = shutdown_source.split("_dispatch_guest_shutdown_backend() {", 1)[1].split(
        "\n}", 1
    )[0]
    assert "/usr/bin/env -i PATH=/usr/bin:/bin HOME=/root" in dispatch
    assert "/usr/bin/bash --noprofile --norc /usr/bin/shutdown" in dispatch
    assert "-h now" not in dispatch
    assert "systemctl" not in dispatch
    assert "schema_version=2" in init_source
    assert "SHUTDOWN_BACKEND=\"autodl-wrapper-v1\"" in init_source
    assert "shutdown_backend_sha256=${SHUTDOWN_BACKEND_SHA256}" in init_source
    assert "/usr/bin/bash --noprofile --norc /usr/bin/shutdown" in bootstrap_source

    shutdown = shlex.quote(
        _bash_path(REPO_ROOT / "scripts" / "cloud" / "lib" / "shutdown.sh")
    )
    probe = tmp_path / "shutdown-backend-digest-probe.sh"
    probe.write_bytes(
        f"""
set -euo pipefail
source {shutdown}
_shutdown_autodl_wrapper_sha256() {{ printf '%s\\n' "${{FAKE_DIGEST}}"; }}
FAKE_DIGEST="$(printf 'a%.0s' {{1..64}})"
declare -gA _REMEMR1_CAPABILITY=(
    [shutdown_backend]=autodl-wrapper-v1
    [shutdown_backend_path]=/usr/bin/shutdown
    [shutdown_backend_sha256]="${{FAKE_DIGEST}}"
)
_shutdown_verify_capability_backend
_REMEMR1_CAPABILITY[shutdown_backend_sha256]="$(printf 'b%.0s' {{1..64}})"
if _shutdown_verify_capability_backend; then exit 91; fi
""".encode("ascii")
    )
    result = subprocess.run(
        [shutil.which("bash"), _bash_path(probe)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr


def test_public_gpu_exports_cuda_toolkit_for_noninteractive_shells():
    source = RUN_GPU.read_text(encoding="utf-8")

    assert 'export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"' in source
    assert 'export PATH="${CUDA_HOME}/bin:${PATH}"' in source
    assert source.index("export CUDA_HOME=") < source.index("exec bash")


def test_shutdown_authorizes_all_four_exact_terminal_marker_classes(tmp_path):
    shutdown = shlex.quote(
        _bash_path(REPO_ROOT / "scripts" / "cloud" / "lib" / "shutdown.sh")
    )
    root = shlex.quote(_bash_path(tmp_path / "terminal-markers"))
    script = f"""
set -euo pipefail
source {shutdown}
root={root}
mkdir -p "${{root}}"

for spec in '.success 0' '.failed 17' '.scientific-stop 42' '.capacity-stop 43'; do
    rm -f -- "${{root}}"/.*-stop "${{root}}"/.success "${{root}}"/.failed
    read -r marker exit_code <<< "${{spec}}"
    printf '%s\n' "${{exit_code}}" > "${{root}}/${{marker}}"
    _shutdown_verify_terminal_marker "${{root}}" "${{exit_code}}"
done

rm -f -- "${{root}}"/.*-stop "${{root}}"/.success "${{root}}"/.failed
printf '42\n' > "${{root}}/.failed"
if _shutdown_verify_terminal_marker "${{root}}" 42; then exit 91; fi
rm -f -- "${{root}}/.failed"
printf '43\n' > "${{root}}/.capacity-stop"
printf '43\n' > "${{root}}/.failed"
if _shutdown_verify_terminal_marker "${{root}}" 43; then exit 92; fi
rm -f -- "${{root}}/.capacity-stop" "${{root}}/.failed"
printf '42\n' > "${{root}}/target"
ln -s target "${{root}}/.scientific-stop"
if _shutdown_verify_terminal_marker "${{root}}" 42; then exit 93; fi
"""

    probe = tmp_path / "shutdown-terminal-marker-probe.sh"
    probe.write_bytes(script.encode("ascii"))
    result = subprocess.run(
        [shutil.which("bash"), _bash_path(probe)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr


def test_bootstrap_log_drain_fails_closed_with_a_leaked_fifo_writer(tmp_path):
    source = BOOTSTRAP.read_text(encoding="utf-8")
    start = source.index("drain_bootstrap_log() {")
    end = source.index("\n}\n\nbootstrap_failure_shutdown_authorized()", start) + 2
    drain_function = source[start:end]
    output_dir = shlex.quote(_bash_path(tmp_path / "bootstrap-drain"))
    script = "\n".join(
        [
            "set -euo pipefail",
            drain_function,
            'BOOTSTRAP_RUN="$(mktemp -d /tmp/rememr1-bootstrap-drain.XXXXXX)"',
            f"OUTPUT_DIR={output_dir}",
            'mkdir -p "${OUTPUT_DIR}"',
            'fifo="${BOOTSTRAP_RUN}/log.fifo"',
            'mkfifo "${fifo}"',
            '/usr/bin/tee -a "${BOOTSTRAP_RUN}/bootstrap.log" < "${fifo}" >/dev/null 8>&- &',
            'BOOTSTRAP_TEE_PID="$!"',
            'exec 3>"${fifo}"',
            'exec 1>"${fifo}" 2>&1',
            'rm -f "${fifo}"',
            'BOOTSTRAP_LOG_DRAINED=no',
            'BOOTSTRAP_LOG_DRAIN_TIMEOUT_SECONDS=1',
            'set +e',
            'drain_bootstrap_log',
            'rc="$?"',
            'set -e',
            'exec 3>&-',
            'printf "%s\\n" "${rc}" > "${OUTPUT_DIR}/result"',
            'printf "%s\\n" "${BOOTSTRAP_LOG_DRAINED}" > "${OUTPUT_DIR}/drained"',
            'cp "${BOOTSTRAP_RUN}/bootstrap.log" "${OUTPUT_DIR}/bootstrap.log"',
            'rm -f "${BOOTSTRAP_RUN}/bootstrap.log" "${BOOTSTRAP_RUN}/log-drain-failed"',
            'rmdir "${BOOTSTRAP_RUN}"',
            'exit "${rc}"',
        ]
    )

    started = time.monotonic()
    probe = tmp_path / "bootstrap-drain-probe.sh"
    probe.write_bytes(script.encode("ascii"))
    result = subprocess.run(
        [shutil.which("bash"), _bash_path(probe)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=5,
    )
    elapsed = time.monotonic() - started

    assert result.returncode == 1, result.stderr
    assert elapsed < 4
    assert (tmp_path / "bootstrap-drain" / "result").read_text(encoding="ascii").strip() == "1"
    assert (tmp_path / "bootstrap-drain" / "drained").read_text(encoding="ascii").strip() == "no"
    assert "[bootstrap] terminal-state-published" in (
        tmp_path / "bootstrap-drain" / "bootstrap.log"
    ).read_text(encoding="utf-8")


def test_test_mode_rejects_symlink_alias_to_production_path(tmp_path):
    runtime = shlex.quote(
        _bash_path(REPO_ROOT / "scripts" / "cloud" / "lib" / "runtime.sh")
    )
    script = f"""
set -euo pipefail
alias_dir="$(mktemp -d /tmp/rememr1-test-mode-alias.XXXXXX)"
cleanup() {{
    unlink "${{alias_dir}}/production" 2>/dev/null || true
    rmdir "${{alias_dir}}" 2>/dev/null || true
}}
trap cleanup EXIT
ln -s /root/autodl-tmp "${{alias_dir}}/production"
source {runtime}
REMEMR1_TEST_MODE=yes
REMEMR1_TEST_SHUTDOWN_LOG="${{alias_dir}}/events.log"
REMEMR1_PROJECT_DIR="${{alias_dir}}/production/project"
PERSIST_ROOT="${{alias_dir}}/persist"
if rememr1_validate_test_mode; then
    echo "test mode accepted a production symlink alias" >&2
    exit 99
fi
"""
    probe = tmp_path / "test-mode-symlink-probe.sh"
    probe.write_bytes(script.encode("ascii"))

    result = subprocess.run(
        [shutil.which("bash"), _bash_path(probe)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    assert "test mode is forbidden for production AutoDL paths" in result.stderr


def test_failure_publishes_exit_code_before_shutdown_request(launcher_tmp_path):
    env, launcher_root, shutdown_log, _ = _write_cloud_fixture(
        launcher_tmp_path, pipeline_rc=23
    )

    _launch(
        env,
        "--phase",
        "cpu",
        "--retry-failed-stage",
        expected_returncodes=(0, 23),
    )
    launcher = _wait_for_terminal(launcher_root)

    assert (launcher / "exit-code").read_text(encoding="ascii").strip() == "23"
    assert (launcher / ".failed").is_file()
    assert not (launcher / ".success").exists()
    terminal = _terminal_json(launcher)
    assert terminal["outcome"] == "failed"
    assert terminal["retryable"] is True
    assert "--retry-failed-stage" in terminal["retry_hint"]
    assert (launcher / "retryable").read_text(encoding="ascii").strip() == "true"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not (launcher / "shutdown-safe").exists():
        time.sleep(0.05)
    assert (launcher / "shutdown-safe").is_file()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not any(
        "shutdown-request " in event for event in _events(shutdown_log)
    ):
        time.sleep(0.05)
    events = _events(shutdown_log)
    exit_index = next(i for i, event in enumerate(events) if "exit-code-written " in event)
    safe_index = next(
        i for i, event in enumerate(events) if "shutdown-safe-written " in event
    )
    shutdown_index = next(i for i, event in enumerate(events) if "shutdown-request " in event)
    assert exit_index < safe_index < shutdown_index
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not (launcher / "shutdown-skipped").exists():
        time.sleep(0.05)
    assert (launcher / "shutdown-skipped").read_text(encoding="ascii").strip() == "test-mode"
    assert not (launcher / "shutdown-dispatched").exists()
    assert "terminal-state-published exit_code=23 outcome=failed" in (
        launcher / "launcher.log"
    ).read_text(encoding="utf-8")
    result_dir = Path((launcher / "pipeline-result").read_text(encoding="utf-8").strip())
    if os.name == "nt":
        # Read the args through the Windows-visible persistent directory instead.
        result_dir = launcher_tmp_path / "persist" / "fake-pipeline-23"
    assert "--phase cpu" in (result_dir / "args").read_text(encoding="ascii")
    assert "--retry-failed-stage" in (result_dir / "args").read_text(encoding="ascii")


def test_r1_launch_binds_marker_bytes_and_failure_requires_new_nonce(
    launcher_tmp_path,
):
    env, launcher_root, _, _ = _write_cloud_fixture(
        launcher_tmp_path, pipeline_rc=23
    )
    persist = launcher_tmp_path / "persist"
    marker = persist / "r1-approval.json"
    budget = persist / "budget.json"
    marker.write_bytes(b'{"status":"approved-once"}\n')
    budget.write_bytes(b'{"projection":"test"}\n')

    _launch(
        env,
        "--phase",
        "gpu-capacity",
        "--offload-profile",
        "r1",
        "--r1-approval",
        _bash_path(marker),
        "--budget-projection",
        _bash_path(budget),
        expected_returncodes=(0, 23),
    )
    launcher = _wait_for_terminal(launcher_root)
    request = json.loads((launcher / "request.json").read_text(encoding="utf-8"))
    expected_sha = hashlib.sha256(marker.read_bytes()).hexdigest()
    assert request["r1_approval_file_sha256"] == expected_sha
    terminal = _terminal_json(launcher)
    assert terminal["retryable"] is False
    assert "new R1 approval nonce" in terminal["retry_hint"]
    assert "retryable=false" in (launcher / "terminal").read_text(encoding="utf-8")


def test_bc_failure_requires_new_budget_generation(launcher_tmp_path):
    env, launcher_root, _, _ = _write_cloud_fixture(
        launcher_tmp_path, pipeline_rc=23
    )
    budget = launcher_tmp_path / "persist" / "budget.json"
    budget.write_bytes(b'{"projection":"test"}\n')

    _launch(
        env,
        "--phase",
        "gpu-bc40",
        "--budget-projection",
        _bash_path(budget),
        expected_returncodes=(0, 23),
    )
    terminal = _terminal_json(_wait_for_terminal(launcher_root))
    assert terminal["retryable"] is False
    assert "new budget projection" in terminal["retry_hint"]
    assert "new immutable generation" in terminal["retry_hint"]


def test_success_publishes_durable_state_before_shutdown_request(launcher_tmp_path):
    env, launcher_root, shutdown_log, _ = _write_cloud_fixture(launcher_tmp_path)

    _launch(env, "cpu")
    launcher = _wait_for_terminal(launcher_root)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not (launcher / "shutdown-skipped").exists():
        time.sleep(0.05)

    assert (launcher / "exit-code").read_text(encoding="ascii").strip() == "0"
    assert (launcher / ".success").is_file()
    terminal = _terminal_json(launcher)
    assert terminal["outcome"] == "success"
    assert terminal["retryable"] is False
    assert (launcher / "retryable").read_text(encoding="ascii").strip() == "false"
    assert (launcher / "shutdown-safe").is_file()
    assert (launcher / "shutdown-skipped").read_text(encoding="ascii").strip() == "test-mode"
    assert not (launcher / "shutdown-dispatched").exists()
    events = _events(shutdown_log)
    written_index = next(
        i for i, event in enumerate(events) if "terminal-state-written " in event
    )
    terminal_sync_index = next(
        i
        for i, event in enumerate(events)
        if "launcher-sync-complete point=terminal-state-sync " in event
    )
    removed_index = next(
        i for i, event in enumerate(events) if "running-marker-removed " in event
    )
    removal_sync_index = next(
        i
        for i, event in enumerate(events)
        if "launcher-sync-complete point=running-marker-sync " in event
    )
    exit_index = next(i for i, event in enumerate(events) if "exit-code-written " in event)
    safe_index = next(
        i for i, event in enumerate(events) if "shutdown-safe-written " in event
    )
    shutdown_index = next(i for i, event in enumerate(events) if "shutdown-request " in event)
    assert (
        written_index
        < terminal_sync_index
        < removed_index
        < removal_sync_index
        < exit_index
        < safe_index
        < shutdown_index
    )


def test_scientific_stop_is_non_retryable_and_shutdown_safe(launcher_tmp_path):
    env, launcher_root, shutdown_log, _ = _write_cloud_fixture(
        launcher_tmp_path, pipeline_rc=42
    )

    _launch(env, "cpu", expected_returncodes=(0, 42))
    launcher = _wait_for_terminal(launcher_root)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not (launcher / "shutdown-skipped").exists():
        time.sleep(0.05)

    assert (launcher / ".scientific-stop").read_text(encoding="ascii").strip() == "42"
    assert not (launcher / ".failed").exists()
    assert not (launcher / ".success").exists()
    assert (launcher / "retryable").read_text(encoding="ascii").strip() == "false"
    terminal = _terminal_json(launcher)
    assert terminal["outcome"] == "scientific-stop"
    assert terminal["exit_code"] == 42
    assert terminal["retryable"] is False
    assert (launcher / "shutdown-safe").is_file()
    events = _events(shutdown_log)
    terminal_index = next(
        i for i, event in enumerate(events) if "terminal-state-written exit_code=42" in event
    )
    removed_index = next(
        i for i, event in enumerate(events) if "running-marker-removed " in event
    )
    safe_index = next(
        i for i, event in enumerate(events) if "shutdown-safe-written " in event
    )
    shutdown_index = next(i for i, event in enumerate(events) if "shutdown-request " in event)
    assert terminal_index < removed_index < safe_index < shutdown_index


def test_capacity_stop_is_non_retryable_and_shutdown_safe(launcher_tmp_path):
    env, launcher_root, shutdown_log, _ = _write_cloud_fixture(
        launcher_tmp_path, pipeline_rc=43
    )

    _launch(env, "cpu", expected_returncodes=(0, 43))
    launcher = _wait_for_terminal(launcher_root)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not (launcher / "shutdown-skipped").exists():
        time.sleep(0.05)

    assert (launcher / ".capacity-stop").read_text(encoding="ascii").strip() == "43"
    assert not (launcher / ".failed").exists()
    assert not (launcher / ".scientific-stop").exists()
    assert (launcher / "retryable").read_text(encoding="ascii").strip() == "false"
    terminal = _terminal_json(launcher)
    assert terminal["outcome"] == "capacity-stop"
    assert terminal["exit_code"] == 43
    assert terminal["retryable"] is False
    assert (launcher / "shutdown-safe").is_file()
    events = _events(shutdown_log)
    terminal_index = next(
        i for i, event in enumerate(events) if "terminal-state-written exit_code=43" in event
    )
    removed_index = next(
        i for i, event in enumerate(events) if "running-marker-removed " in event
    )
    safe_index = next(
        i for i, event in enumerate(events) if "shutdown-safe-written " in event
    )
    shutdown_index = next(i for i, event in enumerate(events) if "shutdown-request " in event)
    assert terminal_index < removed_index < safe_index < shutdown_index


@pytest.mark.parametrize(
    ("failure_point", "running_remains", "expected_event", "forbidden_event"),
    [
        (
            "terminal-state-sync",
            True,
            "launcher-sync-failed point=terminal-state-sync ",
            "running-marker-removed ",
        ),
        (
            "running-marker-remove",
            True,
            "running-marker-remove-failed ",
            "launcher-sync-started point=running-marker-sync ",
        ),
        (
            "running-marker-sync",
            False,
            "launcher-sync-failed point=running-marker-sync ",
            "launcher-sync-complete point=running-marker-sync ",
        ),
    ],
)
def test_terminal_durability_failures_never_become_shutdown_safe(
    launcher_tmp_path,
    failure_point,
    running_remains,
    expected_event,
    forbidden_event,
):
    env, launcher_root, shutdown_log, _ = _write_cloud_fixture(
        launcher_tmp_path, pipeline_rc=23
    )
    env["REMEMR1_TEST_LAUNCHER_FAIL_AT"] = failure_point

    _launch(env, "cpu", expected_returncodes=(0, 23))
    launcher = _wait_for_terminal(launcher_root)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not (launcher / "shutdown-skipped").exists():
        time.sleep(0.05)

    assert (launcher / "exit-code").read_text(encoding="ascii").strip() == "23"
    assert (launcher / ".failed").is_file()
    assert (launcher / ".running").exists() is running_remains
    assert (launcher / "persistence-failed").is_file()
    assert not (launcher / "reserve-released").exists()
    assert not (launcher / "shutdown-safe").exists()
    assert (
        launcher / "shutdown-skipped"
    ).read_text(encoding="ascii").strip() == "durable-state-sync-failed"
    events = _events(shutdown_log)
    assert any(expected_event in event for event in events)
    assert not any(forbidden_event in event for event in events)
    assert not any("shutdown-request " in event for event in events)


def test_zero_length_terminal_reserve_is_rearmed_after_lock(launcher_tmp_path):
    env, launcher_root, _, _ = _write_cloud_fixture(launcher_tmp_path)
    reserve = launcher_tmp_path / "persist" / "cloud" / "terminal-reserve"
    reserve.parent.mkdir(parents=True)
    reserve.write_bytes(b"")

    _launch(env, "cpu", "--keep-running")
    launcher = _wait_for_terminal(launcher_root)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not (launcher / "shutdown-skipped").exists():
        time.sleep(0.05)

    assert (launcher / ".success").is_file()
    assert (launcher / "shutdown-skipped").read_text(encoding="ascii").strip() == "keep-running"
    assert reserve.stat().st_size >= 8 * 1024 * 1024


@pytest.mark.parametrize("option, expected_reason", [("--keep-running", "keep-running"), ("--dry-run", "dry-run")])
def test_non_shutdown_modes_never_request_shutdown(
    launcher_tmp_path, option, expected_reason
):
    env, launcher_root, shutdown_log, _ = _write_cloud_fixture(launcher_tmp_path)

    _launch(env, "cpu", option)
    launcher = _wait_for_terminal(launcher_root)

    assert (launcher / "exit-code").read_text(encoding="ascii").strip() == "0"
    assert (launcher / ".success").is_file()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not (launcher / "shutdown-skipped").exists():
        time.sleep(0.05)
    assert (launcher / "shutdown-skipped").read_text(encoding="ascii").strip() == expected_reason
    assert not any("shutdown-request " in event for event in _events(shutdown_log))


def test_busy_lock_is_terminal_and_never_requests_shutdown(launcher_tmp_path):
    env, launcher_root, shutdown_log, lock_file = _write_cloud_fixture(
        launcher_tmp_path
    )
    reserve = launcher_tmp_path / "persist" / "cloud" / "terminal-reserve"
    reserve.parent.mkdir(parents=True)
    reserve.write_bytes(b"")
    lock_command = (
        f"exec 9>>{shlex.quote(_bash_path(lock_file))}; "
        "flock -n 9 || exit 1; printf 'locked\\n'; sleep 30"
    )
    holder = subprocess.Popen(
        [shutil.which("bash"), "-lc", lock_command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "locked"
        _launch(env, "gpu-gates", expected_returncodes=(75,))
        launcher = _wait_for_terminal(launcher_root)
        assert (launcher / "exit-code").read_text(encoding="ascii").strip() == "75"
        assert (launcher / "lock-busy").is_file()
        assert (launcher / ".failed").is_file()
        assert not (launcher / "shutdown-safe").exists()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not (launcher / "shutdown-skipped").exists():
            time.sleep(0.05)
        assert (launcher / "shutdown-skipped").read_text(encoding="ascii").strip() == "lock-not-acquired"
        assert not any("shutdown-request " in event for event in _events(shutdown_log))
        assert reserve.stat().st_size == 0
    finally:
        holder.terminate()
        try:
            holder.wait(timeout=5)
        except subprocess.TimeoutExpired:
            holder.kill()
            holder.wait(timeout=5)


def test_status_uses_each_launchers_scoped_terminal_pointer(launcher_tmp_path):
    env, launcher_root, _, _ = _write_cloud_fixture(launcher_tmp_path)
    pipeline = launcher_tmp_path / "persist" / "cloud" / "pipelines" / "shared"
    first_state = pipeline / "terminals" / "gpu-bc40" / "r0" / "budget-first"
    second_state = pipeline / "terminals" / "gpu-bc80" / "r0" / "budget-second"
    first_state.mkdir(parents=True)
    second_state.mkdir(parents=True)
    (first_state / "failed-stage").write_text("b20\n", encoding="ascii")
    (second_state / "failed-stage").write_text("b60\n", encoding="ascii")
    (pipeline / "last-terminal").write_bytes(
        (_bash_path(second_state) + "\n").encode("ascii")
    )

    outputs = []
    for name, state in (("first", first_state), ("second", second_state)):
        launcher = launcher_root / name
        launcher.mkdir()
        (launcher / ".failed").write_text("23\n", encoding="ascii")
        (launcher / "pipeline-result").write_bytes(
            (_bash_path(pipeline) + "\n").encode("ascii")
        )
        (launcher / "pipeline-result.terminal").write_bytes(
            (_bash_path(state) + "\n").encode("ascii")
        )
        result = subprocess.run(
            [shutil.which("bash"), _bash_path(STATUS), _bash_path(launcher)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            check=False,
            timeout=15,
        )
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout)

    assert "pipeline_failed-stage=b20" in outputs[0]
    assert "pipeline_failed-stage=b60" not in outputs[0]
    assert "pipeline_failed-stage=b60" in outputs[1]


def test_public_gpu_runs_finalize_gates_and_r0_capacity_under_one_launcher(
    launcher_tmp_path,
):
    env, launcher_root, _, _ = _write_composite_gpu_fixture(launcher_tmp_path)

    _run_public_gpu(env, "--retry-failed-stage", "--keep-running")
    launcher = _wait_for_terminal(launcher_root)
    calls = _composite_calls(launcher_tmp_path)

    assert [call[0] for call in calls] == [
        "cpu-finalize",
        "gpu-gates",
        "gpu-capacity",
    ]
    assert all("--retry-failed-stage" in call[1] for call in calls)
    assert "--offload-profile r0" in calls[2][1]
    assert calls[0][4:11] == ["", "void", "yes", "1", "1", "1", "disabled"]
    assert calls[1][4:11] == [
        "base-cuda",
        "base-nvidia",
        "base-allow",
        "base-hf",
        "base-datasets",
        "base-transformers",
        "base-wandb",
    ]
    assert calls[2][4:11] == calls[1][4:11]
    assert len({call[2] for call in calls}) == 3
    assert len({call[3] for call in calls}) == 3
    assert all(launcher.name in Path(call[3]).name for call in calls)

    final_result = _bash_path(
        launcher_tmp_path / "persist" / "fake-composite-gpu-capacity-0"
    )
    final_terminal = f"{final_result}/terminal"
    assert (launcher / "pipeline-result").read_text(encoding="ascii").strip() == final_result
    assert (
        launcher / "pipeline-result.terminal"
    ).read_text(encoding="ascii").strip() == final_terminal
    assert (
        launcher / "pipeline-result.gpu-capacity"
    ).read_text(encoding="ascii").strip() == final_result
    request = json.loads((launcher / "request.json").read_text(encoding="utf-8"))
    assert request["phase"] == "gpu"
    assert request["offload_profile"] == "r0"
    terminal = _terminal_json(launcher)
    assert terminal["phase"] == "gpu"
    assert terminal["offload_profile"] == "r0"
    assert terminal["pipeline_result"] == final_result
    assert terminal["pipeline_terminal_dir"] == final_terminal
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not (launcher / "shutdown-skipped").exists():
        time.sleep(0.05)
    assert (launcher / "shutdown-skipped").read_text(encoding="ascii").strip() == "keep-running"


@pytest.mark.parametrize(
    (
        "cpu_finalize_rc",
        "gpu_gates_rc",
        "gpu_capacity_rc",
        "expected_phases",
        "expected_rc",
        "terminal_marker",
    ),
    [
        (23, 0, 0, ["cpu-finalize"], 23, ".failed"),
        (0, 42, 0, ["cpu-finalize", "gpu-gates"], 42, ".scientific-stop"),
        (
            0,
            0,
            43,
            ["cpu-finalize", "gpu-gates", "gpu-capacity"],
            43,
            ".capacity-stop",
        ),
    ],
)
def test_public_gpu_short_circuits_and_preserves_the_failing_subphase(
    launcher_tmp_path,
    cpu_finalize_rc,
    gpu_gates_rc,
    gpu_capacity_rc,
    expected_phases,
    expected_rc,
    terminal_marker,
):
    env, launcher_root, _, _ = _write_composite_gpu_fixture(
        launcher_tmp_path,
        cpu_finalize_rc=cpu_finalize_rc,
        gpu_gates_rc=gpu_gates_rc,
        gpu_capacity_rc=gpu_capacity_rc,
    )

    _run_public_gpu(env, "--keep-running", expected_returncodes=(0, expected_rc))
    launcher = _wait_for_terminal(launcher_root)
    calls = _composite_calls(launcher_tmp_path)

    assert [call[0] for call in calls] == expected_phases
    failed_phase = expected_phases[-1]
    expected_result = _bash_path(
        launcher_tmp_path
        / "persist"
        / f"fake-composite-{failed_phase}-{expected_rc}"
    )
    assert (launcher / "pipeline-result").read_text(encoding="ascii").strip() == expected_result
    assert (
        launcher / "pipeline-result.terminal"
    ).read_text(encoding="ascii").strip() == f"{expected_result}/terminal"
    assert (launcher / "composite-subphase").read_text(encoding="ascii").strip() == failed_phase
    assert (launcher / "exit-code").read_text(encoding="ascii").strip() == str(expected_rc)
    assert (launcher / terminal_marker).read_text(encoding="ascii").strip() == str(
        expected_rc
    )
    terminal = _terminal_json(launcher)
    assert terminal["exit_code"] == expected_rc
    assert terminal["pipeline_result"] == expected_result
    if expected_rc in {42, 43}:
        assert terminal["retryable"] is False


def test_public_gpu_does_not_reuse_a_stale_terminal_pointer(launcher_tmp_path):
    env, launcher_root, _, _ = _write_composite_gpu_fixture(
        launcher_tmp_path,
        omit_terminal_phase="gpu-capacity",
    )

    _run_public_gpu(env, "--keep-running", expected_returncodes=(0, 74))
    launcher = _wait_for_terminal(launcher_root)

    final_result = _bash_path(
        launcher_tmp_path / "persist" / "fake-composite-gpu-capacity-0"
    )
    assert (launcher / "pipeline-result").read_text(encoding="ascii").strip() == final_result
    assert not (launcher / "pipeline-result.terminal").exists()
    terminal = _terminal_json(launcher)
    assert terminal["exit_code"] == 74
    assert terminal["outcome"] == "failed"
    assert terminal["pipeline_result"] == final_result
    assert terminal["pipeline_terminal_dir"] == ""
    assert (launcher / "composite-result-transfer-failed").is_file()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not (launcher / "shutdown-skipped").exists():
        time.sleep(0.05)
    assert (
        launcher / "shutdown-skipped"
    ).read_text(encoding="ascii").strip() == "pipeline-state-incomplete"


def test_public_gpu_forwards_dry_run_and_retry_to_every_subphase(launcher_tmp_path):
    env, launcher_root, _, _ = _write_composite_gpu_fixture(launcher_tmp_path)

    _run_public_gpu(env, "--dry-run", "--retry-failed-stage")
    launcher = _wait_for_terminal(launcher_root)
    calls = _composite_calls(launcher_tmp_path)

    assert [call[0] for call in calls] == [
        "cpu-finalize",
        "gpu-gates",
        "gpu-capacity",
    ]
    assert all("--dry-run" in call[1] for call in calls)
    assert all("--retry-failed-stage" in call[1] for call in calls)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not (launcher / "shutdown-skipped").exists():
        time.sleep(0.05)
    assert (launcher / "shutdown-skipped").read_text(encoding="ascii").strip() == "dry-run"


@pytest.mark.parametrize(
    "arguments",
    [
        ("--offload-profile", "r0"),
        ("--r1-approval", "/tmp/forbidden-r1-approval.json"),
        ("--budget-projection", "/tmp/forbidden-budget.json"),
    ],
)
def test_public_gpu_rejects_profile_and_paid_phase_authority(
    launcher_tmp_path,
    arguments,
):
    env, launcher_root, _, _ = _write_composite_gpu_fixture(launcher_tmp_path)

    result = subprocess.run(
        [shutil.which("bash"), _bash_path(RUN_GPU), *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
        timeout=15,
    )

    assert result.returncode == 2
    assert "gpu is fixed to R0" in result.stderr
    assert list(launcher_root.iterdir()) == []
