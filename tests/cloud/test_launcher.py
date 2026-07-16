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
            "FAKE_PIPELINE_RC",
        ]
        existing = env.get("WSLENV", "")
        env["WSLENV"] = ":".join(part for part in [existing, *forwarded] if part)
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


def _wait_for_terminal(launcher_root: Path, timeout: float = 15.0) -> Path:
    deadline = time.monotonic() + timeout
    launcher = None
    while time.monotonic() < deadline:
        directories = [path for path in launcher_root.iterdir() if path.is_dir()]
        if directories:
            launcher = max(directories, key=lambda path: path.name)
            if (launcher / "exit-code").is_file() and (
                (launcher / ".success").is_file() or (launcher / ".failed").is_file()
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
        check=False,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "command.log").read_text(encoding="ascii") == "command-output"


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
mkdir -p "${{root}}"/authorization "${{root}}"/sync "${{root}}"/backend "${{root}}"/success
REMEMR1_TEST_MODE=no
unset REMEMR1_TEST_SHUTDOWN_LOG || true

verify_guest_shutdown_authorization() {{ return 1; }}
rememr1_sync_all() {{ return 0; }}
_dispatch_guest_shutdown_backend() {{ : > "${{root}}/authorization/backend-called"; return 0; }}
if request_guest_shutdown "${{root}}/authorization" cpu 23; then exit 91; fi
[[ "$(<"${{root}}/authorization/shutdown-skipped")" == authorization-failed ]]
[[ ! -e "${{root}}/authorization/shutdown-requested" ]]
[[ ! -e "${{root}}/authorization/shutdown-failed" ]]
[[ ! -e "${{root}}/authorization/backend-called" ]]

verify_guest_shutdown_authorization() {{ return 0; }}
rememr1_sync_all() {{ return 1; }}
_dispatch_guest_shutdown_backend() {{ : > "${{root}}/sync/backend-called"; return 0; }}
if request_guest_shutdown "${{root}}/sync" cpu 23; then exit 92; fi
[[ "$(<"${{root}}/sync/shutdown-skipped")" == pre-dispatch-sync-failed ]]
[[ ! -e "${{root}}/sync/shutdown-requested" ]]
[[ ! -e "${{root}}/sync/shutdown-failed" ]]
[[ ! -e "${{root}}/sync/backend-called" ]]

rememr1_sync_all() {{ return 0; }}
_dispatch_guest_shutdown_backend() {{
    [[ -f "${{root}}/backend/shutdown-requested" ]] || return 98
    : > "${{root}}/backend/backend-called"
    return 1
}}
if request_guest_shutdown "${{root}}/backend" gpu-gates 17; then exit 93; fi
[[ -f "${{root}}/backend/shutdown-requested" ]]
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


def test_success_publishes_durable_state_before_shutdown_request(launcher_tmp_path):
    env, launcher_root, shutdown_log, _ = _write_cloud_fixture(launcher_tmp_path)

    _launch(env, "cpu")
    launcher = _wait_for_terminal(launcher_root)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not (launcher / "shutdown-skipped").exists():
        time.sleep(0.05)

    assert (launcher / "exit-code").read_text(encoding="ascii").strip() == "0"
    assert (launcher / ".success").is_file()
    assert (launcher / "shutdown-safe").is_file()
    assert (launcher / "shutdown-skipped").read_text(encoding="ascii").strip() == "test-mode"
    assert not (launcher / "shutdown-dispatched").exists()
    events = _events(shutdown_log)
    exit_index = next(i for i, event in enumerate(events) if "exit-code-written " in event)
    safe_index = next(
        i for i, event in enumerate(events) if "shutdown-safe-written " in event
    )
    shutdown_index = next(i for i, event in enumerate(events) if "shutdown-request " in event)
    assert exit_index < safe_index < shutdown_index


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
