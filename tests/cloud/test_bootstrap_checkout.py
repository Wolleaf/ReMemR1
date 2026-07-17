import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
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
        [
            bash,
            "-lc",
            "command -v git >/dev/null && command -v timeout >/dev/null",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(
    not _bash_available(), reason="Linux bash/git/timeout required"
)


def _run_bash(script: str, *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", encoding="utf-8", newline="\n", delete=False
    ) as handle:
        handle.write(script)
        script_path = Path(handle.name)
    try:
        return subprocess.run(
            [shutil.which("bash"), _bash_path(script_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout,
        )
    finally:
        script_path.unlink(missing_ok=True)


def _checkout_functions() -> str:
    source = BOOTSTRAP.read_text(encoding="utf-8")
    start = source.index("bootstrap_checkout_exact_revision() {")
    end = source.index('\n\nif [[ -z "${EXPECTED_COMMIT}" ]]', start)
    return source[start:end]


def _create_local_remote(tmp_path: Path) -> tuple[Path, Path, str]:
    remote = tmp_path / "remote.git"
    source = tmp_path / "source"
    script = f"""
set -euo pipefail
git init --quiet --bare --initial-branch=main {shlex.quote(_bash_path(remote))}
git init --quiet --initial-branch=main {shlex.quote(_bash_path(source))}
git -C {shlex.quote(_bash_path(source))} config user.name rememr1-test
git -C {shlex.quote(_bash_path(source))} config user.email rememr1-test@example.invalid
git -C {shlex.quote(_bash_path(source))} config core.autocrlf false
printf '%s\n' first > {shlex.quote(_bash_path(source / 'payload.txt'))}
git -C {shlex.quote(_bash_path(source))} add payload.txt
git -C {shlex.quote(_bash_path(source))} commit --quiet -m first
git -C {shlex.quote(_bash_path(source))} remote add origin {shlex.quote(_bash_path(remote))}
git -C {shlex.quote(_bash_path(source))} push --quiet --set-upstream origin main
git -C {shlex.quote(_bash_path(source))} rev-parse HEAD
"""
    result = _run_bash(script)
    assert result.returncode == 0, result.stderr
    return remote, source, result.stdout.strip().splitlines()[-1]


def _commit_and_push(source: Path, value: str) -> str:
    script = f"""
set -euo pipefail
printf '%s\n' {shlex.quote(value)} > {shlex.quote(_bash_path(source / 'payload.txt'))}
git -C {shlex.quote(_bash_path(source))} add payload.txt
git -C {shlex.quote(_bash_path(source))} commit --quiet -m {shlex.quote(value)}
git -C {shlex.quote(_bash_path(source))} push --quiet origin main
git -C {shlex.quote(_bash_path(source))} rev-parse HEAD
"""
    result = _run_bash(script)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()[-1]


def _prepare_checkout(remote: Path, expected_commit: str, project: Path):
    script = f"""
set -euo pipefail
{_checkout_functions()}
prepare_cloud_checkout \
    {shlex.quote(_bash_path(remote))} \
    {shlex.quote(expected_commit)} \
    {shlex.quote(_bash_path(project))}
"""
    return _run_bash(script)


def _create_incomplete_checkout(remote: Path, project: Path):
    result = _run_bash(
        "git clone --quiet --no-checkout "
        f"{shlex.quote(_bash_path(remote))} {shlex.quote(_bash_path(project))}\n"
    )
    assert result.returncode == 0, result.stderr


def _recover_checkout(remote: Path, project: Path):
    script = f"""
set -euo pipefail
{_checkout_functions()}
recover_incomplete_checkout \
    {shlex.quote(_bash_path(remote))} \
    {shlex.quote(_bash_path(project))}
"""
    return _run_bash(script)


def _recover_and_prepare(
    remote: Path, expected_commit: str, project: Path
) -> subprocess.CompletedProcess[str]:
    script = f"""
set -euo pipefail
{_checkout_functions()}
recover_incomplete_checkout \
    {shlex.quote(_bash_path(remote))} \
    {shlex.quote(_bash_path(project))}
prepare_cloud_checkout \
    {shlex.quote(_bash_path(remote))} \
    {shlex.quote(expected_commit)} \
    {shlex.quote(_bash_path(project))}
"""
    return _run_bash(script)


def _git(project: Path, arguments: str):
    return _run_bash(
        f"git -C {shlex.quote(_bash_path(project))} {arguments}"
    )


def test_first_clone_checks_out_exact_commit_before_atomic_publication(tmp_path):
    remote, _, expected_commit = _create_local_remote(tmp_path)
    project = tmp_path / "project"

    result = _prepare_checkout(remote, expected_commit, project)

    assert result.returncode == 0, result.stderr
    assert _git(project, "rev-parse HEAD").stdout.strip() == expected_commit
    assert _git(project, "symbolic-ref --quiet HEAD").returncode == 1
    assert _git(project, "status --porcelain --untracked-files=all").stdout == ""
    assert (project / "payload.txt").read_text(encoding="ascii") == "first\n"
    assert list(tmp_path.glob("project.clone-staging.*")) == []


def test_failed_first_checkout_preserves_staging_without_occupying_project(tmp_path):
    remote, _, _ = _create_local_remote(tmp_path)
    project = tmp_path / "project"

    result = _prepare_checkout(remote, "f" * 40, project)

    assert result.returncode != 0
    assert "staging preserved for diagnosis" in result.stderr
    assert not project.exists()
    staging = list(tmp_path.glob("project.clone-staging.*"))
    assert len(staging) == 1
    assert (staging[0] / ".git").is_dir()


@pytest.mark.parametrize("dirty_kind", ["tracked", "untracked"])
def test_existing_dirty_checkout_is_rejected_before_fetch_or_checkout(
    tmp_path, dirty_kind
):
    remote, source, first_commit = _create_local_remote(tmp_path)
    project = tmp_path / "project"
    first = _prepare_checkout(remote, first_commit, project)
    assert first.returncode == 0, first.stderr
    second_commit = _commit_and_push(source, "second")
    fetch_head_before = (project / ".git" / "FETCH_HEAD").read_bytes()
    if dirty_kind == "tracked":
        (project / "payload.txt").write_text("locally modified\n", encoding="ascii")
    else:
        (project / "local-only.txt").write_text("untracked\n", encoding="ascii")

    result = _prepare_checkout(remote, second_commit, project)

    assert result.returncode != 0
    assert "refusing to change a dirty cloud checkout" in result.stderr
    assert _git(project, "rev-parse HEAD").stdout.strip() == first_commit
    assert (project / ".git" / "FETCH_HEAD").read_bytes() == fetch_head_before


def test_existing_clean_checkout_can_advance_to_exact_commit(tmp_path):
    remote, source, first_commit = _create_local_remote(tmp_path)
    project = tmp_path / "project"
    first = _prepare_checkout(remote, first_commit, project)
    assert first.returncode == 0, first.stderr
    second_commit = _commit_and_push(source, "second")

    result = _prepare_checkout(remote, second_commit, project)

    assert result.returncode == 0, result.stderr
    assert _git(project, "rev-parse HEAD").stdout.strip() == second_commit
    assert _git(project, "symbolic-ref --quiet HEAD").returncode == 1
    assert _git(project, "status --porcelain --untracked-files=all").stdout == ""
    assert (project / "payload.txt").read_text(encoding="ascii") == "second\n"


def test_explicit_recovery_preserves_old_no_checkout_clone_then_rebuilds(tmp_path):
    remote, _, expected_commit = _create_local_remote(tmp_path)
    project = tmp_path / "project"
    _create_incomplete_checkout(remote, project)
    assert list(project.iterdir()) == [project / ".git"]

    result = _recover_and_prepare(remote, expected_commit, project)

    assert result.returncode == 0, result.stderr
    assert "Preserved incomplete checkout at" in result.stdout
    backups = list(tmp_path.glob("project.incomplete-checkout-backup.*"))
    assert len(backups) == 1
    assert list(backups[0].iterdir()) == [backups[0] / ".git"]
    assert _git(project, "rev-parse HEAD").stdout.strip() == expected_commit
    assert _git(project, "status --porcelain --untracked-files=all").stdout == ""


def test_recovery_flag_is_idempotent_for_absent_and_clean_checkout(tmp_path):
    remote, _, expected_commit = _create_local_remote(tmp_path)
    project = tmp_path / "project"

    first = _recover_and_prepare(remote, expected_commit, project)
    second = _recover_and_prepare(remote, expected_commit, project)

    assert first.returncode == 0, first.stderr
    assert "No incomplete checkout needs recovery" in first.stdout
    assert second.returncode == 0, second.stderr
    assert "Existing checkout is already clean; no recovery needed" in second.stdout
    assert _git(project, "rev-parse HEAD").stdout.strip() == expected_commit
    assert list(tmp_path.glob("project.incomplete-checkout-backup.*")) == []


def test_recovery_syncs_quarantine_before_reporting_success():
    source = BOOTSTRAP.read_text(encoding="utf-8")
    start = source.index("recover_incomplete_checkout() {")
    end = source.index("\n}\n\nprepare_cloud_checkout()", start)
    recovery = source[start:end]

    assert recovery.index('mv -T -- "${project_dir}" "${backup_path}"') < recovery.index(
        'sync -f "${backup_path}"'
    )
    assert recovery.index('sync -f "$(dirname "${backup_path}")"') < recovery.index(
        'echo "Preserved incomplete checkout at ${backup_path}"'
    )


@pytest.mark.parametrize(
    "unsafe_state",
    ["worktree-file", "wrong-origin", "git-lock", "changed-index", "skip-worktree"],
)
def test_explicit_recovery_refuses_any_state_other_than_old_clone_bug(
    tmp_path, unsafe_state
):
    remote, _, _ = _create_local_remote(tmp_path)
    project = tmp_path / "project"
    _create_incomplete_checkout(remote, project)
    if unsafe_state == "worktree-file":
        (project / "user-data.txt").write_text("keep me\n", encoding="ascii")
    elif unsafe_state == "wrong-origin":
        changed = _git(project, "remote set-url origin https://example.invalid/other.git")
        assert changed.returncode == 0, changed.stderr
    elif unsafe_state == "git-lock":
        (project / ".git" / "index.lock").write_bytes(b"")
    elif unsafe_state == "changed-index":
        changed = _git(project, "read-tree HEAD")
        assert changed.returncode == 0, changed.stderr
    else:
        populated = _git(project, "read-tree HEAD")
        assert populated.returncode == 0, populated.stderr
        changed = _git(project, "update-index --skip-worktree payload.txt")
        assert changed.returncode == 0, changed.stderr

    result = _recover_checkout(remote, project)

    assert result.returncode != 0
    assert project.is_dir()
    assert (project / ".git").is_dir()
    assert list(tmp_path.glob("project.incomplete-checkout-backup.*")) == []
