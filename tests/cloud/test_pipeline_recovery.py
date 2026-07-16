import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE = REPO_ROOT / "scripts" / "cloud" / "run_pipeline.sh"
COMMIT = "a" * 40
SHA256 = "b" * 64


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
    result = subprocess.run(
        [bash, "-lc", "command -v flock >/dev/null && command -v sync >/dev/null"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


pytestmark = pytest.mark.skipif(
    not _bash_available(), reason="Linux bash/flock/sync required"
)


def _write_cloud_fixture(tmp_path: Path):
    project = tmp_path / "project"
    persist = tmp_path / "persist"
    env_prefix = persist / "env"
    pipeline_root = persist / "cloud" / "pipelines"
    run_root = persist / "cloud" / "runs"
    events = persist / "events.log"
    python_log = persist / "python.log"
    pipeline_events = persist / "pipeline-events.log"
    g0_sentinel = persist / "g0-failed-once"
    result_file = persist / "pipeline-result"
    lock_file = persist / "cloud" / "pipeline.lock"
    capability = persist / "cloud" / "shutdown-capability"
    launcher_root = persist / "cloud" / "launchers"

    (project / "scripts" / "cloud").mkdir(parents=True)
    (project / "verl" / "trainer" / "config" / "reproduction").mkdir(
        parents=True
    )
    (project / "environment").mkdir()
    env_prefix.joinpath("bin").mkdir(parents=True)
    run_root.mkdir(parents=True)
    pipeline_root.mkdir(parents=True)
    launcher_root.mkdir(parents=True)
    lock_file.touch()
    capability.write_text("test\n", encoding="ascii")
    (project / "environment" / "reproduction-cu130.lock.json").write_text(
        "{}\n", encoding="ascii"
    )
    (project / "environment" / "reproduction-assets.json").write_text(
        "{}\n", encoding="ascii"
    )
    (project / "verl" / "trainer" / "config" / "reproduction" / "test.yaml").write_text(
        "test: true\n", encoding="ascii"
    )
    (project / "scripts" / "cloud" / "setup_git.sh").write_bytes(
        b"#!/usr/bin/env bash\nexit 0\n"
    )

    bundle_paths = []
    for name in ("g0-train", "g0-validation", "g1-train", "g1-validation", "g1-eval"):
        bundle = persist / "data" / name
        bundle.mkdir(parents=True)
        (bundle / "train.parquet").write_bytes(name.encode("ascii"))
        bundle_paths.append(_bash_path(bundle))
    resolved_configs = persist / "resolved-configs"
    resolved_configs.mkdir()

    fake_python = env_prefix / "bin" / "python"
    fake_python_script = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"printf 'python:%s\\n' \"$*\" >> {shlex.quote(_bash_path(events))}\n"
        f"printf '%s\\n' \"$*\" >> {shlex.quote(_bash_path(python_log))}\n"
        "if [[ \"$*\" == *'cloud_state.py verify-handoff'* && "
        "\"$*\" == *'--field'* ]]; then\n"
        + "".join(
            f"  printf '%s\\n' {shlex.quote(value)}\n"
            for value in (
                bundle_paths[0],
                SHA256,
                bundle_paths[1],
                SHA256,
                bundle_paths[2],
                SHA256,
                bundle_paths[3],
                SHA256,
                bundle_paths[4],
                SHA256,
                _bash_path(resolved_configs),
            )
        )
        + "fi\nexit 0\n"
    )
    fake_python.write_bytes(fake_python_script.encode("ascii"))
    fake_python.chmod(0o755)

    stage_runner = tmp_path / "fake_stage_runner.sh"
    stage_runner_script = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "stage=\"$1\"\n"
        f"printf 'stage:%s\\n' \"${{stage}}\" >> {shlex.quote(_bash_path(events))}\n"
        "run=\"${REMEMR1_PERSIST_ROOT}/cloud/runs/fake-${stage}-$(date +%s%N)-$$\"\n"
        "mkdir -p \"${run}/logs\" \"${run}/artifacts\"\n"
        "printf '%s\\n' \"${run}\" > \"${REMEMR1_STAGE_RESULT_FILE}\"\n"
        "printf 'schema_version=1\\nstage=%s\\ngit_commit=%s\\npipeline_dir=%s\\n' "
        "\"${stage}\" \"${REMEMR1_EXPECTED_COMMIT}\" \"${REMEMR1_PIPELINE_DIR}\" "
        "> \"${run}/run.meta\"\n"
        "printf '%s\\n' now > \"${run}/finished-at\"\n"
        "if [[ \"${stage}\" == \"${REMEMR1_TEST_FAIL_STAGE:-}\" ]]; then\n"
        "  printf '23\\n' > \"${run}/.failed\"\n"
        "  exit 23\n"
        "fi\n"
        "if [[ \"${stage}\" == g0 ]]; then\n"
        f"  if [[ ! -e {shlex.quote(_bash_path(g0_sentinel))} ]]; then\n"
        f"    printf first > {shlex.quote(_bash_path(g0_sentinel))}\n"
        "    printf '23\\n' > \"${run}/.failed\"\n"
        "    exit 23\n"
        "  fi\n"
        "  printf '91\\n' > \"${run}/.failed\"\n"
        "  exit 91\n"
        "fi\n"
        "printf '0\\n' > \"${run}/.success\"\n"
    )
    stage_runner.write_bytes(stage_runner_script.encode("ascii"))

    env_file = persist / "cloud.env"
    values = {
        "REMEMR1_PROJECT_DIR": _bash_path(project),
        "REMEMR1_PERSIST_ROOT": _bash_path(persist),
        "REMEMR1_EXPECTED_COMMIT": COMMIT,
        "REMEMR1_CAPABILITY_FILE": _bash_path(capability),
        "REMEMR1_LOCK_FILE": _bash_path(lock_file),
        "REMEMR1_LAUNCHER_ROOT": _bash_path(launcher_root),
        "REMEMR1_PIPELINE_ROOT": _bash_path(pipeline_root),
        "REMEMR1_ENV_PREFIX": _bash_path(env_prefix),
        "REMEMR1_STAGE_RUNNER": _bash_path(stage_runner),
        "REMEMR1_RESULT_FILE": _bash_path(result_file),
    }
    env_file.write_bytes(
        "".join(
            f"export {key}={shlex.quote(value)}\n" for key, value in values.items()
        ).encode("ascii")
    )
    env = os.environ.copy()
    env.update(
        {
            "REMEMR1_CLOUD_ENV": _bash_path(env_file),
            "REMEMR1_TEST_MODE": "yes",
            "REMEMR1_PIPELINE_TEST_EVENT_FILE": _bash_path(pipeline_events),
        }
    )
    if os.name == "nt":
        forwarded = [
            "REMEMR1_CLOUD_ENV",
            "REMEMR1_TEST_MODE",
            "REMEMR1_PIPELINE_TEST_EVENT_FILE",
            "REMEMR1_PIPELINE_TEST_FAIL_SYNC_AT",
            "REMEMR1_TEST_FAIL_STAGE",
        ]
        env["WSLENV"] = ":".join(
            value for value in [env.get("WSLENV", ""), *forwarded] if value
        )
    return env, persist, pipeline_root, events, python_log, lock_file


def _run_pipeline(env, lock_file: Path, *args: str):
    arguments = " ".join(shlex.quote(value) for value in args)
    command = (
        f"exec 9>>{shlex.quote(_bash_path(lock_file))}\n"
        "/usr/bin/flock -n 9\n"
        "export REMEMR1_CLOUD_LOCK_HELD=yes\n"
        f"exec bash {shlex.quote(_bash_path(PIPELINE))} {arguments}\n"
    )
    return subprocess.run(
        [shutil.which("bash"), "-c", command],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=30,
    )


def test_retry_adopts_complete_failed_gpu_artifacts_without_retraining(tmp_path):
    env, persist, pipeline_root, events, python_log, lock_file = _write_cloud_fixture(
        tmp_path
    )
    dry_run = _run_pipeline(env, lock_file, "--phase", "cpu", "--dry-run")
    assert dry_run.returncode == 0, dry_run.stderr
    pipelines = list(pipeline_root.iterdir())
    assert len(pipelines) == 1
    pipeline = pipelines[0]
    handoff = pipeline / "cpu-handoff.json"
    handoff.write_text("{}\n", encoding="ascii")
    (pipeline / ".cpu-ready").write_bytes((_bash_path(handoff) + "\n").encode("ascii"))

    first = _run_pipeline(env, lock_file, "--phase", "gpu-gates")
    assert first.returncode == 23, first.stderr
    failed_runs = [
        path
        for path in (persist / "cloud" / "runs").iterdir()
        if path.name.startswith("fake-g0-") and (path / ".failed").is_file()
    ]
    assert len(failed_runs) == 1
    failed_run = failed_runs[0]
    assert (failed_run / ".failed").read_text(encoding="ascii").strip() == "23"
    assert not Path(str(failed_run) + ".adopted").exists()

    events_before = events.read_text(encoding="utf-8")
    refused = _run_pipeline(env, lock_file, "--phase", "gpu-gates")
    assert refused.returncode != 0
    assert events.read_text(encoding="utf-8") == events_before
    assert (pipeline / ".failed").read_text(encoding="ascii").strip() == "23"
    assert (pipeline / "failed-stage").read_text(encoding="ascii").strip() == "g0"
    terminal_events = (persist / "pipeline-events.log").read_text(
        encoding="ascii"
    ).splitlines()
    assert terminal_events[-5:] == [
        "sync:failure-before-cleanup:.failed:begin:running=yes",
        "sync:failure-before-cleanup:.failed:complete:running=yes",
        "running-removed:failure",
        "sync:failure-after-cleanup:.failed:begin:running=no",
        "sync:failure-after-cleanup:.failed:complete:running=no",
    ]

    retried = _run_pipeline(
        env,
        lock_file,
        "--phase",
        "gpu-gates",
        "--retry-failed-stage",
    )
    assert retried.returncode == 0, retried.stderr
    adopted = Path(str(failed_run) + ".adopted")
    assert (failed_run / ".failed").read_text(encoding="ascii").strip() == "23"
    assert (adopted / ".success").read_text(encoding="ascii").strip() == "0"
    assert (adopted / "original-failure").read_text(encoding="ascii").strip() == "23"
    assert (adopted / "adopted-from-failure").read_text(
        encoding="utf-8"
    ).strip() == _bash_path(failed_run)
    assert (pipeline / "stages" / "g0.run").read_text(
        encoding="utf-8"
    ).strip() == _bash_path(adopted)

    event_lines = events.read_text(encoding="utf-8").splitlines()
    assert event_lines.count("stage:g0") == 1
    assert event_lines.count("stage:gpu-preflight") == 2
    g0_verify = next(
        index
        for index, line in enumerate(event_lines)
        if "verify_training_artifacts.py" in line
        and "checkpoints/g0/global_step_20" in line
    )
    second_preflight = max(
        index for index, line in enumerate(event_lines) if line == "stage:gpu-preflight"
    )
    assert second_preflight < g0_verify

    verifier_calls = [
        line
        for line in python_log.read_text(encoding="utf-8").splitlines()
        if "verify_training_artifacts.py" in line
    ]
    assert any(
        "--expected-step 20" in line
        and "--expected-base-model Qwen/Qwen3.5-0.8B" in line
        and "--expected-train-manifest " + SHA256 in line
        for line in verifier_calls
    )
    resume_calls = [
        line
        for line in verifier_calls
        if "checkpoints/g1-resume2/global_step_2" in line
    ]
    assert len(resume_calls) == 2
    assert all("--expected-resume-from" in line for line in resume_calls)

    downgrade = _run_pipeline(env, lock_file, "--phase", "cpu")
    assert downgrade.returncode != 0
    assert (pipeline / ".success").read_text(encoding="ascii").strip() == "0"
    assert not (pipeline / ".failed").exists()
    terminal_events = (persist / "pipeline-events.log").read_text(
        encoding="ascii"
    ).splitlines()
    assert terminal_events[-5:] == [
        "sync:success-before-cleanup:.success:begin:running=yes",
        "sync:success-before-cleanup:.success:complete:running=yes",
        "running-removed:success",
        "sync:success-after-cleanup:.success:begin:running=no",
        "sync:success-after-cleanup:.success:complete:running=no",
    ]


def test_failed_terminal_sync_keeps_running_and_requires_explicit_retry(tmp_path):
    env, persist, pipeline_root, events, _, lock_file = _write_cloud_fixture(tmp_path)
    dry_run = _run_pipeline(env, lock_file, "--phase", "cpu", "--dry-run")
    assert dry_run.returncode == 0, dry_run.stderr
    pipeline = next(pipeline_root.iterdir())

    env["REMEMR1_TEST_FAIL_STAGE"] = "cpu-preflight"
    env["REMEMR1_PIPELINE_TEST_FAIL_SYNC_AT"] = "failure-before-cleanup"
    failed = _run_pipeline(env, lock_file, "--phase", "cpu")
    assert failed.returncode == 23, failed.stderr
    assert (pipeline / ".running").is_file()
    assert (pipeline / ".failed").read_text(encoding="ascii").strip() == "23"
    assert (
        pipeline / "failed-stage"
    ).read_text(encoding="ascii").strip() == "cpu-preflight"

    events_before = events.read_text(encoding="utf-8")
    refused = _run_pipeline(env, lock_file, "--phase", "cpu")
    assert refused.returncode != 0
    assert events.read_text(encoding="utf-8") == events_before
    assert (pipeline / ".failed").read_text(encoding="ascii").strip() == "23"
