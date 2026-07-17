import hashlib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE = REPO_ROOT / "scripts" / "cloud" / "run_pipeline.sh"
STATUS = REPO_ROOT / "scripts" / "cloud" / "status.sh"
COMMIT = "a" * 40
SHA256 = "b" * 64


def _bash_path(path: Path) -> str:
    path = path.resolve()
    if os.name != "nt":
        return str(path)
    drive, tail = os.path.splitdrive(str(path))
    return f"/mnt/{drive[0].lower()}/{tail.lstrip('\\/').replace(os.sep, '/')}"


def _native_path(path: Path) -> Path:
    if os.name == "nt":
        return Path("\\\\?\\" + str(path.resolve()))
    return path


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
    for name in (
        "g0-train",
        "g0-validation",
        "g1-train",
        "g1-validation",
        "g1-eval",
        "formal-train",
        "formal-validation",
        "length-train",
        "length-validation",
        "hotpot-eval",
        "2wiki-eval",
    ):
        bundle = persist / "data" / name
        bundle.mkdir(parents=True)
        (bundle / "train.parquet").write_bytes(name.encode("ascii"))
        bundle_paths.append(_bash_path(bundle))
    resolved_configs = persist / "resolved-configs"
    resolved_configs.mkdir()
    for name in (
        "g2a_qwen35_2b_5090_r0",
        "g2b_qwen35_2b_5090_step1_r0",
        "g2b_qwen35_2b_5090_resume5_r0",
        "g2_length_stress_qwen35_2b_5090_r0",
    ):
        (resolved_configs / f"{name}.yaml").write_text("{}\n", encoding="ascii")

    fake_python = env_prefix / "bin" / "python"
    fake_python_script = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"printf 'python:%s\\n' \"$*\" >> {shlex.quote(_bash_path(events))}\n"
        f"printf '%s\\n' \"$*\" >> {shlex.quote(_bash_path(python_log))}\n"
        "if [[ \"${1:-}\" == -c && \"${2:-}\" == *'verify_r1_approval_marker'* ]]; then "
        "exit 0; fi\n"
        "if [[ \"${1:-}\" == -c && \"${2:-}\" == *'json.load'* ]]; then "
        "exec /usr/bin/python3 \"$@\"; fi\n"
        "if [[ \"$*\" == *'capacity_evidence.py consume-r1'* ]]; then\n"
        "  output=\n"
        "  while [[ $# -gt 0 ]]; do\n"
        "    if [[ \"$1\" == --output ]]; then output=\"$2\"; break; fi\n"
        "    shift\n"
        "  done\n"
        "  [[ -n \"${output}\" ]]\n"
        "  printf '{\"status\":\"consumed\"}\\n' > \"${output}\"\n"
        "  exit 0\n"
        "fi\n"
        "if [[ \"$*\" == *'cloud_state.py verify-handoff'* && "
        "\"$*\" == *'--field'* ]]; then\n"
        + "".join(
            f"  printf '%s\\n' {shlex.quote(value)}\n"
            for value in (
                "rtx5090-32g-qwen35-2b-v1",
                SHA256,
                SHA256,
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
                bundle_paths[5],
                SHA256,
                bundle_paths[6],
                SHA256,
                bundle_paths[7],
                SHA256,
                bundle_paths[8],
                SHA256,
                bundle_paths[9],
                SHA256,
                bundle_paths[10],
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
        "printf 'experiment_profile_id=%s\nphase=%s\noffload_profile=%s\n' "
        "\"${REMEMR1_EXPERIMENT_PROFILE}\" \"${REMEMR1_PHASE}\" "
        "\"${REMEMR1_OFFLOAD_PROFILE:-}\" >> \"${run}/run.meta\"\n"
        "printf 'budget_projection_sha256=%s\n' "
        "\"${REMEMR1_BUDGET_PROJECTION_SHA256:-}\" >> \"${run}/run.meta\"\n"
        "printf 'r1_approval_marker_sha256=%s\n' "
        "\"${REMEMR1_R1_APPROVAL_MARKER_SHA256:-}\" >> \"${run}/run.meta\"\n"
        "printf 'scope_generation=%s\n' "
        "\"${REMEMR1_SCOPE_GENERATION:-base}\" >> \"${run}/run.meta\"\n"
        "printf '%s\\n' now > \"${run}/finished-at\"\n"
        "if [[ \"${stage}\" == \"${REMEMR1_TEST_FAIL_STAGE:-}\" ]]; then\n"
        "  printf '23\\n' > \"${run}/.failed\"\n"
        "  exit 23\n"
        "fi\n"
        "if [[ \"${stage}\" == \"${REMEMR1_TEST_SCIENTIFIC_STOP_STAGE:-}\" ]]; then\n"
        "  printf '42\n' > \"${run}/.scientific-stop\"\n"
        "  printf 'false\n' > \"${run}/retryable\"\n"
        "  exit 42\n"
        "fi\n"
        "if [[ \"${stage}\" == \"${REMEMR1_TEST_CAPACITY_STOP_STAGE:-}\" ]]; then\n"
        "  mkdir -p \"${run}/evidence\"\n"
        "  printf '{}\n' > \"${run}/evidence/capacity-stop.json\"\n"
        "  printf '43\n' > \"${run}/.capacity-stop\"\n"
        "  printf 'false\n' > \"${run}/retryable\"\n"
        "  exit 43\n"
        "fi\n"
        "if [[ \"${stage}\" == cpu-handoff ]]; then\n"
        "  printf '{}\\n' > \"${REMEMR1_PIPELINE_DIR}/cpu-handoff.json\"\n"
        "fi\n"
        "case \"${stage}\" in\n"
        "  g2a|g2b-step1|g2b-resume5|g2-length-stress)\n"
        "    printf '{}\\n' > \"${run}/telemetry.json\"\n"
        "    ;;\n"
        "  g2-artifacts)\n"
        "    mkdir -p \"${run}/artifacts/capacity-inputs\" \"${REMEMR1_CAPACITY_OUTPUT_DIR}\"\n"
        "    printf '{}\\n' > \"${run}/artifacts/g2-artifacts.json\"\n"
        "    for name in identity selected-configs attempt-metadata telemetry; do\n"
        "      printf '{}\\n' > \"${run}/artifacts/capacity-inputs/${name}.json\"\n"
        "      printf '{}\\n' > \"${REMEMR1_CAPACITY_OUTPUT_DIR}/${name}.json\"\n"
        "    done\n"
        "    ;;\n"
        "esac\n"
        "if [[ \"${stage}\" == capacity-seal ]]; then\n"
        "  mkdir -p \"${REMEMR1_CAPACITY_OUTPUT_DIR}\"\n"
        "  printf '{}\\n' > \"${REMEMR1_CAPACITY_OUTPUT_DIR}/capacity-profile.json\"\n"
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
        "REMEMR1_EXPERIMENT_PROFILE": "rtx5090-32g-qwen35-2b-v1",
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
            "REMEMR1_TEST_SCIENTIFIC_STOP_STAGE",
            "REMEMR1_TEST_CAPACITY_STOP_STAGE",
            "REMEMR1_LAUNCHER_DIR",
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
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
        timeout=30,
    )


def _budget_args(persist: Path, decision: str, generation: str = "initial"):
    projection = persist / f"{decision}-{generation}-projection.json"
    projection_identity = hashlib.sha256(
        f"{decision}:{generation}".encode("ascii")
    ).hexdigest()
    projection.write_text(
        json.dumps(
            {"decision": decision, "projection_sha256": projection_identity},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="ascii",
    )
    digest = hashlib.sha256(projection.read_bytes()).hexdigest()
    return (
        "--budget-projection",
        _bash_path(projection),
        "--budget-projection-file-sha256",
        digest,
    )


def _set_result_file(persist: Path, result_file: Path) -> None:
    env_file = persist / "cloud.env"
    lines = env_file.read_text(encoding="ascii").splitlines()
    replacement = f"export REMEMR1_RESULT_FILE={shlex.quote(_bash_path(result_file))}"
    env_file.write_bytes(
        (
            "\n".join(
            replacement if line.startswith("export REMEMR1_RESULT_FILE=") else line
            for line in lines
            )
            + "\n"
        ).encode("ascii")
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
    gpu_state = pipeline / "terminals" / "gpu-gates" / "r0" / "base"
    assert (failed_run / ".failed").read_text(encoding="ascii").strip() == "23"
    assert not Path(str(failed_run) + ".adopted").exists()
    assert (gpu_state / ".failed").read_text(encoding="ascii").strip() == "23"

    events_before = events.read_text(encoding="utf-8")
    refused = _run_pipeline(env, lock_file, "--phase", "gpu-gates")
    assert refused.returncode != 0
    assert events.read_text(encoding="utf-8") == events_before
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
    assert (pipeline / "stages" / "gpu-gates" / "r0" / "base" / "g0.run").read_text(
        encoding="utf-8"
    ).strip() == _bash_path(adopted)

    event_lines = events.read_text(encoding="utf-8").splitlines()
    assert event_lines.count("stage:g0") == 1
    assert event_lines.count("stage:gpu-preflight") == 2
    g0_verify = next(
        index
        for index, line in enumerate(event_lines)
        if "verify_training_artifacts.py" in line
        and "fake-g0-" in line
        and "/checkpoints/global_step_20" in line
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
        if "fake-g1-resume2-" in line and "/checkpoints/global_step_2" in line
    ]
    assert len(resume_calls) == 2
    assert all("--expected-resume-from" in line for line in resume_calls)

    downgrade = _run_pipeline(env, lock_file, "--phase", "cpu")
    assert downgrade.returncode != 0
    assert (pipeline / ".gpu-gates-ready").is_file()
    assert not (gpu_state / ".failed").exists()
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
    launcher_root = persist / "cloud" / "launchers"
    failed_launcher = launcher_root / "sync-failed"
    failed_launcher.mkdir()
    _set_result_file(persist, failed_launcher / "pipeline-result")
    env["REMEMR1_LAUNCHER_DIR"] = _bash_path(failed_launcher)

    env["REMEMR1_TEST_FAIL_STAGE"] = "cpu-preflight"
    env["REMEMR1_PIPELINE_TEST_FAIL_SYNC_AT"] = "failure-before-cleanup"
    failed = _run_pipeline(env, lock_file, "--phase", "cpu")
    assert failed.returncode == 23, failed.stderr
    cpu_state = pipeline / "terminals" / "cpu" / "none" / "base"
    assert (cpu_state / ".running").is_file()
    assert (cpu_state / ".failed").read_text(encoding="ascii").strip() == "23"
    inhibition = (pipeline / "shutdown-inhibited").read_text(
        encoding="ascii"
    ).strip()
    assert inhibition == f"terminal-sync-or-cleanup-incomplete:{_bash_path(cpu_state)}"

    refused_launcher = launcher_root / "stale-refused"
    refused_launcher.mkdir()
    _set_result_file(persist, refused_launcher / "pipeline-result")
    env["REMEMR1_LAUNCHER_DIR"] = _bash_path(refused_launcher)
    events_before = events.read_text(encoding="utf-8")
    refused = _run_pipeline(env, lock_file, "--phase", "cpu")
    assert refused.returncode != 0
    assert events.read_text(encoding="utf-8") == events_before
    assert (cpu_state / ".failed").read_text(encoding="ascii").strip() == "23"
    snapshot = pipeline / "launcher-terminals" / "stale-refused"
    assert (refused_launcher / "pipeline-result.terminal").read_text(
        encoding="ascii"
    ).strip() == _bash_path(snapshot)
    assert _native_path(snapshot / ".failed").read_text(
        encoding="ascii"
    ).strip() == "1"
    assert _native_path(snapshot / "failed-stage").read_text(
        encoding="ascii"
    ).strip() == "stale-running-requires-retry"
    assert (cpu_state / ".running").is_file()


def test_immutable_generation_running_refusal_has_launcher_snapshot(tmp_path):
    env, persist, pipeline_root, _, _, lock_file = _write_cloud_fixture(tmp_path)
    initialized = _run_pipeline(env, lock_file, "--phase", "cpu", "--dry-run")
    assert initialized.returncode == 0, initialized.stderr
    pipeline = next(pipeline_root.iterdir())
    budget_args = _budget_args(persist, "bc40", "immutable-running")
    projection = json.loads(
        (persist / "bc40-immutable-running-projection.json").read_text(
            encoding="ascii"
        )
    )
    state = (
        pipeline
        / "terminals"
        / "gpu-bc40"
        / "unresolved"
        / f"budget-{projection['projection_sha256']}"
    )
    _native_path(state).mkdir(parents=True)
    _native_path(state / ".running").write_text("999999\n", encoding="ascii")

    launcher = persist / "cloud" / "launchers" / "immutable-refused"
    launcher.mkdir()
    _set_result_file(persist, launcher / "pipeline-result")
    env["REMEMR1_LAUNCHER_DIR"] = _bash_path(launcher)
    refused = _run_pipeline(env, lock_file, "--phase", "gpu-bc40", *budget_args)

    assert refused.returncode == 1, refused.stderr
    assert "cannot be retried" in refused.stderr
    snapshot = pipeline / "launcher-terminals" / "immutable-refused"
    assert (launcher / "pipeline-result.terminal").read_text(
        encoding="ascii"
    ).strip() == _bash_path(snapshot)
    assert _native_path(snapshot / ".failed").read_text(
        encoding="ascii"
    ).strip() == "1"
    assert _native_path(snapshot / "failed-stage").read_text(
        encoding="ascii"
    ).strip() == "immutable-generation-running"
    assert _native_path(state / ".running").is_file()
    assert (pipeline / "shutdown-inhibited").read_text(
        encoding="ascii"
    ).strip() == f"immutable-generation-running:{_bash_path(state)}"


def test_status_does_not_fallback_to_shared_last_terminal(tmp_path):
    env, persist, pipeline_root, _, _, lock_file = _write_cloud_fixture(tmp_path)
    initialized = _run_pipeline(env, lock_file, "--phase", "cpu", "--dry-run")
    assert initialized.returncode == 0, initialized.stderr
    pipeline = next(pipeline_root.iterdir())
    shared_terminal = pipeline / "terminals" / "cpu" / "none" / "base"
    _native_path(shared_terminal).mkdir(parents=True)
    _native_path(shared_terminal / "failed-stage").write_text(
        "unrelated-run\n", encoding="ascii"
    )
    (pipeline / "last-terminal").write_text(
        _bash_path(shared_terminal) + "\n", encoding="ascii"
    )

    launcher = persist / "cloud" / "launchers" / "no-terminal-pointer"
    launcher.mkdir()
    (launcher / ".failed").write_text("1\n", encoding="ascii")
    (launcher / "pipeline-result").write_text(
        _bash_path(pipeline) + "\n", encoding="ascii"
    )
    status = subprocess.run(
        [shutil.which("bash"), _bash_path(STATUS), _bash_path(launcher)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )

    assert status.returncode == 0, status.stderr
    assert "pipeline_terminal_dir=" not in status.stdout
    assert "pipeline_failed-stage=" not in status.stdout


def test_all_phase_dry_runs_publish_the_exact_ordered_dag(tmp_path):
    env, persist, _, _, _, lock_file = _write_cloud_fixture(tmp_path)
    bc40_budget = _budget_args(persist, "bc40")
    bc80_budget = _budget_args(persist, "bc80")
    cases = (
        (
            ("--phase", "cpu", "--dry-run"),
            [
                "cpu-preflight",
                "cpu-environment",
                "cpu-kernel-sources",
                "cpu-assets",
            ],
        ),
        (
            ("--phase", "cpu-finalize", "--dry-run"),
            [
                "cpu-preflight",
                "cpu-data",
                "cpu-tests",
                "cpu-configs",
                "cpu-handoff",
            ],
        ),
        (
            ("--phase", "gpu-gates", "--dry-run"),
            ["gpu-preflight", "g0", "g1-step1", "g1-resume2", "g1-artifacts"],
        ),
        (
            ("--phase", "gpu-capacity", "--offload-profile", "r0", "--dry-run"),
            [
                "gpu-preflight",
                "g2a",
                "g2b-step1",
                "g2b-resume5",
                "g2-length-stress",
                "g2-artifacts",
                "capacity-seal",
            ],
        ),
        (
            ("--phase", "gpu-bc40", *bc40_budget, "--dry-run"),
            [
                "gpu-preflight",
                "b-pilot",
                "c-pilot",
                "pilot-gate",
                "b20",
                "c20",
                "b40",
                "c40",
                "bc40-artifacts",
                "eval40",
                "package40",
            ],
        ),
        (
            ("--phase", "gpu-bc80", *bc80_budget, "--dry-run"),
            [
                "gpu-preflight",
                "b60",
                "c60",
                "b80",
                "c80",
                "bc80-artifacts",
                "eval80",
                "package80",
            ],
        ),
        (
            ("--phase", "gpu-export", "--dry-run"),
            ["gpu-preflight", "export-results"],
        ),
    )

    for arguments, expected in cases:
        result = _run_pipeline(env, lock_file, *arguments)
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == expected


def test_cpu_preparation_and_internal_finalization_publish_distinct_evidence(tmp_path):
    env, _, pipeline_root, events, _, lock_file = _write_cloud_fixture(tmp_path)

    prepared = _run_pipeline(env, lock_file, "--phase", "cpu")
    assert prepared.returncode == 0, prepared.stderr
    pipeline = next(pipeline_root.iterdir())
    seal = pipeline / "cpu-env-ready.seal"
    env_ready = pipeline / ".cpu-env-ready"
    assert seal.is_file()
    assert env_ready.read_text(encoding="utf-8").strip() == _bash_path(seal)
    assert not (pipeline / ".cpu-ready").exists()
    assert events.read_text(encoding="utf-8").splitlines() == [
        "stage:cpu-preflight",
        "stage:cpu-environment",
        "stage:cpu-kernel-sources",
        "stage:cpu-assets",
    ]

    finalized = _run_pipeline(env, lock_file, "--phase", "cpu-finalize")
    assert finalized.returncode == 0, finalized.stderr
    handoff = pipeline / "cpu-handoff.json"
    assert handoff.is_file()
    assert (pipeline / ".cpu-ready").read_text(
        encoding="utf-8"
    ).strip() == _bash_path(handoff)
    stage_events = [
        line
        for line in events.read_text(encoding="utf-8").splitlines()
        if line.startswith("stage:")
    ]
    assert stage_events[-5:] == [
        "stage:cpu-preflight",
        "stage:cpu-data",
        "stage:cpu-tests",
        "stage:cpu-configs",
        "stage:cpu-handoff",
    ]
    assert stage_events.count("stage:cpu-preflight") == 2
    assert _native_path(
        pipeline
        / "stages"
        / "cpu-finalize"
        / "none"
        / "base"
        / "cpu-handoff.run"
    ).is_file()


def test_cpu_finalization_rejects_tampered_environment_evidence_before_stages(
    tmp_path,
):
    env, _, pipeline_root, events, _, lock_file = _write_cloud_fixture(tmp_path)
    prepared = _run_pipeline(env, lock_file, "--phase", "cpu")
    assert prepared.returncode == 0, prepared.stderr
    pipeline = next(pipeline_root.iterdir())
    (pipeline / "cpu-env-ready.seal").write_text("tampered\n", encoding="ascii")
    event_count = len(events.read_text(encoding="utf-8").splitlines())

    finalized = _run_pipeline(env, lock_file, "--phase", "cpu-finalize")

    assert finalized.returncode != 0
    assert "requires valid .cpu-env-ready evidence" in finalized.stderr
    assert len(events.read_text(encoding="utf-8").splitlines()) == event_count
    assert not (pipeline / ".cpu-ready").exists()


def test_cpu_finalization_is_refused_after_gpu_start(tmp_path):
    env, _, pipeline_root, events, _, lock_file = _write_cloud_fixture(tmp_path)
    prepared = _run_pipeline(env, lock_file, "--phase", "cpu")
    assert prepared.returncode == 0, prepared.stderr
    pipeline = next(pipeline_root.iterdir())
    (pipeline / ".gpu-started").write_text("now\n", encoding="ascii")
    event_count = len(events.read_text(encoding="utf-8").splitlines())

    finalized = _run_pipeline(env, lock_file, "--phase", "cpu-finalize")

    assert finalized.returncode != 0
    assert "GPU gates have already started" in finalized.stderr
    assert len(events.read_text(encoding="utf-8").splitlines()) == event_count
    assert not (pipeline / ".cpu-ready").exists()


def test_successful_cpu_finalization_is_reusable_after_gpu_start(tmp_path):
    env, _, pipeline_root, events, _, lock_file = _write_cloud_fixture(tmp_path)
    prepared = _run_pipeline(env, lock_file, "--phase", "cpu")
    assert prepared.returncode == 0, prepared.stderr
    finalized = _run_pipeline(env, lock_file, "--phase", "cpu-finalize")
    assert finalized.returncode == 0, finalized.stderr
    pipeline = next(pipeline_root.iterdir())
    (pipeline / ".gpu-started").write_text("now\n", encoding="ascii")
    event_count = len(events.read_text(encoding="utf-8").splitlines())

    repeated = _run_pipeline(
        env,
        lock_file,
        "--phase",
        "cpu-finalize",
        "--retry-failed-stage",
    )

    assert repeated.returncode == 0, repeated.stderr
    assert "already finalized before GPU start" in repeated.stdout
    assert len(events.read_text(encoding="utf-8").splitlines()) == event_count


def test_successful_r0_capacity_rerun_revalidates_without_downgrading(tmp_path):
    env, persist, pipeline_root, events, _, lock_file = _write_cloud_fixture(tmp_path)
    initialized = _run_pipeline(env, lock_file, "--phase", "cpu", "--dry-run")
    assert initialized.returncode == 0, initialized.stderr
    pipeline = next(pipeline_root.iterdir())
    handoff = pipeline / "cpu-handoff.json"
    handoff.write_text("{}\n", encoding="ascii")
    (pipeline / ".cpu-ready").write_bytes(
        (_bash_path(handoff) + "\n").encode("ascii")
    )

    first_gates = _run_pipeline(env, lock_file, "--phase", "gpu-gates")
    assert first_gates.returncode == 23, first_gates.stderr
    gates = _run_pipeline(
        env,
        lock_file,
        "--phase",
        "gpu-gates",
        "--retry-failed-stage",
    )
    assert gates.returncode == 0, gates.stderr
    first_capacity = _run_pipeline(
        env,
        lock_file,
        "--phase",
        "gpu-capacity",
        "--offload-profile",
        "r0",
    )
    assert first_capacity.returncode == 0, first_capacity.stderr
    capacity_state = pipeline / "terminals" / "gpu-capacity" / "r0" / "base"
    assert _native_path(capacity_state / ".success").read_text(
        encoding="ascii"
    ).strip() == "0"

    repeated = _run_pipeline(
        env,
        lock_file,
        "--phase",
        "gpu-capacity",
        "--offload-profile",
        "r0",
    )

    assert repeated.returncode == 0, repeated.stderr
    assert _native_path(capacity_state / ".success").read_text(
        encoding="ascii"
    ).strip() == "0"
    assert not _native_path(capacity_state / ".failed").exists()
    stage_events = [
        line
        for line in events.read_text(encoding="utf-8").splitlines()
        if line.startswith("stage:")
    ]
    assert stage_events.count("stage:gpu-preflight") == 4
    for stage in (
        "g2a",
        "g2b-step1",
        "g2b-resume5",
        "g2-length-stress",
        "g2-artifacts",
        "capacity-seal",
    ):
        assert stage_events.count(f"stage:{stage}") == 1
    assert _native_path(persist / "pipeline-result.terminal").is_file()


def test_r1_approval_is_claimed_once_and_dry_run_does_not_consume(tmp_path):
    env, persist, pipeline_root, events, _, lock_file = _write_cloud_fixture(tmp_path)
    initialized = _run_pipeline(env, lock_file, "--phase", "cpu", "--dry-run")
    assert initialized.returncode == 0, initialized.stderr
    pipeline = next(pipeline_root.iterdir())
    handoff = pipeline / "cpu-handoff.json"
    handoff.write_text("{}\n", encoding="ascii")
    handoff_pointer = (_bash_path(handoff) + "\n").encode("ascii")
    (pipeline / ".cpu-ready").write_bytes(handoff_pointer)
    (pipeline / ".gpu-gates-ready").write_bytes(handoff_pointer)
    capacity_r0 = pipeline / "capacity" / "r0"
    capacity_r0.mkdir(parents=True)
    (capacity_r0 / "capacity-evidence.json").write_text("{}\n", encoding="ascii")
    (capacity_r0 / "r1-target-identity.json").write_text("{}\n", encoding="ascii")

    budget_args = _budget_args(persist, "bc40", "r1-consumption")
    projection_path = persist / "bc40-r1-consumption-projection.json"
    projection = json.loads(projection_path.read_text(encoding="ascii"))
    marker_sha = "c" * 64
    marker = persist / "r1-approval.json"
    marker.write_text(
        json.dumps(
            {
                "approval_marker_sha256": marker_sha,
                "budget_projection_sha256": projection["projection_sha256"],
                "r0_terminal_sha256": SHA256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="ascii",
    )
    launcher = persist / "cloud" / "launchers" / "launcher-r1"
    launcher.mkdir()
    (launcher / "request.json").write_text("{}\n", encoding="ascii")
    env["REMEMR1_LAUNCHER_DIR"] = _bash_path(launcher)
    args = (
        "--phase",
        "gpu-capacity",
        "--offload-profile",
        "r1",
        "--r1-approval",
        _bash_path(marker),
        *budget_args,
    )

    dry_run = _run_pipeline(env, lock_file, *args, "--dry-run")
    assert dry_run.returncode == 0, dry_run.stderr
    claim = (
        pipeline
        / "capacity"
        / "r1"
        / "approval-consumptions"
        / marker_sha
    )
    assert not claim.exists()

    env["REMEMR1_TEST_FAIL_STAGE"] = "gpu-preflight"
    first = _run_pipeline(env, lock_file, *args)
    assert first.returncode == 23, first.stderr
    assert any(
        "capacity_evidence.py consume-r1" in line
        for line in events.read_text(encoding="utf-8").splitlines()
    )
    stage_count = events.read_text(encoding="utf-8").splitlines().count(
        "stage:gpu-preflight"
    )

    repeated_launcher = persist / "cloud" / "launchers" / "launcher-r1-repeat"
    repeated_launcher.mkdir()
    (repeated_launcher / "request.json").write_text("{}\n", encoding="ascii")
    env["REMEMR1_LAUNCHER_DIR"] = _bash_path(repeated_launcher)
    repeated = _run_pipeline(env, lock_file, *args, "--retry-failed-stage")
    assert repeated.returncode != 0
    assert "already claimed" in repeated.stderr
    assert events.read_text(encoding="utf-8").splitlines().count(
        "stage:gpu-preflight"
    ) == stage_count

    empty_marker_sha = "d" * 64
    empty_marker = persist / "r1-approval-empty-claim.json"
    empty_marker.write_text(
        json.dumps(
            {
                "approval_marker_sha256": empty_marker_sha,
                "budget_projection_sha256": projection["projection_sha256"],
                "r0_terminal_sha256": SHA256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="ascii",
    )
    empty_claim_bash = (
        f"{_bash_path(pipeline)}/capacity/r1/approval-consumptions/"
        f"{empty_marker_sha}"
    )
    created = subprocess.run(
        [
            shutil.which("bash"),
            "-c",
            f"mkdir -p -- {shlex.quote(empty_claim_bash)}",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert created.returncode == 0, created.stderr
    empty_launcher = persist / "cloud" / "launchers" / "launcher-r1-empty"
    empty_launcher.mkdir()
    (empty_launcher / "request.json").write_text("{}\n", encoding="ascii")
    env["REMEMR1_LAUNCHER_DIR"] = _bash_path(empty_launcher)
    empty_args = list(args)
    empty_args[empty_args.index(_bash_path(marker))] = _bash_path(empty_marker)
    refused_empty = _run_pipeline(env, lock_file, *empty_args)
    assert refused_empty.returncode != 0
    assert "already claimed" in refused_empty.stderr


def test_launcher_terminal_snapshots_survive_same_scope_retry(tmp_path):
    env, persist, pipeline_root, events, _, lock_file = _write_cloud_fixture(tmp_path)
    launcher_root = persist / "cloud" / "launchers"
    first_launcher = launcher_root / "first"
    first_launcher.mkdir()
    _set_result_file(persist, first_launcher / "pipeline-result")
    env["REMEMR1_LAUNCHER_DIR"] = _bash_path(first_launcher)
    env["REMEMR1_TEST_FAIL_STAGE"] = "cpu-preflight"

    first = _run_pipeline(env, lock_file, "--phase", "cpu")
    assert first.returncode == 23, first.stderr
    (first_launcher / ".failed").write_text("23\n", encoding="ascii")
    first_pointer = first_launcher / "pipeline-result.terminal"

    refused_launcher = launcher_root / "refused"
    refused_launcher.mkdir()
    _set_result_file(persist, refused_launcher / "pipeline-result")
    env["REMEMR1_LAUNCHER_DIR"] = _bash_path(refused_launcher)
    event_count = len(events.read_text(encoding="utf-8").splitlines())
    refused = _run_pipeline(env, lock_file, "--phase", "cpu")
    assert refused.returncode != 0
    assert "use --retry-failed-stage" in refused.stderr
    assert len(events.read_text(encoding="utf-8").splitlines()) == event_count

    second_launcher = launcher_root / "second"
    second_launcher.mkdir()
    _set_result_file(persist, second_launcher / "pipeline-result")
    env["REMEMR1_LAUNCHER_DIR"] = _bash_path(second_launcher)
    env.pop("REMEMR1_TEST_FAIL_STAGE")
    second = _run_pipeline(
        env,
        lock_file,
        "--phase",
        "cpu",
        "--retry-failed-stage",
    )
    assert second.returncode == 0, second.stderr
    (second_launcher / ".success").write_text("0\n", encoding="ascii")

    first_status = subprocess.run(
        [shutil.which("bash"), _bash_path(STATUS), _bash_path(first_launcher)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )
    second_status = subprocess.run(
        [shutil.which("bash"), _bash_path(STATUS), _bash_path(second_launcher)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )
    assert first_status.returncode == 0, first_status.stderr
    assert second_status.returncode == 0, second_status.stderr
    assert "pipeline_failed-stage=cpu-preflight" in first_status.stdout
    assert "pipeline_failed-stage=" not in second_status.stdout
    assert "/launcher-terminals/first" in first_status.stdout
    assert "/launcher-terminals/second" in second_status.stdout
    assert first_pointer.is_file()
    pipeline = next(pipeline_root.iterdir())
    shared = pipeline / "terminals" / "cpu" / "none" / "base"
    assert (shared / ".success").is_file()
    assert not (shared / ".failed").exists()


@pytest.mark.parametrize(
    "arguments, message",
    [
        (("--phase", "gpu-capacity"), "requires --offload-profile"),
        (
            ("--phase", "gpu-capacity", "--offload-profile", "r1"),
            "requires --r1-approval",
        ),
        (
            ("--phase", "gpu-capacity", "--offload-profile", "r0", "--r1-approval", "/tmp/x"),
            "must not carry",
        ),
        (("--phase", "gpu-bc40"), "requires --budget-projection"),
        (("--phase", "gpu-bc80"), "requires --budget-projection"),
    ],
)
def test_phase_arguments_fail_closed_before_work(tmp_path, arguments, message):
    env, _, _, events, _, lock_file = _write_cloud_fixture(tmp_path)
    result = _run_pipeline(env, lock_file, *arguments)
    assert result.returncode == 2
    assert message in result.stderr
    assert not events.exists()


def test_stage_verifies_complete_training_artifacts_before_publishing_success():
    source = (REPO_ROOT / "scripts" / "cloud" / "run_stage.sh").read_text(
        encoding="utf-8"
    )
    function = source.split("run_bound_training() {", 1)[1].split(
        "\n}\n\nprepare_capacity_approval_args()", 1
    )[0]

    assert "scripts/cloud/verify_training_artifacts.py" in function
    assert "--expected-resume-from" in function
    assert function.index("scripts/cloud/verify_training_artifacts.py") < function.rindex(
        'return "${training_rc}"'
    )
