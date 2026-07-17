#!/usr/bin/env bash
# Start one cloud phase in a detached, durable launcher.
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/runtime.sh
source "${SCRIPT_DIR}/lib/runtime.sh"
# shellcheck source=lib/lock.sh
source "${SCRIPT_DIR}/lib/lock.sh"
# shellcheck source=lib/shutdown.sh
source "${SCRIPT_DIR}/lib/shutdown.sh"

usage() {
    cat <<'EOF'
Usage: launch.sh --phase PHASE [OPTIONS]

Phases:
  cpu, gpu-gates, gpu-capacity, gpu-bc40, gpu-bc80, gpu-export

Options:
  --offload-profile P  Required for gpu-capacity; P is r0 or r1.
  --r1-approval PATH   Required with gpu-capacity --offload-profile r1.
  --budget-projection PATH
                        Required for B/C40, B/C80, and R1 capacity.
  --keep-running        Persist results but do not request guest shutdown.
  --retry-failed-stage  Allow the pipeline to retry its first failed stage.
  --dry-run             Validate/print pipeline actions without shutdown.
  -h, --help            Show this help.
EOF
}

phase=""
keep_running=no
retry_failed_stage=no
dry_run=no
offload_profile=""
r1_approval=""
r1_approval_file_sha256=""
budget_projection=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)
            [[ $# -ge 2 ]] || { echo "--phase requires a value" >&2; exit 2; }
            [[ -z "${phase}" ]] || { echo "phase was provided more than once" >&2; exit 2; }
            phase="$2"
            shift 2
            ;;
        cpu|gpu-gates|gpu-capacity|gpu-bc40|gpu-bc80|gpu-export)
            [[ -z "${phase}" ]] || { echo "phase was provided more than once" >&2; exit 2; }
            phase="$1"
            shift
            ;;
        --offload-profile)
            [[ $# -ge 2 && -z "${offload_profile}" ]] || {
                echo "--offload-profile requires one value" >&2
                exit 2
            }
            offload_profile="$2"
            shift 2
            ;;
        --r1-approval)
            [[ $# -ge 2 && -z "${r1_approval}" ]] || {
                echo "--r1-approval requires one path" >&2
                exit 2
            }
            r1_approval="$2"
            shift 2
            ;;
        --budget-projection)
            [[ $# -ge 2 && -z "${budget_projection}" ]] || {
                echo "--budget-projection requires one path" >&2
                exit 2
            }
            budget_projection="$2"
            shift 2
            ;;
        --keep-running)
            keep_running=yes
            shift
            ;;
        --retry-failed-stage)
            retry_failed_stage=yes
            shift
            ;;
        --dry-run)
            dry_run=yes
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown launch argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done
case "${phase}" in
    cpu|gpu-gates|gpu-capacity|gpu-bc40|gpu-bc80|gpu-export) ;;
    *) echo "invalid phase: ${phase}" >&2; usage >&2; exit 2 ;;
esac
if [[ "${phase}" == "gpu-capacity" ]]; then
    [[ "${offload_profile}" == "r0" || "${offload_profile}" == "r1" ]] || {
        echo "gpu-capacity requires --offload-profile r0 or r1" >&2
        exit 2
    }
    if [[ "${offload_profile}" == "r1" && -z "${r1_approval}" ]]; then
        echo "R1 capacity requires --r1-approval" >&2
        exit 2
    fi
    if [[ "${offload_profile}" == "r1" && -z "${budget_projection}" ]]; then
        echo "R1 capacity requires --budget-projection for the approved B/C40 projection" >&2
        exit 2
    fi
    if [[ "${offload_profile}" == "r0" && \
          ( -n "${r1_approval}" || -n "${budget_projection}" ) ]]; then
        echo "R0 capacity must not carry R1 approval or budget evidence" >&2
        exit 2
    fi
elif [[ "${phase}" == "gpu-bc40" || "${phase}" == "gpu-bc80" ]]; then
    [[ -n "${budget_projection}" ]] || {
        echo "${phase} requires --budget-projection" >&2
        exit 2
    }
    [[ -z "${offload_profile}" && -z "${r1_approval}" ]] || {
        echo "offload and R1 approval arguments are only valid for gpu-capacity" >&2
        exit 2
    }
elif [[ -n "${offload_profile}" || -n "${r1_approval}" || \
        -n "${budget_projection}" ]]; then
    echo "offload and R1 approval arguments are only valid for gpu-capacity" >&2
    exit 2
fi

rememr1_validate_test_mode
CLOUD_ENV="${REMEMR1_CLOUD_ENV:-/root/autodl-tmp/rememr1-cloud.env}"
rememr1_load_cloud_env "${CLOUD_ENV}"
rememr1_require_cloud_env
rememr1_validate_test_mode
export REMEMR1_CLOUD_ENV="${CLOUD_ENV}"

[[ -d "${REMEMR1_PROJECT_DIR}" ]] || { echo "project directory is missing" >&2; exit 1; }
[[ -d "${PERSIST_ROOT}" ]] || { echo "persistent root is missing" >&2; exit 1; }
[[ -f "${CAPABILITY_FILE}" ]] || { echo "shutdown capability is missing" >&2; exit 1; }
mkdir -p -- "${LAUNCHER_ROOT}" "$(dirname -- "${LOCK_FILE}")"
persist_real="$(rememr1_realpath_existing "${PERSIST_ROOT}")"
launcher_root_real="$(rememr1_realpath_existing "${LAUNCHER_ROOT}")"
rememr1_path_is_within "${launcher_root_real}" "${persist_real}" || {
    echo "LAUNCHER_ROOT must be inside PERSIST_ROOT" >&2
    exit 1
}
if [[ -n "${r1_approval}" ]]; then
    [[ "${r1_approval}" == /* && -f "${r1_approval}" && ! -L "${r1_approval}" ]] || {
        echo "R1 approval must be an existing absolute regular file" >&2
        exit 1
    }
    r1_approval="$(rememr1_realpath_existing "${r1_approval}")"
    rememr1_path_is_within "${r1_approval}" "${persist_real}" || {
        echo "R1 approval must be inside PERSIST_ROOT" >&2
        exit 1
    }
    r1_approval_file_sha256="$(sha256sum "${r1_approval}" | awk '{print $1}')"
fi
budget_projection_file_sha256=""
if [[ -n "${budget_projection}" ]]; then
    [[ "${budget_projection}" == /* && -f "${budget_projection}" && \
       ! -L "${budget_projection}" ]] || {
        echo "budget projection must be an existing absolute regular file" >&2
        exit 1
    }
    budget_projection="$(rememr1_realpath_existing "${budget_projection}")"
    rememr1_path_is_within "${budget_projection}" "${persist_real}" || {
        echo "budget projection must be inside PERSIST_ROOT" >&2
        exit 1
    }
    budget_projection_file_sha256="$(sha256sum "${budget_projection}" | awk '{print $1}')"
fi

worker="${SCRIPT_DIR}/launcher_worker.sh"
[[ -f "${worker}" ]] || { echo "launcher worker is missing: ${worker}" >&2; exit 1; }
for executable in /usr/bin/bash /usr/bin/nohup /usr/bin/setsid /usr/bin/sync \
    /usr/bin/timeout; do
    [[ -x "${executable}" ]] || { echo "required executable is missing: ${executable}" >&2; exit 1; }
done

if [[ "${REMEMR1_TEST_MODE:-no}" != "yes" ]]; then
    [[ "${EUID}" -eq 0 ]] || { echo "production launch must run as root" >&2; exit 1; }
    head="$(/usr/bin/git -C "${REMEMR1_PROJECT_DIR}" rev-parse HEAD 2>/dev/null)" || {
        echo "cannot read project HEAD" >&2
        exit 1
    }
    [[ "${head}" == "${EXPECTED_COMMIT}" ]] || {
        echo "project HEAD differs from EXPECTED_COMMIT" >&2
        exit 1
    }
    [[ -z "$(/usr/bin/git -C "${REMEMR1_PROJECT_DIR}" status --porcelain --untracked-files=normal)" ]] || {
        echo "project worktree is dirty" >&2
        exit 1
    }
    if [[ "${keep_running}" != "yes" && "${dry_run}" != "yes" ]]; then
        verify_guest_shutdown_preflight || {
            echo "automatic shutdown preflight failed before launching work" >&2
            exit 1
        }
    fi
fi

stamp="$(/usr/bin/date -u +%Y%m%dT%H%M%SZ)"
short_commit="${EXPECTED_COMMIT:0:12}"
launcher=""
for attempt in 1 2 3 4 5; do
    candidate="${launcher_root_real}/${stamp}-${phase}-${short_commit}-$$-${RANDOM:-0}-${attempt}"
    if mkdir -- "${candidate}" 2>/dev/null; then
        launcher="${candidate}"
        break
    fi
done
[[ -n "${launcher}" ]] || { echo "could not create a unique launcher directory" >&2; exit 1; }
launcher="$(rememr1_realpath_existing "${launcher}")"

started_at="$(rememr1_utc_now)"
request_json="$(printf \
    '{\n  "schema_version": 1,\n  "phase": "%s",\n  "experiment_profile_id": "%s",\n  "expected_commit": "%s",\n  "offload_profile": "%s",\n  "r1_approval": "%s",\n  "r1_approval_file_sha256": "%s",\n  "budget_projection": "%s",\n  "budget_projection_file_sha256": "%s",\n  "keep_running": "%s",\n  "retry_failed_stage": "%s",\n  "dry_run": "%s",\n  "requested_at": "%s"\n}' \
    "$(rememr1_json_escape "${phase}")" \
    "$(rememr1_json_escape "${REMEMR1_EXPERIMENT_PROFILE}")" \
    "${EXPECTED_COMMIT}" "$(rememr1_json_escape "${offload_profile}")" \
    "$(rememr1_json_escape "${r1_approval}")" \
    "${r1_approval_file_sha256}" \
    "$(rememr1_json_escape "${budget_projection}")" \
    "${budget_projection_file_sha256}" \
    "${keep_running}" "${retry_failed_stage}" "${dry_run}" \
    "${started_at}")"
atomic_write "${launcher}/request.json" "${request_json}"
atomic_write "${launcher}/status" "starting"
atomic_write "${launcher}/.starting" "${started_at}"

worker_args=(--phase "${phase}" --launcher-dir "${launcher}")
[[ -n "${offload_profile}" ]] && worker_args+=(--offload-profile "${offload_profile}")
if [[ -n "${r1_approval}" ]]; then
    worker_args+=(--r1-approval "${r1_approval}" \
        --r1-approval-file-sha256 "${r1_approval_file_sha256}")
fi
if [[ -n "${budget_projection}" ]]; then
    worker_args+=(--budget-projection "${budget_projection}" \
        --budget-projection-file-sha256 "${budget_projection_file_sha256}")
fi
[[ "${keep_running}" == "yes" ]] && worker_args+=(--keep-running)
[[ "${retry_failed_stage}" == "yes" ]] && worker_args+=(--retry-failed-stage)
[[ "${dry_run}" == "yes" ]] && worker_args+=(--dry-run)

/usr/bin/nohup /usr/bin/setsid /usr/bin/bash "${worker}" "${worker_args[@]}" \
    >> "${launcher}/launcher.log" 2>&1 < /dev/null &
launcher_pid=$!
atomic_write "${launcher}/launcher-pid" "${launcher_pid}"

printf 'launcher_dir=%s\n' "${launcher}"
printf 'launcher_pid=%s\n' "${launcher_pid}"
printf 'launcher_log=%s/launcher.log\n' "${launcher}"
printf 'launcher_status=%s/status\n' "${launcher}"
printf 'status_command=%q %q\n' "${SCRIPT_DIR}/status.sh" "${launcher}"

ready=no
for _ in {1..150}; do
    if [[ -f "${launcher}/exit-code" ]]; then
        ready=terminal
        break
    fi
    if [[ -f "${launcher}/lock-acquired" ]]; then
        ready=running
        break
    fi
    /usr/bin/sleep 0.1
done
if [[ "${ready}" == "terminal" ]]; then
    early_rc="$(<"${launcher}/exit-code")"
    if [[ "${early_rc}" != "0" ]]; then
        /usr/bin/tail -n 80 "${launcher}/launcher.log" >&2 || true
        exit "${early_rc}"
    fi
elif [[ "${ready}" != "running" ]]; then
    echo "launcher did not acquire the cloud lock or publish an exit code within 15 seconds" >&2
    /usr/bin/tail -n 80 "${launcher}/launcher.log" >&2 || true
    exit 1
fi
