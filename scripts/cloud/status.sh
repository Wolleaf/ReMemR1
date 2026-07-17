#!/usr/bin/env bash
# Read durable launcher state without modifying it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/runtime.sh
source "${SCRIPT_DIR}/lib/runtime.sh"

usage() {
    echo "Usage: status.sh [LAUNCHER_DIR|--latest]" >&2
}

requested="${1:---latest}"
[[ $# -le 1 ]] || { usage; exit 2; }
if [[ -z "${LAUNCHER_ROOT:-}" ]]; then
    CLOUD_ENV="${REMEMR1_CLOUD_ENV:-/root/autodl-tmp/rememr1-cloud.env}"
    rememr1_load_cloud_env "${CLOUD_ENV}"
fi
rememr1_require_cloud_env
root="$(rememr1_realpath_existing "${LAUNCHER_ROOT}")"

if [[ "${requested}" == "--latest" ]]; then
    launcher=""
    shopt -s nullglob
    for candidate in "${root}"/*; do
        [[ -d "${candidate}" ]] || continue
        if [[ -z "${launcher}" || "${candidate##*/}" > "${launcher##*/}" ]]; then
            launcher="${candidate}"
        fi
    done
    shopt -u nullglob
    [[ -n "${launcher}" ]] || { echo "no launcher directories in ${root}" >&2; exit 1; }
else
    launcher="${requested}"
fi
launcher="$(rememr1_realpath_existing "${launcher}")"
rememr1_path_is_within "${launcher}" "${root}" || {
    echo "launcher is outside LAUNCHER_ROOT" >&2
    exit 1
}

state=unknown
if [[ -f "${launcher}/.success" ]]; then
    state=success
elif [[ -f "${launcher}/.scientific-stop" ]]; then
    state=scientific-stop
elif [[ -f "${launcher}/.capacity-stop" ]]; then
    state=capacity-stop
elif [[ -f "${launcher}/lock-busy" ]]; then
    state=lock-busy
elif [[ -f "${launcher}/.failed" ]]; then
    state=failed
elif [[ -f "${launcher}/.running" ]]; then
    worker_pid="$(<"${launcher}/.running")"
    if [[ "${worker_pid}" =~ ^[0-9]+$ ]] && kill -0 "${worker_pid}" 2>/dev/null; then
        state=running
    else
        state=interrupted
    fi
elif [[ -f "${launcher}/.starting" ]]; then
    state=starting
elif [[ -f "${launcher}/status" ]]; then
    IFS= read -r state < "${launcher}/status" || state=unknown
fi

printf 'launcher_dir=%s\n' "${launcher}"
printf 'state=%s\n' "${state}"
for field in launcher-pid worker-pid pid exit-code pipeline-result shutdown-backend shutdown-skipped \
    shutdown-requested shutdown-dispatched shutdown-failed retryable; do
    if [[ -f "${launcher}/${field}" ]]; then
        value="$(<"${launcher}/${field}")"
        printf '%s=%s\n' "${field}" "${value}"
    fi
done
if [[ -s "${launcher}/pipeline-result" ]]; then
    pipeline="$(<"${launcher}/pipeline-result")"
    if [[ -d "${pipeline}" ]]; then
        printf 'pipeline_dir=%s\n' "${pipeline}"
        pipeline_state="${pipeline}"
        if [[ -f "${launcher}/pipeline-result.terminal" && \
              ! -L "${launcher}/pipeline-result.terminal" ]]; then
            candidate="$(<"${launcher}/pipeline-result.terminal")"
            if [[ ( "${candidate}" == "${pipeline}/terminals/"* || \
                    "${candidate}" == "${pipeline}/launcher-terminals/"* ) && \
                  -d "${candidate}" && ! -L "${candidate}" ]]; then
                pipeline_state="${candidate}"
                printf 'pipeline_terminal_dir=%s\n' "${pipeline_state}"
            fi
        fi
        for field in current-phase current-stage current-attempt failed-phase failed-stage \
            scientific-stop-phase scientific-stop-stage capacity-stop-phase \
            capacity-stop-stage retryable last-successful-phase; do
            field_root="${pipeline_state}"
            [[ "${field}" != "last-successful-phase" ]] || field_root="${pipeline}"
            if [[ -f "${field_root}/${field}" ]]; then
                value="$(<"${field_root}/${field}")"
                printf 'pipeline_%s=%s\n' "${field}" "${value}"
            fi
        done
    fi
fi
if [[ -f "${launcher}/terminal.json" ]]; then
    echo "terminal_json:"
    cat -- "${launcher}/terminal.json"
fi
