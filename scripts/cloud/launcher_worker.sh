#!/usr/bin/env bash
# Detached worker: lock, run, publish terminal state, then optionally shut down.
set -uo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/runtime.sh
source "${SCRIPT_DIR}/lib/runtime.sh"
# shellcheck source=lib/lock.sh
source "${SCRIPT_DIR}/lib/lock.sh"
# shellcheck source=lib/shutdown.sh
source "${SCRIPT_DIR}/lib/shutdown.sh"

phase=""
launcher_dir=""
keep_running=no
retry_failed_stage=no
dry_run=no
offload_profile=""
r1_approval=""
r1_approval_file_sha256=""
budget_projection=""
budget_projection_file_sha256=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase) phase="${2:-}"; shift 2 ;;
        --launcher-dir) launcher_dir="${2:-}"; shift 2 ;;
        --keep-running) keep_running=yes; shift ;;
        --retry-failed-stage) retry_failed_stage=yes; shift ;;
        --dry-run) dry_run=yes; shift ;;
        --offload-profile) offload_profile="${2:-}"; shift 2 ;;
        --r1-approval) r1_approval="${2:-}"; shift 2 ;;
        --r1-approval-file-sha256)
            r1_approval_file_sha256="${2:-}"
            shift 2
            ;;
        --budget-projection) budget_projection="${2:-}"; shift 2 ;;
        --budget-projection-file-sha256)
            budget_projection_file_sha256="${2:-}"
            shift 2
            ;;
        *) echo "unknown worker argument: $1" >&2; exit 2 ;;
    esac
done
case "${phase}" in
    cpu|gpu|gpu-gates|gpu-capacity|gpu-bc40|gpu-bc80|gpu-export) ;;
    *) echo "invalid worker phase: ${phase}" >&2; exit 2 ;;
esac
if [[ "${phase}" == "gpu" ]]; then
    [[ "${offload_profile}" == "r0" && -z "${r1_approval}" && \
       -z "${r1_approval_file_sha256}" && -z "${budget_projection}" && \
       -z "${budget_projection_file_sha256}" ]] || exit 2
elif [[ "${phase}" == "gpu-capacity" ]]; then
    [[ "${offload_profile}" == "r0" || "${offload_profile}" == "r1" ]] || exit 2
    [[ "${offload_profile}" != "r1" || \
       ( -n "${r1_approval}" && \
         "${r1_approval_file_sha256}" =~ ^[0-9a-f]{64}$ ) ]] || exit 2
    [[ "${offload_profile}" != "r1" || -n "${budget_projection}" ]] || exit 2
    [[ "${offload_profile}" != "r0" || \
       ( -z "${r1_approval}" && -z "${r1_approval_file_sha256}" && \
         -z "${budget_projection}" ) ]] || exit 2
elif [[ "${phase}" == "gpu-bc40" || "${phase}" == "gpu-bc80" ]]; then
    [[ -n "${budget_projection}" && -z "${offload_profile}" && \
       -z "${r1_approval}" && -z "${r1_approval_file_sha256}" ]] || exit 2
elif [[ -n "${offload_profile}" || -n "${r1_approval}" || \
        -n "${r1_approval_file_sha256}" || \
        -n "${budget_projection}" ]]; then
    exit 2
fi
if [[ -n "${budget_projection}" ]]; then
    [[ "${budget_projection_file_sha256}" =~ ^[0-9a-f]{64}$ ]] || exit 2
elif [[ -n "${budget_projection_file_sha256}" ]]; then
    exit 2
fi
rememr1_validate_test_mode || exit 64
rememr1_require_cloud_env || exit 64
if [[ -z "${launcher_dir}" || ! -d "${launcher_dir}" ]]; then
    echo "launcher directory is missing" >&2
    exit 64
fi
launcher_dir="$(rememr1_realpath_existing "${launcher_dir}")" || exit 64
launcher_root_real="$(rememr1_realpath_existing "${LAUNCHER_ROOT}")" || exit 64
persist_real="$(rememr1_realpath_existing "${PERSIST_ROOT}")" || exit 64
rememr1_path_is_within "${launcher_dir}" "${launcher_root_real}" || exit 64
rememr1_path_is_within "${launcher_dir}" "${persist_real}" || exit 64
if [[ -n "${r1_approval}" ]]; then
    [[ "${r1_approval}" == /* && -f "${r1_approval}" && ! -L "${r1_approval}" ]] || exit 64
    r1_approval="$(rememr1_realpath_existing "${r1_approval}")" || exit 64
    rememr1_path_is_within "${r1_approval}" "${persist_real}" || exit 64
    observed_r1_approval_sha256="$(sha256sum "${r1_approval}" | awk '{print $1}')" || exit 64
    [[ "${observed_r1_approval_sha256}" == \
       "${r1_approval_file_sha256}" ]] || exit 64
fi
if [[ -n "${budget_projection}" ]]; then
    [[ "${budget_projection}" == /* && -f "${budget_projection}" && \
       ! -L "${budget_projection}" ]] || exit 64
    budget_projection="$(rememr1_realpath_existing "${budget_projection}")" || exit 64
    rememr1_path_is_within "${budget_projection}" "${persist_real}" || exit 64
    observed_budget_sha256="$(sha256sum "${budget_projection}" | awk '{print $1}')" || exit 64
    [[ "${observed_budget_sha256}" == "${budget_projection_file_sha256}" ]] || exit 64
fi
REMEMR1_LAUNCHER_DIR="${launcher_dir}"
export REMEMR1_LAUNCHER_DIR

started_at="$(rememr1_utc_now)"
lock_acquired=no
outcome=failed
terminal_publish_failure=write
pipeline_invoked=no
primary_result_file=""
composite_active_phase=""
composite_active_result_file=""

launcher_test_failure_requested() {
    [[ "${REMEMR1_TEST_MODE:-no}" == "yes" && \
       "${REMEMR1_TEST_LAUNCHER_FAIL_AT:-}" == "$1" ]]
}

sync_launcher_state() {
    local point="$1"
    rememr1_test_event "launcher-sync-started point=${point} launcher_dir=${launcher_dir}" || return
    if launcher_test_failure_requested "${point}"; then
        rememr1_test_event \
            "launcher-sync-failed point=${point} launcher_dir=${launcher_dir}" || true
        return 1
    fi
    if ! rememr1_sync_all; then
        rememr1_test_event \
            "launcher-sync-failed point=${point} launcher_dir=${launcher_dir}" || true
        return 1
    fi
    rememr1_test_event \
        "launcher-sync-complete point=${point} launcher_dir=${launcher_dir}"
}

remove_running_marker() {
    if launcher_test_failure_requested "running-marker-remove"; then
        rememr1_test_event \
            "running-marker-remove-failed launcher_dir=${launcher_dir}" || true
        return 1
    fi
    if ! rm -f -- "${launcher_dir}/.running"; then
        rememr1_test_event \
            "running-marker-remove-failed launcher_dir=${launcher_dir}" || true
        return 1
    fi
    rememr1_test_event "running-marker-removed launcher_dir=${launcher_dir}"
}

replace_pointer_from_file() {
    local source="$1"
    local destination="$2"
    local label="$3"
    local value resolved
    if [[ ! -e "${source}" && ! -L "${source}" ]]; then
        rm -f -- "${destination}"
        return 0
    fi
    [[ -f "${source}" && ! -L "${source}" ]] || {
        echo "${label} source is unsafe: ${source}" >&2
        return 1
    }
    value="$(<"${source}")"
    [[ "${value}" == /* && "${value}" != *$'\n'* && \
       -d "${value}" && ! -L "${value}" ]] || {
        echo "${label} source contains an invalid directory pointer" >&2
        return 1
    }
    resolved="$(rememr1_realpath_existing "${value}")" || return
    rememr1_path_is_within "${resolved}" "${persist_real}" || {
        echo "${label} source escaped persistent storage" >&2
        return 1
    }
    atomic_write "${destination}" "${resolved}"
}

publish_active_composite_result() {
    [[ "${phase}" == "gpu" && -n "${composite_active_result_file}" && \
       -n "${primary_result_file}" ]] || return 0
    [[ -f "${composite_active_result_file}" && \
       ! -L "${composite_active_result_file}" ]] || {
        rm -f -- "${primary_result_file}" "${primary_result_file}.terminal"
        echo "composite subphase did not publish a safe pipeline result" >&2
        return 1
    }
    replace_pointer_from_file "${composite_active_result_file}" \
        "${primary_result_file}" "composite pipeline result" || return
    local terminal_source="${composite_active_result_file}.terminal"
    if [[ -e "${terminal_source}" || -L "${terminal_source}" ]]; then
        [[ -e "${composite_active_result_file}" && \
           ! -L "${composite_active_result_file}" ]] || {
            echo "composite terminal pointer exists without a pipeline result" >&2
            return 1
        }
    fi
    if [[ ! -e "${terminal_source}" && ! -L "${terminal_source}" && \
          "${dry_run}" != "yes" ]]; then
        rm -f -- "${primary_result_file}.terminal"
        echo "composite subphase did not publish a pipeline terminal pointer" >&2
        return 1
    fi
    replace_pointer_from_file "${terminal_source}" \
        "${primary_result_file}.terminal" "composite pipeline terminal"
}

publish_terminal_state() {
    local rc="$1"
    local final_outcome="$2"
    local finished_at result_path pipeline_terminal_dir retryable retry_hint terminal_json terminal_text
    terminal_publish_failure=write
    finished_at="$(rememr1_utc_now)"
    result_path=""
    if [[ -f "${REMEMR1_RESULT_FILE:-}" ]]; then
        IFS= read -r result_path < "${REMEMR1_RESULT_FILE}" || result_path=""
    fi
    pipeline_terminal_dir=""
    if [[ -f "${REMEMR1_RESULT_FILE:-}.terminal" && \
          ! -L "${REMEMR1_RESULT_FILE}.terminal" ]]; then
        pipeline_terminal_dir="$(<"${REMEMR1_RESULT_FILE}.terminal")"
    fi
    rm -f -- "${launcher_dir}/.starting" || return
    rm -f -- "${launcher_dir}/.success" "${launcher_dir}/.failed" \
        "${launcher_dir}/.scientific-stop" "${launcher_dir}/.capacity-stop" \
        "${launcher_dir}/retryable" || return
    retry_hint=""
    if [[ "${rc}" -eq 0 ]]; then
        retryable=false
    elif [[ "${rc}" -eq 42 || "${rc}" -eq 43 ]]; then
        retryable=false
        retry_hint="terminal scientific/capacity result; do not retry"
    elif [[ "${pipeline_invoked}" == yes && "${dry_run}" != yes && \
            "${phase}" == gpu-capacity && "${offload_profile}" == r1 ]]; then
        retryable=false
        retry_hint="create a new R1 approval nonce and launch a new immutable generation"
    elif [[ "${pipeline_invoked}" == yes && "${dry_run}" != yes && \
            ( "${phase}" == gpu-bc40 || "${phase}" == gpu-bc80 ) ]]; then
        retryable=false
        retry_hint="create a new budget projection and launch a new immutable generation"
    else
        retryable=true
        retry_hint="inspect the failure, then relaunch with --retry-failed-stage"
    fi
    terminal_json="$(printf \
        '{\n  "schema_version": 1,\n  "phase": "%s",\n  "experiment_profile_id": "%s",\n  "offload_profile": "%s",\n  "budget_projection": "%s",\n  "budget_projection_file_sha256": "%s",\n  "outcome": "%s",\n  "exit_code": %s,\n  "retryable": %s,\n  "retry_hint": "%s",\n  "expected_commit": "%s",\n  "started_at": "%s",\n  "finished_at": "%s",\n  "pipeline_result": "%s",\n  "pipeline_terminal_dir": "%s"\n}' \
        "$(rememr1_json_escape "${phase}")" \
        "$(rememr1_json_escape "${REMEMR1_EXPERIMENT_PROFILE}")" \
        "$(rememr1_json_escape "${offload_profile}")" \
        "$(rememr1_json_escape "${budget_projection}")" \
        "${budget_projection_file_sha256}" \
        "$(rememr1_json_escape "${final_outcome}")" "${rc}" "${retryable}" \
        "$(rememr1_json_escape "${retry_hint}")" \
        "${EXPECTED_COMMIT}" \
        "${started_at}" "${finished_at}" "$(rememr1_json_escape "${result_path}")" \
        "$(rememr1_json_escape "${pipeline_terminal_dir}")")"
    terminal_text="$(printf \
        'phase=%s\noutcome=%s\nexit_code=%s\nretryable=%s\nretry_hint=%s\nstarted_at=%s\nfinished_at=%s\npipeline_result=%s\npipeline_terminal_dir=%s' \
        "${phase}" "${final_outcome}" "${rc}" "${retryable}" "${retry_hint}" \
        "${started_at}" "${finished_at}" "${result_path}" \
        "${pipeline_terminal_dir}")"
    [[ -f "${launcher_dir}/launcher.log" && ! -L "${launcher_dir}/launcher.log" ]] || return
    printf '[launcher] terminal-state-published exit_code=%s outcome=%s\n' \
        "${rc}" "${final_outcome}" >> "${launcher_dir}/launcher.log" || return
    atomic_write "${launcher_dir}/terminal.json" "${terminal_json}" || return
    atomic_write "${launcher_dir}/terminal" "${terminal_text}" || return
    atomic_write "${launcher_dir}/exit-code" "${rc}" || return
    atomic_write "${launcher_dir}/status" "${final_outcome}" || return
    if [[ "${rc}" -eq 0 ]]; then
        atomic_write "${launcher_dir}/.success" "0" || return
    elif [[ "${rc}" -eq 42 ]]; then
        atomic_write "${launcher_dir}/.scientific-stop" "42" || return
    elif [[ "${rc}" -eq 43 ]]; then
        atomic_write "${launcher_dir}/.capacity-stop" "43" || return
    else
        atomic_write "${launcher_dir}/.failed" "${rc}" || return
    fi
    atomic_write "${launcher_dir}/retryable" "${retryable}" || return
    rememr1_test_event \
        "terminal-state-written exit_code=${rc} launcher_dir=${launcher_dir}" || return
    terminal_publish_failure=terminal-state-sync
    sync_launcher_state "terminal-state-sync" || return
    terminal_publish_failure=running-marker-remove
    remove_running_marker || return
    terminal_publish_failure=running-marker-sync
    sync_launcher_state "running-marker-sync" || return
    rememr1_test_event "exit-code-written exit_code=${rc} launcher_dir=${launcher_dir}" || return
    terminal_publish_failure=none
}

finish_worker() {
    local rc="$1"
    trap - EXIT INT TERM
    set +e
    local publish_ok=no reserve sync_rc pipeline_state pipeline_shutdown_inhibited=no
    if [[ "${phase}" == "gpu" && -n "${composite_active_phase}" ]] && \
       ! publish_active_composite_result; then
        pipeline_shutdown_inhibited=yes
        atomic_write "${launcher_dir}/composite-result-transfer-failed" \
            "${composite_active_phase}" 2>/dev/null || true
    fi
    if publish_terminal_state "${rc}" "${outcome}"; then
        publish_ok=yes
    else
        reserve="${REMEMR1_RESERVE_FILE:-${PERSIST_ROOT}/cloud/terminal-reserve}"
        if [[ "${terminal_publish_failure}" == "write" && \
              "${lock_acquired}" == "yes" ]] && \
           rememr1_path_is_within "${reserve}" "${PERSIST_ROOT}" && \
           [[ -f "${reserve}" && ! -L "${reserve}" ]] && \
           /usr/bin/truncate -s 0 -- "${reserve}"; then
            atomic_write "${launcher_dir}/reserve-released" "$(rememr1_utc_now)" 2>/dev/null || true
            if publish_terminal_state "${rc}" "${outcome}"; then
                publish_ok=yes
            fi
        fi
        if [[ "${publish_ok}" != "yes" ]]; then
            atomic_write "${launcher_dir}/persistence-failed" \
                "terminal state publication failed at $(rememr1_utc_now)" 2>/dev/null || true
        fi
    fi

    if [[ "${publish_ok}" == "yes" ]]; then
        sync_rc=0
    else
        sync_rc=1
    fi
    if [[ -s "${REMEMR1_RESULT_FILE:-}" ]]; then
        pipeline_state="$(<"${REMEMR1_RESULT_FILE}")"
        if [[ "${pipeline_state}" == /* && -d "${pipeline_state}" && \
              ! -L "${pipeline_state}" ]] && \
           rememr1_path_is_within "${pipeline_state}" "${persist_real}" && \
           [[ -f "${pipeline_state}/shutdown-inhibited" && \
              ! -L "${pipeline_state}/shutdown-inhibited" ]]; then
            pipeline_shutdown_inhibited=yes
        fi
    fi
    if [[ "${lock_acquired}" == "yes" && "${sync_rc}" -eq 0 && \
          "${pipeline_shutdown_inhibited}" != yes ]]; then
        if atomic_write "${launcher_dir}/shutdown-safe" "$(rememr1_utc_now)"; then
            rememr1_sync_all
            sync_rc=$?
            if [[ "${sync_rc}" -eq 0 ]]; then
                rememr1_test_event "shutdown-safe-written launcher_dir=${launcher_dir}" || sync_rc=1
            fi
        else
            sync_rc=1
        fi
    fi

    if [[ "${lock_acquired}" != "yes" ]]; then
        atomic_write "${launcher_dir}/shutdown-skipped" "lock-not-acquired" 2>/dev/null || true
    elif [[ "${sync_rc}" -ne 0 ]]; then
        rm -f -- "${launcher_dir}/shutdown-safe"
        atomic_write "${launcher_dir}/shutdown-skipped" "durable-state-sync-failed" 2>/dev/null || true
    elif [[ "${pipeline_shutdown_inhibited}" == yes ]]; then
        rm -f -- "${launcher_dir}/shutdown-safe"
        atomic_write "${launcher_dir}/shutdown-skipped" \
            "pipeline-state-incomplete" 2>/dev/null || true
        rememr1_sync_all || true
    elif [[ "${keep_running}" == "yes" ]]; then
        atomic_write "${launcher_dir}/shutdown-skipped" "keep-running" || true
        rememr1_sync_all || true
    elif [[ "${dry_run}" == "yes" ]]; then
        atomic_write "${launcher_dir}/shutdown-skipped" "dry-run" || true
        rememr1_sync_all || true
    elif [[ "${REMEMR1_TEST_MODE:-no}" == "yes" ]]; then
        request_guest_shutdown "${launcher_dir}" "${phase}" "${rc}" || true
        atomic_write "${launcher_dir}/shutdown-skipped" "test-mode" || true
        rememr1_sync_all || true
    else
        if request_guest_shutdown "${launcher_dir}" "${phase}" "${rc}"; then
            atomic_write "${launcher_dir}/shutdown-dispatched" "$(rememr1_utc_now)" || true
        fi
    fi
    exit "${rc}"
}

trap 'finish_worker "$?"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

atomic_write "${launcher_dir}/worker-pid" "$$" || exit 74
atomic_write "${launcher_dir}/pid" "$$" || exit 74
atomic_write "${launcher_dir}/status" "waiting-for-lock" || exit 74
if acquire_cloud_lock; then
    lock_acquired=yes
else
    lock_rc=$?
    if [[ "${lock_rc}" -eq 75 ]]; then
        outcome=lock-busy
        atomic_write "${launcher_dir}/lock-busy" "$(rememr1_utc_now)" || true
        exit 75
    fi
    outcome=lock-error
    exit "${lock_rc}"
fi

rememr1_ensure_terminal_reserve || exit 74
atomic_write "${launcher_dir}/.running" "$$" || exit 74
rm -f -- "${launcher_dir}/.starting" || exit 74
atomic_write "${launcher_dir}/status" "running" || exit 74
REMEMR1_RESULT_FILE="${launcher_dir}/pipeline-result"
export REMEMR1_RESULT_FILE
primary_result_file="${REMEMR1_RESULT_FILE}"

pipeline="${REMEMR1_PROJECT_DIR}/scripts/cloud/run_pipeline.sh"
run_pipeline_phase() {
    local subphase="$1"
    shift
    local launcher_key invocation_root synthetic_launcher subphase_result rc
    local -a args=(--phase "${subphase}" "$@")
    launcher_key="$(basename -- "${launcher_dir}")"
    invocation_root="${launcher_dir}/pipeline-invocations"
    if [[ ! -e "${invocation_root}" && ! -L "${invocation_root}" ]]; then
        mkdir -- "${invocation_root}" || return 74
    fi
    [[ -d "${invocation_root}" && ! -L "${invocation_root}" ]] || return 74
    synthetic_launcher="${invocation_root}/${launcher_key}-${subphase}"
    mkdir -- "${synthetic_launcher}" || return 74
    subphase_result="${launcher_dir}/pipeline-result.${subphase}"
    composite_active_phase="${subphase}"
    composite_active_result_file="${subphase_result}"
    atomic_write "${launcher_dir}/composite-subphase" "${subphase}" || return 74
    rm -f -- "${primary_result_file}" "${primary_result_file}.terminal" || return 74
    [[ "${retry_failed_stage}" == "yes" ]] && args+=(--retry-failed-stage)
    [[ "${dry_run}" == "yes" ]] && args+=(--dry-run)
    if [[ "${subphase}" == "cpu-finalize" ]]; then
        (
            export CUDA_VISIBLE_DEVICES=''
            export NVIDIA_VISIBLE_DEVICES=void
            export REMEMR1_ALLOW_GPU_CPU_PHASE=yes
            export HF_HUB_OFFLINE=1
            export HF_DATASETS_OFFLINE=1
            export TRANSFORMERS_OFFLINE=1
            export WANDB_MODE=disabled
            export REMEMR1_RESULT_FILE="${subphase_result}"
            export REMEMR1_LAUNCHER_DIR="${synthetic_launcher}"
            /usr/bin/bash "${pipeline}" "${args[@]}"
        )
        rc=$?
    else
        (
            export REMEMR1_RESULT_FILE="${subphase_result}"
            export REMEMR1_LAUNCHER_DIR="${synthetic_launcher}"
            /usr/bin/bash "${pipeline}" "${args[@]}"
        )
        rc=$?
    fi
    publish_active_composite_result || return 74
    return "${rc}"
}

if [[ ! -f "${pipeline}" ]]; then
    echo "pipeline script is missing: ${pipeline}" >&2
    pipeline_rc=66
else
    pipeline_invoked=yes
    if [[ "${phase}" == "gpu" ]]; then
        pipeline_rc=0
        run_pipeline_phase cpu-finalize
        pipeline_rc=$?
        if [[ "${pipeline_rc}" -eq 0 ]]; then
            run_pipeline_phase gpu-gates
            pipeline_rc=$?
        fi
        if [[ "${pipeline_rc}" -eq 0 ]]; then
            run_pipeline_phase gpu-capacity --offload-profile r0
            pipeline_rc=$?
        fi
    else
        pipeline_args=(--phase "${phase}")
        [[ -n "${offload_profile}" ]] && pipeline_args+=(--offload-profile "${offload_profile}")
        [[ -n "${r1_approval}" ]] && pipeline_args+=(--r1-approval "${r1_approval}")
        if [[ -n "${budget_projection}" ]]; then
            pipeline_args+=(--budget-projection "${budget_projection}" \
                --budget-projection-file-sha256 "${budget_projection_file_sha256}")
        fi
        [[ "${retry_failed_stage}" == "yes" ]] && pipeline_args+=(--retry-failed-stage)
        [[ "${dry_run}" == "yes" ]] && pipeline_args+=(--dry-run)
        /usr/bin/bash "${pipeline}" "${pipeline_args[@]}"
        pipeline_rc=$?
    fi
fi
if [[ "${pipeline_rc}" -eq 0 ]]; then
    outcome=success
elif [[ "${pipeline_rc}" -eq 42 ]]; then
    outcome=scientific-stop
elif [[ "${pipeline_rc}" -eq 43 ]]; then
    outcome=capacity-stop
else
    outcome=failed
fi
exit "${pipeline_rc}"
