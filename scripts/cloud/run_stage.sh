#!/usr/bin/env bash
# Execute one bounded cloud stage with an immutable run directory and markers.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 STAGE" >&2
    exit 2
fi
STAGE="$1"
CLOUD_ENV="${REMEMR1_CLOUD_ENV:-/root/autodl-tmp/rememr1-cloud.env}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/runtime.sh"
source "${SCRIPT_DIR}/lib/lock.sh"
rememr1_load_cloud_env "${CLOUD_ENV}"
rememr1_require_cloud_env
require_cloud_lock
cd "${REMEMR1_PROJECT_DIR}"

case "${STAGE}" in
    cpu-preflight|cpu-environment|cpu-kernel-sources|cpu-assets|cpu-data|cpu-tests|cpu-configs|cpu-handoff|gpu-preflight|g0|g1-step1|g1-resume2|g1-artifacts|g2a|g2b-step1|g2b-resume5|g2-length-stress|g2-artifacts|capacity-seal|capacity-stop-seal|b-pilot|c-pilot|pilot-gate|b20|c20|b40|c40|bc40-artifacts|eval40|package40|b60|c60|b80|c80|bc80-artifacts|eval80|package80|export-results) ;;
    *) echo "unknown cloud stage: ${STAGE}" >&2; exit 2 ;;
esac

[[ -n "${REMEMR1_PIPELINE_DIR:-}" && -n "${REMEMR1_STAGE_RESULT_FILE:-}" ]] || {
    echo "run_stage.sh must be called by the locked cloud pipeline" >&2
    exit 1
}

PERSIST_REAL="$(rememr1_realpath_existing "${PERSIST_ROOT}")"
CLOUD_ROOT_RAW="${PERSIST_ROOT}/cloud"
[[ -d "${CLOUD_ROOT_RAW}" && ! -L "${CLOUD_ROOT_RAW}" ]] || {
    echo "cloud state root must be a regular directory" >&2
    exit 1
}
CLOUD_ROOT="$(rememr1_realpath_existing "${CLOUD_ROOT_RAW}")"
rememr1_path_is_within "${CLOUD_ROOT}" "${PERSIST_REAL}" || {
    echo "cloud state root escaped persistent storage" >&2
    exit 1
}
RUN_ROOT_RAW="${CLOUD_ROOT}/runs"
[[ -d "${RUN_ROOT_RAW}" && ! -L "${RUN_ROOT_RAW}" ]] || {
    echo "stage run root must be an existing non-symlink directory" >&2
    exit 1
}
RUN_ROOT="$(rememr1_realpath_existing "${RUN_ROOT_RAW}")"
[[ "${RUN_ROOT}" == "${CLOUD_ROOT}/runs" ]] || {
    echo "stage run root escaped the initialized cloud state" >&2
    exit 1
}
[[ -d "${REMEMR1_PIPELINE_DIR}" && ! -L "${REMEMR1_PIPELINE_DIR}" ]] || {
    echo "pipeline state must be an existing non-symlink directory" >&2
    exit 1
}
PIPELINE_REAL="$(rememr1_realpath_existing "${REMEMR1_PIPELINE_DIR}")"
PIPELINE_ROOT="$(rememr1_realpath_existing "${CLOUD_ROOT}/pipelines")"
rememr1_path_is_within "${PIPELINE_REAL}" "${PIPELINE_ROOT}" || {
    echo "pipeline state escaped the initialized pipeline root" >&2
    exit 1
}
ATTEMPT_BASE="${PIPELINE_REAL}/attempts"
[[ -d "${ATTEMPT_BASE}" && ! -L "${ATTEMPT_BASE}" && \
   "$(rememr1_realpath_existing "${ATTEMPT_BASE}")" == "${ATTEMPT_BASE}" ]] || {
    echo "pipeline attempt root is missing or unsafe" >&2
    exit 1
}
[[ -n "${REMEMR1_ATTEMPT_ROOT:-}" && -d "${REMEMR1_ATTEMPT_ROOT}" && \
   ! -L "${REMEMR1_ATTEMPT_ROOT}" ]] || {
    echo "scoped pipeline attempt root is missing or unsafe" >&2
    exit 1
}
ATTEMPT_ROOT="$(rememr1_realpath_existing "${REMEMR1_ATTEMPT_ROOT}")"
rememr1_path_is_within "${ATTEMPT_ROOT}" "${ATTEMPT_BASE}" || {
    echo "scoped pipeline attempt root escaped pipeline state" >&2
    exit 1
}
RESULT_REAL="$(rememr1_realpath_maybe "${REMEMR1_STAGE_RESULT_FILE}")"
rememr1_path_is_within "${RESULT_REAL}" "${ATTEMPT_ROOT}" || {
    echo "stage result file escaped the pipeline attempt root" >&2
    exit 1
}
[[ "$(dirname -- "${RESULT_REAL}")" == "${ATTEMPT_ROOT}" && \
   "$(basename -- "${RESULT_REAL}")" == "${STAGE}-"*.run ]] || {
    echo "stage result file does not match the expected attempt identity" >&2
    exit 1
}
stamp="$(date -u +%Y%m%dT%H%M%SZ)-$$-${RANDOM:-0}"
RUN_DIR="${RUN_ROOT}/${stamp}-${STAGE}"
[[ ! -e "${RUN_DIR}" && ! -L "${RUN_DIR}" ]] || {
    echo "stage run directory already exists: ${RUN_DIR}" >&2
    exit 1
}
mkdir -- "${RUN_DIR}"
RUN_DIR="$(rememr1_realpath_existing "${RUN_DIR}")"
[[ "$(dirname -- "${RUN_DIR}")" == "${RUN_ROOT}" ]] || {
    echo "stage run directory escaped the run root" >&2
    exit 1
}
mkdir -- "${RUN_DIR}/logs" "${RUN_DIR}/artifacts"
LOG_FILE="${RUN_DIR}/logs/stage.log"
STAGE_FINISHED="no"

write_atomic() {
    local destination="$1"
    local value="$2"
    local temporary="${destination}.tmp.$$"
    printf '%s\n' "${value}" > "${temporary}"
    mv "${temporary}" "${destination}"
}

publish_immutable_file() {
    local source="$1"
    local destination="$2"
    local temporary="${destination}.tmp.$$"
    [[ -f "${source}" && ! -L "${source}" ]] || return 1
    if [[ -e "${destination}" || -L "${destination}" ]]; then
        [[ -f "${destination}" && ! -L "${destination}" ]] || return 1
        cmp -s "${source}" "${destination}"
        return
    fi
    cp -- "${source}" "${temporary}" || return
    mv -- "${temporary}" "${destination}" || return
    rememr1_sync_file "${destination}"
}

write_atomic "${REMEMR1_STAGE_RESULT_FILE}" "${RUN_DIR}"
write_atomic "${RUN_DIR}/.running" "$$"
cat > "${RUN_DIR}/run.meta.tmp.$$" <<EOF
schema_version=1
stage=${STAGE}
git_commit=${REMEMR1_EXPECTED_COMMIT}
experiment_profile_id=${REMEMR1_EXPERIMENT_PROFILE}
phase=${REMEMR1_PHASE:-}
offload_profile=${REMEMR1_OFFLOAD_PROFILE:-}
budget_projection_sha256=${REMEMR1_BUDGET_PROJECTION_SHA256:-}
r1_approval_marker_sha256=${REMEMR1_R1_APPROVAL_MARKER_SHA256:-}
scope_generation=${REMEMR1_SCOPE_GENERATION:-base}
pipeline_dir=${REMEMR1_PIPELINE_DIR}
started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
mv "${RUN_DIR}/run.meta.tmp.$$" "${RUN_DIR}/run.meta"

finish_stage() {
    local rc="$1"
    local terminal_marker
    trap - EXIT INT TERM
    if [[ "${rc}" -eq 0 && "${STAGE_FINISHED}" != "yes" ]]; then
        rc=70
    fi
    if [[ "${rc}" -eq 43 && "${REMEMR1_PHASE:-}" != gpu-capacity ]]; then
        rc=70
    fi
    if ! write_atomic "${RUN_DIR}/finished-at" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"; then
        echo "stage terminal timestamp could not be persisted" >&2
        exit 74
    fi
    if [[ "${rc}" -eq 0 ]]; then
        terminal_marker="${RUN_DIR}/.success"
        if ! write_atomic "${terminal_marker}" "0"; then
            echo "stage success marker could not be persisted" >&2
            exit 74
        fi
    elif [[ "${rc}" -eq 42 ]]; then
        terminal_marker="${RUN_DIR}/.scientific-stop"
        if ! write_atomic "${terminal_marker}" "42" ||
           ! write_atomic "${RUN_DIR}/retryable" "false"; then
            echo "stage scientific-stop marker could not be persisted" >&2
            exit 74
        fi
    elif [[ "${rc}" -eq 43 ]]; then
        terminal_marker="${RUN_DIR}/.capacity-stop"
        if ! write_atomic "${terminal_marker}" "43" ||
           ! write_atomic "${RUN_DIR}/retryable" "false"; then
            echo "stage capacity-stop marker could not be persisted" >&2
            exit 74
        fi
    else
        terminal_marker="${RUN_DIR}/.failed"
        if ! write_atomic "${terminal_marker}" "${rc}"; then
            echo "stage failure marker could not be persisted" >&2
            exit 74
        fi
    fi

    if ! rememr1_sync_file "${terminal_marker}"; then
        if [[ "${rc}" -eq 0 || "${rc}" -eq 42 || "${rc}" -eq 43 ]]; then
            rc=74
            rm -f "${RUN_DIR}/.success" "${RUN_DIR}/.scientific-stop" \
                "${RUN_DIR}/.capacity-stop" \
                "${RUN_DIR}/retryable" || true
            terminal_marker="${RUN_DIR}/.failed"
            write_atomic "${terminal_marker}" "${rc}" && \
                rememr1_sync_file "${terminal_marker}" || true
        fi
        exit "${rc}"
    fi
    if ! rm -f "${RUN_DIR}/.running"; then
        if [[ "${rc}" -eq 0 || "${rc}" -eq 42 || "${rc}" -eq 43 ]]; then
            rc=74
            rm -f "${RUN_DIR}/.success" "${RUN_DIR}/.scientific-stop" \
                "${RUN_DIR}/.capacity-stop" \
                "${RUN_DIR}/retryable" || true
            terminal_marker="${RUN_DIR}/.failed"
            write_atomic "${terminal_marker}" "${rc}" && \
                rememr1_sync_file "${terminal_marker}" || true
        fi
        exit "${rc}"
    fi
    if ! rememr1_sync_file "${terminal_marker}"; then
        if [[ "${rc}" -eq 0 || "${rc}" -eq 42 || "${rc}" -eq 43 ]]; then
            rc=74
            rm -f "${RUN_DIR}/.success" "${RUN_DIR}/.scientific-stop" \
                "${RUN_DIR}/.capacity-stop" \
                "${RUN_DIR}/retryable" || true
            terminal_marker="${RUN_DIR}/.failed"
            write_atomic "${terminal_marker}" "${rc}" && \
                rememr1_sync_file "${terminal_marker}" || true
        fi
        exit "${rc}"
    fi
    exit "${rc}"
}
trap 'finish_stage $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_logged() {
    local label="$1"
    local duration="$2"
    shift 2
    local had_errexit=no
    [[ "$-" == *e* ]] && had_errexit=yes
    set +e
    {
        printf '[%s] start %s (timeout=%s)\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${label}" "${duration}"
        printf '[command]'
        printf ' %q' "$@"
        printf '\n'
    } | tee -a "${LOG_FILE}"
    local prefix_status=("${PIPESTATUS[@]}")
    if [[ "${prefix_status[0]}" -ne 0 || "${prefix_status[1]}" -ne 0 ]]; then
        [[ "${had_errexit}" == "yes" ]] && set -e
        return 1
    fi

    timeout --verbose --signal=TERM --kill-after=5m "${duration}" "$@" \
        2>&1 | tee -a "${LOG_FILE}"
    local pipeline_status=("${PIPESTATUS[@]}")
    local command_rc="${pipeline_status[0]}"
    local tee_rc="${pipeline_status[1]}"
    [[ "${had_errexit}" == "yes" ]] && set -e
    if [[ "${command_rc}" -ne 0 ]]; then
        return "${command_rc}"
    fi
    if [[ "${tee_rc}" -ne 0 ]]; then
        return "${tee_rc}"
    fi
    printf '[%s] complete %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${label}" | tee -a "${LOG_FILE}"
}

PYTHON="${REMEMR1_ENV_PREFIX}/bin/python"
RUNTIME_ASSET_MANIFEST="${REMEMR1_PIPELINE_DIR}/runtime-assets.json"
ASSET_REPORT="${REMEMR1_PIPELINE_DIR}/asset-report.json"
DATA_ROOT="${REMEMR1_PERSIST_ROOT}/data"
HANDOFF="${REMEMR1_PIPELINE_DIR}/cpu-handoff.json"
RESOLVED_CONFIG_ROOT="${REMEMR1_PIPELINE_DIR}/resolved-configs"
GPU_EVIDENCE_ROOT="${REMEMR1_PIPELINE_DIR}/gpu-evidence"
GPU_BUILD_LOG="${REMEMR1_PERSIST_ROOT}/evidence/gpu-environment/kernel-build.log"
GPU_FREEZE="${REMEMR1_PERSIST_ROOT}/evidence/pip-freeze.txt"
GPU_BUILD_INFO="${REMEMR1_PIPELINE_DIR}/build-info.json"

require_verified_handoff_context() {
    [[ "${REMEMR1_VERIFIED_HANDOFF:-}" == "${HANDOFF}" ]] || {
        echo "GPU stage lacks the pipeline's verified handoff context" >&2
        return 1
    }
    local observed
    observed="$(sha256sum "${HANDOFF}" | awk '{print $1}')"
    [[ "${observed}" == "${REMEMR1_VERIFIED_HANDOFF_FILE_SHA256:-}" ]] || {
        echo "CPU handoff file changed after pipeline verification" >&2
        return 1
    }
}

stop_ray() {
    run_logged ray-stop 2m \
        "${PYTHON}" -m ray.scripts.scripts stop --force
}

completed_stage_run() {
    local stage="$1"
    local origin_phase origin_profile origin_dir record
    origin_profile="${REMEMR1_OFFLOAD_PROFILE:-none}"
    case "${stage}" in
        cpu-*) origin_phase=cpu; origin_profile="" ;;
        g0|g1-step1|g1-resume2|g1-artifacts)
            origin_phase=gpu-gates
            origin_profile=r0
            ;;
        g2a|g2b-step1|g2b-resume5|g2-length-stress|g2-artifacts|capacity-seal|capacity-stop-seal)
            origin_phase=gpu-capacity
            ;;
        b-pilot|c-pilot|pilot-gate|b20|c20|b40|c40|bc40-artifacts|eval40|package40)
            origin_phase=gpu-bc40
            ;;
        b60|c60|b80|c80|bc80-artifacts|eval80|package80)
            origin_phase=gpu-bc80
            ;;
        export-results) origin_phase=gpu-export ;;
        *) echo "unknown predecessor stage: ${stage}" >&2; return 1 ;;
    esac
    if [[ "${origin_phase}" == "${REMEMR1_PHASE}" ]]; then
        origin_dir="${REMEMR1_STAGE_RECORD_DIR}"
    elif [[ "${origin_phase}" == gpu-gates ]]; then
        origin_dir="${PIPELINE_REAL}/stages/gpu-gates/r0/base"
    elif [[ "${origin_phase}" == gpu-bc40 ]]; then
        [[ -f "${PIPELINE_REAL}/.bc40-stage-record-dir" && \
           ! -L "${PIPELINE_REAL}/.bc40-stage-record-dir" ]] || return 1
        origin_dir="$(<"${PIPELINE_REAL}/.bc40-stage-record-dir")"
    elif [[ "${origin_phase}" == gpu-bc80 ]]; then
        [[ -f "${PIPELINE_REAL}/.bc80-stage-record-dir" && \
           ! -L "${PIPELINE_REAL}/.bc80-stage-record-dir" ]] || return 1
        origin_dir="$(<"${PIPELINE_REAL}/.bc80-stage-record-dir")"
    else
        origin_dir="${PIPELINE_REAL}/stages/${origin_phase}/${origin_profile}/base"
    fi
    [[ "${origin_dir}" == "${PIPELINE_REAL}/stages/"* && \
       -d "${origin_dir}" && ! -L "${origin_dir}" ]] || return 1
    record="${origin_dir}/${stage}.run"
    [[ -f "${record}" && ! -L "${record}" && -s "${record}" ]] || {
        echo "completed stage record is missing: ${stage}" >&2
        return 1
    }
    local run
    run="$(<"${record}")"
    [[ -d "${run}" && ! -L "${run}" ]] || return 1
    run="$(rememr1_realpath_existing "${run}")" || return 1
    rememr1_path_is_within "${run}" "${RUN_ROOT}" || return 1
    [[ -f "${run}/run.meta" && ! -L "${run}/run.meta" ]] || return 1
    grep -Fqx "stage=${stage}" "${run}/run.meta" || return 1
    grep -Fqx "git_commit=${REMEMR1_EXPECTED_COMMIT}" "${run}/run.meta" || return 1
    grep -Fqx "experiment_profile_id=${REMEMR1_EXPERIMENT_PROFILE}" \
        "${run}/run.meta" || return 1
    grep -Fqx "phase=${origin_phase}" "${run}/run.meta" || return 1
    grep -Fqx "offload_profile=${origin_profile}" "${run}/run.meta" || return 1
    grep -Fqx "scope_generation=$(basename -- "${origin_dir}")" \
        "${run}/run.meta" || return 1
    grep -Fqx "pipeline_dir=${PIPELINE_REAL}" "${run}/run.meta" || return 1
    [[ -f "${run}/.success" && "$(<"${run}/.success")" == 0 && \
       ! -e "${run}/.running" && ! -e "${run}/.failed" && \
       ! -e "${run}/.scientific-stop" && \
       ! -e "${run}/.capacity-stop" ]] || return 1
    printf '%s\n' "${run}"
}

run_bound_training() {
    local config_id="$1"
    local duration="$2"
    local predecessor_stage="${3:-}"
    local predecessor_config_id="${4:-}"
    local predecessor_step="${5:-}"
    local step_zero_stage="${6:-}"
    local index="${REMEMR1_CONFIG_ROOT}/index.json"
    local bind_args=(bind --index "${index}" --config-id "${config_id}" \
        --attempt-dir "${RUN_DIR}")
    local predecessor_run checkpoint evidence step_zero_run step_zero
    if [[ -n "${predecessor_stage}" ]]; then
        predecessor_run="$(completed_stage_run "${predecessor_stage}")" || return
        checkpoint="${predecessor_run}/checkpoints/global_step_${predecessor_step}"
        evidence="${checkpoint}/reproduction_extra_state.json"
        [[ -d "${checkpoint}" && -f "${evidence}" && ! -L "${evidence}" ]] || {
            echo "resume predecessor evidence is incomplete: ${predecessor_stage}" >&2
            return 1
        }
        bind_args+=(--resume-config-id "${predecessor_config_id}" \
            --resume-checkpoint-dir "${checkpoint}" \
            --resume-evidence "${evidence}")
    fi
    if [[ -n "${step_zero_stage}" ]]; then
        step_zero_run="$(completed_stage_run "${step_zero_stage}")" || return
        step_zero="${step_zero_run}/evidence/step_zero_fingerprint.json"
        [[ -f "${step_zero}" && ! -L "${step_zero}" ]] || {
            echo "step-zero reference is incomplete: ${step_zero_stage}" >&2
            return 1
        }
        bind_args+=(--step-zero-reference "${step_zero}")
    fi
    run_logged "bind-${config_id}" 10m \
        "${PYTHON}" scripts/cloud/run_resolved_training.py "${bind_args[@]}"
    run_logged "verify-binding-${config_id}" 10m \
        "${PYTHON}" scripts/cloud/run_resolved_training.py verify \
        --attempt-dir "${RUN_DIR}"

    local training_rc cleanup_rc telemetry_rc expected_step="" length_stress=no
    local expected_train_file="" expected_validation_file=""
    local expected_train_manifest="" expected_validation_manifest=""
    local expected_base_model="" expected_revision=""
    local -a telemetry_args artifact_args
    set +e
    run_logged "train-${config_id}" "${duration}" \
        "${PYTHON}" scripts/cloud/run_resolved_training.py run \
        --attempt-dir "${RUN_DIR}"
    training_rc="$?"
    stop_ray
    cleanup_rc="$?"
    set -e
    if [[ "${cleanup_rc}" -ne 0 ]]; then
        echo "Ray cleanup failed after ${config_id} (exit ${cleanup_rc})" >&2
        [[ "${training_rc}" -eq 0 ]] && return "${cleanup_rc}"
    fi
    if [[ "${training_rc}" -eq 0 ]]; then
        case "${config_id}" in
            g0_qwen35_08b) expected_step=20 ;;
            g1_qwen35_2b_step1) expected_step=1 ;;
            g1_qwen35_2b_resume2) expected_step=2 ;;
            g2a_qwen35_2b_5090_*|g2b_qwen35_2b_5090_step1_*) expected_step=1 ;;
            g2b_qwen35_2b_5090_resume5_*) expected_step=5 ;;
            g2_length_stress_qwen35_2b_5090_*)
                expected_step=1
                length_stress=yes
                ;;
            b_pilot_qwen35_2b_5090_*|c_pilot_qwen35_2b_5090_*) expected_step=3 ;;
            [bc]20_qwen35_2b_5090_*) expected_step=20 ;;
            [bc]40_qwen35_2b_5090_*) expected_step=40 ;;
            [bc]60_qwen35_2b_5090_*) expected_step=60 ;;
            [bc]80_qwen35_2b_5090_*) expected_step=80 ;;
            *) echo "strict telemetry contract lacks ${config_id}" >&2; return 2 ;;
        esac
        telemetry_args=(verify-success \
            --attempt-dir "${RUN_DIR}" \
            --expected-config-id "${config_id}" \
            --expected-config-sha256 \
                "$(sha256sum "${REMEMR1_CONFIG_ROOT}/${config_id}.yaml" | awk '{print $1}')" \
            --expected-offload-profile "${REMEMR1_OFFLOAD_PROFILE}" \
            --expected-final-step "${expected_step}")
        [[ "${length_stress}" != yes ]] || telemetry_args+=(--length-stress)
        set +e
        run_logged "verify-telemetry-${config_id}" 10m \
            "${PYTHON}" scripts/cloud/training_telemetry.py \
            "${telemetry_args[@]}"
        telemetry_rc="$?"
        set -e
        [[ "${telemetry_rc}" -eq 0 ]] || return "${telemetry_rc}"
    fi
    if [[ "${training_rc}" -eq 0 && "${length_stress}" != yes ]]; then
        case "${config_id}" in
            g0_qwen35_08b)
                expected_step=20
                expected_train_file="${REMEMR1_G0_TRAIN_PATH}/train.parquet"
                expected_validation_file="${REMEMR1_G0_VALIDATION_PATH}/train.parquet"
                expected_train_manifest="${REMEMR1_G0_TRAIN_SHA256}"
                expected_validation_manifest="${REMEMR1_G0_VALIDATION_SHA256}"
                expected_base_model="Qwen/Qwen3.5-0.8B"
                expected_revision="2fc06364715b967f1860aea9cf38778875588b17"
                ;;
            g1_qwen35_2b_step1)
                expected_step=1
                expected_train_file="${REMEMR1_G1_TRAIN_PATH}/train.parquet"
                expected_validation_file="${REMEMR1_G1_VALIDATION_PATH}/train.parquet"
                expected_train_manifest="${REMEMR1_G1_TRAIN_SHA256}"
                expected_validation_manifest="${REMEMR1_G1_VALIDATION_SHA256}"
                expected_base_model="Qwen/Qwen3.5-2B"
                expected_revision="15852e8c16360a2fea060d615a32b45270f8a8fc"
                ;;
            g1_qwen35_2b_resume2)
                expected_step=2
                expected_train_file="${REMEMR1_G1_TRAIN_PATH}/train.parquet"
                expected_validation_file="${REMEMR1_G1_VALIDATION_PATH}/train.parquet"
                expected_train_manifest="${REMEMR1_G1_TRAIN_SHA256}"
                expected_validation_manifest="${REMEMR1_G1_VALIDATION_SHA256}"
                expected_base_model="Qwen/Qwen3.5-2B"
                expected_revision="15852e8c16360a2fea060d615a32b45270f8a8fc"
                ;;
            *_qwen35_2b_5090_*)
                [[ -n "${expected_step}" ]] || {
                    echo "artifact contract lacks final step for ${config_id}" >&2
                    return 2
                }
                expected_train_file="${REMEMR1_FORMAL_TRAIN_PATH}/train.parquet"
                expected_validation_file="${REMEMR1_FORMAL_VALIDATION_PATH}/train.parquet"
                expected_train_manifest="${REMEMR1_FORMAL_TRAIN_SHA256}"
                expected_validation_manifest="${REMEMR1_FORMAL_VALIDATION_SHA256}"
                expected_base_model="Qwen/Qwen3.5-2B"
                expected_revision="15852e8c16360a2fea060d615a32b45270f8a8fc"
                ;;
            *)
                echo "artifact contract lacks ${config_id}" >&2
                return 2
                ;;
        esac
        artifact_args=(
            --checkpoint-dir "${RUN_DIR}/checkpoints/global_step_${expected_step}"
            --adapter-dir "${RUN_DIR}/artifacts/adapter/global_step_${expected_step}/adapter"
            --expected-step "${expected_step}"
            --expected-train-file "${expected_train_file}"
            --expected-validation-file "${expected_validation_file}"
            --expected-resolved-config "${REMEMR1_CONFIG_ROOT}/${config_id}.yaml"
            --expected-train-manifest "${expected_train_manifest}"
            --expected-validation-manifest "${expected_validation_manifest}"
            --expected-base-model "${expected_base_model}"
            --expected-revision "${expected_revision}"
        )
        if [[ -n "${predecessor_stage}" ]]; then
            artifact_args+=(--expected-resume-from "${checkpoint}")
        fi
        run_logged "verify-artifacts-${config_id}" 2h \
            "${PYTHON}" scripts/cloud/verify_training_artifacts.py \
            "${artifact_args[@]}"
    fi
    return "${training_rc}"
}

prepare_capacity_approval_args() {
    CAPACITY_APPROVAL_ARGS=()
    [[ "${REMEMR1_OFFLOAD_PROFILE}" == r1 ]] || return 0
    local capacity_dir
    capacity_dir="$(dirname -- "${REMEMR1_CAPACITY_PROFILE_PATH}")"
    local approval_record="${capacity_dir}/r1-approval.path"
    local consumption_record="${capacity_dir}/r1-approval-consumption.path"
    local terminal_record="${capacity_dir}/r0-terminal.sha256"
    local budget_record="${capacity_dir}/budget-projection.sha256"
    [[ -f "${approval_record}" && ! -L "${approval_record}" && \
       -f "${consumption_record}" && ! -L "${consumption_record}" && \
       -f "${terminal_record}" && ! -L "${terminal_record}" && \
       -f "${budget_record}" && ! -L "${budget_record}" ]] || {
        echo "R1 capacity approval binding is incomplete" >&2
        return 1
    }
    local approval consumption terminal_sha budget_sha
    approval="$(<"${approval_record}")"
    consumption="$(<"${consumption_record}")"
    terminal_sha="$(<"${terminal_record}")"
    budget_sha="$(<"${budget_record}")"
    [[ "${approval}" == /* && -f "${approval}" && ! -L "${approval}" && \
       "${consumption}" == /* && -f "${consumption}" && \
       ! -L "${consumption}" ]] || return 1
    approval="$(rememr1_realpath_existing "${approval}")" || return 1
    consumption="$(rememr1_realpath_existing "${consumption}")" || return 1
    rememr1_path_is_within "${approval}" "${PERSIST_REAL}" || return 1
    rememr1_path_is_within "${consumption}" "${PERSIST_REAL}" || return 1
    [[ "${terminal_sha}" =~ ^[0-9a-f]{64}$ && \
       "${budget_sha}" =~ ^[0-9a-f]{64}$ ]] || return 1
    CAPACITY_APPROVAL_ARGS=(
        --r1-approval-marker "${approval}"
        --r1-approval-consumption "${consumption}"
        --r0-capacity-evidence \
            "${REMEMR1_PIPELINE_DIR}/capacity/r0/capacity-evidence.json"
        --r0-terminal-sha256 "${terminal_sha}"
        --budget-projection-sha256 "${budget_sha}"
    )
}

bind_eval_artifacts() {
    local level="$1"
    local b_stage="$2"
    local c_stage="$3"
    local b_run c_run probe_dir b_probe c_probe
    b_run="$(completed_stage_run "${b_stage}")" || return
    c_run="$(completed_stage_run "${c_stage}")" || return
    probe_dir="${RUN_DIR}/artifacts/resume-probes"
    mkdir -- "${probe_dir}"
    b_probe="${probe_dir}/${b_stage}.json"
    c_probe="${probe_dir}/${c_stage}.json"
    run_logged "resume-probe-${b_stage}" 30m \
        "${PYTHON}" scripts/cloud/resume_endpoint_probe.py probe \
        --attempt-dir "${b_run}" \
        --output "${b_probe}"
    run_logged "resume-probe-${c_stage}" 30m \
        "${PYTHON}" scripts/cloud/resume_endpoint_probe.py probe \
        --attempt-dir "${c_run}" \
        --output "${c_probe}"
    run_logged "verify-resume-probe-${b_stage}" 10m \
        "${PYTHON}" scripts/cloud/resume_endpoint_probe.py verify \
        --evidence "${b_probe}"
    run_logged "verify-resume-probe-${c_stage}" 10m \
        "${PYTHON}" scripts/cloud/resume_endpoint_probe.py verify \
        --evidence "${c_probe}"
    prepare_capacity_approval_args || return
    run_logged "bind-eval-${level}" 30m \
        "${PYTHON}" scripts/cloud/eval_matrix.py bind \
        --index "${REMEMR1_CONFIG_ROOT}/index.json" \
        --eval-config-id "eval${level}_qwen35_2b_5090" \
        --capacity-profile "${REMEMR1_CAPACITY_PROFILE_PATH}" \
        --b-attempt "${b_run}" \
        --c-attempt "${c_run}" \
        --b-resume-probe "${b_probe}" \
        --c-resume-probe "${c_probe}" \
        --output "${RUN_DIR}/artifacts/eval-binding.json" \
        "${CAPACITY_APPROVAL_ARGS[@]}"
}

case "${STAGE}" in
    cpu-preflight)
        run_logged checkout 5m bash scripts/cloud/setup_git.sh
        if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q .; then
            if [[ "${REMEMR1_ALLOW_GPU_CPU_PHASE:-no}" != "yes" ]]; then
                echo "CPU preparation refuses an instance with an exposed GPU" >&2
                exit 1
            fi
        fi
        for command_name in curl findmnt flock git python3 sha256sum setsid timeout; do
            command -v "${command_name}" >/dev/null 2>&1 || {
                echo "required command is missing: ${command_name}" >&2
                exit 1
            }
        done
        free_bytes="$(df --output=avail -B1 "${REMEMR1_PERSIST_ROOT}" | awk 'NR == 2 {print $1}')"
        min_free_gib="${REMEMR1_MIN_FREE_GIB:-200}"
        [[ "${min_free_gib}" =~ ^[0-9]+$ ]] || {
            echo "REMEMR1_MIN_FREE_GIB must be an integer" >&2
            exit 2
        }
        min_free_bytes="$((min_free_gib * 1024 * 1024 * 1024))"
        [[ "${free_bytes}" =~ ^[0-9]+$ && "${free_bytes}" -ge "${min_free_bytes}" ]] || {
            echo "CPU preparation requires at least ${min_free_gib} GiB free on the persistent volume" >&2
            exit 1
        }
        min_cpu_cores="${REMEMR1_MIN_CPU_CORES:-16}"
        [[ "${min_cpu_cores}" =~ ^[0-9]+$ && "${min_cpu_cores}" -gt 0 ]] || {
            echo "REMEMR1_MIN_CPU_CORES must be a positive integer" >&2
            exit 2
        }
        min_ram_gib="${REMEMR1_MIN_RAM_GIB:-48}"
        [[ "${min_ram_gib}" =~ ^[0-9]+$ && "${min_ram_gib}" -gt 0 ]] || {
            echo "REMEMR1_MIN_RAM_GIB must be a positive integer" >&2
            exit 2
        }
        run_logged host-resource-preflight 5m \
            python3 scripts/cloud/host_resource_probe.py \
            --min-cpu-cores "${min_cpu_cores}" \
            --min-ram-gib "${min_ram_gib}"
        df -h "${REMEMR1_PERSIST_ROOT}" | tee -a "${LOG_FILE}"
        ;;
    cpu-environment)
        run_logged environment-install 4h bash scripts/cloud/install_env.sh
        ;;
    cpu-kernel-sources)
        run_logged kernel-source-prefetch 2h bash scripts/cloud/prepare_kernel_sources.sh
        ;;
    cpu-assets)
        run_logged asset-metadata-resolution 30m \
            "${PYTHON}" scripts/cloud/cloud_state.py resolve-assets \
            --manifest environment/reproduction-assets.json \
            --output "${RUNTIME_ASSET_MANIFEST}"
        run_logged asset-prefetch 18h \
            "${PYTHON}" scripts/reproduction/prefetch_assets.py \
            --manifest "${RUNTIME_ASSET_MANIFEST}" \
            --cache-dir "${HUGGINGFACE_HUB_CACHE}" \
            --download --retries 4 --timeout 300 \
            --report "${ASSET_REPORT}"
        ;;
    cpu-data)
        run_logged data-bundles 12h \
            "${PYTHON}" scripts/cloud/cloud_state.py build-data \
            --manifest "${RUNTIME_ASSET_MANIFEST}" \
            --cache-dir "${HUGGINGFACE_HUB_CACHE}" \
            --data-root "${DATA_ROOT}" \
            --summary "${REMEMR1_PIPELINE_DIR}/data-summary.json"
        ;;
    cpu-tests)
        run_logged reproduction-and-cloud-tests 2h \
            "${PYTHON}" -m pytest -q tests/reproduction tests/cloud
        run_logged compileall 30m \
            "${PYTHON}" -m compileall -q scripts taskutils recurrent verl
        ;;
    cpu-configs)
        run_logged resolve-configs 30m \
            "${PYTHON}" scripts/cloud/resolve_configs.py \
            --data-root "${DATA_ROOT}" \
            --output "${RESOLVED_CONFIG_ROOT}"
        ;;
    cpu-handoff)
        run_logged publish-handoff 4h \
            "${PYTHON}" scripts/cloud/cloud_state.py publish-handoff \
            --output "${HANDOFF}" \
            --commit "${REMEMR1_EXPECTED_COMMIT}" \
            --asset-manifest "${RUNTIME_ASSET_MANIFEST}" \
            --asset-report "${ASSET_REPORT}" \
            --pip-freeze "${REMEMR1_PERSIST_ROOT}/evidence/pip-freeze.cpu.txt" \
            --data-root "${DATA_ROOT}" \
            --kernel-source-root "${REMEMR1_PERSIST_ROOT}/sources" \
            --config-root "${RESOLVED_CONFIG_ROOT}" \
            --persist-root "${REMEMR1_PERSIST_ROOT}"
        ;;
    gpu-preflight)
        export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        export WANDB_MODE=disabled
        require_verified_handoff_context
        run_logged ray-clean-start 2m \
            "${PYTHON}" -m ray.scripts.scripts stop --force
        probe_profile="${REMEMR1_OFFLOAD_PROFILE:-r0}"
        case "${probe_profile}" in
            r0) probe_profile=R0 ;;
            r1) probe_profile=R1 ;;
            *) echo "GPU preflight received an invalid offload profile" >&2; exit 2 ;;
        esac
        if [[ -f "${GPU_BUILD_INFO}" ]] && \
           run_logged environment-revalidation 20m \
               "${PYTHON}" scripts/reproduction/verify_environment.py \
               --build-info "${GPU_BUILD_INFO}" && \
           run_logged gpu-evidence-revalidation 30m \
               "${PYTHON}" scripts/cloud/gpu_probe.py \
               --verify-existing \
               --evidence-root "${GPU_EVIDENCE_ROOT}" \
               --build-log "${GPU_BUILD_LOG}" \
               --pip-freeze "${GPU_FREEZE}" \
               --build-info "${GPU_BUILD_INFO}" \
               --profile "${probe_profile}" \
               --optimizer-steps 20; then
            echo "Existing sm_120 build evidence is valid on this GPU host" | tee -a "${LOG_FILE}"
        else
            run_logged kernel-build 4h bash scripts/cloud/install_gpu_kernels.sh
            run_logged kernel-probes 2h \
                "${PYTHON}" scripts/cloud/gpu_probe.py \
                --evidence-root "${GPU_EVIDENCE_ROOT}" \
                --build-log "${GPU_BUILD_LOG}" \
                --pip-freeze "${GPU_FREEZE}" \
                --build-info "${GPU_BUILD_INFO}" \
                --profile "${probe_profile}" \
                --optimizer-steps 20
            run_logged environment-gate 20m \
                "${PYTHON}" scripts/reproduction/verify_environment.py \
                --build-info "${GPU_BUILD_INFO}"
            run_logged gpu-evidence-gate 30m \
                "${PYTHON}" scripts/cloud/gpu_probe.py \
                --verify-existing \
                --evidence-root "${GPU_EVIDENCE_ROOT}" \
                --build-log "${GPU_BUILD_LOG}" \
                --pip-freeze "${GPU_FREEZE}" \
                --build-info "${GPU_BUILD_INFO}" \
                --profile "${probe_profile}" \
                --optimizer-steps 20
        fi
        ;;
    g0)
        export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
        require_verified_handoff_context
        run_bound_training g0_qwen35_08b 6h
        ;;
    g1-step1)
        export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
        require_verified_handoff_context
        run_bound_training g1_qwen35_2b_step1 6h
        ;;
    g1-resume2)
        export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
        require_verified_handoff_context
        run_bound_training g1_qwen35_2b_resume2 6h \
            g1-step1 g1_qwen35_2b_step1 1
        ;;
    g1-artifacts)
        export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
        require_verified_handoff_context
        predecessor="$(completed_stage_run g1-resume2)"
        adapter="${predecessor}/artifacts/adapter/global_step_2/adapter"
        merged="${RUN_DIR}/artifacts/merged-global_step_2"
        run_logged adapter-verify 30m \
            "${PYTHON}" -m scripts.reproduction.export_adapter verify \
            --adapter-dir "${adapter}"
        run_logged adapter-hf-recurrent-eval 3h \
            "${PYTHON}" scripts/cloud/g1_eval.py \
            --bundle-dir "${REMEMR1_G1_EVAL_PATH}" \
            --expected-manifest-sha256 "${REMEMR1_G1_EVAL_SHA256}" \
            --adapter-dir "${adapter}" \
            --output-dir "${RUN_DIR}/artifacts/hf-recurrent-eval"
        if [[ -d "${merged}" ]]; then
            run_logged merged-artifact-revalidation 30m \
                "${PYTHON}" -c \
                'from verl.utils.checkpoint.reproduction import validate_adapter_export, validate_merged_model_artifact; import sys; adapter = validate_adapter_export(sys.argv[1]); merged = validate_merged_model_artifact(sys.argv[2]); valid = merged["global_step"] == 2 and merged["source_adapter_metadata_sha256"] == adapter.sha256; sys.exit(f"merged artifact identity mismatch: {merged}") if not valid else None' \
                "${adapter}" "${merged}"
        else
            run_logged adapter-merge-reload 2h \
                "${PYTHON}" -m scripts.reproduction.export_adapter merge \
                --adapter-dir "${adapter}" \
                --merged-dir "${merged}" \
                --device cuda --max-new-tokens 4
        fi
        ;;
    g2a)
        export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
        require_verified_handoff_context
        run_bound_training \
            "g2a_qwen35_2b_5090_${REMEMR1_OFFLOAD_PROFILE}" 12h
        ;;
    g2b-step1)
        export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
        require_verified_handoff_context
        run_bound_training \
            "g2b_qwen35_2b_5090_step1_${REMEMR1_OFFLOAD_PROFILE}" 12h
        ;;
    g2b-resume5)
        export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
        require_verified_handoff_context
        run_bound_training \
            "g2b_qwen35_2b_5090_resume5_${REMEMR1_OFFLOAD_PROFILE}" 36h \
            g2b-step1 \
            "g2b_qwen35_2b_5090_step1_${REMEMR1_OFFLOAD_PROFILE}" 1
        ;;
    g2-length-stress)
        export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
        require_verified_handoff_context
        run_bound_training \
            "g2_length_stress_qwen35_2b_5090_${REMEMR1_OFFLOAD_PROFILE}" 24h
        ;;
    g2-artifacts)
        export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
        require_verified_handoff_context
        g2_resume_run="$(completed_stage_run g2b-resume5)"
        capacity_dir="${REMEMR1_CAPACITY_OUTPUT_DIR}"
        aggregate_dir="${RUN_DIR}/artifacts/capacity-inputs"
        mkdir -p "${capacity_dir}"
        run_logged g2-artifact-evidence 2h \
            "${PYTHON}" scripts/cloud/capacity_aggregate.py artifacts \
            --checkpoint-dir "${g2_resume_run}/checkpoints/global_step_5" \
            --adapter-dir "${g2_resume_run}/artifacts/adapter/global_step_5/adapter" \
            --output "${RUN_DIR}/artifacts/g2-artifacts.json"
        run_logged g2-capacity-aggregate 2h \
            "${PYTHON}" scripts/cloud/capacity_aggregate.py aggregate \
            --handoff "${HANDOFF}" \
            --index "${REMEMR1_CONFIG_ROOT}/index.json" \
            --gpu-evidence "${GPU_EVIDENCE_ROOT}/causal-conv1d/bf16-forward.json" \
            --profile "${REMEMR1_OFFLOAD_PROFILE}" \
            --g2a-pointer "${REMEMR1_STAGE_RECORD_DIR}/g2a.run" \
            --g2b-step1-pointer "${REMEMR1_STAGE_RECORD_DIR}/g2b-step1.run" \
            --g2b-resume5-pointer "${REMEMR1_STAGE_RECORD_DIR}/g2b-resume5.run" \
            --length-stress-pointer \
                "${REMEMR1_STAGE_RECORD_DIR}/g2-length-stress.run" \
            --artifact-evidence "${RUN_DIR}/artifacts/g2-artifacts.json" \
            --output-dir "${aggregate_dir}"
        for name in identity selected-configs attempt-metadata telemetry; do
            publish_immutable_file "${aggregate_dir}/${name}.json" \
                "${capacity_dir}/${name}.json"
        done
        ;;
    capacity-stop-seal)
        export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
        require_verified_handoff_context
        case "${REMEMR1_CAPACITY_STOPPED_STAGE:-}" in
            g2a|g2b-step1|g2b-resume5|g2-length-stress) ;;
            *) echo "capacity-stop finalizer lacks a trusted stopped stage" >&2; exit 1 ;;
        esac
        [[ -n "${REMEMR1_CAPACITY_STOPPED_POINTER:-}" ]] || exit 1
        capacity_dir="${REMEMR1_CAPACITY_OUTPUT_DIR}"
        aggregate_dir="${RUN_DIR}/artifacts/capacity-inputs"
        mkdir -p "${capacity_dir}"
        stop_args=(capacity-stop \
            --handoff "${HANDOFF}" \
            --index "${REMEMR1_CONFIG_ROOT}/index.json" \
            --gpu-evidence \
                "${GPU_EVIDENCE_ROOT}/causal-conv1d/bf16-forward.json" \
            --profile "${REMEMR1_OFFLOAD_PROFILE}" \
            --stopped-stage "${REMEMR1_CAPACITY_STOPPED_STAGE}" \
            --stopped-pointer "${REMEMR1_CAPACITY_STOPPED_POINTER}" \
            --output-dir "${aggregate_dir}")
        case "${REMEMR1_CAPACITY_STOPPED_STAGE}" in
            g2b-step1)
                stop_args+=(--g2a-pointer "${REMEMR1_STAGE_RECORD_DIR}/g2a.run")
                ;;
            g2b-resume5)
                stop_args+=(--g2a-pointer "${REMEMR1_STAGE_RECORD_DIR}/g2a.run" \
                    --g2b-step1-pointer \
                        "${REMEMR1_STAGE_RECORD_DIR}/g2b-step1.run")
                ;;
            g2-length-stress)
                stop_args+=(--g2a-pointer "${REMEMR1_STAGE_RECORD_DIR}/g2a.run" \
                    --g2b-step1-pointer \
                        "${REMEMR1_STAGE_RECORD_DIR}/g2b-step1.run" \
                    --g2b-resume5-pointer \
                        "${REMEMR1_STAGE_RECORD_DIR}/g2b-resume5.run")
                ;;
        esac
        run_logged capacity-stop-aggregate 2h \
            "${PYTHON}" scripts/cloud/capacity_aggregate.py "${stop_args[@]}"
        for name in identity selected-configs attempt-metadata telemetry; do
            publish_immutable_file "${aggregate_dir}/${name}.json" \
                "${capacity_dir}/${name}.json"
        done
        capacity_evidence="${RUN_DIR}/artifacts/capacity-evidence.json"
        run_logged capacity-stop-evidence 30m \
            "${PYTHON}" scripts/cloud/capacity_evidence.py evidence \
            --identity "${aggregate_dir}/identity.json" \
            --telemetry "${aggregate_dir}/telemetry.json" \
            --attempt-metadata "${aggregate_dir}/attempt-metadata.json" \
            --selected-configs "${aggregate_dir}/selected-configs.json" \
            --output "${capacity_evidence}"
        publish_immutable_file "${capacity_evidence}" \
            "${capacity_dir}/capacity-evidence.json"
        if [[ "${REMEMR1_OFFLOAD_PROFILE}" == r0 ]]; then
            r1_target="${RUN_DIR}/artifacts/r1-target-identity.json"
            run_logged r1-target-identity 30m \
                "${PYTHON}" scripts/cloud/capacity_aggregate.py target-identity \
                --handoff "${HANDOFF}" \
                --index "${REMEMR1_CONFIG_ROOT}/index.json" \
                --gpu-evidence \
                    "${GPU_EVIDENCE_ROOT}/causal-conv1d/bf16-forward.json" \
                --profile r1 \
                --output "${r1_target}"
            publish_immutable_file "${r1_target}" \
                "${capacity_dir}/r1-target-identity.json"
        fi
        ;;
    capacity-seal)
        export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
        require_verified_handoff_context
        capacity_dir="${REMEMR1_CAPACITY_OUTPUT_DIR}"
        capacity_evidence="${RUN_DIR}/artifacts/capacity-evidence.json"
        capacity_profile="${RUN_DIR}/artifacts/capacity-profile.json"
        run_logged capacity-evidence 30m \
            "${PYTHON}" scripts/cloud/capacity_evidence.py evidence \
            --identity "${capacity_dir}/identity.json" \
            --telemetry "${capacity_dir}/telemetry.json" \
            --attempt-metadata "${capacity_dir}/attempt-metadata.json" \
            --selected-configs "${capacity_dir}/selected-configs.json" \
            --output "${capacity_evidence}"
        publish_immutable_file "${capacity_evidence}" \
            "${capacity_dir}/capacity-evidence.json"
        if [[ "${REMEMR1_OFFLOAD_PROFILE}" == r0 ]]; then
            r1_target="${RUN_DIR}/artifacts/r1-target-identity.json"
            run_logged r1-target-identity 30m \
                "${PYTHON}" scripts/cloud/capacity_aggregate.py target-identity \
                --handoff "${HANDOFF}" \
                --index "${REMEMR1_CONFIG_ROOT}/index.json" \
                --gpu-evidence \
                    "${GPU_EVIDENCE_ROOT}/causal-conv1d/bf16-forward.json" \
                --profile r1 \
                --output "${r1_target}"
            publish_immutable_file "${r1_target}" \
                "${capacity_dir}/r1-target-identity.json"
        fi
        mapfile -t capacity_decision < <("${PYTHON}" -c \
            'import json,sys; c=json.load(open(sys.argv[1], encoding="utf-8"))["classification"]; t=json.load(open(sys.argv[2], encoding="utf-8")); non_green=c["non_green_metrics"]; resource=(bool(non_green) and all(c["metrics"][name]["r1_trigger"] for name in non_green)) or t.get("terminal_reason") in {"oom","allocator_pressure","gpu_memory_headroom"}; print(c["overall"]); print(str(c["r1_eligible"]).lower()); print(c.get("eligibility_reason") or ""); print(str(resource).lower())' \
            "${capacity_evidence}" "${capacity_dir}/telemetry.json")
        [[ "${#capacity_decision[@]}" -eq 4 ]] || exit 1
        if [[ "${capacity_decision[0]}" != green ]]; then
            if [[ "${REMEMR1_OFFLOAD_PROFILE}" == r0 && \
                  "${capacity_decision[1]}" == true ]]; then
                write_atomic "${capacity_dir}/r1-eligibility-reason" \
                    "${capacity_decision[2]}"
            fi
            if [[ "${capacity_decision[3]}" == true ]]; then
                exit 43
            fi
            exit 42
        fi
        seal_args=(generate \
            --identity "${capacity_dir}/identity.json" \
            --telemetry "${capacity_dir}/telemetry.json" \
            --attempt-metadata "${capacity_dir}/attempt-metadata.json" \
            --selected-configs "${capacity_dir}/selected-configs.json" \
            --output "${capacity_profile}")
        if [[ "${REMEMR1_OFFLOAD_PROFILE}" == r1 ]]; then
            run_logged r1-budget-projection-gate 5m \
                "${PYTHON}" scripts/cloud/cost_gate.py verify \
                --projection "${REMEMR1_BUDGET_PROJECTION}"
            mapfile -t approval_fields < <("${PYTHON}" -c \
                'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")); print(value["r0_terminal_sha256"]); print(value["budget_projection_sha256"])' \
                "${REMEMR1_R1_APPROVAL}")
            [[ "${#approval_fields[@]}" -eq 2 ]] || exit 1
            [[ "${approval_fields[1]}" == \
               "${REMEMR1_BUDGET_PROJECTION_SHA256}" ]] || {
                echo "R1 approval budget hash differs from the verified projection" >&2
                exit 1
            }
            seal_args+=(--approval-marker "${REMEMR1_R1_APPROVAL}" \
                --approval-consumption \
                    "${REMEMR1_R1_APPROVAL_CONSUMPTION}" \
                --r0-capacity-evidence \
                    "${REMEMR1_PIPELINE_DIR}/capacity/r0/capacity-evidence.json" \
                --r0-terminal-sha256 "${approval_fields[0]}" \
                --budget-projection-sha256 \
                    "${REMEMR1_BUDGET_PROJECTION_SHA256}")
        fi
        run_logged capacity-profile-seal 30m \
            "${PYTHON}" scripts/cloud/capacity_evidence.py "${seal_args[@]}"
        verify_args=(verify --capacity-profile "${capacity_profile}")
        if [[ "${REMEMR1_OFFLOAD_PROFILE}" == r1 ]]; then
            verify_args+=(--approval-marker "${REMEMR1_R1_APPROVAL}" \
                --approval-consumption \
                    "${REMEMR1_R1_APPROVAL_CONSUMPTION}" \
                --r0-capacity-evidence \
                    "${REMEMR1_PIPELINE_DIR}/capacity/r0/capacity-evidence.json" \
                --r0-terminal-sha256 "${approval_fields[0]}" \
                --budget-projection-sha256 \
                    "${REMEMR1_BUDGET_PROJECTION_SHA256}")
        fi
        run_logged capacity-profile-verify 30m \
            "${PYTHON}" scripts/cloud/capacity_evidence.py "${verify_args[@]}"
        publish_immutable_file "${capacity_evidence}" \
            "${capacity_dir}/capacity-evidence.json"
        publish_immutable_file "${capacity_profile}" \
            "${capacity_dir}/capacity-profile.json"
        if [[ "${REMEMR1_OFFLOAD_PROFILE}" == r1 ]]; then
            write_atomic "${capacity_dir}/r1-approval.path" \
                "${REMEMR1_R1_APPROVAL}"
            write_atomic "${capacity_dir}/r1-approval-consumption.path" \
                "${REMEMR1_R1_APPROVAL_CONSUMPTION}"
            write_atomic "${capacity_dir}/r0-terminal.sha256" \
                "${approval_fields[0]}"
            write_atomic "${capacity_dir}/budget-projection.sha256" \
                "${REMEMR1_BUDGET_PROJECTION_SHA256}"
        fi
        ;;
    b-pilot)
        export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
        require_verified_handoff_context
        run_bound_training \
            "b_pilot_qwen35_2b_5090_${REMEMR1_OFFLOAD_PROFILE}" 24h
        ;;
    c-pilot)
        export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
        require_verified_handoff_context
        run_bound_training \
            "c_pilot_qwen35_2b_5090_${REMEMR1_OFFLOAD_PROFILE}" 24h \
            "" "" "" b-pilot
        ;;
    pilot-gate)
        export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
        require_verified_handoff_context
        b_pilot_run="$(completed_stage_run b-pilot)"
        c_pilot_run="$(completed_stage_run c-pilot)"
        run_logged paired-pilot-gate 30m \
            "${PYTHON}" scripts/cloud/pilot_gate.py \
            --b-evidence "${b_pilot_run}/evidence/pilot.jsonl" \
            --c-evidence "${c_pilot_run}/evidence/pilot.jsonl" \
            --b-step-zero "${b_pilot_run}/evidence/step_zero_fingerprint.json" \
            --c-step-zero "${c_pilot_run}/evidence/step_zero_fingerprint.json" \
            --output "${RUN_DIR}/artifacts/pilot-gate.json"
        ;;
    b20)
        export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
        require_verified_handoff_context
        run_bound_training "b20_qwen35_2b_5090_${REMEMR1_OFFLOAD_PROFILE}" 7d
        ;;
    c20)
        export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
        require_verified_handoff_context
        run_bound_training "c20_qwen35_2b_5090_${REMEMR1_OFFLOAD_PROFILE}" 7d \
            "" "" "" b20
        ;;
    b40)
        export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
        require_verified_handoff_context
        run_bound_training "b40_qwen35_2b_5090_${REMEMR1_OFFLOAD_PROFILE}" 7d \
            b20 "b20_qwen35_2b_5090_${REMEMR1_OFFLOAD_PROFILE}" 20
        ;;
    c40)
        export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
        require_verified_handoff_context
        run_bound_training "c40_qwen35_2b_5090_${REMEMR1_OFFLOAD_PROFILE}" 7d \
            c20 "c20_qwen35_2b_5090_${REMEMR1_OFFLOAD_PROFILE}" 20 b20
        ;;
    b60)
        export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
        require_verified_handoff_context
        run_bound_training "b60_qwen35_2b_5090_${REMEMR1_OFFLOAD_PROFILE}" 7d \
            b40 "b40_qwen35_2b_5090_${REMEMR1_OFFLOAD_PROFILE}" 40
        ;;
    c60)
        export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
        require_verified_handoff_context
        run_bound_training "c60_qwen35_2b_5090_${REMEMR1_OFFLOAD_PROFILE}" 7d \
            c40 "c40_qwen35_2b_5090_${REMEMR1_OFFLOAD_PROFILE}" 40 b20
        ;;
    b80)
        export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
        require_verified_handoff_context
        run_bound_training "b80_qwen35_2b_5090_${REMEMR1_OFFLOAD_PROFILE}" 7d \
            b60 "b60_qwen35_2b_5090_${REMEMR1_OFFLOAD_PROFILE}" 60
        ;;
    c80)
        export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
        require_verified_handoff_context
        run_bound_training "c80_qwen35_2b_5090_${REMEMR1_OFFLOAD_PROFILE}" 7d \
            c60 "c60_qwen35_2b_5090_${REMEMR1_OFFLOAD_PROFILE}" 60 b20
        ;;
    bc40-artifacts)
        require_verified_handoff_context
        bind_eval_artifacts 40 b40 c40
        ;;
    eval40)
        require_verified_handoff_context
        binding_run="$(completed_stage_run bc40-artifacts)"
        run_logged eval40-plan 2h \
            "${PYTHON}" scripts/cloud/eval_matrix.py plan \
            --config "${REMEMR1_CONFIG_ROOT}/eval40_qwen35_2b_5090.yaml" \
            --bindings "${binding_run}/artifacts/eval-binding.json" \
            --output-root "${RUN_DIR}/artifacts/results" \
            --output "${RUN_DIR}/artifacts/plan.json"
        run_logged eval40-matrix 7d \
            "${PYTHON}" scripts/cloud/eval_matrix.py run \
            --plan "${RUN_DIR}/artifacts/plan.json"
        ;;
    package40)
        require_verified_handoff_context
        eval_run="$(completed_stage_run eval40)"
        run_logged package40 2h \
            "${PYTHON}" scripts/cloud/eval_matrix.py package \
            --plan "${eval_run}/artifacts/plan.json" \
            --output "${RUN_DIR}/artifacts/verified-package"
        run_logged verify-package40 30m \
            "${PYTHON}" scripts/cloud/eval_matrix.py verify-package \
            --plan "${eval_run}/artifacts/plan.json" \
            --package-dir "${RUN_DIR}/artifacts/verified-package"
        ;;
    bc80-artifacts)
        require_verified_handoff_context
        bind_eval_artifacts 80 b80 c80
        ;;
    eval80)
        require_verified_handoff_context
        binding_run="$(completed_stage_run bc80-artifacts)"
        run_logged eval80-plan 2h \
            "${PYTHON}" scripts/cloud/eval_matrix.py plan \
            --config "${REMEMR1_CONFIG_ROOT}/eval80_qwen35_2b_5090.yaml" \
            --bindings "${binding_run}/artifacts/eval-binding.json" \
            --output-root "${RUN_DIR}/artifacts/results" \
            --output "${RUN_DIR}/artifacts/plan.json"
        run_logged eval80-matrix 7d \
            "${PYTHON}" scripts/cloud/eval_matrix.py run \
            --plan "${RUN_DIR}/artifacts/plan.json"
        ;;
    package80)
        require_verified_handoff_context
        eval_run="$(completed_stage_run eval80)"
        run_logged package80 2h \
            "${PYTHON}" scripts/cloud/eval_matrix.py package \
            --plan "${eval_run}/artifacts/plan.json" \
            --output "${RUN_DIR}/artifacts/verified-package"
        run_logged verify-package80 30m \
            "${PYTHON}" scripts/cloud/eval_matrix.py verify-package \
            --plan "${eval_run}/artifacts/plan.json" \
            --package-dir "${RUN_DIR}/artifacts/verified-package"
        ;;
    export-results)
        require_verified_handoff_context
        [[ "${REMEMR1_EXPORT_SOURCE}" == /* && \
           -d "${REMEMR1_EXPORT_SOURCE}" && \
           ! -L "${REMEMR1_EXPORT_SOURCE}" ]] || {
            echo "verified export source is missing or unsafe" >&2
            exit 1
        }
        rememr1_path_is_within "${REMEMR1_EXPORT_SOURCE}" "${PERSIST_REAL}" || exit 1
        if [[ -f "${REMEMR1_PIPELINE_DIR}/.bc80-ready" && \
              "$(<"${REMEMR1_PIPELINE_DIR}/.bc80-ready")" == \
              "${REMEMR1_EXPORT_SOURCE}" ]]; then
            eval_run="$(completed_stage_run eval80)"
        else
            eval_run="$(completed_stage_run eval40)"
        fi
        run_logged verify-export-source 30m \
            "${PYTHON}" scripts/cloud/eval_matrix.py verify-package \
            --plan "${eval_run}/artifacts/plan.json" \
            --package-dir "${REMEMR1_EXPORT_SOURCE}"
        mkdir -- "${RUN_DIR}/artifacts/exported-package"
        run_logged export-package-copy 30m \
            cp -a -- "${REMEMR1_EXPORT_SOURCE}/." \
            "${RUN_DIR}/artifacts/exported-package/"
        run_logged verify-export-copy 30m \
            "${PYTHON}" scripts/cloud/eval_matrix.py verify-package \
            --plan "${eval_run}/artifacts/plan.json" \
            --package-dir "${RUN_DIR}/artifacts/exported-package"
        ;;
esac

STAGE_FINISHED="yes"
finish_stage 0
