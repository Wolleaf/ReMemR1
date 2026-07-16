#!/usr/bin/env bash
# Run one resumable CPU or bounded GPU gate DAG under the inherited cloud lock.
set -euo pipefail

PHASE=""
RETRY_FAILED_STAGE="no"
DRY_RUN="no"

usage() {
    echo "Usage: $0 --phase cpu|gpu-gates [--retry-failed-stage] [--dry-run]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase) PHASE="$2"; shift 2 ;;
        --retry-failed-stage) RETRY_FAILED_STAGE="yes"; shift ;;
        --dry-run) DRY_RUN="yes"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done
case "${PHASE}" in
    cpu|gpu-gates) ;;
    *) usage; exit 2 ;;
esac

CLOUD_ENV="${REMEMR1_CLOUD_ENV:-/root/autodl-tmp/rememr1-cloud.env}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/runtime.sh"
rememr1_load_cloud_env "${CLOUD_ENV}"
rememr1_require_cloud_env
cd "${REMEMR1_PROJECT_DIR}"
source "${SCRIPT_DIR}/lib/lock.sh"
require_cloud_lock

PERSIST_REAL="$(rememr1_realpath_existing "${PERSIST_ROOT}")"
CLOUD_ROOT_RAW="${PERSIST_ROOT}/cloud"
[[ -d "${CLOUD_ROOT_RAW}" && ! -L "${CLOUD_ROOT_RAW}" ]] || {
    echo "cloud state root must be a regular directory" >&2
    exit 1
}
CLOUD_ROOT_REAL="$(rememr1_realpath_existing "${CLOUD_ROOT_RAW}")"
rememr1_path_is_within "${CLOUD_ROOT_REAL}" "${PERSIST_REAL}" || {
    echo "cloud state root escaped persistent storage" >&2
    exit 1
}
[[ -n "${REMEMR1_PIPELINE_ROOT:-}" && "${REMEMR1_PIPELINE_ROOT}" == /* && \
   -d "${REMEMR1_PIPELINE_ROOT}" && ! -L "${REMEMR1_PIPELINE_ROOT}" ]] || {
    echo "pipeline root must be an existing non-symlink directory" >&2
    exit 1
}
PIPELINE_ROOT_REAL="$(rememr1_realpath_existing "${REMEMR1_PIPELINE_ROOT}")"
[[ "${PIPELINE_ROOT_REAL}" == "${CLOUD_ROOT_REAL}/pipelines" ]] || {
    echo "pipeline root differs from the initialized cloud state" >&2
    exit 1
}
STAGE_RUN_ROOT_RAW="${CLOUD_ROOT_REAL}/runs"
if [[ -e "${STAGE_RUN_ROOT_RAW}" || -L "${STAGE_RUN_ROOT_RAW}" ]]; then
    [[ -d "${STAGE_RUN_ROOT_RAW}" && ! -L "${STAGE_RUN_ROOT_RAW}" ]] || {
        echo "stage run root must be a regular directory" >&2
        exit 1
    }
else
    mkdir -- "${STAGE_RUN_ROOT_RAW}"
fi
STAGE_RUN_ROOT="$(rememr1_realpath_existing "${STAGE_RUN_ROOT_RAW}")"
[[ "${STAGE_RUN_ROOT}" == "${CLOUD_ROOT_REAL}/runs" ]] || {
    echo "stage run root escaped the initialized cloud state" >&2
    exit 1
}
timeout --verbose --signal=TERM --kill-after=30s 5m \
    bash scripts/cloud/setup_git.sh

CONFIG_TREE_SHA="$(
    find verl/trainer/config/reproduction scripts/cloud -type f \
        \( -name '*.yaml' -o -name '*.py' -o -name '*.sh' \) -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 sha256sum \
        | sha256sum \
        | awk '{print $1}'
)"
LOCK_FILE_SHA="$(sha256sum environment/reproduction-cu130.lock.json | awk '{print $1}')"
ASSET_FILE_SHA="$(sha256sum environment/reproduction-assets.json | awk '{print $1}')"
IDENTITY_TEXT="git_commit=${REMEMR1_EXPECTED_COMMIT}
config_tree_sha256=${CONFIG_TREE_SHA}
environment_lock_file_sha256=${LOCK_FILE_SHA}
asset_manifest_file_sha256=${ASSET_FILE_SHA}"
IDENTITY_SHA="$(printf '%s\n' "${IDENTITY_TEXT}" | sha256sum | awk '{print $1}')"
PIPELINE_DIR="${PIPELINE_ROOT_REAL}/${REMEMR1_EXPECTED_COMMIT}-${IDENTITY_SHA}"
if [[ -e "${PIPELINE_DIR}" || -L "${PIPELINE_DIR}" ]]; then
    [[ -d "${PIPELINE_DIR}" && ! -L "${PIPELINE_DIR}" ]] || {
        echo "pipeline state path is not a regular directory" >&2
        exit 1
    }
else
    mkdir -- "${PIPELINE_DIR}"
fi
PIPELINE_DIR="$(rememr1_realpath_existing "${PIPELINE_DIR}")"
[[ "$(dirname -- "${PIPELINE_DIR}")" == "${PIPELINE_ROOT_REAL}" ]] || {
    echo "pipeline state escaped the pipeline root" >&2
    exit 1
}
export REMEMR1_PIPELINE_DIR="${PIPELINE_DIR}"
for state_directory in attempts stages; do
    candidate="${PIPELINE_DIR}/${state_directory}"
    if [[ -e "${candidate}" || -L "${candidate}" ]]; then
        [[ -d "${candidate}" && ! -L "${candidate}" ]] || {
            echo "pipeline ${state_directory} path is not a regular directory" >&2
            exit 1
        }
    else
        mkdir -- "${candidate}"
    fi
    [[ "$(rememr1_realpath_existing "${candidate}")" == "${candidate}" ]] || {
        echo "pipeline ${state_directory} path escaped its state directory" >&2
        exit 1
    }
done

write_value() {
    atomic_write_text "$1" "$2"
}

identity_tmp="${PIPELINE_DIR}/identity.tmp.$$"
printf '%s\n' "${IDENTITY_TEXT}" > "${identity_tmp}"
[[ ! -L "${PIPELINE_DIR}/identity" ]] || {
    rm -f "${identity_tmp}"
    echo "pipeline identity must not be a symlink" >&2
    exit 1
}
if [[ -f "${PIPELINE_DIR}/identity" ]]; then
    cmp -s "${identity_tmp}" "${PIPELINE_DIR}/identity" || {
        rm -f "${identity_tmp}"
        echo "pipeline identity differs from persisted state" >&2
        exit 1
    }
    rm -f "${identity_tmp}"
else
    mv "${identity_tmp}" "${PIPELINE_DIR}/identity"
fi
if [[ -n "${REMEMR1_RESULT_FILE:-}" ]]; then
    write_value "${REMEMR1_RESULT_FILE}" "${PIPELINE_DIR}"
fi

if [[ "${DRY_RUN}" == "yes" ]]; then
    if [[ "${PHASE}" == "cpu" ]]; then
        printf '%s\n' cpu-preflight cpu-environment cpu-kernel-sources cpu-assets cpu-data cpu-tests cpu-configs cpu-handoff
    else
        printf '%s\n' gpu-preflight g0 g1-step1 g1-resume2 g1-artifacts
    fi
    write_value "${PIPELINE_DIR}/last-dry-run" "${PHASE} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    exit 0
fi

CURRENT_STAGE="initialization"
PIPELINE_FINISHED="no"
STAGE_RUNNER="${REMEMR1_STAGE_RUNNER:-${REMEMR1_PROJECT_DIR}/scripts/cloud/run_stage.sh}"

pipeline_exit() {
    local rc="$?"
    trap - EXIT INT TERM
    rm -f "${PIPELINE_DIR}/.running"
    if [[ "${PIPELINE_FINISHED}" != "yes" ]]; then
        [[ "${rc}" -ne 0 ]] || rc=70
        rm -f "${PIPELINE_DIR}/.success"
        write_value "${PIPELINE_DIR}/.failed" "${rc}"
        write_value "${PIPELINE_DIR}/failed-stage" "${CURRENT_STAGE}"
        write_value "${PIPELINE_DIR}/failed-phase" "${PHASE}"
        rememr1_sync_file "${PIPELINE_DIR}/.failed" || true
    fi
    exit "${rc}"
}

if [[ -f "${PIPELINE_DIR}/.running" ]]; then
    old_pid="$(<"${PIPELINE_DIR}/.running")"
    [[ "${RETRY_FAILED_STAGE}" == "yes" ]] || {
        echo "interrupted pipeline marker (old pid ${old_pid}) requires --retry-failed-stage after log review" >&2
        exit 1
    }
fi
if [[ -f "${PIPELINE_DIR}/.failed" && "${RETRY_FAILED_STAGE}" != "yes" ]]; then
    if [[ -s "${PIPELINE_DIR}/failed-stage" ]]; then
        failed_stage="$(<"${PIPELINE_DIR}/failed-stage")"
    else
        failed_stage="unknown"
    fi
    echo "pipeline previously failed at ${failed_stage}; use --retry-failed-stage after inspection" >&2
    exit 1
fi
if [[ "${PHASE}" == "cpu" && -f "${PIPELINE_DIR}/.gpu-started" ]]; then
    echo "GPU gates have already started; refusing to rerun or downgrade CPU state" >&2
    exit 1
fi

trap pipeline_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
rm -f "${PIPELINE_DIR}/.success" "${PIPELINE_DIR}/.failed" \
    "${PIPELINE_DIR}/failed-stage" "${PIPELINE_DIR}/failed-phase"
write_value "${PIPELINE_DIR}/.running" "$$"
write_value "${PIPELINE_DIR}/current-phase" "${PHASE}"

resolve_verified_stage_run() {
    local run="$1"
    local expected_stage="$2"
    local resolved run_root
    [[ -d "${run}" && ! -L "${run}" ]] || {
        echo "stage run is missing or is a symlink: ${run}" >&2
        return 1
    }
    resolved="$(realpath -e "${run}" 2>/dev/null || true)"
    run_root="${STAGE_RUN_ROOT}"
    [[ -n "${resolved}" && -n "${run_root}" ]] || return 1
    case "${resolved}" in
        "${run_root}/"*) ;;
        *) echo "stage run escaped the persistent run root: ${run}" >&2; return 1 ;;
    esac
    [[ -f "${resolved}/run.meta" && ! -L "${resolved}/run.meta" ]] || return 1
    grep -Fqx "schema_version=1" "${resolved}/run.meta" || return 1
    grep -Fqx "stage=${expected_stage}" "${resolved}/run.meta" || return 1
    grep -Fqx "git_commit=${REMEMR1_EXPECTED_COMMIT}" "${resolved}/run.meta" || return 1
    grep -Fqx "pipeline_dir=${PIPELINE_DIR}" "${resolved}/run.meta" || return 1
    printf '%s\n' "${resolved}"
}

verify_stage_run() {
    local run="$1"
    local expected_stage="$2"
    local resolved
    resolved="$(resolve_verified_stage_run "${run}" "${expected_stage}")" || return
    [[ -f "${resolved}/.success" && ! -L "${resolved}/.success" && \
       "$(<"${resolved}/.success")" == 0 ]] || return 1
    [[ ! -e "${resolved}/.failed" && ! -e "${resolved}/.running" ]] || return 1
}

verify_stage_artifacts() {
    local key="$1"
    local python="${REMEMR1_ENV_PREFIX}/bin/python"
    case "${key}" in
        cpu-handoff)
            timeout --verbose --signal=TERM --kill-after=5m 4h \
                "${python}" scripts/cloud/cloud_state.py verify-handoff \
                --handoff "${PIPELINE_DIR}/cpu-handoff.json" \
                --expected-commit "${REMEMR1_EXPECTED_COMMIT}" || return
            ;;
        gpu-preflight)
            timeout --verbose --signal=TERM --kill-after=1m 20m \
                "${python}" scripts/reproduction/verify_environment.py \
                --build-info "${PIPELINE_DIR}/build-info.json" || return
            timeout --verbose --signal=TERM --kill-after=1m 30m \
                "${python}" scripts/cloud/gpu_probe.py \
                --verify-existing \
                --evidence-root "${PIPELINE_DIR}/gpu-evidence" \
                --build-log "${REMEMR1_PERSIST_ROOT}/evidence/gpu-environment/kernel-build.log" \
                --pip-freeze "${REMEMR1_PERSIST_ROOT}/evidence/pip-freeze.txt" \
                --build-info "${PIPELINE_DIR}/build-info.json" \
                --optimizer-steps 20 || return
            ;;
        g0)
            timeout --verbose --signal=TERM --kill-after=1m 2h \
                "${python}" scripts/cloud/verify_training_artifacts.py \
                --checkpoint-dir "${PIPELINE_DIR}/checkpoints/g0/global_step_20" \
                --adapter-dir "${PIPELINE_DIR}/artifacts/g0/adapter/global_step_20/adapter" \
                --expected-step 20 \
                --expected-train-file "${REMEMR1_G0_TRAIN_PATH}/train.parquet" \
                --expected-validation-file "${REMEMR1_G0_VALIDATION_PATH}/train.parquet" \
                --expected-resolved-config "${REMEMR1_CONFIG_ROOT}/g0_qwen35_08b.yaml" \
                --expected-train-manifest "${REMEMR1_G0_TRAIN_SHA256}" \
                --expected-validation-manifest "${REMEMR1_G0_VALIDATION_SHA256}" \
                --expected-base-model Qwen/Qwen3.5-0.8B \
                --expected-revision 2fc06364715b967f1860aea9cf38778875588b17 || return
            ;;
        g1-step1)
            timeout --verbose --signal=TERM --kill-after=1m 2h \
                "${python}" scripts/cloud/verify_training_artifacts.py \
                --checkpoint-dir "${PIPELINE_DIR}/checkpoints/g1-step1/global_step_1" \
                --adapter-dir "${PIPELINE_DIR}/artifacts/g1-step1/adapter/global_step_1/adapter" \
                --expected-step 1 \
                --expected-train-file "${REMEMR1_G1_TRAIN_PATH}/train.parquet" \
                --expected-validation-file "${REMEMR1_G1_VALIDATION_PATH}/train.parquet" \
                --expected-resolved-config "${REMEMR1_CONFIG_ROOT}/g1_qwen35_2b_step1.yaml" \
                --expected-train-manifest "${REMEMR1_G1_TRAIN_SHA256}" \
                --expected-validation-manifest "${REMEMR1_G1_VALIDATION_SHA256}" \
                --expected-base-model Qwen/Qwen3.5-2B \
                --expected-revision 15852e8c16360a2fea060d615a32b45270f8a8fc || return
            ;;
        g1-resume2)
            timeout --verbose --signal=TERM --kill-after=1m 2h \
                "${python}" scripts/cloud/verify_training_artifacts.py \
                --checkpoint-dir "${PIPELINE_DIR}/checkpoints/g1-resume2/global_step_2" \
                --adapter-dir "${PIPELINE_DIR}/artifacts/g1-resume2/adapter/global_step_2/adapter" \
                --expected-step 2 \
                --expected-train-file "${REMEMR1_G1_TRAIN_PATH}/train.parquet" \
                --expected-validation-file "${REMEMR1_G1_VALIDATION_PATH}/train.parquet" \
                --expected-resolved-config "${REMEMR1_CONFIG_ROOT}/g1_qwen35_2b_resume2.yaml" \
                --expected-train-manifest "${REMEMR1_G1_TRAIN_SHA256}" \
                --expected-validation-manifest "${REMEMR1_G1_VALIDATION_SHA256}" \
                --expected-base-model Qwen/Qwen3.5-2B \
                --expected-revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
                --expected-resume-from "${PIPELINE_DIR}/checkpoints/g1-step1/global_step_1" || return
            ;;
        g1-artifacts)
            timeout --verbose --signal=TERM --kill-after=1m 2h \
                "${python}" scripts/cloud/verify_training_artifacts.py \
                --checkpoint-dir "${PIPELINE_DIR}/checkpoints/g1-resume2/global_step_2" \
                --adapter-dir "${PIPELINE_DIR}/artifacts/g1-resume2/adapter/global_step_2/adapter" \
                --expected-step 2 \
                --expected-train-file "${REMEMR1_G1_TRAIN_PATH}/train.parquet" \
                --expected-validation-file "${REMEMR1_G1_VALIDATION_PATH}/train.parquet" \
                --expected-resolved-config "${REMEMR1_CONFIG_ROOT}/g1_qwen35_2b_resume2.yaml" \
                --expected-train-manifest "${REMEMR1_G1_TRAIN_SHA256}" \
                --expected-validation-manifest "${REMEMR1_G1_VALIDATION_SHA256}" \
                --expected-base-model Qwen/Qwen3.5-2B \
                --expected-revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
                --expected-resume-from "${PIPELINE_DIR}/checkpoints/g1-step1/global_step_1" || return
            timeout --verbose --signal=TERM --kill-after=1m 30m \
                "${python}" scripts/cloud/g1_eval.py \
                --verify-existing \
                --bundle-dir "${REMEMR1_G1_EVAL_PATH}" \
                --expected-manifest-sha256 "${REMEMR1_G1_EVAL_SHA256}" \
                --adapter-dir "${PIPELINE_DIR}/artifacts/g1-resume2/adapter/global_step_2/adapter" \
                --output-dir "${PIPELINE_DIR}/artifacts/g1-resume2/hf-recurrent-eval" || return
            timeout --verbose --signal=TERM --kill-after=1m 2h \
                "${python}" -c \
                'from verl.utils.checkpoint.reproduction import validate_adapter_export, validate_merged_model_artifact; import sys; adapter = validate_adapter_export(sys.argv[1]); merged = validate_merged_model_artifact(sys.argv[2]); valid = merged["global_step"] == 2 and merged["source_adapter_metadata_sha256"] == adapter.sha256; sys.exit(f"merged artifact identity mismatch: {merged}") if not valid else None' \
                "${PIPELINE_DIR}/artifacts/g1-resume2/adapter/global_step_2/adapter" \
                "${PIPELINE_DIR}/artifacts/g1-resume2/merged-global_step_2" || return
            ;;
    esac
}

recover_successful_attempt() {
    local key="$1"
    local candidate run
    local candidates=()
    shopt -s nullglob
    candidates=("${PIPELINE_DIR}/attempts/${key}-"*.run)
    shopt -u nullglob
    local index
    for ((index=${#candidates[@]} - 1; index >= 0; index--)); do
        candidate="${candidates[${index}]}"
        [[ -s "${candidate}" ]] || continue
        run="$(<"${candidate}")"
        if verify_stage_run "${run}" "${key}" 2>/dev/null && \
           verify_stage_artifacts "${key}" >/dev/null 2>&1; then
            write_value "${PIPELINE_DIR}/stages/${key}.run" "${run}" || return
            rememr1_sync_file "${PIPELINE_DIR}/stages/${key}.run" || return
            rm -f "${PIPELINE_DIR}/current-stage" "${PIPELINE_DIR}/current-attempt" || return
            echo "[pipeline] recovered ${key}: ${run}"
            return 0
        fi
    done
    return 1
}

ADOPTED_RUN=""

publish_adopted_stage_run() {
    local key="$1"
    local failed_run="$2"
    local original_rc="$3"
    local source_state="${4:-failed}"
    local run_root adopted staging now metadata
    run_root="$(realpath -e "${REMEMR1_PERSIST_ROOT}/cloud/runs")" || return
    adopted="${failed_run}.adopted"
    case "${adopted}" in
        "${run_root}/"*) ;;
        *) echo "adopted stage run escaped the persistent run root" >&2; return 1 ;;
    esac
    if [[ -e "${adopted}" || -L "${adopted}" ]]; then
        verify_stage_run "${adopted}" "${key}" || return
        [[ -f "${adopted}/original-failure" && \
           ! -L "${adopted}/original-failure" && \
           "$(<"${adopted}/original-failure")" == "${original_rc}" ]] || return 1
        [[ -f "${adopted}/adopted-from-failure" && \
           ! -L "${adopted}/adopted-from-failure" && \
           "$(<"${adopted}/adopted-from-failure")" == "${failed_run}" ]] || return 1
        [[ -f "${adopted}/adoption-source-state" && \
           ! -L "${adopted}/adoption-source-state" && \
           "$(<"${adopted}/adoption-source-state")" == "${source_state}" ]] || return 1
        rememr1_sync_file "${adopted}/.success" || return
        ADOPTED_RUN="${adopted}"
        return 0
    fi

    staging="${run_root}/.adoption-${key}-$$-${RANDOM:-0}"
    [[ ! -e "${staging}" && ! -L "${staging}" ]] || return 1
    mkdir -p "${staging}/logs" "${staging}/artifacts" || return
    now="$(date -u +%Y-%m-%dT%H:%M:%SZ)" || {
        rm -rf -- "${staging}"
        return 1
    }
    metadata="schema_version=1
stage=${key}
git_commit=${REMEMR1_EXPECTED_COMMIT}
pipeline_dir=${PIPELINE_DIR}
started_at=${now}
adopted_at=${now}"
    write_value "${staging}/run.meta" "${metadata}" || {
        rm -rf -- "${staging}"
        return 1
    }
    write_value "${staging}/original-failure" "${original_rc}" || {
        rm -rf -- "${staging}"
        return 1
    }
    write_value "${staging}/adopted-from-failure" "${failed_run}" || {
        rm -rf -- "${staging}"
        return 1
    }
    write_value "${staging}/adoption-source-state" "${source_state}" || {
        rm -rf -- "${staging}"
        return 1
    }
    write_value "${staging}/finished-at" "${now}" || {
        rm -rf -- "${staging}"
        return 1
    }
    write_value "${staging}/logs/stage.log" \
        "adopted verified artifacts from ${source_state} run ${failed_run} (${original_rc})" || {
        rm -rf -- "${staging}"
        return 1
    }
    write_value "${staging}/.success" "0" || {
        rm -rf -- "${staging}"
        return 1
    }
    mv "${staging}" "${adopted}" || {
        rm -rf -- "${staging}"
        return 1
    }
    rememr1_sync_file "${adopted}/.success" || return
    ADOPTED_RUN="${adopted}"
}

recover_failed_artifact_attempt() {
    local key="$1"
    case "${key}" in
        g0|g1-step1|g1-resume2|g1-artifacts) ;;
        *) return 1 ;;
    esac
    [[ "${RETRY_FAILED_STAGE}" == "yes" ]] || return 1

    local candidate run resolved original_rc source_state has_running adoption_attempt record
    local candidates=()
    shopt -s nullglob
    candidates=("${PIPELINE_DIR}/attempts/${key}-"*.run)
    shopt -u nullglob
    local index
    for ((index=${#candidates[@]} - 1; index >= 0; index--)); do
        candidate="${candidates[${index}]}"
        [[ -f "${candidate}" && ! -L "${candidate}" && -s "${candidate}" ]] || continue
        run="$(<"${candidate}")"
        resolved="$(resolve_verified_stage_run "${run}" "${key}" 2>/dev/null)" || continue
        has_running="no"
        if [[ -e "${resolved}/.running" || -L "${resolved}/.running" ]]; then
            [[ -f "${resolved}/.running" && ! -L "${resolved}/.running" ]] || continue
            has_running="yes"
        fi
        if [[ -e "${resolved}/.failed" || -L "${resolved}/.failed" ]]; then
            [[ -f "${resolved}/.failed" && ! -L "${resolved}/.failed" && \
               -f "${resolved}/finished-at" && ! -L "${resolved}/finished-at" && \
               ! -e "${resolved}/.success" && ! -L "${resolved}/.success" ]] || continue
            original_rc="$(<"${resolved}/.failed")"
            [[ "${original_rc}" =~ ^[1-9][0-9]{0,2}$ && \
               "${original_rc}" -ge 1 && "${original_rc}" -le 255 ]] || continue
            source_state="failed"
        elif [[ -e "${resolved}/.success" || -L "${resolved}/.success" ]]; then
            [[ "${has_running}" == "yes" && -f "${resolved}/.success" && \
               ! -L "${resolved}/.success" && "$(<"${resolved}/.success")" == 0 && \
               -f "${resolved}/finished-at" && ! -L "${resolved}/finished-at" ]] || continue
            original_rc="stale-running-after-success"
            source_state="interrupted"
        else
            [[ "${has_running}" == "yes" ]] || continue
            original_rc="interrupted-without-terminal-exit-code"
            source_state="interrupted"
        fi
        if ! verify_stage_artifacts "${key}" >/dev/null 2>&1; then
            continue
        fi

        ADOPTED_RUN=""
        publish_adopted_stage_run \
            "${key}" "${resolved}" "${original_rc}" "${source_state}" || return 2
        [[ -n "${ADOPTED_RUN}" ]] || return 2
        adoption_attempt="${candidate%.run}.adopted.run"
        if [[ -e "${adoption_attempt}" || -L "${adoption_attempt}" ]]; then
            [[ -f "${adoption_attempt}" && ! -L "${adoption_attempt}" && \
               "$(<"${adoption_attempt}")" == "${ADOPTED_RUN}" ]] || return 2
        else
            write_value "${adoption_attempt}" "${ADOPTED_RUN}" || return 2
        fi
        rememr1_sync_file "${adoption_attempt}" || return 2
        record="${PIPELINE_DIR}/stages/${key}.run"
        write_value "${record}" "${ADOPTED_RUN}" || return 2
        rememr1_sync_file "${record}" || return 2
        rm -f "${PIPELINE_DIR}/current-stage" "${PIPELINE_DIR}/current-attempt" || return 2
        echo "[pipeline] adopted ${key}: ${ADOPTED_RUN} (failed run preserved: ${resolved})"
        return 0
    done
    return 1
}

run_stage() {
    local key="$1"
    CURRENT_STAGE="${key}"
    local record="${PIPELINE_DIR}/stages/${key}.run"
    if [[ -s "${record}" && "${key}" != "gpu-preflight" ]]; then
        local recorded
        recorded="$(<"${record}")"
        if verify_stage_run "${recorded}" "${key}" && \
           verify_stage_artifacts "${key}"; then
            echo "[pipeline] reuse ${key}: ${recorded}"
            return 0
        fi
        if [[ "${RETRY_FAILED_STAGE}" != "yes" ]]; then
            echo "persisted stage record is invalid: ${key}" >&2
            return 1
        fi
        echo "[pipeline] retrying invalid persisted stage: ${key}" >&2
    fi
    if [[ "${key}" != "gpu-preflight" ]] && \
       recover_successful_attempt "${key}"; then
        return 0
    fi
    if [[ "${RETRY_FAILED_STAGE}" == "yes" && "${key}" != "gpu-preflight" ]]; then
        local adoption_rc=0
        recover_failed_artifact_attempt "${key}" || adoption_rc="$?"
        if [[ "${adoption_rc}" -eq 0 ]]; then
            return 0
        fi
        if [[ "${adoption_rc}" -ne 1 ]]; then
            echo "failed to publish adopted stage state: ${key}" >&2
            return "${adoption_rc}"
        fi
    fi

    local attempt="${PIPELINE_DIR}/attempts/${key}-$(date -u +%Y%m%dT%H%M%SZ)-$$.run"
    write_value "${PIPELINE_DIR}/current-stage" "${key}"
    write_value "${PIPELINE_DIR}/current-attempt" "${attempt}"
    echo "[pipeline] start ${key}"
    set +e
    REMEMR1_STAGE_RESULT_FILE="${attempt}" bash "${STAGE_RUNNER}" "${key}"
    local rc="$?"
    set -e
    if [[ "${rc}" -ne 0 ]]; then
        [[ -s "${attempt}" ]] && write_value "${PIPELINE_DIR}/last-failed-run" "$(<"${attempt}")"
        return "${rc}"
    fi
    [[ -s "${attempt}" ]] || {
        echo "stage did not publish its run directory: ${key}" >&2
        return 1
    }
    local run
    run="$(<"${attempt}")"
    verify_stage_run "${run}" "${key}"
    verify_stage_artifacts "${key}"
    write_value "${record}" "${run}"
    rememr1_sync_file "${record}"
    rm -f "${PIPELINE_DIR}/current-stage" "${PIPELINE_DIR}/current-attempt"
    echo "[pipeline] complete ${key}: ${run}"
}

finish_phase() {
    local marker="$1"
    local value="$2"
    trap '' INT TERM
    write_value "${marker}" "${value}"
    rm -f "${PIPELINE_DIR}/.running" "${PIPELINE_DIR}/.failed" \
        "${PIPELINE_DIR}/failed-stage" "${PIPELINE_DIR}/failed-phase" \
        "${PIPELINE_DIR}/current-stage" "${PIPELINE_DIR}/current-attempt"
    write_value "${PIPELINE_DIR}/last-successful-phase" "${PHASE}"
    rememr1_sync_file "${marker}" || return
}

if [[ "${PHASE}" == "cpu" ]]; then
    if [[ -f "${PIPELINE_DIR}/.cpu-ready" ]]; then
        CURRENT_STAGE="cpu-handoff-revalidation"
        handoff="$(<"${PIPELINE_DIR}/.cpu-ready")"
        set +e
        timeout --verbose --signal=TERM --kill-after=5m 4h \
            "${REMEMR1_ENV_PREFIX}/bin/python" scripts/cloud/cloud_state.py verify-handoff \
            --handoff "${handoff}" --expected-commit "${REMEMR1_EXPECTED_COMMIT}"
        ready_rc="$?"
        set -e
        if [[ "${ready_rc}" -eq 0 ]]; then
            finish_phase "${PIPELINE_DIR}/.cpu-ready" "${handoff}"
            PIPELINE_FINISHED="yes"
            echo "[pipeline] CPU preparation was already complete: ${handoff}" || true
            exit 0
        fi
        if [[ "${RETRY_FAILED_STAGE}" != "yes" ]]; then
            exit "${ready_rc}"
        fi
        echo "[pipeline] retrying after invalid CPU handoff" >&2
    fi
    for stage in \
        cpu-preflight cpu-environment cpu-kernel-sources cpu-assets \
        cpu-data cpu-tests cpu-configs cpu-handoff; do
        run_stage "${stage}"
    done
    CURRENT_STAGE="cpu-finalization"
    handoff="${PIPELINE_DIR}/cpu-handoff.json"
    finish_phase "${PIPELINE_DIR}/.cpu-ready" "${handoff}"
    PIPELINE_FINISHED="yes"
    echo "[pipeline] CPU handoff ready: ${handoff}" || true
    exit 0
fi

CURRENT_STAGE="cpu-handoff"
[[ -s "${PIPELINE_DIR}/.cpu-ready" ]] || {
    echo "GPU gates require a successful CPU handoff for this exact identity" >&2
    exit 1
}
handoff="$(<"${PIPELINE_DIR}/.cpu-ready")"
set +e
handoff_values="$(timeout --verbose --signal=TERM --kill-after=5m 4h \
    "${REMEMR1_ENV_PREFIX}/bin/python" \
    scripts/cloud/cloud_state.py verify-handoff \
    --handoff "${handoff}" --expected-commit "${REMEMR1_EXPECTED_COMMIT}" \
    --field bundles.gates.g0.train.path \
    --field bundles.gates.g0.train.manifest_sha256 \
    --field bundles.gates.g0.validation.path \
    --field bundles.gates.g0.validation.manifest_sha256 \
    --field bundles.gates.g1.train.path \
    --field bundles.gates.g1.train.manifest_sha256 \
    --field bundles.gates.g1.validation.path \
    --field bundles.gates.g1.validation.manifest_sha256 \
    --field bundles.gates.g1.eval.path \
    --field bundles.gates.g1.eval.manifest_sha256 \
    --field config_root)"
handoff_rc="$?"
set -e
if [[ "${handoff_rc}" -ne 0 ]]; then
    printf '%s\n' "${handoff_values}" >&2
    exit "${handoff_rc}"
fi
mapfile -t HANDOFF_VALUES <<< "${handoff_values}"
[[ "${#HANDOFF_VALUES[@]}" -eq 11 ]] || {
    echo "verified CPU handoff returned an unexpected field count" >&2
    exit 1
}
export REMEMR1_G0_TRAIN_PATH="${HANDOFF_VALUES[0]}"
export REMEMR1_G0_TRAIN_SHA256="${HANDOFF_VALUES[1]}"
export REMEMR1_G0_VALIDATION_PATH="${HANDOFF_VALUES[2]}"
export REMEMR1_G0_VALIDATION_SHA256="${HANDOFF_VALUES[3]}"
export REMEMR1_G1_TRAIN_PATH="${HANDOFF_VALUES[4]}"
export REMEMR1_G1_TRAIN_SHA256="${HANDOFF_VALUES[5]}"
export REMEMR1_G1_VALIDATION_PATH="${HANDOFF_VALUES[6]}"
export REMEMR1_G1_VALIDATION_SHA256="${HANDOFF_VALUES[7]}"
export REMEMR1_G1_EVAL_PATH="${HANDOFF_VALUES[8]}"
export REMEMR1_G1_EVAL_SHA256="${HANDOFF_VALUES[9]}"
export REMEMR1_CONFIG_ROOT="${HANDOFF_VALUES[10]}"
export REMEMR1_VERIFIED_HANDOFF="${handoff}"
export REMEMR1_VERIFIED_HANDOFF_FILE_SHA256="$(sha256sum "${handoff}" | awk '{print $1}')"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
write_value "${PIPELINE_DIR}/.gpu-started" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
rememr1_sync_file "${PIPELINE_DIR}/.gpu-started"
for stage in gpu-preflight g0 g1-step1 g1-resume2 g1-artifacts; do
    run_stage "${stage}"
done
CURRENT_STAGE="gpu-finalization"
finish_phase "${PIPELINE_DIR}/.gpu-gates-ready" "${handoff}"
write_value "${PIPELINE_DIR}/.success" "0"
rememr1_sync_file "${PIPELINE_DIR}/.success"
PIPELINE_FINISHED="yes"
echo "[pipeline] G-1, G0, and G1 gates complete: ${PIPELINE_DIR}" || true
