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
    cpu-preflight|cpu-environment|cpu-kernel-sources|cpu-assets|cpu-data|cpu-tests|cpu-configs|cpu-handoff|gpu-preflight|g0|g1-step1|g1-resume2|g1-artifacts) ;;
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
ATTEMPT_ROOT="${PIPELINE_REAL}/attempts"
[[ -d "${ATTEMPT_ROOT}" && ! -L "${ATTEMPT_ROOT}" && \
   "$(rememr1_realpath_existing "${ATTEMPT_ROOT}")" == "${ATTEMPT_ROOT}" ]] || {
    echo "pipeline attempt root is missing or unsafe" >&2
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

write_atomic "${REMEMR1_STAGE_RESULT_FILE}" "${RUN_DIR}"
write_atomic "${RUN_DIR}/.running" "$$"
cat > "${RUN_DIR}/run.meta.tmp.$$" <<EOF
schema_version=1
stage=${STAGE}
git_commit=${REMEMR1_EXPECTED_COMMIT}
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
    else
        terminal_marker="${RUN_DIR}/.failed"
        if ! write_atomic "${terminal_marker}" "${rc}"; then
            echo "stage failure marker could not be persisted" >&2
            exit 74
        fi
    fi

    if ! rememr1_sync_file "${terminal_marker}"; then
        if [[ "${rc}" -eq 0 ]]; then
            rc=74
            rm -f "${RUN_DIR}/.success" || true
            terminal_marker="${RUN_DIR}/.failed"
            write_atomic "${terminal_marker}" "${rc}" && \
                rememr1_sync_file "${terminal_marker}" || true
        fi
        exit "${rc}"
    fi
    if ! rm -f "${RUN_DIR}/.running"; then
        if [[ "${rc}" -eq 0 ]]; then
            rc=74
            rm -f "${RUN_DIR}/.success" || true
            terminal_marker="${RUN_DIR}/.failed"
            write_atomic "${terminal_marker}" "${rc}" && \
                rememr1_sync_file "${terminal_marker}" || true
        fi
        exit "${rc}"
    fi
    if ! rememr1_sync_file "${terminal_marker}"; then
        if [[ "${rc}" -eq 0 ]]; then
            rc=74
            rm -f "${RUN_DIR}/.success" || true
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

run_training() {
    local config_name="$1"
    local train_path="$2"
    local train_sha="$3"
    local val_path="$4"
    local val_sha="$5"
    local checkpoint_dir="$6"
    local adapter_dir="$7"
    shift 7
    local training_rc cleanup_rc
    set +e
    run_logged "train-${config_name}" 6h \
        "${PYTHON}" -m verl.trainer.main_ppo \
            "--config-name=reproduction/${config_name}" \
            "data.train_files=${train_path}/train.parquet" \
            "data.val_files=${val_path}/train.parquet" \
            "reproduction.data_manifest_sha256=${train_sha}" \
            "reproduction.val_data_manifest_sha256=${val_sha}" \
            "trainer.default_local_dir=${checkpoint_dir}" \
            "reproduction.adapter_export_dir=${adapter_dir}" \
            "$@"
    training_rc="$?"
    stop_ray
    cleanup_rc="$?"
    set -e
    if [[ "${cleanup_rc}" -ne 0 ]]; then
        echo "Ray cleanup failed after ${config_name} (exit ${cleanup_rc})" >&2
        [[ "${training_rc}" -eq 0 ]] && return "${cleanup_rc}"
    fi
    return "${training_rc}"
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
        for command_name in curl findmnt flock git sha256sum setsid timeout; do
            command -v "${command_name}" >/dev/null 2>&1 || {
                echo "required command is missing: ${command_name}" >&2
                exit 1
            }
        done
        free_bytes="$(df --output=avail -B1 "${REMEMR1_PERSIST_ROOT}" | awk 'NR == 2 {print $1}')"
        min_free_gib="${REMEMR1_MIN_FREE_GIB:-80}"
        [[ "${min_free_gib}" =~ ^[0-9]+$ ]] || {
            echo "REMEMR1_MIN_FREE_GIB must be an integer" >&2
            exit 2
        }
        min_free_bytes="$((min_free_gib * 1024 * 1024 * 1024))"
        [[ "${free_bytes}" =~ ^[0-9]+$ && "${free_bytes}" -ge "${min_free_bytes}" ]] || {
            echo "CPU preparation requires at least ${min_free_gib} GiB free on the persistent volume" >&2
            exit 1
        }
        memory_kib="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
        min_ram_gib="${REMEMR1_MIN_RAM_GIB:-48}"
        [[ "${min_ram_gib}" =~ ^[0-9]+$ ]] || {
            echo "REMEMR1_MIN_RAM_GIB must be an integer" >&2
            exit 2
        }
        min_memory_kib="$((min_ram_gib * 1024 * 1024))"
        [[ "${memory_kib}" =~ ^[0-9]+$ && "${memory_kib}" -ge "${min_memory_kib}" ]] || {
            echo "CPU preparation requires at least ${min_ram_gib} GiB RAM for formal parquet materialization" >&2
            exit 1
        }
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
        run_logged hardware-identity 5m "${PYTHON}" -c \
            'import sys, torch; available = torch.cuda.is_available(); n = torch.cuda.get_device_name(0) if available else "unavailable"; c = list(torch.cuda.get_device_capability(0)) if available else []; m = torch.cuda.get_device_properties(0).total_memory if available else 0; valid = available and c == [12, 0] and "RTX PRO 6000" in n.upper() and m >= 90*1024**3; print(n, c, m); sys.exit(f"unsupported GPU identity: {(n, c, m)}") if not valid else None'
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
                --optimizer-steps 20
        fi
        ;;
    g0)
        export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
        require_verified_handoff_context
        train_path="${REMEMR1_G0_TRAIN_PATH}"
        train_sha="${REMEMR1_G0_TRAIN_SHA256}"
        val_path="${REMEMR1_G0_VALIDATION_PATH}"
        val_sha="${REMEMR1_G0_VALIDATION_SHA256}"
        run_training g0_qwen35_08b \
            "${train_path}" "${train_sha}" "${val_path}" "${val_sha}" \
            "${REMEMR1_PIPELINE_DIR}/checkpoints/g0" \
            "${REMEMR1_PIPELINE_DIR}/artifacts/g0/adapter"
        ;;
    g1-step1)
        export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
        require_verified_handoff_context
        train_path="${REMEMR1_G1_TRAIN_PATH}"
        train_sha="${REMEMR1_G1_TRAIN_SHA256}"
        val_path="${REMEMR1_G1_VALIDATION_PATH}"
        val_sha="${REMEMR1_G1_VALIDATION_SHA256}"
        run_training g1_qwen35_2b_step1 \
            "${train_path}" "${train_sha}" "${val_path}" "${val_sha}" \
            "${REMEMR1_PIPELINE_DIR}/checkpoints/g1-step1" \
            "${REMEMR1_PIPELINE_DIR}/artifacts/g1-step1/adapter"
        ;;
    g1-resume2)
        export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
        require_verified_handoff_context
        predecessor="${REMEMR1_PIPELINE_DIR}/checkpoints/g1-step1/global_step_1"
        [[ -d "${predecessor}" ]] || {
            echo "G1 step-1 checkpoint is missing: ${predecessor}" >&2
            exit 1
        }
        train_path="${REMEMR1_G1_TRAIN_PATH}"
        train_sha="${REMEMR1_G1_TRAIN_SHA256}"
        val_path="${REMEMR1_G1_VALIDATION_PATH}"
        val_sha="${REMEMR1_G1_VALIDATION_SHA256}"
        run_training g1_qwen35_2b_resume2 \
            "${train_path}" "${train_sha}" "${val_path}" "${val_sha}" \
            "${REMEMR1_PIPELINE_DIR}/checkpoints/g1-resume2" \
            "${REMEMR1_PIPELINE_DIR}/artifacts/g1-resume2/adapter" \
            "trainer.resume_from_path=${predecessor}"
        ;;
    g1-artifacts)
        export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
        require_verified_handoff_context
        adapter="${REMEMR1_PIPELINE_DIR}/artifacts/g1-resume2/adapter/global_step_2/adapter"
        merged="${REMEMR1_PIPELINE_DIR}/artifacts/g1-resume2/merged-global_step_2"
        run_logged adapter-verify 30m \
            "${PYTHON}" -m scripts.reproduction.export_adapter verify \
            --adapter-dir "${adapter}"
        run_logged adapter-hf-recurrent-eval 3h \
            "${PYTHON}" scripts/cloud/g1_eval.py \
            --bundle-dir "${REMEMR1_G1_EVAL_PATH}" \
            --expected-manifest-sha256 "${REMEMR1_G1_EVAL_SHA256}" \
            --adapter-dir "${adapter}" \
            --output-dir "${REMEMR1_PIPELINE_DIR}/artifacts/g1-resume2/hf-recurrent-eval"
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
esac

STAGE_FINISHED="yes"
finish_stage 0
