#!/usr/bin/env bash
# Initialize a manually checked out revision and launch bounded CPU preparation.
set -euo pipefail
umask 077

usage() {
    cat >&2 <<'EOF'
Usage: prepare_cpu.sh [--keep-running] [--dry-run] [--retry-failed-stage]

The checkout containing this script must be clean. Its current full HEAD is
sealed as the CPU-to-GPU workflow identity before the detached launcher starts.
EOF
}

start_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep-running|--dry-run|--retry-failed-stage)
            start_args+=("$1")
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown CPU preparation argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
git_root="$(git -C "${PROJECT_DIR}" rev-parse --show-toplevel 2>/dev/null)" || {
    echo "CPU preparation requires a Git checkout" >&2
    exit 1
}
git_root="$(realpath -e -- "${git_root}")"
[[ "${git_root}" == "${PROJECT_DIR}" ]] || {
    echo "prepare_cpu.sh must run from the checkout that contains it" >&2
    exit 1
}
expected_commit="$(git -C "${PROJECT_DIR}" rev-parse --verify 'HEAD^{commit}')"
[[ "${expected_commit}" =~ ^[0-9a-f]{40}$ ]] || {
    echo "checkout HEAD is not a full lowercase commit SHA" >&2
    exit 1
}
dirty="$(git -C "${PROJECT_DIR}" status --porcelain --untracked-files=all)"
[[ -z "${dirty}" ]] || {
    echo "CPU preparation refuses a dirty checkout" >&2
    printf '%s\n' "${dirty}" >&2
    exit 1
}
git -C "${PROJECT_DIR}" diff --check --cached

# AutoDL's no-card shell can expose NVIDIA device files from the container
# image. Mask both CUDA selectors and prove the mask again in install_env.sh.
export CUDA_VISIBLE_DEVICES=''
export NVIDIA_VISIBLE_DEVICES=void
export REMEMR1_ALLOW_GPU_CPU_PHASE=yes
export REMEMR1_PROJECT_DIR="${PROJECT_DIR}"
export REMEMR1_EXPECTED_COMMIT="${expected_commit}"
export REMEMR1_MIN_CPU_CORES="${REMEMR1_MIN_CPU_CORES:-1/2}"
export REMEMR1_MIN_RAM_GIB="${REMEMR1_MIN_RAM_GIB:-2}"
export REMEMR1_CPU_LOW_RESOURCE=yes

init_args=(
    --project-dir "${PROJECT_DIR}"
    --expected-commit "${expected_commit}"
    --allow-guest-shutdown
)
[[ -z "${REMEMR1_PERSIST_ROOT:-}" ]] || \
    init_args+=(--persist-root "${REMEMR1_PERSIST_ROOT}")
[[ -z "${REMEMR1_EXPERIMENT_PROFILE:-}" ]] || \
    init_args+=(--experiment-profile "${REMEMR1_EXPERIMENT_PROFILE}")
[[ -z "${REMEMR1_CLOUD_ENV:-}" ]] || \
    init_args+=(--cloud-env "${REMEMR1_CLOUD_ENV}")
[[ -z "${HF_ENDPOINT:-}" ]] || init_args+=(--hf-endpoint "${HF_ENDPOINT}")

bash "${SCRIPT_DIR}/init_cloud.sh" "${init_args[@]}"
exec bash "${SCRIPT_DIR}/start_cpu_prep.sh" "${start_args[@]}"
