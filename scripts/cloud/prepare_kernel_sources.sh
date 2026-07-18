#!/usr/bin/env bash
# Fetch immutable GPU kernel sources during the unmetered CPU/network phase.
set -euo pipefail

CLOUD_ENV="${REMEMR1_CLOUD_ENV:-/root/autodl-tmp/rememr1-cloud.env}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/runtime.sh"
rememr1_load_cloud_env "${CLOUD_ENV}"
rememr1_require_cloud_env
cd "${REMEMR1_PROJECT_DIR}"

SOURCE_ROOT="${REMEMR1_PERSIST_ROOT}/sources"
mkdir -p "${SOURCE_ROOT}"

git_fetch_retry() {
    local repository_path="$1"
    shift
    local attempt
    for attempt in 1 2 3 4; do
        if timeout 15m git \
            -c http.version=HTTP/1.1 \
            -c http.lowSpeedLimit=1024 \
            -c http.lowSpeedTime=60 \
            -C "${repository_path}" fetch "$@"; then
            return 0
        fi
        [[ "${attempt}" -lt 4 ]] || return 1
        sleep "$((attempt * 3))"
    done
}

prepare_repo() {
    local name="$1"
    local repository="$2"
    local commit="$3"
    local destination="${SOURCE_ROOT}/${name}"
    local unmaterialized_checkout=no
    local original_promisor=""
    local original_partial_filter=""
    if [[ ! -d "${destination}/.git" ]]; then
        [[ ! -e "${destination}" ]] || {
            echo "kernel source path exists but is not a git checkout: ${destination}" >&2
            return 1
        }
        git init --quiet "${destination}"
        git -C "${destination}" remote add origin "${repository}"
    fi
    if [[ "$(git -C "${destination}" remote get-url origin)" != "${repository}" ]]; then
        echo "kernel source origin mismatch: ${destination}" >&2
        return 1
    fi
    if [[ ! -e "${destination}/.git/index" && \
          ! -L "${destination}/.git/index" && \
          -z "$(find "${destination}" -mindepth 1 -maxdepth 1 \
              ! -name .git -print -quit)" ]]; then
        unmaterialized_checkout=yes
    fi
    if [[ "${unmaterialized_checkout}" != yes && \
          -n "$(git -C "${destination}" status --porcelain --untracked-files=all)" ]]; then
        echo "kernel source checkout is dirty: ${destination}" >&2
        return 1
    fi
    original_promisor="$(git -C "${destination}" config --bool --get remote.origin.promisor || true)"
    original_partial_filter="$(git -C "${destination}" config --get remote.origin.partialclonefilter || true)"
    if [[ "$(git -C "${destination}" rev-parse --is-shallow-repository)" == true && \
          ("${original_promisor}" == true || -n "${original_partial_filter}") ]]; then
        git_fetch_retry "${destination}" --unshallow --no-filter origin
    fi
    if [[ "${original_promisor}" == true || -n "${original_partial_filter}" ]]; then
        git_fetch_retry "${destination}" --refetch --no-filter origin
        git -C "${destination}" config --unset-all remote.origin.promisor || true
        git -C "${destination}" config --unset-all remote.origin.partialclonefilter || true
    fi
    if ! git -C "${destination}" cat-file -e "${commit}^{commit}" 2>/dev/null; then
        git_fetch_retry "${destination}" \
            --no-tags --depth=1 --no-filter origin "${commit}"
    fi
    git -C "${destination}" checkout --detach "${commit}"
    git -C "${destination}" update-ref refs/heads/rememr1-pinned "${commit}"
    git -C "${destination}" config uploadpack.allowFilter true
    git -C "${destination}" config uploadpack.allowAnySHA1InWant true
    [[ "$(git -C "${destination}" rev-parse HEAD)" == "${commit}" ]] || return 1
    [[ "$(git -C "${destination}" config --bool --get remote.origin.promisor || true)" != true ]] || {
        echo "kernel source checkout remains partial: ${destination}" >&2
        return 1
    }
    if ! GIT_NO_LAZY_FETCH=1 git -C "${destination}" rev-list \
        --objects "${commit}" --missing=error >/dev/null; then
        echo "kernel source pinned snapshot is incomplete: ${destination}" >&2
        [[ -z "${original_promisor}" ]] || \
            git -C "${destination}" config remote.origin.promisor "${original_promisor}"
        [[ -z "${original_partial_filter}" ]] || \
            git -C "${destination}" config remote.origin.partialclonefilter \
                "${original_partial_filter}"
        return 1
    fi
    if ! GIT_NO_LAZY_FETCH=1 git -C "${destination}" fsck --full --no-dangling; then
        [[ -z "${original_promisor}" ]] || \
            git -C "${destination}" config remote.origin.promisor "${original_promisor}"
        [[ -z "${original_partial_filter}" ]] || \
            git -C "${destination}" config remote.origin.partialclonefilter \
                "${original_partial_filter}"
        return 1
    fi
    if [[ -n "$(git -C "${destination}" status --porcelain --untracked-files=all)" ]]; then
        echo "kernel source checkout is dirty after materialization: ${destination}" >&2
        return 1
    fi
}

prepare_repo \
    flash-linear-attention \
    https://github.com/fla-org/flash-linear-attention.git \
    b328e7c611ca205d1908cce9a90b8ca223fc0101
prepare_repo \
    causal-conv1d \
    https://github.com/Dao-AILab/causal-conv1d.git \
    4f6ae4e26ae5fe8af9372f8d312ab25cc4595223

echo "Pinned kernel sources ready: ${SOURCE_ROOT}"
