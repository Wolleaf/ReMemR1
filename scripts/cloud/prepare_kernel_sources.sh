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

prepare_repo() {
    local name="$1"
    local repository="$2"
    local commit="$3"
    local destination="${SOURCE_ROOT}/${name}"
    local unmaterialized_checkout=no
    if [[ ! -d "${destination}/.git" ]]; then
        [[ ! -e "${destination}" ]] || {
            echo "kernel source path exists but is not a git checkout: ${destination}" >&2
            return 1
        }
        git clone --filter=blob:none --no-checkout "${repository}" "${destination}"
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
    git -C "${destination}" fetch --no-tags --depth=1 origin "${commit}"
    git -C "${destination}" checkout --detach "${commit}"
    [[ "$(git -C "${destination}" rev-parse HEAD)" == "${commit}" ]] || return 1
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
