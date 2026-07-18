#!/usr/bin/env bash
# Build the pinned CUDA extensions from CPU-prefetched git objects only.
set -euo pipefail

CLOUD_ENV="${REMEMR1_CLOUD_ENV:-/root/autodl-tmp/rememr1-cloud.env}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/runtime.sh"
rememr1_load_cloud_env "${CLOUD_ENV}"
rememr1_require_cloud_env
cd "${REMEMR1_PROJECT_DIR}"

PYTHON="${REMEMR1_ENV_PREFIX}/bin/python"
SOURCE_ROOT="${REMEMR1_PERSIST_ROOT}/sources"
EVIDENCE_ROOT="${REMEMR1_PERSIST_ROOT}/evidence/gpu-environment"
BUILD_LOG="${EVIDENCE_ROOT}/kernel-build.log"
mkdir -p "${EVIDENCE_ROOT}"

[[ -x "${PYTHON}" ]] || {
    echo "persistent Python environment is missing; complete CPU preparation first" >&2
    exit 1
}
for source in flash-linear-attention causal-conv1d; do
    [[ -d "${SOURCE_ROOT}/${source}/.git" ]] || {
        echo "CPU-prefetched kernel source is missing: ${source}" >&2
        exit 1
    }
done

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
[[ -x "${CUDA_HOME}/bin/nvcc" ]] || {
    echo "CUDA 13.0 nvcc is missing at ${CUDA_HOME}/bin/nvcc" >&2
    exit 1
}
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-16}"
export MAX_JOBS="${MAX_JOBS:-16}"
export TORCH_CUDA_ARCH_LIST="12.0"

# Pip sees the original pinned HTTPS requirements (and records them in
# direct_url.json), while git rewrites those exact URLs to the persistent local
# mirrors. PIP_NO_INDEX and TRANSFORMERS_OFFLINE keep the paid phase offline.
export GIT_CONFIG_COUNT=2
export GIT_CONFIG_KEY_0="url.file://${SOURCE_ROOT}/flash-linear-attention/.insteadOf"
export GIT_CONFIG_VALUE_0="https://github.com/fla-org/flash-linear-attention.git"
export GIT_CONFIG_KEY_1="url.file://${SOURCE_ROOT}/causal-conv1d/.insteadOf"
export GIT_CONFIG_VALUE_1="https://github.com/Dao-AILab/causal-conv1d.git"
export PIP_NO_INDEX=1
# The pinned causal-conv1d setup otherwise tries to download a release wheel.
export CAUSAL_CONV1D_FORCE_BUILD=TRUE

tmp_log="${BUILD_LOG}.tmp.$$"
set +e
"${PYTHON}" -m pip install --no-deps --no-build-isolation --force-reinstall \
    -r environment/reproduction-sm120-kernels.requirements.txt \
    2>&1 | tee "${tmp_log}"
pipe_status=("${PIPESTATUS[@]}")
set -e
producer_rc="${pipe_status[0]}"
tee_rc="${pipe_status[1]}"
if [[ "${producer_rc}" -ne 0 || "${tee_rc}" -ne 0 ]]; then
    mv "${tmp_log}" "${BUILD_LOG}.failed.$(date -u +%Y%m%dT%H%M%SZ).$$" 2>/dev/null || true
    [[ "${producer_rc}" -ne 0 ]] && exit "${producer_rc}"
    exit "${tee_rc}"
fi
mv "${tmp_log}" "${BUILD_LOG}"

"${PYTHON}" -m pip check
"${PYTHON}" - <<'PY'
import json
from pathlib import Path

from scripts.reproduction.verify_environment import (
    load_environment_lock,
    verify_kernel_install_sources,
)

root = Path.cwd()
lock = load_environment_lock(
    root / "environment/reproduction-cu130.lock.json",
    repository_root=root,
)
verify_kernel_install_sources(lock)
print(json.dumps({"status": "pinned_kernel_sources_verified"}, sort_keys=True))
PY

freeze="${REMEMR1_PERSIST_ROOT}/evidence/pip-freeze.txt"
"${PYTHON}" -m pip freeze --all | LC_ALL=C sort > "${freeze}.tmp.$$"
mv "${freeze}.tmp.$$" "${freeze}"
rememr1_sync_file "${BUILD_LOG}"
rememr1_sync_file "${freeze}"
echo "Pinned GPU kernels built offline: ${BUILD_LOG}"
