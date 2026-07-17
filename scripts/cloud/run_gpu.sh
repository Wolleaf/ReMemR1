#!/usr/bin/env bash
# Finish CPU sealing, run bounded GPU gates, then run the fixed R0 capacity phase.
set -euo pipefail

CLOUD_ENV="${REMEMR1_CLOUD_ENV:-/root/autodl-tmp/rememr1-cloud.env}"
[[ -f "${CLOUD_ENV}" ]] || {
    echo "cloud environment is missing; CPU preparation has not been initialized" >&2
    exit 1
}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/runtime.sh"
rememr1_load_cloud_env "${CLOUD_ENV}"
rememr1_require_cloud_env
# AutoDL images can install CUDA under /usr/local/cuda without adding nvcc to
# non-interactive SSH shells. Keep the complete GPU workflow on one toolchain.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
exec bash "${REMEMR1_PROJECT_DIR}/scripts/cloud/launch.sh" --phase gpu "$@"
