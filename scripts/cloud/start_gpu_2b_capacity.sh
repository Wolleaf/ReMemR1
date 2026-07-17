#!/usr/bin/env bash
# Run the paid G2 capacity phase for one explicitly selected offload profile.
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
exec bash "${REMEMR1_PROJECT_DIR}/scripts/cloud/launch.sh" \
    --phase gpu-capacity "$@"
