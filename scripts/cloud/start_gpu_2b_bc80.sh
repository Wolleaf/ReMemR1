#!/usr/bin/env bash
# Continue the paired B/C arms through step 80 and publish the L2 package.
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
exec bash "${REMEMR1_PROJECT_DIR}/scripts/cloud/launch.sh" --phase gpu-bc80 "$@"
