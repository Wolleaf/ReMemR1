#!/usr/bin/env bash
# Revalidate the immutable clean checkout before every cloud phase.
set -euo pipefail

CLOUD_ENV="${REMEMR1_CLOUD_ENV:-/root/autodl-tmp/rememr1-cloud.env}"
[[ -f "${CLOUD_ENV}" ]] || {
    echo "cloud environment is missing; run scripts/cloud/init_cloud.sh first" >&2
    exit 1
}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/runtime.sh"
rememr1_load_cloud_env "${CLOUD_ENV}"
rememr1_require_cloud_env

[[ "${REMEMR1_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || {
    echo "REMEMR1_EXPECTED_COMMIT is invalid" >&2
    exit 1
}
project="$(realpath -e "${REMEMR1_PROJECT_DIR}")"
case "${project}" in
    /root/autodl-tmp/*) ;;
    *) echo "cloud checkout escaped /root/autodl-tmp" >&2; exit 1 ;;
esac
cd "${project}"
[[ "$(git rev-parse --show-toplevel)" == "${project}" ]] || {
    echo "configured project is not the checkout root" >&2
    exit 1
}
[[ "$(git rev-parse HEAD)" == "${REMEMR1_EXPECTED_COMMIT}" ]] || {
    echo "checkout commit differs from the cloud pin" >&2
    exit 1
}
dirty="$(git status --porcelain --untracked-files=all)"
[[ -z "${dirty}" ]] || {
    echo "cloud checkout is dirty; refusing an ambiguous run" >&2
    printf '%s\n' "${dirty}" >&2
    exit 1
}
git diff --check --cached
echo "Verified clean checkout: ${project}@${REMEMR1_EXPECTED_COMMIT}"
