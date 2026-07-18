#!/usr/bin/env bash
# Create the repository-external persistent environment and shutdown capability.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/runtime.sh"
source "${SCRIPT_DIR}/lib/shutdown.sh"

PROJECT_DIR="${REMEMR1_PROJECT_DIR:-/root/autodl-tmp/ReMemR1}"
EXPERIMENT_PROFILE="${REMEMR1_EXPERIMENT_PROFILE:-rtx5090-32g-qwen35-2b-v1}"
PERSIST_ROOT="${REMEMR1_PERSIST_ROOT:-/root/autodl-tmp/rememr1/profiles/${EXPERIMENT_PROFILE}}"
EXPECTED_COMMIT="${REMEMR1_EXPECTED_COMMIT:-}"
CLOUD_ENV="${REMEMR1_CLOUD_ENV:-/root/autodl-tmp/rememr1-cloud.env}"
HF_ENDPOINT_VALUE="${HF_ENDPOINT:-https://huggingface.co}"
ALLOW_GUEST_SHUTDOWN="no"

usage() {
    echo "Usage: $0 --expected-commit SHA [--project-dir DIR] [--persist-root DIR] [--experiment-profile ID] [--cloud-env FILE] [--hf-endpoint HTTPS_URL] [--allow-guest-shutdown]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project-dir) PROJECT_DIR="$2"; shift 2 ;;
        --persist-root) PERSIST_ROOT="$2"; shift 2 ;;
        --experiment-profile) EXPERIMENT_PROFILE="$2"; shift 2 ;;
        --expected-commit) EXPECTED_COMMIT="$2"; shift 2 ;;
        --cloud-env) CLOUD_ENV="$2"; shift 2 ;;
        --hf-endpoint) HF_ENDPOINT_VALUE="$2"; shift 2 ;;
        --allow-guest-shutdown) ALLOW_GUEST_SHUTDOWN="yes"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done

[[ "${EXPERIMENT_PROFILE}" == "rtx5090-32g-qwen35-2b-v1" ]] || {
    echo "unsupported experiment profile: ${EXPERIMENT_PROFILE}" >&2
    exit 2
}

[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || {
    echo "--expected-commit must be a full lowercase commit SHA" >&2
    exit 2
}
[[ "$(uname -s)" == "Linux" && "${EUID}" -eq 0 ]] || {
    echo "cloud initialization requires root on Linux" >&2
    exit 1
}
if grep -Eqi '(microsoft|wsl)' /proc/sys/kernel/osrelease 2>/dev/null; then
    echo "refusing to initialize automatic shutdown inside WSL" >&2
    exit 1
fi
[[ "${PROJECT_DIR}" =~ ^/root/autodl-tmp/[^/]+$ ]] || {
    echo "project directory must be a direct child of /root/autodl-tmp" >&2
    exit 2
}
case "${PERSIST_ROOT}" in
    /root/autodl-tmp/*) ;;
    *) echo "persistent root must be below /root/autodl-tmp" >&2; exit 2 ;;
esac
case "${CLOUD_ENV}" in
    /root/autodl-tmp/*) ;;
    *) echo "cloud env must be below /root/autodl-tmp" >&2; exit 2 ;;
esac
case "${HF_ENDPOINT_VALUE}" in
    https://*) ;;
    *) echo "Hugging Face endpoint must use HTTPS" >&2; exit 2 ;;
esac

PROJECT_DIR="$(realpath -e "${PROJECT_DIR}")"
[[ "${PROJECT_DIR}" =~ ^/root/autodl-tmp/[^/]+$ ]] || {
    echo "resolved project directory escaped the fixed AutoDL location" >&2
    exit 1
}
cd "${PROJECT_DIR}"
PERSIST_ROOT="$(realpath -m "${PERSIST_ROOT}")"
CLOUD_ENV="$(realpath -m "${CLOUD_ENV}")"
case "${PERSIST_ROOT}" in
    /root/autodl-tmp/*) ;;
    *) echo "resolved persistent root escaped /root/autodl-tmp" >&2; exit 1 ;;
esac
case "${CLOUD_ENV}" in
    /root/autodl-tmp/*) ;;
    *) echo "resolved cloud env escaped /root/autodl-tmp" >&2; exit 1 ;;
esac
case "${PERSIST_ROOT}/" in
    "${PROJECT_DIR}/"*)
        echo "persistent root must not be inside the clean checkout" >&2
        exit 1
        ;;
esac
case "${PROJECT_DIR}/" in
    "${PERSIST_ROOT}/"*)
        echo "clean checkout must not be inside the persistent output root" >&2
        exit 1
        ;;
esac
case "${CLOUD_ENV}" in
    "${PROJECT_DIR}"|"${PROJECT_DIR}/"*)
        echo "cloud env must be outside the clean checkout" >&2
        exit 1
        ;;
esac
if [[ -e "${CLOUD_ENV}" || -L "${CLOUD_ENV}" ]]; then
    [[ -f "${CLOUD_ENV}" && ! -L "${CLOUD_ENV}" && \
       "$(stat -c '%u' "${CLOUD_ENV}")" -eq 0 && \
       "$(stat -c '%a' "${CLOUD_ENV}")" == 600 ]] || {
        echo "existing cloud env must be a root-owned 0600 regular file" >&2
        exit 1
    }
    expected_persist_export="$(printf 'export REMEMR1_PERSIST_ROOT=%q' "${PERSIST_ROOT}")"
    expected_lock_export="$(printf 'export REMEMR1_LOCK_FILE=%q' "${PERSIST_ROOT}/cloud/pipeline.lock")"
    grep -Fqx "${expected_persist_export}" "${CLOUD_ENV}" && \
        grep -Fqx "${expected_lock_export}" "${CLOUD_ENV}" || {
            echo "existing cloud env targets another persistent root; refusing to overwrite it" >&2
            exit 1
        }
fi
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || {
    echo "checkout HEAD differs from --expected-commit" >&2
    exit 1
}
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
    echo "cloud checkout must be clean before initialization" >&2
    exit 1
}

command -v findmnt >/dev/null 2>&1 || {
    echo "findmnt is required to prove persistent storage" >&2
    exit 1
}
[[ -x /usr/bin/timeout ]] || {
    echo "/usr/bin/timeout is required for bounded initialization" >&2
    exit 1
}
rememr1_verify_persistent_mount /root/autodl-tmp || {
    echo "/root/autodl-tmp is not an independent durable mount" >&2
    exit 1
}

SHUTDOWN_BACKEND="none"
SHUTDOWN_BACKEND_PATH=""
SHUTDOWN_BACKEND_SHA256="$(printf '0%.0s' {1..64})"
if [[ "${ALLOW_GUEST_SHUTDOWN}" == "yes" ]]; then
    SHUTDOWN_BACKEND="autodl-wrapper-v1"
    SHUTDOWN_BACKEND_PATH="/usr/bin/shutdown"
    SHUTDOWN_BACKEND_SHA256="$(_shutdown_autodl_wrapper_sha256 \
        "${SHUTDOWN_BACKEND_PATH}")" || {
        echo "AutoDL's official /usr/bin/shutdown wrapper is unavailable" >&2
        exit 1
    }
fi

umask 077
mkdir -p \
    "${PERSIST_ROOT}/cache/huggingface" \
    "${PERSIST_ROOT}/cache/pip" \
    "${PERSIST_ROOT}/cache/torch" \
    "${PERSIST_ROOT}/cloud/launchers" \
    "${PERSIST_ROOT}/cloud/pipelines" \
    "${PERSIST_ROOT}/data" \
    "${PERSIST_ROOT}/envs" \
    "${PERSIST_ROOT}/evidence" \
    "${PERSIST_ROOT}/outputs" \
    "${PERSIST_ROOT}/ray" \
    "${PERSIST_ROOT}/sources" \
    "${PERSIST_ROOT}/tmp"

CAPABILITY_FILE="${PERSIST_ROOT}/cloud/shutdown-capability"
LOCK_FILE="${PERSIST_ROOT}/cloud/pipeline.lock"
LAUNCHER_ROOT="${PERSIST_ROOT}/cloud/launchers"
RESERVE_FILE="${PERSIST_ROOT}/cloud/terminal-reserve"
if [[ -e "${LOCK_FILE}" || -L "${LOCK_FILE}" ]]; then
    [[ -f "${LOCK_FILE}" && ! -L "${LOCK_FILE}" ]] || {
        echo "pipeline lock must be a regular non-symlink file" >&2
        exit 1
    }
else
    : > "${LOCK_FILE}"
    chmod 600 "${LOCK_FILE}"
    chown 0:0 "${LOCK_FILE}"
fi
[[ -x /usr/bin/flock ]] || {
    echo "/usr/bin/flock is required for safe cloud initialization" >&2
    exit 1
}
exec {init_lock_fd}>>"${LOCK_FILE}"
/usr/bin/flock -n "${init_lock_fd}" || {
    echo "refusing to initialize cloud state while another launcher is active" >&2
    exit 75
}
if [[ -e "${RESERVE_FILE}" ]]; then
    [[ -f "${RESERVE_FILE}" && ! -L "${RESERVE_FILE}" ]] || {
        echo "terminal reserve must be a regular non-symlink file" >&2
        exit 1
    }
fi
if [[ ! -f "${RESERVE_FILE}" || "$(stat -c '%s' "${RESERVE_FILE}")" -lt 8388608 ]]; then
    /usr/bin/timeout --signal=TERM --kill-after=30s 2m \
        dd if=/dev/zero of="${RESERVE_FILE}.tmp.$$" bs=1M count=8 \
        conv=fsync status=none
    chmod 600 "${RESERVE_FILE}.tmp.$$"
    chown 0:0 "${RESERVE_FILE}.tmp.$$"
    mv "${RESERVE_FILE}.tmp.$$" "${RESERVE_FILE}"
fi

cap_tmp="${CAPABILITY_FILE}.tmp.$$"
cat > "${cap_tmp}" <<EOF
schema_version=2
project_dir=${PROJECT_DIR}
persist_root=${PERSIST_ROOT}
expected_commit=${EXPECTED_COMMIT}
lock_file=${LOCK_FILE}
allow_guest_shutdown=${ALLOW_GUEST_SHUTDOWN}
shutdown_backend=${SHUTDOWN_BACKEND}
shutdown_backend_path=${SHUTDOWN_BACKEND_PATH}
shutdown_backend_sha256=${SHUTDOWN_BACKEND_SHA256}
EOF
chmod 600 "${cap_tmp}"
chown 0:0 "${cap_tmp}"
mv "${cap_tmp}" "${CAPABILITY_FILE}"

env_tmp="${CLOUD_ENV}.tmp.$$"
{
    printf 'export REMEMR1_PROJECT_DIR=%q\n' "${PROJECT_DIR}"
    printf 'export REMEMR1_PERSIST_ROOT=%q\n' "${PERSIST_ROOT}"
    printf 'export REMEMR1_EXPERIMENT_PROFILE=%q\n' "${EXPERIMENT_PROFILE}"
    printf 'export REMEMR1_EXPECTED_COMMIT=%q\n' "${EXPECTED_COMMIT}"
    printf 'export REMEMR1_CLOUD_ENV=%q\n' "${CLOUD_ENV}"
    printf 'export REMEMR1_CAPABILITY_FILE=%q\n' "${CAPABILITY_FILE}"
    printf 'export REMEMR1_LOCK_FILE=%q\n' "${LOCK_FILE}"
    printf 'export REMEMR1_LAUNCHER_ROOT=%q\n' "${LAUNCHER_ROOT}"
    printf 'export REMEMR1_RESERVE_FILE=%q\n' "${RESERVE_FILE}"
    printf 'export REMEMR1_PIPELINE_ROOT=%q\n' "${PERSIST_ROOT}/cloud/pipelines"
    printf 'export REMEMR1_ENV_PREFIX=%q\n' "${PERSIST_ROOT}/envs/reproduction-cu130"
    printf 'export HF_HOME=%q\n' "${PERSIST_ROOT}/cache/huggingface"
    printf 'export HF_ENDPOINT=%q\n' "${HF_ENDPOINT_VALUE}"
    printf 'export HUGGINGFACE_HUB_CACHE=%q\n' "${PERSIST_ROOT}/cache/huggingface/hub"
    printf 'export PIP_CACHE_DIR=%q\n' "${PERSIST_ROOT}/cache/pip"
    printf 'export TORCH_HOME=%q\n' "${PERSIST_ROOT}/cache/torch"
    printf 'export RAY_TMPDIR=%q\n' /root/autodl-tmp/ray
    printf 'export TMPDIR=%q\n' "${PERSIST_ROOT}/tmp"
    printf 'export PYTHONPATH=%q\n' "${PROJECT_DIR}"
    printf 'export PYTHONOPTIMIZE=%q\n' ''
    printf 'export FLA_TILELANG=%q\n' 0
    printf 'export OMP_NUM_THREADS=%q\n' 1
} > "${env_tmp}"
chmod 600 "${env_tmp}"
chown 0:0 "${env_tmp}"
mv "${env_tmp}" "${CLOUD_ENV}"
/usr/bin/timeout --signal=TERM --kill-after=30s 2m sync -f "${CLOUD_ENV}"
/usr/bin/timeout --signal=TERM --kill-after=30s 2m sync -f "${CAPABILITY_FILE}"

echo "Cloud state initialized: ${CLOUD_ENV}"
echo "Pinned checkout: ${PROJECT_DIR}@${EXPECTED_COMMIT}"
echo "Persistent root: ${PERSIST_ROOT}"
echo "Experiment profile: ${EXPERIMENT_PROFILE}"
