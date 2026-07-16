#!/usr/bin/env bash
# Clone one immutable revision, initialize persistent cloud state, and launch CPU preparation.
set -euo pipefail
umask 077

REPO_URL="${REMEMR1_REPO_URL:-https://github.com/Wolleaf/ReMemR1.git}"
BRANCH="${REMEMR1_BRANCH:-reproduction/qwen35-plan}"
EXPECTED_COMMIT="${REMEMR1_EXPECTED_COMMIT:-}"
PROJECT_DIR="${REMEMR1_PROJECT_DIR:-/root/autodl-tmp/ReMemR1}"
PHASE="cpu"
ALLOW_GUEST_SHUTDOWN="no"
START_ARGS=()

usage() {
    cat >&2 <<'EOF'
Usage: bootstrap.sh [--repo URL] [--branch NAME] [--expected-commit SHA]
                    [--project-dir DIR] [--phase cpu|init-only]
                    [--allow-guest-shutdown] [--keep-running] [--dry-run]

Without --expected-commit, the branch tip is resolved once and recorded as a
full immutable commit before checkout. Supply the SHA explicitly for a fully
operator-pinned launch.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo) REPO_URL="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --expected-commit) EXPECTED_COMMIT="$2"; shift 2 ;;
        --project-dir) PROJECT_DIR="$2"; shift 2 ;;
        --phase) PHASE="$2"; shift 2 ;;
        --allow-guest-shutdown) ALLOW_GUEST_SHUTDOWN="yes"; shift ;;
        --keep-running|--dry-run) START_ARGS+=("$1"); shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done

case "${PHASE}" in
    cpu|init-only) ;;
    *) usage; exit 2 ;;
esac
shutdown_disabled="no"
for start_arg in "${START_ARGS[@]}"; do
    [[ "${start_arg}" == "--keep-running" || "${start_arg}" == "--dry-run" ]] && \
        shutdown_disabled="yes"
done
if [[ "${PHASE}" == "cpu" && "${ALLOW_GUEST_SHUTDOWN}" != "yes" && \
      "${shutdown_disabled}" != "yes" ]]; then
    echo "automatic mode requires explicit --allow-guest-shutdown" >&2
    exit 2
fi
if [[ -n "${EXPECTED_COMMIT}" && ! "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "--expected-commit must be a full lowercase commit SHA" >&2
    exit 2
fi
if [[ "$(uname -s)" != "Linux" || "${EUID}" -ne 0 ]]; then
    echo "bootstrap must run as root on the Linux cloud guest" >&2
    exit 1
fi
if [[ "${REMEMR1_TEST_MODE:-no}" != "no" || \
      -n "${REMEMR1_TEST_SHUTDOWN_LOG:-}" ]]; then
    echo "bootstrap does not support the launcher test hook" >&2
    exit 2
fi
for command_name in findmnt flock git mkfifo realpath sync tee timeout; do
    command -v "${command_name}" >/dev/null 2>&1 || {
        echo "required command is missing: ${command_name}" >&2
        exit 1
    }
done
verify_bootstrap_persistent_mount() {
    local path="$1"
    local target source fstype major_minor root_source root_major_minor
    target="$(findmnt -n -o TARGET --target "${path}")" || return
    target="$(realpath -e -- "${target}")" || return
    [[ "${target}" == "/root/autodl-tmp" ]] || return 1
    source="$(findmnt -n -o SOURCE --target "${path}")" || return
    fstype="$(findmnt -n -o FSTYPE --target "${path}")" || return
    major_minor="$(findmnt -n -o MAJ:MIN --target "${path}")" || return
    root_source="$(findmnt -n -o SOURCE --target /)" || return
    root_major_minor="$(findmnt -n -o MAJ:MIN --target /)" || return
    case "${fstype,,}" in
        overlay|tmpfs|ramfs|rootfs|devtmpfs|squashfs) return 1 ;;
    esac
    [[ -n "${source}" && -n "${major_minor}" && \
       "${source}" != "${root_source}" && \
       "${major_minor}" != "${root_major_minor}" ]]
}
if [[ "${shutdown_disabled}" != "yes" && \
      ! -x /usr/sbin/shutdown && ! -x /sbin/shutdown && ! -x /usr/bin/systemctl ]]; then
    echo "no supported shutdown backend is available" >&2
    exit 1
fi
[[ "${PROJECT_DIR}" =~ ^/root/autodl-tmp/[^/]+$ ]] || {
    echo "--project-dir must be a direct child of /root/autodl-tmp" >&2
    exit 2
}
if grep -Eqi '(microsoft|wsl)' /proc/sys/kernel/osrelease 2>/dev/null; then
    echo "refusing cloud bootstrap inside WSL" >&2
    exit 1
fi
verify_bootstrap_persistent_mount /root/autodl-tmp || {
    echo "/root/autodl-tmp is not an independent durable mount" >&2
    exit 1
}

BOOTSTRAP_ROOT="${REMEMR1_PERSIST_ROOT:-/root/autodl-tmp/rememr1}"
case "${BOOTSTRAP_ROOT}" in
    /root/autodl-tmp/*) ;;
    *) echo "bootstrap persistent root must be below /root/autodl-tmp" >&2; exit 2 ;;
esac
BOOTSTRAP_LOCK="${BOOTSTRAP_ROOT}/cloud/pipeline.lock"
mkdir -p "${BOOTSTRAP_ROOT}/cloud/bootstrap" "$(dirname "${BOOTSTRAP_LOCK}")"
exec 8>>"${BOOTSTRAP_LOCK}"
if ! flock -n 8; then
    echo "cloud workspace lock is busy; refusing bootstrap and shutdown" >&2
    exit 75
fi
BOOTSTRAP_LOCK_HELD="yes"
BOOTSTRAP_INITIALIZED="no"
BOOTSTRAP_RUN="${BOOTSTRAP_ROOT}/cloud/bootstrap/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir "${BOOTSTRAP_RUN}"
BOOTSTRAP_LOG_FIFO="${BOOTSTRAP_RUN}/.bootstrap-log-pipe"
mkfifo "${BOOTSTRAP_LOG_FIFO}"
/usr/bin/tee -a "${BOOTSTRAP_RUN}/bootstrap.log" < "${BOOTSTRAP_LOG_FIFO}" 8>&- &
BOOTSTRAP_TEE_PID=$!
exec > "${BOOTSTRAP_LOG_FIFO}" 2>&1
rm -f "${BOOTSTRAP_LOG_FIFO}"
BOOTSTRAP_LOG_DRAINED="no"
BOOTSTRAP_LOG_DRAIN_TIMEOUT_SECONDS=30

drain_bootstrap_log() {
    local tee_wait_rc=0 tee_wait_timed_out=no timer_pid
    [[ "${BOOTSTRAP_LOG_DRAINED}" == "no" ]] || return 0
    printf '[bootstrap] terminal-state-published\n'
    exec 1>&- 2>&-
    trap 'tee_wait_timed_out=yes' USR1
    (
        exec 8>&-
        /usr/bin/sleep "${BOOTSTRAP_LOG_DRAIN_TIMEOUT_SECONDS}"
        kill -USR1 "$$"
    ) &
    timer_pid=$!
    if wait "${BOOTSTRAP_TEE_PID}"; then
        tee_wait_rc=0
    else
        tee_wait_rc=$?
    fi
    kill -TERM "${timer_pid}" 2>/dev/null || true
    wait "${timer_pid}" 2>/dev/null || true
    trap - USR1
    if [[ "${tee_wait_timed_out}" == "yes" ]]; then
        kill -TERM "${BOOTSTRAP_TEE_PID}" 2>/dev/null || true
        kill -KILL "${BOOTSTRAP_TEE_PID}" 2>/dev/null || true
    fi
    if [[ "${tee_wait_rc}" -eq 0 && "${tee_wait_timed_out}" == "no" ]] && \
       grep -Fqx '[bootstrap] terminal-state-published' \
        "${BOOTSTRAP_RUN}/bootstrap.log"; then
        BOOTSTRAP_LOG_DRAINED="yes"
        return 0
    fi
    printf '%s\n' \
        "bootstrap tee failed while draining (rc=${tee_wait_rc}, timeout=${tee_wait_timed_out})" > \
        "${BOOTSTRAP_RUN}/log-drain-failed"
    return 1
}

bootstrap_failure_shutdown_authorized() {
    local descriptor_path lock_path cloud_env
    [[ "${REMEMR1_TEST_MODE:-no}" == "no" && \
       -z "${REMEMR1_TEST_SHUTDOWN_LOG:-}" ]] || return 1
    [[ "$(uname -s)" == "Linux" && "${EUID}" -eq 0 ]] || return 1
    ! grep -Eqi '(microsoft|wsl)' /proc/sys/kernel/osrelease 2>/dev/null || return 1
    [[ "${PROJECT_DIR}" =~ ^/root/autodl-tmp/[^/]+$ ]] || return 1
    descriptor_path="$(realpath -e "/proc/$$/fd/8" 2>/dev/null)" || return 1
    lock_path="$(realpath -e "${BOOTSTRAP_LOCK}" 2>/dev/null)" || return 1
    [[ "${descriptor_path}" == "${lock_path}" ]] || return 1
    flock -n 8 || return 1
    verify_bootstrap_persistent_mount "${BOOTSTRAP_ROOT}" || return 1
    [[ -f "${BOOTSTRAP_RUN}/bootstrap.log" && \
       -f "${BOOTSTRAP_RUN}/exit-code" && \
       -f "${BOOTSTRAP_RUN}/terminal" ]] || return 1
    if [[ "${BOOTSTRAP_INITIALIZED}" == "yes" ]]; then
        cloud_env="${REMEMR1_CLOUD_ENV:-/root/autodl-tmp/rememr1-cloud.env}"
        (
            # Reuse the production preflight once the pinned checkout initialized it.
            source "${PROJECT_DIR}/scripts/cloud/lib/runtime.sh"
            source "${PROJECT_DIR}/scripts/cloud/lib/shutdown.sh"
            rememr1_load_cloud_env "${cloud_env}" &&
                rememr1_require_cloud_env &&
                rememr1_validate_test_mode &&
                verify_guest_shutdown_preflight
        ) || return 1
    fi
}

bootstrap_exit() {
    local rc="$?"
    trap - EXIT INT TERM
    if [[ "${rc}" -ne 0 ]]; then
        printf '%s\n' "${rc}" > "${BOOTSTRAP_RUN}/exit-code.tmp"
        mv "${BOOTSTRAP_RUN}/exit-code.tmp" "${BOOTSTRAP_RUN}/exit-code"
        printf '%s\n' "failed" > "${BOOTSTRAP_RUN}/terminal"
        drain_bootstrap_log || true
        if [[ "${BOOTSTRAP_LOCK_HELD}" != "yes" ]]; then
            exec 8>>"${BOOTSTRAP_LOCK}"
            if flock -n 8; then
                BOOTSTRAP_LOCK_HELD="yes"
            fi
        fi
        if [[ "${rc}" != "75" && "${BOOTSTRAP_LOCK_HELD}" == "yes" && \
              "${BOOTSTRAP_LOG_DRAINED}" == "yes" && \
              "${ALLOW_GUEST_SHUTDOWN}" == "yes" && \
              "${shutdown_disabled}" != "yes" ]]; then
            if bootstrap_failure_shutdown_authorized && \
               timeout --signal=TERM --kill-after=30s 5m /usr/bin/sync; then
                printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > \
                    "${BOOTSTRAP_RUN}/shutdown-safe"
                if timeout --signal=TERM --kill-after=30s 5m /usr/bin/sync; then
                    printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > \
                        "${BOOTSTRAP_RUN}/shutdown-requested"
                    if timeout --signal=TERM --kill-after=30s 2m \
                       /usr/bin/sync -f "${BOOTSTRAP_RUN}/shutdown-requested"; then
                        timeout --signal=TERM --kill-after=30s 2m /usr/sbin/shutdown -h now || \
                        timeout --signal=TERM --kill-after=30s 2m /sbin/shutdown -h now || \
                        timeout --signal=TERM --kill-after=30s 2m /usr/bin/systemctl poweroff || \
                            printf '%s\n' "shutdown command failed" > "${BOOTSTRAP_RUN}/shutdown-failed"
                    else
                        printf '%s\n' "shutdown-request-marker-sync-failed" > \
                            "${BOOTSTRAP_RUN}/shutdown-skipped"
                    fi
                else
                    printf '%s\n' "shutdown-safe-sync-failed" > \
                        "${BOOTSTRAP_RUN}/shutdown-skipped"
                fi
            else
                printf '%s\n' "authorization-or-sync-failed" > \
                    "${BOOTSTRAP_RUN}/shutdown-skipped"
            fi
        fi
    else
        printf '%s\n' "0" > "${BOOTSTRAP_RUN}/exit-code"
        printf '%s\n' "success" > "${BOOTSTRAP_RUN}/terminal"
        drain_bootstrap_log || true
    fi
    exit "${rc}"
}
trap bootstrap_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ -z "${EXPECTED_COMMIT}" ]]; then
    remote_line=""
    for attempt in 1 2 3; do
        if remote_line="$(timeout --signal=TERM --kill-after=30s 5m \
            git -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=60 \
            ls-remote --exit-code "${REPO_URL}" "refs/heads/${BRANCH}")"; then
            break
        fi
        [[ "${attempt}" -lt 3 ]] && sleep "$((attempt * 2))"
    done
    EXPECTED_COMMIT="$(printf '%s\n' "${remote_line}" | awk 'NR == 1 {print $1}')"
    [[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || {
        echo "could not resolve branch ${BRANCH} to exactly one commit" >&2
        exit 1
    }
    echo "Resolved ${BRANCH} once: ${EXPECTED_COMMIT}"
fi

mkdir -p "$(dirname "${PROJECT_DIR}")"
if [[ -e "${PROJECT_DIR}" && ! -d "${PROJECT_DIR}/.git" ]]; then
    echo "project path exists but is not a git checkout: ${PROJECT_DIR}" >&2
    exit 1
fi
if [[ ! -d "${PROJECT_DIR}/.git" ]]; then
    clone_staging="${PROJECT_DIR}.clone-staging.$$"
    [[ ! -e "${clone_staging}" ]] || {
        echo "clone staging path already exists: ${clone_staging}" >&2
        exit 1
    }
    timeout --signal=TERM --kill-after=1m 30m \
        git -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=60 \
        clone --filter=blob:none --no-checkout "${REPO_URL}" "${clone_staging}"
    mv "${clone_staging}" "${PROJECT_DIR}"
fi

cd "${PROJECT_DIR}"
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
    echo "refusing to change a dirty cloud checkout: ${PROJECT_DIR}" >&2
    exit 1
fi
timeout --signal=TERM --kill-after=1m 15m \
    git -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=60 \
    fetch --no-tags --depth=1 origin "${EXPECTED_COMMIT}"
timeout --signal=TERM --kill-after=30s 5m \
    git checkout --detach "${EXPECTED_COMMIT}"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || {
    echo "checked-out commit does not match the resolved revision" >&2
    exit 1
}

init_args=(
    --project-dir "${PROJECT_DIR}"
    --expected-commit "${EXPECTED_COMMIT}"
)
[[ "${ALLOW_GUEST_SHUTDOWN}" == "yes" ]] && init_args+=(--allow-guest-shutdown)
timeout --signal=TERM --kill-after=1m 10m \
    bash scripts/cloud/init_cloud.sh "${init_args[@]}"
BOOTSTRAP_INITIALIZED="yes"

if [[ "${PHASE}" == "init-only" ]]; then
    echo "Cloud checkout initialized at ${PROJECT_DIR}@${EXPECTED_COMMIT}"
    exit 0
fi
flock -u 8
exec 8>&-
BOOTSTRAP_LOCK_HELD="no"
bash scripts/cloud/start_cpu_prep.sh "${START_ARGS[@]}"
