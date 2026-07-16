#!/usr/bin/env bash
# One non-blocking advisory lock serializes all work on the persistent volume.

_rememr1_flock_path() {
    if [[ -x /usr/bin/flock ]]; then
        printf '%s\n' /usr/bin/flock
    elif [[ -x /bin/flock ]]; then
        printf '%s\n' /bin/flock
    else
        echo "flock is required for cloud execution" >&2
        return 1
    fi
}

verify_cloud_lock_fd() {
    local expected="${1:-${LOCK_FILE:-}}"
    local flock_path descriptor_path expected_path
    [[ -n "${expected}" ]] || return 1
    [[ "${REMEMR1_CLOUD_LOCK_HELD:-no}" == "yes" ]] || return 1
    [[ -e /proc/$$/fd/9 ]] || return 1
    descriptor_path="$(/usr/bin/readlink -- /proc/$$/fd/9)" || return
    descriptor_path="$(/usr/bin/realpath -e -- "${descriptor_path}")" || return
    expected_path="$(/usr/bin/realpath -e -- "${expected}")" || return
    [[ "${descriptor_path}" == "${expected_path}" ]] || return 1
    flock_path="$(_rememr1_flock_path)" || return
    "${flock_path}" -n 9
}

require_cloud_lock() {
    local expected="${LOCK_FILE:-${REMEMR1_LOCK_FILE:-}}"
    if [[ -n "${LOCK_FILE:-}" && -n "${REMEMR1_LOCK_FILE:-}" && \
          "${LOCK_FILE}" != "${REMEMR1_LOCK_FILE}" ]]; then
        echo "LOCK_FILE and REMEMR1_LOCK_FILE disagree" >&2
        return 1
    fi
    if ! verify_cloud_lock_fd "${expected}"; then
        echo "the current process does not hold the cloud lock on FD 9" >&2
        return 1
    fi
}

acquire_cloud_lock() {
    if [[ -z "${LOCK_FILE:-}" || "${LOCK_FILE}" != /* ]]; then
        echo "LOCK_FILE must be an absolute path" >&2
        return 2
    fi
    local flock_path
    flock_path="$(_rememr1_flock_path)" || return
    mkdir -p -- "$(dirname -- "${LOCK_FILE}")" || return
    if [[ "${REMEMR1_CLOUD_LOCK_HELD:-no}" == "yes" ]]; then
        if ! verify_cloud_lock_fd "${LOCK_FILE}"; then
            echo "inherited cloud lock descriptor is invalid" >&2
            return 1
        fi
    else
        exec 9>>"${LOCK_FILE}"
        chmod 0600 -- "${LOCK_FILE}" || {
            exec 9>&-
            return 1
        }
        if ! "${flock_path}" -n 9; then
            exec 9>&-
            echo "cloud workspace lock is busy: ${LOCK_FILE}" >&2
            return 75
        fi
        REMEMR1_CLOUD_LOCK_HELD=yes
        export REMEMR1_CLOUD_LOCK_HELD
    fi
    REMEMR1_LOCK_FILE="${LOCK_FILE}"
    export REMEMR1_LOCK_FILE
    if [[ -n "${REMEMR1_LAUNCHER_DIR:-}" ]]; then
        atomic_write "${REMEMR1_LAUNCHER_DIR}/lock-acquired" \
            "lock_file=$(/usr/bin/realpath -e -- "${LOCK_FILE}") pid=$$" || return
    fi
}
