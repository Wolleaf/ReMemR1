#!/usr/bin/env bash
# Shared primitives for durable cloud launcher state.

rememr1_utc_now() {
    /usr/bin/date -u +%Y-%m-%dT%H:%M:%SZ
}

rememr1_json_escape() {
    local value="${1-}"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    value="${value//$'\n'/\\n}"
    value="${value//$'\r'/\\r}"
    value="${value//$'\t'/\\t}"
    printf '%s' "${value}"
}

atomic_write() {
    if [[ $# -lt 1 || $# -gt 2 ]]; then
        echo "usage: atomic_write PATH [VALUE]" >&2
        return 2
    fi
    local target="$1"
    local parent tmp
    parent="$(dirname -- "${target}")"
    if [[ ! -d "${parent}" ]]; then
        echo "atomic_write parent does not exist: ${parent}" >&2
        return 1
    fi
    tmp="${target}.tmp.$$-${RANDOM:-0}"
    if [[ $# -eq 2 ]]; then
        if ! printf '%s\n' "$2" > "${tmp}"; then
            rm -f -- "${tmp}"
            return 1
        fi
    elif ! cat > "${tmp}"; then
        rm -f -- "${tmp}"
        return 1
    fi
    if ! mv -f -- "${tmp}" "${target}"; then
        rm -f -- "${tmp}"
        return 1
    fi
}

atomic_write_text() {
    if [[ $# -ne 2 ]]; then
        echo "usage: atomic_write_text PATH VALUE" >&2
        return 2
    fi
    atomic_write "$1" "$2"
}

run_logged() {
    if [[ $# -lt 2 ]]; then
        echo "usage: run_logged LOG COMMAND [ARG ...]" >&2
        return 2
    fi
    local log_file="$1"
    shift
    local had_errexit=no
    local -a pipeline_status
    local rc
    [[ "$-" == *e* ]] && had_errexit=yes
    set +e
    "$@" 2>&1 | /usr/bin/tee -a -- "${log_file}"
    pipeline_status=("${PIPESTATUS[@]}")
    if [[ "${pipeline_status[0]}" -ne 0 ]]; then
        rc="${pipeline_status[0]}"
    else
        rc="${pipeline_status[1]}"
    fi
    [[ "${had_errexit}" == "yes" ]] && set -e
    return "${rc}"
}

sha256_file() {
    if [[ $# -ne 1 || ! -f "$1" ]]; then
        echo "sha256_file requires a regular file" >&2
        return 2
    fi
    local output digest
    if [[ -x /usr/bin/sha256sum ]]; then
        output="$(/usr/bin/sha256sum -- "$1")" || return
    elif [[ -x /bin/sha256sum ]]; then
        output="$(/bin/sha256sum -- "$1")" || return
    elif [[ -x /usr/bin/shasum ]]; then
        output="$(/usr/bin/shasum -a 256 -- "$1")" || return
    else
        echo "no SHA-256 implementation is available" >&2
        return 1
    fi
    digest="${output%%[[:space:]]*}"
    if [[ ! "${digest}" =~ ^[0-9a-f]{64}$ ]]; then
        echo "invalid SHA-256 output" >&2
        return 1
    fi
    printf '%s\n' "${digest}"
}

rememr1_realpath_existing() {
    /usr/bin/realpath -e -- "$1"
}

rememr1_realpath_maybe() {
    /usr/bin/realpath -m -- "$1"
}

rememr1_path_is_within() {
    if [[ $# -ne 2 ]]; then
        return 2
    fi
    local child root
    child="$(rememr1_realpath_maybe "$1")" || return
    root="$(rememr1_realpath_maybe "$2")" || return
    if [[ "${root}" == "/" ]]; then
        [[ "${child}" == /* ]]
        return
    fi
    [[ "${child}" == "${root}" || "${child}" == "${root}/"* ]]
}

rememr1_verify_persistent_mount() {
    if [[ $# -ne 1 || ! -x /usr/bin/findmnt || ! -x /usr/bin/realpath ]]; then
        echo "persistent mount verification requires one path, findmnt, and realpath" >&2
        return 2
    fi
    local path target source fstype major_minor root_source root_major_minor
    path="$(/usr/bin/realpath -e -- "$1")" || return
    target="$(/usr/bin/findmnt -n -o TARGET --target "${path}")" || return
    target="$(/usr/bin/realpath -e -- "${target}")" || return
    [[ "${target}" == "/root/autodl-tmp" ]] || {
        echo "persistent data must be governed by the /root/autodl-tmp mount" >&2
        return 1
    }
    source="$(/usr/bin/findmnt -n -o SOURCE --target "${path}")" || return
    fstype="$(/usr/bin/findmnt -n -o FSTYPE --target "${path}")" || return
    major_minor="$(/usr/bin/findmnt -n -o MAJ:MIN --target "${path}")" || return
    root_source="$(/usr/bin/findmnt -n -o SOURCE --target /)" || return
    root_major_minor="$(/usr/bin/findmnt -n -o MAJ:MIN --target /)" || return
    case "${fstype,,}" in
        overlay|tmpfs|ramfs|rootfs|devtmpfs|squashfs)
            echo "ephemeral filesystem is not accepted for persistent data: ${fstype}" >&2
            return 1
            ;;
    esac
    [[ -n "${source}" && -n "${major_minor}" && \
       "${source}" != "${root_source}" && \
       "${major_minor}" != "${root_major_minor}" ]] || {
        echo "/root/autodl-tmp shares the host root backing device" >&2
        return 1
    }
    return 0
}

rememr1_validate_test_mode() {
    case "${REMEMR1_TEST_MODE:-no}" in
        no|"")
            if [[ -n "${REMEMR1_TEST_SHUTDOWN_LOG:-}" ]]; then
                echo "REMEMR1_TEST_SHUTDOWN_LOG requires REMEMR1_TEST_MODE=yes" >&2
                return 1
            fi
            ;;
        yes)
            if [[ -z "${REMEMR1_TEST_SHUTDOWN_LOG:-}" || \
                  "${REMEMR1_TEST_SHUTDOWN_LOG}" != /* ]]; then
                echo "test mode requires an absolute REMEMR1_TEST_SHUTDOWN_LOG" >&2
                return 1
            fi
            local candidate canonical_candidate
            for candidate in "${REMEMR1_PROJECT_DIR:-}" "${PERSIST_ROOT:-${REMEMR1_PERSIST_ROOT:-}}"; do
                case "${candidate}" in
                    /root/autodl-tmp|/root/autodl-tmp/*)
                        echo "test mode is forbidden for production AutoDL paths" >&2
                        return 1
                        ;;
                esac
                [[ -n "${candidate}" ]] || continue
                canonical_candidate="$(rememr1_realpath_maybe "${candidate}")" || {
                    echo "test mode could not canonicalize a configured path" >&2
                    return 1
                }
                case "${canonical_candidate}" in
                    /root/autodl-tmp|/root/autodl-tmp/*)
                        echo "test mode is forbidden for production AutoDL paths" >&2
                        return 1
                        ;;
                esac
            done
            ;;
        *)
            echo "REMEMR1_TEST_MODE must be yes or no" >&2
            return 1
            ;;
    esac
    return 0
}

rememr1_ensure_terminal_reserve() {
    local raw_reserve="${REMEMR1_RESERVE_FILE:-${PERSIST_ROOT}/cloud/terminal-reserve}"
    local reserve persist parent temporary size
    [[ ! -L "${raw_reserve}" ]] || {
        echo "terminal reserve must not be a symlink" >&2
        return 1
    }
    persist="$(rememr1_realpath_existing "${PERSIST_ROOT}")" || return
    reserve="$(rememr1_realpath_maybe "${raw_reserve}")" || return
    rememr1_path_is_within "${reserve}" "${persist}" || {
        echo "terminal reserve escaped persistent storage" >&2
        return 1
    }
    if [[ -e "${reserve}" ]]; then
        [[ -f "${reserve}" && ! -L "${reserve}" ]] || {
            echo "terminal reserve must be a regular non-symlink file" >&2
            return 1
        }
        size="$(/usr/bin/stat -c '%s' -- "${reserve}")" || return
        [[ "${size}" =~ ^[0-9]+$ ]] || return 1
        if [[ "${size}" -ge 8388608 ]]; then
            return 0
        fi
    fi
    for executable in /usr/bin/dd /usr/bin/mv; do
        [[ -x "${executable}" ]] || {
            echo "terminal reserve dependency is missing: ${executable}" >&2
            return 1
        }
    done
    parent="$(dirname -- "${reserve}")"
    mkdir -p -- "${parent}" || return
    temporary="${reserve}.tmp.$$-${RANDOM:-0}"
    if ! /usr/bin/timeout --signal=TERM --kill-after=30s 2m \
        /usr/bin/dd if=/dev/zero of="${temporary}" bs=1M count=8 \
        conv=fsync status=none; then
        rm -f -- "${temporary}"
        return 1
    fi
    chmod 0600 -- "${temporary}" || {
        rm -f -- "${temporary}"
        return 1
    }
    /usr/bin/mv -f -- "${temporary}" "${reserve}" || {
        rm -f -- "${temporary}"
        return 1
    }
}

rememr1_sync_all() {
    if [[ ! -x /usr/bin/timeout || ! -x /usr/bin/sync ]]; then
        echo "bounded sync dependencies are unavailable" >&2
        return 1
    fi
    /usr/bin/timeout --signal=TERM --kill-after=30s 5m /usr/bin/sync
}

rememr1_sync_file() {
    if [[ $# -ne 1 || ! -e "$1" ]]; then
        echo "bounded file sync requires an existing path" >&2
        return 2
    fi
    if [[ ! -x /usr/bin/timeout || ! -x /usr/bin/sync ]]; then
        echo "bounded sync dependencies are unavailable" >&2
        return 1
    fi
    /usr/bin/timeout --signal=TERM --kill-after=30s 2m /usr/bin/sync -f -- "$1"
}

rememr1_test_event() {
    if [[ "${REMEMR1_TEST_MODE:-no}" != "yes" ]]; then
        return 0
    fi
    rememr1_validate_test_mode || return
    printf '%s\t%s\n' "$(rememr1_utc_now)" "$*" >> "${REMEMR1_TEST_SHUTDOWN_LOG}" || return
    return 0
}

rememr1_load_cloud_env() {
    if [[ $# -ne 1 ]]; then
        echo "usage: rememr1_load_cloud_env FILE" >&2
        return 2
    fi
    local env_file="$1"
    if [[ ! -f "${env_file}" || -L "${env_file}" ]]; then
        echo "cloud environment file must be a regular non-symlink: ${env_file}" >&2
        return 1
    fi
    if [[ "${REMEMR1_TEST_MODE:-no}" != "yes" ]]; then
        if [[ "$('/usr/bin/stat' -c '%u' -- "${env_file}")" != "0" || \
              "$('/usr/bin/stat' -c '%a' -- "${env_file}")" != "600" ]]; then
            echo "cloud environment file must be root-owned mode 0600" >&2
            return 1
        fi
    fi
    local had_allexport=no
    [[ "$-" == *a* ]] && had_allexport=yes
    set -a
    # The production ownership check makes sourcing this root-authored file safe.
    if ! source "${env_file}"; then
        [[ "${had_allexport}" == "no" ]] && set +a
        echo "failed to load cloud environment: ${env_file}" >&2
        return 1
    fi
    [[ "${had_allexport}" == "no" ]] && set +a
}

rememr1_require_cloud_env() {
    local canonical alias canonical_value alias_value
    for canonical in PERSIST_ROOT EXPECTED_COMMIT CAPABILITY_FILE LOCK_FILE LAUNCHER_ROOT; do
        alias="REMEMR1_${canonical}"
        canonical_value="${!canonical:-}"
        alias_value="${!alias:-}"
        if [[ -n "${canonical_value}" && -n "${alias_value}" && \
              "${canonical_value}" != "${alias_value}" ]]; then
            echo "${canonical} and ${alias} disagree" >&2
            return 1
        fi
        if [[ -z "${canonical_value}" && -n "${alias_value}" ]]; then
            printf -v "${canonical}" '%s' "${alias_value}"
        elif [[ -n "${canonical_value}" && -z "${alias_value}" ]]; then
            printf -v "${alias}" '%s' "${canonical_value}"
        fi
        export "${canonical}" "${alias}"
    done

    local name
    for name in REMEMR1_PROJECT_DIR PERSIST_ROOT EXPECTED_COMMIT CAPABILITY_FILE \
        LOCK_FILE LAUNCHER_ROOT; do
        if [[ -z "${!name:-}" ]]; then
            echo "missing required cloud setting: ${name}" >&2
            return 1
        fi
    done
    if [[ ! "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
        echo "EXPECTED_COMMIT must be a full lowercase commit SHA" >&2
        return 1
    fi
    for name in REMEMR1_PROJECT_DIR PERSIST_ROOT CAPABILITY_FILE LOCK_FILE LAUNCHER_ROOT; do
        if [[ "${!name}" != /* ]]; then
            echo "${name} must be an absolute path" >&2
            return 1
        fi
    done
}
