#!/usr/bin/env bash
# Fail-closed guest shutdown authorization and dispatch.

_shutdown_reject() {
    echo "[shutdown] safety gate rejected shutdown: $*" >&2
    return 1
}

_shutdown_autodl_wrapper_sha256() {
    if [[ $# -ne 1 ]]; then
        _shutdown_reject "AutoDL shutdown wrapper path is required"
        return
    fi
    local path="$1" mode
    [[ "${path}" == /usr/bin/shutdown && -f "${path}" && ! -L "${path}" && \
       -x "${path}" ]] || \
        _shutdown_reject "AutoDL shutdown wrapper is missing or unsafe" || return
    [[ "$(/usr/bin/stat -c '%u' -- "${path}")" == 0 ]] || \
        _shutdown_reject "AutoDL shutdown wrapper must be root-owned" || return
    mode="$(/usr/bin/stat -c '%a' -- "${path}")" || return
    [[ "${mode}" =~ ^[0-7]{3}$ ]] || \
        _shutdown_reject "AutoDL shutdown wrapper mode is invalid" || return
    (( (8#${mode} & 8#022) == 0 )) || \
        _shutdown_reject "AutoDL shutdown wrapper is group/world writable" || return
    [[ -f /etc/autodl-init && ! -L /etc/autodl-init && \
       "$(/usr/bin/stat -c '%u' -- /etc/autodl-init)" == 0 ]] || \
        _shutdown_reject "AutoDL guest marker is missing or unsafe" || return
    sha256_file "${path}"
}

_shutdown_verify_capability_backend() {
    local observed
    [[ "${_REMEMR1_CAPABILITY[shutdown_backend]}" == autodl-wrapper-v1 && \
       "${_REMEMR1_CAPABILITY[shutdown_backend_path]}" == /usr/bin/shutdown && \
       "${_REMEMR1_CAPABILITY[shutdown_backend_sha256]}" =~ ^[0-9a-f]{64}$ ]] || \
        _shutdown_reject "capability does not select the AutoDL shutdown wrapper" || return
    observed="$(_shutdown_autodl_wrapper_sha256 \
        "${_REMEMR1_CAPABILITY[shutdown_backend_path]}")" || return
    [[ "${observed}" == "${_REMEMR1_CAPABILITY[shutdown_backend_sha256]}" ]] || \
        _shutdown_reject "AutoDL shutdown wrapper digest changed" || return
}

_shutdown_parse_capability() {
    local capability_file="$1"
    declare -gA _REMEMR1_CAPABILITY=()
    local line key value
    while IFS= read -r line || [[ -n "${line}" ]]; do
        [[ -n "${line}" && "${line}" == *=* ]] || \
            _shutdown_reject "malformed capability file" || return
        key="${line%%=*}"
        value="${line#*=}"
        case "${key}" in
            schema_version|project_dir|persist_root|expected_commit|lock_file|allow_guest_shutdown|shutdown_backend|shutdown_backend_path|shutdown_backend_sha256) ;;
            *) _shutdown_reject "unknown capability key: ${key}" || return ;;
        esac
        if [[ -n "${_REMEMR1_CAPABILITY[${key}]+set}" ]]; then
            _shutdown_reject "duplicate capability key: ${key}" || return
        fi
        _REMEMR1_CAPABILITY["${key}"]="${value}"
    done < "${capability_file}"
    local required
    for required in schema_version project_dir persist_root expected_commit lock_file \
        allow_guest_shutdown shutdown_backend shutdown_backend_path \
        shutdown_backend_sha256; do
        [[ -n "${_REMEMR1_CAPABILITY[${required}]+set}" ]] || \
            _shutdown_reject "missing capability key: ${required}" || return
    done
}

_shutdown_verify_terminal_marker() {
    local launcher="$1"
    local exit_code="$2"
    local name marker_count=0
    for name in .success .failed .scientific-stop .capacity-stop; do
        if [[ -e "${launcher}/${name}" || -L "${launcher}/${name}" ]]; then
            marker_count=$((marker_count + 1))
        fi
    done
    [[ "${marker_count}" -eq 1 ]] || {
        _shutdown_reject "launcher terminal marker is ambiguous"
        return
    }
    case "${exit_code}" in
        0) name=.success ;;
        42) name=.scientific-stop ;;
        43) name=.capacity-stop ;;
        *) name=.failed ;;
    esac
    [[ -f "${launcher}/${name}" && ! -L "${launcher}/${name}" && \
       "$(<"${launcher}/${name}")" == "${exit_code}" ]] || {
        _shutdown_reject "launcher terminal marker does not match exit-code"
        return
    }
}

verify_guest_shutdown_preflight() {
    rememr1_require_cloud_env || return
    rememr1_validate_test_mode || return
    [[ "${REMEMR1_TEST_MODE:-no}" != "yes" ]] || \
        _shutdown_reject "test mode cannot authorize production shutdown" || return
    local autodl_root project persist launcher_root capability lock head dirty
    [[ "$(/usr/bin/uname -s)" == "Linux" ]] || \
        _shutdown_reject "host is not Linux" || return
    if /usr/bin/grep -Eiq '(microsoft|wsl)' /proc/version 2>/dev/null || \
       [[ -n "${WSL_DISTRO_NAME:-}" ]]; then
        _shutdown_reject "WSL is never authorized to power off the host" || return
    fi
    [[ "${EUID}" -eq 0 ]] || _shutdown_reject "EUID must be 0" || return
    local executable
    for executable in /usr/bin/bash /usr/bin/env /usr/bin/findmnt /usr/bin/git /usr/bin/realpath \
        /usr/bin/stat /usr/bin/sync /usr/bin/timeout; do
        [[ -x "${executable}" ]] || \
            _shutdown_reject "required shutdown executable is missing: ${executable}" || return
    done
    autodl_root="$(/usr/bin/realpath -e -- /root/autodl-tmp)" || return
    project="$(/usr/bin/realpath -e -- "${REMEMR1_PROJECT_DIR}")" || return
    persist="$(/usr/bin/realpath -e -- "${PERSIST_ROOT}")" || return
    launcher_root="$(/usr/bin/realpath -e -- "${LAUNCHER_ROOT}")" || return
    capability="$(/usr/bin/realpath -e -- "${CAPABILITY_FILE}")" || return
    lock="$(/usr/bin/realpath -e -- "${LOCK_FILE}")" || return
    [[ "${project}" =~ ^/root/autodl-tmp/[^/]+$ ]] || \
        _shutdown_reject "project must be a fixed direct child of /root/autodl-tmp" || return
    rememr1_path_is_within "${persist}" "${autodl_root}" || \
        _shutdown_reject "persistent root escaped /root/autodl-tmp" || return
    rememr1_path_is_within "${launcher_root}" "${persist}" || \
        _shutdown_reject "launcher root is outside persistent storage" || return
    rememr1_path_is_within "${capability}" "${persist}" || \
        _shutdown_reject "capability is outside persistent storage" || return
    rememr1_path_is_within "${lock}" "${persist}" || \
        _shutdown_reject "lock is outside persistent storage" || return
    rememr1_verify_persistent_mount "${persist}" || \
        _shutdown_reject "persistent root is not an independent durable mount" || return
    [[ -f "${capability}" && ! -L "${CAPABILITY_FILE}" ]] || \
        _shutdown_reject "capability must be a regular non-symlink" || return
    [[ "$(/usr/bin/stat -c '%u' -- "${capability}")" == "0" && \
       "$(/usr/bin/stat -c '%a' -- "${capability}")" == "600" ]] || \
        _shutdown_reject "capability must be root-owned mode 0600" || return
    _shutdown_parse_capability "${capability}" || return
    [[ "${_REMEMR1_CAPABILITY[schema_version]}" == "2" && \
       "${_REMEMR1_CAPABILITY[allow_guest_shutdown]}" == "yes" ]] || \
        _shutdown_reject "capability does not authorize guest shutdown" || return
    _shutdown_verify_capability_backend || return
    [[ "$(/usr/bin/realpath -e -- "${_REMEMR1_CAPABILITY[project_dir]}")" == "${project}" && \
       "$(/usr/bin/realpath -e -- "${_REMEMR1_CAPABILITY[persist_root]}")" == "${persist}" && \
       "$(/usr/bin/realpath -e -- "${_REMEMR1_CAPABILITY[lock_file]}")" == "${lock}" && \
       "${_REMEMR1_CAPABILITY[expected_commit]}" == "${EXPECTED_COMMIT}" ]] || \
        _shutdown_reject "capability identity mismatch" || return
    head="$(/usr/bin/git -C "${project}" rev-parse HEAD 2>/dev/null)" || return
    dirty="$(/usr/bin/git -C "${project}" status --porcelain --untracked-files=all)" || return
    [[ "${head}" == "${EXPECTED_COMMIT}" && -z "${dirty}" ]] || \
        _shutdown_reject "checkout is not the pinned clean commit" || return
}

verify_guest_shutdown_authorization() {
    if [[ $# -ne 1 ]]; then
        _shutdown_reject "launcher directory argument is required"
        return
    fi
    rememr1_require_cloud_env || return
    local launcher_dir="$1"
    local autodl_root project persist launcher_root launcher capability lock
    local head dirty capability_uid capability_mode

    [[ "$(/usr/bin/uname -s)" == "Linux" ]] || \
        _shutdown_reject "host is not Linux" || return
    if /usr/bin/grep -Eiq '(microsoft|wsl)' /proc/version 2>/dev/null || \
       [[ -n "${WSL_DISTRO_NAME:-}" ]]; then
        _shutdown_reject "WSL is never authorized to power off the host" || return
    fi
    [[ "${EUID}" -eq 0 ]] || _shutdown_reject "EUID must be 0" || return
    local executable
    for executable in /usr/bin/bash /usr/bin/env /usr/bin/findmnt /usr/bin/git /usr/bin/realpath \
        /usr/bin/stat /usr/bin/sync /usr/bin/timeout; do
        [[ -x "${executable}" ]] || \
            _shutdown_reject "required shutdown executable is missing: ${executable}" || return
    done
    autodl_root="$(/usr/bin/realpath -e -- /root/autodl-tmp)" || \
        _shutdown_reject "AutoDL persistent mount path does not exist" || return

    project="$(/usr/bin/realpath -e -- "${REMEMR1_PROJECT_DIR}")" || \
        _shutdown_reject "project directory does not exist" || return
    persist="$(/usr/bin/realpath -e -- "${PERSIST_ROOT}")" || \
        _shutdown_reject "persistent root does not exist" || return
    launcher_root="$(/usr/bin/realpath -e -- "${LAUNCHER_ROOT}")" || \
        _shutdown_reject "launcher root does not exist" || return
    launcher="$(/usr/bin/realpath -e -- "${launcher_dir}")" || \
        _shutdown_reject "launcher directory does not exist" || return
    capability="$(/usr/bin/realpath -e -- "${CAPABILITY_FILE}")" || \
        _shutdown_reject "capability file does not exist" || return
    lock="$(/usr/bin/realpath -e -- "${LOCK_FILE}")" || \
        _shutdown_reject "lock file does not exist" || return

    [[ "${project}" =~ ^/root/autodl-tmp/[^/]+$ ]] || \
        _shutdown_reject "project must be a fixed direct child of /root/autodl-tmp" || return
    [[ "${persist}" != "/" ]] || _shutdown_reject "persistent root cannot be /" || return
    rememr1_path_is_within "${persist}" "${autodl_root}" || \
        _shutdown_reject "persistent root escaped /root/autodl-tmp" || return
    rememr1_path_is_within "${launcher_root}" "${persist}" || \
        _shutdown_reject "launcher root is outside persistent storage" || return
    rememr1_path_is_within "${launcher}" "${launcher_root}" || \
        _shutdown_reject "launcher status is outside launcher root" || return
    rememr1_path_is_within "${capability}" "${persist}" || \
        _shutdown_reject "capability is outside persistent storage" || return
    rememr1_path_is_within "${lock}" "${persist}" || \
        _shutdown_reject "lock is outside persistent storage" || return

    [[ -x /usr/bin/findmnt ]] || _shutdown_reject "findmnt is unavailable" || return
    rememr1_verify_persistent_mount "${persist}" || \
        _shutdown_reject "persistent root is not an independent durable mount" || return

    [[ -f "${capability}" && ! -L "${CAPABILITY_FILE}" ]] || \
        _shutdown_reject "capability must be a regular non-symlink" || return
    capability_uid="$(/usr/bin/stat -c '%u' -- "${capability}")" || return
    capability_mode="$(/usr/bin/stat -c '%a' -- "${capability}")" || return
    [[ "${capability_uid}" == "0" && "${capability_mode}" == "600" ]] || \
        _shutdown_reject "capability must be root-owned mode 0600" || return
    _shutdown_parse_capability "${capability}" || return
    [[ "${_REMEMR1_CAPABILITY[schema_version]}" == "2" ]] || \
        _shutdown_reject "unsupported capability schema" || return
    [[ "${_REMEMR1_CAPABILITY[allow_guest_shutdown]}" == "yes" ]] || \
        _shutdown_reject "capability does not allow guest shutdown" || return
    _shutdown_verify_capability_backend || return
    [[ "$(/usr/bin/realpath -e -- "${_REMEMR1_CAPABILITY[project_dir]}")" == "${project}" ]] || \
        _shutdown_reject "capability project mismatch" || return
    [[ "$(/usr/bin/realpath -e -- "${_REMEMR1_CAPABILITY[persist_root]}")" == "${persist}" ]] || \
        _shutdown_reject "capability persistent root mismatch" || return
    [[ "$(/usr/bin/realpath -e -- "${_REMEMR1_CAPABILITY[lock_file]}")" == "${lock}" ]] || \
        _shutdown_reject "capability lock mismatch" || return
    [[ "${_REMEMR1_CAPABILITY[expected_commit]}" == "${EXPECTED_COMMIT}" ]] || \
        _shutdown_reject "capability commit mismatch" || return

    [[ -x /usr/bin/git ]] || _shutdown_reject "git is unavailable" || return
    head="$(/usr/bin/git -C "${project}" rev-parse HEAD 2>/dev/null)" || \
        _shutdown_reject "cannot read project HEAD" || return
    [[ "${head}" == "${EXPECTED_COMMIT}" ]] || \
        _shutdown_reject "project HEAD differs from EXPECTED_COMMIT" || return
    dirty="$(/usr/bin/git -C "${project}" status --porcelain --untracked-files=normal)" || \
        _shutdown_reject "cannot inspect project worktree" || return
    [[ -z "${dirty}" ]] || _shutdown_reject "project worktree is dirty" || return
    verify_cloud_lock_fd "${lock}" || \
        _shutdown_reject "worker does not hold the configured lock on FD 9" || return

    [[ -f "${launcher}/shutdown-safe" && -f "${launcher}/exit-code" && \
       -f "${launcher}/terminal.json" && -f "${launcher}/lock-acquired" ]] || \
        _shutdown_reject "durable terminal launcher state is incomplete" || return
    [[ ! -e "${launcher}/.running" && ! -e "${launcher}/.starting" ]] || \
        _shutdown_reject "launcher still has a nonterminal marker" || return
    local exit_code
    exit_code="$(<"${launcher}/exit-code")"
    [[ "${exit_code}" =~ ^[0-9]+$ && "${exit_code}" -le 255 ]] || \
        _shutdown_reject "launcher exit-code is invalid" || return
    _shutdown_verify_terminal_marker "${launcher}" "${exit_code}" || return
    [[ -f "${launcher}/launcher.log" && ! -L "${launcher}/launcher.log" ]] || \
        _shutdown_reject "launcher log is missing or unsafe" || return
    /usr/bin/grep -Fq "terminal-state-published exit_code=${exit_code}" \
        "${launcher}/launcher.log" || \
        _shutdown_reject "launcher log lacks its terminal sentinel" || return
}

_record_shutdown_skipped() {
    local launcher_dir="$1"
    local reason="$2"
    rm -f -- "${launcher_dir}/shutdown-backend" \
        "${launcher_dir}/shutdown-requested" 2>/dev/null || true
    atomic_write "${launcher_dir}/shutdown-skipped" "${reason}" 2>/dev/null || true
    rememr1_sync_all >/dev/null 2>&1 || true
}

_dispatch_guest_shutdown_backend() {
    _shutdown_verify_capability_backend || return 2
    if /usr/bin/timeout --signal=TERM --kill-after=30s 2m \
        /usr/bin/env -i PATH=/usr/bin:/bin HOME=/root \
        /usr/bin/bash --noprofile --norc /usr/bin/shutdown; then
        return 0
    fi
    return 1
}

request_guest_shutdown() {
    if [[ $# -ne 3 ]]; then
        echo "usage: request_guest_shutdown LAUNCHER_DIR PHASE EXIT_CODE" >&2
        return 2
    fi
    local launcher_dir="$1"
    local phase="$2"
    local exit_code="$3"
    if [[ "${REMEMR1_TEST_MODE:-no}" == "yes" ]]; then
        rememr1_validate_test_mode || return
        rememr1_test_event \
            "shutdown-request phase=${phase} exit_code=${exit_code} launcher_dir=${launcher_dir}"
        return 0
    fi
    rememr1_validate_test_mode || return
    if ! verify_guest_shutdown_authorization "${launcher_dir}"; then
        _record_shutdown_skipped "${launcher_dir}" "authorization-failed"
        return 1
    fi
    if ! rememr1_sync_all; then
        _record_shutdown_skipped "${launcher_dir}" "pre-dispatch-sync-failed"
        return 1
    fi
    if ! atomic_write "${launcher_dir}/shutdown-backend" \
        "${_REMEMR1_CAPABILITY[shutdown_backend]}"; then
        _record_shutdown_skipped "${launcher_dir}" "backend-marker-write-failed"
        return 1
    fi
    if ! atomic_write "${launcher_dir}/shutdown-requested" \
        "$(rememr1_utc_now) phase=${phase} exit_code=${exit_code}"; then
        _record_shutdown_skipped "${launcher_dir}" "request-marker-write-failed"
        return 1
    fi
    if ! rememr1_sync_all; then
        _record_shutdown_skipped "${launcher_dir}" "request-marker-sync-failed"
        return 1
    fi
    echo "[shutdown] terminal state is durable; requesting guest shutdown"
    local dispatch_rc
    if _dispatch_guest_shutdown_backend; then
        return 0
    else
        dispatch_rc=$?
    fi
    if [[ "${dispatch_rc}" -eq 2 ]]; then
        _record_shutdown_skipped "${launcher_dir}" "backend-revalidation-failed"
        return 1
    fi
    atomic_write "${launcher_dir}/shutdown-failed" "$(rememr1_utc_now)" || true
    rememr1_sync_all || true
    echo "[shutdown] guest shutdown failed; stop the instance in the provider console" >&2
    return 1
}
