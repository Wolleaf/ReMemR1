#!/usr/bin/env bash
# Run one resumable cloud phase DAG under the inherited cloud lock.
set -euo pipefail

PHASE=""
RETRY_FAILED_STAGE="no"
DRY_RUN="no"
OFFLOAD_PROFILE=""
R1_APPROVAL=""
BUDGET_PROJECTION=""
BUDGET_PROJECTION_FILE_SHA256=""

usage() {
    echo "Usage: $0 --phase PHASE [--offload-profile r0|r1] [OPTIONS]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase) PHASE="$2"; shift 2 ;;
        --retry-failed-stage) RETRY_FAILED_STAGE="yes"; shift ;;
        --dry-run) DRY_RUN="yes"; shift ;;
        --offload-profile) OFFLOAD_PROFILE="${2:-}"; shift 2 ;;
        --r1-approval) R1_APPROVAL="${2:-}"; shift 2 ;;
        --budget-projection) BUDGET_PROJECTION="${2:-}"; shift 2 ;;
        --budget-projection-file-sha256)
            BUDGET_PROJECTION_FILE_SHA256="${2:-}"
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done
case "${PHASE}" in
    cpu|gpu-gates|gpu-capacity|gpu-bc40|gpu-bc80|gpu-export) ;;
    *) usage; exit 2 ;;
esac
if [[ "${PHASE}" == "gpu-capacity" ]]; then
    [[ "${OFFLOAD_PROFILE}" == "r0" || "${OFFLOAD_PROFILE}" == "r1" ]] || {
        echo "gpu-capacity requires --offload-profile r0 or r1" >&2
        exit 2
    }
    [[ "${OFFLOAD_PROFILE}" != "r1" || -n "${R1_APPROVAL}" ]] || {
        echo "R1 capacity requires --r1-approval" >&2
        exit 2
    }
    [[ "${OFFLOAD_PROFILE}" != "r1" || -n "${BUDGET_PROJECTION}" ]] || {
        echo "R1 capacity requires --budget-projection" >&2
        exit 2
    }
    [[ "${OFFLOAD_PROFILE}" != "r0" || \
       ( -z "${R1_APPROVAL}" && -z "${BUDGET_PROJECTION}" ) ]] || {
        echo "R0 capacity must not carry R1 approval or budget evidence" >&2
        exit 2
    }
elif [[ "${PHASE}" == "gpu-bc40" || "${PHASE}" == "gpu-bc80" ]]; then
    [[ -n "${BUDGET_PROJECTION}" ]] || {
        echo "${PHASE} requires --budget-projection" >&2
        exit 2
    }
    [[ -z "${OFFLOAD_PROFILE}" && -z "${R1_APPROVAL}" ]] || {
        echo "offload and approval arguments are only valid for gpu-capacity" >&2
        exit 2
    }
elif [[ -n "${OFFLOAD_PROFILE}" || -n "${R1_APPROVAL}" || \
        -n "${BUDGET_PROJECTION}" ]]; then
    echo "offload and approval arguments are only valid for gpu-capacity" >&2
    exit 2
fi
if [[ -n "${BUDGET_PROJECTION}" ]]; then
    [[ "${BUDGET_PROJECTION_FILE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
        echo "budget projection requires its launcher-bound file SHA-256" >&2
        exit 2
    }
elif [[ -n "${BUDGET_PROJECTION_FILE_SHA256}" ]]; then
    echo "budget projection SHA-256 was provided without a projection" >&2
    exit 2
fi

CLOUD_ENV="${REMEMR1_CLOUD_ENV:-/root/autodl-tmp/rememr1-cloud.env}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/runtime.sh"
rememr1_load_cloud_env "${CLOUD_ENV}"
rememr1_require_cloud_env
cd "${REMEMR1_PROJECT_DIR}"
source "${SCRIPT_DIR}/lib/lock.sh"
require_cloud_lock

PERSIST_REAL="$(rememr1_realpath_existing "${PERSIST_ROOT}")"
if [[ -n "${R1_APPROVAL}" ]]; then
    [[ "${R1_APPROVAL}" == /* && -f "${R1_APPROVAL}" && ! -L "${R1_APPROVAL}" ]] || {
        echo "R1 approval is missing or unsafe" >&2
        exit 1
    }
    R1_APPROVAL="$(rememr1_realpath_existing "${R1_APPROVAL}")"
    rememr1_path_is_within "${R1_APPROVAL}" "${PERSIST_REAL}" || {
        echo "R1 approval escaped persistent storage" >&2
        exit 1
    }
fi
if [[ -n "${BUDGET_PROJECTION}" ]]; then
    [[ "${BUDGET_PROJECTION}" == /* && -f "${BUDGET_PROJECTION}" && \
       ! -L "${BUDGET_PROJECTION}" ]] || {
        echo "budget projection is missing or unsafe" >&2
        exit 1
    }
    BUDGET_PROJECTION="$(rememr1_realpath_existing "${BUDGET_PROJECTION}")"
    rememr1_path_is_within "${BUDGET_PROJECTION}" "${PERSIST_REAL}" || {
        echo "budget projection escaped persistent storage" >&2
        exit 1
    }
    observed_budget_file_sha256="$(sha256sum "${BUDGET_PROJECTION}" | awk '{print $1}')"
    [[ "${observed_budget_file_sha256}" == "${BUDGET_PROJECTION_FILE_SHA256}" ]] || {
        echo "budget projection changed after launcher admission" >&2
        exit 1
    }
fi
CLOUD_ROOT_RAW="${PERSIST_ROOT}/cloud"
[[ -d "${CLOUD_ROOT_RAW}" && ! -L "${CLOUD_ROOT_RAW}" ]] || {
    echo "cloud state root must be a regular directory" >&2
    exit 1
}
CLOUD_ROOT_REAL="$(rememr1_realpath_existing "${CLOUD_ROOT_RAW}")"
rememr1_path_is_within "${CLOUD_ROOT_REAL}" "${PERSIST_REAL}" || {
    echo "cloud state root escaped persistent storage" >&2
    exit 1
}
[[ -n "${REMEMR1_PIPELINE_ROOT:-}" && "${REMEMR1_PIPELINE_ROOT}" == /* && \
   -d "${REMEMR1_PIPELINE_ROOT}" && ! -L "${REMEMR1_PIPELINE_ROOT}" ]] || {
    echo "pipeline root must be an existing non-symlink directory" >&2
    exit 1
}
PIPELINE_ROOT_REAL="$(rememr1_realpath_existing "${REMEMR1_PIPELINE_ROOT}")"
[[ "${PIPELINE_ROOT_REAL}" == "${CLOUD_ROOT_REAL}/pipelines" ]] || {
    echo "pipeline root differs from the initialized cloud state" >&2
    exit 1
}
STAGE_RUN_ROOT_RAW="${CLOUD_ROOT_REAL}/runs"
if [[ -e "${STAGE_RUN_ROOT_RAW}" || -L "${STAGE_RUN_ROOT_RAW}" ]]; then
    [[ -d "${STAGE_RUN_ROOT_RAW}" && ! -L "${STAGE_RUN_ROOT_RAW}" ]] || {
        echo "stage run root must be a regular directory" >&2
        exit 1
    }
else
    mkdir -- "${STAGE_RUN_ROOT_RAW}"
fi
STAGE_RUN_ROOT="$(rememr1_realpath_existing "${STAGE_RUN_ROOT_RAW}")"
[[ "${STAGE_RUN_ROOT}" == "${CLOUD_ROOT_REAL}/runs" ]] || {
    echo "stage run root escaped the initialized cloud state" >&2
    exit 1
}
timeout --verbose --signal=TERM --kill-after=30s 5m \
    bash scripts/cloud/setup_git.sh

BUDGET_PROJECTION_SHA256=""
BUDGET_DECISION=""
R1_APPROVAL_MARKER_SHA256=""
if [[ -n "${BUDGET_PROJECTION}" ]]; then
    timeout --verbose --signal=TERM --kill-after=30s 5m \
        "${REMEMR1_ENV_PREFIX}/bin/python" scripts/cloud/cost_gate.py verify \
        --projection "${BUDGET_PROJECTION}"
    mapfile -t budget_fields < <("${REMEMR1_ENV_PREFIX}/bin/python" -c \
        'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")); print(value.get("decision", "")); print(value.get("projection_sha256", ""))' \
        "${BUDGET_PROJECTION}")
    [[ "${#budget_fields[@]}" -eq 2 && \
       "${budget_fields[1]}" =~ ^[0-9a-f]{64}$ ]] || {
        echo "verified budget projection returned invalid identity fields" >&2
        exit 1
    }
    BUDGET_DECISION="${budget_fields[0]}"
    BUDGET_PROJECTION_SHA256="${budget_fields[1]}"
    expected_budget_decision="bc40"
    [[ "${PHASE}" != "gpu-bc80" ]] || expected_budget_decision="bc80"
    [[ "${BUDGET_DECISION}" == "${expected_budget_decision}" ]] || {
        echo "budget projection decision does not match ${PHASE}" >&2
        exit 1
    }
    if [[ "${PHASE}" == "gpu-capacity" ]]; then
        mapfile -t approval_identity < <("${REMEMR1_ENV_PREFIX}/bin/python" -c \
            'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")); print(value.get("budget_projection_sha256", "")); print(value.get("approval_marker_sha256", ""))' \
            "${R1_APPROVAL}")
        [[ "${#approval_identity[@]}" -eq 2 && \
           "${approval_identity[1]}" =~ ^[0-9a-f]{64}$ ]] || {
            echo "R1 approval marker identity is invalid" >&2
            exit 1
        }
        approval_budget_sha256="${approval_identity[0]}"
        R1_APPROVAL_MARKER_SHA256="${approval_identity[1]}"
        [[ "${approval_budget_sha256}" == "${BUDGET_PROJECTION_SHA256}" ]] || {
            echo "R1 approval is not bound to the verified B/C40 budget projection" >&2
            exit 1
        }
    fi
fi

CONFIG_TREE_SHA="$(
    find verl/trainer/config/reproduction scripts/cloud -type f \
        \( -name '*.yaml' -o -name '*.py' -o -name '*.sh' \) -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 sha256sum \
        | sha256sum \
        | awk '{print $1}'
)"
LOCK_FILE_SHA="$(sha256sum environment/reproduction-cu130.lock.json | awk '{print $1}')"
ASSET_FILE_SHA="$(sha256sum environment/reproduction-assets.json | awk '{print $1}')"
IDENTITY_TEXT="git_commit=${REMEMR1_EXPECTED_COMMIT}
experiment_profile_id=${REMEMR1_EXPERIMENT_PROFILE}
config_tree_sha256=${CONFIG_TREE_SHA}
environment_lock_file_sha256=${LOCK_FILE_SHA}
asset_manifest_file_sha256=${ASSET_FILE_SHA}"
IDENTITY_SHA="$(printf '%s\n' "${IDENTITY_TEXT}" | sha256sum | awk '{print $1}')"
PIPELINE_DIR="${PIPELINE_ROOT_REAL}/${REMEMR1_EXPECTED_COMMIT}-${IDENTITY_SHA}"
if [[ -e "${PIPELINE_DIR}" || -L "${PIPELINE_DIR}" ]]; then
    [[ -d "${PIPELINE_DIR}" && ! -L "${PIPELINE_DIR}" ]] || {
        echo "pipeline state path is not a regular directory" >&2
        exit 1
    }
else
    mkdir -- "${PIPELINE_DIR}"
fi
PIPELINE_DIR="$(rememr1_realpath_existing "${PIPELINE_DIR}")"
[[ "$(dirname -- "${PIPELINE_DIR}")" == "${PIPELINE_ROOT_REAL}" ]] || {
    echo "pipeline state escaped the pipeline root" >&2
    exit 1
}
export REMEMR1_PIPELINE_DIR="${PIPELINE_DIR}"
export REMEMR1_PHASE="${PHASE}"
export REMEMR1_OFFLOAD_PROFILE="${OFFLOAD_PROFILE}"
export REMEMR1_R1_APPROVAL="${R1_APPROVAL}"
export REMEMR1_BUDGET_PROJECTION="${BUDGET_PROJECTION}"
export REMEMR1_BUDGET_PROJECTION_SHA256="${BUDGET_PROJECTION_SHA256}"
export REMEMR1_BUDGET_DECISION="${BUDGET_DECISION}"
export REMEMR1_R1_APPROVAL_MARKER_SHA256="${R1_APPROVAL_MARKER_SHA256}"
for state_directory in attempts stages; do
    candidate="${PIPELINE_DIR}/${state_directory}"
    if [[ -e "${candidate}" || -L "${candidate}" ]]; then
        [[ -d "${candidate}" && ! -L "${candidate}" ]] || {
            echo "pipeline ${state_directory} path is not a regular directory" >&2
            exit 1
        }
    else
        mkdir -- "${candidate}"
    fi
    [[ "$(rememr1_realpath_existing "${candidate}")" == "${candidate}" ]] || {
        echo "pipeline ${state_directory} path escaped its state directory" >&2
        exit 1
    }
done

write_value() {
    atomic_write_text "$1" "$2"
}

identity_tmp="${PIPELINE_DIR}/identity.tmp.$$"
printf '%s\n' "${IDENTITY_TEXT}" > "${identity_tmp}"
[[ ! -L "${PIPELINE_DIR}/identity" ]] || {
    rm -f "${identity_tmp}"
    echo "pipeline identity must not be a symlink" >&2
    exit 1
}
if [[ -f "${PIPELINE_DIR}/identity" ]]; then
    cmp -s "${identity_tmp}" "${PIPELINE_DIR}/identity" || {
        rm -f "${identity_tmp}"
        echo "pipeline identity differs from persisted state" >&2
        exit 1
    }
    rm -f "${identity_tmp}"
else
    mv "${identity_tmp}" "${PIPELINE_DIR}/identity"
fi
if [[ -n "${REMEMR1_RESULT_FILE:-}" ]]; then
    write_value "${REMEMR1_RESULT_FILE}" "${PIPELINE_DIR}"
fi

phase_stages() {
    case "$1" in
        cpu)
            printf '%s\n' cpu-preflight cpu-environment cpu-kernel-sources \
                cpu-assets cpu-data cpu-tests cpu-configs cpu-handoff
            ;;
        gpu-gates)
            printf '%s\n' gpu-preflight g0 g1-step1 g1-resume2 g1-artifacts
            ;;
        gpu-capacity)
            printf '%s\n' gpu-preflight g2a g2b-step1 g2b-resume5 \
                g2-length-stress g2-artifacts capacity-seal
            ;;
        gpu-bc40)
            printf '%s\n' gpu-preflight b-pilot c-pilot pilot-gate b20 c20 \
                b40 c40 bc40-artifacts eval40 package40
            ;;
        gpu-bc80)
            printf '%s\n' gpu-preflight b60 c60 b80 c80 bc80-artifacts \
                eval80 package80
            ;;
        gpu-export)
            printf '%s\n' gpu-preflight export-results
            ;;
    esac
}

if [[ "${PHASE}" == gpu-capacity && "${OFFLOAD_PROFILE}" == r1 ]]; then
    r0_capacity_evidence="${PIPELINE_DIR}/capacity/r0/capacity-evidence.json"
    r1_target_identity="${PIPELINE_DIR}/capacity/r0/r1-target-identity.json"
    r0_terminal_sha256="$("${REMEMR1_ENV_PREFIX}/bin/python" -c \
        'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("r0_terminal_sha256", ""))' \
        "${R1_APPROVAL}")"
    [[ "${r0_terminal_sha256}" =~ ^[0-9a-f]{64}$ ]] || {
        echo "R1 approval has an invalid R0 terminal hash" >&2
        exit 1
    }
    "${REMEMR1_ENV_PREFIX}/bin/python" -c \
        'import json,sys; from scripts.cloud.capacity_evidence import verify_r1_approval_marker; load=lambda p: json.load(open(p, encoding="utf-8")); verify_r1_approval_marker(load(sys.argv[1]), r1_identity=load(sys.argv[2]), r0_capacity_evidence=load(sys.argv[3]), r0_terminal_sha256=sys.argv[4], budget_projection_sha256=sys.argv[5])' \
        "${R1_APPROVAL}" "${r1_target_identity}" "${r0_capacity_evidence}" \
        "${r0_terminal_sha256}" "${BUDGET_PROJECTION_SHA256}"
fi

if [[ "${DRY_RUN}" == "yes" ]]; then
    phase_stages "${PHASE}"
    write_value "${PIPELINE_DIR}/last-dry-run" "${PHASE} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    exit 0
fi

if [[ "${PHASE}" == gpu-capacity && "${OFFLOAD_PROFILE}" == r1 ]]; then
    [[ -n "${REMEMR1_LAUNCHER_DIR:-}" && \
       "${REMEMR1_LAUNCHER_DIR}" == /* && \
       -d "${REMEMR1_LAUNCHER_DIR}" && \
       ! -L "${REMEMR1_LAUNCHER_DIR}" ]] || {
        echo "R1 approval consumption requires its immutable launcher directory" >&2
        exit 1
    }
    launcher_root_real="$(rememr1_realpath_existing "${REMEMR1_LAUNCHER_ROOT}")"
    launcher_dir_real="$(rememr1_realpath_existing "${REMEMR1_LAUNCHER_DIR}")"
    [[ "${launcher_root_real}" == "${CLOUD_ROOT_REAL}/launchers" ]] || {
        echo "launcher root differs from the initialized cloud state" >&2
        exit 1
    }
    rememr1_path_is_within "${launcher_dir_real}" "${launcher_root_real}" || {
        echo "R1 approval launcher escaped the initialized launcher root" >&2
        exit 1
    }
    for claim_parent in \
        "${PIPELINE_DIR}/capacity" \
        "${PIPELINE_DIR}/capacity/r1" \
        "${PIPELINE_DIR}/capacity/r1/approval-consumptions"; do
        if [[ -e "${claim_parent}" || -L "${claim_parent}" ]]; then
            [[ -d "${claim_parent}" && ! -L "${claim_parent}" ]] || {
                echo "R1 approval claim parent is unsafe" >&2
                exit 1
            }
        else
            mkdir -- "${claim_parent}"
        fi
        [[ "$(rememr1_realpath_existing "${claim_parent}")" == \
           "${claim_parent}" ]] || {
            echo "R1 approval claim parent escaped pipeline state" >&2
            exit 1
        }
    done
    approval_claim_dir="${PIPELINE_DIR}/capacity/r1/approval-consumptions/${R1_APPROVAL_MARKER_SHA256}"
    if ! mkdir -- "${approval_claim_dir}"; then
        echo "R1 approval marker was already claimed and cannot be reused" >&2
        exit 1
    fi
    rememr1_sync_all || {
        echo "failed to persist the one-time R1 approval claim" >&2
        exit 1
    }
    REMEMR1_R1_APPROVAL_CONSUMPTION="${approval_claim_dir}/consumption.json"
    timeout --verbose --signal=TERM --kill-after=30s 5m \
        "${REMEMR1_ENV_PREFIX}/bin/python" \
        scripts/cloud/capacity_evidence.py consume-r1 \
        --approval-marker "${R1_APPROVAL}" \
        --r1-identity "${r1_target_identity}" \
        --launcher-dir "${launcher_dir_real}" \
        --pipeline-dir "${PIPELINE_DIR}" \
        --output "${REMEMR1_R1_APPROVAL_CONSUMPTION}" || {
            echo "R1 approval claim could not be sealed; the marker remains consumed" >&2
            exit 1
        }
    rememr1_sync_file "${REMEMR1_R1_APPROVAL_CONSUMPTION}" || {
        echo "failed to persist R1 approval consumption evidence" >&2
        exit 1
    }
    export REMEMR1_R1_APPROVAL_CONSUMPTION
fi

SCOPE_OFFLOAD="${OFFLOAD_PROFILE:-none}"
case "${PHASE}" in
    cpu) SCOPE_OFFLOAD=none ;;
    gpu-gates) SCOPE_OFFLOAD=r0 ;;
    gpu-bc40|gpu-bc80|gpu-export)
        SCOPE_OFFLOAD=unresolved
        if [[ -f "${PIPELINE_DIR}/.capacity-ready" && \
              ! -L "${PIPELINE_DIR}/.capacity-ready" ]]; then
            scope_profile="$(<"${PIPELINE_DIR}/.capacity-ready")"
            if [[ "${scope_profile}" == /* && -f "${scope_profile}" && \
                  ! -L "${scope_profile}" ]]; then
                scope_selected="$("${REMEMR1_ENV_PREFIX}/bin/python" -c \
                    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("selected_profile", ""))' \
                    "${scope_profile}" 2>/dev/null || true)"
                case "${scope_selected}" in
                    R0) SCOPE_OFFLOAD=r0 ;;
                    R1) SCOPE_OFFLOAD=r1 ;;
                esac
            fi
        fi
        ;;
esac
if [[ "${SCOPE_OFFLOAD}" == r0 || "${SCOPE_OFFLOAD}" == r1 ]]; then
    OFFLOAD_PROFILE="${SCOPE_OFFLOAD}"
    export REMEMR1_OFFLOAD_PROFILE="${OFFLOAD_PROFILE}"
fi
SCOPE_GENERATION=base
case "${PHASE}" in
    gpu-bc40|gpu-bc80)
        SCOPE_GENERATION="budget-${BUDGET_PROJECTION_SHA256}"
        ;;
    gpu-capacity)
        if [[ "${SCOPE_OFFLOAD}" == r1 ]]; then
            SCOPE_GENERATION="approval-${R1_APPROVAL_MARKER_SHA256}-budget-${BUDGET_PROJECTION_SHA256}"
        fi
        ;;
esac

activate_phase_scope() {
    local base candidate
    for base in attempts stages terminals; do
        candidate="${PIPELINE_DIR}/${base}/${PHASE}/${SCOPE_OFFLOAD}/${SCOPE_GENERATION}"
        if [[ -e "${candidate}" || -L "${candidate}" ]]; then
            [[ -d "${candidate}" && ! -L "${candidate}" ]] || {
                echo "scoped ${base} path is unsafe" >&2
                return 1
            }
        else
            mkdir -p -- "${candidate}"
        fi
        [[ "$(rememr1_realpath_existing "${candidate}")" == "${candidate}" ]] || {
            echo "scoped ${base} path escaped pipeline state" >&2
            return 1
        }
    done
    REMEMR1_ATTEMPT_ROOT="${PIPELINE_DIR}/attempts/${PHASE}/${SCOPE_OFFLOAD}/${SCOPE_GENERATION}"
    REMEMR1_STAGE_RECORD_DIR="${PIPELINE_DIR}/stages/${PHASE}/${SCOPE_OFFLOAD}/${SCOPE_GENERATION}"
    PIPELINE_STATE_DIR="${PIPELINE_DIR}/terminals/${PHASE}/${SCOPE_OFFLOAD}/${SCOPE_GENERATION}"
    REMEMR1_SCOPE_GENERATION="${SCOPE_GENERATION}"
    export REMEMR1_ATTEMPT_ROOT REMEMR1_STAGE_RECORD_DIR REMEMR1_SCOPE_GENERATION
}
activate_phase_scope
PERMANENT_SCIENTIFIC_DIR="${PIPELINE_DIR}/scientific-stops/${PHASE}/${SCOPE_OFFLOAD}"
if [[ -e "${PERMANENT_SCIENTIFIC_DIR}" || -L "${PERMANENT_SCIENTIFIC_DIR}" ]]; then
    [[ -d "${PERMANENT_SCIENTIFIC_DIR}" && \
       ! -L "${PERMANENT_SCIENTIFIC_DIR}" ]] || {
        echo "permanent scientific-stop ledger is unsafe" >&2
        exit 1
    }
else
    mkdir -p -- "${PERMANENT_SCIENTIFIC_DIR}"
fi
if [[ "${PHASE}" == gpu-capacity ]]; then
    if [[ "${SCOPE_OFFLOAD}" == r1 ]]; then
        REMEMR1_CAPACITY_OUTPUT_DIR="${PIPELINE_DIR}/capacity/r1/generations/${SCOPE_GENERATION}"
    else
        REMEMR1_CAPACITY_OUTPUT_DIR="${PIPELINE_DIR}/capacity/r0"
    fi
    export REMEMR1_CAPACITY_OUTPUT_DIR
fi
if [[ -n "${BUDGET_PROJECTION_SHA256}" ]]; then
    write_value "${PIPELINE_STATE_DIR}/budget-projection.sha256" \
        "${BUDGET_PROJECTION_SHA256}"
    rememr1_sync_file "${PIPELINE_STATE_DIR}/budget-projection.sha256"
fi
write_value "${PIPELINE_DIR}/last-terminal" "${PIPELINE_STATE_DIR}"
rememr1_sync_file "${PIPELINE_DIR}/last-terminal"
LAUNCHER_TERMINAL_SNAPSHOT=""
RESULT_TERMINAL_POINTER=""
if [[ -n "${REMEMR1_RESULT_FILE:-}" ]]; then
    RESULT_TERMINAL_POINTER="${REMEMR1_RESULT_FILE}.terminal"
    result_terminal_real="$(rememr1_realpath_maybe "${RESULT_TERMINAL_POINTER}")"
    rememr1_path_is_within "${result_terminal_real}" "${PERSIST_REAL}" || {
        echo "launcher terminal pointer escaped persistent storage" >&2
        exit 1
    }
    if [[ -n "${REMEMR1_LAUNCHER_DIR:-}" ]]; then
        launcher_root_real="$(rememr1_realpath_existing "${REMEMR1_LAUNCHER_ROOT}")"
        launcher_dir_real="$(rememr1_realpath_existing "${REMEMR1_LAUNCHER_DIR}")"
        rememr1_path_is_within "${launcher_dir_real}" "${launcher_root_real}" || {
            echo "launcher terminal snapshot escaped the launcher root" >&2
            exit 1
        }
        launcher_key="$(basename -- "${launcher_dir_real}")"
        [[ "${launcher_key}" =~ ^[A-Za-z0-9._-]+$ ]] || {
            echo "launcher identity is unsafe for terminal snapshotting" >&2
            exit 1
        }
        snapshot_root="${PIPELINE_DIR}/launcher-terminals"
        if [[ -e "${snapshot_root}" || -L "${snapshot_root}" ]]; then
            [[ -d "${snapshot_root}" && ! -L "${snapshot_root}" ]] || exit 1
        else
            mkdir -- "${snapshot_root}"
        fi
        LAUNCHER_TERMINAL_SNAPSHOT="${snapshot_root}/${launcher_key}"
        [[ ! -e "${LAUNCHER_TERMINAL_SNAPSHOT}" && \
           ! -L "${LAUNCHER_TERMINAL_SNAPSHOT}" ]] || {
            echo "launcher terminal snapshot already exists" >&2
            exit 1
        }
    else
        write_value "${RESULT_TERMINAL_POINTER}" "${PIPELINE_STATE_DIR}"
        rememr1_sync_file "${RESULT_TERMINAL_POINTER}"
    fi
fi

CURRENT_STAGE="initialization"
PIPELINE_FINISHED="no"
STAGE_RUNNER="${REMEMR1_STAGE_RUNNER:-${REMEMR1_PROJECT_DIR}/scripts/cloud/run_stage.sh}"

pipeline_test_event() {
    local event_file="${REMEMR1_PIPELINE_TEST_EVENT_FILE:-}"
    [[ "${REMEMR1_TEST_MODE:-no}" == "yes" && -n "${event_file}" ]] || return 0
    case "${event_file}" in
        "${PERSIST_REAL}/"*) ;;
        *) return 1 ;;
    esac
    [[ ! -L "${event_file}" ]] || return 1
    printf '%s\n' "$1" >> "${event_file}"
}

publish_launcher_terminal_snapshot() {
    [[ -n "${LAUNCHER_TERMINAL_SNAPSHOT}" ]] || return 0
    local override_kind="${1:-}"
    local override_phase="${2:-}"
    local override_stage="${3:-}"
    local staging="${LAUNCHER_TERMINAL_SNAPSHOT}.tmp.$$"
    local field terminal_count=0
    local -a fields=(
        current-phase current-stage current-attempt failed-phase failed-stage
        scientific-stop-phase scientific-stop-stage capacity-stop-phase
        capacity-stop-stage retryable budget-projection.sha256
        .success .failed .scientific-stop .capacity-stop
    )
    [[ ! -e "${staging}" && ! -L "${staging}" ]] || return 1
    mkdir -- "${staging}" || return
    for field in "${fields[@]}"; do
        if [[ -n "${override_kind}" && \
              "${field}" == .success || -n "${override_kind}" && \
              "${field}" == .failed || -n "${override_kind}" && \
              "${field}" == .scientific-stop || -n "${override_kind}" && \
              "${field}" == .capacity-stop ]]; then
            continue
        fi
        if [[ -e "${PIPELINE_STATE_DIR}/${field}" || \
              -L "${PIPELINE_STATE_DIR}/${field}" ]]; then
            [[ -f "${PIPELINE_STATE_DIR}/${field}" && \
               ! -L "${PIPELINE_STATE_DIR}/${field}" ]] || return 1
            cp -- "${PIPELINE_STATE_DIR}/${field}" "${staging}/${field}" || return
        fi
    done
    if [[ -n "${override_kind}" ]]; then
        case "${override_kind}" in
            success)
                write_value "${staging}/.success" 0 || return
                ;;
            failed)
                write_value "${staging}/.failed" 1 || return
                write_value "${staging}/failed-phase" "${override_phase}" || return
                write_value "${staging}/failed-stage" "${override_stage}" || return
                ;;
            scientific-stop)
                write_value "${staging}/.scientific-stop" 42 || return
                write_value "${staging}/scientific-stop-phase" \
                    "${override_phase}" || return
                write_value "${staging}/scientific-stop-stage" \
                    "${override_stage}" || return
                write_value "${staging}/retryable" false || return
                ;;
            capacity-stop)
                write_value "${staging}/.capacity-stop" 43 || return
                write_value "${staging}/capacity-stop-phase" \
                    "${override_phase}" || return
                write_value "${staging}/capacity-stop-stage" \
                    "${override_stage}" || return
                write_value "${staging}/retryable" false || return
                ;;
            *) return 2 ;;
        esac
    fi
    for field in .success .failed .scientific-stop .capacity-stop; do
        [[ ! -f "${staging}/${field}" ]] || terminal_count=$((terminal_count + 1))
    done
    [[ "${terminal_count}" -eq 1 ]] || {
        echo "launcher terminal snapshot requires exactly one terminal marker" >&2
        return 1
    }
    write_value "${staging}/operational-state" "${PIPELINE_STATE_DIR}" || return
    write_value "${staging}/launcher-dir" "${REMEMR1_LAUNCHER_DIR}" || return
    rememr1_sync_all || return
    mv -- "${staging}" "${LAUNCHER_TERMINAL_SNAPSHOT}" || return
    rememr1_sync_all || return
    write_value "${RESULT_TERMINAL_POINTER}" "${LAUNCHER_TERMINAL_SNAPSHOT}" || return
    rememr1_sync_file "${RESULT_TERMINAL_POINTER}"
}

publish_shutdown_inhibition() {
    local reason="$1"
    if ! write_value "${PIPELINE_DIR}/shutdown-inhibited" "${reason}" ||
       ! rememr1_sync_file "${PIPELINE_DIR}/shutdown-inhibited"; then
        echo "shutdown inhibition could not be durably published" >&2
        return 74
    fi
}

publish_early_terminal_snapshot() {
    if ! publish_launcher_terminal_snapshot "$@"; then
        publish_shutdown_inhibition \
            "launcher-terminal-snapshot-failed:${PIPELINE_STATE_DIR}" || true
        return 74
    fi
}

pipeline_sync_terminal() {
    local marker="$1"
    local point="$2"
    local running="no"
    [[ -f "${PIPELINE_STATE_DIR}/.running" ]] && running="yes"
    pipeline_test_event "sync:${point}:${marker##*/}:begin:running=${running}" || return
    if [[ "${REMEMR1_TEST_MODE:-no}" == "yes" &&
          "${REMEMR1_PIPELINE_TEST_FAIL_SYNC_AT:-}" == "${point}" ]]; then
        pipeline_test_event \
            "sync:${point}:${marker##*/}:injected-failure:running=${running}" || true
        return 75
    fi
    rememr1_sync_file "${marker}" || return
    pipeline_test_event "sync:${point}:${marker##*/}:complete:running=${running}"
}

remove_running_after_terminal() {
    local marker="$1"
    local point="$2"
    shift 2
    rm -f -- "$@" || return
    rm -f -- "${PIPELINE_STATE_DIR}/.running" || return
    if ! pipeline_test_event "running-removed:${point}"; then
        write_value "${PIPELINE_STATE_DIR}/.running" "$$" || true
        return 1
    fi
    if ! pipeline_sync_terminal "${marker}" "${point}-after-cleanup"; then
        write_value "${PIPELINE_STATE_DIR}/.running" "$$" || true
        return 1
    fi
}

pipeline_exit() {
    local rc="$?"
    trap - EXIT INT TERM
    if [[ "${PIPELINE_FINISHED}" != "yes" ]]; then
        [[ "${rc}" -ne 0 ]] || rc=70
        if [[ "${rc}" -eq 43 ]]; then
            rm -f "${PIPELINE_STATE_DIR}/.failed" \
                "${PIPELINE_STATE_DIR}/failed-stage" \
                "${PIPELINE_STATE_DIR}/failed-phase" \
                "${PIPELINE_STATE_DIR}/.scientific-stop" || true
            if write_value "${PIPELINE_STATE_DIR}/capacity-stop-stage" "${CURRENT_STAGE}" &&
               write_value "${PIPELINE_STATE_DIR}/capacity-stop-phase" "${PHASE}" &&
               write_value "${PIPELINE_STATE_DIR}/retryable" "false" &&
               write_value "${PIPELINE_STATE_DIR}/.capacity-stop" "43" &&
               pipeline_sync_terminal \
                   "${PIPELINE_STATE_DIR}/.capacity-stop" \
                   "capacity-stop-before-cleanup"; then
                remove_running_after_terminal \
                    "${PIPELINE_STATE_DIR}/.capacity-stop" "capacity-stop" \
                    "${PIPELINE_STATE_DIR}/.success" \
                    "${PIPELINE_STATE_DIR}/current-stage" \
                    "${PIPELINE_STATE_DIR}/current-attempt" || true
            else
                echo "capacity-stop state could not be durably published; retaining .running" >&2
            fi
        elif [[ "${rc}" -eq 42 ]]; then
            rm -f "${PIPELINE_STATE_DIR}/.failed" \
                "${PIPELINE_STATE_DIR}/failed-stage" \
                "${PIPELINE_STATE_DIR}/failed-phase" \
                "${PIPELINE_STATE_DIR}/.capacity-stop" \
                "${PIPELINE_STATE_DIR}/capacity-stop-stage" \
                "${PIPELINE_STATE_DIR}/capacity-stop-phase" || true
            if write_value "${PERMANENT_SCIENTIFIC_DIR}/stage" "${CURRENT_STAGE}" &&
               write_value "${PERMANENT_SCIENTIFIC_DIR}/phase" "${PHASE}" &&
               write_value "${PERMANENT_SCIENTIFIC_DIR}/offload-profile" \
                   "${SCOPE_OFFLOAD}" &&
               write_value "${PERMANENT_SCIENTIFIC_DIR}/.scientific-stop" "42" &&
               rememr1_sync_file \
                   "${PERMANENT_SCIENTIFIC_DIR}/.scientific-stop" &&
               write_value "${PIPELINE_STATE_DIR}/scientific-stop-stage" "${CURRENT_STAGE}" &&
               write_value "${PIPELINE_STATE_DIR}/scientific-stop-phase" "${PHASE}" &&
               write_value "${PIPELINE_STATE_DIR}/retryable" "false" &&
               write_value "${PIPELINE_STATE_DIR}/.scientific-stop" "42" &&
               pipeline_sync_terminal \
                   "${PIPELINE_STATE_DIR}/.scientific-stop" \
                   "scientific-stop-before-cleanup"; then
                remove_running_after_terminal \
                    "${PIPELINE_STATE_DIR}/.scientific-stop" "scientific-stop" \
                    "${PIPELINE_STATE_DIR}/.success" \
                    "${PIPELINE_STATE_DIR}/current-stage" \
                    "${PIPELINE_STATE_DIR}/current-attempt" || true
            else
                echo "scientific-stop state could not be durably published; retaining .running" >&2
            fi
        else
            rm -f "${PIPELINE_STATE_DIR}/.scientific-stop" \
                "${PIPELINE_STATE_DIR}/scientific-stop-stage" \
                "${PIPELINE_STATE_DIR}/scientific-stop-phase" \
                "${PIPELINE_STATE_DIR}/.capacity-stop" \
                "${PIPELINE_STATE_DIR}/capacity-stop-stage" \
                "${PIPELINE_STATE_DIR}/capacity-stop-phase" \
                "${PIPELINE_STATE_DIR}/retryable" || true
            if write_value "${PIPELINE_STATE_DIR}/failed-stage" "${CURRENT_STAGE}" &&
               write_value "${PIPELINE_STATE_DIR}/failed-phase" "${PHASE}" &&
               write_value "${PIPELINE_STATE_DIR}/.failed" "${rc}" &&
               pipeline_sync_terminal \
                   "${PIPELINE_STATE_DIR}/.failed" "failure-before-cleanup"; then
                remove_running_after_terminal \
                    "${PIPELINE_STATE_DIR}/.failed" "failure" \
                    "${PIPELINE_STATE_DIR}/.success" || true
            else
                echo "pipeline failure state could not be durably published; retaining .running" >&2
            fi
        fi
        if [[ -e "${PIPELINE_STATE_DIR}/.running" || \
              -L "${PIPELINE_STATE_DIR}/.running" ]]; then
            if ! publish_shutdown_inhibition \
                "terminal-sync-or-cleanup-incomplete:${PIPELINE_STATE_DIR}"; then
                rc=74
            fi
        fi
        if [[ ! -e "${PIPELINE_STATE_DIR}/.running" && \
              ! -L "${PIPELINE_STATE_DIR}/.running" ]] && \
           ! publish_launcher_terminal_snapshot; then
            publish_shutdown_inhibition \
                "launcher-terminal-snapshot-failed:${PIPELINE_STATE_DIR}" || true
            rc=74
        fi
    fi
    exit "${rc}"
}

if [[ -f "${PERMANENT_SCIENTIFIC_DIR}/.scientific-stop" ]]; then
    permanent_stage="unknown"
    [[ ! -s "${PERMANENT_SCIENTIFIC_DIR}/stage" ]] || \
        permanent_stage="$(<"${PERMANENT_SCIENTIFIC_DIR}/stage")"
    publish_early_terminal_snapshot scientific-stop "${PHASE}" \
        "${permanent_stage}" || exit "$?"
    echo "phase/profile has a permanent non-retryable scientific stop" >&2
    exit 42
fi
if [[ -f "${PIPELINE_STATE_DIR}/.scientific-stop" ]]; then
    publish_early_terminal_snapshot || exit "$?"
    echo "pipeline reached a non-retryable scientific stop for this identity" >&2
    exit 42
fi
if [[ -f "${PIPELINE_STATE_DIR}/.capacity-stop" ]]; then
    publish_early_terminal_snapshot || exit "$?"
    echo "pipeline reached a non-retryable capacity stop for this phase/profile" >&2
    exit 43
fi
IMMUTABLE_GENERATION=no
if [[ "${PHASE}" == gpu-bc40 || "${PHASE}" == gpu-bc80 || \
      ( "${PHASE}" == gpu-capacity && "${SCOPE_OFFLOAD}" == r1 ) ]]; then
    IMMUTABLE_GENERATION=yes
fi
if [[ "${IMMUTABLE_GENERATION}" == yes && \
      ( -e "${PIPELINE_STATE_DIR}/.running" || \
        -e "${PIPELINE_STATE_DIR}/.failed" ) ]]; then
    if [[ -e "${PIPELINE_STATE_DIR}/.running" || \
          -L "${PIPELINE_STATE_DIR}/.running" ]]; then
        publish_shutdown_inhibition \
            "immutable-generation-running:${PIPELINE_STATE_DIR}" || exit "$?"
        publish_early_terminal_snapshot failed "${PHASE}" \
            "immutable-generation-running" || exit "$?"
    elif [[ -f "${PIPELINE_STATE_DIR}/.failed" ]]; then
        publish_early_terminal_snapshot || exit "$?"
    fi
    echo "this budget/approval generation cannot be retried; create a new immutable generation" >&2
    exit 1
fi
if [[ -e "${PIPELINE_STATE_DIR}/.running" || \
      -L "${PIPELINE_STATE_DIR}/.running" ]]; then
    [[ -f "${PIPELINE_STATE_DIR}/.running" && \
       ! -L "${PIPELINE_STATE_DIR}/.running" ]] || {
        publish_shutdown_inhibition \
            "unsafe-running-marker:${PIPELINE_STATE_DIR}" || exit "$?"
        publish_early_terminal_snapshot failed "${PHASE}" \
            "unsafe-running-marker" || exit "$?"
        echo "pipeline running marker is unsafe" >&2
        exit 1
    }
    old_pid="$(<"${PIPELINE_STATE_DIR}/.running")"
    [[ "${RETRY_FAILED_STAGE}" == "yes" ]] || {
        publish_shutdown_inhibition \
            "stale-running:${PIPELINE_STATE_DIR}" || exit "$?"
        publish_early_terminal_snapshot failed "${PHASE}" \
            "stale-running-requires-retry" || exit "$?"
        echo "interrupted pipeline marker (old pid ${old_pid}) requires --retry-failed-stage after log review" >&2
        exit 1
    }
fi
if [[ -f "${PIPELINE_STATE_DIR}/.failed" && "${RETRY_FAILED_STAGE}" != "yes" ]]; then
    if [[ -s "${PIPELINE_STATE_DIR}/failed-stage" ]]; then
        failed_stage="$(<"${PIPELINE_STATE_DIR}/failed-stage")"
    else
        failed_stage="unknown"
    fi
    publish_early_terminal_snapshot || exit "$?"
    echo "pipeline previously failed at ${failed_stage}; use --retry-failed-stage after inspection" >&2
    exit 1
fi
if [[ "${PHASE}" == "cpu" && -f "${PIPELINE_DIR}/.gpu-started" ]]; then
    if [[ -f "${PIPELINE_STATE_DIR}/.success" ]]; then
        publish_early_terminal_snapshot || exit "$?"
    fi
    echo "GPU gates have already started; refusing to rerun or downgrade CPU state" >&2
    exit 1
fi

trap pipeline_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
rm -f "${PIPELINE_STATE_DIR}/.success" "${PIPELINE_STATE_DIR}/retryable"
write_value "${PIPELINE_STATE_DIR}/.running" "$$"
write_value "${PIPELINE_STATE_DIR}/current-phase" "${PHASE}"

resolve_verified_stage_run() {
    local run="$1"
    local expected_stage="$2"
    local resolved run_root
    [[ -d "${run}" && ! -L "${run}" ]] || {
        echo "stage run is missing or is a symlink: ${run}" >&2
        return 1
    }
    resolved="$(realpath -e "${run}" 2>/dev/null || true)"
    run_root="${STAGE_RUN_ROOT}"
    [[ -n "${resolved}" && -n "${run_root}" ]] || return 1
    case "${resolved}" in
        "${run_root}/"*) ;;
        *) echo "stage run escaped the persistent run root: ${run}" >&2; return 1 ;;
    esac
    [[ -f "${resolved}/run.meta" && ! -L "${resolved}/run.meta" ]] || return 1
    grep -Fqx "schema_version=1" "${resolved}/run.meta" || return 1
    grep -Fqx "stage=${expected_stage}" "${resolved}/run.meta" || return 1
    grep -Fqx "git_commit=${REMEMR1_EXPECTED_COMMIT}" "${resolved}/run.meta" || return 1
    grep -Fqx "experiment_profile_id=${REMEMR1_EXPERIMENT_PROFILE}" \
        "${resolved}/run.meta" || return 1
    grep -Fqx "phase=${PHASE}" "${resolved}/run.meta" || return 1
    grep -Fqx "offload_profile=${OFFLOAD_PROFILE}" "${resolved}/run.meta" || return 1
    grep -Fqx "budget_projection_sha256=${BUDGET_PROJECTION_SHA256}" \
        "${resolved}/run.meta" || return 1
    grep -Fqx "r1_approval_marker_sha256=${R1_APPROVAL_MARKER_SHA256}" \
        "${resolved}/run.meta" || return 1
    grep -Fqx "scope_generation=${SCOPE_GENERATION}" \
        "${resolved}/run.meta" || return 1
    grep -Fqx "pipeline_dir=${PIPELINE_DIR}" "${resolved}/run.meta" || return 1
    printf '%s\n' "${resolved}"
}

verify_stage_run() {
    local run="$1"
    local expected_stage="$2"
    local resolved
    resolved="$(resolve_verified_stage_run "${run}" "${expected_stage}")" || return
    [[ -f "${resolved}/.success" && ! -L "${resolved}/.success" && \
       "$(<"${resolved}/.success")" == 0 ]] || return 1
    [[ ! -e "${resolved}/.failed" && ! -e "${resolved}/.running" && \
       ! -e "${resolved}/.scientific-stop" && \
       ! -e "${resolved}/.capacity-stop" ]] || return 1
}

stage_record_path() {
    local stage="$1"
    local origin_phase="${PHASE}"
    local origin_profile="${OFFLOAD_PROFILE:-none}"
    local origin_dir=""
    case "${stage}" in
        cpu-*) origin_phase=cpu; origin_profile=none ;;
        g0|g1-step1|g1-resume2|g1-artifacts)
            origin_phase=gpu-gates
            origin_profile=r0
            ;;
        g2a|g2b-step1|g2b-resume5|g2-length-stress|g2-artifacts|capacity-seal|capacity-stop-seal)
            origin_phase=gpu-capacity
            ;;
        b-pilot|c-pilot|pilot-gate|b20|c20|b40|c40|bc40-artifacts|eval40|package40)
            origin_phase=gpu-bc40
            ;;
        b60|c60|b80|c80|bc80-artifacts|eval80|package80)
            origin_phase=gpu-bc80
            ;;
        export-results) origin_phase=gpu-export ;;
    esac
    if [[ "${origin_phase}" == "${PHASE}" ]]; then
        origin_dir="${REMEMR1_STAGE_RECORD_DIR}"
    elif [[ "${origin_phase}" == gpu-gates ]]; then
        origin_dir="${PIPELINE_DIR}/stages/gpu-gates/r0/base"
    elif [[ "${origin_phase}" == gpu-bc40 ]]; then
        [[ -f "${PIPELINE_DIR}/.bc40-stage-record-dir" && \
           ! -L "${PIPELINE_DIR}/.bc40-stage-record-dir" ]] || return 1
        origin_dir="$(<"${PIPELINE_DIR}/.bc40-stage-record-dir")"
    elif [[ "${origin_phase}" == gpu-bc80 ]]; then
        [[ -f "${PIPELINE_DIR}/.bc80-stage-record-dir" && \
           ! -L "${PIPELINE_DIR}/.bc80-stage-record-dir" ]] || return 1
        origin_dir="$(<"${PIPELINE_DIR}/.bc80-stage-record-dir")"
    else
        origin_dir="${PIPELINE_DIR}/stages/${origin_phase}/${origin_profile}/base"
    fi
    [[ "${origin_dir}" == "${PIPELINE_DIR}/stages/"* && \
       -d "${origin_dir}" && ! -L "${origin_dir}" ]] || return 1
    printf '%s\n' "${origin_dir}/${stage}.run"
}

verify_2b_training_artifacts() {
    local key="$1"
    local run="$2"
    local python="$3"
    local config_id step predecessor="" predecessor_step="" length_stress=no
    case "${key}" in
        g2a) config_id="g2a_qwen35_2b_5090_${OFFLOAD_PROFILE}"; step=1 ;;
        g2b-step1) config_id="g2b_qwen35_2b_5090_step1_${OFFLOAD_PROFILE}"; step=1 ;;
        g2b-resume5)
            config_id="g2b_qwen35_2b_5090_resume5_${OFFLOAD_PROFILE}"
            step=5; predecessor=g2b-step1; predecessor_step=1
            ;;
        g2-length-stress)
            config_id="g2_length_stress_qwen35_2b_5090_${OFFLOAD_PROFILE}"
            step=1
            length_stress=yes
            ;;
        b-pilot) config_id="b_pilot_qwen35_2b_5090_${OFFLOAD_PROFILE}"; step=3 ;;
        c-pilot) config_id="c_pilot_qwen35_2b_5090_${OFFLOAD_PROFILE}"; step=3 ;;
        b20) config_id="b20_qwen35_2b_5090_${OFFLOAD_PROFILE}"; step=20 ;;
        c20) config_id="c20_qwen35_2b_5090_${OFFLOAD_PROFILE}"; step=20 ;;
        b40)
            config_id="b40_qwen35_2b_5090_${OFFLOAD_PROFILE}"
            step=40; predecessor=b20; predecessor_step=20
            ;;
        c40)
            config_id="c40_qwen35_2b_5090_${OFFLOAD_PROFILE}"
            step=40; predecessor=c20; predecessor_step=20
            ;;
        b60)
            config_id="b60_qwen35_2b_5090_${OFFLOAD_PROFILE}"
            step=60; predecessor=b40; predecessor_step=40
            ;;
        c60)
            config_id="c60_qwen35_2b_5090_${OFFLOAD_PROFILE}"
            step=60; predecessor=c40; predecessor_step=40
            ;;
        b80)
            config_id="b80_qwen35_2b_5090_${OFFLOAD_PROFILE}"
            step=80; predecessor=b60; predecessor_step=60
            ;;
        c80)
            config_id="c80_qwen35_2b_5090_${OFFLOAD_PROFILE}"
            step=80; predecessor=c60; predecessor_step=60
            ;;
        *) return 1 ;;
    esac
    local -a telemetry_args=(verify-success \
        --attempt-dir "${run}" \
        --expected-config-id "${config_id}" \
        --expected-config-sha256 \
            "$(sha256sum "${REMEMR1_CONFIG_ROOT}/${config_id}.yaml" | awk '{print $1}')" \
        --expected-offload-profile "${OFFLOAD_PROFILE}" \
        --expected-final-step "${step}")
    [[ "${length_stress}" != yes ]] || telemetry_args+=(--length-stress)
    timeout --verbose --signal=TERM --kill-after=30s 10m \
        "${python}" scripts/cloud/training_telemetry.py \
        "${telemetry_args[@]}" || return
    [[ "${length_stress}" != yes ]] || return 0
    local args=(
        --checkpoint-dir "${run}/checkpoints/global_step_${step}"
        --adapter-dir "${run}/artifacts/adapter/global_step_${step}/adapter"
        --expected-step "${step}"
        --expected-train-file "${REMEMR1_FORMAL_TRAIN_PATH}/train.parquet"
        --expected-validation-file "${REMEMR1_FORMAL_VALIDATION_PATH}/train.parquet"
        --expected-resolved-config "${REMEMR1_CONFIG_ROOT}/${config_id}.yaml"
        --expected-train-manifest "${REMEMR1_FORMAL_TRAIN_SHA256}"
        --expected-validation-manifest "${REMEMR1_FORMAL_VALIDATION_SHA256}"
        --expected-base-model Qwen/Qwen3.5-2B
        --expected-revision 15852e8c16360a2fea060d615a32b45270f8a8fc
    )
    if [[ -n "${predecessor}" ]]; then
        local predecessor_record predecessor_run
        predecessor_record="$(stage_record_path "${predecessor}")" || return
        [[ -f "${predecessor_record}" && ! -L "${predecessor_record}" ]] || return 1
        predecessor_run="$(<"${predecessor_record}")"
        args+=(--expected-resume-from \
            "${predecessor_run}/checkpoints/global_step_${predecessor_step}")
    fi
    timeout --verbose --signal=TERM --kill-after=1m 2h \
        "${python}" scripts/cloud/verify_training_artifacts.py "${args[@]}"
}

verify_stage_artifacts() {
    local key="$1"
    local run="${2:-}"
    local python="${REMEMR1_ENV_PREFIX}/bin/python"
    case "${key}" in
        cpu-handoff)
            timeout --verbose --signal=TERM --kill-after=5m 4h \
                "${python}" scripts/cloud/cloud_state.py verify-handoff \
                --handoff "${PIPELINE_DIR}/cpu-handoff.json" \
                --expected-commit "${REMEMR1_EXPECTED_COMMIT}" || return
            ;;
        gpu-preflight)
            timeout --verbose --signal=TERM --kill-after=1m 20m \
                "${python}" scripts/reproduction/verify_environment.py \
                --build-info "${PIPELINE_DIR}/build-info.json" || return
            timeout --verbose --signal=TERM --kill-after=1m 30m \
                "${python}" scripts/cloud/gpu_probe.py \
                --verify-existing \
                --evidence-root "${PIPELINE_DIR}/gpu-evidence" \
                --build-log "${REMEMR1_PERSIST_ROOT}/evidence/gpu-environment/kernel-build.log" \
                --pip-freeze "${REMEMR1_PERSIST_ROOT}/evidence/pip-freeze.txt" \
                --build-info "${PIPELINE_DIR}/build-info.json" \
                --profile "${REMEMR1_OFFLOAD_PROFILE^^}" \
                --optimizer-steps 20 || return
            ;;
        g0)
            [[ -n "${run}" ]] || return 1
            timeout --verbose --signal=TERM --kill-after=1m 20m \
                "${python}" scripts/cloud/run_resolved_training.py verify \
                --attempt-dir "${run}" || return
            timeout --verbose --signal=TERM --kill-after=1m 2h \
                "${python}" scripts/cloud/verify_training_artifacts.py \
                --checkpoint-dir "${run}/checkpoints/global_step_20" \
                --adapter-dir "${run}/artifacts/adapter/global_step_20/adapter" \
                --expected-step 20 \
                --expected-train-file "${REMEMR1_G0_TRAIN_PATH}/train.parquet" \
                --expected-validation-file "${REMEMR1_G0_VALIDATION_PATH}/train.parquet" \
                --expected-resolved-config "${REMEMR1_CONFIG_ROOT}/g0_qwen35_08b.yaml" \
                --expected-train-manifest "${REMEMR1_G0_TRAIN_SHA256}" \
                --expected-validation-manifest "${REMEMR1_G0_VALIDATION_SHA256}" \
                --expected-base-model Qwen/Qwen3.5-0.8B \
                --expected-revision 2fc06364715b967f1860aea9cf38778875588b17 || return
            ;;
        g1-step1)
            [[ -n "${run}" ]] || return 1
            timeout --verbose --signal=TERM --kill-after=1m 20m \
                "${python}" scripts/cloud/run_resolved_training.py verify \
                --attempt-dir "${run}" || return
            timeout --verbose --signal=TERM --kill-after=1m 2h \
                "${python}" scripts/cloud/verify_training_artifacts.py \
                --checkpoint-dir "${run}/checkpoints/global_step_1" \
                --adapter-dir "${run}/artifacts/adapter/global_step_1/adapter" \
                --expected-step 1 \
                --expected-train-file "${REMEMR1_G1_TRAIN_PATH}/train.parquet" \
                --expected-validation-file "${REMEMR1_G1_VALIDATION_PATH}/train.parquet" \
                --expected-resolved-config "${REMEMR1_CONFIG_ROOT}/g1_qwen35_2b_step1.yaml" \
                --expected-train-manifest "${REMEMR1_G1_TRAIN_SHA256}" \
                --expected-validation-manifest "${REMEMR1_G1_VALIDATION_SHA256}" \
                --expected-base-model Qwen/Qwen3.5-2B \
                --expected-revision 15852e8c16360a2fea060d615a32b45270f8a8fc || return
            ;;
        g1-resume2)
            [[ -n "${run}" ]] || return 1
            local g1_step1_run
            g1_step1_run="$(<"${REMEMR1_STAGE_RECORD_DIR}/g1-step1.run")" || return
            timeout --verbose --signal=TERM --kill-after=1m 20m \
                "${python}" scripts/cloud/run_resolved_training.py verify \
                --attempt-dir "${run}" || return
            timeout --verbose --signal=TERM --kill-after=1m 2h \
                "${python}" scripts/cloud/verify_training_artifacts.py \
                --checkpoint-dir "${run}/checkpoints/global_step_2" \
                --adapter-dir "${run}/artifacts/adapter/global_step_2/adapter" \
                --expected-step 2 \
                --expected-train-file "${REMEMR1_G1_TRAIN_PATH}/train.parquet" \
                --expected-validation-file "${REMEMR1_G1_VALIDATION_PATH}/train.parquet" \
                --expected-resolved-config "${REMEMR1_CONFIG_ROOT}/g1_qwen35_2b_resume2.yaml" \
                --expected-train-manifest "${REMEMR1_G1_TRAIN_SHA256}" \
                --expected-validation-manifest "${REMEMR1_G1_VALIDATION_SHA256}" \
                --expected-base-model Qwen/Qwen3.5-2B \
                --expected-revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
                --expected-resume-from "${g1_step1_run}/checkpoints/global_step_1" || return
            ;;
        g1-artifacts)
            [[ -n "${run}" ]] || return 1
            local g1_resume_run
            g1_resume_run="$(<"${REMEMR1_STAGE_RECORD_DIR}/g1-resume2.run")" || return
            timeout --verbose --signal=TERM --kill-after=1m 2h \
                "${python}" scripts/cloud/verify_training_artifacts.py \
                --checkpoint-dir "${g1_resume_run}/checkpoints/global_step_2" \
                --adapter-dir "${g1_resume_run}/artifacts/adapter/global_step_2/adapter" \
                --expected-step 2 \
                --expected-train-file "${REMEMR1_G1_TRAIN_PATH}/train.parquet" \
                --expected-validation-file "${REMEMR1_G1_VALIDATION_PATH}/train.parquet" \
                --expected-resolved-config "${REMEMR1_CONFIG_ROOT}/g1_qwen35_2b_resume2.yaml" \
                --expected-train-manifest "${REMEMR1_G1_TRAIN_SHA256}" \
                --expected-validation-manifest "${REMEMR1_G1_VALIDATION_SHA256}" \
                --expected-base-model Qwen/Qwen3.5-2B \
                --expected-revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
                --expected-resume-from "$(<"${REMEMR1_STAGE_RECORD_DIR}/g1-step1.run")/checkpoints/global_step_1" || return
            timeout --verbose --signal=TERM --kill-after=1m 30m \
                "${python}" scripts/cloud/g1_eval.py \
                --verify-existing \
                --bundle-dir "${REMEMR1_G1_EVAL_PATH}" \
                --expected-manifest-sha256 "${REMEMR1_G1_EVAL_SHA256}" \
                --adapter-dir "${g1_resume_run}/artifacts/adapter/global_step_2/adapter" \
                --output-dir "${run}/artifacts/hf-recurrent-eval" || return
            timeout --verbose --signal=TERM --kill-after=1m 2h \
                "${python}" -c \
                'from verl.utils.checkpoint.reproduction import validate_adapter_export, validate_merged_model_artifact; import sys; adapter = validate_adapter_export(sys.argv[1]); merged = validate_merged_model_artifact(sys.argv[2]); valid = merged["global_step"] == 2 and merged["source_adapter_metadata_sha256"] == adapter.sha256; sys.exit(f"merged artifact identity mismatch: {merged}") if not valid else None' \
                "${g1_resume_run}/artifacts/adapter/global_step_2/adapter" \
                "${run}/artifacts/merged-global_step_2" || return
            ;;
        g2a|g2b-step1|g2b-resume5|g2-length-stress|b-pilot|c-pilot|b20|c20|b40|c40|b60|c60|b80|c80)
            [[ -n "${run}" && -f "${run}/telemetry.json" && \
               ! -L "${run}/telemetry.json" ]] || return 1
            timeout --verbose --signal=TERM --kill-after=1m 20m \
                "${python}" scripts/cloud/run_resolved_training.py verify \
                --attempt-dir "${run}" || return
            verify_2b_training_artifacts "${key}" "${run}" "${python}" || return
            if [[ "${key}" == b-pilot || "${key}" == c-pilot ]]; then
                [[ -f "${run}/evidence/pilot.jsonl" && \
                   ! -L "${run}/evidence/pilot.jsonl" && \
                   -f "${run}/evidence/step_zero_fingerprint.json" && \
                   ! -L "${run}/evidence/step_zero_fingerprint.json" ]] || return 1
            fi
            ;;
        g2-artifacts)
            [[ -n "${run}" && \
               -f "${run}/artifacts/g2-artifacts.json" && \
               ! -L "${run}/artifacts/g2-artifacts.json" ]] || return 1
            for name in identity selected-configs attempt-metadata telemetry; do
                [[ -f "${run}/artifacts/capacity-inputs/${name}.json" && \
                   ! -L "${run}/artifacts/capacity-inputs/${name}.json" ]] || return 1
                [[ -f "${REMEMR1_CAPACITY_OUTPUT_DIR}/${name}.json" && \
                   ! -L "${REMEMR1_CAPACITY_OUTPUT_DIR}/${name}.json" ]] || return 1
                cmp -s "${run}/artifacts/capacity-inputs/${name}.json" \
                    "${REMEMR1_CAPACITY_OUTPUT_DIR}/${name}.json" || return
            done
            ;;
        capacity-stop-seal)
            [[ -n "${run}" && \
               -f "${run}/artifacts/capacity-evidence.json" && \
               ! -L "${run}/artifacts/capacity-evidence.json" ]] || return 1
            for name in identity selected-configs attempt-metadata telemetry; do
                [[ -f "${run}/artifacts/capacity-inputs/${name}.json" && \
                   ! -L "${run}/artifacts/capacity-inputs/${name}.json" ]] || return 1
                cmp -s "${run}/artifacts/capacity-inputs/${name}.json" \
                    "${REMEMR1_CAPACITY_OUTPUT_DIR}/${name}.json" || return
            done
            cmp -s "${run}/artifacts/capacity-evidence.json" \
                "${REMEMR1_CAPACITY_OUTPUT_DIR}/capacity-evidence.json" || return
            timeout --verbose --signal=TERM --kill-after=30s 10m \
                "${python}" -c \
                'import json,sys; from scripts.cloud.capacity_evidence import verify_capacity_evidence; value=json.load(open(sys.argv[1], encoding="utf-8")); verified=verify_capacity_evidence(value); assert verified["classification"]["overall"] != "green"' \
                "${run}/artifacts/capacity-evidence.json" || return
            ;;
        capacity-seal)
            local capacity_dir verify_args approval consumption terminal_sha budget_sha
            capacity_dir="${REMEMR1_CAPACITY_OUTPUT_DIR}"
            verify_args=(verify --capacity-profile \
                "${capacity_dir}/capacity-profile.json")
            if [[ "${OFFLOAD_PROFILE}" == r1 ]]; then
                approval="$(<"${capacity_dir}/r1-approval.path")" || return
                consumption="$(<"${capacity_dir}/r1-approval-consumption.path")" || return
                terminal_sha="$(<"${capacity_dir}/r0-terminal.sha256")" || return
                budget_sha="$(<"${capacity_dir}/budget-projection.sha256")" || return
                verify_args+=(--approval-marker "${approval}" \
                    --approval-consumption "${consumption}" \
                    --r0-capacity-evidence \
                        "${PIPELINE_DIR}/capacity/r0/capacity-evidence.json" \
                    --r0-terminal-sha256 "${terminal_sha}" \
                    --budget-projection-sha256 "${budget_sha}")
            fi
            timeout --verbose --signal=TERM --kill-after=30s 30m \
                "${python}" scripts/cloud/capacity_evidence.py \
                "${verify_args[@]}" || return
            ;;
        pilot-gate)
            [[ -n "${run}" && \
               -f "${run}/artifacts/pilot-gate.json" && \
               ! -L "${run}/artifacts/pilot-gate.json" ]] || return 1
            ;;
        bc40-artifacts|bc80-artifacts)
            [[ -n "${run}" && -f "${run}/artifacts/eval-binding.json" && \
               ! -L "${run}/artifacts/eval-binding.json" ]] || return 1
            local level
            [[ "${key}" == bc40-artifacts ]] && level=40 || level=80
            timeout --verbose --signal=TERM --kill-after=1m 2h \
                "${python}" scripts/cloud/eval_matrix.py plan \
                --config "${REMEMR1_CONFIG_ROOT}/eval${level}_qwen35_2b_5090.yaml" \
                --bindings "${run}/artifacts/eval-binding.json" \
                --output-root "${run}/artifacts/verification-results" \
                --dry-run || return
            ;;
        eval40|eval80)
            [[ -n "${run}" && -f "${run}/artifacts/plan.json" && \
               ! -L "${run}/artifacts/plan.json" ]] || return 1
            timeout --verbose --signal=TERM --kill-after=1m 2h \
                "${python}" -c \
                'from pathlib import Path; import sys; from scripts.cloud import eval_matrix as e; plan=e.load_plan(Path(sys.argv[1])); [e.verify_cell_output(plan, cell) for cell in plan["cells"]]' \
                "${run}/artifacts/plan.json" || return
            ;;
        package40|package80)
            [[ -n "${run}" && \
               -d "${run}/artifacts/verified-package" && \
               ! -L "${run}/artifacts/verified-package" ]] || return 1
            local eval_stage eval_record eval_run
            [[ "${key}" == package40 ]] && eval_stage=eval40 || eval_stage=eval80
            eval_record="$(stage_record_path "${eval_stage}")" || return
            [[ -f "${eval_record}" && ! -L "${eval_record}" ]] || return 1
            eval_run="$(<"${eval_record}")"
            timeout --verbose --signal=TERM --kill-after=1m 30m \
                "${python}" scripts/cloud/eval_matrix.py verify-package \
                --plan "${eval_run}/artifacts/plan.json" \
                --package-dir "${run}/artifacts/verified-package" || return
            ;;
        export-results)
            [[ -n "${run}" && \
               -d "${run}/artifacts/exported-package" && \
               ! -L "${run}/artifacts/exported-package" ]] || return 1
            local eval_stage eval_record eval_run
            if [[ -f "${PIPELINE_DIR}/.bc80-ready" && \
                  "$(<"${PIPELINE_DIR}/.bc80-ready")" == \
                  "${REMEMR1_EXPORT_SOURCE:-}" ]]; then
                eval_stage=eval80
            else
                eval_stage=eval40
            fi
            eval_record="$(stage_record_path "${eval_stage}")" || return
            eval_run="$(<"${eval_record}")" || return
            timeout --verbose --signal=TERM --kill-after=1m 30m \
                "${python}" scripts/cloud/eval_matrix.py verify-package \
                --plan "${eval_run}/artifacts/plan.json" \
                --package-dir "${run}/artifacts/exported-package" || return
            ;;
    esac
}

recover_successful_attempt() {
    local key="$1"
    local candidate run
    local candidates=()
    shopt -s nullglob
    candidates=("${REMEMR1_ATTEMPT_ROOT}/${key}-"*.run)
    shopt -u nullglob
    local index
    for ((index=${#candidates[@]} - 1; index >= 0; index--)); do
        candidate="${candidates[${index}]}"
        [[ -s "${candidate}" ]] || continue
        run="$(<"${candidate}")"
        if verify_stage_run "${run}" "${key}" 2>/dev/null && \
           verify_stage_artifacts "${key}" "${run}" >/dev/null 2>&1; then
            write_value "${REMEMR1_STAGE_RECORD_DIR}/${key}.run" "${run}" || return
            rememr1_sync_file "${REMEMR1_STAGE_RECORD_DIR}/${key}.run" || return
            rm -f "${PIPELINE_STATE_DIR}/current-stage" \
                "${PIPELINE_STATE_DIR}/current-attempt" || return
            echo "[pipeline] recovered ${key}: ${run}"
            return 0
        fi
    done
    return 1
}

ADOPTED_RUN=""

publish_adopted_stage_run() {
    local key="$1"
    local failed_run="$2"
    local original_rc="$3"
    local source_state="${4:-failed}"
    local run_root adopted staging now metadata
    run_root="$(realpath -e "${REMEMR1_PERSIST_ROOT}/cloud/runs")" || return
    adopted="${failed_run}.adopted"
    case "${adopted}" in
        "${run_root}/"*) ;;
        *) echo "adopted stage run escaped the persistent run root" >&2; return 1 ;;
    esac
    if [[ -e "${adopted}" || -L "${adopted}" ]]; then
        verify_stage_run "${adopted}" "${key}" || return
        [[ -f "${adopted}/original-failure" && \
           ! -L "${adopted}/original-failure" && \
           "$(<"${adopted}/original-failure")" == "${original_rc}" ]] || return 1
        [[ -f "${adopted}/adopted-from-failure" && \
           ! -L "${adopted}/adopted-from-failure" && \
           "$(<"${adopted}/adopted-from-failure")" == "${failed_run}" ]] || return 1
        [[ -f "${adopted}/adoption-source-state" && \
           ! -L "${adopted}/adoption-source-state" && \
           "$(<"${adopted}/adoption-source-state")" == "${source_state}" ]] || return 1
        rememr1_sync_file "${adopted}/.success" || return
        ADOPTED_RUN="${adopted}"
        return 0
    fi

    staging="${run_root}/.adoption-${key}-$$-${RANDOM:-0}"
    [[ ! -e "${staging}" && ! -L "${staging}" ]] || return 1
    mkdir -p "${staging}/logs" "${staging}/artifacts" || return
    now="$(date -u +%Y-%m-%dT%H:%M:%SZ)" || {
        rm -rf -- "${staging}"
        return 1
    }
    metadata="schema_version=1
stage=${key}
git_commit=${REMEMR1_EXPECTED_COMMIT}
experiment_profile_id=${REMEMR1_EXPERIMENT_PROFILE}
phase=${PHASE}
offload_profile=${OFFLOAD_PROFILE}
budget_projection_sha256=${BUDGET_PROJECTION_SHA256}
r1_approval_marker_sha256=${R1_APPROVAL_MARKER_SHA256}
scope_generation=${SCOPE_GENERATION}
pipeline_dir=${PIPELINE_DIR}
started_at=${now}
adopted_at=${now}"
    write_value "${staging}/run.meta" "${metadata}" || {
        rm -rf -- "${staging}"
        return 1
    }
    write_value "${staging}/original-failure" "${original_rc}" || {
        rm -rf -- "${staging}"
        return 1
    }
    write_value "${staging}/adopted-from-failure" "${failed_run}" || {
        rm -rf -- "${staging}"
        return 1
    }
    write_value "${staging}/adoption-source-state" "${source_state}" || {
        rm -rf -- "${staging}"
        return 1
    }
    write_value "${staging}/finished-at" "${now}" || {
        rm -rf -- "${staging}"
        return 1
    }
    write_value "${staging}/logs/stage.log" \
        "adopted verified artifacts from ${source_state} run ${failed_run} (${original_rc})" || {
        rm -rf -- "${staging}"
        return 1
    }
    write_value "${staging}/.success" "0" || {
        rm -rf -- "${staging}"
        return 1
    }
    mv "${staging}" "${adopted}" || {
        rm -rf -- "${staging}"
        return 1
    }
    rememr1_sync_file "${adopted}/.success" || return
    ADOPTED_RUN="${adopted}"
}

recover_failed_artifact_attempt() {
    local key="$1"
    case "${key}" in
        g0|g1-step1|g1-resume2|g1-artifacts) ;;
        *) return 1 ;;
    esac
    [[ "${RETRY_FAILED_STAGE}" == "yes" ]] || return 1

    local candidate run resolved original_rc source_state has_running adoption_attempt record
    local candidates=()
    shopt -s nullglob
    candidates=("${REMEMR1_ATTEMPT_ROOT}/${key}-"*.run)
    shopt -u nullglob
    local index
    for ((index=${#candidates[@]} - 1; index >= 0; index--)); do
        candidate="${candidates[${index}]}"
        [[ -f "${candidate}" && ! -L "${candidate}" && -s "${candidate}" ]] || continue
        run="$(<"${candidate}")"
        resolved="$(resolve_verified_stage_run "${run}" "${key}" 2>/dev/null)" || continue
        # A scientific stop is an immutable terminal decision, never recoverable evidence.
        [[ ! -e "${resolved}/.scientific-stop" && \
           ! -L "${resolved}/.scientific-stop" && \
           ! -e "${resolved}/.capacity-stop" && \
           ! -L "${resolved}/.capacity-stop" ]] || continue
        has_running="no"
        if [[ -e "${resolved}/.running" || -L "${resolved}/.running" ]]; then
            [[ -f "${resolved}/.running" && ! -L "${resolved}/.running" ]] || continue
            has_running="yes"
        fi
        if [[ -e "${resolved}/.failed" || -L "${resolved}/.failed" ]]; then
            [[ -f "${resolved}/.failed" && ! -L "${resolved}/.failed" && \
               -f "${resolved}/finished-at" && ! -L "${resolved}/finished-at" && \
               ! -e "${resolved}/.success" && ! -L "${resolved}/.success" ]] || continue
            original_rc="$(<"${resolved}/.failed")"
            [[ "${original_rc}" =~ ^[1-9][0-9]{0,2}$ && \
               "${original_rc}" -ge 1 && "${original_rc}" -le 255 ]] || continue
            source_state="failed"
        elif [[ -e "${resolved}/.success" || -L "${resolved}/.success" ]]; then
            [[ "${has_running}" == "yes" && -f "${resolved}/.success" && \
               ! -L "${resolved}/.success" && "$(<"${resolved}/.success")" == 0 && \
               -f "${resolved}/finished-at" && ! -L "${resolved}/finished-at" ]] || continue
            original_rc="stale-running-after-success"
            source_state="interrupted"
        else
            [[ "${has_running}" == "yes" ]] || continue
            original_rc="interrupted-without-terminal-exit-code"
            source_state="interrupted"
        fi
        if ! verify_stage_artifacts "${key}" "${resolved}" >/dev/null 2>&1; then
            continue
        fi

        ADOPTED_RUN=""
        publish_adopted_stage_run \
            "${key}" "${resolved}" "${original_rc}" "${source_state}" || return 2
        [[ -n "${ADOPTED_RUN}" ]] || return 2
        adoption_attempt="${candidate%.run}.adopted.run"
        if [[ -e "${adoption_attempt}" || -L "${adoption_attempt}" ]]; then
            [[ -f "${adoption_attempt}" && ! -L "${adoption_attempt}" && \
               "$(<"${adoption_attempt}")" == "${ADOPTED_RUN}" ]] || return 2
        else
            write_value "${adoption_attempt}" "${ADOPTED_RUN}" || return 2
        fi
        rememr1_sync_file "${adoption_attempt}" || return 2
        record="${REMEMR1_STAGE_RECORD_DIR}/${key}.run"
        write_value "${record}" "${ADOPTED_RUN}" || return 2
        rememr1_sync_file "${record}" || return 2
        rm -f "${PIPELINE_STATE_DIR}/current-stage" \
            "${PIPELINE_STATE_DIR}/current-attempt" || return 2
        echo "[pipeline] adopted ${key}: ${ADOPTED_RUN} (failed run preserved: ${resolved})"
        return 0
    done
    return 1
}

run_stage() {
    local key="$1"
    CURRENT_STAGE="${key}"
    local record="${REMEMR1_STAGE_RECORD_DIR}/${key}.run"
    if [[ -s "${record}" && "${key}" != "gpu-preflight" ]]; then
        local recorded
        recorded="$(<"${record}")"
        if verify_stage_run "${recorded}" "${key}" && \
           verify_stage_artifacts "${key}" "${recorded}"; then
            echo "[pipeline] reuse ${key}: ${recorded}"
            return 0
        fi
        if [[ "${RETRY_FAILED_STAGE}" != "yes" ]]; then
            echo "persisted stage record is invalid: ${key}" >&2
            return 1
        fi
        echo "[pipeline] retrying invalid persisted stage: ${key}" >&2
    fi
    if [[ "${key}" != "gpu-preflight" ]] && \
       recover_successful_attempt "${key}"; then
        return 0
    fi
    if [[ "${RETRY_FAILED_STAGE}" == "yes" && "${key}" != "gpu-preflight" ]]; then
        local adoption_rc=0
        recover_failed_artifact_attempt "${key}" || adoption_rc="$?"
        if [[ "${adoption_rc}" -eq 0 ]]; then
            return 0
        fi
        if [[ "${adoption_rc}" -ne 1 ]]; then
            echo "failed to publish adopted stage state: ${key}" >&2
            return "${adoption_rc}"
        fi
    fi

    local attempt="${REMEMR1_ATTEMPT_ROOT}/${key}-$(date -u +%Y%m%dT%H%M%SZ)-$$.run"
    write_value "${PIPELINE_STATE_DIR}/current-stage" "${key}"
    write_value "${PIPELINE_STATE_DIR}/current-attempt" "${attempt}"
    echo "[pipeline] start ${key}"
    set +e
    REMEMR1_STAGE_RESULT_FILE="${attempt}" bash "${STAGE_RUNNER}" "${key}"
    local rc="$?"
    set -e
    if [[ "${rc}" -ne 0 ]]; then
        if [[ -s "${attempt}" && "${rc}" -eq 43 ]]; then
            write_value "${PIPELINE_STATE_DIR}/last-capacity-stop-run" \
                "$(<"${attempt}")"
        elif [[ -s "${attempt}" ]]; then
            write_value "${PIPELINE_STATE_DIR}/last-failed-run" "$(<"${attempt}")"
        fi
        return "${rc}"
    fi
    [[ -s "${attempt}" ]] || {
        echo "stage did not publish its run directory: ${key}" >&2
        return 1
    }
    local run
    run="$(<"${attempt}")"
    verify_stage_run "${run}" "${key}"
    verify_stage_artifacts "${key}" "${run}"
    write_value "${record}" "${run}"
    rememr1_sync_file "${record}"
    rm -f "${PIPELINE_STATE_DIR}/current-stage" \
        "${PIPELINE_STATE_DIR}/current-attempt"
    echo "[pipeline] complete ${key}: ${run}"
}

finish_phase() {
    local marker="$1"
    local value="$2"
    local terminal_marker="${3:-${marker}}"
    trap '' INT TERM
    write_value "${marker}" "${value}" || return 74
    write_value "${PIPELINE_DIR}/last-successful-phase" "${PHASE}" || return 74
    if [[ "${terminal_marker}" != "${marker}" ]]; then
        write_value "${terminal_marker}" "0" || return 74
    fi
    pipeline_sync_terminal "${terminal_marker}" "success-before-cleanup" || return 74
    remove_running_after_terminal \
        "${terminal_marker}" "success" \
        "${PIPELINE_STATE_DIR}/.failed" "${PIPELINE_STATE_DIR}/failed-stage" \
        "${PIPELINE_STATE_DIR}/failed-phase" \
        "${PIPELINE_STATE_DIR}/current-stage" \
        "${PIPELINE_STATE_DIR}/current-attempt" \
        "${PIPELINE_STATE_DIR}/.scientific-stop" \
        "${PIPELINE_STATE_DIR}/scientific-stop-stage" \
        "${PIPELINE_STATE_DIR}/scientific-stop-phase" \
        "${PIPELINE_STATE_DIR}/.capacity-stop" \
        "${PIPELINE_STATE_DIR}/capacity-stop-stage" \
        "${PIPELINE_STATE_DIR}/capacity-stop-phase" \
        "${PIPELINE_STATE_DIR}/retryable" || return 74
    publish_launcher_terminal_snapshot || {
        write_value "${PIPELINE_DIR}/shutdown-inhibited" \
            "launcher-terminal-snapshot-failed:${PIPELINE_STATE_DIR}" || true
        return 74
    }
}

if [[ "${PHASE}" == "cpu" ]]; then
    if [[ -f "${PIPELINE_DIR}/.cpu-ready" ]]; then
        CURRENT_STAGE="cpu-handoff-revalidation"
        handoff="$(<"${PIPELINE_DIR}/.cpu-ready")"
        set +e
        timeout --verbose --signal=TERM --kill-after=5m 4h \
            "${REMEMR1_ENV_PREFIX}/bin/python" scripts/cloud/cloud_state.py verify-handoff \
            --handoff "${handoff}" --expected-commit "${REMEMR1_EXPECTED_COMMIT}"
        ready_rc="$?"
        set -e
        if [[ "${ready_rc}" -eq 0 ]]; then
            finish_phase "${PIPELINE_DIR}/.cpu-ready" "${handoff}" \
                "${PIPELINE_STATE_DIR}/.success"
            PIPELINE_FINISHED="yes"
            echo "[pipeline] CPU preparation was already complete: ${handoff}" || true
            exit 0
        fi
        if [[ "${RETRY_FAILED_STAGE}" != "yes" ]]; then
            exit "${ready_rc}"
        fi
        echo "[pipeline] retrying after invalid CPU handoff" >&2
    fi
    for stage in \
        cpu-preflight cpu-environment cpu-kernel-sources cpu-assets \
        cpu-data cpu-tests cpu-configs cpu-handoff; do
        run_stage "${stage}"
    done
    CURRENT_STAGE="cpu-finalization"
    handoff="${PIPELINE_DIR}/cpu-handoff.json"
    finish_phase "${PIPELINE_DIR}/.cpu-ready" "${handoff}" \
        "${PIPELINE_STATE_DIR}/.success"
    PIPELINE_FINISHED="yes"
    echo "[pipeline] CPU handoff ready: ${handoff}" || true
    exit 0
fi

CURRENT_STAGE="cpu-handoff"
[[ -s "${PIPELINE_DIR}/.cpu-ready" ]] || {
    echo "GPU phases require a successful CPU handoff for this exact identity" >&2
    exit 1
}
handoff="$(<"${PIPELINE_DIR}/.cpu-ready")"
set +e
handoff_values="$(timeout --verbose --signal=TERM --kill-after=5m 4h \
    "${REMEMR1_ENV_PREFIX}/bin/python" \
    scripts/cloud/cloud_state.py verify-handoff \
    --handoff "${handoff}" --expected-commit "${REMEMR1_EXPECTED_COMMIT}" \
    --field experiment_profile_id \
    --field handoff_sha256 \
    --field config_tree_sha256 \
    --field bundles.gates.g0.train.path \
    --field bundles.gates.g0.train.manifest_sha256 \
    --field bundles.gates.g0.validation.path \
    --field bundles.gates.g0.validation.manifest_sha256 \
    --field bundles.gates.g1.train.path \
    --field bundles.gates.g1.train.manifest_sha256 \
    --field bundles.gates.g1.validation.path \
    --field bundles.gates.g1.validation.manifest_sha256 \
    --field bundles.gates.g1.eval.path \
    --field bundles.gates.g1.eval.manifest_sha256 \
    --field bundles.formal.train.path \
    --field bundles.formal.train.manifest_sha256 \
    --field bundles.formal.validation.path \
    --field bundles.formal.validation.manifest_sha256 \
    --field bundles.capacity.length_stress.train.path \
    --field bundles.capacity.length_stress.train.manifest_sha256 \
    --field bundles.capacity.length_stress.validation.path \
    --field bundles.capacity.length_stress.validation.manifest_sha256 \
    --field bundles.formal.eval.hotpotqa.path \
    --field bundles.formal.eval.hotpotqa.manifest_sha256 \
    --field bundles.formal.eval.2wikimultihopqa.path \
    --field bundles.formal.eval.2wikimultihopqa.manifest_sha256 \
    --field config_root)"
handoff_rc="$?"
set -e
if [[ "${handoff_rc}" -ne 0 ]]; then
    printf '%s\n' "${handoff_values}" >&2
    exit "${handoff_rc}"
fi
mapfile -t HANDOFF_VALUES <<< "${handoff_values}"
[[ "${#HANDOFF_VALUES[@]}" -eq 26 ]] || {
    echo "verified CPU handoff returned an unexpected field count" >&2
    exit 1
}
[[ "${HANDOFF_VALUES[0]}" == "${REMEMR1_EXPERIMENT_PROFILE}" ]] || {
    echo "verified handoff experiment profile differs from pipeline identity" >&2
    exit 1
}
export REMEMR1_HANDOFF_SHA256="${HANDOFF_VALUES[1]}"
export REMEMR1_ACTIVE_CONFIG_TREE_SHA256="${HANDOFF_VALUES[2]}"
export REMEMR1_G0_TRAIN_PATH="${HANDOFF_VALUES[3]}"
export REMEMR1_G0_TRAIN_SHA256="${HANDOFF_VALUES[4]}"
export REMEMR1_G0_VALIDATION_PATH="${HANDOFF_VALUES[5]}"
export REMEMR1_G0_VALIDATION_SHA256="${HANDOFF_VALUES[6]}"
export REMEMR1_G1_TRAIN_PATH="${HANDOFF_VALUES[7]}"
export REMEMR1_G1_TRAIN_SHA256="${HANDOFF_VALUES[8]}"
export REMEMR1_G1_VALIDATION_PATH="${HANDOFF_VALUES[9]}"
export REMEMR1_G1_VALIDATION_SHA256="${HANDOFF_VALUES[10]}"
export REMEMR1_G1_EVAL_PATH="${HANDOFF_VALUES[11]}"
export REMEMR1_G1_EVAL_SHA256="${HANDOFF_VALUES[12]}"
export REMEMR1_FORMAL_TRAIN_PATH="${HANDOFF_VALUES[13]}"
export REMEMR1_FORMAL_TRAIN_SHA256="${HANDOFF_VALUES[14]}"
export REMEMR1_FORMAL_VALIDATION_PATH="${HANDOFF_VALUES[15]}"
export REMEMR1_FORMAL_VALIDATION_SHA256="${HANDOFF_VALUES[16]}"
export REMEMR1_LENGTH_STRESS_TRAIN_PATH="${HANDOFF_VALUES[17]}"
export REMEMR1_LENGTH_STRESS_TRAIN_SHA256="${HANDOFF_VALUES[18]}"
export REMEMR1_LENGTH_STRESS_VALIDATION_PATH="${HANDOFF_VALUES[19]}"
export REMEMR1_LENGTH_STRESS_VALIDATION_SHA256="${HANDOFF_VALUES[20]}"
export REMEMR1_HOTPOT_EVAL_PATH="${HANDOFF_VALUES[21]}"
export REMEMR1_HOTPOT_EVAL_SHA256="${HANDOFF_VALUES[22]}"
export REMEMR1_2WIKI_EVAL_PATH="${HANDOFF_VALUES[23]}"
export REMEMR1_2WIKI_EVAL_SHA256="${HANDOFF_VALUES[24]}"
export REMEMR1_CONFIG_ROOT="${HANDOFF_VALUES[25]}"
export REMEMR1_VERIFIED_HANDOFF="${handoff}"
export REMEMR1_VERIFIED_HANDOFF_FILE_SHA256="$(sha256sum "${handoff}" | awk '{print $1}')"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
write_value "${PIPELINE_DIR}/.gpu-started" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
rememr1_sync_file "${PIPELINE_DIR}/.gpu-started"

require_phase_predecessor() {
    local marker="$1"
    local label="$2"
    [[ -f "${marker}" && ! -L "${marker}" && -s "${marker}" ]] || {
        echo "${label} predecessor is missing or unsafe" >&2
        return 1
    }
}

load_capacity_profile_selection() {
    require_phase_predecessor "${PIPELINE_DIR}/.capacity-ready" "capacity" || return
    REMEMR1_CAPACITY_PROFILE_PATH="$(<"${PIPELINE_DIR}/.capacity-ready")"
    [[ "${REMEMR1_CAPACITY_PROFILE_PATH}" == /* && \
       -f "${REMEMR1_CAPACITY_PROFILE_PATH}" && \
       ! -L "${REMEMR1_CAPACITY_PROFILE_PATH}" ]] || {
        echo "capacity profile path is missing or unsafe" >&2
        return 1
    }
    rememr1_path_is_within "${REMEMR1_CAPACITY_PROFILE_PATH}" "${PERSIST_REAL}" || {
        echo "capacity profile escaped persistent storage" >&2
        return 1
    }
    local selected
    selected="$("${REMEMR1_ENV_PREFIX}/bin/python" -c \
        'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")); print(value.get("selected_profile", ""))' \
        "${REMEMR1_CAPACITY_PROFILE_PATH}")" || return
    case "${selected}" in
        R0) OFFLOAD_PROFILE="r0" ;;
        R1) OFFLOAD_PROFILE="r1" ;;
        *) echo "capacity profile has an invalid selected profile" >&2; return 1 ;;
    esac
    [[ "${SCOPE_OFFLOAD}" == "${OFFLOAD_PROFILE}" ]] || {
        echo "capacity profile selection differs from the phase terminal scope" >&2
        return 1
    }
    local capacity_dir verify_args approval consumption terminal_sha budget_sha
    capacity_dir="$(dirname -- "${REMEMR1_CAPACITY_PROFILE_PATH}")"
    verify_args=(verify --capacity-profile "${REMEMR1_CAPACITY_PROFILE_PATH}")
    if [[ "${OFFLOAD_PROFILE}" == r1 ]]; then
        [[ -f "${capacity_dir}/r1-approval.path" && \
           ! -L "${capacity_dir}/r1-approval.path" && \
           -f "${capacity_dir}/r1-approval-consumption.path" && \
           ! -L "${capacity_dir}/r1-approval-consumption.path" && \
           -f "${capacity_dir}/r0-terminal.sha256" && \
           ! -L "${capacity_dir}/r0-terminal.sha256" && \
           -f "${capacity_dir}/budget-projection.sha256" && \
           ! -L "${capacity_dir}/budget-projection.sha256" ]] || {
            echo "R1 capacity approval binding is incomplete" >&2
            return 1
        }
        approval="$(<"${capacity_dir}/r1-approval.path")"
        consumption="$(<"${capacity_dir}/r1-approval-consumption.path")"
        terminal_sha="$(<"${capacity_dir}/r0-terminal.sha256")"
        budget_sha="$(<"${capacity_dir}/budget-projection.sha256")"
        [[ "${approval}" == /* && -f "${approval}" && ! -L "${approval}" && \
           "${consumption}" == /* && -f "${consumption}" && \
           ! -L "${consumption}" && \
           "${terminal_sha}" =~ ^[0-9a-f]{64}$ && \
           "${budget_sha}" =~ ^[0-9a-f]{64}$ ]] || return 1
        approval="$(rememr1_realpath_existing "${approval}")" || return
        consumption="$(rememr1_realpath_existing "${consumption}")" || return
        rememr1_path_is_within "${approval}" "${PERSIST_REAL}" || return
        rememr1_path_is_within "${consumption}" "${PERSIST_REAL}" || return
        approved_bc40_budget="${BUDGET_PROJECTION_SHA256}"
        if [[ "${PHASE}" != gpu-bc40 ]]; then
            [[ -f "${PIPELINE_DIR}/.bc40-budget-projection.sha256" && \
               ! -L "${PIPELINE_DIR}/.bc40-budget-projection.sha256" ]] || return 1
            approved_bc40_budget="$(<"${PIPELINE_DIR}/.bc40-budget-projection.sha256")"
        fi
        [[ "${approved_bc40_budget}" == "${budget_sha}" ]] || {
            echo "R1 approval is not bound to this pipeline's verified B/C40 budget" >&2
            return 1
        }
        verify_args+=(--approval-marker "${approval}" \
            --approval-consumption "${consumption}" \
            --r0-capacity-evidence \
                "${PIPELINE_DIR}/capacity/r0/capacity-evidence.json" \
            --r0-terminal-sha256 "${terminal_sha}" \
            --budget-projection-sha256 "${budget_sha}")
    fi
    timeout --verbose --signal=TERM --kill-after=30s 30m \
        "${REMEMR1_ENV_PREFIX}/bin/python" \
        scripts/cloud/capacity_evidence.py "${verify_args[@]}" || return
    export REMEMR1_CAPACITY_PROFILE_PATH OFFLOAD_PROFILE
    export REMEMR1_OFFLOAD_PROFILE="${OFFLOAD_PROFILE}"
    if [[ "${OFFLOAD_PROFILE}" == r1 ]]; then
        export REMEMR1_R1_APPROVAL="${approval}"
        export REMEMR1_R1_APPROVAL_CONSUMPTION="${consumption}"
    fi
}

case "${PHASE}" in
    gpu-gates)
        OFFLOAD_PROFILE="r0"
        export REMEMR1_OFFLOAD_PROFILE="${OFFLOAD_PROFILE}"
        for stage in $(phase_stages "${PHASE}"); do
            run_stage "${stage}"
        done
        CURRENT_STAGE="gpu-finalization"
        write_value "${PIPELINE_DIR}/.gpu-gates-stage-record-dir" \
            "${REMEMR1_STAGE_RECORD_DIR}"
        rememr1_sync_file "${PIPELINE_DIR}/.gpu-gates-stage-record-dir"
        finish_phase "${PIPELINE_DIR}/.gpu-gates-ready" "${handoff}" \
            "${PIPELINE_STATE_DIR}/.success"
        echo "[pipeline] G-1, G0, and G1 gates complete: ${PIPELINE_DIR}" || true
        ;;
    gpu-capacity)
        require_phase_predecessor "${PIPELINE_DIR}/.gpu-gates-ready" "GPU gates"
        [[ ! -e "${PIPELINE_DIR}/.capacity-ready" && \
           ! -L "${PIPELINE_DIR}/.capacity-ready" ]] || {
            echo "capacity profile is already sealed; refusing another capacity run" >&2
            exit 1
        }
        export REMEMR1_OFFLOAD_PROFILE="${OFFLOAD_PROFILE}"
        export REMEMR1_R1_APPROVAL="${R1_APPROVAL}"
        if [[ -f "${PIPELINE_STATE_DIR}/last-capacity-stop-pointer" && \
              ! -L "${PIPELINE_STATE_DIR}/last-capacity-stop-pointer" && \
              -f "${PIPELINE_STATE_DIR}/last-capacity-stop-stage" && \
              ! -L "${PIPELINE_STATE_DIR}/last-capacity-stop-stage" ]]; then
            [[ "${RETRY_FAILED_STAGE}" == yes ]] || {
                echo "capacity-stop evidence finalization requires explicit retry" >&2
                exit 1
            }
            export REMEMR1_CAPACITY_STOPPED_POINTER="$(<"${PIPELINE_STATE_DIR}/last-capacity-stop-pointer")"
            export REMEMR1_CAPACITY_STOPPED_STAGE="$(<"${PIPELINE_STATE_DIR}/last-capacity-stop-stage")"
            CURRENT_STAGE="capacity-stop-seal"
            run_stage capacity-stop-seal
            rm -f "${PIPELINE_DIR}/shutdown-inhibited"
            CURRENT_STAGE="${REMEMR1_CAPACITY_STOPPED_STAGE}"
            exit 43
        fi
        for stage in $(phase_stages "${PHASE}"); do
            CURRENT_STAGE="${stage}"
            set +e
            (set -e; run_stage "${stage}")
            stage_rc="$?"
            set -e
            if [[ "${stage_rc}" -eq 43 ]]; then
                stopped_attempt_record="$(<"${PIPELINE_STATE_DIR}/current-attempt")"
                [[ "${stopped_attempt_record}" == \
                   "${REMEMR1_ATTEMPT_ROOT}/${stage}-"*.run && \
                   -f "${stopped_attempt_record}" && \
                   ! -L "${stopped_attempt_record}" ]] || {
                    echo "capacity stop did not preserve its immutable attempt pointer" >&2
                    exit 1
                }
                if [[ "${stage}" == capacity-seal ]]; then
                    stopped_run="$(<"${stopped_attempt_record}")"
                    set +e
                    "${REMEMR1_ENV_PREFIX}/bin/python" -c \
                        'import json,sys; from scripts.cloud.capacity_evidence import validate_identity,verify_capacity_evidence; evidence=verify_capacity_evidence(json.load(open(sys.argv[1], encoding="utf-8"))); c=evidence["classification"]; non_green=c["non_green_metrics"]; assert c["overall"] != "green"; assert non_green and all(c["metrics"][name]["r1_trigger"] for name in non_green); assert open(sys.argv[1], "rb").read() == open(sys.argv[2], "rb").read(); profile=sys.argv[3]; target=sys.argv[4]; (validate_identity(json.load(open(target, encoding="utf-8"))) if profile == "r0" else None)' \
                        "${REMEMR1_CAPACITY_OUTPUT_DIR}/capacity-evidence.json" \
                        "${stopped_run}/artifacts/capacity-evidence.json" \
                        "${OFFLOAD_PROFILE}" \
                        "${REMEMR1_CAPACITY_OUTPUT_DIR}/r1-target-identity.json"
                    evidence_rc="$?"
                    set -e
                    if [[ "${evidence_rc}" -ne 0 ]]; then
                        write_value "${PIPELINE_DIR}/shutdown-inhibited" \
                            "capacity-evidence-revalidation-failed:${PIPELINE_STATE_DIR}"
                        rememr1_sync_file "${PIPELINE_DIR}/shutdown-inhibited"
                        exit "${evidence_rc}"
                    fi
                    rm -f "${PIPELINE_DIR}/shutdown-inhibited"
                    CURRENT_STAGE="${stage}"
                    exit 43
                fi
                export REMEMR1_CAPACITY_STOPPED_STAGE="${stage}"
                export REMEMR1_CAPACITY_STOPPED_POINTER="${stopped_attempt_record}"
                write_value "${PIPELINE_STATE_DIR}/last-capacity-stop-stage" "${stage}"
                write_value "${PIPELINE_STATE_DIR}/last-capacity-stop-pointer" \
                    "${stopped_attempt_record}"
                rememr1_sync_file \
                    "${PIPELINE_STATE_DIR}/last-capacity-stop-pointer"
                set +e
                (set -e; run_stage capacity-stop-seal)
                finalize_rc="$?"
                set -e
                if [[ "${finalize_rc}" -ne 0 ]]; then
                    write_value "${PIPELINE_DIR}/shutdown-inhibited" \
                        "capacity-stop-finalization-failed:${PIPELINE_STATE_DIR}"
                    rememr1_sync_file "${PIPELINE_DIR}/shutdown-inhibited"
                    exit "${finalize_rc}"
                fi
                rm -f "${PIPELINE_DIR}/shutdown-inhibited"
                CURRENT_STAGE="${stage}"
                exit 43
            fi
            [[ "${stage_rc}" -eq 0 ]] || exit "${stage_rc}"
        done
        CURRENT_STAGE="capacity-finalization"
        capacity_profile="${REMEMR1_CAPACITY_OUTPUT_DIR}/capacity-profile.json"
        [[ -f "${capacity_profile}" && ! -L "${capacity_profile}" ]] || {
            echo "capacity phase did not publish capacity-profile.json" >&2
            exit 1
        }
        write_value "${PIPELINE_DIR}/.capacity-stage-record-dir" \
            "${REMEMR1_STAGE_RECORD_DIR}"
        rememr1_sync_file "${PIPELINE_DIR}/.capacity-stage-record-dir"
        finish_phase "${PIPELINE_DIR}/.capacity-ready" "${capacity_profile}" \
            "${PIPELINE_STATE_DIR}/.success"
        echo "[pipeline] capacity profile ready: ${capacity_profile}" || true
        ;;
    gpu-bc40)
        load_capacity_profile_selection
        if [[ -f "${PIPELINE_DIR}/.bc40-ready" && \
              ! -L "${PIPELINE_DIR}/.bc40-ready" ]]; then
            package="$(<"${PIPELINE_DIR}/.bc40-ready")"
            prior_stage_dir="$(<"${PIPELINE_DIR}/.bc40-stage-record-dir")"
            [[ "${prior_stage_dir}" == "${PIPELINE_DIR}/stages/gpu-bc40/"* && \
               -f "${prior_stage_dir}/eval40.run" && \
               -f "${prior_stage_dir}/package40.run" ]] || exit 1
            prior_eval_run="$(<"${prior_stage_dir}/eval40.run")"
            prior_package_run="$(<"${prior_stage_dir}/package40.run")"
            [[ "${package}" == "${prior_package_run}/artifacts/verified-package" ]] || exit 1
            timeout --verbose --signal=TERM --kill-after=1m 30m \
                "${REMEMR1_ENV_PREFIX}/bin/python" \
                scripts/cloud/eval_matrix.py verify-package \
                --plan "${prior_eval_run}/artifacts/plan.json" \
                --package-dir "${package}"
            CURRENT_STAGE="bc40-revalidation"
            finish_phase "${PIPELINE_DIR}/.bc40-ready" "${package}" \
                "${PIPELINE_STATE_DIR}/.success"
            PIPELINE_FINISHED=yes
            exit 0
        fi
        for stage in $(phase_stages "${PHASE}"); do
            run_stage "${stage}"
        done
        CURRENT_STAGE="bc40-finalization"
        package_run="$(<"${REMEMR1_STAGE_RECORD_DIR}/package40.run")"
        package="${package_run}/artifacts/verified-package"
        [[ -d "${package}" && ! -L "${package}" ]] || {
            echo "B/C40 phase did not publish its verified package" >&2
            exit 1
        }
        write_value "${PIPELINE_DIR}/.bc40-stage-record-dir" \
            "${REMEMR1_STAGE_RECORD_DIR}"
        write_value "${PIPELINE_DIR}/.bc40-budget-projection.sha256" \
            "${BUDGET_PROJECTION_SHA256}"
        rememr1_sync_file "${PIPELINE_DIR}/.bc40-stage-record-dir"
        rememr1_sync_file "${PIPELINE_DIR}/.bc40-budget-projection.sha256"
        finish_phase "${PIPELINE_DIR}/.bc40-ready" "${package}" \
            "${PIPELINE_STATE_DIR}/.success"
        ;;
    gpu-bc80)
        require_phase_predecessor "${PIPELINE_DIR}/.bc40-ready" "B/C40"
        load_capacity_profile_selection
        if [[ -f "${PIPELINE_DIR}/.bc80-ready" && \
              ! -L "${PIPELINE_DIR}/.bc80-ready" ]]; then
            package="$(<"${PIPELINE_DIR}/.bc80-ready")"
            prior_stage_dir="$(<"${PIPELINE_DIR}/.bc80-stage-record-dir")"
            [[ "${prior_stage_dir}" == "${PIPELINE_DIR}/stages/gpu-bc80/"* && \
               -f "${prior_stage_dir}/eval80.run" && \
               -f "${prior_stage_dir}/package80.run" ]] || exit 1
            prior_eval_run="$(<"${prior_stage_dir}/eval80.run")"
            prior_package_run="$(<"${prior_stage_dir}/package80.run")"
            [[ "${package}" == "${prior_package_run}/artifacts/verified-package" ]] || exit 1
            timeout --verbose --signal=TERM --kill-after=1m 30m \
                "${REMEMR1_ENV_PREFIX}/bin/python" \
                scripts/cloud/eval_matrix.py verify-package \
                --plan "${prior_eval_run}/artifacts/plan.json" \
                --package-dir "${package}"
            CURRENT_STAGE="bc80-revalidation"
            finish_phase "${PIPELINE_DIR}/.bc80-ready" "${package}" \
                "${PIPELINE_STATE_DIR}/.success"
            PIPELINE_FINISHED=yes
            exit 0
        fi
        for stage in $(phase_stages "${PHASE}"); do
            run_stage "${stage}"
        done
        CURRENT_STAGE="bc80-finalization"
        package_run="$(<"${REMEMR1_STAGE_RECORD_DIR}/package80.run")"
        package="${package_run}/artifacts/verified-package"
        [[ -d "${package}" && ! -L "${package}" ]] || {
            echo "B/C80 phase did not publish its verified package" >&2
            exit 1
        }
        write_value "${PIPELINE_DIR}/.bc80-stage-record-dir" \
            "${REMEMR1_STAGE_RECORD_DIR}"
        write_value "${PIPELINE_DIR}/.bc80-budget-projection.sha256" \
            "${BUDGET_PROJECTION_SHA256}"
        rememr1_sync_file "${PIPELINE_DIR}/.bc80-stage-record-dir"
        rememr1_sync_file "${PIPELINE_DIR}/.bc80-budget-projection.sha256"
        finish_phase "${PIPELINE_DIR}/.bc80-ready" "${package}" \
            "${PIPELINE_STATE_DIR}/.success"
        ;;
    gpu-export)
        if [[ -f "${PIPELINE_DIR}/.bc80-ready" ]]; then
            export REMEMR1_EXPORT_SOURCE="$(<"${PIPELINE_DIR}/.bc80-ready")"
        else
            require_phase_predecessor "${PIPELINE_DIR}/.bc40-ready" "B/C40"
            export REMEMR1_EXPORT_SOURCE="$(<"${PIPELINE_DIR}/.bc40-ready")"
        fi
        load_capacity_profile_selection
        for stage in $(phase_stages "${PHASE}"); do
            run_stage "${stage}"
        done
        CURRENT_STAGE="export-finalization"
        export_run="$(<"${REMEMR1_STAGE_RECORD_DIR}/export-results.run")"
        export_root="${export_run}/artifacts/exported-package"
        [[ -d "${export_root}" && ! -L "${export_root}" ]] || {
            echo "export phase did not publish a verified result" >&2
            exit 1
        }
        finish_phase "${PIPELINE_DIR}/.export-ready" "${export_root}" \
            "${PIPELINE_STATE_DIR}/.success"
        ;;
esac
PIPELINE_FINISHED="yes"
exit 0
