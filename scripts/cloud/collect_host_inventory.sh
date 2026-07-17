#!/usr/bin/env bash
set -uo pipefail

section() {
    printf '\n===== %s =====\n' "$1"
}

run() {
    printf '$'
    printf ' %q' "$@"
    printf '\n'
    "$@" 2>&1
    local rc=$?
    if (( rc != 0 )); then
        printf '[exit=%d]\n' "${rc}"
    fi
    return 0
}

read_file() {
    local path="$1"
    printf '%s\n' "--- ${path} ---"
    if [[ -r "${path}" ]]; then
        sed -n '1,240p' "${path}"
    else
        printf '%s\n' '[unavailable]'
    fi
}

section "identity"
run date --iso-8601=seconds
run uname -a
read_file /etc/os-release

section "cpu"
run nproc
run lscpu

section "memory"
run free -b
run sh -c "grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo"

section "cgroup"
read_file /proc/self/cgroup
run sh -c "grep -E ' - cgroup2? ' /proc/self/mountinfo"
shopt -s nullglob
cgroup_files=(
    /sys/fs/cgroup/cpu.max
    /sys/fs/cgroup/cpuset.cpus
    /sys/fs/cgroup/cpuset.cpus.effective
    /sys/fs/cgroup/memory.max
    /sys/fs/cgroup/memory.current
    /sys/fs/cgroup/memory.swap.max
    /sys/fs/cgroup/memory.swap.current
    /sys/fs/cgroup/cpu/cpu.cfs_quota_us
    /sys/fs/cgroup/cpu/cpu.cfs_period_us
    /sys/fs/cgroup/cpuset/cpuset.cpus
    /sys/fs/cgroup/memory/memory.limit_in_bytes
    /sys/fs/cgroup/memory/memory.usage_in_bytes
    /sys/fs/cgroup/memory/memory.memsw.usage_in_bytes
)
for path in "${cgroup_files[@]}"; do
    [[ -e "${path}" ]] && read_file "${path}"
done

section "persistent-volume"
run findmnt -T /root/autodl-tmp -o TARGET,SOURCE,FSTYPE,OPTIONS,MAJ:MIN
run df -B1 -T /root/autodl-tmp
run stat -f -c 'type=%T block_size=%S blocks=%b available=%a' /root/autodl-tmp

section "gpu"
if command -v nvidia-smi >/dev/null 2>&1; then
    run nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,memory.free --format=csv,noheader,nounits
    run nvidia-smi --query-gpu=index,compute_cap --format=csv,noheader,nounits
    run nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits
else
    printf '%s\n' 'nvidia-smi unavailable'
fi

section "toolchain"
run python3 --version
run sh -c 'command -v nvcc && nvcc --version'
run sh -c 'python3 -c "import torch; print(\"torch=\" + torch.__version__); print(\"torch_cuda=\" + str(torch.version.cuda)); print(\"cuda_available=\" + str(torch.cuda.is_available()))"'

section "safe-environment"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES-<unset>}"
printf 'NVIDIA_VISIBLE_DEVICES=%s\n' "${NVIDIA_VISIBLE_DEVICES-<unset>}"

section "collector"
run sha256sum "$0"
