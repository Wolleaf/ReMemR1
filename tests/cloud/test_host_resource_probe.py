from pathlib import Path

import pytest

from scripts.cloud.host_resource_probe import (
    GIB,
    ProbeError,
    check_minimums,
    probe_resources,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write(root: Path, logical_path: str, value: str = "") -> Path:
    path = root.joinpath(*Path(logical_path).parts[1:])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="ascii")
    return path


def _base_proc(root: Path, cgroup: str, mountinfo: str, ram_gib: int = 64) -> None:
    _write(root, "/proc/meminfo", f"MemTotal:       {ram_gib * 1024 * 1024} kB\n")
    _write(root, "/proc/self/cgroup", cgroup)
    _write(root, "/proc/self/mountinfo", mountinfo)


def _v2_fixture(
    root: Path,
    *,
    memory_max: str = "max",
    cpu_max: str = "max 100000",
    cpuset: str = "0-31",
) -> None:
    _base_proc(
        root,
        "0::/\n",
        "36 25 0:32 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
    )
    _write(root, "/sys/fs/cgroup/memory.max", memory_max + "\n")
    _write(root, "/sys/fs/cgroup/cpu.max", cpu_max + "\n")
    _write(root, "/sys/fs/cgroup/cpuset.cpus.effective", cpuset + "\n")


def test_v2_unlimited_limits_use_host_resources(tmp_path):
    _v2_fixture(tmp_path)

    result = probe_resources(tmp_path, host_cpu_count=32)

    assert result.cgroup_mode == "v2"
    assert result.memory_limit_bytes is None
    assert result.cpu_quota is None
    assert result.effective_memory_bytes == 64 * GIB
    assert result.effective_cpu_cores == 32
    assert check_minimums(result, min_cpu_cores=24, min_ram_gib=48) == []


def test_v2_half_core_and_two_gib_fail_both_minimums(tmp_path):
    _v2_fixture(
        tmp_path,
        memory_max=str(2 * GIB),
        cpu_max="50000 100000",
    )

    result = probe_resources(tmp_path, host_cpu_count=32)
    violations = check_minimums(result, min_cpu_cores=24, min_ram_gib=48)

    assert result.effective_cpu_cores.numerator == 1
    assert result.effective_cpu_cores.denominator == 2
    assert result.as_dict()["effective_cpu"]["millicores_floor"] == 500
    assert result.effective_memory_bytes == 2 * GIB
    assert len(violations) == 2
    assert "500 millicores" in violations[0]
    assert "2147483648 bytes" in violations[1]


def test_cpuset_is_counted_without_double_counting_overlaps(tmp_path):
    _v2_fixture(tmp_path, cpuset="0-3,2-5,8,10-12")

    result = probe_resources(tmp_path, host_cpu_count=64)

    assert result.cpuset_cpu_count == 10
    assert result.effective_cpu_cores == 10


def test_v1_memory_cpu_and_cpuset_fallback(tmp_path):
    mountinfo = "".join(
        [
            "29 23 0:26 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n",
            "30 23 0:27 / /sys/fs/cgroup/cpu rw - cgroup cgroup rw,cpu,cpuacct\n",
            "31 23 0:28 / /sys/fs/cgroup/cpuset rw - cgroup cgroup rw,cpuset\n",
        ]
    )
    _base_proc(
        tmp_path,
        "5:memory:/job\n4:cpu,cpuacct:/job\n3:cpuset:/job\n",
        mountinfo,
        ram_gib=96,
    )
    _write(tmp_path, "/sys/fs/cgroup/memory/memory.limit_in_bytes", str(2**63 - 4096))
    _write(tmp_path, "/sys/fs/cgroup/memory/job/memory.limit_in_bytes", str(60 * GIB))
    _write(tmp_path, "/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "-1\n")
    _write(tmp_path, "/sys/fs/cgroup/cpu/cpu.cfs_period_us", "100000\n")
    _write(tmp_path, "/sys/fs/cgroup/cpu/job/cpu.cfs_quota_us", "3000000\n")
    _write(tmp_path, "/sys/fs/cgroup/cpu/job/cpu.cfs_period_us", "100000\n")
    _write(tmp_path, "/sys/fs/cgroup/cpuset/cpuset.cpus", "0-63\n")
    _write(tmp_path, "/sys/fs/cgroup/cpuset/job/cpuset.cpus", "0-23\n")

    result = probe_resources(tmp_path, host_cpu_count=64)

    assert result.cgroup_mode == "v1"
    assert result.memory_limit_bytes == 60 * GIB
    assert result.cpu_quota == 30
    assert result.cpuset_cpu_count == 24
    assert result.effective_cpu_cores == 24
    assert check_minimums(result, min_cpu_cores=24, min_ram_gib=48) == []


def test_malformed_limit_fails_closed(tmp_path):
    _v2_fixture(tmp_path, cpu_max="not-a-quota")

    with pytest.raises(ProbeError, match="malformed CPU quota"):
        probe_resources(tmp_path, host_cpu_count=32)


def test_unreadable_required_limit_fails_closed(tmp_path):
    _v2_fixture(tmp_path)
    memory_max = tmp_path / "sys" / "fs" / "cgroup" / "memory.max"
    memory_max.unlink()
    memory_max.mkdir()

    with pytest.raises(ProbeError, match="cannot read"):
        probe_resources(tmp_path, host_cpu_count=32)


def test_cpu_preflight_uses_probe_with_profile_defaults():
    source = (REPO_ROOT / "scripts" / "cloud" / "run_stage.sh").read_text(
        encoding="utf-8"
    )
    preflight = source.split("    cpu-preflight)", 1)[1].split("        ;;", 1)[0]

    assert 'min_cpu_cores="${REMEMR1_MIN_CPU_CORES:-24}"' in preflight
    assert 'min_ram_gib="${REMEMR1_MIN_RAM_GIB:-48}"' in preflight
    assert "python3 scripts/cloud/host_resource_probe.py" in preflight
    assert preflight.index("run_logged checkout") < preflight.index("host-resource-preflight")
