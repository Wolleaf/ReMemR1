import argparse
from fractions import Fraction
from pathlib import Path

import pytest

from scripts.cloud.host_resource_probe import (
    GIB,
    ProbeError,
    _positive_fraction,
    check_minimums,
    create_memory_telemetry_probe,
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
    ram_gib: int = 64,
) -> None:
    _base_proc(
        root,
        "0::/\n",
        "36 25 0:32 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
        ram_gib=ram_gib,
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


def test_v2_half_core_and_two_gib_support_low_resource_minimums(tmp_path):
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
    assert check_minimums(
        result,
        min_cpu_cores=Fraction(1, 2),
        min_ram_gib=2,
    ) == []


def test_low_resource_minimums_reject_less_than_half_core_or_two_gib(tmp_path):
    _v2_fixture(
        tmp_path,
        memory_max=str(2 * GIB - 1),
        cpu_max="49999 100000",
    )

    result = probe_resources(tmp_path, host_cpu_count=32)
    violations = check_minimums(
        result,
        min_cpu_cores=Fraction(1, 2),
        min_ram_gib=2,
    )

    assert len(violations) == 2
    assert "requires at least 1/2 effective CPU cores" in violations[0]
    assert "requires at least 2 GiB effective RAM" in violations[1]


def test_positive_fraction_accepts_integer_and_fraction_values():
    assert _positive_fraction("1/2") == Fraction(1, 2)
    assert _positive_fraction("16") == Fraction(16, 1)


@pytest.mark.parametrize("value", ["0", "-1/2", "1/0", "not-a-number"])
def test_positive_fraction_rejects_nonpositive_or_malformed_values(value):
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer or fraction"):
        _positive_fraction(value)


def test_active_profile_accepts_16_cores_and_90_gb_but_rejects_15_cores(tmp_path):
    _v2_fixture(
        tmp_path / "accepted",
        memory_max=str(90_000_000_000),
        cpuset="0-15",
        ram_gib=90,
    )
    accepted = probe_resources(tmp_path / "accepted", host_cpu_count=16)
    assert check_minimums(accepted, min_cpu_cores=16, min_ram_gib=48) == []

    _v2_fixture(
        tmp_path / "rejected",
        memory_max=str(90 * GIB),
        cpuset="0-14",
        ram_gib=90,
    )
    rejected = probe_resources(tmp_path / "rejected", host_cpu_count=16)
    violations = check_minimums(rejected, min_cpu_cores=16, min_ram_gib=48)
    assert len(violations) == 1
    assert "requires at least 16 effective CPU cores" in violations[0]


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


def test_v2_telemetry_uses_limiting_ancestor_counters_over_host_fallback(tmp_path):
    _base_proc(
        tmp_path,
        "0::/parent/job\n",
        "36 25 0:32 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
        ram_gib=256,
    )
    _write(tmp_path, "/sys/fs/cgroup/memory.max", str(200 * GIB))
    _write(tmp_path, "/sys/fs/cgroup/parent/memory.max", str(90 * GIB))
    _write(tmp_path, "/sys/fs/cgroup/parent/job/memory.max", "max\n")
    current = _write(
        tmp_path,
        "/sys/fs/cgroup/parent/memory.current",
        str(61 * GIB),
    )
    _write(tmp_path, "/sys/fs/cgroup/parent/memory.swap.current", str(3 * GIB))

    probe = create_memory_telemetry_probe(
        tmp_path,
        host_total_memory_bytes=256 * GIB,
    )
    first = probe.sample(
        fallback_used_memory_bytes=180 * GIB,
        fallback_swap_used_bytes=12 * GIB,
    )

    assert probe.cgroup_mode == "v2"
    assert first.total == 90 * GIB
    assert first.used == 61 * GIB
    assert first.swap_used == 3 * GIB

    current.write_text(str(62 * GIB), encoding="ascii")
    second = probe.sample(
        fallback_used_memory_bytes=181 * GIB,
        fallback_swap_used_bytes=13 * GIB,
    )
    assert second.used == 62 * GIB


def test_v1_telemetry_derives_swap_from_memsw_over_host_fallback(tmp_path):
    _base_proc(
        tmp_path,
        "5:memory:/job\n",
        "29 23 0:26 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n",
        ram_gib=192,
    )
    _write(
        tmp_path,
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        str(2**63 - 4096),
    )
    _write(
        tmp_path,
        "/sys/fs/cgroup/memory/job/memory.limit_in_bytes",
        str(96 * GIB),
    )
    _write(
        tmp_path,
        "/sys/fs/cgroup/memory/job/memory.usage_in_bytes",
        str(70 * GIB),
    )
    _write(
        tmp_path,
        "/sys/fs/cgroup/memory/job/memory.memsw.usage_in_bytes",
        str(72 * GIB),
    )

    probe = create_memory_telemetry_probe(
        tmp_path,
        host_total_memory_bytes=192 * GIB,
    )
    snapshot = probe.sample(
        fallback_used_memory_bytes=150 * GIB,
        fallback_swap_used_bytes=11 * GIB,
    )

    assert probe.cgroup_mode == "v1"
    assert snapshot.total == 96 * GIB
    assert snapshot.used == 70 * GIB
    assert snapshot.swap_used == 2 * GIB


def test_memory_telemetry_uses_host_values_without_a_cgroup(tmp_path):
    _base_proc(tmp_path, "", "", ram_gib=128)

    probe = create_memory_telemetry_probe(
        tmp_path,
        host_total_memory_bytes=128 * GIB,
    )
    snapshot = probe.sample(
        fallback_used_memory_bytes=40 * GIB,
        fallback_swap_used_bytes=1 * GIB,
    )

    assert probe.cgroup_mode == "none"
    assert snapshot.total == 128 * GIB
    assert snapshot.used == 40 * GIB
    assert snapshot.swap_used == 1 * GIB


def test_cgroup_telemetry_fails_closed_without_a_swap_counter(tmp_path):
    _base_proc(
        tmp_path,
        "0::/job\n",
        "36 25 0:32 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
        ram_gib=128,
    )
    _write(tmp_path, "/sys/fs/cgroup/memory.max", str(128 * GIB))
    _write(tmp_path, "/sys/fs/cgroup/job/memory.max", str(90 * GIB))
    _write(tmp_path, "/sys/fs/cgroup/job/memory.current", str(40 * GIB))

    probe = create_memory_telemetry_probe(
        tmp_path,
        host_total_memory_bytes=128 * GIB,
    )
    with pytest.raises(ProbeError, match="swap usage counter is unavailable"):
        probe.sample(
            fallback_used_memory_bytes=50 * GIB,
            fallback_swap_used_bytes=0,
        )


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

    assert 'min_cpu_cores="${REMEMR1_MIN_CPU_CORES:-1/2}"' in preflight
    assert 'min_ram_gib="${REMEMR1_MIN_RAM_GIB:-2}"' in preflight
    assert 'rememr1_find_host_python "${REMEMR1_ENV_PREFIX}"' in preflight
    assert '"${host_python}" scripts/cloud/host_resource_probe.py' in preflight
    assert preflight.index("run_logged checkout") < preflight.index("host-resource-preflight")

    runtime = (REPO_ROOT / "scripts" / "cloud" / "lib" / "runtime.sh").read_text(
        encoding="utf-8"
    )
    assert "sys.version_info[0] == 3" in runtime
    assert "sys.version_info[:2] >= (3, 10)" in runtime
    assert "type -P python3" in runtime
    assert '[[ -n "${candidate}" && -f "${candidate}" && -x "${candidate}" ]]' in runtime
