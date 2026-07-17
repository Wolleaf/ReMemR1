#!/usr/bin/env python3
"""Fail-closed host and cgroup resource probe for CPU preparation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


GIB = 1024 * 1024 * 1024


class ProbeError(RuntimeError):
    """Raised when a resource limit cannot be determined safely."""


@dataclass(frozen=True)
class MountInfo:
    root: PurePosixPath
    mount_point: PurePosixPath
    fs_type: str
    mount_source: str
    super_options: Tuple[str, ...]


@dataclass(frozen=True)
class Membership:
    unified: Optional[PurePosixPath]
    controllers: Dict[str, PurePosixPath]


@dataclass(frozen=True)
class ResourceProbe:
    cgroup_mode: str
    host_cpu_cores: int
    cpu_quota: Optional[Fraction]
    cpu_quota_source: str
    cpuset_cpu_count: Optional[int]
    cpuset_source: str
    effective_cpu_cores: Fraction
    host_memory_bytes: int
    memory_limit_bytes: Optional[int]
    memory_limit_source: str
    effective_memory_bytes: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "schema_version": 1,
            "cgroup_mode": self.cgroup_mode,
            "host_cpu_cores": self.host_cpu_cores,
            "cpu_quota": (
                None
                if self.cpu_quota is None
                else {
                    "numerator": self.cpu_quota.numerator,
                    "denominator": self.cpu_quota.denominator,
                }
            ),
            "cpu_quota_source": self.cpu_quota_source,
            "cpuset_cpu_count": self.cpuset_cpu_count,
            "cpuset_source": self.cpuset_source,
            "effective_cpu": {
                "numerator": self.effective_cpu_cores.numerator,
                "denominator": self.effective_cpu_cores.denominator,
                "millicores_floor": (
                    self.effective_cpu_cores.numerator * 1000
                    // self.effective_cpu_cores.denominator
                ),
            },
            "host_memory_bytes": self.host_memory_bytes,
            "memory_limit_bytes": self.memory_limit_bytes,
            "memory_limit_source": self.memory_limit_source,
            "effective_memory_bytes": self.effective_memory_bytes,
        }


def _unescape_mount_path(value: str) -> str:
    for encoded, decoded in (
        ("\\040", " "),
        ("\\011", "\t"),
        ("\\012", "\n"),
        ("\\134", "\\"),
    ):
        value = value.replace(encoded, decoded)
    return value


def parse_mountinfo(text: str) -> List[MountInfo]:
    mounts: List[MountInfo] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        if " - " not in line:
            raise ProbeError("malformed /proc/self/mountinfo line %d" % line_number)
        prefix, suffix = line.split(" - ", 1)
        prefix_fields = prefix.split()
        suffix_fields = suffix.split()
        if len(prefix_fields) < 6 or len(suffix_fields) < 3:
            raise ProbeError("malformed /proc/self/mountinfo line %d" % line_number)
        fs_type = suffix_fields[0]
        if fs_type not in {"cgroup", "cgroup2"}:
            continue
        root = PurePosixPath(_unescape_mount_path(prefix_fields[3]))
        mount_point = PurePosixPath(_unescape_mount_path(prefix_fields[4]))
        if not root.is_absolute() or not mount_point.is_absolute():
            raise ProbeError("cgroup mount paths must be absolute")
        mounts.append(
            MountInfo(
                root=root,
                mount_point=mount_point,
                fs_type=fs_type,
                mount_source=suffix_fields[1],
                super_options=tuple(suffix_fields[2].split(",")),
            )
        )
    return mounts


def parse_membership(text: str) -> Membership:
    unified: Optional[PurePosixPath] = None
    controllers: Dict[str, PurePosixPath] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split(":", 2)
        if len(fields) != 3 or not fields[0].isdigit():
            raise ProbeError("malformed /proc/self/cgroup line %d" % line_number)
        hierarchy, raw_controllers, raw_path = fields
        path = PurePosixPath(raw_path)
        if not path.is_absolute():
            raise ProbeError("cgroup membership path must be absolute")
        if raw_controllers == "":
            if hierarchy != "0" or unified is not None:
                raise ProbeError("malformed unified cgroup membership")
            unified = path
            continue
        for controller in raw_controllers.split(","):
            if not controller:
                raise ProbeError("empty cgroup v1 controller name")
            previous = controllers.get(controller)
            if previous is not None and previous != path:
                raise ProbeError("conflicting cgroup membership for %s" % controller)
            controllers[controller] = path
    return Membership(unified=unified, controllers=controllers)


def parse_cpuset(value: str, source: str) -> int:
    value = value.strip()
    if not value:
        return 0
    cpus = set()
    for item in value.split(","):
        fields = item.split("-", 1)
        if len(fields) == 1:
            if not fields[0].isdigit():
                raise ProbeError("malformed cpuset in %s" % source)
            start = end = int(fields[0])
        else:
            if not fields[0].isdigit() or not fields[1].isdigit():
                raise ProbeError("malformed cpuset in %s" % source)
            start, end = int(fields[0]), int(fields[1])
            if end < start:
                raise ProbeError("descending cpuset range in %s" % source)
        cpus.update(range(start, end + 1))
    return len(cpus)


def parse_memtotal(text: str) -> int:
    matches = []
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] == "MemTotal:":
            matches.append(fields)
    if len(matches) != 1:
        raise ProbeError("/proc/meminfo must contain exactly one MemTotal entry")
    fields = matches[0]
    if len(fields) != 3 or not fields[1].isdigit() or fields[2] != "kB":
        raise ProbeError("malformed MemTotal entry in /proc/meminfo")
    value = int(fields[1]) * 1024
    if value <= 0:
        raise ProbeError("host memory must be positive")
    return value


def _rooted(filesystem_root: Path, logical_path: PurePosixPath) -> Path:
    if not logical_path.is_absolute():
        raise ProbeError("internal probe path is not absolute")
    return filesystem_root.joinpath(*logical_path.parts[1:])


def _read_required(path: Path, description: str) -> str:
    try:
        return path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise ProbeError("cannot read %s (%s): %s" % (description, path, error)) from error


def _read_optional(path: Path, description: str) -> Optional[str]:
    try:
        return path.read_text(encoding="ascii")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError) as error:
        raise ProbeError("cannot read %s (%s): %s" % (description, path, error)) from error


def _mount_target(mount: MountInfo, membership: PurePosixPath) -> PurePosixPath:
    if membership == PurePosixPath("/"):
        return mount.mount_point
    try:
        relative = membership.relative_to(mount.root)
    except ValueError as error:
        raise ProbeError("cgroup membership is outside its mounted hierarchy") from error
    return mount.mount_point / relative


def _mount_for(
    mounts: Sequence[MountInfo],
    membership: PurePosixPath,
    fs_type: str,
    controller: Optional[str] = None,
) -> Tuple[MountInfo, PurePosixPath]:
    candidates = []
    for mount in mounts:
        if mount.fs_type != fs_type:
            continue
        if controller is not None:
            tokens = set(mount.super_options)
            tokens.update(mount.mount_source.split(","))
            tokens.update(mount.mount_point.name.split(","))
            if controller not in tokens:
                continue
        try:
            target = _mount_target(mount, membership)
        except ProbeError:
            continue
        candidates.append((len(mount.root.parts), mount, target))
    if not candidates:
        label = fs_type if controller is None else "%s controller" % controller
        raise ProbeError("cannot resolve the %s cgroup mount" % label)
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def _ancestors(target: PurePosixPath, mount_point: PurePosixPath) -> Iterable[PurePosixPath]:
    current = target
    while True:
        yield current
        if current == mount_point:
            return
        parent = current.parent
        if parent == current:
            raise ProbeError("cgroup target escaped its mount")
        current = parent


def _parse_memory_limit(value: str, source: str, allow_max: bool) -> Optional[int]:
    value = value.strip()
    if allow_max and value == "max":
        return None
    if not value.isdigit():
        raise ProbeError("malformed memory limit in %s" % source)
    return int(value)


def _parse_cpu_limit(value: str, source: str, v2: bool) -> Optional[Fraction]:
    fields = value.split()
    if v2:
        if len(fields) != 2 or not fields[1].isdigit() or int(fields[1]) <= 0:
            raise ProbeError("malformed CPU quota in %s" % source)
        if fields[0] == "max":
            return None
        if not fields[0].isdigit():
            raise ProbeError("malformed CPU quota in %s" % source)
        quota = int(fields[0])
        if quota <= 0:
            raise ProbeError("CPU quota must be positive in %s" % source)
        return Fraction(quota, int(fields[1]))
    raise ProbeError("internal error: v1 CPU quota requires separate files")


def _collect_v2_memory(
    filesystem_root: Path, mount: MountInfo, target: PurePosixPath
) -> Tuple[Optional[int], str]:
    limits: List[Tuple[int, str]] = []
    for logical_dir in _ancestors(target, mount.mount_point):
        logical_file = logical_dir / "memory.max"
        source = str(logical_file)
        value = _read_required(_rooted(filesystem_root, logical_file), source)
        limit = _parse_memory_limit(value, source, allow_max=True)
        if limit is not None:
            limits.append((limit, source))
    if not limits:
        return None, "cgroup-v2:unlimited"
    return min(limits, key=lambda item: item[0])


def _collect_v2_cpu(
    filesystem_root: Path, mount: MountInfo, target: PurePosixPath
) -> Tuple[Optional[Fraction], str]:
    limits: List[Tuple[Fraction, str]] = []
    for logical_dir in _ancestors(target, mount.mount_point):
        logical_file = logical_dir / "cpu.max"
        source = str(logical_file)
        value = _read_required(_rooted(filesystem_root, logical_file), source)
        limit = _parse_cpu_limit(value, source, v2=True)
        if limit is not None:
            limits.append((limit, source))
    if not limits:
        return None, "cgroup-v2:unlimited"
    return min(limits, key=lambda item: item[0])


def _collect_v1_memory(
    filesystem_root: Path, mount: MountInfo, target: PurePosixPath
) -> Tuple[Optional[int], str]:
    limits: List[Tuple[int, str]] = []
    for logical_dir in _ancestors(target, mount.mount_point):
        logical_file = logical_dir / "memory.limit_in_bytes"
        source = str(logical_file)
        value = _read_required(_rooted(filesystem_root, logical_file), source)
        limit = _parse_memory_limit(value, source, allow_max=False)
        if limit is not None:
            limits.append((limit, source))
    if not limits:
        return None, "cgroup-v1:unlimited"
    return min(limits, key=lambda item: item[0])


def _collect_v1_cpu(
    filesystem_root: Path, mount: MountInfo, target: PurePosixPath
) -> Tuple[Optional[Fraction], str]:
    limits: List[Tuple[Fraction, str]] = []
    for logical_dir in _ancestors(target, mount.mount_point):
        quota_file = logical_dir / "cpu.cfs_quota_us"
        period_file = logical_dir / "cpu.cfs_period_us"
        quota_source = str(quota_file)
        period_source = str(period_file)
        quota_value = _read_required(
            _rooted(filesystem_root, quota_file), quota_source
        ).strip()
        period_value = _read_required(
            _rooted(filesystem_root, period_file), period_source
        ).strip()
        if not period_value.isdigit() or int(period_value) <= 0:
            raise ProbeError("malformed CPU period in %s" % period_source)
        if quota_value == "-1":
            continue
        if not quota_value.isdigit() or int(quota_value) <= 0:
            raise ProbeError("malformed CPU quota in %s" % quota_source)
        limits.append(
            (Fraction(int(quota_value), int(period_value)), quota_source)
        )
    if not limits:
        return None, "cgroup-v1:unlimited"
    return min(limits, key=lambda item: item[0])


def _collect_cpuset(
    filesystem_root: Path,
    mount: MountInfo,
    target: PurePosixPath,
    v2: bool,
) -> Tuple[Optional[int], str]:
    names = ("cpuset.cpus.effective", "cpuset.cpus") if v2 else ("cpuset.cpus",)
    saw_file = False
    for logical_dir in _ancestors(target, mount.mount_point):
        for name in names:
            logical_file = logical_dir / name
            source = str(logical_file)
            value = _read_optional(_rooted(filesystem_root, logical_file), source)
            if value is None:
                continue
            saw_file = True
            count = parse_cpuset(value, source)
            if count > 0:
                return count, source
    if saw_file:
        return 0, "cgroup-cpuset:empty"
    return None, "cgroup-cpuset:not-present"


def _detect_host_cpu_count() -> int:
    counts = []
    cpu_count = os.cpu_count()
    if cpu_count is not None and cpu_count > 0:
        counts.append(cpu_count)
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity_count = len(os.sched_getaffinity(0))
        except OSError:
            affinity_count = 0
        if affinity_count > 0:
            counts.append(affinity_count)
    if not counts:
        raise ProbeError("cannot determine the host CPU count")
    return min(counts)


def probe_resources(
    filesystem_root: Path = Path("/"), host_cpu_count: Optional[int] = None
) -> ResourceProbe:
    if host_cpu_count is None:
        host_cpu_count = _detect_host_cpu_count()
    if host_cpu_count <= 0:
        raise ProbeError("host CPU count must be positive")

    meminfo = _read_required(
        _rooted(filesystem_root, PurePosixPath("/proc/meminfo")),
        "/proc/meminfo",
    )
    host_memory = parse_memtotal(meminfo)
    membership = parse_membership(
        _read_required(
            _rooted(filesystem_root, PurePosixPath("/proc/self/cgroup")),
            "/proc/self/cgroup",
        )
    )
    mounts: List[MountInfo] = []
    if membership.unified is not None or membership.controllers:
        mounts = parse_mountinfo(
            _read_required(
                _rooted(filesystem_root, PurePosixPath("/proc/self/mountinfo")),
                "/proc/self/mountinfo",
            )
        )

    if membership.unified is not None and membership.controllers:
        mode = "hybrid"
    elif membership.unified is not None:
        mode = "v2"
    elif membership.controllers:
        mode = "v1"
    else:
        mode = "none"

    if "memory" in membership.controllers:
        mount, target = _mount_for(
            mounts, membership.controllers["memory"], "cgroup", "memory"
        )
        memory_limit, memory_source = _collect_v1_memory(
            filesystem_root, mount, target
        )
    elif membership.unified is not None:
        mount, target = _mount_for(mounts, membership.unified, "cgroup2")
        memory_limit, memory_source = _collect_v2_memory(
            filesystem_root, mount, target
        )
    else:
        memory_limit, memory_source = None, "host-only:no-memory-controller"

    if "cpu" in membership.controllers:
        mount, target = _mount_for(
            mounts, membership.controllers["cpu"], "cgroup", "cpu"
        )
        cpu_quota, cpu_source = _collect_v1_cpu(filesystem_root, mount, target)
    elif membership.unified is not None:
        mount, target = _mount_for(mounts, membership.unified, "cgroup2")
        cpu_quota, cpu_source = _collect_v2_cpu(filesystem_root, mount, target)
    else:
        cpu_quota, cpu_source = None, "host-only:no-cpu-controller"

    if "cpuset" in membership.controllers:
        mount, target = _mount_for(
            mounts, membership.controllers["cpuset"], "cgroup", "cpuset"
        )
        cpuset_count, cpuset_source = _collect_cpuset(
            filesystem_root, mount, target, v2=False
        )
    elif membership.unified is not None:
        mount, target = _mount_for(mounts, membership.unified, "cgroup2")
        cpuset_count, cpuset_source = _collect_cpuset(
            filesystem_root, mount, target, v2=True
        )
    else:
        cpuset_count, cpuset_source = None, "host-only:no-cpuset-controller"

    cpu_candidates = [Fraction(host_cpu_count, 1)]
    if cpu_quota is not None:
        cpu_candidates.append(cpu_quota)
    if cpuset_count is not None:
        cpu_candidates.append(Fraction(cpuset_count, 1))
    effective_cpu = min(cpu_candidates)
    effective_memory = min(
        host_memory,
        host_memory if memory_limit is None else memory_limit,
    )
    return ResourceProbe(
        cgroup_mode=mode,
        host_cpu_cores=host_cpu_count,
        cpu_quota=cpu_quota,
        cpu_quota_source=cpu_source,
        cpuset_cpu_count=cpuset_count,
        cpuset_source=cpuset_source,
        effective_cpu_cores=effective_cpu,
        host_memory_bytes=host_memory,
        memory_limit_bytes=memory_limit,
        memory_limit_source=memory_source,
        effective_memory_bytes=effective_memory,
    )


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return "%d/%d" % (value.numerator, value.denominator)


def check_minimums(
    probe: ResourceProbe, min_cpu_cores: int, min_ram_gib: int
) -> List[str]:
    violations = []
    if probe.effective_cpu_cores < min_cpu_cores:
        millicores = (
            probe.effective_cpu_cores.numerator * 1000
            // probe.effective_cpu_cores.denominator
        )
        violations.append(
            "CPU preparation requires at least %d effective CPU cores; detected %s "
            "cores (%d millicores) after host, cgroup quota, and cpuset limits"
            % (
                min_cpu_cores,
                _format_fraction(probe.effective_cpu_cores),
                millicores,
            )
        )
    required_memory = min_ram_gib * GIB
    if probe.effective_memory_bytes < required_memory:
        violations.append(
            "CPU preparation requires at least %d GiB effective RAM; detected %d bytes "
            "after host and cgroup limits"
            % (min_ram_gib, probe.effective_memory_bytes)
        )
    return violations


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-cpu-cores", required=True, type=int)
    parser.add_argument("--min-ram-gib", required=True, type=int)
    args = parser.parse_args(argv)
    if args.min_cpu_cores <= 0 or args.min_ram_gib <= 0:
        parser.error("minimum CPU cores and RAM must be positive integers")
    try:
        probe = probe_resources()
    except ProbeError as error:
        print("host resource probe failed closed: %s" % error, file=sys.stderr)
        return 2
    print(json.dumps(probe.as_dict(), sort_keys=True, separators=(",", ":")))
    violations = check_minimums(probe, args.min_cpu_cores, args.min_ram_gib)
    for violation in violations:
        print(violation, file=sys.stderr)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
