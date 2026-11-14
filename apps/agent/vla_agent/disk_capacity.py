"""Sample native filesystems without waiting on storage in the heartbeat loop."""

from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys
from threading import Thread
from time import monotonic

import psutil

from .state import utc_now


SAMPLE_INTERVAL = 60
SAMPLE_TIMEOUT = 10
MAX_MOUNTS = 128
MAX_ALIASES = 4
LOCAL_FILESYSTEMS = frozenset({
    "apfs", "bcachefs", "btrfs", "exfat", "ext2", "ext3", "ext4", "f2fs", "hfs", "hfsplus", "jfs",
    "msdos", "ntfs", "ntfs3", "reiserfs", "ufs", "vfat", "xfs", "zfs",
})
MOUNT_ESCAPE = re.compile(r"\\([0-7]{3})")


@dataclass(frozen=True)
class Mount:
    path: str
    filesystem: str
    device: str


def _linux_mounts(lines):
    for line in lines:
        left, separator, right = line.partition(" - ")
        fields, detail = left.split(), right.split()
        if not separator or len(fields) < 6 or not detail:
            continue
        mountpoint = MOUNT_ESCAPE.sub(lambda match: chr(int(match[1], 8)), fields[4])
        filesystem = detail[0]
        if filesystem in LOCAL_FILESYSTEMS or (mountpoint == "/" and filesystem == "overlay"):
            yield Mount(mountpoint, filesystem, fields[2])


def _mounts():
    if sys.platform.startswith("linux"):
        with Path("/proc/self/mountinfo").open(encoding="utf-8", errors="surrogateescape") as lines:
            return list(_linux_mounts(lines))
    return [Mount(part.mountpoint, part.fstype, part.device) for part in psutil.disk_partitions()
            if part.fstype in LOCAL_FILESYSTEMS]


def _disk_bytes(path):
    details = psutil.disk_usage(path)
    if details.total <= 0 or details.used < 0 or details.free < 0 or details.used + details.free > details.total:
        raise ValueError("invalid disk capacity")
    return {"total_bytes": details.total, "used_bytes": details.used, "free_bytes": details.free}


def _read_mount(mount):
    result = {"mountpoint": mount.path, "filesystem": mount.filesystem}
    try:
        if sys.platform.startswith("linux"):
            device = os.stat(mount.path).st_dev
            if f"{os.major(device)}:{os.minor(device)}" != mount.device:
                raise OSError("mount changed")
        return {**result, **_disk_bytes(mount.path), "error": None}
    except (psutil.Error, OSError, ValueError) as exc:
        return {**result, "total_bytes": None, "used_bytes": None, "free_bytes": None, "error": type(exc).__name__}


def _local_disks():
    mounts = {}
    # Choose a stable, shallow representative for bind mounts and repeated devices.
    for mount in sorted(_mounts(), key=lambda item: (item.path.rstrip("/").count("/"), item.path)):
        if (not mount.path.startswith("/") or len(mount.path) > 4096
                or any(ord(char) < 32 or 127 <= ord(char) <= 159 or 0xD800 <= ord(char) <= 0xDFFF for char in mount.path)):
            continue
        aliases = mounts.setdefault((mount.device, mount.filesystem), [])
        if len(aliases) < MAX_ALIASES and mount not in aliases:
            aliases.append(mount)
    disks = {}
    for aliases in list(mounts.values())[:MAX_MOUNTS]:
        first = _read_mount(aliases[0])
        result = first
        if first["error"]:
            for alias in aliases[1:]:
                candidate = _read_mount(alias)
                if candidate["error"] is None:
                    result = candidate
                    break
        mountpoint = result["mountpoint"]
        if mountpoint not in disks or disks[mountpoint]["error"] is not None:
            disks[mountpoint] = result
    error = "NoLocalMounts" if not disks else "MountLimitReached" if len(mounts) > MAX_MOUNTS else None
    return {"mounts": list(disks.values()), "error": error}


class _DiskSample:
    """One in-flight reader, including when a kernel stat call stalls."""

    def __init__(self, collect):
        self.collect = collect
        self.worker = None
        self.started = None
        self.started_at = None
        self.key = None
        self.result = None

    def read(self, key=None):
        current = monotonic()
        if self.worker is not None and not self.worker.is_alive():
            self.worker = None
        completed, _ = self.result or (None, None)
        if self.started is not None and current - self.started > SAMPLE_TIMEOUT and completed != self.started:
            self.result = (self.started, {"sampled_at": self.started_at, "error": "CollectionTimeout"})
        if self.worker is None and (self.started is None or key != self.key or current - self.started >= SAMPLE_INTERVAL):
            started_at = utc_now()
            self.started = current
            self.started_at = started_at
            if key != self.key:
                self.result = None
            self.key = key

            def collect():
                try:
                    values = self.collect() if key is None else self.collect(key)
                except (psutil.Error, OSError, ValueError) as exc:
                    values = {"error": type(exc).__name__}
                if monotonic() - current <= SAMPLE_TIMEOUT:
                    self.result = (current, {"sampled_at": started_at, **values})

            self.worker = Thread(target=collect, daemon=True, name="vla-disk-sample")
            try:
                self.worker.start()
            except RuntimeError:
                self.worker = None
                self.result = (current, {"sampled_at": started_at, "error": "CollectionUnavailable"})
        if self.key != key:
            return None
        return self.result[1] if self.result is not None else None


class DiskCapacityMonitor:
    def __init__(self):
        self.local = _DiskSample(_local_disks)
        # Keep legacy output history, even if it resides on a separate network mount.
        self.output = _DiskSample(_disk_bytes)

    def heartbeat_metrics(self, output_root):
        metrics = {}
        if (local := self.local.read()) is not None:
            metrics["disk_capacity"] = {"mounts": [], "error": None, **local}
        if (output := self.output.read(output_root)) is not None:
            metrics["disk_total_bytes"] = output.get("total_bytes")
            metrics["disk_free_bytes"] = output.get("free_bytes")
            if output.get("error"):
                metrics["collection_errors"] = {"disk": output["error"]}
        return metrics
