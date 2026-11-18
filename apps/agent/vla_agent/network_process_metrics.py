"""Estimate process rates and daily usage from attributable TCP sockets on Linux."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from ipaddress import IPv6Address, ip_address
import json
import os
from pathlib import Path
import re
from selectors import DefaultSelector, EVENT_READ
import shutil
import socket
import stat
import subprocess
import sys
from tempfile import NamedTemporaryFile
from time import monotonic
from uuid import uuid4

import psutil

from .network_metrics import _boot_id


MAX_OUTPUT_BYTES = 2 * 1024 * 1024
MAX_SOCKET_ROWS = 4096
MAX_PROCESSES = 10
MAX_DAY_PROCESSES = 4096
MAX_DAY_FILE_BYTES = 2 * 1024 * 1024
COMMAND_TIMEOUT_SECONDS = 2
MAX_INTERVAL_SECONDS = 60
SYSTEM_PATH = "/usr/sbin:/usr/bin:/sbin:/bin"
DAY_TIMEZONE = timezone(timedelta(hours=8), "Asia/Shanghai")

_OWNER = r'\("(?:[^"\\]|\\.)*",pid=(\d{1,10}),fd=(\d{1,10})\)'
_OWNERS = re.compile(rf"(?:^|\s)users:\(({_OWNER}(?:,{_OWNER})*)\)")
_OWNER_ROW = re.compile(_OWNER)
_SOCKET_ROW = re.compile(r"^\s*(?:ESTAB\s+)?\d+\s+\d+\s+(\S+)\s+(\S+)\s+(.*)$")
_SOCKET_FIELDS = re.compile(r"(?:^|\s)(ino|sk|bytes_received|bytes_acked):([0-9a-fA-F]{1,20})(?=\s|$)")


@dataclass(frozen=True)
class SocketSample:
    pid: int
    started: int
    name: str
    rx_bytes: int
    tx_bytes: int


def _ss_snapshot():
    executable = shutil.which("ss", path=SYSTEM_PATH)
    if executable is None:
        raise OSError("ss unavailable")
    deadline = monotonic() + COMMAND_TIMEOUT_SECONDS
    with subprocess.Popen([executable, "-H", "-O", "-t", "-n", "-i", "-p", "-e", "state", "established"],
                          stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          env={"LC_ALL": "C", "PATH": SYSTEM_PATH}) as process:
        try:
            data = bytearray()
            with DefaultSelector() as selector:
                selector.register(process.stdout, EVENT_READ)
                while True:
                    remaining = deadline - monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise TimeoutError("ss timed out")
                    chunk = os.read(process.stdout.fileno(), 65536)
                    if not chunk:
                        break
                    if len(data) + len(chunk) > MAX_OUTPUT_BYTES:
                        raise OSError("ss output exceeded limit")
                    data.extend(chunk)
            if process.wait(timeout=max(0, deadline - monotonic())):
                raise OSError("ss returned an error")
            return data.decode("utf-8", errors="replace")
        finally:
            if process.poll() is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass


def _address(value):
    address = ip_address(value.strip("[]").split("%", 1)[0])
    if isinstance(address, IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


def _local_addresses():
    local = set()
    for interfaces in psutil.net_if_addrs().values():
        for interface in interfaces:
            if interface.family in (socket.AF_INET, socket.AF_INET6):
                try:
                    local.add(_address(interface.address))
                except ValueError:
                    continue
    return local


def _process_identity(pid, *, strict=False):
    try:
        value = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
        identity, closing, rest = value.rpartition(")")
        _, opening, raw_name = identity.partition("(")
        if not closing or not opening:
            raise ValueError("Process stat malformed")
        fields = rest.split()
        started = int(fields[19])  # /proc/pid/stat field 22, starttime.
        if fields[0] in {"Z", "X", "x"}:
            return None
    except (FileNotFoundError, ProcessLookupError):
        return None
    except (OSError, IndexError, ValueError):
        if strict:
            raise
        return None
    name = re.sub(r"[^A-Za-z0-9_.+@-]", "_", raw_name)[:64]
    return started, name if re.search(r"[A-Za-z]", name) else f"process-{pid}"


def _owns_socket(pid, fd, inode):
    try:
        return os.readlink(f"/proc/{pid}/fd/{fd}") == f"socket:[{inode}]"
    except OSError:
        return False


def _capture_sockets():
    rows = _ss_snapshot().splitlines()
    if len(rows) > MAX_SOCKET_ROWS:
        raise OSError("ss socket row limit exceeded")
    local = _local_addresses()
    samples = {}
    identities = {}
    partial = False
    for line in rows:
        row = _SOCKET_ROW.fullmatch(line)
        if row is None:
            if line.strip():
                partial = True
            continue
        source, peer, tail = row.groups()
        try:
            local_ip = _address(source.rsplit(":", 1)[0])
            remote_ip = _address(peer.rsplit(":", 1)[0])
        except ValueError:
            partial = True
            continue
        if local_ip.is_loopback or remote_ip.is_loopback or remote_ip in local:
            continue
        owners = _OWNERS.search(tail)
        if owners is None:
            partial = True
            continue
        entries = {(int(pid), int(fd)) for pid, fd in _OWNER_ROW.findall(owners.group(1))}
        pids = {pid for pid, _ in entries}
        stats = dict(_SOCKET_FIELDS.findall(tail[owners.end():]))
        if len(pids) != 1 or "ino" not in stats or "sk" not in stats:
            partial = True
            continue
        pid = pids.pop()
        try:
            inode = int(stats["ino"])
            cookie = int(stats["sk"], 16)
            rx_bytes = int(stats.get("bytes_received", "0"))
            tx_bytes = int(stats.get("bytes_acked", "0"))
        except ValueError:
            partial = True
            continue
        if not (0 < pid < 2**31 and inode > 0 and 0 < cookie < 2**64 - 1
                and 0 <= rx_bytes < 2**64 and 0 <= tx_bytes < 2**64
                and any(_owns_socket(pid, fd, inode) for _, fd in entries)):
            partial = True
            continue
        if pid not in identities:
            identities[pid] = _process_identity(pid)
        identity = identities[pid]
        if identity is None:
            partial = True
            continue
        started, name = identity
        samples[(cookie, inode, source, peer)] = SocketSample(pid, started, name, rx_bytes, tx_bytes)
    return samples, partial


def _today():
    return datetime.now(DAY_TIMEZONE).date().isoformat()


def _stored_total(row):
    return (
        isinstance(row, dict)
        and isinstance(row.get("instance_id"), str)
        and re.fullmatch(r"[0-9a-f]{32}", row["instance_id"])
        and type(row.get("pid")) is int and 0 < row["pid"] < 2**31
        and isinstance(row.get("name"), str) and len(row["name"]) <= 64
        and re.fullmatch(r"[A-Za-z0-9_.+@-]+", row["name"])
        and re.search(r"[A-Za-z]", row["name"])
        and all(type(row.get(key)) is int and 0 <= row[key] < 2**128 for key in ("rx_bytes", "tx_bytes"))
    )


class _DailyProcessUsage:
    def __init__(self, state_dir):
        self.path = Path(state_dir) / "network-process-today.json" if state_dir is not None else None
        self.boot_id = _boot_id() or uuid4().hex
        self.date = None
        self.rows = {}
        self.dirty = False

    def _instance_id(self, pid, started):
        return sha256(f"{self.boot_id}:{pid}:{started}".encode()).hexdigest()[:32]

    def begin_day(self, date):
        if date == self.date:
            return
        previous_day = self.date
        self.date = date
        self.rows = {}
        self.dirty = previous_day is not None
        if previous_day is not None or self.path is None:
            return
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as stream:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    return
                data = stream.read(MAX_DAY_FILE_BYTES + 1)
            if len(data) > MAX_DAY_FILE_BYTES:
                return
            record = json.loads(data)
        except (OSError, ValueError, RecursionError):
            return
        if not isinstance(record, dict) or record.get("date") != date:
            self.dirty = True
            return
        records = record.get("processes")
        if not isinstance(records, list):
            return
        for row in records[:MAX_DAY_PROCESSES]:
            if _stored_total(row):
                values = {key: row[key] for key in ("instance_id", "pid", "name", "rx_bytes", "tx_bytes")}
                self.rows[values["instance_id"]] = values

    def add(self, totals):
        for (pid, started), (name, rx, tx) in totals.items():
            instance_id = self._instance_id(pid, started)
            row = self.rows.get(instance_id)
            if row is None:
                row = {"instance_id": instance_id, "pid": pid, "name": name,
                       "rx_bytes": 0, "tx_bytes": 0}
                self.rows[instance_id] = row
            row["name"] = name
            row["rx_bytes"] += rx
            row["tx_bytes"] += tx
            self.dirty = True

    def _persist(self):
        if not self.dirty or self.path is None:
            return
        temporary = None
        try:
            with NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".network-process-today-",
                                    dir=self.path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                json.dump({"date": self.date, "processes": list(self.rows.values())},
                          stream, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except OSError:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
        else:
            self.dirty = False

    def _active(self, row):
        try:
            current = _process_identity(row["pid"], strict=True)
        except (OSError, IndexError, ValueError):
            return None
        return current is not None and self._instance_id(row["pid"], current[0]) == row["instance_id"]

    def report(self):
        ordered = sorted(self.rows.values(), key=lambda row: (
            -(row["rx_bytes"] + row["tx_bytes"]), row["pid"], row["instance_id"],
        ))
        if len(ordered) > MAX_DAY_PROCESSES:
            self.rows = {row["instance_id"]: row for row in ordered[:MAX_DAY_PROCESSES]}
        self._persist()
        # The initial baseline and intervals that straddle midnight may omit bytes.
        return {"date": self.date, "partial": True,
                "processes": [{**row, "active": self._active(row)}
                              for row in ordered[:min(MAX_PROCESSES, MAX_DAY_PROCESSES)]]}


class NetworkProcessMonitor:
    def __init__(self, state_dir=None):
        self.previous = {}
        self.previous_at = None
        self.previous_day = None
        self.previous_partial = False
        self.daily = _DailyProcessUsage(state_dir)

    def sample(self):
        if not sys.platform.startswith("linux"):
            return {"status": "unavailable", "processes": []}
        capture_day = _today()
        try:
            observed, partial = _capture_sockets()
        except (OSError, psutil.Error, subprocess.SubprocessError, NotImplementedError):
            self.previous = {}
            self.previous_at = None
            self.previous_day = None
            self.previous_partial = False
            self.daily.begin_day(_today())
            return {"status": "unavailable", "processes": [], "today": self.daily.report()}
        timestamp = monotonic()
        duration = timestamp - self.previous_at if self.previous_at is not None else 0
        previous, old_partial = self.previous, self.previous_partial
        today, previous_day = _today(), self.previous_day
        self.previous_day = today if capture_day == today else None
        self.daily.begin_day(today)
        self.previous, self.previous_at, self.previous_partial = observed, timestamp, partial
        if not 0 < duration <= MAX_INTERVAL_SECONDS:
            return {"status": "warming_up", "processes": [], "partial": partial, "today": self.daily.report()}
        totals = {}
        for key, current in observed.items():
            earlier = previous.get(key)
            if earlier is None or (earlier.pid, earlier.started) != (current.pid, current.started):
                continue
            rx = current.rx_bytes - earlier.rx_bytes
            tx = current.tx_bytes - earlier.tx_bytes
            if min(rx, tx) < 0 or not (rx or tx):
                continue
            entry = totals.setdefault((current.pid, current.started), [current.name, 0, 0])
            entry[1] += rx
            entry[2] += tx
        if capture_day == today == previous_day:
            self.daily.add(totals)
        ranked = sorted(totals.items(), key=lambda item: (-(item[1][1] + item[1][2]), item[0]))[:MAX_PROCESSES]
        return {"status": "ready", "partial": partial or old_partial, "today": self.daily.report(), "processes": [
            {"pid": pid, "name": name, "rx_bytes_per_second": rx / duration, "tx_bytes_per_second": tx / duration}
            for (pid, _), (name, rx, tx) in ranked
        ]}
