"""Estimate process rates from attributable established TCP sockets on Linux."""

from dataclasses import dataclass
from ipaddress import IPv6Address, ip_address
import os
from pathlib import Path
import re
from selectors import DefaultSelector, EVENT_READ
import shutil
import socket
import subprocess
import sys
from time import monotonic

import psutil


MAX_OUTPUT_BYTES = 2 * 1024 * 1024
MAX_SOCKET_ROWS = 4096
MAX_PROCESSES = 10
COMMAND_TIMEOUT_SECONDS = 2
MAX_INTERVAL_SECONDS = 60
SYSTEM_PATH = "/usr/sbin:/usr/bin:/sbin:/bin"

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


def _process_identity(pid):
    try:
        value = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
        identity, closing, rest = value.rpartition(")")
        _, opening, raw_name = identity.partition("(")
        if not closing or not opening:
            return None
        started = int(rest.split()[19])  # /proc/pid/stat field 22, starttime.
    except (OSError, IndexError, ValueError):
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


class NetworkProcessMonitor:
    def __init__(self):
        self.previous = {}
        self.previous_at = None
        self.previous_partial = False

    def sample(self):
        if not sys.platform.startswith("linux"):
            return {"status": "unavailable", "processes": []}
        try:
            observed, partial = _capture_sockets()
        except (OSError, psutil.Error, subprocess.SubprocessError, NotImplementedError):
            self.previous = {}
            self.previous_at = None
            self.previous_partial = False
            return {"status": "unavailable", "processes": []}
        timestamp = monotonic()
        duration = timestamp - self.previous_at if self.previous_at is not None else 0
        previous, old_partial = self.previous, self.previous_partial
        self.previous, self.previous_at, self.previous_partial = observed, timestamp, partial
        if not 0 < duration <= MAX_INTERVAL_SECONDS:
            return {"status": "warming_up", "processes": [], "partial": partial}
        totals = {}
        for key, current in observed.items():
            earlier = previous.get(key)
            if earlier is None or (earlier.pid, earlier.started) != (current.pid, current.started):
                continue
            rx = current.rx_bytes - earlier.rx_bytes
            tx = current.tx_bytes - earlier.tx_bytes
            if min(rx, tx) < 0 or not (rx or tx):
                continue
            entry = totals.setdefault(current.pid, [current.name, 0, 0])
            entry[1] += rx
            entry[2] += tx
        ranked = sorted(totals.items(), key=lambda item: (-(item[1][1] + item[1][2]), item[0]))[:MAX_PROCESSES]
        return {"status": "ready", "partial": partial or old_partial, "processes": [
            {"pid": pid, "name": name, "rx_bytes_per_second": rx / duration, "tx_bytes_per_second": tx / duration}
            for pid, (name, rx, tx) in ranked
        ]}
