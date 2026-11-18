"""Sample interface counters locally without opening a network connection."""

from pathlib import Path
import socket
import sys
from time import monotonic
from uuid import UUID

import psutil


MAX_INTERFACES = 64
LOOPBACK_NAMES = {"lo", "lo0"}
ROUTE_UP = 0x1
ROUTE_REJECT = 0x200


def _default_route(lines, available, *, ipv6=False):
    candidates = []
    for line in lines:
        fields = line.split()
        try:
            if ipv6:
                destination, prefix, _, source_prefix, _, metric, _, _, flags, name = fields
                if int(destination, 16) or int(prefix, 16) or int(source_prefix, 16):
                    continue
                metric = int(metric, 16)
            else:
                name, destination, _, flags, _, _, metric, mask, *_ = fields
                if int(destination, 16) or int(mask, 16):
                    continue
                metric = int(metric)
            route_flags = int(flags, 16)
        except ValueError:
            continue
        if name in available and route_flags & ROUTE_UP and not route_flags & ROUTE_REJECT:
            candidates.append((metric, name))
    return min(candidates)[1] if candidates else None


def _default_interface(available):
    if not sys.platform.startswith("linux"):
        return None
    for path, ipv6 in (("/proc/net/route", False), ("/proc/net/ipv6_route", True)):
        try:
            with Path(path).open(encoding="utf-8", errors="replace") as lines:
                result = _default_route(lines, available, ipv6=ipv6)
        except OSError:
            continue
        if result is not None:
            return result
    return None


def _boot_id():
    if sys.platform.startswith("linux"):
        try:
            return str(UUID(Path("/proc/sys/kernel/random/boot_id").read_text().strip()))
        except (OSError, ValueError):
            pass
    try:
        return f"boot-time:{int(psutil.boot_time())}"
    except (psutil.Error, OSError, NotImplementedError, ValueError):
        return None


def _ifindex(name):
    try:
        return socket.if_nametoindex(name)
    except OSError:
        # The interface may have disappeared after the counter read.
        return None


def _valid_name(name):
    return (name not in LOOPBACK_NAMES and 0 < len(name) <= 64
            and all(char.isprintable() and not char.isspace() for char in name))


class NetworkMonitor:
    def __init__(self):
        self.boot_id = _boot_id()
        self.previous_at = None
        self.previous = {}

    def sample(self):
        try:
            # Expose actual kernel resets to the rate and history calculations.
            counters = psutil.net_io_counters(pernic=True, nowrap=False)
        except (psutil.Error, OSError, NotImplementedError) as error:
            self.previous_at = None
            self.previous = {}
            return {"boot_id": self.boot_id, "default_interface": None,
                    "interfaces": [], "error": type(error).__name__}

        sampled_at = monotonic()
        elapsed = sampled_at - self.previous_at if self.previous_at is not None else 0
        names = {name for name, value in counters.items() if _valid_name(name)
                 and 0 <= value.bytes_recv < 2**64 and 0 <= value.bytes_sent < 2**64}
        default = _default_interface(names)
        names = sorted(names, key=lambda name: (name != default, name))[:MAX_INTERFACES]
        current = {}
        interfaces = []
        for name in names:
            index = _ifindex(name)
            rx_bytes, tx_bytes = counters[name].bytes_recv, counters[name].bytes_sent
            current[name] = (index, rx_bytes, tx_bytes)
            rx_rate = tx_rate = None
            previous = self.previous.get(name)
            if previous is not None and previous[0] == index and elapsed > 0:
                rx_delta, tx_delta = rx_bytes - previous[1], tx_bytes - previous[2]
                if min(rx_delta, tx_delta) >= 0:
                    rx_rate, tx_rate = rx_delta / elapsed, tx_delta / elapsed
            interfaces.append({"name": name, "ifindex": index, "rx_bytes": rx_bytes, "tx_bytes": tx_bytes,
                               "rx_bytes_per_second": rx_rate, "tx_bytes_per_second": tx_rate})

        self.previous_at = sampled_at
        self.previous = current
        return {"boot_id": self.boot_id, "default_interface": default, "interfaces": interfaces}
