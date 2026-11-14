"""Evaluate current local filesystem capacity from authenticated Agent heartbeats."""

from datetime import datetime, timedelta
from pathlib import PurePosixPath

from .common import digest

THRESHOLD_PERCENT = 90
MAX_SAMPLE_AGE = timedelta(minutes=3)
CLOCK_SKEW = timedelta(seconds=30)
MAX_MOUNTS = 128
MAX_CAPACITY = (1 << 63) - 1


def _timestamp(value):
    if not isinstance(value, str) or len(value) > 40:
        return None
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError:
        return None
    return stamp if stamp.tzinfo is not None else None


def _mountpoint(value):
    if (not isinstance(value, str) or not value.startswith("/") or len(value) > 4096
            or ".." in value.split("/")
            or any(ord(c) < 32 or 127 <= ord(c) <= 159 or 0xD800 <= ord(c) <= 0xDFFF for c in value)):
        return None
    return str(PurePosixPath("/" + value.lstrip("/")))


def _capacity(row):
    total, used, free = (row.get(key) for key in ("total_bytes", "used_bytes", "free_bytes"))
    if any(type(value) is not int or not 0 <= value <= MAX_CAPACITY for value in (total, used, free)):
        return None
    available = used + free
    if not 0 < available <= total or used * 100 < available * THRESHOLD_PERCENT:
        return None
    return {
        "total_bytes": total, "used_bytes": used, "free_bytes": free,
        "usage_percent": round(100 * used / available, 1),
        "threshold_percent": THRESHOLD_PERCENT,
    }


def disk_capacity_alerts(node, moment, heartbeat_timeout_seconds):
    if node.status not in {"ONLINE", "RECOVERING"}:
        return []
    last_seen = _timestamp(node.last_seen)
    if last_seen is None or moment - last_seen > timedelta(seconds=heartbeat_timeout_seconds):
        return []
    report = (node.metrics or {}).get("disk_capacity")
    if not isinstance(report, dict):
        return []
    sampled_at = _timestamp(report.get("sampled_at"))
    if (sampled_at is None or sampled_at > last_seen + CLOCK_SKEW or moment - sampled_at > MAX_SAMPLE_AGE):
        return []
    mounts = report.get("mounts")
    if not isinstance(mounts, list) or len(mounts) > MAX_MOUNTS:
        return []

    alerts = {}
    for row in mounts:
        if not isinstance(row, dict) or row.get("error") is not None:
            continue
        mountpoint = _mountpoint(row.get("mountpoint"))
        details = _capacity(row)
        if mountpoint is None or details is None:
            continue
        previous = alerts.get(mountpoint)
        if previous and previous["details"]["usage_percent"] >= details["usage_percent"]:
            continue
        alerts[mountpoint] = {
            "id": "disk:" + node.id + ":" + digest(mountpoint),
            "severity": "warning",
            "code": "DISK_CAPACITY_HIGH",
            "title": "磁盘空间告警",
            "node_id": node.id,
            "run_id": None,
            "message": f"{node.effective_name} 的磁盘 {mountpoint} 已使用 {details['usage_percent']:g}%",
            "timestamp": report["sampled_at"],
            "details": {"mountpoint": mountpoint, **details},
        }
    return list(alerts.values())
