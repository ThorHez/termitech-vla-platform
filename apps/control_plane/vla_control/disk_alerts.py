"""Evaluate current local filesystem capacity from authenticated Agent heartbeats."""

from datetime import datetime, timedelta
from pathlib import PurePosixPath

from sqlalchemy import select

from .common import digest
from .remote_models import RemoteConnection, RemoteDeployment

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
    if not 0 < available <= total:
        return None
    return {
        "total_bytes": total, "used_bytes": used, "free_bytes": free,
        "usage_percent": round(100 * used / available, 1),
        "threshold_percent": THRESHOLD_PERCENT,
    }


def _observed_mounts(node, moment, heartbeat_timeout_seconds):
    if node.status not in {"ONLINE", "RECOVERING"}:
        return
    last_seen = _timestamp(node.last_seen)
    if last_seen is None or moment - last_seen > timedelta(seconds=heartbeat_timeout_seconds):
        return
    report = (node.metrics or {}).get("disk_capacity")
    if not isinstance(report, dict):
        return
    sampled_at = _timestamp(report.get("sampled_at"))
    if (sampled_at is None or sampled_at > last_seen + CLOCK_SKEW or moment - sampled_at > MAX_SAMPLE_AGE):
        return
    mounts = report.get("mounts")
    if not isinstance(mounts, list) or len(mounts) > MAX_MOUNTS:
        return

    for row in mounts:
        if not isinstance(row, dict) or row.get("error") is not None:
            continue
        mountpoint = _mountpoint(row.get("mountpoint"))
        details = _capacity(row)
        if mountpoint is not None and details is not None:
            yield mountpoint, details, sampled_at, last_seen


def _machine_keys(session, node_ids):
    """SSH keys may be cloned across machines. Require the endpoint too."""
    query = (select(RemoteDeployment.node_id, RemoteConnection.host, RemoteConnection.port,
                    RemoteConnection.fingerprint, RemoteDeployment.agent_config)
             .join(RemoteConnection, RemoteConnection.id == RemoteDeployment.connection_id)
             .where(RemoteDeployment.node_id.in_(node_ids), RemoteDeployment.state == "SUCCEEDED",
                    RemoteConnection.fingerprint != "")
             .order_by(RemoteDeployment.created_at.desc(), RemoteDeployment.id.desc()))
    keys = {}
    for node_id, host, port, fingerprint, config in session.execute(query):
        if (config or {}).get("_operation", "install") in {"install", "upgrade"}:
            keys.setdefault(node_id, "ssh:" + digest([host.casefold(), port, fingerprint]))
    return keys


def disk_capacity_alerts(session, nodes, moment, heartbeat_timeout_seconds):
    if not nodes:
        return []
    latest = {}
    machine_keys = _machine_keys(session, [node.id for node in nodes])
    for node in nodes:
        machine = machine_keys.get(node.id, "node:" + node.id)
        for mountpoint, details, sampled_at, last_seen in _observed_mounts(node, moment, heartbeat_timeout_seconds):
            key = machine, mountpoint
            order = sampled_at, last_seen, node.id
            if key not in latest or order >= latest[key][0]:
                latest[key] = (order, node, details)

    alerts = []
    for (machine, mountpoint), ((sampled_at, _, _), node, details) in latest.items():
        used, free = details["used_bytes"], details["free_bytes"]
        if used * 100 < (used + free) * THRESHOLD_PERCENT:
            continue
        alerts.append({
            "id": "disk:" + digest([machine, mountpoint]),
            "severity": "warning",
            "code": "DISK_CAPACITY_HIGH",
            "title": "磁盘空间告警",
            "node_id": node.id,
            "run_id": None,
            "message": f"{node.effective_name} 的磁盘 {mountpoint} 已使用 {details['usage_percent']:g}%",
            "timestamp": sampled_at.isoformat(),
            "details": {"mountpoint": mountpoint, **details},
        })
    return sorted(alerts, key=lambda alert: (-alert["details"]["usage_percent"], alert["id"]))
