"""Bounded machine resource snapshots and portable time-bucket aggregation.

History reflects accepted heartbeats, including collection failures. No process,
environment, command, or training payload is copied into the history table.
"""

from datetime import datetime, timedelta
import math
from typing import Literal

from fastapi import Query
from sqlalchemy import delete, select, tuple_

from .common import APIError, now
from .models import Node, NodeResourceSample


SAMPLE_INTERVAL_SECONDS = 30
RETENTION_DAYS = 30
CLEANUP_BATCH_SIZE = 1000
WINDOWS = {
    "1h": (3600, 30),
    "6h": (6 * 3600, 60),
    "24h": (24 * 3600, 300),
    "7d": (7 * 86400, 1800),
    "30d": (30 * 86400, 7200),
}
SCALAR_METRICS = ("cpu_percent", "memory_percent", "disk_percent")
GPU_METRICS = ("utilization_pct", "memory_percent")


def number(value):
    if type(value) not in (int, float):
        return None
    # Reject unbounded JSON integers as unavailable rather than allowing float
    # division/conversion to fail the entire heartbeat transaction later.
    try:
        return value if math.isfinite(float(value)) else None
    except OverflowError:
        return None


def percent(value):
    value = number(value)
    return value if value is not None and 0 <= value <= 100 else None


def ratio_percent(used, total):
    used, total = number(used), number(total)
    if used is None or total is None or total <= 0 or not 0 <= used <= total:
        return None
    return 100 * (used / total)


def snapshot(metrics, gpus):
    total, free = number(metrics.get("disk_total_bytes")), number(metrics.get("disk_free_bytes"))
    disk_percent = None
    if total is not None and free is not None and total > 0 and 0 <= free <= total:
        disk_percent = 100 * (1 - free / total)
    return {
        "cpu_percent": percent(metrics.get("cpu_percent")),
        "memory_percent": percent(metrics.get("memory_percent")),
        "disk_percent": disk_percent,
        "gpus": [{
            "uuid": gpu["uuid"], "name": gpu.get("name") or gpu["uuid"],
            "utilization_pct": percent(gpu.get("utilization_pct")),
            "memory_percent": ratio_percent(gpu.get("memory_used_mb"), gpu.get("memory_total_mb")),
        } for gpu in gpus],
    }


def capture_sample(session, node):
    """Call while holding the heartbeat's node lock, in its transaction."""
    timestamp = node.last_seen
    moment = datetime.fromisoformat(timestamp)
    latest = session.scalar(select(NodeResourceSample.timestamp).where(
        NodeResourceSample.node_id == node.id).order_by(NodeResourceSample.timestamp.desc()).limit(1))
    if latest is not None and (moment - datetime.fromisoformat(latest)).total_seconds() < SAMPLE_INTERVAL_SECONDS:
        return
    session.add(NodeResourceSample(node_id=node.id, timestamp=timestamp,
                                   metrics=snapshot(node.metrics, node.gpus)))
    cutoff = (moment - timedelta(days=RETENTION_DAYS)).isoformat(timespec="microseconds")
    # An indexed global batch also ages out disconnected nodes while any Agent
    # is reporting. Work stays bounded independently of total historical size.
    expired = list(session.execute(select(NodeResourceSample.node_id, NodeResourceSample.timestamp).where(
        NodeResourceSample.timestamp < cutoff
    ).order_by(NodeResourceSample.timestamp).limit(CLEANUP_BATCH_SIZE)))
    if expired:
        session.execute(delete(NodeResourceSample).where(
            tuple_(NodeResourceSample.node_id, NodeResourceSample.timestamp).in_(expired)))


def _accumulator(keys):
    return {key: [0.0, 0] for key in keys}


def _add(accumulator, value):
    for key, pair in accumulator.items():
        metric = percent(value.get(key))
        if metric is not None:
            pair[0] += metric
            pair[1] += 1


def _means(accumulator):
    return {key: round(total / count, 3) if count else None for key, (total, count) in accumulator.items()}


def history(session, node_id, window, end):
    if session.get(Node, node_id) is None:
        raise APIError(404, "NOT_FOUND", "Node not found")
    duration, bucket_seconds = WINDOWS[window]
    end_moment = datetime.fromisoformat(end)
    start_moment = end_moment - timedelta(seconds=duration)
    start = start_moment.isoformat(timespec="microseconds")
    bucket_count = math.ceil(duration / bucket_seconds)
    buckets = {}
    # At most one sample per 30 seconds and a 30-day window bound this scan;
    # streaming avoids materializing every snapshot in application memory.
    samples = session.scalars(select(NodeResourceSample).where(
        NodeResourceSample.node_id == node_id,
        NodeResourceSample.timestamp >= start, NodeResourceSample.timestamp <= end,
    ).order_by(NodeResourceSample.timestamp).execution_options(yield_per=1000))
    for sample in samples:
        index = min(bucket_count - 1, int((datetime.fromisoformat(sample.timestamp) - start_moment).total_seconds() // bucket_seconds))
        bucket = buckets.setdefault(index, {"sample_count": 0, "metrics": _accumulator(SCALAR_METRICS), "gpus": {}})
        bucket["sample_count"] += 1
        _add(bucket["metrics"], sample.metrics)
        for gpu in sample.metrics["gpus"]:
            gpu_bucket = bucket["gpus"].setdefault(gpu["uuid"], {
                "uuid": gpu["uuid"], "name": gpu["name"], "metrics": _accumulator(GPU_METRICS)})
            gpu_bucket["name"] = gpu["name"]
            _add(gpu_bucket["metrics"], gpu)
    points = []
    if buckets:
        for index in range(bucket_count):
            bucket = buckets.get(index)
            points.append({
                "timestamp": (start_moment + timedelta(seconds=index * bucket_seconds)).isoformat(timespec="microseconds"),
                "sample_count": bucket["sample_count"] if bucket else 0,
                **(_means(bucket["metrics"]) if bucket else dict.fromkeys(SCALAR_METRICS)),
                "gpus": [{"uuid": gpu["uuid"], "name": gpu["name"], **_means(gpu["metrics"])}
                         for gpu in sorted(bucket["gpus"].values(), key=lambda item: item["uuid"])] if bucket else [],
            })
    return {"node_id": node_id, "window": window, "start": start, "end": end,
            "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS, "bucket_seconds": bucket_seconds,
            "retention_days": RETENTION_DAYS, "points": points}


def register_routes(app, dependencies, db):
    @app.get("/api/v1/nodes/{node_id}/resource-history", dependencies=dependencies)
    def resource_history(node_id: str, window: Literal["1h", "6h", "24h", "7d", "30d"] = Query(default="24h")):
        with db.read() as session:
            return history(session, node_id, window, now())
