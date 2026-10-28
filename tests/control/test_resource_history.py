"""Machine history is durable, bounded and independent of training payloads."""

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import delete, func, inspect, select

import vla_control.main as main_module
import vla_control.resource_history as resource_history
from vla_control.main import create_app
from vla_control.models import Node, NodeResourceSample

from .conftest import ADMIN, gpu


END = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)


def timestamp(moment):
    return moment.isoformat(timespec="microseconds")


def clear_samples(platform):
    with platform.app.state.db.write() as session:
        session.execute(delete(NodeResourceSample))


def heartbeat_at(platform, node, monkeypatch, moment, metrics=None, gpus=None, **extra):
    monkeypatch.setattr(main_module, "now", lambda: timestamp(moment))
    return platform.client.post("/agent/v1/heartbeat", headers=node["headers"], json={
        "node_id": node["node_id"], "metrics": metrics or {}, "gpus": [] if gpus is None else gpus, **extra})


def get_history(platform, node, monkeypatch, window="24h", end=END, headers=ADMIN):
    monkeypatch.setattr(resource_history, "now", lambda: timestamp(end))
    return platform.client.get(f"/api/v1/nodes/{node['node_id']}/resource-history", headers=headers,
                               params={"window": window})


def samples(platform):
    with platform.app.state.db.read() as session:
        return list(session.scalars(select(NodeResourceSample).order_by(NodeResourceSample.timestamp)))


def test_accepted_heartbeat_persists_only_resource_scalars_and_survives_reopen(platform, monkeypatch):
    node = platform.register()
    clear_samples(platform)
    result = heartbeat_at(platform, node, monkeypatch, END - timedelta(seconds=15), metrics={
        "cpu_percent": 0, "memory_percent": 25, "disk_total_bytes": 1000, "disk_free_bytes": 400,
        "runtime_checks": [{"secret": "do-not-store"}], "collection_errors": {"path": "private"},
    }, gpus=[gpu(memory_total_mb=100, memory_used_mb=30, utilization_pct=45,
                 processes=[{"pid": 1234, "private": "do-not-store"}])])
    assert result.status_code == 200, result.text
    stored = samples(platform)
    assert len(stored) == 1
    assert stored[0].timestamp == timestamp(END - timedelta(seconds=15))
    assert stored[0].metrics == {"cpu_percent": 0, "memory_percent": 25, "disk_percent": 60,
        "gpus": [{"uuid": "GPU-1", "name": "test gpu", "utilization_pct": 45, "memory_percent": 30}]}
    app = create_app(platform.settings)
    monkeypatch.setattr(resource_history, "now", lambda: timestamp(END))
    with TestClient(app) as client:
        response = client.get(f"/api/v1/nodes/{node['node_id']}/resource-history?window=1h", headers=ADMIN)
        assert response.status_code == 200
        last = response.json()["points"][-1]
        assert last["cpu_percent"] == 0
        assert last["gpus"][0]["memory_percent"] == 30
        assert "do-not-store" not in response.text
        assert "node_resource_samples" in inspect(app.state.db.engine).get_table_names()


def test_rate_limit_and_transaction_rejection_do_not_record_extra_samples(platform, monkeypatch):
    node = platform.register()
    clear_samples(platform)
    for seconds in (0, 1, 29, 30, 31):
        assert heartbeat_at(platform, node, monkeypatch, END + timedelta(seconds=seconds),
                            metrics={"cpu_percent": seconds}).status_code == 200
    stored = samples(platform)
    assert [sample.metrics["cpu_percent"] for sample in stored] == [0, 30]
    response = heartbeat_at(platform, node, monkeypatch, END + timedelta(seconds=60),
        active_attempts=[{"attempt_id": "unknown-attempt", "pid": 3, "state": "RUNNING"}])
    assert response.status_code == 403
    assert len(samples(platform)) == 2
    other = platform.register(name="other")
    response = platform.client.post("/agent/v1/heartbeat", headers=node["headers"], json={"node_id": other["node_id"]})
    assert response.status_code == 403


def test_bucket_averages_gaps_and_missing_values_are_not_zero(platform, monkeypatch):
    node = platform.register()
    clear_samples(platform)
    for seconds, cpu in ((-110, 20), (-80, 80)):
        assert heartbeat_at(platform, node, monkeypatch, END + timedelta(seconds=seconds),
            metrics={"cpu_percent": cpu, "memory_percent": None, "disk_total_bytes": 0, "disk_free_bytes": 0},
            gpus=[gpu(name=None, memory_total_mb=100, memory_used_mb=None, utilization_pct=cpu)]).status_code == 200
    result = get_history(platform, node, monkeypatch, "6h").json()
    assert len(result["points"]) == 360
    average, gap = result["points"][-2:]
    assert average["sample_count"] == 2
    assert average["cpu_percent"] == 50
    assert average["memory_percent"] is None
    assert average["disk_percent"] is None
    assert average["gpus"] == [{"uuid": "GPU-1", "name": "GPU-1", "utilization_pct": 50, "memory_percent": None}]
    assert gap["sample_count"] == 0
    assert gap["cpu_percent"] is None
    assert gap["gpus"] == []


def test_invalid_metrics_are_omitted_without_losing_valid_zero_or_gpu_identity(platform, monkeypatch):
    node = platform.register()
    clear_samples(platform)
    assert heartbeat_at(platform, node, monkeypatch, END - timedelta(seconds=10), metrics={
        "cpu_percent": "100", "memory_percent": True, "disk_total_bytes": 100, "disk_free_bytes": 200},
        gpus=[gpu(memory_total_mb=0, memory_used_mb=0, utilization_pct=0)]).status_code == 200
    point = get_history(platform, node, monkeypatch, "1h").json()["points"][-1]
    assert point["cpu_percent"] is None
    assert point["memory_percent"] is None
    assert point["disk_percent"] is None
    assert point["gpus"][0]["memory_percent"] is None
    assert point["gpus"][0]["utilization_pct"] == 0
    assert resource_history.percent(float("nan")) is None
    assert resource_history.percent(float("inf")) is None
    assert resource_history.percent(-1) is None
    assert resource_history.percent(101) is None
    assert resource_history.percent(10 ** 1000) is None
    assert heartbeat_at(platform, node, monkeypatch, END + timedelta(seconds=30), metrics={
        "cpu_percent": 10 ** 1000, "disk_total_bytes": 10 ** 1000, "disk_free_bytes": 5 * 10 ** 999}).status_code == 200
    assert samples(platform)[-1].metrics["disk_percent"] is None
    assert heartbeat_at(platform, node, monkeypatch, END + timedelta(seconds=60), metrics={
        "disk_total_bytes": 10 ** 1000, "disk_free_bytes": 1.0}).status_code == 200
    assert samples(platform)[-1].metrics["disk_percent"] is None


@pytest.mark.parametrize("window,count,bucket", [("1h", 120, 30), ("6h", 360, 60), ("24h", 288, 300),
                                                ("7d", 336, 1800), ("30d", 360, 7200)])
def test_time_windows_are_bounded_and_empty_history_is_not_fabricated(platform, monkeypatch, window, count, bucket):
    node = platform.register()
    clear_samples(platform)
    empty = get_history(platform, node, monkeypatch, window).json()
    assert empty["points"] == []
    assert empty["retention_days"] == 30
    assert empty["sample_interval_seconds"] == 30
    assert empty["bucket_seconds"] == bucket
    assert heartbeat_at(platform, node, monkeypatch, END, metrics={"cpu_percent": 1}).status_code == 200
    result = get_history(platform, node, monkeypatch, window).json()
    assert len(result["points"]) == count
    assert sum(point["sample_count"] for point in result["points"]) == 1
    assert result["points"] == sorted(result["points"], key=lambda item: item["timestamp"])


def test_retention_excludes_expired_samples_and_cleanup_is_global_and_bounded(platform, monkeypatch):
    active = platform.register(name="active")
    retired = platform.register(name="retired")
    clear_samples(platform)
    with platform.app.state.db.write() as session:
        session.get(Node, retired["node_id"]).status = "REMOVED"
        for index in range(resource_history.CLEANUP_BATCH_SIZE + 2):
            session.add(NodeResourceSample(node_id=retired["node_id"],
                timestamp=timestamp(END - timedelta(days=31, seconds=index * 30)),
                metrics={"cpu_percent": 100, "memory_percent": None, "disk_percent": None, "gpus": []}))
    assert get_history(platform, retired, monkeypatch, "30d").json()["points"] == []
    assert heartbeat_at(platform, active, monkeypatch, END, metrics={"cpu_percent": 5}).status_code == 200
    with platform.app.state.db.read() as session:
        remaining = session.scalar(select(func.count()).select_from(NodeResourceSample).where(
            NodeResourceSample.node_id == retired["node_id"]))
    assert remaining == 2
    assert get_history(platform, active, monkeypatch, "30d").json()["points"][-1]["cpu_percent"] == 5


@pytest.mark.parametrize("state", ["OFFLINE", "DISCONNECTED", "REMOVED"])
def test_history_survives_machine_lifecycle_and_stays_scoped(platform, monkeypatch, state):
    first, second = platform.register(name="first"), platform.register(name="second")
    clear_samples(platform)
    assert heartbeat_at(platform, first, monkeypatch, END - timedelta(seconds=5), metrics={"cpu_percent": 12}).status_code == 200
    assert heartbeat_at(platform, second, monkeypatch, END - timedelta(seconds=5), metrics={"cpu_percent": 94}).status_code == 200
    with platform.app.state.db.write() as session:
        session.get(Node, first["node_id"]).status = state
    first_points = get_history(platform, first, monkeypatch, "1h").json()["points"]
    assert first_points[-1]["cpu_percent"] == 12
    assert get_history(platform, second, monkeypatch, "1h").json()["points"][-1]["cpu_percent"] == 94


def test_history_is_read_only_platform_permission_and_validates_inputs(platform, monkeypatch):
    node = platform.register()
    url = f"/api/v1/nodes/{node['node_id']}/resource-history"
    assert platform.client.get(url).status_code == 401
    assert platform.client.get(url, headers=node["headers"]).status_code == 401
    assert platform.client.get(url, headers=ADMIN, params={"window": "365d"}).status_code == 422
    assert platform.client.get("/api/v1/nodes/missing/resource-history", headers=ADMIN).status_code == 404
    platform.app.state.auth.create_user("history-viewer", "history-viewer-password-123", "viewer")
    client = TestClient(platform.app, base_url="https://testserver")
    try:
        response = client.post("/api/v1/auth/login", json={"username": "history-viewer", "password": "history-viewer-password-123"})
        assert response.status_code == 200
        assert client.get(url).status_code == 200
    finally:
        client.close()
