"""Disk capacity warnings use fresh, authenticated filesystem samples."""

import pytest

from vla_control.common import now, offset_time
from vla_control.models import Node
from .conftest import ADMIN

ALERTS = "/api/v1/alerts"


def mount(mountpoint, used, free, total=1000, **overrides):
    return {"mountpoint": mountpoint, "filesystem": "ext4", "total_bytes": total,
            "used_bytes": used, "free_bytes": free, "error": None, **overrides}


def report(platform, node, mounts, sampled_at=None, error=None):
    metrics = {"disk_capacity": {"sampled_at": sampled_at or now(), "mounts": mounts, "error": error}}
    response = platform.client.post("/agent/v1/heartbeat", headers=node["headers"],
                                    json={"node_id": node["node_id"], "metrics": metrics})
    assert response.status_code == 200, response.text


def alerts(platform):
    response = platform.client.get(ALERTS, headers=ADMIN)
    assert response.status_code == 200, response.text
    return response.json()


def test_disk_alarm_includes_boundary_tracks_mount_and_clears_after_recovery(platform):
    node = platform.register(name="storage-worker")
    renamed = platform.client.patch(f"/api/v1/nodes/{node['node_id']}", headers=ADMIN, json={"name": "训练机"})
    assert renamed.status_code == 200, renamed.text
    report(platform, node, [
        mount("/", 320, 680),
        mount("/data1", 908, 92),
        mount("/data2", 810, 90),  # 10% reserved, exactly 90% of usable space.
        mount("/data3", 899, 101),
        mount("/data4", 8999, 1001, total=10_000),  # Rounds to 90.0, still below threshold.
    ])
    high, boundary = alerts(platform)
    assert high["details"]["mountpoint"] == "/data1"
    assert boundary["details"] == {"mountpoint": "/data2", "threshold_percent": 90,
                                   "usage_percent": 90, "total_bytes": 1000, "used_bytes": 810, "free_bytes": 90}
    assert boundary["severity"] == "warning" and boundary["code"] == "DISK_CAPACITY_HIGH"
    assert boundary["node_id"] == node["node_id"] and boundary["run_id"] is None
    assert boundary["message"] == "训练机 的磁盘 /data2 已使用 90%"
    assert alerts(platform) == [high, boundary]

    report(platform, node, [mount("/data1//.", 910, 90), mount("/data2", 800, 100)])
    remaining, = alerts(platform)
    assert remaining["id"] == high["id"] and remaining["details"]["usage_percent"] == 91
    report(platform, node, [mount("/data1", 890, 110)])
    assert alerts(platform) == []


def test_bad_rows_do_not_hide_a_valid_disk_warning_or_create_duplicate(platform):
    node = platform.register()
    report(platform, node, [
        mount("/bad", 1001, 0),
        mount("/bad2", 700, 600),
        mount("/bad3", 0, 0),
        mount("relative", 920, 80),
        mount("/../unsafe", 970, 30),
        mount("/bool", True, 0),
        mount("/large", 1 << 80, 0),
        mount("/broken", 990, 10, error="PermissionError"),
        None,
        mount("/data", 955, 45),
        mount("/data/./", 935, 65),
    ])
    warning, = alerts(platform)
    assert warning["details"]["mountpoint"] == "/data"
    assert warning["details"]["usage_percent"] == 95.5


@pytest.mark.parametrize("reported", [offset_time(-181), offset_time(120), "2026-01-01", "invalid"])
def test_stale_future_or_malformed_sample_does_not_warn(platform, reported):
    node = platform.register()
    report(platform, node, [mount("/data", 980, 20)], sampled_at=reported)
    assert alerts(platform) == []


def test_retired_offline_and_legacy_nodes_do_not_report_disk_alerts(platform):
    node = platform.register()
    assert alerts(platform) == []
    report(platform, node, [mount("/data", 990, 10)])
    for status in ("OFFLINE", "DISCONNECTED", "REMOVED"):
        with platform.app.state.db.write() as session:
            session.get(Node, node["node_id"]).status = status
        assert all(row["code"] != "DISK_CAPACITY_HIGH" for row in alerts(platform))


def test_expired_heartbeat_does_not_report_old_disk_capacity(platform):
    node = platform.register()
    report(platform, node, [mount("/data", 990, 10)])
    with platform.app.state.db.write() as session:
        session.get(Node, node["node_id"]).last_seen = offset_time(-91)
    assert alerts(platform) == []
