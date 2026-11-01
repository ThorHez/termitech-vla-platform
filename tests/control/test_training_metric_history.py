"""Metric history stays complete even when training logs exceed the live tail."""

import math

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import inspect, select

from vla_control.common import now, uid
from vla_control.models import Attempt, Event

from .conftest import ADMIN


def scheduled(platform):
    node = platform.register()
    run = platform.run(node)
    platform.schedule()
    attempt = platform.get_run(run)["attempts"][0]
    return run, attempt


def add_events(platform, attempt, rows):
    with platform.app.state.db.write() as session:
        session.add_all(Event(attempt_id=attempt["id"], seq=seq, type=kind,
                              timestamp=now(), data=data, digest="a" * 64)
                        for seq, kind, data in rows)


def metrics(platform, run, **params):
    response = platform.client.get(f"/api/v1/runs/{run['id']}/metrics", headers=ADMIN, params=params)
    assert response.status_code == 200, response.text
    return response.json()


def next_attempt(platform, run, number):
    item = {"id": uid(), "number": number}
    with platform.app.state.db.write() as session:
        session.add(Attempt(id=item["id"], run_id=run["id"], node_id=run["node_id"], number=number,
                            execution_token=uid(), created_at=now(), state="FAILED"))
    return item


def test_early_metrics_survive_more_than_5000_log_records(platform):
    run, attempt = scheduled(platform)
    add_events(platform, attempt, [(1, "METRIC", {"step": 0, "loss": 0.9})]
               + [(seq, "LOG", {"text": "training output"}) for seq in range(2, 6003)]
               + [(6003, "METRIC", {"step": 20700, "loss": 0.0035})])
    first = metrics(platform, run, limit=1)
    assert first == {"attempt_id": attempt["id"], "items": [first["items"][0]],
                     "next_seq": 1, "has_more": True, "total_count": 2}
    assert first["items"][0]["data"] == {"step": 0, "loss": 0.9}
    second = metrics(platform, run, after_seq=first["next_seq"], limit=1)
    assert [row["seq"] for row in second["items"]] == [6003]
    assert second["items"][0]["data"]["step"] == 20700
    assert second["next_seq"] == 6003 and not second["has_more"] and second["total_count"] == 2
    assert all(row["type"] == "METRIC" for row in first["items"] + second["items"])
    assert metrics(platform, run, after_seq=6003)["items"] == []


def test_metric_only_pagination_is_lossless_and_bounded(platform):
    run, attempt = scheduled(platform)
    add_events(platform, attempt, [(seq, "METRIC", {"step": seq, "loss": 1 / seq}) for seq in range(1, 2003)])
    cursor, loaded, page_sizes = 0, [], []
    while True:
        page = metrics(platform, run, after_seq=cursor)
        assert page["attempt_id"] == attempt["id"] and page["total_count"] == 2002
        loaded.extend(row["seq"] for row in page["items"])
        page_sizes.append(len(page["items"]))
        cursor = page["next_seq"]
        if not page["has_more"]:
            break
    assert loaded == list(range(1, 2003)) and page_sizes == [1000, 1000, 2]


def test_total_count_exposes_late_history_below_a_live_cursor(platform):
    run, attempt = scheduled(platform)
    add_events(platform, attempt, [(10, "METRIC", {"step": 900, "loss": 0.2})])
    assert metrics(platform, run)["total_count"] == 1
    # Agent event ingestion accepts late earlier sequence numbers. A client can
    # notice the changed count and reload instead of silently losing that point.
    add_events(platform, attempt, [(2, "METRIC", {"step": 0, "loss": 0.9})])
    update = metrics(platform, run, after_seq=10)
    assert update["items"] == [] and update["total_count"] == 2 and update["next_seq"] == 10
    assert [row["data"]["step"] for row in metrics(platform, run)["items"]] == [0, 900]


def test_attempt_cursor_never_mixes_restarts_and_other_runs(platform):
    run, old = scheduled(platform)
    add_events(platform, old, [(1, "METRIC", {"step": 100, "loss": 9})])
    latest = next_attempt(platform, run, 2)
    add_events(platform, latest, [(1, "METRIC", {"step": 0, "loss": 0.8}),
                                  (2, "METRIC", {"step": 100, "loss": 0.7})])
    assert metrics(platform, run)["attempt_id"] == latest["id"]
    assert [row["data"]["loss"] for row in metrics(platform, run)["items"]] == [0.8, 0.7]
    explicit = metrics(platform, run, attempt_id=old["id"])
    assert explicit["attempt_id"] == old["id"] and explicit["items"][0]["data"]["loss"] == 9
    assert not metrics(platform, run, attempt_id=old["id"], after_seq=1)["items"]

    other = platform.run(platform.register(name="other-worker"))
    platform.schedule()
    foreign_attempt = platform.get_run(other)["attempts"][0]
    response = platform.client.get(f"/api/v1/runs/{run['id']}/metrics", headers=ADMIN,
                                   params={"attempt_id": foreign_attempt["id"]})
    assert response.status_code == 404


def test_unstarted_and_empty_attempts_return_stable_empty_pages(platform):
    run = platform.run(platform.register())
    assert metrics(platform, run) == {"attempt_id": None, "items": [], "next_seq": 0,
                                      "has_more": False, "total_count": 0}
    platform.schedule()
    attempt = platform.get_run(run)["attempts"][0]
    assert metrics(platform, run, after_seq=15) == {"attempt_id": attempt["id"], "items": [],
                                                  "next_seq": 15, "has_more": False, "total_count": 0}


def test_sparse_invalid_zero_and_nonfinite_metrics_are_not_dropped(platform):
    run, attempt = scheduled(platform)
    values = [{"step": 0, "loss": 0, "other": -0.5}, {"step": 1, "loss": None},
              {"step": 2, "other": 1.2}, {"step": 3, "loss": "NaN", "enabled": False},
              {"step": 4, "loss": math.nan}, {"step": 5, "loss": math.inf, "other": -math.inf}]
    add_events(platform, attempt, [(seq, "METRIC", value) for seq, value in enumerate(values, 1)])
    page = metrics(platform, run)
    assert page["total_count"] == 6
    assert [row["data"] for row in page["items"]] == values[:4] + [
        {"step": 4, "loss": None}, {"step": 5, "loss": None, "other": None}]


@pytest.mark.parametrize("params", [{"after_seq": -1}, {"after_seq": 2**63}, {"after_seq": "1.2"},
                                  {"limit": 0}, {"limit": 1001}, {"attempt_id": ""},
                                  {"attempt_id": "x" * 37}])
def test_invalid_history_parameters_are_rejected(platform, params):
    response = platform.client.get("/api/v1/runs/missing/metrics", headers=ADMIN, params=params)
    assert response.status_code == 422


def test_metric_history_permissions_match_other_read_only_run_routes(platform):
    run, _ = scheduled(platform)
    path = f"/api/v1/runs/{run['id']}/metrics"
    assert platform.client.get(path).status_code == 401
    platform.app.state.auth.create_user("metric-viewer", "metric-history-password-123", "viewer")
    with TestClient(platform.app, base_url="https://testserver") as viewer:
        login = viewer.post("/api/v1/auth/login", json={"username": "metric-viewer",
                                                       "password": "metric-history-password-123"})
        assert login.status_code == 200, login.text
        assert viewer.get(path).status_code == 200
        assert viewer.get("/api/v1/runs/missing/metrics").status_code == 404


def test_existing_database_gets_metric_index_without_changing_events(platform):
    run, attempt = scheduled(platform)
    add_events(platform, attempt, [(1, "LOG", {"text": "log"}),
                                  (2, "METRIC", {"step": 0, "loss": 0})])
    db = platform.app.state.db
    with db.read() as session:
        before = [(row.attempt_id, row.seq, row.digest, row.data) for row in session.scalars(select(Event))]
    with db.engine.begin() as connection:
        connection.exec_driver_sql("DROP INDEX ix_events_attempt_type_seq")
    db.initialize()
    index = next(item for item in inspect(db.engine).get_indexes("events")
                 if item["name"] == "ix_events_attempt_type_seq")
    assert index["column_names"] == ["attempt_id", "type", "seq"]
    with db.read() as session:
        assert [(row.attempt_id, row.seq, row.digest, row.data) for row in session.scalars(select(Event))] == before
    assert metrics(platform, run)["items"][0]["data"] == {"step": 0, "loss": 0}
