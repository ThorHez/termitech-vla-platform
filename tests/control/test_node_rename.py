"""A machine label changes only console metadata, independent of Agent lifecycle."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import inspect, select

from vla_control.auth_models import UserSession
from vla_control.db import Database
from vla_control.models import Asset, Command, Node, Run

from .conftest import ADMIN, ENROLL


PASSWORD = "isolated-rename-tests-password"


def rename(platform, node, name, **extra):
    return platform.client.patch(f"/api/v1/nodes/{node['node_id']}", headers=ADMIN,
                                 json={"name": name, **extra})


def node_row(platform, node_id):
    with platform.app.state.db.read() as session:
        return dict(session.execute(select(Node.__table__).where(Node.id == node_id)).mappings().one())


def browser(platform, role="admin"):
    platform.app.state.auth.create_user("rename-" + role, PASSWORD, role)
    client = TestClient(platform.app, base_url="https://testserver")
    response = client.post("/api/v1/auth/login", json={"username": "rename-" + role, "password": PASSWORD})
    assert response.status_code == 200, response.text
    client.headers["X-CSRF-Token"] = response.json()["csrf_token"]
    return client


def test_rename_updates_all_console_reads_and_keeps_identity_and_references(platform):
    node = platform.register(name="agent-host")
    asset = platform.client.post("/api/v1/assets", headers=ADMIN, json={"name": "data", "kind": "DATASET",
        "node_id": node["node_id"], "path": "/machine/dataset", "version": "1"}).json()
    run = platform.run(node)
    platform.schedule()
    before = node_row(platform, node["node_id"])
    commands = platform.commands(node)
    response = rename(platform, node, "  A100 训练机一号  ")
    assert response.status_code == 200, response.text
    assert response.json()["name"] == "A100 训练机一号"
    assert response.json()["hostname"] == "agent-host"
    after = node_row(platform, node["node_id"])
    assert after == {**before, "display_name": "A100 训练机一号"}
    assert platform.commands(node) == commands
    detail = platform.client.get(f"/api/v1/nodes/{node['node_id']}", headers=ADMIN).json()
    assert detail["name"] == "A100 训练机一号"
    assert detail["assets"][0]["id"] == asset["id"]
    assert detail["runs"][0]["id"] == run["id"]
    assert detail["profiles"][0]["node_id"] == node["node_id"]
    assert platform.client.get("/api/v1/nodes", headers=ADMIN).json()[0]["name"] == "A100 训练机一号"
    with platform.app.state.db.read() as session:
        assert session.get(Asset, asset["id"]).node_id == node["node_id"]
        assert session.get(Run, run["id"]).node_id == node["node_id"]


def test_heartbeat_and_idempotent_registration_preserve_custom_label(platform):
    node = platform.register(name="agent-host", key="stable-registration")
    assert rename(platform, node, "custom label").status_code == 200
    platform.heartbeat(node)
    repeated = platform.register(name="agent-host", key="stable-registration")
    assert repeated["node_id"] == node["node_id"] and repeated["token"] == node["token"]
    row = node_row(platform, node["node_id"])
    assert row["name"] == "agent-host" and row["display_name"] == "custom label"
    assert platform.client.get("/api/v1/nodes", headers=ADMIN).json()[0]["name"] == "custom label"


@pytest.mark.parametrize("status", ["OFFLINE", "RECOVERING", "DISCONNECTED", "REMOVED"])
def test_unavailable_or_retired_machine_label_can_be_edited_without_contacting_agent(platform, status):
    node = platform.register()
    with platform.app.state.db.write() as session:
        session.get(Node, node["node_id"]).status = status
    before = node_row(platform, node["node_id"])
    response = rename(platform, node, "离线设备")
    assert response.status_code == 200 and response.json()["status"] == status
    assert node_row(platform, node["node_id"]) == {**before, "display_name": "离线设备"}
    with platform.app.state.db.read() as session:
        assert list(session.scalars(select(Command))) == []
    if status in {"OFFLINE", "RECOVERING"}:
        assert platform.client.get("/api/v1/alerts", headers=ADMIN).json()[0]["message"] == f"离线设备: {status}"


def test_duplicate_labels_are_allowed_because_machine_id_remains_identity(platform):
    first, second = platform.register(), platform.register(name="other")
    assert rename(platform, first, "同一名称").status_code == 200
    assert rename(platform, second, "同一名称").status_code == 200
    rows = platform.client.get("/api/v1/nodes", headers=ADMIN).json()
    assert {row["id"] for row in rows} == {first["node_id"], second["node_id"]}
    assert [row["name"] for row in rows] == ["同一名称", "同一名称"]


@pytest.mark.parametrize("name", ["", " \t\n", "x" * 201, None, 12, [], "middle\nname", "a\x00b", "a\x1fb", "a\x7fb", "a\x85b", "a\x9fb"])
def test_invalid_name_cannot_modify_machine(platform, name):
    node = platform.register()
    before = node_row(platform, node["node_id"])
    response = rename(platform, node, name)
    assert response.status_code == 422, response.text
    assert node_row(platform, node["node_id"]) == before


def test_name_length_applies_after_trimming_and_no_other_fields_are_mutable(platform):
    node = platform.register()
    assert rename(platform, node, "  " + "机" * 200 + "  ").status_code == 200
    before = node_row(platform, node["node_id"])
    response = rename(platform, node, "new label", hostname="different-host", roots=["/"])
    assert response.status_code == 422
    assert node_row(platform, node["node_id"]) == before
    missing = platform.client.patch("/api/v1/nodes/missing", headers=ADMIN, json={"name": "label"})
    assert missing.status_code == 404


@pytest.mark.parametrize("role,status", [("admin", 200), ("operator", 403), ("viewer", 403)])
def test_rename_requires_machine_management_permission_but_not_password_reauthentication(platform, role, status):
    node = platform.register()
    client = browser(platform, role)
    try:
        assert client.get("/api/v1/auth/me").json()["reauth_expires_at"] is None
        response = client.patch(f"/api/v1/nodes/{node['node_id']}", json={"name": "allowed label"})
        assert response.status_code == status, response.text
        if role == "admin":
            # Connection removal retains its independent recent-password gate.
            response = client.post(f"/api/v1/nodes/{node['node_id']}/disconnect", json={})
            assert response.status_code == 403 and response.json()["error"]["code"] == "REAUTH_REQUIRED"
        else:
            assert response.json()["error"]["code"] == "FORBIDDEN"
            assert node_row(platform, node["node_id"])["display_name"] is None
    finally:
        client.close()


def test_rename_requires_browser_csrf_and_rejects_agent_and_anonymous_credentials(platform):
    node = platform.register()
    path = f"/api/v1/nodes/{node['node_id']}"
    for headers in ({}, ENROLL, node["headers"]):
        response = platform.client.patch(path, headers=headers, json={"name": "unauthorized label"})
        assert response.status_code == 401
    client = browser(platform)
    try:
        csrf = client.headers.pop("X-CSRF-Token")
        response = client.patch(path, json={"name": "missing csrf"})
        assert response.status_code == 403 and response.json()["error"]["code"] == "CSRF_REJECTED"
        response = client.patch(path, headers={"X-CSRF-Token": csrf, "Origin": "https://foreign.example"},
                                json={"name": "foreign origin"})
        assert response.status_code == 403 and response.json()["error"]["code"] == "ORIGIN_REJECTED"
        assert node_row(platform, node["node_id"])["display_name"] is None
    finally:
        client.close()


def test_concurrent_additive_label_migration_preserves_sessions_and_existing_machine_rows(platform):
    node = platform.register(name="legacy-name")
    asset = platform.client.post("/api/v1/assets", headers=ADMIN, json={"name": "data", "kind": "DATASET",
        "node_id": node["node_id"], "path": "/machine/dataset", "version": "1"}).json()
    run = platform.run(node)
    client = browser(platform)
    try:
        before = node_row(platform, node["node_id"])
        with platform.app.state.db.read() as session:
            session_before = dict(session.execute(select(UserSession.__table__)).mappings().one())
        with platform.app.state.db.engine.begin() as connection:
            connection.exec_driver_sql("ALTER TABLE nodes DROP COLUMN display_name")
        barrier = Barrier(2)

        def migrate(_):
            separate = Database(platform.settings.database_url)
            try:
                barrier.wait(timeout=5)
                separate.initialize()
            finally:
                separate.close()

        with ThreadPoolExecutor(max_workers=2) as workers:
            list(workers.map(migrate, range(2)))
        assert "display_name" in {column["name"] for column in inspect(platform.app.state.db.engine).get_columns("nodes")}
        assert node_row(platform, node["node_id"]) == before
        with platform.app.state.db.read() as session:
            assert dict(session.execute(select(UserSession.__table__)).mappings().one()) == session_before
            assert session.get(Asset, asset["id"]).node_id == node["node_id"]
            assert session.get(Run, run["id"]).node_id == node["node_id"]
        assert client.get("/api/v1/auth/me").status_code == 200
        assert client.get(f"/api/v1/nodes/{node['node_id']}").json()["name"] == "legacy-name"
        assert client.patch(f"/api/v1/nodes/{node['node_id']}", json={"name": "upgraded label"}).status_code == 200
        platform.app.state.db.initialize()
        assert client.get(f"/api/v1/nodes/{node['node_id']}").json()["name"] == "upgraded label"
        assert client.get("/api/v1/auth/me").status_code == 200
    finally:
        client.close()
