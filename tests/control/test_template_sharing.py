"""Workspace templates are opt-in, readable by peers and editable by their owner."""

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select

from vla_control.auth_models import AuditEvent
from vla_control.main import create_app
from vla_control.settings import Settings

from .test_dataset_templates import PROFILE


PASSWORD = "template-sharing-fixture-123"
TRAINING_CONTENT = {
    "schema_version": 1,
    "request": {"name": "", "node_id": "", "profile_id": "", "template_id": "openpi-sft",
                "inputs": {}, "parameters": {"batch_size": 32}, "gpu_count": 1},
    "step": 0,
    "advanced_recipe": False,
    "edited_parameters": [],
    "edited_stage_parameters": [],
    "entrypoint_edited": False,
    "special_values": [],
}


def login(client, username):
    client.cookies.clear()
    response = client.post("/api/v1/auth/login", json={"username": username, "password": PASSWORD})
    assert response.status_code == 200, response.text
    client.headers["X-CSRF-Token"] = response.json()["csrf_token"]
    return response.json()["user"]["id"]


@pytest.mark.parametrize("route,table,audit_action,config_key,config", [
    ("training-templates", "training_templates", "training_templates.sharing", "content", TRAINING_CONTENT),
    ("dataset-format-templates", "dataset_format_templates", "dataset_templates.sharing", "profile", PROFILE),
])
def test_legacy_templates_can_be_shared_and_revoked_by_their_owner(tmp_path, route, table, audit_action,
                                                                 config_key, config):
    app = create_app(Settings(database_url="sqlite:///" + str(tmp_path / "templates.sqlite3"),
                             enrollment_token="test-agent", scheduler_enabled=False,
                             session_cookie_secure=False))
    for username, role in (("alice", "admin"), ("bob", "operator"), ("viewer", "viewer")):
        app.state.auth.create_user(username, PASSWORD, role)
    path = "/api/v1/" + route
    body = {"name": "同名模板", config_key: config}
    with TestClient(app) as client:
        owner = login(client, "alice")
        response = client.post(path, json=body)
        assert response.status_code == 201, response.text
        original = response.json()
        detail = path + "/" + original["id"]
        sharing = detail + "/sharing"

        # Reproduce the pre-sharing schema with an existing template and live session.
        with app.state.db.engine.begin() as connection:
            connection.exec_driver_sql(f"ALTER TABLE {table} DROP COLUMN shared")
        app.state.db.initialize()
        assert client.get(detail).json() == original
        assert original["shared"] is False and original["is_owner"] is True

        login(client, "bob")
        assert client.get(path).json() == []
        assert client.get(detail).status_code == 404
        login(client, "alice")
        response = client.put(sharing, json={"expected_revision": original["revision"], "shared": True})
        assert response.status_code == 200, response.text
        published = response.json()
        assert published["shared"] is True and published["revision"] == original["revision"] + 1

        peer = login(client, "bob")
        rows = client.get(path).json()
        assert len(rows) == 1 and rows[0]["id"] == original["id"] and config_key not in rows[0]
        assert rows[0]["shared"] is True and rows[0]["is_owner"] is False
        shared = client.get(detail).json()
        assert shared[config_key] == original[config_key] and shared["is_owner"] is False
        revision = {"expected_revision": published["revision"]}
        assert client.put(sharing, json={**revision, "shared": False}).status_code == 404
        assert client.put(detail, json={**revision, **body,
            **({"description": ""} if config_key == "content" else {})}).status_code == 404
        assert client.request("DELETE", detail, json=revision).status_code == 404
        own = client.post(path, json=body).json()
        assert own["id"] != original["id"] and own["shared"] is False
        assert {row["id"] for row in client.get(path).json()} == {original["id"], own["id"]}

        login(client, "viewer")
        assert client.get(detail).status_code == 403
        login(client, "alice")
        response = client.post(path, json=body)
        assert response.status_code == 200, response.text
        updated = response.json()
        assert updated["shared"] is True and updated["id"] == original["id"]
        assert client.put(sharing, json={**revision, "shared": False}).status_code == 409
        response = client.put(sharing, json={"expected_revision": updated["revision"], "shared": False})
        assert response.status_code == 200 and response.json()["shared"] is False
        login(client, "bob")
        assert client.get(detail).status_code == 404
        assert [row["id"] for row in client.get(path).json()] == [own["id"]]

        with app.state.db.read() as session:
            events = list(session.scalars(select(AuditEvent).where(AuditEvent.action == audit_action)))
        assert sum(event.actor_id == owner and event.status == 200 for event in events) == 2
        denial = next(event for event in events if event.actor_id == peer and event.status == 404)
        assert denial.details["target_id"] == original["id"] and "target_name" not in denial.details
        assert all(config_key not in event.details for event in events)
