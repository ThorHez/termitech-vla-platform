"""Independent routed security and raw-format checks on disposable state."""

from copy import deepcopy
import json

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import func, select

from vla_control.auth_models import User, UserSession
from vla_control.common import offset_time
from vla_control.dataset_template_models import DatasetFormatTemplate
from vla_control.main import create_app
from vla_control.models import Asset, Command, Node, Run
from vla_control.settings import Settings


PASSWORD = "independent-dataset-format-123"
BASE = "/api/v1/dataset-format-templates"
PROFILE = {
    "schema_version": 1, "format": "lerobot", "robot_type": "ARX5", "arm_mode": "single",
    "action_representation": "joint", "action_mode": "absolute", "rotation_representation": "none",
    "cameras": [{"name": "wrist", "key": "observation.images.wrist", "kind": "rgb"}],
    "task_description": "Flatten the cloth.\nKeep the same source layout.", "fps": 20,
    "feature_keys": ["observation.state", "action", "observation.images.wrist"],
    "field_annotations": [
        {"key": "observation.state", "role": "state", "representation": "joint", "value_mode": "absolute",
         "component": "joint", "arm": "left", "shape": [7]},
        {"key": "action", "role": "action", "representation": "joint", "value_mode": "absolute",
         "component": "joint", "arm": "left", "shape": [50, 7], "horizon": 50},
    ],
}


@pytest.fixture
def secured(tmp_path):
    app = create_app(Settings(database_url="sqlite:///" + str(tmp_path / "format-security.db"),
        enrollment_token="isolated-format-agent", session_cookie_secure=False, scheduler_enabled=False))
    for name, role in (("format-admin", "admin"), ("format-writer", "operator"), ("format-reader", "viewer")):
        app.state.auth.create_user(name, PASSWORD, role)
    with TestClient(app) as client:
        yield app, client


def login(client, name="format-admin"):
    client.cookies.clear()
    for key in ("X-CSRF-Token", "X-Dataset-Template-User"):
        client.headers.pop(key, None)
    result = client.post("/api/v1/auth/login", json={"username": name, "password": PASSWORD})
    assert result.status_code == 200, result.text
    value = result.json()
    client.headers["X-CSRF-Token"] = value["csrf_token"]
    client.headers["X-Dataset-Template-User"] = value["user"]["id"]
    return value


def save(client, **changes):
    return client.post(BASE, json={"name": "Same ARX5 format", "profile": deepcopy(PROFILE), **changes})


def count(app):
    with app.state.db.read() as session:
        return session.scalar(select(func.count()).select_from(DatasetFormatTemplate))


def test_templates_require_real_session_and_csrf_before_storage(secured):
    app, client = secured
    assert client.get(BASE).status_code == 401
    assert save(client).status_code == 401
    login(client)
    client.headers.pop("X-CSRF-Token")
    denied = save(client)
    assert denied.status_code == 403 and denied.json()["error"]["code"] == "CSRF_REJECTED"
    assert count(app) == 0


def test_reader_cannot_list_or_mutate_private_formats(secured):
    app, client = secured
    login(client, "format-reader")
    assert client.get(BASE).status_code == 403
    assert save(client).status_code == 403
    assert count(app) == 0


@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "DELETE"])
def test_stale_account_identity_header_rejects_every_operation(secured, method):
    app, client = secured
    first = login(client)
    owned = save(client).json()
    login(client, "format-writer")
    client.headers["X-Dataset-Template-User"] = first["user"]["id"]
    path = BASE if method in {"GET", "POST"} else BASE + "/" + owned["id"]
    body = {"name": "wrong account", "profile": PROFILE} if method == "POST" else (
        {"expected_revision": 1, "name": "wrong account", "profile": PROFILE} if method == "PUT"
        else {"expected_revision": 1} if method == "DELETE" else None)
    result = client.request(method, path, json=body)
    assert result.status_code == 403
    assert result.json()["error"]["code"] == "DATASET_TEMPLATE_USER_CHANGED"
    assert count(app) == 1


def test_admin_cannot_read_rename_or_delete_another_accounts_format(secured):
    app, client = secured
    login(client, "format-writer")
    other = save(client).json()
    login(client, "format-admin")
    assert client.get(BASE).json() == []
    for identity in (other["id"], "missing-template"):
        path = BASE + "/" + identity
        responses = [client.get(path),
            client.put(path, json={"expected_revision": 1, "name": "stolen", "profile": PROFILE}),
            client.request("DELETE", path, json={"expected_revision": 1})]
        assert all(response.status_code == 404 and response.json()["error"]["code"] == "NOT_FOUND"
                   for response in responses)
    assert count(app) == 1


@pytest.mark.parametrize("change", ["disabled", "expired", "idle", "downgraded", "password-change"])
def test_revoked_authentication_cannot_access_existing_templates(secured, change):
    app, client = secured
    current = login(client)
    saved = save(client).json()
    with app.state.db.write() as session:
        user = session.get(User, current["user"]["id"])
        login_row = session.scalar(select(UserSession).where(UserSession.user_id == user.id))
        if change == "disabled":
            user.enabled = False
        elif change == "expired":
            login_row.expires_at = offset_time(-1)
        elif change == "idle":
            login_row.last_activity_at = offset_time(-app.state.settings.session_idle_seconds - 1)
        elif change == "downgraded":
            user.role = "viewer"
        else:
            user.must_change_password = True
    for response in (client.get(BASE), client.get(BASE + "/" + saved["id"]), save(client)):
        assert response.status_code in {401, 403}
    assert count(app) == 1


def test_auth_revocation_while_reading_body_is_checked_before_save(secured):
    app, client = secured
    current = login(client)
    payload = json.dumps({"name": "must not save", "profile": PROFILE}).encode()

    def stream():
        yield payload[:1]
        with app.state.db.write() as session:
            session.get(User, current["user"]["id"]).enabled = False
        yield payload[1:]

    result = client.post(BASE, content=stream(), headers={"Content-Type": "application/json"})
    assert result.status_code == 401
    assert count(app) == 0


@pytest.mark.parametrize("extra", [{"processing": None}, {"path": "/another/machine/data"},
    {"node_id": "another-node"}, {"norm_asset_id": "another-norm"}, {"action_horizon": 50},
    {"processor": {"name": "ExecutableFactory"}}, {"code_dir": "/project"}])
def test_asset_locations_and_training_configuration_cannot_enter_formats(secured, extra):
    app, client = secured
    login(client)
    result = save(client, profile={**PROFILE, **extra})
    assert result.status_code == 422
    assert count(app) == 0


def test_raw_horizon_is_round_tripped_without_creating_training_or_asset_state(secured):
    app, client = secured
    login(client)
    stored = save(client)
    assert stored.status_code == 201, stored.text
    template = stored.json()
    action = template["profile"]["field_annotations"][1]
    assert action["shape"] == [50, 7] and action["horizon"] == 50
    assert "processing" not in template["profile"] and "user_id" not in template
    listed = client.get(BASE)
    assert listed.headers["cache-control"] == "no-store"
    assert all("profile" not in row and "user_id" not in row for row in listed.json())
    assert client.get(BASE + "/" + template["id"]).json() == template
    changed = deepcopy(PROFILE)
    changed["field_annotations"][1]["shape"] = [7]
    changed["field_annotations"][1]["horizon"] = 50
    assert save(client, name="single-frame wrong declaration", profile=changed).status_code == 422
    with app.state.db.read() as session:
        for model in (Asset, Node, Run, Command):
            assert session.scalar(select(func.count()).select_from(model)) == 0


def test_cross_origin_save_is_rejected_without_leaking_profile_values(secured):
    app, client = secured
    login(client)
    result = client.post(BASE, json={"name": "private format", "profile": PROFILE},
                         headers={"Origin": "https://foreign.example"})
    assert result.status_code == 403 and result.json()["error"]["code"] == "ORIGIN_REJECTED"
    assert "Flatten the cloth" not in result.text and count(app) == 0


def test_legacy_admin_token_has_no_private_template_storage(tmp_path):
    app = create_app(Settings(database_url="sqlite:///" + str(tmp_path / "legacy-format.db"),
        enrollment_token="isolated-agent", admin_token="legacy-accountless-admin", legacy_admin_enabled=True,
        session_cookie_secure=False, scheduler_enabled=False))
    with TestClient(app) as client:
        client.headers["Authorization"] = "Bearer legacy-accountless-admin"
        assert client.get("/api/v1/nodes").status_code == 200
        result = save(client)
        assert result.status_code == 403 and result.json()["error"]["code"] == "DATASET_TEMPLATE_ACCOUNT_REQUIRED"
        assert count(app) == 0
