"""Machine labels and enrollment secrets use the original, still-active session."""

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import delete, select

from vla_control.auth_models import UserSession
from vla_control.common import offset_time
from vla_control.models import Node

from .test_onboarding import request_body


PASSWORD = "management-request-revalidation-password"


@pytest.fixture(params=["rename", "agent-config"])
def management_request(platform, request):
    node = platform.register()
    users, clients = {}, {}
    for name in ("admin-one", "admin-two"):
        users[name] = platform.app.state.auth.create_user(name, PASSWORD, "admin")
        client = TestClient(platform.app, base_url="https://testserver")
        logged = client.post("/api/v1/auth/login", json={"username": name, "password": PASSWORD})
        assert logged.status_code == 200, logged.text
        client.headers["X-CSRF-Token"] = logged.json()["csrf_token"]
        assert client.post("/api/v1/auth/reauth", json={"password": PASSWORD}).status_code == 200
        clients[name] = client
    surface = request.param
    target = "/api/v1/agent-configs" if surface == "agent-config" else f"/api/v1/nodes/{node['node_id']}"
    def send(name="admin-one", **kwargs):
        client = clients[name]
        if surface == "agent-config":
            return client.post(target, json=request_body(), **kwargs)
        return client.patch(target, json={"name": "updated-machine-label"}, **kwargs)
    try:
        yield platform, node, users, clients, surface, target, send
    finally:
        for client in clients.values():
            client.close()


def assert_no_effect(platform, node, response):
    assert "config" not in response.json()
    assert platform.settings.enrollment_token not in response.text
    with platform.app.state.db.read() as session:
        assert session.get(Node, node["node_id"]).display_name is None


def revoke_after_initial_authentication(platform, target, user_id, change, monkeypatch):
    original = platform.app.state.auth.require
    changed = []
    def require(request, permission=None):
        principal = original(request, permission)
        if request.url.path == target and not changed:
            changed.append(True)
            if change in {"disabled", "demoted"}:
                platform.app.state.auth.update_user(None, user_id,
                    {"enabled": False} if change == "disabled" else {"role": "operator"})
            else:
                with platform.app.state.db.write() as session:
                    platform.app.state.auth.lock_accounts(session)
                    if change == "logout":
                        session.execute(delete(UserSession).where(UserSession.user_id == user_id))
                    else:
                        row = session.scalar(select(UserSession).where(UserSession.user_id == user_id))
                        if change == "idle":
                            row.last_activity_at = offset_time(-3601)
                        elif change == "expired":
                            row.expires_at = offset_time(-1)
                        else:
                            row.reauthenticated_at = offset_time(-301)
        return principal
    monkeypatch.setattr(platform.app.state.auth, "require", require)
    return changed


@pytest.mark.parametrize("change", ["disabled", "demoted", "logout", "idle", "expired"])
def test_request_cannot_change_machine_or_return_enrollment_secret_after_revocation(management_request, monkeypatch, change):
    platform, node, users, _, _, target, send = management_request
    changed = revoke_after_initial_authentication(platform, target, users["admin-one"]["id"], change, monkeypatch)
    response = send()
    assert changed and response.status_code in {401, 403}, response.text
    assert_no_effect(platform, node, response)


def test_recent_password_is_rechecked_only_for_sensitive_configuration_generation(management_request, monkeypatch):
    platform, node, users, _, surface, target, send = management_request
    changed = revoke_after_initial_authentication(platform, target, users["admin-one"]["id"], "reauth-expired", monkeypatch)
    response = send()
    assert changed
    if surface == "agent-config":
        assert response.status_code == 403 and response.json()["error"]["code"] == "REAUTH_REQUIRED"
        assert_no_effect(platform, node, response)
    else:
        assert response.status_code == 200, response.text
        with platform.app.state.db.read() as session:
            assert session.get(Node, node["node_id"]).display_name == "updated-machine-label"


def test_old_account_assertion_cannot_use_replacement_administrator_cookie(management_request):
    platform, node, users, _, _, _, send = management_request
    response = send("admin-two", headers={"X-Operation-User": users["admin-one"]["id"]})
    assert response.status_code == 403 and response.json()["error"]["code"] == "ACCOUNT_CHANGED"
    assert_no_effect(platform, node, response)


def test_current_administrator_still_has_the_expected_successful_behavior(management_request):
    platform, node, _, _, surface, _, send = management_request
    response = send()
    assert response.status_code == (201 if surface == "agent-config" else 200), response.text
    assert response.headers["Cache-Control"] == "no-store"
    if surface == "agent-config":
        assert response.json()["config"]["enrollment_token"] == platform.settings.enrollment_token
        with platform.app.state.db.read() as session:
            assert session.get(Node, node["node_id"]).display_name is None
    else:
        with platform.app.state.db.read() as session:
            assert session.get(Node, node["node_id"]).display_name == "updated-machine-label"
