"""Every human API is protected, including direct calls outside the web UI."""

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from vla_control.authorization import AUTH_ROUTES, PERMISSIONS, permission_for
from vla_control.main import create_app
from vla_control.settings import Settings
from vla_control.users import create_initial_admin


PASSWORD = "isolated-admin-password-123"


@pytest.fixture
def secured(tmp_path):
    app = create_app(Settings(database_url="sqlite:///" + str(tmp_path / "permissions.db"),
                              enrollment_token="isolated-enrollment", scheduler_enabled=False,
                              session_cookie_secure=False))
    create_initial_admin(app.state.db, "initial-admin", PASSWORD)
    with TestClient(app) as client:
        yield app, client


def login(client, username="initial-admin", password=PASSWORD):
    client.cookies.clear()
    result = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert result.status_code == 200, result.text
    client.headers["X-CSRF-Token"] = result.json()["csrf_token"]
    return result.json()


def account(client, role):
    login(client)
    proof = client.post("/api/v1/auth/reauth", json={"password": PASSWORD})
    assert proof.status_code == 200, proof.text
    result = client.post("/api/v1/users", json={"username": role + "-user", "password": PASSWORD, "role": role})
    assert result.status_code == 201, result.text
    login(client, role + "-user")
    result = client.post("/api/v1/auth/password", json={"current_password": PASSWORD, "new_password": PASSWORD + "-changed"})
    assert result.status_code == 200, result.text
    client.headers["X-CSRF-Token"] = result.json()["csrf_token"]


def concrete(path):
    import re
    return re.sub(r"\{[^}]+\}", "missing-resource", path)


def test_all_registered_human_routes_have_an_explicit_permission(secured):
    app, _ = secured
    registered = {(method, route.path) for route in app.routes if isinstance(route, APIRoute)
                  and route.path.startswith("/api/v1/") for method in route.methods}
    assert registered == set(PERMISSIONS) | AUTH_ROUTES
    for method, path in registered:
        assert permission_for(method, path) is not None or (method, path) in AUTH_ROUTES


def test_unmapped_management_route_fails_closed_even_for_admin(secured):
    app, client = secured

    @app.post("/api/v1/forgot-to-protect")
    def unclassified():
        pytest.fail("An unclassified endpoint must never execute")

    login(client)
    result = client.post("/api/v1/forgot-to-protect")
    assert result.status_code == 403
    assert result.json()["error"]["code"] == "PERMISSION_NOT_CONFIGURED"


@pytest.mark.parametrize("role", ["viewer", "operator"])
def test_roles_cannot_call_restricted_routes_directly(secured, role):
    _, client = secured
    account(client, role)
    allowed = {"platform.read"} if role == "viewer" else {"platform.read", "training.write", "assets.write", "recipes.write"}
    for (method, path), permission in PERMISSIONS.items():
        if permission not in allowed:
            result = client.request(method, concrete(path), json={} if method != "GET" else None)
            assert result.status_code == 403, (role, method, path, result.text)
            assert result.json()["error"]["code"] == "FORBIDDEN"


def test_operator_training_asset_recipe_requests_reach_validation(secured):
    _, client = secured
    account(client, "operator")
    for (method, path), permission in PERMISSIONS.items():
        if permission in {"training.write", "assets.write", "recipes.write"}:
            result = client.request(method, concrete(path), json={})
            assert result.status_code in {404, 409, 422}, (method, path, result.text)


def test_cookie_and_agent_credentials_are_separate_and_denials_are_audited(secured):
    _, client = secured
    authentication = login(client)
    # A human session cannot enroll an Agent or acknowledge machine events.
    assert client.post("/agent/v1/register", json={}).status_code == 401
    assert client.post("/agent/v1/events/batch", json={}).status_code == 401
    client.headers.pop("X-CSRF-Token")
    denied = client.post("/api/v1/assets", json={"password": "never-record-this"})
    assert denied.status_code == 403
    client.headers["X-CSRF-Token"] = authentication["csrf_token"]
    audit = client.get("/api/v1/audit")
    assert audit.status_code == 200
    assert any(row["path"] == "/api/v1/assets" and row["status"] == 403 and row["actor_username"] == "initial-admin"
               for row in audit.json())
    assert "never-record-this" not in audit.text
    assert client.get("/api/v1/overview").headers["cache-control"] == "no-store"


def test_proxy_root_path_cannot_bypass_authentication_or_permissions(secured):
    app, _ = secured
    app.state.auth.create_user("prefix-viewer", PASSWORD, "viewer")
    app.state.auth.create_user("prefix-operator", PASSWORD, "operator", force_change=True)
    # No second lifespan: this is the same isolated database and running app.
    client = TestClient(app, root_path="/platform")
    try:
        prefix = "/platform/api/v1"
        assert client.get(prefix + "/overview").status_code == 401
        assert client.post(prefix + "/assets", json={}).status_code == 401
        response = client.post(prefix + "/auth/login", json={"username": "prefix-viewer", "password": PASSWORD})
        assert response.status_code == 200
        assert "Path=/platform/api/v1" in response.headers["set-cookie"]
        client.headers["X-CSRF-Token"] = response.json()["csrf_token"]
        assert client.get(prefix + "/overview").status_code == 200
        assert client.get(prefix + "/overview").headers["cache-control"] == "no-store"
        assert client.get(prefix + "/users").status_code == 403
        assert client.post(prefix + "/assets", json={}).status_code == 403
        assert client.post(prefix + "/auth/logout").status_code == 204
        assert not client.cookies.get("vla_session")
        response = client.post(prefix + "/auth/login", json={"username": "prefix-operator", "password": PASSWORD})
        assert response.status_code == 200
        client.headers["X-CSRF-Token"] = response.json()["csrf_token"]
        assert client.get(prefix + "/auth/me").status_code == 200
        assert client.get(prefix + "/overview").json()["error"]["code"] == "PASSWORD_CHANGE_REQUIRED"
        response = client.post(prefix + "/auth/password", json={"current_password": PASSWORD, "new_password": PASSWORD + "changed"})
        assert response.status_code == 200
        client.headers["X-CSRF-Token"] = response.json()["csrf_token"]
        assert client.post(prefix + "/run-previews", json={}).status_code == 422
        assert client.post(prefix + "/remote-deployments", json={}).status_code == 403
    finally:
        client.close()
