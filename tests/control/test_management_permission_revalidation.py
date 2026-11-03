"""Sensitive management commands bind their actor after fresh authorization."""

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import delete, event, select

from vla_control.auth_models import UserSession
from vla_control.common import offset_time
from vla_control.models import Command

from .test_multiuser_task_permissions import team  # noqa: F401
from .test_profile_additions import body as add_body, capable
from .test_profile_rebindings import body as rebind_body
from .test_recipe_details import register
from .test_recipe_writeback import saved
from .test_storage_configurations import body as storage_body


PASSWORD = "management-revalidation-test-password"


@pytest.fixture(params=["storage", "profile", "add-profile", "writeback"])
def management(platform, request):
    surface = request.param
    node = register(platform)
    if surface == "storage":
        path = f"/api/v1/nodes/{node['node_id']}/storage-configurations"
        body, kind, permission = storage_body(), "STORAGE_CONFIGURE", "machines.manage"
    elif surface == "writeback":
        _, recipe, _ = saved(platform, node)
        path = f"/api/v1/recipe-revisions/{recipe['latest_revision']['id']}/writeback"
        body, kind, permission = {"config_name": "pi0_management_check"}, "RECIPE_WRITE", "recipes.writeback"
    else:
        path = f"/api/v1/nodes/{node['node_id']}/" + ("profiles" if surface == "add-profile" else "profile-rebindings")
        body, kind, permission = (add_body() if surface == "add-profile" else rebind_body()), "PROFILE_REBIND", "machines.manage"
        if surface == "add-profile":
            capable(platform, node)
    users, clients = {}, {}
    for name in ("admin-one", "admin-two"):
        users[name] = platform.app.state.auth.create_user(name, PASSWORD, "admin")
        client = TestClient(platform.app, base_url="https://testserver")
        login = client.post("/api/v1/auth/login", json={"username": name, "password": PASSWORD})
        assert login.status_code == 200, login.text
        client.headers["X-CSRF-Token"] = login.json()["csrf_token"]
        assert client.post("/api/v1/auth/reauth", json={"password": PASSWORD}).status_code == 200
        clients[name] = client
    try:
        yield platform, path, body, kind, permission, users, clients, node
    finally:
        for client in clients.values():
            client.close()


@pytest.mark.parametrize("change", ["disabled", "demoted", "logout", "idle", "expired", "reauth-expired"])
def test_sensitive_management_rechecks_identity_and_proof_before_creating_commands(management, monkeypatch, change):
    platform, path, body, kind, _, users, clients, _ = management
    user_id = users["admin-one"]["id"]
    original = platform.app.state.auth.require
    changed = []
    def revoke_after_dependency(request, permission=None):
        principal = original(request, permission)
        if request.url.path == path and not changed:
            changed.append(True)
            if change in {"disabled", "demoted"}:
                platform.app.state.auth.update_user(None, user_id,
                    {"enabled": False} if change == "disabled" else {"role": "operator"})
            else:
                with platform.app.state.db.write() as session:
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
    monkeypatch.setattr(platform.app.state.auth, "require", revoke_after_dependency)
    response = clients["admin-one"].post(path, json=body)
    assert changed and response.status_code in {401, 403}, response.text
    if change == "reauth-expired":
        assert response.json()["error"]["code"] == "REAUTH_REQUIRED"
    with platform.app.state.db.read() as session:
        assert list(session.scalars(select(Command).where(Command.type == kind))) == []


def test_new_management_command_binds_actor_and_dedup_does_not_transfer_it(management):
    platform, path, body, kind, permission, users, clients, _ = management
    first = clients["admin-one"].post(path, json=body)
    assert first.status_code == 202, first.text
    second = clients["admin-two"].post(path, json=body)
    assert second.status_code == 202 and second.json()["id"] == first.json()["id"], second.text
    with platform.app.state.db.read() as session:
        commands = list(session.scalars(select(Command).where(Command.type == kind)))
        assert len(commands) == 1
        assert commands[0].actor_user_id == users["admin-one"]["id"]
        assert commands[0].required_permission == permission


def test_logout_takes_account_guard_before_removing_the_session(team):
    platform, _, _, clients = team
    statements = []
    def record(_connection, _cursor, statement, _parameters, _context, _many):
        statements.append(statement.lower())
    event.listen(platform.app.state.db.engine, "before_cursor_execute", record)
    try:
        assert clients["alice"].post("/api/v1/auth/logout").status_code == 204
    finally:
        event.remove(platform.app.state.db.engine, "before_cursor_execute", record)
    guard = next(index for index, statement in enumerate(statements) if "from auth_guard" in statement)
    removal = next(index for index, statement in enumerate(statements) if statement.startswith("delete from user_sessions"))
    assert guard < removal


@pytest.mark.parametrize("management", ["writeback"], indirect=True)
@pytest.mark.parametrize("denial", ["role-revoked", "actor-unknown"])
def test_writeback_can_be_resubmitted_after_undelivered_authorization_rejection(management, denial):
    platform, path, body, _, _, users, clients, node = management
    first = clients["admin-one"].post(path, json=body)
    assert first.status_code == 202, first.text
    first_id = first.json()["id"]
    if denial == "role-revoked":
        platform.app.state.auth.update_user(None, users["admin-one"]["id"], {"role": "operator"})
    else:
        with platform.app.state.db.write() as session:
            session.get(Command, first_id).actor_user_id = None
    assert first_id not in {row["id"] for row in platform.commands(node)}
    with platform.app.state.db.read() as session:
        command = session.get(Command, first_id)
        assert command.status == "REJECTED" and command.delivered_at is None
        assert command.result["error"]["code"] == (
            "COMMAND_AUTHORIZATION_REVOKED" if denial == "role-revoked" else "COMMAND_ACTOR_REQUIRED")
    platform.app.state.auth.update_user(None, users["admin-one"]["id"], {"role": "admin"})
    client = clients["admin-one"]
    client.cookies.clear()
    login = client.post("/api/v1/auth/login", json={"username": "admin-one", "password": PASSWORD})
    assert login.status_code == 200, login.text
    client.headers["X-CSRF-Token"] = login.json()["csrf_token"]
    assert client.post("/api/v1/auth/reauth", json={"password": PASSWORD}).status_code == 200
    second = client.post(path, json=body)
    assert second.status_code == 202 and second.json()["id"] != first_id, second.text
    with platform.app.state.db.read() as session:
        old = session.get(Command, first_id)
        fresh = session.get(Command, second.json()["id"])
        assert old.status == "REJECTED" and old.delivered_at is None
        assert old.actor_user_id == (users["admin-one"]["id"] if denial == "role-revoked" else None)
        assert fresh.actor_user_id == users["admin-one"]["id"]
        assert fresh.status == "PENDING"


@pytest.mark.parametrize("management", ["writeback"], indirect=True)
@pytest.mark.parametrize("uncertain", ["delivered", "commit-receipt", "different-error"])
def test_writeback_does_not_reissue_delivered_or_uncertain_rejection(management, uncertain):
    from vla_control.common import now
    platform, path, body, _, _, users, clients, _ = management
    first = clients["admin-one"].post(path, json=body)
    assert first.status_code == 202, first.text
    with platform.app.state.db.write() as session:
        command = session.get(Command, first.json()["id"])
        command.status, command.acknowledged_at = "REJECTED", now()
        command.result = {"error": {"code": "COMMAND_AUTHORIZATION_REVOKED", "message": "Rejected"}}
        if uncertain == "delivered":
            command.delivered_at = now()
        elif uncertain == "commit-receipt":
            command.result = {**command.result, "commit_sha": "a" * 40}
        else:
            command.result = {"error": {"code": "RECIPE_SOURCE_CHANGED", "message": "Source changed"}}
    repeated = clients["admin-two"].post(path, json=body)
    assert repeated.status_code == 202 and repeated.json()["id"] == first.json()["id"], repeated.text
    with platform.app.state.db.read() as session:
        rows = list(session.scalars(select(Command).where(Command.type == "RECIPE_WRITE")))
        assert len(rows) == 1 and rows[0].actor_user_id == users["admin-one"]["id"]
