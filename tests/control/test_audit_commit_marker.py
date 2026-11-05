"""Explicit audit events suppress fallback only after their outer commit."""

from types import SimpleNamespace

import pytest
from sqlalchemy import delete, select
from starlette.requests import Request
from starlette.websockets import WebSocket

from vla_control.auth_models import AuditEvent
from vla_control.authorization import PERMISSIONS
from vla_control.common import APIError

from .test_audit_events import PASSWORD, audit  # noqa: F401


def request(*, websocket=False):
    scope = {"type": "websocket" if websocket else "http", "path": "/api/v1/assets/example/archive",
             "headers": [], "query_string": b"token=never-copy-query", "scheme": "wss" if websocket else "https",
             "server": ("testserver", 443), "client": ("testclient", 1234),
             "route": SimpleNamespace(path="/api/v1/assets/{asset_id}/archive")}
    if websocket:
        return WebSocket(scope, receive=None, send=None)
    return Request({**scope, "method": "POST"})


def events(platform):
    with platform.app.state.db.read() as session:
        return list(session.scalars(select(AuditEvent).order_by(AuditEvent.created_at, AuditEvent.id)))


def test_marker_is_set_only_after_successful_outer_commit(audit):
    platform, _ = audit
    original = request()
    with platform.app.state.db.write() as session:
        platform.app.state.auth._audit(session, original, "assets.archive", 200)
        assert not getattr(original.state, "audit_recorded_id", None)
    assert original.state.audit_recorded_id == events(platform)[0].id
    assert events(platform)[0].path == "/api/v1/assets/{asset_id}/archive"


def test_rollback_cannot_suppress_the_failed_request_record(audit):
    platform, _ = audit
    original = request()
    with pytest.raises(RuntimeError):
        with platform.app.state.db.write() as session:
            platform.app.state.auth._audit(session, original, "assets.archive", 200)
            raise RuntimeError("abort isolated transaction")
    assert not getattr(original.state, "audit_recorded_id", None) and events(platform) == []
    platform.app.state.auth.audit(original, "assets.archive", 500, {"error_code": "INTERNAL_ERROR"})
    recorded = events(platform)
    assert len(recorded) == 1 and recorded[0].status == 500
    assert original.state.audit_recorded_id == recorded[0].id


def test_savepoint_commit_waits_for_outer_commit(audit):
    platform, _ = audit
    original = request()
    with platform.app.state.db.write() as session:
        with session.begin_nested():
            platform.app.state.auth._audit(session, original, "assets.archive", 200)
        assert not getattr(original.state, "audit_recorded_id", None)
    assert len(events(platform)) == 1 and original.state.audit_recorded_id == events(platform)[0].id


def test_outer_rollback_after_savepoint_commit_never_sets_marker(audit):
    platform, _ = audit
    original = request()
    with pytest.raises(RuntimeError):
        with platform.app.state.db.write() as session:
            with session.begin_nested():
                platform.app.state.auth._audit(session, original, "assets.archive", 200)
            raise RuntimeError("outer rollback")
    assert not getattr(original.state, "audit_recorded_id", None) and events(platform) == []


def test_savepoint_rollback_keeps_only_the_committed_outer_marker(audit):
    platform, _ = audit
    original = request()
    with platform.app.state.db.write() as session:
        platform.app.state.auth._audit(session, original, "assets.archive", 200)
        with pytest.raises(RuntimeError):
            with session.begin_nested():
                platform.app.state.auth._audit(session, original, "users.update", 200)
                session.flush()
                raise RuntimeError("nested rollback")
        assert not getattr(original.state, "audit_recorded_id", None)
    recorded = events(platform)
    assert len(recorded) == 1 and recorded[0].action == "assets.archive"
    assert original.state.audit_recorded_id == recorded[0].id


def test_multiple_explicit_websocket_events_are_never_deduplicated(audit):
    platform, _ = audit
    socket = request(websocket=True)
    platform.app.state.auth.audit(socket, "terminal.open", 200, {"terminal_id": "terminal"})
    first = socket.state.audit_recorded_id
    platform.app.state.auth.audit(socket, "terminal.close", 200, {"terminal_id": "terminal"})
    recorded = events(platform)
    assert [row.action for row in recorded] == ["terminal.open", "terminal.close"]
    assert first != socket.state.audit_recorded_id and len(recorded) == 2


def test_http_auth_and_user_events_are_recorded_once_with_safe_context(audit):
    platform, client = audit
    assert client.post("/api/v1/auth/reauth", json={"password": PASSWORD}).status_code == 200
    created = client.post("/api/v1/users", json={"username": "audit-created", "password": PASSWORD, "role": "viewer"})
    assert created.status_code == 201, created.text
    recorded = events(platform)
    assert [row.action for row in recorded] == ["auth.reauth", "users.create"]
    assert recorded[1].details["target_id"] == created.json()["id"]
    assert recorded[1].details["target_name"] == "audit-created"
    assert recorded[1].details["operation_label"] and recorded[1].details["request_id"]
    assert PASSWORD not in str([row.details for row in recorded])
    # Failed authentication is a committed explicit audit even though the HTTP
    # handler raises outside its transaction. The middleware must not duplicate it.
    failed = client.post("/api/v1/auth/login?token=never-copy-query",
        json={"username": "audit-admin", "password": "never-copy-wrong-password"})
    assert failed.status_code == 401
    recorded = events(platform)
    assert [row.action for row in recorded] == ["auth.reauth", "users.create", "auth.login_failed"]
    assert recorded[-1].path == "/api/v1/auth/login"
    assert "never-copy" not in str([row.details for row in recorded])


def test_http_rollback_still_receives_one_failed_fallback_record(audit, monkeypatch):
    platform, client = audit
    path = "/api/v1/audit-rollback-example"
    monkeypatch.setitem(PERMISSIONS, ("POST", path), "assets.write")

    @platform.app.post(path)
    def abort(request: Request):
        with platform.app.state.db.write() as session:
            platform.app.state.auth._audit(session, request, "users.update", 200)
            raise APIError(409, "ISOLATED_ROLLBACK", "isolated rollback")

    response = client.post(path)
    assert response.status_code == 409
    recorded = events(platform)
    assert len(recorded) == 1 and recorded[0].status == 409
    assert recorded[0].action == "api.mutation" and recorded[0].details["error_code"] == "ISOLATED_ROLLBACK"


def test_unmatched_path_and_unknown_details_never_copy_raw_secrets(audit):
    platform, _ = audit
    original = Request({"type": "http", "method": "POST", "path": "/api/v1/secret-path-value",
                        "headers": [], "query_string": b"token=secret-query-value",
                        "scheme": "https", "server": ("testserver", 443)})
    platform.app.state.auth.audit(original, "api.mutation", 404,
        {"body": "secret-body-value", "password": "secret-password-value", "token": "secret-token-value",
         "terminal_input": "secret-terminal-value", "attempted_username": "secret/invalid-username",
         "counts": {"created": 2, "token": "secret-count-value"}})
    event = events(platform)[0]
    assert event.path == "/api/v1/<unmatched>" and "secret-" not in str(event.details)
    assert event.details["counts"] == {"created": 2}


def test_anonymous_failed_login_keeps_actor_null_and_searchable_normalized_attempt(audit):
    from fastapi.testclient import TestClient
    platform, administrator = audit
    platform.app.state.auth.create_user("bob", PASSWORD, "operator")
    with platform.app.state.db.write() as session:
        session.execute(delete(AuditEvent))
    anonymous = TestClient(platform.app, base_url="https://testserver")
    try:
        response = anonymous.post("/api/v1/auth/login", json={"username": "  BOB  ", "password": "unrecorded-invalid-password"})
        assert response.status_code == 401
    finally:
        anonymous.close()
    recorded = events(platform)
    assert len(recorded) == 1 and recorded[0].action == "auth.login_failed"
    assert recorded[0].actor_id is None and recorded[0].actor_username is None
    assert recorded[0].details["attempted_username"] == "bob"
    response = administrator.get("/api/v1/audit/events", params={"q": "bob", "actor_id": "unattributed"})
    assert response.status_code == 200 and [item["id"] for item in response.json()["items"]] == [recorded[0].id]
    assert "unrecorded-invalid-password" not in response.text


def test_authenticated_failed_account_switch_keeps_the_real_actor(audit):
    platform, client = audit
    alice = platform.app.state.auth.create_user("alice", PASSWORD, "operator")
    bob = platform.app.state.auth.create_user("bob", PASSWORD, "operator")
    authenticated = client.post("/api/v1/auth/login", json={"username": "alice", "password": PASSWORD})
    assert authenticated.status_code == 200
    client.headers["X-CSRF-Token"] = authenticated.json()["csrf_token"]
    with platform.app.state.db.write() as session:
        session.execute(delete(AuditEvent))
    response = client.post("/api/v1/auth/login", json={"username": "BoB", "password": "unrecorded-invalid-password"})
    assert response.status_code == 401
    recorded = events(platform)
    assert len(recorded) == 1 and recorded[0].action == "auth.login_failed"
    assert recorded[0].actor_id == alice["id"] and recorded[0].actor_username == "alice"
    assert recorded[0].actor_id != bob["id"] and recorded[0].details["attempted_username"] == "bob"
    assert "unrecorded-invalid-password" not in str(recorded[0].details)
