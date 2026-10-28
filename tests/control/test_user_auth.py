"""Identity/authorization invariants using independent databases, never live data."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta
import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import func, select

from vla_control import auth as auth_module
from vla_control.auth import AuthManager, COOKIE_NAME, PERMISSIONS, hash_password, register_auth_routes, verify_password
from vla_control.auth_models import AuditEvent, LoginThrottle, User, UserSession
from vla_control.common import APIError, now, offset_time
from vla_control.db import Database
from vla_control.settings import Settings
from vla_control.users import create_initial_admin, main as users_cli

PASSWORD = "correct-horse-battery"
NEW_PASSWORD = "different-horse-battery"


@pytest.fixture
def identity(tmp_path):
    db = Database("sqlite:///" + str(tmp_path / "identity.db"))
    db.initialize()
    settings = Settings(database_url=str(db.engine.url), enrollment_token="separate-agent-credential", session_cookie_secure=False)
    manager = AuthManager(db, settings)
    app = FastAPI()

    @app.exception_handler(APIError)
    async def api_error(_, exc):
        return JSONResponse(status_code=exc.status, content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}})

    register_auth_routes(app, manager)

    @app.get("/api/v1/test/read")
    def read(request: Request):
        return {"user": manager.require(request, "platform.read").username}

    @app.post("/api/v1/test/write")
    def write(request: Request):
        return {"user": manager.require(request, "training.write").username}

    client = TestClient(app)
    yield db, manager, app, client
    client.close()
    db.close()


def login(client, username="admin", password=PASSWORD):
    response = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return response.json()


def headers(payload):
    return {"X-CSRF-Token": payload["csrf_token"]}


def admin(identity):
    _, manager, _, client = identity
    user = manager.create_user("admin", PASSWORD)
    payload = login(client)
    proof = client.post("/api/v1/auth/reauth", headers=headers(payload), json={"password": PASSWORD})
    assert proof.status_code == 200, proof.text
    return user, proof.json()


def test_password_hashing_uses_salt_and_rejects_bad_storage():
    first, second = hash_password(PASSWORD), hash_password(PASSWORD)
    assert first != second and PASSWORD not in first
    assert verify_password(PASSWORD, first)
    assert not verify_password("wrong", first)
    assert not verify_password(PASSWORD, "scrypt$9999999999$8$3$x$x")
    assert not verify_password(PASSWORD, "garbage")
    assert not verify_password("x" * 257, first)
    with pytest.raises(APIError):
        hash_password("too-short")


def test_default_secure_and_optional_legacy_settings(monkeypatch):
    monkeypatch.delenv("VLA_SESSION_COOKIE_SECURE", raising=False)
    monkeypatch.delenv("VLA_LEGACY_ADMIN_ENABLED", raising=False)
    settings = Settings.from_env()
    assert settings.session_cookie_secure and not settings.legacy_admin_enabled
    Settings(enrollment_token="agents-only").validate()
    with pytest.raises(ValueError):
        Settings(enrollment_token="same", admin_token="same").validate()
    with pytest.raises(ValueError):
        Settings(enrollment_token="x", legacy_admin_enabled=True).validate()


def test_first_admin_creation_is_atomic_and_no_public_bootstrap(identity):
    db, _, _, client = identity
    user = create_initial_admin(db, " Admin ", PASSWORD)
    assert user["username"] == "admin" and not user["must_change_password"]
    with pytest.raises(APIError) as exc:
        create_initial_admin(db, "another", PASSWORD)
    assert exc.value.code == "ADMIN_ALREADY_INITIALIZED"
    assert client.post("/api/v1/auth/register", json={}).status_code == 404
    assert client.post("/api/v1/auth/bootstrap", json={}).status_code == 404


def test_login_session_is_hashed_cookie_http_only_and_no_token_in_payload(identity):
    db, _, _, client = identity
    _, payload = admin(identity)
    assert set(payload) == {"user", "permissions", "csrf_token", "expires_at", "idle_expires_at", "session_idle_seconds", "reauth_expires_at", "server_time"}
    assert set(payload["permissions"]) == PERMISSIONS
    raw = client.cookies.get(COOKIE_NAME)
    with db.read() as session:
        row = session.scalar(select(UserSession))
        assert row.token_hash != raw and raw not in repr(row.__dict__)
        assert len(row.token_hash) == 64
    response = client.get("/api/v1/auth/me")
    assert {key: value for key, value in response.json().items() if key != "server_time"} == {key: value for key, value in payload.items() if key != "server_time"}
    assert response.json()["server_time"] >= payload["server_time"]
    assert response.headers["cache-control"] == "no-store"
    assert PASSWORD not in response.text and "password_hash" not in response.text


def test_secure_cookie_attributes(identity):
    _, manager, app, _ = identity
    manager.settings = replace(manager.settings, session_cookie_secure=True)
    manager.create_user("admin", PASSWORD)
    with TestClient(app, base_url="https://controller.example") as client:
        response = client.post("/api/v1/auth/login", json={"username": "admin", "password": PASSWORD})
        cookie = response.headers["set-cookie"]
        assert "HttpOnly" in cookie and "Secure" in cookie and "SameSite=strict" in cookie
        assert "Path=/api/v1" in cookie and "Max-Age=28800" in cookie
        assert client.get("/api/v1/auth/me").status_code == 200


def test_unauthenticated_paths_do_not_accept_agent_credentials(identity):
    _, _, _, client = identity
    assert client.get("/api/v1/auth/me").status_code == 401
    assert client.get("/api/v1/test/read", headers={"Authorization": "Bearer separate-agent-credential"}).status_code == 401
    assert client.post("/api/v1/users", json={"username": "bob", "password": PASSWORD, "role": "admin"}).status_code == 401


@pytest.mark.parametrize("path,body", [
    ("/api/v1/auth/logout", None),
    ("/api/v1/auth/password", {"current_password": PASSWORD, "new_password": NEW_PASSWORD}),
    ("/api/v1/users", {"username": "worker", "password": PASSWORD, "role": "viewer"}),
    ("/api/v1/test/write", {}),
])
def test_all_authenticated_unsafe_requests_require_csrf(identity, path, body):
    *_, client = identity
    admin(identity)
    response = client.post(path, json=body)
    assert response.status_code == 403 and response.json()["error"]["code"] == "CSRF_REJECTED"


@pytest.mark.parametrize("origin", ["https://evil.example", "null", "http://testserver.evil", "http://[", "http://testserver:81"])
def test_login_rejects_foreign_or_malformed_origin(identity, origin):
    *_, client = identity
    response = client.post("/api/v1/auth/login", headers={"Origin": origin}, json={"username": "missing", "password": PASSWORD})
    assert response.status_code == 403


def test_same_origin_and_authenticated_login_csrf(identity):
    *_, client = identity
    _, payload = admin(identity)
    body = {"username": "admin", "password": PASSWORD}
    assert client.post("/api/v1/auth/login", json=body).status_code == 403
    assert client.post("/api/v1/auth/login", json=body, headers={**headers(payload), "Origin": "http://testserver"}).status_code == 200


@pytest.mark.parametrize("role,write_status", [("viewer", 403), ("operator", 200), ("admin", 200)])
def test_roles_are_enforced_on_backend(identity, role, write_status):
    _, manager, _, client = identity
    manager.create_user("someone", PASSWORD, role)
    payload = login(client, "someone")
    assert client.get("/api/v1/test/read").status_code == 200
    assert client.post("/api/v1/test/write", headers=headers(payload)).status_code == write_status
    expected = 200 if role == "admin" else 403
    assert client.get("/api/v1/users").status_code == expected
    assert client.get("/api/v1/audit").status_code == expected


def test_expired_session_is_rejected(identity):
    db, _, _, client = identity
    admin(identity)
    with db.write() as session:
        row = session.scalar(select(UserSession))
        row.expires_at = offset_time(-1)
    assert client.get("/api/v1/auth/me").status_code == 401


def test_logout_revokes_server_side_session(identity):
    _, _, app, client = identity
    _, payload = admin(identity)
    cookie = client.cookies.get(COOKIE_NAME)
    assert client.post("/api/v1/auth/logout", headers=headers(payload)).status_code == 204
    assert client.get("/api/v1/auth/me").status_code == 401
    with TestClient(app) as other:
        other.cookies.set(COOKIE_NAME, cookie, path="/api/v1")
        assert other.get("/api/v1/auth/me").status_code == 401


def test_admin_created_user_must_change_and_password_rotates_all_sessions(identity):
    db, manager, app, client = identity
    _, payload = admin(identity)
    response = client.post("/api/v1/users", headers=headers(payload), json={"username": "worker", "role": "operator", "password": PASSWORD})
    assert response.status_code == 201 and response.json()["must_change_password"]
    with TestClient(app) as worker, TestClient(app) as another:
        first = login(worker, "worker")
        login(another, "worker")
        old_cookie = worker.cookies.get(COOKIE_NAME)
        assert worker.get("/api/v1/test/read").json()["error"]["code"] == "PASSWORD_CHANGE_REQUIRED"
        assert worker.get("/api/v1/auth/me").status_code == 200
        changed = worker.post("/api/v1/auth/password", headers=headers(first), json={"current_password": PASSWORD, "new_password": NEW_PASSWORD})
        assert changed.status_code == 200 and not changed.json()["user"]["must_change_password"]
        assert worker.cookies.get(COOKIE_NAME) != old_cookie
        assert changed.json()["csrf_token"] != first["csrf_token"]
        assert another.get("/api/v1/auth/me").status_code == 401
        assert worker.post("/api/v1/test/write", headers=headers(first)).status_code == 403
        assert worker.post("/api/v1/test/write", headers=headers(changed.json())).status_code == 200
        assert worker.post("/api/v1/auth/logout", headers=headers(changed.json())).status_code == 204
        assert login(worker, "worker", NEW_PASSWORD)["user"]["username"] == "worker"


@pytest.mark.parametrize("change", [{"enabled": False}, {"role": "viewer"}])
def test_last_enabled_admin_cannot_be_disabled_or_demoted(identity, change):
    *_, client = identity
    user, payload = admin(identity)
    response = client.patch("/api/v1/users/" + user["id"], headers=headers(payload), json=change)
    assert response.status_code == 409 and response.json()["error"]["code"] == "LAST_ADMIN_REQUIRED"


def test_concurrent_demotions_preserve_one_admin(identity):
    db, manager, _, _ = identity
    first = manager.create_user("first", PASSWORD)
    second = manager.create_user("second", PASSWORD)
    # Offline callers have the same account invariant and DB guard.
    def demote(user_id):
        try:
            manager.update_user(None, user_id, {"role": "viewer"})
            return 200
        except APIError as exc:
            return exc.status
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(demote, [first["id"], second["id"]]))
    assert sorted(results) == [200, 409]
    with db.read() as session:
        assert session.scalar(select(func.count()).select_from(User).where(User.enabled.is_(True), User.role == "admin")) == 1


@pytest.mark.parametrize("action", ["disable", "role", "reset"])
def test_admin_changes_immediately_revoke_affected_sessions(identity, action):
    _, manager, app, client = identity
    _, payload = admin(identity)
    target = manager.create_user("worker", PASSWORD, "operator")
    with TestClient(app) as worker:
        login(worker, "worker")
        if action == "reset":
            response = client.post(f"/api/v1/users/{target['id']}/reset-password", headers=headers(payload), json={"password": NEW_PASSWORD})
        else:
            change = {"enabled": False} if action == "disable" else {"role": "viewer"}
            response = client.patch(f"/api/v1/users/{target['id']}", headers=headers(payload), json=change)
        assert response.status_code == 200
        assert worker.get("/api/v1/auth/me").status_code == 401
        if action == "disable":
            failure = worker.post("/api/v1/auth/login", json={"username": "worker", "password": PASSWORD})
            assert failure.status_code == 401 and failure.json()["error"]["code"] == "INVALID_CREDENTIALS"


def test_login_lockout_persists_across_manager_instances(identity):
    db, manager, _, client = identity
    manager.create_user("admin", PASSWORD)
    for _ in range(5):
        response = client.post("/api/v1/auth/login", json={"username": "admin", "password": "wrong"})
        assert response.status_code == 401
    restarted = AuthManager(db, manager.settings)
    app = FastAPI()
    @app.exception_handler(APIError)
    async def handler(_, exc):
        return JSONResponse(status_code=exc.status, content={"error": {"code": exc.code}})
    register_auth_routes(app, restarted)
    with TestClient(app) as next_client:
        assert next_client.post("/api/v1/auth/login", json={"username": "admin", "password": PASSWORD}).status_code == 429
    with db.write() as session:
        for row in session.scalars(select(LoginThrottle)):
            row.locked_until = offset_time(-1)
            row.window_started_at = offset_time(-901)
    assert login(client)["user"]["username"] == "admin"


def test_unknown_and_disabled_credentials_have_uniform_errors(identity):
    _, manager, _, client = identity
    disabled = manager.create_user("disabled", PASSWORD, "viewer")
    manager.update_user(None, disabled["id"], {"enabled": False})
    responses = [client.post("/api/v1/auth/login", json={"username": username, "password": PASSWORD}) for username in ("missing", "disabled")]
    assert responses[0].status_code == responses[1].status_code == 401
    assert responses[0].json() == responses[1].json()


def test_legacy_token_explicit_and_disappears_after_first_user(identity):
    _, manager, _, client = identity
    token = {"Authorization": "Bearer old-shared-token"}
    manager.settings = replace(manager.settings, admin_token="old-shared-token")
    assert client.get("/api/v1/test/read", headers=token).status_code == 401
    manager.settings = replace(manager.settings, legacy_admin_enabled=True)
    assert client.get("/api/v1/test/read", headers=token).status_code == 200
    manager.create_user("admin", PASSWORD)
    assert client.get("/api/v1/test/read", headers=token).status_code == 401


def test_audit_redacts_request_body_query_and_unapproved_details(identity):
    db, manager, _, client = identity
    _, payload = admin(identity)
    request = Request({"type": "http", "method": "POST", "path": "/api/v1/runs", "raw_path": b"/api/v1/runs",
                       "query_string": b"token=do-not-copy", "scheme": "http", "server": ("testserver", 80),
                       "client": ("127.0.0.1", 1234), "headers": [(b"authorization", b"Bearer private-token")]})
    manager.audit(request, "request.mutation", 403, {"password": PASSWORD, "token": "secret", "permission": "training.write", "code": "FORBIDDEN"})
    response = client.get("/api/v1/audit")
    assert response.status_code == 200
    assert PASSWORD not in response.text and "private-token" not in response.text and "do-not-copy" not in response.text
    assert response.json()[0]["details"] == {"permission": "training.write", "code": "FORBIDDEN"}
    assert any(row["action"] == "auth.login" for row in response.json())


def test_username_case_insensitive_unique_and_invalid_role(identity):
    _, manager, _, _ = identity
    manager.create_user("Admin", PASSWORD)
    with pytest.raises(APIError) as exc:
        manager.create_user("ADMIN", PASSWORD)
    assert exc.value.code == "USERNAME_TAKEN"
    with pytest.raises(APIError) as exc:
        manager.create_user("bad username", PASSWORD)
    assert exc.value.code == "INVALID_USERNAME"
    with pytest.raises(APIError):
        manager.create_user("worker", PASSWORD, "superuser")


def test_offline_cli_requires_database_and_never_prints_password(identity, monkeypatch, capsys):
    db, _, _, _ = identity
    monkeypatch.delenv("VLA_DATABASE_URL", raising=False)
    with pytest.raises(SystemExit):
        users_cli(["create-admin", "--username", "admin"])
    monkeypatch.setattr("vla_control.users.getpass.getpass", lambda _: PASSWORD)
    users_cli(["create-admin", "--username", "admin", "--database-url", str(db.engine.url)])
    assert PASSWORD not in capsys.readouterr().out
    monkeypatch.setattr("vla_control.users.getpass.getpass", lambda _: NEW_PASSWORD)
    users_cli(["reset-password", "--username", "admin", "--database-url", str(db.engine.url)])
    with db.read() as session:
        user = session.scalar(select(User).where(User.username == "admin"))
        assert verify_password(NEW_PASSWORD, user.password_hash)
        assert not user.must_change_password
    assert NEW_PASSWORD not in capsys.readouterr().out


def test_initialize_additive_and_idempotent(identity):
    db, manager, _, _ = identity
    user = manager.create_user("admin", PASSWORD)
    db.initialize()
    with db.read() as session:
        assert session.get(User, user["id"]).username == "admin"
        assert session.scalar(select(func.count()).select_from(User)) == 1


def test_unicode_passwords_and_non_ascii_csrf_do_not_cause_server_error(identity):
    _, manager, _, client = identity
    original = "密码包含中文字符-123456"
    changed = "更新密码包含中文字符-654321"
    manager.create_user("unicode", original)
    payload = login(client, "unicode", original)
    result = client.post("/api/v1/auth/password", headers=headers(payload), json={"current_password": original, "new_password": changed})
    assert result.status_code == 200
    assert client.post("/api/v1/test/write", headers=[(b"X-CSRF-Token", b"\xe9")]).status_code == 403


def test_sessions_are_bounded_per_user(identity):
    db, manager, app, _ = identity
    manager.create_user("admin", PASSWORD)
    oldest = None
    for _ in range(11):
        with TestClient(app) as browser:
            login(browser)
            oldest = oldest or browser.cookies.get(COOKIE_NAME)
    with db.read() as session:
        assert session.scalar(select(func.count()).select_from(UserSession)) == 10
    with TestClient(app) as browser:
        browser.cookies.set(COOKIE_NAME, oldest, path="/api/v1")
        assert browser.get("/api/v1/auth/me").status_code == 401


def test_session_policy_defaults_and_environment(monkeypatch):
    for name in ("VLA_SESSION_TTL_SECONDS", "VLA_SESSION_IDLE_SECONDS", "VLA_SESSION_REAUTH_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings.from_env()
    assert (settings.session_ttl_seconds, settings.session_idle_seconds, settings.session_reauth_seconds) == (28800, 3600, 300)
    monkeypatch.setenv("VLA_SESSION_TTL_SECONDS", "14400")
    monkeypatch.setenv("VLA_SESSION_IDLE_SECONDS", "600")
    monkeypatch.setenv("VLA_SESSION_REAUTH_SECONDS", "120")
    assert Settings.from_env().session_ttl_seconds == 14400
    assert Settings.from_env().session_idle_seconds == 600
    assert Settings.from_env().session_reauth_seconds == 120
    for changes in ({"session_idle_seconds": 59}, {"session_idle_seconds": 28801}, {"session_reauth_seconds": 29}, {"session_reauth_seconds": 901}):
        with pytest.raises(ValueError):
            replace(Settings(enrollment_token="test"), **changes).validate()


def test_only_explicit_human_activity_extends_idle_deadline(identity):
    db, _, _, client = identity
    _, payload = admin(identity)
    with db.write() as session:
        row = session.scalar(select(UserSession))
        row.last_activity_at = offset_time(-1800)
        original_activity = row.last_activity_at
        original_expiry = row.expires_at
    for _ in range(3):
        assert client.get("/api/v1/auth/me").status_code == 200
        assert client.get("/api/v1/test/read").status_code == 200
    with db.read() as session:
        assert session.scalar(select(UserSession)).last_activity_at == original_activity
    proof = client.post("/api/v1/auth/reauth", headers=headers(payload), json={"password": PASSWORD})
    assert proof.status_code == 200
    with db.read() as session:
        assert session.scalar(select(UserSession)).last_activity_at == original_activity
    refreshed = client.post("/api/v1/auth/activity", headers=headers(payload), json={})
    assert refreshed.status_code == 200
    assert refreshed.json()["expires_at"] <= original_expiry
    with db.read() as session:
        row = session.scalar(select(UserSession))
        assert row.last_activity_at > original_activity
        assert row.expires_at == original_expiry


@pytest.mark.parametrize("expired_field,offset", [("last_activity_at", -3601), ("created_at", -28801)])
def test_idle_or_absolute_expiry_blocks_polling_and_cannot_be_revived(identity, expired_field, offset):
    db, _, _, client = identity
    _, payload = admin(identity)
    with db.write() as session:
        row = session.scalar(select(UserSession))
        # A later stored expiry cannot bypass the current idle or absolute limit.
        row.expires_at = offset_time(86400)
        setattr(row, expired_field, offset_time(offset))
    assert client.get("/api/v1/auth/me").status_code == 401
    assert client.get("/api/v1/test/read").status_code == 401
    assert client.post("/api/v1/auth/activity", headers=headers(payload), json={}).status_code == 401
    assert client.post("/api/v1/auth/reauth", headers=headers(payload), json={"password": PASSWORD}).status_code == 401


@pytest.mark.parametrize("elapsed,status", [(1800, 200), (3599, 200), (3600, 401)])
def test_default_idle_timeout_is_one_hour_at_the_exact_boundary(identity, monkeypatch, elapsed, status):
    db, _, _, client = identity
    admin(identity)
    with db.read() as session:
        activity = datetime.fromisoformat(session.scalar(select(UserSession)).last_activity_at)
    stamp = (activity + timedelta(seconds=elapsed)).isoformat(timespec="microseconds")
    monkeypatch.setattr(auth_module, "now", lambda: stamp)
    assert client.get("/api/v1/auth/me").status_code == status
    assert client.get("/api/v1/test/read").status_code == status


def test_default_absolute_eight_hour_boundary_survives_recent_activity(identity, monkeypatch):
    db, _, _, client = identity
    _, payload = admin(identity)
    with db.write() as session:
        row = session.scalar(select(UserSession))
        created = datetime.fromisoformat(row.created_at)
        original_expiry = row.expires_at
        row.last_activity_at = (created + timedelta(seconds=27000)).isoformat(timespec="microseconds")
    deadline = (created + timedelta(seconds=28800)).isoformat(timespec="microseconds")
    stamp = (created + timedelta(seconds=28799)).isoformat(timespec="microseconds")
    monkeypatch.setattr(auth_module, "now", lambda: stamp)
    assert client.get("/api/v1/auth/me").status_code == 200
    refreshed = client.post("/api/v1/auth/activity", headers=headers(payload), json={})
    assert refreshed.status_code == 200
    assert refreshed.json()["expires_at"] == refreshed.json()["idle_expires_at"] == deadline
    with db.read() as session:
        row = session.scalar(select(UserSession))
        assert row.expires_at == original_expiry
        assert row.last_activity_at == stamp
    stamp = deadline
    assert client.get("/api/v1/auth/me").status_code == 401
    assert client.post("/api/v1/auth/activity", headers=headers(payload), json={}).status_code == 401
    assert client.post("/api/v1/auth/reauth", headers=headers(payload), json={"password": PASSWORD}).status_code == 401


def test_longer_default_policy_does_not_extend_existing_short_session(identity, monkeypatch):
    db, _, _, client = identity
    _, payload = admin(identity)
    with db.write() as session:
        row = session.scalar(select(UserSession))
        created = datetime.fromisoformat(row.created_at)
        row.expires_at = (created + timedelta(seconds=7200)).isoformat(timespec="microseconds")
        row.last_activity_at = (created + timedelta(seconds=7190)).isoformat(timespec="microseconds")
        stored_expiry = row.expires_at
    stamp = (created + timedelta(seconds=7199)).isoformat(timespec="microseconds")
    monkeypatch.setattr(auth_module, "now", lambda: stamp)
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 200
    assert response.json()["expires_at"] == stored_expiry
    stamp = stored_expiry
    assert client.get("/api/v1/auth/me").status_code == 401
    assert client.post("/api/v1/auth/activity", headers=headers(payload), json={}).status_code == 401


def test_sensitive_mutation_requires_separate_fresh_password_proof(identity):
    db, manager, _, client = identity
    manager.create_user("admin", PASSWORD)
    payload = login(client)
    mutation = lambda: client.post("/api/v1/users", headers=headers(payload), json={"username": "worker", "password": PASSWORD, "role": "viewer"})
    denied = mutation()
    assert denied.status_code == 403 and denied.json()["error"]["code"] == "REAUTH_REQUIRED"
    wrong = client.post("/api/v1/auth/reauth", headers=headers(payload), json={"password": "incorrect"})
    assert wrong.status_code == 403 and wrong.json()["error"]["code"] == "INVALID_PASSWORD"
    assert client.get("/api/v1/auth/me").status_code == 200
    proof = client.post("/api/v1/auth/reauth", headers=headers(payload), json={"password": PASSWORD})
    assert proof.status_code == 200 and proof.json()["reauth_expires_at"]
    assert mutation().status_code == 201
    with db.write() as session:
        session.scalar(select(UserSession)).reauthenticated_at = offset_time(-301)
    stale = client.patch("/api/v1/users/missing", headers=headers(payload), json={"enabled": False})
    assert stale.status_code == 403 and stale.json()["error"]["code"] == "REAUTH_REQUIRED"


def test_reauthentication_throttle_survives_new_sessions_and_does_not_leak_password(identity):
    db, manager, app, client = identity
    manager.create_user("admin", PASSWORD)
    payload = login(client)
    for _ in range(5):
        result = client.post("/api/v1/auth/reauth", headers=headers(payload), json={"password": "never-copy-this"})
        assert result.status_code == 403
    # Re-login does not clear a separate reauthentication account throttle.
    with TestClient(app) as other:
        next_payload = login(other)
        result = other.post("/api/v1/auth/reauth", headers=headers(next_payload), json={"password": PASSWORD})
        assert result.status_code == 429
    audit = client.get("/api/v1/audit")
    assert "never-copy-this" not in audit.text and PASSWORD not in audit.text
    assert any(row["action"] == "auth.reauth_failed" for row in audit.json())


def test_activity_and_reauthentication_require_csrf(identity):
    *_, client = identity
    admin(identity)
    for path, body in (("activity", {}), ("reauth", {"password": PASSWORD})):
        result = client.post("/api/v1/auth/" + path, json=body)
        assert result.status_code == 403 and result.json()["error"]["code"] == "CSRF_REJECTED"


@pytest.mark.parametrize("exhaustion_path", ["password", "reauth"])
def test_password_change_and_reauthentication_share_guess_limit(identity, exhaustion_path):
    _, manager, _, client = identity
    manager.create_user("admin", PASSWORD)
    payload = login(client)
    body = {"password": "wrong-password"} if exhaustion_path == "reauth" else {"current_password": "wrong-password", "new_password": NEW_PASSWORD}
    for _ in range(5):
        result = client.post("/api/v1/auth/" + exhaustion_path, headers=headers(payload), json=body)
        assert result.status_code == 403 and result.json()["error"]["code"] == "INVALID_PASSWORD"
    for path, body in (("reauth", {"password": PASSWORD}), ("password", {"current_password": PASSWORD, "new_password": NEW_PASSWORD})):
        result = client.post("/api/v1/auth/" + path, headers=headers(payload), json=body)
        assert result.status_code == 429


def test_session_upgrade_revokes_old_cookies_and_preserves_users_and_business_tables(tmp_path):
    import sqlite3
    from sqlalchemy import inspect
    path = tmp_path / "upgrade.db"
    db = Database("sqlite:///" + str(path))
    db.initialize()
    manager = AuthManager(db, Settings(enrollment_token="test"))
    user = manager.create_user("admin", PASSWORD)
    with db.engine.begin() as connection:
        # A separate sentinel represents an existing business table and data.
        connection.exec_driver_sql("CREATE TABLE upgrade_business_sentinel (id TEXT PRIMARY KEY)")
        connection.exec_driver_sql("INSERT INTO upgrade_business_sentinel VALUES ('keep-me')")
    db.close()
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE user_sessions")
        connection.execute("CREATE TABLE user_sessions (id VARCHAR(36) PRIMARY KEY, user_id VARCHAR(36), token_hash VARCHAR(64), csrf_token VARCHAR(100), created_at VARCHAR(40), expires_at VARCHAR(40))")
        connection.execute("INSERT INTO user_sessions VALUES (?, ?, ?, ?, ?, ?)", ("old-cookie", user["id"], "old-hash", "old-csrf", now(), offset_time(28800)))
    upgraded = Database("sqlite:///" + str(path))
    upgraded.initialize()
    assert {"last_activity_at", "reauthenticated_at"} <= {column["name"] for column in inspect(upgraded.engine).get_columns("user_sessions")}
    with upgraded.read() as session:
        assert session.scalar(select(func.count()).select_from(UserSession)) == 0
        assert session.get(User, user["id"]).username == "admin"
    with upgraded.engine.connect() as connection:
        assert connection.exec_driver_sql("SELECT id FROM upgrade_business_sentinel").scalar_one() == "keep-me"
    # Reinitializing an upgraded database must keep newly issued sessions.
    with upgraded.write() as session:
        upgraded_manager = AuthManager(upgraded, Settings(enrollment_token="test"))
        upgraded_manager._new_session(session, session.get(User, user["id"]))
    upgraded.initialize()
    with upgraded.read() as session:
        assert session.scalar(select(func.count()).select_from(UserSession)) == 1
    upgraded.close()
