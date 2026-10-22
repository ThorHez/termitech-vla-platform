"""Independent security regression checks against the complete routed application."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from threading import Barrier

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import func, select

from vla_control import auth as auth_module
from vla_control.auth import COOKIE_NAME
from vla_control.auth_models import User, UserSession
from vla_control.common import now, offset_time
from vla_control.db import Database
from vla_control.main import create_app
from vla_control.models import Attempt, Command, Node, Run
from vla_control.settings import Settings
from vla_control.users import create_initial_admin


PASSWORD = "independent-policy-check-123"


@pytest.fixture
def secured(tmp_path):
    app = create_app(Settings(database_url="sqlite:///" + str(tmp_path / "review.db"),
                              enrollment_token="independent-agent-enrollment",
                              session_cookie_secure=False, scheduler_enabled=False))
    create_initial_admin(app.state.db, "review-admin", PASSWORD)
    with TestClient(app) as client:
        yield app, client


def login(client, username="review-admin"):
    client.cookies.clear()
    response = client.post("/api/v1/auth/login", json={"username": username, "password": PASSWORD})
    assert response.status_code == 200, response.text
    client.headers["X-CSRF-Token"] = response.json()["csrf_token"]
    return response.json()


def session_values(app):
    with app.state.db.read() as transaction:
        row = transaction.scalar(select(UserSession).order_by(UserSession.created_at.desc()))
        return {name: getattr(row, name) for name in
                ("id", "created_at", "expires_at", "last_activity_at", "reauthenticated_at")}


def modify_session(app, **values):
    with app.state.db.write() as transaction:
        row = transaction.scalar(select(UserSession).order_by(UserSession.created_at.desc()))
        for name, value in values.items():
            setattr(row, name, value)


def test_real_sensitive_routes_challenge_before_any_mutation(secured):
    app, client = secured
    authentication = login(client)
    assert authentication["session_idle_seconds"] == 900
    assert authentication["reauth_expires_at"] is None
    row = session_values(app)
    absolute = datetime.fromisoformat(authentication["expires_at"])
    idle = datetime.fromisoformat(authentication["idle_expires_at"])
    created = datetime.fromisoformat(row["created_at"])
    assert absolute - created == timedelta(seconds=7200)
    assert idle - datetime.fromisoformat(row["last_activity_at"]) == timedelta(seconds=900)
    requests = [
        ("/api/v1/users", {"username": "new-viewer", "password": PASSWORD, "role": "viewer"}),
        ("/api/v1/nodes/missing/uninstall", {}),
        ("/api/v1/recipe-revisions/missing/writeback", {}),
    ]
    for path, body in requests:
        response = client.post(path, json=body)
        assert response.status_code == 403, (path, response.text)
        assert response.json()["error"]["code"] == "REAUTH_REQUIRED"
    assert len(client.get("/api/v1/users").json()) == 1
    proof = client.post("/api/v1/auth/reauth", json={"password": PASSWORD})
    assert proof.status_code == 200, proof.text
    result = client.post(requests[0][0], json=requests[0][1])
    assert result.status_code == 201, result.text


def test_viewer_is_forbidden_before_reauthentication_challenge(secured):
    app, client = secured
    app.state.auth.create_user("review-viewer", PASSWORD, "viewer")
    login(client, "review-viewer")
    response = client.post("/api/v1/nodes/missing/uninstall", json={})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "FORBIDDEN"


def test_polling_cannot_extend_idle_or_resurrect_expired_session(secured):
    app, client = secured
    login(client)
    modify_session(app, last_activity_at=offset_time(-800))
    before = session_values(app)
    for path in ("/api/v1/auth/me", "/api/v1/overview", "/api/v1/runs"):
        assert client.get(path).status_code == 200
    assert session_values(app) == before
    modify_session(app, last_activity_at=offset_time(-901))
    for method, path, body in [("GET", "/api/v1/auth/me", None),
                               ("POST", "/api/v1/auth/activity", {}),
                               ("POST", "/api/v1/auth/reauth", {"password": PASSWORD})]:
        response = client.request(method, path, json=body)
        assert response.status_code == 401, (path, response.text)


@pytest.mark.parametrize("path,body", [
    ("/api/v1/auth/activity", {}),
    ("/api/v1/auth/reauth", {"password": PASSWORD}),
])
def test_activity_and_reauthentication_require_csrf(secured, path, body):
    app, client = secured
    login(client)
    before = session_values(app)
    client.headers.pop("X-CSRF-Token")
    response = client.post(path, json=body)
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "CSRF_REJECTED"
    assert session_values(app) == before


def test_activity_body_cannot_set_server_timestamps_or_elevation(secured):
    app, client = secured
    login(client)
    modify_session(app, last_activity_at=offset_time(-600))
    before = session_values(app)
    future = offset_time(86400)
    response = client.post("/api/v1/auth/activity", json={
        "last_activity_at": future, "expires_at": future, "reauthenticated_at": future})
    assert response.status_code in {200, 422}, response.text
    after = session_values(app)
    assert after["expires_at"] == before["expires_at"]
    assert after["reauthenticated_at"] is None
    assert after["last_activity_at"] < future


def test_older_eight_hour_expiry_cannot_bypass_two_hour_limit(secured):
    app, client = secured
    login(client)
    modify_session(app, created_at=offset_time(-7201), expires_at=offset_time(20000),
                   last_activity_at=offset_time(-1))
    assert client.get("/api/v1/auth/me").status_code == 401
    assert client.post("/api/v1/auth/activity", json={}).status_code == 401


def test_password_proof_does_not_extend_idle_or_absolute_and_is_session_scoped(secured):
    app, client = secured
    login(client)
    modify_session(app, last_activity_at=offset_time(-500))
    before = session_values(app)
    response = client.post("/api/v1/auth/reauth", json={"password": PASSWORD})
    assert response.status_code == 200, response.text
    after = session_values(app)
    assert after["last_activity_at"] == before["last_activity_at"]
    assert after["expires_at"] == before["expires_at"]
    assert after["reauthenticated_at"] is not None
    first_cookie = client.cookies.get(COOKIE_NAME)
    first_csrf = client.headers["X-CSRF-Token"]
    login(client)
    assert client.post("/api/v1/nodes/missing/uninstall", json={}).json()["error"]["code"] == "REAUTH_REQUIRED"
    # The independently authenticated browser retains only its own proof.
    first = TestClient(app)
    try:
        first.cookies.set(COOKIE_NAME, first_cookie, path="/api/v1")
        first.headers["X-CSRF-Token"] = first_csrf
        response = first.post("/api/v1/nodes/missing/uninstall", json={},
                              headers={"Idempotency-Key": "review-missing-node"})
        assert response.status_code == 404, response.text
    finally:
        first.close()


def test_reverse_proxy_prefix_cannot_skip_sensitive_password_proof(secured):
    app, _ = secured
    client = TestClient(app, root_path="/platform")
    try:
        response = client.post("/platform/api/v1/auth/login", json={"username": "review-admin", "password": PASSWORD})
        assert response.status_code == 200, response.text
        client.headers["X-CSRF-Token"] = response.json()["csrf_token"]
        denied = client.post("/platform/api/v1/nodes/missing/uninstall", json={})
        assert denied.status_code == 403
        assert denied.json()["error"]["code"] == "REAUTH_REQUIRED"
        assert client.post("/platform/api/v1/auth/reauth", json={"password": PASSWORD}).status_code == 200
        assert client.post("/platform/api/v1/nodes/missing/uninstall", json={},
                           headers={"Idempotency-Key": "review-prefix-missing-node"}).status_code == 404
    finally:
        client.close()


def test_password_verification_cannot_restore_session_that_expired_during_hashing(secured, monkeypatch):
    app, client = secured
    login(client)
    original = auth_module.verify_password

    def expire_after_verification(password, encoded):
        valid = original(password, encoded)
        modify_session(app, last_activity_at=offset_time(-901))
        return valid

    monkeypatch.setattr(auth_module, "verify_password", expire_after_verification)
    response = client.post("/api/v1/auth/reauth", json={"password": PASSWORD})
    assert response.status_code == 401, response.text
    assert session_values(app)["reauthenticated_at"] is None


def test_user_creation_rechecks_proof_after_credential_hashing(secured, monkeypatch):
    app, client = secured
    login(client)
    assert client.post("/api/v1/auth/reauth", json={"password": PASSWORD}).status_code == 200
    original = auth_module.hash_password

    def expire_proof_after_hashing(password):
        encoded = original(password)
        modify_session(app, reauthenticated_at=offset_time(-301))
        return encoded

    monkeypatch.setattr(auth_module, "hash_password", expire_proof_after_hashing)
    response = client.post("/api/v1/users", json={"username": "race-created", "password": PASSWORD, "role": "viewer"})
    assert response.status_code == 403, response.text
    assert response.json()["error"]["code"] == "REAUTH_REQUIRED"
    assert len(client.get("/api/v1/users").json()) == 1


@pytest.mark.parametrize("expiry", ["idle", "absolute"])
def test_browser_expiry_does_not_disable_agent_heartbeat_or_cancel_training(secured, expiry):
    app, client = secured
    login(client)
    registration = client.post("/agent/v1/register", headers={
        "Authorization": "Bearer independent-agent-enrollment"}, json={
            "name": "independent-worker", "hostname": "independent-worker", "roots": ["/fixture"],
            "output_root": "/fixture/output", "profiles": [{
                "id": "runtime", "name": "fixture", "code_dir": "/fixture/code", "python": "/usr/bin/python3",
                "adapter": "smoke", "recipe": "smoke"}]})
    assert registration.status_code == 201, registration.text
    node = registration.json()
    stamp = now()
    with app.state.db.write() as transaction:
        transaction.add(Run(id="active-run", name="Unrelated training", node_id=node["node_id"],
                            profile_id="runtime", template_id="smoke", state="RUNNING", created_at=stamp,
                            updated_at=stamp, inputs={}, parameters={}, gpu_count=0, code_expectation={}))
        transaction.flush()
        transaction.add(Attempt(id="active-attempt", run_id="active-run", node_id=node["node_id"],
                                execution_token="independent-running-token", state="RUNNING", created_at=stamp,
                                started_at=stamp, pid=1234))
    if expiry == "idle":
        modify_session(app, last_activity_at=offset_time(-901))
    else:
        modify_session(app, created_at=offset_time(-7201))
    assert client.get("/api/v1/runs/active-run").status_code == 401
    headers = {"Authorization": "Bearer " + node["token"]}
    heartbeat = client.post("/agent/v1/heartbeat", headers=headers, json={
        "node_id": node["node_id"], "metrics": {"still_alive": True}, "active_attempts": [{
            "attempt_id": "active-attempt", "pid": 1234, "state": "RUNNING"}]})
    assert heartbeat.status_code == 200, heartbeat.text
    commands = client.get("/agent/v1/commands", headers=headers, params={"node_id": node["node_id"]})
    assert commands.status_code == 200 and commands.json() == []
    with app.state.db.read() as transaction:
        assert transaction.get(Run, "active-run").state == "RUNNING"
        assert transaction.get(Attempt, "active-attempt").state == "RUNNING"
        assert transaction.get(Node, node["node_id"]).status == "ONLINE"
        assert transaction.scalar(select(func.count()).select_from(Command)) == 0


def test_concurrent_old_schema_upgrade_revokes_sessions_and_preserves_metadata(secured):
    app, client = secured
    login(client)
    stamp = now()
    with app.state.db.write() as transaction:
        transaction.add(Node(id="preserved-machine", token_hash="a" * 64, name="Preserved machine",
                             hostname="preserved", status="ONLINE", last_seen=stamp, created_at=stamp,
                             roots=["/fixture"], output_root="/fixture/output"))
    with app.state.db.engine.begin() as connection:
        connection.exec_driver_sql("ALTER TABLE user_sessions DROP COLUMN last_activity_at")
        connection.exec_driver_sql("ALTER TABLE user_sessions DROP COLUMN reauthenticated_at")
    barrier = Barrier(2)

    def upgrade(_):
        separate = Database(app.state.settings.database_url)
        try:
            barrier.wait(timeout=5)
            separate.initialize()
        finally:
            separate.close()

    with ThreadPoolExecutor(max_workers=2) as workers:
        list(workers.map(upgrade, range(2)))
    with app.state.db.read() as transaction:
        assert transaction.scalar(select(func.count()).select_from(UserSession)) == 0
        assert transaction.scalar(select(func.count()).select_from(User)) == 1
        node = transaction.get(Node, "preserved-machine")
        assert node.status == "ONLINE" and node.token_hash == "a" * 64
    # A second initialization must not invalidate sessions created after upgrade.
    login(client)
    app.state.db.initialize()
    assert client.get("/api/v1/auth/me").status_code == 200
