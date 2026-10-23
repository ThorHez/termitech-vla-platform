"""Independent adversarial checks; disposable DB, no lifespan or real sockets.

These checks attack trust boundaries without dispatching a training command,
starting an SSH manager, or using a deployed account/credential.
"""

from contextlib import ExitStack
import json
import re
import socket
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest
from sqlalchemy import func, select

from vla_agent.safety import AgentError, allowed_path
from vla_control.auth import COOKIE_NAME
from vla_control.auth_models import User
from vla_control.authorization import PERMISSIONS
from vla_control.common import now, token_hash, uid
from vla_control.main import create_app
from vla_control.models import Command, Node
from vla_control.remote_deployment import DeploymentManager
from vla_control.remote_schemas import SSHAuthenticationInput
from vla_control.settings import Settings
from vla_control.ssh_remote import RemoteError, _install_path


ORIGIN = "https://security.invalid"
PASSWORD = "disposable-security-password-492"


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("A security contract test attempted a real connection or remote-manager startup")

    monkeypatch.setenv("VLA_REMOTE_STATE_DIR", str(tmp_path / "isolated-remote-state"))
    monkeypatch.delenv("VLA_WEB_DIST", raising=False)
    # The ASGI transport is in memory. Never enter TestClient's lifespan: the
    # production lifespan intentionally starts SSH tunnels even with no scheduler.
    with ExitStack() as stack:
        stack.enter_context(patch.object(socket.socket, "connect", forbidden))
        stack.enter_context(patch.object(socket.socket, "connect_ex", forbidden))
        stack.enter_context(patch.object(socket, "create_connection", forbidden))
        stack.enter_context(patch.object(DeploymentManager, "start", forbidden))
        app = create_app(Settings(database_url="sqlite:///" + str(tmp_path / "isolated-audit.db"),
                                  enrollment_token="disposable-enrollment-secret", scheduler_enabled=False,
                                  require_https=True, public_url=ORIGIN, legacy_admin_enabled=False))
        for role in ("admin", "operator", "viewer"):
            app.state.auth.create_user("audit-" + role, PASSWORD, role)
        client = TestClient(app, base_url=ORIGIN, follow_redirects=False)
        try:
            yield app, client
            assert app.state.remote_manager._started is False
            assert not app.state.remote_manager._tunnels
        finally:
            client.close()
            app.state.db.close()


def login(client, username="audit-admin"):
    client.cookies.clear()
    client.headers.pop("X-CSRF-Token", None)
    response = client.post("/api/v1/auth/login", json={"username": username, "password": PASSWORD},
                           headers={"Origin": ORIGIN})
    assert response.status_code == 200, response.text
    client.headers["Origin"] = ORIGIN
    client.headers["X-CSRF-Token"] = response.json()["csrf_token"]
    return response.json()


def concrete(path):
    return re.sub(r"\{[^}]+\}", "missing-security-resource", path)


def test_every_management_route_rejects_anonymous_before_side_effects(isolated):
    _, client = isolated
    for (method, path) in PERMISSIONS:
        response = client.request(method, concrete(path), json={} if method != "GET" else None)
        assert response.status_code == 401, (method, path, response.text)
        assert response.json()["error"]["code"] == "UNAUTHORIZED"


def test_method_overrides_and_encoded_paths_never_grant_anonymous_access(isolated):
    _, client = isolated
    targets = ("/api/v1/users", "/api%2fv1%2fusers", "/api/v1/%75sers", "/api/v1//users",
               "/API/v1/users", "/api/v1/../v1/users", "/api/v1/users/", "/api/v1/users%00")
    for target in targets:
        for method in ("GET", "POST", "PATCH", "PUT", "DELETE", "HEAD", "OPTIONS"):
            response = client.request(method, target, headers={"X-HTTP-Method-Override": "GET"},
                                      json={} if method not in {"GET", "HEAD"} else None)
            assert response.status_code in {307, 401, 403, 404, 405}, (method, target, response.status_code)
            if response.status_code == 307:
                # Canonical slash redirects contain no credential or capability.
                response = client.get(response.headers["Location"])
                assert response.status_code == 401


def test_sql_metacharacters_do_not_authenticate_or_mutate_account_inventory(isolated):
    app, client = isolated
    with app.state.db.read() as session:
        before = session.scalar(select(func.count()).select_from(User))
    for username in ("' OR 1=1 --", "audit-admin'--", '" OR "1"="1', "admin; DROP TABLE users;--"):
        response = client.post("/api/v1/auth/login", json={"username": username, "password": PASSWORD},
                               headers={"Origin": ORIGIN})
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "INVALID_CREDENTIALS"
        assert COOKIE_NAME not in response.cookies
    with app.state.db.read() as session:
        assert session.scalar(select(func.count()).select_from(User)) == before == 3
    assert login(client)["user"]["username"] == "audit-admin"


def test_fixed_cookie_stale_csrf_and_signed_out_cookie_cannot_be_replayed(isolated):
    app, client = isolated
    client.cookies.set(COOKIE_NAME, "attacker-chosen-cookie", domain="security.invalid", path="/api/v1")
    response = client.post("/api/v1/auth/login", json={"username": "audit-admin", "password": PASSWORD},
                           headers={"Origin": ORIGIN})
    assert response.status_code == 200
    first = response.json()
    first_cookie = client.cookies.get(COOKIE_NAME)
    assert first_cookie != "attacker-chosen-cookie"
    client.headers.update({"Origin": ORIGIN, "X-CSRF-Token": first["csrf_token"]})
    response = client.post("/api/v1/auth/login", json={"username": "audit-admin", "password": PASSWORD})
    assert response.status_code == 200
    second = response.json()
    second_cookie = client.cookies.get(COOKIE_NAME)
    assert second_cookie != first_cookie and second["csrf_token"] != first["csrf_token"]
    assert client.post("/api/v1/auth/activity").json()["error"]["code"] == "CSRF_REJECTED"
    client.headers["X-CSRF-Token"] = second["csrf_token"]
    assert client.post("/api/v1/auth/logout").status_code == 204
    replay = TestClient(app, base_url=ORIGIN)
    try:
        for cookie in (first_cookie, second_cookie):
            replay.cookies.set(COOKIE_NAME, cookie, domain="security.invalid", path="/api/v1")
            assert replay.get("/api/v1/auth/me").status_code == 401
    finally:
        replay.close()


def test_cross_origin_mutations_and_simple_login_bodies_are_rejected(isolated):
    _, client = isolated
    payload = login(client)
    for origin in ("https://security.invalid.evil.invalid", "null", "https://user@security.invalid",
                   "https://security.invalid:444", "http://security.invalid", "https://security.invalid/path"):
        response = client.post("/api/v1/auth/activity", headers={"Origin": origin})
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "ORIGIN_REJECTED"
    client.cookies.clear()
    client.headers.pop("X-CSRF-Token", None)
    body = json.dumps({"username": "audit-admin", "password": PASSWORD})
    for content_type in ("text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=foo"):
        response = client.post("/api/v1/auth/login", content=body,
                               headers={"Origin": ORIGIN, "Content-Type": content_type})
        assert response.status_code == 422
        assert COOKIE_NAME not in response.cookies
    assert payload["csrf_token"] not in response.text


def test_agent_credentials_cannot_cross_node_or_human_boundaries(isolated):
    app, client = isolated
    records = []
    with app.state.db.write() as session:
        for index in range(2):
            node_id, token = uid(), "disposable-node-token-" + str(index)
            session.add(Node(id=node_id, token_hash=token_hash(token), name="synthetic-" + str(index),
                             hostname="synthetic.invalid", roots=["/isolated"], output_root="/isolated/out",
                             created_at=now(), last_seen=now(), status="ONLINE", gpus=[], metrics={}))
            records.append((node_id, token))
        session.flush()
        command_id = uid()
        session.add(Command(id=command_id, node_id=records[1][0], type="CANCEL", status="PENDING",
                            created_at=now(), payload={"attempt_id": "never-dispatched", "execution_token": "synthetic"}))
    owned_id, owned_token = records[0]
    foreign_id, foreign_token = records[1]
    headers = {"Authorization": "Bearer " + owned_token}
    assert client.get("/agent/v1/commands", params={"node_id": foreign_id}, headers=headers).status_code == 403
    assert client.post("/agent/v1/heartbeat", json={"node_id": foreign_id}, headers=headers).status_code == 403
    assert client.post("/agent/v1/events/batch", json={"node_id": foreign_id, "events": []}, headers=headers).status_code == 403
    assert client.post(f"/agent/v1/commands/{command_id}/ack", headers=headers,
                       json={"node_id": owned_id, "status": "APPLIED", "result": {}}).status_code == 403
    assert client.get("/api/v1/users", headers=headers).status_code == 401
    assert client.post("/agent/v1/register", json={}, headers=headers).status_code == 401
    login(client)
    assert client.get("/agent/v1/commands", params={"node_id": owned_id}).status_code == 401
    body = client.get("/api/v1/nodes").text
    assert owned_token not in body and foreign_token not in body
    assert "token_hash" not in body and "registration_key" not in body
    with app.state.db.read() as session:
        assert session.get(Command, command_id).status == "PENDING"


def test_filesystem_root_rejects_dotdot_prefix_collision_and_symlink_escape(tmp_path):
    root = tmp_path / "managed"
    root.mkdir()
    private = tmp_path / "managed-private"
    private.mkdir()
    marker = private / "sensitive.txt"
    marker.write_text("synthetic-private-content")
    (root / "escape").symlink_to(private, target_is_directory=True)
    for candidate in (str(marker), str(root / "escape/sensitive.txt"), str(root / "../managed-private/sensitive.txt")):
        with pytest.raises(AgentError) as rejected:
            allowed_path(candidate, [str(root)])
        assert rejected.value.code == "PATH_OUTSIDE_ROOTS"
        assert "synthetic-private-content" not in str(rejected.value)
    assert allowed_path(str(root), [str(root)]) == root


@pytest.mark.parametrize("host", ["127.0.0.1; id", "$(id)", "localhost`id`", "-oProxyCommand=id", "x\nHost evil", "https://host/path"])
def test_ssh_host_shell_metacharacters_never_reach_transport(host):
    with pytest.raises(ValidationError):
        SSHAuthenticationInput(host=host, username="root", password="synthetic")


@pytest.mark.parametrize("path", ["/tmp/agent;id", "/tmp/$(id)", "/tmp/`id`", "/tmp/agent\n", "/tmp/../etc", "/"])
def test_ssh_install_shell_metacharacters_and_protected_roots_are_rejected(path):
    with pytest.raises(RemoteError) as rejected:
        _install_path(path)
    assert rejected.value.code == "SSH_INSTALL_PATH_INVALID"
