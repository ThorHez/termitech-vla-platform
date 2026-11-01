"""Independent HTTP/WebSocket perimeter checks for the routed terminal feature.

All state is temporary and every attempted SSH construction is a test failure.
"""

import time

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select
from starlette.websockets import WebSocketDisconnect

from vla_control.auth import COOKIE_NAME
from vla_control.auth_models import User, UserSession
from vla_control.common import now, offset_time, uid
from vla_control.main import create_app
from vla_control.models import Node
from vla_control.remote_models import RemoteConnection, RemoteDeployment
from vla_control.settings import Settings
from vla_control.transport_security import uvicorn_options

from .test_node_terminals import authenticate, terminal, ticket as channel_ticket, ORIGIN as CHANNEL_ORIGIN


ORIGIN = "https://terminal.example:8443"
PASSWORD = "isolated-terminal-review-password"
NODE_ID = "review-node"
POST = f"/api/v1/nodes/{NODE_ID}/terminal-sessions"
WS = "/api/v1/node-terminals/unknown-session/ws"


@pytest.fixture
def secured(tmp_path, monkeypatch):
    app = create_app(Settings(database_url="sqlite:///" + str(tmp_path / "terminal-review.db"),
                              enrollment_token="separate-agent-credential", scheduler_enabled=False,
                              public_url=ORIGIN, require_https=True))
    app.state.auth.create_user("terminal-admin", PASSWORD, "admin")
    stamp = now()
    with app.state.db.write() as session:
        session.add(Node(id=NODE_ID, name="review node", hostname="review-node", token_hash="review-hash",
                         status="OFFLINE", last_seen=stamp, created_at=stamp, roots=["/machine"],
                         output_root="/machine/output", gpus=[], metrics={}))
        connection = RemoteConnection(id=uid(), host="192.0.2.77", port=22, username="review-user",
            encrypted_password="encrypted-test-value", fingerprint="SHA256:synthetic-review-fingerprint",
            probe_result={}, install_dir="/machine/agent", created_at=stamp, expires_at=offset_time(-100))
        session.add(connection)
        session.flush()
        session.add(RemoteDeployment(id=uid(), connection_id=connection.id, idempotency_key=uid(),
            request_digest="d" * 64, target_key=uid(), onboarding_id=uid(), node_id=NODE_ID,
            state="SUCCEEDED", stage="SUCCEEDED", logs=[], history=[], agent_config={"_operation": "install"},
            install_dir="/machine/agent", control_plane_url=ORIGIN, connection_mode="direct",
            tunnel_state="NOT_REQUIRED", created_at=stamp, updated_at=stamp, finished_at=stamp))
    remote_attempts = []

    def unexpected_remote(*_args, **_kwargs):
        remote_attempts.append(True)
        raise AssertionError("Denied terminal requests must not construct an SSH client")

    monkeypatch.setattr(app.state.remote_manager, "_remote", unexpected_remote)
    with TestClient(app, base_url=ORIGIN) as client:
        yield app, client
    assert remote_attempts == []


def login(client, username="terminal-admin"):
    client.cookies.clear()
    result = client.post("/api/v1/auth/login", json={"username": username, "password": PASSWORD})
    assert result.status_code == 200, result.text
    client.headers["X-CSRF-Token"] = result.json()["csrf_token"]
    return result.json()


def ticket(client):
    assert client.post("/api/v1/auth/reauth", json={"password": PASSWORD}).status_code == 200
    response = client.post(POST, json={})
    assert response.status_code == 201, response.text
    return response.json()


def websocket_denied(client, *, url=None, headers=None, issued=None):
    """No denied request may receive ready or binary terminal data."""
    supplied_headers = dict(headers or {"Origin": ORIGIN})
    cookie = client.cookies.get(COOKIE_NAME)
    if cookie:
        # A hostile Host is checked even when a valid cookie is explicitly
        # supplied, rather than passing only because the cookie jar omitted it.
        supplied_headers["Cookie"] = f"{COOKIE_NAME}={cookie}"
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(url or "wss://terminal.example:8443" + (issued["websocket_path"] if issued else WS),
                                      headers=supplied_headers) as websocket:
            websocket.send_json({"type": "auth", "ticket": issued["ticket"] if issued else "invalid",
                                 "cols": 80, "rows": 24})
            while True:
                message = websocket.receive()
                if message["type"] == "websocket.close":
                    raise WebSocketDisconnect(message.get("code", 1000))
                assert message.get("bytes") is None
                assert '"ready"' not in message.get("text", "")


def test_terminal_post_requires_cookie_login_and_csrf(secured):
    _, client = secured
    response = client.post(POST, json={})
    assert response.status_code == 401
    login(client)
    client.headers.pop("X-CSRF-Token")
    response = client.post(POST, json={})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "CSRF_REJECTED"


def test_terminal_post_requires_recent_password_before_node_lookup(secured):
    _, client = secured
    login(client)
    response = client.post(POST, json={})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "REAUTH_REQUIRED"
    response = client.post("/api/v1/auth/reauth", json={"password": PASSWORD})
    assert response.status_code == 200
    assert client.post("/api/v1/nodes/unknown-node/terminal-sessions", json={}).status_code == 404


@pytest.mark.parametrize("role", ["viewer", "operator"])
def test_non_admin_cannot_acquire_or_use_terminal(secured, role):
    app, client = secured
    app.state.auth.create_user("terminal-" + role, PASSWORD, role)
    login(client, "terminal-" + role)
    response = client.post(POST, json={})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "FORBIDDEN"
    websocket_denied(client)


def test_unauthenticated_terminal_websocket_is_rejected(secured):
    websocket_denied(secured[1])


def test_legacy_administrator_bearer_cannot_grant_an_interactive_shell(tmp_path, monkeypatch):
    app = create_app(Settings(database_url="sqlite:///" + str(tmp_path / "legacy-terminal.db"),
        enrollment_token="separate-agent-credential", admin_token="isolated-legacy-administrator",
        legacy_admin_enabled=True, scheduler_enabled=False, public_url=ORIGIN, require_https=True))
    attempted = []
    monkeypatch.setattr(app.state.remote_manager, "_remote", lambda *_args: attempted.append(True))
    with TestClient(app, base_url=ORIGIN) as client:
        client.headers["Authorization"] = "Bearer isolated-legacy-administrator"
        # Legacy auth is demonstrably active on ordinary management reads.
        assert client.get("/api/v1/nodes").status_code == 200
        response = client.post(POST, json={})
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "ACCOUNT_REQUIRED"
        websocket_denied(client)
    assert attempted == []


@pytest.mark.parametrize("origin", [
    "https://evil.example", "https://terminal.example:9443", "http://terminal.example:8443",
    "null", "https://terminal.example:8443/extra", "https://user@terminal.example:8443",
    "https://terminal.example:8443?query=secret", "https://terminal.example:8443#fragment",
])
def test_websocket_origin_cannot_be_forged_to_bypass_cookie_boundary(secured, origin):
    _, client = secured
    login(client)
    websocket_denied(client, headers={"Origin": origin}, issued=ticket(client))


def test_missing_websocket_origin_is_rejected_even_with_valid_cookie(secured):
    _, client = secured
    login(client)
    websocket_denied(client, headers={"X-Review": "origin-is-absent"}, issued=ticket(client))


@pytest.mark.parametrize("url", [
    "ws://terminal.example:8443", "wss://evil.example:8443", "wss://terminal.example:9443",
])
def test_websocket_rejects_plaintext_and_unapproved_host(secured, url):
    _, client = secured
    login(client)
    issued = ticket(client)
    websocket_denied(client, url=url + issued["websocket_path"], issued=issued)


@pytest.mark.parametrize("change", ["expired", "idle", "disabled", "force-password-change", "downgrade"])
def test_invalidated_login_cannot_start_websocket(secured, change):
    app, client = secured
    login(client)
    issued = ticket(client)
    with app.state.db.write() as session:
        user = session.scalar(select(User).where(User.username == "terminal-admin"))
        row = session.scalar(select(UserSession).where(UserSession.user_id == user.id))
        if change == "expired":
            row.expires_at = offset_time(-1)
        elif change == "idle":
            row.last_activity_at = offset_time(-3601)
        elif change == "disabled":
            user.enabled = False
        elif change == "force-password-change":
            user.must_change_password = True
        else:
            user.role = "viewer"
    websocket_denied(client, issued=issued)


def test_websocket_server_has_a_bounded_transport_and_no_compression():
    settings = Settings(enrollment_token="separate", scheduler_enabled=False)
    options = uvicorn_options(settings)
    assert options["ws"] == "websockets-sansio"
    assert options["ws_max_size"] == 32 * 1024
    assert options["ws_per_message_deflate"] is False


@pytest.mark.parametrize("expiry", ["absolute-policy", "platform-idle", "terminal-idle", "must-change-password"])
def test_running_terminal_applies_both_eight_hour_and_one_hour_deadlines(terminal, expiry):
    issued = channel_ticket(terminal)
    with terminal.client.websocket_connect(issued["websocket_path"], headers=CHANNEL_ORIGIN) as websocket:
        authenticate(websocket, issued)
        assert websocket.receive_json()["type"] == "ready"
        assert websocket.receive_bytes()
        with terminal.db.write() as session:
            login_row = session.scalar(select(UserSession).where(UserSession.user_id == terminal.user_id))
            if expiry == "absolute-policy":
                # An old cookie cannot bypass the current absolute policy even
                # if a historical stored expiry was longer.
                login_row.created_at = offset_time(-28801)
                login_row.expires_at = offset_time(86400)
            elif expiry == "platform-idle":
                login_row.last_activity_at = offset_time(-3601)
            elif expiry == "must-change-password":
                session.get(User, terminal.user_id).must_change_password = True
        if expiry == "terminal-idle":
            terminal.manager._terminals[issued["id"]].last_input = time.monotonic() - 3601
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert error["code"] in {"TERMINAL_SESSION_ENDED", "TERMINAL_IDLE_TIMEOUT"}
        assert websocket.receive_json()["type"] == "closed"
    assert terminal.opened[0].closed.wait(1)


def test_running_terminal_does_not_require_fresh_password_every_five_minutes(terminal):
    issued = channel_ticket(terminal)
    with terminal.client.websocket_connect(issued["websocket_path"], headers=CHANNEL_ORIGIN) as websocket:
        authenticate(websocket, issued)
        assert websocket.receive_json()["type"] == "ready"
        assert websocket.receive_bytes()
        with terminal.db.write() as session:
            login_row = session.scalar(select(UserSession).where(UserSession.user_id == terminal.user_id))
            login_row.reauthenticated_at = offset_time(-301)
            login_row.last_activity_at = offset_time(-60)
            original_expiry, original_created = login_row.expires_at, login_row.created_at
        websocket.send_json({"type": "input", "data": "still-authorized\r"})
        assert websocket.receive_bytes() == b"still-authorized\r"
        with terminal.db.read() as session:
            login_row = session.scalar(select(UserSession).where(UserSession.user_id == terminal.user_id))
            assert login_row.last_activity_at > offset_time(-10)
            assert login_row.expires_at == original_expiry
            assert login_row.created_at == original_created


def test_ping_and_resize_do_not_extend_human_activity_or_terminal_input_deadline(terminal):
    issued = channel_ticket(terminal)
    with terminal.client.websocket_connect(issued["websocket_path"], headers=CHANNEL_ORIGIN) as websocket:
        authenticate(websocket, issued)
        assert websocket.receive_json()["type"] == "ready"
        assert websocket.receive_bytes()
        with terminal.db.write() as session:
            row = session.scalar(select(UserSession).where(UserSession.user_id == terminal.user_id))
            row.last_activity_at = offset_time(-90)
            original_activity = row.last_activity_at
        last_input = terminal.manager._terminals[issued["id"]].last_input
        websocket.send_json({"type": "resize", "cols": 110, "rows": 40})
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json() == {"type": "pong"}
        with terminal.db.read() as session:
            row = session.scalar(select(UserSession).where(UserSession.user_id == terminal.user_id))
            assert row.last_activity_at == original_activity
        assert terminal.manager._terminals[issued["id"]].last_input == last_input
