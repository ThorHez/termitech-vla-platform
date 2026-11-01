"""Terminals use synthetic SSH channels and isolated account databases only."""

import asyncio
from collections import deque
import threading
import time
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import delete, select
from starlette.websockets import WebSocketDisconnect

from vla_control import node_terminals as module
from vla_control.auth import AuthManager, register_auth_routes
from vla_control.auth_models import AuditEvent, User, UserSession
from vla_control.common import APIError, now, offset_time, uid
from vla_control.db import Database
from vla_control.models import Allocation, Command, Node
from vla_control.node_terminals import NodeTerminalManager, register_routes
from vla_control.remote_models import RemoteConnection, RemoteDeployment
from vla_control.settings import Settings


PASSWORD = "terminal-fixture-password"
ORIGIN = {"origin": "http://testserver"}


class Channel:
    def __init__(self):
        self.outputs = deque()
        self.lock = threading.Lock()
        self.closed = False
        self.writes = []
        self.sizes = []
        self.shells = 0

    def settimeout(self, timeout):
        self.timeout = timeout

    def get_pty(self, **kwargs):
        self.pty = kwargs

    def invoke_shell(self):
        self.shells += 1
        with self.lock:
            self.outputs.append(b"fixture prompt\r\n")

    def recv_ready(self):
        with self.lock:
            return bool(self.outputs)

    def recv(self, count):
        with self.lock:
            item = self.outputs.popleft()
            assert len(item) <= count
            return item

    def send(self, data):
        self.writes.append(data)
        with self.lock:
            self.outputs.append(data)
        return len(data)

    def resize_pty(self, **kwargs):
        self.sizes.append(kwargs)

    def exit_status_ready(self):
        return self.closed

    def close(self):
        self.closed = True


class Remote:
    def __init__(self, channel=None):
        self.channel = channel or Channel()
        self.closed = threading.Event()
        self.opened = []
        self.client = SimpleNamespace(get_transport=lambda: self)

    def connect(self):
        return self

    def is_active(self):
        return not self.closed.is_set()

    def open_session(self, **kwargs):
        self.opened.append(kwargs)
        return self.channel

    def close(self):
        self.channel.close()
        self.closed.set()


@pytest.fixture
def terminal(tmp_path, monkeypatch):
    db = Database("sqlite:///" + str(tmp_path / "terminals.db"))
    db.initialize()
    settings = Settings(database_url=str(db.engine.url), enrollment_token="unused-agent-token",
                        session_cookie_secure=False)
    auth = AuthManager(db, settings)
    admin = auth.create_user("admin", PASSWORD)
    opened = []

    def remote(connection):
        assert connection.host == "192.0.2.20" and connection.username == "fixture-ssh-user"
        assert connection.fingerprint == "SHA256:pinned-fixture"
        item = Remote()
        opened.append(item)
        return item

    manager = NodeTerminalManager(db, settings, SimpleNamespace(_remote=remote), auth)
    app = FastAPI()

    @app.exception_handler(APIError)
    async def api_error(_, exc):
        return JSONResponse(status_code=exc.status, content={"error": {"code": exc.code, "message": exc.message}})

    register_auth_routes(app, auth)
    register_routes(app, manager)
    node_id, connection_id = uid(), uid()
    with db.write() as session:
        session.add(Node(id=node_id, token_hash="a" * 64, name="fixture", hostname="fixture", status="OFFLINE",
                         last_seen=offset_time(-1000), created_at=now(), roots=[], output_root="/unused", gpus=[], metrics={}))
        session.add(RemoteConnection(id=connection_id, host="192.0.2.20", port=22, username="fixture-ssh-user",
                    encrypted_password="encrypted-not-returned", fingerprint="SHA256:pinned-fixture",
                    probe_result={}, install_dir="/agent", created_at=now(), expires_at=offset_time(-3600)))
        session.flush()
        session.add(RemoteDeployment(id=uid(), connection_id=connection_id, idempotency_key=uid(),
                    request_digest="b" * 64, target_key=uid(), onboarding_id=uid(), node_id=node_id,
                    state="SUCCEEDED", stage="SUCCEEDED", logs=[], history=[], agent_config={"_operation": "install"},
                    install_dir="/agent", control_plane_url="https://controller", connection_mode="direct",
                    tunnel_state="NOT_REQUIRED", created_at=now(), updated_at=now()))
    monkeypatch.setattr(module, "CHECK_INTERVAL", 0.02)
    with TestClient(app) as client:
        login = client.post("/api/v1/auth/login", json={"username": "admin", "password": PASSWORD}).json()
        headers = {**ORIGIN, "x-csrf-token": login["csrf_token"]}
        proof = client.post("/api/v1/auth/reauth", headers=headers, json={"password": PASSWORD})
        assert proof.status_code == 200
        value = SimpleNamespace(db=db, auth=auth, manager=manager, app=app, client=client, headers=headers,
                                node_id=node_id, connection_id=connection_id, opened=opened, user_id=admin["id"])
        yield value
        manager.close()
        assert all(item.closed.wait(1) for item in opened)
    db.close()


def ticket(terminal):
    response = terminal.client.post(f"/api/v1/nodes/{terminal.node_id}/terminal-sessions", headers=terminal.headers)
    assert response.status_code == 201, response.text
    return response.json()


def authenticate(ws, value, **changes):
    ws.send_json({"type": "auth", "ticket": value["ticket"], "cols": 100, "rows": 30, **changes})


def test_ticket_requires_csrf_and_recent_password_and_does_not_connect(terminal):
    url = f"/api/v1/nodes/{terminal.node_id}/terminal-sessions"
    response = terminal.client.post(url, headers=ORIGIN)
    assert response.status_code == 403 and response.json()["error"]["code"] == "CSRF_REJECTED"
    with terminal.db.write() as session:
        for login in session.scalars(select(UserSession)):
            login.reauthenticated_at = None
    response = terminal.client.post(url, headers=terminal.headers)
    assert response.status_code == 403 and response.json()["error"]["code"] == "REAUTH_REQUIRED"
    assert not terminal.opened


def test_offline_agent_ticket_is_cookie_bound_ephemeral_and_contains_no_secrets(terminal):
    value = ticket(terminal)
    assert value["target"] == {"host": "192.0.2.20", "port": 22, "username": "fixture-ssh-user"}
    assert value["ticket"] not in value["websocket_path"]
    assert now() < value["expires_at"] <= offset_time(61)
    assert "encrypted-not-returned" not in str(value)
    assert not terminal.opened
    with terminal.db.read() as session:
        assert list(session.scalars(select(Command))) == []
        assert list(session.scalars(select(Allocation))) == []


def test_pty_echo_resize_ping_and_close_preserve_last_output(terminal):
    value = ticket(terminal)
    with terminal.client.websocket_connect(value["websocket_path"], headers=ORIGIN) as ws:
        authenticate(ws, value)
        assert ws.receive_json() == {"type": "ready"}
        assert ws.receive_bytes() == b"fixture prompt\r\n"
        ws.send_json({"type": "input", "data": "echo fixture\n"})
        assert ws.receive_bytes() == b"echo fixture\n"
        ws.send_json({"type": "resize", "cols": 120, "rows": 40})
        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "pong"}
        remote = terminal.opened[0]
        assert remote.opened == [{"window_size": 128 * 1024, "max_packet_size": 16 * 1024, "timeout": 15}]
        assert remote.channel.pty == {"term": "xterm-256color", "width": 100, "height": 30}
        assert remote.channel.sizes == [{"width": 120, "height": 40}]
        with remote.channel.lock:
            remote.channel.outputs.append(b"final line\r\n")
        remote.channel.closed = True
        assert ws.receive_bytes() == b"final line\r\n"
        assert ws.receive_json()["type"] == "closed"
    assert remote.closed.wait(1)
    with terminal.db.read() as session:
        audit = list(session.scalars(select(AuditEvent).where(AuditEvent.action.like("terminal.%"))))
        assert {row.action for row in audit} == {"terminal.open", "terminal.close"}
        assert all("echo fixture" not in str(row.details) and value["ticket"] not in str(row.details) for row in audit)


@pytest.mark.parametrize("headers", [{}, {"origin": "https://evil.example"}, {"origin": "null"},
                                      {"origin": "http://testserver", "host": "evil.example"}])
def test_bad_handshake_is_rejected_before_ssh(terminal, headers):
    value = ticket(terminal)
    with pytest.raises(WebSocketDisconnect):
        with terminal.client.websocket_connect(value["websocket_path"], headers=headers):
            pass
    assert not terminal.opened


def test_ticket_is_single_use_and_does_not_allow_parallel_handshakes(terminal):
    value = ticket(terminal)
    with terminal.client.websocket_connect(value["websocket_path"], headers=ORIGIN) as first:
        with pytest.raises(WebSocketDisconnect):
            with terminal.client.websocket_connect(value["websocket_path"], headers=ORIGIN):
                pass
        authenticate(first, value)
        assert first.receive_json()["type"] == "ready"
        assert first.receive_bytes()
    with pytest.raises(WebSocketDisconnect):
        with terminal.client.websocket_connect(value["websocket_path"], headers=ORIGIN):
            pass
    assert len(terminal.opened) == 1


def test_wrong_ticket_opens_no_ssh_and_invalidates_claim(terminal):
    value = ticket(terminal)
    with terminal.client.websocket_connect(value["websocket_path"], headers=ORIGIN) as ws:
        authenticate(ws, value, ticket="wrong")
        assert ws.receive_json()["code"] == "TERMINAL_TICKET_INVALID"
        assert ws.receive_json()["type"] == "closed"
    assert value["id"] not in terminal.manager._terminals
    assert not terminal.opened


def test_claimed_ticket_expiry_rejects_auth_without_early_capacity_release(terminal):
    value = ticket(terminal)
    with terminal.client.websocket_connect(value["websocket_path"], headers=ORIGIN) as ws:
        record = terminal.manager._terminals[value["id"]]
        record.expires = 0
        with terminal.manager._lock:
            terminal.manager._prune()
        assert value["id"] in terminal.manager._terminals
        authenticate(ws, value)
        assert ws.receive_json()["code"] == "TERMINAL_TICKET_INVALID"
        assert ws.receive_json()["type"] == "closed"
    assert not terminal.opened


def test_first_frame_timeout_revokes_pending_connection(terminal, monkeypatch):
    monkeypatch.setattr(module, "AUTH_TIMEOUT", 0.02)
    value = ticket(terminal)
    with terminal.client.websocket_connect(value["websocket_path"], headers=ORIGIN) as ws:
        assert ws.receive_json()["code"] == "TERMINAL_TIMEOUT"
        assert ws.receive_json()["type"] == "closed"
    assert value["id"] not in terminal.manager._terminals and not terminal.opened


def test_ticket_limits_and_expiry_are_bounded(terminal):
    one, _ = ticket(terminal), ticket(terminal)
    response = terminal.client.post(f"/api/v1/nodes/{terminal.node_id}/terminal-sessions", headers=terminal.headers)
    assert response.status_code == 429
    terminal.manager._terminals[one["id"]].expires = 0
    assert ticket(terminal)["id"] != one["id"]
    with pytest.raises(WebSocketDisconnect):
        with terminal.client.websocket_connect(one["websocket_path"], headers=ORIGIN):
            pass


@pytest.mark.parametrize("mutation", ["logout", "disabled", "role", "binding", "retired"])
def test_active_terminal_closes_when_session_or_node_binding_is_revoked(terminal, mutation):
    value = ticket(terminal)
    with terminal.client.websocket_connect(value["websocket_path"], headers=ORIGIN) as ws:
        authenticate(ws, value)
        assert ws.receive_json()["type"] == "ready"
        assert ws.receive_bytes()
        with terminal.db.write() as session:
            if mutation == "logout":
                session.execute(delete(UserSession))
            elif mutation == "disabled":
                session.get(User, terminal.user_id).enabled = False
            elif mutation == "role":
                session.get(User, terminal.user_id).role = "viewer"
            elif mutation == "retired":
                session.get(Node, terminal.node_id).status = "DISCONNECTED"
            else:
                session.get(RemoteConnection, terminal.connection_id).fingerprint = "SHA256:changed"
        assert ws.receive_json()["type"] == "error"
        assert ws.receive_json()["type"] == "closed"
    assert terminal.opened[0].closed.wait(1)


def test_ssh_identity_change_during_connect_prevents_shell(terminal):
    remote = Remote()
    def connect():
        with terminal.db.write() as session:
            session.get(RemoteConnection, terminal.connection_id).fingerprint = "SHA256:changed"
    remote.connect = connect
    terminal.manager.remote_manager._remote = lambda _connection: remote
    value = ticket(terminal)
    with terminal.client.websocket_connect(value["websocket_path"], headers=ORIGIN) as ws:
        authenticate(ws, value)
        assert ws.receive_json()["code"] == "TERMINAL_BINDING_CHANGED"
        assert ws.receive_json()["type"] == "closed"
    assert remote.closed.wait(1) and remote.channel.shells == 0


def test_invalid_frame_stops_remote_without_echoing_unsafe_content(terminal):
    value = ticket(terminal)
    with terminal.client.websocket_connect(value["websocket_path"], headers=ORIGIN) as ws:
        authenticate(ws, value)
        assert ws.receive_json()["type"] == "ready"
        assert ws.receive_bytes()
        ws.send_json({"type": "input", "data": "private input", "unexpected": True})
        error = ws.receive_json()
        assert error["code"] == "TERMINAL_FRAME_INVALID" and "private input" not in str(error)
        assert ws.receive_json()["type"] == "closed"
    assert terminal.opened[0].closed.wait(1)


def test_slow_output_and_oversized_frames_are_bounded(monkeypatch):
    monkeypatch.setattr(module, "SEND_TIMEOUT", 0.01)
    async def run():
        queue = asyncio.Queue(maxsize=1)
        channel = Channel()
        channel.outputs.extend([b"a", b"b"])
        record = SimpleNamespace(channel=channel, stop=threading.Event())
        with pytest.raises(TimeoutError):
            await NodeTerminalManager._output(None, record, queue)
        assert queue.qsize() == 1
        class Socket:
            async def receive(self):
                return {"type": "websocket.receive", "text": "a" * (module.MAX_FRAME + 1)}
        with pytest.raises(APIError) as error:
            await NodeTerminalManager._message(Socket())
        assert error.value.code == "TERMINAL_FRAME_INVALID"
    asyncio.run(run())


def test_shutdown_interrupts_connect_without_opening_shell_or_blocking(terminal):
    entered = threading.Event()
    remote = Remote()
    def connect():
        entered.set()
        assert remote.closed.wait(2)
    remote.connect = connect
    terminal.manager.remote_manager._remote = lambda _connection: remote
    value = ticket(terminal)
    with terminal.client.websocket_connect(value["websocket_path"], headers=ORIGIN) as ws:
        authenticate(ws, value)
        assert entered.wait(1)
        started = time.monotonic()
        terminal.manager.close()
        assert time.monotonic() - started < 0.1
        assert ws.receive_json()["type"] == "error"
        assert ws.receive_json()["type"] == "closed"
    assert remote.closed.wait(1) and remote.channel.shells == 0


def test_closing_transports_continue_to_count_against_limit(terminal, monkeypatch):
    monkeypatch.setattr(module, "GLOBAL_LIMIT", 1)
    value = ticket(terminal)
    record = terminal.manager._terminals[value["id"]]
    record.consumed = True
    release, closing = threading.Event(), threading.Event()
    def close():
        closing.set()
        assert release.wait(2)
    record.remote = SimpleNamespace(close=close)
    terminal.manager._stop(record, "TEST")
    assert closing.wait(1)
    try:
        response = terminal.client.post(f"/api/v1/nodes/{terminal.node_id}/terminal-sessions", headers=terminal.headers)
        assert response.status_code == 429
    finally:
        release.set()
