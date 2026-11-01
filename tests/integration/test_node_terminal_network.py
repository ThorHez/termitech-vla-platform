"""Real TLS/WebSocket/SSH wire tests; the SSH peer never executes commands."""

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import socket
import ssl
import threading
import time

import httpx
import paramiko
import pytest
import uvicorn
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.sync.client import connect

from vla_control.common import now, offset_time, uid
from vla_control.main import create_app
from vla_control.models import Node
from vla_control.remote_models import RemoteConnection, RemoteDeployment
from vla_control.settings import Settings
from vla_control.ssh_remote import _fingerprint
from vla_control.transport_security import uvicorn_options


PASSWORD = "isolated-terminal-network-123"
SSH_PASSWORD = "synthetic-ssh-password"
ROOT = Path(__file__).resolve().parents[2]


class SyntheticSSH:
    """An authenticated SSH PTY echo peer, deliberately without shell execution."""

    def __init__(self):
        self.key = paramiko.RSAKey.generate(2048)
        self.socket = socket.socket()
        self.socket.bind(("127.0.0.1", 0))
        self.socket.listen(8)
        self.socket.settimeout(.1)
        self.port = self.socket.getsockname()[1]
        self.stopping = threading.Event()
        self.connections = []
        self.workers = []
        self.sizes = []
        self.shells = 0
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self.accept, daemon=True)
        self.thread.start()

    def accept(self):
        while not self.stopping.is_set():
            try:
                stream, _ = self.socket.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            worker = threading.Thread(target=self.serve, args=(stream,), daemon=True)
            self.workers.append(worker)
            worker.start()

    def serve(self, stream):
        peer = self
        shell = threading.Event()

        class Server(paramiko.ServerInterface):
            def check_auth_password(self, username, password):
                return paramiko.AUTH_SUCCESSFUL if (username, password) == ("fixture", SSH_PASSWORD) else paramiko.AUTH_FAILED

            def get_allowed_auths(self, username):
                return "password"

            def check_channel_request(self, kind, channel_id):
                return paramiko.OPEN_SUCCEEDED if kind == "session" else paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED

            def check_channel_pty_request(self, channel, term, width, height, pixelwidth, pixelheight, modes):
                peer.sizes.append((width, height))
                return term == b"xterm-256color"

            def check_channel_window_change_request(self, channel, width, height, pixelwidth, pixelheight):
                peer.sizes.append((width, height))
                return True

            def check_channel_shell_request(self, channel):
                peer.shells += 1
                shell.set()
                return True

        transport = paramiko.Transport(stream)
        self.connections.append(transport)
        try:
            transport.add_server_key(self.key)
            transport.start_server(server=Server())
            channel = transport.accept(5)
            if channel is None or not shell.wait(5):
                return
            channel.settimeout(.1)
            channel.sendall("\x1b[32m隔离 SSH 终端\x1b[0m\r\n$ ".encode())
            while not self.stopping.is_set():
                try:
                    data = channel.recv(16384)
                except socket.timeout:
                    continue
                if not data:
                    return
                channel.sendall(data)
        except (EOFError, OSError, paramiko.SSHException):
            pass
        finally:
            transport.close()
            self.closed.set()

    def close(self):
        self.stopping.set()
        self.socket.close()
        for transport in self.connections:
            transport.close()
        self.thread.join(3)
        for worker in self.workers:
            worker.join(3)


@pytest.fixture
def terminal_network(tmp_path):
    spec = importlib.util.spec_from_file_location("terminal_tls", ROOT / "scripts/generate_local_tls.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    tls = generator.generate(tmp_path / "tls")
    context = ssl.create_default_context(cafile=tls["ca_certificate"])
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    origin = f"https://127.0.0.1:{port}"
    settings = Settings(database_url="sqlite:///" + str(tmp_path / "network.db"),
                        enrollment_token="isolated-terminal-agent", scheduler_enabled=False,
                        require_https=True, public_url=origin,
                        tls_cert_file=tls["certificate"], tls_key_file=tls["private_key"])
    app = create_app(settings)
    app.state.auth.create_user("network-admin", PASSWORD, "admin")
    ssh = SyntheticSSH()
    node_id, connection_id, stamp = uid(), uid(), now()
    with app.state.db.write() as session:
        session.add(Node(id=node_id, name="isolated-peer", hostname="localhost", status="OFFLINE",
                         token_hash="a" * 64, created_at=stamp, last_seen=stamp, roots=["/fixture"], output_root="/fixture/out"))
        session.add(RemoteConnection(id=connection_id, host="127.0.0.1", port=ssh.port, username="fixture",
                    encrypted_password=app.state.remote_manager.vault.encrypt(SSH_PASSWORD), fingerprint=_fingerprint(ssh.key),
                    probe_result={}, install_dir="/fixture/agent", created_at=stamp, expires_at=offset_time(-60)))
        session.flush()
        session.add(RemoteDeployment(id=uid(), connection_id=connection_id, idempotency_key=uid(), request_digest="b" * 64,
                    target_key=uid(), onboarding_id=uid(), node_id=node_id, state="SUCCEEDED", stage="SUCCEEDED",
                    logs=[], history=[], agent_config={"_operation": "install"}, install_dir="/fixture/agent",
                    control_plane_url=origin, connection_mode="direct", tunnel_state="NOT_REQUIRED",
                    created_at=stamp, updated_at=stamp, finished_at=stamp))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", **uvicorn_options(settings)))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            assert thread.is_alive() and time.monotonic() < deadline, "isolated TLS server failed to start"
            time.sleep(.02)
        with httpx.Client(base_url=origin, verify=context, trust_env=False, timeout=5, headers={"Origin": origin}) as client:
            login = client.post("/api/v1/auth/login", json={"username": "network-admin", "password": PASSWORD})
            assert login.status_code == 200, login.text
            client.headers["X-CSRF-Token"] = login.json()["csrf_token"]
            assert client.post("/api/v1/auth/reauth", json={"password": PASSWORD}).status_code == 200
            yield app, client, ssh, node_id, origin, context
    finally:
        server.should_exit = True
        thread.join(15)
        listener.close()
        ssh.close()
        assert not thread.is_alive(), "isolated WebSocket server failed to stop"


def ticket(fixture):
    _, client, _, node_id, *_ = fixture
    response = client.post(f"/api/v1/nodes/{node_id}/terminal-sessions")
    assert response.status_code in {200, 201}, response.text
    return response.json()


@contextmanager
def websocket(fixture, value, *, origin=None):
    _, client, _, _, base, context = fixture
    cookie = "; ".join(f"{key}={value}" for key, value in client.cookies.items())
    with connect(base.replace("https:", "wss:") + value["websocket_path"], ssl=context, proxy=None,
                 origin=origin or base, additional_headers={"Cookie": cookie}, open_timeout=5, close_timeout=2) as ws:
        ws.send(json.dumps({"type": "auth", "ticket": value["ticket"], "cols": 100, "rows": 30}))
        yield ws


def test_tls_websocket_pinned_ssh_pty_unicode_control_input_resize_and_cleanup(terminal_network):
    value = ticket(terminal_network)
    ssh = terminal_network[2]
    with websocket(terminal_network, value) as ws:
        assert json.loads(ws.recv(timeout=5))["type"] == "ready"
        assert "隔离 SSH 终端" in ws.recv(timeout=5).decode()
        raw = "中文\t\x03\x1b[A\r\n"
        ws.send(json.dumps({"type": "input", "data": raw}))
        assert ws.recv(timeout=5).decode() == raw
        ws.send(json.dumps({"type": "resize", "cols": 132, "rows": 44}))
        ws.send(json.dumps({"type": "ping"}))
        assert json.loads(ws.recv(timeout=5))["type"] == "pong"
        deadline = time.monotonic() + 2
        while (132, 44) not in ssh.sizes and time.monotonic() < deadline:
            time.sleep(.02)
        assert ssh.sizes == [(100, 30), (132, 44)]
    assert ssh.closed.wait(5)
    assert ssh.shells == 1


def test_actual_websocket_transport_rejects_oversized_frame(terminal_network):
    with websocket(terminal_network, ticket(terminal_network)) as ws:
        assert json.loads(ws.recv(timeout=5))["type"] == "ready"
        assert isinstance(ws.recv(timeout=5), bytes)
        ws.send(json.dumps({"type": "input", "data": "x" * 40000}))
        with pytest.raises(ConnectionClosed) as error:
            ws.recv(timeout=5)
        assert error.value.rcvd.code == 1009


def test_real_upgrade_rejects_foreign_origin_without_ssh_connection(terminal_network):
    with pytest.raises(InvalidStatus) as error:
        with websocket(terminal_network, ticket(terminal_network), origin="https://foreign.example"):
            pytest.fail("foreign origin must not upgrade")
    assert error.value.response.status_code == 403
    assert not terminal_network[2].connections


def test_logout_ends_real_websocket_and_ssh_channel(terminal_network):
    _, client, ssh, *_ = terminal_network
    with websocket(terminal_network, ticket(terminal_network)) as ws:
        assert json.loads(ws.recv(timeout=5))["type"] == "ready"
        assert isinstance(ws.recv(timeout=5), bytes)
        assert client.post("/api/v1/auth/logout").status_code == 204
        message = json.loads(ws.recv(timeout=5))
        assert message["type"] == "error" and message["code"] == "TERMINAL_SESSION_ENDED"
        assert json.loads(ws.recv(timeout=5))["type"] == "closed"
        with pytest.raises(ConnectionClosed):
            ws.recv(timeout=5)
    assert ssh.closed.wait(5)
