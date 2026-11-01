"""Account-bound, ephemeral SSH terminals. No terminal content is persisted.

Each accepted terminal owns a new pinned SSH client. It never borrows an Agent
tunnel, changes an execution profile, or reserves GPUs. Tickets live only in
this single control-plane process and are invalidated by restart.
"""

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import secrets
import socket
import threading
import time
from urllib.parse import urlsplit

import anyio
from fastapi import Request, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from .auth import COOKIE_NAME, Principal, ROLE_PERMISSIONS
from .auth_models import User, UserSession
from .checkpoint_transfers import _connection
from .common import APIError, now, offset_time, uid
from .models import Node
from .remote_models import RemoteDeployment
from .ssh_remote import RemoteError
from .transport_security import TransportSecurityMiddleware


MAX_FRAME = 32 * 1024
OUTPUT_CHUNK = 16 * 1024
OUTPUT_QUEUE = 16  # 256 KiB, in addition to the bounded SSH receive window.
TICKET_SECONDS = 60
AUTH_TIMEOUT = 5
SEND_TIMEOUT = 5
CHECK_INTERVAL = 2
USER_LIMIT = 2
NODE_LIMIT = 2
GLOBAL_LIMIT = 8


def _digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _identity(connection):
    # A credential change invalidates an issued ticket without retaining a
    # second plaintext password or exposing a credential-derived value.
    return tuple(getattr(connection, key) for key in ("id", "host", "port", "username", "fingerprint")) + (
        _digest(connection.encrypted_password),)


def _dimensions(cols, rows):
    if type(cols) is not int or type(rows) is not int or not 1 <= cols <= 500 or not 1 <= rows <= 200:
        raise APIError(422, "TERMINAL_SIZE_INVALID", "终端列数需为 1–500，行数需为 1–200")
    return cols, rows


def _origin(value):
    try:
        if not value or any(c in value for c in "\\\r\n\t "):
            raise ValueError
        parsed = urlsplit(value)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.path not in {"", "/"} or parsed.query or parsed.fragment
                or parsed.netloc.endswith(":")):
            raise ValueError
        return parsed.scheme, parsed.hostname.lower(), parsed.port or (443 if parsed.scheme == "https" else 80)
    except (ValueError, UnicodeError):
        raise APIError(403, "ORIGIN_REJECTED", "终端请求来源不被允许") from None


@dataclass
class _Terminal:
    id: str
    ticket_digest: str
    user_id: str
    username: str
    session_id: str
    node_id: str
    identity: tuple
    expires: float
    expires_at: str
    consumed: bool = False
    claimed: bool = False
    cleanup_started: bool = False
    connecting: bool = False
    connect_done: threading.Event = field(default_factory=threading.Event)
    stop: threading.Event = field(default_factory=threading.Event)
    remote: object = None
    channel: object = None
    last_input: float = field(default_factory=time.monotonic)
    last_touch: float = 0
    reason: str = "TERMINAL_CLOSED"


class NodeTerminalManager:
    def __init__(self, db, settings, remote_manager, auth):
        self.db, self.settings = db, settings
        self.remote_manager, self.auth = remote_manager, auth
        self._lock = threading.RLock()
        self._terminals: dict[str, _Terminal] = {}
        self._closing = False
        self._transport = TransportSecurityMiddleware(None, settings)

    def _principal(self, connection) -> Principal:
        # A legacy bearer token must never grant an interactive shell.
        if not connection.cookies.get(COOKIE_NAME):
            raise APIError(401, "ACCOUNT_REQUIRED", "请使用管理员账户登录后打开终端")
        principal = self.auth.current(connection)
        if principal.legacy or not principal.session_id:
            raise APIError(401, "ACCOUNT_REQUIRED", "终端需要有效的用户登录会话")
        if principal.must_change_password:
            raise APIError(403, "PASSWORD_CHANGE_REQUIRED", "请先修改初始密码")
        if "machines.manage" not in principal.permissions:
            raise APIError(403, "FORBIDDEN", "当前账户没有机器终端权限")
        return principal

    def _bound_connection(self, session, node_id):
        node = session.get(Node, node_id)
        if node is None:
            raise APIError(404, "NOT_FOUND", "机器不存在")
        if node.status in {"DISCONNECTED", "REMOVED"}:
            raise APIError(409, "NODE_CONNECTION_REMOVED", "机器已断开或卸载，请重新接入")
        if session.scalar(select(RemoteDeployment.id).where(RemoteDeployment.node_id == node_id,
                RemoteDeployment.state.in_({"QUEUED", "RUNNING", "WAITING_AGENT"})).limit(1)):
            raise APIError(409, "NODE_MAINTENANCE", "机器正在执行接入管理操作，请稍后打开终端")
        connection = _connection(session, node_id)
        if not connection.fingerprint or not connection.encrypted_password:
            raise APIError(409, "TERMINAL_SSH_UNAVAILABLE", "机器没有可用的已验证 SSH 管理凭据")
        return connection

    def create(self, request: Request, node_id: str):
        self.auth.require(request, "machines.manage")
        principal = self._principal(request)
        with self.db.read() as session:
            self.auth.require_recent_password(session.get(UserSession, principal.session_id))
            connection = self._bound_connection(session, node_id)
            identity = _identity(connection)
        raw = secrets.token_urlsafe(32)
        record = _Terminal(uid(), _digest(raw), principal.user_id, principal.username, principal.session_id,
                           node_id, identity, time.monotonic() + TICKET_SECONDS, offset_time(TICKET_SECONDS))
        with self._lock:
            self._prune()
            if self._closing:
                raise APIError(503, "TERMINAL_SHUTTING_DOWN", "控制台正在关闭，请稍后重试")
            current = list(self._terminals.values())
            if (len(current) >= GLOBAL_LIMIT or sum(r.user_id == record.user_id for r in current) >= USER_LIMIT
                    or sum(r.node_id == node_id for r in current) >= NODE_LIMIT):
                raise APIError(429, "TERMINAL_LIMIT", "终端连接数已达上限，请关闭已有终端后重试")
            self._terminals[record.id] = record
        # No secrets in the URL, server logs, audit details, or durable storage.
        return {"id": record.id, "ticket": raw, "websocket_path": f"/api/v1/node-terminals/{record.id}/ws",
                "target": {"username": identity[3], "host": identity[1], "port": identity[2]},
                "expires_at": record.expires_at}

    def _prune(self):
        for key, record in list(self._terminals.items()):
            if not record.consumed and not record.claimed and record.expires <= time.monotonic():
                self._terminals.pop(key, None)

    def _check_handshake(self, websocket):
        scope = websocket.scope
        if self.settings.require_https and scope.get("scheme") != "wss":
            raise APIError(403, "HTTPS_REQUIRED", "请通过 HTTPS 打开机器终端")
        raw_hosts = [v for k, v in scope.get("headers", []) if k.lower() == b"host"]
        raw_origins = [v for k, v in scope.get("headers", []) if k.lower() == b"origin"]
        if len(raw_hosts) != 1 or len(raw_origins) != 1:
            raise APIError(403, "ORIGIN_REJECTED", "终端必须从控制台页面打开")
        secure = scope.get("scheme") == "wss"
        host_scope = {**scope, "scheme": "https" if secure else "http"}
        if self._transport.origin and not self._transport.allowed_host(host_scope):
            raise APIError(403, "HOST_REJECTED", "终端请求地址不被允许")
        try:
            host, supplied = raw_hosts[0].decode("ascii"), raw_origins[0].decode("ascii")
        except UnicodeError:
            raise APIError(403, "ORIGIN_REJECTED", "终端请求来源不被允许") from None
        if _origin(supplied) != _origin(("https://" if secure else "http://") + host):
            raise APIError(403, "ORIGIN_REJECTED", "终端请求来源不被允许")

    def _consume(self, terminal_id, ticket, principal):
        if not isinstance(ticket, str) or len(ticket) > 200:
            raise APIError(403, "TERMINAL_TICKET_INVALID", "终端连接凭证无效或已过期")
        with self._lock:
            self._prune()
            record = self._terminals.get(terminal_id)
            if (self._closing or record is None or record.consumed or record.expires <= time.monotonic()
                    or record.user_id != principal.user_id
                    or record.session_id != principal.session_id
                    or not hmac.compare_digest(record.ticket_digest, _digest(ticket))):
                raise APIError(403, "TERMINAL_TICKET_INVALID", "终端连接凭证无效或已过期")
            record.consumed = True
            record.ticket_digest = ""
            return record

    def _validate_live(self, record, *, touch=False, recent=False):
        with self.db.write() if touch else self.db.read() as session:
            if touch:
                self.auth.lock_accounts(session)
            user, login = session.get(User, record.user_id), session.get(UserSession, record.session_id)
            if (not user or not user.enabled or user.must_change_password or not login
                    or login.user_id != record.user_id or not self.auth.session_active(login)
                    or "machines.manage" not in ROLE_PERMISSIONS.get(user.role, frozenset())):
                raise APIError(401, "TERMINAL_SESSION_ENDED", "登录状态或权限已变更，请重新登录")
            if recent:
                self.auth.require_recent_password(login)
            connection = self._bound_connection(session, record.node_id)
            if _identity(connection) != record.identity:
                raise APIError(409, "TERMINAL_BINDING_CHANGED", "机器 SSH 绑定已变更，请重新打开终端")
            if touch:
                login.last_activity_at = now()
            return connection

    def _connect(self, record, cols, rows):
        try:
            self._open(record, cols, rows)
        finally:
            record.connect_done.set()

    def _open(self, record, cols, rows):
        connection = self._validate_live(record, recent=True)
        remote = self.remote_manager._remote(connection)
        with self._lock:
            if self._closing or record.stop.is_set():
                raise APIError(503, "TERMINAL_CLOSED", "终端连接已关闭")
            record.remote = remote
            record.connecting = True
        remote.connect()
        if record.stop.is_set():
            remote.close()
            raise APIError(409, "TERMINAL_CLOSED", "终端连接已关闭")
        # Recheck after a possibly slow password exchange before opening shell.
        self._validate_live(record)
        transport = remote.client.get_transport()
        if transport is None or not transport.is_active():
            raise APIError(502, "SSH_DISCONNECTED", "SSH 连接已断开，请重新打开终端")
        channel = transport.open_session(window_size=128 * 1024, max_packet_size=OUTPUT_CHUNK, timeout=15)
        with self._lock:
            record.channel = channel
            if self._closing or record.stop.is_set():
                channel.close()
                raise APIError(409, "TERMINAL_CLOSED", "终端连接已关闭")
        channel.settimeout(0.5)
        channel.get_pty(term="xterm-256color", width=cols, height=rows)
        self._validate_live(record)
        if record.stop.is_set():
            channel.close()
            raise APIError(409, "TERMINAL_CLOSED", "终端连接已关闭")
        channel.invoke_shell()

    @staticmethod
    async def _send(websocket, value):
        lock = getattr(websocket, "_node_terminal_send_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            websocket._node_terminal_send_lock = lock
        async def write():
            async with lock:
                action = websocket.send_bytes(value) if isinstance(value, bytes) else websocket.send_json(value)
                await action
        await asyncio.wait_for(write(), SEND_TIMEOUT)

    @staticmethod
    async def _message(websocket):
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(message.get("code", 1000))
        raw = message.get("text")
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_FRAME:
            raise APIError(422, "TERMINAL_FRAME_INVALID", "终端消息超出限制或格式不正确")
        try:
            value = json.loads(raw)
        except (ValueError, RecursionError):
            raise APIError(422, "TERMINAL_FRAME_INVALID", "终端消息必须使用 JSON 格式") from None
        if not isinstance(value, dict):
            raise APIError(422, "TERMINAL_FRAME_INVALID", "终端消息格式不正确")
        return value

    async def _input(self, websocket, record):
        # A bounded byte budget prevents endless small frames flooding SSH.
        budget, refill = 128 * 1024, time.monotonic()
        while not record.stop.is_set():
            message = await self._message(websocket)
            kind = message.get("type")
            if kind == "input" and set(message) == {"type", "data"}:
                data = message["data"]
                if not isinstance(data, str):
                    raise APIError(422, "TERMINAL_FRAME_INVALID", "终端输入格式不正确")
                encoded = data.encode("utf-8")
                current = time.monotonic()
                budget = min(128 * 1024, budget + (current - refill) * 64 * 1024)
                refill = current
                if len(encoded) > budget:
                    raise APIError(429, "TERMINAL_INPUT_LIMIT", "终端输入过快，请重新打开连接")
                budget -= len(encoded)
                if encoded:
                    if current - record.last_touch >= 30:
                        await asyncio.to_thread(self._validate_live, record, touch=True)
                        record.last_touch = current
                    record.last_input = current
                    while encoded and not record.stop.is_set():
                        count = await asyncio.wait_for(asyncio.to_thread(record.channel.send, encoded), SEND_TIMEOUT)
                        if count <= 0:
                            raise APIError(502, "SSH_DISCONNECTED", "SSH 连接已关闭")
                        encoded = encoded[count:]
            elif kind == "resize" and set(message) == {"type", "cols", "rows"}:
                cols, rows = _dimensions(message["cols"], message["rows"])
                await asyncio.wait_for(asyncio.to_thread(record.channel.resize_pty, width=cols, height=rows), SEND_TIMEOUT)
            elif kind == "ping" and set(message) == {"type"}:
                await self._send(websocket, {"type": "pong"})
            else:
                raise APIError(422, "TERMINAL_FRAME_INVALID", "不支持的终端消息")

    async def _output(self, record, queue):
        channel = record.channel
        while not record.stop.is_set():
            if channel.recv_ready():
                value = channel.recv(OUTPUT_CHUNK)
                if not value:
                    break
                await asyncio.wait_for(queue.put(value), SEND_TIMEOUT)
            elif channel.closed or channel.exit_status_ready():
                break
            else:
                await asyncio.sleep(0.02)
        await asyncio.wait_for(queue.put(None), SEND_TIMEOUT)

    async def _writer(self, websocket, queue):
        while True:
            value = await queue.get()
            if value is None:
                return
            await self._send(websocket, value)

    async def _watch(self, record):
        while not record.stop.is_set():
            if time.monotonic() - record.last_input >= self.settings.session_idle_seconds:
                raise APIError(401, "TERMINAL_IDLE_TIMEOUT", "终端长时间无输入，连接已关闭")
            await asyncio.to_thread(self._validate_live, record)
            await asyncio.sleep(CHECK_INTERVAL)

    def _stop(self, record, reason):
        with self._lock:
            if record.cleanup_started:
                return
            record.cleanup_started = True
            record.reason = reason
            record.stop.set()
            if record.remote is None:
                self._terminals.pop(record.id, None)
                return
        # Never block the ASGI event loop or shutdown on a network close/join.
        def cleanup():
            try:
                with suppress(Exception):
                    if record.channel is not None:
                        record.channel.close()
                with suppress(Exception):
                    if record.remote is not None:
                        record.remote.close()
                # SSHRemote.connect can race close before its own client is
                # assigned. Its caller checks stop again and closes afterward;
                # retain this capacity slot until that handshake actually ends.
                if record.connecting:
                    record.connect_done.wait()
            finally:
                with self._lock:
                    self._terminals.pop(record.id, None)
        threading.Thread(target=cleanup, name="vla-terminal-close", daemon=True).start()

    async def websocket(self, websocket: WebSocket, terminal_id: str):
        accepted, record, pending, claimed, tasks = False, None, None, False, []
        reason = "TERMINAL_CLOSED"
        try:
            self._check_handshake(websocket)
            principal = await asyncio.to_thread(self._principal, websocket)
            # Reject unknown/replayed IDs before allocating a browser socket.
            with self._lock:
                self._prune()
                pending = self._terminals.get(terminal_id)
                if (self._closing or pending is None or pending.consumed or pending.claimed or pending.user_id != principal.user_id
                        or pending.session_id != principal.session_id):
                    raise APIError(403, "TERMINAL_TICKET_INVALID", "终端连接凭证无效或已过期")
                pending.claimed = True
                claimed = True
            await websocket.accept()
            accepted = True
            message = await asyncio.wait_for(self._message(websocket), AUTH_TIMEOUT)
            if message.get("type") != "auth" or set(message) != {"type", "ticket", "cols", "rows"}:
                raise APIError(403, "TERMINAL_TICKET_INVALID", "终端连接凭证无效或已过期")
            cols, rows = _dimensions(message["cols"], message["rows"])
            record = self._consume(terminal_id, message["ticket"], principal)
            await asyncio.wait_for(asyncio.to_thread(self._connect, record, cols, rows), 35)
            await self._send(websocket, {"type": "ready"})
            await asyncio.to_thread(self.auth.audit, websocket, "terminal.open", 200,
                                    {"permission": "machines.manage", "node_id": record.node_id,
                                     "terminal_id": record.id, "ssh_username": record.identity[3]})
            queue = asyncio.Queue(maxsize=OUTPUT_QUEUE)
            tasks = [asyncio.create_task(self._input(websocket, record)),
                     asyncio.create_task(self._output(record, queue)),
                     asyncio.create_task(self._writer(websocket, queue)),
                     asyncio.create_task(self._watch(record))]
            remaining = set(tasks)
            while remaining:
                done, remaining = await asyncio.wait(remaining, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
                # The producer's EOF sentinel lets the writer flush every last
                # byte before it completes; other completion ends the session.
                if any(task is not tasks[1] for task in done):
                    break
        except APIError as exc:
            reason = exc.code
            if accepted:
                with suppress(Exception):
                    await self._send(websocket, {"type": "error", "code": exc.code, "message": exc.message})
        except RemoteError as exc:
            reason = exc.code
            if accepted:
                with suppress(Exception):
                    await self._send(websocket, {"type": "error", "code": exc.code, "message": exc.message})
        except (WebSocketDisconnect, asyncio.CancelledError):
            reason = "TERMINAL_DISCONNECTED"
        except (TimeoutError, socket.timeout):
            reason = "TERMINAL_TIMEOUT"
            if accepted:
                with suppress(Exception):
                    await self._send(websocket, {"type": "error", "code": reason, "message": "终端连接超时或接收过慢，请重新打开"})
        except Exception:
            reason = "TERMINAL_CONNECTION_FAILED"
            if accepted:
                with suppress(Exception):
                    await self._send(websocket, {"type": "error", "code": reason, "message": "终端连接中断，请重新打开"})
        finally:
            if record is not None:
                if record.stop.is_set():
                    reason = record.reason
                self._stop(record, reason)
            elif pending is not None and claimed:
                with self._lock:
                    self._terminals.pop(pending.id, None)
            for task in tasks:
                task.cancel()
            # ASGI servers cancel endpoint tasks on disconnect/shutdown. Shield
            # only the bounded cleanup, never the SSH connection or shell work.
            with anyio.CancelScope(shield=True):
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                if accepted:
                    with suppress(Exception):
                        await self._send(websocket, {"type": "closed", "code": reason})
                with suppress(Exception):
                    await asyncio.wait_for(websocket.close(code=1000 if accepted else 1008), SEND_TIMEOUT)
                if record is not None:
                    with suppress(Exception):
                        await asyncio.wait_for(asyncio.to_thread(self.auth.audit, websocket, "terminal.close", 200,
                                                {"code": reason, "node_id": record.node_id,
                                                 "terminal_id": record.id, "ssh_username": record.identity[3]}), SEND_TIMEOUT)

    def close(self):
        with self._lock:
            self._closing = True
            current = list(self._terminals.values())
        for record in current:
            self._stop(record, "TERMINAL_SHUTTING_DOWN")


def register_routes(app, manager):
    @app.post("/api/v1/nodes/{node_id}/terminal-sessions", status_code=201)
    def create_terminal(node_id: str, request: Request):
        return manager.create(request, node_id)

    @app.websocket("/api/v1/node-terminals/{terminal_id}/ws")
    async def terminal(websocket: WebSocket, terminal_id: str):
        await manager.websocket(websocket, terminal_id)
