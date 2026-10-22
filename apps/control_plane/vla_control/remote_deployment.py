"""Durable SSH onboarding jobs and control-plane-owned reverse tunnels.

Only the SSH transport sees decrypted passwords. No enrollment credential is
persisted in a job, exposed by a response, or accepted as an arbitrary log line.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
from pathlib import Path
import queue
import stat
import threading
import time
from urllib.parse import urlsplit, urlunsplit

from cryptography.fernet import Fernet, InvalidToken
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import select, text
from sqlalchemy.engine import make_url

from .common import APIError, TERMINAL, digest, now, offset_time, uid
from .models import Allocation, Command, Node, Profile, Run
from .onboarding import _path, generate_agent_config, resolve_agent_config_defaults
from .remote_models import RemoteConnection, RemoteDeployment
from .remote_schemas import RemoteDeploymentInput, SSHAuthenticationInput, SSHConnectionInput, SSHInstallCheckInput, SSHPathSuggestInput


JOB_TERMINAL = {"SUCCEEDED", "FAILED", "INTERRUPTED", "DISCONNECTED", "UNINSTALLED"}
RETRYABLE = {"FAILED", "INTERRUPTED"}
DISABLED_TUNNELS = {"SUPERSEDED", "DISCONNECTED", "UNINSTALLED"}
logger = logging.getLogger(__name__)


def _state_directory(settings) -> Path:
    override = os.getenv("VLA_REMOTE_STATE_DIR")
    if override:
        return Path(override).expanduser().absolute()
    parsed = make_url(settings.database_url)
    if parsed.get_backend_name() == "sqlite" and parsed.database not in {None, "", ":memory:"}:
        return Path(parsed.database).expanduser().absolute().parent / "remote-management"
    return Path(".state/remote-management").absolute()


class _CredentialVault:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if directory.is_symlink():
            raise ValueError("Remote management state directory cannot be a symlink")
        os.chmod(directory, 0o700)
        path = directory / "credentials.key"
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(Fernet.generate_key())
                stream.flush()
                os.fsync(stream.fileno())
        if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
            raise ValueError("Remote management credential key must be a regular private file")
        os.chmod(path, 0o600)
        self._fernet = Fernet(path.read_bytes())

    def encrypt(self, password: str) -> str:
        return self._fernet.encrypt(password.encode()).decode()

    def decrypt(self, value: str) -> str:
        try:
            return self._fernet.decrypt(value.encode()).decode()
        except (InvalidToken, UnicodeError, ValueError) as exc:
            raise APIError(503, "SSH_CREDENTIAL_UNAVAILABLE", "远程凭据无法解密；请重新测试 SSH 连接") from exc


def _loopback_url(url: str) -> bool:
    hostname = urlsplit(url).hostname or ""
    if hostname.lower() in {"localhost", "localhost."}:
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


class DeploymentManager:
    """One manager per control-plane process; database jobs survive restarts.

    This initial deployment model expects one API process. Worker ownership is
    process-local, while target conflict checks are transactionally persisted.
    """

    def __init__(self, db, settings, *, remote_factory=None, tunnel_factory=None):
        self.db = db
        self.settings = settings
        self.state_dir = _state_directory(settings)
        self.vault = _CredentialVault(self.state_dir)
        self._remote_factory = remote_factory
        self._tunnel_factory = tunnel_factory
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._workers: dict[str, threading.Thread] = {}
        self._tunnels: dict[str, object] = {}
        self._tunnel_retries: dict[str, tuple[float, int]] = {}
        self._monitor: threading.Thread | None = None
        self._started = False
        self._wait_seconds = float(os.getenv("VLA_REMOTE_AGENT_WAIT_SECONDS", "180"))
        self._path_queries = threading.BoundedSemaphore(4)

    def _remote(self, connection: RemoteConnection):
        factory = self._remote_factory
        if factory is None:
            from .ssh_remote import SSHRemote
            factory = SSHRemote
        return factory(host=connection.host, port=connection.port, username=connection.username,
                       password=self.vault.decrypt(connection.encrypted_password), state_dir=self.state_dir,
                       expected_fingerprint=connection.fingerprint or None)

    def _new_tunnel(self, connection: RemoteConnection, remote_port: int):
        factory = self._tunnel_factory
        if factory is None:
            from .ssh_remote import ManagedTunnel
            factory = ManagedTunnel
        return factory(host=connection.host, port=connection.port, username=connection.username,
                       password=self.vault.decrypt(connection.encrypted_password), state_dir=self.state_dir,
                       remote_port=remote_port, target_host=os.getenv("VLA_TUNNEL_TARGET_HOST", "127.0.0.1"),
                       target_port=int(os.getenv("VLA_TUNNEL_TARGET_PORT", "8000")),
                       expected_fingerprint=connection.fingerprint)

    def start(self):
        with self._lock:
            if self._started:
                return
            self._started = True
        with self.db.write() as session:
            for job in session.scalars(select(RemoteDeployment).where(RemoteDeployment.state.not_in(JOB_TERMINAL))):
                self._change(job, "INTERRUPTED", "INTERRUPTED", "中台已重启，未完成的部署已中断；请检查后重试")
                job.error = {"code": "DEPLOYMENT_INTERRUPTED", "message": "中台已重启，部署未完成；可使用重试重新检查并安装"}
                if job.connection_mode == "ssh-tunnel":
                    job.tunnel_state = "FAILED"
        self._monitor = threading.Thread(target=self._monitor_loop, name="vla-remote-monitor", daemon=True)
        self._monitor.start()

    def close(self):
        self._stop.set()
        with self._lock:
            tunnels = list(self._tunnels.values())
            self._tunnels.clear()
            workers = list(self._workers.values())
        for tunnel in tunnels:
            try:
                tunnel.close()
            except Exception:
                pass
        if self._monitor:
            self._monitor.join(timeout=5)
        for worker in workers:
            worker.join(timeout=2)

    def _safe(self, text: str, connection: RemoteConnection | None = None) -> str:
        from .ssh_remote import sanitize_install_output
        secrets = [self.settings.enrollment_token, self.settings.admin_token]
        if connection is not None:
            try:
                secrets.append(self.vault.decrypt(connection.encrypted_password))
            except APIError:
                pass
        return sanitize_install_output(str(text), secrets)

    def _log_path(self, job_id: str) -> Path:
        # IDs originate from persisted jobs, never from a caller-supplied path.
        if not job_id or Path(job_id).name != job_id or job_id in {".", ".."}:
            raise ValueError("Invalid deployment identity")
        return self.state_dir / "deployment-logs" / (job_id + ".jsonl")

    def _save_log(self, job, entry: dict) -> None:
        """Keep full sanitized diagnostics independently of the bounded UI tail."""
        try:
            path = self._log_path(job.id)
            path.parent.mkdir(mode=0o700, exist_ok=True)
            if path.parent.is_symlink():
                raise OSError("Deployment log directory cannot be a symlink")
            path.parent.chmod(0o700)
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags, 0o600)
            with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
                os.fchmod(stream.fileno(), 0o600)
                # Seed earlier records when an old deployment first uses file logs.
                if os.fstat(stream.fileno()).st_size == 0:
                    for previous in job.logs:
                        safe = {**previous, "message": self._safe(previous.get("message", ""))}
                        stream.write(json.dumps(safe, ensure_ascii=False) + "\n")
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            # A diagnostic disk failure must not abort or falsely fail installation.
            logger.warning("Unable to persist deployment diagnostics for %s", job.id)

    def _append_log(self, job, stage: str, message: str, timestamp: str | None = None):
        entry = {"timestamp": timestamp or now(), "stage": stage, "message": self._safe(message)}
        self._save_log(job, entry)
        job.logs = [*job.logs, entry][-300:]

    def _change(self, job, state: str, stage: str, message: str | None = None):
        timestamp = now()
        if not job.history or job.state != state or job.stage != stage:
            job.history = [*job.history, {"state": state, "stage": stage, "timestamp": timestamp}][-200:]
        job.state, job.stage, job.updated_at = state, stage, timestamp
        if state in JOB_TERMINAL:
            job.finished_at = timestamp
        if message:
            self._append_log(job, stage, message, timestamp)

    def _log(self, job_id: str, stage: str, message: str):
        if self._stop.is_set():
            return
        with self.db.write() as session:
            job = session.get(RemoteDeployment, job_id)
            if job is None or job.state in JOB_TERMINAL:
                return
            connection = session.get(RemoteConnection, job.connection_id)
            self._change(job, "RUNNING", self._safe(stage, connection).upper()[:64], self._safe(message, connection))

    def _probe_view(self, probe: dict, connection: RemoteConnection | None = None):
        checks = [{"name": self._safe(check.get("name", "检查"), connection), "ok": check.get("ok") is True,
                   "message": self._safe(check.get("message", ""), connection)}
                  for check in probe.get("checks", []) if isinstance(check, dict)][:100]
        result = {"hostname": self._safe(probe.get("hostname", ""), connection), "checks": checks,
                  "can_install": probe.get("can_install") is True}
        existing = probe.get("existing_agent")
        if isinstance(existing, dict):
            # The transport reads only the Agent's public configuration fields.
            clean = {key: existing[key] for key in ("node_id", "name", "roots", "output_root", "state_dir", "active_attempts") if key in existing}
            clean["profiles"] = []
            for profile in existing.get("profiles", []):
                if not isinstance(profile, dict):
                    continue
                item = {key: profile[key] for key in ("id", "name", "code_dir", "python", "adapter", "recipe", "execution_mode") if key in profile}
                clean["profiles"].append(item)
            result["existing_agent"] = clean
        return result

    def test_connection(self, body: SSHConnectionInput):
        install_dir = self._install_path(body.install_dir)
        connection = RemoteConnection(id=uid(), host=body.host, port=body.port, username=body.username,
            encrypted_password=self.vault.encrypt(body.password.get_secret_value()), fingerprint="",
            probe_result={}, install_dir=install_dir, created_at=now(), expires_at=offset_time(86400))
        return self._test_connection(connection, install_dir)

    def test_existing_connection(self, connection_id: str, body: SSHInstallCheckInput):
        install_dir = self._install_path(body.install_dir)
        return self._test_connection(self._stored_connection(connection_id), install_dir)

    def _test_connection(self, connection: RemoteConnection, install_dir: str):
        remote = self._remote(connection)
        result: queue.Queue = queue.Queue(maxsize=1)

        def probe_remote():
            try:
                remote.connect()
                result.put((remote.fingerprint, remote.probe(install_dir), None))
            except Exception as exc:
                result.put((None, None, exc))
            finally:
                remote.close()

        worker = threading.Thread(target=probe_remote, name="vla-ssh-probe", daemon=True)
        worker.start()
        try:
            fingerprint, probe, error = result.get(timeout=18)
        except queue.Empty:
            remote.close()
            raise APIError(504, "SSH_TEST_TIMEOUT", "SSH 连接检查超过 18 秒，请检查网络或 SSH 服务后重试")
        if error is not None:
            failure = self._transport_error(error, 422)
            raise APIError(failure.status, failure.code, self._safe(failure.message, connection))
        connection.fingerprint = fingerprint
        connection.probe_result = self._probe_view(probe, connection)
        connection.install_dir = install_dir
        with self.db.write() as session:
            session.add(connection)
        return {"connection_id": connection.id, "host": connection.host, "port": connection.port,
                "username": connection.username, "fingerprint": connection.fingerprint,
                "expires_at": connection.expires_at, **connection.probe_result}

    def connect_connection(self, body: SSHAuthenticationInput):
        """Authenticate browsing before an installation directory is selected."""
        connection = RemoteConnection(id=uid(), host=body.host, port=body.port, username=body.username,
            encrypted_password=self.vault.encrypt(body.password.get_secret_value()), fingerprint="",
            probe_result={}, install_dir="", created_at=now(), expires_at=offset_time(86400))
        remote = self._remote(connection)
        result: queue.Queue = queue.Queue(maxsize=1)

        def connect_remote():
            try:
                remote.connect()
                result.put((remote.fingerprint, None))
            except Exception as exc:
                result.put((None, exc))
            finally:
                remote.close()

        threading.Thread(target=connect_remote, name="vla-ssh-connect", daemon=True).start()
        try:
            fingerprint, error = result.get(timeout=18)
        except queue.Empty:
            remote.close()
            raise APIError(504, "SSH_TEST_TIMEOUT", "SSH 连接超过 18 秒，请检查网络或 SSH 服务后重试")
        if error is not None:
            failure = self._transport_error(error, 422)
            raise APIError(failure.status, failure.code, self._safe(failure.message, connection))
        connection.fingerprint = fingerprint
        with self.db.write() as session:
            session.add(connection)
        return {"connection_id": connection.id, "host": connection.host, "port": connection.port,
                "username": connection.username, "fingerprint": fingerprint, "expires_at": connection.expires_at}

    def _stored_connection(self, connection_id: str) -> RemoteConnection:
        with self.db.read() as session:
            connection = session.get(RemoteConnection, connection_id)
            if connection is None:
                raise APIError(404, "SSH_CONNECTION_NOT_FOUND", "SSH 连接不存在，请重新连接")
            if not connection.encrypted_password:
                raise APIError(409, "SSH_CONNECTION_REVOKED", "此 SSH 凭据已被移除，请重新输入密码并连接")
            if connection.expires_at <= now():
                raise APIError(409, "SSH_CONNECTION_EXPIRED", "SSH 连接已过期，请重新输入密码并连接")
            if not connection.fingerprint:
                raise APIError(409, "SSH_CONNECTION_UNVERIFIED", "SSH 机器身份尚未验证，请重新连接")
        return connection

    def path_suggestions(self, connection_id: str, body: SSHPathSuggestInput):
        from .path_suggestions import validate_prefix

        # SSH permissions provide the pre-install scope. No Agent or managed
        # roots exist yet, and browsing never executes a caller-provided path.
        validate_prefix(body.prefix, ["/"])
        connection = self._stored_connection(connection_id)
        if not self._path_queries.acquire(blocking=False):
            raise APIError(429, "SSH_PATH_BUSY", "路径联想请求较多，请稍后重试或手动填写")
        try:
            remote = self._remote(connection)
        except Exception:
            self._path_queries.release()
            raise
        result: queue.Queue = queue.Queue(maxsize=1)

        def browse_remote():
            try:
                remote.connect()
                result.put((remote.path_suggestions(body.prefix, limit=body.limit,
                                                   directory_only=body.directory_only), None))
            except Exception as exc:
                result.put((None, exc))
            finally:
                try:
                    remote.close()
                finally:
                    self._path_queries.release()

        threading.Thread(target=browse_remote, name="vla-ssh-paths", daemon=True).start()
        try:
            paths, error = result.get(timeout=12)
        except queue.Empty:
            remote.close()
            raise APIError(504, "SSH_PATH_TIMEOUT", "路径联想超时，可继续手动填写或稍后重试")
        if error is not None:
            failure = self._transport_error(error)
            raise APIError(failure.status, failure.code, self._safe(failure.message, connection))
        return {"connection_id": connection.id, "status": "READY", **paths}

    @staticmethod
    def _install_path(path: str):
        value = _path(path, "install_dir")
        if value in {"/", "/etc", "/usr", "/var", "/home", "/root", "/tmp", "/data", "/data2"}:
            raise APIError(422, "INVALID_INSTALL_DIR", "Agent 安装目录必须是独立子目录，不能使用系统或数据盘根目录")
        return value

    @staticmethod
    def _transport_error(exc: Exception, status: int = 502):
        if isinstance(exc, APIError):
            return exc
        from .ssh_remote import RemoteError
        if isinstance(exc, RemoteError):
            return APIError(status, exc.code, exc.message)
        # Never serialize str(exc): SSH libraries and subprocess failures can
        # embed credentials, remote commands, or unrestricted remote output.
        return APIError(status, "SSH_OPERATION_FAILED", "远程操作失败；请检查 SSH 连接、目录权限和服务状态后重试")

    @staticmethod
    def _assert_idle(session, node_id: str | None):
        if not node_id:
            return
        session.scalar(select(Node).where(Node.id == node_id).with_for_update().execution_options(populate_existing=True))
        run = session.scalar(select(Run.id).where(Run.node_id == node_id, Run.state.not_in(TERMINAL)).limit(1))
        command = session.scalar(select(Command.id).where(Command.node_id == node_id, Command.status == "PENDING").limit(1))
        allocation = session.scalar(select(Allocation.node_id).where(Allocation.node_id == node_id).limit(1))
        if run or command or allocation:
            raise APIError(409, "NODE_BUSY", "该机器仍有训练、资源预约或待执行命令，暂不能管理 Agent")

    @staticmethod
    def _assert_no_other_job(session, node_id: str | None, exclude_id: str | None = None):
        if not node_id:
            return
        query = select(RemoteDeployment).where(RemoteDeployment.node_id == node_id, RemoteDeployment.state.not_in(JOB_TERMINAL))
        if exclude_id:
            query = query.where(RemoteDeployment.id != exclude_id)
        other = session.scalar(query.limit(1))
        if other:
            raise APIError(409, "REMOTE_NODE_MANAGED", "该机器已有正在执行的管理操作，请等待完成后再修改配置",
                           {"deployment_id": other.id})

    def create(self, body: RemoteDeploymentInput, key: str, request_url: str, *, retry=False):
        if not key or not key.strip() or len(key) > 200:
            raise APIError(422, "IDEMPOTENCY_KEY_REQUIRED", "Provide an Idempotency-Key of 1 to 200 characters")
        prefix = self._install_path(body.install_dir)
        request_digest = digest(body.model_dump())
        with self.db.read() as session:
            existing = session.scalar(select(RemoteDeployment).where(RemoteDeployment.idempotency_key == key))
            if existing:
                if existing.request_digest != request_digest:
                    raise APIError(409, "IDEMPOTENCY_CONFLICT", "该幂等键已用于不同的部署配置")
                return self._view(session, existing)
        # Validate config and generate onboarding provenance before queuing.
        default = None
        if body.agent.control_plane_url is None:
            default = resolve_agent_config_defaults(os.getenv("VLA_AGENT_CONTROL_PLANE_URL"), request_url)["control_plane_url"]
            if default is None:
                # Unlike downloaded manual configs, this flow owns and verifies
                # an SSH tunnel, so a private HTTP origin needs no guessed public
                # endpoint. The temporary port is replaced before installation.
                default = "http://127.0.0.1:" + os.getenv("VLA_TUNNEL_TARGET_PORT", "8000")
        generated = generate_agent_config(body.agent, self.settings.enrollment_token, default)
        config = {key: value for key, value in generated["config"].items() if key != "enrollment_token"}
        mode = "ssh-tunnel" if _loopback_url(config["control_plane_url"]) else "direct"
        with self.db.write() as session:
            existing = session.scalar(select(RemoteDeployment).where(RemoteDeployment.idempotency_key == key))
            if existing:
                if existing.request_digest != request_digest:
                    raise APIError(409, "IDEMPOTENCY_CONFLICT", "该幂等键已用于不同的部署配置")
                return self._view(session, existing)
            connection = session.get(RemoteConnection, body.connection_id)
            if not connection:
                raise APIError(404, "SSH_CONNECTION_NOT_FOUND", "SSH 连接不存在，请重新测试连接")
            if not connection.encrypted_password:
                raise APIError(409, "SSH_CONNECTION_REVOKED", "此 SSH 凭据已被移除，请重新输入密码并测试连接")
            if not retry and connection.expires_at < now():
                raise APIError(409, "SSH_CONNECTION_EXPIRED", "SSH 连接检查已过期，请重新输入密码并测试连接")
            if connection.install_dir != prefix:
                raise APIError(409, "SSH_TARGET_CHANGED", "安装目录已变更，请重新测试 SSH 连接")
            if not connection.probe_result.get("can_install"):
                raise APIError(409, "SSH_PREREQUISITES_FAILED", "目标机器未通过安装条件检查，请修复后重新测试连接")
            node_id = (connection.probe_result.get("existing_agent") or {}).get("node_id")
            if node_id:
                existing_node = session.get(Node, node_id)
                if existing_node and existing_node.status == "REMOVED":
                    raise APIError(409, "REMOVED_AGENT_IDENTITY", "机器仍保留已卸载 Agent 的旧身份，请检查安装状态并使用新的 Agent 状态目录")
            self._assert_idle(session, node_id)
            target = digest({"host": connection.host, "port": connection.port, "install_dir": prefix})
            if self.db.engine.dialect.name == "postgresql":
                # New machines have no Node row to lock yet. Serialize their
                # target claims as well as updates through a transaction lock.
                session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": int(target[:16], 16) - (1 << 63)})
            active = session.scalar(select(RemoteDeployment).where(RemoteDeployment.target_key == target,
                RemoteDeployment.state.not_in(JOB_TERMINAL)).limit(1))
            if active:
                raise APIError(409, "REMOTE_TARGET_MANAGED", "该目录已有部署或正在接入，请查看现有部署记录",
                               {"deployment_id": active.id})
            prior_node = session.scalar(select(RemoteDeployment.node_id).join(Node, Node.id == RemoteDeployment.node_id).where(
                RemoteDeployment.target_key == target, RemoteDeployment.node_id.is_not(None), Node.status != "REMOVED",
                RemoteDeployment.state != "UNINSTALLED").order_by(RemoteDeployment.created_at.desc()).limit(1))
            node_id = node_id or prior_node
            self._assert_idle(session, node_id)
            # PostgreSQL may have waited on the Node lock while a removal
            # committed. Refresh ORM identities loaded before that wait.
            session.refresh(connection)
            if not connection.encrypted_password:
                raise APIError(409, "SSH_CONNECTION_REVOKED", "此 SSH 凭据已被移除，请重新输入密码并测试连接")
            if node_id:
                # Host aliases can resolve to one registered physical machine.
                # The Node lock above serializes these claims even when their
                # host-derived target keys differ.
                self._assert_no_other_job(session, node_id)
                reconnect_node = session.get(Node, node_id, populate_existing=True)
                if reconnect_node and reconnect_node.status == "REMOVED":
                    raise APIError(409, "REMOVED_AGENT_IDENTITY", "机器仍保留已卸载 Agent 的旧身份，请重新测试 SSH 连接")
                if reconnect_node and reconnect_node.status == "DISCONNECTED":
                    # The retained Agent identity can authenticate again only
                    # after a new administrator-authorized SSH onboarding job.
                    reconnect_node.status = "REGISTERED"
            timestamp = now()
            job = RemoteDeployment(id=uid(), connection_id=connection.id, idempotency_key=key,
                request_digest=request_digest, target_key=target, onboarding_id=generated["id"], node_id=node_id,
                state="QUEUED", stage="QUEUED", logs=[], history=[], agent_config=config, install_dir=prefix,
                control_plane_url=config["control_plane_url"], connection_mode=mode,
                tunnel_state="STARTING" if mode == "ssh-tunnel" else "NOT_REQUIRED", tunnel_remote_port=None,
                created_at=timestamp, updated_at=timestamp, started_at=None, finished_at=None, error=None)
            self._change(job, "QUEUED", "QUEUED", "部署已排队，等待连接目标机器")
            session.add(job)
            session.flush()
            view = self._view(session, job)
        self._launch(job.id)
        return view

    def _launch(self, job_id: str):
        if self._stop.is_set():
            return
        worker = threading.Thread(target=self._execute, args=(job_id,), name=f"vla-deploy-{job_id[:8]}", daemon=True)
        with self._lock:
            self._workers[job_id] = worker
        worker.start()

    @staticmethod
    def _operation(job) -> str:
        return job.agent_config.get("_operation", "install")

    @staticmethod
    def _required_node(session, node_id):
        node = session.get(Node, node_id, populate_existing=True)
        if not node:
            raise APIError(404, "NODE_NOT_FOUND", "机器不存在")
        return node

    def _node_management(self, session, node_id):
        node = self._required_node(session, node_id)
        jobs = list(session.scalars(select(RemoteDeployment).where(RemoteDeployment.node_id == node_id).order_by(RemoteDeployment.created_at.desc())))
        installations = [job for job in jobs if self._operation(job) == "install"]
        latest = installations[0] if installations else None
        connection = session.get(RemoteConnection, latest.connection_id) if latest else None
        busy = None
        if node.status in {"DISCONNECTED", "REMOVED"}:
            busy = "机器已断开或卸载；如需再次使用，请重新通过 SSH 接入"
        elif any(job.state not in JOB_TERMINAL for job in jobs):
            busy = "机器仍有未完成的管理操作，请等待完成后再操作"
        else:
            try:
                self._assert_idle(session, node_id)
            except APIError as exc:
                busy = exc.message
        uninstall_reason = busy
        if not uninstall_reason and (not latest or not connection or not connection.encrypted_password):
            uninstall_reason = "没有可用的 SSH 管理凭据；请先通过 SSH 重新接入机器"
        return {"node_id": node.id, "node_status": node.status, "managed": latest is not None,
                "host": connection.host if connection else None, "port": connection.port if connection else None,
                "username": connection.username if connection else None, "install_dir": latest.install_dir if latest else None,
                "deployment_id": latest.id if latest else None, "can_disconnect": busy is None,
                "can_uninstall": uninstall_reason is None, "reason": busy or uninstall_reason,
                "disconnect_reason": busy, "uninstall_reason": uninstall_reason}, latest, connection

    def management(self, node_id):
        with self.db.read() as session:
            return self._node_management(session, node_id)[0]

    def _retire_node(self, session, node_id: str, state: str):
        node = self._required_node(session, node_id)
        node.status = "REMOVED" if state == "UNINSTALLED" else "DISCONNECTED"
        for profile in session.scalars(select(Profile).where(Profile.node_id == node_id)):
            profile.active = False
        connections = set()
        for job in session.scalars(select(RemoteDeployment).where(RemoteDeployment.node_id == node_id)):
            job.tunnel_state = state
            job.updated_at = now()
            connections.add(job.connection_id)
        targets = {(connection.host, connection.port) for connection in session.scalars(select(RemoteConnection).where(
            RemoteConnection.id.in_(connections)))}
        # Tests performed through a second hostname may not yet have created a
        # deployment. Revoke those credentials for the same known identity too.
        for connection in session.scalars(select(RemoteConnection)):
            if (connection.id in connections or (connection.host, connection.port) in targets or
                    (connection.probe_result.get("existing_agent") or {}).get("node_id") == node_id):
                connection.encrypted_password = ""
                connection.expires_at = now()

    def _close_node_tunnels(self, node_id: str):
        with self.db.read() as session:
            job_ids = list(session.scalars(select(RemoteDeployment.id).where(RemoteDeployment.node_id == node_id)))
        with self._lock:
            tunnels = [self._tunnels.pop(job_id) for job_id in job_ids if job_id in self._tunnels]
            for job_id in job_ids:
                self._tunnel_retries.pop(job_id, None)
        for tunnel in tunnels:
            tunnel.close()

    def _publish_tunnel(self, job_id, tunnel):
        # Serialize publication with removal's final close. A tunnel reconnect
        # already in flight when a node is removed must never resurrect it.
        with self._lock:
            with self.db.read() as session:
                job = session.get(RemoteDeployment, job_id)
                node = session.get(Node, job.node_id) if job and job.node_id else None
                allowed = (job is not None and job.tunnel_state not in DISABLED_TUNNELS
                           and self._operation(job) == "install"
                           and (node is None or node.status not in {"DISCONNECTED", "REMOVED"})
                           and not self._stop.is_set())
            if allowed:
                self._tunnels[job_id] = tunnel
                return True
        tunnel.close()
        return False

    def remove(self, node_id: str, operation: str, key: str | None):
        if not key or not key.strip() or len(key) > 200:
            raise APIError(422, "IDEMPOTENCY_KEY_REQUIRED", "Provide an Idempotency-Key of 1 to 200 characters")
        request_digest = digest({"node_id": node_id, "operation": operation})
        with self.db.write() as session:
            self._required_node(session, node_id)
            session.scalar(select(Node).where(Node.id == node_id).with_for_update().execution_options(populate_existing=True))
            existing = session.scalar(select(RemoteDeployment).where(RemoteDeployment.idempotency_key == key))
            if existing:
                if existing.request_digest != request_digest:
                    raise APIError(409, "IDEMPOTENCY_CONFLICT", "该幂等键已用于不同的管理操作")
                return self._view(session, existing)
            self._assert_idle(session, node_id)
            info, source, connection = self._node_management(session, node_id)
            reason = info["disconnect_reason"] if operation == "disconnect" else info["uninstall_reason"]
            if reason:
                raise APIError(409, "NODE_MANAGEMENT_UNAVAILABLE", reason)
            if connection is None:
                # Unmanaged Agents can still be disconnected without inventing
                # SSH access. This inert row only anchors the audit job's FK.
                connection = RemoteConnection(id=uid(), host=self._required_node(session, node_id).hostname, port=22,
                    username="", encrypted_password="", fingerprint="", probe_result={"unmanaged": True}, install_dir="",
                    created_at=now(), expires_at=now())
                session.add(connection)
                session.flush()
            timestamp = now()
            config = {**(source.agent_config if source else {}), "_operation": operation}
            job = RemoteDeployment(id=uid(), connection_id=connection.id, idempotency_key=key, request_digest=request_digest,
                target_key=source.target_key if source else digest({"node_id": node_id}), onboarding_id=uid(), node_id=node_id,
                state="QUEUED", stage="QUEUED", logs=[], history=[], agent_config=config,
                install_dir=source.install_dir if source else "", control_plane_url=source.control_plane_url if source else "",
                connection_mode=source.connection_mode if source else "direct", tunnel_state="NOT_REQUIRED", tunnel_remote_port=None,
                created_at=timestamp, updated_at=timestamp, started_at=None, finished_at=None, error=None)
            session.add(job)
            session.flush()
            if operation == "disconnect":
                self._retire_node(session, node_id, "DISCONNECTED")
                self._change(job, "DISCONNECTED", "DISCONNECTED", "已断开中台连接并移除 SSH 管理凭据；远端 Agent 服务仍保留")
            else:
                self._change(job, "QUEUED", "QUEUED", "卸载已排队，将通过 SSH 再次检查机器状态后执行")
            view = self._view(session, job)
        if operation == "disconnect":
            self._close_node_tunnels(node_id)
        else:
            self._launch(job.id)
        return view

    def _execute_uninstall(self, job_id):
        remote = None
        try:
            with self.db.write() as session:
                job = session.get(RemoteDeployment, job_id)
                if not job or job.state != "QUEUED":
                    return
                self._assert_idle(session, job.node_id)
                connection = session.get(RemoteConnection, job.connection_id)
                job.started_at = now()
                self._change(job, "RUNNING", "UNINSTALLING", "正在连接目标机器并检查 Agent 是否可以安全卸载")
                prefix, node_id = job.install_dir, job.node_id
            remote = self._remote(connection)
            remote.connect()
            if self._stop.is_set():
                return
            probe = remote.probe(prefix)
            remote_node = (probe.get("existing_agent") or {}).get("node_id")
            if remote_node and remote_node != node_id:
                raise APIError(409, "REMOTE_NODE_CHANGED", "远端 Agent 身份与待卸载机器不一致，已停止操作；请重新检查 SSH 接入")
            result = remote.uninstall(prefix, lambda stage, message: self._log(job_id, stage, message))
            if self._stop.is_set():
                return
            if not isinstance(result, dict) or result.get("service_removed") is not True:
                raise APIError(502, "AGENT_UNINSTALL_NOT_CONFIRMED", "远端未确认 Agent 服务已卸载，中台连接和凭据仍保留")
            safe_result = {"backup_dir": self._safe(result.get("backup_dir", ""), connection),
                "removed_paths": [self._safe(path, connection) for path in result.get("removed_paths", [])][:100],
                "service_removed": True}
            with self.db.write() as session:
                job = session.get(RemoteDeployment, job_id)
                self._assert_idle(session, job.node_id)
                job.agent_config = {**job.agent_config, "_operation_result": safe_result}
                self._retire_node(session, node_id, "UNINSTALLED")
                self._change(job, "UNINSTALLED", "UNINSTALLED", "Agent 已从远端卸载，中台连接与 SSH 凭据已移除；训练数据和历史记录保留")
                job.error = None
            self._close_node_tunnels(node_id)
        except Exception as exc:
            if not self._stop.is_set():
                error = self._transport_error(exc)
                with self.db.write() as session:
                    job = session.get(RemoteDeployment, job_id)
                    connection = session.get(RemoteConnection, job.connection_id)
                    message = self._safe(error.message, connection)
                    self._change(job, "FAILED", "FAILED", message)
                    job.error = {"code": error.code, "message": message}
        finally:
            if remote is not None:
                remote.close()
            with self._lock:
                self._workers.pop(job_id, None)

    def _execute(self, job_id: str):
        with self.db.read() as session:
            job = session.get(RemoteDeployment, job_id)
            uninstall = job is not None and self._operation(job) == "uninstall"
        if uninstall:
            self._execute_uninstall(job_id)
            return
        remote = None
        try:
            with self.db.write() as session:
                job = session.get(RemoteDeployment, job_id)
                if job is None or job.state != "QUEUED":
                    return
                connection = session.get(RemoteConnection, job.connection_id)
                self._assert_idle(session, job.node_id)
                self._change(job, "RUNNING", "CONNECTING", "正在通过已验证的 SSH 主机密钥连接目标机器")
                job.started_at = now()
                config = dict(job.agent_config)
                prefix, mode = job.install_dir, job.connection_mode
            remote = self._remote(connection)
            remote.connect()
            if self._stop.is_set():
                return
            self._log(job_id, "CHECKING", "正在重新检查系统、目录和现有 Agent 状态")
            probe = remote.probe(prefix)
            if not probe.get("can_install"):
                raise APIError(409, "SSH_PREREQUISITES_FAILED", "目标机器未通过安装条件检查，请修复目录权限或系统依赖后重试")
            existing = probe.get("existing_agent") or {}
            with self.db.write() as session:
                job = session.get(RemoteDeployment, job_id)
                actual_node_id = existing.get("node_id") or job.node_id
                self._assert_idle(session, actual_node_id)
                self._assert_no_other_job(session, actual_node_id, job.id)
                actual_node = session.get(Node, actual_node_id) if actual_node_id else None
                if actual_node and actual_node.status in {"DISCONNECTED", "REMOVED"}:
                    raise APIError(409, "NODE_CONNECTION_REMOVED", "此机器的连接已被移除，请重新通过 SSH 测试并接入")
                job.node_id = actual_node_id
            if mode == "ssh-tunnel":
                self._log(job_id, "TUNNEL", "正在为此机器建立独立的 SSH 反向通道")
                tunnel = self._new_tunnel(connection, 0)
                tunnel.start()
                if self._stop.is_set():
                    tunnel.close()
                    return
                port = tunnel.actual_port
                if not self._publish_tunnel(job_id, tunnel):
                    if self._stop.is_set():
                        return
                    raise APIError(409, "NODE_CONNECTION_REMOVED", "此机器的连接已被移除，SSH 通道不会重新启用")
                parsed = urlsplit(config["control_plane_url"])
                config["control_plane_url"] = urlunsplit((parsed.scheme, f"127.0.0.1:{port}", parsed.path, "", ""))
                with self.db.write() as session:
                    job = session.get(RemoteDeployment, job_id)
                    job.tunnel_remote_port, job.tunnel_state = port, "HEALTHY"
                    job.control_plane_url = config["control_plane_url"]
                    job.agent_config = config
            config["enrollment_token"] = self.settings.enrollment_token
            bundle = Path(os.getenv("VLA_AGENT_BUNDLE", ".local/agent-dist/vla-agent-bundle.tar.gz"))
            if not bundle.is_file():
                raise APIError(503, "AGENT_PACKAGE_UNAVAILABLE", "中台尚未准备 Agent 安装包，请管理员先构建安装包")
            remote.install(bundle, config, prefix, lambda stage, message: self._log(job_id, stage, message))
            config.pop("enrollment_token", None)
            remote.close()
            remote = None
            if self._stop.is_set():
                return
            with self.db.write() as session:
                job = session.get(RemoteDeployment, job_id)
                self._change(job, "WAITING_AGENT", "WAITING_AGENT", "安装命令已完成，等待 Agent 新心跳和训练 Python 环境检查")
            deadline = time.monotonic() + self._wait_seconds
            while not self._stop.wait(0.5):
                if self._confirm_agent(job_id):
                    self._supersede_older(job_id)
                    return
                if time.monotonic() >= deadline:
                    raise APIError(504, "AGENT_ONLINE_TIMEOUT", "Agent 已安装，但未在等待时间内通过上线和训练环境检查；请检查服务与连接地址后重试")
        except Exception as exc:
            if not self._stop.is_set():
                error = self._transport_error(exc)
                with self.db.write() as session:
                    job = session.get(RemoteDeployment, job_id)
                    connection = session.get(RemoteConnection, job.connection_id)
                    message = self._safe(error.message, connection)
                    self._change(job, "FAILED", "FAILED", message)
                    job.error = {"code": error.code, "message": message}
                # Preserve a working tunnel when installation succeeded but a
                # training environment check failed: the Agent remains usable.
        finally:
            if remote is not None:
                remote.close()
            with self._lock:
                self._workers.pop(job_id, None)

    def _confirm_agent(self, job_id: str) -> bool:
        with self.db.write() as session:
            job = session.get(RemoteDeployment, job_id)
            profile_rows = list(session.scalars(select(Profile).where(Profile.active.is_(True))))
            by_node = {}
            for profile in profile_rows:
                if profile.metadata_json.get("onboarding_id") == job.onboarding_id:
                    by_node.setdefault(profile.node_id, {})[profile.id] = profile
            expected = job.agent_config["profiles"]
            for node_id, profiles in by_node.items():
                node = session.get(Node, node_id)
                if (not node or node.status != "ONLINE" or node.last_seen <= job.started_at
                        or (job.node_id and job.node_id != node.id)):
                    continue
                if any(p["id"] not in profiles or any(getattr(profiles[p["id"]], field) != p.get(field)
                       for field in ("code_dir", "python", "adapter", "recipe", "execution_mode")) for p in expected):
                    continue
                job.node_id = node.id
                # A new authenticated heartbeat carrying the exact generated
                # onboarding ID and interpreter/code fields proves the updated
                # local config was loaded. Existing agents retain identity and
                # do not re-register their public node fields on upgrades.
                node.name = job.agent_config["name"]
                node.roots = job.agent_config["roots"]
                node.output_root = job.agent_config["output_root"]
                checks = {check.get("profile_id"): check for check in node.metrics.get("runtime_checks", [])}
                if any(checks.get(p["id"], {}).get("status") == "FAILED" for p in expected):
                    message = "Agent 已上线，但训练 Python 环境检查失败；请在机器详情查看检查结果，修正路径后重新接入"
                    self._change(job, "FAILED", "FAILED", message)
                    job.error = {"code": "RUNTIME_CHECK_FAILED", "message": message}
                    return True
                if not all(checks.get(p["id"], {}).get("status") == "READY" for p in expected):
                    continue
                self._change(job, "SUCCEEDED", "ONLINE", "Agent 已上线，训练 Python 环境检查通过")
                job.error = None
                return True
        return False

    def _supersede_older(self, job_id: str):
        retired = []
        with self.db.write() as session:
            current = session.get(RemoteDeployment, job_id)
            if not current or current.state != "SUCCEEDED":
                return
            for old in session.scalars(select(RemoteDeployment).where(RemoteDeployment.target_key == current.target_key,
                    RemoteDeployment.id != current.id, RemoteDeployment.created_at < current.created_at,
                    RemoteDeployment.tunnel_state != "SUPERSEDED")):
                old.tunnel_state = "SUPERSEDED"
                retired.append(old.id)
        for old_id in retired:
            with self._lock:
                tunnel = self._tunnels.pop(old_id, None)
            if tunnel is not None:
                tunnel.close()

    def _monitor_loop(self):
        while not self._stop.is_set():
            try:
                self._maintain_tunnels()
            except Exception:
                # Individual restoration errors are persisted below. A database
                # outage must not kill the monitor or disclose SSH exceptions.
                pass
            self._stop.wait(3)

    def _maintain_tunnels(self):
        with self.db.read() as session:
            jobs = list(session.scalars(select(RemoteDeployment).where(RemoteDeployment.connection_mode == "ssh-tunnel",
                RemoteDeployment.tunnel_remote_port.is_not(None), RemoteDeployment.tunnel_state.not_in(DISABLED_TUNNELS),
                RemoteDeployment.state.in_({"SUCCEEDED", "WAITING_AGENT", "FAILED", "INTERRUPTED"}))))
            connections = {job.connection_id: session.get(RemoteConnection, job.connection_id) for job in jobs}
        for job in jobs:
            if self._stop.is_set():
                return
            with self._lock:
                tunnel = self._tunnels.get(job.id)
            if self._operation(job) != "install":
                continue
            if tunnel is not None and tunnel.healthy:
                self._tunnel_retries.pop(job.id, None)
                if job.tunnel_state != "HEALTHY":
                    self._tunnel_status(job.id, "HEALTHY", "SSH 通道已恢复")
                continue
            retry_at, attempts = self._tunnel_retries.get(job.id, (0, 0))
            if time.monotonic() < retry_at:
                continue
            self._tunnel_status(job.id, "RECONNECTING", "正在恢复此机器的 SSH 通道")
            try:
                if tunnel is not None:
                    tunnel.close()
                tunnel = self._new_tunnel(connections[job.connection_id], job.tunnel_remote_port)
                tunnel.start()
                if self._stop.is_set():
                    tunnel.close()
                    return
                if tunnel.actual_port != job.tunnel_remote_port:
                    tunnel.close()
                    raise APIError(502, "TUNNEL_PORT_CHANGED", "SSH 通道端口恢复失败")
                if not self._publish_tunnel(job.id, tunnel):
                    continue
                self._tunnel_status(job.id, "HEALTHY", "SSH 通道已就绪")
                self._tunnel_retries.pop(job.id, None)
            except Exception:
                self._tunnel_retries[job.id] = (time.monotonic() + min(60, 3 * 2 ** min(attempts, 5)), attempts + 1)
                self._tunnel_status(job.id, "FAILED", "SSH 通道重连失败，将继续重试；请检查机器网络和 SSH 凭据")

    def _tunnel_status(self, job_id: str, state: str, message: str):
        if self._stop.is_set():
            return
        with self.db.write() as session:
            job = session.get(RemoteDeployment, job_id)
            if job and job.tunnel_state not in DISABLED_TUNNELS and job.tunnel_state != state:
                job.tunnel_state = state
                timestamp = now()
                job.updated_at = timestamp
                self._append_log(job, "TUNNEL", message, timestamp)

    def _view(self, session, job):
        connection = session.get(RemoteConnection, job.connection_id)
        return {"id": job.id, "connection_id": job.connection_id, "host": connection.host, "port": connection.port,
                "username": connection.username, "install_dir": job.install_dir, "onboarding_id": job.onboarding_id,
                "node_id": job.node_id, "state": job.state, "stage": job.stage, "logs": job.logs, "history": job.history,
                "agent": {key: value for key, value in job.agent_config.items() if not key.startswith("_")},
                "operation": self._operation(job), "operation_result": job.agent_config.get("_operation_result"),
                "connection_mode": job.connection_mode, "tunnel_state": job.tunnel_state,
                "error": job.error, "created_at": job.created_at, "updated_at": job.updated_at,
                "started_at": job.started_at, "finished_at": job.finished_at}

    def list(self):
        with self.db.read() as session:
            return [self._view(session, job) for job in session.scalars(select(RemoteDeployment).order_by(RemoteDeployment.created_at.desc()).limit(100))]

    def get(self, job_id: str):
        with self.db.read() as session:
            job = session.get(RemoteDeployment, job_id)
            if not job:
                raise APIError(404, "REMOTE_DEPLOYMENT_NOT_FOUND", "远程部署不存在")
            return self._view(session, job)

    def download_logs(self, job_id: str):
        with self.db.read() as session:
            job = session.get(RemoteDeployment, job_id)
            if not job:
                raise APIError(404, "REMOTE_DEPLOYMENT_NOT_FOUND", "远程部署不存在")
            fallback = list(job.logs)
            path = self._log_path(job.id)
        try:
            if path.parent.is_symlink():
                raise OSError("Deployment log directory cannot be a symlink")
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            stream = os.fdopen(descriptor, "r", encoding="utf-8", errors="replace")
        except OSError:
            stream = None

        def lines():
            def clean(entry):
                return {"timestamp": str(entry.get("timestamp", "")), "stage": str(entry.get("stage", "")),
                        "message": self._safe(entry.get("message", ""))}

            def identity(entry):
                return entry["timestamp"], entry["stage"], entry["message"]

            retained = sorted((clean(entry) for entry in fallback), key=lambda entry: entry["timestamp"])
            if stream is None:
                for entry in retained:
                    yield json.dumps(entry, ensure_ascii=False) + "\n"
                return
            pending = iter(retained)
            next_retained = next(pending, None)
            retained_keys = {identity(entry) for entry in retained}
            seen = set()
            with stream:
                for line in stream:
                    # Log files are application-owned; re-redact before export.
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(entry, dict) and isinstance(entry.get("message"), str):
                        entry = clean(entry)
                        # Repair recoverable file-write gaps using the durable DB tail.
                        while next_retained is not None and next_retained["timestamp"] < entry["timestamp"]:
                            if identity(next_retained) not in seen:
                                yield json.dumps(next_retained, ensure_ascii=False) + "\n"
                            next_retained = next(pending, None)
                        key = identity(entry)
                        if key in retained_keys:
                            seen.add(key)
                        yield json.dumps(entry, ensure_ascii=False) + "\n"
            for remaining in ([next_retained] if next_retained is not None else []):
                if identity(remaining) not in seen:
                    yield json.dumps(remaining, ensure_ascii=False) + "\n"
            for remaining in pending:
                if identity(remaining) not in seen:
                    yield json.dumps(remaining, ensure_ascii=False) + "\n"
        return lines()

    def retry(self, job_id: str, request_url: str, key: str | None):
        old = self.get(job_id)
        if old["state"] not in RETRYABLE:
            raise APIError(409, "DEPLOYMENT_NOT_RETRYABLE", "只有失败或中断的部署可以重试")
        if old["operation"] == "uninstall":
            return self.remove(old["node_id"], "uninstall", key or f"retry-{job_id}-{uid()}")
        config = dict(old["agent"])
        config["profiles"] = [{field: profile[field] for field in ("id", "name", "code_dir", "python", "adapter", "recipe", "execution_mode")
                               if field in profile} for profile in config["profiles"]]
        config = {field: config[field] for field in ("name", "control_plane_url", "roots", "output_root", "state_dir", "profiles")}
        body = RemoteDeploymentInput(connection_id=old["connection_id"], install_dir=old["install_dir"], agent=config)
        return self.create(body, key or f"retry-{job_id}-{uid()}", request_url, retry=True)


def register_remote_routes(app: FastAPI, admin, manager: DeploymentManager):
    def response(value, status=200):
        return JSONResponse(status_code=status, content=value, headers={"Cache-Control": "no-store"})

    @app.post("/api/v1/ssh-connections/test", dependencies=admin)
    def test_connection(body: SSHConnectionInput):
        return response(manager.test_connection(body))

    @app.post("/api/v1/ssh-connections/connect", dependencies=admin)
    def connect_connection(body: SSHAuthenticationInput):
        return response(manager.connect_connection(body))

    @app.post("/api/v1/ssh-connections/{connection_id}/path-suggestions", dependencies=admin)
    def ssh_path_suggestions(connection_id: str, body: SSHPathSuggestInput):
        return response(manager.path_suggestions(connection_id, body))

    @app.post("/api/v1/ssh-connections/{connection_id}/test", dependencies=admin)
    def test_existing_connection(connection_id: str, body: SSHInstallCheckInput):
        return response(manager.test_existing_connection(connection_id, body))

    @app.get("/api/v1/nodes/{node_id}/management", dependencies=admin)
    def node_management(node_id: str):
        return response(manager.management(node_id))

    @app.post("/api/v1/nodes/{node_id}/disconnect", dependencies=admin)
    def node_disconnect(node_id: str, idempotency_key: str | None = Header(default=None)):
        return response(manager.remove(node_id, "disconnect", idempotency_key))

    @app.post("/api/v1/nodes/{node_id}/uninstall", dependencies=admin, status_code=202)
    def node_uninstall(node_id: str, idempotency_key: str | None = Header(default=None)):
        return response(manager.remove(node_id, "uninstall", idempotency_key), 202)

    @app.get("/api/v1/remote-deployments", dependencies=admin)
    def list_deployments():
        return response(manager.list())

    @app.post("/api/v1/remote-deployments", dependencies=admin, status_code=202)
    def create_deployment(body: RemoteDeploymentInput, request: Request, idempotency_key: str | None = Header(default=None)):
        return response(manager.create(body, idempotency_key, str(request.base_url)), 202)

    @app.get("/api/v1/remote-deployments/{job_id}", dependencies=admin)
    def deployment_detail(job_id: str):
        return response(manager.get(job_id))

    @app.get("/api/v1/remote-deployments/{job_id}/logs/download", dependencies=admin)
    def deployment_logs_download(job_id: str):
        return StreamingResponse(manager.download_logs(job_id), media_type="application/x-ndjson",
                                 headers={"Cache-Control": "no-store",
                                          "Content-Disposition": f'attachment; filename="deployment-{job_id}.jsonl"'})

    @app.post("/api/v1/remote-deployments/{job_id}/retry", dependencies=admin, status_code=202)
    def retry_deployment(job_id: str, request: Request, idempotency_key: str | None = Header(default=None)):
        return response(manager.retry(job_id, str(request.base_url), idempotency_key), 202)
