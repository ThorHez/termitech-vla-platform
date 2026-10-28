"""Durable, resumable machine-to-machine checkpoint synchronization."""

from copy import deepcopy
import hashlib
import json
import logging
import math
from pathlib import PurePosixPath
import re
import threading
import time
from types import SimpleNamespace

from pydantic import Field, field_validator
from sqlalchemy import or_, select
from vla_protocol import canonical_digest

from .checkpoint_transfer_models import CheckpointTransfer, TRANSFER_ACTIVE, TRANSFER_RETRYABLE
from .common import APIError, now, offset_time, require_path, uid
from .models import Asset, Command, Node, Profile, Run
from .profile_rebindings import require_not_pending
from .recipe_writeback import require_no_pending_writeback
from .remote_models import RemoteConnection, RemoteDeployment
from .schemas import InputModel


LOG = logging.getLogger(__name__)
EXPORT_COMMANDS = {"POLICY_EXPORT", "POLICY_EXPORT_STATUS", "POLICY_EXPORT_CANCEL"}
PUBLIC_FIELDS = ("id", "name", "source_node_id", "source_run_id", "source_checkpoint_path", "target_node_id",
    "target_profile_id", "target_directory", "target_path", "state", "bytes_total", "bytes_transferred", "files_total",
    "files_transferred", "speed_bytes_per_second", "progress_pct", "attempt", "logs", "error", "manifest_sha256",
    "portable_digest", "created_at", "updated_at", "finished_at")


class TransferInput(InputModel):
    source_run_id: str = Field(min_length=1, max_length=36)
    source_checkpoint_path: str = Field(min_length=1, max_length=4096)
    target_node_id: str = Field(min_length=1, max_length=36)
    target_profile_id: str = Field(min_length=1, max_length=200)
    target_directory: str = Field(min_length=1, max_length=4096)
    name: str | None = Field(default=None, max_length=200)
    record_id: str | None = Field(default=None, min_length=1, max_length=36)
    record_name: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("name", "record_name")
    @classmethod
    def name_text(cls, value):
        if value is not None and (not value.strip() or any(ord(char) < 32 for char in value)):
            raise ValueError("同步名称不能为空或包含控制字符")
        return value.strip() if value is not None else value


def _transfer_warnings(session, row):
    from .policy_deployments import _source_warnings
    binding = row.binding_snapshot
    warnings = _source_warnings(binding.get("source_training") if isinstance(binding, dict) else None)
    if warnings or session is None:
        return warnings
    receipts = select(Command.result["warnings"]).where(Command.node_id == row.source_node_id,
        Command.type.in_({"POLICY_EXPORT", "POLICY_EXPORT_STATUS"}), Command.status == "APPLIED",
        Command.payload["transfer_id"].as_string() == row.id,
        Command.payload["transfer_attempt"].as_integer() == row.attempt,
        Command.payload["export_id"].as_string() == _export_id(row))
    for receipt in session.scalars(receipts):
        if warnings := _source_warnings({"warnings": receipt}):
            return warnings
    return []


def transfer_dict(row, session=None):
    result = {key: deepcopy(getattr(row, key)) for key in PUBLIC_FIELDS}
    if result["files_transferred"] == -1:
        result["files_transferred"] = None
    result["warnings"] = _transfer_warnings(session, row)
    from .deployment_record_models import DeploymentRecordTransfer
    ownership = session.get(DeploymentRecordTransfer, row.id) if session is not None else None
    result["record_id"] = ownership.record_id if ownership is not None else None
    return result


def active_transfers(session, node_id, profile_id=None, code_dir=None):
    result = []
    for row in session.scalars(select(CheckpointTransfer).where(CheckpointTransfer.state.in_(TRANSFER_ACTIVE),
            or_(CheckpointTransfer.source_node_id == node_id, CheckpointTransfer.target_node_id == node_id))):
        profile = row.binding_snapshot["source_profile" if row.source_node_id == node_id else "target_profile"]
        if profile_id is None and code_dir is None or profile["id"] == profile_id or profile["code_dir"] == code_dir:
            result.append(row)
    return result


def _path(value, roots):
    if any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value):
        raise APIError(422, "TRANSFER_PATH_INVALID", "同步路径不能包含控制字符")
    require_path(value, roots)
    return str(PurePosixPath(value))


def _profile(profile):
    return {key: deepcopy(getattr(profile, key)) for key in ("id", "node_id", "code_dir", "python", "adapter", "execution_mode")}


def _online(session, node_id, settings):
    node = session.scalar(select(Node).where(Node.id == node_id).with_for_update())
    if node is None:
        raise APIError(404, "NOT_FOUND", "同步机器不存在")
    if node.status != "ONLINE" or node.last_seen < offset_time(-settings.heartbeat_timeout_seconds):
        raise APIError(409, "NODE_OFFLINE", "源机器与部署机器都需在线，才能同步并验证模型")
    if session.scalar(select(RemoteDeployment.id).where(RemoteDeployment.node_id == node.id,
            RemoteDeployment.state.in_({"QUEUED", "RUNNING", "WAITING_AGENT"})).limit(1)):
        raise APIError(409, "NODE_MAINTENANCE", "机器正在更新 Agent，请等待完成后同步")
    return node


def _require_portable_agent(node, settings):
    # POLICY_DEPLOY_START predates portable checkpoints. An old Agent can
    # otherwise ignore the new payload and serve a different named config.
    capabilities = (node.metrics or {}).get("capabilities")
    if (node.status != "ONLINE" or node.last_seen < offset_time(-settings.heartbeat_timeout_seconds)
            or not isinstance(capabilities, list) or "POLICY_DEPLOY_PORTABLE_V1" not in capabilities):
        raise APIError(409, "POLICY_DEPLOY_AGENT_UNSUPPORTED",
            "目标机器 Agent 尚未确认支持跨机器推理配置，请先在机器资源中更新目标 Agent 并等待新心跳")


def _connection(session, node_id):
    # Only installation evidence bound to this exact node grants use of a saved
    # SSH credential. A client cannot submit a host or another connection id.
    for job in session.scalars(select(RemoteDeployment).where(RemoteDeployment.node_id == node_id,
            RemoteDeployment.state == "SUCCEEDED").order_by(RemoteDeployment.created_at.desc(), RemoteDeployment.id.desc())):
        if (job.agent_config or {}).get("_operation") not in (None, "install"):
            continue
        connection = session.get(RemoteConnection, job.connection_id)
        if connection is not None and connection.fingerprint:
            return connection
    raise APIError(409, "TRANSFER_SSH_UNAVAILABLE", "该机器尚无已绑定的 SSH 接入记录，请先在机器资源中完成 SSH 接入")


def _bindings(session, body, settings):
    from .policy_deployments import _source_training

    run = session.get(Run, body.source_run_id)
    if run is None:
        raise APIError(404, "NOT_FOUND", "来源训练任务不存在")
    if run.template_id != "openpi-sft":
        raise APIError(422, "TRANSFER_SOURCE_UNSUPPORTED", "当前支持 OpenPI 动作策略权重同步")
    if run.node_id == body.target_node_id:
        raise APIError(422, "TRANSFER_SAME_NODE", "源机器与部署机器相同，请直接使用本地权重部署")
    # Stable lock ordering for transfers in opposite directions.
    nodes = {node_id: _online(session, node_id, settings) for node_id in sorted({run.node_id, body.target_node_id})}
    source, target = nodes[run.node_id], nodes[body.target_node_id]
    _require_portable_agent(target, settings)
    source_profile = session.get(Profile, (source.id, run.profile_id))
    target_profile = session.get(Profile, (target.id, body.target_profile_id))
    for profile in (source_profile, target_profile):
        if profile is None or not profile.active or profile.adapter != "openpi-sft" or profile.execution_mode != "native":
            raise APIError(422, "TRANSFER_PROFILE_UNSUPPORTED", "源和目标机器均需配置可用的原生 OpenPI 运行环境")
        require_not_pending(session, profile.node_id, profile.id, profile.code_dir)
        require_no_pending_writeback(session, profile.node_id, profile.code_dir)
    source_path = _path(body.source_checkpoint_path, source.roots)
    target_directory = _path(body.target_directory, target.roots)
    if target_directory == "/":
        raise APIError(422, "TRANSFER_PATH_INVALID", "请选择独立的目标权重目录")
    training = _source_training(session, SimpleNamespace(source_run_id=run.id, node_id=run.node_id,
        profile_id=run.profile_id, checkpoint_path=source_path))
    source_connection, target_connection = _connection(session, source.id), _connection(session, target.id)
    return {"source_node_id": source.id, "source_profile": _profile(source_profile), "target_profile": _profile(target_profile),
        "source_roots": deepcopy(source.roots), "target_roots": deepcopy(target.roots), "source_training": training,
        "source_connection_id": source_connection.id, "target_connection_id": target_connection.id}


def _check_snapshot(session, row, settings):
    body = TransferInput(source_run_id=row.source_run_id, source_checkpoint_path=row.source_checkpoint_path,
        target_node_id=row.target_node_id, target_profile_id=row.target_profile_id, target_directory=row.target_directory)
    current = _bindings(session, body, settings)
    if current != row.binding_snapshot:
        raise APIError(409, "TRANSFER_BINDING_CHANGED", "同步的机器目录、运行环境、SSH 接入或训练配置已变化，请重新创建同步任务")
    return current


def _verify_saved_ssh_credentials(session, bindings, manager):
    """Validate the pinned identities before spending time on CPU export.

    SSHRemote construction decrypts and validates saved connection fields only;
    connecting and host-key verification remain in the existing transfer worker.
    """
    for side, label in (("source", "源机器"), ("target", "目标机器")):
        connection = session.get(RemoteConnection, bindings[side + "_connection_id"])
        if connection is None:
            raise APIError(409, "TRANSFER_SSH_UNAVAILABLE", label + "缺少已绑定的 SSH 接入记录，请先在机器资源中完成 SSH 接入")
        remote = None
        try:
            remote = manager.remote_manager._remote(connection)
        except Exception as exc:
            if getattr(exc, "code", None) == "SSH_CREDENTIAL_UNAVAILABLE":
                raise APIError(503, "TRANSFER_" + side.upper() + "_SSH_CREDENTIAL_UNAVAILABLE",
                    label + "已保存的 SSH 凭据无法解密；请在机器资源中重新测试 SSH 连接",
                    {"side": side}) from None
            raise APIError(422, "TRANSFER_" + side.upper() + "_SSH_CONFIGURATION_INVALID",
                label + "已保存的 SSH 连接信息无效；请在机器资源中重新测试 SSH 连接",
                {"side": side}) from None
        finally:
            if remote is not None:
                remote.close()


def _export_id(row):
    return f"{row.id}-{row.attempt}"


def _pending_exports(session, export_id):
    return [command for command in session.scalars(select(Command).where(Command.type.in_({"POLICY_EXPORT", "POLICY_EXPORT_STATUS"}),
        Command.status == "PENDING")) if command.payload.get("export_id") == export_id]


def _queue_export(session, row, kind="POLICY_EXPORT"):
    payload = {"export_id": _export_id(row), "transfer_id": row.id, "transfer_attempt": row.attempt}
    if kind == "POLICY_EXPORT":
        payload.update(profile_id=row.binding_snapshot["source_profile"]["id"], checkpoint_path=row.source_checkpoint_path,
            source_training=deepcopy(row.binding_snapshot["source_training"]), expires_at=offset_time(120))
    command = Command(id=uid(), node_id=row.source_node_id, type=kind, status="PENDING", created_at=now(), payload=payload)
    session.add(command)
    return command


def _log(row, message):
    row.logs = (row.logs + f"{now()} {message}\n")[-65536:]
    row.updated_at = now()


def _manifest(value, portable):
    if not isinstance(value, dict) or value.get("schema_version") != 1 or not isinstance(value.get("files"), list) or not 1 <= len(value["files"]) <= 20000:
        raise APIError(502, "TRANSFER_INVALID_RESULT", "权重校验清单格式无效")
    paths = set()
    total = 0
    for item in value["files"]:
        if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
            raise APIError(502, "TRANSFER_INVALID_RESULT", "权重校验清单文件字段无效")
        path = item["path"]
        if (not isinstance(path, str) or not path or len(path) > 4096 or path.startswith("/") or ".." in PurePosixPath(path).parts
                or str(PurePosixPath(path)) != path or path in paths or any(ord(char) < 32 for char in path)
                or type(item["size"]) is not int or not 0 <= item["size"] < 2**63
                or not isinstance(item["sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"])):
            raise APIError(502, "TRANSFER_INVALID_RESULT", "权重校验清单包含无效或重复路径")
        paths.add(path)
        total += item["size"]
    if (total >= 2**63 or value.get("bytes_total") != total or value.get("files_total") != len(paths)
            or not ("model.safetensors" in paths or any(path.startswith("params/") for path in paths))
            or not any(path.startswith("assets/") and path.endswith("norm_stats.json") for path in paths)):
        raise APIError(502, "TRANSFER_INVALID_RESULT", "权重校验清单缺少模型、归一化参数或总量不一致")
    package = json.dumps(portable, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    sidecar = next((item for item in value["files"] if item["path"] == "termitech-policy.json"), None)
    if sidecar != {"path": "termitech-policy.json", "size": len(package), "sha256": hashlib.sha256(package).hexdigest()}:
        raise APIError(502, "TRANSFER_INVALID_RESULT", "权重清单中的推理配置与已确认导出不一致")
    return value


def acknowledge_export(session, command, node, status, value, manager=None):
    if command.status != "PENDING":
        return
    row = session.scalar(select(CheckpointTransfer).where(CheckpointTransfer.id == command.payload.get("transfer_id"),
        CheckpointTransfer.source_node_id == node.id).with_for_update())
    if row is None:
        raise APIError(409, "TRANSFER_NOT_FOUND", "同步任务不存在")
    clean = {}
    try:
        if not isinstance(value, dict) or value.get("state") not in {"RUNNING", "READY", "FAILED"}:
            # Older Agents reject unknown commands without a state.
            if isinstance(value, dict) and isinstance(value.get("error"), dict) and value["error"].get("code") == "COMMAND_UNSUPPORTED":
                clean = {"state": "FAILED", "error": {"code": "TRANSFER_AGENT_UNSUPPORTED", "message": "源机器 Agent 尚不支持推理配置导出，请先更新 Agent"}}
            else:
                raise ValueError("Invalid export state")
        else:
            clean["state"] = value["state"]
            if value.get("error"):
                error = value["error"]
                if not isinstance(error, dict) or not isinstance(error.get("code"), str) or not isinstance(error.get("message"), str):
                    raise ValueError("Invalid export error")
                clean["error"] = {"code": error["code"][:128], "message": error["message"][:2000]}
            if clean["state"] == "READY":
                config, config_digest = value.get("portable_config"), value.get("portable_digest")
                if not isinstance(config, dict) or len(json.dumps(config, ensure_ascii=False, allow_nan=False).encode()) > 512 * 1024:
                    raise ValueError("Portable configuration exceeds bounded JSON contract")
                if canonical_digest(config) != config_digest:
                    raise ValueError("Portable configuration digest differs")
                clean.update(portable_config=deepcopy(config), portable_digest=config_digest)
        if status == "REJECTED" and clean.get("state") != "FAILED":
            raise ValueError("Rejected export has no failure evidence")
    except (ValueError, TypeError, OverflowError, RecursionError, UnicodeError):
        clean = {"state": "FAILED", "error": {"code": "TRANSFER_EXPORT_INVALID", "message": "源 Agent 返回的推理配置导出凭据无效"}}
    if clean.get("error") and manager is not None:
        clean["error"]["message"] = manager._safe(session, row, clean["error"]["message"])
    command.status = status if command.type == "POLICY_EXPORT_CANCEL" else "REJECTED" if clean["state"] == "FAILED" else "APPLIED"
    if command.status == "APPLIED":
        from .policy_deployments import _source_warnings
        if warnings := _source_warnings(value):
            clean["warnings"] = warnings
    command.result, command.acknowledged_at = deepcopy(clean), now()
    if row.state not in TRANSFER_ACTIVE or command.payload.get("transfer_attempt") != row.attempt:
        return
    if clean["state"] == "READY":
        if row.portable_digest and row.portable_digest != clean["portable_digest"]:
            row.state = "FAILED"
            row.error = {"code": "TRANSFER_EXPORT_CHANGED", "message": "重试时推理配置发生变化，请重新创建同步任务"}
            row.finished_at = now()
        else:
            row.portable_config, row.portable_digest = clean["portable_config"], clean["portable_digest"]
            _log(row, "推理配置已从源机器导出并通过摘要校验")
    elif clean["state"] == "FAILED":
        row.state, row.error, row.finished_at = "FAILED", clean.get("error") or {"code": "TRANSFER_EXPORT_FAILED", "message": "源机器未完成推理配置导出"}, now()
        _log(row, row.error["message"])
    else:
        row.updated_at = now()


class CheckpointTransferManager:
    def __init__(self, db, settings, remote_manager, *, transfer_function=None):
        self.db, self.settings, self.remote_manager = db, settings, remote_manager
        self.transfer_function = transfer_function
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._workers = {}
        self._remotes = {}
        self._semaphore = threading.BoundedSemaphore(2)
        self._export_timeout = 900

    def start(self):
        with self.db.write() as session:
            for row in session.scalars(select(CheckpointTransfer).where(CheckpointTransfer.state.in_(TRANSFER_ACTIVE))):
                row.state, row.finished_at = "INTERRUPTED", now()
                row.error = {"code": "TRANSFER_INTERRUPTED", "message": "控制台已重启，同步已中断；已传输的部分文件保留，可重试"}
                _log(row, row.error["message"])
                self._retire_exports(session, row)

    def close(self):
        self._stop.set()
        with self._lock:
            remotes = [remote for pair in self._remotes.values() for remote in pair]
            workers = list(self._workers.values())
        for remote in remotes:
            try:
                remote.close()
            except Exception:
                pass
        with self.db.write() as session:
            for row in session.scalars(select(CheckpointTransfer).where(CheckpointTransfer.state.in_(TRANSFER_ACTIVE))):
                row.state, row.finished_at = "INTERRUPTED", now()
                row.error = {"code": "TRANSFER_INTERRUPTED", "message": "控制台服务停止，同步已中断；部分文件保留，可重试"}
                _log(row, row.error["message"])
                self._retire_exports(session, row)
        for worker in workers:
            worker.join(timeout=5)

    @staticmethod
    def _retire_exports(session, row):
        for command in _pending_exports(session, _export_id(row)):
            command.status, command.acknowledged_at = "REJECTED", now()
            command.result = {"error": {"code": "TRANSFER_INACTIVE", "message": "同步已取消或中断，不再执行此导出命令"}}
        # A source worker may already have acknowledged RUNNING, or be just
        # about to launch. The source Agent persists a cancellation tombstone.
        cancellation = [command for command in session.scalars(select(Command).where(Command.node_id == row.source_node_id,
            Command.type == "POLICY_EXPORT_CANCEL")) if command.payload.get("export_id") == _export_id(row)]
        if not cancellation:
            _queue_export(session, row, "POLICY_EXPORT_CANCEL")

    def _launch(self, transfer_id):
        with self._lock:
            current = self._workers.get(transfer_id)
            if current and current.is_alive():
                return
            worker = threading.Thread(target=self._run, args=(transfer_id,), name="vla-checkpoint-transfer", daemon=True)
            self._workers[transfer_id] = worker
            worker.start()

    def _cancelled(self, transfer_id, attempt):
        if self._stop.is_set():
            return True
        with self.db.read() as session:
            row = session.get(CheckpointTransfer, transfer_id)
            return row is None or row.state not in TRANSFER_ACTIVE or row.attempt != attempt

    def _progress(self, transfer_id, attempt, event):
        with self.db.write() as session:
            row = session.scalar(select(CheckpointTransfer).where(CheckpointTransfer.id == transfer_id).with_for_update())
            if row is None or row.attempt != attempt or row.state not in TRANSFER_ACTIVE:
                return
            stage = event.get("stage")
            if stage in {"PREPARING", "TRANSFERRING", "VERIFYING"}:
                row.state = stage
            for source, target in (("bytes_done", "bytes_transferred"), ("bytes_total", "bytes_total"),
                    ("files_done", "files_transferred"), ("files_total", "files_total")):
                if type(event.get(source)) is int and event[source] >= 0:
                    setattr(row, target, event[source])
                elif source == "files_done" and source in event and event[source] is None:
                    row.files_transferred = -1
            for source, target in (("speed_bps", "speed_bytes_per_second"), ("percent", "progress_pct")):
                if type(event.get(source)) in (int, float) and math.isfinite(event[source]) and event[source] >= 0:
                    setattr(row, target, min(100, event[source]) if source == "percent" else event[source])
            if event.get("message"):
                # Transport scripts emit operational receipts, not arbitrary
                # remote stdout or credentials. Still apply the shared redactor.
                _log(row, self._safe(session, row, str(event["message"])))
            row.updated_at = now()

    def _safe(self, session, row, message):
        for key in ("source_connection_id", "target_connection_id"):
            connection = session.get(RemoteConnection, row.binding_snapshot[key])
            message = self.remote_manager._safe(message, connection)
        return message

    def _run(self, transfer_id):
        acquired = False
        remotes = []
        attempt = None
        try:
            while not self._stop.is_set():
                if self._semaphore.acquire(timeout=.2):
                    acquired = True
                    break
            if not acquired:
                return
            with self.db.write() as session:
                row = session.get(CheckpointTransfer, transfer_id)
                if row is None or row.state not in TRANSFER_ACTIVE:
                    return
                attempt = row.attempt
                _check_snapshot(session, row, self.settings)
                row = session.scalar(select(CheckpointTransfer).where(CheckpointTransfer.id == transfer_id)
                    .with_for_update().execution_options(populate_existing=True))
                if row.state not in TRANSFER_ACTIVE or row.attempt != attempt:
                    return
                row.state = "EXPORTING"
                _log(row, "等待源 Agent 导出可移植推理配置")
            deadline = time.monotonic() + self._export_timeout
            last_poll = 0
            while not self._cancelled(transfer_id, attempt):
                with self.db.write() as session:
                    row = session.get(CheckpointTransfer, transfer_id)
                    if row.portable_config is not None:
                        break
                    if not _pending_exports(session, _export_id(row)) and time.monotonic() - last_poll >= 2:
                        _queue_export(session, row, "POLICY_EXPORT_STATUS")
                        last_poll = time.monotonic()
                if time.monotonic() >= deadline:
                    raise APIError(504, "TRANSFER_EXPORT_TIMEOUT", "等待源机器导出推理配置超时，请检查 Agent 后重试")
                self._stop.wait(.2)
            if self._cancelled(transfer_id, attempt):
                return
            with self.db.write() as session:
                row = session.get(CheckpointTransfer, transfer_id)
                bindings = _check_snapshot(session, row, self.settings)
                row = session.scalar(select(CheckpointTransfer).where(CheckpointTransfer.id == transfer_id)
                    .with_for_update().execution_options(populate_existing=True))
                if row.state not in TRANSFER_ACTIVE or row.attempt != attempt:
                    return
                source_connection = session.get(RemoteConnection, bindings["source_connection_id"])
                target_connection = session.get(RemoteConnection, bindings["target_connection_id"])
                source_remote = self.remote_manager._remote(source_connection)
                target_remote = self.remote_manager._remote(target_connection)
                remotes = [source_remote, target_remote]
                spec = {key: deepcopy(getattr(row, key)) for key in ("source_checkpoint_path", "target_directory", "portable_config", "portable_digest")}
                row.state = "PREPARING"
                _log(row, "正在通过已绑定 SSH 连接检查源权重与目标存储")
            with self._lock:
                self._remotes[transfer_id] = remotes
            for remote in remotes:
                if self._cancelled(transfer_id, attempt):
                    return
                remote.connect()
            transfer_function = self.transfer_function
            if transfer_function is None:
                from .checkpoint_transfer_transport import transfer_checkpoint
                transfer_function = transfer_checkpoint
            result = transfer_function(source_remote, target_remote, spec["source_checkpoint_path"], spec["target_directory"], transfer_id,
                bindings["source_roots"], bindings["target_roots"], progress=lambda event: self._progress(transfer_id, attempt, event),
                cancelled=lambda: self._cancelled(transfer_id, attempt), extra_files={"termitech-policy.json": json.dumps(
                    {"config": spec["portable_config"], "digest": spec["portable_digest"]}, sort_keys=True, ensure_ascii=False,
                    separators=(",", ":"), allow_nan=False).encode()})
            if self._cancelled(transfer_id, attempt):
                return
            with self.db.write() as session:
                row = session.get(CheckpointTransfer, transfer_id)
                _check_snapshot(session, row, self.settings)
                row = session.scalar(select(CheckpointTransfer).where(CheckpointTransfer.id == transfer_id)
                    .with_for_update().execution_options(populate_existing=True))
                if row.attempt != attempt or row.state not in TRANSFER_ACTIVE:
                    return
                manifest = _manifest(result["manifest"], {"config": spec["portable_config"], "digest": spec["portable_digest"]})
                if canonical_digest(manifest) != result["manifest_sha256"]:
                    raise APIError(502, "TRANSFER_INVALID_RESULT", "权重同步返回的校验清单摘要不一致")
                expected_path = str(PurePosixPath(row.target_directory) / row.id)
                if result["target_path"] != expected_path:
                    raise APIError(502, "TRANSFER_INVALID_RESULT", "权重同步的发布目录与任务不一致")
                if result["bytes_total"] != manifest["bytes_total"] or result["files_total"] != manifest["files_total"]:
                    raise APIError(502, "TRANSFER_INVALID_RESULT", "权重同步总量与文件校验清单不一致")
                row.target_path, row.manifest, row.manifest_sha256 = result["target_path"], deepcopy(manifest), result["manifest_sha256"]
                row.bytes_total = row.bytes_transferred = result["bytes_total"]
                row.files_total = row.files_transferred = result["files_total"]
                row.progress_pct, row.speed_bytes_per_second = 100, 0
                row.state, row.error, row.finished_at = "SUCCEEDED", None, now()
                _log(row, "推理 checkpoint（已排除根目录 train_state/）与推理配置已校验并发布，等待用户确认部署")
                session.add(Asset(id=uid(), node_id=row.target_node_id, kind="MODEL", name=row.name,
                    path=row.target_path, version=row.manifest_sha256[:16], state="AVAILABLE", created_at=now(),
                    metadata_json={"format": "openpi-transferred", "transfer_id": row.id, "source_run_id": row.source_run_id,
                        "source_node_id": row.source_node_id, "manifest_sha256": row.manifest_sha256,
                        "portable_digest": row.portable_digest, "evaluated": False}))
        except Exception as exc:
            if attempt is not None:
                code = getattr(exc, "code", "TRANSFER_FAILED")
                message = getattr(exc, "message", str(exc))
                with self.db.write() as session:
                    row = session.scalar(select(CheckpointTransfer).where(CheckpointTransfer.id == transfer_id).with_for_update())
                    if row is not None and row.attempt == attempt and row.state in TRANSFER_ACTIVE:
                        row.state, row.finished_at = "FAILED", now()
                        row.error = {"code": str(code)[:128], "message": self._safe(session, row, str(message))[:2000]}
                        _log(row, row.error["message"])
                        self._retire_exports(session, row)
                LOG.info("Checkpoint transfer %s failed: %s", transfer_id, code)
        finally:
            for remote in remotes:
                try:
                    remote.close()
                except Exception:
                    pass
            with self._lock:
                self._remotes.pop(transfer_id, None)
            if acquired:
                self._semaphore.release()


def portable_deployment(session, body, settings):
    """Only a verified transfer may rebind a source run onto another machine."""
    row = session.get(CheckpointTransfer, body.transfer_id)
    if row is None:
        raise APIError(404, "NOT_FOUND", "权重同步任务不存在")
    if row.state != "SUCCEEDED" or row.portable_config is None or row.manifest is None:
        raise APIError(409, "TRANSFER_NOT_READY", "权重同步尚未完成并通过校验，不能部署")
    if (body.node_id != row.target_node_id or body.profile_id != row.target_profile_id
            or body.checkpoint_path != row.target_path or body.source_run_id != row.source_run_id):
        raise APIError(422, "TRANSFER_DEPLOYMENT_MISMATCH", "部署机器、环境、目录与来源训练必须匹配已校验的同步任务")
    target = session.get(Node, row.target_node_id)
    profile = session.get(Profile, (row.target_node_id, row.target_profile_id))
    if (target is None or profile is None or not profile.active or target.roots != row.binding_snapshot["target_roots"]
            or _profile(profile) != row.binding_snapshot["target_profile"]):
        raise APIError(409, "TRANSFER_BINDING_CHANGED", "目标机器的目录或运行环境已变化，请重新同步模型")
    _require_portable_agent(target, settings)
    if canonical_digest(row.portable_config) != row.portable_digest or canonical_digest(row.manifest) != row.manifest_sha256:
        raise APIError(409, "TRANSFER_RECEIPT_INVALID", "同步凭据摘要不一致，请重新同步模型")
    _manifest(row.manifest, {"config": row.portable_config, "digest": row.portable_digest})
    return {"config": deepcopy(row.portable_config), "digest": row.portable_digest,
        "checkpoint_manifest_digest": row.manifest_sha256, "manifest": deepcopy(row.manifest),
        "transfer_id": row.id, "source_node_id": row.source_node_id, "source_run_id": row.source_run_id}


def register_routes(app, admin, manager):
    db, settings = manager.db, manager.settings

    @app.get("/api/v1/checkpoint-transfers", dependencies=admin)
    def list_transfers():
        with db.read() as session:
            return [transfer_dict(row, session) for row in session.scalars(select(CheckpointTransfer).order_by(
                CheckpointTransfer.created_at.desc(), CheckpointTransfer.id.desc()))]

    @app.get("/api/v1/checkpoint-transfers/{transfer_id}", dependencies=admin)
    def get_transfer(transfer_id: str):
        with db.read() as session:
            row = session.get(CheckpointTransfer, transfer_id)
            if row is None:
                raise APIError(404, "NOT_FOUND", "同步任务不存在")
            return transfer_dict(row, session)

    @app.post("/api/v1/checkpoint-transfers", dependencies=admin, status_code=202)
    def create_transfer(body: TransferInput):
        with db.write() as session:
            bindings = _bindings(session, body, settings)
            from .deployment_record_models import DeploymentRecordTransfer
            from .deployment_records import create_record, require_preparation_record
            record = require_preparation_record(session, body.record_id) if body.record_id is not None else None
            active = list(session.scalars(select(CheckpointTransfer).where(CheckpointTransfer.state.in_(TRANSFER_ACTIVE))))
            if len(active) >= 20:
                raise APIError(429, "TRANSFER_BUSY", "当前同步任务较多，请等待已有任务完成")
            legacy_deduplication = body.record_id is None and body.record_name is None
            for existing in active:
                ownership = session.get(DeploymentRecordTransfer, existing.id)
                same_record = record is not None and ownership is not None and ownership.record_id == record.id
                if ((same_record or legacy_deduplication)
                        and existing.source_run_id == body.source_run_id and existing.source_checkpoint_path == body.source_checkpoint_path
                        and existing.target_node_id == body.target_node_id and existing.target_profile_id == body.target_profile_id
                        and existing.target_directory == body.target_directory):
                    if ownership is None:
                        from .deployment_records import transfer_record
                        transfer_record(session, existing)
                    return transfer_dict(existing, session)
            _verify_saved_ssh_credentials(session, bindings, manager)
            run = session.get(Run, body.source_run_id)
            stamp = now()
            row = CheckpointTransfer(id=uid(), name=body.name or (run.name + " · 权重同步")[:200],
                source_node_id=bindings["source_node_id"], source_run_id=run.id,
                source_checkpoint_path=_path(body.source_checkpoint_path, bindings["source_roots"]),
                target_node_id=body.target_node_id, target_profile_id=body.target_profile_id,
                target_directory=_path(body.target_directory, bindings["target_roots"]), state="QUEUED", attempt=1,
                bytes_total=0, bytes_transferred=0, files_total=0, files_transferred=0, speed_bytes_per_second=0,
                progress_pct=0, logs="", binding_snapshot=bindings, created_at=stamp, updated_at=stamp)
            session.add(row)
            session.flush()
            if record is None:
                record = create_record(session, row.id, body.record_name or row.name, row.created_at)
            session.add(DeploymentRecordTransfer(transfer_id=row.id, record_id=record.id,
                provenance={"origin": "created_transfer", "linked_at": stamp}))
            session.flush()
            _queue_export(session, row)
            result = transfer_dict(row, session)
        manager._launch(row.id)
        return result

    @app.post("/api/v1/checkpoint-transfers/{transfer_id}/retry", dependencies=admin, status_code=202)
    def retry_transfer(transfer_id: str):
        with manager._lock:
            worker = manager._workers.get(transfer_id)
            if worker is not None and worker.is_alive():
                raise APIError(409, "TRANSFER_CANCELLING", "上一次同步仍在结束，请稍后重试")
        with db.write() as session:
            row = session.get(CheckpointTransfer, transfer_id)
            if row is None:
                raise APIError(404, "NOT_FOUND", "同步任务不存在")
            if row.state not in TRANSFER_RETRYABLE:
                raise APIError(409, "TRANSFER_NOT_RETRYABLE", "只有失败、取消或中断的同步可以重试")
            bindings = _check_snapshot(session, row, settings)
            _verify_saved_ssh_credentials(session, bindings, manager)
            row = session.scalar(select(CheckpointTransfer).where(CheckpointTransfer.id == transfer_id)
                .with_for_update().execution_options(populate_existing=True))
            if row.state not in TRANSFER_RETRYABLE:
                raise APIError(409, "TRANSFER_NOT_RETRYABLE", "同步状态已变化，请刷新后重试")
            manager._retire_exports(session, row)
            row.attempt += 1
            row.state, row.error, row.finished_at = "QUEUED", None, None
            row.target_path, row.speed_bytes_per_second = None, 0
            # A new inventory may exclude training-only state or reflect a
            # different verified partial. Recount it before claiming progress.
            row.bytes_total = row.bytes_transferred = 0
            row.files_total = row.files_transferred = 0
            row.progress_pct = 0
            _log(row, "重试同步，将重新核对源权重并复用已传输的部分文件")
            # Re-export even when a prior attempt produced a config, so code
            # drift cannot silently reuse an obsolete portable receipt.
            row.portable_config = None
            _queue_export(session, row)
            result = transfer_dict(row, session)
        manager._launch(transfer_id)
        return result

    @app.post("/api/v1/checkpoint-transfers/{transfer_id}/cancel", dependencies=admin, status_code=202)
    def cancel_transfer(transfer_id: str):
        with db.write() as session:
            row = session.scalar(select(CheckpointTransfer).where(CheckpointTransfer.id == transfer_id).with_for_update())
            if row is None:
                raise APIError(404, "NOT_FOUND", "同步任务不存在")
            if row.state in TRANSFER_ACTIVE:
                row.state, row.finished_at, row.speed_bytes_per_second = "CANCELLED", now(), 0
                _log(row, "用户取消同步；已传输的部分文件保留，未删除源权重或已发布模型")
                manager._retire_exports(session, row)
            result = transfer_dict(row, session)
        with manager._lock:
            remotes = list(manager._remotes.get(transfer_id, []))
        for remote in remotes:
            try:
                remote.close()
            except Exception:
                pass
        return result
