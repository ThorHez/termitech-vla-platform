"""Catalogue deletion history and inode-bound Agent file deletion."""

from pathlib import PurePosixPath
import re

from fastapi import Request
from pydantic import Field
from sqlalchemy import or_, select
from sqlalchemy.orm import aliased

from .asset_transfer_models import ASSET_TRANSFER_ACTIVE, AssetTransfer
from .auth_models import User
from .checkpoint_transfer_models import TRANSFER_ACTIVE, CheckpointTransfer
from .common import APIError, TERMINAL, absolute_path, now, offset_time, uid
from .dataset_batch_models import DatasetScan
from .models import Allocation, Asset, AssetDeletion, Attempt, Command, Node, Profile, Run
from .policy_deployment_models import RESOURCE_STATES, PolicyDeployment
from .remote_models import RemoteConnection, RemoteDeployment
from .resource_authorization import bind_command_actor, request_user_id, require_active_actor
from .schemas import InputModel

INSPECT = "ASSET_PURGE_INSPECT"
PURGE = "ASSET_PURGE"
COMMANDS = {INSPECT, PURGE}
PERMISSION = "assets.purge"
HEX = re.compile(r"[0-9a-f]{64}")
ERROR_CODE = re.compile(r"[A-Z][A-Z0-9_]{0,95}")
FINAL = {"DELETED", "MISSING"}
DELIVERY_TIMEOUT = 10 * 60
MAX_PROTECTED_PATHS = 10_000
MAX_PROTECTED_PATHS_BYTES = 1024 * 1024


class PurgeInput(InputModel):
    confirm_path: str = Field(min_length=1, max_length=4096)
    inspection_id: str = Field(min_length=1, max_length=36)


def record_archive(session, asset, request):
    if session.get(AssetDeletion, asset.id) is None:
        session.add(AssetDeletion(asset_id=asset.id, archived_at=now(),
                                  archived_by_user_id=request_user_id(request)))


def _path(value):
    if (not isinstance(value, str) or not absolute_path(value)
            or any(ord(c) < 32 or 127 <= ord(c) <= 159 or 0xD800 <= ord(c) <= 0xDFFF for c in value)):
        return None
    return PurePosixPath("/" + value.lstrip("/"))


def _overlap(path, other):
    return other is not None and (path == other or path in other.parents or other in path.parents)


def _machine_node_ids(session, node):
    """A second Agent on the same machine can access the same local files."""
    identities = select(RemoteConnection.host, RemoteConnection.port, RemoteConnection.fingerprint,
                        RemoteDeployment.node_id).join(RemoteDeployment, RemoteDeployment.connection_id == RemoteConnection.id)
    current = identities.where(RemoteDeployment.node_id == node.id, RemoteDeployment.state == "SUCCEEDED")
    addresses, fingerprints = set(), set()
    for host, port, fingerprint, _ in session.execute(current):
        if host not in {"127.0.0.1", "localhost", "::1"}:
            addresses.add((host, port))
        if fingerprint:
            fingerprints.add(fingerprint)
    candidates = set(session.scalars(select(Node.id).where(Node.hostname == node.hostname))) if node.hostname else set()
    if addresses or fingerprints:
        for host, port, fingerprint, node_id in session.execute(identities.where(RemoteDeployment.state == "SUCCEEDED")):
            if fingerprint in fingerprints or (host, port) in addresses:
                candidates.add(node_id)
    return candidates | {node.id}


def _purging_nodes(session):
    return set(session.scalars(select(Command.node_id).where(Command.type == PURGE, Command.status == "PENDING",
        or_(Command.delivered_at.is_not(None), Command.created_at > offset_time(-DELIVERY_TIMEOUT)))))


def file_purge_pending(session, node_id, pending=None):
    pending = _purging_nodes(session) if pending is None else pending
    if not pending:
        return False
    node = session.get(Node, node_id)
    return node is not None and bool(pending & _machine_node_ids(session, node))


def require_no_file_purge(session, node_ids):
    """Serialize work admission with the purging request's AuthGuard."""
    pending = _purging_nodes(session)
    if any(file_purge_pending(session, node_id, pending) for node_id in set(node_ids)):
        raise APIError(409, "ASSET_PURGE_PENDING", "机器正在删除资产文件，请等待 Agent 返回结果后再操作")


def _node_error(node, heartbeat_timeout):
    if node.status != "ONLINE" or node.last_seen < offset_time(-heartbeat_timeout):
        return APIError(409, "NODE_OFFLINE", "机器尚未在线，无法检查或删除文件")
    capabilities = (node.metrics or {}).get("capabilities")
    if not isinstance(capabilities, list) or PURGE not in capabilities:
        return APIError(409, "ASSET_PURGE_AGENT_UNSUPPORTED", "此机器的 Agent 尚不支持删除资产文件，请先更新 Agent")
    return None


def _protect_roots(path, nodes, profiles):
    protected = set()
    for sibling in nodes:
        for value in (*sibling.roots, sibling.output_root):
            root = _path(value)
            if root is not None and (path == root or path in root.parents):
                raise APIError(409, "ASSET_PURGE_UNSAFE_PATH", "资产路径包含机器的资源根目录")
        if _overlap(path, _path(sibling.cache_root)):
            raise APIError(409, "ASSET_PURGE_UNSAFE_PATH", "资产路径与机器缓存目录重叠")
        if _path(sibling.cache_root) is not None:
            protected.add(sibling.cache_root)
    for profile in profiles:
        code, python = _path(profile.code_dir), _path(profile.python)
        if (code is not None and (path == code or path in code.parents)) or _overlap(path, python):
            raise APIError(409, "ASSET_PURGE_UNSAFE_PATH", "资产路径包含已登记的训练代码或 Python 环境")
        if python is not None:
            protected.add(profile.python)
        if code is not None:
            protected.update(str(code / name) for name in (".git", ".venv", "src", "scripts", "packages", "third_party"))
    return protected


def _require_target(session, asset, node, heartbeat_timeout):
    if error := _node_error(node, heartbeat_timeout):
        raise error
    path = _path(asset.path)
    roots = [_path(root) for root in (*node.roots, node.output_root)]
    if path is None or asset.path.startswith("//") or not any(root is not None and root in path.parents for root in roots):
        raise APIError(409, "ASSET_PURGE_UNSAFE_PATH", "资产路径必须位于机器登记目录内，并且不能是登记根目录")
    if asset.kind not in {"DATASET", "MODEL", "NORM"}:
        raise APIError(409, "ASSET_PURGE_UNSUPPORTED_KIND", "此类资产仅支持删除平台登记")
    node_ids = _machine_node_ids(session, node)
    protected_paths = _protect_roots(path, session.scalars(select(Node).where(Node.id.in_(node_ids))),
                                      session.scalars(select(Profile).where(Profile.node_id.in_(node_ids))))
    registered = session.execute(select(Asset.path, Asset.name, Asset.state).where(
        Asset.node_id.in_(node_ids), Asset.id != asset.id,
        or_(Asset.state != "ARCHIVED", Asset.created_at > asset.created_at)))
    for other, name, state in registered:
        other_path = _path(other)
        if state == "ARCHIVED":
            if path == other_path:
                raise APIError(409, "ASSET_PURGE_PATH_REUSED", "此路径有较新的删除登记，请从较新的记录核对文件")
            continue
        if _overlap(path, other_path):
            raise APIError(409, "ASSET_PURGE_PATH_IN_USE", f"此路径与其他已登记资产「{name}」重叠，请先确认其用途")
        if other_path is not None:
            protected_paths.add(other)
    for deployment in session.scalars(select(RemoteDeployment).where(
            RemoteDeployment.node_id.in_(node_ids), RemoteDeployment.state == "SUCCEEDED")):
        if (deployment.agent_config or {}).get("_operation", "install") in {"install", "upgrade"}:
            protected_paths.update(value for value in (
                deployment.install_dir, (deployment.agent_config or {}).get("state_dir"),
            ) if _path(value) is not None)
    if (len(protected_paths) > MAX_PROTECTED_PATHS
            or sum(len(path.encode("utf-8")) for path in protected_paths) > MAX_PROTECTED_PATHS_BYTES):
        raise APIError(409, "ASSET_PURGE_CATALOGUE_LIMIT", "机器登记目录过多，无法完成所有路径的重叠检查")
    return node_ids, sorted(protected_paths)


def _require_idle(session, node_ids, ignored_command_id=None):
    tasks = (
        select(Run.id).where(Run.node_id.in_(node_ids), Run.state.not_in(TERMINAL)),
        select(Attempt.id).where(Attempt.node_id.in_(node_ids), Attempt.state.not_in(TERMINAL)),
        select(Allocation.attempt_id).where(Allocation.node_id.in_(node_ids)),
        select(PolicyDeployment.id).where(PolicyDeployment.node_id.in_(node_ids), PolicyDeployment.state.in_(RESOURCE_STATES)),
        select(AssetTransfer.id).where(AssetTransfer.state.in_(ASSET_TRANSFER_ACTIVE), AssetTransfer.deleted_at.is_(None),
                                       or_(AssetTransfer.source_node_id.in_(node_ids), AssetTransfer.target_node_id.in_(node_ids))),
        select(CheckpointTransfer.id).where(CheckpointTransfer.state.in_(TRANSFER_ACTIVE),
            or_(CheckpointTransfer.source_node_id.in_(node_ids), CheckpointTransfer.target_node_id.in_(node_ids))),
        select(DatasetScan.id).where(DatasetScan.node_id.in_(node_ids), DatasetScan.state.in_({"QUEUED", "SCANNING"})),
        select(RemoteDeployment.id).where(RemoteDeployment.node_id.in_(node_ids),
                                         RemoteDeployment.state.in_({"QUEUED", "RUNNING", "WAITING_AGENT"})),
        select(Command.id).where(Command.node_id.in_(node_ids), Command.status == "PENDING", Command.type != INSPECT,
                                 Command.id != (ignored_command_id or ""),
                                 or_(Command.type != PURGE, Command.delivered_at.is_not(None),
                                     Command.created_at > offset_time(-DELIVERY_TIMEOUT))),
    )
    if any(session.scalar(query.limit(1)) is not None for query in tasks):
        raise APIError(409, "ASSET_PURGE_NODE_BUSY", "机器还有训练、校验、同步或其他后台操作，请等待完成后再删除文件")


def _status(command):
    if command is None:
        return None
    if command.status == "PENDING":
        if not command.delivered_at and command.created_at <= offset_time(-DELIVERY_TIMEOUT):
            return "FAILED"
        return "RUNNING" if command.delivered_at else "PENDING"
    if command.status == "APPLIED":
        return command.result["status"]
    return "FAILED"


def _command_view(command):
    if command is None:
        return None
    result = command.result or {}
    error = result.get("error") or {}
    status = _status(command)
    if status == "FAILED" and command.status == "PENDING":
        error = _expired_error()
    return {"id": command.id, "status": status, "requested_at": command.created_at,
            "finished_at": command.acknowledged_at, "error_code": error.get("code"),
            "error_message": error.get("message")}


def _needs_new_inspection(command):
    result = (command.result or {}) if command is not None else {}
    return (result.get("error", {}).get("code") == "ASSET_PURGE_IDENTITY_CHANGED"
            and result.get("staged") is not True)


def _recovery_required(command):
    return (command is not None and command.status == "REJECTED"
            and (command.result or {}).get("error", {}).get("code") == "ASSET_PURGE_INDETERMINATE")


def _enqueue(session, asset, command_type, request, **extra):
    command = Command(id=uid(), node_id=asset.node_id, type=command_type, status="PENDING", created_at=now(),
                      payload={"asset_id": asset.id, "path": asset.path, **extra})
    session.add(bind_command_actor(command, request, PERMISSION))
    session.flush()
    return command


def _queued_error(session, command, heartbeat_timeout):
    """Recheck mutable references before the first destructive delivery."""
    asset = session.get(Asset, command.payload["asset_id"])
    record = session.get(AssetDeletion, command.payload["asset_id"])
    if (asset is None or asset.state != "ARCHIVED" or record is None or record.purge_command_id != command.id
            or asset.path != command.payload["path"] or asset.node_id != command.node_id):
        return APIError(409, "ASSET_PURGE_RECORD_CHANGED", "资产登记记录已变化，请刷新删除台账")
    node = session.scalar(select(Node).where(Node.id == command.node_id).with_for_update())
    try:
        node_ids, protected_paths = _require_target(session, asset, node, heartbeat_timeout)
        _require_idle(session, node_ids, ignored_command_id=command.id)
        command.payload = {**command.payload, "protected_paths": protected_paths}
    except APIError as error:
        return error
    return None


def prepare_delivery(session, command, heartbeat_timeout):
    if command.created_at <= offset_time(-DELIVERY_TIMEOUT):
        _reject(command, _expired_error())
        return False
    if command.type != PURGE:
        return True
    if error := _queued_error(session, command, heartbeat_timeout):
        _reject(command, {"code": error.code, "message": error.message})
        return False
    return True


def _reject(command, error):
    command.status, command.acknowledged_at = "REJECTED", now()
    command.result = {"error": error}


def _expired_error():
    return {"code": "ASSET_PURGE_DELIVERY_EXPIRED", "message": "Agent 未在有效期内收到请求，请重新提交"}


def _expire_undelivered(command):
    if _status(command) == "FAILED" and command.status == "PENDING":
        _reject(command, _expired_error())


def acknowledge(session, command, status, result):
    """Only a matching authenticated Agent result can release a delivered purge."""
    from .service import apply_ack

    if command.delivered_at is None:
        if command.status == "REJECTED":
            return
        raise APIError(409, "ASSET_PURGE_NOT_DELIVERED", "资产删除命令尚未下发")
    if status == "APPLIED":
        allowed = {"READY", "MISSING"} if command.type == INSPECT else FINAL
        if (result.get("asset_id") == command.payload["asset_id"] and result.get("path") == command.payload["path"]
                and isinstance(result.get("status"), str) and result["status"] in allowed):
            stored = {"status": result["status"]}
            if command.type == INSPECT:
                identity = result.get("identity")
                if not ((stored["status"] == "MISSING" and identity is None)
                        or (stored["status"] == "READY" and isinstance(identity, str) and HEX.fullmatch(identity))):
                    raise APIError(422, "ASSET_PURGE_INVALID_RESULT", "Agent 返回的文件身份无效")
                stored["identity"] = identity
            apply_ack(session, command, status, stored)
            return
        raise APIError(422, "ASSET_PURGE_INVALID_RESULT", "Agent 回执与待删除的资产或路径不一致")
    error = result.get("error")
    if not isinstance(error, dict) or not isinstance(error.get("code"), str) or not ERROR_CODE.fullmatch(error["code"]):
        raise APIError(422, "ASSET_PURGE_INVALID_RESULT", "Agent 返回的错误回执无效")
    message = error.get("message")
    if isinstance(message, str):
        message = " ".join(message.split())[:600]
    stored = {"error": {"code": error["code"], "message": message or "Agent 未能完成这次操作，请检查机器上的删除记录"}}
    if result.get("staged") is True:
        stored["staged"] = True
    apply_ack(session, command, "REJECTED", stored)


def _load(session, asset_id):
    asset = session.scalar(select(Asset).where(Asset.id == asset_id).with_for_update())
    if asset is None or asset.state != "ARCHIVED":
        raise APIError(404, "ASSET_DELETION_NOT_FOUND", "找不到已删除的资产登记")
    node = session.scalar(select(Node).where(Node.id == asset.node_id).with_for_update())
    record = session.get(AssetDeletion, asset.id)
    if record is None:
        record = AssetDeletion(asset_id=asset.id)
        session.add(record)
    return asset, node, record


def register_routes(app, dependencies, db, settings):
    @app.get("/api/v1/assets/deletions", dependencies=dependencies)
    def list_deletions():
        inspection, purge = aliased(Command), aliased(Command)
        with db.read() as session:
            query = (select(Asset, Node, AssetDeletion, User.username, inspection, purge)
                     .join(Node, Node.id == Asset.node_id)
                     .outerjoin(AssetDeletion, AssetDeletion.asset_id == Asset.id)
                     .outerjoin(User, User.id == AssetDeletion.archived_by_user_id)
                     .outerjoin(inspection, inspection.id == AssetDeletion.inspection_command_id)
                     .outerjoin(purge, purge.id == AssetDeletion.purge_command_id)
                     .where(Asset.state == "ARCHIVED")
                     .order_by(AssetDeletion.archived_at.desc().nulls_last(), Asset.created_at.desc(), Asset.id))
            rows = []
            for asset, node, deletion, actor, check, command in session.execute(query):
                node_error = _node_error(node, settings.heartbeat_timeout_seconds)
                status = _status(command)
                blocked = "文件清理状态待核对，请由管理员检查机器上的暂存目录" if _recovery_required(command) else None
                blocked = blocked or (node_error.message if node_error else None)
                rows.append({"id": asset.id, "name": asset.name, "kind": asset.kind, "node_id": node.id,
                             "node_name": node.effective_name, "path": asset.path,
                             "archived_at": deletion.archived_at if deletion else None, "archived_by": actor,
                             "inspection": _command_view(check), "file_deletion": _command_view(command),
                             "requires_new_inspection": _needs_new_inspection(command),
                             "can_purge": blocked is None and status not in FINAL | {"PENDING", "RUNNING"},
                             "purge_block_reason": blocked})
            return rows

    @app.post("/api/v1/assets/{asset_id}/purge-inspections", dependencies=dependencies, status_code=202)
    def inspect_files(asset_id: str, request: Request):
        with db.write() as session:
            require_active_actor(session, request, PERMISSION)
            asset, node, record = _load(session, asset_id)
            purge = session.get(Command, record.purge_command_id) if record.purge_command_id else None
            if purge is not None and not _needs_new_inspection(purge):
                raise APIError(409, "ASSET_PURGE_ALREADY_CONFIRMED", "文件删除已确认，重试会使用原文件检查记录")
            if record.inspection_command_id:
                previous = session.get(Command, record.inspection_command_id)
                if previous is not None:
                    if _status(previous) in {"PENDING", "RUNNING"}:
                        return _command_view(previous)
                    _expire_undelivered(previous)
            _, protected_paths = _require_target(session, asset, node, settings.heartbeat_timeout_seconds)
            require_no_file_purge(session, [node.id])
            command = _enqueue(session, asset, INSPECT, request, protected_paths=protected_paths)
            record.inspection_command_id = command.id
            return _command_view(command)

    @app.post("/api/v1/assets/{asset_id}/purge", dependencies=dependencies, status_code=202)
    def purge_files(asset_id: str, body: PurgeInput, request: Request):
        with db.write() as session:
            require_active_actor(session, request, PERMISSION)
            if request_user_id(request) is None:
                raise APIError(403, "ASSET_PURGE_ADMIN_SESSION_REQUIRED", "请使用管理员账户登录后删除机器文件")
            asset, node, record = _load(session, asset_id)
            if body.confirm_path != asset.path:
                raise APIError(422, "ASSET_PURGE_CONFIRMATION_MISMATCH", "输入的路径与待删除资产的完整路径不一致")
            if body.inspection_id != record.inspection_command_id:
                raise APIError(409, "ASSET_PURGE_INSPECTION_CHANGED", "文件检查记录已变化，请刷新台账后确认")
            inspection = session.get(Command, body.inspection_id)
            if (inspection is None or inspection.type != INSPECT or inspection.status != "APPLIED"
                    or inspection.node_id != node.id or inspection.payload.get("asset_id") != asset.id
                    or inspection.payload.get("path") != asset.path):
                raise APIError(409, "ASSET_PURGE_INSPECTION_REQUIRED", "请先等待 Agent 完成文件检查")
            if record.purge_command_id:
                previous = session.get(Command, record.purge_command_id)
                if _status(previous) in FINAL | {"PENDING", "RUNNING"}:
                    return _command_view(previous)
                if _recovery_required(previous):
                    raise APIError(409, "ASSET_PURGE_INDETERMINATE", "文件清理状态待核对，请检查机器上的暂存目录")
                if _needs_new_inspection(previous) and inspection.created_at <= previous.created_at:
                    raise APIError(409, "ASSET_PURGE_INSPECTION_CHANGED", "文件已变化，请重新检查路径后确认")
                _expire_undelivered(previous)
            node_ids, protected_paths = _require_target(session, asset, node, settings.heartbeat_timeout_seconds)
            _require_idle(session, node_ids)
            command = _enqueue(session, asset, PURGE, request, identity=inspection.result["identity"],
                               protected_paths=protected_paths)
            record.purge_command_id = command.id
            return _command_view(command)
