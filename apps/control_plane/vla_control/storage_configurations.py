"""Idle-node storage changes require a matching, durable Agent acknowledgement."""

from copy import deepcopy
from pathlib import PurePosixPath

from fastapi.responses import JSONResponse
from pydantic import Field
from sqlalchemy import or_, select

from .common import APIError, TERMINAL, now, offset_time, require_path, uid, under_root
from .models import Allocation, Attempt, Command, Node, Preview, Run
from .policy_deployment_models import PolicyDeployment, RESOURCE_STATES
from .remote_models import RemoteDeployment
from .schemas import InputModel


COMMAND = "STORAGE_CONFIGURE"
TTL_SECONDS = 120
ERRORS = {
    "STORAGE_CONFIGURE_INVALID": "请填写受管目录内的完整绝对路径",
    "STORAGE_CONFIGURE_CONFLICT": "目录配置已变化，请刷新后重试",
    "STORAGE_CONFIGURE_PENDING": "目录配置正在保存或等待机器确认，请等待完成",
    "STORAGE_CONFIGURE_BUSY": "机器仍有训练、部署或配置操作，请等待完成后修改目录",
    "STORAGE_CONFIGURE_UNSUPPORTED": "当前 Agent 不支持修改存储目录，请先更新 Agent",
    "STORAGE_CONFIGURE_EXPIRED": "目录修改命令已过期，机器未修改配置，请重试",
    "STORAGE_CONFIGURE_UNWRITABLE": "目标目录无法写入，请检查训练用户的目录权限",
    "STORAGE_CONFIGURE_FAILED": "机器未完成目录配置，请检查 Agent 操作记录",
    "STORAGE_CONFIGURE_CONFIRMATION_PENDING": "尚未确认机器上的持久化结果，训练保持锁定，请等待 Agent 回执",
    "STORAGE_CONFIGURE_INVALID_RESULT": "机器回执不完整，训练保持锁定，请检查 Agent 操作记录",
}


class StorageInput(InputModel):
    expected_output_root: str = Field(min_length=1, max_length=4096)
    expected_cache_root: str | None = Field(default=None, min_length=1, max_length=4096)
    output_root: str = Field(min_length=1, max_length=4096)
    cache_root: str | None = Field(default=None, min_length=1, max_length=4096)


def _error(code):
    return {"code": code, "message": ERRORS.get(code, ERRORS["STORAGE_CONFIGURE_FAILED"])}


def _path(value, roots):
    if (not isinstance(value, str) or not value or len(value) > 4096
            or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value)):
        raise APIError(422, "STORAGE_CONFIGURE_INVALID", ERRORS["STORAGE_CONFIGURE_INVALID"])
    try:
        value.encode("utf-8")
        require_path(value, roots)
    except (APIError, UnicodeError) as exc:
        raise APIError(422, "STORAGE_CONFIGURE_INVALID", ERRORS["STORAGE_CONFIGURE_INVALID"]) from exc
    path = str(PurePosixPath(value))
    if path == "/":
        raise APIError(422, "STORAGE_CONFIGURE_INVALID", ERRORS["STORAGE_CONFIGURE_INVALID"])
    return path


def pending(session, node_id):
    return list(session.scalars(select(Command).where(Command.node_id == node_id,
        Command.type == COMMAND, Command.status == "PENDING")))


def require_not_pending(session, node_id):
    if pending(session, node_id):
        raise APIError(409, "STORAGE_CONFIGURE_PENDING", ERRORS["STORAGE_CONFIGURE_PENDING"])


def _in_use(session, node_id):
    if session.scalar(select(Allocation.node_id).where(Allocation.node_id == node_id).limit(1)):
        return True
    if session.scalar(select(Run.id).where(Run.node_id == node_id, Run.state.not_in(TERMINAL)).limit(1)):
        return True
    if session.scalar(select(Attempt.id).where(Attempt.node_id == node_id, Attempt.state.not_in(TERMINAL)).limit(1)):
        return True
    if session.scalar(select(PolicyDeployment.id).where(PolicyDeployment.node_id == node_id,
            PolicyDeployment.state.in_(RESOURCE_STATES)).limit(1)):
        return True
    if session.scalar(select(RemoteDeployment.id).where(RemoteDeployment.node_id == node_id,
            RemoteDeployment.state.not_in({"SUCCEEDED", "FAILED", "INTERRUPTED", "DISCONNECTED", "UNINSTALLED"})).limit(1)):
        return True
    from .checkpoint_transfer_models import CheckpointTransfer, TRANSFER_ACTIVE
    return bool(session.scalar(select(CheckpointTransfer.id).where(
        or_(CheckpointTransfer.source_node_id == node_id, CheckpointTransfer.target_node_id == node_id),
        CheckpointTransfer.state.in_(TRANSFER_ACTIVE)).limit(1)))


def _require_idle(session, node_id, *, except_command=None):
    commands = select(Command.id).where(Command.node_id == node_id, Command.status == "PENDING",
        Command.type.in_({COMMAND, "PROFILE_REBIND", "RECIPE_WRITE", "PREFLIGHT", "START", "CANCEL",
                         "POLICY_DEPLOY_START", "POLICY_DEPLOY_STOP", "POLICY_EXPORT"}))
    if except_command is not None:
        commands = commands.where(Command.id != except_command)
    if _in_use(session, node_id) or session.scalar(commands.limit(1)):
        raise APIError(409, "STORAGE_CONFIGURE_BUSY", ERRORS["STORAGE_CONFIGURE_BUSY"])


def query_dict(command):
    result = {"id": command.id, "node_id": command.node_id, "status": "PENDING", "error": None,
              "requested": {key: command.payload[key] for key in ("output_root", "cache_root")}, "storage": None}
    receipt = command.result if isinstance(command.result, dict) else {}
    if command.status == "APPLIED":
        result.update(status="READY", storage=deepcopy(receipt["storage"]))
    elif command.status != "PENDING":
        result.update(status="FAILED", error=deepcopy(receipt.get("error", _error("STORAGE_CONFIGURE_FAILED"))))
    elif receipt.get("error") or command.payload["expires_at"] < now():
        result["error"] = deepcopy(receipt.get("error", _error("STORAGE_CONFIGURE_CONFIRMATION_PENDING")))
    return result


def create_query(session, node, body):
    data = body.model_dump()
    for key in data:
        if data[key] is not None:
            data[key] = _path(data[key], node.roots)
    current = node.output_root, node.cache_root
    target = data["output_root"], data["cache_root"]
    commands = session.scalars(select(Command).where(Command.node_id == node.id, Command.type == COMMAND)
        .order_by(Command.created_at.desc(), Command.id.desc()))
    for command in commands:
        if (all(command.payload.get(key) == value for key, value in data.items())
                and (command.status == "PENDING" or command.status == "APPLIED" and current == target
                     and not pending(session, node.id))):
            return query_dict(command)
    require_not_pending(session, node.id)
    if current != (data["expected_output_root"], data["expected_cache_root"]):
        raise APIError(409, "STORAGE_CONFIGURE_CONFLICT", ERRORS["STORAGE_CONFIGURE_CONFLICT"])
    capabilities = (node.metrics or {}).get("capabilities")
    if isinstance(capabilities, list) and COMMAND not in capabilities:
        raise APIError(409, "STORAGE_CONFIGURE_UNSUPPORTED", ERRORS["STORAGE_CONFIGURE_UNSUPPORTED"])
    _require_idle(session, node.id)
    command = Command(id=uid(), node_id=node.id, type=COMMAND, status="PENDING", created_at=now(),
                      payload={**data, "expires_at": offset_time(TTL_SECONDS)})
    session.add(command)
    session.flush()
    return query_dict(command)


def _directory(value, path):
    if path is None:
        return None if value is None else False
    if (not isinstance(value, dict) or value.get("path") != path
            or any(type(value.get(key)) is not int or not 0 <= value[key] <= 2**63 - 1
                   for key in ("free_bytes", "total_bytes"))
            or value["free_bytes"] > value["total_bytes"]):
        return False
    return {key: value[key] for key in ("path", "free_bytes", "total_bytes")}


def _receipt(result, payload, status):
    if not isinstance(result, dict):
        return None
    error = result.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        if code == "COMMAND_UNSUPPORTED":
            code = "STORAGE_CONFIGURE_UNSUPPORTED"
        if error.get("persisted") is False or code == "STORAGE_CONFIGURE_UNSUPPORTED":
            code = code if isinstance(code, str) and code in ERRORS else "STORAGE_CONFIGURE_FAILED"
            return {"persisted": False, "error": _error(code)}
        return None
    directories = result.get("directories")
    if (status != "APPLIED" or result.get("persisted") is not True
            or not {"output_root", "cache_root"} <= result.keys()
            or any(result.get(key) != payload[key] for key in ("output_root", "cache_root"))
            or not isinstance(directories, dict) or set(directories) != {"output", "cache"}):
        return None
    output = _directory(directories["output"], payload["output_root"])
    cache = _directory(directories["cache"], payload["cache_root"])
    if output is False or cache is False:
        return None
    return {"persisted": True, "output_root": payload["output_root"], "cache_root": payload["cache_root"],
            "directories": {"output": output, "cache": cache}}


def acknowledge(session, command, node, status, result):
    clean = _receipt(result, command.payload, status)
    if command.status != "PENDING":
        if clean != (command.result or {}).get("agent_result"):
            raise APIError(409, "ACK_CONFLICT", "This command already has a different acknowledgement")
        return
    if clean is None:
        command.result = {"error": _error("STORAGE_CONFIGURE_INVALID_RESULT")}
        return
    if clean.get("error"):
        command.status, command.acknowledged_at = "REJECTED", now()
        command.result = {"agent_result": clean, "error": clean["error"]}
        return
    if (node.output_root, node.cache_root) != (command.payload["expected_output_root"], command.payload["expected_cache_root"]):
        command.result = {"error": _error("STORAGE_CONFIGURE_CONFIRMATION_PENDING")}
        return
    try:
        _require_idle(session, node.id, except_command=command.id)
    except APIError:
        command.result = {"error": _error("STORAGE_CONFIGURE_CONFIRMATION_PENDING")}
        return
    node.output_root, node.cache_root = clean["output_root"], clean["cache_root"]
    for preview in session.scalars(select(Preview).where(Preview.node_id == node.id,
            Preview.status.in_({"PENDING", "READY"})).with_for_update()):
        preview.status, preview.error = "FAILED", {"code": "STORAGE_CHANGED_SINCE_PREFLIGHT", "message": "训练存储目录已修改，请重新预检"}
    storage = {key: clean[key] for key in ("output_root", "cache_root", "directories")}
    command.status, command.acknowledged_at = "APPLIED", now()
    command.result = {"agent_result": clean, "storage": storage}
    session.flush()


def snapshot_error(session, node_id, evidence):
    """Old Agent proofs cannot authorize a node after a storage configuration change."""
    node = session.get(Node, node_id)
    if node is None:
        return {"code": "NODE_OFFLINE", "message": "机器不存在，请刷新后重新预检"}
    configured = node.cache_root is not None or session.scalar(select(Command.id).where(
        Command.node_id == node_id, Command.type == COMMAND, Command.status == "APPLIED").limit(1)) is not None
    storage = evidence.get("storage_configuration") if isinstance(evidence, dict) else None
    if storage is None:
        if configured:
            return {"code": "AGENT_STORAGE_CONFIGURATION_UNSUPPORTED", "message": "Agent 未确认当前存储目录，请更新 Agent 后重新预检"}
        return None
    if (not isinstance(storage, dict) or set(storage) != {"output_root", "cache_root"}
            or storage != {"output_root": node.output_root, "cache_root": node.cache_root}):
        return {"code": "STORAGE_CHANGED_SINCE_PREFLIGHT", "message": "Agent 实际存储目录与当前配置不一致，请重新预检"}
    if node.cache_root is not None:
        plan = evidence.get("plan")
        cache = plan.get("training_cache") if isinstance(plan, dict) else None
        paths = cache.get("paths") if isinstance(cache, dict) else None
        if (not isinstance(cache, dict) or not isinstance(cache.get("datasets_cache"), str)
                or cache["datasets_cache"] != str(PurePosixPath(node.cache_root) / "huggingface/datasets")
                or not isinstance(paths, dict) or not paths
                or any(not isinstance(path, str) or not under_root(path, [node.cache_root]) for path in paths.values())):
            return {"code": "STORAGE_CHANGED_SINCE_PREFLIGHT", "message": "Agent 实际训练缓存未使用当前缓存目录，请重新预检"}
    return None


def register_routes(app, admin, db, require_available_node, settings):
    def response(value, status=200):
        return JSONResponse(content=value, status_code=status, headers={"Cache-Control": "no-store"})

    @app.post("/api/v1/nodes/{node_id}/storage-configurations", dependencies=admin, status_code=202)
    def create(node_id: str, body: StorageInput):
        with db.write() as session:
            node = require_available_node(session, node_id)
            if node.last_seen < offset_time(-settings.heartbeat_timeout_seconds):
                raise APIError(409, "NODE_OFFLINE", "机器心跳已过期，暂时无法修改目录")
            return response(create_query(session, node, body), 202)

    @app.get("/api/v1/nodes/{node_id}/storage-configurations", dependencies=admin)
    def latest(node_id: str):
        with db.read() as session:
            if session.get(Node, node_id) is None:
                raise APIError(404, "NOT_FOUND", "Node not found")
            command = session.scalar(select(Command).where(Command.node_id == node_id, Command.type == COMMAND)
                .order_by(Command.created_at.desc(), Command.id.desc()).limit(1))
            return response({"operation": query_dict(command) if command else None})

    @app.get("/api/v1/nodes/{node_id}/storage-configurations/{operation_id}", dependencies=admin)
    def get(node_id: str, operation_id: str):
        with db.read() as session:
            command = session.get(Command, operation_id)
            if command is None or command.node_id != node_id or command.type != COMMAND:
                raise APIError(404, "NOT_FOUND", "Storage configuration operation not found")
            return response(query_dict(command))
