"""Read-only dataset metadata queries through the owning machine's Agent."""

import posixpath

from fastapi.responses import JSONResponse
from pydantic import Field
from sqlalchemy import select
from vla_protocol.dataset_inspection import normalize_dataset_inspection

from .common import APIError, now, offset_time, require_path, uid
from .models import Command
from .schemas import InputModel


QUERY_TTL_SECONDS = 20
CACHE_SECONDS = 10
MAX_PENDING = 10
COMMAND = "DATASET_INSPECT"
ERRORS = {
    "DATASET_INSPECT_TIMEOUT": "读取数据集元信息超时，可重试或继续手动填写",
    "DATASET_INSPECT_AGENT_UNSUPPORTED": "当前 Agent 尚不支持读取数据集字段，请更新 Agent 后重试",
    "DATASET_INSPECT_NOT_FOUND": "目录中未找到 LeRobot meta/info.json，请核对所选数据集目录",
    "DATASET_INSPECT_PERMISSION_DENIED": "Agent 无权读取该数据集元信息，请检查目录权限",
    "DATASET_INSPECT_INVALID_METADATA": "数据集元信息格式无效，无法可靠推断，请核对 meta/info.json",
    "DATASET_INSPECT_LIMIT": "数据集元信息超出安全读取范围，可继续手动填写",
    "DATASET_INSPECT_INVALID_RESULT": "Agent 返回的数据集字段结果无效，请更新后重试",
    "DATASET_INSPECT_FAILED": "无法读取数据集元信息，可重试或继续手动填写",
    "PATH_OUTSIDE_ROOTS": "该数据集路径不在机器允许访问的目录内，或存在逃逸链接",
}


class InspectInput(InputModel):
    path: str = Field(min_length=1, max_length=4096)


def _error(code):
    return {"code": code, "message": ERRORS[code]}


def validated_path(path, roots):
    if (not isinstance(path, str) or not 1 <= len(path) <= 4096
            or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in path)):
        raise APIError(422, "INVALID_DATASET_PATH", "数据集路径不能包含控制字符")
    require_path(path, roots)
    return "/" + posixpath.normpath(path).lstrip("/")


def expire_pending(session, node_id=None):
    query = select(Command).where(Command.type == COMMAND, Command.status == "PENDING")
    if node_id is not None:
        query = query.where(Command.node_id == node_id)
    timestamp = now()
    for command in session.scalars(query.with_for_update()):
        expiry = command.payload.get("expires_at")
        if not isinstance(expiry, str) or expiry <= timestamp:
            command.status = "REJECTED"
            command.result = {"error": _error("DATASET_INSPECT_TIMEOUT")}
            command.acknowledged_at = timestamp
    session.flush()


def validated_result(command, roots, value):
    try:
        if validated_path(command.payload["path"], roots) != command.payload["path"]:
            return {"error": _error("PATH_OUTSIDE_ROOTS")}
    except (APIError, ValueError, TypeError, KeyError):
        return {"error": _error("PATH_OUTSIDE_ROOTS")}
    if not isinstance(value, dict):
        return {"error": _error("DATASET_INSPECT_INVALID_RESULT")}
    if value.get("error") is not None:
        supplied = value["error"].get("code") if isinstance(value["error"], dict) else None
        code = {"COMMAND_UNSUPPORTED": "DATASET_INSPECT_AGENT_UNSUPPORTED",
                "DATASET_INSPECT_EXPIRED": "DATASET_INSPECT_TIMEOUT"}.get(supplied, supplied) if isinstance(supplied, str) else None
        return {"error": _error(code if code in ERRORS else "DATASET_INSPECT_FAILED")}
    try:
        return {"inspection": normalize_dataset_inspection(value, expected_path=command.payload["path"])}
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError, UnicodeError):
        return {"error": _error("DATASET_INSPECT_INVALID_RESULT")}


def query_dict(command, roots):
    result = {"id": command.id, "node_id": command.node_id, "path": command.payload["path"],
              "status": "PENDING", "inspection": None, "error": None}
    if command.status == "PENDING":
        try:
            validated_path(command.payload["path"], roots)
        except APIError:
            result.update(status="FAILED", error=_error("PATH_OUTSIDE_ROOTS"))
        return result
    clean = validated_result(command, roots, command.result)
    if command.status != "APPLIED" or "error" in clean:
        result.update(status="FAILED", error=clean.get("error") or _error("DATASET_INSPECT_FAILED"))
    else:
        result.update(status="READY", inspection=clean["inspection"])
    return result


def create_query(session, node, path):
    path = validated_path(path, node.roots)
    expire_pending(session, node.id)
    recent = select(Command).where(Command.node_id == node.id, Command.type == COMMAND,
                                   Command.created_at >= offset_time(-CACHE_SECONDS)).order_by(Command.created_at.desc())
    for candidate in session.scalars(recent):
        if candidate.payload.get("path") == path:
            existing = query_dict(candidate, node.roots)
            if existing["status"] in {"PENDING", "READY"}:
                return existing
    pending = select(Command.id).where(Command.node_id == node.id, Command.type == COMMAND,
                                       Command.status == "PENDING").limit(MAX_PENDING)
    if len(list(session.scalars(pending))) >= MAX_PENDING:
        raise APIError(429, "DATASET_INSPECT_BUSY", "当前读取请求较多，请稍后重试")
    command = Command(id=uid(), node_id=node.id, type=COMMAND, status="PENDING", created_at=now(),
                      payload={"path": path, "expires_at": offset_time(QUERY_TTL_SECONDS)})
    session.add(command)
    return query_dict(command, node.roots)


def acknowledge(session, command, node, status, value):
    from .service import apply_ack

    expire_pending(session, node.id)
    if command.status != "PENDING":
        return  # A timeout or the first terminal result is immutable on retries.
    clean = validated_result(command, node.roots, value)
    if status == "REJECTED" and "error" not in clean:
        clean = {"error": _error("DATASET_INSPECT_FAILED")}
    # Persist only the protocol whitelist; suggestions never mutate an asset.
    stored = {"error": clean["error"]} if "error" in clean else clean["inspection"]
    apply_ack(session, command, "REJECTED" if "error" in clean else "APPLIED", stored)


def register_routes(app, admin, db, require_available_node, settings):
    def response(value, status=200):
        return JSONResponse(status_code=status, content=value, headers={"Cache-Control": "no-store"})

    def available(session, node_id):
        node = require_available_node(session, node_id)
        if node.status != "ONLINE" or node.last_seen < offset_time(-settings.heartbeat_timeout_seconds):
            raise APIError(409, "NODE_OFFLINE", "机器尚未在线，无法读取磁盘中的数据集字段；仍可手动填写")
        return node

    @app.post("/api/v1/nodes/{node_id}/dataset-inspections", dependencies=admin, status_code=202)
    def create(node_id: str, body: InspectInput):
        with db.write() as session:
            node = available(session, node_id)
            return response(create_query(session, node, body.path), 202)

    @app.get("/api/v1/nodes/{node_id}/dataset-inspections/{query_id}", dependencies=admin)
    def get(node_id: str, query_id: str):
        with db.write() as session:
            node = require_available_node(session, node_id)
            command = session.get(Command, query_id)
            if command is None or command.node_id != node.id or command.type != COMMAND:
                raise APIError(404, "NOT_FOUND", "Dataset metadata query not found")
            expire_pending(session, node.id)
            return response(query_dict(command, node.roots))
