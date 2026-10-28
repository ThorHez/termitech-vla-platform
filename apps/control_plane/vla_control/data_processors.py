"""Bounded, machine-local processor catalogs; discovery never grants acceptance."""

import json
from pathlib import PurePosixPath
import re

from fastapi.responses import JSONResponse
from pydantic import Field
from sqlalchemy import select

from .common import APIError, now, offset_time, uid, under_root
from .models import Command, Profile
from .recipe_details import _hex, _literal, _safe_text
from .recipe_suggestions import matching_profile, profile_digest, require_profile, SOURCES
from .schemas import InputModel


QUERY_TTL_SECONDS = 60
CACHE_SECONDS = 30
MAX_PENDING = 5
ERRORS = {
    "DATA_PROCESSORS_TIMEOUT": "读取数据处理器超时，请检查训练环境后重试",
    "AGENT_DATA_PROCESSORS_UNSUPPORTED": "当前 Agent 不支持独立数据处理器，请更新 Agent",
    "DATA_PROCESSORS_PROFILE_CHANGED": "训练环境已变化，请重新读取数据处理器",
    "DATA_PROCESSORS_INVALID_RESULT": "Agent 返回的数据处理器目录无效，请重试",
    "DATA_PROCESSORS_FAILED": "无法从训练环境读取数据处理器，请检查本机环境",
    "DATA_PROCESSORS_OUTPUT_LIMIT": "数据处理器目录超过读取上限，请检查训练环境",
    "PROFILE_UNAVAILABLE": "该训练环境当前不可用，请重新选择训练环境",
    "RUNTIME_UNAVAILABLE": "训练 Python 当前不可执行，请检查训练环境",
    "RECIPE_SOURCE_UNAVAILABLE": "训练环境缺少可读取的 OpenPI config.py",
    "RECIPE_SOURCE_CHANGED": "读取期间训练代码发生变化，请重新读取数据处理器",
    "CODE_VERSION_UNAVAILABLE": "无法确认训练代码版本，请检查本机 Git 仓库",
    "PATH_OUTSIDE_ROOTS": "训练代码不在机器允许访问的目录内",
}
TYPES = {"boolean", "integer", "number", "string", "array", "object", "null"}


class CatalogInput(InputModel):
    profile_id: str = Field(min_length=1, max_length=200)


def _error(code):
    return {"code": code, "message": ERRORS[code]}


def _source(value, code_dir=None, *, pinned=False):
    if not isinstance(value, dict):
        return None
    path = value.get("path")
    if (not _safe_text(path, 4096) or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts
            or not path.endswith(".py") or PurePosixPath(path).as_posix() != path
            or not _hex(value.get("sha256"), {64})):
        return None
    if not pinned:
        return {"path": path, "sha256": value["sha256"]}
    if (path not in SOURCES or value.get("code_dir") != code_dir
            or not _hex(value.get("commit_sha"), {40, 64}) or not _hex(value.get("changes_digest"), {64})
            or not isinstance(value.get("dirty"), bool)):
        return None
    return {key: value[key] for key in ("path", "sha256", "code_dir", "commit_sha", "changes_digest", "dirty")}


def _parameter_path(value):
    return (isinstance(value, str) and len(value) <= 256
            and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*", value) is not None
            and all(not part.startswith("_") for part in value.split(".")))


def _entry(value):
    from vla_adapters.data_processors import normalize_processor

    if (not isinstance(value, dict) or not isinstance(value.get("name"), str)
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,199}", value["name"]) is None
            or not _safe_text(value.get("description"), 4096, multiline=True)
            or not isinstance(value.get("available"), bool)):
        raise ValueError("invalid processor identity")
    parameters, unsupported = value.get("parameters"), value.get("unsupported_parameters")
    if not isinstance(parameters, list) or len(parameters) > 128 or not isinstance(unsupported, list) or len(unsupported) > 128:
        raise ValueError("invalid processor parameters")
    clean, paths = [], set()
    for parameter in parameters:
        if (not isinstance(parameter, dict) or not _parameter_path(parameter.get("path"))
                or parameter["path"] in paths or parameter.get("type") not in TYPES
                or not isinstance(parameter.get("required"), bool)):
            raise ValueError("invalid parameter descriptor")
        # Required fields without defaults still pass the shared path policy.
        normalize_processor({"name": value["name"], "parameters": {parameter["path"]: parameter.get("default")}})
        row = {key: parameter[key] for key in ("path", "type", "required")}
        for key in ("default", "choices", "items", "value_type", "values", "container"):
            if key in parameter:
                if not _literal(parameter[key]) or len(json.dumps(parameter[key], ensure_ascii=True)) > 8192:
                    raise ValueError("invalid parameter literal")
                row[key] = parameter[key]
        if "nullable" in parameter:
            if not isinstance(parameter["nullable"], bool):
                raise ValueError("invalid nullable marker")
            row["nullable"] = parameter["nullable"]
        if "choices" in row and (not isinstance(row["choices"], list) or len(row["choices"]) > 64):
            raise ValueError("invalid choices")
        clean.append(row)
        paths.add(row["path"])
    excluded = []
    for parameter in unsupported:
        if (not isinstance(parameter, dict) or not _parameter_path(parameter.get("path"))
                or not _safe_text(parameter.get("reason"), 512, multiline=True)
                or not isinstance(parameter.get("required"), bool)):
            raise ValueError("invalid unsupported descriptor")
        excluded.append({key: parameter[key] for key in ("path", "reason", "required")})
    result = {"name": value["name"], "description": value["description"], "parameters": clean,
              "unsupported_parameters": excluded, "available": value["available"]}
    if "reason" in value:
        if not _safe_text(value["reason"], 2048, multiline=True):
            raise ValueError("invalid processor availability reason")
        result["reason"] = value["reason"]
    if "source" in value:
        source = _source(value["source"])
        if source is None:
            raise ValueError("invalid processor source")
        result["source"] = source
    if "dataset_contract" in value:
        from vla_adapters.dataset_contracts import normalize_dataset_contract
        if not value["available"] or "source" not in result:
            raise ValueError("dataset contract requires an available traceable processor")
        result["dataset_contract"] = normalize_dataset_contract(value["dataset_contract"])
    if "composition_schema_version" in value:
        if (value["name"] != "ComposableDataConfig" or type(value["composition_schema_version"]) is not int
                or value["composition_schema_version"] != 1 or not value["available"] or "source" not in result
                or clean or excluded):
            raise ValueError("invalid composition capability")
        result["composition_schema_version"] = 1
    return result


def validated_result(command, roots, result):
    invalid = {"error": _error("DATA_PROCESSORS_INVALID_RESULT")}
    if not isinstance(result, dict):
        return invalid
    if result.get("error") is not None:
        error = result["error"]
        code = error.get("code") if isinstance(error, dict) else None
        code = {"COMMAND_UNSUPPORTED": "AGENT_DATA_PROCESSORS_UNSUPPORTED",
                "DATA_PROCESSORS_EXPIRED": "DATA_PROCESSORS_TIMEOUT"}.get(code, code)
        return {"error": _error(code if isinstance(code, str) and code in ERRORS else "DATA_PROCESSORS_FAILED")}
    code_dir = command.payload.get("code_dir")
    if (result.get("profile_id") != command.payload["profile_id"] or not _safe_text(code_dir, 4096)
            or not code_dir.startswith("/") or ".." in PurePosixPath(code_dir).parts or not under_root(code_dir, roots)):
        return invalid
    source = _source(result.get("source"), code_dir, pinned=True)
    entries = result.get("entries")
    if source is None or not isinstance(entries, list) or len(entries) > 128:
        return invalid
    try:
        clean = [_entry(entry) for entry in entries]
        if len({entry["name"] for entry in clean}) != len(clean) or len(json.dumps(clean, ensure_ascii=True)) > 256 * 1024:
            return invalid
    except (TypeError, ValueError, RecursionError):
        return invalid
    return {"profile_id": command.payload["profile_id"], "entries": clean, "source": source}


def expire_pending(session, node_id=None):
    query = select(Command).where(Command.type == "DATA_PROCESSORS", Command.status == "PENDING")
    if node_id is not None:
        query = query.where(Command.node_id == node_id)
    timestamp = now()
    for command in session.scalars(query.with_for_update()):
        expiry = command.payload.get("expires_at")
        if not isinstance(expiry, str) or expiry <= timestamp:
            command.status, command.result, command.acknowledged_at = "REJECTED", {"error": _error("DATA_PROCESSORS_TIMEOUT")}, timestamp
    session.flush()


def query_dict(command, profile, roots):
    result = {"id": command.id, "node_id": command.node_id, "profile_id": command.payload["profile_id"],
              "status": "PENDING", "catalog": None, "error": None}
    if not matching_profile(command, profile, roots):
        result.update(status="FAILED", error=_error("DATA_PROCESSORS_PROFILE_CHANGED"))
    elif command.status != "PENDING":
        clean = validated_result(command, roots, command.result)
        if command.status != "APPLIED" or "error" in clean:
            result.update(status="FAILED", error=clean.get("error") or _error("DATA_PROCESSORS_FAILED"))
        else:
            result.update(status="READY", catalog=clean)
    return result


def create_query(session, node, profile_id):
    profile = require_profile(session, node.id, profile_id)
    digest = profile_digest(profile, node.roots)
    expire_pending(session, node.id)
    recent = select(Command).where(Command.node_id == node.id, Command.type == "DATA_PROCESSORS",
                                   Command.created_at >= offset_time(-CACHE_SECONDS), Command.status.in_({"PENDING", "APPLIED"}))
    for command in session.scalars(recent.order_by(Command.created_at.desc())):
        if command.payload.get("profile_digest") == digest:
            cached = query_dict(command, profile, node.roots)
            if cached["status"] in {"PENDING", "READY"}:
                return cached
    pending = select(Command.id).where(Command.node_id == node.id, Command.type == "DATA_PROCESSORS", Command.status == "PENDING").limit(MAX_PENDING)
    if len(list(session.scalars(pending))) >= MAX_PENDING:
        raise APIError(429, "DATA_PROCESSORS_BUSY", "数据处理器读取请求较多，请稍后重试")
    command = Command(id=uid(), node_id=node.id, type="DATA_PROCESSORS", status="PENDING", created_at=now(),
                      payload={"profile_id": profile_id, "profile_digest": digest, "code_dir": profile.code_dir,
                               "expires_at": offset_time(QUERY_TTL_SECONDS)})
    session.add(command)
    return query_dict(command, profile, node.roots)


def acknowledge(session, command, node, status, result):
    from .service import apply_ack

    expire_pending(session, node.id)
    profile = session.get(Profile, (node.id, command.payload["profile_id"]))
    matches = matching_profile(command, profile, node.roots)
    prior = command.result.get("error") if isinstance(command.result, dict) else None
    if command.status != "PENDING" and (not matches or isinstance(prior, dict) and prior.get("code") in {
            "DATA_PROCESSORS_TIMEOUT", "DATA_PROCESSORS_PROFILE_CHANGED"}):
        return
    clean = validated_result(command, node.roots, result) if matches else {"error": _error("DATA_PROCESSORS_PROFILE_CHANGED")}
    if status == "REJECTED" and "error" not in clean:
        clean = {"error": _error("DATA_PROCESSORS_FAILED")}
    apply_ack(session, command, "REJECTED" if "error" in clean else "APPLIED", clean)


def register_routes(app, admin, db, require_available_node, settings):
    def response(value, status=200):
        return JSONResponse(status_code=status, content=value, headers={"Cache-Control": "no-store"})

    def available(session, node_id):
        node = require_available_node(session, node_id)
        if node.last_seen < offset_time(-settings.heartbeat_timeout_seconds):
            raise APIError(409, "NODE_OFFLINE", "机器心跳已过期，暂时无法读取数据处理器")
        return node

    @app.post("/api/v1/nodes/{node_id}/data-processors", dependencies=admin, status_code=202)
    def create(node_id: str, body: CatalogInput):
        with db.write() as session:
            node = available(session, node_id)
            return response(create_query(session, node, body.profile_id), 202)

    @app.get("/api/v1/nodes/{node_id}/data-processors/{query_id}", dependencies=admin)
    def get(node_id: str, query_id: str):
        with db.write() as session:
            node = available(session, node_id)
            command = session.get(Command, query_id)
            if command is None or command.node_id != node.id or command.type != "DATA_PROCESSORS":
                raise APIError(404, "NOT_FOUND", "Data processor query not found")
            expire_pending(session, node.id)
            return response(query_dict(command, session.get(Profile, (node.id, command.payload["profile_id"])), node.roots))
