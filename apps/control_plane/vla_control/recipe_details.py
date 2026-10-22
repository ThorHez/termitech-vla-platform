"""Bounded source details for one machine-local, registered OpenPI recipe."""

import json
import math
from pathlib import PurePosixPath
import re

from pydantic import Field
from sqlalchemy import select

from .common import APIError, now, offset_time, uid, under_root
from .models import Command
from .recipe_suggestions import matching_profile, profile_digest, require_profile, SOURCES
from .schemas import InputModel


QUERY_TTL_SECONDS = 20
CACHE_SECONDS = 10
DETAIL_VALID_SECONDS = 300
MAX_PENDING = 10
MAX_FIELDS = 500
WARNINGS = {"DYNAMIC_REGISTRY_UNSUPPORTED", "DISCOVERY_LIMIT_REACHED", "INVALID_RECIPE_NAME",
            "DETAIL_FIELDS_TRUNCATED", "DETAIL_EXPRESSION_TRUNCATED", "UNRESOLVED_DEFAULTS", "DETAIL_TEXT_SANITIZED"}
ERRORS = {
    "RECIPE_DETAIL_TIMEOUT": "读取训练配方详情超时，请稍后重试",
    "AGENT_RECIPE_DETAIL_UNSUPPORTED": "当前 Agent 不支持训练配方详情，请更新 Agent",
    "RECIPE_PROFILE_CHANGED": "训练环境配置已变化，请重新读取训练配方详情",
    "PROFILE_UNAVAILABLE": "该训练环境当前不可用，请重新选择训练环境",
    "RECIPE_SOURCE_UNAVAILABLE": "无法读取该训练环境的配方配置",
    "RECIPE_SOURCE_INVALID": "该训练环境的配方配置无法解析",
    "RECIPE_SOURCE_TOO_LARGE": "配方配置超过读取上限",
    "RECIPE_SOURCE_CHANGED": "读取期间训练代码发生变化，请重新读取详情",
    "CODE_VERSION_UNAVAILABLE": "无法确认该训练环境的代码版本",
    "PATH_OUTSIDE_ROOTS": "训练代码不在机器允许访问的目录内",
    "RECIPE_DETAIL_UNSUPPORTED": "该配方由动态代码生成，当前无法读取完整静态详情",
    "RECIPE_NOT_FOUND": "该代码版本中未找到所选训练配方",
    "RECIPE_DETAIL_INVALID_RESULT": "Agent 返回的训练配方详情无效，请重试",
    "RECIPE_DETAIL_FAILED": "暂时无法读取训练配方详情，请稍后重试",
}


class RecipeDetailInput(InputModel):
    profile_id: str = Field(min_length=1, max_length=200)
    recipe: str = Field(min_length=1, max_length=255, pattern=r"^[A-Za-z0-9_-]+$")


def _error(code):
    return {"code": code, "message": ERRORS[code]}


def _safe_text(value, limit, multiline=False):
    allowed = {9, 10, 13} if multiline else set()
    return (isinstance(value, str) and len(value) <= limit
            and not any((ord(char) < 32 and ord(char) not in allowed) or 127 <= ord(char) < 160 for char in value))


def _hex(value, lengths):
    return isinstance(value, str) and len(value) in lengths and re.fullmatch(r"[0-9a-f]+", value) is not None


def _literal(value, depth=0):
    if depth > 4:
        return False
    if value is None or isinstance(value, (bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, str):
        return _safe_text(value, 4096, multiline=True)
    if isinstance(value, list):
        return len(value) <= 64 and all(_literal(item, depth + 1) for item in value)
    if isinstance(value, dict):
        return len(value) <= 64 and all(_safe_text(key, 128) and _literal(item, depth + 1) for key, item in value.items())
    return False


def expire_pending(session, node_id=None):
    query = select(Command).where(Command.type == "RECIPE_DETAIL", Command.status == "PENDING")
    if node_id is not None:
        query = query.where(Command.node_id == node_id)
    timestamp = now()
    for command in session.scalars(query.with_for_update()):
        expiry = command.payload.get("expires_at")
        if not isinstance(expiry, str) or expiry <= timestamp:
            command.status, command.result, command.acknowledged_at = "REJECTED", {"error": _error("RECIPE_DETAIL_TIMEOUT")}, timestamp
    session.flush()


def validated_result(command, roots, result):
    invalid = {"error": _error("RECIPE_DETAIL_INVALID_RESULT")}
    if not isinstance(result, dict):
        return invalid
    if result.get("error") is not None:
        error = result["error"]
        code = error.get("code") if isinstance(error, dict) else None
        if code == "COMMAND_UNSUPPORTED":
            code = "AGENT_RECIPE_DETAIL_UNSUPPORTED"
        elif code == "RECIPE_QUERY_EXPIRED":
            code = "RECIPE_DETAIL_TIMEOUT"
        return {"error": _error(code if isinstance(code, str) and code in ERRORS else "RECIPE_DETAIL_FAILED")}
    if result.get("profile_id") != command.payload["profile_id"] or result.get("recipe") != command.payload["recipe"]:
        return invalid
    source = result.get("source")
    if not isinstance(source, dict):
        return invalid
    path, code_dir, line, end_line = source.get("path"), source.get("code_dir"), source.get("line"), source.get("end_line")
    if (not isinstance(path, str) or path not in SOURCES or not _safe_text(code_dir, 4096)
            or not code_dir.startswith("/") or ".." in PurePosixPath(code_dir).parts or not under_root(code_dir, roots)
            or type(line) is not int or type(end_line) is not int or not 1 <= line <= end_line <= 1_000_000
            or not _hex(source.get("sha256"), {64}) or not _hex(source.get("commit_sha"), {40, 64})
            or not _hex(source.get("changes_digest"), {64}) or not isinstance(source.get("dirty"), bool)):
        return invalid
    fields, warnings, source_code = result.get("fields"), result.get("warnings"), result.get("source_code")
    if (not isinstance(fields, list) or len(fields) > MAX_FIELDS or not _safe_text(source_code, 32768, multiline=True)
            or not isinstance(warnings, list) or len(warnings) > len(WARNINGS)
            or any(not isinstance(code, str) or code not in WARNINGS for code in warnings)
            or len(set(warnings)) != len(warnings)):
        return invalid
    clean_fields, seen, characters = [], set(), 0
    for field in fields:
        if not isinstance(field, dict):
            return invalid
        name, kind, expression, origin = field.get("path"), field.get("kind"), field.get("expression"), field.get("origin")
        if (not isinstance(name, str) or len(name) > 512
                or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*|\[[0-9]+\])*", name) is None
                or name in seen or kind not in ("literal", "expression") or origin not in ("recipe", "default")
                or not _safe_text(expression, 2048, multiline=True)):
            return invalid
        clean = {"path": name, "kind": kind, "expression": expression, "origin": origin}
        if kind == "literal":
            if "value" not in field or not _literal(field["value"]):
                return invalid
            if len(json.dumps(field["value"], ensure_ascii=False, allow_nan=False)) > 4096:
                return invalid
            clean["value"] = field["value"]
        clean_fields.append(clean)
        seen.add(name)
        characters += len(name) + len(expression)
        if characters > 65536:
            return invalid
    return {"profile_id": command.payload["profile_id"], "recipe": command.payload["recipe"],
            "source": {key: source[key] for key in ("path", "sha256", "line", "end_line", "commit_sha", "code_dir", "changes_digest", "dirty")},
            "fields": clean_fields, "source_code": source_code, "warnings": list(warnings)}


def query_dict(command, profile, roots):
    response = {"id": command.id, "node_id": command.node_id, "profile_id": command.payload["profile_id"],
                "recipe": command.payload["recipe"], "status": "PENDING", "detail": None, "error": None}
    if not matching_profile(command, profile, roots):
        response.update(status="FAILED", error=_error("RECIPE_PROFILE_CHANGED"))
    elif command.status != "PENDING":
        clean = validated_result(command, roots, command.result)
        if command.status != "APPLIED" or "error" in clean:
            response.update(status="FAILED", error=clean.get("error") or _error("RECIPE_DETAIL_FAILED"))
        else:
            response.update(status="READY", detail=clean)
    return response


def create_query(session, node, profile_id, recipe):
    profile = require_profile(session, node.id, profile_id)
    snapshot = profile_digest(profile, node.roots)
    expire_pending(session, node.id)
    recent = select(Command).where(Command.node_id == node.id, Command.type == "RECIPE_DETAIL",
                                   Command.created_at >= offset_time(-CACHE_SECONDS), Command.status.in_({"PENDING", "APPLIED"}))
    for command in session.scalars(recent.order_by(Command.created_at.desc())):
        if command.payload.get("profile_digest") == snapshot and command.payload.get("recipe") == recipe:
            cached = query_dict(command, profile, node.roots)
            if cached["status"] in {"PENDING", "READY"}:
                return cached
    pending = select(Command.id).where(Command.node_id == node.id, Command.type == "RECIPE_DETAIL", Command.status == "PENDING").limit(MAX_PENDING)
    if len(list(session.scalars(pending))) >= MAX_PENDING:
        raise APIError(429, "RECIPE_DETAIL_BUSY", "配方详情读取请求较多，请稍后重试")
    command = Command(id=uid(), node_id=node.id, type="RECIPE_DETAIL", status="PENDING", created_at=now(),
                      payload={"profile_id": profile_id, "recipe": recipe, "profile_digest": snapshot, "expires_at": offset_time(QUERY_TTL_SECONDS)})
    session.add(command)
    return query_dict(command, profile, node.roots)


def acknowledge(session, command, node, status, result):
    from .models import Profile
    from .service import apply_ack

    expire_pending(session, node.id)
    profile = session.get(Profile, (node.id, command.payload["profile_id"]))
    matches = matching_profile(command, profile, node.roots)
    previous_error = command.result.get("error") if isinstance(command.result, dict) else None
    if command.status != "PENDING" and (not matches or isinstance(previous_error, dict) and previous_error.get("code") in {
            "RECIPE_DETAIL_TIMEOUT", "RECIPE_PROFILE_CHANGED"}):
        return
    clean = validated_result(command, node.roots, result) if matches else {"error": _error("RECIPE_PROFILE_CHANGED")}
    if status == "REJECTED" and "error" not in clean:
        clean = {"error": _error("RECIPE_DETAIL_FAILED")}
    apply_ack(session, command, "REJECTED" if "error" in clean else "APPLIED", clean)
