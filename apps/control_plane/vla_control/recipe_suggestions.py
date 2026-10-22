"""Read-only recipe discovery over the machine-local Agent command queue."""

import re

from sqlalchemy import select
from vla_adapters import resolve_recipe
from vla_protocol.recipe_search import recipe_matches, recipe_sort_key

from .common import APIError, digest, now, offset_time, uid
from .models import Command, Profile
from .serialization import profile_dict


QUERY_TTL_SECONDS = 15
CACHE_SECONDS = 10
MAX_PENDING = 20
MAX_ENTRIES = 50
MATCH_MODE = "contains-v1"
SOURCES = {"src/openpi/training/config.py", "openpi/training/config.py"}
WARNINGS = {"DYNAMIC_REGISTRY_UNSUPPORTED", "INVALID_RECIPE_NAME", "DISCOVERY_LIMIT_REACHED"}
ERRORS = {
    "RECIPE_SUGGEST_TIMEOUT": "训练配方联想超时，可手动填写或稍后重试",
    "AGENT_RECIPE_SUGGEST_UNSUPPORTED": "当前 Agent 不支持训练配方联想，请更新 Agent；仍可手动填写",
    "RECIPE_PROFILE_CHANGED": "训练环境配置已变化，请重新获取训练配方",
    "PROFILE_UNAVAILABLE": "该训练环境当前不可用，请重新选择训练环境",
    "RECIPE_SOURCE_UNAVAILABLE": "无法读取该训练环境的配方配置，可手动填写",
    "RECIPE_SOURCE_INVALID": "该训练环境的配方配置无法解析，可手动填写",
    "RECIPE_SOURCE_TOO_LARGE": "配方配置超过读取上限，可手动填写",
    "PATH_OUTSIDE_ROOTS": "训练代码不在机器允许访问的目录内",
    "RECIPE_SUGGEST_INVALID_RESULT": "Agent 返回的训练配方联想结果无效，可手动填写",
    "RECIPE_SUGGEST_FAILED": "暂时无法获取训练配方，可手动填写",
}


def _valid_name(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,255}", value) is not None


def _error(code: str) -> dict:
    return {"code": code, "message": ERRORS[code]}


def profile_digest(profile: Profile, roots: list[str]) -> str:
    # Include the trust metadata: old discovery results must not outlive a
    # profile edit, deactivation, or a change of machine-local access roots.
    return digest({"profile": profile_dict(profile), "roots": roots})


def matching_profile(command: Command, profile: Profile | None, roots: list[str]) -> bool:
    return (profile is not None and profile.active and profile.adapter == "openpi-sft"
            and profile.node_id == command.node_id and profile.id == command.payload["profile_id"]
            and profile_digest(profile, roots) == command.payload.get("profile_digest"))


def require_profile(session, node_id: str, profile_id: str) -> Profile:
    profile = session.get(Profile, (node_id, profile_id))
    if profile is None or not profile.active:
        raise APIError(404, "PROFILE_UNAVAILABLE", ERRORS["PROFILE_UNAVAILABLE"])
    if profile.adapter != "openpi-sft":
        raise APIError(422, "RECIPE_PROFILE_UNSUPPORTED", "该训练环境无需查询 OpenPI 训练配方")
    return profile


def expire_pending(session, node_id: str | None = None) -> None:
    query = select(Command).where(Command.type == "RECIPE_SUGGEST", Command.status == "PENDING")
    if node_id is not None:
        query = query.where(Command.node_id == node_id)
    timestamp = now()
    for command in session.scalars(query.with_for_update()):
        expires_at = command.payload.get("expires_at", "")
        if not isinstance(expires_at, str) or expires_at <= timestamp:
            command.status = "REJECTED"
            command.result = {"error": _error("RECIPE_SUGGEST_TIMEOUT")}
            command.acknowledged_at = timestamp
    session.flush()


def validated_result(command: Command, result) -> dict:
    """Never return raw config contents, paths, errors, or Agent trust claims."""
    invalid = {"error": _error("RECIPE_SUGGEST_INVALID_RESULT")}
    if not isinstance(result, dict):
        return invalid
    if result.get("error") is not None:
        error = result["error"]
        code = error.get("code") if isinstance(error, dict) else None
        if code == "COMMAND_UNSUPPORTED":
            code = "AGENT_RECIPE_SUGGEST_UNSUPPORTED"
        elif code == "RECIPE_QUERY_EXPIRED":
            code = "RECIPE_SUGGEST_TIMEOUT"
        return {"error": _error(code if isinstance(code, str) and code in ERRORS else "RECIPE_SUGGEST_FAILED")}
    prefix = command.payload["prefix"]
    if (result.get("profile_id") != command.payload["profile_id"] or result.get("prefix") != prefix
            or not isinstance(result.get("truncated"), bool) or not isinstance(result.get("source"), str)
            or result["source"] not in SOURCES):
        return invalid
    warnings = result.get("warnings")
    if (not isinstance(warnings, list) or len(warnings) > len(WARNINGS)
            or any(not isinstance(code, str) or code not in WARNINGS for code in warnings)
            or len(set(warnings)) != len(warnings)):
        return invalid
    entries = result.get("entries")
    if not isinstance(entries, list) or len(entries) > min(command.payload["limit"], MAX_ENTRIES):
        return invalid
    names = set()
    for entry in entries:
        name = entry.get("name") if isinstance(entry, dict) else None
        if not _valid_name(name) or not recipe_matches(name, prefix) or name in names:
            return invalid
        names.add(name)
    return {"profile_id": command.payload["profile_id"], "prefix": prefix,
            "entries": [{"name": name} for name in sorted(names, key=lambda name: recipe_sort_key(name, prefix))], "truncated": result["truncated"],
            "source": result["source"], "warnings": list(warnings)}


def acknowledge(command: Command, profile: Profile | None, roots: list[str], status: str, result: dict) -> tuple[str, dict]:
    if not matching_profile(command, profile, roots):
        return "REJECTED", {"error": _error("RECIPE_PROFILE_CHANGED")}
    clean = validated_result(command, result)
    if status == "REJECTED" and "error" not in clean:
        clean = {"error": _error("RECIPE_SUGGEST_FAILED")}
    return ("REJECTED" if "error" in clean else "APPLIED"), clean


def query_dict(command: Command, profile: Profile | None, roots: list[str]) -> dict:
    result = {"id": command.id, "node_id": command.node_id, "profile_id": command.payload["profile_id"],
              "prefix": command.payload["prefix"], "status": "PENDING", "entries": [], "truncated": False,
              "source": None, "warnings": [], "error": None}
    if not matching_profile(command, profile, roots):
        result.update(status="FAILED", error=_error("RECIPE_PROFILE_CHANGED"))
        return result
    if command.status == "PENDING":
        return result
    clean = validated_result(command, command.result)
    if command.status != "APPLIED" or "error" in clean:
        result.update(status="FAILED", error=clean.get("error") or _error("RECIPE_SUGGEST_FAILED"))
        return result
    known = {entry["name"] for entry in clean["entries"]}
    metadata = profile.metadata_json or {}
    registered = {profile.recipe} if _valid_name(profile.recipe) else set()
    validated = metadata.get("validated_recipes", [])
    if isinstance(validated, list):
        registered.update(name for name in validated if _valid_name(name))
    keyword = command.payload["prefix"]
    registered = {name for name in registered if recipe_matches(name, keyword)}
    merged = sorted(known | registered, key=lambda name: recipe_sort_key(name, keyword))
    entries = []
    serialized_profile = profile_dict(profile)
    for name in merged[:command.payload["limit"]]:
        try:
            resolve_recipe(serialized_profile, name)
            is_validated = True
        except ValueError:
            is_validated = False
        entries.append({"name": name, "validated": is_validated,
                        "source": "both" if name in known and name in registered else ("code" if name in known else "registered")})
    result.update(status="READY", entries=entries, source=clean["source"], warnings=clean["warnings"],
                  truncated=clean["truncated"] or len(merged) > command.payload["limit"])
    return result


def create_query(session, node, profile_id: str, prefix: str, limit: int) -> dict:
    if not isinstance(prefix, str) or re.fullmatch(r"[A-Za-z0-9_-]{0,255}", prefix) is None:
        raise APIError(422, "INVALID_RECIPE_PREFIX", "训练配方关键字只能包含英文字母、数字、下划线和连字符")
    profile = require_profile(session, node.id, profile_id)
    snapshot = profile_digest(profile, node.roots)
    expire_pending(session, node.id)
    recent = select(Command).where(
        Command.node_id == node.id, Command.type == "RECIPE_SUGGEST",
        Command.created_at >= offset_time(-CACHE_SECONDS), Command.status.in_({"PENDING", "APPLIED"}),
    ).order_by(Command.created_at.desc())
    for command in session.scalars(recent):
        if (command.payload.get("profile_digest") == snapshot and command.payload.get("prefix") == prefix
                and command.payload.get("limit") == limit and command.payload.get("match_mode") == MATCH_MODE):
            value = query_dict(command, profile, node.roots)
            if value["status"] in {"PENDING", "READY"}:
                return value
    pending = select(Command.id).where(
        Command.node_id == node.id, Command.type == "RECIPE_SUGGEST", Command.status == "PENDING",
    ).limit(MAX_PENDING)
    if len(list(session.scalars(pending))) >= MAX_PENDING:
        raise APIError(429, "RECIPE_SUGGEST_BUSY", "训练配方联想请求较多，请稍后重试；仍可手动填写")
    command = Command(id=uid(), node_id=node.id, type="RECIPE_SUGGEST", status="PENDING",
                      payload={"profile_id": profile_id, "profile_digest": snapshot, "prefix": prefix,
                               "limit": limit, "match_mode": MATCH_MODE,
                               "expires_at": offset_time(QUERY_TTL_SECONDS)}, created_at=now())
    session.add(command)
    return query_dict(command, profile, node.roots)
