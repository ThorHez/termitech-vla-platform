"""Bounded, machine-local path queries over the existing Agent command queue."""

from pathlib import PurePosixPath
import posixpath

from sqlalchemy import select

from .common import APIError, now, offset_time, uid, under_root
from .models import Command, Profile


QUERY_TTL_SECONDS = 15
CACHE_SECONDS = 10
MAX_PENDING = 20
MAX_ENTRIES = 50

ERRORS = {
    "PATH_SUGGEST_TIMEOUT": "路径联想超时，可继续手动填写路径，或稍后重试",
    "AGENT_PATH_SUGGEST_UNSUPPORTED": "当前 Agent 不支持路径联想，请更新 Agent；仍可手动填写路径",
    "PATH_OUTSIDE_ROOTS": "该路径不在机器允许访问的目录内",
    "PATH_OUTSIDE_PROJECT": "训练入口必须在所选训练环境的项目目录内选择",
    "PATH_PROFILE_CHANGED": "训练环境的项目目录已变化，请重新获取路径联想",
    "PATH_NOT_FOUND": "该目录不存在，请检查路径后重试",
    "PATH_NOT_DIRECTORY": "该路径不是目录，可继续手动填写文件路径",
    "PATH_PERMISSION_DENIED": "Agent 无权列出该目录，可继续手动填写路径",
    "PATH_UNAVAILABLE": "该目录不存在或暂不可访问，可继续手动填写路径",
    "PATH_SUGGEST_INVALID_RESULT": "Agent 返回的路径联想结果无效，可继续手动填写路径",
    "PATH_SUGGEST_FAILED": "暂时无法获取路径联想，可继续手动填写路径",
}


def _safe_text(value) -> bool:
    return isinstance(value, str) and len(value) <= 4096 and not any(
        ord(char) < 32 or 127 <= ord(char) < 160 for char in value
    )


def _absolute(value) -> bool:
    return (_safe_text(value) and value.startswith("/")
            and ".." not in PurePosixPath(value).parts)


def _normalized(value: str) -> str:
    # Match the Agent's lexical spelling without resolving machine-local links.
    return "/" + posixpath.normpath(value).lstrip("/")


def _under_roots(path: str, roots: list[str]) -> bool:
    return under_root(_normalized(path), [_normalized(root) for root in roots])


def validate_prefix(prefix: str, roots: list[str]) -> None:
    """Permit rooted paths and incomplete ancestors that can suggest a root."""
    if not _safe_text(prefix) or (prefix and not _absolute(prefix)):
        raise APIError(422, "INVALID_PATH_PREFIX", "路径前缀必须是绝对路径，且不能包含控制字符或上级目录")
    normalized = _normalized(prefix) if prefix else ""
    if prefix and not _under_roots(prefix, roots) and not any(_normalized(root).startswith(normalized) for root in roots):
        raise APIError(422, "PATH_OUTSIDE_ROOTS", ERRORS["PATH_OUTSIDE_ROOTS"])


def _error(code: str) -> dict:
    return {"code": code, "message": ERRORS[code]}


def expire_pending(session, node_id: str | None = None) -> None:
    query = select(Command).where(Command.type == "PATH_SUGGEST", Command.status == "PENDING")
    if node_id is not None:
        query = query.where(Command.node_id == node_id)
    timestamp = now()
    for command in session.scalars(query.with_for_update()):
        expires_at = command.payload.get("expires_at", "")
        if not isinstance(expires_at, str) or expires_at <= timestamp:
            command.status = "REJECTED"
            command.result = {"error": _error("PATH_SUGGEST_TIMEOUT")}
            command.acknowledged_at = timestamp
    session.flush()


def matching_profile(command: Command, profile: Profile | None) -> bool:
    return (profile is not None and profile.active and profile.node_id == command.node_id
            and profile.id == command.payload.get("profile_id") and profile.adapter == "openpi-sft"
            and profile.execution_mode == "native" and profile.code_dir == command.payload.get("base_path"))


def validated_result(command: Command, roots: list[str], result) -> dict:
    """Never serialize unrestricted Agent output, even from historical ACKs."""
    invalid = {"error": _error("PATH_SUGGEST_INVALID_RESULT")}
    if not isinstance(result, dict):
        return invalid
    error = result.get("error")
    if error is not None:
        code = error.get("code") if isinstance(error, dict) else None
        if code == "COMMAND_UNSUPPORTED":
            code = "AGENT_PATH_SUGGEST_UNSUPPORTED"
        elif code == "PATH_QUERY_EXPIRED":
            code = "PATH_SUGGEST_TIMEOUT"
        return {"error": _error(code if isinstance(code, str) and code in ERRORS else "PATH_SUGGEST_FAILED")}
    scoped = command.payload.get("profile_id") is not None
    if scoped:
        if (result.get("profile_id") != command.payload["profile_id"]
                or result.get("base_path") != command.payload.get("base_path")
                or command.payload.get("python_only") is not True
                or not _absolute(command.payload.get("base_path"))):
            return invalid
        # A project query is narrower than the node's generic asset browser.
        roots = [command.payload["base_path"]]
    prefix = command.payload["prefix"]
    if result.get("prefix") != prefix or not isinstance(result.get("truncated"), bool):
        return invalid
    entries = result.get("entries")
    limit = min(command.payload["limit"], MAX_ENTRIES)
    if not isinstance(entries, list) or len(entries) > limit:
        return invalid
    if "parent" not in result:
        return invalid
    parent = result["parent"]
    if parent is not None and (not _absolute(parent) or not _under_roots(parent, roots)):
        return invalid
    clean_entries = []
    seen = set()
    normalized_prefix = (_normalized(prefix) if prefix.endswith("/") else
                         posixpath.join(_normalized(posixpath.dirname(prefix)), posixpath.basename(prefix))) if prefix else ""
    if prefix.endswith("/") and normalized_prefix != "/":
        normalized_prefix += "/"
    for entry in entries:
        if not isinstance(entry, dict):
            return invalid
        name, path, kind = entry.get("name"), entry.get("path"), entry.get("kind")
        root_slash = parent is None and name == "/" and path == "/" and any(_normalized(root) == "/" for root in roots)
        if (not _safe_text(name) or not name or name in {".", ".."} or ("/" in name and not root_slash)
                or not _absolute(path) or not _under_roots(path, roots) or kind not in ("directory", "file")
                or (scoped and kind == "file" and not path.endswith(".py"))):
            return invalid
        normalized_path = _normalized(path)
        if ((PurePosixPath(path).name != name and not root_slash) or normalized_path in seen
                or not (normalized_path + ("/" if kind == "directory" else "")).startswith(normalized_prefix)):
            return invalid
        if parent is None:
            if kind != "directory" or not any(normalized_path == _normalized(root) for root in roots):
                return invalid
        elif PurePosixPath(normalized_path).parent != PurePosixPath(_normalized(parent)):
            return invalid
        seen.add(normalized_path)
        clean_entries.append({"name": name, "path": path, "kind": kind})
    return {"prefix": prefix, "parent": parent, "entries": clean_entries, "truncated": result["truncated"],
            **({"profile_id": command.payload["profile_id"], "base_path": command.payload["base_path"]} if scoped else {})}


def acknowledge(command: Command, roots: list[str], status: str, result: dict, profile: Profile | None = None) -> tuple[str, dict]:
    if command.payload.get("profile_id") is not None and not matching_profile(command, profile):
        return "REJECTED", {"error": _error("PATH_PROFILE_CHANGED")}
    clean = validated_result(command, roots, result)
    if status == "REJECTED" and "error" not in clean:
        clean = {"error": _error("PATH_SUGGEST_FAILED")}
    return ("REJECTED" if "error" in clean else "APPLIED"), clean


def query_dict(command: Command, roots: list[str], profile: Profile | None = None) -> dict:
    result = {"id": command.id, "node_id": command.node_id, "status": "PENDING",
              "prefix": command.payload["prefix"], "parent": None, "entries": [], "truncated": False, "error": None}
    if command.payload.get("profile_id") is not None:
        result.update(profile_id=command.payload["profile_id"], base_path=command.payload["base_path"])
        if not matching_profile(command, profile):
            result.update(status="FAILED", error=_error("PATH_PROFILE_CHANGED"))
            return result
    if command.status == "PENDING":
        return result
    clean = validated_result(command, roots, command.result)
    if command.status != "APPLIED" or "error" in clean:
        result.update(status="FAILED", error=clean.get("error") or _error("PATH_SUGGEST_FAILED"))
    else:
        result.update(clean, status="READY")
    return result


def create_query(session, node, prefix: str, limit: int, profile_id: str | None = None) -> dict:
    profile, scope = None, {}
    if profile_id is not None:
        profile = session.get(Profile, (node.id, profile_id))
        if profile is None or not profile.active:
            raise APIError(422, "PROFILE_NODE_MISMATCH", "训练环境不属于当前机器或已停用")
        if profile.adapter != "openpi-sft" or profile.execution_mode != "native":
            raise APIError(422, "ENTRYPOINT_PROFILE_UNSUPPORTED", "所选训练环境不支持 OpenPI 训练入口联想")
        base_path = profile.code_dir
        if not _absolute(base_path) or not _under_roots(base_path, node.roots):
            raise APIError(422, "PATH_OUTSIDE_ROOTS", ERRORS["PATH_OUTSIDE_ROOTS"])
        prefix = prefix or _normalized(base_path).rstrip("/") + "/"
        if not _absolute(prefix) or not _under_roots(prefix, [base_path]):
            raise APIError(422, "PATH_OUTSIDE_PROJECT", ERRORS["PATH_OUTSIDE_PROJECT"])
        scope = {"profile_id": profile_id, "base_path": base_path, "python_only": True}
    else:
        validate_prefix(prefix, node.roots)
    expire_pending(session, node.id)
    recent = select(Command).where(
        Command.node_id == node.id, Command.type == "PATH_SUGGEST",
        Command.created_at >= offset_time(-CACHE_SECONDS), Command.status.in_({"PENDING", "APPLIED"}),
    ).order_by(Command.created_at.desc())
    for command in session.scalars(recent):
        if (command.payload.get("prefix") == prefix and command.payload.get("limit") == limit
                and all(command.payload.get(key) == scope.get(key) for key in ("profile_id", "base_path", "python_only"))):
            result = query_dict(command, node.roots, profile)
            if result["status"] in {"PENDING", "READY"}:
                return result
    pending = select(Command.id).where(
        Command.node_id == node.id, Command.type == "PATH_SUGGEST", Command.status == "PENDING",
    ).limit(MAX_PENDING)
    if len(list(session.scalars(pending))) >= MAX_PENDING:
        raise APIError(429, "PATH_SUGGEST_BUSY", "路径联想请求较多，请稍后重试；仍可手动填写路径")
    command = Command(id=uid(), node_id=node.id, type="PATH_SUGGEST", status="PENDING",
                      payload={"prefix": prefix, "limit": limit, "expires_at": offset_time(QUERY_TTL_SECONDS), **scope}, created_at=now())
    session.add(command)
    return query_dict(command, node.roots, profile)
