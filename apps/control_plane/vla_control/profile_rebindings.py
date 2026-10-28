"""Agent-owned runtime changes with durable delivery and conservative reconciliation.

The server never changes a profile merely because a request was queued or a
heartbeat reported new paths. A matching persisted Agent receipt is required.
"""

from copy import deepcopy
from datetime import datetime
import json
from pathlib import PurePosixPath

from fastapi.responses import JSONResponse
from pydantic import Field
from sqlalchemy import select

from .common import APIError, TERMINAL, now, offset_time, require_path, uid
from .models import Attempt, Command, Node, Preview, Profile, Run
from .policy_deployment_models import PolicyDeployment, RESOURCE_STATES
from .schemas import InputModel
from .serialization import profile_dict


TTL_SECONDS = 120
PATH_FIELDS = ("expected_code_dir", "expected_python", "code_dir", "python")
ERRORS = {
    "AGENT_PROFILE_REBIND_UNSUPPORTED": "当前 Agent 不支持切换训练环境，请先更新 Agent",
    "PROFILE_REBIND_INVALID": "训练环境路径或操作参数无效",
    "PROFILE_REBIND_EXPIRED": "切换命令已过期，Agent 未执行修改，请重试",
    "PROFILE_REBIND_CONFLICT": "训练环境已变化，请刷新后重新选择",
    "PROFILE_REBIND_BUSY": "该训练环境仍有训练、真机部署或代码操作，请等待完成",
    "PROFILE_REBIND_RUNTIME_UNAVAILABLE": "新的训练环境未通过 Agent 检查，请核对代码目录和 Python",
    "PROFILE_REBIND_FAILED": "Agent 未完成训练环境切换，请检查机器上的操作记录",
    "PROFILE_REBIND_CONFIRMATION_PENDING": "尚未确认 Agent 的持久化结果，训练保持锁定；请等待 Agent 回执",
    "PROFILE_REBIND_INVALID_RESULT": "Agent 的切换凭据不完整，训练保持锁定；请检查 Agent 操作记录",
}


class RebindInput(InputModel):
    profile_id: str = Field(min_length=1, max_length=200)
    expected_code_dir: str = Field(min_length=1, max_length=4096)
    expected_python: str = Field(min_length=1, max_length=4096)
    code_dir: str = Field(min_length=1, max_length=4096)
    python: str = Field(min_length=1, max_length=4096)


def _error(code):
    return {"code": code, "message": ERRORS.get(code, ERRORS["PROFILE_REBIND_FAILED"])}


def _path(value, roots=None):
    if (not isinstance(value, str) or not value or len(value) > 4096
            or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value)):
        raise APIError(422, "PROFILE_REBIND_INVALID", ERRORS["PROFILE_REBIND_INVALID"])
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise APIError(422, "PROFILE_REBIND_INVALID", ERRORS["PROFILE_REBIND_INVALID"]) from exc
    require_path(value, roots)
    return str(PurePosixPath(value))


def _pair(profile):
    return str(PurePosixPath(profile.code_dir)), str(PurePosixPath(profile.python))


def _set_runtime_check(node, profile_id, check):
    existing = (node.metrics or {}).get("runtime_checks", [])
    existing = existing if isinstance(existing, list) else []
    node.metrics = {**(node.metrics or {}), "runtime_checks": [deepcopy(row) for row in existing
        if isinstance(row, dict) and row.get("profile_id") != profile_id] + [deepcopy(check)]}


def pending(session, node_id, profile_id=None):
    commands = session.scalars(select(Command).where(Command.node_id == node_id,
        Command.type == "PROFILE_REBIND", Command.status == "PENDING"))
    return [command for command in commands if profile_id is None or command.payload.get("profile_id") == profile_id]


def require_not_pending(session, node_id, profile_id=None, code_dir=None):
    from .storage_configurations import require_not_pending as require_storage_not_pending
    require_storage_not_pending(session, node_id)
    wanted = str(PurePosixPath(code_dir)) if code_dir is not None else None
    for command in pending(session, node_id):
        if (profile_id is None and wanted is None or command.payload["profile_id"] == profile_id
                or wanted in {command.payload["expected_code_dir"], command.payload["code_dir"]}):
            raise APIError(409, "PROFILE_REBIND_PENDING", "训练环境正在切换或等待持久化确认，请等待完成后再操作")


def _in_use(session, node_id, profile_id):
    from .checkpoint_transfers import active_transfers
    run = session.scalar(select(Run.id).where(Run.node_id == node_id, Run.profile_id == profile_id,
                                             Run.state.not_in(TERMINAL)).limit(1))
    attempt = session.scalar(select(Attempt.id).join(Run, Run.id == Attempt.run_id).where(
        Attempt.node_id == node_id, Run.profile_id == profile_id, Attempt.state.not_in(TERMINAL)).limit(1))
    deployment = session.scalar(select(PolicyDeployment.id).where(
        PolicyDeployment.node_id == node_id, PolicyDeployment.profile_id == profile_id,
        PolicyDeployment.state.in_(RESOURCE_STATES)).limit(1))
    return bool(run or attempt or deployment or active_transfers(session, node_id, profile_id))


def _require_idle(session, node_id, profile_id, code_dirs):
    if _in_use(session, node_id, profile_id):
        raise APIError(409, "PROFILE_REBIND_BUSY", ERRORS["PROFILE_REBIND_BUSY"])
    # The old and destination checkout may also be used by another profile.
    profiles = {p.id for p in session.scalars(select(Profile).where(Profile.node_id == node_id))
                if str(PurePosixPath(p.code_dir)) in code_dirs}
    from .checkpoint_transfers import active_transfers
    if any(active_transfers(session, node_id, code_dir=directory) for directory in code_dirs):
        raise APIError(409, "PROFILE_REBIND_BUSY", ERRORS["PROFILE_REBIND_BUSY"])
    deployment = session.scalar(select(PolicyDeployment.id).where(
        PolicyDeployment.node_id == node_id, PolicyDeployment.profile_id.in_(profiles),
        PolicyDeployment.state.in_(RESOURCE_STATES)).limit(1))
    if deployment:
        raise APIError(409, "PROFILE_REBIND_BUSY", ERRORS["PROFILE_REBIND_BUSY"])
    for run in session.scalars(select(Run).where(Run.node_id == node_id, Run.state.not_in(TERMINAL))):
        if run.profile_id in profiles or (run.code_expectation or {}).get("code_dir") in code_dirs:
            raise APIError(409, "PROFILE_REBIND_BUSY", ERRORS["PROFILE_REBIND_BUSY"])
    for attempt in session.scalars(select(Attempt).where(Attempt.node_id == node_id, Attempt.state.not_in(TERMINAL))):
        run = session.get(Run, attempt.run_id)
        if (run is not None and run.profile_id in profiles
                or (attempt.code_snapshot or {}).get("code_dir") in code_dirs):
            raise APIError(409, "PROFILE_REBIND_BUSY", ERRORS["PROFILE_REBIND_BUSY"])
    for command in session.scalars(select(Command).where(Command.node_id == node_id,
            Command.status == "PENDING", Command.type.in_({"START", "PREFLIGHT", "RECIPE_WRITE"}))):
        data = command.payload
        source = (data.get("definition") or {}).get("source", {})
        preview = session.get(Preview, data.get("preview_id")) if data.get("preview_id") else None
        attempt = session.get(Attempt, data.get("attempt_id")) if data.get("attempt_id") else None
        run = session.get(Run, attempt.run_id) if attempt else None
        if (data.get("profile_id") in profiles | {profile_id}
                or source.get("code_dir") in code_dirs
                or preview is not None and preview.request.get("profile_id") in profiles | {profile_id}
                or run is not None and run.profile_id in profiles | {profile_id}):
            raise APIError(409, "PROFILE_REBIND_BUSY", ERRORS["PROFILE_REBIND_BUSY"])


def query_dict(command):
    result = {"id": command.id, "node_id": command.node_id, "profile_id": command.payload["profile_id"],
        "requested": {key: command.payload[key] for key in ("code_dir", "python")},
        "status": "PENDING", "error": None, "profile": None, "runtime_check": None}
    receipt = command.result if isinstance(command.result, dict) else {}
    if command.status == "APPLIED":
        result.update(status="READY", profile=deepcopy(receipt["profile"]), runtime_check=deepcopy(receipt["runtime_check"]))
    elif command.status != "PENDING":
        result.update(status="FAILED", error=deepcopy(receipt.get("error", _error("PROFILE_REBIND_FAILED"))))
    elif receipt.get("error") or command.payload["expires_at"] < now():
        result["error"] = deepcopy(receipt.get("error", _error("PROFILE_REBIND_CONFIRMATION_PENDING")))
    return result


def create_query(session, node, body):
    data = body.model_dump()
    for key in PATH_FIELDS:
        data[key] = _path(data[key], node.roots if key == "code_dir" else None)
    commands = list(session.scalars(select(Command).where(Command.node_id == node.id,
        Command.type == "PROFILE_REBIND").order_by(Command.created_at.desc(), Command.id.desc())))
    profile = session.get(Profile, (node.id, data["profile_id"]))
    # Lost HTTP responses and repeated clicks reuse the durable pending/success operation.
    for command in commands:
        if (all(command.payload.get(key) == value for key, value in data.items())
                and (command.status == "PENDING" or command.status == "APPLIED" and profile is not None
                     and profile.active and _pair(profile) == (data["code_dir"], data["python"])
                     and not pending(session, node.id, data["profile_id"]))):
            return query_dict(command)
    if profile is None or not profile.active:
        raise APIError(422, "PROFILE_NODE_MISMATCH", "训练环境不属于该机器或已停用")
    if profile.adapter != "openpi-sft" or profile.execution_mode != "native":
        raise APIError(422, "PROFILE_REBIND_UNSUPPORTED", "仅支持切换本机原生 OpenPI 训练环境")
    if _pair(profile) != (data["expected_code_dir"], data["expected_python"]):
        raise APIError(409, "PROFILE_REBIND_CONFLICT", ERRORS["PROFILE_REBIND_CONFLICT"])
    require_not_pending(session, node.id, profile.id)
    for checkout in {data["expected_code_dir"], data["code_dir"]}:
        require_not_pending(session, node.id, code_dir=checkout)
    _require_idle(session, node.id, profile.id, {data["expected_code_dir"], data["code_dir"]})
    command = Command(id=uid(), node_id=node.id, type="PROFILE_REBIND", status="PENDING", created_at=now(),
                      payload={**data, "expires_at": offset_time(TTL_SECONDS)})
    session.add(command)
    session.flush()
    return query_dict(command)


def _runtime(value, profile_id, roots, python):
    required = {"code_directory", "python_interpreter", "git_repository", "interpreter_probe", "openpi_import", "jax_import",
                "selected_project", "selected_project_import"}
    if (not isinstance(value, dict) or value.get("profile_id") != profile_id or value.get("status") != "READY"
            or not isinstance(value.get("checks"), list) or not 1 <= len(value["checks"]) <= 32):
        return None
    for key in ("checked_at", "python_version", "executable", "prefix", "module_origin"):
        if not isinstance(value.get(key), str) or not 1 <= len(value[key]) <= 4096:
            return None
    try:
        datetime.fromisoformat(value["checked_at"])
        _path(value["executable"])
        _path(value["prefix"])
        _path(value["module_origin"], roots)
    except (ValueError, APIError):
        return None
    checks = []
    for item in value["checks"]:
        if (not isinstance(item, dict) or item.get("ok") is not True
                or not isinstance(item.get("name"), str) or not 1 <= len(item["name"]) <= 100
                or not isinstance(item.get("message"), str) or len(item["message"]) > 4096):
            return None
        checks.append({key: item[key] for key in ("name", "ok", "message")})
    if not required <= {item["name"] for item in checks}:
        return None
    named = {item["name"]: item["message"] for item in checks}
    if len(named) != len(checks):
        return None
    try:
        directory = _path(named["code_directory"], roots)
        repository = _path(named["git_repository"], roots)
        require_path(directory, [repository])
        require_path(value["module_origin"], [repository])
        package = str(PurePosixPath(directory) / "src/openpi")
        require_path(value["module_origin"], [package])
        if (named["python_interpreter"] != python or named["openpi_import"] != value["module_origin"]
                or named["selected_project"] != str(PurePosixPath(package) / "training/config.py")
                or named["selected_project_import"] != value["module_origin"]
                or named["interpreter_probe"] != f"Python {value['python_version']} at {value['executable']}"):
            return None
    except APIError:
        return None
    clean = {key: value[key] for key in ("profile_id", "status", "checked_at", "python_version", "executable", "prefix", "module_origin")}
    clean["checks"] = checks
    try:
        length = len(json.dumps(clean, ensure_ascii=False).encode("utf-8"))
    except (TypeError, ValueError, UnicodeError):
        return None
    if length > 64 * 1024:
        return None
    return clean


def acknowledge(session, command, node, status, result):
    payload = command.payload
    error = result.get("error") if isinstance(result, dict) else None
    if isinstance(error, dict):
        code = error.get("code")
        code = "AGENT_PROFILE_REBIND_UNSUPPORTED" if code == "COMMAND_UNSUPPORTED" else code
        known_unmodified = error.get("persisted") is False or code == "AGENT_PROFILE_REBIND_UNSUPPORTED"
        clean = {"error": _error(code if isinstance(code, str) and code in ERRORS else "PROFILE_REBIND_FAILED"), "persisted": False} if known_unmodified else None
    else:
        runtime = _runtime(result.get("runtime_check"), payload["profile_id"], node.roots, payload["python"]) if isinstance(result, dict) else None
        clean = ({"profile_id": payload["profile_id"], "code_dir": payload["code_dir"], "python": payload["python"],
                  "runtime_check": runtime, "persisted": True}
                 if status == "APPLIED" and isinstance(result, dict) and result.get("persisted") is True and runtime is not None
                 and all(result.get(key) == payload[key] for key in ("profile_id", "code_dir", "python")) else None)
    if command.status != "PENDING":
        if clean != (command.result or {}).get("agent_result"):
            raise APIError(409, "ACK_CONFLICT", "This command already has a different acknowledgement")
        return
    if clean is None:
        command.result = {"error": _error("PROFILE_REBIND_CONFIRMATION_PENDING" if error else "PROFILE_REBIND_INVALID_RESULT")}
        return
    if "error" in clean:
        command.status, command.result, command.acknowledged_at = "REJECTED", {"agent_result": clean, "error": clean["error"]}, now()
        return
    profile = session.get(Profile, (node.id, payload["profile_id"]))
    if (profile is None or not profile.active or _pair(profile) != (payload["expected_code_dir"], payload["expected_python"])
            or _in_use(session, node.id, payload["profile_id"])):
        command.result = {"error": _error("PROFILE_REBIND_CONFIRMATION_PENDING")}
        return
    profile.code_dir, profile.python = payload["code_dir"], payload["python"]
    _set_runtime_check(node, profile.id, clean["runtime_check"])
    for preview in session.scalars(select(Preview).where(Preview.node_id == node.id, Preview.status.in_({"PENDING", "READY"}))):
        if preview.request.get("profile_id") == profile.id:
            preview.status, preview.error = "FAILED", {"code": "PROFILE_CHANGED_SINCE_PREFLIGHT", "message": "训练环境已切换，请重新预检"}
    command.status, command.acknowledged_at = "APPLIED", now()
    command.result = {"agent_result": clean, "profile": profile_dict(profile), "runtime_check": clean["runtime_check"],
                      "heartbeat_confirmed": False}
    session.flush()


def sync_blocked_profiles(session, node_id, incoming):
    """Hold pending paths; ignore the first known stale heartbeat after success."""
    blocked = {command.payload["profile_id"] for command in pending(session, node_id)}
    seen = set()
    for command in session.scalars(select(Command).where(Command.node_id == node_id,
            Command.type == "PROFILE_REBIND", Command.status == "APPLIED").order_by(Command.created_at.desc(), Command.id.desc())):
        profile_id = command.payload["profile_id"]
        if profile_id in seen:
            continue
        seen.add(profile_id)
        if (command.result or {}).get("heartbeat_confirmed") or profile_id in blocked:
            continue
        entry = incoming.get(profile_id)
        if entry is None:
            continue
        pair = (str(PurePosixPath(entry["code_dir"])), str(PurePosixPath(entry["python"])))
        old = command.payload["expected_code_dir"], command.payload["expected_python"]
        target = command.payload["code_dir"], command.payload["python"]
        if pair == old and old != target:
            blocked.add(profile_id)
            node = session.get(Node, node_id)
            if node is not None:
                _set_runtime_check(node, profile_id, command.result["runtime_check"])
        else:
            # A matching fresh heartbeat establishes convergence. A deliberate
            # later Agent configuration change retains normal sync semantics.
            command.result = {**command.result, "heartbeat_confirmed": True}
    return blocked


def register_routes(app, admin, db, require_available_node, settings):
    def response(value, status=200):
        return JSONResponse(status_code=status, content=value, headers={"Cache-Control": "no-store"})

    @app.post("/api/v1/nodes/{node_id}/profile-rebindings", dependencies=admin, status_code=202)
    def create(node_id: str, body: RebindInput):
        with db.write() as session:
            node = require_available_node(session, node_id)
            if node.last_seen < offset_time(-settings.heartbeat_timeout_seconds):
                raise APIError(409, "NODE_OFFLINE", "机器心跳已过期，暂时无法切换训练环境")
            return response(create_query(session, node, body), 202)

    @app.get("/api/v1/nodes/{node_id}/profile-rebindings", dependencies=admin)
    def latest(node_id: str, profile_id: str):
        with db.read() as session:
            node = session.get(Node, node_id)
            if node is None:
                raise APIError(404, "NOT_FOUND", "Node not found")
            operation = session.scalar(select(Command).where(Command.node_id == node_id,
                Command.type == "PROFILE_REBIND", Command.payload["profile_id"].as_string() == profile_id)
                .order_by(Command.created_at.desc(), Command.id.desc()).limit(1))
            return response({"operation": query_dict(operation) if operation else None})

    @app.get("/api/v1/nodes/{node_id}/profile-rebindings/{operation_id}", dependencies=admin)
    def get(node_id: str, operation_id: str):
        with db.read() as session:
            command = session.get(Command, operation_id)
            if command is None or command.type != "PROFILE_REBIND" or command.node_id != node_id:
                raise APIError(404, "NOT_FOUND", "Profile rebind operation not found")
            return response(query_dict(command))
