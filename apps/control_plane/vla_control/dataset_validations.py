"""Durable, dataset-only CPU validation independent of training authorization."""

from copy import deepcopy
import json
import math
from pathlib import PurePosixPath
import re

from fastapi import Request
from pydantic import Field
from sqlalchemy import select
from sqlalchemy.orm import object_session

from .common import APIError, digest, now, offset_time, require_path, uid
from .models import Asset, Command, Node, Profile
from .schemas import InputModel
from .resource_authorization import (
    bind_command_actor, execution_permission_authorized, request_user_id, require_active_actor,
)
from .account_lifecycle import task_execution_authorized


COMMAND = "DATASET_VALIDATE"
INTERNAL_KEY = "_dataset_validation"
SCOPE = "dataset_source"
QUEUE_TTL_SECONDS = 24 * 60 * 60
MAX_RESULT_BYTES = 256 * 1024
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
ERRORS = {
    "DATASET_VALIDATION_PROFILE_REQUIRED": "有多个校验环境，请选择本数据集使用的环境后重新校验",
    "DATASET_VALIDATION_PROFILE_UNAVAILABLE": "没有可用的原生 OpenPI 校验环境，请先配置环境",
    "DATASET_VALIDATION_NODE_OFFLINE": "机器离线，登记已保存；恢复在线后可重新校验",
    "DATASET_VALIDATION_AGENT_UNSUPPORTED": "当前 Agent 尚不支持数据源校验，登记已保存",
    "DATASET_VALIDATION_INPUT_CHANGED": "数据集登记信息或校验环境已改变，请重新校验",
    "DATASET_VALIDATION_EXPIRED": "后台校验请求已过期，请重新校验",
    "DATASET_VALIDATION_INVALID_RESULT": "Agent 返回的校验结果无效，请重新校验",
    "DATASET_VALIDATION_FAILED": "数据源校验未通过，请查看具体检查项",
    "DATASET_VALIDATION_CANCELLED": "基础校验已取消；任务尚未下发，数据集和训练均保留",
}


class ValidationInput(InputModel):
    profile_id: str | None = Field(default=None, min_length=1, max_length=200, strict=True)


def _error(code, message=None):
    return {"code": code, "message": message or ERRORS.get(code, ERRORS["DATASET_VALIDATION_FAILED"])}


def _metadata(asset):
    return {key: deepcopy(value) for key, value in (asset.metadata_json or {}).items() if key != INTERNAL_KEY}


def input_digest(asset, profile, node):
    runtime = None if profile is None else {key: getattr(profile, key) for key in (
        "id", "node_id", "code_dir", "python", "adapter", "recipe", "execution_mode", "docker_image", "active", "metadata_json")}
    return digest({"asset_id": asset.id, "node_id": asset.node_id, "path": asset.path,
                   "version": asset.version, "kind": asset.kind, "metadata": _metadata(asset),
                   "profile": runtime, "roots": node.roots if node is not None else None})


def _write(asset, report):
    asset.metadata_json = {**(asset.metadata_json or {}), INTERNAL_KEY: deepcopy(report)}


def _context(session, asset, profile_id):
    # Keep ORM references for this request to avoid repeating the same two
    # reads for every asset in a list. A write still sees its tracked objects.
    cache = session.info.setdefault("dataset_validation_context", {})
    node_key, profile_key = ("node", asset.node_id), ("profile", asset.node_id, profile_id)
    if node_key not in cache:
        cache[node_key] = session.get(Node, asset.node_id)
    if profile_key not in cache:
        cache[profile_key] = session.get(Profile, (asset.node_id, profile_id)) if profile_id else None
    return cache[node_key], cache[profile_key]


def validation_dict(asset, session=None):
    if asset.kind != "DATASET":
        return None
    stored = (asset.metadata_json or {}).get(INTERNAL_KEY)
    if not isinstance(stored, dict):
        return None
    report = deepcopy(stored)
    session = session or object_session(asset)
    if session is not None and report.get("status") in {"PENDING", "SUCCEEDED", "FAILED"}:
        node, profile = _context(session, asset, report.get("profile_id"))
        if (asset.state in {"ARCHIVED", "UNAVAILABLE"} or node is None or profile is None
                or input_digest(asset, profile, node) != report.get("input_digest")):
            report.update(status="STALE", error=_error("DATASET_VALIDATION_INPUT_CHANGED"))
    if report.get("status") == "PENDING" and report.get("expires_at", "") <= now():
        report.update(status="FAILED", error=_error("DATASET_VALIDATION_EXPIRED"))
    return report


def invalidate_validation(session, asset, reason="DATASET_VALIDATION_INPUT_CHANGED"):
    stored = (asset.metadata_json or {}).get(INTERNAL_KEY)
    if not isinstance(stored, dict):
        return
    report = {**stored, "status": "STALE", "error": _error(reason)}
    _write(asset, report)
    command = session.get(Command, stored.get("id")) if stored.get("id") else None
    if command is not None and command.type == COMMAND and command.status == "PENDING":
        command.status, command.result, command.acknowledged_at = "REJECTED", {"error": _error(reason)}, now()


def reject_undelivered_validation(session, command, error):
    """Reject local-only work and its matching report, locking Asset first.

    Delivered commands keep their retry/late-ack semantics. The report ID and
    digest prevent an old command from overwriting a newer validation report.
    No other registered metadata or actor attribution changes here.
    """
    if command.type != COMMAND:
        return False
    asset = session.scalar(select(Asset).where(Asset.id == command.payload.get("asset_id"))
        .with_for_update().execution_options(populate_existing=True))
    command = session.scalar(select(Command).where(Command.id == command.id)
        .with_for_update().execution_options(populate_existing=True))
    if command is None or command.type != COMMAND or command.status != "PENDING" or command.delivered_at is not None:
        return False
    stamp = now()
    clean = _error(str(error.get("code") or "COMMAND_AUTHORIZATION_REVOKED")[:128],
                   str(error.get("message") or "校验发起账户或权限已失效，请由管理员重新提交")[:2000])
    stored = (asset.metadata_json or {}).get(INTERNAL_KEY) if asset is not None else None
    if (isinstance(stored, dict) and stored.get("status") == "PENDING" and stored.get("id") == command.id
            and asset.node_id == command.node_id and stored.get("input_digest") == command.payload.get("input_digest")):
        _write(asset, {**stored, "status": "FAILED", "error": deepcopy(clean), "completed_at": stamp})
    command.status, command.result, command.acknowledged_at = "REJECTED", {"error": clean, "checks": []}, stamp
    session.flush()
    return True


def validation_execution_authorized(session, command, *, allow_legacy=False):
    """Recheck ownership of manual work and the provenance of registration/copy work."""
    if command.type != COMMAND:
        return False
    source_id = command.payload.get("source_transfer_id")
    registration = command.payload.get("source_registration")
    if ((registration is not None and registration is not True)
            or (source_id is not None and registration is not None)):
        return False
    owned_retry = source_id is None and registration is None and command.required_permission == "datasets.validate"
    permission = ("assets.transfer" if source_id is not None else
                  "datasets.register" if registration is True else
                  "datasets.validate" if owned_retry else "assets.write")
    legacy = command.actor_user_id is None and command.required_permission is None
    if (command.required_permission != permission and not legacy
            or not execution_permission_authorized(session, command.actor_user_id, permission, allow_legacy=allow_legacy)):
        return False
    if registration is True or owned_retry:
        asset = session.get(Asset, command.payload.get("asset_id"))
        if asset is None or asset.kind != "DATASET" or asset.state in {"ARCHIVED", "UNAVAILABLE"}:
            return False
        if owned_retry and (command.actor_user_id is None or asset.registered_by_user_id != command.actor_user_id):
            return False
        stored = (asset.metadata_json or {}).get(INTERNAL_KEY)
        if not isinstance(stored, dict):
            return False
        if registration is True and stored.get("registration") != {
            "asset_id": asset.id, "node_id": asset.node_id, "path": asset.path,
            "created_at": asset.created_at, "actor_user_id": command.actor_user_id,
            "input_digest": command.payload.get("input_digest"),
        }:
            return False
        node, profile = _context(session, asset, command.payload.get("profile_id"))
        return bool(node is not None and profile is not None
                    and stored.get("status") == "PENDING" and stored.get("id") == command.id
                    and stored.get("profile_id") == command.payload.get("profile_id")
                    and stored.get("input_digest") == command.payload.get("input_digest")
                    and command.node_id == asset.node_id and command.payload.get("path") == asset.path
                    and command.payload.get("registered_profile") == _metadata(asset).get("dataset_profile")
                    and input_digest(asset, profile, node) == command.payload.get("input_digest"))
    if source_id is None:
        return True
    if not isinstance(source_id, str) or not source_id:
        return False
    from .asset_transfer_models import AssetTransfer
    transfer = session.get(AssetTransfer, source_id)
    asset = session.get(Asset, command.payload.get("asset_id"))
    if transfer is None or asset is None:
        return False
    actor = transfer.execution_user_id if transfer.execution_user_id is not None else transfer.owner_user_id
    provenance = (asset.metadata_json or {}).get("asset_sync")
    return bool(transfer.state == "SUCCEEDED" and transfer.target_asset_id == asset.id
                and transfer.target_node_id == command.node_id == asset.node_id
                and asset.kind == "DATASET" and command.actor_user_id == actor
                and isinstance(provenance, dict) and provenance.get("transfer_id") == transfer.id
                and task_execution_authorized(session, transfer.owner_user_id, actor, allow_legacy=allow_legacy))


def cancel_pending_validation(session, asset):
    """Cancel only this asset's proven undelivered validation command.

    The authenticated route holds AuthGuard, also held by command delivery, so
    delivered_at cannot change between this check and commit. Keep Asset before
    Command ordering consistent with validation acknowledgement and asset edits.
    """
    stored = (asset.metadata_json or {}).get(INTERNAL_KEY)
    command = session.scalar(select(Command).where(Command.id == stored.get("id"))
        .with_for_update().execution_options(populate_existing=True)) if isinstance(stored, dict) and stored.get("id") else None
    matching = (command is not None and command.type == COMMAND and command.node_id == asset.node_id
                and isinstance(command.payload, dict) and command.payload.get("asset_id") == asset.id
                and command.payload.get("input_digest") == stored.get("input_digest"))
    if not matching:
        raise APIError(409, "DATASET_VALIDATION_NOT_PENDING", "本数据集没有可取消的当前基础校验任务")
    if command.delivered_at is not None:
        raise APIError(409, "DATASET_VALIDATION_ALREADY_DELIVERED",
                       "基础校验任务已下发，不能确认执行进程已停止；请等待校验完成后再升级 Agent")
    error = command.result.get("error") if isinstance(command.result, dict) else None
    if (stored.get("status") == "CANCELLED" and command.status == "REJECTED"
            and isinstance(error, dict) and error.get("code") == "DATASET_VALIDATION_CANCELLED"):
        return deepcopy(stored)
    if stored.get("status") != "PENDING" or command.status != "PENDING":
        raise APIError(409, "DATASET_VALIDATION_NOT_PENDING", "本数据集没有可取消的当前基础校验任务")
    stamp = now()
    error = _error("DATASET_VALIDATION_CANCELLED")
    report = {**stored, "status": "CANCELLED", "error": deepcopy(error), "completed_at": stamp}
    _write(asset, report)
    command.status, command.result, command.acknowledged_at = "REJECTED", {"error": error, "checks": []}, stamp
    session.flush()
    return deepcopy(report)


def _reuse_pending_validation(session, asset, previous):
    session.scalar(select(Asset).where(Asset.id == asset.id).with_for_update()
        .execution_options(populate_existing=True))
    command = session.scalar(select(Command).where(Command.id == previous.get("id"))
        .with_for_update().execution_options(populate_existing=True)) if previous.get("id") else None
    if command is None or command.type != COMMAND or command.status != "PENDING":
        return False
    if command.delivered_at is not None:
        return True
    if validation_execution_authorized(session, command, allow_legacy=True):
        return True
    code = "COMMAND_ACTOR_REQUIRED" if command.actor_user_id is None else "COMMAND_AUTHORIZATION_REVOKED"
    reject_undelivered_validation(session, command,
        {"code": code, "message": "校验发起账户或权限已失效，旧校验未下发，请使用有权限的账户重新提交"})
    return False


def queue_validation(session, asset, profile_id=None, *, heartbeat_timeout_seconds=90, actor_request=None,
                     actor_user_id=None, source_transfer_id=None, source_registration=False):
    if asset.kind != "DATASET":
        return None
    if source_registration and source_transfer_id is not None:
        raise ValueError("dataset validation must have exactly one source")
    node = session.get(Node, asset.node_id)
    if node is None:
        raise APIError(404, "NOT_FOUND", "机器不存在")
    if asset.state in {"ARCHIVED", "UNAVAILABLE"}:
        raise APIError(409, "DATASET_VALIDATION_ASSET_UNAVAILABLE", "数据集已归档或不可用，不能发起校验")
    require_path(asset.path, node.roots)
    choices = list(session.scalars(select(Profile).where(Profile.node_id == node.id, Profile.active.is_(True),
        Profile.adapter == "openpi-sft", Profile.execution_mode == "native").order_by(Profile.id)))
    chosen = next((item for item in choices if item.id == profile_id), None) if profile_id else (choices[0] if len(choices) == 1 else None)
    if profile_id and chosen is None:
        raise APIError(422, "DATASET_VALIDATION_PROFILE_UNAVAILABLE", "请选择本机有效的原生 OpenPI 校验环境")
    # A new registration receives its own server-written receipt. Never reuse
    # or invalidate a command referenced by supplied/imported internal metadata.
    previous = None if source_registration else validation_dict(asset, session)
    if (previous and previous.get("status") == "PENDING" and chosen is not None
            and previous.get("profile_id") == chosen.id and previous.get("input_digest") == input_digest(asset, chosen, node)
            and _reuse_pending_validation(session, asset, previous)):
        return previous
    reason = None
    if chosen is None:
        reason = "DATASET_VALIDATION_PROFILE_REQUIRED" if choices else "DATASET_VALIDATION_PROFILE_UNAVAILABLE"
    elif node.status != "ONLINE" or node.last_seen < offset_time(-heartbeat_timeout_seconds):
        reason = "DATASET_VALIDATION_NODE_OFFLINE"
    elif not isinstance((node.metrics or {}).get("capabilities"), list) or COMMAND not in node.metrics["capabilities"]:
        reason = "DATASET_VALIDATION_AGENT_UNSUPPORTED"
    if previous and previous.get("id"):
        invalidate_validation(session, asset)
    timestamp = now()
    report = {"id": None, "scope": SCOPE, "status": "BLOCKED" if reason else "PENDING",
              "profile_id": chosen.id if chosen else None, "input_digest": input_digest(asset, chosen, node),
              "requested_at": timestamp, "completed_at": None, "expires_at": offset_time(QUEUE_TTL_SECONDS),
              "error": _error(reason) if reason else None, "checks": [], "source_fingerprint": None,
              "timestamp_contract": None, "raw_samples": None, "runtime": None}
    if reason is None:
        command = Command(id=uid(), node_id=node.id, type=COMMAND, status="PENDING", created_at=timestamp,
            payload={"asset_id": asset.id, "path": asset.path, "profile_id": chosen.id,
                     "registered_profile": deepcopy(_metadata(asset).get("dataset_profile")),
                     "input_digest": report["input_digest"], "expires_at": report["expires_at"]})
        if source_transfer_id is not None:
            command.payload = {**command.payload, "source_transfer_id": source_transfer_id}
            command.actor_user_id, command.required_permission = actor_user_id, "assets.transfer"
        else:
            if source_registration:
                permission = "datasets.register"
            elif actor_request is not None and actor_request.state.principal.role != "admin":
                permission = "datasets.validate"
            else:
                permission = "assets.write"
            bind_command_actor(command, actor_request, permission)
        if source_registration:
            command.payload = {**command.payload, "source_registration": True}
            report["registration"] = {"asset_id": asset.id, "node_id": asset.node_id, "path": asset.path,
                                      "created_at": asset.created_at, "actor_user_id": command.actor_user_id,
                                      "input_digest": report["input_digest"]}
        session.add(command)
        report["id"] = command.id
    _write(asset, report)
    session.flush()
    return deepcopy(report)


def _text(value, maximum=2000):
    if not isinstance(value, str) or not value or len(value) > maximum or any(ord(c) < 32 and c not in "\n\t" for c in value):
        raise ValueError("invalid text")
    value.encode("utf-8")
    return value


def _integer(value, minimum=0, maximum=25_000_000):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("invalid integer")
    return value


def _hash(value):
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ValueError("invalid fingerprint")
    return value


def _keys(value, names):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError("invalid result fields")


def _checks(value):
    if not isinstance(value, list) or not 1 <= len(value) <= 64:
        raise ValueError("invalid checks")
    for check in value:
        _keys(check, ("name", "ok", "message"))
        _text(check["name"], 100)
        _text(check["message"])
        if type(check["ok"]) is not bool:
            raise ValueError("invalid check result")
    return deepcopy(value)


def _canonical_path(value):
    _text(value, 4096)
    path = PurePosixPath(value)
    if (not path.is_absolute() or ".." in path.parts
            or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value)):
        raise ValueError("invalid dataset path")
    # Compare remote POSIX spellings without resolving them on the controller.
    # Parent traversal and different symlink targets cannot prove equivalence.
    return "/" + str(path).lstrip("/")


def validated_result(command, value):
    try:
        if len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode()) > MAX_RESULT_BYTES:
            raise ValueError("result too large")
        allowed = {"schema_version", "scope", "path", "profile_id", "source_fingerprint", "checks",
                   "timestamp_contract", "raw_samples", "runtime", "error"}
        if not isinstance(value, dict) or set(value) - allowed:
            raise ValueError("unexpected result fields")
        if value.get("error") is not None:
            _keys(value["error"], ("code", "message"))
            code, message = _text(value["error"]["code"], 100), _text(value["error"]["message"])
            if re.fullmatch(r"[A-Z][A-Z0-9_]*", code) is None:
                raise ValueError("invalid error code")
            return {"error": _error(code, message), "checks": _checks(value["checks"]) if value.get("checks") else []}
        _keys(value, allowed - {"error"})
        if (type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["scope"] != SCOPE
                or _canonical_path(value["path"]) != _canonical_path(command.payload["path"])
                or value["profile_id"] != command.payload["profile_id"]):
            raise ValueError("result does not match command")
        _hash(value["source_fingerprint"])
        checks = _checks(value["checks"])
        required_checks = {"source_metadata", "dataset_timestamps", "raw_samples", "dataset_cache"}
        if not required_checks.issubset({check["name"] for check in checks if check["ok"]}):
            raise ValueError("missing successful dataset checks")
        contract = value["timestamp_contract"]
        _keys(contract, ("schema_version", "status", "scope", "effective_fps", "expected_interval_seconds", "tolerance_s",
            "checked_rows", "checked_pairs", "checked_episodes", "parquet_files", "metadata_sha256", "episodes_sha256",
            "data_manifest_digest", "runtime"))
        if type(contract["schema_version"]) is not int or contract["schema_version"] != 1 or contract["status"] != "verified":
            raise ValueError("timestamp validation did not finish")
        _text(contract["scope"], 256)
        for field in ("effective_fps", "expected_interval_seconds", "tolerance_s"):
            item = contract[field]
            if type(item) not in (int, float) or not math.isfinite(item) or item < 0:
                raise ValueError("invalid timestamp numeric field")
        if not 0 < contract["effective_fps"] <= 1000 or not math.isclose(contract["expected_interval_seconds"], 1 / contract["effective_fps"], rel_tol=1e-10):
            raise ValueError("invalid timestamp frequency")
        declared_fps = (command.payload.get("registered_profile") or {}).get("fps")
        if declared_fps is not None and contract["effective_fps"] != declared_fps:
            raise ValueError("timestamp evidence does not match the registered FPS")
        frames = _integer(contract["checked_rows"], 1)
        episodes = _integer(contract["checked_episodes"], 1, frames)
        if _integer(contract["checked_pairs"]) != frames - episodes:
            raise ValueError("partial timestamp validation")
        _integer(contract["parquet_files"], 1, 20000)
        for key in ("metadata_sha256", "episodes_sha256", "data_manifest_digest"):
            _hash(contract[key])
        timestamp_runtime = contract["runtime"]
        _keys(timestamp_runtime, ("module", "module_path", "module_sha256", "constructor_signature", "validator", "validator_path", "validator_sha256"))
        for key in timestamp_runtime:
            _hash(timestamp_runtime[key]) if key.endswith("sha256") else _text(timestamp_runtime[key], 4096)
        runtime = value["runtime"]
        _keys(runtime, ("python_executable", "python_version", "loader_path", "loader_sha256", "validator_path", "validator_sha256"))
        for key in runtime:
            _hash(runtime[key]) if key.endswith("sha256") else _text(runtime[key], 4096)
        if any(runtime[key] != timestamp_runtime[key] for key in ("validator_path", "validator_sha256")):
            raise ValueError("runtime and timestamp evidence have different validators")
        samples = value["raw_samples"]
        _keys(samples, ("selection", "checked_samples", "sample_indices", "samples"))
        indices = sorted({0, frames // 2, frames - 1})
        if (samples["selection"] != "first_middle_last" or type(samples["checked_samples"]) is not int
                or samples["checked_samples"] != len(indices) or samples["sample_indices"] != indices
                or any(type(index) is not int for index in samples["sample_indices"])
                or not isinstance(samples["samples"], list) or len(samples["samples"]) != len(indices)):
            raise ValueError("invalid bounded samples")
        for index, sample in zip(indices, samples["samples"]):
            _keys(sample, ("index", "fields"))
            if type(sample["index"]) is not int or sample["index"] != index or not isinstance(sample["fields"], dict) or not 1 <= len(sample["fields"]) <= 256:
                raise ValueError("invalid sample fields")
            for key, field in sample["fields"].items():
                _text(key, 256)
                _keys(field, ("shape", "dtype"))
                _text(field["dtype"], 64)
                if not isinstance(field["shape"], list) or len(field["shape"]) > 5:
                    raise ValueError("invalid sample shape")
                for dimension in field["shape"]:
                    _integer(dimension, 1, 1_000_000)
        result = deepcopy(value)
        if any(not check["ok"] for check in checks):
            result["error"] = _error("DATASET_VALIDATION_FAILED")
        return result
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError, UnicodeError):
        return {"error": _error("DATASET_VALIDATION_INVALID_RESULT"), "checks": []}


def acknowledge(session, command, node, status, value):
    # Asset before Command matches asset edits; the caller does not take the
    # generic ACK route's Node lock for this metadata-only operation.
    asset = session.scalar(select(Asset).where(Asset.id == command.payload.get("asset_id")).with_for_update())
    command = session.scalar(select(Command).where(Command.id == command.id).with_for_update().execution_options(populate_existing=True))
    if command.status != "PENDING":
        return
    report = validation_dict(asset, session) if asset is not None else None
    matching = (asset is not None and asset.node_id == node.id and report is not None
                and report.get("id") == command.id and report.get("input_digest") == command.payload.get("input_digest"))
    if not matching or report["status"] == "STALE":
        clean = {"error": _error("DATASET_VALIDATION_INPUT_CHANGED"), "checks": []}
        if matching:
            _write(asset, {**report, "status": "STALE", "error": clean["error"]})
    elif command.payload.get("expires_at", "") <= now():
        clean = {"error": _error("DATASET_VALIDATION_EXPIRED"), "checks": []}
        _write(asset, {**report, "status": "FAILED", "error": clean["error"], "completed_at": now()})
    else:
        clean = validated_result(command, value)
        if status != "APPLIED" and "error" not in clean:
            clean["error"] = _error("DATASET_VALIDATION_FAILED")
        _write(asset, {**report, **{key: deepcopy(clean.get(key)) for key in (
            "source_fingerprint", "timestamp_contract", "raw_samples", "runtime")},
            "checks": clean.get("checks", []), "error": clean.get("error"),
            "status": "FAILED" if clean.get("error") else "SUCCEEDED", "completed_at": now()})
    command.status, command.result, command.acknowledged_at = ("REJECTED" if clean.get("error") else "APPLIED"), clean, now()


def register_routes(app, admin, db, settings):
    def required(session, asset_id, *, lock=False):
        query = select(Asset).where(Asset.id == asset_id)
        asset = session.scalar(query.with_for_update() if lock else query)
        if asset is None or asset.kind != "DATASET":
            raise APIError(404, "NOT_FOUND", "数据集资产不存在")
        return asset

    @app.post("/api/v1/assets/{asset_id}/validations", dependencies=admin, status_code=202)
    def create(asset_id: str, body: ValidationInput, request: Request):
        with db.write() as session:
            require_active_actor(session, request, "datasets.validate")
            asset = required(session, asset_id, lock=True)
            if request.state.principal.role != "admin" and (
                asset.registered_by_user_id is None or asset.registered_by_user_id != request_user_id(request)
            ):
                raise APIError(403, "RESOURCE_ACCESS_DENIED", "只能重新校验自己登记的数据集")
            return queue_validation(session, asset, body.profile_id,
                                    heartbeat_timeout_seconds=settings.heartbeat_timeout_seconds, actor_request=request)

    @app.get("/api/v1/assets/{asset_id}/validation", dependencies=admin)
    def get(asset_id: str):
        with db.read() as session:
            return validation_dict(required(session, asset_id), session)

    @app.post("/api/v1/assets/{asset_id}/validation/cancel", dependencies=admin)
    def cancel(asset_id: str, request: Request):
        with db.write() as session:
            require_active_actor(session, request, "assets.write")
            return cancel_pending_validation(session, required(session, asset_id, lock=True))
