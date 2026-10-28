"""Deploy local OpenPI inference through the existing machine Agent.

Every process mutation is a durable Agent command. A queued request, an open TCP
port, or an offline machine never establish that a model is serving or stopped.
"""

from copy import deepcopy
import json
import math
from pathlib import PurePosixPath
import re
from types import SimpleNamespace
from typing import Literal

from pydantic import Field, field_validator, model_validator
from sqlalchemy import select
from vla_protocol.config_preview import validate_generated_config
from vla_protocol.gpu_policy import gpu_deployment_error, normalize_gpu_policy

from .common import APIError, TERMINAL, now, offset_time, require_path, uid
from .models import Allocation, Asset, Attempt, Command, Node, Profile, Run
from .policy_deployment_models import PolicyDeployment, RESOURCE_STATES
from .profile_rebindings import require_not_pending
from .recipe_writeback import require_no_pending_writeback
from .schemas import InputModel


COMMANDS = {"POLICY_DEPLOY_START", "POLICY_DEPLOY_STATUS", "POLICY_DEPLOY_STOP", "POLICY_DEPLOY_TEST"}
STATUS_INTERVAL_SECONDS = 15
STATUS_STALE_SECONDS = 45
CHECKPOINT_SOURCE_WARNING = {
    "code": "POLICY_DEPLOY_CHECKPOINT_SOURCE_MISMATCH",
    "message": "权重目录不属于所选训练的已记录产物；将继续使用该训练配置，请确认模型结构和归一化参数一致。",
}


class _PortableReceiptMismatch(ValueError):
    pass


class _GpuPolicyReceiptMismatch(ValueError):
    pass


class _ExecutionReceiptMismatch(ValueError):
    pass


class _BackendReceiptMismatch(ValueError):
    pass


class CreateInput(InputModel):
    name: str = Field(min_length=1, max_length=200)
    node_id: str = Field(min_length=1, max_length=36)
    profile_id: str = Field(min_length=1, max_length=200)
    checkpoint_path: str = Field(min_length=1, max_length=4096)
    config_name: str | None = Field(default=None, min_length=1, max_length=255, pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
    inference_backend: Literal["auto", "jax", "pytorch"] = "auto"
    port: int = Field(default=8000, ge=1024, le=65535, strict=True)
    gpu_uuid: str | None = Field(default=None, min_length=1, max_length=200)
    gpu_mode: Literal["exclusive", "shared"] = "exclusive"
    gpu_memory_mb: int | None = Field(default=None, ge=256, le=1048576, strict=True)
    default_prompt: str | None = Field(default=None, max_length=4096)
    source_run_id: str | None = Field(default=None, min_length=1, max_length=36)
    transfer_id: str | None = Field(default=None, min_length=1, max_length=36)
    record_id: str | None = Field(default=None, min_length=1, max_length=36)

    @model_validator(mode="after")
    def gpu_policy(self):
        normalize_gpu_policy(self.gpu_mode, self.gpu_memory_mb)
        return self

    @field_validator("name")
    @classmethod
    def name_text(cls, value):
        if not value.strip() or any(ord(char) < 32 for char in value):
            raise ValueError("部署名称不能为空或包含控制字符")
        return value.strip()


class TestInput(InputModel):
    observation_path: str = Field(min_length=1, max_length=4096)


def _path(value, roots):
    if any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value):
        raise APIError(422, "INVALID_PATH", "路径不能包含控制字符")
    require_path(value, roots)
    return str(PurePosixPath(value))


def _latest_start(session, deployment_id):
    return session.scalar(select(Command).where(Command.type == "POLICY_DEPLOY_START",
        Command.payload["deployment_id"].as_string() == deployment_id)
        .order_by(Command.created_at.desc(), Command.id.desc()).limit(1))


def _pending(session, deployment_id):
    latest = _latest_start(session, deployment_id)
    execution_id = latest.payload.get("execution_id") if latest is not None else None
    return [command for command in session.scalars(select(Command).where(
        Command.type.in_(COMMANDS), Command.status == "PENDING").order_by(Command.created_at, Command.id))
        if command.payload.get("deployment_id") == deployment_id
        and command.payload.get("execution_id") == execution_id
        and (latest is None or command.node_id == latest.node_id and command.created_at >= latest.created_at)]


def deployment_dict(session, row, settings):
    from .checkpoint_transfer_models import CheckpointTransfer, PolicyDeploymentTransfer
    from .deployment_record_models import DeploymentRecordRuntime
    node = session.get(Node, row.node_id)
    connected = node is not None and node.status == "ONLINE" and node.last_seen >= offset_time(-settings.heartbeat_timeout_seconds)
    pending = _pending(session, row.id)
    result = {key: deepcopy(getattr(row, key)) for key in (
        "id", "name", "node_id", "profile_id", "checkpoint_path", "config_name", "inference_backend", "resolved_backend", "port", "gpu_uuid", "gpu_mode", "gpu_memory_mb",
        "default_prompt", "source_run_id", "state", "endpoint", "pid", "logs", "checks", "error",
        "test_result", "created_at", "updated_at")}
    ownership = session.get(DeploymentRecordRuntime, row.id)
    result["record_id"] = ownership.record_id if ownership is not None else None
    result.update(reported_state=row.state, connection_state="ONLINE" if connected else "OFFLINE",
                  status_stale=row.state in RESOURCE_STATES and row.updated_at < offset_time(-STATUS_STALE_SECONDS),
                  pending_command=pending[0].type if pending else None,
                  pending_command_id=pending[0].id if pending else None)
    latest = _latest_start(session, row.id)
    if latest is not None and latest.payload.get("execution_id") is not None:
        result["execution_id"] = latest.payload["execution_id"]
    if (not connected or result["status_stale"]) and row.state in RESOURCE_STATES:
        result["state"] = "UNKNOWN"
    transfer = session.get(PolicyDeploymentTransfer, row.id)
    source_code = None
    frozen = None
    if transfer is not None:
        result["transfer_id"] = transfer.transfer_id
        exported = session.get(CheckpointTransfer, transfer.transfer_id)
        binding = exported.binding_snapshot if exported is not None else None
        frozen = binding.get("source_training") if isinstance(binding, dict) else None
        source_code = frozen.get("code_expectation") if isinstance(frozen, dict) else None
    elif row.source_run_id:
        start = session.scalar(select(Command).where(Command.node_id == row.node_id,
            Command.type == "POLICY_DEPLOY_START", Command.payload["deployment_id"].as_string() == row.id)
            .order_by(Command.created_at, Command.id).limit(1))
        frozen = start.payload.get("source_training") if start is not None else None
        source_code = frozen.get("code_expectation") if isinstance(frozen, dict) else None
    result["config_source"] = "training" if row.source_run_id or transfer is not None else "existing"
    result["source_training_recipe"] = _snapshot_recipe_name(source_code)
    result["warnings"] = _source_warnings(frozen)
    if transfer is not None and exported is not None:
        from .checkpoint_transfers import _transfer_warnings
        result["warnings"] = _transfer_warnings(session, exported)
    if result["config_source"] == "training" and not result["warnings"]:
        receipts = select(Command.result["warnings"]).where(Command.node_id == row.node_id,
            Command.type.in_({"POLICY_DEPLOY_START", "POLICY_DEPLOY_STATUS"}), Command.status == "APPLIED",
            Command.payload["deployment_id"].as_string() == row.id)
        for warnings in session.scalars(receipts):
            result["warnings"] = _source_warnings({"warnings": warnings})
            if result["warnings"]:
                break
    return result


def _enqueue(session, row, kind, payload=None):
    values = {"deployment_id": row.id, **deepcopy(payload or {})}
    if kind != "POLICY_DEPLOY_START":
        latest = _latest_start(session, row.id)
        if latest is not None and latest.payload.get("execution_id") is not None:
            values["execution_id"] = latest.payload["execution_id"]
    command = Command(id=uid(), node_id=row.node_id, type=kind, status="PENDING", created_at=now(),
                      payload=values)
    session.add(command)
    session.flush()
    return command


def queue_status_checks(session, node, settings):
    """Keep serving evidence current even when no operator has a page open.

    Called under the scheduler's existing node lock. It only queues bounded
    probes and never treats lack of response as process-exit evidence.
    """
    if node.status != "ONLINE" or node.last_seen < offset_time(-settings.heartbeat_timeout_seconds):
        return
    cutoff = offset_time(-STATUS_INTERVAL_SECONDS)
    for row in session.scalars(select(PolicyDeployment).where(PolicyDeployment.node_id == node.id,
            PolicyDeployment.state.in_(RESOURCE_STATES), PolicyDeployment.updated_at <= cutoff).with_for_update()):
        if not _pending(session, row.id):
            _enqueue(session, row, "POLICY_DEPLOY_STATUS")


def _error(code, message):
    return {"code": code, "message": message}


def _gpu_policy_matches(value, expected):
    if (not isinstance(value, dict) or set(value) != {"mode", "memory_mb", "reserve_mb"}
            or type(value.get("reserve_mb")) is not int):
        return False
    try:
        return normalize_gpu_policy(value["mode"], value["memory_mb"]) == value == expected
    except (TypeError, ValueError):
        return False


def _clean_result(value, row, portable=None, execution_id=None):
    if not isinstance(value, dict):
        raise ValueError("Agent result is not an object")
    clean = {}
    if execution_id is not None:
        if value.get("execution_id") != execution_id:
            raise _ExecutionReceiptMismatch("Agent did not confirm this deployment execution")
        clean["execution_id"] = execution_id
    state = value.get("state")
    if state is not None:
        if state not in {"STARTING", "RUNNING", "STOPPED", "FAILED"}:
            raise ValueError("invalid deployment state")
        clean["state"] = state
    requested_backend = getattr(row, "inference_backend", "auto")
    resolved_backend = value.get("resolved_backend")
    if "resolved_backend" in value:
        if not isinstance(resolved_backend, str) or resolved_backend not in {"jax", "pytorch"}:
            raise _BackendReceiptMismatch("Agent did not return a supported inference backend")
        if requested_backend != "auto" and resolved_backend != requested_backend:
            raise _BackendReceiptMismatch("Agent inference backend differs from the deployment request")
        if getattr(row, "resolved_backend", None) not in {None, resolved_backend}:
            raise _BackendReceiptMismatch("Agent inference backend changed within the same execution")
        clean["resolved_backend"] = resolved_backend
    if requested_backend != "auto" and state in {"STARTING", "RUNNING"} and resolved_backend != requested_backend:
        raise _BackendReceiptMismatch("Agent did not confirm the requested inference backend")
    expected_gpu_policy = normalize_gpu_policy(row.gpu_mode, row.gpu_memory_mb)
    if row.gpu_mode == "shared" and state in {"STARTING", "RUNNING"} and not _gpu_policy_matches(value.get("gpu_policy"), expected_gpu_policy):
        raise _GpuPolicyReceiptMismatch("Agent did not confirm the frozen shared GPU policy")
    if "gpu_policy" in value:
        if not _gpu_policy_matches(value["gpu_policy"], expected_gpu_policy):
            raise _GpuPolicyReceiptMismatch("Agent GPU policy differs from the deployment request")
        clean["gpu_policy"] = deepcopy(expected_gpu_policy)
    if "pid" in value:
        pid = value["pid"]
        if pid is not None and (type(pid) is not int or not 0 < pid <= 2**31 - 1):
            raise ValueError("invalid process identity")
        clean["pid"] = pid
    if "endpoint" in value:
        if value["endpoint"] not in {None, f"ws://127.0.0.1:{row.port}"}:
            raise ValueError("deployment endpoint differs from local binding")
        clean["endpoint"] = value["endpoint"]
    if "logs" in value:
        if not isinstance(value["logs"], str):
            raise ValueError("invalid logs")
        clean["logs"] = value["logs"][-65536:]
    if "checks" in value:
        checks = value["checks"]
        if not isinstance(checks, list) or len(checks) > 64:
            raise ValueError("invalid checks")
        clean["checks"] = []
        for item in checks:
            if (not isinstance(item, dict) or type(item.get("ok")) is not bool
                    or not isinstance(item.get("name"), str) or not 1 <= len(item["name"]) <= 128
                    or not isinstance(item.get("message"), str) or len(item["message"]) > 4096):
                raise ValueError("invalid check")
            clean["checks"].append({key: item[key] for key in ("name", "ok", "message")})
    if state == "RUNNING" and (not clean.get("pid") or clean.get("endpoint") != f"ws://127.0.0.1:{row.port}"
            or not any(item["name"] == "websocket_metadata" and item["ok"] is True for item in clean.get("checks", []))):
        raise ValueError("running requires process and WebSocket metadata evidence")
    if state == "RUNNING" and portable is not None and (
            value.get("portable_digest") != portable.portable_digest
            or value.get("checkpoint_manifest_digest") != portable.manifest_sha256):
        raise _PortableReceiptMismatch("portable configuration or checkpoint receipt differs")
    for key in ("portable_digest", "checkpoint_manifest_digest"):
        if key in value:
            supplied = value[key]
            if not isinstance(supplied, str) or len(supplied) != 64 or any(char not in "0123456789abcdef" for char in supplied):
                raise ValueError("invalid portable receipt digest")
            clean[key] = supplied
    error = value.get("error")
    if error is not None:
        if (not isinstance(error, dict) or not isinstance(error.get("code"), str)
                or not isinstance(error.get("message"), str)):
            raise ValueError("invalid error")
        code = error["code"][:128]
        clean["error"] = (_error("POLICY_DEPLOY_AGENT_UNSUPPORTED", "当前 Agent 尚不支持真机部署，请先在机器资源中更新 Agent")
                          if code == "COMMAND_UNSUPPORTED" else _error(code, error["message"][:4096]))
    else:
        clean["error"] = None
    if "test_result" in value:
        test = value["test_result"]
        if test is not None:
            if (not isinstance(test, dict) or test.get("status") not in {"RUNNING", "SUCCEEDED", "FAILED"}
                    or len(json.dumps(test, allow_nan=False).encode()) > 65536):
                raise ValueError("invalid inference test receipt")
            for key in ("observation_path", "output_path"):
                if key in test and (not isinstance(test[key], str) or not 1 <= len(test[key]) <= 4096
                                    or not test[key].startswith("/") or any(ord(char) < 32 for char in test[key])):
                    raise ValueError("invalid inference result path")
            if "action_shape" in test and (not isinstance(test["action_shape"], list)
                    or not 1 <= len(test["action_shape"]) <= 8
                    or any(type(size) is not int or not 0 < size <= 100000 for size in test["action_shape"])):
                raise ValueError("invalid inference action shape")
            if "latency_ms" in test and (type(test["latency_ms"]) not in (int, float)
                    or not math.isfinite(test["latency_ms"]) or test["latency_ms"] < 0):
                raise ValueError("invalid inference latency")
            if "finite" in test and type(test["finite"]) is not bool:
                raise ValueError("invalid inference finite flag")
            if "error" in test:
                error = test["error"]
                if (not isinstance(error, dict) or not isinstance(error.get("code"), str)
                        or not isinstance(error.get("message"), str)):
                    raise ValueError("invalid inference error")
                test = {**test, "error": _error(error["code"][:128], error["message"][:4096])}
            if test["status"] == "SUCCEEDED" and (test.get("finite") is not True or not test.get("action_shape")):
                raise ValueError("successful inference requires finite action shape")
            test = {key: deepcopy(test[key]) for key in ("status", "observation_path", "output_path", "action_shape",
                    "latency_ms", "finite", "error") if key in test}
        clean["test_result"] = test
    return clean


def acknowledge(session, command, node, status, value):
    # Receipt retry never changes a later status or restarts the process.
    if command.status != "PENDING":
        return
    row = session.scalar(select(PolicyDeployment).where(
        PolicyDeployment.id == command.payload.get("deployment_id"), PolicyDeployment.node_id == node.id).with_for_update())
    if row is None:
        raise APIError(409, "POLICY_DEPLOY_NOT_FOUND", "部署记录不存在")
    latest = _latest_start(session, row.id)
    if latest is not None and (command.created_at < latest.created_at
            or command.payload.get("execution_id") != latest.payload.get("execution_id")):
        command.status = "REJECTED"
        command.result = {"error": _error("POLICY_DEPLOY_EXECUTION_SUPERSEDED", "该命令属于已结束的旧执行，不再改变当前部署状态")}
        command.acknowledged_at = now()
        return
    from .checkpoint_transfer_models import CheckpointTransfer, PolicyDeploymentTransfer
    association = session.get(PolicyDeploymentTransfer, row.id)
    portable = session.get(CheckpointTransfer, association.transfer_id) if association is not None else None
    try:
        if association is not None and isinstance(value, dict) and value.get("state") == "RUNNING" and (portable is None or portable.state != "SUCCEEDED"):
            raise _PortableReceiptMismatch("transfer receipt is unavailable")
        clean = _clean_result(value, row, portable, command.payload.get("execution_id"))
        if status == "APPLIED" and "state" not in clean:
            raise ValueError("missing deployment state")
        if status == "APPLIED" and (warnings := _source_warnings(value)):
            clean["warnings"] = warnings
    except _ExecutionReceiptMismatch:
        clean = {"error": _error("POLICY_DEPLOY_EXECUTION_MISMATCH",
            "Agent 未确认本次执行标识，服务状态待确认；请更新 Agent 后刷新或停止服务")}
        status = "REJECTED"
    except _GpuPolicyReceiptMismatch:
        clean = {"error": _error("POLICY_DEPLOY_GPU_POLICY_MISMATCH",
            "Agent 未确认本次共享 GPU 模式与显存预算，服务状态待确认；请更新 Agent 后刷新或停止服务")}
        status = "REJECTED"
    except _BackendReceiptMismatch:
        clean = {"error": _error("POLICY_DEPLOY_BACKEND_MISMATCH",
            "Agent 未确认所选推理框架，服务状态待确认；请更新 Agent 后刷新或停止服务")}
        status = "REJECTED"
    except _PortableReceiptMismatch:
        clean = {"error": _error("POLICY_DEPLOY_PORTABLE_MISMATCH",
            "目标 Agent 未确认所选模型与推理配置的摘要，服务状态待确认；请更新 Agent 后停止或重新部署")}
        status = "REJECTED"
    except (ValueError, TypeError, OverflowError, UnicodeError):
        clean = {"error": _error("POLICY_DEPLOY_INVALID_RESULT", "Agent 未返回有效部署凭据，请刷新状态或停止服务")}
        status = "REJECTED"
    command.status = status
    command.result = deepcopy(clean)
    command.acknowledged_at = now()
    # STOP is prioritized over older status/test commands in the delivery
    # queue. A delayed receipt must not resurrect a service already stopped.
    if row.observed_command_at is not None and command.created_at < row.observed_command_at:
        return
    row.observed_command_at = command.created_at
    if status == "REJECTED":
        row.error = clean.get("error") or _error("POLICY_DEPLOY_FAILED", "Agent 未完成该部署操作")
        # Only a known unsupported/expired START proves that it did not launch.
        refused = command.type == "POLICY_DEPLOY_START" and row.error["code"] in {
            "POLICY_DEPLOY_AGENT_UNSUPPORTED", "POLICY_DEPLOY_EXPIRED"}
        if clean.get("state") in {"FAILED", "STOPPED"} or refused:
            row.state = clean.get("state", "FAILED")
        elif command.type == "POLICY_DEPLOY_TEST" and clean.get("state") == "RUNNING":
            row.state = "RUNNING"
        else:
            row.state = "UNKNOWN"
    else:
        row.state = clean["state"]
        row.error = clean.get("error")
    for key in ("pid", "endpoint", "logs", "checks", "test_result", "resolved_backend"):
        if key in clean:
            setattr(row, key, deepcopy(clean[key]))
    if row.state in {"FAILED", "STOPPED"}:
        row.endpoint = None
        row.pid = None
    session.flush()
    if any(pending.type == "POLICY_DEPLOY_STOP" for pending in _pending(session, row.id)):
        row.state = "STOPPING"
    row.updated_at = now()


def _snapshot_recipe_name(code):
    """Read provenance without trusting historical client-supplied labels."""
    if not isinstance(code, dict):
        return None
    preflight = code.get("automatic_preflight")
    plan = preflight.get("plan") if isinstance(preflight, dict) else None
    recipe = code.get("recipe")
    planned_recipe = plan.get("recipe") if isinstance(plan, dict) else None
    if recipe is not None and planned_recipe is not None and recipe != planned_recipe:
        return None
    name = recipe if recipe is not None else planned_recipe
    return name if isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,254}", name) else None


def _training_config_name(code):
    """Recover the trace name only from the confirmed training snapshot.

    This is not a registry lookup: the Agent builds inference from the frozen
    training or transferred policy, even if this recipe is absent on the target.
    """
    if not isinstance(code, dict):
        raise APIError(422, "POLICY_DEPLOY_SOURCE_CONFIG_MISSING", "来源训练缺少完整生成配置，无法可靠恢复模型；请使用包含生成配置的训练任务")
    preflight = code.get("automatic_preflight")
    plan = preflight.get("plan") if isinstance(preflight, dict) else None
    if not isinstance(plan, dict):
        raise APIError(422, "POLICY_DEPLOY_SOURCE_CONFIG_MISSING", "来源训练缺少完整生成配置，无法可靠恢复模型；请使用包含生成配置的训练任务")
    if code.get("training_stage") == "value" or plan.get("training_stage") == "value":
        raise APIError(422, "POLICY_DEPLOY_VALUE_UNSUPPORTED", "Value 阶段模型不提供机器人动作策略，请选择 SFT 或 ACP 权重")
    if len(plan.get("datasets", [])) > 1:
        raise APIError(422, "POLICY_DEPLOY_MULTI_DATASET_UNSUPPORTED", "多数据组训练的推理归一化选择尚未支持，请使用已明确单数据组处理配置的权重")
    try:
        generated = validate_generated_config(plan.get("generated_config"))
    except (ValueError, TypeError):
        raise APIError(422, "POLICY_DEPLOY_SOURCE_CONFIG_MISSING", "来源训练缺少完整生成配置，无法可靠恢复模型；请使用包含生成配置的训练任务") from None
    if code.get("confirmed_config_sha256") != generated["sha256"]:
        raise APIError(422, "POLICY_DEPLOY_SOURCE_CONFIG_MISMATCH", "来源训练的生成配置确认凭据不一致，无法部署")
    recipe, planned_recipe = code.get("recipe"), plan.get("recipe")
    if recipe is not None and planned_recipe is not None and recipe != planned_recipe:
        raise APIError(422, "POLICY_DEPLOY_SOURCE_CONFIG_MISMATCH", "来源训练配方与确认配置不一致，无法部署")
    name = _snapshot_recipe_name(code)
    if name is None:
        raise APIError(422, "POLICY_DEPLOY_SOURCE_CONFIG_MISSING", "来源训练缺少有效的冻结配方名称，无法生成推理配置")
    return name


def _source_warnings(source):
    """Canonicalize the one attribution warning; ignore arbitrary Agent text."""
    warnings = source.get("warnings") if isinstance(source, dict) else None
    if isinstance(warnings, list) and any(isinstance(item, dict) and item.get("code") == CHECKPOINT_SOURCE_WARNING["code"]
                                         for item in warnings[:64]):
        return [deepcopy(CHECKPOINT_SOURCE_WARNING)]
    return []


def _source_training(session, body):
    if body.source_run_id is None:
        return None
    run = session.get(Run, body.source_run_id)
    if run is None or run.node_id != body.node_id or run.profile_id != body.profile_id:
        raise APIError(422, "POLICY_DEPLOY_SOURCE_MISMATCH", "来源训练任务必须属于所选机器和运行环境")
    if run.state not in TERMINAL:
        raise APIError(409, "POLICY_DEPLOY_SOURCE_ACTIVE", "来源训练尚未结束，请等待完成后部署权重")
    attempts = list(session.scalars(select(Attempt).where(Attempt.run_id == run.id, Attempt.node_id == run.node_id)))
    attempt_ids = [attempt.id for attempt in attempts]
    checkpoint_roots = [attempt.output_dir for attempt in attempts if attempt.output_dir]
    if attempt_ids:
        checkpoint_roots.extend(asset.path for asset in session.scalars(select(Asset).where(
            Asset.origin_attempt_id.in_(attempt_ids), Asset.node_id == run.node_id,
            Asset.kind == "MODEL", Asset.state != "ARCHIVED")))
    checkpoint_roots = sorted(set(checkpoint_roots))
    warnings = []
    try:
        require_path(body.checkpoint_path, checkpoint_roots)
    except APIError:
        warnings.append(deepcopy(CHECKPOINT_SOURCE_WARNING))
    code = run.code_expectation or {}
    _training_config_name(code)
    result = {"run_id": run.id, "code_expectation": deepcopy(code), "checkpoint_roots": checkpoint_roots}
    if warnings:
        result["warnings"] = warnings
    return result


def _start_profile(session, node, profile_id):
    profile = session.get(Profile, (node.id, profile_id))
    if profile is None or not profile.active:
        raise APIError(422, "PROFILE_NODE_MISMATCH", "运行环境不属于该机器或已停用")
    if profile.adapter != "openpi-sft" or profile.execution_mode != "native":
        raise APIError(422, "POLICY_DEPLOY_PROFILE_UNSUPPORTED", "当前只支持原生 OpenPI 运行环境")
    require_not_pending(session, node.id, profile.id, profile.code_dir)
    require_no_pending_writeback(session, node.id, profile.code_dir)
    return profile


def _stopped_execution_confirmed(session, row, latest):
    if latest is None or row.observed_command_at is None or row.pid is not None or row.endpoint is not None:
        return False
    commands = session.scalars(select(Command).where(Command.node_id == row.node_id,
        Command.type.in_(COMMANDS), Command.status.in_({"APPLIED", "REJECTED"}),
        Command.payload["deployment_id"].as_string() == row.id,
        Command.created_at == row.observed_command_at))
    for command in commands:
        if command.created_at < latest.created_at or command.payload.get("execution_id") != latest.payload.get("execution_id"):
            continue
        result = command.result or {}
        if result.get("state") == row.state:
            return True
        if row.state == "FAILED" and command.type == "POLICY_DEPLOY_START" and (result.get("error") or {}).get("code") in {
                "POLICY_DEPLOY_AGENT_UNSUPPORTED", "POLICY_DEPLOY_EXPIRED"}:
            return True
    return False


def _admit_resources(session, node, body):
    from .scheduler import gpu_available

    active = list(session.scalars(select(PolicyDeployment).where(
        PolicyDeployment.node_id == node.id, PolicyDeployment.state.in_(RESOURCE_STATES))))
    if any(row.port == body.port for row in active):
        raise APIError(409, "POLICY_DEPLOY_PORT_BUSY", "该端口已有部署或尚未确认停止，请选择其他端口")
    reserved = {row.gpu_uuid for row in session.scalars(select(Allocation).where(Allocation.node_id == node.id))}
    reserved.update(row.gpu_uuid for row in active if row.gpu_uuid)
    capabilities = (node.metrics or {}).get("capabilities")
    if getattr(body, "inference_backend", "auto") != "auto" and (not isinstance(capabilities, list)
            or "POLICY_DEPLOY_BACKEND_V1" not in capabilities):
        raise APIError(409, "POLICY_DEPLOY_BACKEND_UNSUPPORTED", "目标 Agent 尚不支持指定 JAX / PyTorch 推理框架，请先更新 Agent")
    if body.gpu_mode == "shared" and (not isinstance(capabilities, list)
            or "POLICY_DEPLOY_SHARED_GPU_V1" not in capabilities):
        raise APIError(409, "POLICY_DEPLOY_SHARED_GPU_UNSUPPORTED", "目标 Agent 尚不支持共享 GPU 部署，请先更新 Agent")
    def admission_error(gpu):
        if body.gpu_mode == "exclusive":
            return None if gpu_available(gpu) else _error("GPU_UNAVAILABLE", "GPU 未满足独占空闲要求")
        return gpu_deployment_error(gpu, body.gpu_mode, body.gpu_memory_mb)
    available = [gpu for gpu in node.gpus if gpu.get("uuid") not in reserved and admission_error(gpu) is None]
    gpu_uuid = body.gpu_uuid
    if gpu_uuid is not None and not any(gpu.get("uuid") == gpu_uuid for gpu in available):
        selected = next((gpu for gpu in node.gpus if gpu.get("uuid") == gpu_uuid), None)
        if body.gpu_mode == "shared" and selected is not None and gpu_uuid not in reserved:
            error = admission_error(selected)
            if error is not None:
                raise APIError(409, "POLICY_DEPLOY_" + error["code"], error["message"])
        raise APIError(409, "POLICY_DEPLOY_GPU_UNAVAILABLE", "所选 GPU 不存在、正在使用或缺少可靠空闲信息")
    if gpu_uuid is None and node.gpus:
        if not available:
            raise APIError(409, "POLICY_DEPLOY_GPU_UNAVAILABLE", "当前没有可用 GPU，请等待训练或其他部署释放资源")
        gpu_uuid = available[0]["uuid"]
    if body.gpu_mode == "shared" and gpu_uuid is None:
        raise APIError(409, "POLICY_DEPLOY_GPU_UNAVAILABLE", "共享部署需要已确认显存及进程类型的 GPU")
    if gpu_uuid is None and session.scalar(select(Run.id).where(
            Run.node_id == node.id, Run.state.not_in(TERMINAL)).limit(1)):
        raise APIError(409, "POLICY_DEPLOY_MACHINE_BUSY", "该机器仍有训练任务，请等待任务完成")
    return gpu_uuid


def register_routes(app, admin, db, require_available_node, settings):
    def online(session, node_id):
        node = require_available_node(session, node_id)
        if node.last_seen < offset_time(-settings.heartbeat_timeout_seconds):
            raise APIError(409, "NODE_OFFLINE", "机器心跳已过期，无法执行部署操作")
        return node

    def required(session, deployment_id):
        row = session.get(PolicyDeployment, deployment_id)
        if row is None:
            raise APIError(404, "NOT_FOUND", "部署记录不存在")
        # Every writer takes node then deployment, matching Agent ACK and the
        # scheduler lock order on PostgreSQL as well as SQLite.
        online(session, row.node_id)
        return session.scalar(select(PolicyDeployment).where(PolicyDeployment.id == deployment_id)
                              .with_for_update().execution_options(populate_existing=True))

    @app.get("/api/v1/policy-deployments", dependencies=admin)
    def list_deployments():
        with db.read() as session:
            return [deployment_dict(session, row, settings) for row in session.scalars(select(PolicyDeployment)
                    .order_by(PolicyDeployment.created_at.desc(), PolicyDeployment.id.desc()))]

    @app.get("/api/v1/policy-deployments/{deployment_id}", dependencies=admin)
    def get_deployment(deployment_id: str):
        with db.read() as session:
            row = session.get(PolicyDeployment, deployment_id)
            if row is None:
                raise APIError(404, "NOT_FOUND", "部署记录不存在")
            return deployment_dict(session, row, settings)

    @app.post("/api/v1/policy-deployments", dependencies=admin, status_code=202)
    def create_deployment(body: CreateInput):
        with db.write() as session:
            node = online(session, body.node_id)
            profile = _start_profile(session, node, body.profile_id)
            checkpoint_path = _path(body.checkpoint_path, node.roots)
            portable = None
            if body.transfer_id is not None:
                from .checkpoint_transfer_models import CheckpointTransfer
                from .checkpoint_transfers import portable_deployment
                portable = portable_deployment(session, body, settings)
                source = None
                transfer = session.get(CheckpointTransfer, body.transfer_id)
                snapshot = (transfer.binding_snapshot or {}).get("source_training") or {}
                config_name = _training_config_name(snapshot.get("code_expectation"))
            else:
                source = _source_training(session, body)
                config_name = _training_config_name(source["code_expectation"]) if source is not None else body.config_name
            if config_name is None:
                raise APIError(422, "POLICY_DEPLOY_CONFIG_REQUIRED", "未关联训练任务的权重需要选择目标运行环境中已有的推理配置")
            from .deployment_record_models import DeploymentRecordRuntime
            from .deployment_records import create_record, record_runtime, require_preparation_record, transfer_record
            record = None
            if portable is not None:
                record = transfer_record(session, transfer)
                if body.record_id is not None:
                    explicit = require_preparation_record(session, body.record_id)
                    if explicit.id != record.id:
                        raise APIError(422, "DEPLOYMENT_RECORD_CONFLICT", "所选同步与部署记录的明确归属不一致")
                    record = explicit
                elif record_runtime(session, record.id) is not None:
                    # Legacy clients may reuse an exact verified artifact. Its
                    # owning workflow remains unchanged; the new runtime has a
                    # separate parent and keeps only the existing selected FK.
                    record = None
                else:
                    record = require_preparation_record(session, record.id)
            elif body.record_id is not None:
                record = require_preparation_record(session, body.record_id)
            gpu_uuid = _admit_resources(session, node, body)
            timestamp = now()
            row = PolicyDeployment(id=uid(), name=body.name, node_id=node.id, profile_id=profile.id,
                checkpoint_path=checkpoint_path, config_name=config_name, port=body.port, gpu_uuid=gpu_uuid,
                inference_backend=body.inference_backend, resolved_backend=None,
                gpu_mode=body.gpu_mode, gpu_memory_mb=body.gpu_memory_mb,
                default_prompt=body.default_prompt, source_run_id=body.source_run_id, state="PENDING",
                logs="", checks=[], created_at=timestamp, updated_at=timestamp)
            session.add(row)
            session.flush()
            if record is None:
                record = create_record(session, row.id, body.name, row.created_at)
            record.name = body.name
            session.add(DeploymentRecordRuntime(deployment_id=row.id, record_id=record.id))
            session.flush()
            if portable is not None:
                from .checkpoint_transfer_models import PolicyDeploymentTransfer
                session.add(PolicyDeploymentTransfer(deployment_id=row.id, transfer_id=body.transfer_id))
            payload = {key: getattr(row, key) for key in ("profile_id", "checkpoint_path", "config_name", "inference_backend", "port", "gpu_uuid", "gpu_mode", "gpu_memory_mb", "default_prompt")}
            payload["expires_at"] = offset_time(120)
            if source is not None:
                payload["source_training"] = source
            if portable is not None:
                payload["portable_policy"] = portable
            _enqueue(session, row, "POLICY_DEPLOY_START", payload)
            return deployment_dict(session, row, settings)

    @app.post("/api/v1/policy-deployments/{deployment_id}/start", dependencies=admin, status_code=202)
    def restart_deployment(deployment_id: str):
        from .checkpoint_transfer_models import PolicyDeploymentTransfer

        with db.write() as session:
            row = required(session, deployment_id)
            node = session.get(Node, row.node_id)
            capabilities = (node.metrics or {}).get("capabilities")
            if not isinstance(capabilities, list) or "POLICY_DEPLOY_RESTART_V1" not in capabilities:
                raise APIError(409, "POLICY_DEPLOY_RESTART_UNSUPPORTED", "目标 Agent 尚不支持安全重启，请先更新 Agent 并等待新心跳")
            latest = _latest_start(session, row.id)
            pending = _pending(session, row.id)
            if (row.state == "PENDING" and latest is not None and latest.payload.get("execution_id") is not None
                    and latest.status == "PENDING" and len(pending) == 1 and pending[0].id == latest.id):
                return deployment_dict(session, row, settings)
            if row.state not in {"STOPPED", "FAILED"}:
                raise APIError(409, "POLICY_DEPLOY_NOT_RESTARTABLE", "仅已确认停止或失败的部署可以重新启动；请先刷新或停止当前服务")
            if not _stopped_execution_confirmed(session, row, latest):
                raise APIError(409, "POLICY_DEPLOY_STOP_UNCONFIRMED", "尚未确认旧执行已经停止，请先刷新状态或停止服务")
            if any(command.created_at > row.observed_command_at for command in pending):
                raise APIError(409, "POLICY_DEPLOY_BUSY", "部署仍有待确认的操作，请等待结果后重新启动")
            profile = _start_profile(session, node, row.profile_id)
            _path(row.checkpoint_path, node.roots)
            payload = {key: getattr(row, key) for key in ("profile_id", "checkpoint_path", "config_name", "inference_backend", "port",
                "gpu_uuid", "gpu_mode", "gpu_memory_mb", "default_prompt")}
            association = session.get(PolicyDeploymentTransfer, row.id)
            if association is not None:
                from .checkpoint_transfers import portable_deployment
                portable = portable_deployment(session, SimpleNamespace(node_id=row.node_id, profile_id=row.profile_id,
                    checkpoint_path=row.checkpoint_path, source_run_id=row.source_run_id, transfer_id=association.transfer_id), settings)
                if portable != latest.payload.get("portable_policy"):
                    raise APIError(422, "POLICY_DEPLOY_SOURCE_CONFIG_MISMATCH", "同步权重或推理配置与原部署绑定不一致，无法重启")
                payload["portable_policy"] = portable
            elif row.source_run_id is not None:
                frozen = latest.payload.get("source_training")
                if not isinstance(frozen, dict) or frozen.get("run_id") != row.source_run_id:
                    raise APIError(422, "POLICY_DEPLOY_SOURCE_CONFIG_MISSING", "原部署缺少可恢复的冻结训练配置，无法重启")
                payload["config_name"] = _training_config_name(frozen.get("code_expectation"))
                if frozen["code_expectation"].get("code_dir") != profile.code_dir:
                    raise APIError(422, "POLICY_DEPLOY_SOURCE_MISMATCH", "当前运行环境代码目录与原部署冻结配置不一致")
                payload["source_training"] = deepcopy(frozen)
            elif latest.payload.get("source_training") is not None or latest.payload.get("portable_policy") is not None:
                raise APIError(422, "POLICY_DEPLOY_SOURCE_CONFIG_MISMATCH", "原部署来源绑定不一致，无法重启")
            payload["gpu_uuid"] = _admit_resources(session, node, row)
            payload.update(execution_id=uid(), expires_at=offset_time(120))
            for command in pending:
                command.status = "REJECTED"
                command.result = {"error": _error("POLICY_DEPLOY_EXECUTION_SUPERSEDED", "旧执行已确认停止，该待执行命令已由重启替代")}
                command.acknowledged_at = now()
            row.state, row.error, row.pid, row.endpoint, row.test_result = "PENDING", None, None, None, None
            row.resolved_backend = None
            row.logs, row.checks = "", []
            row.gpu_uuid, row.config_name = payload["gpu_uuid"], payload["config_name"]
            row.updated_at = now()
            started = _enqueue(session, row, "POLICY_DEPLOY_START", payload)
            row.observed_command_at = started.created_at
            return deployment_dict(session, row, settings)

    @app.post("/api/v1/policy-deployments/{deployment_id}/refresh", dependencies=admin, status_code=202)
    def refresh_deployment(deployment_id: str):
        with db.write() as session:
            row = required(session, deployment_id)
            if not _pending(session, row.id):
                _enqueue(session, row, "POLICY_DEPLOY_STATUS")
            return deployment_dict(session, row, settings)

    @app.post("/api/v1/policy-deployments/{deployment_id}/stop", dependencies=admin, status_code=202)
    def stop_deployment(deployment_id: str):
        with db.write() as session:
            row = required(session, deployment_id)
            pending = _pending(session, row.id)
            if row.state not in {"STOPPED", "FAILED"} or pending:
                if not any(command.type == "POLICY_DEPLOY_STOP" for command in pending):
                    _enqueue(session, row, "POLICY_DEPLOY_STOP")
                row.state, row.updated_at = "STOPPING", now()
            return deployment_dict(session, row, settings)

    @app.post("/api/v1/policy-deployments/{deployment_id}/test", dependencies=admin, status_code=202)
    def test_deployment(deployment_id: str, body: TestInput):
        with db.write() as session:
            row = required(session, deployment_id)
            node = session.get(Node, row.node_id)
            if row.state != "RUNNING":
                raise APIError(409, "POLICY_DEPLOY_NOT_RUNNING", "推理服务尚未通过就绪检查，请先刷新状态")
            if _pending(session, row.id) or (row.test_result or {}).get("status") == "RUNNING":
                raise APIError(409, "POLICY_DEPLOY_BUSY", "部署仍有待完成的操作或测试，请等待结果")
            observation = _path(body.observation_path, node.roots)
            _enqueue(session, row, "POLICY_DEPLOY_TEST", {"observation_path": observation})
            return deployment_dict(session, row, settings)
