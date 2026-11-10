"""Human operation metadata, without request bodies, secrets or polling noise."""

from dataclasses import dataclass
import json
import logging
import re
import time

import anyio
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request

from .authorization import request_route_path
from .common import uid


logger = logging.getLogger(__name__)
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]{1,200}\Z")


@dataclass(frozen=True)
class Operation:
    action: str
    label: str
    category: str
    target_type: str
    parameter: str | None = None


OPERATIONS = {}
ACTIONS = {}


def _operation(action, label, category, target_type, method, path, parameter=None):
    value = Operation(action, label, category, target_type, parameter)
    OPERATIONS[(method, "/api/v1" + path)] = value
    ACTIONS[action] = value


for action, label, category, target, method, path, parameter in (
    ("auth.login", "登录", "auth", "account", "POST", "/auth/login", None),
    ("auth.logout", "退出登录", "auth", "account", "POST", "/auth/logout", None),
    ("auth.password", "修改本人密码", "auth", "account", "POST", "/auth/password", None),
    ("auth.reauth", "确认敏感操作身份", "auth", "account", "POST", "/auth/reauth", None),
    ("users.create", "创建用户", "users", "user", "POST", "/users", None),
    ("users.update", "修改用户信息或权限", "users", "user", "PATCH", "/users/{user_id}", "user_id"),
    ("users.reset_password", "重置用户密码", "users", "user", "POST", "/users/{user_id}/reset-password", "user_id"),
    ("assets.register", "登记资产", "assets", "asset", "POST", "/assets", None),
    ("assets.archive", "删除资产登记", "assets", "asset", "POST", "/assets/{asset_id}/archive", "asset_id"),
    ("assets.update_profile", "修改数据集特征", "assets", "asset", "PATCH", "/assets/{asset_id}/dataset-profile", "asset_id"),
    ("assets.bind_norm", "修改归一化绑定", "assets", "asset", "PUT", "/assets/{asset_id}/norm-binding", "asset_id"),
    ("assets.validate", "发起数据集校验", "assets", "asset", "POST", "/assets/{asset_id}/validations", "asset_id"),
    ("assets.cancel_validation", "取消数据集校验", "assets", "asset", "POST", "/assets/{asset_id}/validation/cancel", "asset_id"),
    ("datasets.scan", "扫描数据集目录", "assets", "dataset_scan", "POST", "/nodes/{node_id}/dataset-scans", None),
    ("datasets.register_batch", "批量登记数据集", "assets", "dataset_scan", "POST", "/nodes/{node_id}/dataset-scans/{scan_id}/register", "scan_id"),
    ("dataset_templates.save", "保存个人数据集格式模板", "templates", "dataset_template", "POST", "/dataset-format-templates", None),
    ("dataset_templates.update", "修改个人数据集格式模板", "templates", "dataset_template", "PUT", "/dataset-format-templates/{template_id}", "template_id"),
    ("dataset_templates.sharing", "更改数据集格式模板共享范围", "templates", "dataset_template", "PUT", "/dataset-format-templates/{template_id}/sharing", "template_id"),
    ("dataset_templates.delete", "删除个人数据集格式模板", "templates", "dataset_template", "DELETE", "/dataset-format-templates/{template_id}", "template_id"),
    ("queries.paths", "查询机器路径", "queries", "query", "POST", "/nodes/{node_id}/path-suggestions", None),
    ("queries.dataset_fields", "读取数据集字段", "queries", "query", "POST", "/nodes/{node_id}/dataset-inspections", None),
    ("queries.recipes", "查询机器配方", "queries", "query", "POST", "/nodes/{node_id}/recipe-suggestions", None),
    ("queries.recipe_details", "读取机器配方详情", "queries", "query", "POST", "/nodes/{node_id}/recipe-details", None),
    ("queries.processors", "查询数据处理器", "queries", "query", "POST", "/nodes/{node_id}/data-processors", None),
    ("queries.inference_configs", "查询推理配置", "queries", "query", "POST", "/nodes/{node_id}/inference-config-suggestions", None),
    ("transfers.create", "创建资产同步任务", "transfers", "asset_transfer", "POST", "/asset-transfers", None),
    ("transfers.retry", "重试资产同步", "transfers", "asset_transfer", "POST", "/asset-transfers/{transfer_id}/retry", "transfer_id"),
    ("transfers.cancel", "取消资产同步", "transfers", "asset_transfer", "POST", "/asset-transfers/{transfer_id}/cancel", "transfer_id"),
    ("transfers.delete", "删除资产同步记录", "transfers", "asset_transfer", "DELETE", "/asset-transfers/{transfer_id}", "transfer_id"),
    ("transfers.create_batch", "批量创建资产同步", "transfers", "asset_transfer_batch", "POST", "/asset-transfer-batches", None),
    ("runs.preflight", "发起训练预检", "training", "run_preview", "POST", "/run-previews", None),
    ("runs.create", "创建训练任务", "training", "run", "POST", "/runs", None),
    ("runs.cancel", "取消训练任务", "training", "run", "POST", "/runs/{run_id}/cancel", "run_id"),
    ("runs.delete", "删除训练记录", "training", "run", "DELETE", "/runs/{run_id}", "run_id"),
    ("resources.reserve", "预留 GPU", "resources", "node", "PUT", "/nodes/{node_id}/gpu-reservations", "node_id"),
    ("resources.release", "释放 GPU 预留", "resources", "node", "DELETE", "/nodes/{node_id}/gpu-reservations/{gpu_uuid}", "node_id"),
    ("deployments.create", "创建推理部署", "deployments", "policy_deployment", "POST", "/policy-deployments", None),
    ("deployments.refresh", "刷新推理部署状态", "deployments", "policy_deployment", "POST", "/policy-deployments/{deployment_id}/refresh", "deployment_id"),
    ("deployments.start", "启动推理服务", "deployments", "policy_deployment", "POST", "/policy-deployments/{deployment_id}/start", "deployment_id"),
    ("deployments.stop", "停止推理服务", "deployments", "policy_deployment", "POST", "/policy-deployments/{deployment_id}/stop", "deployment_id"),
    ("deployments.test", "测试推理服务", "deployments", "policy_deployment", "POST", "/policy-deployments/{deployment_id}/test", "deployment_id"),
    ("deployments.delete_record", "删除部署记录", "deployments", "deployment_record", "DELETE", "/deployment-records/{record_id}", "record_id"),
    ("checkpoints.transfer", "创建模型部署同步", "transfers", "checkpoint_transfer", "POST", "/checkpoint-transfers", None),
    ("checkpoints.retry", "重试模型部署同步", "transfers", "checkpoint_transfer", "POST", "/checkpoint-transfers/{transfer_id}/retry", "transfer_id"),
    ("checkpoints.cancel", "取消模型部署同步", "transfers", "checkpoint_transfer", "POST", "/checkpoint-transfers/{transfer_id}/cancel", "transfer_id"),
    ("drafts.save", "保存个人训练草稿", "drafts", "training_draft", "PUT", "/training-draft", None),
    ("drafts.clear", "清空个人训练草稿", "drafts", "training_draft", "DELETE", "/training-draft", None),
    ("training_templates.create", "保存个人训练模板", "templates", "training_template", "POST", "/training-templates", None),
    ("training_templates.update", "修改个人训练模板", "templates", "training_template", "PUT", "/training-templates/{template_id}", "template_id"),
    ("training_templates.sharing", "更改训练模板共享范围", "templates", "training_template", "PUT", "/training-templates/{template_id}/sharing", "template_id"),
    ("training_templates.delete", "删除个人训练模板", "templates", "training_template", "DELETE", "/training-templates/{template_id}", "template_id"),
    ("recipes.create", "登记公共配方", "recipes", "recipe", "POST", "/recipes", None),
    ("recipes.revise", "修订公共配方", "recipes", "recipe", "POST", "/recipes/{recipe_id}/revisions", "recipe_id"),
    ("recipes.writeback", "回写机器配方代码", "recipes", "recipe_writeback", "POST", "/recipe-revisions/{revision_id}/writeback", None),
    ("machines.agent_config", "生成 Agent 接入配置", "machines", "agent_config", "POST", "/agent-configs", None),
    ("machines.ssh_test", "测试 SSH 连接", "machines", "ssh_connection", "POST", "/ssh-connections/test", None),
    ("machines.ssh_connect", "建立 SSH 连接", "machines", "ssh_connection", "POST", "/ssh-connections/connect", None),
    ("machines.ssh_retest", "重新测试 SSH 连接", "machines", "ssh_connection", "POST", "/ssh-connections/{connection_id}/test", "connection_id"),
    ("queries.ssh_paths", "查询 SSH 接入路径", "queries", "ssh_connection", "POST", "/ssh-connections/{connection_id}/path-suggestions", "connection_id"),
    ("terminal.create_session", "申请机器终端会话", "machines", "terminal", "POST", "/nodes/{node_id}/terminal-sessions", None),
    ("machines.storage", "修改机器存储目录", "machines", "node", "POST", "/nodes/{node_id}/storage-configurations", "node_id"),
    ("machines.rebind_profile", "切换训练环境", "machines", "node", "POST", "/nodes/{node_id}/profile-rebindings", "node_id"),
    ("machines.add_profile", "添加训练环境", "machines", "node", "POST", "/nodes/{node_id}/profiles", "node_id"),
    ("machines.rename", "修改机器名称", "machines", "node", "PATCH", "/nodes/{node_id}", "node_id"),
    ("machines.disconnect", "断开机器", "machines", "node", "POST", "/nodes/{node_id}/disconnect", "node_id"),
    ("machines.uninstall", "卸载机器 Agent", "machines", "node", "POST", "/nodes/{node_id}/uninstall", "node_id"),
    ("machines.upgrade", "升级机器 Agent", "machines", "node", "POST", "/nodes/{node_id}/upgrade", "node_id"),
    ("machines.onboard", "创建机器接入任务", "machines", "remote_deployment", "POST", "/remote-deployments", None),
    ("machines.retry_onboarding", "重试机器接入任务", "machines", "remote_deployment", "POST", "/remote-deployments/{job_id}/retry", "job_id"),
    ("machines.download_agent", "下载 Agent 安装包", "machines", "agent_package", "GET", "/agent-package", None),
    ("machines.download_logs", "下载机器接入日志", "machines", "remote_deployment", "GET", "/remote-deployments/{job_id}/logs/download", "job_id"),
):
    _operation(action, label, category, target, method, path, parameter)

for action, label, category, target in (
    ("auth.login_failed", "登录失败", "auth", "account"),
    ("auth.reauth_failed", "敏感操作身份确认失败", "auth", "account"),
    ("auth.password_failed", "修改本人密码失败", "auth", "account"),
    ("auth.sessions_revoked", "撤销登录会话", "auth", "user"),
    ("terminal.open", "打开机器终端", "machines", "terminal"),
    ("terminal.open_failed", "机器终端连接失败", "machines", "terminal"),
    ("terminal.close", "关闭机器终端", "machines", "terminal"),
):
    ACTIONS[action] = Operation(action, label, category, target)

FALLBACK = Operation("api.mutation", "API 操作", "api", "resource")


def _text(value, maximum=200):
    if not isinstance(value, str):
        return None
    return "".join(character if ord(character) >= 32 and not 127 <= ord(character) <= 159 else " "
                   for character in value)[:maximum]


def _identifier(value):
    return value if isinstance(value, str) and IDENTIFIER.fullmatch(value) else None


def operation_for(request, action=None):
    if action in ACTIONS:
        return ACTIONS[action]
    method, path = getattr(request, "method", "WS"), request_route_path(request)
    if (method, path) in OPERATIONS:
        return OPERATIONS[(method, path)]
    if method == "GET":
        return Operation("api.read", "访问资源", "api", "resource")
    return FALLBACK


def prime_context(db, request):
    """Read only an existing object's display label after authentication.

    Physical record deletion and large responses cannot supply a name later.
    Never load configuration bodies, passwords or credentials for auditing.
    The read closes before the business handler takes its transaction locks.
    """
    if request.method in {"GET", "HEAD", "OPTIONS"}:
        return
    spec = operation_for(request)
    target = _identifier(request.scope.get("path_params", {}).get(spec.parameter)) if spec.parameter else None
    if not target:
        return
    from sqlalchemy import func, select
    from .asset_transfer_models import AssetTransfer
    from .auth_models import User
    from .checkpoint_transfer_models import CheckpointTransfer
    from .dataset_template_models import DatasetFormatTemplate
    from .deployment_record_models import DeploymentRecord
    from .models import Asset, Node, Run, TrainingTemplate
    from .policy_deployment_models import PolicyDeployment
    from .recipe_models import SavedRecipe
    models = {"asset": Asset, "run": Run, "node": Node, "user": User,
              "asset_transfer": AssetTransfer, "checkpoint_transfer": CheckpointTransfer,
              "policy_deployment": PolicyDeployment, "deployment_record": DeploymentRecord,
              "recipe": SavedRecipe, "training_template": TrainingTemplate,
              "dataset_template": DatasetFormatTemplate}
    model = models.get(spec.target_type)
    if model is None:
        return
    column = func.coalesce(Node.display_name, Node.name) if model is Node else (
        User.display_name if model is User else model.name)
    query = select(column).where(model.id == target)
    if model in (TrainingTemplate, DatasetFormatTemplate):
        principal = getattr(request.state, "principal", None)
        if principal is None or principal.legacy:
            return
        query = query.where(model.user_id == principal.user_id)
    with db.read() as session:
        label = session.scalar(query)
    if _text(label):
        request.state.operation_target_name = _text(label)


def context_details(request, action, details=None):
    """Enrich trusted audit metadata. Never read request bodies or query values."""
    result = dict(details or {})
    if request is None:
        spec = ACTIONS.get(action, FALLBACK)
    else:
        spec = operation_for(request, action)
    result.update(operation_label=spec.label, category=spec.category, target_type=spec.target_type)
    target = _identifier(result.get("target_id")) or _identifier(result.get("target_user_id")) or _identifier(result.get("terminal_id"))
    if target:
        result["target_id"] = target
    if not result.get("target_name") and result.get("target_username"):
        result["target_name"] = _text(result["target_username"])
    if request is not None:
        parameters = request.scope.get("path_params", {})
        target = (_identifier(parameters.get(spec.parameter)) if spec.parameter else None) or target
        if target:
            result["target_id"] = target
        if not result.get("target_name") and getattr(request.state, "operation_target_name", None):
            result["target_name"] = request.state.operation_target_name
        node = _identifier(parameters.get("node_id"))
        if node:
            result["node_id"] = node
        request_id = getattr(request.state, "operation_request_id", None)
        if request_id:
            result["request_id"] = request_id
        started = getattr(request.state, "operation_started", None)
        if started is not None:
            result["duration_ms"] = min(2**31 - 1, max(0, int((time.monotonic() - started) * 1000)))
    return result


def response_details(request, payload, status):
    """Project a bounded JSON response into a fixed metadata allowlist."""
    spec = operation_for(request)
    details = context_details(request, spec.action)
    if not isinstance(payload, dict):
        return details
    if status >= 400:
        error = payload.get("error")
        if isinstance(error, dict) and _identifier(error.get("code")):
            details["error_code"] = error["code"]
        return details
    if not spec.parameter:
        for field in ("id", "run_id", "transfer_id", "scan_id", "operation_id", "connection_id", "terminal_id", "session_id", "template_id"):
            if _identifier(payload.get(field)):
                details["target_id"] = payload[field]
                break
    for field in ("name", "display_name"):
        value = _text(payload.get(field))
        if value:
            details["target_name"] = value
            break
    if isinstance(payload.get("kind"), str) and payload["kind"] in {"DATASET", "MODEL", "NORM"}:
        details["asset_kind"] = payload["kind"]
    for field in ("node_id", "template_id"):
        if _identifier(payload.get(field)):
            details[field] = payload[field]
    for field in ("created_count", "skipped_count", "count"):
        value = payload.get(field)
        if type(value) is int and 0 <= value <= 2**31 - 1:
            details[field] = value
    if spec.action == "transfers.create_batch" and isinstance(payload.get("transfers"), list):
        transfers = payload["transfers"]
        details["count"] = min(len(transfers), 2**31 - 1)
        identifiers = [value["id"] for value in transfers if isinstance(value, dict) and _identifier(value.get("id"))]
        if identifiers:
            details["related_ids"] = ",".join(identifiers[:3])[:200]
    return details


def should_audit(request, status):
    path = request_route_path(request)
    # Router/proxy redirects have not performed the requested business action.
    # Its destination request records the actual outcome instead.
    if 300 <= status < 400:
        return False
    if not path.startswith("/api/v1/") or path in {"/api/v1/auth/activity", "/api/v1/auth/me"}:
        return False
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        return True
    return request.method == "GET" and (
        (request.method, path) in OPERATIONS or status == 403)


class OperationAuditMiddleware:
    """Observe response metadata without consuming/rebuilding response streams."""

    def __init__(self, app, auth):
        self.app, self.auth = app, auth

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request = Request(scope)
        request.state.operation_request_id = uid()
        request.state.operation_started = time.monotonic()
        status, capture, overflow = 500, False, False
        body = bytearray()

        async def observe(message):
            nonlocal status, capture, overflow
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = dict(message.get("headers", []))
                capture = should_audit(request, status) and headers.get(b"content-type", b"").startswith(b"application/json")
            elif message["type"] == "http.response.body" and capture and not overflow:
                value = message.get("body", b"")
                if len(body) + len(value) > MAX_RESPONSE_BYTES:
                    body.clear()
                    overflow = True
                else:
                    body.extend(value)
            await send(message)

        try:
            await self.app(scope, receive, observe)
        finally:
            if should_audit(request, status) and not getattr(request.state, "audit_recorded_id", None):
                payload = None
                if body and not overflow:
                    try:
                        payload = json.loads(body)
                    except (ValueError, UnicodeError, RecursionError):
                        pass
                details = response_details(request, payload, status)
                action = operation_for(request).action
                # A client disconnect must not suppress a committed operation's
                # record. Audit storage failure cannot turn a committed business
                # action into a misleading HTTP failure that encourages retries.
                with anyio.CancelScope(shield=True):
                    try:
                        await run_in_threadpool(self.auth.audit, request, action, status, details)
                    except Exception:
                        logger.error("Unable to persist operation audit; request_id=%s", request.state.operation_request_id)
