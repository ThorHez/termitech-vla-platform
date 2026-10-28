"""HTTP API for local-data training orchestration."""

from contextlib import asynccontextmanager
from copy import deepcopy
import hmac
import logging
import os
from pathlib import Path
import secrets
import threading

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import case, func, select, text
from sqlalchemy.exc import IntegrityError
from vla_adapters import get_templates

from . import schemas
from .auth import AuthManager, register_auth_routes
from .authorization import permission_for, request_route_path
from .common import ACTIVE, APIError, TERMINAL, digest, now, offset_time, require_path, token_hash, uid, valid_snapshot
from .db import Database
from .models import Allocation, Asset, Attempt, Command, Event, Idempotency, Node, Preview, Profile, Run, RunInput
from .onboarding import generate_agent_config, resolve_agent_config_defaults
from .path_suggestions import acknowledge as acknowledge_path_query, create_query, expire_pending as expire_path_queries, matching_profile as matching_path_profile, query_dict as path_query_dict
from .recipe_suggestions import acknowledge as acknowledge_recipe_query, create_query as create_recipe_query, expire_pending as expire_recipe_queries, matching_profile as matching_recipe_profile, query_dict as recipe_query_dict
from .recipe_details import acknowledge as acknowledge_recipe_detail
from .data_processors import acknowledge as acknowledge_data_processors, register_routes as register_data_processor_routes
from .dataset_inspections import acknowledge as acknowledge_dataset_inspection, expire_pending as expire_dataset_inspections, register_routes as register_dataset_inspection_routes
from .recipe_library import register_recipe_routes
from .recipe_writeback import acknowledge as acknowledge_recipe_writeback, register_routes as register_writeback_routes, require_no_pending_writeback
from .profile_rebindings import acknowledge as acknowledge_profile_rebind, register_routes as register_profile_rebinding_routes
from .storage_configurations import acknowledge as acknowledge_storage_configuration, register_routes as register_storage_configuration_routes
from .policy_deployments import acknowledge as acknowledge_policy_deployment, register_routes as register_policy_deployment_routes, COMMANDS as POLICY_DEPLOYMENT_COMMANDS
from .checkpoint_transfers import (CheckpointTransferManager, acknowledge_export,
                                  register_routes as register_checkpoint_transfer_routes, EXPORT_COMMANDS)
from .policy_deployment_models import PolicyDeployment, RESOURCE_STATES as POLICY_RESOURCE_STATES
from .remote_deployment import DeploymentManager, register_remote_routes
from .remote_models import RemoteDeployment
from .scheduler import enqueue, gpu_available, resolve_inputs, tick
from .serialization import asset_dict, node_dict, preview_dict, profile_dict, run_dict
from .service import apply_ack, automatic_preflight_error, config_confirmation_error, get_required, ingest_events, sync_profiles, validate_request
from .settings import Settings
from .transport_security import TransportSecurityMiddleware, uvicorn_options
from .training_drafts import register_routes as register_training_draft_routes
from .training_templates import register_routes as register_training_template_routes
from .resource_history import capture_sample, register_routes as register_resource_history_routes
from .deployment_records import register_routes as register_deployment_record_routes

logger = logging.getLogger(__name__)
RETIRED_NODE_STATES = {"DISCONNECTED", "REMOVED"}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.validate()
    db = Database(settings.database_url)
    db.initialize()
    auth = AuthManager(db, settings)
    remote_manager = DeploymentManager(db, settings)
    transfer_manager = CheckpointTransferManager(db, settings, remote_manager)
    stop = threading.Event()

    def scheduler_loop():
        while not stop.wait(settings.scheduler_interval_seconds):
            try:
                tick(db, settings)
            except Exception:
                logger.exception("Scheduler pass failed; reservations retained")

    @asynccontextmanager
    async def lifespan(_):
        worker = None
        remote_manager.start()
        transfer_manager.start()
        if settings.scheduler_enabled:
            worker = threading.Thread(target=scheduler_loop, name="vla-scheduler", daemon=True)
            worker.start()
        try:
            yield
        finally:
            stop.set()
            transfer_manager.close()
            remote_manager.close()
            if worker:
                worker.join(timeout=35)
            db.close()

    def require_human(request: Request):
        route_path = request_route_path(request)
        if not route_path.startswith("/api/v1/"):
            return
        permission = permission_for(request.method, route_path)
        if permission is not None:
            return auth.require(request, permission)

    app = FastAPI(title="Termitech VLA Control Plane", version="0.1.0", lifespan=lifespan,
                  dependencies=[Depends(require_human)])
    app.state.db = db
    app.state.settings = settings
    app.state.remote_manager = remote_manager
    app.state.transfer_manager = transfer_manager
    app.state.auth = auth

    @app.middleware("http")
    async def human_request_audit(request: Request, call_next):
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            if request_route_path(request).startswith("/api/v1/"):
                response.headers["Cache-Control"] = "no-store"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Referrer-Policy"] = "same-origin"
            response.headers["Content-Security-Policy"] = "frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
            return response
        finally:
            route_path = request_route_path(request)
            if (route_path.startswith("/api/v1/")
                    and (not route_path.startswith("/api/v1/auth/") or status >= 400)
                    and request.method not in {"GET", "HEAD", "OPTIONS"}):
                # Do not capture request bodies, query strings, passwords, Agent
                # credentials or SSH secrets in the audit trail.
                from starlette.concurrency import run_in_threadpool
                await run_in_threadpool(auth.audit, request, "api.mutation", status)

    app.add_middleware(TransportSecurityMiddleware, settings=settings)
    register_auth_routes(app, auth)

    @app.exception_handler(APIError)
    async def api_error(_, exc):
        return JSONResponse(status_code=exc.status, content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}})

    @app.exception_handler(RequestValidationError)
    async def validation_error(_, exc):
        errors = [{"location": list(item["loc"]), "message": item["msg"], "type": item["type"]} for item in exc.errors()]
        return JSONResponse(status_code=422, content={"error": {"code": "VALIDATION_ERROR", "message": "Invalid request", "details": {"errors": errors}}})

    @app.exception_handler(IntegrityError)
    async def integrity_error(_, exc):
        logger.warning("Database rejected a conflicting mutation: %s", type(exc).__name__)
        return JSONResponse(status_code=409, content={"error": {"code": "CONCURRENT_MODIFICATION", "message": "Conflicting update; retry with the same idempotency key", "details": {}}})

    def bearer(value: str | None) -> str:
        if not value or not value.startswith("Bearer ") or not value[7:]:
            raise APIError(401, "UNAUTHORIZED", "Bearer authentication is required")
        return value[7:]

    def require_enrollment(authorization: str | None = Header(default=None)):
        if not hmac.compare_digest(bearer(authorization), settings.enrollment_token):
            raise APIError(401, "UNAUTHORIZED", "Enrollment credential is invalid")

    def require_agent(authorization: str | None = Header(default=None)) -> str:
        hashed = token_hash(bearer(authorization))
        with db.read() as session:
            node = session.scalar(select(Node).where(Node.token_hash == hashed))
        if not node:
            raise APIError(401, "UNAUTHORIZED", "Node credential is invalid")
        assert_agent_connected(node)
        return node.id

    def assert_agent_connected(node):
        if node.status in RETIRED_NODE_STATES:
            raise APIError(401, "NODE_CONNECTION_REMOVED", "该机器连接已删除，请通过中台重新接入")

    def check_owner(actual: str, supplied: str):
        if actual != supplied:
            raise APIError(403, "NODE_OWNERSHIP_MISMATCH", "Credential does not belong to the requested node")

    def key_required(key: str | None) -> str:
        if not key or not key.strip() or len(key) > 200:
            raise APIError(422, "IDEMPOTENCY_KEY_REQUIRED", "Provide an Idempotency-Key of 1 to 200 characters")
        return key

    admin = [Depends(require_human)]
    register_training_draft_routes(app, admin, db, auth)
    register_training_template_routes(app, admin, db, auth)
    register_resource_history_routes(app, admin, db)
    register_remote_routes(app, admin, remote_manager)

    def require_available_node(session, node_id):
        node = session.scalar(select(Node).where(Node.id == node_id).with_for_update()
                              .execution_options(populate_existing=True))
        if node is None:
            raise APIError(404, "NOT_FOUND", "Node not found")
        if node.status != "ONLINE":
            raise APIError(409, "NODE_OFFLINE", "机器当前不可用，请重新接入或等待节点恢复在线")
        if session.scalar(select(RemoteDeployment.id).where(
            RemoteDeployment.node_id == node_id,
            RemoteDeployment.state.in_({"QUEUED", "RUNNING", "WAITING_AGENT"}),
        ).limit(1)):
            raise APIError(409, "NODE_MAINTENANCE", "该机器正在部署 Agent，请等待接入完成后再进行训练预检或提交")
        return node

    register_recipe_routes(app, admin, db, require_available_node, settings)
    register_data_processor_routes(app, admin, db, require_available_node, settings)
    register_dataset_inspection_routes(app, admin, db, require_available_node, settings)
    register_writeback_routes(app, admin, db, require_available_node, settings)
    register_profile_rebinding_routes(app, admin, db, require_available_node, settings)
    register_storage_configuration_routes(app, admin, db, require_available_node, settings)
    register_policy_deployment_routes(app, admin, db, require_available_node, settings)
    register_checkpoint_transfer_routes(app, admin, transfer_manager)
    register_deployment_record_routes(app, admin, db, settings)

    @app.get("/healthz")
    def health():
        with db.read() as session:
            session.execute(text("SELECT 1"))
        return {"status": "ok"}

    @app.get("/api/v1/templates", dependencies=admin)
    def templates():
        return get_templates()

    def agent_config_defaults(request: Request):
        return resolve_agent_config_defaults(os.getenv("VLA_AGENT_CONTROL_PLANE_URL"), str(request.base_url))

    @app.get("/api/v1/agent-config-defaults", dependencies=admin)
    def agent_config_defaults_get(request: Request):
        return JSONResponse(content=agent_config_defaults(request), headers={"Cache-Control": "no-store"})

    @app.post("/api/v1/agent-configs", dependencies=admin, status_code=201)
    def agent_config_create(body: schemas.AgentConfigInput, request: Request):
        # An explicit advanced override need not use the platform default.
        default_url = agent_config_defaults(request)["control_plane_url"] if body.control_plane_url is None else None
        return JSONResponse(status_code=201, content=generate_agent_config(body, settings.enrollment_token, default_url),
                            headers={"Cache-Control": "no-store"})

    @app.get("/api/v1/agent-package", dependencies=admin)
    def agent_package():
        package = Path(os.getenv("VLA_AGENT_BUNDLE", ".local/agent-dist/vla-agent-bundle.tar.gz"))
        if not package.is_file():
            raise APIError(503, "AGENT_PACKAGE_UNAVAILABLE", "Agent installation package is unavailable; ask the administrator to build the Agent bundle and configure VLA_AGENT_BUNDLE")
        return FileResponse(package, media_type="application/gzip", filename="vla-agent-bundle.tar.gz",
                            headers={"Cache-Control": "no-store"})

    @app.get("/api/v1/overview", dependencies=admin)
    def overview():
        with db.read() as session:
            nodes = list(session.scalars(select(Node).where(Node.status.not_in(RETIRED_NODE_STATES))))
            allocations = {(a.node_id, a.gpu_uuid) for a in session.scalars(select(Allocation))}
            allocations.update((item.node_id, item.gpu_uuid) for item in session.scalars(select(PolicyDeployment).where(
                PolicyDeployment.state.in_(POLICY_RESOURCE_STATES))) if item.gpu_uuid)
            return {"nodes_online": sum(n.status == "ONLINE" for n in nodes), "nodes_total": len(nodes),
                    "gpus_total": sum(len(n.gpus) for n in nodes),
                    "gpus_available": sum(n.status == "ONLINE" and (n.id, gpu["uuid"]) not in allocations and gpu_available(gpu) for n in nodes for gpu in n.gpus),
                    "runs_active": session.scalar(select(func.count()).select_from(Run).where(Run.state.not_in(TERMINAL))),
                    "runs_failed": session.scalar(select(func.count()).select_from(Run).where(Run.state == "FAILED")),
                    "assets_total": session.scalar(select(func.count()).select_from(Asset).where(Asset.state != "ARCHIVED"))}

    @app.get("/api/v1/nodes", dependencies=admin)
    def nodes(include_inactive: bool = False):
        with db.read() as session:
            query = select(Node).order_by(Node.created_at)
            if not include_inactive:
                query = query.where(Node.status.not_in(RETIRED_NODE_STATES))
            return [node_dict(session, node) for node in session.scalars(query)]

    @app.get("/api/v1/nodes/{node_id}", dependencies=admin)
    def node_detail(node_id: str):
        with db.read() as session:
            result = node_dict(session, get_required(session, Node, node_id, "Node"))
            result["assets"] = [asset_dict(a) for a in session.scalars(select(Asset).where(Asset.node_id == node_id))]
            result["profiles"] = [profile_dict(p) for p in session.scalars(select(Profile).where(Profile.node_id == node_id))]
            result["runs"] = [run_dict(session, r) for r in session.scalars(select(Run).where(Run.node_id == node_id).order_by(Run.created_at.desc()).limit(100))]
            return result

    @app.patch("/api/v1/nodes/{node_id}", dependencies=admin)
    def node_update(node_id: str, body: schemas.NodeUpdateInput):
        with db.write() as session:
            node = session.scalar(select(Node).where(Node.id == node_id).with_for_update())
            if node is None:
                raise APIError(404, "NOT_FOUND", "Node not found")
            node.display_name = body.name
            session.flush()
            return node_dict(session, node)

    @app.get("/api/v1/profiles", dependencies=admin)
    def profiles(node_id: str | None = None):
        with db.read() as session:
            query = select(Profile).where(Profile.active.is_(True))
            if node_id:
                query = query.where(Profile.node_id == node_id)
            return [profile_dict(p) for p in session.scalars(query)]

    def require_path_query_node(session, node_id):
        node = require_available_node(session, node_id)
        if node.last_seen < offset_time(-settings.heartbeat_timeout_seconds):
            raise APIError(409, "NODE_OFFLINE", "机器心跳已过期，暂时无法获取路径联想")
        return node

    @app.post("/api/v1/nodes/{node_id}/path-suggestions", dependencies=admin, status_code=202)
    def path_suggestions_create(node_id: str, body: schemas.PathSuggestInput):
        with db.write() as session:
            node = require_path_query_node(session, node_id)
            return create_query(session, node, body.prefix, body.limit, body.profile_id)

    @app.get("/api/v1/nodes/{node_id}/path-suggestions/{query_id}", dependencies=admin)
    def path_suggestions_get(node_id: str, query_id: str):
        with db.write() as session:
            node = require_path_query_node(session, node_id)
            command = get_required(session, Command, query_id, "Path suggestion")
            if command.node_id != node.id or command.type != "PATH_SUGGEST":
                raise APIError(404, "NOT_FOUND", "Path suggestion not found")
            expire_path_queries(session, node.id)
            profile = session.get(Profile, (node.id, command.payload["profile_id"])) if command.payload.get("profile_id") else None
            return path_query_dict(command, node.roots, profile)

    @app.post("/api/v1/nodes/{node_id}/recipe-suggestions", dependencies=admin, status_code=202)
    def recipe_suggestions_create(node_id: str, body: schemas.RecipeSuggestInput):
        with db.write() as session:
            node = require_available_node(session, node_id)
            if node.last_seen < offset_time(-settings.heartbeat_timeout_seconds):
                raise APIError(409, "NODE_OFFLINE", "机器心跳已过期，暂时无法获取训练配方")
            return create_recipe_query(session, node, body.profile_id, body.prefix, body.limit)

    @app.get("/api/v1/nodes/{node_id}/recipe-suggestions/{query_id}", dependencies=admin)
    def recipe_suggestions_get(node_id: str, query_id: str):
        with db.write() as session:
            node = require_available_node(session, node_id)
            if node.last_seen < offset_time(-settings.heartbeat_timeout_seconds):
                raise APIError(409, "NODE_OFFLINE", "机器心跳已过期，暂时无法获取训练配方")
            command = get_required(session, Command, query_id, "Recipe suggestion")
            if command.node_id != node.id or command.type != "RECIPE_SUGGEST":
                raise APIError(404, "NOT_FOUND", "Recipe suggestion not found")
            expire_recipe_queries(session, node.id)
            profile = session.get(Profile, (node.id, command.payload["profile_id"]))
            return recipe_query_dict(command, profile, node.roots)

    @app.post("/api/v1/nodes/{node_id}/inference-config-suggestions", dependencies=admin, status_code=202)
    def inference_config_suggestions_create(node_id: str, body: schemas.RecipeSuggestInput):
        with db.write() as session:
            node = require_available_node(session, node_id)
            if node.last_seen < offset_time(-settings.heartbeat_timeout_seconds):
                raise APIError(409, "NODE_OFFLINE", "机器心跳已过期，暂时无法获取推理配置")
            return create_recipe_query(session, node, body.profile_id, body.prefix, body.limit, code_only=True)

    @app.get("/api/v1/nodes/{node_id}/inference-config-suggestions/{query_id}", dependencies=admin)
    def inference_config_suggestions_get(node_id: str, query_id: str):
        with db.write() as session:
            node = require_available_node(session, node_id)
            if node.last_seen < offset_time(-settings.heartbeat_timeout_seconds):
                raise APIError(409, "NODE_OFFLINE", "机器心跳已过期，暂时无法获取推理配置")
            command = get_required(session, Command, query_id, "Inference config suggestion")
            if command.node_id != node.id or command.type != "RECIPE_SUGGEST":
                raise APIError(404, "NOT_FOUND", "Inference config suggestion not found")
            expire_recipe_queries(session, node.id)
            profile = session.get(Profile, (node.id, command.payload["profile_id"]))
            return recipe_query_dict(command, profile, node.roots, code_only=True)

    @app.get("/api/v1/assets", dependencies=admin)
    def assets(node_id: str | None = None, kind: str | None = None):
        with db.read() as session:
            query = select(Asset).order_by(Asset.created_at.desc())
            if node_id:
                query = query.where(Asset.node_id == node_id)
            if kind:
                query = query.where(Asset.kind == kind)
            return [asset_dict(a) for a in session.scalars(query)]

    @app.post("/api/v1/assets", dependencies=admin, status_code=201)
    def create_asset(body: schemas.AssetInput):
        with db.write() as session:
            node = get_required(session, Node, body.node_id, "Node")
            require_path(body.path, node.roots)
            asset = Asset(id=uid(), node_id=node.id, name=body.name, kind=body.kind, path=body.path,
                          version=body.version, metadata_json=body.metadata, state="REGISTERED", created_at=now())
            session.add(asset)
            session.flush()
            return asset_dict(asset)

    @app.post("/api/v1/assets/{asset_id}/archive", dependencies=admin)
    def archive_asset(asset_id: str):
        with db.write() as session:
            asset = get_required(session, Asset, asset_id, "Asset")
            session.scalar(select(Node).where(Node.id == asset.node_id).with_for_update())
            active_run = session.scalar(select(RunInput.run_id).join(Run, RunInput.run_id == Run.id).where(RunInput.asset_id == asset_id, Run.state.not_in(TERMINAL)).limit(1))
            if active_run:
                raise APIError(409, "ASSET_IN_USE", "An active training run still references this asset")
            asset.state = "ARCHIVED"
            return asset_dict(asset)

    @app.patch("/api/v1/assets/{asset_id}/dataset-profile", dependencies=admin)
    def update_dataset_profile(asset_id: str, body: schemas.AssetDatasetProfileInput):
        with db.write() as session:
            asset = session.scalar(select(Asset).where(Asset.id == asset_id).with_for_update())
            if asset is None:
                raise APIError(404, "NOT_FOUND", "Asset not found")
            if asset.kind != "DATASET":
                raise APIError(409, "ASSET_KIND_MISMATCH", "只有数据集资产可以编辑数据集描述")
            if asset.state == "ARCHIVED":
                raise APIError(409, "ASSET_ARCHIVED", "已归档资产不能编辑数据集描述")
            asset.metadata_json = {**asset.metadata_json,
                                   "dataset_profile": body.dataset_profile.model_dump(mode="json", exclude_none=True)}
            session.flush()
            return asset_dict(asset)

    @app.post("/api/v1/run-previews", dependencies=admin, status_code=202)
    def preview_create(body: schemas.PreviewInput):
        data = body.model_dump()
        with db.write() as session:
            node, profile, parameters = validate_request(session, data)
            require_available_node(session, node.id)
            require_no_pending_writeback(session, node.id, profile.code_dir)
            if node.last_seen < offset_time(-settings.heartbeat_timeout_seconds):
                raise APIError(409, "NODE_OFFLINE", "Node heartbeat has expired")
            data["parameters"] = parameters
            # The asset description belongs to this preflight, not a later
            # asset edit or an Agent-supplied source snapshot. Legacy tasks
            # retain their original command shape when no profile is present.
            dataset_profiles = {}
            for role, asset_id in data["inputs"].items():
                asset = session.get(Asset, asset_id)
                if asset.kind == "DATASET" and isinstance(asset.metadata_json.get("dataset_profile"), dict):
                    dataset_profiles[role] = deepcopy(asset.metadata_json["dataset_profile"])
            if dataset_profiles:
                data["dataset_profiles"] = dataset_profiles
            from vla_adapters.composition_spec import COMPOSABLE_PROCESSOR, validate_composition_profile_input
            for index, binding in enumerate(data.get("dataset_bindings") or []):
                processor = binding.get("processor") or {}
                if processor.get("name") != COMPOSABLE_PROCESSOR:
                    continue
                declared = dataset_profiles.get(f"dataset_{index}")
                if declared is None and index == 0:
                    declared = dataset_profiles.get("dataset")
                try:
                    validate_composition_profile_input(declared, processor.get("parameters", {}).get("composition"))
                except ValueError as exc:
                    message = str(exc)
                    code = message.split(":", 1)[0] if ":" in message else "INVALID_COMPOSITION"
                    raise APIError(422, code, message) from exc
            preview = Preview(id=uid(), node_id=node.id, request=data, status="PENDING", checks=[],
                              created_at=now(), expires_at=offset_time(settings.preview_ttl_seconds))
            session.add(preview)
            enqueue(session, node.id, "PREFLIGHT", {"preview_id": preview.id, "profile_id": body.profile_id,
                "template_id": body.template_id, "recipe": data["recipe"], "inputs": resolve_inputs(session, data),
                "parameters": parameters, "gpu_count": body.gpu_count, "entrypoint": data.get("entrypoint"),
                **({"auto_config": data["auto_config"]} if data.get("auto_config") is not None else {}),
                **({"training_stage": data["training_stage"], "stage_config": data["stage_config"]} if data.get("training_stage") is not None else {}),
                **({"dataset_bindings": data["dataset_bindings"]} if data.get("dataset_bindings") is not None else {}),
                **({"recipe_definition": data["recipe_definition"], "node_id": node.id} if data.get("recipe_definition") else {})})
            return {"id": preview.id, "status": preview.status}

    @app.get("/api/v1/run-previews/{preview_id}", dependencies=admin)
    def preview_detail(preview_id: str):
        with db.read() as session:
            preview = get_required(session, Preview, preview_id, "Preflight")
            result = preview_dict(preview)
            if preview.status in {"PENDING", "READY"} and preview.expires_at < now():
                result["status"] = "EXPIRED"
                result["error"] = {"code": "PREFLIGHT_EXPIRED", "message": "Preflight expired"}
            return result

    @app.post("/api/v1/runs", dependencies=admin, status_code=201)
    def run_create(body: schemas.RunInput, idempotency_key: str | None = Header(default=None)):
        key = key_required(idempotency_key)
        request_digest = digest(body.model_dump())
        with db.write() as session:
            existing = session.get(Idempotency, key)
            if existing:
                if existing.digest != request_digest:
                    raise APIError(409, "IDEMPOTENCY_CONFLICT", "Idempotency key was used for a different request")
                return run_dict(session, session.get(Run, existing.run_id))
            baseline = get_required(session, Preview, body.preflight_id, "Preflight")
            session.scalar(select(Node).where(Node.id == baseline.node_id).with_for_update())
            # A concurrent submission may have committed while PostgreSQL
            # waited for the node lock; recheck before consuming the preflight.
            existing = session.get(Idempotency, key)
            if existing:
                if existing.digest != request_digest:
                    raise APIError(409, "IDEMPOTENCY_CONFLICT", "Idempotency key was used for a different request")
                return run_dict(session, session.get(Run, existing.run_id))
            preview = session.scalar(select(Preview).where(Preview.id == body.preflight_id).with_for_update().execution_options(populate_existing=True))
            if not preview:
                raise APIError(404, "NOT_FOUND", "Preflight not found")
            if preview.run_id:
                raise APIError(409, "PREFLIGHT_ALREADY_USED", "This preflight already created a training run")
            if preview.expires_at < now():
                raise APIError(409, "PREFLIGHT_EXPIRED", "Run a new preflight before submission")
            if preview.status != "READY":
                raise APIError(409, "PREFLIGHT_NOT_READY", "A successful preflight is required")
            data = preview.request
            if "recipe" not in data:
                raise APIError(409, "AGENT_RECIPE_UNSUPPORTED", "旧版预检未绑定训练配方，请更新 Agent 后重新预检")
            node, profile, _ = validate_request(session, data, frozen=True)
            require_no_pending_writeback(session, node.id, profile.code_dir)
            if not valid_snapshot(preview.code_snapshot):
                raise APIError(409, "CODE_VERSION_UNAVAILABLE", "预检未记录完整代码版本，请重新预检")
            if "recipe" not in preview.code_snapshot:
                raise APIError(409, "AGENT_RECIPE_UNSUPPORTED", "预检未记录训练配方，请更新 Agent 后重新预检")
            if preview.code_snapshot["recipe"] != data["recipe"]:
                raise APIError(409, "RECIPE_MISMATCH", "预检记录的训练配方与所选配方不一致，请重新预检")
            if preview.code_snapshot.get("recipe_definition") != data.get("recipe_definition"):
                raise APIError(409, "RECIPE_DEFINITION_MISMATCH", "预检记录的自定义配方版本不一致，请重新预检")
            preflight_error = automatic_preflight_error(session, data, preview.code_snapshot, preview.checks)
            if preflight_error:
                raise APIError(409, preflight_error["code"], preflight_error["message"])
            confirmation_error = config_confirmation_error(data, preview.code_snapshot, body.confirmed_config_sha256)
            if confirmation_error:
                raise APIError(409, confirmation_error["code"], confirmation_error["message"])
            session.scalar(select(Node).where(Node.id == node.id).with_for_update())
            require_available_node(session, node.id)
            if node.last_seen < offset_time(-settings.heartbeat_timeout_seconds):
                raise APIError(409, "NODE_OFFLINE", "Node heartbeat has expired")
            created = now()
            run_fields = {key: value for key, value in data.items() if key not in {"recipe", "recipe_revision_id", "recipe_definition", "entrypoint", "dataset_bindings", "training_stage", "stage_config", "dataset_profiles", "auto_config"}}
            expectation = {key: deepcopy(value) for key, value in preview.code_snapshot.items() if key != "dataset_profiles"}
            if data["template_id"] in {"openpi-sft", "recap"}:
                expectation["confirmed_config_sha256"] = body.confirmed_config_sha256
                expectation["config_confirmed_at"] = created
            if "dataset_profiles" in data:
                expectation["dataset_profiles"] = deepcopy(data["dataset_profiles"])
            run = Run(id=uid(), **run_fields, state="QUEUED", code_expectation=expectation,
                      created_at=created, updated_at=created, queue_reason="AWAITING_SCHEDULER")
            session.add(run)
            session.flush()
            for role, asset_id in data["inputs"].items():
                session.add(RunInput(run_id=run.id, node_id=run.node_id, role=role, asset_id=asset_id))
            session.add(Idempotency(key=key, digest=request_digest, run_id=run.id, created_at=created))
            preview.run_id = run.id
            session.flush()
            return run_dict(session, run)

    @app.get("/api/v1/runs", dependencies=admin)
    def runs():
        with db.read() as session:
            return [run_dict(session, run) for run in session.scalars(select(Run).order_by(Run.created_at.desc()).limit(500))]

    @app.get("/api/v1/runs/{run_id}", dependencies=admin)
    def run_detail(run_id: str):
        with db.read() as session:
            return run_dict(session, get_required(session, Run, run_id, "Run"))

    @app.post("/api/v1/runs/{run_id}/cancel", dependencies=admin, status_code=202)
    def cancel_run(run_id: str):
        with db.write() as session:
            run = get_required(session, Run, run_id, "Run")
            session.scalar(select(Node).where(Node.id == run.node_id).with_for_update())
            run = session.scalar(select(Run).where(Run.id == run_id).with_for_update().execution_options(populate_existing=True))
            if run.state in TERMINAL or run.state == "CANCELLING" or run.previous_state == "CANCELLING":
                return run_dict(session, run)
            run.updated_at = now()
            if run.state == "QUEUED":
                run.state, run.finished_at, run.queue_reason = "CANCELLED", now(), None
            else:
                attempt = session.scalar(select(Attempt).where(Attempt.run_id == run.id, Attempt.state.not_in(TERMINAL)).with_for_update())
                if not attempt:
                    raise APIError(409, "INVALID_TRANSITION", "Active execution could not be found")
                enqueue(session, run.node_id, "CANCEL", {"attempt_id": attempt.id, "execution_token": attempt.execution_token})
                if run.state == "UNKNOWN":
                    run.previous_state = attempt.previous_state = "CANCELLING"
                else:
                    run.state = attempt.state = "CANCELLING"
            session.flush()
            return run_dict(session, run)

    @app.get("/api/v1/runs/{run_id}/events", dependencies=admin)
    def run_events(run_id: str, after_seq: int = Query(default=0, ge=0), limit: int = Query(default=1000, ge=1, le=1000)):
        with db.read() as session:
            get_required(session, Run, run_id, "Run")
            events = session.scalars(select(Event).join(Attempt, Event.attempt_id == Attempt.id).where(Attempt.run_id == run_id, Event.seq > after_seq).order_by(Event.seq).limit(limit))
            return [{"attempt_id": e.attempt_id, "seq": e.seq, "type": e.type, "timestamp": e.timestamp, "data": e.data} for e in events]

    @app.get("/api/v1/runs/{run_id}/logs", dependencies=admin)
    def run_logs(run_id: str, after_seq: int = Query(default=0, ge=0), limit: int = Query(default=1000, ge=1, le=1000)):
        with db.read() as session:
            get_required(session, Run, run_id, "Run")
            events = list(session.scalars(select(Event).join(Attempt, Event.attempt_id == Attempt.id).where(Attempt.run_id == run_id, Event.seq > after_seq, Event.type == "LOG").order_by(Event.seq).limit(limit)))
            return {"items": [{"seq": e.seq, "text": e.data["text"], "timestamp": e.timestamp} for e in events], "next_seq": events[-1].seq if events else after_seq}

    @app.get("/api/v1/alerts", dependencies=admin)
    def alerts():
        with db.read() as session:
            result = []
            for node in session.scalars(select(Node).where(Node.status.in_(["OFFLINE", "RECOVERING"]))):
                result.append({"id": "node:" + node.id, "severity": "error", "code": "NODE_" + node.status,
                               "node_id": node.id, "run_id": None, "message": f"{node.effective_name}: {node.status}", "timestamp": node.last_seen})
            for run in session.scalars(select(Run).where(Run.state.in_(["FAILED", "UNKNOWN"])).order_by(Run.updated_at.desc()).limit(100)):
                error = run.error or {"code": "EXECUTION_UNKNOWN", "message": "Awaiting agent reconciliation; GPU reservation is retained"}
                result.append({"id": "run:" + run.id, "severity": "error", "code": error.get("code", "TRAINING_FAILED"),
                               "node_id": run.node_id, "run_id": run.id, "message": error.get("message", "Training failed"), "timestamp": run.updated_at})
            return result

    def registration_token(node_id, key):
        return hmac.new(settings.enrollment_token.encode(), ("vla-node:" + node_id + ":" + key).encode(), "sha256").hexdigest()

    @app.post("/agent/v1/register", dependencies=[Depends(require_enrollment)], status_code=201)
    def register(body: schemas.RegisterInput, idempotency_key: str | None = Header(default=None)):
        for root in body.roots:
            require_path(root)
        require_path(body.output_root)
        if body.cache_root is not None:
            require_path(body.cache_root, body.roots)
        if len({p.id for p in body.profiles}) != len(body.profiles):
            raise APIError(422, "DUPLICATE_PROFILE", "Profile IDs must be unique on each node")
        for profile in body.profiles:
            require_path(profile.code_dir, body.roots)
            require_path(profile.python)
        registration_key = digest(key_required(idempotency_key)) if idempotency_key is not None else None
        registration_data = body.model_dump()
        if body.cache_root is None:
            registration_data.pop("cache_root")  # Preserve older registration idempotency digests.
        body_digest = digest(registration_data)
        with db.write() as session:
            if registration_key:
                existing = session.scalar(select(Node).where(Node.registration_key == registration_key).with_for_update())
                if existing:
                    assert_agent_connected(existing)
                    if existing.registration_digest != body_digest:
                        raise APIError(409, "IDEMPOTENCY_CONFLICT", "Registration key was used with different node configuration")
                    token = registration_token(existing.id, registration_key)
                    if token_hash(token) != existing.token_hash:
                        raise APIError(409, "ENROLLMENT_SECRET_CHANGED", "Original registration credential changed; recover the saved node credentials")
                    return {"node_id": existing.id, "token": token}
            node_id = uid()
            token = registration_token(node_id, registration_key) if registration_key else secrets.token_urlsafe(48)
            node = Node(id=node_id, token_hash=token_hash(token), registration_key=registration_key, registration_digest=body_digest,
                        name=body.name, hostname=body.hostname, roots=body.roots, output_root=body.output_root, cache_root=body.cache_root,
                        created_at=now(), last_seen=now(), status="REGISTERED", gpus=[], metrics={})
            session.add(node)
            session.flush()
            for value in body.profiles:
                data = value.model_dump()
                data["metadata_json"] = data.pop("metadata")
                session.add(Profile(node_id=node_id, **data))
            return {"node_id": node_id, "token": token}

    @app.post("/agent/v1/heartbeat")
    def heartbeat(body: schemas.HeartbeatInput, node_id: str = Depends(require_agent)):
        check_owner(node_id, body.node_id)
        if len({gpu.uuid for gpu in body.gpus}) != len(body.gpus):
            raise APIError(422, "DUPLICATE_GPU_UUID", "Reported GPU UUIDs must be unique")
        with db.write() as session:
            node = session.scalar(select(Node).where(Node.id == node_id).with_for_update())
            assert_agent_connected(node)
            node.last_seen, node.gpus, node.metrics = now(), [gpu.model_dump() for gpu in body.gpus], body.metrics
            capture_sample(session, node)
            reported_ids = {active.attempt_id for active in body.active_attempts}

            def mark_unknown(attempt):
                run = session.scalar(select(Run).where(Run.id == attempt.run_id).with_for_update())
                if attempt.state not in TERMINAL and attempt.state != "UNKNOWN":
                    attempt.previous_state, attempt.state = attempt.state, "UNKNOWN"
                if run.state not in TERMINAL and run.state != "UNKNOWN":
                    run.previous_state, run.state = run.state, "UNKNOWN"
                    run.updated_at = now()

            for active in body.active_attempts:
                attempt = session.scalar(select(Attempt).where(Attempt.id == active.attempt_id).with_for_update())
                if attempt is None or attempt.node_id != node_id:
                    raise APIError(403, "NODE_OWNERSHIP_MISMATCH", "Attempt belongs to another node")
                if active.state == "UNKNOWN" or active.state in TERMINAL:
                    mark_unknown(attempt)
                elif attempt.state == "UNKNOWN" and active.state in {"RUNNING", "STARTING", "PREPARING", "CANCELLING", "STOPPING"}:
                    run = session.scalar(select(Run).where(Run.id == attempt.run_id).with_for_update())
                    restored = "CANCELLING" if run.previous_state == "CANCELLING" or active.state in {"CANCELLING", "STOPPING"} else ("RUNNING" if active.state == "RUNNING" else "STARTING")
                    run.state = attempt.state = restored
                    run.previous_state = attempt.previous_state = None
                    run.updated_at = now()
            # Absence from a fresh heartbeat is uncertainty, never proof of
            # exit. STARTING has a normal dispatch-to-local-registration gap.
            for attempt in session.scalars(select(Attempt).where(Attempt.node_id == node_id, Attempt.state.in_(["RUNNING", "CANCELLING"])).with_for_update()):
                if attempt.id not in reported_ids and attempt.started_at is not None:
                    mark_unknown(attempt)
            session.flush()
            unresolved = session.scalar(select(Attempt.id).where(Attempt.node_id == node_id, Attempt.state == "UNKNOWN").limit(1))
            node.status = "RECOVERING" if unresolved else "ONLINE"
            result = {"ok": True}
            if body.profiles is not None:
                result["profile_sync"] = sync_profiles(session, node, [profile.model_dump() for profile in body.profiles])
            return result

    @app.get("/agent/v1/commands")
    def commands(node_id: str, owner: str = Depends(require_agent)):
        check_owner(owner, node_id)
        with db.read() as session:
            # A slow preflight backlog must never hide lifecycle commands behind
            # the delivery limit. Keep delivery stable and read-only within each
            # priority, including START before its later CANCEL, so retries
            # return the same unacknowledged commands.
            priority = case((Command.type.in_(["START", "CANCEL", "POLICY_DEPLOY_START", "POLICY_DEPLOY_STOP", "POLICY_EXPORT_CANCEL"]), 0),
                            (Command.type == "PREFLIGHT", 2), else_=1)
            rows = session.scalars(select(Command).where(Command.node_id == node_id, Command.status == "PENDING")
                                   .order_by(priority, Command.created_at, Command.id).limit(100))
            return [{"id": row.id, "type": row.type, "payload": row.payload} for row in rows]

    @app.post("/agent/v1/commands/{command_id}/ack")
    def command_ack(command_id: str, body: schemas.AckInput, node_id: str = Depends(require_agent)):
        check_owner(node_id, body.node_id)
        with db.write() as session:
            node = session.scalar(select(Node).where(Node.id == node_id).with_for_update())
            assert_agent_connected(node)
            command = session.scalar(select(Command).where(Command.id == command_id).with_for_update())
            if command is None or command.node_id != node_id:
                raise APIError(403, "NODE_OWNERSHIP_MISMATCH", "Command is not assigned to this node")
            if command.type == "PATH_SUGGEST":
                expire_path_queries(session, node_id)
                previous_error = command.result.get("error") if isinstance(command.result, dict) else None
                if (command.status == "REJECTED" and isinstance(command.result, dict)
                        and isinstance(previous_error, dict) and previous_error.get("code") == "PATH_SUGGEST_TIMEOUT"):
                    return {"ok": True}
                profile = session.get(Profile, (node_id, command.payload["profile_id"])) if command.payload.get("profile_id") else None
                if (command.status != "PENDING" and command.payload.get("profile_id") is not None
                        and not matching_path_profile(command, profile)):
                    return {"ok": True}
                status, result = acknowledge_path_query(command, node.roots, body.status, body.result, profile)
                apply_ack(session, command, status, result)
            elif command.type == "RECIPE_WRITE":
                acknowledge_recipe_writeback(session, command, node, body.status, body.result)
            elif command.type == "PROFILE_REBIND":
                acknowledge_profile_rebind(session, command, node, body.status, body.result)
            elif command.type == "STORAGE_CONFIGURE":
                acknowledge_storage_configuration(session, command, node, body.status, body.result)
            elif command.type in POLICY_DEPLOYMENT_COMMANDS:
                acknowledge_policy_deployment(session, command, node, body.status, body.result)
            elif command.type in EXPORT_COMMANDS:
                acknowledge_export(session, command, node, body.status, body.result, transfer_manager)
            elif command.type == "RECIPE_DETAIL":
                acknowledge_recipe_detail(session, command, node, body.status, body.result)
            elif command.type == "DATA_PROCESSORS":
                acknowledge_data_processors(session, command, node, body.status, body.result)
            elif command.type == "DATASET_INSPECT":
                acknowledge_dataset_inspection(session, command, node, body.status, body.result)
            elif command.type == "RECIPE_SUGGEST":
                expire_recipe_queries(session, node_id)
                profile = session.get(Profile, (node_id, command.payload["profile_id"]))
                previous_error = command.result.get("error") if isinstance(command.result, dict) else None
                if (command.status != "PENDING" and
                        (not matching_recipe_profile(command, profile, node.roots)
                         or isinstance(previous_error, dict) and previous_error.get("code") in {
                             "RECIPE_SUGGEST_TIMEOUT", "RECIPE_PROFILE_CHANGED"})):
                    return {"ok": True}
                status, result = acknowledge_recipe_query(command, profile, node.roots, body.status, body.result)
                apply_ack(session, command, status, result)
            else:
                apply_ack(session, command, body.status, body.result)
            return {"ok": True}

    @app.post("/agent/v1/events/batch")
    def events_batch(body: schemas.EventBatchInput, node_id: str = Depends(require_agent)):
        check_owner(node_id, body.node_id)
        with db.write() as session:
            node = session.scalar(select(Node).where(Node.id == node_id).with_for_update())
            assert_agent_connected(node)
            accepted = ingest_events(session, node, [event.model_dump() for event in body.events])
            session.flush()
            if node.status == "RECOVERING" and node.last_seen >= offset_time(-settings.heartbeat_timeout_seconds):
                unresolved = session.scalar(select(Attempt.id).where(Attempt.node_id == node_id, Attempt.state == "UNKNOWN").limit(1))
                if unresolved is None:
                    node.status = "ONLINE"
            return {"accepted": accepted}

    static_dir = os.getenv("VLA_WEB_DIST")
    if static_dir and Path(static_dir, "index.html").is_file():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="web")
    return app


def main():
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    settings = Settings.from_env()
    uvicorn.run(create_app(settings), host=os.getenv("VLA_HOST", "127.0.0.1"),
                port=int(os.getenv("VLA_PORT", "8000")), **uvicorn_options(settings))


if __name__ == "__main__":
    main()
