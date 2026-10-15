"""HTTP API for local-data training orchestration."""

from contextlib import asynccontextmanager
import hmac
import logging
import os
from pathlib import Path
import secrets
import threading

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from vla_adapters import get_templates

from . import schemas
from .common import ACTIVE, APIError, TERMINAL, digest, now, offset_time, require_path, token_hash, uid
from .db import Database
from .models import Allocation, Asset, Attempt, Command, Event, Idempotency, Node, Preview, Profile, Run, RunInput
from .scheduler import enqueue, gpu_available, resolve_inputs, tick
from .serialization import asset_dict, node_dict, preview_dict, profile_dict, run_dict
from .service import apply_ack, get_required, ingest_events, sync_profiles, validate_request
from .settings import Settings

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.validate()
    db = Database(settings.database_url)
    db.initialize()
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
        if settings.scheduler_enabled:
            worker = threading.Thread(target=scheduler_loop, name="vla-scheduler", daemon=True)
            worker.start()
        try:
            yield
        finally:
            stop.set()
            if worker:
                worker.join(timeout=35)
            db.close()

    app = FastAPI(title="Termitech VLA Control Plane", version="0.1.0", lifespan=lifespan)
    app.state.db = db
    app.state.settings = settings

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

    def require_admin(authorization: str | None = Header(default=None)):
        if not hmac.compare_digest(bearer(authorization), settings.admin_token):
            raise APIError(401, "UNAUTHORIZED", "Administrator credential is invalid")

    def require_enrollment(authorization: str | None = Header(default=None)):
        if not hmac.compare_digest(bearer(authorization), settings.enrollment_token):
            raise APIError(401, "UNAUTHORIZED", "Enrollment credential is invalid")

    def require_agent(authorization: str | None = Header(default=None)) -> str:
        hashed = token_hash(bearer(authorization))
        with db.read() as session:
            node_id = session.scalar(select(Node.id).where(Node.token_hash == hashed))
        if not node_id:
            raise APIError(401, "UNAUTHORIZED", "Node credential is invalid")
        return node_id

    def check_owner(actual: str, supplied: str):
        if actual != supplied:
            raise APIError(403, "NODE_OWNERSHIP_MISMATCH", "Credential does not belong to the requested node")

    def key_required(key: str | None) -> str:
        if not key or not key.strip() or len(key) > 200:
            raise APIError(422, "IDEMPOTENCY_KEY_REQUIRED", "Provide an Idempotency-Key of 1 to 200 characters")
        return key

    admin = [Depends(require_admin)]

    @app.get("/healthz")
    def health():
        with db.read() as session:
            session.execute(text("SELECT 1"))
        return {"status": "ok"}

    @app.get("/api/v1/templates", dependencies=admin)
    def templates():
        return get_templates()

    @app.get("/api/v1/overview", dependencies=admin)
    def overview():
        with db.read() as session:
            nodes = list(session.scalars(select(Node)))
            allocations = {(a.node_id, a.gpu_uuid) for a in session.scalars(select(Allocation))}
            return {"nodes_online": sum(n.status == "ONLINE" for n in nodes), "nodes_total": len(nodes),
                    "gpus_total": sum(len(n.gpus) for n in nodes),
                    "gpus_available": sum(n.status == "ONLINE" and (n.id, gpu["uuid"]) not in allocations and gpu_available(gpu) for n in nodes for gpu in n.gpus),
                    "runs_active": session.scalar(select(func.count()).select_from(Run).where(Run.state.not_in(TERMINAL))),
                    "runs_failed": session.scalar(select(func.count()).select_from(Run).where(Run.state == "FAILED")),
                    "assets_total": session.scalar(select(func.count()).select_from(Asset).where(Asset.state != "ARCHIVED"))}

    @app.get("/api/v1/nodes", dependencies=admin)
    def nodes():
        with db.read() as session:
            return [node_dict(session, node) for node in session.scalars(select(Node).order_by(Node.created_at))]

    @app.get("/api/v1/nodes/{node_id}", dependencies=admin)
    def node_detail(node_id: str):
        with db.read() as session:
            result = node_dict(session, get_required(session, Node, node_id, "Node"))
            result["assets"] = [asset_dict(a) for a in session.scalars(select(Asset).where(Asset.node_id == node_id))]
            result["profiles"] = [profile_dict(p) for p in session.scalars(select(Profile).where(Profile.node_id == node_id))]
            result["runs"] = [run_dict(session, r) for r in session.scalars(select(Run).where(Run.node_id == node_id).order_by(Run.created_at.desc()).limit(100))]
            return result

    @app.get("/api/v1/profiles", dependencies=admin)
    def profiles(node_id: str | None = None):
        with db.read() as session:
            query = select(Profile).where(Profile.active.is_(True))
            if node_id:
                query = query.where(Profile.node_id == node_id)
            return [profile_dict(p) for p in session.scalars(query)]

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

    @app.post("/api/v1/run-previews", dependencies=admin, status_code=202)
    def preview_create(body: schemas.PreviewInput):
        data = body.model_dump()
        with db.write() as session:
            node, _, parameters = validate_request(session, data)
            if node.last_seen < offset_time(-settings.heartbeat_timeout_seconds):
                raise APIError(409, "NODE_OFFLINE", "Node heartbeat has expired")
            data["parameters"] = parameters
            preview = Preview(id=uid(), node_id=node.id, request=data, status="PENDING", checks=[],
                              created_at=now(), expires_at=offset_time(settings.preview_ttl_seconds))
            session.add(preview)
            enqueue(session, node.id, "PREFLIGHT", {"preview_id": preview.id, "profile_id": body.profile_id,
                "template_id": body.template_id, "inputs": resolve_inputs(session, data),
                "parameters": parameters, "gpu_count": body.gpu_count})
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
            node, _, _ = validate_request(session, data)
            session.scalar(select(Node).where(Node.id == node.id).with_for_update())
            if node.last_seen < offset_time(-settings.heartbeat_timeout_seconds):
                raise APIError(409, "NODE_OFFLINE", "Node heartbeat has expired")
            created = now()
            run = Run(id=uid(), **data, state="QUEUED", code_expectation=preview.code_snapshot,
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
                               "node_id": node.id, "run_id": None, "message": f"{node.name}: {node.status}", "timestamp": node.last_seen})
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
        if len({p.id for p in body.profiles}) != len(body.profiles):
            raise APIError(422, "DUPLICATE_PROFILE", "Profile IDs must be unique on each node")
        for profile in body.profiles:
            require_path(profile.code_dir, body.roots)
            require_path(profile.python)
        registration_key = digest(key_required(idempotency_key)) if idempotency_key is not None else None
        body_digest = digest(body.model_dump())
        with db.write() as session:
            if registration_key:
                existing = session.scalar(select(Node).where(Node.registration_key == registration_key).with_for_update())
                if existing:
                    if existing.registration_digest != body_digest:
                        raise APIError(409, "IDEMPOTENCY_CONFLICT", "Registration key was used with different node configuration")
                    token = registration_token(existing.id, registration_key)
                    if token_hash(token) != existing.token_hash:
                        raise APIError(409, "ENROLLMENT_SECRET_CHANGED", "Original registration credential changed; recover the saved node credentials")
                    return {"node_id": existing.id, "token": token}
            node_id = uid()
            token = registration_token(node_id, registration_key) if registration_key else secrets.token_urlsafe(48)
            node = Node(id=node_id, token_hash=token_hash(token), registration_key=registration_key, registration_digest=body_digest,
                        name=body.name, hostname=body.hostname, roots=body.roots, output_root=body.output_root,
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
            node.last_seen, node.gpus, node.metrics = now(), [gpu.model_dump() for gpu in body.gpus], body.metrics
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
            rows = session.scalars(select(Command).where(Command.node_id == node_id, Command.status == "PENDING").order_by(Command.created_at, Command.id).limit(100))
            return [{"id": row.id, "type": row.type, "payload": row.payload} for row in rows]

    @app.post("/agent/v1/commands/{command_id}/ack")
    def command_ack(command_id: str, body: schemas.AckInput, node_id: str = Depends(require_agent)):
        check_owner(node_id, body.node_id)
        with db.write() as session:
            session.scalar(select(Node).where(Node.id == node_id).with_for_update())
            command = session.scalar(select(Command).where(Command.id == command_id).with_for_update())
            if command is None or command.node_id != node_id:
                raise APIError(403, "NODE_OWNERSHIP_MISMATCH", "Command is not assigned to this node")
            apply_ack(session, command, body.status, body.result)
            return {"ok": True}

    @app.post("/agent/v1/events/batch")
    def events_batch(body: schemas.EventBatchInput, node_id: str = Depends(require_agent)):
        check_owner(node_id, body.node_id)
        with db.write() as session:
            node = session.scalar(select(Node).where(Node.id == node_id).with_for_update())
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
    uvicorn.run(create_app(), host=os.getenv("VLA_HOST", "127.0.0.1"), port=int(os.getenv("VLA_PORT", "8000")))


if __name__ == "__main__":
    main()
