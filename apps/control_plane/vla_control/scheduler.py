"""Database-backed scheduling; multiple workers share the same reservation locks."""

import logging
import secrets
import time

from sqlalchemy import select

from .common import ACTIVE, now, offset_time, uid
from .db import Database
from .models import Allocation, Asset, Attempt, Command, Node, Preview, Run
from .path_suggestions import expire_pending as expire_path_queries
from .recipe_suggestions import expire_pending as expire_recipe_queries
from .recipe_details import expire_pending as expire_recipe_details
from .recipe_writeback import pending_for_checkout
from .remote_models import RemoteDeployment
from .settings import Settings

logger = logging.getLogger(__name__)


def gpu_available(gpu: dict) -> bool:
    """Unknown process telemetry is not proof that a device is available."""
    if gpu.get("processes") != []:
        return False
    utilization = gpu.get("utilization_pct")
    memory = gpu.get("memory_used_mb")
    total = gpu.get("memory_total_mb")
    if utilization is None or memory is None or total is None:
        return False
    return utilization <= 5 and memory <= max(128, total * 0.02)


def enqueue(session, node_id: str, kind: str, payload: dict) -> Command:
    command = Command(id=uid(), node_id=node_id, type=kind, payload=payload, created_at=now(), status="PENDING")
    session.add(command)
    return command


def resolve_inputs(session, run_or_request) -> dict:
    inputs = run_or_request.inputs if hasattr(run_or_request, "inputs") else run_or_request["inputs"]
    result = {}
    for role, asset_id in inputs.items():
        asset = session.get(Asset, asset_id)
        result[role] = {"id": asset.id, "path": asset.path, "kind": asset.kind}
    return result


def tick(db: Database, settings: Settings) -> int:
    """Do one bounded scheduling pass, returning the number of dispatched runs."""
    dispatched = 0
    cutoff = offset_time(-settings.heartbeat_timeout_seconds)
    with db.read() as session:
        node_ids = list(session.scalars(select(Node.id).order_by(Node.id)))
    for node_id in node_ids:
        with db.write() as session:
            node = session.scalar(select(Node).where(Node.id == node_id).with_for_update())
            if node.status in {"DISCONNECTED", "REMOVED"}:
                continue
            if node.last_seen < cutoff:
                node.status = "OFFLINE"
                for run in session.scalars(select(Run).where(Run.node_id == node_id, Run.state.in_(ACTIVE)).with_for_update()):
                    if run.state != "UNKNOWN":
                        run.previous_state, run.state = run.state, "UNKNOWN"
                        run.updated_at = now()
                    for attempt in session.scalars(select(Attempt).where(Attempt.run_id == run.id, Attempt.state.in_(ACTIVE)).with_for_update()):
                        if attempt.state != "UNKNOWN":
                            attempt.previous_state, attempt.state = attempt.state, "UNKNOWN"
            queued = list(session.scalars(select(Run).where(Run.node_id == node_id, Run.state == "QUEUED").order_by(Run.created_at).with_for_update()))
            maintenance = session.scalar(select(RemoteDeployment.id).where(
                RemoteDeployment.node_id == node_id,
                RemoteDeployment.state.in_({"QUEUED", "RUNNING", "WAITING_AGENT"}),
            ).limit(1))
            if maintenance:
                for run in queued:
                    run.queue_reason = "NODE_MAINTENANCE"
                continue
            if node.status != "ONLINE":
                for run in queued:
                    run.queue_reason = "NODE_RECOVERING" if node.status == "RECOVERING" else "NODE_OFFLINE"
                continue
            reserved = {row.gpu_uuid for row in session.scalars(select(Allocation).where(Allocation.node_id == node_id))}
            free = [gpu["uuid"] for gpu in node.gpus if gpu["uuid"] not in reserved and gpu_available(gpu)]
            for run in queued:
                if pending_for_checkout(session, node_id, run.code_expectation.get("code_dir")):
                    run.queue_reason = "RECIPE_WRITEBACK_PENDING"
                    continue
                assets = [session.get(Asset, asset_id) for asset_id in run.inputs.values()]
                if any(asset is None or asset.state != "READY" for asset in assets):
                    run.state, run.finished_at, run.updated_at = "FAILED", now(), now()
                    run.error = {"code": "INPUT_UNAVAILABLE", "message": "An input asset is no longer available"}
                    run.queue_reason = None
                    continue
                if run.gpu_count > len(free):
                    run.queue_reason = "GPU_UNAVAILABLE"
                    continue
                chosen = free[:run.gpu_count]
                attempt = Attempt(id=uid(), run_id=run.id, node_id=node_id, number=1,
                                  execution_token=secrets.token_hex(32), state="STARTING",
                                  created_at=now(), gpu_uuids=chosen, last_event_seq=0)
                session.add(attempt)
                session.flush()
                for gpu_uuid in chosen:
                    session.add(Allocation(node_id=node_id, gpu_uuid=gpu_uuid, attempt_id=attempt.id, created_at=now()))
                # Reservation, execution identity, state and durable command are
                # committed together. A delivery retry reuses the same command.
                enqueue(session, node_id, "START", {
                    "run_id": run.id, "attempt_id": attempt.id, "execution_token": attempt.execution_token,
                    "profile_id": run.profile_id, "template_id": run.template_id,
                    "inputs": resolve_inputs(session, run), "parameters": run.parameters,
                    "gpu_uuids": chosen, "code_expectation": run.code_expectation,
                    "entrypoint": run.code_expectation.get("requested_entrypoint"),
                    **({"training_stage": run.code_expectation["training_stage"], "stage_config": run.code_expectation.get("stage_config", {})} if run.code_expectation.get("training_stage") is not None else {}),
                    **({"dataset_bindings": run.code_expectation["dataset_bindings"]} if run.code_expectation.get("dataset_bindings") is not None else {}),
                    **({"recipe": run.code_expectation["recipe"]} if "recipe" in run.code_expectation else {}),
                    **({"recipe_definition": run.code_expectation["recipe_definition"], "node_id": node_id} if run.code_expectation.get("recipe_definition") else {}),
                })
                run.state, run.queue_reason, run.updated_at = "STARTING", None, now()
                free = free[run.gpu_count:]
                dispatched += 1
    with db.write() as session:
        expire_path_queries(session)
        expire_recipe_queries(session)
        expire_recipe_details(session)
        for preview in session.scalars(select(Preview).where(Preview.status.in_(["PENDING", "READY"]), Preview.expires_at < now()).with_for_update()):
            preview.status = "EXPIRED"
            preview.error = {"code": "PREFLIGHT_EXPIRED", "message": "Preflight expired; request a new check"}
    return dispatched


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = Settings.from_env()
    settings.validate()
    db = Database(settings.database_url)
    db.initialize()
    try:
        while True:
            try:
                tick(db, settings)
            except Exception:
                logger.exception("Scheduler pass failed; will retry without releasing allocations")
            time.sleep(settings.scheduler_interval_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        db.close()


if __name__ == "__main__":
    main()
