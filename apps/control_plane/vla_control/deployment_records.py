"""Durable deployment workflows with exact transfer and runtime relationships."""

from copy import deepcopy

from sqlalchemy import select

from .auth_models import AuditEvent
from .checkpoint_transfer_models import CheckpointTransfer, PolicyDeploymentTransfer, TRANSFER_ACTIVE, TRANSFER_TERMINAL
from .common import APIError, now, uid
from .deployment_record_models import DeploymentRecord, DeploymentRecordRuntime, DeploymentRecordTransfer
from .models import Command
from .policy_deployment_models import PolicyDeployment


def _record_name(name):
    return (name.removesuffix(" · 权重同步").strip() or "部署记录")[:200]


def backfill_deployment_records(connection):
    """Run under initialize's shared guard; never infer relationships from assets.

    Existing shared transfers keep their first exact runtime owner. Later
    runtimes retain their original explicit selected-transfer reference.
    """
    records, runtimes, transfers = (model.__table__ for model in (
        DeploymentRecord, DeploymentRecordRuntime, DeploymentRecordTransfer))
    runtime_table, transfer_table, exact = (model.__table__ for model in (
        PolicyDeployment, CheckpointTransfer, PolicyDeploymentTransfer))

    def make_record(candidate, name, created_at):
        record_id = candidate
        if connection.scalar(select(records.c.id).where(records.c.id == record_id)) is not None:
            record_id = uid()
        connection.execute(records.insert().values(id=record_id, name=_record_name(name), created_at=created_at))
        return record_id

    for row in connection.execute(select(runtime_table).order_by(runtime_table.c.created_at, runtime_table.c.id)).mappings():
        if connection.scalar(select(runtimes.c.record_id).where(runtimes.c.deployment_id == row["id"])) is not None:
            continue
        transfer_id = connection.scalar(select(exact.c.transfer_id).where(exact.c.deployment_id == row["id"]))
        owner = connection.scalar(select(transfers.c.record_id).where(transfers.c.transfer_id == transfer_id)) if transfer_id else None
        if owner is not None and connection.scalar(select(runtimes.c.deployment_id).where(runtimes.c.record_id == owner)) is None:
            record_id = owner
            connection.execute(records.update().where(records.c.id == record_id).values(name=row["name"]))
        else:
            record_id = make_record(row["id"], row["name"], row["created_at"])
        connection.execute(runtimes.insert().values(deployment_id=row["id"], record_id=record_id))
        if transfer_id is not None and owner is None:
            connection.execute(transfers.insert().values(transfer_id=transfer_id, record_id=record_id,
                provenance={"origin": "backfill_exact_runtime_transfer", "linked_at": row["created_at"]}))
    for row in connection.execute(select(transfer_table).order_by(transfer_table.c.created_at, transfer_table.c.id)).mappings():
        if connection.scalar(select(transfers.c.record_id).where(transfers.c.transfer_id == row["id"])) is not None:
            continue
        record_id = make_record(row["id"], row["name"], row["created_at"])
        connection.execute(transfers.insert().values(transfer_id=row["id"], record_id=record_id,
            provenance={"origin": "backfill_standalone_transfer", "linked_at": row["created_at"]}))


def resolve_record(session, identifier, *, lock=False):
    """Canonical IDs take priority; aliases use only the durable relationship FK."""
    record = session.get(DeploymentRecord, identifier)
    if record is None:
        link = session.get(DeploymentRecordRuntime, identifier)
        if link is None:
            link = session.get(DeploymentRecordTransfer, identifier)
        record = session.get(DeploymentRecord, link.record_id) if link is not None else None
    visited = set()
    while record is not None and record.merged_into_id is not None:
        if record.id in visited:
            raise APIError(409, "DEPLOYMENT_RECORD_CONFLICT", "部署记录的历史归属关系无效")
        visited.add(record.id)
        record = session.get(DeploymentRecord, record.merged_into_id)
    if record is None:
        raise APIError(404, "DEPLOYMENT_RECORD_NOT_FOUND", "部署记录不存在")
    if lock:
        record = session.scalar(select(DeploymentRecord).where(DeploymentRecord.id == record.id)
            .with_for_update().execution_options(populate_existing=True))
    return record


def record_runtime(session, record_id):
    link = session.scalar(select(DeploymentRecordRuntime).where(DeploymentRecordRuntime.record_id == record_id))
    return session.get(PolicyDeployment, link.deployment_id) if link is not None else None


def require_preparation_record(session, record_id):
    record = resolve_record(session, record_id, lock=True)
    if record_runtime(session, record.id) is not None:
        raise APIError(409, "DEPLOYMENT_RECORD_HAS_RUNTIME", "该部署记录已有推理服务，请在原记录中启动或停止服务")
    return record


def create_record(session, record_id, name, created_at):
    if session.get(DeploymentRecord, record_id) is not None:
        record_id = uid()
    record = DeploymentRecord(id=record_id, name=_record_name(name), created_at=created_at)
    session.add(record)
    session.flush()
    return record


def transfer_record(session, transfer):
    link = session.get(DeploymentRecordTransfer, transfer.id)
    if link is None:
        record = create_record(session, transfer.id, transfer.name, transfer.created_at)
        session.add(DeploymentRecordTransfer(transfer_id=transfer.id, record_id=record.id,
            provenance={"origin": "standalone_transfer", "linked_at": now()}))
        session.flush()
        return record
    return resolve_record(session, link.record_id, lock=True)


def _children(session, record):
    runtime = record_runtime(session, record.id)
    selected = session.get(PolicyDeploymentTransfer, runtime.id) if runtime is not None else None
    owned = list(session.scalars(select(CheckpointTransfer).join(DeploymentRecordTransfer,
        DeploymentRecordTransfer.transfer_id == CheckpointTransfer.id).where(DeploymentRecordTransfer.record_id == record.id)))
    if selected is not None and not any(row.id == selected.transfer_id for row in owned):
        # Historical shared artifacts are referenced only through the existing
        # exact runtime FK. They do not acquire ownership in another workflow.
        referenced = session.get(CheckpointTransfer, selected.transfer_id)
        if referenced is not None:
            owned.append(referenced)
    owned.sort(key=lambda row: (row.created_at, row.id), reverse=True)
    return runtime, selected, owned


def record_dict(session, record, settings, *, detail=False):
    from .policy_deployments import deployment_dict
    runtime, selected, transfers = _children(session, record)
    serialized_runtime = deployment_dict(session, runtime, settings) if runtime is not None else None
    active = next((row for row in transfers if row.state in TRANSFER_ACTIVE), None)
    representative = active or (transfers[0] if transfers else None)
    if runtime is not None:
        state = serialized_runtime["state"]
        node_id, profile_id, source_run_id = runtime.node_id, runtime.profile_id, runtime.source_run_id
    else:
        state = "PREPARING" if active else "READY_TO_DEPLOY" if representative and representative.state == "SUCCEEDED" else representative.state if representative else "DRAFT"
        node_id = representative.target_node_id if representative else None
        profile_id = representative.target_profile_id if representative else None
        source_run_id = representative.source_run_id if representative else None
    dates = [record.created_at, *[row.updated_at for row in transfers]]
    if runtime is not None:
        dates.append(runtime.updated_at)
    for provenance in session.scalars(select(DeploymentRecordTransfer.provenance).where(DeploymentRecordTransfer.record_id == record.id)):
        if isinstance(provenance, dict) and isinstance(provenance.get("linked_at"), str):
            dates.append(provenance["linked_at"])
    result = {"id": record.id, "name": record.name, "node_id": node_id, "profile_id": profile_id,
        "source_run_id": source_run_id, "state": state, "created_at": record.created_at, "updated_at": max(dates),
        "deployment_id": runtime.id if runtime else None,
        "selected_transfer_id": selected.transfer_id if selected else representative.id if runtime is None and representative else None,
        "transfer_ids": [row.id for row in transfers]}
    if detail:
        from .checkpoint_transfers import transfer_dict
        result.update(deployment=serialized_runtime,
            transfers=[transfer_dict(row, session) for row in transfers])
    return result


def _historical_portable_witness(session, transfer):
    """Recover only an exact durable export receipt, without editing old rows."""
    from vla_protocol import canonical_digest
    if transfer.portable_config is not None:
        return (transfer.portable_config, None) if canonical_digest(transfer.portable_config) == transfer.portable_digest else (None, None)
    if transfer.state != "CANCELLED" or not transfer.portable_digest:
        return None, None
    receipts = session.scalars(select(Command).where(Command.node_id == transfer.source_node_id,
        Command.type.in_({"POLICY_EXPORT", "POLICY_EXPORT_STATUS"}), Command.status == "APPLIED",
        Command.payload["transfer_id"].as_string() == transfer.id).order_by(Command.created_at.desc(), Command.id.desc()))
    for command in receipts:
        payload, result = command.payload or {}, command.result or {}
        if not isinstance(payload, dict) or not isinstance(result, dict):
            continue
        attempt = payload.get("transfer_attempt")
        if (type(attempt) is not int or not 1 <= attempt <= transfer.attempt
                or payload.get("export_id") != transfer.id + "-" + str(attempt)):
            continue
        if command.type == "POLICY_EXPORT":
            training = payload.get("source_training")
            if not isinstance(training, dict) or training.get("run_id") != transfer.source_run_id:
                continue
        config = result.get("portable_config")
        if (result.get("state") == "READY" and isinstance(config, dict)
                and result.get("portable_digest") == transfer.portable_digest
                and canonical_digest(config) == transfer.portable_digest):
            return config, command.id
    return None, None


def _assign_historical_transfer(session, transfer_id, record_id, *, actor_id=None, actor_username="system"):
    """Explicit private repair, never a broad run/node matching migration."""
    from vla_protocol import canonical_digest
    target = resolve_record(session, record_id, lock=True)
    transfer = session.scalar(select(CheckpointTransfer).where(CheckpointTransfer.id == transfer_id)
        .with_for_update().execution_options(populate_existing=True))
    link = session.get(DeploymentRecordTransfer, transfer_id)
    if transfer is None or link is None:
        raise APIError(404, "NOT_FOUND", "历史同步或原部署记录不存在")
    if link.record_id == target.id:
        return target
    source = resolve_record(session, link.record_id, lock=True)
    if transfer.state not in TRANSFER_TERMINAL or record_runtime(session, source.id) is not None:
        raise APIError(409, "DEPLOYMENT_RECORD_CONFLICT", "只能将未关联推理服务且已结束的历史同步归入部署记录")
    runtime, selected, _ = _children(session, target)
    reference = session.get(CheckpointTransfer, selected.transfer_id) if runtime is not None and selected is not None else None
    source_config, witness_command_id = _historical_portable_witness(session, transfer)
    fields = ("source_node_id", "source_run_id", "source_checkpoint_path", "target_node_id", "target_profile_id", "target_directory")
    if (reference is None or reference.state != "SUCCEEDED" or any(getattr(transfer, key) != getattr(reference, key) for key in fields)
            or transfer.binding_snapshot != reference.binding_snapshot or not transfer.portable_digest
            or transfer.portable_digest != reference.portable_digest or source_config is None
            or reference.portable_config is None or canonical_digest(source_config) != transfer.portable_digest
            or canonical_digest(reference.portable_config) != reference.portable_digest):
        raise APIError(422, "DEPLOYMENT_RECORD_TRANSFER_MISMATCH", "历史同步与所选部署的完整来源、目标、冻结配置或推理摘要不一致")
    stamp = now()
    original = deepcopy(link.provenance or {})
    link.record_id = target.id
    link.provenance = {"origin": "explicit_historical_assignment", "previous_record_id": source.id,
        "previous_provenance": original, "linked_at": stamp, "reference_transfer_id": reference.id,
        "binding_sha256": canonical_digest(transfer.binding_snapshot), "portable_digest": transfer.portable_digest,
        "portable_witness_command_id": witness_command_id}
    session.flush()
    if (session.scalar(select(DeploymentRecordTransfer.transfer_id).where(DeploymentRecordTransfer.record_id == source.id).limit(1)) is None
            and record_runtime(session, source.id) is None):
        source.merged_into_id = target.id
    session.add(AuditEvent(id=uid(), created_at=stamp, actor_id=actor_id, actor_username=actor_username,
        action="deployment_record.attach_historical_transfer", method="INTERNAL", path="/api/v1/deployment-records/" + target.id,
        status=200, details={"record_id": target.id, "previous_record_id": source.id, "transfer_id": transfer.id,
            "reference_transfer_id": reference.id, "binding_sha256": canonical_digest(transfer.binding_snapshot),
            "portable_witness_command_id": witness_command_id}, remote_addr="local"))
    session.flush()
    return target


def register_routes(app, admin, db, settings):
    @app.get("/api/v1/deployment-records", dependencies=admin)
    def list_records():
        with db.read() as session:
            return [record_dict(session, row, settings) for row in session.scalars(select(DeploymentRecord).where(
                DeploymentRecord.merged_into_id.is_(None)).order_by(DeploymentRecord.created_at.desc(), DeploymentRecord.id.desc()))]

    @app.get("/api/v1/deployment-records/{record_id}", dependencies=admin)
    def get_record(record_id: str):
        with db.read() as session:
            return record_dict(session, resolve_record(session, record_id), settings, detail=True)
