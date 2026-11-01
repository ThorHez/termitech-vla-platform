"""Human-visible training records, with execution and asset lineage retained."""

from sqlalchemy import select

from .common import APIError, TERMINAL, now
from .checkpoint_transfer_models import CheckpointTransfer, PolicyDeploymentTransfer
from .deployment_record_models import DeploymentRecord, DeploymentRecordRuntime, DeploymentRecordTransfer
from .models import Allocation, Attempt, Node, Run
from .policy_deployment_models import PolicyDeployment


def require_visible_run(session, run_id, *, lock=False):
    query = select(Run).where(Run.id == run_id, Run.deleted_at.is_(None))
    if lock:
        query = query.with_for_update().execution_options(populate_existing=True)
    run = session.scalar(query)
    if run is None:
        raise APIError(404, "NOT_FOUND", "Run not found")
    return run


def require_replayable_run(session, run_id):
    run = session.get(Run, run_id)
    if run is None or run.deleted_at is not None:
        raise APIError(409, "RUN_DELETED", "该训练任务记录已删除，请重新创建训练并预检")
    return run


def delete_run_record(session, run_id):
    run = session.get(Run, run_id)
    if run is None:
        raise APIError(404, "NOT_FOUND", "Run not found")
    # Match lifecycle mutation lock ordering so completion/cancellation and
    # duplicate deletes cannot race against this terminal-state check.
    session.scalar(select(Node).where(Node.id == run.node_id).with_for_update())
    run = session.scalar(select(Run).where(Run.id == run_id).with_for_update()
                         .execution_options(populate_existing=True))
    if run.deleted_at is not None:
        return
    active_attempt = session.scalar(select(Attempt.id).where(
        Attempt.run_id == run_id, Attempt.state.not_in(TERMINAL)).limit(1))
    allocation = session.scalar(select(Allocation.attempt_id).join(
        Attempt, Allocation.attempt_id == Attempt.id).where(Attempt.run_id == run_id).limit(1))
    if run.state not in TERMINAL or active_attempt is not None or allocation is not None:
        raise APIError(409, "RUN_NOT_TERMINAL", "只能删除已结束的训练任务记录，请先取消任务并等待执行结束")
    runtime_records = select(DeploymentRecord.id).join(DeploymentRecordRuntime,
        DeploymentRecordRuntime.record_id == DeploymentRecord.id).join(PolicyDeployment,
        PolicyDeployment.id == DeploymentRecordRuntime.deployment_id).where(
        DeploymentRecord.deleted_at.is_(None), PolicyDeployment.source_run_id == run_id)
    transfer_records = select(DeploymentRecord.id).join(DeploymentRecordTransfer,
        DeploymentRecordTransfer.record_id == DeploymentRecord.id).join(CheckpointTransfer,
        CheckpointTransfer.id == DeploymentRecordTransfer.transfer_id).where(
        DeploymentRecord.deleted_at.is_(None), CheckpointTransfer.source_run_id == run_id)
    referenced_records = select(DeploymentRecord.id).join(DeploymentRecordRuntime,
        DeploymentRecordRuntime.record_id == DeploymentRecord.id).join(PolicyDeploymentTransfer,
        PolicyDeploymentTransfer.deployment_id == DeploymentRecordRuntime.deployment_id).join(CheckpointTransfer,
        CheckpointTransfer.id == PolicyDeploymentTransfer.transfer_id).where(
        DeploymentRecord.deleted_at.is_(None), CheckpointTransfer.source_run_id == run_id)
    linked_record = session.scalar(runtime_records.union(transfer_records, referenced_records).limit(1))
    if linked_record is not None:
        raise APIError(409, "RUN_HAS_DEPLOYMENTS", "该训练任务有关联部署记录，请先删除部署记录后再删除训练任务",
                       {"deployment_record_id": linked_record})
    # Keep attempts/events, consumed previews and idempotency keys intact:
    # registered checkpoints, existing deployments and delayed Agent delivery
    # still depend on these records. No machine-side command is issued.
    run.deleted_at = now()
