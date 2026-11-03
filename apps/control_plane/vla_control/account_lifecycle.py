"""Account checks for queued execution, independent of browser session expiry.

Short transactions that decide new execution or GPU reservation ownership lock
AuthGuard first, before Node/Run/resource rows. Account mutations use the same
guard, so revocation and a new dispatch cannot commit in the opposite order.
Never use these helpers to stop an already claimed execution or free Allocation.
"""

from sqlalchemy import select

from .auth_models import AuthGuard, User
from .common import now
from .models import Allocation, Attempt, GPUReservation, Run
from .policy_deployment_models import PolicyDeployment, RESOURCE_STATES


def lock_execution_guard(session) -> None:
    session.execute(select(AuthGuard).where(AuthGuard.id == 1).with_for_update()).scalar_one()


def training_execution_authorized(session, user_id, *, allow_legacy=False) -> bool:
    if user_id is None:
        return bool(allow_legacy and session.scalar(select(User.id).limit(1)) is None)
    from .auth import ROLE_PERMISSIONS
    user = session.get(User, user_id, populate_existing=True)
    return bool(user and user.enabled and not user.must_change_password
                and "training.write" in ROLE_PERMISSIONS.get(user.role, frozenset()))


def task_execution_authorized(session, owner_user_id, execution_user_id=None, *, allow_legacy=False) -> bool:
    """Explicit intervention stays valid only while its actor is an admin."""
    actor = execution_user_id if execution_user_id is not None else owner_user_id
    if not training_execution_authorized(session, actor, allow_legacy=allow_legacy):
        return False
    if actor == owner_user_id:
        return True
    user = session.get(User, actor, populate_existing=True)
    return bool(user and user.role == "admin")


def revoke_pending_work(session, user_id, reason="ACCOUNT_EXECUTION_REVOKED") -> None:
    """Freeze queued work and release idle reservations in an account mutation.

The caller already holds AuthGuard. Existing Attempts and allocations remain
intact, even if the account is disabled while a training process is running.
"""
    for run in session.scalars(select(Run).where(
            Run.owner_user_id == user_id, Run.state == "QUEUED").with_for_update()):
        run.queue_reason, run.updated_at = reason, now()
    active = set(session.execute(select(Allocation.node_id, Allocation.gpu_uuid)
        .join(Attempt, Attempt.id == Allocation.attempt_id)
        .join(Run, Run.id == Attempt.run_id).where(Run.owner_user_id == user_id)))
    active.update((row.node_id, row.gpu_uuid) for row in session.scalars(select(PolicyDeployment).where(
        (PolicyDeployment.execution_user_id == user_id)
        | (PolicyDeployment.execution_user_id.is_(None) & (PolicyDeployment.owner_user_id == user_id)),
        PolicyDeployment.state.in_(RESOURCE_STATES))) if row.gpu_uuid)
    for row in session.scalars(select(GPUReservation).where(GPUReservation.user_id == user_id)):
        if (row.node_id, row.gpu_uuid) not in active:
            session.delete(row)
