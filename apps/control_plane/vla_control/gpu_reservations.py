"""Per-account GPU reservations, enforced at every platform admission boundary."""

from datetime import datetime, timedelta, timezone

from fastapi import Request
from pydantic import Field, field_validator
from sqlalchemy import select

from .auth_models import User
from .common import APIError, now, offset_time
from .gpu_selection import requested_gpu_uuids
from .models import Allocation, Attempt, GPUReservation, Node, Run
from .policy_deployment_models import PolicyDeployment, RESOURCE_STATES
from .schemas import InputModel
from .resource_authorization import require_active_actor


class ReservationInput(InputModel):
    gpu_uuids: list[str] = Field(max_length=256)

    @field_validator("gpu_uuids", mode="before")
    @classmethod
    def validate_devices(cls, value):
        if value == []:
            return value
        return requested_gpu_uuids(value, len(value) if type(value) is list else 0)


def request_owner(request: Request) -> str | None:
    """Legacy bootstrap credentials have no personal reservation rights."""
    principal = request.state.principal
    return None if principal.legacy else principal.user_id


def reservation_expiry(row):
    """Legacy rows receive the default eight-hour policy, never infinity."""
    try:
        expiry = datetime.fromisoformat(row.expires_at) if row.expires_at else (
            datetime.fromisoformat(row.created_at) + timedelta(seconds=28800))
        if expiry.tzinfo is None:
            return None
        return expiry.astimezone(timezone.utc).isoformat(timespec="microseconds")
    except (ValueError, TypeError, OverflowError):
        return None


def _active_reservations(session, node_id=None):
    query = select(GPUReservation, User).join(User, User.id == GPUReservation.user_id)
    if node_id is not None:
        query = query.where(GPUReservation.node_id == node_id)
    from .auth import ROLE_PERMISSIONS
    stamp = now()
    return [(row, user, expiry) for row, user in session.execute(query)
            if (expiry := reservation_expiry(row)) is not None and expiry > stamp
            and user.enabled and not user.must_change_password
            and "training.write" in ROLE_PERMISSIONS.get(user.role, frozenset())]


def reservation_owners(session, node_id):
    return {row.gpu_uuid: row.user_id for row, _, _ in _active_reservations(session, node_id)}


def reservation_dicts(session, node_id):
    rows = sorted(_active_reservations(session, node_id), key=lambda item: (item[0].created_at, item[0].gpu_uuid))
    return [{"node_id": row.node_id, "gpu_uuid": row.gpu_uuid, "user_id": row.user_id,
             "username": user.username, "display_name": user.display_name, "created_at": row.created_at,
             "expires_at": expiry}
            for row, user, expiry in rows]


def _prune_inactive_reservations(session, node_id):
    active_keys = {(row.node_id, row.gpu_uuid) for row, _, _ in _active_reservations(session, node_id)}
    for row in session.scalars(select(GPUReservation).where(GPUReservation.node_id == node_id)):
        if (row.node_id, row.gpu_uuid) not in active_keys:
            session.delete(row)
    session.flush()


def require_gpu_access(session, node_id, user_id, gpu_uuids):
    if not gpu_uuids:
        return
    owners = reservation_owners(session, node_id)
    blocked = [uuid for uuid in gpu_uuids if uuid in owners and owners[uuid] != user_id]
    if blocked:
        raise APIError(409, "GPU_RESERVED_BY_OTHER_USER", "所选 GPU 已由其他用户预留，请选择其他 GPU", {"gpu_uuids": blocked})


def _node(session, node_id, *, lock=False):
    query = select(Node).where(Node.id == node_id)
    if lock:
        query = query.with_for_update().execution_options(populate_existing=True)
    node = session.scalar(query)
    if node is None:
        raise APIError(404, "NOT_FOUND", "机器不存在")
    return node


def _validate_additions(session, node, user_id, additions, settings):
    if not additions:
        return
    from .remote_models import RemoteDeployment
    if session.scalar(select(RemoteDeployment.id).where(RemoteDeployment.node_id == node.id,
            RemoteDeployment.state.in_({"QUEUED", "RUNNING", "WAITING_AGENT"})).limit(1)):
        raise APIError(409, "NODE_MAINTENANCE", "机器正在管理或升级 Agent，请等待完成后新增 GPU 预留")
    if node.status != "ONLINE" or node.last_seen < offset_time(-settings.heartbeat_timeout_seconds):
        raise APIError(409, "NODE_OFFLINE", "机器离线或心跳已过期，暂时不能新增 GPU 预留")
    inventory = {gpu["uuid"]: gpu for gpu in node.gpus}
    if not additions <= inventory.keys():
        raise APIError(422, "GPU_NOT_FOUND", "所选 GPU 不属于该机器，请刷新后重新选择")
    # The node lock serializes these checks with training and deployment
    # admissions. An existing task is never stopped to grant a reservation.
    occupants = {}
    rows = session.execute(select(Allocation.gpu_uuid, Run.owner_user_id)
        .join(Attempt, Attempt.id == Allocation.attempt_id)
        .join(Run, Run.id == Attempt.run_id).where(Allocation.node_id == node.id))
    for uuid, owner in rows:
        occupants.setdefault(uuid, []).append(owner)
    for row in session.scalars(select(PolicyDeployment).where(
            PolicyDeployment.node_id == node.id, PolicyDeployment.state.in_(RESOURCE_STATES))):
        if row.gpu_uuid:
            actor = row.execution_user_id if row.execution_user_id is not None else row.owner_user_id
            occupants.setdefault(row.gpu_uuid, []).append(actor)
    from .scheduler import gpu_available
    for uuid in additions:
        owners = occupants.get(uuid, [])
        if any(owner != user_id for owner in owners) or not owners and not gpu_available(inventory[uuid]):
            raise APIError(409, "GPU_IN_USE", "GPU 正在被其他任务或机器进程占用，当前无法预留", {"gpu_uuid": uuid})


def register_routes(app, dependencies, db, auth, settings):
    @app.get("/api/v1/gpu-reservation-policy", dependencies=dependencies)
    def reservation_policy(request: Request):
        user_id = request_owner(request)
        with db.read() as session:
            count = sum(1 for row, _, _ in _active_reservations(session) if row.user_id == user_id)
        return {"max_per_user": settings.gpu_reservation_max_per_user,
                "ttl_seconds": settings.gpu_reservation_ttl_seconds,
                "reserved_count": count,
                "remaining_count": max(0, settings.gpu_reservation_max_per_user - count)}

    @app.get("/api/v1/nodes/{node_id}/gpu-reservations", dependencies=dependencies)
    def list_reservations(node_id: str):
        with db.read() as session:
            _node(session, node_id)
            return reservation_dicts(session, node_id)

    @app.put("/api/v1/nodes/{node_id}/gpu-reservations", dependencies=dependencies)
    def save_reservations(node_id: str, body: ReservationInput, request: Request):
        user_id = request_owner(request)
        if user_id is None:
            raise APIError(403, "GPU_RESERVATION_ACCOUNT_REQUIRED", "请使用个人账户登录后申请 GPU 预留")
        with db.write() as session:
            # AuthGuard precedes Node/resource locks. It also serializes the
            # per-account count across simultaneous requests on different nodes.
            require_active_actor(session, request)
            node = _node(session, node_id, lock=True)
            _prune_inactive_reservations(session, node_id)
            rows = list(session.scalars(select(GPUReservation).where(GPUReservation.node_id == node_id)))
            owners = {row.gpu_uuid: row.user_id for row in rows}
            selected = set(body.gpu_uuids)
            blocked = [uuid for uuid in body.gpu_uuids if uuid in owners and owners[uuid] != user_id]
            if blocked:
                raise APIError(409, "GPU_ALREADY_RESERVED", "GPU 已由其他用户预留，请刷新后重新选择", {"gpu_uuids": blocked})
            other_nodes = sum(1 for row, _, _ in _active_reservations(session)
                              if row.user_id == user_id and row.node_id != node_id)
            if selected and other_nodes + len(selected) > settings.gpu_reservation_max_per_user:
                raise APIError(409, "GPU_RESERVATION_LIMIT", "GPU 预留数量超过当前账户上限", {
                    "limit": settings.gpu_reservation_max_per_user, "reserved_on_other_nodes": other_nodes})
            additions = selected - owners.keys()
            _validate_additions(session, node, user_id, additions, settings)
            for row in rows:
                if row.user_id == user_id and row.gpu_uuid not in selected:
                    session.delete(row)
            stamp = now()
            expiry = offset_time(settings.gpu_reservation_ttl_seconds)
            for uuid in body.gpu_uuids:
                if uuid in additions:
                    session.add(GPUReservation(node_id=node_id, gpu_uuid=uuid, user_id=user_id,
                                               created_at=stamp, expires_at=expiry))
                else:
                    next(row for row in rows if row.gpu_uuid == uuid).expires_at = expiry
            session.flush()
            return reservation_dicts(session, node_id)

    @app.delete("/api/v1/nodes/{node_id}/gpu-reservations/{gpu_uuid}", dependencies=dependencies)
    def release_reservation(node_id: str, gpu_uuid: str, request: Request):
        principal = request.state.principal
        with db.write() as session:
            require_active_actor(session, request)
            _node(session, node_id, lock=True)
            row = session.get(GPUReservation, (node_id, gpu_uuid))
            if row is None:
                raise APIError(404, "NOT_FOUND", "GPU 预留不存在")
            if row.user_id != request_owner(request) and "machines.manage" not in principal.permissions:
                raise APIError(403, "FORBIDDEN", "只能释放自己的 GPU 预留")
            session.delete(row)
            return {"node_id": node_id, "gpu_uuid": gpu_uuid, "deleted": True}
