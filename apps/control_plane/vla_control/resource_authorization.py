"""Account-bound task ownership, independent of shared resource visibility."""

import hashlib

from sqlalchemy import select

from .auth_models import AuthGuard, User, UserSession
from .common import APIError, now


COMMAND_AUTHORIZATION_ERRORS = {
    "COMMAND_AUTHORIZATION_REVOKED": "原查询发起账户或权限已失效，查询未下发，请重新查询",
    "COMMAND_ACTOR_REQUIRED": "无法确认原查询发起账户，查询未下发，请重新查询",
}


def request_user_id(request):
    principal = request.state.principal
    return None if principal.legacy else principal.user_id


def require_expected_user(request, header="x-operation-user"):
    expected = request.headers.get(header)
    if expected is not None and expected != request.state.principal.user_id:
        raise APIError(403, "ACCOUNT_CHANGED", "登录账户已改变，请重新执行此操作")


def require_resource_owner(request, owner_user_id):
    principal = request.state.principal
    if principal.role == "admin":
        return
    if owner_user_id is None or owner_user_id != request_user_id(request):
        raise APIError(403, "RESOURCE_ACCESS_DENIED", "只能管理自己创建的任务；历史无归属任务由管理员管理")


def require_active_actor(session, request, permission="training.write"):
    """Serialize admission with account revocation before taking resource locks.

    AuthGuard must be the first mutation lock. A request admitted before a
    concurrent disable/role change must not enqueue new work after it commits.
    """
    session.execute(select(AuthGuard).where(AuthGuard.id == 1).with_for_update()).scalar_one()
    require_expected_user(request)
    principal = request.state.principal
    if principal.legacy:
        if session.scalar(select(User.id).limit(1)) is None:
            return
        raise APIError(403, "ACCOUNT_CHANGED", "账户权限已变更，请重新登录")
    from .auth import ROLE_PERMISSIONS
    user = session.get(User, principal.user_id, populate_existing=True)
    active = session.get(UserSession, principal.session_id) if principal.session_id else None
    manager = getattr(request.app.state, "auth", None)
    if (user is None or not user.enabled or active is None
            or active.user_id != principal.user_id or user.role != principal.role
            or manager is not None and not manager.session_active(active)):
        raise APIError(401, "UNAUTHORIZED", "登录账户已不可用，请重新登录")
    if user.must_change_password:
        raise APIError(403, "PASSWORD_CHANGE_REQUIRED", "请先修改初始密码")
    if permission not in ROLE_PERMISSIONS.get(user.role, frozenset()):
        raise APIError(403, "FORBIDDEN", "当前账户没有此操作的权限")
    from .authorization import SENSITIVE_ROUTES, request_route_path
    if manager is not None and (request.method, request_route_path(request)) in SENSITIVE_ROUTES:
        manager.require_recent_password(active)


def bind_command_actor(command, request, permission):
    """Persist the initiating account on a newly accepted asynchronous command."""
    if request is not None:
        command.actor_user_id = request_user_id(request)
        command.required_permission = permission
    return command


def execution_permission_authorized(session, actor_user_id, permission, *, allow_legacy=False):
    """Queued work survives session expiry, but cannot outlive account rights."""
    if actor_user_id is None:
        return bool(allow_legacy and session.scalar(select(User.id).limit(1)) is None)
    from .auth import ROLE_PERMISSIONS
    actor = session.get(User, actor_user_id, populate_existing=True)
    return bool(actor and actor.enabled and not actor.must_change_password
                and permission in ROLE_PERMISSIONS.get(actor.role, frozenset()))


def query_reusable(session, command, actor_request, permission):
    """Reuse shared query results without transferring the initiating account.

    Accepted work survives logout, and a delivered command retains retry
    semantics. Only an undelivered pending query is subject to actor revocation.
    The new caller can create a fresh query, but cannot adopt the old command.
    Empty-user offline fixtures retain their explicit legacy behavior.
    """
    command_permission = {
        "PATH_SUGGEST": "assets.query", "DATASET_INSPECT": "assets.query",
        "RECIPE_DETAIL": "recipes.query", "DATA_PROCESSORS": "recipes.query",
        "RECIPE_SUGGEST": "training.write" if command.payload.get("code_only", False) else "recipes.query",
    }.get(command.type, permission)
    if command_permission != permission:
        return False
    if command.status == "APPLIED":
        return True
    if command.status != "PENDING":
        return False
    if command.delivered_at is not None:
        return True
    legacy = (command.actor_user_id is None
              and session.scalar(select(User.id).limit(1)) is None)
    metadata_valid = command.required_permission == command_permission or (
        legacy and command.required_permission is None)
    if metadata_valid and execution_permission_authorized(
            session, command.actor_user_id, command_permission, allow_legacy=legacy):
        return True
    code = ("COMMAND_ACTOR_REQUIRED" if command.actor_user_id is None
            else "COMMAND_AUTHORIZATION_REVOKED")
    command.status = "REJECTED"
    command.result = {"error": {"code": code, "message": COMMAND_AUTHORIZATION_ERRORS[code]}}
    command.acknowledged_at = now()
    session.flush()
    return False


def scoped_request_key(request, key):
    """Preserve old offline fixture keys; human receipts are account-scoped."""
    user_id = request_user_id(request)
    if user_id is None:
        return key
    return "user:" + hashlib.sha256((user_id + "\0" + key).encode()).hexdigest()


def owner_fields(owner_user_id):
    return {"owner_user_id": owner_user_id}
