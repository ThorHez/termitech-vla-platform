"""Human accounts, expiring sessions and server-enforced role permissions.

Node enrollment and per-Agent tokens deliberately do not pass through this module.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import base64
import hashlib
import hmac
import re
import secrets
from urllib.parse import urlsplit

from fastapi import FastAPI, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import delete, func, select

from .auth_models import AuditEvent, AuthGuard, LoginThrottle, User, UserSession
from .authorization import SENSITIVE_ROUTES, request_route_path
from .common import APIError, now, offset_time, token_hash, uid
from .db import Database
from .settings import Settings

COOKIE_NAME = "vla_session"
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
ROLES = ("admin", "operator", "viewer")
PERMISSIONS = frozenset({"platform.read", "training.write", "assets.write", "recipes.write",
                         "machines.manage", "recipes.writeback", "users.manage", "audit.read"})
ROLE_PERMISSIONS = {
    "admin": PERMISSIONS,
    "operator": frozenset({"platform.read", "training.write", "assets.write", "recipes.write"}),
    "viewer": frozenset({"platform.read"}),
}
USERNAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,79}$")
PASSWORD_MIN = 12
PASSWORD_MAX = 256
THROTTLE_WINDOW = 900
THROTTLE_ROWS = 10000
_AUDIT_FIELDS = {"target_user_id", "target_username", "role", "enabled", "permission", "code", "count"}


def normalize_username(value: str) -> str:
    return value.strip().lower()


def validate_username(value: str) -> str:
    value = normalize_username(value)
    if not USERNAME.fullmatch(value):
        raise APIError(422, "INVALID_USERNAME", "用户名须为 3 至 80 位小写字母、数字、点、下划线或连字符，并以字母或数字开头")
    return value


def validate_password(value: str) -> None:
    if not PASSWORD_MIN <= len(value) <= PASSWORD_MAX:
        raise APIError(422, "WEAK_PASSWORD", f"密码长度须为 {PASSWORD_MIN} 至 {PASSWORD_MAX} 个字符")


def hash_password(value: str) -> str:
    validate_password(value)
    salt = secrets.token_bytes(16)
    derived = hashlib.scrypt(value.encode(), salt=salt, n=32768, r=8, p=3, maxmem=64 * 1024 * 1024, dklen=32)
    return "scrypt$32768$8$3$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(derived).decode()


def verify_password(value: str, encoded: str) -> bool:
    if not isinstance(value, str) or len(value) > PASSWORD_MAX:
        return False
    try:
        algorithm, n, r, p, salt, expected = encoded.split("$")
        # Stored parameters are bounded to keep a corrupted database from
        # turning authentication into an unbounded memory allocation.
        if (algorithm, n, r, p) != ("scrypt", "32768", "8", "3"):
            return False
        decoded_salt = base64.b64decode(salt, validate=True)
        decoded_expected = base64.b64decode(expected, validate=True)
        if len(decoded_salt) != 16 or len(decoded_expected) != 32:
            return False
        actual = hashlib.scrypt(value.encode(), salt=decoded_salt, n=32768, r=8, p=3,
                                maxmem=64 * 1024 * 1024, dklen=32)
        return hmac.compare_digest(actual, decoded_expected)
    except (ValueError, TypeError):
        return False


# A fixed, non-secret hash gives absent and disabled accounts the same password
# derivation work as an incorrect password on a real account.
_DUMMY_HASH = hash_password("no-account-password-placeholder")


def user_dict(user: User) -> dict:
    return {field: getattr(user, field) for field in (
        "id", "username", "display_name", "role", "enabled", "must_change_password", "created_at", "updated_at")}


@dataclass(frozen=True)
class Principal:
    user_id: str
    username: str
    role: str
    permissions: frozenset[str]
    session_id: str | None = None
    legacy: bool = False
    must_change_password: bool = False


class AuthManager:
    def __init__(self, db: Database, settings: Settings):
        self.db = db
        self.settings = settings

    def lock_accounts(self, session) -> None:
        session.execute(select(AuthGuard).where(AuthGuard.id == 1).with_for_update()).scalar_one()

    @staticmethod
    def check_origin(request: Request) -> None:
        if request.method in SAFE_METHODS:
            return
        origin = request.headers.get("origin")
        if not origin:
            return
        try:
            supplied = urlsplit(origin)
            expected = urlsplit(str(request.base_url))
            supplied_origin = (supplied.scheme, supplied.hostname, supplied.port or (443 if supplied.scheme == "https" else 80))
            expected_origin = (expected.scheme, expected.hostname, expected.port or (443 if expected.scheme == "https" else 80))
        except ValueError:
            raise APIError(403, "ORIGIN_REJECTED", "请求来源不被允许") from None
        if supplied.scheme not in {"http", "https"} or supplied.username or supplied.password or supplied.path not in {"", "/"} or supplied.query or supplied.fragment or supplied_origin != expected_origin:
            raise APIError(403, "ORIGIN_REJECTED", "请求来源不被允许")

    def current(self, request: Request) -> Principal:
        raw = request.cookies.get(COOKIE_NAME, "")
        if raw and len(raw) <= 200:
            with self.db.read() as session:
                row = session.scalar(select(UserSession).where(UserSession.token_hash == token_hash(raw)))
                user = session.get(User, row.user_id) if self.session_active(row) else None
            if row and user and user.enabled:
                principal = Principal(user.id, user.username, user.role, ROLE_PERMISSIONS.get(user.role, frozenset()),
                                      row.id, False, user.must_change_password)
                request.state.principal = principal
                return principal
        authorization = request.headers.get("authorization", "")
        if self.settings.legacy_admin_enabled and self.settings.admin_token and authorization.startswith("Bearer "):
            with self.db.read() as session:
                any_users = session.scalar(select(User.id).limit(1)) is not None
            if not any_users and hmac.compare_digest(authorization[7:].encode(), self.settings.admin_token.encode()):
                principal = Principal("legacy-admin", "legacy-admin", "admin", PERMISSIONS, legacy=True)
                request.state.principal = principal
                return principal
        raise APIError(401, "UNAUTHORIZED", "请登录后继续")

    def require(self, request: Request, permission: str | None = None) -> Principal:
        self.check_origin(request)
        principal = self.current(request)
        if request.method not in SAFE_METHODS and not principal.legacy:
            with self.db.read() as session:
                row = session.get(UserSession, principal.session_id)
            supplied = request.headers.get("x-csrf-token", "")
            if not row or not supplied or not hmac.compare_digest(supplied.encode(), row.csrf_token.encode()):
                raise APIError(403, "CSRF_REJECTED", "请求校验已失效，请刷新页面后重试")
        if principal.must_change_password and request_route_path(request) not in {
            "/api/v1/auth/me", "/api/v1/auth/password", "/api/v1/auth/logout", "/api/v1/auth/activity"}:
            raise APIError(403, "PASSWORD_CHANGE_REQUIRED", "请先修改初始密码")
        if permission is not None and permission not in principal.permissions:
            raise APIError(403, "FORBIDDEN", "当前账户没有此操作的权限", {"permission": permission})
        if not principal.legacy and (request.method, request_route_path(request)) in SENSITIVE_ROUTES:
            with self.db.read() as session:
                row = session.get(UserSession, principal.session_id)
                self.require_recent_password(row)
        return principal

    def deadlines(self, row: UserSession) -> tuple[str, str, str | None]:
        """Always apply the current policy, including to cookies issued earlier."""
        created = datetime.fromisoformat(row.created_at)
        activity = datetime.fromisoformat(row.last_activity_at or "")
        expiry = min(datetime.fromisoformat(row.expires_at),
                     created + timedelta(seconds=self.settings.session_ttl_seconds))
        idle = min(expiry, activity + timedelta(seconds=self.settings.session_idle_seconds))
        proof = min(expiry, idle, datetime.fromisoformat(row.reauthenticated_at)
                    + timedelta(seconds=self.settings.session_reauth_seconds)) if row.reauthenticated_at else None
        return (expiry.isoformat(timespec="microseconds"), idle.isoformat(timespec="microseconds"),
                proof.isoformat(timespec="microseconds") if proof else None)

    def session_active(self, row: UserSession | None) -> bool:
        if row is None:
            return False
        try:
            expiry, idle, _ = self.deadlines(row)
            return min(expiry, idle) > now()
        except (ValueError, TypeError, OverflowError):
            return False

    def require_recent_password(self, row: UserSession | None) -> None:
        if not self.session_active(row):
            raise APIError(401, "UNAUTHORIZED", "登录状态已过期，请重新登录")
        proof = self.deadlines(row)[2]
        if proof is None or proof <= now():
            raise APIError(403, "REAUTH_REQUIRED", "此操作需要再次验证当前密码")

    def _audit(self, session, request: Request | None, action: str, status: int, details: dict | None = None,
               actor_id: str | None = None, actor_username: str | None = None) -> None:
        principal = getattr(request.state, "principal", None) if request else None
        clean = {key: value for key, value in (details or {}).items()
                 if key in _AUDIT_FIELDS and (value is None or isinstance(value, (str, bool, int)))}
        clean = {key: value[:200] if isinstance(value, str) else value for key, value in clean.items()}
        session.add(AuditEvent(id=uid(), created_at=now(), actor_id=actor_id or (principal.user_id if principal else None),
                               actor_username=actor_username or (principal.username if principal else None),
                               action=action[:120], method=request.method if request else "CLI",
                               path=request.url.path[:600] if request else "local-console", status=status, details=clean,
                               remote_addr=(request.client.host[:100] if request and request.client else "local")))

    def audit(self, request: Request | None, action: str, status: int, details: dict | None = None) -> None:
        """Record request metadata only; passwords, tokens, bodies and queries are never copied."""
        with self.db.write() as session:
            self._audit(session, request, action, status, details)

    def _new_session(self, session, user: User) -> tuple[dict, str]:
        raw = secrets.token_urlsafe(48)
        row = UserSession(id=uid(), user_id=user.id, token_hash=token_hash(raw), csrf_token=secrets.token_urlsafe(32),
                          created_at=now(), expires_at=offset_time(self.settings.session_ttl_seconds),
                          last_activity_at=now(), reauthenticated_at=None)
        session.execute(delete(UserSession).where(UserSession.expires_at <= now()))
        session.add(row)
        session.flush()
        stale = session.scalars(select(UserSession.id).where(UserSession.user_id == user.id)
                                .order_by(UserSession.created_at.desc(), UserSession.id.desc()).offset(10)).all()
        if stale:
            session.execute(delete(UserSession).where(UserSession.id.in_(stale)))
        return self.payload(user, row), raw

    def payload(self, user: User, row: UserSession) -> dict:
        expiry, idle, proof = self.deadlines(row)
        return {"user": user_dict(user), "permissions": sorted(ROLE_PERMISSIONS.get(user.role, frozenset())),
                "csrf_token": row.csrf_token, "expires_at": expiry, "idle_expires_at": idle,
                "session_idle_seconds": self.settings.session_idle_seconds, "reauth_expires_at": proof,
                "server_time": now()}

    @staticmethod
    def cookie_path(request: Request | None = None) -> str:
        root = request.scope.get("root_path", "").rstrip("/") if request else ""
        return root + "/api/v1"

    def set_cookie(self, response: Response, raw: str, request: Request | None = None) -> None:
        response.set_cookie(COOKIE_NAME, raw, max_age=self.settings.session_ttl_seconds, path=self.cookie_path(request),
                            secure=self.settings.session_cookie_secure, httponly=True, samesite="strict")
        response.headers["Cache-Control"] = "no-store"

    def _throttle_keys(self, request: Request, username: str) -> tuple[str, str]:
        remote = request.client.host if request.client else "unknown"
        return token_hash("user:" + username), token_hash("ip:" + remote)

    def _verification_keys(self, request: Request, user_id: str) -> tuple[str, str]:
        remote = request.client.host if request.client else "unknown"
        return token_hash("reauth:user:" + user_id), token_hash("reauth:ip:" + remote)

    def _check_throttle(self, session, keys: tuple[str, str]) -> None:
        stamp = now()
        for key in keys:
            row = session.get(LoginThrottle, key)
            if row and row.locked_until and row.locked_until > stamp:
                raise APIError(429, "LOGIN_RATE_LIMITED", "登录尝试过于频繁，请稍后再试")

    def _record_failure(self, session, keys: tuple[str, str]) -> None:
        stamp = now()
        session.execute(delete(LoginThrottle).where(LoginThrottle.expires_at <= stamp))
        for index, key in enumerate(keys):
            row = session.get(LoginThrottle, key)
            if row is None:
                row = LoginThrottle(key=key, failures=0, window_started_at=stamp, expires_at=offset_time(THROTTLE_WINDOW * 2))
                session.add(row)
            if row.window_started_at <= (datetime.now(timezone.utc) - timedelta(seconds=THROTTLE_WINDOW)).isoformat(timespec="microseconds"):
                row.failures = 0
                row.window_started_at = stamp
            row.failures += 1
            row.expires_at = offset_time(THROTTLE_WINDOW * 2)
            if row.failures >= (5 if index == 0 else 30):
                row.locked_until = offset_time(THROTTLE_WINDOW)
        session.flush()
        excess = (session.scalar(select(func.count()).select_from(LoginThrottle)) or 0) - THROTTLE_ROWS
        if excess > 0:
            oldest = session.scalars(select(LoginThrottle.key).order_by(LoginThrottle.expires_at).limit(excess)).all()
            session.execute(delete(LoginThrottle).where(LoginThrottle.key.in_(oldest)))

    def activity(self, request: Request) -> dict:
        principal = self.require(request)
        if principal.legacy:
            raise APIError(401, "ACCOUNT_REQUIRED", "请先创建账户")
        with self.db.write() as session:
            self.lock_accounts(session)
            user = session.get(User, principal.user_id)
            row = session.get(UserSession, principal.session_id)
            if not user or not user.enabled or not self.session_active(row):
                raise APIError(401, "UNAUTHORIZED", "登录状态已过期，请重新登录")
            row.last_activity_at = now()
            return self.payload(user, row)

    def reauthenticate(self, request: Request, password: str) -> dict:
        principal = self.require(request)
        if principal.legacy:
            raise APIError(401, "ACCOUNT_REQUIRED", "请先创建账户")
        keys = self._verification_keys(request, principal.user_id)
        with self.db.read() as session:
            self._check_throttle(session, keys)
            user = session.get(User, principal.user_id)
        valid = verify_password(password, user.password_hash if user and user.enabled else _DUMMY_HASH)
        failure = None
        result = None
        with self.db.write() as session:
            self.lock_accounts(session)
            self._check_throttle(session, keys)
            current = session.get(User, principal.user_id)
            row = session.get(UserSession, principal.session_id)
            if not current or not current.enabled or not self.session_active(row):
                raise APIError(401, "UNAUTHORIZED", "登录状态已过期，请重新登录")
            if not valid or not hmac.compare_digest(current.password_hash, user.password_hash):
                self._record_failure(session, keys)
                self._audit(session, request, "auth.reauth_failed", 403)
                failure = APIError(403, "INVALID_PASSWORD", "当前密码不正确")
            else:
                session.execute(delete(LoginThrottle).where(LoginThrottle.key == keys[0]))
                row.reauthenticated_at = now()
                result = self.payload(current, row)
                self._audit(session, request, "auth.reauth", 200)
        if failure:
            raise failure
        return result

    def login(self, request: Request, response: Response, username: str, password: str) -> dict:
        self.check_origin(request)
        # Switching an already authenticated browser also needs its CSRF token.
        try:
            existing = self.current(request)
        except APIError:
            existing = None
        if existing:
            self.require(request)
        username = normalize_username(username)
        keys = self._throttle_keys(request, username)
        with self.db.read() as session:
            self._check_throttle(session, keys)
            user = session.scalar(select(User).where(User.username == username))
        valid = verify_password(password, user.password_hash if user and user.enabled else _DUMMY_HASH)
        result = None
        failure = None
        with self.db.write() as session:
            self.lock_accounts(session)
            self._check_throttle(session, keys)
            # Re-read after hashing: password changes and disables invalidate
            # an authentication attempt that started before the change.
            current = session.get(User, user.id) if user else None
            valid = bool(valid and current and current.enabled and hmac.compare_digest(current.password_hash, user.password_hash))
            stamp = now()
            session.execute(delete(LoginThrottle).where(LoginThrottle.expires_at <= stamp))
            if not valid:
                self._record_failure(session, keys)
                self._audit(session, request, "auth.login_failed", 401)
                failure = APIError(401, "INVALID_CREDENTIALS", "用户名或密码不正确")
            else:
                session.execute(delete(LoginThrottle).where(LoginThrottle.key == keys[0]))
                if existing and existing.session_id:
                    session.execute(delete(UserSession).where(UserSession.id == existing.session_id))
                result, raw = self._new_session(session, current)
                request.state.principal = Principal(current.id, current.username, current.role,
                                                    ROLE_PERMISSIONS[current.role], must_change_password=current.must_change_password)
                self._audit(session, request, "auth.login", 200)
        if failure:
            raise failure
        self.set_cookie(response, raw, request)
        return result

    def _authorize_account_mutation(self, session, request: Request | None) -> None:
        if request is None:
            return
        principal = getattr(request.state, "principal", None)
        if principal and principal.legacy:
            if self.settings.legacy_admin_enabled and not session.scalar(select(User.id).limit(1)):
                return
        elif principal:
            user = session.get(User, principal.user_id)
            active = session.get(UserSession, principal.session_id) if principal.session_id else None
            if not user or not user.enabled or not self.session_active(active):
                raise APIError(401, "UNAUTHORIZED", "登录状态已变更，请重新登录")
            if user.role == "admin" and not user.must_change_password:
                self.require_recent_password(active)
                return
        raise APIError(403, "FORBIDDEN", "账户权限已变更，请重新登录")

    def create_user(self, username: str, password: str, role: str = "admin", display_name: str | None = None,
                    *, request: Request | None = None, force_change: bool = False, initial_only: bool = False) -> dict:
        username = validate_username(username)
        if role not in ROLES:
            raise APIError(422, "INVALID_ROLE", "角色须为 admin、operator 或 viewer")
        encoded = hash_password(password)
        with self.db.write() as session:
            self.lock_accounts(session)
            self._authorize_account_mutation(session, request)
            if initial_only and session.scalar(select(User.id).limit(1)):
                raise APIError(409, "ADMIN_ALREADY_INITIALIZED", "平台已存在账户，不能重复初始化管理员")
            if session.scalar(select(User.id).where(User.username == username)):
                raise APIError(409, "USERNAME_TAKEN", "用户名已存在")
            stamp = now()
            user = User(id=uid(), username=username, display_name=(display_name or username).strip()[:120], role=role,
                        enabled=True, password_hash=encoded, must_change_password=force_change,
                        created_at=stamp, updated_at=stamp)
            session.add(user)
            session.flush()
            self._audit(session, request, "users.create", 201, {"target_user_id": user.id, "target_username": user.username, "role": role})
            return user_dict(user)

    def reset_password(self, user_id: str, password: str, request: Request | None = None,
                       *, force_change: bool = True) -> dict:
        encoded = hash_password(password)
        with self.db.write() as session:
            self.lock_accounts(session)
            self._authorize_account_mutation(session, request)
            user = session.get(User, user_id)
            if not user:
                raise APIError(404, "NOT_FOUND", "账户不存在")
            user.password_hash = encoded
            user.must_change_password = force_change
            user.updated_at = now()
            session.execute(delete(UserSession).where(UserSession.user_id == user.id))
            self._audit(session, request, "users.reset_password", 200, {"target_user_id": user.id})
            return user_dict(user)

    def update_user(self, request: Request | None, user_id: str, changes: dict) -> dict:
        with self.db.write() as session:
            self.lock_accounts(session)
            self._authorize_account_mutation(session, request)
            user = session.get(User, user_id)
            if not user:
                raise APIError(404, "NOT_FOUND", "账户不存在")
            role = changes.get("role", user.role)
            enabled = changes.get("enabled", user.enabled)
            if role not in ROLES:
                raise APIError(422, "INVALID_ROLE", "角色须为 admin、operator 或 viewer")
            if user.enabled and user.role == "admin" and (not enabled or role != "admin"):
                enabled_admins = session.scalar(select(func.count()).select_from(User).where(User.enabled.is_(True), User.role == "admin"))
                if enabled_admins <= 1:
                    raise APIError(409, "LAST_ADMIN_REQUIRED", "至少须保留一个启用的管理员账户")
            revoke = user.role != role or user.enabled != enabled
            user.role = role
            user.enabled = enabled
            if "display_name" in changes:
                user.display_name = changes["display_name"].strip() or user.username
            user.updated_at = now()
            if revoke:
                session.execute(delete(UserSession).where(UserSession.user_id == user.id))
            self._audit(session, request, "users.update", 200, {"target_user_id": user.id, "role": role, "enabled": enabled})
            return user_dict(user)


class LoginInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=PASSWORD_MAX)


class PasswordInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    current_password: str = Field(min_length=1, max_length=PASSWORD_MAX)
    new_password: str = Field(min_length=PASSWORD_MIN, max_length=PASSWORD_MAX)


class ReauthenticationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    password: str = Field(min_length=1, max_length=PASSWORD_MAX)


class UserCreateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=3, max_length=80)
    display_name: str | None = Field(default=None, max_length=120)
    role: str = "viewer"
    password: str = Field(min_length=PASSWORD_MIN, max_length=PASSWORD_MAX)


class UserUpdateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    display_name: str | None = Field(default=None, max_length=120)
    role: str | None = None
    enabled: bool | None = None


class ResetPasswordInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    password: str = Field(min_length=PASSWORD_MIN, max_length=PASSWORD_MAX)


def register_auth_routes(app: FastAPI, manager: AuthManager) -> None:
    @app.get("/api/v1/auth/me")
    def me(request: Request, response: Response):
        principal = manager.require(request)
        if principal.legacy:
            raise APIError(401, "ACCOUNT_REQUIRED", "请通过服务器控制台创建管理员账户后登录")
        with manager.db.read() as session:
            user = session.get(User, principal.user_id)
            row = session.get(UserSession, principal.session_id)
            if not user or not user.enabled or not manager.session_active(row):
                raise APIError(401, "UNAUTHORIZED", "登录状态已变更，请重新登录")
            response.headers["Cache-Control"] = "no-store"
            return manager.payload(user, row)

    @app.post("/api/v1/auth/login")
    def login(body: LoginInput, request: Request, response: Response):
        return manager.login(request, response, body.username, body.password)

    @app.post("/api/v1/auth/activity")
    def activity(request: Request, response: Response):
        response.headers["Cache-Control"] = "no-store"
        return manager.activity(request)

    @app.post("/api/v1/auth/reauth")
    def reauthenticate(body: ReauthenticationInput, request: Request, response: Response):
        response.headers["Cache-Control"] = "no-store"
        return manager.reauthenticate(request, body.password)

    @app.post("/api/v1/auth/logout", status_code=204)
    def logout(request: Request, response: Response):
        principal = manager.require(request)
        with manager.db.write() as session:
            if principal.session_id:
                session.execute(delete(UserSession).where(UserSession.id == principal.session_id))
            manager._audit(session, request, "auth.logout", 204)
        response.delete_cookie(COOKIE_NAME, path=manager.cookie_path(request), secure=manager.settings.session_cookie_secure, httponly=True, samesite="strict")
        response.headers["Cache-Control"] = "no-store"

    @app.post("/api/v1/auth/password")
    def password(body: PasswordInput, request: Request, response: Response):
        principal = manager.require(request)
        if principal.legacy:
            raise APIError(401, "ACCOUNT_REQUIRED", "请先创建账户")
        keys = manager._verification_keys(request, principal.user_id)
        with manager.db.read() as session:
            manager._check_throttle(session, keys)
            user = session.get(User, principal.user_id)
        if not user or not verify_password(body.current_password, user.password_hash):
            with manager.db.write() as session:
                manager.lock_accounts(session)
                manager._check_throttle(session, keys)
                current = session.get(User, principal.user_id)
                active = session.get(UserSession, principal.session_id)
                if not current or not current.enabled or not manager.session_active(active):
                    raise APIError(401, "UNAUTHORIZED", "登录状态已变更，请重新登录")
                manager._record_failure(session, keys)
                manager._audit(session, request, "auth.password_failed", 403)
            raise APIError(403, "INVALID_PASSWORD", "当前密码不正确")
        if hmac.compare_digest(body.current_password.encode(), body.new_password.encode()):
            raise APIError(422, "PASSWORD_UNCHANGED", "新密码不能与当前密码相同")
        encoded = hash_password(body.new_password)
        with manager.db.write() as session:
            manager.lock_accounts(session)
            manager._check_throttle(session, keys)
            current = session.get(User, principal.user_id)
            active_session = session.get(UserSession, principal.session_id)
            if not current or not current.enabled or not manager.session_active(active_session) or not hmac.compare_digest(current.password_hash, user.password_hash):
                raise APIError(401, "UNAUTHORIZED", "登录状态已变更，请重新登录")
            current.password_hash = encoded
            current.must_change_password = False
            current.updated_at = now()
            session.execute(delete(UserSession).where(UserSession.user_id == current.id))
            session.execute(delete(LoginThrottle).where(LoginThrottle.key == keys[0]))
            result, raw = manager._new_session(session, current)
            manager._audit(session, request, "auth.password", 200)
        manager.set_cookie(response, raw, request)
        return result

    @app.get("/api/v1/users")
    def users(request: Request):
        manager.require(request, "users.manage")
        with manager.db.read() as session:
            return [user_dict(user) for user in session.scalars(select(User).order_by(User.username))]

    @app.post("/api/v1/users", status_code=201)
    def create(body: UserCreateInput, request: Request):
        manager.require(request, "users.manage")
        return manager.create_user(body.username, body.password, body.role, body.display_name, request=request, force_change=True)

    @app.patch("/api/v1/users/{user_id}")
    def update(user_id: str, body: UserUpdateInput, request: Request):
        manager.require(request, "users.manage")
        changes = body.model_dump(exclude_unset=True)
        if any(value is None for value in changes.values()):
            raise APIError(422, "INVALID_UPDATE", "更新字段不能为 null")
        return manager.update_user(request, user_id, changes)

    @app.post("/api/v1/users/{user_id}/reset-password")
    def reset(user_id: str, body: ResetPasswordInput, request: Request):
        manager.require(request, "users.manage")
        return manager.reset_password(user_id, body.password, request)

    @app.get("/api/v1/audit")
    def audit(request: Request, limit: int = Query(default=100, ge=1, le=500)):
        manager.require(request, "audit.read")
        with manager.db.read() as session:
            rows = session.scalars(select(AuditEvent).order_by(AuditEvent.created_at.desc(), AuditEvent.id.desc()).limit(limit)).all()
            return [{field: getattr(row, field) for field in (
                "id", "created_at", "actor_id", "actor_username", "action", "method", "path", "status", "details", "remote_addr")}
                    for row in rows]
