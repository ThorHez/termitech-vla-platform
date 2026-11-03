"""Bounded SSH metadata discovery and atomic, account-owned dataset batches."""

from copy import deepcopy
import hashlib
import json
from pathlib import PurePosixPath
import threading
import time

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import Field, StrictBool, ValidationError, field_validator, model_validator
from sqlalchemy import func, select, text
from starlette.concurrency import run_in_threadpool

from .auth import ROLE_PERMISSIONS
from .auth_models import User
from .checkpoint_transfers import _connection
from .common import APIError, digest, now, offset_time, require_path, uid
from .dataset_batch_models import DatasetBatchRegistration, DatasetScan
from .dataset_norm_bindings import _lock_norm_path, canonical_norm_path, resolve_norm_path, validate_norm_binding
from .dataset_profiles import normalize_dataset_profile
from .dataset_validations import queue_validation
from .models import Asset, Node
from .remote_models import RemoteConnection, RemoteDeployment
from .schemas import InputModel
from .serialization import asset_dict
from .account_lifecycle import lock_execution_guard
from .resource_authorization import require_active_actor


WORKING = {"QUEUED", "SCANNING"}
MAX_ENTRIES = 200
MAX_BODY = 1024 * 1024
SCAN_TTL = 1800


def _path(value, roots=None, *, allow_root=False):
    if (not isinstance(value, str) or not value.strip() or len(value) > 4096 or "\\" in value
            or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in value)):
        raise APIError(422, "DATASET_SCAN_PATH_INVALID", "请填写有效的数据集父目录绝对路径")
    value = value.strip()
    require_path(value, roots)
    path = "/" + str(PurePosixPath(value)).lstrip("/")
    if path == "/" and not allow_root:
        raise APIError(422, "DATASET_SCAN_PATH_INVALID", "请选择数据集所在目录，不能扫描文件系统根目录")
    try:
        path.encode("utf-8")
    except UnicodeError:
        raise APIError(422, "DATASET_SCAN_PATH_INVALID", "目录包含无效文本") from None
    return path


def _text(value, maximum, label):
    if (not isinstance(value, str) or not value.strip() or len(value.strip()) > maximum
            or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in value)):
        raise ValueError(label + "不能为空、过长或包含控制字符")
    value.encode("utf-8")
    return value.strip()


class ScanInput(InputModel):
    path: str = Field(min_length=1, max_length=4096, strict=True)
    recursive: StrictBool = True
    max_depth: int = Field(default=1, ge=1, le=8, strict=True)


class BatchEntry(InputModel):
    path: str = Field(min_length=1, max_length=4096, strict=True)
    name: str = Field(min_length=1, max_length=200, strict=True)
    version: str = Field(default="v1", min_length=1, max_length=200, strict=True)
    profile: dict | None = None
    norm_asset_id: str | None = Field(default=None, min_length=1, max_length=36, strict=True)
    norm_path: str | None = Field(default=None, min_length=1, max_length=4096, strict=True)

    @field_validator("name", "version")
    @classmethod
    def nonempty(cls, value):
        return _text(value, 200, "名称与版本")

    @model_validator(mode="after")
    def declarations(self):
        if "norm_path" in self.model_fields_set:
            if self.norm_path is None or "norm_asset_id" in self.model_fields_set:
                raise ValueError("norm_path 与 norm_asset_id 只能选择一个，路径不能为空")
        if self.profile is not None:
            if "processing" in self.profile:
                raise ValueError("批量登记只使用源字段特征，不保存训练组合")
            self.profile = normalize_dataset_profile(self.profile)
        return self


class BatchInput(InputModel):
    idempotency_key: str = Field(min_length=1, max_length=200, strict=True)
    entries: list[BatchEntry] = Field(min_length=1, max_length=MAX_ENTRIES)

    @field_validator("idempotency_key")
    @classmethod
    def key(cls, value):
        return _text(value, 200, "幂等标识")


def _identity(connection):
    return {key: getattr(connection, key) for key in ("id", "host", "port", "username", "fingerprint")} | {
        "credential_sha256": hashlib.sha256(connection.encrypted_password.encode()).hexdigest()}


def _binding(session, node_id, path):
    node = session.get(Node, node_id)
    if node is None:
        raise APIError(404, "NOT_FOUND", "机器不存在")
    if node.status in {"DISCONNECTED", "REMOVED"}:
        raise APIError(409, "NODE_CONNECTION_REMOVED", "机器已断开，请重新接入后扫描")
    if session.scalar(select(RemoteDeployment.id).where(RemoteDeployment.node_id == node_id,
            RemoteDeployment.state.in_({"QUEUED", "RUNNING", "WAITING_AGENT"})).limit(1)):
        raise APIError(409, "NODE_MAINTENANCE", "机器正在执行接入管理操作，请稍后扫描")
    _path(path, node.roots)
    try:
        connection = _connection(session, node_id)
    except APIError:
        raise APIError(409, "DATASET_SCAN_SSH_UNAVAILABLE", "机器尚无已验证的 SSH 接入记录，请先完成机器接入") from None
    if not connection.encrypted_password:
        raise APIError(409, "DATASET_SCAN_SSH_UNAVAILABLE", "机器 SSH 凭据不可用，请重新测试连接")
    return node, connection, {"roots": deepcopy(node.roots), "connection": _identity(connection)}


def _owner_exists(session, user_id, *, lock=False):
    query = select(User).where(User.id == user_id)
    if lock:
        query = query.with_for_update()
    user = session.scalar(query.execution_options(populate_existing=True))
    if user is None or not user.enabled:
        raise APIError(401, "UNAUTHORIZED", "登录账户已不可用")
    if user.must_change_password or "assets.write" not in ROLE_PERMISSIONS.get(user.role, frozenset()):
        raise APIError(403, "FORBIDDEN", "当前账户没有资产登记权限")


def _owned(session, scan_id, node_id, user_id, *, lock=False):
    query = select(DatasetScan).where(DatasetScan.id == scan_id, DatasetScan.node_id == node_id,
                                     DatasetScan.user_id == user_id)
    if lock:
        query = query.with_for_update()
    row = session.scalar(query.execution_options(populate_existing=True))
    if row is None:
        raise APIError(404, "NOT_FOUND", "数据集扫描不存在")
    return row


def _expire_result_bodies(session):
    # Keep small ownership/receipt references, not expired megabyte previews.
    for row in session.scalars(select(DatasetScan).where(DatasetScan.expires_at <= now(),
            DatasetScan.result.is_not(None), DatasetScan.state.not_in(WORKING)).limit(1000)):
        row.result = None


def _existing(session, node_id):
    found = {}
    for asset in session.scalars(select(Asset).where(Asset.node_id == node_id, Asset.kind == "DATASET")
                                .order_by(Asset.created_at, Asset.id)):
        try:
            found.setdefault(_path(asset.path), asset)
        except APIError:
            continue
    return found


def scan_dict(row, session):
    result = {key: deepcopy(getattr(row, key)) for key in ("id", "node_id", "path", "recursive", "max_depth",
              "state", "created_at", "updated_at", "expires_at", "error")}
    result.update(entries=[], warnings=[], truncated=False, scanned_directories=0, skipped_directories=0)
    if row.expires_at <= now():
        result.update(state="FAILED", error={"code": "DATASET_SCAN_EXPIRED", "message": "扫描已过期，请重新扫描目录"})
    elif row.result:
        result.update(deepcopy(row.result))
        existing = _existing(session, row.node_id)
        for entry in result["entries"]:
            asset = existing.get(entry["path"])
            if asset is not None:
                entry.update(existing_asset_id=asset.id, existing_state=asset.state)
    return result


def _scan_result(value, path, roots):
    if not isinstance(value, dict) or value.get("path") != path or not isinstance(value.get("entries"), list):
        raise APIError(502, "DATASET_SCAN_INVALID_RESULT", "扫描返回结果无效，请重新扫描")
    if len(value["entries"]) > MAX_ENTRIES:
        raise APIError(502, "DATASET_SCAN_INVALID_RESULT", "扫描候选数量超过限制")
    entries, seen = [], set()
    for item in value["entries"]:
        if not isinstance(item, dict):
            raise APIError(502, "DATASET_SCAN_INVALID_RESULT", "扫描候选结果无效")
        target = _path(item.get("path"), roots)
        parent = PurePosixPath(path)
        if target in seen or (PurePosixPath(target) != parent and parent not in PurePosixPath(target).parents):
            raise APIError(502, "DATASET_SCAN_INVALID_RESULT", "扫描返回重复或目录外的候选路径")
        seen.add(target)
        name = _text(item.get("name"), 200, "候选名称")
        summary = item.get("summary", {})
        if not isinstance(summary, dict):
            raise APIError(502, "DATASET_SCAN_INVALID_RESULT", "扫描元信息摘要无效")
        clean = {}
        for key in ("fps", "total_episodes", "total_frames"):
            raw = summary.get(key)
            if type(raw) in (int, float) and raw >= 0 and raw < 10**16:
                clean[key] = raw
        for key, limit in (("robot_type", 100), ("task_description", 2000)):
            raw = summary.get(key)
            if isinstance(raw, str) and len(raw) <= limit and not any(ord(c) < 32 and c not in "\n\t" for c in raw):
                clean[key] = raw
        error = item.get("error")
        if error is not None:
            if not isinstance(error, dict) or not isinstance(error.get("code"), str):
                raise APIError(502, "DATASET_SCAN_INVALID_RESULT", "扫描错误摘要无效")
            error = {"code": error["code"][:128], "message": str(error.get("message", "元信息无法解析"))[:500]}
        hashes = {}
        for key in ("info_sha256", "feature_signature"):
            raw = item.get(key)
            if raw is not None and (not isinstance(raw, str) or len(raw) != 64 or any(c not in "0123456789abcdef" for c in raw)):
                raise APIError(502, "DATASET_SCAN_INVALID_RESULT", "扫描元信息摘要指纹无效")
            hashes[key] = raw
        if error is None and hashes["info_sha256"] is None:
            raise APIError(502, "DATASET_SCAN_INVALID_RESULT", "候选缺少元信息指纹")
        entries.append({"path": target, "relative_path": str(PurePosixPath(target).relative_to(parent)),
                        "name": name, "summary": clean, "error": error, **hashes})
    warnings = value.get("warnings", [])
    if not isinstance(warnings, list) or len(warnings) > 100 or any(not isinstance(v, str) or len(v) > 1000 for v in warnings):
        raise APIError(502, "DATASET_SCAN_INVALID_RESULT", "扫描提示无效")
    counts = {key: value.get(key, 0) for key in ("scanned_directories", "skipped_directories")}
    if (type(counts["scanned_directories"]) is not int or not 0 <= counts["scanned_directories"] <= 5000
            or type(counts["skipped_directories"]) is not int or not 0 <= counts["skipped_directories"] <= 100000
            or type(value.get("truncated")) is not bool):
        raise APIError(502, "DATASET_SCAN_INVALID_RESULT", "扫描统计无效")
    result = {"entries": entries, "warnings": warnings, "truncated": value["truncated"], **counts}
    if len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode()) > MAX_BODY:
        raise APIError(502, "DATASET_SCAN_INVALID_RESULT", "扫描摘要超过大小限制")
    return result


def _node_registration_lock(session, node_id):
    if session.get_bind().dialect.name == "postgresql":
        key = int.from_bytes(hashlib.sha256(("dataset-batch\0" + node_id).encode()).digest()[:8], "big", signed=True)
        session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": key})


def _admission_lock(session):
    if session.get_bind().dialect.name == "postgresql":
        key = int.from_bytes(hashlib.sha256(b"dataset-scan-admission").digest()[:8], "big", signed=True)
        session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": key})


class DatasetBatchManager:
    def __init__(self, db, settings, remote_manager, auth, scanner=None):
        self.db, self.settings, self.remote_manager, self.auth = db, settings, remote_manager, auth
        self.scanner = scanner
        self._lock = threading.RLock()
        self._semaphore = threading.BoundedSemaphore(2)
        self._closing = False
        self._started = False
        self._threads = set()
        self._remotes = {}
        self._cancel = threading.Event()

    def start(self):
        with self._lock:
            if self._started:
                return
            self._started, self._closing = True, False
            self._cancel.clear()
        with self.db.write() as session:
            _expire_result_bodies(session)
            for row in session.scalars(select(DatasetScan).where(DatasetScan.state.in_(WORKING))):
                row.state, row.updated_at = "FAILED", now()
                row.error = {"code": "DATASET_SCAN_INTERRUPTED", "message": "控制台已重启，请重新扫描目录"}

    def close(self):
        with self._lock:
            self._closing = True
            self._cancel.set()
            remotes, threads = list(self._remotes.values()), list(self._threads)
        for remote in remotes:
            try:
                remote.close()
            except Exception:
                pass
        for thread in threads:
            thread.join(timeout=.5)
        with self.db.write() as session:
            for row in session.scalars(select(DatasetScan).where(DatasetScan.state.in_(WORKING))):
                row.state, row.updated_at = "FAILED", now()
                row.error = {"code": "DATASET_SCAN_INTERRUPTED", "message": "扫描已中断，请重新扫描目录"}

    def create(self, user_id, node_id, body, request=None):
        path = _path(body.path)
        with self.db.write() as session:
            if request is not None:
                require_active_actor(session, request, "assets.write")
            else:
                lock_execution_guard(session)
            _owner_exists(session, user_id, lock=True)
            _admission_lock(session)
            _expire_result_bodies(session)
            _, _, snapshot = _binding(session, node_id, path)
            count = session.scalar(select(func.count()).select_from(DatasetScan).where(DatasetScan.state.in_(WORKING)))
            own = session.scalar(select(func.count()).select_from(DatasetScan).where(
                DatasetScan.user_id == user_id, DatasetScan.expires_at > now()))
            if count >= 20 or own >= 100:
                raise APIError(429, "DATASET_SCAN_LIMIT", "扫描任务过多，请等待已有扫描完成或过期后重试")
            with self._lock:
                if self._closing:
                    raise APIError(503, "DATASET_SCAN_SHUTTING_DOWN", "控制台正在重启，请稍后扫描")
            stamp = now()
            row = DatasetScan(id=uid(), user_id=user_id, node_id=node_id, path=path,
                recursive=body.recursive, max_depth=body.max_depth, state="QUEUED", binding_snapshot=snapshot,
                result=None, error=None, created_at=stamp, updated_at=stamp, expires_at=offset_time(SCAN_TTL))
            session.add(row)
            session.flush()
            result = scan_dict(row, session)
        self._launch(row.id)
        return result

    def _launch(self, scan_id):
        def execute():
            try:
                self._run(scan_id)
            finally:
                with self._lock:
                    self._threads.discard(threading.current_thread())
        thread = threading.Thread(target=execute, name="vla-dataset-scan", daemon=True)
        with self._lock:
            if self._closing:
                return
            self._threads.add(thread)
        thread.start()

    def _run(self, scan_id):
        acquired, remote = False, None
        try:
            while not self._cancel.is_set():
                if self._semaphore.acquire(timeout=.1):
                    acquired = True
                    break
            if not acquired:
                return
            with self.db.write() as session:
                lock_execution_guard(session)
                row = session.get(DatasetScan, scan_id)
                if row is None or row.state != "QUEUED":
                    return
                _owner_exists(session, row.user_id)
                _, connection, snapshot = _binding(session, row.node_id, row.path)
                if row.expires_at <= now() or snapshot != row.binding_snapshot:
                    raise APIError(409, "DATASET_SCAN_BINDING_CHANGED", "扫描的目录或 SSH 绑定已改变，请重新扫描")
                row.state, row.updated_at = "SCANNING", now()
                spec = {"path": row.path, "roots": deepcopy(snapshot["roots"]), "recursive": row.recursive, "max_depth": row.max_depth}
            remote = self.remote_manager._remote(connection)
            with self._lock:
                if self._closing:
                    return
                self._remotes[scan_id] = remote
            remote.connect()
            if self._cancel.is_set():
                return
            scanner = self.scanner
            if scanner is None:
                from .dataset_scan_transport import scan_dataset_roots
                scanner = scan_dataset_roots
            value = scanner(remote, spec["path"], spec["roots"], recursive=spec["recursive"],
                            max_depth=spec["max_depth"], max_datasets=MAX_ENTRIES, cancelled=self._cancel.is_set)
            result = _scan_result(value, spec["path"], spec["roots"])
            with self.db.write() as session:
                row = session.get(DatasetScan, scan_id)
                if self._cancel.is_set() or row is None or row.state != "SCANNING":
                    return
                _owner_exists(session, row.user_id)
                _, _, current = _binding(session, row.node_id, row.path)
                if current != row.binding_snapshot:
                    raise APIError(409, "DATASET_SCAN_BINDING_CHANGED", "机器 SSH 绑定或允许目录已改变，请重新扫描")
                row.result, row.error, row.state, row.updated_at = result, None, "READY", now()
        except Exception as error:
            code = getattr(error, "code", "DATASET_SCAN_FAILED")
            message = getattr(error, "message", None) or "无法完成元信息扫描，请检查机器 SSH 连接、访问权限或稍后重试"
            with self.db.write() as session:
                row = session.get(DatasetScan, scan_id)
                if row is not None and row.state in WORKING:
                    row.state, row.updated_at = "FAILED", now()
                    row.error = {"code": str(code)[:128], "message": str(message)[:1000]}
        finally:
            if remote is not None:
                try:
                    remote.close()
                except Exception:
                    pass
            with self._lock:
                self._remotes.pop(scan_id, None)
            if acquired:
                self._semaphore.release()

    def get(self, user_id, node_id, scan_id):
        with self.db.read() as session:
            _owner_exists(session, user_id)
            row = _owned(session, scan_id, node_id, user_id)
            return scan_dict(row, session)

    def register(self, user_id, node_id, scan_id, body, request=None):
        data = body.model_dump(exclude_unset=True)
        canonical = []
        for entry in body.entries:
            canonical.append(_path(entry.path))
        if len(set(canonical)) != len(canonical):
            raise APIError(422, "DATASET_BATCH_DUPLICATE_PATH", "同一批次不能重复登记相同目录")
        request_digest = digest({"node_id": node_id, "scan_id": scan_id, "request": data})
        with self.db.write() as session:
            if request is not None:
                require_active_actor(session, request, "assets.write")
            else:
                lock_execution_guard(session)
            _owner_exists(session, user_id, lock=True)
            previous = session.scalar(select(DatasetBatchRegistration).where(
                DatasetBatchRegistration.user_id == user_id,
                DatasetBatchRegistration.request_key == body.idempotency_key))
            if previous is not None:
                if previous.request_digest != request_digest:
                    raise APIError(409, "DATASET_BATCH_IDEMPOTENCY_CONFLICT", "此提交标识已用于其他配置，请刷新后重试")
                return deepcopy(previous.result)
            scan = _owned(session, scan_id, node_id, user_id, lock=True)
            if scan.expires_at <= now():
                raise APIError(410, "DATASET_SCAN_EXPIRED", "扫描已过期，请重新扫描后登记")
            if scan.state != "READY" or not scan.result:
                raise APIError(409, "DATASET_SCAN_NOT_READY", "请先完成数据集目录扫描")
            node, _, snapshot = _binding(session, node_id, scan.path)
            if snapshot != scan.binding_snapshot:
                raise APIError(409, "DATASET_SCAN_BINDING_CHANGED", "机器 SSH 绑定或允许目录已改变，请重新扫描")
            admitted = {entry["path"]: entry for entry in scan.result["entries"] if entry.get("error") is None}
            for path in canonical:
                _path(path, node.roots)
                if path not in admitted:
                    raise APIError(422, "DATASET_BATCH_PATH_NOT_SCANNED", "只可登记本次扫描发现且元信息有效的数据集目录")
            _node_registration_lock(session, node_id)
            existing = _existing(session, node_id)
            pending = [(entry, path) for entry, path in zip(body.entries, canonical) if path not in existing]
            # Claim every automatic NORM path before taking NORM row locks, and
            # take known NORM references before INSERT's implicit Node FK lock.
            norm_paths = sorted({canonical_norm_path(entry.norm_path) for entry, _ in pending if entry.norm_path is not None})
            for path in norm_paths:
                require_path(path, node.roots)
                _lock_norm_path(session, node_id, path)
            norm_ids = {entry.norm_asset_id for entry, _ in pending if entry.norm_asset_id is not None}
            for norm in session.scalars(select(Asset).where(Asset.node_id == node_id, Asset.kind == "NORM")):
                try:
                    if canonical_norm_path(norm.path) in norm_paths:
                        norm_ids.add(norm.id)
                except APIError:
                    continue
            for norm_id in sorted(norm_ids):
                session.scalar(select(Asset).where(Asset.id == norm_id).with_for_update()
                               .execution_options(populate_existing=True))
            for entry, _ in pending:
                if entry.norm_path is None:
                    validate_norm_binding(session, kind="DATASET", node_id=node_id, norm_asset_id=entry.norm_asset_id)
            names = {canonical_norm_path(entry.norm_path): entry.name for entry, _ in pending if entry.norm_path is not None}
            norms = {path: resolve_norm_path(session, node_id=node_id, path=path, dataset_name=names[path]).id
                     for path in norm_paths}
            created, skipped = [], []
            for entry, path in zip(body.entries, canonical):
                if old := existing.get(path):
                    skipped.append({"path": path, "asset_id": old.id, "state": old.state,
                                    "reason": "已登记相同目录，保留原有资产"})
                    continue
                source = admitted[path]
                metadata = {"batch_registration": {"scan_id": scan.id, "info_sha256": source["info_sha256"],
                            "feature_signature": source.get("feature_signature"), "metadata_only": True}}
                if entry.profile is not None:
                    metadata["dataset_profile"] = deepcopy(entry.profile)
                norm_id = norms[canonical_norm_path(entry.norm_path)] if entry.norm_path is not None else entry.norm_asset_id
                asset = Asset(id=uid(), node_id=node_id, name=entry.name, kind="DATASET", path=path,
                              version=entry.version, state="REGISTERED", metadata_json=metadata,
                              norm_asset_id=norm_id, created_at=now())
                session.add(asset)
                session.flush()
                queue_validation(session, asset, heartbeat_timeout_seconds=self.settings.heartbeat_timeout_seconds,
                                 actor_request=request)
                created.append(asset_dict(asset))
            result = {"scan_id": scan.id, "idempotency_key": body.idempotency_key, "created": created,
                      "skipped": skipped, "created_count": len(created), "skipped_count": len(skipped)}
            session.add(DatasetBatchRegistration(id=uid(), user_id=user_id, scan_id=scan.id,
                        request_key=body.idempotency_key, request_digest=request_digest, result=result, created_at=now()))
            session.flush()
            return result


async def _body(request, model):
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY:
            raise APIError(413, "DATASET_BATCH_TOO_LARGE", "批量登记请求不能超过 1 MiB")
        chunks.append(chunk)
    try:
        def reject(_):
            raise ValueError("Non-JSON constant")
        def unique(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("Duplicate property")
                result[key] = value
            return result
        value = json.loads(b"".join(chunks), parse_constant=reject, object_pairs_hook=unique)
        return model.model_validate(value)
    except (ValueError, TypeError, UnicodeError, RecursionError, ValidationError):
        raise APIError(422, "DATASET_BATCH_INVALID", "批量登记请求无效，请检查字段、模板和归一化配置") from None


def register_routes(app, dependencies, manager):
    def owner(request):
        principal = manager.auth.require(request, "assets.write")
        if principal.legacy or not principal.session_id:
            raise APIError(403, "DATASET_BATCH_ACCOUNT_REQUIRED", "请使用个人账户登录后批量登记数据集")
        for name in ("x-dataset-scan-user", "x-dataset-batch-user"):
            expected = request.headers.get(name)
            if expected is not None and expected != principal.user_id:
                raise APIError(403, "DATASET_BATCH_USER_CHANGED", "登录账户已改变，请重新打开批量登记")
        return principal.user_id

    @app.post("/api/v1/nodes/{node_id}/dataset-scans", dependencies=dependencies, status_code=202)
    async def create(node_id: str, request: Request):
        user_id = owner(request)
        body = await _body(request, ScanInput)
        owner(request)
        result = await run_in_threadpool(manager.create, user_id, node_id, body, request)
        return JSONResponse(result, status_code=202, headers={"Cache-Control": "no-store"})

    @app.get("/api/v1/nodes/{node_id}/dataset-scans/{scan_id}", dependencies=dependencies)
    def get(node_id: str, scan_id: str, request: Request):
        return JSONResponse(manager.get(owner(request), node_id, scan_id), headers={"Cache-Control": "no-store"})

    @app.post("/api/v1/nodes/{node_id}/dataset-scans/{scan_id}/register", dependencies=dependencies)
    async def register(node_id: str, scan_id: str, request: Request):
        user_id = owner(request)
        body = await _body(request, BatchInput)
        owner(request)
        return JSONResponse(await run_in_threadpool(manager.register, user_id, node_id, scan_id, body, request),
                            headers={"Cache-Control": "no-store"})
