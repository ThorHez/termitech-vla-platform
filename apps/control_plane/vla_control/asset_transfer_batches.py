"""Atomically admit small groups to the existing registered-asset copier."""

from copy import deepcopy
import json
from pathlib import PurePosixPath

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import Field, ValidationError, field_validator
from sqlalchemy import select
from sqlalchemy.exc import DBAPIError
from starlette.concurrency import run_in_threadpool
from vla_protocol import canonical_digest

from .asset_transfer_batch_models import AssetTransferBatch
from .asset_transfer_models import AssetTransfer
from .asset_transfers import (AssetTransferInput, _admission_lock, _bindings, _check_companion_paths,
    _norm_sync, _target_available, active_asset_transfers, asset_transfer_dict, component)
from .auth import ROLE_PERMISSIONS
from .auth_models import User
from .checkpoint_transfers import _online, _verify_saved_ssh_credentials
from .common import APIError, now, require_path, uid
from .models import Asset
from .schemas import InputModel
from .account_lifecycle import lock_execution_guard
from .resource_authorization import require_active_actor, require_expected_user


MAX_ENTRIES = 20
MAX_BODY = 128 * 1024


def _text(value):
    if (not value.strip() or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in value)):
        raise ValueError("同步请求不能包含空白或控制字符")
    value = value.strip()
    value.encode("utf-8")
    return value


class AssetTransferBatchEntry(InputModel):
    source_asset_id: str = Field(min_length=1, max_length=36, strict=True)
    target_name: str = Field(min_length=1, max_length=160, strict=True)
    include_norm_stats: bool = Field(default=False, strict=True)
    exclude_train_state: bool = Field(default=False, strict=True)
    expected_norm_asset_id: str | None = Field(default=None, min_length=1, max_length=36, strict=True)

    @field_validator("source_asset_id", "target_name", "expected_norm_asset_id")
    @classmethod
    def valid_text(cls, value, info):
        if value is None:
            return None
        value = _text(value)
        return component(value) if info.field_name == "target_name" else value


class AssetTransferBatchInput(InputModel):
    idempotency_key: str = Field(min_length=1, max_length=200, strict=True)
    target_node_id: str = Field(min_length=1, max_length=36, strict=True)
    target_directory: str = Field(min_length=1, max_length=4096, strict=True)
    entries: list[AssetTransferBatchEntry] = Field(min_length=1, max_length=MAX_ENTRIES)

    @field_validator("idempotency_key", "target_node_id")
    @classmethod
    def valid_text(cls, value):
        return _text(value)

    @field_validator("target_directory")
    @classmethod
    def valid_directory(cls, value):
        value = _text(value)
        if "\\" in value:
            raise ValueError("目标目录需要有效的绝对路径")
        require_path(value)
        return "/" + str(PurePosixPath(value)).lstrip("/")


def _owner(session, user_id):
    user = session.scalar(select(User).where(User.id == user_id).with_for_update()
                          .execution_options(populate_existing=True))
    if user is None or not user.enabled:
        raise APIError(401, "UNAUTHORIZED", "登录账户已不可用")
    if user.must_change_password or "assets.transfer" not in ROLE_PERMISSIONS.get(user.role, frozenset()):
        raise APIError(403, "FORBIDDEN", "当前账户没有资产同步权限")


def _lock_inputs(session, body):
    # Never wait on an Asset while holding Node locks. NOWAIT also avoids a
    # cycle with legacy dataset -> NORM binders when this batch contains both
    # a dataset and one of its companion NORMs as independently selected rows.
    sources = {}
    for asset_id in sorted(entry.source_asset_id for entry in body.entries):
        asset = session.scalar(select(Asset).where(Asset.id == asset_id).with_for_update(nowait=True)
                               .execution_options(populate_existing=True))
        if asset is None:
            raise APIError(404, "NOT_FOUND", "来源资产不存在")
        sources[asset_id] = asset
    companions = {sources[entry.source_asset_id].norm_asset_id for entry in body.entries if entry.include_norm_stats}
    for asset_id in sorted(companions - sources.keys() - {None}):
        session.scalar(select(Asset).where(Asset.id == asset_id).with_for_update(nowait=True)
                       .execution_options(populate_existing=True))
    # All row locks for known inputs are held now. Source binding changes are
    # blocked by their dataset row locks until all queued transfers commit.
    return sources


def _overlap(left, right):
    return left == right or left in right.parents or right in left.parents


def _validate_plans(session, plans):
    destinations = []
    for _, bindings, companion in plans:
        names = [bindings["target_name"]] + ([companion["target_name"]] if companion else [])
        for name in names:
            _target_available(session, bindings["target_node_id"], bindings["target_directory"], name)
            path = PurePosixPath(bindings["target_directory"]) / name
            if any(_overlap(path, old) for old in destinations):
                raise APIError(422, "ASSET_TRANSFER_BATCH_TARGET_CONFLICT", "本批次包含重复或重叠的目标目录，请修改目标目录名")
            destinations.append(path)
        if companion:
            _check_companion_paths(bindings, companion["target_name"])
    # Distinct registered node ids can point at the same physical SSH machine.
    # Protect every source from every destination in this group, not only the
    # source/target pair checked by the single-transfer contract.
    for _, bindings, _ in plans:
        source, target = bindings["source_connection"], bindings["target_connection"]
        if (source["fingerprint"] == target["fingerprint"] or
                (source["host"], source["port"]) == (target["host"], target["port"])):
            for key in ("source_asset", "norm_asset"):
                if key in bindings and any(_overlap(PurePosixPath(bindings[key]["path"]), path) for path in destinations):
                    raise APIError(422, "ASSET_TRANSFER_PATH_OVERLAP", "本批次目标目录不能与同一物理机器上的任一来源资产重叠")


def create_batch(manager, user_id, body, *, request=None):
    ids = [entry.source_asset_id for entry in body.entries]
    if len(set(ids)) != len(ids):
        raise APIError(422, "ASSET_TRANSFER_BATCH_DUPLICATE_SOURCE", "本批次不能重复选择同一来源资产")
    names = [entry.target_name for entry in body.entries]
    if len(set(names)) != len(names):
        raise APIError(422, "ASSET_TRANSFER_BATCH_TARGET_CONFLICT", "本批次目标目录名不能重复")
    digest_fields = body.model_dump()
    # A default false flag must preserve receipts created before filtering was
    # introduced, including explicit false browser submissions after upgrade.
    for entry in digest_fields['entries']:
        if not entry['exclude_train_state']:
            entry.pop('exclude_train_state')
    request_digest = canonical_digest(digest_fields)
    launches = []
    try:
        with manager.db.write() as session:
            if request is not None:
                require_active_actor(session, request, "assets.transfer")
            else:
                lock_execution_guard(session)
            _owner(session, user_id)
            # This is the same first lock used by single create and retry, so
            # global capacity and target overlap checks cannot race each other.
            _admission_lock(session)
            previous = session.scalar(select(AssetTransferBatch).where(
                AssetTransferBatch.user_id == user_id, AssetTransferBatch.request_key == body.idempotency_key))
            if previous is not None:
                if previous.request_digest != request_digest:
                    raise APIError(409, "ASSET_TRANSFER_BATCH_IDEMPOTENCY_CONFLICT", "此提交标识已用于其他同步配置，请重新提交")
                return deepcopy(previous.result)
            if len(active_asset_transfers(session)) + len(body.entries) > MAX_ENTRIES:
                raise APIError(429, "ASSET_TRANSFER_BUSY", "当前同步任务余量不足以排入整批，请减少选择或等待已有任务完成")
            sources = _lock_inputs(session, body)
            nodes = {asset.node_id for asset in sources.values()} | {body.target_node_id}
            for node_id in sorted(nodes):
                _online(session, node_id, manager.settings)
            plans, origins = [], set()
            for entry in body.entries:
                single = AssetTransferInput(**entry.model_dump(), target_node_id=body.target_node_id,
                                            target_directory=body.target_directory)
                bindings = _bindings(session, single, manager.settings)
                source = bindings["source_asset"]
                identity = (source["node_id"], "/" + str(PurePosixPath(source["path"])).lstrip("/"))
                if identity in origins:
                    raise APIError(422, "ASSET_TRANSFER_BATCH_DUPLICATE_SOURCE", "本批次包含指向同一来源路径的资产，请保留一个")
                origins.add(identity)
                transfer_id = uid()
                companion = _norm_sync(transfer_id, bindings["norm_asset"]) if entry.include_norm_stats else None
                plans.append((transfer_id, bindings, companion))
                _verify_saved_ssh_credentials(session, bindings, manager)
            _validate_plans(session, plans)
            batch_id, transfers = uid(), []
            for index, (transfer_id, bindings, companion) in enumerate(plans):
                source = bindings["source_asset"]
                row = AssetTransfer(id=transfer_id, owner_user_id=user_id, execution_user_id=user_id,
                    request_key=f"asset-batch:{batch_id}:{index}",
                    request_digest=canonical_digest({"batch": batch_id, "index": index, "bindings": bindings}),
                    name=(source["name"] + " · 资产同步")[:200],
                    source_asset_id=source["id"], source_node_id=source["node_id"], source_path=source["path"],
                    source_kind=source["kind"], target_node_id=body.target_node_id,
                    target_directory=bindings["target_directory"], target_name=bindings["target_name"],
                    include_norm_stats=companion is not None, norm_sync=companion,
                    exclude_train_state=bindings.get('exclude_train_state', False),
                    state="QUEUED", attempt=1, logs="", binding_snapshot=bindings, created_at=now(), updated_at=now())
                session.add(row)
                session.flush()
                transfers.append(asset_transfer_dict(row))
                launches.append(row.id)
            result = {"idempotency_key": body.idempotency_key, "transfers": transfers}
            session.add(AssetTransferBatch(id=batch_id, user_id=user_id, request_key=body.idempotency_key,
                        request_digest=request_digest, result=result, created_at=now()))
            session.flush()
    except DBAPIError as error:
        if getattr(error.orig, "sqlstate", None) == "55P03" or getattr(error.orig, "pgcode", None) == "55P03":
            raise APIError(409, "ASSET_TRANSFER_SOURCE_BUSY", "来源资产正在变更，请稍后重试整批同步") from None
        raise
    # Workers never observe only part of the batch. A frozen replay above does
    # not relaunch previously committed jobs, even if their state later changes.
    for transfer_id in launches:
        manager._launch(transfer_id)
    return result


async def _body(request):
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY:
            raise APIError(413, "ASSET_TRANSFER_BATCH_TOO_LARGE", "批量同步请求不能超过 128 KiB")
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
        return AssetTransferBatchInput.model_validate(value)
    except (ValueError, TypeError, UnicodeError, RecursionError, ValidationError):
        raise APIError(422, "ASSET_TRANSFER_BATCH_INVALID", "批量同步请求无效，请检查资产、目标目录和归一化参数选择") from None


def register_routes(app, admin, manager, auth):
    def owner(request):
        principal = auth.require(request, "assets.transfer")
        if principal.legacy or not principal.session_id:
            raise APIError(403, "ASSET_TRANSFER_BATCH_ACCOUNT_REQUIRED", "请使用个人账户登录后批量同步资产")
        expected = request.headers.get("x-asset-transfer-batch-user")
        if expected is not None and expected != principal.user_id:
            raise APIError(403, "ASSET_TRANSFER_BATCH_USER_CHANGED", "登录账户已改变，请重新打开批量同步")
        return principal.user_id

    @app.post("/api/v1/asset-transfer-batches", dependencies=admin, status_code=202)
    async def create(request: Request):
        require_expected_user(request)
        user_id = owner(request)
        body = await _body(request)
        owner(request)
        result = await run_in_threadpool(create_batch, manager, user_id, body, request=request)
        return JSONResponse(result, status_code=202, headers={"Cache-Control": "no-store"})
