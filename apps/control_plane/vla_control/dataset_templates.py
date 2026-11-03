"""Account-owned, declarative dataset formats; never asset or training writes."""

import json

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import defer
from starlette.concurrency import run_in_threadpool

from .auth import ROLE_PERMISSIONS
from .auth_models import User
from .common import APIError, now, uid
from .dataset_profiles import DatasetProfile
from .dataset_template_models import DatasetFormatTemplate
from .account_lifecycle import lock_execution_guard
from .resource_authorization import require_active_actor


MAX_REQUEST_BYTES = 64 * 1024
MAX_TEMPLATES_PER_USER = 100
MAX_EXPECTED_REVISION = 2**63 - 2


def _invalid(message="数据集格式模板无效，请检查数据集特征"):
    return APIError(422, "DATASET_TEMPLATE_INVALID", message)


def _conflict():
    return APIError(409, "DATASET_TEMPLATE_CONFLICT", "模板已在其他窗口更新，请重新读取后再保存或删除")


def _not_found():
    # The same response covers missing and another account's identity.
    return APIError(404, "NOT_FOUND", "数据集格式模板不存在")


def _name(value):
    if type(value) is not str:
        raise _invalid("模板名称必须为文本")
    value = value.strip()
    if not value or len(value) > 100 or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in value):
        raise _invalid("模板名称须为 1 至 100 个字符，且不能包含控制字符")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise _invalid("模板名称包含无效文本") from error
    return value


def normalize_profile(value):
    # Templates describe source fields only. Legacy assets may retain a
    # historical processing composition, but it must be excluded deliberately.
    if type(value) is not dict or "processing" in value:
        raise _invalid("格式模板仅保存数据集源字段特征，不包含训练组合配置")
    try:
        profile = DatasetProfile.model_validate(value).model_dump(mode="json", exclude_none=True)
        profile.pop("processing", None)
        encoded = json.dumps(profile, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValidationError, ValueError, TypeError, UnicodeError, RecursionError) as error:
        # Never echo arbitrary profile values into an error or audit record.
        raise _invalid() from error
    if len(encoded) > MAX_REQUEST_BYTES:
        raise APIError(413, "DATASET_TEMPLATE_TOO_LARGE", "格式模板不能超过 64 KiB")
    return profile


async def _body(request, operation):
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_REQUEST_BYTES:
            raise APIError(413, "DATASET_TEMPLATE_TOO_LARGE", "格式模板请求不能超过 64 KiB")
        chunks.append(chunk)
    try:
        def reject_constant(_):
            raise ValueError("Non-JSON constant")

        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate JSON key")
                result[key] = value
            return result

        value = json.loads(b"".join(chunks), parse_constant=reject_constant, object_pairs_hook=unique_keys)
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise _invalid("模板请求必须为有效 JSON") from error
    fields = {"name", "profile"} if operation == "create" else (
        {"expected_revision", "name", "profile"} if operation == "update" else {"expected_revision"})
    if type(value) is not dict or set(value) != fields:
        raise _invalid("模板请求字段无效，用户归属由当前登录账户确定")
    if operation != "create":
        revision = value["expected_revision"]
        if type(revision) is not int or not 1 <= revision <= MAX_EXPECTED_REVISION:
            raise _invalid("模板版本必须为有效的正整数")
    if operation != "delete":
        value["name"] = _name(value["name"])
        value["profile"] = normalize_profile(value["profile"])
    return value


def _representation(row, include_profile=True):
    result = {key: getattr(row, key) for key in ("id", "name", "revision", "created_at", "updated_at")}
    if include_profile:
        result["profile"] = row.profile
    return result


def _owned(session, user_id, template_id):
    row = session.scalar(select(DatasetFormatTemplate).where(
        DatasetFormatTemplate.id == template_id, DatasetFormatTemplate.user_id == user_id))
    if row is None:
        raise _not_found()
    return row


def _lock_owner(session, user_id, request=None):
    # Serialize this owner's upserts/capacity checks across PostgreSQL workers;
    # SQLite writes already hold BEGIN IMMEDIATE. A unique index is the final guard.
    if request is not None:
        require_active_actor(session, request, "dataset-templates.write")
    else:
        lock_execution_guard(session)
    user = session.scalar(select(User).where(User.id == user_id).with_for_update())
    if user is None or not user.enabled:
        raise APIError(401, "UNAUTHORIZED", "登录账户已不可用")
    if user.must_change_password:
        raise APIError(403, "PASSWORD_CHANGE_REQUIRED", "请先修改初始密码")
    if "dataset-templates.write" not in ROLE_PERMISSIONS.get(user.role, frozenset()):
        raise APIError(403, "FORBIDDEN", "当前账户没有资产编辑权限")


def create_template(db, user_id, value, request=None):
    with db.write() as session:
        _lock_owner(session, user_id, request)
        row = session.scalar(select(DatasetFormatTemplate).where(
            DatasetFormatTemplate.user_id == user_id, DatasetFormatTemplate.name == value["name"]))
        created = row is None
        stamp = now()
        if created:
            count = session.scalar(select(func.count()).select_from(DatasetFormatTemplate).where(
                DatasetFormatTemplate.user_id == user_id))
            if count >= MAX_TEMPLATES_PER_USER:
                raise APIError(409, "DATASET_TEMPLATE_LIMIT", "每个账户最多保存 100 个数据集格式模板，可更新同名模板或删除旧模板")
            row = DatasetFormatTemplate(id=uid(), user_id=user_id, name=value["name"], profile=value["profile"],
                                        revision=1, created_at=stamp, updated_at=stamp)
            session.add(row)
        else:
            if row.revision >= MAX_EXPECTED_REVISION:
                raise _conflict()
            row.profile = value["profile"]
            row.revision += 1
            row.updated_at = stamp
        session.flush()
        result = _representation(row)
    return result, 201 if created else 200


def update_template(db, user_id, template_id, value, request=None):
    with db.write() as session:
        _lock_owner(session, user_id, request)
        row = _owned(session, user_id, template_id)
        revision = value["expected_revision"]
        if row.revision != revision or row.revision >= MAX_EXPECTED_REVISION:
            raise _conflict()
        duplicate = session.scalar(select(DatasetFormatTemplate.id).where(
            DatasetFormatTemplate.user_id == user_id, DatasetFormatTemplate.name == value["name"],
            DatasetFormatTemplate.id != template_id).limit(1))
        if duplicate is not None:
            raise APIError(409, "DATASET_TEMPLATE_NAME_CONFLICT", "当前账户已有同名格式模板，可按名称保存以覆盖该模板")
        stamp = now()
        changed = session.execute(update(DatasetFormatTemplate).where(
            DatasetFormatTemplate.id == template_id, DatasetFormatTemplate.user_id == user_id,
            DatasetFormatTemplate.revision == revision,
        ).values(name=value["name"], profile=value["profile"], revision=revision + 1, updated_at=stamp),
            execution_options={"synchronize_session": False})
        if changed.rowcount != 1:
            raise _conflict()
        return {"id": template_id, "name": value["name"], "profile": value["profile"],
                "revision": revision + 1, "created_at": row.created_at, "updated_at": stamp}


def delete_template(db, user_id, template_id, revision, request=None):
    with db.write() as session:
        _lock_owner(session, user_id, request)
        _owned(session, user_id, template_id)
        changed = session.execute(delete(DatasetFormatTemplate).where(
            DatasetFormatTemplate.id == template_id, DatasetFormatTemplate.user_id == user_id,
            DatasetFormatTemplate.revision == revision))
        if changed.rowcount != 1:
            raise _conflict()
    return {"id": template_id, "deleted": True}


def register_routes(app, dependencies, db, auth):
    def owner(request):
        principal = auth.require(request, "dataset-templates.write")
        if principal.legacy or not principal.session_id:
            raise APIError(403, "DATASET_TEMPLATE_ACCOUNT_REQUIRED", "请使用个人账户登录后使用数据集格式模板")
        expected = request.headers.get("x-dataset-template-user")
        if expected is not None and expected != principal.user_id:
            raise APIError(403, "DATASET_TEMPLATE_USER_CHANGED", "登录账户已改变，未操作其他账户的格式模板")
        return principal.user_id

    def response(value, status=200):
        return JSONResponse(value, status_code=status, headers={"Cache-Control": "no-store"})

    @app.get("/api/v1/dataset-format-templates", dependencies=dependencies)
    def list_templates(request: Request):
        user_id = owner(request)
        with db.read() as session:
            rows = session.scalars(select(DatasetFormatTemplate).where(DatasetFormatTemplate.user_id == user_id)
                .options(defer(DatasetFormatTemplate.profile))
                .order_by(DatasetFormatTemplate.updated_at.desc(), DatasetFormatTemplate.id))
            return response([_representation(row, include_profile=False) for row in rows])

    @app.get("/api/v1/dataset-format-templates/{template_id}", dependencies=dependencies)
    def read(template_id: str, request: Request):
        user_id = owner(request)
        with db.read() as session:
            return response(_representation(_owned(session, user_id, template_id)))

    @app.post("/api/v1/dataset-format-templates", dependencies=dependencies, status_code=201)
    async def create(request: Request):
        user_id = owner(request)
        value = await _body(request, "create")
        owner(request)
        result, status = await run_in_threadpool(create_template, db, user_id, value, request)
        return response(result, status)

    @app.put("/api/v1/dataset-format-templates/{template_id}", dependencies=dependencies)
    async def save(template_id: str, request: Request):
        user_id = owner(request)
        value = await _body(request, "update")
        owner(request)
        return response(await run_in_threadpool(update_template, db, user_id, template_id, value, request))

    @app.delete("/api/v1/dataset-format-templates/{template_id}", dependencies=dependencies)
    async def remove(template_id: str, request: Request):
        user_id = owner(request)
        value = await _body(request, "delete")
        owner(request)
        return response(await run_in_threadpool(delete_template, db, user_id, template_id, value["expected_revision"], request))
