"""Account-owned reusable training inputs, separate from preflight acceptance."""

import json

from fastapi import Request
from fastapi.responses import JSONResponse
from sqlalchemy import delete, select, update
from starlette.concurrency import run_in_threadpool

from .common import APIError, now, uid
from .auth_models import User
from .models import TrainingTemplate
from .training_drafts import MAX_CONTENT_BYTES, MAX_EXPECTED_REVISION, bounded_content


MAX_REQUEST_BYTES = MAX_CONTENT_BYTES + 32 * 1024


def _invalid(message="训练模板必须为受限 JSON 配置"):
    return APIError(422, "TRAINING_TEMPLATE_INVALID", message)


def _conflict():
    return APIError(409, "TRAINING_TEMPLATE_CONFLICT", "训练模板已在其他窗口更新，请重新读取后再保存或删除")


def _name_conflict():
    return APIError(409, "TRAINING_TEMPLATE_NAME_CONFLICT", "当前账户已有同名训练模板，请使用按名称保存或选择其他名称")


def _not_found():
    # Keep absent and foreign-account identities indistinguishable, even for admins.
    return APIError(404, "NOT_FOUND", "训练模板不存在")


def _text(value, name, maximum, *, optional=False, multiline=False):
    if type(value) is not str:
        raise _invalid(name + "必须为文本")
    value = value.strip()
    if (not optional and not value or len(value) > maximum
            or any((ord(char) < 32 and not (multiline and char in "\n\r\t"))
                   or 127 <= ord(char) <= 159 for char in value)):
        raise _invalid(f"{name}无效或超过 {maximum} 字符")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise _invalid(name + "包含无效文本") from error
    return value


async def _body(request, *, operation):
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_REQUEST_BYTES:
            raise APIError(413, "TRAINING_TEMPLATE_TOO_LARGE", "训练模板不能超过 512 KiB")
        chunks.append(chunk)
    try:
        def invalid_constant(_):
            raise ValueError("Non-JSON numeric constant")
        value = json.loads(b"".join(chunks), parse_constant=invalid_constant)
    except (ValueError, TypeError, RecursionError, UnicodeError) as error:
        raise _invalid() from error
    expected = ({"name", "content"}, {"description"}) if operation == "create" else (
        ({"expected_revision", "name", "description", "content"}, set()) if operation == "update"
        else ({"expected_revision"}, set()))
    required, optional = expected
    if type(value) is not dict or not required <= set(value) or set(value) - required - optional:
        raise _invalid("模板请求字段无效，用户归属由当前登录账户确定")
    if operation != "create":
        revision = value["expected_revision"]
        if type(revision) is not int or not 1 <= revision <= MAX_EXPECTED_REVISION:
            raise _invalid("模板版本必须为有效的正整数")
    if operation != "delete":
        value["name"] = _text(value["name"], "模板名称", 100)
        value["description"] = _text(value.get("description", ""), "模板说明", 2000, optional=True, multiline=True)
        try:
            value["content"] = bounded_content(value["content"])
        except APIError as error:
            raise APIError(error.status, error.code.replace("TRAINING_DRAFT_", "TRAINING_TEMPLATE_", 1),
                           error.message.replace("训练草稿", "训练模板"), error.details) from error
    return value


def _representation(row, *, include_content=True):
    result = {"id": row.id, "name": row.name, "description": row.description,
              "revision": row.revision, "created_at": row.created_at, "updated_at": row.updated_at}
    if include_content:
        result["content"] = row.content
    return result


def _owned(session, user_id, template_id):
    row = session.scalar(select(TrainingTemplate).where(
        TrainingTemplate.id == template_id, TrainingTemplate.user_id == user_id))
    if row is None:
        raise _not_found()
    return row


def _lock_owner(session, user_id):
    # Serialize mutations for one account on PostgreSQL. SQLite writers already
    # hold BEGIN IMMEDIATE; the unique owner/name index supplies a final guard.
    if session.scalar(select(User.id).where(User.id == user_id).with_for_update()) is None:
        raise APIError(401, "UNAUTHORIZED", "登录账户已不可用")


def create_template(db, user_id, value):
    with db.write() as session:
        _lock_owner(session, user_id)
        stamp = now()
        row = session.scalar(select(TrainingTemplate).where(
            TrainingTemplate.user_id == user_id, TrainingTemplate.name == value["name"]))
        created = row is None
        if created:
            row = TrainingTemplate(id=uid(), user_id=user_id, name=value["name"], description=value["description"],
                                   content=value["content"], revision=1, created_at=stamp, updated_at=stamp)
            session.add(row)
        else:
            row.description, row.content = value["description"], value["content"]
            row.revision += 1
            row.updated_at = stamp
        session.flush()
        result = _representation(row)
    return result, 201 if created else 200


def update_template(db, user_id, template_id, value):
    revision = value["expected_revision"]
    with db.write() as session:
        _lock_owner(session, user_id)
        row = _owned(session, user_id, template_id)
        if row.revision != revision:
            raise _conflict()
        if session.scalar(select(TrainingTemplate.id).where(
                TrainingTemplate.user_id == user_id, TrainingTemplate.name == value["name"],
                TrainingTemplate.id != template_id).limit(1)) is not None:
            raise _name_conflict()
        stamp = now()
        changed = session.execute(update(TrainingTemplate).where(
            TrainingTemplate.id == template_id, TrainingTemplate.user_id == user_id,
            TrainingTemplate.revision == revision,
        ).values(name=value["name"], description=value["description"], content=value["content"],
                 revision=revision + 1, updated_at=stamp), execution_options={"synchronize_session": False})
        if changed.rowcount != 1:
            raise _conflict()
        result = {"id": template_id, "name": value["name"], "description": value["description"],
                  "content": value["content"], "revision": revision + 1, "created_at": row.created_at,
                  "updated_at": stamp}
    return result


def delete_template(db, user_id, template_id, revision):
    with db.write() as session:
        _lock_owner(session, user_id)
        _owned(session, user_id, template_id)
        changed = session.execute(delete(TrainingTemplate).where(
            TrainingTemplate.id == template_id, TrainingTemplate.user_id == user_id,
            TrainingTemplate.revision == revision))
        if changed.rowcount != 1:
            raise _conflict()
    return {"id": template_id, "deleted": True}


def register_routes(app, dependencies, db, auth):
    def owner(request):
        principal = auth.require(request, "training.write")
        if principal.legacy:
            raise APIError(403, "TRAINING_TEMPLATE_ACCOUNT_REQUIRED", "请使用个人账户登录后保存训练模板")
        expected_user = request.headers.get("x-training-template-user")
        if expected_user is not None and expected_user != principal.user_id:
            raise APIError(403, "TRAINING_TEMPLATE_USER_CHANGED", "登录账户已改变，未操作其他账户的训练模板")
        return principal.user_id

    def response(value, status=200):
        return JSONResponse(value, status_code=status, headers={"Cache-Control": "no-store"})

    @app.get("/api/v1/training-templates", dependencies=dependencies)
    def list_templates(request: Request):
        user_id = owner(request)
        with db.read() as session:
            rows = session.scalars(select(TrainingTemplate).where(TrainingTemplate.user_id == user_id)
                                   .order_by(TrainingTemplate.updated_at.desc(), TrainingTemplate.id))
            return response([_representation(row, include_content=False) for row in rows])

    @app.post("/api/v1/training-templates", dependencies=dependencies, status_code=201)
    async def create(request: Request):
        user_id = owner(request)
        value = await _body(request, operation="create")
        result, status = await run_in_threadpool(create_template, db, user_id, value)
        return response(result, status)

    @app.get("/api/v1/training-templates/{template_id}", dependencies=dependencies)
    def read(template_id: str, request: Request):
        user_id = owner(request)
        with db.read() as session:
            return response(_representation(_owned(session, user_id, template_id)))

    @app.put("/api/v1/training-templates/{template_id}", dependencies=dependencies)
    async def save(template_id: str, request: Request):
        user_id = owner(request)
        value = await _body(request, operation="update")
        return response(await run_in_threadpool(update_template, db, user_id, template_id, value))

    @app.delete("/api/v1/training-templates/{template_id}", dependencies=dependencies)
    async def remove(template_id: str, request: Request):
        user_id = owner(request)
        value = await _body(request, operation="delete")
        return response(await run_in_threadpool(delete_template, db, user_id, template_id, value["expected_revision"]))
