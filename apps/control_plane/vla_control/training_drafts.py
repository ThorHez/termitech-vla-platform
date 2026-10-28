"""Bounded, account-owned wizard drafts, independent of training acceptance."""

import json
import math
import re

from fastapi import Request
from fastapi.responses import JSONResponse
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from starlette.concurrency import run_in_threadpool

from .common import APIError, now
from .models import TrainingDraft


MAX_CONTENT_BYTES = 512 * 1024
MAX_REQUEST_BYTES = MAX_CONTENT_BYTES + 4096
MAX_DEPTH = 48
MAX_NODES = 50000
MAX_EXPECTED_REVISION = 2**53 - 2
_FORBIDDEN = {
    "previewid", "confirmedconfigsha256", "codesnapshot", "idempotencykey",
    "executiontoken", "csrftoken", "password", "admintoken", "enrollmenttoken",
    "accesstoken", "refreshtoken", "sessiontoken", "authorization", "cookie", "passwordhash",
}


def _invalid(message="训练草稿必须为受限 JSON 配置"):
    return APIError(422, "TRAINING_DRAFT_INVALID", message)


def bounded_content(value):
    """Accept incomplete inert data, while excluding credentials and approvals."""
    if type(value) is not dict:
        raise _invalid()
    stack, nodes = [(value, 0)], 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if depth > MAX_DEPTH or nodes > MAX_NODES:
            raise _invalid("训练草稿嵌套过深或包含过多配置项")
        if type(item) is dict:
            nodes += len(item)
            for key, entry in item.items():
                if type(key) is not str:
                    raise _invalid()
                normalized = re.sub(r"[^a-z0-9]", "", key.casefold())
                if normalized in _FORBIDDEN:
                    raise _invalid("训练草稿不能保存凭据、预检结果或启动确认")
                stack.append((entry, depth + 1))
        elif type(item) is list:
            stack.extend((entry, depth + 1) for entry in item)
        elif item is not None and type(item) not in {str, int, float, bool}:
            raise _invalid()
        elif type(item) is float and not math.isfinite(item):
            raise _invalid("草稿中的非有限数值需先编码为草稿占位信息")
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, RecursionError, UnicodeError) as error:
        raise _invalid() from error
    if len(encoded) > MAX_CONTENT_BYTES:
        raise APIError(413, "TRAINING_DRAFT_TOO_LARGE", "训练草稿不能超过 512 KiB")
    return value


async def _body(request: Request, *, save: bool):
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_REQUEST_BYTES:
            raise APIError(413, "TRAINING_DRAFT_TOO_LARGE", "训练草稿不能超过 512 KiB")
        chunks.append(chunk)
    try:
        def invalid_constant(_):
            raise ValueError("Non-JSON numeric constant")
        value = json.loads(b"".join(chunks), parse_constant=invalid_constant)
    except (ValueError, TypeError, RecursionError, UnicodeError) as error:
        raise _invalid() from error
    expected = {"expected_revision", "content"} if save else {"expected_revision"}
    if type(value) is not dict or set(value) != expected:
        raise _invalid("草稿请求字段无效，用户归属由当前登录账户确定")
    revision = value["expected_revision"]
    if type(revision) is not int or not 0 <= revision <= MAX_EXPECTED_REVISION:
        raise _invalid("草稿版本必须为有效的非负整数")
    return revision, bounded_content(value["content"]) if save else None


def _representation(row):
    return ({"content": row.content, "revision": row.revision, "updated_at": row.updated_at}
            if row is not None else {"content": None, "revision": 0, "updated_at": None})


def _conflict():
    return APIError(409, "TRAINING_DRAFT_CONFLICT", "训练草稿已在其他窗口更新，请重新读取后再保存")


def mutate_draft(db, user_id, expected_revision, content):
    stamp = now()
    try:
        with db.write() as session:
            existing = session.get(TrainingDraft, user_id)
            if existing is None:
                if expected_revision != 0:
                    raise _conflict()
                session.add(TrainingDraft(user_id=user_id, revision=1, content=content, updated_at=stamp))
                session.flush()
            else:
                result = session.execute(update(TrainingDraft).where(
                    TrainingDraft.user_id == user_id,
                    TrainingDraft.revision == expected_revision,
                ).values(content=content, revision=expected_revision + 1, updated_at=stamp))
                if result.rowcount != 1:
                    raise _conflict()
    except IntegrityError as error:
        # Two initial saves may both observe an absent row on PostgreSQL.
        # The losing insert is rolled back instead of overwriting the winner.
        raise _conflict() from error
    return {"content": content, "revision": expected_revision + 1, "updated_at": stamp}


def register_routes(app, dependencies, db, auth):
    def owner(request):
        principal = auth.require(request, "training.write")
        if principal.legacy:
            raise APIError(403, "TRAINING_DRAFT_ACCOUNT_REQUIRED", "请使用个人账户登录后保存训练草稿")
        expected_user = request.headers.get("x-training-draft-user")
        if expected_user is not None and expected_user != principal.user_id:
            # This is an assertion, never an alternate owner selector. A queued
            # save from the previous account must not use replacement cookies.
            raise APIError(403, "TRAINING_DRAFT_USER_CHANGED", "登录账户已改变，未保存到其他账户的训练草稿")
        return principal.user_id

    def response(value):
        return JSONResponse(value, headers={"Cache-Control": "no-store"})

    @app.get("/api/v1/training-draft", dependencies=dependencies)
    def read(request: Request):
        user_id = owner(request)
        with db.read() as session:
            return response(_representation(session.get(TrainingDraft, user_id)))

    @app.put("/api/v1/training-draft", dependencies=dependencies)
    async def save(request: Request):
        user_id = owner(request)
        revision, content = await _body(request, save=True)
        return response(await run_in_threadpool(mutate_draft, db, user_id, revision, content))

    @app.delete("/api/v1/training-draft", dependencies=dependencies)
    async def clear(request: Request):
        user_id = owner(request)
        revision, _ = await _body(request, save=False)
        return response(await run_in_threadpool(mutate_draft, db, user_id, revision, None))
