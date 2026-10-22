"""Durable, checkout-scoped recipe source writeback and Git commit receipts.

Unlike read-only discovery commands, a mutation is never expired by the control
plane: a delayed acknowledgement may describe an already committed change.
"""

import copy
from pathlib import PurePosixPath
from types import SimpleNamespace

from fastapi.responses import JSONResponse
from pydantic import Field
from sqlalchemy import select

from .common import APIError, now, offset_time, uid
from .models import Attempt, Command, Preview, Profile, Run
from . import recipe_details
from .recipe_models import RecipeRevision, SavedRecipe
from .recipe_suggestions import matching_profile, profile_digest, require_profile
from .schemas import InputModel


WRITE_TTL_SECONDS = 300
ERRORS = {
    "AGENT_RECIPE_WRITE_UNSUPPORTED": "当前 Agent 不支持配方回写，请更新 Agent 后再使用",
    "RECIPE_WRITE_DETAIL_UNAVAILABLE": "源码已提交，但提交后配方详情读取失败；请重新读取该配置名称后保存",
    "RECIPE_WRITE_INVALID_RESULT": "Agent 返回的配方提交凭据无效，请检查机器上的操作记录",
    "RECIPE_WRITE_FAILED": "配方回写失败，请检查机器上的操作记录",
    "INVALID_RECIPE_WRITE": "配方回写参数无效，请重新读取配方后保存",
    "RECIPE_WRITE_BUSY": "该代码目录正在执行其他操作，请等待操作完成",
    "RECIPE_SOURCE_DIRTY": "config.py 存在未提交修改，请处理后再回写",
    "RECIPE_NAME_CONFLICT": "代码中已存在该配方名称，请使用新的配置名称",
    "RECIPE_COMMIT_FAILED": "配方 Git 提交未完成，请检查机器上的操作记录",
    "RECIPE_GIT_BUSY": "代码仓库正在执行其他 Git 操作，请等待操作完成",
    "RECIPE_OPERATION_CONFLICT": "操作标识与已有提交记录不一致，请检查机器上的操作记录",
    "RECIPE_WRITE_EXPIRED": "配方回写命令已过期，未启动新的源码修改",
    "RECIPE_PROFILE_CHANGED": "训练环境配置已变化，请检查机器上的操作记录",
    "RECIPE_REVISION_CHANGED": "配方已有更新修订，提交结果未覆盖当前修订",
    "PROFILE_UNAVAILABLE": "该训练环境当前不可用",
    "RECIPE_SOURCE_CHANGED": "训练代码与保存配方时不一致，请重新读取详情并保存",
    "RECIPE_SOURCE_UNAVAILABLE": "无法读取配方配置文件",
    "RECIPE_SOURCE_INVALID": "配方配置文件无法解析",
    "RECIPE_SOURCE_TOO_LARGE": "配方配置超过处理上限",
    "RECIPE_NOT_FOUND": "原配方已不存在，请重新读取详情",
    "RECIPE_NAME_EXISTS": "代码中已存在该配方名称，请使用新的配置名称",
    "RECIPE_WRITE_UNSUPPORTED": "该配方的源码结构暂不支持安全回写",
    "RECIPE_OVERRIDES_INVALID": "修改项与当前配方的可编辑字段不一致",
    "RECIPE_ASSET_FIELDS_MANAGED": "数据路径、归一化、模型路径和混采权重由训练任务的资产绑定管理，请移除这些修改并保存新修订后再回写",
    "RECIPE_CHECKOUT_BUSY": "该代码目录有训练、预检或待调度任务，暂时无法修改源码",
    "GIT_WORKTREE_DIRTY": "代码目录存在未提交修改，请处理后再回写",
    "GIT_IDENTITY_REQUIRED": "训练代码仓库未配置 Git 提交身份",
    "GIT_COMMIT_FAILED": "配方 Git 提交失败，请检查机器上的操作记录",
    "GIT_UNAVAILABLE": "训练代码目录不是可用的 Git 仓库",
    "PATH_OUTSIDE_ROOTS": "训练代码不在机器允许访问的目录内",
    "CODE_VERSION_UNAVAILABLE": "无法确认训练代码版本",
}


class WritebackInput(InputModel):
    config_name: str = Field(min_length=1, max_length=255, pattern=r"^[A-Za-z0-9_-]+$")


def _error(code):
    return {"code": code, "message": ERRORS[code]}


def _checkout(value):
    return str(PurePosixPath(value)) if isinstance(value, str) and value.startswith("/") else None


def pending_for_checkout(session, node_id, code_dir):
    """Check the same checkout across profiles, including uncertain operations."""
    wanted = _checkout(code_dir)
    for command in session.scalars(select(Command).where(
            Command.node_id == node_id, Command.type == "RECIPE_WRITE", Command.status == "PENDING")):
        definition = command.payload.get("definition")
        source = definition.get("source", {}) if isinstance(definition, dict) else {}
        if _checkout(source.get("code_dir")) == wanted:
            return True
    return False


def require_checkout_idle(session, node_id, code_dir):
    """Reject source edits while any execution may still use the checkout."""
    wanted = _checkout(code_dir)
    profiles = {profile.id for profile in session.scalars(select(Profile).where(Profile.node_id == node_id))
                if _checkout(profile.code_dir) == wanted}
    active = {"QUEUED", "STARTING", "PREPARING", "RUNNING", "CANCELLING", "STOPPING", "UNKNOWN"}
    for run in session.scalars(select(Run).where(Run.node_id == node_id, Run.state.in_(active))):
        if run.profile_id in profiles or _checkout((run.code_expectation or {}).get("code_dir")) == wanted:
            raise APIError(409, "RECIPE_CHECKOUT_BUSY", ERRORS["RECIPE_CHECKOUT_BUSY"])
    for attempt in session.scalars(select(Attempt).where(Attempt.node_id == node_id, Attempt.state.in_(active))):
        run = session.get(Run, attempt.run_id)
        if (run is not None and run.profile_id in profiles
                or _checkout((attempt.code_snapshot or {}).get("code_dir")) == wanted):
            raise APIError(409, "RECIPE_CHECKOUT_BUSY", ERRORS["RECIPE_CHECKOUT_BUSY"])
    for preview in session.scalars(select(Preview).where(
            Preview.node_id == node_id, Preview.status.in_({"PENDING", "READY"}), Preview.run_id.is_(None),
            Preview.expires_at > now())):
        if (preview.request.get("profile_id") in profiles
                or _checkout((preview.code_snapshot or {}).get("code_dir")) == wanted):
            raise APIError(409, "RECIPE_CHECKOUT_BUSY", ERRORS["RECIPE_CHECKOUT_BUSY"])


def require_no_pending_writeback(session, node_id, code_dir):
    if pending_for_checkout(session, node_id, code_dir):
        raise APIError(409, "RECIPE_WRITEBACK_PENDING", "配方源码正在回写并提交，请等待结果后再预检或启动训练")


def require_recipe_not_pending(session, recipe_id):
    for command in session.scalars(select(Command).where(Command.type == "RECIPE_WRITE", Command.status == "PENDING")):
        definition = command.payload.get("definition")
        if isinstance(definition, dict) and definition.get("recipe_id") == recipe_id:
            raise APIError(409, "RECIPE_WRITEBACK_PENDING", "该配方正在回写并提交，请等待结果后再保存下一修订")


def validated_result(command, roots, result):
    invalid = {"error": _error("RECIPE_WRITE_INVALID_RESULT")}
    if not isinstance(result, dict):
        return invalid
    if result.get("error") is not None:
        error = result["error"]
        code = error.get("code") if isinstance(error, dict) else None
        code = {"COMMAND_UNSUPPORTED": "AGENT_RECIPE_WRITE_UNSUPPORTED",
                "RECIPE_QUERY_EXPIRED": "RECIPE_WRITE_EXPIRED"}.get(code, code)
        return {"error": _error(code if isinstance(code, str) and code in ERRORS else "RECIPE_WRITE_FAILED")}
    definition = command.payload["definition"]
    commit = result.get("commit_sha")
    if (result.get("operation_id") != command.id or result.get("definition_digest") != definition["digest"]
            or result.get("profile_id") != command.payload["profile_id"]
            or result.get("base_recipe") != definition["base_recipe"]
            or result.get("config_name") != command.payload["config_name"]
            or result.get("previous_commit_sha") != definition["source"]["commit_sha"]
            or not recipe_details._hex(commit, {40, 64}) or commit == result.get("previous_commit_sha")
            or result.get("changed_paths") != [definition["source"]["path"]]):
        return invalid
    receipt = {"operation_id": command.id, "definition_digest": definition["digest"],
               "profile_id": command.payload["profile_id"], "base_recipe": definition["base_recipe"],
               "config_name": command.payload["config_name"], "previous_commit_sha": result["previous_commit_sha"],
               "commit_sha": commit, "changed_paths": list(result["changed_paths"])}
    if result.get("detail") is None and isinstance(result.get("warning"), dict):
        # The commit identity is complete, but no detail exists from which to
        # build a new revision. Preserve the committed SHA without rebasing.
        return {**receipt, "detail": None, "error": _error("RECIPE_WRITE_DETAIL_UNAVAILABLE")}
    detail_command = SimpleNamespace(payload={"profile_id": command.payload["profile_id"],
                                               "recipe": command.payload["config_name"]})
    detail = recipe_details.validated_result(detail_command, roots, result.get("detail"))
    if ("error" in detail or detail["source"]["commit_sha"] != commit
            or _checkout(detail["source"]["code_dir"]) != _checkout(definition["source"]["code_dir"])
            or detail["source"]["path"] != definition["source"]["path"]):
        return invalid
    return {**receipt, "detail": detail}


def query_dict(command):
    result = {"id": command.id, "node_id": command.node_id, "profile_id": command.payload["profile_id"],
              "recipe_revision_id": command.payload["definition"]["id"], "config_name": command.payload["config_name"],
              "status": "PENDING", "commit_sha": None, "recipe": None, "error": None}
    if command.status == "PENDING":
        return result
    receipt = command.result if isinstance(command.result, dict) else {}
    if command.status == "APPLIED" and isinstance(receipt.get("recipe"), dict) and recipe_details._hex(receipt.get("commit_sha"), {40, 64}):
        result.update(status="SUCCEEDED", commit_sha=receipt["commit_sha"], recipe=copy.deepcopy(receipt["recipe"]))
    else:
        error = receipt.get("error")
        code = error.get("code") if isinstance(error, dict) else None
        result.update(status="FAILED", error=_error(code if isinstance(code, str) and code in ERRORS else "RECIPE_WRITE_FAILED"),
                      commit_sha=receipt.get("commit_sha") if recipe_details._hex(receipt.get("commit_sha"), {40, 64}) else None)
    return result


def acknowledge(session, command, node, status, result):
    """Atomically accept the commit and append one materialized revision."""
    from .recipe_library import _new_revision, recipe_dict

    clean = validated_result(command, node.roots, result)
    if status != "APPLIED" and "error" not in clean:
        clean = {"error": _error("RECIPE_WRITE_FAILED")}
    if command.status != "PENDING":
        # Compare the Agent's sanitized receipt, not the enriched control result.
        if (command.result or {}).get("agent_result") != clean:
            raise APIError(409, "ACK_CONFLICT", "This command already has a different acknowledgement")
        return
    agent_result = copy.deepcopy(clean)
    profile = session.get(Profile, (node.id, command.payload["profile_id"]))
    if "error" not in clean and not matching_profile(command, profile, node.roots):
        clean = {"error": _error("RECIPE_PROFILE_CHANGED")}
    definition = command.payload["definition"]
    recipe = session.scalar(select(SavedRecipe).where(SavedRecipe.id == definition["recipe_id"]).with_for_update())
    stored = {"agent_result": agent_result}
    if recipe_details._hex(agent_result.get("commit_sha"), {40, 64}):
        stored["commit_sha"] = agent_result["commit_sha"]
    if "error" not in clean and (recipe is None or recipe.latest_revision != definition["revision"]):
        stored["error"] = _error("RECIPE_REVISION_CHANGED")
    elif "error" in clean:
        stored["error"] = clean["error"]
    else:
        revision = _new_revision(session, recipe, clean["detail"], {}, recipe.description, recipe.latest_revision + 1)
        stored.update(commit_sha=clean["commit_sha"], recipe_revision_id=revision.id, recipe=recipe_dict(session, recipe))
    command.status = "REJECTED" if "error" in stored else "APPLIED"
    command.result, command.acknowledged_at = stored, now()
    session.flush()


def create_query(session, node, revision_id, config_name):
    from .recipe_library import _get, _verified_revision, require_asset_managed_overrides_absent

    revision = _get(session, RecipeRevision, revision_id, "Recipe revision")
    recipe = _get(session, SavedRecipe, revision.recipe_id, "Recipe")
    if recipe.node_id != node.id:
        raise APIError(422, "RECIPE_REVISION_SCOPE_MISMATCH", "配方不属于当前机器")
    # Stable identity comes before freshness checks: never produce a second
    # mutation merely because an acknowledgement or HTTP response was lost.
    for command in session.scalars(select(Command).where(
            Command.node_id == node.id, Command.type == "RECIPE_WRITE").order_by(Command.created_at, Command.id)):
        if (command.payload.get("definition", {}).get("id") == revision.id
                and command.payload.get("config_name") == config_name):
            return query_dict(command)
    recipe = session.scalar(select(SavedRecipe).where(SavedRecipe.id == recipe.id).with_for_update()
                            .execution_options(populate_existing=True))
    if recipe.latest_revision != revision.revision:
        raise APIError(409, "RECIPE_REVISION_CHANGED", "请使用当前最新配方修订进行回写")
    definition = _verified_revision(revision, recipe)
    require_asset_managed_overrides_absent(definition)
    profile = require_profile(session, node.id, recipe.profile_id)
    if _checkout(definition["source"]["code_dir"]) != _checkout(profile.code_dir):
        raise APIError(409, "RECIPE_PROFILE_CHANGED", ERRORS["RECIPE_PROFILE_CHANGED"])
    require_no_pending_writeback(session, node.id, profile.code_dir)
    require_checkout_idle(session, node.id, profile.code_dir)
    command_id = uid()
    command = Command(id=command_id, node_id=node.id, type="RECIPE_WRITE", status="PENDING", created_at=now(),
                      payload={"operation_id": command_id, "profile_id": profile.id,
                               "profile_digest": profile_digest(profile, node.roots), "definition": copy.deepcopy(definition),
                               "config_name": config_name, "expires_at": offset_time(WRITE_TTL_SECONDS)})
    session.add(command)
    session.flush()
    return query_dict(command)


def register_routes(app, admin, db, require_available_node, settings):
    def response(value, status=200):
        return JSONResponse(status_code=status, content=value, headers={"Cache-Control": "no-store"})

    @app.post("/api/v1/recipe-revisions/{revision_id}/writeback", dependencies=admin, status_code=202)
    def create(revision_id: str, body: WritebackInput):
        with db.write() as session:
            revision = session.get(RecipeRevision, revision_id)
            if revision is None:
                raise APIError(404, "NOT_FOUND", "Recipe revision not found")
            recipe = session.get(SavedRecipe, revision.recipe_id)
            node = require_available_node(session, recipe.node_id)
            if node.last_seen < offset_time(-settings.heartbeat_timeout_seconds):
                raise APIError(409, "NODE_OFFLINE", "机器心跳已过期，暂时无法回写配方")
            return response(create_query(session, node, revision_id, body.config_name), 202)

    @app.get("/api/v1/recipe-writebacks/{operation_id}", dependencies=admin)
    def get(operation_id: str):
        with db.read() as session:
            command = session.get(Command, operation_id)
            if command is None or command.type != "RECIPE_WRITE":
                raise APIError(404, "NOT_FOUND", "Recipe writeback not found")
            return response(query_dict(command))
