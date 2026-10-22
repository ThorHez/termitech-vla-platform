"""Immutable, data-only recipe revisions cloned from authenticated Agent details."""

import copy

from fastapi.responses import JSONResponse
from pydantic import Field, field_validator
from sqlalchemy import select
from vla_adapters.recipe_definitions import (definition_digest, is_asset_managed_field, validate_definition,
                                            validate_overrides, validate_overrides_against_fields)

from .common import APIError, now, offset_time, uid
from .models import Command, Profile
from . import recipe_details
from .recipe_models import RecipeRevision, SavedRecipe
from .schemas import InputModel


class RecipeCreateInput(InputModel):
    node_id: str = Field(min_length=1, max_length=36)
    profile_id: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=4000)
    detail_query_id: str = Field(min_length=1, max_length=36)
    overrides: dict = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def name_is_display_text(cls, value):
        value = value.strip()
        if not value or any(ord(char) < 32 or 127 <= ord(char) < 160 for char in value):
            raise ValueError("配方名称不能为空或包含控制字符")
        return value


class RecipeRevisionInput(InputModel):
    detail_query_id: str = Field(min_length=1, max_length=36)
    overrides: dict = Field(default_factory=dict)
    description: str | None = Field(default=None, max_length=4000)


def _get(session, model, identity, label):
    value = session.get(model, identity)
    if value is None:
        raise APIError(404, "NOT_FOUND", f"{label} not found")
    return value


def _verified_revision(revision, recipe):
    try:
        definition = validate_definition(revision.definition)
    except (ValueError, TypeError):
        raise APIError(409, "RECIPE_REVISION_INVALID", "保存的配方修订内容无效，请重新创建修订") from None
    if (definition["id"] != revision.id or definition["recipe_id"] != recipe.id
            or definition["revision"] != revision.revision or definition["node_id"] != recipe.node_id
            or definition["profile_id"] != recipe.profile_id):
        raise APIError(409, "RECIPE_REVISION_INVALID", "保存的配方修订身份不一致")
    return definition


def get_recipe_definition(session, revision_id, node_id, profile_id):
    """Resolve a saved revision for the run pipeline; acceptance is checked there."""
    revision = _get(session, RecipeRevision, revision_id, "Recipe revision")
    recipe = _get(session, SavedRecipe, revision.recipe_id, "Recipe")
    if recipe.node_id != node_id or recipe.profile_id != profile_id:
        raise APIError(422, "RECIPE_REVISION_SCOPE_MISMATCH", "保存的配方修订不属于当前机器和训练环境")
    definition = _verified_revision(revision, recipe)
    require_asset_managed_overrides_absent(definition)
    return definition


def require_asset_managed_overrides_absent(definition):
    """Legacy revisions are readable, but need a new revision before reuse."""
    try:
        validate_overrides(definition["overrides"])
    except ValueError as exc:
        if str(exc).startswith("RECIPE_ASSET_FIELDS_MANAGED:"):
            raise APIError(422, "RECIPE_ASSET_FIELDS_MANAGED",
                           "配方包含由资产绑定管理的数据路径、归一化、模型路径或混采权重，请移除这些修改并另存新修订，再在训练任务中选择资产") from None
        raise


def revision_dict(session, revision, recipe=None):
    recipe = recipe or _get(session, SavedRecipe, revision.recipe_id, "Recipe")
    definition = _verified_revision(revision, recipe)
    profile = session.get(Profile, (recipe.node_id, recipe.profile_id))
    approvals = (profile.metadata_json or {}).get("validated_recipe_revisions", []) if profile and profile.active else []
    asset_conflicts = [path for path in definition["overrides"] if is_asset_managed_field(path)]
    try:
        require_asset_managed_overrides_absent(definition)
    except APIError as exc:
        if exc.code != "RECIPE_ASSET_FIELDS_MANAGED":
            raise
        asset_conflicts = asset_conflicts or list(definition["overrides"])
    validated = not asset_conflicts and isinstance(approvals, list) and definition["digest"] in approvals
    return {"id": revision.id, "recipe_id": recipe.id, "revision": revision.revision, "definition": definition,
            "detail": copy.deepcopy(revision.source_detail), "description": revision.description,
            "created_at": revision.created_at, "validated": validated,
            "status": "VALIDATED" if validated else "UNVALIDATED",
            "asset_managed_overrides": asset_conflicts}


def recipe_dict(session, recipe):
    revision = session.scalar(select(RecipeRevision).where(RecipeRevision.recipe_id == recipe.id,
                                                          RecipeRevision.revision == recipe.latest_revision))
    if revision is None:
        raise APIError(409, "RECIPE_REVISION_INVALID", "保存的配方缺少当前修订")
    latest = revision_dict(session, revision, recipe)
    return {"id": recipe.id, "node_id": recipe.node_id, "profile_id": recipe.profile_id,
            "name": recipe.name, "description": recipe.description, "created_at": recipe.created_at,
            "latest_revision": latest, "validated": latest["validated"], "status": latest["status"]}


def _save_detail(session, node, profile_id, query_id):
    command = _get(session, Command, query_id, "Recipe detail")
    if (command.type != "RECIPE_DETAIL" or command.node_id != node.id
            or command.payload.get("profile_id") != profile_id):
        raise APIError(422, "RECIPE_DETAIL_SCOPE_MISMATCH", "配方详情不属于当前机器和训练环境")
    recipe_details.expire_pending(session, node.id)
    profile = session.get(Profile, (node.id, profile_id))
    detail = recipe_details.query_dict(command, profile, node.roots)
    if detail["status"] != "READY":
        raise APIError(409, "RECIPE_DETAIL_REQUIRED", "请先成功读取当前配方详情，再保存新配方")
    if command.created_at < offset_time(-recipe_details.DETAIL_VALID_SECONDS):
        raise APIError(409, "RECIPE_DETAIL_EXPIRED", "配方详情已过期，请重新读取后保存")
    return detail["detail"]


def _new_revision(session, recipe, detail, overrides, description, number):
    try:
        overrides = validate_overrides_against_fields(overrides, detail["fields"])
    except (ValueError, TypeError) as exc:
        if str(exc).startswith("RECIPE_ASSET_FIELDS_MANAGED:"):
            raise APIError(422, "RECIPE_ASSET_FIELDS_MANAGED",
                           "数据路径、归一化、模型路径和混采权重由训练任务的资产绑定管理，不能保存为配方修改") from None
        raise APIError(422, "RECIPE_OVERRIDES_INVALID", "修改项包含不支持的字段或超出允许范围的值") from None
    revision_id = uid()
    definition = {"schema_version": 1, "id": revision_id, "recipe_id": recipe.id, "name": recipe.name,
                  "revision": number, "base_recipe": detail["recipe"], "node_id": recipe.node_id,
                  "profile_id": recipe.profile_id, "source": copy.deepcopy(detail["source"]), "overrides": overrides}
    definition["digest"] = definition_digest(definition)
    try:
        definition = validate_definition(definition)
    except (ValueError, TypeError):
        raise APIError(409, "RECIPE_DEFINITION_INVALID", "无法生成有效的配方修订，请重新读取配方详情") from None
    revision = RecipeRevision(id=revision_id, recipe_id=recipe.id, revision=number, definition=definition,
                              source_detail=copy.deepcopy(detail), description=description, created_at=now())
    session.add(revision)
    recipe.latest_revision = number
    recipe.description = description
    session.flush()
    return revision


def register_recipe_routes(app, admin, db, require_available_node, settings):
    def response(value, status=200):
        return JSONResponse(status_code=status, content=value, headers={"Cache-Control": "no-store"})

    def available(session, node_id):
        node = require_available_node(session, node_id)
        if node.last_seen < offset_time(-settings.heartbeat_timeout_seconds):
            raise APIError(409, "NODE_OFFLINE", "机器心跳已过期，暂时无法读取或保存配方")
        return node

    @app.post("/api/v1/nodes/{node_id}/recipe-details", dependencies=admin, status_code=202)
    def details_create(node_id: str, body: recipe_details.RecipeDetailInput):
        with db.write() as session:
            node = available(session, node_id)
            return response(recipe_details.create_query(session, node, body.profile_id, body.recipe), 202)

    @app.get("/api/v1/nodes/{node_id}/recipe-details/{query_id}", dependencies=admin)
    def details_get(node_id: str, query_id: str):
        with db.write() as session:
            node = available(session, node_id)
            command = _get(session, Command, query_id, "Recipe detail")
            if command.node_id != node.id or command.type != "RECIPE_DETAIL":
                raise APIError(404, "NOT_FOUND", "Recipe detail not found")
            recipe_details.expire_pending(session, node.id)
            profile = session.get(Profile, (node.id, command.payload["profile_id"]))
            return response(recipe_details.query_dict(command, profile, node.roots))

    @app.get("/api/v1/recipes", dependencies=admin)
    def recipes_list(node_id: str | None = None, profile_id: str | None = None):
        with db.read() as session:
            query = select(SavedRecipe).order_by(SavedRecipe.created_at.desc(), SavedRecipe.id)
            if node_id is not None:
                query = query.where(SavedRecipe.node_id == node_id)
            if profile_id is not None:
                query = query.where(SavedRecipe.profile_id == profile_id)
            return response([recipe_dict(session, recipe) for recipe in session.scalars(query)])

    @app.get("/api/v1/recipes/{recipe_id}", dependencies=admin)
    def recipe_get(recipe_id: str):
        with db.read() as session:
            return response(recipe_dict(session, _get(session, SavedRecipe, recipe_id, "Recipe")))

    @app.get("/api/v1/recipe-revisions/{revision_id}", dependencies=admin)
    def revision_get(revision_id: str):
        with db.read() as session:
            return response(revision_dict(session, _get(session, RecipeRevision, revision_id, "Recipe revision")))

    @app.post("/api/v1/recipes", dependencies=admin, status_code=201)
    def recipe_create(body: RecipeCreateInput):
        with db.write() as session:
            node = available(session, body.node_id)
            detail = _save_detail(session, node, body.profile_id, body.detail_query_id)
            duplicate = select(SavedRecipe.id).where(SavedRecipe.node_id == node.id, SavedRecipe.profile_id == body.profile_id,
                                                     SavedRecipe.name == body.name).limit(1)
            if session.scalar(duplicate):
                raise APIError(409, "RECIPE_NAME_EXISTS", "当前机器和训练环境中已有同名配方，请更换名称")
            recipe = SavedRecipe(id=uid(), node_id=node.id, profile_id=body.profile_id, name=body.name,
                                 description=body.description, latest_revision=1, created_at=now())
            session.add(recipe)
            session.flush()
            _new_revision(session, recipe, detail, body.overrides, body.description, 1)
            return response(recipe_dict(session, recipe), 201)

    @app.post("/api/v1/recipes/{recipe_id}/revisions", dependencies=admin, status_code=201)
    def revision_create(recipe_id: str, body: RecipeRevisionInput):
        with db.write() as session:
            # Lock the node before the recipe, matching run/profile mutations.
            existing = _get(session, SavedRecipe, recipe_id, "Recipe")
            node = available(session, existing.node_id)
            recipe = session.scalar(select(SavedRecipe).where(SavedRecipe.id == recipe_id).with_for_update()
                                    .execution_options(populate_existing=True))
            from .recipe_writeback import require_recipe_not_pending
            require_recipe_not_pending(session, recipe.id)
            detail = _save_detail(session, node, recipe.profile_id, body.detail_query_id)
            description = recipe.description if body.description is None else body.description
            _new_revision(session, recipe, detail, body.overrides, description, recipe.latest_revision + 1)
            return response(recipe_dict(session, recipe), 201)
