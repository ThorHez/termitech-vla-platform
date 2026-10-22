"""Cloned recipes freeze authenticated source and never inherit base acceptance."""

import copy

import pytest

from vla_control.common import offset_time, uid
from vla_control.models import Command, Node, Profile
from vla_control.recipe_library import get_recipe_definition
from vla_control.recipe_models import RecipeRevision, SavedRecipe
from vla_adapters.recipe_definitions import definition_digest

from .conftest import ADMIN
from .test_recipe_details import create as create_detail, ready, register


def save(platform, node, query_id, **changes):
    body = {"node_id": node["node_id"], "profile_id": "default", "name": "我的抓取配方", "description": "低学习率",
            "detail_query_id": query_id, "overrides": {"batch_size": 16, "lr_schedule.peak_lr": 0.00001}, **changes}
    return platform.client.post("/api/v1/recipes", headers=ADMIN, json=body)


def get(platform, recipe_id):
    return platform.client.get(f"/api/v1/recipes/{recipe_id}", headers=ADMIN)


def revise(platform, recipe_id, query_id, **changes):
    return platform.client.post(f"/api/v1/recipes/{recipe_id}/revisions", headers=ADMIN,
                                json={"detail_query_id": query_id, "overrides": {"batch_size": 64}, **changes})


def test_create_freezes_agent_source_and_draft_revision(platform):
    node = register(platform)
    detail = ready(platform, node)
    response = save(platform, node, detail["id"])
    assert response.status_code == 201, response.text
    saved = response.json()
    assert saved["name"] == "我的抓取配方" and saved["status"] == "UNVALIDATED" and saved["validated"] is False
    revision = saved["latest_revision"]
    definition = revision["definition"]
    assert definition["base_recipe"] == "pi0_base" and definition["name"] == saved["name"]
    assert definition["source"] == detail["detail"]["source"] and definition["digest"] == definition_digest(definition)
    assert definition["id"] == revision["id"] and definition["revision"] == 1
    assert revision["detail"] == detail["detail"]
    assert get(platform, saved["id"]).json() == saved
    assert platform.client.get("/api/v1/recipes", headers=ADMIN).json() == [saved]
    with platform.app.state.db.read() as session:
        loaded = get_recipe_definition(session, revision["id"], node["node_id"], "default")
        assert loaded == definition
        loaded["overrides"]["batch_size"] = 999
        assert session.get(RecipeRevision, revision["id"]).definition["overrides"]["batch_size"] == 16
    assert response.headers["Cache-Control"] == "no-store"


def test_source_fields_and_validation_cannot_be_supplied_by_browser(platform):
    node = register(platform)
    detail = ready(platform, node)
    for extra in ({"source": detail["detail"]["source"]}, {"validated": True}, {"definition": {}}, {"base_recipe": "other"}):
        assert save(platform, node, detail["id"], **extra).status_code == 422
    saved = save(platform, node, detail["id"]).json()
    assert saved["validated"] is False


def test_every_library_route_requires_admin(platform):
    node = register(platform)
    saved = save(platform, node, ready(platform, node)["id"]).json()
    recipe_id, revision_id = saved["id"], saved["latest_revision"]["id"]
    for url in ("/api/v1/recipes", f"/api/v1/recipes/{recipe_id}", f"/api/v1/recipe-revisions/{revision_id}"):
        assert platform.client.get(url).status_code == 401
    assert platform.client.post("/api/v1/recipes", json={}).status_code == 401
    assert platform.client.post(f"/api/v1/recipes/{recipe_id}/revisions", json={}).status_code == 401


def test_unique_name_scope_and_list_filters(platform):
    node, other = register(platform), register(platform, name="other")
    detail, other_detail = ready(platform, node), ready(platform, other)
    first = save(platform, node, detail["id"]).json()
    duplicate = save(platform, node, detail["id"])
    assert duplicate.status_code == 409 and duplicate.json()["error"]["code"] == "RECIPE_NAME_EXISTS"
    second = save(platform, other, other_detail["id"]).json()
    assert second["id"] != first["id"]
    assert platform.client.get("/api/v1/recipes", headers=ADMIN, params={"node_id": node["node_id"], "profile_id": "default"}).json() == [first]
    assert platform.client.get("/api/v1/recipes", headers=ADMIN, params={"profile_id": "missing"}).json() == []


@pytest.mark.parametrize("overrides", [
    {"batch_size": True}, {"batch_size": 0}, {"batch_size": 1.5}, {"lr_schedule.peak_lr": "0.001"},
    {"lr_schedule.peak_lr": 10}, {"python": "/bin/sh"}, {"model": "arbitrary()"},
])
def test_overrides_are_typed_allowlisted_and_not_code(platform, overrides):
    node = register(platform)
    response = save(platform, node, ready(platform, node)["id"], overrides=overrides)
    assert response.status_code == 422 and response.json()["error"]["code"] == "RECIPE_OVERRIDES_INVALID"
    assert platform.client.get("/api/v1/recipes", headers=ADMIN).json() == []


@pytest.mark.parametrize("path,value", [
    ("data.datasets[1].repo_id", "/other/node"), ("data.datasets[1].assets.asset_id", "other"),
    ("data.datasets[1].assets.assets_dir", "/other/norm"), ("data.weights", [1, 100]),
    ("model.pretrained_path", "/other/weights"),
])
def test_asset_fields_are_rejected_even_when_agent_reports_editable_literals(platform, path, value):
    from .test_recipe_details import result
    node = register(platform)
    source = result()
    source["fields"].append({"path": path, "kind": "literal", "value": value, "expression": repr(value), "origin": "recipe"})
    detail = ready(platform, node, value=source)
    response = save(platform, node, detail["id"], overrides={path: value})
    assert response.status_code == 422 and response.json()["error"]["code"] == "RECIPE_ASSET_FIELDS_MANAGED"
    assert platform.client.get("/api/v1/recipes", headers=ADMIN).json() == []


def test_legacy_asset_overrides_remain_readable_but_block_run_and_writeback(platform):
    from vla_control.common import APIError
    node = register(platform)
    detail = ready(platform, node)
    saved = save(platform, node, detail["id"]).json()
    revision_id = saved["latest_revision"]["id"]
    with platform.app.state.db.write() as session:
        revision = session.get(RecipeRevision, revision_id)
        original = copy.deepcopy(revision.definition)
        original["overrides"]["data.datasets[1].assets.assets_dir"] = "/legacy/norm"
        original["digest"] = definition_digest(original)
        revision.definition = original
        session.get(Profile, (node["node_id"], "default")).metadata_json = {
            "validated_recipe_revisions": [original["digest"]],
        }
    historical = platform.client.get(f"/api/v1/recipe-revisions/{revision_id}", headers=ADMIN).json()
    assert historical["definition"] == original and historical["validated"] is False
    assert historical["asset_managed_overrides"] == ["data.datasets[1].assets.assets_dir"]
    with platform.app.state.db.read() as session:
        with pytest.raises(APIError) as error:
            get_recipe_definition(session, revision_id, node["node_id"], "default")
        assert error.value.code == "RECIPE_ASSET_FIELDS_MANAGED"
    response = platform.client.post(f"/api/v1/recipe-revisions/{revision_id}/writeback", headers=ADMIN,
                                    json={"config_name": "should_not_commit"})
    assert response.status_code == 422 and response.json()["error"]["code"] == "RECIPE_ASSET_FIELDS_MANAGED"
    assert not [command for command in platform.commands(node) if command["type"] == "RECIPE_WRITE"]
    fresh = ready(platform, node)
    clean = revise(platform, saved["id"], fresh["id"], overrides={"batch_size": 16})
    assert clean.status_code == 201
    assert platform.client.get(f"/api/v1/recipe-revisions/{revision_id}", headers=ADMIN).json()["definition"] == original


def test_save_needs_recent_ready_and_matching_detail(platform):
    node, other = register(platform), register(platform, name="other")
    pending = create_detail(platform, node).json()
    assert save(platform, node, pending["id"]).json()["error"]["code"] == "RECIPE_DETAIL_REQUIRED"
    detail = ready(platform, node)
    assert save(platform, other, detail["id"]).json()["error"]["code"] == "RECIPE_DETAIL_SCOPE_MISMATCH"
    with platform.app.state.db.write() as session:
        session.get(Command, detail["id"]).created_at = offset_time(-301)
    assert save(platform, node, detail["id"]).json()["error"]["code"] == "RECIPE_DETAIL_EXPIRED"
    fresh = ready(platform, node)
    with platform.app.state.db.write() as session:
        session.get(Profile, (node["node_id"], "default")).python = "/machine/changed-python"
    assert save(platform, node, fresh["id"]).json()["error"]["code"] == "RECIPE_DETAIL_REQUIRED"


def test_revision_is_immutable_and_does_not_inherit_acceptance(platform):
    node = register(platform)
    detail = ready(platform, node)
    saved = save(platform, node, detail["id"]).json()
    old_revision = saved["latest_revision"]
    old_digest = old_revision["definition"]["digest"]
    with platform.app.state.db.write() as session:
        session.get(Profile, (node["node_id"], "default")).metadata_json = {
            "validated": True, "validated_recipes": ["pi0_base"], "validated_recipe_revisions": [old_digest],
        }
    assert get(platform, saved["id"]).json()["validated"] is True
    detail = ready(platform, node)
    response = revise(platform, saved["id"], detail["id"], description="第二版")
    assert response.status_code == 201, response.text
    revised = response.json()
    assert revised["latest_revision"]["revision"] == 2 and revised["validated"] is False
    assert revised["latest_revision"]["definition"]["digest"] != old_digest
    historical = platform.client.get(f"/api/v1/recipe-revisions/{old_revision['id']}", headers=ADMIN).json()
    assert historical["definition"] == old_revision["definition"] and historical["detail"] == old_revision["detail"]
    assert historical["validated"] is True and revised["description"] == "第二版"
    assert platform.client.put(f"/api/v1/recipe-revisions/{old_revision['id']}", headers=ADMIN, json={}).status_code == 405


def test_definition_scope_check_and_tamper_detection(platform):
    node = register(platform)
    saved = save(platform, node, ready(platform, node)["id"]).json()
    revision_id = saved["latest_revision"]["id"]
    from vla_control.common import APIError
    with platform.app.state.db.read() as session:
        with pytest.raises(APIError) as error:
            get_recipe_definition(session, revision_id, uid(), "default")
        assert error.value.code == "RECIPE_REVISION_SCOPE_MISMATCH"
    with platform.app.state.db.write() as session:
        revision = session.get(RecipeRevision, revision_id)
        definition = copy.deepcopy(revision.definition)
        definition["overrides"]["batch_size"] = 128
        revision.definition = definition
    response = platform.client.get(f"/api/v1/recipe-revisions/{revision_id}", headers=ADMIN)
    assert response.status_code == 409 and response.json()["error"]["code"] == "RECIPE_REVISION_INVALID"


def test_library_stays_readable_when_machine_offline_but_save_is_blocked(platform):
    node = register(platform)
    detail = ready(platform, node)
    saved = save(platform, node, detail["id"]).json()
    with platform.app.state.db.write() as session:
        session.get(Node, node["node_id"]).status = "OFFLINE"
    assert get(platform, saved["id"]).status_code == 200
    assert revise(platform, saved["id"], detail["id"]).status_code == 409
    assert save(platform, node, detail["id"], name="different").status_code == 409


@pytest.mark.parametrize("name", [" ", "bad\nname", "x" * 201])
def test_display_name_is_bounded_text(platform, name):
    node = register(platform)
    assert save(platform, node, ready(platform, node)["id"], name=name).status_code == 422
