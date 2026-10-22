"""Source mutations keep stable command identity and immutable recipe lineage."""

import copy

import pytest
from sqlalchemy import func, select

from vla_control.common import now, offset_time, uid
from vla_control.models import Attempt, Command, Node, Preview, Profile, Run
from vla_control.recipe_models import RecipeRevision, SavedRecipe
from vla_control.recipe_writeback import pending_for_checkout

from .conftest import ADMIN
from .test_recipe_details import ready, register, result as detail_result
from .test_recipe_library import get as get_recipe, revise, save


def saved(platform, node=None, value=None):
    node = node or register(platform)
    detail = ready(platform, node, value=value)
    response = save(platform, node, detail["id"])
    assert response.status_code == 201, response.text
    return node, response.json(), detail


def create(platform, recipe, config_name="pi0_custom", headers=ADMIN):
    revision = recipe["latest_revision"]
    return platform.client.post(f"/api/v1/recipe-revisions/{revision['id']}/writeback", headers=headers,
                                json={"config_name": config_name})


def get(platform, operation, headers=ADMIN):
    return platform.client.get(f"/api/v1/recipe-writebacks/{operation['id']}", headers=headers)


def result(platform, node, operation):
    command = next(item for item in platform.commands(node) if item["id"] == operation["id"])
    payload = command["payload"]
    source = copy.deepcopy(payload["definition"]["source"])
    source.update(commit_sha="d" * 40, sha256="e" * 64, changes_digest="f" * 64, line=110, end_line=120)
    detail = detail_result(recipe=payload["config_name"], source=source,
                           source_code=f"TrainConfig(name='{payload['config_name']}', batch_size=16)")
    detail["fields"][0].update(value=16, expression="16")
    return {"operation_id": operation["id"], "definition_digest": payload["definition"]["digest"],
            "profile_id": "default", "base_recipe": payload["definition"]["base_recipe"],
            "config_name": payload["config_name"], "previous_commit_sha": payload["definition"]["source"]["commit_sha"],
            "commit_sha": "d" * 40, "changed_paths": [source["path"]], "detail": detail}


def ack(platform, node, operation, value, status="APPLIED"):
    return platform.client.post(f"/agent/v1/commands/{operation['id']}/ack", headers=node["headers"],
                                json={"node_id": node["node_id"], "status": status, "result": value})


def test_writeback_materializes_one_new_revision_and_ack_retry_is_idempotent(platform):
    node, recipe, _ = saved(platform)
    old = copy.deepcopy(recipe["latest_revision"])
    response = create(platform, recipe)
    assert response.status_code == 202 and response.headers["Cache-Control"] == "no-store"
    operation = response.json()
    assert operation["status"] == "PENDING" and operation["recipe"] is None
    assert create(platform, recipe).json() == operation
    command = next(item for item in platform.commands(node) if item["id"] == operation["id"])
    assert command["type"] == "RECIPE_WRITE" and command["payload"]["definition"] == old["definition"]
    value = result(platform, node, operation)
    value["private_agent_output"] = "never persisted"
    value["detail"]["fields"][0]["editable"] = True
    assert ack(platform, node, operation, value).status_code == 200
    completed = get(platform, operation).json()
    assert completed["status"] == "SUCCEEDED" and completed["commit_sha"] == "d" * 40
    current = completed["recipe"]["latest_revision"]
    assert current["revision"] == 2 and current["definition"]["base_recipe"] == "pi0_custom"
    assert current["definition"]["overrides"] == {} and current["definition"]["source"]["commit_sha"] == "d" * 40
    assert current["validated"] is False
    assert platform.client.get(f"/api/v1/recipe-revisions/{old['id']}", headers=ADMIN).json() == old
    assert ack(platform, node, operation, value).status_code == 200
    assert create(platform, recipe).json() == completed
    with platform.app.state.db.read() as session:
        assert session.scalar(select(func.count()).select_from(RecipeRevision)) == 2
        assert "private_agent_output" not in session.get(Command, operation["id"]).result["agent_result"]
    changed = copy.deepcopy(value)
    changed["commit_sha"] = "a" * 40
    assert ack(platform, node, operation, changed).status_code == 409


def test_writeback_auth_browser_source_rejection_and_agent_ownership(platform):
    node, recipe, _ = saved(platform)
    for headers in ({}, node["headers"]):
        assert create(platform, recipe, headers=headers).status_code == 401
    operation = create(platform, recipe).json()
    assert get(platform, operation, headers={}).status_code == 401
    other = register(platform, name="other")
    assert ack(platform, other, operation, {"error": {"code": "COMMAND_UNSUPPORTED"}}).status_code == 403
    revision_id = recipe["latest_revision"]["id"]
    response = platform.client.post(f"/api/v1/recipe-revisions/{revision_id}/writeback", headers=ADMIN,
                                    json={"config_name": "x", "source": {"path": "/etc/passwd"}})
    assert response.status_code == 422
    assert get(platform, {"id": recipe["id"]}).status_code == 404


@pytest.mark.parametrize("name", ["", "invalid/name", "x y", "配置", "x" * 256, "x\n"])
def test_config_name_is_bounded_ascii(platform, name):
    _, recipe, _ = saved(platform)
    assert create(platform, recipe, name).status_code == 422


@pytest.mark.parametrize("mutate", [
    lambda value: value.update(operation_id=uid()),
    lambda value: value.update(definition_digest="a" * 64),
    lambda value: value.update(profile_id="other"),
    lambda value: value.update(config_name="other"),
    lambda value: value.update(base_recipe="other"),
    lambda value: value.update(previous_commit_sha="a" * 40),
    lambda value: value.update(commit_sha="invalid"),
    lambda value: value.update(changed_paths=["unrelated.py"]),
    lambda value: value["detail"]["source"].update(commit_sha="a" * 40),
    lambda value: value["detail"]["source"].update(code_dir="/machine/other"),
    lambda value: value["detail"].update(recipe="other"),
])
def test_untrusted_receipts_never_create_revision(platform, mutate):
    node, recipe, _ = saved(platform)
    operation = create(platform, recipe).json()
    value = result(platform, node, operation)
    mutate(value)
    assert ack(platform, node, operation, value).status_code == 200
    failed = get(platform, operation).json()
    assert failed["status"] == "FAILED" and failed["error"]["code"] == "RECIPE_WRITE_INVALID_RESULT"
    assert get_recipe(platform, recipe["id"]).json() == recipe
    assert create(platform, recipe).json() == failed


@pytest.mark.parametrize("code,expected", [("COMMAND_UNSUPPORTED", "AGENT_RECIPE_WRITE_UNSUPPORTED"),
    ("GIT_WORKTREE_DIRTY", "GIT_WORKTREE_DIRTY"), ("RECIPE_SOURCE_CHANGED", "RECIPE_SOURCE_CHANGED"),
    ("PRIVATE_REMOTE_ERROR", "RECIPE_WRITE_FAILED")])
def test_safe_failures_do_not_rebase_or_requeue(platform, code, expected):
    node, recipe, _ = saved(platform)
    operation = create(platform, recipe).json()
    value = {"error": {"code": code, "message": "private paths and command output"}}
    assert ack(platform, node, operation, value, "REJECTED").status_code == 200
    failed = get(platform, operation).json()
    assert failed["status"] == "FAILED" and failed["error"]["code"] == expected
    assert "private" not in str(failed)
    assert get_recipe(platform, recipe["id"]).json() == recipe
    assert ack(platform, node, operation, value, "REJECTED").status_code == 200
    assert create(platform, recipe).json() == failed


def test_pending_job_never_expires_and_blocks_recipe_edits_and_preview(platform):
    node, recipe, detail = saved(platform)
    operation = create(platform, recipe).json()
    with platform.app.state.db.write() as session:
        command = session.get(Command, operation["id"])
        payload = dict(command.payload, expires_at=offset_time(-1000))
        command.payload = payload
        assert pending_for_checkout(session, node["node_id"], "/machine/code")
        assert not pending_for_checkout(session, node["node_id"], "/machine/other")
    platform.schedule()
    assert get(platform, operation).json()["status"] == "PENDING"
    assert revise(platform, recipe["id"], detail["id"]).status_code == 409
    assert create(platform, recipe, "other_name").json()["error"]["code"] == "RECIPE_WRITEBACK_PENDING"
    response = platform.client.post("/api/v1/run-previews", headers=ADMIN, json={
        "name": "training", "node_id": node["node_id"], "profile_id": "default", "template_id": "openpi-sft",
        "recipe": "pi0_base", "inputs": {}, "parameters": {}, "gpu_count": 1})
    # Validation of inputs precedes mutation gating; use a smoke profile sharing
    # the checkout to exercise the guard without requiring local input assets.
    with platform.app.state.db.write() as session:
        session.add(Profile(node_id=node["node_id"], id="smoke", name="smoke", code_dir="/machine/code",
                            python="/usr/bin/python3", adapter="smoke", recipe="smoke", active=True))
    response = platform.client.post("/api/v1/run-previews", headers=ADMIN, json={
        "name": "training", "node_id": node["node_id"], "profile_id": "smoke", "template_id": "smoke",
        "inputs": {}, "parameters": {"steps": 1}, "gpu_count": 1})
    assert response.status_code == 409 and response.json()["error"]["code"] == "RECIPE_WRITEBACK_PENDING"


@pytest.mark.parametrize("state", ["QUEUED", "STARTING", "PREPARING", "RUNNING", "CANCELLING", "STOPPING", "UNKNOWN"])
def test_active_checkout_prevents_writeback_dispatch(platform, state):
    node, recipe, _ = saved(platform)
    with platform.app.state.db.write() as session:
        session.add(Run(id=uid(), name="active", node_id=node["node_id"], profile_id="default", template_id="openpi-sft",
                        state=state, created_at=now(), updated_at=now(), inputs={}, parameters={}, gpu_count=1,
                        code_expectation={"code_dir": "/machine/code"}))
    response = create(platform, recipe)
    assert response.status_code == 409 and response.json()["error"]["code"] == "RECIPE_CHECKOUT_BUSY"
    assert not any(command["type"] == "RECIPE_WRITE" for command in platform.commands(node))


@pytest.mark.parametrize("status", ["PENDING", "READY"])
def test_live_preflight_prevents_writeback_dispatch(platform, status):
    node, recipe, _ = saved(platform)
    with platform.app.state.db.write() as session:
        session.add(Preview(id=uid(), node_id=node["node_id"], status=status,
                            request={"profile_id": "default"}, created_at=now(), expires_at=offset_time(60)))
    response = create(platform, recipe)
    assert response.status_code == 409 and response.json()["error"]["code"] == "RECIPE_CHECKOUT_BUSY"


def test_profile_change_during_job_preserves_retry_receipt_and_old_revision(platform):
    node, recipe, _ = saved(platform)
    operation = create(platform, recipe).json()
    value = result(platform, node, operation)
    with platform.app.state.db.write() as session:
        session.get(Profile, (node["node_id"], "default")).python = "/machine/other-python"
    assert ack(platform, node, operation, value).status_code == 200
    assert get(platform, operation).json()["error"]["code"] == "RECIPE_PROFILE_CHANGED"
    assert ack(platform, node, operation, value).status_code == 200
    assert get_recipe(platform, recipe["id"]).json() == recipe


def test_latest_revision_conflict_never_overwrites_newer_recipe(platform):
    node, recipe, _ = saved(platform)
    operation = create(platform, recipe).json()
    value = result(platform, node, operation)
    with platform.app.state.db.write() as session:
        session.get(SavedRecipe, recipe["id"]).latest_revision = 99
    assert ack(platform, node, operation, value).status_code == 200
    assert get(platform, operation).json()["error"]["code"] == "RECIPE_REVISION_CHANGED"
    with platform.app.state.db.read() as session:
        assert session.get(SavedRecipe, recipe["id"]).latest_revision == 99
        assert session.scalar(select(func.count()).select_from(RecipeRevision)) == 1


def test_checkout_alias_profile_is_also_guarded(platform):
    node, recipe, _ = saved(platform)
    with platform.app.state.db.write() as session:
        session.add(Profile(node_id=node["node_id"], id="alias", name="alias", code_dir="/machine/code/",
                            python="/usr/bin/python3", adapter="smoke", recipe="smoke", active=True))
        session.flush()
        session.add(Run(id=uid(), name="active", node_id=node["node_id"], profile_id="alias", template_id="smoke",
                        state="RUNNING", created_at=now(), updated_at=now(), inputs={}, parameters={}, gpu_count=1,
                        code_expectation={}))
    assert create(platform, recipe).json()["error"]["code"] == "RECIPE_CHECKOUT_BUSY"


def test_generic_literals_require_observed_matching_leaf(platform):
    fields = detail_result()["fields"] + [
        {"path": "data.state_pad_dim", "kind": "literal", "value": 128, "expression": "128", "origin": "recipe"},
        {"path": "data.datasets[0].base_config.use_quantile_norm", "kind": "literal", "value": True, "expression": "True", "origin": "recipe"},
        {"path": "data.datasets[0].base_config.action_sequence_keys", "kind": "literal", "value": [], "expression": "[]", "origin": "recipe"}]
    node = register(platform)
    detail = ready(platform, node, detail_result(fields=fields))
    response = save(platform, node, detail["id"], overrides={"data.state_pad_dim": 256,
                                                          "data.datasets[0].base_config.use_quantile_norm": False,
                                                          "data.datasets[0].base_config.action_sequence_keys": ["actions"]})
    assert response.status_code == 201, response.text
    for overrides in ({"data.state_pad_dim": True}, {"data.absent": 123}, {"model": "arbitrary()"}):
        assert save(platform, node, detail["id"], name=uid(), overrides=overrides).status_code == 422


def test_commit_receipt_preserves_unrelated_dirty_files(platform):
    node, recipe, _ = saved(platform)
    operation = create(platform, recipe).json()
    value = result(platform, node, operation)
    value["detail"]["source"]["dirty"] = True
    assert ack(platform, node, operation, value).status_code == 200
    completed = get(platform, operation).json()
    assert completed["status"] == "SUCCEEDED"
    assert completed["recipe"]["latest_revision"]["definition"]["source"]["dirty"] is True


def test_committed_source_with_missing_detail_keeps_sha_without_materializing(platform):
    node, recipe, _ = saved(platform)
    operation = create(platform, recipe).json()
    value = result(platform, node, operation)
    value.update(detail=None, warning={"code": "PRIVATE_ERROR", "message": "private machine details"})
    assert ack(platform, node, operation, value).status_code == 200
    completed = get(platform, operation).json()
    assert completed["status"] == "FAILED" and completed["commit_sha"] == "d" * 40
    assert completed["error"]["code"] == "RECIPE_WRITE_DETAIL_UNAVAILABLE"
    assert completed["recipe"] is None and "private" not in str(completed)
    assert get_recipe(platform, recipe["id"]).json() == recipe
    assert ack(platform, node, operation, value).status_code == 200
    assert create(platform, recipe).json() == completed


def test_unresolved_attempt_blocks_write_even_if_run_state_is_terminal(platform):
    node, recipe, _ = saved(platform)
    run_id = uid()
    with platform.app.state.db.write() as session:
        session.add(Run(id=run_id, name="reported done", node_id=node["node_id"], profile_id="default",
                        template_id="openpi-sft", state="FAILED", created_at=now(), updated_at=now(), inputs={},
                        parameters={}, gpu_count=1, code_expectation={}))
        session.flush()
        session.add(Attempt(id=uid(), run_id=run_id, node_id=node["node_id"], number=1, execution_token=uid(),
                            state="UNKNOWN", created_at=now(), gpu_uuids=[], code_snapshot=None))
    response = create(platform, recipe)
    assert response.status_code == 409 and response.json()["error"]["code"] == "RECIPE_CHECKOUT_BUSY"


def test_unrelated_checkout_run_does_not_block_and_offline_receipt_stays_readable(platform):
    node, recipe, _ = saved(platform)
    with platform.app.state.db.write() as session:
        session.add(Profile(node_id=node["node_id"], id="other", name="other", code_dir="/machine/other",
                            python="/usr/bin/python3", adapter="smoke", recipe="smoke", active=True))
        session.flush()
        session.add(Run(id=uid(), name="other training", node_id=node["node_id"], profile_id="other",
                        template_id="smoke", state="RUNNING", created_at=now(), updated_at=now(), inputs={},
                        parameters={}, gpu_count=1, code_expectation={"code_dir": "/machine/other"}))
    response = create(platform, recipe)
    assert response.status_code == 202
    operation = response.json()
    with platform.app.state.db.write() as session:
        session.get(Node, node["node_id"]).status = "OFFLINE"
    assert get(platform, operation).json() == operation
