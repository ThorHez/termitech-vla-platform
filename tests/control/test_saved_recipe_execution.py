"""Saved recipe revisions are automatically checked and frozen per run."""

import copy

import pytest

from vla_control.common import uid
from vla_control.models import Profile

from .conftest import ADMIN, automatic_checks, automatic_preflight, confirmed_run_request, snapshot
from .test_recipe_details import ready, register
from .test_recipe_library import revise, save


def prepared(platform, overrides=None):
    node = register(platform)
    saved = save(platform, node, ready(platform, node)["id"], overrides=overrides or {
        "batch_size": 16, "num_train_steps": 100, "lr_schedule.peak_lr": 0.00002,
    }).json()
    assert "latest_revision" in saved, saved
    inputs = {}
    for role, kind in (("dataset", "DATASET"), ("model", "MODEL"), ("norm", "NORM")):
        response = platform.client.post("/api/v1/assets", headers=ADMIN, json={
            "node_id": node["node_id"], "name": role, "kind": kind, "path": f"/machine/{role}", "version": "1",
        })
        assert response.status_code == 201, response.text
        inputs[role] = response.json()["id"]
    return node, saved, inputs


def approve(platform, node, *saved_or_revision, **metadata):
    approvals = []
    for value in saved_or_revision:
        revision = value.get("latest_revision", value)
        approvals.append(revision["definition"]["digest"])
    with platform.app.state.db.write() as session:
        session.get(Profile, (node["node_id"], "default")).metadata_json = {
            "validated_recipe_revisions": approvals, **metadata,
        }


def preview(platform, node, saved, inputs, **changes):
    body = {"name": "saved recipe run", "node_id": node["node_id"], "profile_id": "default", "template_id": "openpi-sft",
            "recipe_revision_id": saved["latest_revision"]["id"], "inputs": inputs, "parameters": {}, "gpu_count": 1, **changes}
    return platform.client.post("/api/v1/run-previews", headers=ADMIN, json=body)


def acknowledge_preview(platform, node, pending, *, definition="expected", recipe="pi0_base"):
    pending = pending.json() if hasattr(pending, "json") else pending
    command = next(item for item in platform.commands(node)
                   if item["type"] == "PREFLIGHT" and item["payload"]["preview_id"] == pending["id"])
    payload = command["payload"]
    code = snapshot(recipe=recipe, commit_sha="b" * 40, changes_digest="c" * 64,
                    module_origin="/machine/code/src/openpi/__init__.py", runtime_digest="d" * 64)
    if definition == "expected":
        code["recipe_definition"] = payload["recipe_definition"]
    elif definition is not None:
        code["recipe_definition"] = definition
    code["automatic_preflight"] = automatic_preflight(payload)
    response = platform.client.post(f"/agent/v1/commands/{command['id']}/ack", headers=node["headers"], json={
        "node_id": node["node_id"], "status": "APPLIED", "result": {
            "checks": automatic_checks(), "code_snapshot": code,
        },
    })
    assert response.status_code == 200, response.text
    return platform.client.get(f"/api/v1/run-previews/{pending['id']}", headers=ADMIN).json(), command


def submit(platform, checked):
    return platform.client.post("/api/v1/runs", headers={**ADMIN, "Idempotency-Key": uid()}, json=confirmed_run_request(checked))


def test_saved_recipe_reaches_agent_without_manual_approval(platform):
    node, saved, inputs = prepared(platform)
    for metadata in ({}, {"validated": True}, {"validated": True, "validated_recipes": ["pi0_base"]},
                     {"validated_recipe_revisions": [saved["latest_revision"]["id"]]},
                     {"validated_recipe_revisions": ["wrong-digest"]}):
        with platform.app.state.db.write() as session:
            session.get(Profile, (node["node_id"], "default")).metadata_json = metadata
        response = preview(platform, node, saved, inputs)
        assert response.status_code == 202 and response.json()["status"] == "PENDING"
        with platform.app.state.db.read() as session:
            assert session.get(Profile, (node["node_id"], "default")).metadata_json == metadata


def test_frozen_definition_is_in_preflight_request_ack_and_run(platform):
    node, saved, inputs = prepared(platform)
    approve(platform, node, saved)
    checked, command = acknowledge_preview(platform, node, preview(platform, node, saved, inputs))
    definition = saved["latest_revision"]["definition"]
    assert command["payload"]["recipe_definition"] == definition
    assert command["payload"]["node_id"] == node["node_id"]
    assert checked["status"] == "READY"
    assert checked["request"]["recipe_definition"] == definition
    assert checked["request"]["recipe_revision_id"] == definition["id"]
    assert checked["code_snapshot"]["recipe_definition"] == definition
    response = submit(platform, checked)
    assert response.status_code == 201, response.text
    run = response.json()
    assert run["recipe"] == "pi0_base" and run["recipe_definition"] == definition
    assert run["code_expectation"]["recipe_definition"] == definition


@pytest.mark.parametrize("definition", [None, {}])
def test_missing_or_wrong_definition_from_old_agent_fails_preflight(platform, definition):
    node, saved, inputs = prepared(platform)
    approve(platform, node, saved)
    checked, _ = acknowledge_preview(platform, node, preview(platform, node, saved, inputs), definition=definition)
    assert checked["status"] == "FAILED" and checked["error"]["code"] == "RECIPE_DEFINITION_MISMATCH"
    assert submit(platform, checked).json()["error"]["code"] == "PREFLIGHT_NOT_READY"


def test_new_revision_does_not_redirect_queued_run_or_start_command(platform):
    node, saved, inputs = prepared(platform)
    approve(platform, node, saved)
    checked, _ = acknowledge_preview(platform, node, preview(platform, node, saved, inputs))
    run = submit(platform, checked).json()
    original = copy.deepcopy(saved["latest_revision"]["definition"])
    second_detail = ready(platform, node)
    revised_response = revise(platform, saved["id"], second_detail["id"], overrides={"batch_size": 64})
    assert revised_response.status_code == 201, revised_response.text
    revised = revised_response.json()
    assert revised["latest_revision"]["revision"] == 2 and revised["validated"] is False
    assert platform.schedule() == 1
    start = next(item for item in platform.commands(node) if item["type"] == "START")
    assert start["payload"]["node_id"] == node["node_id"]
    assert start["payload"]["recipe_definition"] == original
    assert start["payload"]["code_expectation"]["recipe_definition"] == original
    assert start["payload"]["parameters"]["batch_size"] == 16
    current = platform.get_run(run)
    assert current["recipe_definition"] == original and current["recipe_definition"]["revision"] == 1


def test_wrong_node_and_same_name_different_profile_cannot_use_revision(platform):
    node, saved, inputs = prepared(platform)
    approve(platform, node, saved)
    other = register(platform, name="other")
    with platform.app.state.db.write() as session:
        session.get(Profile, (other["node_id"], "default")).metadata_json = {
            "validated_recipe_revisions": [saved["latest_revision"]["definition"]["digest"]],
        }
        session.add(Profile(node_id=node["node_id"], id="second", name="second", code_dir="/machine/code",
                            python="/usr/bin/python3", adapter="openpi-sft", recipe="pi0_base", active=True,
                            metadata_json={"validated_recipe_revisions": [saved["latest_revision"]["definition"]["digest"]]}))
    for target, changes in ((other, {}), (node, {"profile_id": "second"})):
        response = preview(platform, target, saved, inputs, **changes)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "RECIPE_REVISION_SCOPE_MISMATCH"


def test_revision_base_name_cannot_be_changed_by_task_request(platform):
    node, saved, inputs = prepared(platform)
    approve(platform, node, saved)
    response = preview(platform, node, saved, inputs, recipe="pi0_different")
    assert response.status_code == 422 and response.json()["error"]["code"] == "RECIPE_DEFINITION_MISMATCH"


def test_explicit_task_parameters_override_recipe_values_only_for_that_run(platform):
    node, saved, inputs = prepared(platform)
    approve(platform, node, saved)
    checked, _ = acknowledge_preview(platform, node, preview(platform, node, saved, inputs,
                                                            parameters={"steps": 3, "batch_size": 8}))
    assert checked["request"]["parameters"] == {"steps": 3, "batch_size": 8, "learning_rate": 0.00002}
    run = submit(platform, checked).json()
    assert run["parameters"] == checked["request"]["parameters"]
    definition = run["recipe_definition"]
    assert definition["overrides"]["num_train_steps"] == 100 and definition["overrides"]["batch_size"] == 16
    another = preview(platform, node, saved, inputs)
    assert another.status_code == 202
    another_checked, _ = acknowledge_preview(platform, node, another)
    assert another_checked["request"]["parameters"] == {"steps": 100, "batch_size": 16, "learning_rate": 0.00002}


def test_legacy_approval_removal_does_not_replace_automatic_evidence(platform):
    node, saved, inputs = prepared(platform)
    approve(platform, node, saved)
    checked, _ = acknowledge_preview(platform, node, preview(platform, node, saved, inputs))
    approve(platform, node)
    response = submit(platform, checked)
    assert response.status_code == 201, response.text


@pytest.mark.parametrize("reported", [None, {}])
def test_started_event_with_missing_or_changed_definition_is_cancelled(platform, reported):
    node, saved, inputs = prepared(platform)
    approve(platform, node, saved)
    checked, _ = acknowledge_preview(platform, node, preview(platform, node, saved, inputs))
    run = submit(platform, checked).json()
    assert platform.schedule() == 1
    run = platform.get_run(run)
    attempt = run["attempts"][0]
    code = copy.deepcopy(run["code_expectation"])
    if reported is None:
        code.pop("recipe_definition")
    else:
        code["recipe_definition"] = reported
    response = platform.emit(node, attempt["id"], 1, "STARTED", {"pid": 1234, "code_snapshot": code,
                                                                "output_dir": "/machine/output/" + attempt["id"]})
    assert response.status_code == 200, response.text
    current = platform.get_run(run)
    assert current["state"] == "CANCELLING" and current["error"]["code"] == "RECIPE_DEFINITION_MISMATCH"
    assert any(command["type"] == "CANCEL" for command in platform.commands(node))


def test_started_event_accepts_exact_frozen_revision(platform):
    node, saved, inputs = prepared(platform)
    approve(platform, node, saved)
    checked, _ = acknowledge_preview(platform, node, preview(platform, node, saved, inputs))
    run = submit(platform, checked).json()
    platform.schedule()
    platform.start(node, run)
    current = platform.get_run(run)
    assert current["state"] == "RUNNING"
    assert current["attempts"][0]["code_snapshot"]["recipe_definition"] == saved["latest_revision"]["definition"]
