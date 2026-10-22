"""Training recipes are frozen per task and checked automatically by the Agent."""

import pytest

from vla_control.models import Preview, Profile, Run

from .conftest import ADMIN, snapshot


def body(node, **changes):
    return {"name": "recipe task", "node_id": node["node_id"], "profile_id": "default",
            "template_id": "openpi-sft", "gpu_count": 1, **changes}


def inputs(platform, node):
    result = {}
    for role, kind in (("dataset", "DATASET"), ("model", "MODEL"), ("norm", "NORM")):
        response = platform.client.post("/api/v1/assets", headers=ADMIN, json={
            "name": role, "kind": kind, "node_id": node["node_id"], "path": "/machine/" + role})
        assert response.status_code == 201
        result[role] = response.json()["id"]
    return result


def request_preview(platform, node, **changes):
    return platform.client.post("/api/v1/run-previews", headers=ADMIN, json=body(node, **changes))


def ack_preview(platform, node, identifier, code):
    command = next(command for command in platform.commands(node)
                   if command["type"] == "PREFLIGHT" and command["payload"]["preview_id"] == identifier)
    response = platform.client.post(f"/agent/v1/commands/{command['id']}/ack", headers=node["headers"], json={
        "node_id": node["node_id"], "status": "APPLIED", "result": {
            "checks": [{"name": "runtime and recipe", "ok": True}], "code_snapshot": code}})
    assert response.status_code == 200, response.text
    return platform.client.get(f"/api/v1/run-previews/{identifier}", headers=ADMIN).json(), command


def test_environment_without_default_recipe_registers_but_training_needs_selection(platform):
    node = platform.register(adapter="openpi-sft", recipe=None, metadata={"validated_recipes": ["sft_a"]})
    response = request_preview(platform, node)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "RECIPE_REQUIRED"
    assert "选择训练配方" in response.json()["error"]["message"]
    # Registering the environment did not need a default or pretend to validate it.
    assert platform.client.get("/api/v1/profiles", headers=ADMIN).json()[0]["recipe"] is None


def test_selected_recipe_is_frozen_into_preflight_run_and_start_command(platform):
    node = platform.register(adapter="openpi-sft", recipe=None, metadata={"validated_recipes": ["sft_a", "sft_b"]})
    local_inputs = inputs(platform, node)
    preview = platform.preview(node, inputs=local_inputs, template="openpi-sft", recipe="sft_b")
    assert preview["status"] == "READY"
    assert preview["request"]["recipe"] == preview["code_snapshot"]["recipe"] == "sft_b"
    response = platform.client.post("/api/v1/runs", headers={**ADMIN, "Idempotency-Key": "sft-b"},
                                    json={"preflight_id": preview["id"]})
    assert response.status_code == 201, response.text
    run = response.json()
    assert run["recipe"] == run["code_expectation"]["recipe"] == "sft_b"
    assert platform.schedule() == 1
    start = next(c for c in platform.commands(node) if c["type"] == "START")
    assert start["payload"]["recipe"] == "sft_b"
    assert start["payload"]["code_expectation"]["recipe"] == "sft_b"


@pytest.mark.parametrize("code_recipe,error", [(None, "AGENT_RECIPE_UNSUPPORTED"), ("sft_b", "RECIPE_MISMATCH")])
def test_agent_must_echo_the_selected_preflight_recipe(platform, code_recipe, error):
    node = platform.register(adapter="openpi-sft", recipe=None, metadata={"validated_recipes": ["sft_a"]})
    response = request_preview(platform, node, inputs=inputs(platform, node), recipe="sft_a")
    assert response.status_code == 202
    code = snapshot(recipe=code_recipe)
    if code_recipe is None:
        del code["recipe"]
    preview, command = ack_preview(platform, node, response.json()["id"], code)
    assert command["payload"]["recipe"] == "sft_a"
    assert preview["status"] == "FAILED"
    assert preview["error"]["code"] == error
    assert "更新 Agent" in preview["error"]["message"]


def test_task_override_reaches_agent_without_legacy_profile_validation(platform):
    node = platform.register(adapter="openpi-sft", recipe="sft_a", metadata={"validated": True})
    response = request_preview(platform, node, inputs=inputs(platform, node), recipe="sft_b")
    assert response.status_code == 202 and response.json()["status"] == "PENDING"
    preview = platform.preview(node, inputs=inputs(platform, node), template="openpi-sft")
    assert preview["status"] == "READY" and preview["request"]["recipe"] == "sft_a"


def test_explicit_recipe_needs_no_allowlist_entry(platform):
    node = platform.register(adapter="openpi-sft", recipe=None, metadata={"validated_recipes": ["sft_a", "sft_b"]})
    local_inputs = inputs(platform, node)
    response = request_preview(platform, node, inputs=local_inputs, recipe="sft_c")
    assert response.status_code == 202 and response.json()["status"] == "PENDING"
    for recipe in ("sft_a", "sft_b", "sft_c"):
        preview = platform.preview(node, inputs=local_inputs, template="openpi-sft", recipe=recipe)
        assert preview["status"] == "READY" and preview["request"]["recipe"] == recipe


def test_submission_uses_automatic_evidence_instead_of_legacy_allowlist(platform):
    node = platform.register(adapter="openpi-sft", recipe=None, metadata={"validated_recipes": ["sft_a"]})
    preview = platform.preview(node, inputs=inputs(platform, node), template="openpi-sft", recipe="sft_a")
    with platform.app.state.db.write() as session:
        session.get(Profile, (node["node_id"], "default")).metadata_json = {"validated_recipes": []}
    response = platform.client.post("/api/v1/runs", headers={**ADMIN, "Idempotency-Key": "revoked"},
                                    json={"preflight_id": preview["id"]})
    assert response.status_code == 201, response.text


def test_wrong_started_recipe_is_stopped_without_releasing_reservations(platform):
    node = platform.register()
    run = platform.run(node)
    platform.schedule()
    attempt = platform.get_run(run)["attempts"][0]
    response = platform.emit(node, attempt["id"], 1, "STARTED", {"pid": 42,
        "code_snapshot": {**run["code_expectation"], "recipe": "another_recipe"}, "output_dir": "/machine/output/run"})
    assert response.status_code == 200
    result = platform.get_run(run)
    assert result["state"] == "CANCELLING" and result["error"]["code"] == "RECIPE_MISMATCH"
    assert any(c["type"] == "CANCEL" for c in platform.commands(node))
    resource = platform.client.get(f"/api/v1/nodes/{node['node_id']}", headers=ADMIN).json()
    assert resource["gpus"][0]["reserved_by"] == attempt["id"]


def test_legacy_run_without_recipe_remains_readable_and_uses_legacy_payload(platform):
    node = platform.register()
    run = platform.run(node)
    with platform.app.state.db.write() as session:
        stored = session.get(Run, run["id"])
        stored.code_expectation = {key: value for key, value in stored.code_expectation.items() if key != "recipe"}
    assert platform.get_run(run)["recipe"] is None
    assert platform.schedule() == 1
    command = next(c for c in platform.commands(node) if c["type"] == "START")
    assert "recipe" not in command["payload"]
    attempt = platform.get_run(run)["attempts"][0]
    response = platform.emit(node, attempt["id"], 1, "STARTED", {"pid": 42,
        "code_snapshot": {**command["payload"]["code_expectation"], "recipe": "smoke"},
        "output_dir": "/machine/output/legacy"})
    assert response.status_code == 200
    assert platform.get_run(run)["state"] == "RUNNING"


@pytest.mark.parametrize("snapshot_has_recipe", [True, False])
def test_legacy_ready_preview_requires_new_recipe_aware_preflight(platform, snapshot_has_recipe):
    node = platform.register()
    preview = platform.preview(node)
    with platform.app.state.db.write() as session:
        stored = session.get(Preview, preview["id"])
        stored.request = {key: value for key, value in stored.request.items() if key != "recipe"}
        if not snapshot_has_recipe:
            stored.code_snapshot = {key: value for key, value in stored.code_snapshot.items() if key != "recipe"}
    response = platform.client.post("/api/v1/runs", headers={**ADMIN, "Idempotency-Key": "old-preview"},
                                    json={"preflight_id": preview["id"]})
    assert response.status_code == 409 and response.json()["error"]["code"] == "AGENT_RECIPE_UNSUPPORTED"


@pytest.mark.parametrize("recipe", ["path/to/config", "name;command", "recipe.name", "x" * 256])
def test_task_recipe_accepts_only_named_configurations(platform, recipe):
    node = platform.register(adapter="openpi-sft")
    response = request_preview(platform, node, recipe=recipe)
    assert response.status_code == 422 and response.json()["error"]["code"] == "VALIDATION_ERROR"
