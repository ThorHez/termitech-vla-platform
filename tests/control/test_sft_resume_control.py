"""True SFT resume uses a separately approved run and an immutable checkpoint source."""

from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace

import pytest

from vla_control.common import APIError, now
from vla_control.models import Asset, Attempt, Node, Run
from vla_control.training_resume import freeze_resume_source
from vla_protocol import canonical_digest

from .conftest import ADMIN, automatic_checks, automatic_preflight, confirmed_run_request, snapshot
from .test_task_recipe import inputs


def create_source(platform, state, step=2):
    node = platform.register(adapter="openpi-sft")
    original = platform.run(node, template="openpi-sft", inputs=inputs(platform, node))
    assert platform.schedule() == 1
    attempt = platform.get_run(original)["attempts"][0]
    output_dir = f"/machine/output/{attempt['id']}"
    execution = {key: value for key, value in original["code_expectation"].items()
                 if key not in {"confirmed_config_sha256", "config_confirmed_at"}}
    execution["captured_at"] = now()
    assert platform.emit(node, attempt["id"], 1, "STARTED", {"pid": 1234, "code_snapshot": execution,
        "output_dir": output_dir}).status_code == 200
    path = f"{output_dir}/checkpoints/{original['recipe']}/platform/{step}"
    output = {"kind": "MODEL", "name": "checkpoint", "version": str(step), "path": path,
              "metadata": {"format": "openpi-orbax", "step": step}}
    if state == "SUCCEEDED":
        completed = platform.emit(node, attempt["id"], 2, state, {
            "exit_code": 0, "code_changed_during_run": False, "outputs": [output]})
    else:
        assert platform.emit(node, attempt["id"], 2, "ARTIFACT", {"outputs": [output]}).status_code == 200
        completed = platform.emit(node, attempt["id"], 3, state, {
            "exit_code": 1 if state == "FAILED" else -15, "code_changed_during_run": False})
    assert completed.status_code == 200, completed.text
    original = platform.get_run(original)
    asset = original["outputs"][0]
    descriptor = {"asset_id": asset["id"], "run_id": original["id"], "attempt_id": attempt["id"],
                  "path": path, "output_dir": output_dir, "step": step}
    request = {"name": "continuation", "node_id": node["node_id"], "profile_id": original["profile_id"],
               "template_id": "openpi-sft", "recipe": original["recipe"], "resume": True,
               "gpu_count": original["gpu_count"], "parameters": {**original["parameters"], "steps": 8},
               "inputs": {**original["inputs"], "model": asset["id"]}}
    return node, original, descriptor, request


@pytest.fixture
def source(platform):
    return create_source(platform, "SUCCEEDED")


def preview(platform, node, request, *, echo=True, source_plan_drift=False):
    response = platform.client.post("/api/v1/run-previews", headers=ADMIN, json=request)
    assert response.status_code == 202, response.text
    identifier = response.json()["id"]
    command = next(row for row in platform.commands(node)
                   if row["type"] == "PREFLIGHT" and row["payload"]["preview_id"] == identifier)
    payload = command["payload"]
    evidence = automatic_preflight(payload)
    checks = automatic_checks()
    if echo:
        evidence["plan"]["resume_source"] = payload["resume_source"]
        original = platform.get_run({"id": payload["resume_source"]["run_id"]})
        evidence["plan"]["resume_checkpoint"] = {
            "checkpoint_metadata_sha256": "a" * 64, "params_manifest_digest": "b" * 64,
            "train_state_metadata_sha256": "c" * 64, "train_state_manifest_digest": "d" * 64,
            "source_effective_config_sha256": "e" * 64, "commit_timestamp_nsecs": 42, "train_state_bytes": 400,
            "source_execution_plan_digest": original["code_expectation"]["automatic_preflight"]["plan_digest"],
        }
        if source_plan_drift:
            evidence["plan"]["resume_checkpoint"]["source_execution_plan_digest"] = "f" * 64
        generated = evidence["plan"]["generated_config"]
        generated["source"] += "FROZEN_REQUEST = " + repr({"resume_source": payload["resume_source"]}) + "\n"
        generated["sha256"] = sha256(generated["source"].encode()).hexdigest()
        checks.append({"name": "resume_checkpoint", "ok": True, "message": "checked"})
        evidence["plan_digest"] = canonical_digest(evidence["plan"])
    code = snapshot(recipe=payload["recipe"], commit_sha="b" * 40, automatic_preflight=evidence)
    ack = platform.client.post(f"/agent/v1/commands/{command['id']}/ack", headers=node["headers"], json={
        "node_id": node["node_id"], "status": "APPLIED", "result": {"checks": checks, "code_snapshot": code}})
    assert ack.status_code == 200, ack.text
    return platform.client.get(f"/api/v1/run-previews/{identifier}", headers=ADMIN).json(), payload


def submit(platform, checked):
    return platform.client.post("/api/v1/runs", headers={**ADMIN, "Idempotency-Key": checked["id"]},
                                json=confirmed_run_request(checked))


def test_resume_is_confirmed_dispatched_and_recorded_as_a_new_run(platform, source):
    node, original, descriptor, request = source
    checked, command = preview(platform, node, request)
    assert checked["status"] == "READY" and command["resume_source"] == descriptor
    created = submit(platform, checked)
    assert created.status_code == 201, created.text
    run = created.json()
    assert run["id"] != original["id"] and run["resume"] is True
    assert run["resume_source"] == run["code_expectation"]["resume_source"] == descriptor
    assert run["inputs"]["model"] == descriptor["asset_id"] and run["parameters"]["steps"] == 8
    archived = platform.client.post(f"/api/v1/assets/{descriptor['asset_id']}/archive", headers=ADMIN)
    assert archived.status_code == 409 and archived.json()["error"]["code"] == "ASSET_IN_USE"
    assert platform.schedule() == 1
    start = next(row["payload"] for row in platform.commands(node)
                 if row["type"] == "START" and row["payload"]["run_id"] == run["id"])
    assert start["resume_source"] == start["code_expectation"]["automatic_preflight"]["plan"]["resume_source"] == descriptor
    assert platform.get_run(original) == original


@pytest.mark.parametrize("state", ["FAILED", "CANCELLED"])
def test_committed_checkpoint_survives_failed_or_cancelled_source(platform, state):
    node, original, descriptor, request = create_source(platform, state)
    checked, command = preview(platform, node, request)
    assert original["state"] == state
    assert checked["status"] == "READY" and command["resume_source"] == descriptor


def test_committed_step_zero_is_valid_if_total_advances_beyond_restored_state(platform):
    node, _, descriptor, request = create_source(platform, "CANCELLED", step=0)
    request["parameters"]["steps"] = 2
    checked, command = preview(platform, node, request)
    assert checked["status"] == "READY" and command["resume_source"] == descriptor
    request["parameters"]["steps"] = 1
    response = platform.client.post("/api/v1/run-previews", headers=ADMIN, json=request)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "SFT_RESUME_STEPS_INVALID"


def test_old_agent_cannot_accept_resume_as_an_ordinary_warm_start(platform, source):
    node, _, _, request = source
    checked, _ = preview(platform, node, request, echo=False)
    assert checked["status"] == "FAILED"
    assert checked["error"]["code"] == "SFT_RESUME_PLAN_MISMATCH"
    assert submit(platform, checked).status_code == 409


def test_checkpoint_proof_must_match_original_approved_execution_plan(platform, source):
    node, _, _, request = source
    checked, _ = preview(platform, node, request, source_plan_drift=True)
    assert checked["status"] == "FAILED"
    assert checked["error"]["code"] == "SFT_RESUME_PLAN_MISMATCH"


@pytest.mark.parametrize("changes,code", [
    ({"entrypoint": "scripts/train.py"}, "SFT_RESUME_UNSUPPORTED"),
    ({"recipe": "another_recipe"}, "SFT_RESUME_CONFIG_MISMATCH"),
    ({"parameters": {"steps": 8, "learning_rate": 0.00001}}, "SFT_RESUME_CONFIG_MISMATCH"),
    ({"parameters": {"steps": 3}}, "SFT_RESUME_STEPS_INVALID"),
])
def test_resume_does_not_accept_mutable_training_settings(platform, source, changes, code):
    _, _, _, original = source
    request = deepcopy(original)
    for key, value in changes.items():
        if key == "parameters":
            request[key].update(value)
        else:
            request[key] = value
    response = platform.client.post("/api/v1/run-previews", headers=ADMIN, json=request)
    assert response.status_code in {409, 422}, response.text
    assert response.json()["error"]["code"] == code


def test_unowned_resume_source_is_not_adoptable_by_an_operator(platform, source):
    node, _, _, request = source
    operator = SimpleNamespace(state=SimpleNamespace(principal=SimpleNamespace(role="operator", user_id="operator", legacy=False)))
    with platform.app.state.db.read() as session, pytest.raises(APIError) as failure:
        freeze_resume_source(session, session.get(Node, node["node_id"]), deepcopy(request), request["parameters"], operator)
    assert failure.value.code == "RESOURCE_ACCESS_DENIED"


@pytest.mark.parametrize("moment", ["submission", "scheduler"])
def test_deleted_source_is_rejected_at_later_lifecycle_boundaries(platform, source, moment):
    node, original, _, request = source
    checked, _ = preview(platform, node, request)
    if moment == "scheduler":
        created = submit(platform, checked)
        assert created.status_code == 201, created.text
    with platform.app.state.db.write() as session:
        session.get(Run, original["id"]).deleted_at = now()
    if moment == "submission":
        rejected = submit(platform, checked)
        assert rejected.status_code == 409
        assert rejected.json()["error"]["code"] == "SFT_RESUME_SOURCE_UNAVAILABLE"
    else:
        assert platform.schedule() == 0
        run = platform.get_run(created.json())
        assert run["state"] == "FAILED" and run["attempts"] == []
        assert run["error"]["code"] == "SFT_RESUME_SOURCE_UNAVAILABLE"


@pytest.mark.parametrize("mutation", ["unknown_lineage", "running_attempt", "code_changed", "inference_only"])
def test_invalid_checkpoint_lineage_never_creates_resume_preview(platform, source, mutation):
    _, _, descriptor, request = source
    with platform.app.state.db.write() as session:
        asset, attempt = session.get(Asset, descriptor["asset_id"]), session.get(Attempt, descriptor["attempt_id"])
        if mutation == "unknown_lineage":
            asset.origin_attempt_id = None
        elif mutation == "running_attempt":
            attempt.state = "RUNNING"
        elif mutation == "code_changed":
            attempt.code_changed_during_run = True
        else:
            asset.metadata_json = {"format": "safetensors", "step": 2}
    response = platform.client.post("/api/v1/run-previews", headers=ADMIN, json=request)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "SFT_RESUME_SOURCE_UNAVAILABLE"
