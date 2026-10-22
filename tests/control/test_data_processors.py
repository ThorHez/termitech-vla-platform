"""Processor catalogs are bounded discovery, and chosen processors are frozen."""

import copy
import json

import pytest
from vla_protocol import canonical_digest

from vla_control.common import offset_time
from vla_control.models import Command, Node, Profile

from .conftest import ADMIN, automatic_preflight, snapshot
from .test_automatic_preflight import acknowledge, submit
from .test_multi_dataset_control import configured
from .test_recipe_details import register
from .test_task_recipe import request_preview


def create(platform, node, headers=ADMIN, **changes):
    return platform.client.post(f"/api/v1/nodes/{node['node_id']}/data-processors", headers=headers,
                                json={"profile_id": "default", **changes})


def get(platform, node, query, headers=ADMIN):
    return platform.client.get(f"/api/v1/nodes/{node['node_id']}/data-processors/{query['id']}", headers=headers)


def result():
    return {"profile_id": "default", "source": {"code_dir": "/machine/code", "path": "src/openpi/training/config.py",
            "sha256": "a" * 64, "commit_sha": "b" * 40, "changes_digest": "c" * 64, "dirty": False},
            "entries": [{"name": "IndependentUmiData", "description": "独立数据处理器", "available": True,
                         "source": {"path": "src/openpi/training/config.py", "sha256": "a" * 64},
                         "parameters": [{"path": "image_horizon", "type": "integer", "default": 1, "required": False},
                                        {"path": "base_config.prompt_from_task", "type": "boolean", "default": True, "required": False},
                                        {"path": "base_config.action_sequence_keys", "type": "array", "default": [], "required": False,
                                         "items": {"type": "string"}, "container": "tuple"}],
                         "unsupported_parameters": [{"path": "data_transforms", "reason": "保留 Python 默认构造器", "required": False}]}]}


def ack(platform, node, query, value=None, status="APPLIED"):
    return platform.client.post(f"/agent/v1/commands/{query['id']}/ack", headers=node["headers"], json={
        "node_id": node["node_id"], "status": status, "result": result() if value is None else value})


def test_catalog_scope_cache_sanitized_ack_and_no_training_acceptance(platform):
    node = register(platform)
    response = create(platform, node)
    assert response.status_code == 202 and response.headers["Cache-Control"] == "no-store"
    query = response.json()
    assert query["status"] == "PENDING" and query["catalog"] is None
    assert create(platform, node).json()["id"] == query["id"]
    command = platform.commands(node)[0]
    assert command["type"] == "DATA_PROCESSORS" and command["payload"]["code_dir"] == "/machine/code"
    raw = result()
    raw["private_output"] = "not visible"
    raw["entries"][0]["execute"] = "/bin/sh"
    assert ack(platform, node, query, raw).status_code == 200
    ready = get(platform, node, query).json()
    assert ready["status"] == "READY" and ready["catalog"] == result()
    assert create(platform, node).json() == ready
    assert ack(platform, node, query, raw).status_code == 200
    with platform.app.state.db.read() as session:
        assert session.get(Profile, (node["node_id"], "default")).metadata_json == {}
        assert session.get(Command, query["id"]).result == result()


def test_catalog_auth_scope_and_unavailable_environment(platform):
    node, other = register(platform), register(platform, name="other")
    query = create(platform, node).json()
    for headers in ({}, node["headers"]):
        assert create(platform, node, headers=headers).status_code == 401
        assert get(platform, node, query, headers=headers).status_code == 401
    assert get(platform, other, query).status_code == 404
    assert ack(platform, other, query).status_code == 403
    assert create(platform, node, profile_id="missing").status_code == 404
    assert create(platform, node, source={}).status_code == 422
    with platform.app.state.db.write() as session:
        session.get(Node, node["node_id"]).last_seen = offset_time(-1000)
    assert create(platform, node).status_code == 409


@pytest.mark.parametrize("mutate", [
    lambda value: value.update(profile_id="other"),
    lambda value: value["source"].update(path="../config.py"),
    lambda value: value["source"].update(code_dir="/outside/code"),
    lambda value: value["source"].update(sha256="invalid"),
    lambda value: value["entries"][0].update(name="__import__('os')"),
    lambda value: value["entries"][0]["source"].update(path="/outside/processor.py"),
    lambda value: value["entries"][0]["parameters"][0].update(path="repo_id"),
    lambda value: value["entries"][0]["parameters"][0].update(default="x" * 10000),
    lambda value: value["entries"].append(copy.deepcopy(value["entries"][0])),
])
def test_catalog_rejects_unbounded_resource_or_source_descriptors(platform, mutate):
    node = register(platform)
    query = create(platform, node).json()
    raw = result()
    mutate(raw)
    assert ack(platform, node, query, raw).status_code == 200
    checked = get(platform, node, query).json()
    assert checked["status"] == "FAILED" and checked["error"]["code"] == "DATA_PROCESSORS_INVALID_RESULT"
    assert checked["catalog"] is None


def test_timeout_and_profile_change_discard_late_catalogs(platform):
    node = register(platform)
    query = create(platform, node).json()
    with platform.app.state.db.write() as session:
        command = session.get(Command, query["id"])
        command.payload = {**command.payload, "expires_at": offset_time(-1)}
    assert get(platform, node, query).json()["error"]["code"] == "DATA_PROCESSORS_TIMEOUT"
    assert ack(platform, node, query).status_code == 200
    assert get(platform, node, query).json()["error"]["code"] == "DATA_PROCESSORS_TIMEOUT"
    next_query = create(platform, node).json()
    assert next_query["id"] != query["id"]
    with platform.app.state.db.write() as session:
        session.get(Profile, (node["node_id"], "default")).python = "/machine/new/python"
    assert ack(platform, node, next_query).status_code == 200
    assert get(platform, node, next_query).json()["error"]["code"] == "DATA_PROCESSORS_PROFILE_CHANGED"


def test_old_agent_error_is_sanitized_and_never_empty_success(platform):
    node = register(platform)
    query = create(platform, node).json()
    assert ack(platform, node, query, {"error": {"code": "COMMAND_UNSUPPORTED", "message": "private traceback"}}, "REJECTED").status_code == 200
    checked = get(platform, node, query).json()
    assert checked["status"] == "FAILED" and checked["error"]["code"] == "AGENT_DATA_PROCESSORS_UNSUPPORTED"
    assert "private" not in str(checked)


def pending_processor(platform):
    node, inputs, rows = configured(platform)
    for row in rows:
        row.pop("recipe_index")
        row["processor"] = {"name": "IndependentUmiData", "parameters": {"image_horizon": 2, "base_config.prompt_from_task": False}}
    response = request_preview(platform, node, inputs=inputs, dataset_bindings=rows)
    assert response.status_code == 202, response.text
    identifier = response.json()["id"]
    command = next(item for item in platform.commands(node) if item["type"] == "PREFLIGHT")
    code = snapshot(recipe="mixed", dataset_bindings=rows, automatic_preflight=automatic_preflight(command["payload"]))
    return node, identifier, command, code, rows


def test_processor_selection_is_independent_of_recipe_indices_and_frozen_through_start(platform):
    node, identifier, command, code, rows = pending_processor(platform)
    assert command["payload"]["dataset_bindings"] == rows
    assert acknowledge(platform, node, identifier, command, code)["status"] == "READY"
    run_response = submit(platform, identifier)
    assert run_response.status_code == 201, run_response.text
    run = run_response.json()
    assert run["dataset_bindings"] == rows and all("recipe_index" not in row for row in rows)
    assert platform.schedule() == 1
    start = next(item for item in platform.commands(node) if item["type"] == "START")
    assert start["payload"]["dataset_bindings"] == rows


@pytest.mark.parametrize("mutation,expected", [("missing", "AGENT_DATA_PROCESSOR_UNSUPPORTED"),
    ("name", "DATA_PROCESSOR_MISMATCH"), ("parameter", "DATA_PROCESSOR_MISMATCH"),
    ("factory", "DATA_PROCESSOR_MISMATCH"), ("identity", "DATA_PROCESSOR_MISMATCH")])
def test_effective_plan_must_prove_actual_processor_and_parameters(platform, mutation, expected):
    node, identifier, command, code, _ = pending_processor(platform)
    plan = json.loads(json.dumps(code["automatic_preflight"]["plan"]))
    row = plan["datasets"][-1]
    if mutation == "missing":
        row.pop("processor")
    elif mutation == "name":
        row["processor"]["name"] = "OtherData"
    elif mutation == "parameter":
        row["processor"]["parameters"]["image_horizon"] = 999
    elif mutation == "factory":
        row["factory"] = "OtherData"
    else:
        row["processor_identity"]["source"]["path"] = "../outside.py"
    code["automatic_preflight"] = {**code["automatic_preflight"], "plan": plan, "plan_digest": canonical_digest(plan)}
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "FAILED" and checked["error"]["code"] == expected


@pytest.mark.parametrize("selection", [{}, {"recipe_index": 0, "processor": {"name": "SomeData", "parameters": {}}},
    {"processor": {"name": "bad.name", "parameters": {}}},
    {"processor": {"name": "SomeData", "parameters": {"repo_id": "/hidden"}}},
    {"processor": {"name": "SomeData", "parameters": {"image_horizon": 10 ** 100}}}])
def test_new_bindings_require_one_safe_processor_choice(platform, selection):
    node, inputs, rows = configured(platform)
    rows[0].pop("recipe_index")
    rows[0].update(selection)
    response = request_preview(platform, node, inputs=inputs, dataset_bindings=rows)
    assert response.status_code == 422
    assert not platform.commands(node)
