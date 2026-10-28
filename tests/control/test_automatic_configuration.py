"""Automatically assembled configurations replace named recipes in the normal flow."""

import copy

import pytest
from vla_protocol import canonical_digest

from .conftest import ADMIN, automatic_preflight, confirmed_run_request, snapshot
from .test_automatic_preflight import acknowledge
from .test_multi_dataset_control import configured
from .test_task_recipe import request_preview


AUTO_CONFIG = {"model_type": "pi05", "finetune_mode": "full", "action_horizon": 50, "action_dim": 32}


def automatic_request(platform, **changes):
    node = platform.register(adapter="openpi-sft", recipe=None, metadata={})
    _, inputs, rows = configured(platform, node)
    for row in rows:
        row.pop("recipe_index")
        row["processor"] = {"name": "LeRobotUmiDataConfigPadded_V4", "parameters": {}}
    request = {"inputs": inputs, "dataset_bindings": rows, "auto_config": copy.deepcopy(AUTO_CONFIG), **changes}
    return node, request


def pending(platform):
    node, request = automatic_request(platform)
    response = request_preview(platform, node, **request)
    assert response.status_code == 202, response.text
    identifier = response.json()["id"]
    command = next(item for item in platform.commands(node) if item["type"] == "PREFLIGHT" and item["payload"]["preview_id"] == identifier)
    payload = command["payload"]
    code = snapshot(recipe=payload["recipe"], auto_config=payload["auto_config"], dataset_bindings=payload["dataset_bindings"],
                    automatic_preflight=automatic_preflight(payload))
    return node, identifier, command, code


def test_no_named_recipe_needed_for_automatic_preflight_or_training(platform):
    node, identifier, command, code = pending(platform)
    assert command["payload"]["auto_config"] == AUTO_CONFIG
    assert command["payload"]["recipe"]
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "READY", checked
    assert checked["request"]["auto_config"] == AUTO_CONFIG
    response = platform.client.post("/api/v1/runs", headers={**ADMIN, "Idempotency-Key": identifier}, json=confirmed_run_request(checked))
    assert response.status_code == 201, response.text
    run = response.json()
    assert run["auto_config"] == AUTO_CONFIG
    assert run["code_expectation"]["auto_config"] == AUTO_CONFIG
    assert platform.schedule() == 1
    start = next(item for item in platform.commands(node) if item["type"] == "START")
    assert start["payload"]["auto_config"] == AUTO_CONFIG
    assert start["payload"]["recipe"] == command["payload"]["recipe"]
    assert start["payload"]["confirmed_config_sha256"] == checked["code_snapshot"]["automatic_preflight"]["plan"]["generated_config"]["sha256"]


@pytest.mark.parametrize("field,value", [("recipe", "existing_code_recipe"), ("recipe_revision_id", "some-saved-revision")])
def test_automatic_configuration_cannot_mix_with_code_recipe(platform, field, value):
    node, request = automatic_request(platform, **{field: value})
    response = request_preview(platform, node, **request)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "AUTO_CONFIG_CONFLICT"
    assert platform.commands(node) == []


def test_automatic_configuration_requires_explicit_processor_for_every_dataset(platform):
    node, request = automatic_request(platform)
    request["dataset_bindings"][-1].pop("processor")
    request["dataset_bindings"][-1]["recipe_index"] = 0
    response = request_preview(platform, node, **request)
    assert response.status_code == 422 and response.json()["error"]["code"] == "AUTO_CONFIG_PROCESSOR_REQUIRED"
    assert platform.commands(node) == []


def test_automatic_configuration_is_unavailable_for_smoke(platform):
    node = platform.register()
    response = request_preview(platform, node, template_id="smoke", auto_config=AUTO_CONFIG, gpu_count=0)
    assert response.status_code == 422 and response.json()["error"]["code"] == "AUTO_CONFIG_UNSUPPORTED"
    assert platform.commands(node) == []


@pytest.mark.parametrize("field,value", [("model_type", "unknown"), ("finetune_mode", "arbitrary"),
                                         ("action_horizon", 0), ("action_horizon", True),
                                         ("action_dim", -1), ("action_dim", "32")])
def test_invalid_auto_model_settings_never_dispatch(platform, field, value):
    node, request = automatic_request(platform)
    request["auto_config"][field] = value
    response = request_preview(platform, node, **request)
    assert response.status_code == 422
    assert platform.commands(node) == []


@pytest.mark.parametrize("where", ["snapshot", "plan"])
def test_agent_must_echo_the_exact_automatic_model_configuration(platform, where):
    node, identifier, command, code = pending(platform)
    if where == "snapshot":
        code["auto_config"] = {**AUTO_CONFIG, "finetune_mode": "lora"}
    else:
        evidence = code["automatic_preflight"]
        evidence["plan"]["auto_config"] = {**AUTO_CONFIG, "action_horizon": 25}
        evidence["plan_digest"] = canonical_digest(evidence["plan"])
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "FAILED"
    assert checked["error"]["code"] in {"AUTO_CONFIG_MISMATCH", "AGENT_PREFLIGHT_MISMATCH"}


def test_old_agent_without_automatic_configuration_echo_is_rejected(platform):
    node, identifier, command, code = pending(platform)
    code.pop("auto_config")
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "FAILED" and checked["error"]["code"] == "AGENT_AUTO_CONFIG_UNSUPPORTED"
