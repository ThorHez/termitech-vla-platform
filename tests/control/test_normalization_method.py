"""Normalization choices stay immutable through preflight, dispatch and execution."""

import copy
import hashlib

import pytest
from vla_protocol import canonical_digest
from vla_control.models import Preview

from .conftest import ADMIN, automatic_preflight, confirmed_run_request, snapshot
from .test_automatic_configuration import automatic_request
from .test_automatic_preflight import acknowledge, submit
from .test_task_recipe import inputs, request_preview


def pending_choice(platform, method="minmax", *, automatic=False):
    if automatic:
        node, request = automatic_request(platform)
    else:
        node = platform.register(adapter="openpi-sft", recipe="native_recipe", metadata={})
        request = {"inputs": inputs(platform, node)}
    if method is not None:
        request["normalization_method"] = method
    response = request_preview(platform, node, **request)
    assert response.status_code == 202, response.text
    identifier = response.json()["id"]
    command = next(row for row in platform.commands(node)
                   if row["type"] == "PREFLIGHT" and row["payload"]["preview_id"] == identifier)
    payload = command["payload"]
    evidence = automatic_preflight(payload)
    if method is not None:
        evidence["plan"]["normalization_method"] = method
        source = "FROZEN_REQUEST = " + repr({"normalization_method": method}) + "\n"
        evidence["plan"]["generated_config"].update(source=source, sha256=hashlib.sha256(source.encode()).hexdigest())
        for data in evidence["plan"]["datasets"]:
            data["normalization_method"] = method
    evidence["plan_digest"] = canonical_digest(evidence["plan"])
    optional = {key: payload[key] for key in ("auto_config", "dataset_bindings", "normalization_method") if key in payload}
    code = snapshot(recipe=payload["recipe"], automatic_preflight=evidence, **optional)
    return node, identifier, command, code


@pytest.mark.parametrize("method", ["minmax", "quantile", "zscore"])
@pytest.mark.parametrize("automatic", [False, True])
def test_explicit_method_survives_preflight_run_serialization_start_and_receipt(platform, method, automatic):
    node, identifier, command, code = pending_choice(platform, method, automatic=automatic)
    assert command["payload"]["normalization_method"] == method
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "READY", checked
    assert checked["request"]["normalization_method"] == method
    response = submit(platform, identifier)
    assert response.status_code == 201, response.text
    run = response.json()
    assert run["normalization_method"] == run["code_expectation"]["normalization_method"] == method
    assert platform.schedule() == 1
    start = next(row for row in platform.commands(node) if row["type"] == "START")
    assert start["payload"]["normalization_method"] == method
    platform.start(node, run)
    assert platform.get_run(run)["state"] == "RUNNING"


def test_legacy_request_never_gets_an_implicit_normalization_method(platform):
    node, identifier, command, code = pending_choice(platform, None)
    assert "normalization_method" not in command["payload"]
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "READY"
    assert "normalization_method" not in checked["request"]
    run = submit(platform, identifier).json()
    assert "normalization_method" not in run and "normalization_method" not in run["code_expectation"]
    assert platform.schedule() == 1
    start = next(row for row in platform.commands(node) if row["type"] == "START")
    assert "normalization_method" not in start["payload"]


@pytest.mark.parametrize("location,mutation", [("snapshot", "missing"), ("plan", "missing"),
                                              ("snapshot", "wrong"), ("plan", "wrong")])
def test_agent_must_confirm_the_exact_choice_in_snapshot_and_plan(platform, location, mutation):
    node, identifier, command, code = pending_choice(platform)
    target = code if location == "snapshot" else code["automatic_preflight"]["plan"]
    if mutation == "missing": target.pop("normalization_method")
    else: target["normalization_method"] = "quantile"
    code["automatic_preflight"]["plan_digest"] = canonical_digest(code["automatic_preflight"]["plan"])
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "FAILED"
    assert checked["error"]["code"] == ("AGENT_NORMALIZATION_UNSUPPORTED" if mutation == "missing" else "NORMALIZATION_METHOD_MISMATCH")


def test_changed_method_invalidates_a_previously_ready_preview(platform):
    node, identifier, command, code = pending_choice(platform)
    assert acknowledge(platform, node, identifier, command, code)["status"] == "READY"
    with platform.app.state.db.write() as session:
        row = session.get(Preview, identifier)
        row.request = {**row.request, "normalization_method": "quantile"}
    response = submit(platform, identifier)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "NORMALIZATION_METHOD_MISMATCH"


@pytest.mark.parametrize("source", ["CONFIG = {}\n", "FROZEN_REQUEST = {'normalization_method': 'quantile'}\n"])
def test_method_echo_alone_cannot_authorize_a_config_that_omits_or_changes_the_choice(platform, source):
    node, identifier, command, code = pending_choice(platform)
    plan = code["automatic_preflight"]["plan"]
    plan["generated_config"].update(source=source, sha256=hashlib.sha256(source.encode()).hexdigest())
    code["automatic_preflight"]["plan_digest"] = canonical_digest(plan)
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "FAILED" and checked["error"]["code"] == "NORMALIZATION_METHOD_MISMATCH"


def test_started_receipt_cannot_change_the_frozen_method(platform):
    node, identifier, command, code = pending_choice(platform)
    assert acknowledge(platform, node, identifier, command, code)["status"] == "READY"
    run = submit(platform, identifier).json()
    assert platform.schedule() == 1
    run = platform.get_run(run)
    actual = copy.deepcopy(run["code_expectation"])
    actual["normalization_method"] = "quantile"
    response = platform.emit(node, run["attempts"][0]["id"], 1, "STARTED", {
        "pid": 1234, "code_snapshot": actual, "output_dir": "/machine/output/current-attempt"})
    assert response.status_code == 200, response.text
    current = platform.get_run(run)
    assert current["state"] == "CANCELLING" and current["error"]["code"] == "NORMALIZATION_METHOD_MISMATCH"
    assert any(command["type"] == "CANCEL" for command in platform.commands(node))


@pytest.mark.parametrize("value", ["", "min_max", "q01_q99", True, 1, [], {}])
def test_invalid_method_is_rejected_before_command_dispatch(platform, value):
    node = platform.register(adapter="openpi-sft")
    response = request_preview(platform, node, normalization_method=value)
    assert response.status_code == 422 and platform.commands(node) == []


def test_smoke_rejects_explicit_method_instead_of_claiming_it_took_effect(platform):
    node = platform.register()
    response = request_preview(platform, node, template_id="smoke", normalization_method="minmax")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "NORMALIZATION_METHOD_UNSUPPORTED"
    assert platform.commands(node) == []
