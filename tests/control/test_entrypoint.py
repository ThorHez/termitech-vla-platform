"""A selected project script is validated, frozen, and checked at execution."""

import copy

import pytest
from vla_protocol import canonical_digest

from vla_control.models import Preview

from .conftest import ADMIN, automatic_checks, automatic_preflight, snapshot
from .test_automatic_preflight import acknowledge, submit
from .test_task_recipe import inputs, request_preview


ENTRYPOINT = "scripts/train_multi.py"


def pending(platform, **changes):
    node = platform.register(adapter="openpi-sft", recipe="umi", metadata={})
    response = request_preview(platform, node, inputs=inputs(platform, node), entrypoint=ENTRYPOINT, **changes)
    assert response.status_code == 202, response.text
    identifier = response.json()["id"]
    command = next(item for item in platform.commands(node) if item["type"] == "PREFLIGHT")
    evidence = automatic_preflight(command["payload"])
    evidence["plan"].update(requested_entrypoint=ENTRYPOINT, entrypoint="/machine/code/" + ENTRYPOINT)
    evidence["plan_digest"] = canonical_digest(evidence["plan"])
    code = snapshot(recipe="umi", requested_entrypoint=ENTRYPOINT, requested_entrypoint_sha256="d" * 64,
                    automatic_preflight=evidence)
    return node, identifier, command, code


def test_entrypoint_is_frozen_forwarded_and_serialized_without_database_column(platform):
    node, identifier, command, code = pending(platform)
    assert command["payload"]["entrypoint"] == ENTRYPOINT
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "READY" and checked["request"]["entrypoint"] == ENTRYPOINT
    response = submit(platform, identifier)
    assert response.status_code == 201, response.text
    run = response.json()
    assert run["entrypoint"] == ENTRYPOINT
    assert run["code_expectation"]["automatic_preflight"]["plan"]["requested_entrypoint"] == ENTRYPOINT
    assert platform.schedule() == 1
    start = next(item for item in platform.commands(node) if item["type"] == "START")
    assert start["payload"]["entrypoint"] == ENTRYPOINT
    assert start["payload"]["code_expectation"]["requested_entrypoint"] == ENTRYPOINT
    platform.start(node, run)
    assert platform.get_run(run)["state"] == "RUNNING"


@pytest.mark.parametrize("entrypoint", ["", "/machine/code/train.py", "../train.py", "scripts/../train.py",
    "./train.py", "scripts/./train.py", "scripts//train.py", "scripts/", "train.sh", "scripts\\train.py",
    "scripts/evil\n.py", "scripts/evil\x00.py", "scripts/evil\x85.py", "scripts/evil\u202e.py", "x" * 4097 + ".py"])
def test_invalid_entrypoints_never_dispatch(platform, entrypoint):
    node = platform.register(adapter="openpi-sft")
    response = request_preview(platform, node, inputs=inputs(platform, node), entrypoint=entrypoint)
    assert response.status_code == 422
    assert platform.commands(node) == []


@pytest.mark.parametrize("mutation,expected", [
    ("no_snapshot_key", "AGENT_ENTRYPOINT_UNSUPPORTED"), ("no_plan_key", "AGENT_ENTRYPOINT_UNSUPPORTED"),
    ("snapshot_mismatch", "ENTRYPOINT_MISMATCH"), ("plan_mismatch", "ENTRYPOINT_MISMATCH"),
    ("outside_project", "ENTRYPOINT_MISMATCH"), ("digest", "AGENT_PREFLIGHT_INVALID"),
])
def test_explicit_entry_requires_matching_current_agent_evidence(platform, mutation, expected):
    node, identifier, command, code = pending(platform)
    plan = code["automatic_preflight"]["plan"]
    if mutation == "no_snapshot_key":
        code.pop("requested_entrypoint")
    elif mutation == "no_plan_key":
        plan.pop("requested_entrypoint")
    elif mutation == "snapshot_mismatch":
        code["requested_entrypoint"] = "scripts/other.py"
    elif mutation == "plan_mismatch":
        plan["requested_entrypoint"] = "scripts/other.py"
    elif mutation == "outside_project":
        plan["entrypoint"] = "/machine/sibling/train.py"
    else:
        plan["entrypoint"] = "/machine/code/other.py"
    if mutation != "digest":
        code["automatic_preflight"]["plan_digest"] = canonical_digest(plan)
    checked = acknowledge(platform, node, identifier, command, code)
    assert checked["status"] == "FAILED" and checked["error"]["code"] == expected


def test_internal_project_symlink_can_have_different_canonical_script_path(platform):
    node, identifier, command, code = pending(platform)
    evidence = code["automatic_preflight"]
    evidence["plan"]["entrypoint"] = "/machine/code/internal/train.py"
    evidence["plan_digest"] = canonical_digest(evidence["plan"])
    assert acknowledge(platform, node, identifier, command, code)["status"] == "READY"


@pytest.mark.parametrize("missing", ["snapshot", "plan"])
def test_submission_rechecks_explicit_entry_evidence(platform, missing):
    node, identifier, command, code = pending(platform)
    assert acknowledge(platform, node, identifier, command, code)["status"] == "READY"
    with platform.app.state.db.write() as session:
        preview = session.get(Preview, identifier)
        changed = copy.deepcopy(preview.code_snapshot)
        if missing == "snapshot":
            changed.pop("requested_entrypoint")
        else:
            changed["automatic_preflight"]["plan"].pop("requested_entrypoint")
            changed["automatic_preflight"]["plan_digest"] = canonical_digest(changed["automatic_preflight"]["plan"])
        preview.code_snapshot = changed
    response = submit(platform, identifier)
    assert response.status_code == 409 and response.json()["error"]["code"] == "AGENT_ENTRYPOINT_UNSUPPORTED"


@pytest.mark.parametrize("mutation,expected", [("missing", "AGENT_ENTRYPOINT_UNSUPPORTED"),
    ("requested", "ENTRYPOINT_MISMATCH"), ("sha256", "CODE_CHANGED_SINCE_PREFLIGHT")])
def test_started_entry_drift_is_cancelled_and_keeps_gpu_reserved(platform, mutation, expected):
    node, identifier, command, code = pending(platform)
    acknowledge(platform, node, identifier, command, code)
    run = submit(platform, identifier).json()
    platform.schedule()
    attempt = platform.get_run(run)["attempts"][0]
    changed = copy.deepcopy(code)
    if mutation == "missing":
        changed.pop("requested_entrypoint")
    elif mutation == "requested":
        changed["requested_entrypoint"] = "other.py"
    else:
        changed["requested_entrypoint_sha256"] = "e" * 64
    response = platform.emit(node, attempt["id"], 1, "STARTED", {"pid": 1234, "code_snapshot": changed,
        "output_dir": "/machine/output/" + attempt["id"]})
    assert response.status_code == 200, response.text
    current = platform.get_run(run)
    assert current["state"] == "CANCELLING" and current["error"]["code"] == expected
    resource = platform.client.get(f"/api/v1/nodes/{node['node_id']}", headers=ADMIN).json()
    assert resource["gpus"][0]["reserved_by"] == attempt["id"]
    assert any(command["type"] == "CANCEL" for command in platform.commands(node))


def test_agent_cannot_invent_an_explicit_entrypoint_for_automatic_request(platform):
    node = platform.register(adapter="openpi-sft", recipe="umi", metadata={})
    response = request_preview(platform, node, inputs=inputs(platform, node))
    identifier = response.json()["id"]
    command = next(item for item in platform.commands(node) if item["type"] == "PREFLIGHT")
    code = snapshot(recipe="umi", requested_entrypoint="other.py", automatic_preflight=automatic_preflight(command["payload"]))
    checked = acknowledge(platform, node, identifier, command, code, checks=automatic_checks())
    assert checked["status"] == "FAILED" and checked["error"]["code"] == "ENTRYPOINT_MISMATCH"
