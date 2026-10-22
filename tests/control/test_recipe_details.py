"""Recipe source details are bounded, authenticated and tied to one environment."""

import copy

import pytest
from sqlalchemy import select

from vla_control.common import now, offset_time, uid
from vla_control.models import Command, Node, Profile

from .conftest import ADMIN


def register(platform, **changes):
    return platform.register(adapter="openpi-sft", recipe="pi0_base", metadata={}, **changes)


def create(platform, node, recipe="pi0_base", profile_id="default", headers=ADMIN):
    return platform.client.post(f"/api/v1/nodes/{node['node_id']}/recipe-details", headers=headers,
                                json={"profile_id": profile_id, "recipe": recipe})


def get(platform, node, query_id, headers=ADMIN):
    return platform.client.get(f"/api/v1/nodes/{node['node_id']}/recipe-details/{query_id}", headers=headers)


def result(**changes):
    return {"profile_id": "default", "recipe": "pi0_base",
            "source": {"path": "src/openpi/training/config.py", "sha256": "a" * 64, "line": 80, "end_line": 100,
                       "commit_sha": "b" * 40, "code_dir": "/machine/code", "changes_digest": "c" * 64, "dirty": False},
            "fields": [{"path": "batch_size", "kind": "literal", "value": 32, "expression": "32", "origin": "recipe"},
                       {"path": "model", "kind": "expression", "expression": "Pi0Config()", "origin": "recipe"},
                       {"path": "ema_decay", "kind": "literal", "value": 0.99, "expression": "0.99", "origin": "default"}],
            "source_code": "TrainConfig(name='pi0_base', batch_size=32, model=Pi0Config())", "warnings": [], **changes}


def ack(platform, node, query_id, value=None, status="APPLIED"):
    return platform.client.post(f"/agent/v1/commands/{query_id}/ack", headers=node["headers"],
                                json={"node_id": node["node_id"], "status": status,
                                      "result": result() if value is None else value})


def ready(platform, node, value=None):
    query = create(platform, node).json()
    response = ack(platform, node, query["id"], value)
    assert response.status_code == 200, response.text
    return get(platform, node, query["id"]).json()


def test_async_detail_contract_cache_and_safe_ack(platform):
    node = register(platform)
    pending = create(platform, node)
    assert pending.status_code == 202
    query = pending.json()
    assert query["status"] == "PENDING" and query["detail"] is None
    assert pending.headers["Cache-Control"] == "no-store"
    assert create(platform, node).json()["id"] == query["id"]
    command = platform.commands(node)[0]
    assert command["type"] == "RECIPE_DETAIL" and command["payload"]["recipe"] == "pi0_base"
    raw = result(private_agent_output="not displayed")
    raw["source"]["unexpected"] = "not persisted"
    raw["fields"][0]["editable"] = True
    assert ack(platform, node, query["id"], raw).status_code == 200
    completed = get(platform, node, query["id"]).json()
    assert completed["status"] == "READY" and completed["detail"] == result()
    assert ack(platform, node, query["id"], raw).status_code == 200
    assert create(platform, node).json()["id"] == query["id"]
    with platform.app.state.db.read() as session:
        stored = session.get(Command, query["id"])
        assert stored.result == result() and stored.acknowledged_at


def test_admin_scope_unavailable_profile_and_wrong_node(platform):
    node, other = register(platform), register(platform, name="other")
    for headers in ({}, node["headers"], {"Authorization": "Bearer wrong"}):
        assert create(platform, node, headers=headers).status_code == 401
    query_id = create(platform, node).json()["id"]
    assert get(platform, node, query_id, headers={}).status_code == 401
    assert get(platform, other, query_id).status_code == 404
    assert ack(platform, other, query_id).status_code == 403
    assert create(platform, node, profile_id="missing").status_code == 404
    assert create(platform, {"node_id": uid()}).status_code == 404
    assert create(platform, platform.register(name="smoke")).status_code == 422


@pytest.mark.parametrize("recipe", ["", "invalid/name", "x y", "配方", "x" * 256, "x\n"])
def test_invalid_recipe_is_not_dispatched(platform, recipe):
    node = register(platform)
    assert create(platform, node, recipe=recipe).status_code == 422
    assert platform.commands(node) == []


@pytest.mark.parametrize("state", ["OFFLINE", "RECOVERING", "DISCONNECTED", "REMOVED"])
def test_unavailable_machine_cannot_read_or_poll_details(platform, state):
    node = register(platform)
    query_id = create(platform, node).json()["id"]
    with platform.app.state.db.write() as session:
        session.get(Node, node["node_id"]).status = state
    assert create(platform, node).status_code == 409
    assert get(platform, node, query_id).status_code == 409


def test_stale_heartbeat_and_profile_changes_are_not_reused(platform):
    node = register(platform)
    query_id = create(platform, node).json()["id"]
    with platform.app.state.db.write() as session:
        session.get(Node, node["node_id"]).last_seen = offset_time(-3600)
    assert create(platform, node).json()["error"]["code"] == "NODE_OFFLINE"
    platform.heartbeat(node)
    with platform.app.state.db.write() as session:
        session.get(Profile, (node["node_id"], "default")).python = "/machine/other-python"
    stale = get(platform, node, query_id).json()
    assert stale["status"] == "FAILED" and stale["error"]["code"] == "RECIPE_PROFILE_CHANGED"
    assert ack(platform, node, query_id).status_code == 200
    assert ack(platform, node, query_id).status_code == 200
    assert create(platform, node).json()["id"] != query_id


def test_timeout_late_ack_and_scheduler_expiry(platform):
    node = register(platform)
    query_id = create(platform, node).json()["id"]
    with platform.app.state.db.write() as session:
        command = session.get(Command, query_id)
        command.payload = {**command.payload, "expires_at": offset_time(-1)}
    failed = get(platform, node, query_id).json()
    assert failed["error"]["code"] == "RECIPE_DETAIL_TIMEOUT"
    assert ack(platform, node, query_id).status_code == 200
    assert get(platform, node, query_id).json() == failed
    second_id = create(platform, node).json()["id"]
    with platform.app.state.db.write() as session:
        command = session.get(Command, second_id)
        command.payload = {**command.payload, "expires_at": offset_time(-1)}
        session.get(Node, node["node_id"]).status = "REMOVED"
    platform.schedule()
    with platform.app.state.db.read() as session:
        assert session.get(Command, second_id).status == "REJECTED"


def test_cache_scope_and_pending_rate_bound(platform):
    node = register(platform)
    first = create(platform, node).json()["id"]
    assert create(platform, node, recipe="pi0_other").json()["id"] != first
    for index in range(8):
        assert create(platform, node, recipe=f"pi0_{index}").status_code == 202
    assert create(platform, node, recipe="overflow").status_code == 429
    assert create(platform, node).json()["id"] == first
    with platform.app.state.db.write() as session:
        for command in session.scalars(select(Command).where(Command.node_id == node["node_id"])):
            command.payload = {**command.payload, "expires_at": offset_time(-1)}
    assert create(platform, node).status_code == 202


@pytest.mark.parametrize(("code", "expected"), [
    ("COMMAND_UNSUPPORTED", "AGENT_RECIPE_DETAIL_UNSUPPORTED"), ("RECIPE_QUERY_EXPIRED", "RECIPE_DETAIL_TIMEOUT"),
    ("CODE_VERSION_UNAVAILABLE", "CODE_VERSION_UNAVAILABLE"), ("RECIPE_DETAIL_UNSUPPORTED", "RECIPE_DETAIL_UNSUPPORTED"),
    ("RECIPE_SOURCE_CHANGED", "RECIPE_SOURCE_CHANGED"), ("RAW_REMOTE_ERROR", "RECIPE_DETAIL_FAILED"),
])
def test_agent_errors_are_safe_and_actionable(platform, code, expected):
    node = register(platform)
    query_id = create(platform, node).json()["id"]
    assert ack(platform, node, query_id, {"error": {"code": code, "message": "private traceback"}}, "REJECTED").status_code == 200
    failed = get(platform, node, query_id).json()
    assert failed["status"] == "FAILED" and failed["error"]["code"] == expected and "traceback" not in str(failed)


@pytest.mark.parametrize("change", [
    {"profile_id": "other"}, {"recipe": "other"}, {"source_code": "x" * 32769}, {"source_code": "bad\x00"},
    {"warnings": ["private output"]}, {"warnings": [[]]}, {"warnings": ["UNRESOLVED_DEFAULTS"] * 2},
    {"fields": "not a list"}, {"fields": [{}]},
    {"fields": [{"path": "unsafe/path", "kind": "expression", "expression": "x", "origin": "recipe"}]},
    {"fields": [{"path": "x", "kind": "expression", "expression": "x" * 2049, "origin": "recipe"}]},
    {"fields": [{"path": "x", "kind": "literal", "expression": "x", "origin": "recipe"}]},
    {"fields": [{"path": "x", "kind": "literal", "value": [1] * 65, "expression": "x", "origin": "recipe"}]},
    {"fields": [{"path": "x", "kind": "expression", "expression": "x", "origin": "recipe"}] * 2},
])
def test_malformed_details_are_not_exposed_or_cached(platform, change):
    node = register(platform)
    query_id = create(platform, node).json()["id"]
    assert ack(platform, node, query_id, result(**change)).status_code == 200
    failed = get(platform, node, query_id).json()
    assert failed["status"] == "FAILED" and failed["detail"] is None
    assert failed["error"]["code"] == "RECIPE_DETAIL_INVALID_RESULT"
    assert create(platform, node).json()["id"] != query_id


@pytest.mark.parametrize("source_changes", [
    {"path": "/etc/passwd"}, {"path": []}, {"sha256": "g" * 64}, {"commit_sha": None}, {"changes_digest": "f"},
    {"dirty": 1}, {"code_dir": "/outside/code"}, {"code_dir": "/machine/../other"}, {"line": True},
    {"line": 101}, {"end_line": 1_000_001},
])
def test_source_identity_and_pins_are_bounded(platform, source_changes):
    node = register(platform)
    value = result()
    value["source"].update(source_changes)
    completed = ready(platform, node, value)
    assert completed["status"] == "FAILED" and completed["detail"] is None


def test_expression_values_stay_unresolved_and_default_origins_survive(platform):
    node = register(platform)
    value = result(warnings=["DETAIL_TEXT_SANITIZED", "UNRESOLVED_DEFAULTS"])
    completed = ready(platform, node, value)
    assert completed["status"] == "READY"
    expression = completed["detail"]["fields"][1]
    assert expression["kind"] == "expression" and "value" not in expression
    assert completed["detail"]["fields"][2]["origin"] == "default"


def test_historical_unsafe_details_are_revalidated(platform):
    node = register(platform)
    completed = ready(platform, node)
    with platform.app.state.db.write() as session:
        command = session.get(Command, completed["id"])
        unsafe = copy.deepcopy(command.result)
        unsafe["source"]["path"] = "/private/config.py"
        command.result = unsafe
    assert get(platform, node, completed["id"]).json()["status"] == "FAILED"
