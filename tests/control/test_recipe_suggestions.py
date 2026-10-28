"""Discovered recipes remain machine-local hints, never new training permission."""

import pytest
from sqlalchemy import select

from vla_control.common import now, offset_time, uid
from vla_control.models import Command, Node, Profile
from vla_control.remote_models import RemoteConnection, RemoteDeployment

from .conftest import ADMIN


def register(platform, **changes):
    return platform.register(adapter="openpi-sft", recipe=None, metadata={}, **changes)


def create(platform, node, profile_id="default", prefix="pi0", limit=30, headers=ADMIN):
    return platform.client.post(f"/api/v1/nodes/{node['node_id']}/recipe-suggestions", headers=headers,
                                json={"profile_id": profile_id, "prefix": prefix, "limit": limit})


def get(platform, node, query_id, headers=ADMIN):
    return platform.client.get(f"/api/v1/nodes/{node['node_id']}/recipe-suggestions/{query_id}", headers=headers)


def result(**changes):
    return {"profile_id": "default", "prefix": "pi0", "entries": [{"name": "pi0_alpha"}, {"name": "pi0_beta"}],
            "truncated": False, "source": "src/openpi/training/config.py", "warnings": [], **changes}


def ack(platform, node, query_id, value=None, status="APPLIED"):
    return platform.client.post(f"/agent/v1/commands/{query_id}/ack", headers=node["headers"],
                                json={"node_id": node["node_id"], "status": status,
                                      "result": result() if value is None else value})


def edit_profile(platform, node, **changes):
    with platform.app.state.db.write() as session:
        profile = session.get(Profile, (node["node_id"], "default"))
        for field, value in changes.items():
            setattr(profile, field, value)


def test_admin_only_and_unknown_node(platform):
    node = register(platform)
    for headers in ({}, node["headers"], {"Authorization": "Bearer wrong"}):
        assert create(platform, node, headers=headers).status_code == 401
    query_id = create(platform, node).json()["id"]
    assert get(platform, node, query_id, headers={}).status_code == 401
    assert create(platform, {"node_id": uid()}).status_code == 404


def test_async_contract_safe_ack_and_cache(platform):
    node = register(platform)
    pending = create(platform, node)
    assert pending.status_code == 202
    query = pending.json()
    assert query["status"] == "PENDING" and query["entries"] == [] and query["error"] is None
    command = platform.commands(node)[0]
    assert command["type"] == "RECIPE_SUGGEST"
    assert command["payload"]["profile_id"] == "default"
    assert command["payload"]["prefix"] == "pi0" and command["payload"]["limit"] == 30
    assert command["payload"]["match_mode"] == "contains-v1"
    assert command["payload"]["expires_at"] > now() and command["payload"]["profile_digest"]
    assert create(platform, node).json()["id"] == query["id"]
    assert ack(platform, node, query["id"], result(unrestricted_output="private source")).status_code == 200
    ready = get(platform, node, query["id"]).json()
    assert ready["status"] == "READY"
    assert ready["entries"] == [{"name": "pi0_alpha", "validated": False, "source": "code"},
                                {"name": "pi0_beta", "validated": False, "source": "code"}]
    assert "private source" not in str(ready)
    assert create(platform, node).json()["id"] == query["id"]
    assert ack(platform, node, query["id"]).status_code == 200
    with platform.app.state.db.read() as session:
        stored = session.get(Command, query["id"])
        assert stored.status == "APPLIED" and stored.acknowledged_at
        assert "unrestricted_output" not in stored.result


def test_discovery_never_bypasses_required_training_inputs(platform):
    node = platform.register(adapter="openpi-sft", recipe="pi0_alpha", metadata={
        "validated": True, "validated_recipes": ["pi0_gamma", "invalid/name", 3],
    })
    query_id = create(platform, node).json()["id"]
    assert ack(platform, node, query_id).status_code == 200
    assert get(platform, node, query_id).json()["entries"] == [
        {"name": "pi0_alpha", "validated": True, "source": "both"},
        {"name": "pi0_beta", "validated": False, "source": "code"},
        {"name": "pi0_gamma", "validated": True, "source": "registered"},
    ]
    response = platform.client.post("/api/v1/run-previews", headers=ADMIN, json={
        "name": "discovery is not validation", "node_id": node["node_id"], "profile_id": "default",
        "template_id": "openpi-sft", "recipe": "pi0_beta", "inputs": {}, "parameters": {}, "gpu_count": 1,
    })
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INPUT_REQUIRED"


def test_false_or_nonboolean_legacy_flag_does_not_validate_default(platform):
    node = register(platform)
    edit_profile(platform, node, recipe="pi0_alpha", metadata_json={"validated": "true"})
    query_id = create(platform, node).json()["id"]
    ack(platform, node, query_id)
    assert get(platform, node, query_id).json()["entries"][0] == {
        "name": "pi0_alpha", "validated": False, "source": "both"}


def test_merge_filters_keyword_and_caps_output(platform):
    node = register(platform)
    edit_profile(platform, node, recipe="other", metadata_json={"validated_recipes": ["pi0_aaa", "other"]})
    query_id = create(platform, node, limit=2).json()["id"]
    ack(platform, node, query_id, result(truncated=True))
    ready = get(platform, node, query_id).json()
    assert [entry["name"] for entry in ready["entries"]] == ["pi0_aaa", "pi0_alpha"]
    assert ready["truncated"] is True


def test_inference_config_suggestions_share_cache_and_only_return_existing_code(platform):
    node = register(platform)
    edit_profile(platform, node, recipe="pi0_alpha", metadata_json={"validated_recipes": ["pi0_aaa"]})
    query = create(platform, node, limit=2).json()
    path = f"/api/v1/nodes/{node['node_id']}/inference-config-suggestions"
    payload = {"profile_id": "default", "prefix": "pi0", "limit": 2}
    assert platform.client.post(path, headers=ADMIN, json=payload).json()["id"] == query["id"]
    assert ack(platform, node, query["id"]).status_code == 200
    ready = platform.client.get(path + "/" + query["id"], headers=ADMIN).json()
    assert ready["status"] == "READY"
    assert ready["entries"] == [{"name": "pi0_alpha", "validated": False, "source": "both"},
                                {"name": "pi0_beta", "validated": False, "source": "code"}]
    assert ready["truncated"] is False
    cached = platform.client.post(path, headers=ADMIN, json=payload).json()
    assert cached == ready
    assert len(platform.commands(node)) == 0
    assert [entry["name"] for entry in get(platform, node, query["id"]).json()["entries"]] == ["pi0_aaa", "pi0_alpha"]


@pytest.mark.parametrize("role,expected", [("operator", 202), ("viewer", 403)])
def test_inference_discovery_uses_deployment_permission_without_changing_recipe_permissions(platform, monkeypatch, role, expected):
    from vla_control.auth import ROLE_PERMISSIONS
    from vla_control.authorization import PERMISSIONS

    node = register(platform)
    # Exercise a deployer with no recipe editing permission while retaining the actual role model.
    monkeypatch.setitem(ROLE_PERMISSIONS, "operator", frozenset({"platform.read", "training.write"}))
    password = "test-user-password"
    platform.app.state.auth.create_user("deployment-user", password, role)
    platform.client.base_url = "https://testserver"
    login = platform.client.post("/api/v1/auth/login", json={"username": "deployment-user", "password": password})
    assert login.status_code == 200, login.text
    headers = {"X-CSRF-Token": login.json()["csrf_token"]}
    path = f"/api/v1/nodes/{node['node_id']}/inference-config-suggestions"
    payload = {"profile_id": "default", "prefix": "pi0", "limit": 30}
    response = platform.client.post(path, headers=headers, json=payload)
    assert response.status_code == expected, response.text
    assert platform.client.post(path.replace("inference-config", "recipe"), headers=headers, json=payload).status_code == 403
    assert PERMISSIONS[("POST", "/api/v1/nodes/{node_id}/recipe-suggestions")] == "recipes.write"
    if expected == 202:
        assert platform.client.get(path + "/" + response.json()["id"]).status_code == 200


@pytest.mark.parametrize("keyword", ["elec", "ELEC", "ElEc"])
def test_middle_keyword_is_case_insensitive_for_code_and_registered_recipes(platform, keyword):
    node = register(platform)
    edit_profile(platform, node, recipe="pi05_ElecMeta", metadata_json={"validated_recipes": ["pi0_elec_registered"]})
    query_id = create(platform, node, prefix=keyword).json()["id"]
    payload = result(prefix=keyword, entries=[{"name": "pi05_ElecMeta"}, {"name": "pi0_elec_code"}])
    assert ack(platform, node, query_id, payload).status_code == 200
    ready = get(platform, node, query_id).json()
    assert ready["status"] == "READY"
    assert ready["entries"] == [
        {"name": "pi05_ElecMeta", "validated": False, "source": "both"},
        {"name": "pi0_elec_code", "validated": False, "source": "code"},
        {"name": "pi0_elec_registered", "validated": True, "source": "registered"},
    ]


def test_prefix_matches_rank_first_before_keyword_matches_and_truncation(platform):
    node = register(platform)
    edit_profile(platform, node, metadata_json={"validated_recipes": ["ELEC_registered", "aaa_elec_registered", "unrelated"]})
    query_id = create(platform, node, prefix="elec", limit=3).json()["id"]
    payload = result(prefix="elec", entries=[{"name": "aaa_elec"}, {"name": "elec_z"}, {"name": "Elec_A"}])
    assert ack(platform, node, query_id, payload).status_code == 200
    ready = get(platform, node, query_id).json()
    assert [entry["name"] for entry in ready["entries"]] == ["Elec_A", "ELEC_registered", "elec_z"]
    assert ready["truncated"] is True


def test_matching_sort_is_stable_for_case_variants(platform):
    node = register(platform)
    query_id = create(platform, node, prefix="elec").json()["id"]
    assert ack(platform, node, query_id, result(prefix="elec", entries=[{"name": "elec"}, {"name": "ELEC"}, {"name": "Elec"}])).status_code == 200
    assert [entry["name"] for entry in get(platform, node, query_id).json()["entries"]] == ["ELEC", "Elec", "elec"]


def test_nonmatching_keyword_ack_is_rejected(platform):
    node = register(platform)
    query_id = create(platform, node, prefix="elec").json()["id"]
    assert ack(platform, node, query_id, result(prefix="elec", entries=[{"name": "pi0_other"}])).status_code == 200
    failed = get(platform, node, query_id).json()
    assert failed["status"] == "FAILED"
    assert failed["error"]["code"] == "RECIPE_SUGGEST_INVALID_RESULT"


@pytest.mark.parametrize("old_mode", [None, "prefix-v1"])
def test_old_prefix_cache_is_not_reused_for_keyword_search(platform, old_mode):
    node = register(platform)
    old_id = create(platform, node, prefix="elec").json()["id"]
    assert ack(platform, node, old_id, result(prefix="elec", entries=[])).status_code == 200
    with platform.app.state.db.write() as session:
        command = session.get(Command, old_id)
        payload = dict(command.payload)
        if old_mode is None:
            payload.pop("match_mode")
        else:
            payload["match_mode"] = old_mode
        command.payload = payload
    fresh = create(platform, node, prefix="elec").json()
    assert fresh["id"] != old_id
    assert create(platform, node, prefix="elec").json()["id"] == fresh["id"]


@pytest.mark.parametrize("prefix", ["", "pi0", "PI0", "a-b_c3"])
def test_named_prefixes_are_accepted(platform, prefix):
    assert create(platform, register(platform), prefix=prefix).status_code == 202


@pytest.mark.parametrize("prefix", ["pi0/", "x y", "配方", "x\n", "x\x00", "x\x7f", "x" * 256])
def test_invalid_prefix_is_not_dispatched(platform, prefix):
    node = register(platform)
    assert create(platform, node, prefix=prefix).status_code == 422
    assert platform.commands(node) == []


@pytest.mark.parametrize("limit", [0, 51, True, "30", 1.5])
def test_limit_is_bounded_integer(platform, limit):
    assert create(platform, register(platform), limit=limit).status_code == 422


def test_profile_must_belong_to_selected_node_be_active_and_support_openpi(platform):
    node = register(platform)
    assert create(platform, node, profile_id="missing").status_code == 404
    edit_profile(platform, node, active=False)
    assert create(platform, node).status_code == 404
    smoke = platform.register(name="smoke")
    assert create(platform, smoke).json()["error"]["code"] == "RECIPE_PROFILE_UNSUPPORTED"


@pytest.mark.parametrize("state", ["OFFLINE", "RECOVERING", "DISCONNECTED", "REMOVED"])
def test_unavailable_nodes_cannot_query_or_poll(platform, state):
    node = register(platform)
    query_id = create(platform, node).json()["id"]
    with platform.app.state.db.write() as session:
        session.get(Node, node["node_id"]).status = state
    assert create(platform, node).status_code == 409
    assert get(platform, node, query_id).status_code == 409


def test_expired_heartbeat_blocks_queries(platform):
    node = register(platform)
    query_id = create(platform, node).json()["id"]
    with platform.app.state.db.write() as session:
        session.get(Node, node["node_id"]).last_seen = offset_time(-3600)
    assert create(platform, node).json()["error"]["code"] == "NODE_OFFLINE"
    assert get(platform, node, query_id).json()["error"]["code"] == "NODE_OFFLINE"


def test_maintenance_blocks_queries(platform):
    node = register(platform)
    timestamp, connection_id = now(), uid()
    with platform.app.state.db.write() as session:
        session.add(RemoteConnection(id=connection_id, host="worker", port=22, username="root", encrypted_password="sealed",
                                     fingerprint="fingerprint", probe_result={}, install_dir="/machine/agent",
                                     created_at=timestamp, expires_at=offset_time(120)))
        session.flush()
        session.add(RemoteDeployment(id=uid(), connection_id=connection_id, idempotency_key=uid(), request_digest="hash",
                                     target_key="worker", onboarding_id=uid(), node_id=node["node_id"], state="RUNNING",
                                     stage="INSTALL", logs=[], history=[], agent_config={}, install_dir="/machine/agent",
                                     control_plane_url="http://control", connection_mode="DIRECT", tunnel_state="NONE",
                                     created_at=timestamp, updated_at=timestamp))
    assert create(platform, node).json()["error"]["code"] == "NODE_MAINTENANCE"


def test_node_and_command_type_isolation(platform):
    node, other = register(platform), register(platform, name="other")
    query_id = create(platform, node).json()["id"]
    assert get(platform, other, query_id).status_code == 404
    assert ack(platform, other, query_id).status_code == 403
    with platform.app.state.db.write() as session:
        wrong = Command(id=uid(), node_id=node["node_id"], type="ASSET_VALIDATE", status="PENDING", payload={}, created_at=now())
        session.add(wrong)
        wrong_id = wrong.id
    assert get(platform, node, wrong_id).status_code == 404


def test_timeout_and_late_ack_are_terminal(platform):
    node = register(platform)
    query_id = create(platform, node).json()["id"]
    with platform.app.state.db.write() as session:
        command = session.get(Command, query_id)
        command.payload = {**command.payload, "expires_at": offset_time(-1)}
    expired = get(platform, node, query_id).json()
    assert expired["status"] == "FAILED" and expired["error"]["code"] == "RECIPE_SUGGEST_TIMEOUT"
    assert ack(platform, node, query_id).status_code == 200
    assert get(platform, node, query_id).json() == expired
    assert platform.commands(node) == []


def test_scheduler_expires_queries_for_retired_node(platform):
    node = register(platform)
    query_id = create(platform, node).json()["id"]
    with platform.app.state.db.write() as session:
        session.get(Node, node["node_id"]).status = "REMOVED"
        command = session.get(Command, query_id)
        command.payload = {**command.payload, "expires_at": offset_time(-1)}
    platform.schedule()
    with platform.app.state.db.read() as session:
        command = session.get(Command, query_id)
        assert command.status == "REJECTED" and command.acknowledged_at


@pytest.mark.parametrize("changes", [
    {"code_dir": "/machine/other"}, {"python": "/machine/other-python"}, {"active": False},
    {"recipe": "pi0_default"}, {"metadata_json": {"validated_recipes": ["pi0_alpha"]}},
])
def test_profile_change_invalidates_pending_and_cached_results(platform, changes):
    node = register(platform)
    query_id = create(platform, node).json()["id"]
    edit_profile(platform, node, **changes)
    failed = get(platform, node, query_id).json()
    assert failed["status"] == "FAILED" and failed["error"]["code"] == "RECIPE_PROFILE_CHANGED"
    assert ack(platform, node, query_id).status_code == 200
    assert ack(platform, node, query_id).status_code == 200
    assert get(platform, node, query_id).json() == failed
    if changes.get("active") is not False:
        assert create(platform, node).json()["id"] != query_id


def test_applied_result_is_hidden_after_profile_or_root_change(platform):
    node = register(platform)
    query_id = create(platform, node).json()["id"]
    ack(platform, node, query_id)
    with platform.app.state.db.write() as session:
        session.get(Node, node["node_id"]).roots = ["/machine/other"]
    failed = get(platform, node, query_id).json()
    assert failed["error"]["code"] == "RECIPE_PROFILE_CHANGED" and failed["entries"] == []
    assert ack(platform, node, query_id).status_code == 200


def test_cache_scopes_limit_prefix_node_profile_and_age(platform):
    node, other = register(platform), register(platform, name="other")
    query_id = create(platform, node).json()["id"]
    assert create(platform, node, limit=10).json()["id"] != query_id
    assert create(platform, node, prefix="other").json()["id"] != query_id
    assert create(platform, other).json()["id"] != query_id
    with platform.app.state.db.write() as session:
        original = session.get(Profile, (node["node_id"], "default"))
        session.add(Profile(node_id=node["node_id"], id="second", name="second", code_dir=original.code_dir,
                            python=original.python, adapter=original.adapter, recipe=None, metadata_json={}, active=True))
        session.get(Command, query_id).created_at = offset_time(-11)
    assert create(platform, node, profile_id="second").json()["id"] != query_id
    assert create(platform, node).json()["id"] != query_id


def test_pending_bound_and_expiration_release_capacity(platform):
    node = register(platform)
    for index in range(20):
        assert create(platform, node, prefix=f"pi{index}").status_code == 202
    assert create(platform, node, prefix="new").status_code == 429
    assert create(platform, node, prefix="pi0").status_code == 202
    with platform.app.state.db.write() as session:
        command = session.scalar(select(Command).where(Command.node_id == node["node_id"]).limit(1))
        command.payload = {**command.payload, "expires_at": offset_time(-1)}
    assert create(platform, node, prefix="new").status_code == 202


@pytest.mark.parametrize("change", [
    {"profile_id": "other"}, {"prefix": "other"}, {"truncated": 1}, {"source": "/private/config.py"},
    {"source": []}, {"source": None}, {"warnings": ["raw secret text"]}, {"warnings": "unexpected"},
    {"warnings": [[]]}, {"warnings": ["INVALID_RECIPE_NAME"] * 2}, {"entries": "not a list"},
    {"entries": [{"name": "pi0_alpha"}] * 2}, {"entries": [{"name": "other"}]},
    {"entries": [{"name": "pi0/unsafe"}]}, {"entries": [{"name": []}]}, {"entries": [None]},
    {"entries": [{"name": "pi0_alpha"}] * 31}, {"error": {"code": [], "message": "private source"}},
])
def test_untrusted_results_fail_safely(platform, change):
    node = register(platform)
    query_id = create(platform, node).json()["id"]
    assert ack(platform, node, query_id, result(**change)).status_code == 200
    failed = get(platform, node, query_id).json()
    assert failed["status"] == "FAILED" and failed["entries"] == []
    assert failed["error"]["code"] in {"RECIPE_SUGGEST_INVALID_RESULT", "RECIPE_SUGGEST_FAILED"}
    assert "private source" not in str(failed)
    assert create(platform, node).json()["id"] != query_id


def test_only_whitelisted_warnings_and_sources_are_exposed(platform):
    node = register(platform)
    query_id = create(platform, node).json()["id"]
    value = result(warnings=["DYNAMIC_REGISTRY_UNSUPPORTED", "INVALID_RECIPE_NAME", "DISCOVERY_LIMIT_REACHED"],
                   source="openpi/training/config.py")
    assert ack(platform, node, query_id, value).status_code == 200
    ready = get(platform, node, query_id).json()
    assert ready["status"] == "READY" and ready["warnings"] == value["warnings"] and ready["source"] == value["source"]


@pytest.mark.parametrize(("code", "expected"), [
    ("COMMAND_UNSUPPORTED", "AGENT_RECIPE_SUGGEST_UNSUPPORTED"), ("RECIPE_QUERY_EXPIRED", "RECIPE_SUGGEST_TIMEOUT"),
    ("PROFILE_UNAVAILABLE", "PROFILE_UNAVAILABLE"), ("RECIPE_SOURCE_INVALID", "RECIPE_SOURCE_INVALID"),
    ("UNEXPECTED_REMOTE_TRACEBACK", "RECIPE_SUGGEST_FAILED"),
])
def test_safe_agent_errors_and_old_agent_upgrade(platform, code, expected):
    node = register(platform)
    query_id = create(platform, node).json()["id"]
    assert ack(platform, node, query_id, {"error": {"code": code, "message": "private source"}}, "REJECTED").status_code == 200
    failed = get(platform, node, query_id).json()
    assert failed["error"]["code"] == expected and "private source" not in str(failed)


def test_historical_unsafe_or_agent_trust_claims_are_not_exposed(platform):
    node = register(platform)
    query_id = create(platform, node).json()["id"]
    with platform.app.state.db.write() as session:
        command = session.get(Command, query_id)
        command.status, command.acknowledged_at = "APPLIED", now()
        command.result = result(entries=[{"name": "pi0_alpha", "validated": True, "source": "registered"}])
    assert get(platform, node, query_id).json()["entries"] == [{"name": "pi0_alpha", "validated": False, "source": "code"}]
    with platform.app.state.db.write() as session:
        session.get(Command, query_id).result = result(source="secret config contents")
    assert get(platform, node, query_id).json()["status"] == "FAILED"
