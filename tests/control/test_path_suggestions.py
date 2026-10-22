"""Path browsing stays on the selected, available machine and inside its roots."""

import pytest
from sqlalchemy import select

from vla_control.common import now, offset_time, uid
from vla_control.models import Command, Node
from vla_control.remote_models import RemoteConnection, RemoteDeployment

from .conftest import ADMIN


def create(platform, node, prefix="/machine/", limit=30, headers=ADMIN):
    return platform.client.post(f"/api/v1/nodes/{node['node_id']}/path-suggestions", headers=headers,
                                json={"prefix": prefix, "limit": limit})


def get(platform, node, query_id, headers=ADMIN):
    return platform.client.get(f"/api/v1/nodes/{node['node_id']}/path-suggestions/{query_id}", headers=headers)


def result(prefix="/machine/", parent="/machine", **changes):
    return {"prefix": prefix, "parent": parent,
            "entries": [{"name": "data", "path": "/machine/data/", "kind": "directory"},
                        {"name": "stats.json", "path": "/machine/stats.json", "kind": "file"}],
            "truncated": False, **changes}


def ack(platform, node, query_id, value=None, status="APPLIED"):
    return platform.client.post(f"/agent/v1/commands/{query_id}/ack", headers=node["headers"],
                                json={"node_id": node["node_id"], "status": status,
                                      "result": result() if value is None else value})


def test_admin_only_and_unknown_node(platform):
    node = platform.register()
    for headers in ({}, node["headers"], {"Authorization": "Bearer wrong"}):
        assert create(platform, node, headers=headers).status_code == 401
    query_id = create(platform, node).json()["id"]
    assert get(platform, node, query_id, headers={}).status_code == 401
    assert create(platform, {"node_id": uid()}).status_code == 404


def test_async_command_contract_safe_ack_and_cache(platform):
    node = platform.register()
    pending = create(platform, node)
    assert pending.status_code == 202
    query = pending.json()
    assert query["status"] == "PENDING"
    assert query["entries"] == [] and query["error"] is None
    command = platform.commands(node)[0]
    assert command["type"] == "PATH_SUGGEST"
    assert command["payload"]["prefix"] == "/machine/"
    assert command["payload"]["limit"] == 30
    assert command["payload"]["expires_at"] > now()
    assert create(platform, node).json()["id"] == query["id"]
    raw = result(unrestricted_output="do not expose")
    assert ack(platform, node, query["id"], raw).status_code == 200
    ready = get(platform, node, query["id"]).json()
    assert ready["status"] == "READY"
    assert ready["entries"] == result()["entries"]
    assert "unrestricted_output" not in ready
    assert create(platform, node).json()["id"] == query["id"]
    with platform.app.state.db.read() as session:
        stored = session.get(Command, query["id"])
        assert stored.status == "APPLIED" and stored.acknowledged_at
        assert "unrestricted_output" not in stored.result


@pytest.mark.parametrize("prefix", ["", "/", "/m", "/machine", "/machine/", "/machine/data"])
def test_prefixes_allow_roots_and_their_incomplete_ancestors(platform, prefix):
    node = platform.register()
    assert create(platform, node, prefix=prefix).status_code == 202


@pytest.mark.parametrize("prefix", ["relative", "machine/", " /machine", "/machine/../etc", "/machine/..", "/machine/\n", "/machine/\x00", "/machine/\x7f", "/machine/\x85", "/etc", "/machine-other"])
def test_invalid_or_unmanaged_prefix_is_never_dispatched(platform, prefix):
    node = platform.register()
    assert create(platform, node, prefix=prefix).status_code == 422
    assert platform.commands(node) == []


@pytest.mark.parametrize("limit", [0, 51, True, "30", 1.5])
def test_limit_is_a_bounded_integer(platform, limit):
    node = platform.register()
    assert create(platform, node, limit=limit).status_code == 422


@pytest.mark.parametrize("state", ["OFFLINE", "RECOVERING", "DISCONNECTED", "REMOVED"])
def test_unavailable_nodes_cannot_be_queried_or_polled(platform, state):
    node = platform.register()
    query_id = create(platform, node).json()["id"]
    with platform.app.state.db.write() as session:
        session.get(Node, node["node_id"]).status = state
    for response in (create(platform, node), get(platform, node, query_id)):
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "NODE_OFFLINE"


def test_online_label_with_expired_heartbeat_is_unavailable(platform):
    node = platform.register()
    with platform.app.state.db.write() as session:
        session.get(Node, node["node_id"]).last_seen = offset_time(-3600)
    assert create(platform, node).json()["error"]["code"] == "NODE_OFFLINE"


def test_queries_respect_deployment_maintenance_lock(platform):
    node = platform.register()
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
    response = create(platform, node)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "NODE_MAINTENANCE"


def test_node_and_command_type_isolation(platform):
    node, other = platform.register(), platform.register(name="other")
    query_id = create(platform, node).json()["id"]
    assert get(platform, other, query_id).status_code == 404
    assert ack(platform, other, query_id).status_code == 403
    with platform.app.state.db.write() as session:
        wrong = Command(id=uid(), node_id=node["node_id"], type="ASSET_VALIDATE", status="PENDING", payload={}, created_at=now())
        session.add(wrong)
        wrong_id = wrong.id
    assert get(platform, node, wrong_id).status_code == 404


def test_timeout_get_and_late_ack_are_terminal(platform):
    node = platform.register()
    query_id = create(platform, node).json()["id"]
    with platform.app.state.db.write() as session:
        session.get(Command, query_id).payload = {"prefix": "/machine/", "limit": 30, "expires_at": offset_time(-1)}
    expired = get(platform, node, query_id).json()
    assert expired["status"] == "FAILED"
    assert expired["error"]["code"] == "PATH_SUGGEST_TIMEOUT"
    assert ack(platform, node, query_id).status_code == 200
    assert get(platform, node, query_id).json() == expired
    assert platform.commands(node) == []
    with platform.app.state.db.read() as session:
        assert session.get(Command, query_id).acknowledged_at


def test_scheduler_expires_queries_even_for_retired_nodes(platform):
    node = platform.register()
    query_id = create(platform, node).json()["id"]
    with platform.app.state.db.write() as session:
        session.get(Node, node["node_id"]).status = "REMOVED"
        session.get(Command, query_id).payload = {"prefix": "/machine/", "limit": 30, "expires_at": offset_time(-1)}
    platform.schedule()
    with platform.app.state.db.read() as session:
        command = session.get(Command, query_id)
        assert command.status == "REJECTED" and command.acknowledged_at


def test_request_limit_and_expiration_free_capacity(platform):
    node = platform.register()
    for index in range(20):
        assert create(platform, node, prefix=f"/machine/{index}").status_code == 202
    assert create(platform, node, prefix="/machine/new").status_code == 429
    assert create(platform, node, prefix="/machine/0").status_code == 202
    with platform.app.state.db.write() as session:
        command = session.scalar(select(Command).where(Command.node_id == node["node_id"]).limit(1))
        command.payload = {**command.payload, "expires_at": offset_time(-1)}
    assert create(platform, node, prefix="/machine/new").status_code == 202


def test_cache_is_scoped_by_limit_prefix_node_and_age(platform):
    node, other = platform.register(), platform.register(name="other")
    query_id = create(platform, node).json()["id"]
    assert create(platform, node, limit=10).json()["id"] != query_id
    assert create(platform, node, prefix="/machine/other").json()["id"] != query_id
    assert create(platform, other).json()["id"] != query_id
    with platform.app.state.db.write() as session:
        session.get(Command, query_id).created_at = offset_time(-11)
    assert create(platform, node).json()["id"] != query_id


@pytest.mark.parametrize("change", [
    {"prefix": "/machine/other"}, {"truncated": 1}, {"truncated": None}, {"parent": "/etc"},
    {"entries": [{"name": "passwd", "path": "/etc/passwd", "kind": "file"}]},
    {"entries": [{"name": "data", "path": "/machine/data", "kind": "symlink"}]},
    {"entries": [{"name": "data", "path": "/machine/data", "kind": []}]},
    {"entries": [{"name": "wrong", "path": "/machine/data", "kind": "directory"}]},
    {"entries": [{"name": "foo", "path": "/machine/../foo", "kind": "file"}]},
    {"entries": [{"name": "foo\n", "path": "/machine/foo\n", "kind": "file"}]},
    {"entries": [{"name": "foo", "path": "/machine/nested/foo", "kind": "file"}]},
    {"entries": [{"name": "foo", "path": "relative/foo", "kind": "file"}]},
    {"entries": [{"name": "foo", "path": "/machine/foo", "kind": "file"}] * 31},
    {"entries": "not a list"}, {"error": {"code": [], "message": "private remote output"}},
])
def test_untrusted_agent_result_fails_safely(platform, change):
    node = platform.register()
    query_id = create(platform, node).json()["id"]
    assert ack(platform, node, query_id, result(**change)).status_code == 200
    failed = get(platform, node, query_id).json()
    assert failed["status"] == "FAILED" and failed["entries"] == []
    assert failed["error"]["code"] in {"PATH_SUGGEST_INVALID_RESULT", "PATH_SUGGEST_FAILED"}
    assert "private remote output" not in str(failed)
    assert create(platform, node).json()["id"] != query_id


def test_old_agent_error_is_actionable_and_does_not_expose_raw_output(platform):
    node = platform.register()
    query_id = create(platform, node).json()["id"]
    error = {"error": {"code": "COMMAND_UNSUPPORTED", "message": "secret remote command"}}
    assert ack(platform, node, query_id, error, status="REJECTED").status_code == 200
    failed = get(platform, node, query_id).json()
    assert failed["error"]["code"] == "AGENT_PATH_SUGGEST_UNSUPPORTED"
    assert "secret remote command" not in str(failed)
    assert "手动" in failed["error"]["message"]


def test_root_suggestions_and_entry_prefix_validation(platform):
    node = platform.register()
    query_id = create(platform, node, prefix="/ma").json()["id"]
    roots = result(prefix="/ma", parent=None, entries=[{"name": "machine", "path": "/machine/", "kind": "directory"}])
    assert ack(platform, node, query_id, roots).status_code == 200
    assert get(platform, node, query_id).json()["status"] == "READY"
    query_id = create(platform, node, prefix="/machine/stat").json()["id"]
    assert ack(platform, node, query_id, result(prefix="/machine/stat")).status_code == 200
    assert get(platform, node, query_id).json()["error"]["code"] == "PATH_SUGGEST_INVALID_RESULT"


def test_historical_unsafe_result_is_not_exposed(platform):
    node = platform.register()
    query_id = create(platform, node).json()["id"]
    with platform.app.state.db.write() as session:
        command = session.get(Command, query_id)
        command.status, command.acknowledged_at = "APPLIED", now()
        command.result = result(entries=[{"name": "passwd", "path": "/etc/passwd", "kind": "file"}])
    assert get(platform, node, query_id).json()["status"] == "FAILED"


def test_explicit_filesystem_root_has_usable_root_name(platform):
    node = platform.register()
    with platform.app.state.db.write() as session:
        session.get(Node, node["node_id"]).roots = ["/"]
    query_id = create(platform, node, prefix="").json()["id"]
    value = result(prefix="", parent=None, entries=[{"name": "/", "path": "/", "kind": "directory"}])
    assert ack(platform, node, query_id, value).status_code == 200
    assert get(platform, node, query_id).json()["status"] == "READY"


@pytest.mark.parametrize(("prefix", "parent", "entries"), [
    ("/./ma", None, [{"name": "machine", "path": "/machine/", "kind": "directory"}]),
    ("//machine/", "/machine", [{"name": "foo", "path": "/machine/foo/", "kind": "directory"}]),
    ("/machine//fo", "/machine", [{"name": "foo", "path": "/machine/foo/", "kind": "directory"}]),
    ("/machine/./fo", "/machine", [{"name": "foo", "path": "/machine/foo/", "kind": "directory"}]),
    ("/machine/.", "/machine", [{"name": ".hidden", "path": "/machine/.hidden/", "kind": "directory"}]),
])
def test_normalized_agent_spelling_preserves_original_prefix(platform, prefix, parent, entries):
    node = platform.register()
    response = create(platform, node, prefix=prefix)
    assert response.status_code == 202
    query_id = response.json()["id"]
    assert ack(platform, node, query_id, result(prefix=prefix, parent=parent, entries=entries)).status_code == 200
    ready = get(platform, node, query_id).json()
    assert ready["status"] == "READY" and ready["prefix"] == prefix


def scoped_create(platform, node, prefix="", profile_id="default"):
    return platform.client.post(f"/api/v1/nodes/{node['node_id']}/path-suggestions", headers=ADMIN,
                                json={"prefix": prefix, "profile_id": profile_id})


def scoped_result(**changes):
    return {"prefix": "/machine/code/", "profile_id": "default", "base_path": "/machine/code",
            "parent": "/machine/code", "entries": [
                {"name": "scripts", "path": "/machine/code/scripts/", "kind": "directory"},
                {"name": "train.py", "path": "/machine/code/train.py", "kind": "file"}],
            "truncated": False, **changes}


def test_project_query_defaults_to_code_directory_and_filters_python_files(platform):
    node = platform.register(adapter="openpi-sft")
    response = scoped_create(platform, node)
    assert response.status_code == 202, response.text
    query = response.json()
    assert query["prefix"] == "/machine/code/"
    assert query["profile_id"] == "default" and query["base_path"] == "/machine/code"
    command = platform.commands(node)[0]
    assert command["payload"]["profile_id"] == "default"
    assert command["payload"]["base_path"] == "/machine/code"
    assert command["payload"]["python_only"] is True
    assert ack(platform, node, query["id"], scoped_result()).status_code == 200
    ready = get(platform, node, query["id"]).json()
    assert ready["status"] == "READY" and ready["entries"] == scoped_result()["entries"]
    assert ready["base_path"] == "/machine/code"


@pytest.mark.parametrize("prefix", ["/", "/mach", "/machine", "/machine/", "/machine/dataset", "/machine/code-other",
    "/machine/code/../dataset", "scripts/", "/machine/code/\n"])
def test_project_query_rejects_ancestors_siblings_and_traversal_before_dispatch(platform, prefix):
    node = platform.register(adapter="openpi-sft")
    response = scoped_create(platform, node, prefix=prefix)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "PATH_OUTSIDE_PROJECT"
    assert platform.commands(node) == []


@pytest.mark.parametrize("mutation", ["missing_echo", "wrong_profile", "wrong_base", "sibling", "non_python", "outside_parent"])
def test_project_ack_must_echo_scope_and_stay_inside_project(platform, mutation):
    node = platform.register(adapter="openpi-sft")
    query_id = scoped_create(platform, node).json()["id"]
    value = scoped_result()
    if mutation == "missing_echo":
        value.pop("profile_id")
        value.pop("base_path")
    elif mutation == "wrong_profile":
        value["profile_id"] = "other"
    elif mutation == "wrong_base":
        value["base_path"] = "/machine"
    elif mutation == "sibling":
        value["entries"] = [{"name": "other.py", "path": "/machine/other.py", "kind": "file"}]
    elif mutation == "non_python":
        value["entries"] = [{"name": "secrets.txt", "path": "/machine/code/secrets.txt", "kind": "file"}]
    else:
        value["parent"] = "/machine"
    assert ack(platform, node, query_id, value).status_code == 200
    failed = get(platform, node, query_id).json()
    assert failed["status"] == "FAILED" and failed["entries"] == []
    assert failed["error"]["code"] == "PATH_SUGGEST_INVALID_RESULT"


def test_project_queries_do_not_reuse_generic_or_another_profile_cache(platform):
    from vla_control.models import Profile

    node = platform.register(adapter="openpi-sft")
    with platform.app.state.db.write() as session:
        session.add(Profile(node_id=node["node_id"], id="other", name="other runtime", code_dir="/machine/code",
                            python="/usr/bin/python3", adapter="openpi-sft", metadata_json={}, execution_mode="native"))
    generic = create(platform, node, prefix="/machine/code/").json()["id"]
    scoped = scoped_create(platform, node).json()["id"]
    other = scoped_create(platform, node, profile_id="other").json()["id"]
    assert len({generic, scoped, other}) == 3
    assert scoped_create(platform, node).json()["id"] == scoped
    with platform.app.state.db.write() as session:
        session.get(Profile, (node["node_id"], "default")).code_dir = "/machine/another_project"
    changed = scoped_create(platform, node).json()
    assert changed["id"] != scoped and changed["base_path"] == "/machine/another_project"


def test_project_queries_require_active_openpi_profile_on_selected_node(platform):
    from vla_control.models import Profile

    node, other = platform.register(adapter="openpi-sft"), platform.register(adapter="openpi-sft", name="other")
    with platform.app.state.db.write() as session:
        session.add(Profile(node_id=other["node_id"], id="other-only", name="other runtime", code_dir="/machine/code",
                            python="/usr/bin/python3", adapter="openpi-sft", metadata_json={}, execution_mode="native"))
    response = scoped_create(platform, node, profile_id="other-only")
    assert response.status_code == 422 and response.json()["error"]["code"] == "PROFILE_NODE_MISMATCH"
    with platform.app.state.db.write() as session:
        session.get(Profile, (node["node_id"], "default")).active = False
    assert scoped_create(platform, node).json()["error"]["code"] == "PROFILE_NODE_MISMATCH"
    smoke = platform.register(name="smoke")
    assert scoped_create(platform, smoke).json()["error"]["code"] == "ENTRYPOINT_PROFILE_UNSUPPORTED"
    for profile_id in ("", "x" * 201):
        assert scoped_create(platform, other, profile_id=profile_id).status_code == 422


def test_profile_changes_invalidate_pending_and_historical_project_results(platform):
    from vla_control.models import Profile

    node = platform.register(adapter="openpi-sft")
    query_id = scoped_create(platform, node).json()["id"]
    assert ack(platform, node, query_id, scoped_result()).status_code == 200
    assert get(platform, node, query_id).json()["status"] == "READY"
    with platform.app.state.db.write() as session:
        session.get(Profile, (node["node_id"], "default")).code_dir = "/machine/another_project"
    failed = get(platform, node, query_id).json()
    assert failed["error"]["code"] == "PATH_PROFILE_CHANGED" and failed["entries"] == []
    assert ack(platform, node, query_id, scoped_result()).status_code == 200
    pending_id = scoped_create(platform, node).json()["id"]
    with platform.app.state.db.write() as session:
        session.get(Profile, (node["node_id"], "default")).active = False
    assert ack(platform, node, pending_id, scoped_result()).status_code == 200
    assert get(platform, node, pending_id).json()["error"]["code"] == "PATH_PROFILE_CHANGED"
