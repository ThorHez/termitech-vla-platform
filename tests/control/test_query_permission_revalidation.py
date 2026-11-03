"""Machine queries keep their initiating account through admission and reuse."""

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import delete, select

from vla_control.auth_models import UserSession
from vla_control.common import offset_time
from vla_control.models import Command
from vla_control.resource_authorization import query_reusable

from .test_data_processors import result as catalog_result
from .test_dataset_inspection_queries import result as inspection_result
from .test_path_suggestions import result as path_result
from .test_recipe_details import register, result as detail_result
from .test_recipe_suggestions import result as recipe_result


PASSWORD = "query-revalidation-test-password"
SURFACES = {
    "paths": ("path-suggestions", {"prefix": "/machine/", "limit": 30}, "PATH_SUGGEST", "assets.query", path_result),
    "recipes": ("recipe-suggestions", {"profile_id": "default", "prefix": "pi0", "limit": 30}, "RECIPE_SUGGEST", "recipes.query", recipe_result),
    "inference": ("inference-config-suggestions", {"profile_id": "default", "prefix": "pi0", "limit": 30}, "RECIPE_SUGGEST", "training.write", recipe_result),
    "details": ("recipe-details", {"profile_id": "default", "recipe": "pi0_base"}, "RECIPE_DETAIL", "recipes.query", detail_result),
    "processors": ("data-processors", {"profile_id": "default"}, "DATA_PROCESSORS", "recipes.query", catalog_result),
    "inspections": ("dataset-inspections", {"path": "/machine/data"}, "DATASET_INSPECT", "assets.query", inspection_result),
}


@pytest.fixture(params=list(SURFACES))
def query(platform, request):
    node = register(platform)
    suffix, body, kind, permission, result = SURFACES[request.param]
    path = f"/api/v1/nodes/{node['node_id']}/{suffix}"
    users, clients = {}, {}
    for name, role in (("admin", "admin"), ("alice", "operator"), ("bob", "operator")):
        users[name] = platform.app.state.auth.create_user(name, PASSWORD, role)
        client = TestClient(platform.app, base_url="https://testserver")
        login = client.post("/api/v1/auth/login", json={"username": name, "password": PASSWORD})
        assert login.status_code == 200, login.text
        client.headers["X-CSRF-Token"] = login.json()["csrf_token"]
        clients[name] = client
    try:
        yield platform, node, path, body, kind, permission, result, users, clients
    finally:
        for client in clients.values():
            client.close()


def accepted(client, path, body):
    response = client.post(path, json=body)
    assert response.status_code == 202, response.text
    return response.json()


def revoke(platform, user_id, change):
    if change in {"disabled", "demoted"}:
        platform.app.state.auth.update_user(None, user_id,
            {"enabled": False} if change == "disabled" else {"role": "viewer"})
    else:
        with platform.app.state.db.write() as session:
            if change == "logout":
                session.execute(delete(UserSession).where(UserSession.user_id == user_id))
            else:
                active = session.scalar(select(UserSession).where(UserSession.user_id == user_id))
                active.expires_at = offset_time(-1)


@pytest.mark.parametrize("change", ["disabled", "demoted", "logout", "expired"])
def test_query_submission_rechecks_actor_after_dependency(query, monkeypatch, change):
    platform, _, path, body, kind, _, _, users, clients = query
    original = platform.app.state.auth.require
    changed = []

    def revoke_after_dependency(request, permission=None):
        principal = original(request, permission)
        if request.url.path == path and not changed:
            changed.append(True)
            revoke(platform, users["alice"]["id"], change)
        return principal

    monkeypatch.setattr(platform.app.state.auth, "require", revoke_after_dependency)
    response = clients["alice"].post(path, json=body)
    assert changed and response.status_code in {401, 403}, response.text
    with platform.app.state.db.read() as session:
        assert list(session.scalars(select(Command).where(Command.type == kind))) == []


def test_query_dedup_keeps_original_actor(query):
    platform, _, path, body, kind, permission, _, users, clients = query
    first = accepted(clients["alice"], path, body)
    second = accepted(clients["bob"], path, body)
    assert second["id"] == first["id"]
    with platform.app.state.db.read() as session:
        rows = list(session.scalars(select(Command).where(Command.type == kind)))
        assert len(rows) == 1
        assert rows[0].actor_user_id == users["alice"]["id"]
        assert rows[0].required_permission == permission


@pytest.mark.parametrize("denial", ["disabled", "demoted", "unknown", "metadata"])
def test_query_first_delivery_rechecks_original_account(query, denial):
    platform, node, path, body, _, _, _, users, clients = query
    first = accepted(clients["alice"], path, body)
    if denial in {"disabled", "demoted"}:
        revoke(platform, users["alice"]["id"], denial)
    else:
        with platform.app.state.db.write() as session:
            command = session.get(Command, first["id"])
            if denial == "unknown":
                command.actor_user_id = None
            else:
                command.required_permission = None
    assert first["id"] not in {row["id"] for row in platform.commands(node)}
    failed = clients["bob"].get(path + "/" + first["id"]).json()
    expected = "COMMAND_ACTOR_REQUIRED" if denial == "unknown" else "COMMAND_AUTHORIZATION_REVOKED"
    assert failed["status"] == "FAILED" and failed["error"]["code"] == expected
    assert "查询" in failed["error"]["message"] or "账户" in failed["error"]["message"]
    with platform.app.state.db.read() as session:
        command = session.get(Command, first["id"])
        assert command.status == "REJECTED" and command.delivered_at is None
        assert command.actor_user_id == (None if denial == "unknown" else users["alice"]["id"])


@pytest.mark.parametrize("denial", ["demoted", "unknown"])
def test_query_reuse_retires_only_invalid_undelivered_command(query, denial):
    platform, _, path, body, _, permission, _, users, clients = query
    first = accepted(clients["alice"], path, body)
    if denial == "demoted":
        revoke(platform, users["alice"]["id"], denial)
    else:
        with platform.app.state.db.write() as session:
            session.get(Command, first["id"]).actor_user_id = None
    second = accepted(clients["bob"], path, body)
    assert second["id"] != first["id"]
    with platform.app.state.db.read() as session:
        old, fresh = session.get(Command, first["id"]), session.get(Command, second["id"])
        assert old.status == "REJECTED" and old.delivered_at is None
        assert old.result["error"]["code"] == (
            "COMMAND_AUTHORIZATION_REVOKED" if denial == "demoted" else "COMMAND_ACTOR_REQUIRED")
        assert fresh.actor_user_id == users["bob"]["id"] and fresh.required_permission == permission
    failed = clients["bob"].get(path + "/" + first["id"]).json()
    assert failed["status"] == "FAILED" and failed["error"]["code"] == old.result["error"]["code"]


def test_delivered_query_retries_keep_actor_after_revocation(query):
    platform, node, path, body, _, _, _, users, clients = query
    first = accepted(clients["alice"], path, body)
    assert first["id"] in {row["id"] for row in platform.commands(node)}
    revoke(platform, users["alice"]["id"], "demoted")
    assert accepted(clients["bob"], path, body)["id"] == first["id"]
    assert first["id"] in {row["id"] for row in platform.commands(node)}
    with platform.app.state.db.read() as session:
        command = session.get(Command, first["id"])
        assert command.actor_user_id == users["alice"]["id"] and command.delivered_at is not None


def test_accepted_query_survives_normal_logout(query):
    platform, node, path, body, _, _, _, users, clients = query
    first = accepted(clients["alice"], path, body)
    assert clients["alice"].post("/api/v1/auth/logout").status_code == 204
    assert accepted(clients["bob"], path, body)["id"] == first["id"]
    assert first["id"] in {row["id"] for row in platform.commands(node)}
    with platform.app.state.db.read() as session:
        assert session.get(Command, first["id"]).actor_user_id == users["alice"]["id"]


def test_authorization_rejection_cannot_be_revived_by_late_query_ack(query):
    platform, node, path, body, _, _, result, users, clients = query
    first = accepted(clients["alice"], path, body)
    revoke(platform, users["alice"]["id"], "demoted")
    accepted(clients["bob"], path, body)  # Retire Alice's undelivered command during dedup.
    ack = platform.client.post(f"/agent/v1/commands/{first['id']}/ack", headers=node["headers"],
        json={"node_id": node["node_id"], "status": "APPLIED", "result": result()})
    assert ack.status_code == 200, ack.text
    failed = clients["bob"].get(path + "/" + first["id"]).json()
    assert failed["status"] == "FAILED" and failed["error"]["code"] == "COMMAND_AUTHORIZATION_REVOKED"


def test_applied_query_cache_remains_shared_without_adopting_actor(query):
    platform, node, path, body, _, _, result, users, clients = query
    first = accepted(clients["alice"], path, body)
    assert first["id"] in {row["id"] for row in platform.commands(node)}
    ack = platform.client.post(f"/agent/v1/commands/{first['id']}/ack", headers=node["headers"],
        json={"node_id": node["node_id"], "status": "APPLIED", "result": result()})
    assert ack.status_code == 200, ack.text
    revoke(platform, users["alice"]["id"], "demoted")
    cached = accepted(clients["bob"], path, body)
    assert cached["id"] == first["id"] and cached["status"] == "READY"
    with platform.app.state.db.read() as session:
        assert session.get(Command, first["id"]).actor_user_id == users["alice"]["id"]


@pytest.mark.parametrize("query", ["recipes"], indirect=True)
def test_recipe_and_inference_queries_do_not_share_pending_execution_permission(query):
    platform, _, path, body, _, _, _, users, clients = query
    regular = accepted(clients["alice"], path, body)
    inference = accepted(clients["bob"], path.replace("recipe-suggestions", "inference-config-suggestions"), body)
    assert inference["id"] != regular["id"]
    with platform.app.state.db.write() as session:
        original = session.get(Command, regular["id"])
        assert not query_reusable(session, original, None, "training.write")
        assert original.status == "PENDING" and original.actor_user_id == users["alice"]["id"]
        assert original.required_permission == "recipes.query"
        assert session.get(Command, inference["id"]).required_permission == "training.write"
