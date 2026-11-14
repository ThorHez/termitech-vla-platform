"""The accepted shared-read, own-task-write contract uses real routed objects."""

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select

from vla_control.models import Idempotency, Preview, Run

from .conftest import Platform


PASSWORD = "multi-user-contract-password"


@pytest.fixture
def team(platform):
    node = platform.register()
    users, clients = {}, {}
    for username, role in (("team-admin", "admin"), ("alice", "operator"),
                           ("bob", "operator"), ("reader", "viewer")):
        users[username] = platform.app.state.auth.create_user(username, PASSWORD, role)
        client = TestClient(platform.app, base_url="https://testserver")
        login = client.post("/api/v1/auth/login", json={"username": username, "password": PASSWORD})
        assert login.status_code == 200, login.text
        client.headers["X-CSRF-Token"] = login.json()["csrf_token"]
        clients[username] = client
    try:
        yield platform, node, users, clients
    finally:
        for client in clients.values():
            client.close()


def actor(team, name="alice"):
    platform, _, _, clients = team
    return Platform(clients[name], platform.app, platform.settings)


def submit(client, preview, key="same-client-key"):
    return client.post("/api/v1/runs", headers={"Idempotency-Key": key}, json={"preflight_id": preview["id"]})


@pytest.mark.parametrize("outsider", ["bob", "reader"])
def test_other_accounts_cannot_cancel_or_delete_but_can_read(team, outsider):
    _, node, users, clients = team
    run = actor(team).run(node)
    path = f"/api/v1/runs/{run['id']}"
    assert run["owner_user_id"] == users["alice"]["id"]
    assert clients[outsider].get(path).status_code == 200
    assert clients[outsider].get(path + "/logs").status_code == 200
    denied = clients[outsider].post(path + "/cancel")
    assert denied.status_code == 403, denied.text
    assert clients["alice"].get(path).json()["state"] == "QUEUED"
    assert clients["alice"].post(path + "/cancel").status_code == 202
    assert clients[outsider].delete(path).status_code == 403
    assert clients["alice"].delete(path).status_code == 204


@pytest.mark.parametrize("owner", ["alice", None])
def test_admin_manages_owned_and_historical_tasks_without_claiming_them(team, owner):
    platform, node, users, clients = team
    run = actor(team).run(node)
    if owner is None:
        with platform.app.state.db.write() as session:
            session.get(Run, run["id"]).owner_user_id = None
    path = f"/api/v1/runs/{run['id']}"
    if owner is None:
        assert clients["alice"].post(path + "/cancel").status_code == 403
    cancelled = clients["team-admin"].post(path + "/cancel")
    assert cancelled.status_code == 202, cancelled.text
    assert cancelled.json()["owner_user_id"] == (users[owner]["id"] if owner else None)
    assert clients["team-admin"].delete(path).status_code == 204


@pytest.mark.parametrize("outsider", ["bob", "team-admin"])
def test_preflight_cannot_be_submitted_by_another_account_even_admin(team, outsider):
    platform, node, _, clients = team
    preview = actor(team).preview(node)
    assert clients[outsider].get(f"/api/v1/run-previews/{preview['id']}").status_code == 200
    denied = submit(clients[outsider], preview)
    assert denied.status_code == 403
    assert denied.json()["error"]["code"] == "PREFLIGHT_ACCOUNT_MISMATCH"
    with platform.app.state.db.read() as session:
        assert session.get(Preview, preview["id"]).run_id is None
    assert submit(clients["alice"], preview).status_code == 201


def test_same_idempotency_key_is_independent_per_account_and_replays_own_run(team):
    _, node, users, clients = team
    previews = {name: actor(team, name).preview(node) for name in ("alice", "bob")}
    runs = {name: submit(clients[name], previews[name]).json() for name in previews}
    assert runs["alice"]["id"] != runs["bob"]["id"]
    for name in previews:
        replay = submit(clients[name], previews[name])
        assert replay.status_code == 201, replay.text
        assert replay.json()["id"] == runs[name]["id"]
        assert replay.json()["owner_user_id"] == users[name]["id"]


@pytest.mark.parametrize("name", ["alice", "bob", "reader"])
@pytest.mark.parametrize("kind", ["DATASET", "NORM", "MODEL"])
def test_existing_public_assets_and_recipes_are_admin_managed(team, name, kind):
    from .test_dataset_profiles import profile

    _, node, _, clients = team
    admin = clients["team-admin"]
    asset = admin.post("/api/v1/assets", json={"node_id": node["node_id"], "name": "public dataset",
        "kind": kind, "path": "/machine/assets/public", "version": "v1", "metadata": {}})
    assert asset.status_code == 201, asset.text
    identifier = asset.json()["id"]
    assert asset.json()["registered_by_user_id"] == team[2]["team-admin"]["id"]
    for method, path, body in (
        ("POST", f"/assets/{identifier}/archive", {}),
        ("PATCH", f"/assets/{identifier}/dataset-profile", {"dataset_profile": profile()}),
        ("PUT", f"/assets/{identifier}/norm-binding", {"norm_asset_id": None}),
        ("POST", "/recipes", {}),
        ("POST", "/recipes/unknown/revisions", {}),
    ):
        result = clients[name].request(method, "/api/v1" + path, json=body)
        expected = 409 if name != "reader" and kind != "DATASET" and method in {"PATCH", "PUT"} else 403
        assert result.status_code == expected, (method, path, result.text)
    validation = clients[name].post(f"/api/v1/assets/{identifier}/validations", json={})
    assert validation.status_code == (403 if name == "reader" else 202 if kind == "DATASET" else 404), validation.text
    assert clients[name].get("/api/v1/assets").status_code == 200
    assert clients[name].get("/api/v1/recipes").status_code == 200
    assert admin.post(f"/api/v1/assets/{identifier}/archive").status_code == 200


def test_account_assertion_rejects_a_delayed_old_account_mutation(team):
    _, node, users, clients = team
    run = actor(team, "bob").run(node)
    path = f"/api/v1/runs/{run['id']}/cancel"
    denied = clients["bob"].post(path, headers={"X-Operation-User": users["alice"]["id"]})
    assert denied.status_code == 403
    assert denied.json()["error"]["code"] == "ACCOUNT_CHANGED"
    assert clients["bob"].get(f"/api/v1/runs/{run['id']}").json()["state"] == "QUEUED"
    assert clients["bob"].post(path, headers={"X-Operation-User": users["bob"]["id"]}).status_code == 202


def test_members_exposes_only_display_identity_to_shared_readers(team):
    _, _, users, clients = team
    result = clients["reader"].get("/api/v1/members")
    assert result.status_code == 200
    assert {row["id"] for row in result.json()} == {row["id"] for row in users.values()}
    assert all(set(row) == {"id", "username", "display_name"} for row in result.json())


def test_receipt_from_before_account_scoping_replays_only_for_its_owner(team):
    platform, node, _, clients = team
    preview = actor(team).preview(node)
    first = submit(clients["alice"], preview, "old-global-key")
    assert first.status_code == 201, first.text
    with platform.app.state.db.write() as session:
        receipt = session.scalar(select(Idempotency).where(Idempotency.run_id == first.json()["id"]))
        receipt.key = "old-global-key"
    own = submit(clients["alice"], preview, "old-global-key")
    assert own.status_code == 201 and own.json()["id"] == first.json()["id"]
    other_preview = actor(team, "bob").preview(node)
    other = submit(clients["bob"], other_preview, "old-global-key")
    assert other.status_code == 201 and other.json()["id"] != first.json()["id"]
