"""Shared reads never grant another account control of executable workflows."""

import pytest
from sqlalchemy import select

from vla_control.asset_transfer_models import AssetTransfer
from vla_control.checkpoint_transfer_models import CheckpointTransfer
from vla_control.deployment_record_models import DeploymentRecord
from vla_control.models import Command
from vla_control.policy_deployment_models import PolicyDeployment

from .test_asset_transfers import create as copy_asset, fixture, run_copy
from .test_asset_transfer_batches import BATCH, body as batch_body, group
from .test_checkpoint_transfers import ack_export, complete, create as copy_checkpoint, transfer
from .test_policy_deployment_restart import enable_restart, terminal
from .test_policy_deployments import action, create as create_policy, ready
from .test_recipe_details import register


PASSWORD = "isolated-resource-ownership-password"


def accounts(platform):
    return {name: platform.app.state.auth.create_user(name, PASSWORD, role)["id"]
            for name, role in (("owner-admin", "admin"), ("alice", "operator"),
                               ("bob", "operator"), ("reader", "viewer"))}


def login(platform, username):
    client = platform.client
    client.base_url = "https://testserver"
    client.cookies.clear()
    client.headers.pop("X-Asset-Transfer-Batch-User", None)
    response = client.post("/api/v1/auth/login", json={"username": username, "password": PASSWORD})
    assert response.status_code == 200, response.text
    client.headers["X-CSRF-Token"] = response.json()["csrf_token"]


def denied(response):
    assert response.status_code == 403, response.text
    assert response.json()["error"]["code"] == "RESOURCE_ACCESS_DENIED"


def test_policy_controls_and_parent_associations_require_owner(platform):
    node = register(platform)
    users = accounts(platform)
    login(platform, "alice")
    deployment = ready(platform, node)
    assert deployment["owner_user_id"] == users["alice"]
    record_id = deployment["record_id"]
    login(platform, "bob")
    for verb in ("start", "stop", "refresh", "test"):
        denied(action(platform, deployment, verb, **({"observation_path": "/machine/sample.json"} if verb == "test" else {})))
    denied(platform.client.delete("/api/v1/deployment-records/" + record_id))
    denied(create_policy(platform, node, record_id=record_id, port=8001))
    login(platform, "reader")
    assert platform.client.get("/api/v1/policy-deployments/" + deployment["id"]).status_code == 200
    assert platform.client.get("/api/v1/deployment-records/" + record_id).json()["owner_user_id"] == users["alice"]
    assert action(platform, deployment, "stop").status_code == 403
    login(platform, "alice")
    assert action(platform, deployment, "stop").status_code == 202


def test_admin_restart_preserves_author_but_uses_execution_actors_gpu_rights(platform):
    node = register(platform)
    enable_restart(platform, node)
    users = accounts(platform)
    login(platform, "alice")
    deployment, _ = terminal(platform, node)
    reserve = platform.client.put(f"/api/v1/nodes/{node['node_id']}/gpu-reservations", json={"gpu_uuids": ["GPU-1"]})
    assert reserve.status_code == 200, reserve.text
    login(platform, "owner-admin")
    blocked = action(platform, deployment, "start")
    assert blocked.status_code == 409 and blocked.json()["error"]["code"] == "GPU_RESERVED_BY_OTHER_USER"
    released = platform.client.delete(f"/api/v1/nodes/{node['node_id']}/gpu-reservations/GPU-1")
    assert released.status_code == 200, released.text
    reserved = platform.client.put(f"/api/v1/nodes/{node['node_id']}/gpu-reservations", json={"gpu_uuids": ["GPU-1"]})
    assert reserved.status_code == 200, reserved.text
    response = action(platform, deployment, "start")
    assert response.status_code == 202, response.text
    assert response.json()["owner_user_id"] == users["alice"]
    assert response.json()["execution_user_id"] == users["owner-admin"]
    assert platform.client.get("/api/v1/deployment-records/" + deployment["record_id"]).json()["owner_user_id"] == users["alice"]


def test_historical_unowned_policy_and_record_are_admin_only(platform):
    node = register(platform)
    deployment, _ = terminal(platform, node)
    accounts(platform)
    login(platform, "bob")
    denied(action(platform, deployment, "refresh"))
    denied(platform.client.delete("/api/v1/deployment-records/" + deployment["record_id"]))
    login(platform, "owner-admin")
    assert platform.client.delete("/api/v1/deployment-records/" + deployment["record_id"]).status_code == 204


def test_asset_copy_ownership_and_account_scoped_idempotency(fixture):
    platform, *_ = fixture
    users = accounts(platform)
    login(platform, "alice")
    first = copy_asset(fixture, idempotency_key="shared-browser-key").json()
    assert first["owner_user_id"] == users["alice"]
    assert copy_asset(fixture, idempotency_key="shared-browser-key").json() == first
    login(platform, "bob")
    other = copy_asset(fixture, idempotency_key="shared-browser-key", target_name="bob-copy")
    assert other.status_code == 202, other.text
    assert other.json()["id"] != first["id"] and other.json()["owner_user_id"] == users["bob"]
    path = "/api/v1/asset-transfers/" + first["id"]
    denied(platform.client.post(path + "/cancel"))
    denied(platform.client.post(path + "/retry"))
    denied(platform.client.delete(path))
    assert platform.client.get(path).status_code == 200
    login(platform, "owner-admin")
    assert platform.client.post(path + "/cancel").status_code == 202
    retried = platform.client.post(path + "/retry")
    assert retried.status_code == 202 and retried.json()["owner_user_id"] == users["alice"]
    login(platform, "alice")
    assert platform.client.post(path + "/cancel").status_code == 202
    assert platform.client.delete(path).status_code == 204


def test_batch_children_keep_personal_ownership(group):
    platform = group[0][0]
    response = group[1].post(BATCH, json=batch_body(group))
    assert response.status_code == 202, response.text
    children = response.json()["transfers"]
    assert all(row["owner_user_id"] == group[4] for row in children)
    accounts(platform)
    login(platform, "bob")
    for child in children:
        denied(platform.client.post("/api/v1/asset-transfers/" + child["id"] + "/cancel"))
    with platform.app.state.db.read() as session:
        assert {row.owner_user_id for row in session.scalars(select(AssetTransfer))} == {group[4]}


def test_checkpoint_copy_can_use_shared_source_but_cannot_attach_other_workflow(transfer):
    platform, _, _, target, run, *_ = transfer
    users = accounts(platform)
    assert run.owner_user_id is None  # Shared historical trained model remains usable.
    login(platform, "alice")
    first = copy_checkpoint(transfer).json()
    assert first["owner_user_id"] == users["alice"]
    parent = platform.client.get("/api/v1/deployment-records/" + first["record_id"]).json()
    assert parent["owner_user_id"] == users["alice"]
    login(platform, "bob")
    path = "/api/v1/checkpoint-transfers/" + first["id"]
    denied(platform.client.post(path + "/cancel"))
    denied(platform.client.post(path + "/retry"))
    denied(copy_checkpoint(transfer, record_id=first["record_id"]))
    denied(create_policy(platform, target, transfer_id=first["id"], source_run_id=run.id))
    second = copy_checkpoint(transfer, target_directory="/machine/bob-models")
    assert second.status_code == 202, second.text
    assert second.json()["id"] != first["id"] and second.json()["owner_user_id"] == users["bob"]
    login(platform, "owner-admin")
    assert platform.client.post(path + "/cancel").status_code == 202
    restarted = platform.client.post(path + "/retry")
    assert restarted.status_code == 202, restarted.text
    assert restarted.json()["owner_user_id"] == users["alice"]
    added = copy_checkpoint(transfer, record_id=first["record_id"], source_checkpoint_path="/machine/output/owned/20000")
    assert added.status_code == 202, added.text
    assert added.json()["owner_user_id"] == users["alice"]
    login(platform, "alice")
    assert platform.client.post("/api/v1/checkpoint-transfers/" + added.json()["id"] + "/cancel").status_code == 202


def test_admin_added_runtime_inherits_existing_workflow_owner(transfer):
    platform, _, _, target, *_ = transfer
    users = accounts(platform)
    login(platform, "alice")
    child = copy_checkpoint(transfer).json()
    login(platform, "owner-admin")
    response = create_policy(platform, target, record_id=child["record_id"])
    assert response.status_code == 202, response.text
    deployment = response.json()
    assert deployment["owner_user_id"] == users["alice"]
    assert deployment["execution_user_id"] == users["owner-admin"]
    login(platform, "alice")
    assert action(platform, deployment, "stop").status_code == 202


@pytest.mark.parametrize("kind", ["asset", "checkpoint"])
def test_historical_unowned_transfers_remain_admin_managed(kind, fixture, transfer):
    selected = fixture if kind == "asset" else transfer
    platform, *_ = selected
    response = copy_asset(selected) if kind == "asset" else copy_checkpoint(selected)
    item = response.json()
    assert item["owner_user_id"] is None
    accounts(platform)
    login(platform, "bob")
    path = "/api/v1/" + ("asset-transfers/" if kind == "asset" else "checkpoint-transfers/") + item["id"]
    denied(platform.client.post(path + "/cancel"))
    denied(platform.client.post(path + "/retry"))
    login(platform, "owner-admin")
    cancelled = platform.client.post(path + "/cancel")
    assert cancelled.status_code == 202 and cancelled.json()["owner_user_id"] is None


@pytest.mark.parametrize("kind", ["asset", "checkpoint"])
def test_revoked_account_cannot_start_queued_transfer(kind, fixture, transfer, monkeypatch):
    selected = fixture if kind == "asset" else transfer
    platform, manager, *_ = selected
    users = accounts(platform)
    login(platform, "alice")
    response = copy_asset(selected) if kind == "asset" else copy_checkpoint(selected)
    assert response.status_code == 202, response.text
    item = response.json()
    platform.app.state.auth.update_user(None, users["alice"], {"enabled": False})
    monkeypatch.setattr(manager.remote_manager, "_remote", lambda *_: pytest.fail("Revoked queued work must not open SSH"))
    manager._run(item["id"])
    model = AssetTransfer if kind == "asset" else CheckpointTransfer
    with platform.app.state.db.read() as session:
        row = session.get(model, item["id"])
        assert row.state == "INTERRUPTED" and row.error["code"] == "ACCOUNT_EXECUTION_REVOKED"
        assert row.owner_user_id == users["alice"]
        if kind == "checkpoint":
            commands = list(session.scalars(select(Command).where(Command.type == "POLICY_EXPORT")))
            assert commands and all(command.status == "REJECTED" for command in commands)


def test_stale_account_header_never_enqueues_transfer(fixture):
    platform, *_ = fixture
    users = accounts(platform)
    login(platform, "bob")
    platform.client.headers["X-Operation-User"] = users["alice"]
    response = copy_asset(fixture)
    assert response.status_code == 403 and response.json()["error"]["code"] == "ACCOUNT_CHANGED"
    with platform.app.state.db.read() as session:
        assert list(session.scalars(select(AssetTransfer))) == []


@pytest.mark.parametrize("historical", [False, True])
def test_admin_can_start_disabled_or_historical_authors_policy_without_claiming_it(platform, historical):
    node = register(platform)
    enable_restart(platform, node)
    if historical:
        deployment, _ = terminal(platform, node)
        users = accounts(platform)
        owner = None
    else:
        users = accounts(platform)
        login(platform, "alice")
        deployment, _ = terminal(platform, node)
        owner = users["alice"]
        platform.app.state.auth.update_user(None, owner, {"enabled": False})
    login(platform, "bob")
    denied(action(platform, deployment, "start"))
    login(platform, "owner-admin")
    response = action(platform, deployment, "start")
    assert response.status_code == 202, response.text
    assert response.json()["owner_user_id"] == owner
    assert response.json()["execution_user_id"] == users["owner-admin"]
    delivered = platform.commands(node)
    assert any(command["type"] == "POLICY_DEPLOY_START" and command["payload"]["deployment_id"] == deployment["id"]
               for command in delivered)
    assert platform.client.get("/api/v1/deployment-records/" + deployment["record_id"]).json()["owner_user_id"] == owner


@pytest.mark.parametrize("kind", ["asset", "checkpoint"])
@pytest.mark.parametrize("historical", [False, True])
def test_admin_retry_executes_disabled_or_historical_authors_transfer(kind, historical, fixture, transfer, monkeypatch):
    selected = fixture if kind == "asset" else transfer
    platform, _, source, *_ = selected
    if historical:
        response = copy_asset(selected) if kind == "asset" else copy_checkpoint(selected)
        users = accounts(platform)
        owner = None
    else:
        users = accounts(platform)
        login(platform, "alice")
        response = copy_asset(selected) if kind == "asset" else copy_checkpoint(selected)
        owner = users["alice"]
        platform.app.state.auth.update_user(None, owner, {"enabled": False})
    assert response.status_code == 202, response.text
    original = response.json()
    model = AssetTransfer if kind == "asset" else CheckpointTransfer
    with platform.app.state.db.write() as session:
        session.get(model, original["id"]).state = "INTERRUPTED"
    base = "/api/v1/" + ("asset-transfers/" if kind == "asset" else "checkpoint-transfers/") + original["id"]
    login(platform, "bob")
    denied(platform.client.post(base + "/retry"))
    login(platform, "owner-admin")
    response = platform.client.post(base + "/retry")
    assert response.status_code == 202, response.text
    retried = response.json()
    assert retried["owner_user_id"] == owner and retried["execution_user_id"] == users["owner-admin"]
    if kind == "asset":
        completed = run_copy(selected, retried["id"], monkeypatch)
    else:
        export = next(command for command in platform.commands(source)
                      if command["type"] == "POLICY_EXPORT" and command["payload"]["transfer_id"] == retried["id"])
        assert ack_export(selected, export).status_code == 200
        complete(selected, retried)
        completed = platform.client.get(base).json()
    assert completed["state"] == "SUCCEEDED", completed
    assert completed["owner_user_id"] == owner and completed["execution_user_id"] == users["owner-admin"]


@pytest.mark.parametrize("kind", ["asset", "checkpoint"])
def test_demoted_admin_cannot_start_pending_intervention_on_another_authors_transfer(kind, fixture, transfer, monkeypatch):
    selected = fixture if kind == "asset" else transfer
    platform, manager, *_ = selected
    users = accounts(platform)
    platform.app.state.auth.create_user("remaining-admin", PASSWORD, "admin")
    login(platform, "alice")
    response = copy_asset(selected) if kind == "asset" else copy_checkpoint(selected)
    assert response.status_code == 202, response.text
    original = response.json()
    model = AssetTransfer if kind == "asset" else CheckpointTransfer
    with platform.app.state.db.write() as session:
        session.get(model, original["id"]).state = "INTERRUPTED"
    login(platform, "owner-admin")
    base = "/api/v1/" + ("asset-transfers/" if kind == "asset" else "checkpoint-transfers/") + original["id"]
    retried = platform.client.post(base + "/retry")
    assert retried.status_code == 202, retried.text
    platform.app.state.auth.update_user(None, users["owner-admin"], {"role": "operator"})
    monkeypatch.setattr(manager.remote_manager, "_remote", lambda *_: pytest.fail("Revoked admin intervention must not open SSH"))
    manager._run(original["id"])
    with platform.app.state.db.read() as session:
        row = session.get(model, original["id"])
        assert row.state == "INTERRUPTED" and row.error["code"] == "ACCOUNT_EXECUTION_REVOKED"
        assert row.owner_user_id == users["alice"] and row.execution_user_id == users["owner-admin"]
