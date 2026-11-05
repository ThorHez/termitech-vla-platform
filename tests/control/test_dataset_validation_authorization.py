"""Manual and derived dataset checks retain their real initiating authority."""

from copy import deepcopy

import pytest
from vla_control.asset_transfer_models import AssetTransfer
from vla_control.common import now, uid
from vla_control.dataset_validations import INTERNAL_KEY
from vla_control.models import Asset, Command, Node, Profile

from .test_asset_transfers import create as copy_asset, fixture, run_copy
from .test_dataset_batches import batch, scan, submit
from .test_dataset_validations import ack, create, get, register, result, worker
from .test_resource_ownership import PASSWORD, accounts, login


def manual(platform):
    node = worker(platform)
    users = accounts(platform)
    users["remaining-admin"] = platform.app.state.auth.create_user("remaining-admin", PASSWORD, "admin")["id"]
    login(platform, "owner-admin")
    asset = register(platform, node)
    initial = asset["validation"]
    assert any(item["id"] == initial["id"] for item in platform.commands(node))
    assert ack(platform, node, initial, result(asset)).status_code == 200
    explicit = create(platform, asset)
    assert explicit.status_code == 202, explicit.text
    asset["validation"] = explicit.json()
    return node, users, asset


def command(platform, identifier):
    with platform.app.state.db.read() as session:
        return session.get(Command, identifier)


def test_explicit_revalidation_binds_admin_and_reuses_normal_pending(platform):
    node, users, asset = manual(platform)
    report = asset["validation"]
    original = command(platform, report["id"])
    assert original.actor_user_id == users["owner-admin"] and original.required_permission == "assets.write"
    assert "source_transfer_id" not in original.payload
    login(platform, "remaining-admin")
    assert create(platform, asset).json()["id"] == report["id"]
    assert command(platform, report["id"]).actor_user_id == users["owner-admin"]
    assert any(item["id"] == report["id"] for item in platform.commands(node))
    assert ack(platform, node, report, result(asset)).status_code == 200
    fresh = create(platform, asset).json()
    assert fresh["id"] != report["id"]
    assert command(platform, fresh["id"]).actor_user_id == users["remaining-admin"]


@pytest.mark.parametrize("change", ["disabled", "demoted", "unknown"])
def test_revoked_or_unknown_manual_validation_rejects_command_and_matching_report(platform, change):
    node, users, asset = manual(platform)
    report = asset["validation"]
    if change == "unknown":
        with platform.app.state.db.write() as session:
            row = session.get(Command, report["id"])
            row.actor_user_id = row.required_permission = None
    else:
        platform.app.state.auth.update_user(None, users["owner-admin"],
            {"enabled": False} if change == "disabled" else {"role": "operator"})
    assert not any(item["id"] == report["id"] for item in platform.commands(node))
    with platform.app.state.db.read() as session:
        old = session.get(Command, report["id"])
        stored = session.get(Asset, asset["id"])
        saved = deepcopy(stored.metadata_json)
        assert old.status == "REJECTED" and old.delivered_at is None
        assert saved[INTERNAL_KEY]["status"] == "FAILED" and saved[INTERNAL_KEY]["completed_at"]
        assert saved[INTERNAL_KEY]["error"]["code"] == (
            "COMMAND_ACTOR_REQUIRED" if change == "unknown" else "COMMAND_AUTHORIZATION_REVOKED")
        assert {key: value for key, value in saved.items() if key != INTERNAL_KEY} == asset["metadata"]
    # A late synthetic Agent success cannot revive rejected local-only work.
    assert ack(platform, node, report, result(asset)).status_code == 200
    with platform.app.state.db.read() as session:
        assert session.get(Asset, asset["id"]).metadata_json == saved
    login(platform, "remaining-admin")
    fresh = create(platform, asset).json()
    assert fresh["id"] != report["id"] and fresh["status"] == "PENDING"
    assert command(platform, fresh["id"]).actor_user_id == users["remaining-admin"]
    assert command(platform, report["id"]).actor_user_id == (None if change == "unknown" else users["owner-admin"])
    assert ack(platform, node, report, result(asset)).status_code == 200
    assert get(platform, asset)["id"] == fresh["id"] and get(platform, asset)["status"] == "PENDING"


@pytest.mark.parametrize("change", ["demoted", "unknown", "wrong-permission"])
def test_resubmit_replaces_unusable_pending_work_without_adopting_its_actor(platform, change):
    _, users, asset = manual(platform)
    previous = asset["validation"]
    if change == "demoted":
        platform.app.state.auth.update_user(None, users["owner-admin"], {"role": "operator"})
    else:
        with platform.app.state.db.write() as session:
            row = session.get(Command, previous["id"])
            if change == "unknown":
                row.actor_user_id = row.required_permission = None
            else:
                row.required_permission = "platform.read"
    login(platform, "remaining-admin")
    fresh = create(platform, asset).json()
    assert fresh["id"] != previous["id"] and fresh["status"] == "PENDING"
    old = command(platform, previous["id"])
    assert old.status == "REJECTED" and old.delivered_at is None
    assert old.actor_user_id == (None if change == "unknown" else users["owner-admin"])
    assert command(platform, fresh["id"]).actor_user_id == users["remaining-admin"]


def test_delivered_manual_validation_keeps_retry_reuse_and_late_ack_after_revocation(platform):
    node, users, asset = manual(platform)
    report = asset["validation"]
    assert any(item["id"] == report["id"] for item in platform.commands(node))
    platform.app.state.auth.update_user(None, users["owner-admin"], {"enabled": False})
    login(platform, "remaining-admin")
    assert create(platform, asset).json()["id"] == report["id"]
    assert any(item["id"] == report["id"] for item in platform.commands(node))
    assert command(platform, report["id"]).actor_user_id == users["owner-admin"]
    assert ack(platform, node, report, result(asset)).status_code == 200
    assert get(platform, asset)["status"] == "SUCCEEDED"


def test_validly_accepted_validation_survives_logout(platform):
    node, _, asset = manual(platform)
    report = asset["validation"]
    assert platform.client.post("/api/v1/auth/logout").status_code == 204
    assert any(item["id"] == report["id"] for item in platform.commands(node))
    assert ack(platform, node, report, result(asset)).status_code == 200


def test_batch_registration_auto_validations_bind_original_admin(batch):
    app, _, _, node_id, _, authenticated, _ = batch
    with app.state.db.write() as session:
        node = session.get(Node, node_id)
        node.status, node.last_seen, node.metrics = "ONLINE", now(), {"capabilities": ["DATASET_VALIDATE"]}
        session.add(Profile(node_id=node_id, id="default", name="validation fixture", code_dir="/machine/code",
            python="/usr/bin/python3", adapter="openpi-sft", recipe="pi0_base", execution_mode="native", metadata_json={}))
    ready = scan(batch)
    response = submit(batch, ready)
    assert response.status_code == 200, response.text
    created = response.json()["created"]
    assert len(created) == 2
    with app.state.db.read() as session:
        for asset in created:
            report = asset["validation"]
            queued = session.get(Command, report["id"])
            assert report["status"] == "PENDING"
            assert queued.actor_user_id == authenticated["user"]["id"] and queued.required_permission == "datasets.register"
            assert queued.payload["source_registration"] is True


def copied(fixture, monkeypatch, *, admin_intervention=False):
    platform, _, _, target, *_ = fixture
    with platform.app.state.db.write() as session:
        session.get(Node, target["node_id"]).metrics = {"capabilities": ["DATASET_VALIDATE"]}
    users = accounts(platform)
    users["remaining-admin"] = platform.app.state.auth.create_user("remaining-admin", PASSWORD, "admin")["id"]
    login(platform, "alice")
    response = copy_asset(fixture)
    assert response.status_code == 202, response.text
    item = response.json()
    if admin_intervention:
        with platform.app.state.db.write() as session:
            session.get(AssetTransfer, item["id"]).state = "INTERRUPTED"
        login(platform, "owner-admin")
        retried = platform.client.post("/api/v1/asset-transfers/" + item["id"] + "/retry")
        assert retried.status_code == 202, retried.text
    finished = run_copy(fixture, item["id"], monkeypatch)
    assert finished["state"] == "SUCCEEDED"
    asset = next(row for row in platform.client.get("/api/v1/assets").json() if row["id"] == finished["target_asset_id"])
    return platform, target, users, finished, asset, asset["validation"]


def test_operator_copy_derived_validation_uses_exact_output_without_granting_manual_rights(fixture, monkeypatch):
    platform, target, users, source, asset, report = copied(fixture, monkeypatch)
    queued = command(platform, report["id"])
    assert queued.actor_user_id == users["alice"] and queued.required_permission == "assets.transfer"
    assert queued.payload["source_transfer_id"] == source["id"]
    assert create(platform, asset).status_code == 403
    assert any(item["id"] == report["id"] for item in platform.commands(target))
    assert ack(platform, target, report, result(asset)).status_code == 200
    assert get(platform, asset)["status"] == "SUCCEEDED"


@pytest.mark.parametrize("change", [{"enabled": False}, {"role": "viewer"}])
def test_revoked_copy_actor_cannot_start_derived_validation_but_copy_stays_succeeded(fixture, monkeypatch, change):
    platform, target, users, source, asset, report = copied(fixture, monkeypatch)
    platform.app.state.auth.update_user(None, users["alice"], change)
    assert not any(item["id"] == report["id"] for item in platform.commands(target))
    with platform.app.state.db.read() as session:
        assert session.get(Command, report["id"]).status == "REJECTED"
        assert session.get(Asset, asset["id"]).metadata_json[INTERNAL_KEY]["status"] == "FAILED"
        assert session.get(AssetTransfer, source["id"]).state == "SUCCEEDED"


@pytest.mark.parametrize("corruption", ["missing-source", "target-id", "target-node", "actor", "provenance", "state"])
def test_forged_derived_validation_source_is_rejected(fixture, monkeypatch, corruption):
    platform, target, users, source, asset, report = copied(fixture, monkeypatch)
    with platform.app.state.db.write() as session:
        queued = session.get(Command, report["id"])
        transfer = session.get(AssetTransfer, source["id"])
        if corruption == "missing-source":
            queued.payload = {**queued.payload, "source_transfer_id": uid()}
        elif corruption == "target-id":
            transfer.target_asset_id = None
        elif corruption == "target-node":
            transfer.target_node_id = transfer.source_node_id
        elif corruption == "actor":
            queued.actor_user_id = users["bob"]
        elif corruption == "provenance":
            target_asset = session.get(Asset, asset["id"])
            metadata = deepcopy(target_asset.metadata_json)
            metadata["asset_sync"]["transfer_id"] = uid()
            target_asset.metadata_json = metadata
        else:
            transfer.state = "FAILED"
    assert not any(item["id"] == report["id"] for item in platform.commands(target))
    assert command(platform, report["id"]).status == "REJECTED"


def test_demoted_admin_cannot_start_validation_derived_from_intervention_on_another_owners_copy(fixture, monkeypatch):
    platform, target, users, source, _, report = copied(fixture, monkeypatch, admin_intervention=True)
    assert source["owner_user_id"] == users["alice"] and source["execution_user_id"] == users["owner-admin"]
    platform.app.state.auth.update_user(None, users["owner-admin"], {"role": "operator"})
    assert not any(item["id"] == report["id"] for item in platform.commands(target))
    assert command(platform, report["id"]).status == "REJECTED"
