"""Registration queues a bounded check with its original actor."""

from copy import deepcopy

import pytest

from vla_control.common import uid
from vla_control.dataset_validations import INTERNAL_KEY, queue_validation
from vla_control.models import Asset, Command

from .test_dataset_validations import ack, create, get, register, result, worker
from .test_resource_ownership import PASSWORD, accounts, login


def registered(platform, username="alice"):
    node = worker(platform)
    users = accounts(platform)
    users["remaining-admin"] = platform.app.state.auth.create_user("remaining-admin", PASSWORD, "admin")["id"]
    login(platform, username)
    return node, users, register(platform, node)


def queued(platform, asset):
    with platform.app.state.db.read() as session:
        return session.get(Command, asset["validation"]["id"])


def test_operator_registration_queues_only_its_new_dataset_check(platform):
    node, users, asset = registered(platform)
    command = queued(platform, asset)
    assert command.actor_user_id == users["alice"] and command.required_permission == "datasets.register"
    assert command.payload["source_registration"] is True and "source_transfer_id" not in command.payload
    receipt = asset["validation"]["registration"]
    assert receipt["asset_id"] == asset["id"] and receipt["actor_user_id"] == users["alice"]
    pending = create(platform, asset)
    assert pending.status_code == 202, pending.text
    assert pending.json()["id"] == command.id
    assert platform.client.post(f"/api/v1/assets/{asset['id']}/validation/cancel").status_code == 403
    assert any(item["id"] == command.id for item in platform.commands(node))
    assert ack(platform, node, asset["validation"], result(asset)).status_code == 200
    assert get(platform, asset)["status"] == "SUCCEEDED"
    fresh = create(platform, asset)
    assert fresh.status_code == 202, fresh.text
    assert fresh.json()["id"] != command.id


@pytest.mark.parametrize("change", [{"enabled": False}, {"role": "viewer"}])
def test_registration_validation_is_rejected_before_delivery_after_revocation(platform, change):
    node, users, asset = registered(platform)
    platform.app.state.auth.update_user(None, users["alice"], change)
    assert not any(item["id"] == asset["validation"]["id"] for item in platform.commands(node))
    with platform.app.state.db.read() as session:
        command = session.get(Command, asset["validation"]["id"])
        stored = session.get(Asset, asset["id"])
        assert command.status == "REJECTED" and command.delivered_at is None
        assert stored.state == "REGISTERED"
        assert stored.metadata_json[INTERNAL_KEY]["status"] == "FAILED"
        saved = deepcopy(stored.metadata_json)
    assert ack(platform, node, asset["validation"], result(asset)).status_code == 200
    with platform.app.state.db.read() as session:
        assert session.get(Asset, asset["id"]).metadata_json == saved


def test_admin_demoted_to_operator_retains_accepted_registration_rights(platform):
    node, users, asset = registered(platform, "owner-admin")
    platform.app.state.auth.update_user(None, users["owner-admin"], {"role": "operator"})
    assert any(item["id"] == asset["validation"]["id"] for item in platform.commands(node))
    assert ack(platform, node, asset["validation"], result(asset)).status_code == 200
    login(platform, "owner-admin")
    assert get(platform, asset)["status"] == "SUCCEEDED"
    assert create(platform, asset).status_code == 202


@pytest.mark.parametrize("change", ["disabled", "logout"])
def test_delivered_registration_retains_retry_and_late_ack(platform, change):
    node, users, asset = registered(platform)
    identifier = asset["validation"]["id"]
    assert any(item["id"] == identifier for item in platform.commands(node))
    if change == "disabled":
        platform.app.state.auth.update_user(None, users["alice"], {"enabled": False})
    else:
        assert platform.client.post("/api/v1/auth/logout").status_code == 204
    assert any(item["id"] == identifier for item in platform.commands(node))
    assert ack(platform, node, asset["validation"], result(asset)).status_code == 200
    with platform.app.state.db.read() as session:
        assert session.get(Asset, asset["id"]).metadata_json[INTERNAL_KEY]["status"] == "SUCCEEDED"


def test_accepted_registration_survives_logout_before_delivery(platform):
    node, _, asset = registered(platform)
    assert platform.client.post("/api/v1/auth/logout").status_code == 204
    assert any(item["id"] == asset["validation"]["id"] for item in platform.commands(node))


def test_admin_revalidation_reuses_valid_registration_then_replaces_revoked_actor_without_adoption(platform):
    node, users, asset = registered(platform)
    old = asset["validation"]
    login(platform, "remaining-admin")
    assert create(platform, asset).json()["id"] == old["id"]
    assert queued(platform, asset).actor_user_id == users["alice"]
    platform.app.state.auth.update_user(None, users["alice"], {"role": "viewer"})
    fresh = create(platform, asset)
    assert fresh.status_code == 202, fresh.text
    assert fresh.json()["id"] != old["id"] and "registration" not in fresh.json()
    with platform.app.state.db.read() as session:
        original = session.get(Command, old["id"])
        replacement = session.get(Command, fresh.json()["id"])
        assert original.status == "REJECTED" and original.actor_user_id == users["alice"]
        assert replacement.actor_user_id == users["remaining-admin"] and replacement.required_permission == "assets.write"
    assert ack(platform, node, old, result(asset)).status_code == 200
    assert get(platform, asset)["id"] == fresh.json()["id"] and get(platform, asset)["status"] == "PENDING"


@pytest.mark.parametrize("corruption", ["asset", "node", "path", "digest", "profile", "actor", "created-at",
    "report-id", "report-digest", "receipt", "registered-profile", "source-transfer", "false-registration"])
def test_registration_binding_cannot_be_forged_or_redirected(platform, corruption):
    node, users, asset = registered(platform)
    with platform.app.state.db.write() as session:
        command = session.get(Command, asset["validation"]["id"])
        row = session.get(Asset, asset["id"])
        metadata = deepcopy(row.metadata_json)
        report = metadata[INTERNAL_KEY]
        if corruption in {"asset", "path", "digest", "profile"}:
            key = {"asset": "asset_id", "path": "path", "digest": "input_digest", "profile": "profile_id"}[corruption]
            command.payload = {**command.payload, key: uid() if corruption == "asset" else "/machine/another"}
        elif corruption == "node":
            report["registration"]["node_id"] = uid()
        elif corruption == "actor":
            command.actor_user_id = users["bob"]
        elif corruption == "created-at":
            row.created_at = "2000-01-01T00:00:00+00:00"
        elif corruption == "report-id":
            report["id"] = uid()
        elif corruption == "report-digest":
            report["input_digest"] = "f" * 64
        elif corruption == "receipt":
            report.pop("registration")
        elif corruption == "registered-profile":
            command.payload = {**command.payload, "registered_profile": None}
        elif corruption == "source-transfer":
            command.payload = {**command.payload, "source_transfer_id": uid()}
            command.required_permission = "assets.transfer"
        else:
            command.payload = {**command.payload, "source_registration": False}
        row.metadata_json = metadata
    assert not any(item["id"] == asset["validation"]["id"] for item in platform.commands(node))
    assert queued(platform, asset).status == "REJECTED"


def test_new_registration_never_reuses_or_rejects_an_imported_report_reference(platform):
    _, _, asset = registered(platform)
    original_id = asset["validation"]["id"]
    with platform.app.state.db.write() as session:
        original = session.get(Asset, asset["id"])
        imported = Asset(id=uid(), node_id=original.node_id, kind="DATASET", name="imported", path=original.path,
                         version=original.version, metadata_json=deepcopy(original.metadata_json),
                         state="REGISTERED", created_at=original.created_at)
        session.add(imported)
        session.flush()
        fresh = queue_validation(session, imported, source_registration=True)
        assert fresh["id"] != original_id and fresh["registration"]["asset_id"] == imported.id
        assert session.get(Command, original_id).status == "PENDING"
