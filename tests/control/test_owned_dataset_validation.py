"""Operators can revalidate datasets under their recorded registration ownership."""

from copy import deepcopy

import pytest
from sqlalchemy import func, select

from vla_control.dataset_validations import INTERNAL_KEY
from vla_control.models import Asset, Command

from .test_dataset_registration_validation import registered
from .test_dataset_validations import ack, create, get, result
from .test_resource_ownership import login


def finish_registration(platform, node, asset):
    report = asset["validation"]
    assert any(item["id"] == report["id"] for item in platform.commands(node))
    response = ack(platform, node, report, result(asset))
    assert response.status_code == 200, response.text
    assert get(platform, asset)["status"] == "SUCCEEDED"


def test_owner_can_request_and_complete_a_fresh_validation(platform):
    node, users, asset = registered(platform)
    finish_registration(platform, node, asset)

    response = create(platform, asset)
    assert response.status_code == 202, response.text
    report = response.json()
    assert report["id"] != asset["validation"]["id"] and report["status"] == "PENDING"
    with platform.app.state.db.read() as session:
        command = session.get(Command, report["id"])
        assert command.actor_user_id == users["alice"]
        assert command.required_permission == "datasets.validate"
        assert command.payload["asset_id"] == asset["id"]
        assert "source_registration" not in command.payload and "source_transfer_id" not in command.payload

    assert create(platform, asset).json()["id"] == report["id"]
    cancel = platform.client.post(f"/api/v1/assets/{asset['id']}/validation/cancel")
    assert cancel.status_code == 403
    assert any(item["id"] == report["id"] for item in platform.commands(node))
    assert ack(platform, node, report, result(asset)).status_code == 200
    completed = get(platform, asset)
    assert completed["id"] == report["id"] and completed["status"] == "SUCCEEDED"


@pytest.mark.parametrize("caller,registrar,error_code", [
    ("alice", "bob", "RESOURCE_ACCESS_DENIED"),
    ("alice", None, "RESOURCE_ACCESS_DENIED"),
    ("reader", "reader", "FORBIDDEN"),
])
def test_manual_validation_requires_write_authority_and_recorded_registrar(platform, caller, registrar, error_code):
    node, users, asset = registered(platform, "bob" if registrar == "bob" else "alice")
    finish_registration(platform, node, asset)
    with platform.app.state.db.write() as session:
        stored = session.get(Asset, asset["id"])
        stored.registered_by_user_id = users[registrar] if registrar else None
        # Client-controlled metadata cannot grant ownership of the asset.
        stored.metadata_json = {**stored.metadata_json, "registered_by_user_id": users[caller]}
        before = deepcopy(stored.metadata_json)
        count = session.scalar(select(func.count()).select_from(Command))

    login(platform, caller)
    denied = create(platform, asset)
    assert denied.status_code == 403, denied.text
    assert denied.json()["error"]["code"] == error_code
    with platform.app.state.db.read() as session:
        assert session.get(Asset, asset["id"]).metadata_json == before
        assert session.scalar(select(func.count()).select_from(Command)) == count

    login(platform, "remaining-admin")
    allowed = create(platform, asset)
    assert allowed.status_code == 202, allowed.text
    with platform.app.state.db.read() as session:
        command = session.get(Command, allowed.json()["id"])
        assert command.actor_user_id == users["remaining-admin"]
        assert command.required_permission == "assets.write"


@pytest.mark.parametrize("change", ["registrar", "role"])
def test_operator_validation_rechecks_ownership_and_role_before_delivery(platform, change):
    node, users, asset = registered(platform)
    finish_registration(platform, node, asset)
    response = create(platform, asset)
    assert response.status_code == 202, response.text
    report = response.json()

    if change == "registrar":
        with platform.app.state.db.write() as session:
            session.get(Asset, asset["id"]).registered_by_user_id = users["bob"]
    else:
        platform.app.state.auth.update_user(None, users["alice"], {"role": "viewer"})

    assert not any(item["id"] == report["id"] for item in platform.commands(node))
    with platform.app.state.db.read() as session:
        command = session.get(Command, report["id"])
        stored = session.get(Asset, asset["id"])
        metadata = stored.metadata_json
        assert command.status == "REJECTED" and command.delivered_at is None
        assert command.result["error"]["code"] == "COMMAND_AUTHORIZATION_REVOKED"
        assert stored.state == "REGISTERED"
        assert metadata[INTERNAL_KEY]["id"] == report["id"] and metadata[INTERNAL_KEY]["status"] == "FAILED"
        assert {key: value for key, value in metadata.items() if key != INTERNAL_KEY} == asset["metadata"]
