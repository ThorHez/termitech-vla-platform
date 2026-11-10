"""Training operators can validate shared datasets without taking ownership."""

from copy import deepcopy

import pytest
from sqlalchemy import func, select

from vla_control.dataset_validations import INTERNAL_KEY
from vla_control.models import Asset, Command, Profile

from .test_dataset_registration_validation import registered
from .test_dataset_validations import ack, create, get, result
from .test_resource_ownership import login


def finish_registration(platform, node, asset):
    report = asset["validation"]
    assert any(item["id"] == report["id"] for item in platform.commands(node))
    response = ack(platform, node, report, result(asset))
    assert response.status_code == 200, response.text
    assert get(platform, asset)["status"] == "SUCCEEDED"


@pytest.mark.parametrize("registrar", ["alice", "bob", None])
def test_operator_can_request_and_complete_validation_regardless_of_registrar(platform, registrar):
    node, users, asset = registered(platform, registrar or "bob")
    finish_registration(platform, node, asset)
    if registrar is None:
        with platform.app.state.db.write() as session:
            stored = session.get(Asset, asset["id"])
            stored.registered_by_user_id, stored.state = None, "READY"
    login(platform, "alice")

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
    with platform.app.state.db.read() as session:
        stored = session.get(Asset, asset["id"])
        assert stored.registered_by_user_id == (users[registrar] if registrar else None)
        assert {key: value for key, value in stored.metadata_json.items() if key != INTERNAL_KEY} == asset["metadata"]


@pytest.mark.parametrize("state", ["ARCHIVED", "UNAVAILABLE"])
def test_operator_cannot_validate_shared_dataset_unavailable_for_training(platform, state):
    node, _, asset = registered(platform, "bob")
    finish_registration(platform, node, asset)
    with platform.app.state.db.write() as session:
        session.get(Asset, asset["id"]).state = state
    login(platform, "alice")
    response = create(platform, asset)
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "DATASET_VALIDATION_ASSET_UNAVAILABLE"


def test_viewer_cannot_validate_even_if_recorded_as_registrar(platform):
    node, users, asset = registered(platform)
    finish_registration(platform, node, asset)
    with platform.app.state.db.write() as session:
        stored = session.get(Asset, asset["id"])
        stored.registered_by_user_id = users["reader"]
        before = deepcopy(stored.metadata_json)
        count = session.scalar(select(func.count()).select_from(Command))

    login(platform, "reader")
    denied = create(platform, asset)
    assert denied.status_code == 403, denied.text
    assert denied.json()["error"]["code"] == "FORBIDDEN"
    with platform.app.state.db.read() as session:
        assert session.get(Asset, asset["id"]).metadata_json == before
        assert session.scalar(select(func.count()).select_from(Command)) == count

    login(platform, "remaining-admin")
    allowed = create(platform, asset)
    assert allowed.status_code == 202, allowed.text
    with platform.app.state.db.read() as session:
        command = session.get(Command, allowed.json()["id"])
        assert command.actor_user_id == users["remaining-admin"]
        assert command.required_permission == "datasets.validate"


def test_existing_operator_command_is_delivered_after_registrar_changes(platform):
    node, users, asset = registered(platform)
    finish_registration(platform, node, asset)
    report = create(platform, asset).json()
    with platform.app.state.db.write() as session:
        session.get(Asset, asset["id"]).registered_by_user_id = users["bob"]
    assert any(item["id"] == report["id"] for item in platform.commands(node))
    assert ack(platform, node, report, result(asset)).status_code == 200


def test_operator_validation_rechecks_role_before_delivery(platform):
    node, users, asset = registered(platform)
    finish_registration(platform, node, asset)
    response = create(platform, asset)
    assert response.status_code == 202, response.text
    report = response.json()

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


@pytest.mark.parametrize("condition", ["pending", "delivered", "revoked"])
def test_shared_refresh_coalesces_same_profile_and_does_not_replace_live_pending_work(platform, condition):
    node, users, asset = registered(platform)
    finish_registration(platform, node, asset)
    with platform.app.state.db.write() as session:
        session.add(Profile(node_id=node["node_id"], id="another", name="Second", code_dir="/machine/other",
                            python="/usr/bin/python3", adapter="openpi-sft"))
    original = create(platform, asset, profile_id="default").json()
    login(platform, "bob")
    assert create(platform, asset, profile_id="default").json()["id"] == original["id"]

    if condition == "delivered":
        assert any(item["id"] == original["id"] for item in platform.commands(node))
    elif condition == "revoked":
        platform.app.state.auth.update_user(None, users["alice"], {"role": "viewer"})
    response = create(platform, asset, profile_id="another")
    if condition == "revoked":
        assert response.status_code == 202, response.text
        assert response.json()["id"] != original["id"] and response.json()["profile_id"] == "another"
    else:
        assert response.status_code == 409, response.text
        assert response.json()["error"]["code"] == "DATASET_VALIDATION_ALREADY_PENDING"
        assert get(platform, asset)["id"] == original["id"]
    with platform.app.state.db.read() as session:
        old = session.get(Command, original["id"])
        assert old.actor_user_id == users["alice"]
        assert old.status == ("REJECTED" if condition == "revoked" else "PENDING")
