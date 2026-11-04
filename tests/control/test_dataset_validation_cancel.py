"""Only a proven undelivered dataset check can be synchronously cancelled."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select

from vla_control.common import APIError, now, uid
from vla_control.dataset_validations import INTERNAL_KEY
from vla_control.models import Allocation, Asset, Attempt, Command, Node, Run

from .conftest import ADMIN, ENROLL
from .test_dataset_validations import ack, create, get, register, result, worker
from .test_policy_deployments import add_run
from .test_resource_ownership import PASSWORD, accounts, login


def cancel(platform, asset, **kwargs):
    return platform.client.post(f"/api/v1/assets/{asset['id']}/validation/cancel",
                                headers={**ADMIN, **kwargs.pop("headers", {})}, **kwargs)


def command_snapshot(row):
    return {key: deepcopy(getattr(row, key)) for key in (
        "id", "node_id", "type", "payload", "created_at", "actor_user_id", "required_permission", "delivered_at")}


def active_training(platform, node):
    with platform.app.state.db.write() as session:
        run = add_run(session, node, state="RUNNING", parameters={"steps": 1000})
        attempt = Attempt(id=uid(), run_id=run.id, node_id=node["node_id"], number=1, execution_token=uid(),
                          state="RUNNING", created_at=now(), started_at=now(), gpu_uuids=["GPU-1"],
                          latest_metrics={"step": 23}, resources={"runner_pid": 123, "runtime_digest": "a" * 64})
        session.add(attempt)
        session.flush()
        session.add(Allocation(node_id=node["node_id"], gpu_uuid="GPU-1", attempt_id=attempt.id, created_at=now()))
        current = session.get(Node, node["node_id"])
        current.metrics = {**current.metrics, "training_supervisor": {"protocol_version": 1, "upgrade_safe": True,
                           "active_attempts": 1, "active_attempt_ids": [attempt.id]}}
        return run.id, attempt.id


def test_cancel_preserves_queue_identity_actor_payload_dataset_and_running_training(platform):
    node = worker(platform)
    users = accounts(platform)
    login(platform, "owner-admin")
    asset = register(platform, node)
    original_report = asset["validation"]
    run_id, attempt_id = active_training(platform, node)
    with platform.app.state.db.read() as session:
        queued = session.get(Command, original_report["id"])
        original = command_snapshot(queued)
        original_asset = {key: deepcopy(getattr(session.get(Asset, asset["id"]), key)) for key in
                          ("path", "name", "version", "state", "norm_asset_id")}
        attempt = session.get(Attempt, attempt_id)
        frozen_attempt = {key: deepcopy(getattr(attempt, key)) for key in
                          ("execution_token", "gpu_uuids", "latest_metrics", "resources", "started_at")}
        assert queued.actor_user_id == users["owner-admin"]
        with pytest.raises(APIError) as failure:
            platform.app.state.remote_manager._assert_upgrade_safe(session, node["node_id"])
        assert failure.value.code == "NODE_BUSY"
    response = cancel(platform, asset)
    assert response.status_code == 200, response.text
    cancelled = response.json()
    assert cancelled["id"] == original_report["id"] and cancelled["status"] == "CANCELLED"
    assert cancelled["error"]["code"] == "DATASET_VALIDATION_CANCELLED" and cancelled["completed_at"]
    assert get(platform, asset) == cancelled
    with platform.app.state.db.read() as session:
        queued = session.get(Command, original_report["id"])
        assert command_snapshot(queued) == original
        assert queued.status == "REJECTED" and queued.acknowledged_at == cancelled["completed_at"]
        assert queued.result == {"error": cancelled["error"], "checks": []}
        stored = session.get(Asset, asset["id"])
        assert {key: getattr(stored, key) for key in original_asset} == original_asset
        assert {key: value for key, value in stored.metadata_json.items() if key != INTERNAL_KEY} == asset["metadata"]
        assert session.get(Run, run_id).state == session.get(Attempt, attempt_id).state == "RUNNING"
        assert {key: getattr(session.get(Attempt, attempt_id), key) for key in frozen_attempt} == frozen_attempt
        assert session.get(Allocation, (node["node_id"], "GPU-1")).attempt_id == attempt_id
        # The original upgrade guard is unchanged: removing queued work is enough
        # only because the active immutable runner still has a fresh safe proof.
        platform.app.state.remote_manager._assert_upgrade_safe(session, node["node_id"])
    assert not any(row["id"] == original_report["id"] for row in platform.commands(node))


def test_cancel_is_idempotent_and_late_ack_cannot_revive_cancelled_work(platform):
    node = worker(platform)
    asset = register(platform, node)
    first = cancel(platform, asset)
    assert first.status_code == 200
    with platform.app.state.db.read() as session:
        acknowledged = session.get(Command, first.json()["id"]).acknowledged_at
    assert cancel(platform, asset).json() == first.json()
    assert ack(platform, node, first.json(), result(asset)).status_code == 200
    assert get(platform, asset) == first.json()
    with platform.app.state.db.read() as session:
        assert session.get(Command, first.json()["id"]).acknowledged_at == acknowledged
        assert len(list(session.scalars(select(Command).where(Command.type == "DATASET_VALIDATE")))) == 1
    fresh = create(platform, asset)
    assert fresh.status_code == 202 and fresh.json()["id"] != first.json()["id"]
    assert ack(platform, node, first.json(), result(asset)).status_code == 200
    assert get(platform, asset)["id"] == fresh.json()["id"] and get(platform, asset)["status"] == "PENDING"


def test_delivered_validation_cannot_be_reported_as_stopped(platform):
    node = worker(platform)
    asset = register(platform, node)
    assert any(row["id"] == asset["validation"]["id"] for row in platform.commands(node))
    with platform.app.state.db.read() as session:
        queued = session.get(Command, asset["validation"]["id"])
        original = command_snapshot(queued)
        report = deepcopy(session.get(Asset, asset["id"]).metadata_json[INTERNAL_KEY])
    response = cancel(platform, asset)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "DATASET_VALIDATION_ALREADY_DELIVERED"
    with platform.app.state.db.read() as session:
        queued = session.get(Command, asset["validation"]["id"])
        assert command_snapshot(queued) == original and queued.status == "PENDING" and queued.result is None
        assert session.get(Asset, asset["id"]).metadata_json[INTERNAL_KEY] == report
    assert ack(platform, node, asset["validation"], result(asset)).status_code == 200
    assert get(platform, asset)["status"] == "SUCCEEDED"


@pytest.mark.parametrize("status", ["SUCCEEDED", "FAILED", "STALE", "BLOCKED"])
def test_only_current_pending_report_is_cancellable(platform, status):
    node = worker(platform)
    asset = register(platform, node)
    with platform.app.state.db.write() as session:
        stored = session.get(Asset, asset["id"])
        stored.metadata_json = {**stored.metadata_json, INTERNAL_KEY: {**stored.metadata_json[INTERNAL_KEY], "status": status}}
    response = cancel(platform, asset)
    assert response.status_code == 409 and response.json()["error"]["code"] == "DATASET_VALIDATION_NOT_PENDING"
    with platform.app.state.db.read() as session:
        assert session.get(Command, asset["validation"]["id"]).status == "PENDING"


@pytest.mark.parametrize("mismatch", ["missing-command", "wrong-command-type", "wrong-node", "wrong-asset", "wrong-digest", "unknown-command-state"])
def test_cancel_never_changes_unmatched_or_uncertain_commands(platform, mismatch):
    node = worker(platform)
    asset = register(platform, node)
    with platform.app.state.db.write() as session:
        row = session.get(Command, asset["validation"]["id"])
        if mismatch == "missing-command":
            session.delete(row)
        elif mismatch == "wrong-command-type":
            row.type = "PREFLIGHT"
        elif mismatch == "wrong-node":
            other = Node(id=uid(), token_hash=uid(), name="other", hostname="other", roots=["/machine"],
                         output_root="/machine/output", status="ONLINE", last_seen=now(), created_at=now(), gpus=[], metrics={})
            session.add(other)
            session.flush()
            row.node_id = other.id
        elif mismatch == "wrong-asset":
            row.payload = {**row.payload, "asset_id": uid()}
        elif mismatch == "wrong-digest":
            row.payload = {**row.payload, "input_digest": "different"}
        else:
            row.status = "UNKNOWN"
    response = cancel(platform, asset)
    assert response.status_code == 409 and response.json()["error"]["code"] == "DATASET_VALIDATION_NOT_PENDING"
    with platform.app.state.db.read() as session:
        assert session.get(Asset, asset["id"]).metadata_json[INTERNAL_KEY]["status"] == "PENDING"


def test_cancel_has_dataset_scope_and_does_not_cancel_another_dataset(platform):
    node = worker(platform)
    asset = register(platform, node)
    other = register(platform, node, name="other", path="/machine/other")
    assert cancel(platform, asset).status_code == 200
    assert get(platform, other)["status"] == "PENDING"
    assert [row["id"] for row in platform.commands(node)] == [other["validation"]["id"]]
    assert platform.client.post("/api/v1/assets/missing/validation/cancel", headers=ADMIN).status_code == 404
    non_dataset = register(platform, node, kind="NORM", name="norm", path="/machine/norm.json", metadata={})
    assert cancel(platform, non_dataset).status_code == 404


@pytest.mark.parametrize("role,expected", [("admin", 200), ("operator", 403), ("viewer", 403)])
def test_cancel_requires_human_assets_write_and_csrf(platform, role, expected):
    node = worker(platform)
    asset = register(platform, node)
    path = f"/api/v1/assets/{asset['id']}/validation/cancel"
    for headers in ({}, ENROLL, node["headers"]):
        assert platform.client.post(path, headers=headers).status_code == 401
    password = "dataset-cancellation-password"
    platform.app.state.auth.create_user("cancel-" + role, password, role)
    with TestClient(platform.app, base_url="https://testserver") as client:
        signed_in = client.post("/api/v1/auth/login", json={"username": "cancel-" + role, "password": password})
        assert signed_in.status_code == 200
        assert client.post(path).status_code == 403
        response = client.post(path, headers={"X-CSRF-Token": signed_in.json()["csrf_token"]})
        assert response.status_code == expected, response.text
    with platform.app.state.db.read() as session:
        assert session.get(Command, asset["validation"]["id"]).status == ("REJECTED" if role == "admin" else "PENDING")


def test_cancel_checks_current_account_and_does_not_adopt_original_command_actor(platform):
    node = worker(platform)
    users = accounts(platform)
    second = platform.app.state.auth.create_user("second-admin", PASSWORD, "admin")["id"]
    login(platform, "owner-admin")
    asset = register(platform, node)
    assert cancel(platform, asset, headers={"X-Operation-User": second}).status_code == 403
    with platform.app.state.db.read() as session:
        assert session.get(Command, asset["validation"]["id"]).status == "PENDING"
    login(platform, "second-admin")
    assert cancel(platform, asset).status_code == 200
    with platform.app.state.db.read() as session:
        assert session.get(Command, asset["validation"]["id"]).actor_user_id == users["owner-admin"]


def test_delivery_and_cancel_are_serialized_without_claiming_an_executing_check_stopped(platform):
    node = worker(platform)
    asset = register(platform, node)
    barrier = Barrier(2)

    def deliver():
        barrier.wait(timeout=5)
        return platform.commands(node)

    def stop():
        barrier.wait(timeout=5)
        return cancel(platform, asset)

    with ThreadPoolExecutor(max_workers=2) as executor:
        delivery = executor.submit(deliver)
        cancellation = executor.submit(stop)
        delivered, response = delivery.result(timeout=10), cancellation.result(timeout=10)
    with platform.app.state.db.read() as session:
        queued = session.get(Command, asset["validation"]["id"])
        if response.status_code == 200:
            assert queued.status == "REJECTED" and queued.delivered_at is None
            assert all(row["id"] != queued.id for row in delivered)
        else:
            assert response.status_code == 409
            assert response.json()["error"]["code"] == "DATASET_VALIDATION_ALREADY_DELIVERED"
            assert queued.status == "PENDING" and queued.delivered_at is not None
            assert any(row["id"] == queued.id for row in delivered)
